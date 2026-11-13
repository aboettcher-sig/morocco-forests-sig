"""Minimum mapping unit (MMU) for the final change map, and the GeoPackage writer.

Changed pixels are grouped into patches of touching pixels. Patches smaller than
`min_pixels` are dropped from the final result. By default, pixels touching at a
corner also count as touching (8-connectivity, like GEE's connectedPixelCount).

Re-apply to a finished run without downloading anything again:

    python postprocess.py outputs/tangier_target2023_exp2021-2022
    python postprocess.py outputs/<run> --min-pixels 5 --connectivity 4

Columns added to pixels.csv:
    changed_before_mmu   change after the NBR / amplitude / pre-NBR masks
    patch_id             id of the touching-pixel patch (-1 if not changed)
    patch_size           number of pixels in that patch (0 if not changed)
    changed              final change: changed_before_mmu and patch_size >= min_pixels
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import ndimage

PIXEL_SIZE = 10.0  # m, Sentinel-2 sub-pixels of the 30 m grid (EPSG:26191)


def apply_mmu(res: pd.DataFrame, min_pixels: int = 3, connectivity: int = 8) -> pd.DataFrame:
    """Return a copy of `res` with patch columns and an MMU-filtered `changed`."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    res = res.copy()
    base = res["changed_before_mmu"] if "changed_before_mmu" in res else res["changed"]
    base = base.astype(bool).to_numpy()
    res["changed_before_mmu"] = base

    col = np.rint((res.x_26191 - res.x_26191.min()) / PIXEL_SIZE).astype(int).to_numpy()
    row = np.rint((res.y_26191.max() - res.y_26191) / PIXEL_SIZE).astype(int).to_numpy()
    grid = np.zeros((row.max() + 1, col.max() + 1), dtype=bool)
    grid[row[base], col[base]] = True

    structure = np.ones((3, 3), bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(grid, structure=structure)
    sizes = np.bincount(labels.ravel())

    lab = labels[row, col]
    res["patch_id"] = np.where(base, lab, -1)
    res["patch_size"] = np.where(base, sizes[lab], 0)
    res["changed"] = base & (res["patch_size"].to_numpy() >= min_pixels)
    return res


# ---------------------------------------------------------------------------
# Footprints (adapted from true_change_handoff, HISTORIC_DISTURBANCE_METHODS.md
# sections 6 and 11). Alternative to the MMU: it also uses the change dates and
# the surrounding pixels.
# ---------------------------------------------------------------------------
FOOTPRINT_DEFAULTS = dict(
    link_distance_m=14.2,   # two changed pixels are linked if their centres are this close (14.2 m = touching,
                            # diagonals included, for 10 m pixels; no gaps)...
    link_days=90,           # ...and their dates differ by at most this many days
    date_column="first_alert_day",  # pixel date used for linking, footprint date and onset; "julianDay" = GEE timing
    min_pixels=3,           # smallest footprint kept
    window_days=90,         # NBR departure is measured over this many days after the footprint date
    ring_m=300.0,           # nearby non-member pixels: within this distance of any member
    min_ring_pixels=10,     # a footprint needs at least this many nearby pixels to be tested
    ring_max_m=1500.0,      # too few clean pixels within ring_m (everything around changed too): look further
                            # out (600, 1000, ... up to this distance) for the nearest undisturbed pixels;
                            # none found -> "review: no clean surroundings" (not mapped as change)
    min_contrast=0.08,      # core departure - nearby departure must be at least this (NBR)
    exclude_ring_candidates=True,  # leave BULC-D candidates (probCls1 >= 0.5) out of the surroundings, so a
                                   # fire that also burned the surroundings does not cancel the contrast
    onset_days=45,          # onset: first group of linked members whose dates are this close...
    onset_min_pixels=2,     # ...and that has at least this many pixels
)


def _components(n, i, j):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def nbr_departure(y_tgt: np.ndarray, fit_tgt: np.ndarray, day: int, window_days: int,
                  rows: np.ndarray, min_obs: int = 3) -> np.ndarray:
    """Median of (expected - observed) NBR in the bins from `day` to `day + window_days`.

    Positive = NBR below what the harmonic fit of the expectation years predicts.
    NaN when the window holds fewer than `min_obs` clear bins.
    """
    k0 = max(int(day) // 2 - 1, 0)          # julianDay = 2 x event number, event 1 = bin 0
    k1 = min(k0 + window_days // 2, y_tgt.shape[1])
    # fit_tgt None: y_tgt already holds the departure (expected - observed)
    dep = (y_tgt[rows, k0:k1].astype(float) if fit_tgt is None
           else fit_tgt[rows, k0:k1] - y_tgt[rows, k0:k1])
    n = np.sum(~np.isnan(dep), axis=1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(dep, axis=1)
    return np.where(n >= min_obs, med, np.nan)


def _onset(x, y, jd, link_m, onset_days, min_px):
    """First group of >= min_px linked members whose dates are within onset_days.

    Members are taken in date order; at each date only members dated in the
    preceding onset_days count. Returns (onset = earliest date in that group,
    coherence date = the date the group first exists), or (nan, nan).
    """
    from scipy.spatial import cKDTree
    order = np.argsort(jd, kind="stable")
    for t in np.unique(jd[order]):
        sel = np.flatnonzero((jd <= t) & (jd >= t - onset_days))
        if sel.size < min_px:
            continue
        pairs = cKDTree(np.c_[x[sel], y[sel]]).query_pairs(link_m, output_type="ndarray")
        lab = _components(sel.size, pairs[:, 0], pairs[:, 1]) if len(pairs) else np.arange(sel.size)
        sizes = np.bincount(lab)
        ok = np.flatnonzero(sizes >= min_px)
        if ok.size:
            members = sel[np.isin(lab, ok)]
            return float(jd[members].min()), float(t)
    return np.nan, np.nan


def apply_footprints(res: pd.DataFrame, y_tgt: np.ndarray, fit_tgt: np.ndarray, origin=None, **kw):
    """Group changed pixels into footprints and test each one against its surroundings.

    Starts from `changed_before_mmu` (change after the NBR / amplitude / pre-NBR
    masks). Returns (res with fp_* columns and `changed_footprint`, one row per footprint).
    `y_tgt` / `fit_tgt` are the target-year series and fit, in the row order of `res`.
    `origin` (continuous runs): start date of bin 0, so that day d is the bin
    starting on origin + d - 2 days; adds fp_date / fp_onset_date / fp_date_month.
    """
    from scipy.spatial import cKDTree
    p = {**FOOTPRINT_DEFAULTS, **kw}
    res = res.copy()
    base = (res["changed_before_mmu"] if "changed_before_mmu" in res else res["changed"]).astype(bool).to_numpy()
    xy = res[["x_26191", "y_26191"]].to_numpy(float)
    col = p["date_column"] if p["date_column"] in res else "julianDay"
    jd_all = res[col].to_numpy(float)
    jd_all = np.where(np.isnan(jd_all), res["julianDay"].to_numpy(float), jd_all)   # no lasting alert: GEE date
    idx = np.flatnonzero(base)
    cand = (res["candidate_changed"].astype(bool).to_numpy()
            if p["exclude_ring_candidates"] and "candidate_changed" in res else np.zeros(len(res), bool))

    # 1. Link changed pixels that are close in space AND in time.
    pairs = cKDTree(xy[idx]).query_pairs(p["link_distance_m"], output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.abs(jd_all[idx][pairs[:, 0]] - jd_all[idx][pairs[:, 1]]) <= p["link_days"]]
    lab = _components(idx.size, pairs[:, 0], pairs[:, 1]) if len(pairs) else np.arange(idx.size)

    tree_all = cKDTree(xy)
    water = res["water"].astype(bool).to_numpy() if "water" in res else np.zeros(len(res), bool)
    clean = ~cand & ~water
    member_fp = np.full(len(res), -1)
    member_fp[idx] = lab
    rows = []
    for fp in np.unique(lab):
        m = idx[lab == fp]
        jd = jd_all[m]
        # 2. One date for the whole footprint: the median of its members' dates.
        date = int(2 * round(np.median(jd) / 2))
        core = nbr_departure(y_tgt, fit_tgt, date, p["window_days"], m)
        # 3. Nearby non-members, measured at the same date.
        near = np.unique(np.concatenate(tree_all.query_ball_point(xy[m], p["ring_m"])))
        ring_all = near[member_fp[near] != fp]
        ring = ring_all[clean[ring_all]]          # disturbed neighbours and water do not count as "surroundings"
        ring_r = p["ring_m"]
        if ring.size < p["min_ring_pixels"] and p["ring_max_m"] > p["ring_m"]:
            # everything nearby changed too: compare with the nearest undisturbed pixels further out
            tree_m = cKDTree(xy[m])
            radii = sorted({r for r in (600.0, 1000.0) if p["ring_m"] < r < p["ring_max_m"]} | {float(p["ring_max_m"])})
            lo, hi = xy[m].min(axis=0), xy[m].max(axis=0)
            for r in radii:
                box = np.flatnonzero(clean & (member_fp != fp) & np.all((xy >= lo - r) & (xy <= hi + r), axis=1))
                d = tree_m.query(xy[box], distance_upper_bound=r)[0] if box.size else np.array([])
                far = box[np.isfinite(d)]
                if far.size >= p["min_ring_pixels"]:
                    ring, ring_r = far, r
                    break
        ring_dep = nbr_departure(y_tgt, fit_tgt, date, p["window_days"], ring) if ring.size else np.array([])
        core_drop = float(np.nanmedian(core)) if np.isfinite(core).any() else np.nan
        ring_drop = float(np.nanmedian(ring_dep)) if np.isfinite(ring_dep).sum() >= p["min_ring_pixels"] else np.nan
        contrast = core_drop - ring_drop
        # 4. Onset: the first date at which >= 2 linked members changed within 45 days.
        onset, coherent = _onset(xy[m, 0], xy[m, 1], jd, p["link_distance_m"], p["onset_days"], p["onset_min_pixels"])
        if m.size < p["min_pixels"]:
            status = "rejected: too small"
        elif np.isnan(core_drop):
            status = "rejected: no clear observations after its date"   # e.g. dated in the last days of the year
        elif np.isnan(ring_drop) and ring_all.size >= p["min_ring_pixels"] and ring.size < p["min_ring_pixels"]:
            status = "review: no clean surroundings"   # everything within ring_max_m changed too; not mapped
        elif np.isnan(ring_drop):
            status = "rejected: no nearby pixels to compare"
        elif not (contrast >= p["min_contrast"]):
            status = "rejected: no stronger than surroundings"
        else:
            status = "accepted"
        rows.append(dict(fp_id=int(fp), fp_size=int(m.size), fp_julianDay=date,
                         fp_month=(pd.Timestamp(2023, 1, 1) + pd.Timedelta(days=min(date, 364))).month
                         if origin is None else (pd.Timestamp(origin) + pd.Timedelta(days=date - 2)).month,
                         fp_onset=onset, fp_coherence=coherent,
                         fp_date_min=float(jd.min()), fp_date_max=float(jd.max()),
                         fp_core_drop=core_drop, fp_ring_drop=ring_drop, fp_contrast=contrast,
                         fp_ring_pixels=int(ring.size), fp_ring_m=float(ring_r),
                         fp_ring_excluded=int(ring_all.size - ring.size), fp_status=status))
    fps = pd.DataFrame(rows)
    if origin is not None and len(fps):
        o = pd.Timestamp(origin)
        fps["fp_date"] = (o + pd.to_timedelta(fps.fp_julianDay - 2, unit="D")).dt.strftime("%Y-%m-%d")
        fps["fp_onset_date"] = (o + pd.to_timedelta(fps.fp_onset - 2, unit="D")).dt.strftime("%Y-%m-%d")
        fps["fp_date_month"] = fps.fp_date.str[:7]
        fps["fp_year"] = fps.fp_date.str[:4].astype(int)
        fps["fp_onset_year"] = pd.to_numeric(fps.fp_onset_date.str[:4], errors="coerce").astype("Int64")
        fps["fp_first_date"] = (o + pd.to_timedelta(fps.fp_date_min - 2, unit="D")).dt.strftime("%Y-%m-%d")
        fps["fp_last_date"] = (o + pd.to_timedelta(fps.fp_date_max - 2, unit="D")).dt.strftime("%Y-%m-%d")
    for c in [c for c in res.columns if c.startswith("fp_")] + ["changed_footprint"]:
        res.pop(c) if c in res else None
    res["fp_id"] = member_fp
    if len(fps):
        res = res.merge(fps, on="fp_id", how="left")
    res["changed_footprint"] = res.get("fp_status", pd.Series(index=res.index, dtype=object)).astype(str).str.startswith("accepted").to_numpy()
    return res, fps


def write_footprint_layers(res: pd.DataFrame, fps: pd.DataFrame, out_dir: str) -> None:
    """Add 'footprints' (dissolved polygons, every footprint with its status) and
    'change_footprint' (accepted pixels) to change.gpkg; write footprints.csv."""
    import geopandas as gpd
    from shapely.geometry import box
    fps.to_csv(os.path.join(out_dir, "footprints.csv"), index=False)
    m = res[res.fp_id >= 0]
    h = PIXEL_SIZE / 2
    g = gpd.GeoDataFrame(m[["fp_id"]], geometry=[box(x - h, y - h, x + h, y + h) for x, y in zip(m.x_26191, m.y_26191)],
                         crs="EPSG:26191").dissolve("fp_id").reset_index().merge(fps, on="fp_id")
    gpkg = os.path.join(out_dir, "change.gpkg")
    g.to_file(gpkg, layer="footprints", driver="GPKG")
    acc = res[res.changed_footprint]
    gpd.GeoDataFrame(acc.drop(columns=["x_26191", "y_26191"]),
                     geometry=[box(x - h, y - h, x + h, y + h) for x, y in zip(acc.x_26191, acc.y_26191)],
                     crs="EPSG:26191").to_file(gpkg, layer="change_footprint", driver="GPKG")


def write_gpkg(res: pd.DataFrame, out_dir: str, aoi_wkt: str | None = None) -> str | None:
    try:
        import geopandas as gpd
        from shapely import wkt as shp_wkt
        from shapely.geometry import box
    except ImportError:
        print("geopandas not installed: skipped change.gpkg (pixels.csv has x/y in EPSG:26191)")
        return None
    h = PIXEL_SIZE / 2
    geom = [box(x - h, y - h, x + h, y + h) for x, y in zip(res.x_26191, res.y_26191)]
    gdf = gpd.GeoDataFrame(res.drop(columns=["x_26191", "y_26191"]), geometry=geom, crs="EPSG:26191")
    gpkg = os.path.join(out_dir, "change.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    gdf[gdf.changed].to_file(gpkg, layer="change", driver="GPKG")
    if "changed_before_mmu" in gdf:
        gdf[gdf.changed_before_mmu].to_file(gpkg, layer="change_before_mmu", driver="GPKG")
    gdf[gdf.candidate_changed].to_file(gpkg, layer="candidates", driver="GPKG")
    if aoi_wkt:
        gpd.GeoSeries([shp_wkt.loads(aoi_wkt)], crs="EPSG:26191").to_file(gpkg, layer="aoi", driver="GPKG")
    if "water" in gdf and gdf.water.any():
        w = gdf[gdf.water.astype(bool)]
        gpd.GeoDataFrame({"water": [1]}, geometry=[w.geometry.union_all()], crs="EPSG:26191").explode(
            index_parts=False).to_file(gpkg, layer="water", driver="GPKG")
    return gpkg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--min-pixels", type=int, default=3)
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    ap.add_argument("--footprints", action="store_true",
                    help="also group changed pixels into footprints (distance + date + contrast with surroundings)")
    for k, v in FOOTPRINT_DEFAULTS.items():
        kind = (lambda t: t.lower() in ("1", "true", "yes", "on")) if isinstance(v, bool) else type(v)
        ap.add_argument("--fp-" + k.replace("_", "-"), type=kind, default=None, help=f"footprint {k} (default {v})")
    a = ap.parse_args()

    csv = os.path.join(a.run_dir, "pixels.csv")
    res = apply_mmu(pd.read_csv(csv), a.min_pixels, a.connectivity)
    pfile = os.path.join(a.run_dir, "params.json")
    params = json.load(open(pfile)) if os.path.exists(pfile) else {}
    params.update(mmu_pixels=a.min_pixels, mmu_connectivity=a.connectivity)
    fps = None
    if a.footprints:
        fp_kw = {**FOOTPRINT_DEFAULTS, **params.get("footprints", {}),
                 **{k: getattr(a, "fp_" + k) for k in FOOTPRINT_DEFAULTS if getattr(a, "fp_" + k) is not None}}
        arr = np.load(os.path.join(a.run_dir, "arrays.npz"))
        cont = params.get("mode") == "continuous"
        if "first_alert_day" not in res:
            import bulcd_core as core
            res["first_alert_day"] = core.first_sustained_alert(
                arr["prob1_history"].astype(float), params.get("alert_threshold", 0.5),
                params.get("alert_hold_days", 60), params.get("day_step", 2))
            params.setdefault("alert_threshold", 0.5); params.setdefault("alert_hold_days", 60)
        assert (arr["cell_id"] == res.cell_id.to_numpy()).all(), "arrays.npz and pixels.csv rows differ"
        yk, fk = ("nbr", "fit") if cont else ("y_tgt", "fit_tgt")
        fit_ = arr[fk].astype(float) + (arr["regional"].astype(float) if cont and arr["regional"].ndim == 2 else 0)
        res, fps = apply_footprints(res, arr[yk].astype(float), fit_,
                                    origin=np.datetime64(params["origin"]) if cont else None, **fp_kw)
        params["footprints"] = fp_kw
    res.to_csv(csv, index=False)
    json.dump(params, open(pfile, "w"), indent=2)
    write_gpkg(res, a.run_dir, params.get("aoi_wkt_26191"))
    print(f"{a.run_dir}: {int(res.changed_before_mmu.sum())} changed pixels before MMU -> "
          f"{int(res.changed.sum())} after MMU >= {a.min_pixels} px ({a.connectivity}-connected), "
          f"{res.loc[res.changed, 'patch_id'].nunique()} patches")
    if fps is not None:
        write_footprint_layers(res, fps, a.run_dir)
        print(f"footprints: {len(fps)} groups -> {int(fps.fp_status.str.startswith('accepted').sum())} accepted, "
              f"{int(res.changed_footprint.sum())} pixels")
        print(fps.fp_status.value_counts().to_string())


if __name__ == "__main__":
    main()
