"""Pair each large change footprint with an adjacent, unchanged forest control area.

    python make_controls.py outputs/<continuous run> [--min-size 100] [--min-nbr 0.3]

For every accepted footprint larger than --min-size pixels, grow a control area of
the same number of pixels from the nearest eligible pixels around it. Eligible
(10 m sub-pixel):
  * never alerted: probCls1 < 0.5 in every 2-day bin of the monitoring period
    (so no change event of any kind, recovered ones included);
  * forest-like seasonality: harmonic amplitude of NBR (area-wide signal removed,
    whole record) <= the run's amp_threshold (0.15), as in the change masks;
  * pre-change NBR (median before the footprint's onset, same dates as the footprint, up to
    30 days before it) within --pixel-tol (0.15) of the footprint's;
  * not water (water mask) and at least --gap-m (20 m) from any changed pixel,
    so mixed edge pixels are left out.
A control is grown as a compact area from a starting pixel, through touching eligible
pixels (closest first, pixels whose pre-change NBR is closer to the footprint's earlier),
to the footprint's size. Starting points are tried nearest to the footprint first (within
--max-dist-m, 1000 m); the first one whose control has a pre-change median NBR within
--nbr-tol (0.05) of the footprint's, and at least --min-nbr (0.3), is kept. Larger
footprints choose first; a pixel belongs to one control only.

Writes, in the run folder:
  controls.gpkg  layers 'footprints' (the large footprints) and 'controls'
                 (one polygon per control, same pair_id = fp_id)
  controls.csv   one row per pair: sizes, median NBR, amplitude, distances
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PIX = 10.0


def pixel_stats(run_dir, monitor_from):
    """Per pixel (pixels.csv order): max probCls1 while monitoring, median NBR, harmonic amplitude."""
    n = sum(1 for _ in open(os.path.join(run_dir, "pixels.csv"))) - 1
    maxp, med, amp = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    files = sorted(glob.glob(os.path.join(run_dir, "tiles", "*.npz"))) or [os.path.join(run_dir, "arrays.npz")]
    for f in files:
        t = np.load(f)
        rows = t["rows"] if "rows" in t.files else np.arange(n)
        d = t["bin_start"].astype("datetime64[D]")
        mon = d >= np.datetime64(monitor_from)
        maxp[rows] = np.nanmax(t["prob1"][:, mon].astype(np.float32), axis=1)
        y = t["nbr"].astype(np.float32)
        med[rows] = np.nanmedian(y, axis=1)
        reg = t["regional"].astype(np.float32) if t["regional"].ndim == 2 else 0
        yc = y - reg
        doy = (d - d.astype("datetime64[Y]")).astype(int) + 1
        w = 2 * np.pi * doy / 365.25
        X = np.c_[np.ones(len(d)), np.cos(w), np.sin(w)]                 # (T, 3)
        m = np.isfinite(yc)
        y0 = np.where(m, yc, 0)
        xtx = (m.astype(np.float64) @ np.einsum("ti,tj->tij", X, X).reshape(len(d), 9)).reshape(-1, 3, 3)
        xty = y0.astype(np.float64) @ X
        ok = m.sum(axis=1) >= 20
        b = np.full((len(rows), 3), np.nan)
        b[ok] = np.linalg.solve(xtx[ok], xty[ok][..., None])[..., 0]
        amp[rows] = np.hypot(b[:, 1], b[:, 2])
    return maxp, med, amp


def grow(seed_dist, prio, grid, xy_grid, target, used):
    """Nearest-first growth through 8-connected eligible pixels.

    seed_dist: {idx: distance to the footprint} to start from; prio: the same for every
    eligible pixel within reach (a pixel not in prio cannot be used)."""
    heap = [(dist, i) for i, dist in seed_dist.items()]
    heapq.heapify(heap)
    seen, out, patches = set(seed_dist), [], 1
    rest = sorted((d, i) for i, d in prio.items())                  # for starting a new patch, nearest first
    k = 0
    while len(out) < target:
        if not heap:                                                 # patch exhausted: start the next-nearest one
            while k < len(rest) and (rest[k][1] in seen or used[rest[k][1]]):
                k += 1
            if k == len(rest):
                break
            seen.add(rest[k][1]); heapq.heappush(heap, rest[k]); patches += 1
        dist, i = heapq.heappop(heap)
        if used[i]:
            continue
        out.append(i)
        cx, cy = xy_grid[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = grid.get((cx + dx, cy + dy))
                if j is not None and j not in seen and j in prio and not used[j]:
                    seen.add(j)
                    heapq.heappush(heap, (prio[j], j))
    return out, patches


def series(run_dir, n):
    """(n, T) float16 NBR of every pixel (pixels.csv order), and the bin dates."""
    out, dates = None, None
    files = sorted(glob.glob(os.path.join(run_dir, "tiles", "*.npz"))) or [os.path.join(run_dir, "arrays.npz")]
    for f in files:
        t = np.load(f)
        rows = t["rows"] if "rows" in t.files else np.arange(t["nbr"].shape[0])
        dates = t["bin_start"].astype("datetime64[D]")
        if out is None:
            out = np.full((n, len(dates)), np.nan, np.float16)
        out[rows] = t["nbr"]
    return out, dates


def window_median(S, idx, sel):
    """Median over pixels of each pixel's median NBR in the bins `sel`."""
    if len(idx) == 0:
        return np.nan
    with np.errstate(all="ignore"):
        v = np.nanmedian(S[np.asarray(idx, int)][:, sel].astype(np.float32), axis=1)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--min-size", type=int, default=100, help="footprints with more than this many pixels")
    ap.add_argument("--min-nbr", type=float, default=0.3, help="minimum pre-change median NBR of a control pixel")
    ap.add_argument("--nbr-tol", type=float, default=0.05,
                    help="control pixel pre-change NBR within this of the footprint's pre-change NBR")
    ap.add_argument("--pixel-tol", type=float, default=0.15,
                    help="control pixels' own pre-change NBR within this of the footprint's (the control median "
                         "must be within --nbr-tol)")
    ap.add_argument("--nbr-weight", type=float, default=1000.0,
                    help="growth order: metres of distance one NBR unit of mismatch is worth")
    ap.add_argument("--seed-spacing-m", type=float, default=100.0)
    ap.add_argument("--max-seeds", type=int, default=60)
    ap.add_argument("--gap-m", type=float, default=20.0, help="minimum distance from any changed pixel")
    ap.add_argument("--max-dist-m", type=float, default=1000.0, help="controls are searched within this distance")
    a = ap.parse_args()
    import geopandas as gpd
    from shapely.geometry import box

    params = json.load(open(os.path.join(a.run_dir, "params.json")))
    amp_max = params.get("amp_threshold", 0.15)
    px = pd.read_csv(os.path.join(a.run_dir, "pixels.csv"),
                     usecols=lambda c: c in {"cell_id", "subpixel_id", "x_26191", "y_26191", "candidate_changed",
                                             "changed_before_mmu", "changed_footprint", "fp_id", "water"})
    fps = pd.read_csv(os.path.join(a.run_dir, "footprints.csv"))
    big = fps[fps.fp_status.str.startswith("accepted") & (fps.fp_size > a.min_size)].sort_values("fp_size", ascending=False)
    print(f"{len(big)} accepted footprints > {a.min_size} px; computing pixel statistics...")
    maxp, med, amp = pixel_stats(a.run_dir, params["monitor_from"])
    px["maxprob"], px["median_nbr"], px["amp"] = maxp, med, amp

    xy = px[["x_26191", "y_26191"]].to_numpy(float)
    changed = (px.candidate_changed | px.get("changed_before_mmu", False) | px.changed_footprint).to_numpy(bool)
    water = px["water"].to_numpy(bool) if "water" in px else np.zeros(len(px), bool)
    d_changed = cKDTree(xy[changed]).query(xy, distance_upper_bound=a.gap_m + 1e-6)[0] if changed.any() else np.full(len(px), np.inf)
    eligible = ((maxp < 0.5) & (amp <= amp_max) & ~water & ~np.isfinite(d_changed) & np.isfinite(maxp))
    print(f"  never alerted, amplitude <= {amp_max}, not water, >= {a.gap_m:.0f} m from change: "
          f"{int(eligible.sum())} of {len(px)} pixels (NBR matched per footprint below)")
    S, dates = series(a.run_dir, len(px))

    x0, y0 = xy.min(axis=0)
    xy_grid = np.round((xy - [x0, y0]) / PIX).astype(int)
    grid = {(cx, cy): i for i, (cx, cy) in enumerate(map(tuple, xy_grid))}
    tree = cKDTree(xy)
    used = np.zeros(len(px), bool)
    rows, ctrl_geoms, fp_geoms = [], [], []
    members, controls = {}, {}
    h = PIX / 2
    for _, f in big.iterrows():
        m = np.flatnonzero(px.fp_id.to_numpy() == f.fp_id)
        tm = cKDTree(xy[m])
        near = np.unique(np.concatenate(tree.query_ball_point(xy[m], a.max_dist_m)))
        near = near[eligible[near] & ~used[near]]
        on = np.datetime64(f.fp_onset_date if isinstance(f.fp_onset_date, str) else f.fp_date)
        before = dates < on - np.timedelta64(30, "D")
        fp_before = window_median(S, m, before)
        target = int(f.fp_size)
        ctrl, patches = [], 0
        if near.size:
            with np.errstate(all="ignore"):
                wmed = np.nanmedian(S[near][:, before].astype(np.float32), axis=1)
            keep = np.isfinite(wmed) & (np.abs(wmed - fp_before) <= a.pixel_tol)
            near, wmed = near[keep], wmed[keep]
        if near.size and np.isfinite(fp_before):
            wm = dict(zip(near.tolist(), wmed.tolist()))
            # candidate starting points: thinned to one per seed_spacing_m, nearest to the footprint first
            dfp = tm.query(xy[near])[0]
            seeds, taken = [], np.empty((0, 2))
            for k in np.argsort(dfp):
                q = xy[near[k]]
                if taken.size == 0 or np.hypot(*(taken - q).T).min() >= a.seed_spacing_m:
                    taken = np.vstack([taken, q]); seeds.append(int(near[k]))
                if len(seeds) >= a.max_seeds:
                    break
            best = None
            for sd in seeds:
                # compact growth around the seed; pixels with NBR closer to the footprint's come first
                ds = np.hypot(*(xy[near] - xy[sd]).T)
                prio = {i: float(d_ + a.nbr_weight * abs(wm[i] - fp_before)) for i, d_ in zip(near.tolist(), ds)}
                reg, pt = grow({sd: 0.0}, prio, grid, xy_grid, target, used)
                rm = float(np.median([wm[i] for i in reg]))
                if abs(rm - fp_before) > a.nbr_tol or rm < a.min_nbr:
                    continue
                if len(reg) >= target:
                    best = (reg, pt); break                        # nearest start that gives a full, matched control
                if best is None or len(reg) > len(best[0]):
                    best = (reg, pt)
            if best is not None:
                ctrl, patches = best
        used[ctrl] = True
        members[int(f.fp_id)], controls[int(f.fp_id)] = m, np.array(ctrl, int)
        c = np.array(ctrl, int)
        rows.append(dict(pair_id=int(f.fp_id), fp_size=int(f.fp_size), fp_date=f.fp_date, fp_onset_date=f.fp_onset_date,
                         ctrl_size=int(c.size), ctrl_complete=bool(c.size >= f.fp_size), ctrl_patches=int(patches),
                         ctrl_median_nbr=float(np.nanmedian(med[c])) if c.size else np.nan,
                         ctrl_amplitude=float(np.nanmedian(amp[c])) if c.size else np.nan,
                         ctrl_maxprob=float(np.nanmax(maxp[c])) if c.size else np.nan,
                         ctrl_dist_min_m=float(tm.query(xy[c])[0].min()) if c.size else np.nan,
                         ctrl_dist_median_m=float(np.median(tm.query(xy[c])[0])) if c.size else np.nan,
                         ctrl_dist_max_m=float(tm.query(xy[c])[0].max()) if c.size else np.nan))
        if c.size:
            ctrl_geoms.append(gpd.GeoSeries([box(x - h, y - h, x + h, y + h) for x, y in xy[c]]).union_all())
        else:
            ctrl_geoms.append(None)
        fp_geoms.append(gpd.GeoSeries([box(x - h, y - h, x + h, y + h) for x, y in xy[m]]).union_all())
    out = pd.DataFrame(rows)
    out.insert(out.columns.get_loc("ctrl_size"), "fp_before_below_min", False)
    # NBR before and after the footprint's onset, for the footprint and its control (same dates)
    for col in ["fp_nbr_before", "fp_nbr_after", "ctrl_nbr_before", "ctrl_nbr_after"]:
        out[col] = np.nan
    for r, k in enumerate(out.pair_id):
        on = np.datetime64(out.at[r, "fp_onset_date"] if isinstance(out.at[r, "fp_onset_date"], str) else out.at[r, "fp_date"])
        before = dates < on - np.timedelta64(30, "D")
        after = dates >= on + np.timedelta64(30, "D")
        out.loc[r, ["fp_nbr_before", "fp_nbr_after", "ctrl_nbr_before", "ctrl_nbr_after"]] = [
            window_median(S, members[k], before), window_median(S, members[k], after),
            window_median(S, controls[k], before), window_median(S, controls[k], after)]
    out["fp_before_below_min"] = out.fp_nbr_before < a.min_nbr - a.nbr_tol   # no control can match under --min-nbr
    out["nbr_before_diff"] = out.ctrl_nbr_before - out.fp_nbr_before
    out = out.round(3)
    out.to_csv(os.path.join(a.run_dir, "controls.csv"), index=False)
    gpkg = os.path.join(a.run_dir, "controls.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    fcols = ["fp_id", "fp_size", "fp_date", "fp_onset_date", "fp_onset_year", "fp_year", "fp_status"]
    gpd.GeoDataFrame(big[fcols].reset_index(drop=True).assign(pair_id=big.fp_id.to_numpy()),
                     geometry=fp_geoms, crs="EPSG:26191").to_file(gpkg, layer="footprints", driver="GPKG")
    ok = [g is not None for g in ctrl_geoms]
    gpd.GeoDataFrame(out[ok].reset_index(drop=True), geometry=[g for g in ctrl_geoms if g is not None],
                     crs="EPSG:26191").to_file(gpkg, layer="controls", driver="GPKG")
    print(f"\n{len(out)} footprints: {int(out.ctrl_complete.sum())} with a full-size control, "
          f"{int(((out.ctrl_size > 0) & ~out.ctrl_complete).sum())} partial, {int((out.ctrl_size == 0).sum())} none")
    print(out[["pair_id", "fp_size", "ctrl_size", "ctrl_patches", "nbr_before_diff", "fp_nbr_before", "fp_nbr_after", "ctrl_nbr_before",
               "ctrl_nbr_after", "ctrl_median_nbr", "ctrl_amplitude", "ctrl_dist_median_m", "ctrl_dist_max_m"]].to_string(index=False))
    print(f"\n-> {gpkg}\n-> {os.path.join(a.run_dir, 'controls.csv')}")


if __name__ == "__main__":
    main()
