"""Python port of users/fpelletier/SIG:682-fao-disturbance/1.1-bulcdCaller.

Runs BULC-D on pixel time series from the FAO PostGIS database and writes the
results locally.

    export FAO_DB_PASSWORD='...'          # or PGPASSWORD; otherwise you are asked
    python run_bulcd.py                    # the Tangier AOI from the GEE script
    python run_bulcd.py --aoi-id 12        # a study area from aoi_boundaries
    python run_bulcd.py --bbox -5.73 35.72 -5.70 35.74 --target-year 2024 --expectation-years 2022 2023

Outputs go to outputs/<run name>/:
    pixels.csv      every 10 m pixel in the AOI with all metrics
    change.gpkg     layer 'change' (final filtered change, = the GEE export) and
                    layer 'candidates' (all pixels with probCls1 >= 0.5)
    arrays.npz      the time series, used by plot_pixel.py
    params.json     the parameters of the run
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import bulcd_core as core
import db_loader as db
import postprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# --- REGION OF INTEREST (the `geometry` import of the GEE script) ---
DEFAULT_BBOX = (-5.7262642950592, 35.724414469943994, -5.696137723403926, 35.74301694802804)

# --- DEFINE PARAMETERS ---
PARAMS = dict(
    target_year=[2023],
    expectation_years=[2021, 2022],
    first_doy=1,
    last_doy=365,
    sensors=["S2", "L8", "L9"],
    which_index="NBR",
    day_step=2,
    # -- Advanced analysis --
    numerator_factor=0.8,        # sensitivity dictionary
    denominator_factor=0.1,
    nbr_threshold=0.4,           # post-change NBR must be <= this
    amp_threshold=0.15,          # harmonic amplitude must be <= this
    pre_nbr_threshold=0.15,      # pre-change mean NBR must be > this
    timing_threshold=0.3,
    alert_threshold=0.5,         # first_alert_day: first day probCls1 >= this...
    alert_hold_days=60,          # ...and stays there this long (not in GEE; used by the footprints)
    mmu_pixels=3,                # minimum mapping unit: smallest patch of touching changed pixels kept
    mmu_connectivity=8,          # 8 = corners touch too, 4 = edges only
    # Footprints (from true_change_handoff): link changed pixels close in space AND time,
    # give each group one date, keep it only if it dropped more than its surroundings.
    # Written next to the MMU result (column `changed_footprint`, layers `footprints`
    # and `change_footprint`); the MMU column `changed` is unchanged.
    footprints=dict(postprocess.FOOTPRINT_DEFAULTS),
    modality=dict(bimodal=False, constant=True, linear=False, trimodal=False, unimodal=True),
)
# Note: `cloudCover = 70` filtered Landsat scenes by their CLOUD_COVER property in
# GEE. Scene cloud cover is not stored in the database, so only the per-pixel
# QA mask is applied here.


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="lon/lat bounding box")
    g.add_argument("--aoi-id", type=int, help="aoi_id in the aoi_boundaries table")
    g.add_argument("--wkt", help="AOI geometry as WKT in EPSG:4326")
    g.add_argument("--aoi-file", help="vector file (GeoPackage, shapefile...) whose features, merged, are the AOI")
    ap.add_argument("--aoi-layer", help="layer of --aoi-file (default: the first one)")
    ap.add_argument("--target-year", type=int, nargs="+")
    ap.add_argument("--expectation-years", type=int, nargs="+")
    ap.add_argument("--sensors", nargs="+", choices=["S2", "L8", "L9"])
    ap.add_argument("--name", help="run folder name (default: derived from AOI and years)")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    ap.add_argument("--workers", type=int, default=3, help="parallel database connections")
    return ap.parse_args()


def main():
    a = parse_args()
    p = dict(PARAMS)
    if a.target_year:
        p["target_year"] = a.target_year
    if a.expectation_years:
        p["expectation_years"] = a.expectation_years
    if a.sensors:
        p["sensors"] = a.sensors
    if p["which_index"] != "NBR":
        raise NotImplementedError("only NBR is ported")

    if a.aoi_id is not None:
        aoi, aoi_label = db.aoi_sql(aoi_id=a.aoi_id), f"aoi{a.aoi_id}"
    elif a.wkt:
        aoi, aoi_label = db.aoi_sql(wkt=a.wkt), "wkt"
    elif a.aoi_file:
        import geopandas as gpd
        gdf = gpd.read_file(a.aoi_file, layer=a.aoi_layer)
        if gdf.crs is None:
            raise SystemExit(f"{a.aoi_file} has no CRS")
        union = gdf.to_crs(26191).union_all()
        aoi = db.aoi_sql(wkt_26191=union.wkt)
        aoi_label = os.path.splitext(os.path.basename(a.aoi_file))[0]
    else:
        bbox = a.bbox or DEFAULT_BBOX
        aoi, aoi_label = db.aoi_sql(bbox=bbox), "bbox" if a.bbox else "tangier"
    name = a.name or f"{aoi_label}_target{'-'.join(map(str, p['target_year']))}" \
                      f"_exp{'-'.join(map(str, p['expectation_years']))}"
    out_dir = os.path.join(HERE, "outputs", name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    conn = db.connect()
    try:
        cells, pixels, aoi_wkt = db.load_pixels(conn, aoi)
    finally:
        conn.close()
    print(f"AOI: {len(cells)} cells, {int(pixels.in_aoi.sum())} pixels of 10 m inside the AOI")

    cal_exp = [core.bin_calendar(y, p["first_doy"], p["last_doy"], p["day_step"]) for y in p["expectation_years"]]
    cal_tgt = [core.bin_calendar(y, p["first_doy"], p["last_doy"], p["day_step"]) for y in p["target_year"]]
    print("Gathering NBR from the database (cached after the first run)...")
    series = db.build_series(cells, cal_exp + cal_tgt, p["sensors"], workers=a.workers, refresh=a.refresh)
    y_exp = np.concatenate(series[:len(cal_exp)], axis=1).astype(float)
    y_tgt = np.concatenate(series[len(cal_exp):], axis=1).astype(float)

    # Work only on pixels inside the AOI (GEE clips to it).
    keep = pixels.in_aoi.to_numpy()
    px = pixels[keep].reset_index(drop=True)
    y_exp, y_tgt = y_exp[keep], y_tgt[keep]

    # --- RUN THE ALGORITHM ---
    r = core.get_probability_of_decrease(
        y_exp, cal_exp, y_tgt, cal_tgt,
        first_doy=p["first_doy"], modality=p["modality"],
        numerator_factor=p["numerator_factor"], denominator_factor=p["denominator_factor"],
        timing_threshold=p["timing_threshold"])

    # --- POST-CHANGE NBR ---
    post_mean, post_median = core.post_change_mean(y_tgt, r["first_change"])

    # --- Masks applied by the caller ---
    with np.errstate(invalid="ignore"):
        low_nbr = post_mean <= p["nbr_threshold"]
        low_amp = r["amplitude"] <= p["amp_threshold"]
        high_pre = r["expectation_mean"] > p["pre_nbr_threshold"]
    final = r["was_changed"] & low_nbr & low_amp & high_pre

    probs = r["final_probs"]
    res = px[["cell_id", "subpixel_id", "box_id"]].copy()
    res["probCls1"], res["probCls2"], res["probCls3"] = probs[:, 0], probs[:, 1], probs[:, 2]
    res["candidate_changed"] = r["was_changed"]
    res["changed"] = final
    res["julianDay"] = r["julian_day"]
    res["month"] = r["month"]
    res["first_alert_day"] = core.first_sustained_alert(r["prob1_history"], p["alert_threshold"],
                                                        p["alert_hold_days"], p["day_step"])
    res["post_change_nbr_mean"] = post_mean
    res["post_change_nbr_median"] = post_median
    res["amplitude"] = r["amplitude"]
    res["pre_change_nbr_mean"] = r["expectation_mean"]
    res["expectation_sd"] = r["expectation_sd"]
    res["n_obs_expectation"] = np.sum(~np.isnan(y_exp), axis=1)
    res["n_obs_target"] = np.sum(~np.isnan(y_tgt), axis=1)
    res["x_26191"], res["y_26191"] = px.x, px.y

    # --- Minimum mapping unit: drop patches of fewer than mmu_pixels touching pixels ---
    res = postprocess.apply_mmu(res, p["mmu_pixels"], p["mmu_connectivity"])
    res, fps = postprocess.apply_footprints(res, y_tgt, r["target_fit"], **p["footprints"])
    res.to_csv(os.path.join(out_dir, "pixels.csv"), index=False)
    postprocess.write_gpkg(res, out_dir, aoi_wkt)
    postprocess.write_footprint_layers(res, fps, out_dir)

    np.savez_compressed(
        os.path.join(out_dir, "arrays.npz"),
        cell_id=px.cell_id.to_numpy(), subpixel_id=px.subpixel_id.to_numpy(),
        x=px.x.to_numpy(), y=px.y.to_numpy(),
        y_exp=y_exp.astype(np.float32), y_tgt=y_tgt.astype(np.float32),
        fit_exp=r["expect_fit"].astype(np.float32), fit_tgt=r["target_fit"].astype(np.float32),
        doy_exp=np.concatenate([c.nominal_doy for c in cal_exp]),
        doy_tgt=np.concatenate([c.nominal_doy for c in cal_tgt]),
        year_exp=np.concatenate([[c.year] * c.n for c in cal_exp]),
        year_tgt=np.concatenate([[c.year] * c.n for c in cal_tgt]),
        prob1_history=r["prob1_history"].astype(np.float32),
        events=r["events"])
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({**p, "aoi_wkt_26191": aoi_wkt, "cloud_mask_s2": "SCL in (4,5,6) and MSK_CLDPRB < 10",
                   "water_mask": None}, f, indent=2)

    n = len(res)
    print(f"\nDone in {time.time() - t0:.0f} s -> {out_dir}")
    print(f"  pixels in AOI:              {n}")
    print(f"  candidate changed (p>=0.5): {int(res.candidate_changed.sum())} ({res.candidate_changed.mean():.1%})")
    print(f"  changed after masks:        {int(res.changed_before_mmu.sum())} ({res.changed_before_mmu.mean():.1%})")
    print(f"  final changed (MMU >= {p['mmu_pixels']} px): {int(res.changed.sum())} ({res.changed.mean():.1%}) "
          f"in {res.loc[res.changed, 'patch_id'].nunique()} patches")
    if res.changed.any():
        print("  final changes by month:", res[res.changed].month.astype(int).value_counts().sort_index().to_dict())
    acc = fps[fps.fp_status.str.startswith("accepted")]
    print(f"  footprints: {len(fps)} groups, {len(acc)} accepted = {int(res.changed_footprint.sum())} pixels "
          f"({res.changed_footprint.mean():.1%}); by month:", acc.groupby("fp_month").fp_size.sum().to_dict())


if __name__ == "__main__":
    main()
