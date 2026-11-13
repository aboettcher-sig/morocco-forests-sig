"""Continuous BULC-D: one unbroken run over the whole record with a moving baseline.

Not the GEE algorithm (run_bulcd.py stays the GEE port). Inspired by the
true_change handoff's adaptive detector: hold a prior state, accumulate evidence
against it, and learn a new state after a lasting change.

    python bulcd_continuous.py --aoi-file ../box205_landsat_coverage/sentinelTest2.gpkg
    python bulcd_continuous.py --aoi-file ... --report-from 2025-01-01 --name st2_cont_2025
    python bulcd_continuous.py --box-id 162

How it runs, per 10 m pixel, every 2-day bin from --monitor-from to the end of the data:
  1. Baseline: harmonic fit (constant + cos + sin of the day of year) on the
     previous 24 months of the pixel's own bins (baseline_months), refitted on
     the first bin of every month. Bins observed while probCls1 >= 0.3, or with
     Z <= -1, are never used in a baseline (a change must not leak into the
     expected curve). Monitoring starts in 2022 with 2021 as the first baseline.
  2. Regional correction: in every bin, the median departure (NBR - expected)
     of the clear pixels that show no evidence of change (probCls1 < 0.3 and
     NBR less than 0.15 below expected), per 2 km block, is subtracted. Drought or a wet year moves the whole area at
     once and is removed; a local disturbance is not. Baselines are fitted on
     the corrected series too.
  3. BULC update with the GEE Z-score bins and transition matrix:
     Z = 0.8 (corrected NBR - expected) / max(SD, 0.1). No reset on 1 January.
  4. Alert: probCls1 >= 0.5. While alerted, the baseline is frozen. An alert
     that holds 60 days (alert_hold_days) opens an EVENT, dated to the start of
     the alert; its onset is the last rise above 0.3 before it.
  5. An open event ends
       - "new normal" after 12 months (new_normal_months): the baseline is
         refitted on the data since the alert start and the probabilities go
         back to the prior, so a later change can be detected;
       - "recovered" if NBR comes back near the baseline frozen at the alert start
         (smoothed departure < 0.05) and stays there for 120 days. The
         probability alone is not used: once the season brings the expected curve
         close to an already-low NBR, probCls1 can fall while nothing recovered;
       - "open" if the record ends first.
  "recovered" events are kept in events.csv but are not mapped as change.
  6. Caller masks, as in GEE, per event: post-change NBR <= 0.4, harmonic
     amplitude <= 0.15, baseline mean NBR > 0.15.

Outputs (outputs/<name>/):
  events.csv     every event of every pixel (all dates, all statuses)
  pixels.csv     one row per pixel for the report period (first event starting
                 in it), with the MMU and footprint columns, like run_bulcd.py
  change.gpkg    same layers as run_bulcd.py (change, footprints, ...)
  footprints.csv, arrays.npz (series, fits, probabilities), params.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

import bulcd_core as core
import db_loader as db
import postprocess

HERE = os.path.dirname(os.path.abspath(__file__))

PARAMS = dict(
    data_from="2021-01-01",      # first bin of the series (start of the first baseline)
    monitor_from="2022-01-01",   # first bin BULC monitors (the first baseline is then 2021 only)
    sensors=["S2", "L8", "L9"],
    day_step=2,
    baseline_months=24,          # moving baseline window
    min_baseline_obs=20,         # fewer clear bins in the window: keep the previous baseline
    numerator_factor=0.8,
    denominator_factor=0.1,
    regional_correction=True,    # subtract the area-wide departure of each bin (as the handoff's common-mode step)
    regional_block_m=2000,       # ...computed per block of this size (whole AOI if a block has too few pixels)
    regional_min_obs=100,        # clear, undisturbed pixels needed to estimate a block's signal; fewer: use the
                                 # 3 x 3 blocks around it, then the whole run area, then no correction
    water_mask=True,             # Sentinel-2 SCL: sub-pixels classed as water (6) in at least water_min_share of their
    water_min_share=0.10,        # clear observations are never mapped and are left out of the area-wide estimate and
                                 # of the footprint surroundings (stand-in for the GEE water mask)
    regional_clean_months=24,    # pixels alerted (probCls1 >= alert_threshold) within this many months are left out
                                 # of the area-wide estimate, so burn scars and their regrowth do not drive it
    regional_ignore_drop=0.15,   # pixels this far below expected are left out of the area-wide estimate, so a
                                 # fire or clearing covering most of the area is not mistaken for weather
    evidence_threshold=0.3,      # bins observed at or above this probCls1 never enter a baseline
    alert_threshold=0.5,         # alert / baseline frozen
    alert_hold_days=60,          # an alert that holds this long opens an event
    new_normal_months=12,        # open event -> new baseline learned after this long
    recover_days=120,            # open event closed as "recovered" after NBR has been back near the
    recover_departure=0.05,      # pre-change baseline (frozen at the alert start; smoothed departure below
    recover_smoothing=0.1,       # this) for this long. Smoothing: weight of each new observation.
    baseline_exclude_z=-1.0,     # bins with Z at or below this never enter a baseline (no slow absorption)
    nbr_threshold=0.4,           # caller masks, as in GEE
    amp_threshold=0.15,
    pre_nbr_threshold=0.15,
    mmu_pixels=3,
    mmu_connectivity=8,
    footprints=dict(postprocess.FOOTPRINT_DEFAULTS),
)
NAMES = ["constant", "cos", "sin"]


def continuous_calendar(d_from: str, last_start: date, step: int) -> core.Calendar:
    d0 = date.fromisoformat(d_from)
    starts = [d0 + timedelta(days=k) for k in range(0, (last_start - d0).days + 1, step)]
    return core.Calendar(year=d0.year, starts=starts, step=step)


def season_design(starts) -> np.ndarray:
    """[1, cos 2*pi*phase, sin 2*pi*phase] with phase = day of year of the bin centre / year length."""
    ph = []
    for s in starts:
        c = s + timedelta(days=1)
        n = 366 if (c.year % 4 == 0 and (c.year % 100 != 0 or c.year % 400 == 0)) else 365
        ph.append((c - date(c.year, 1, 1)).days / n)
    ph = np.asarray(ph)
    return np.column_stack([np.ones_like(ph), np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)])


class Baseline:
    """Per-pixel harmonic baseline, refitted with batched normal equations."""

    def __init__(self, P):
        self.coef = np.full((P, 3), np.nan)
        self.sd = np.full(P, np.nan)
        self.mean = np.full(P, np.nan)

    def refit(self, rows, y, use, X, lo, hi, base_start, min_obs):
        if rows.size == 0 or hi <= lo:
            return
        Y = y[rows, lo:hi].astype(float)
        W = use[rows, lo:hi] & (np.arange(lo, hi)[None, :] >= base_start[rows, None])
        Y0 = np.where(W, Y, 0.0)
        w = W.astype(float)
        x = X[lo:hi]
        pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        S = w @ np.column_stack([x[:, i] * x[:, j] for i, j in pairs])
        xtx = np.empty((rows.size, 3, 3))
        for k, (i, j) in enumerate(pairs):
            xtx[:, i, j] = xtx[:, j, i] = S[:, k]
        xty = Y0 @ x
        yty = (Y0 * Y0).sum(axis=1)
        n = W.sum(axis=1)
        ok = (n >= max(min_obs, 4)) & (np.abs(np.linalg.det(xtx)) > 1e-9)
        if not ok.any():
            return
        b = np.linalg.solve(xtx[ok], xty[ok][..., None])[..., 0]
        rss = yty[ok] - 2 * np.einsum("pi,pi->p", b, xty[ok]) + np.einsum("pi,pij,pj->p", b, xtx[ok], b)
        r = rows[ok]
        self.coef[r] = b
        self.sd[r] = np.sqrt(np.maximum(rss, 0) / (n[ok] - 1))
        self.mean[r] = Y0[ok].sum(axis=1) / n[ok]


def regional_signal(d, eligible, block, min_obs):
    """Median of d over eligible pixels, per block.

    `block` is (bx, by, ub, inv): block column/row per pixel, the unique blocks (n, 2) and
    each pixel's index into them. A block with fewer than `min_obs` eligible pixels uses the
    3 x 3 blocks around it, then the whole run area; if that is short too, no correction (0).
    """
    bx, by, ub, inv = block
    ok = eligible & np.isfinite(d)
    n = ok.sum()
    if n < min_obs:
        return np.zeros(d.shape[0])
    v, kx, ky = d[ok], bx[ok], by[ok]
    whole = np.median(v)
    k = inv[ok]
    cnt = np.bincount(k, minlength=len(ub))
    vals = np.full(len(ub), whole)
    order = np.argsort(k, kind="stable")
    split = np.split(v[order], np.cumsum(cnt)[:-1])
    for j in range(len(ub)):
        if cnt[j] >= min_obs:
            vals[j] = np.median(split[j])
        elif len(ub) > 1:
            nb = (np.abs(kx - ub[j, 0]) <= 1) & (np.abs(ky - ub[j, 1]) <= 1)
            if nb.sum() >= min_obs:
                vals[j] = np.median(v[nb])
    return vals[inv]


def run_continuous(y, starts, p, xy=None, log=print, water=None):
    """y: (P, T) bin-median NBR; xy: (P, 2) coordinates in m for the regional blocks.

    Returns (events DataFrame, fit (P, T), probCls1 history (P, T), regional signal (P, T) or None).
    `fit` is the smooth seasonal baseline; the expected NBR BULC compared with is fit + regional.
    """
    P, T = y.shape
    y_raw = y
    reg = None
    if p.get("regional_correction"):
        if xy is None:
            bx = by = np.zeros(P, np.int64)
        else:
            bx = np.floor(xy[:, 0] / p["regional_block_m"]).astype(np.int64)
            by = np.floor(xy[:, 1] / p["regional_block_m"]).astype(np.int64)
        ub, inv = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
        block = (bx, by, ub, inv.ravel())
        clean_bins = int(round(p.get("regional_clean_months", 0) * 30.44 / p["day_step"]))
        reg = np.zeros((P, T), np.float32)
        y = y.astype(np.float32).copy()
    X = season_design(starts)
    step = p["day_step"]
    m0 = next(i for i, s in enumerate(starts) if s >= date.fromisoformat(p["monitor_from"]))
    win = int(round(p["baseline_months"] * 30.44 / step))
    hold = p["alert_hold_days"] // step
    nn_bins = int(round(p["new_normal_months"] * 30.44 / step))
    rec_bins = p["recover_days"] // step
    month_start = np.array([i == 0 or starts[i].month != starts[i - 1].month for i in range(T)])

    bp = core.BULCParams()
    k = bp.n_classes
    prior = np.full(k, (1 - bp.initializing_leveler) / k); prior[bp.base_class - 1] += bp.initializing_leveler
    table = np.vstack([np.ones(k), bp.transition_matrix * bp.transition_leveler + (1 - bp.transition_leveler) / k])
    post_min = (1 - bp.posterior_leveler) / k

    use = np.isfinite(y)                         # bins allowed in a baseline
    base = Baseline(P)
    base_start = np.zeros(P, np.int64)           # earliest bin a baseline may use (moves at a new normal)
    probs = np.tile(prior, (P, 1))
    fit = np.full((P, T), np.nan, np.float32)
    hist = np.full((P, T), np.nan, np.float32)

    run_start = np.full(P, -1)                   # first bin of the current alert run (>= 0.5)
    ev_open = np.zeros(P, bool)
    ev_start = np.full(P, -1)
    ev_onset = np.full(P, -1)
    low = np.zeros(P, np.int64)                  # open events: consecutive bins with NBR back near the old baseline
    ewma = np.full(P, np.nan)                    # open events: smoothed departure from the frozen baseline
    reset_at = np.full(P, m0)                    # bin of the last reset (onset search starts here)
    frz = {"coef": np.full((P, 3), np.nan), "sd": np.full(P, np.nan), "mean": np.full(P, np.nan)}
    events = []
    not_water = np.ones(P, bool) if water is None else ~np.asarray(water, bool)
    last_alert = np.full(P, -10**9)              # last bin with probCls1 >= alert_threshold

    def close(rows, t, status):
        for i in rows:
            events.append((i, ev_start[i], ev_onset[i], t, status, *frz["coef"][i], frz["sd"][i], frz["mean"][i]))

    base.refit(np.arange(P), y, use, X, max(m0 - win, 0), m0, base_start, p["min_baseline_obs"])
    if reg is not None:
        # signal of the initial baseline period, then refit the baseline on the corrected series
        f0 = base.coef @ X[:m0].T
        for t in range(m0):
            d0 = y_raw[:, t] - f0[:, t]
            with np.errstate(invalid="ignore"):
                ok0 = (d0 > -p["regional_ignore_drop"]) & not_water
            reg[:, t] = regional_signal(d0, ok0, block, p["regional_min_obs"])
        y[:, :m0] = y_raw[:, :m0] - reg[:, :m0]
        base.refit(np.arange(P), y, use, X, max(m0 - win, 0), m0, base_start, p["min_baseline_obs"])
    with np.errstate(invalid="ignore"):
        fit[:, :m0] = (base.coef @ X[:m0].T).astype(np.float32)

    t0 = time.time()
    for t in range(m0, T):
        alerted = probs[:, 0] >= p["alert_threshold"]
        if month_start[t] and t > m0:
            rows = np.flatnonzero(~ev_open & ~alerted)
            base.refit(rows, y, use, X, max(t - win, 0), t, base_start, p["min_baseline_obs"])

        f = base.coef @ X[t]
        if reg is not None:
            d_raw = y_raw[:, t] - f
            with np.errstate(invalid="ignore"):
                quiet = ((probs[:, 0] < p["evidence_threshold"]) & (d_raw > -p["regional_ignore_drop"])
                         & ~ev_open & (t - last_alert > clean_bins) & not_water)
            reg[:, t] = regional_signal(d_raw, quiet, block, p["regional_min_obs"])
            y[:, t] = y_raw[:, t] - reg[:, t]
        with np.errstate(invalid="ignore"):
            denom = np.maximum(base.sd, p["denominator_factor"])
            z = np.clip((y[:, t] - f) * p["numerator_factor"] / denom, -10, 10)
        ev = core.z_to_bins(z)
        v = ev > 0
        if v.any():
            num = probs[v] * table[ev[v]]
            probs[v] = num / num.sum(axis=1, keepdims=True) * bp.posterior_leveler + post_min
        p1 = probs[:, 0]
        fit[:, t], hist[:, t] = f, p1
        last_alert[p1 >= p["alert_threshold"]] = t
        use[p1 >= p["evidence_threshold"], t] = False
        with np.errstate(invalid="ignore"):
            use[z <= p["baseline_exclude_z"], t] = False

        a = p1 >= p["alert_threshold"]
        # alert runs of pixels without an open event
        closed = ~ev_open
        run_start[closed & a & (run_start < 0)] = t
        run_start[closed & ~a] = -1
        frozen_now = closed & a & (run_start == t)
        frz["coef"][frozen_now] = base.coef[frozen_now]
        frz["sd"][frozen_now] = base.sd[frozen_now]
        frz["mean"][frozen_now] = base.mean[frozen_now]
        opening = np.flatnonzero(closed & (run_start >= 0) & (t - run_start + 1 >= hold))
        for i in opening:
            s = run_start[i]
            h = hist[i, reset_at[i]:s + 1]
            lowpos = np.flatnonzero(h <= p["evidence_threshold"])
            ev_onset[i] = reset_at[i] + (lowpos[-1] + 1 if lowpos.size else 0)
            ev_start[i] = s
        ev_open[opening] = True
        low[opening] = 0
        ewma[opening] = np.nan

        # open events: recovery (NBR back near the frozen pre-change baseline) or new normal
        o = ev_open & ~np.isin(np.arange(P), opening)
        with np.errstate(invalid="ignore"):
            d = frz["coef"] @ X[t] - y[:, t]                 # departure from the pre-change baseline
        upd = o & np.isfinite(d)
        first_obs = upd & np.isnan(ewma)
        ewma[first_obs] = d[first_obs]
        later = upd & ~first_obs
        ewma[later] += p["recover_smoothing"] * (d[later] - ewma[later])
        back = o & (ewma < p["recover_departure"])
        low[back] += 1
        low[o & ~back] = 0
        rec = np.flatnonzero(o & (low >= rec_bins))
        close(rec, t, "recovered")
        ev_open[rec] = False; run_start[rec] = -1; reset_at[rec] = t
        nn = np.flatnonzero(ev_open & (t - ev_start >= nn_bins))
        if nn.size:
            close(nn, t, "new normal")
            for i in nn:
                s = ev_start[i]
                use[i, s:t + 1] = np.isfinite(y[i, s:t + 1])
                base_start[i] = s
            base.refit(nn, y, use, X, int(ev_start[nn].min()), t + 1, base_start, p["min_baseline_obs"])
            probs[nn] = prior
            ev_open[nn] = False; run_start[nn] = -1; reset_at[nn] = t + 1
        if (t - m0) % 90 == 0:
            log(f"  {starts[t]}: {int(ev_open.sum())} open events, {len(events)} closed ({time.time() - t0:.0f} s)")

    # end of record
    close(np.flatnonzero(ev_open), T - 1, "open")
    short = np.flatnonzero(~ev_open & (run_start >= 0))      # alert still on at the end, shorter than the hold
    for i in short:
        s = run_start[i]
        h = hist[i, reset_at[i]:s + 1]
        lowpos = np.flatnonzero(h <= p["evidence_threshold"])
        ev_start[i] = s
        ev_onset[i] = reset_at[i] + (lowpos[-1] + 1 if lowpos.size else 0)
    close(short, T - 1, "open (alert shorter than the hold)")

    cols = ["pix", "start", "onset", "end", "status", "b0", "b1", "b2", "sd", "pre_change_nbr_mean"]
    ev = pd.DataFrame(events, columns=cols)
    ev["amplitude"] = np.hypot(ev.b1, ev.b2)
    post = []
    for i, s in zip(ev.pix, ev.start):
        v = y_raw[i, s:]
        v = v[np.isfinite(v)][:5]
        post.append(v.mean() if v.size else np.nan)
    ev["post_change_nbr_mean"] = post
    ev["probCls1_max"] = [np.nanmax(hist[i, s:e + 1]) for i, s, e in zip(ev.pix, ev.start, ev.end)]
    return ev.sort_values(["pix", "start"]).reset_index(drop=True), fit, hist, reg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    g.add_argument("--aoi-id", type=int)
    g.add_argument("--aoi-file")
    g.add_argument("--box-id", type=int, help="a box of the database grid (grid_30m_morocco.box_id)")
    ap.add_argument("--aoi-layer")
    ap.add_argument("--monitor-from", default=PARAMS["monitor_from"])
    ap.add_argument("--report-from", help="report events whose alert starts on/after this date (default: monitor-from)")
    ap.add_argument("--report-to", help="...and before this date (default: end of the data)")
    ap.add_argument("--name")
    ap.add_argument("--sensors", nargs="+", choices=["S2", "L8", "L9"], help="default: S2 L8 L9")
    ap.add_argument("--max-tile-pixels", type=int, default=300_000,
                    help="larger AOIs are processed in tiles of regional_block_m (2 km), then mapped together")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pre-nbr", type=float, help=f"pre-change NBR mask: keep pixels above this (default {PARAMS['pre_nbr_threshold']})")
    a = ap.parse_args()
    p = {**PARAMS, "monitor_from": a.monitor_from}
    if a.sensors:
        p["sensors"] = a.sensors
    if a.pre_nbr is not None:
        p["pre_nbr_threshold"] = a.pre_nbr

    if a.aoi_id is not None:
        aoi, label = db.aoi_sql(aoi_id=a.aoi_id), f"aoi{a.aoi_id}"
    elif a.box_id is not None:
        aoi = ("(select ST_Union(geom) from grid_30m_morocco where box_id = %s)", (a.box_id,))
        label = f"box{a.box_id}"
    elif a.aoi_file:
        import geopandas as gpd
        gdf = gpd.read_file(a.aoi_file, layer=a.aoi_layer)
        aoi = db.aoi_sql(wkt_26191=gdf.to_crs(26191).union_all().wkt)
        label = os.path.splitext(os.path.basename(a.aoi_file))[0]
    else:
        from run_bulcd import DEFAULT_BBOX
        aoi, label = db.aoi_sql(bbox=a.bbox or DEFAULT_BBOX), "bbox" if a.bbox else "tangier"
    r0 = a.report_from or p["monitor_from"]
    tag = "" if sorted(p["sensors"]) == sorted(PARAMS["sensors"]) else "_" + "".join(p["sensors"]) + "only"
    name = a.name or f"{label}_continuous{tag}_report{r0[:4]}" + (f"-{a.report_to[:4]}" if a.report_to else "-end")
    out = os.path.join(HERE, "outputs", name)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    conn = db.connect()
    try:
        cells, pixels, aoi_wkt = db.load_pixels(conn, aoi)
    finally:
        conn.close()
    cal = continuous_calendar(p["data_from"], date(max(db.DATA_YEARS), 12, 29), p["day_step"])
    print(f"AOI: {len(cells)} cells; {cal.n} bins of {p['day_step']} days, {cal.starts[0]} to {cal.starts[-1]}")
    keep = pixels.in_aoi.to_numpy()
    px = pixels[keep].reset_index(drop=True)
    db.ensure_cached(cells, sorted({d.year for d in cal.starts}), p["sensors"], workers=a.workers)
    starts = np.array(cal.starts, dtype="datetime64[D]")
    if p.get("water_mask"):
        wc = db.water_counts(cells, workers=a.workers)
        wc = px[["cell_id", "subpixel_id"]].merge(wc, how="left", on=["cell_id", "subpixel_id"])
        with np.errstate(invalid="ignore", divide="ignore"):
            px["water_share"] = (wc.n_water / wc.n_clear).to_numpy()
        px["water"] = (px.water_share >= p["water_min_share"]).fillna(False).to_numpy(bool)
        print(f"  water mask: {int(px.water.sum())} of {len(px)} sub-pixels water in >= {p['water_min_share']:.0%} "
              f"of clear Sentinel-2 observations")
    else:
        px["water_share"], px["water"] = np.nan, False

    # Large AOIs: one tile per regional block (cells grouped by centroid), so the
    # area-wide correction is computed exactly as in a single run.
    cxy = cells.assign(cx=(cells.xmin + cells.xmax) / 2, cy=(cells.ymin + cells.ymax) / 2).set_index("cell_id")
    bm = p["regional_block_m"]
    if len(px) > a.max_tile_pixels:
        tkey = np.floor(cxy.cx / bm).astype(np.int64) * 1_000_003 + np.floor(cxy.cy / bm).astype(np.int64)
    else:
        tkey = pd.Series(0, index=cxy.index)
    px_tile = tkey.reindex(px.cell_id.to_numpy()).to_numpy()
    tiles = np.unique(px_tile)
    single = len(tiles) == 1
    if not single:
        os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
    print(f"  {len(px)} pixels inside the AOI, {len(tiles)} tile(s). Running BULC continuously...")
    dep = np.full((len(px), cal.n), np.nan, np.float16)         # expected - observed NBR, as BULC saw it
    evs, tile_index = [], []
    xy_all = px[["x", "y"]].to_numpy(float)
    for k, tk in enumerate(tiles):
        rows = np.flatnonzero(px_tile == tk)
        tc = cells[cells.cell_id.isin(np.unique(px.cell_id.to_numpy()[rows]))].reset_index(drop=True)
        ser = db.build_series(tc, [cal], p["sensors"], workers=a.workers, log=lambda m: None)[0]
        pos = pd.Series(np.arange(len(tc)), index=tc.cell_id.to_numpy()).reindex(px.cell_id.to_numpy()[rows]).to_numpy()
        y = ser[pos * 9 + px.subpixel_id.to_numpy()[rows]]
        del ser
        ev_t, fit, hist, reg = run_continuous(y, cal.starts, p, xy=xy_all[rows], log=print if single else (lambda *m: None),
                                              water=px.water.to_numpy()[rows])
        expected = fit + reg if reg is not None else fit
        dep[rows] = (expected - y).astype(np.float16)
        ev_t["pix"] = rows[ev_t.pix.to_numpy()]
        evs.append(ev_t)
        arr = dict(rows=rows, cell_id=px.cell_id.to_numpy()[rows], subpixel_id=px.subpixel_id.to_numpy()[rows],
                   x=px.x.to_numpy()[rows], y=px.y.to_numpy()[rows], bin_start=starts,
                   nbr=y.astype(np.float16), fit=fit.astype(np.float16), prob1=hist.astype(np.float16),
                   regional=(reg if reg is not None else np.zeros(1)).astype(np.float16))
        if single:
            arr.pop("rows")
            np.savez_compressed(os.path.join(out, "arrays.npz"), **arr)
        else:
            fn = f"tile_{k:03d}.npz"
            np.savez_compressed(os.path.join(out, "tiles", fn), **arr)
            tile_index.append(fn)
            print(f"  tile {k + 1}/{len(tiles)}: {len(rows)} pixels, {len(ev_t)} events ({time.time() - t0:.0f} s)")
        del y, fit, hist, reg, expected
    ev = pd.concat(evs).sort_values(["pix", "start"]).reset_index(drop=True)
    ev.insert(0, "cell_id", px.cell_id.to_numpy()[ev.pix])
    ev.insert(1, "subpixel_id", px.subpixel_id.to_numpy()[ev.pix])
    for c in ("start", "onset", "end"):
        ev[c + "_date"] = starts[ev[c].to_numpy()]
    with np.errstate(invalid="ignore"):
        ev["passes_masks"] = ((ev.post_change_nbr_mean <= p["nbr_threshold"]) & (ev.amplitude <= p["amp_threshold"])
                              & (ev.pre_change_nbr_mean > p["pre_nbr_threshold"]))
    ev.drop(columns=["b0", "b1", "b2"]).to_csv(os.path.join(out, "events.csv"), index=False)

    # --- one row per pixel for the report period ---
    rs = np.datetime64(r0)
    re_ = np.datetime64(a.report_to) if a.report_to else starts[-1] + 1
    inper = ev[(ev.start_date >= rs) & (ev.start_date < re_) & (ev.status != "recovered")]
    first = inper.groupby("pix").head(1).set_index("pix")
    first_ok = inper[inper.passes_masks & ~px.water.to_numpy()[inper.pix.to_numpy()]].groupby("pix").head(1).set_index("pix")
    res = px[["cell_id", "subpixel_id", "box_id"]].copy()
    res["candidate_changed"] = res.index.isin(first.index)          # any alert-event starting in the period
    res["changed"] = res.index.isin(first_ok.index)
    pick = first.copy()
    pick.loc[first_ok.index] = first_ok                              # prefer the first event that passes the masks
    pick = pick.reindex(res.index)
    res["n_events"] = inper.groupby("pix").size().reindex(res.index).fillna(0).astype(int)
    res["event_status"] = pick.status
    res["alert_date"] = pick.start_date
    res["onset_date"] = pick.onset_date
    res["end_date"] = pick.end_date
    res["first_alert_day"] = (pick.start + 1) * p["day_step"]       # bin day, the scale the footprints use
    res["julianDay"] = (pick.onset + 1) * p["day_step"]
    res["alert_year"] = pd.to_datetime(pick.start_date).dt.year.astype("Int64")
    res["month"] = pd.to_datetime(pick.start_date).dt.month
    res["onset_year"] = pd.to_datetime(pick.onset_date).dt.year.astype("Int64")
    res["probCls1_max"] = pick.probCls1_max
    res["post_change_nbr_mean"] = pick.post_change_nbr_mean
    res["amplitude"] = pick.amplitude
    res["pre_change_nbr_mean"] = pick.pre_change_nbr_mean
    res["water_share"] = px.water_share.round(3)
    res["water"] = px.water.to_numpy()
    res["x_26191"], res["y_26191"] = px.x, px.y

    res = postprocess.apply_mmu(res, p["mmu_pixels"], p["mmu_connectivity"])
    origin = starts[0]
    res, fps = postprocess.apply_footprints(res, dep, None, origin=origin, **p["footprints"])
    res.to_csv(os.path.join(out, "pixels.csv"), index=False)
    postprocess.write_gpkg(res, out, aoi_wkt)
    postprocess.write_footprint_layers(res, fps, out)
    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump({**p, "mode": "continuous", "origin": str(origin), "report_from": r0,
                   "report_to": a.report_to, "aoi_wkt_26191": aoi_wkt, "tiles": tile_index,
                   "cloud_mask_s2": "SCL in (4,5,6) and MSK_CLDPRB < 10"}, f, indent=2)

    acc = fps[fps.fp_status.str.startswith("accepted")] if len(fps) else fps
    print(f"\nDone in {time.time() - t0:.0f} s -> {out}")
    print(f"  events (all dates): {len(ev)}; by status: {ev.status.value_counts().to_dict()}")
    print(f"  report period {r0} to {a.report_to or 'end'}: {int(res.candidate_changed.sum())} pixels with an event, "
          f"{int(res.changed.sum())} after masks and MMU, {len(acc)} footprints kept = {int(res.changed_footprint.sum())} px")
    if len(acc):
        print("  footprint pixels by month:", acc.groupby("fp_date_month").fp_size.sum().to_dict())


if __name__ == "__main__":
    main()
