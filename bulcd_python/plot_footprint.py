"""Diagnostic plot of one footprint, in the style of the true_change handoff's
footprint_diagnostics.png.

    python plot_footprint.py outputs/<run> 37
    python plot_footprint.py outputs/<run> 37 12 5      # several footprints, one row each

Per footprint, three panels share the time axis:
  1. raw NBR acquisitions of the members (Sentinel-2 per sub-pixel, Landsat per
     30 m cell), the footprint monthly median, and the expected NBR (harmonic fit
     of the expectation years, median over members);
  2. departure from the expected NBR (fit - observed, 30-day running median):
     faint lines = members, thick = member median, grey = pixels within ring_m;
  3. BULC-D probability of decrease (probCls1) in the target year(s).
Red line = onset, red band = range of member dates, black dashed = footprint date,
blue band = the window used for the contrast test.
Writes outputs/<run>/footprint_<ids>.png
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GREY, ORANGE, PURPLE, LIGHT, DARK, RED, BLUE = "#aeb8b1", "#d97706", "#5d3a9b", "#7aa58e", "#174f38", "#b42318", "#2563eb"


def bin_dates(years):
    out, k = [], {}
    for y in years:
        k[y] = k.get(y, -1) + 1
        out.append(np.datetime64(date(int(y), 1, 1) + timedelta(2 * k[y] + 1)))
    return np.array(out)


def raw_obs(cells, subs, years):
    """Raw (date, nbr) for S2 member sub-pixels and Landsat member cells, from the cache."""
    batches = json.load(open(os.path.join(CACHE, "index.json")))["batches"]
    want = set(zip(cells, subs))
    out = {"S2": [], "L": []}
    for kind in out:
        for y in years:
            todo = set(cells)
            for b in reversed(batches):                       # newest batch wins, as in build_series
                if b["kind"] != kind or str(y) not in b["files"]:
                    continue
                hit = todo & set(b["cells"])
                if not hit:
                    continue
                o = np.load(os.path.join(CACHE, b["files"][str(y)]))["obs"]
                o = np.unique(o[np.isin(o["cell_id"], list(hit))])
                if kind == "S2":
                    o = o[[(c, s) in want for c, s in zip(o["cell_id"], o["sub"])]]
                o = o[np.isfinite(o["nbr"]) & (o["nbr"] <= 0.8)]
                d = np.datetime64(f"{y}-01-01") + o["day"].astype("timedelta64[D]")
                out[kind].append(pd.DataFrame({"date": d, "nbr": o["nbr"]}))
                todo -= hit
    return {k: (pd.concat(v) if v else pd.DataFrame(columns=["date", "nbr"])) for k, v in out.items()}


def running_median(a, bins=15, min_obs=3):
    return pd.DataFrame(a.T).rolling(bins, center=True, min_periods=min_obs).median().to_numpy().T


def jd_date(year, jd):
    return np.datetime64(date(year, 1, 1) + timedelta(int(jd)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("fp_ids", type=int, nargs="+")
    a = ap.parse_args()

    res = pd.read_csv(os.path.join(a.run_dir, "pixels.csv"))
    fps = pd.read_csv(os.path.join(a.run_dir, "footprints.csv")).set_index("fp_id")
    params = json.load(open(os.path.join(a.run_dir, "params.json")))
    tiles = params.get("tiles") or []
    if tiles:   # large continuous run: series stored per tile, read only the rows a footprint needs
        tdir = os.path.join(a.run_dir, "tiles")
        tile_rows = {fn: np.load(os.path.join(tdir, fn))["rows"] for fn in tiles}
        arr = np.load(os.path.join(tdir, tiles[0]))
    else:
        arr = np.load(os.path.join(a.run_dir, "arrays.npz"))
    ring_m = params.get("footprints", {}).get("ring_m", 300.0)
    window = params.get("footprints", {}).get("window_days", 90)
    cont = params.get("mode") == "continuous"
    if cont:
        # continuous run: one series over the whole record; day d = bin starting origin + d - 2
        if not tiles:
            y_all, fit_all, hist = (arr[k].astype(float) for k in ("nbr", "fit", "prob1"))
            reg_all = arr["regional"].astype(float) if arr["regional"].ndim == 2 else np.zeros_like(y_all)
        t_all = arr["bin_start"] + np.timedelta64(1, "D")
        t_hist = t_all
        origin = np.datetime64(params["origin"])
        to_date = lambda d: origin + np.timedelta64(int(d) - 2, "D")
        years = sorted(set(arr["bin_start"].astype("datetime64[Y]").astype(int) + 1970))
        hist_title = "BULC-D probability of decrease, continuous (no reset on 1 January)"
    else:
        tgt_year = int(params["target_year"][0])
        y_all = np.concatenate([arr["y_exp"], arr["y_tgt"]], axis=1).astype(float)
        fit_all = np.concatenate([arr["fit_exp"], arr["fit_tgt"]], axis=1).astype(float)
        t_all = bin_dates(np.concatenate([arr["year_exp"], arr["year_tgt"]]))
        hist = arr["prob1_history"].astype(float)
        t_hist = np.array([np.datetime64(f"{tgt_year}-01-01") + np.timedelta64(2 * j, "D") for j in range(hist.shape[1])])
        to_date = lambda d: jd_date(tgt_year, d)
        years = sorted(set(arr["year_exp"].tolist() + arr["year_tgt"].tolist()))
        hist_title = f"BULC-D probability of decrease, {tgt_year}"
        reg_all = np.zeros_like(y_all)
    xy = res[["x_26191", "y_26191"]].to_numpy(float)
    tree = cKDTree(xy)

    def fetch(idx):
        """(nbr, fit, prob1, regional) for global pixel rows idx, in that order."""
        if not tiles:
            return y_all[idx], fit_all[idx], hist[idx], reg_all[idx]
        out = [np.full((len(idx), len(t_all)), np.nan) for _ in range(4)]
        where = pd.Series(np.arange(len(idx)), index=idx)
        for fn, rws in tile_rows.items():
            hit = np.flatnonzero(np.isin(rws, idx))
            if hit.size == 0:
                continue
            z = np.load(os.path.join(tdir, fn))
            dst = where.loc[rws[hit]].to_numpy()
            for o, key in zip(out, ("nbr", "fit", "prob1", "regional")):
                if key == "regional" and z[key].ndim != 2:
                    o[dst] = 0.0
                else:
                    o[dst] = z[key][hit].astype(float)
        return out

    n = len(a.fp_ids)
    fig, axes = plt.subplots(3, n, figsize=(15 * n ** 0 if n == 1 else 9 * n, 11), sharex=True, squeeze=False,
                             height_ratios=[3, 2.2, 1.6])
    for col, fid in enumerate(a.fp_ids):
        if fid not in fps.index:
            raise SystemExit(f"fp_id {fid} not in footprints.csv")
        f = fps.loc[fid]
        m = np.flatnonzero(res.fp_id.to_numpy() == fid)
        near = np.unique(np.concatenate(tree.query_ball_point(xy[m], ring_m)))
        ring = near[res.fp_id.to_numpy()[near] != fid]
        obs = raw_obs(res.cell_id.to_numpy()[m].tolist(), res.subpixel_id.to_numpy()[m].tolist(), years)
        ax1, ax2, ax3 = axes[:, col]
        Y, F, H, R = fetch(np.concatenate([m, ring]))
        ml, rl = np.arange(len(m)), np.arange(len(m), len(m) + len(ring))

        # 1. raw acquisitions, monthly median, expected NBR
        ax1.scatter(obs["S2"].date, obs["S2"].nbr, s=4, color=GREY, alpha=.35, lw=0,
                    label=f"Sentinel-2 acquisitions ({len(m)} member pixels)")
        if len(obs["L"]):
            ax1.scatter(obs["L"].date, obs["L"].nbr, s=9, color="#4b5563", alpha=.55, marker="s", lw=0,
                        label=f"Landsat acquisitions ({res.cell_id.iloc[m].nunique()} member cells)")
        allobs = pd.concat([obs["S2"], obs["L"]])
        mon = allobs.set_index(pd.to_datetime(allobs.date)).nbr.resample("MS").median()
        ax1.plot(mon.index + pd.Timedelta(days=14), mon.values, color=ORANGE, lw=1.3, label="footprint monthly median")
        for v in F[ml]:
            ax1.plot(t_all, v, color=PURPLE, alpha=.08, lw=.6)
        ax1.plot(t_all, np.median(F[ml], axis=0), color=PURPLE, lw=2, label="expected NBR (harmonic fit, median)")
        ax1.set_ylabel("NBR")
        ax1.set_title(f"fp {fid} · {int(f.fp_size)} pixels · {f.fp_status} · raw and monthly data", fontsize=10)

        # 2. departure from expected
        with np.errstate(all="ignore"):
            dep_m = running_median(F[ml] + R[ml] - Y[ml])      # departure as BULC saw it
            dep_r = running_median(F[rl] + R[rl] - Y[rl])
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med_m, med_r = np.nanmedian(dep_m, axis=0), np.nanmedian(dep_r, axis=0)
        for v in dep_m:
            ax2.plot(t_all, v, color=LIGHT, alpha=.25, lw=.7)
        ax2.plot(t_all, med_r, color="#6b7280", lw=1.8, ls="--", label=f"surroundings, median ({len(ring):,} px within {ring_m:.0f} m)")
        ax2.plot(t_all, med_m, color=DARK, lw=2.1, label="members, median")
        ax2.axhline(0, color="#888", lw=.8)
        ax2.set_ylabel("expected − observed NBR\n(30-day running median)" + ("\narea-wide signal removed" if cont else ""))
        ax2.set_title(f"departure {f.fp_core_drop:.3f}; surroundings {f.fp_ring_drop:.3f}; "
                      f"contrast {f.fp_contrast:.3f} (≥ 0.08 to keep)", fontsize=10)

        # 3. probability of decrease
        for v in H[ml]:
            ax3.plot(t_hist, v, color=LIGHT, alpha=.3, lw=.7)
        ax3.plot(t_hist, np.median(H[ml], axis=0), color=DARK, lw=2.1, label="members, median")
        for yv, lab in ((0.5, "0.5 candidate"), (0.3, "0.3 timing")):
            ax3.axhline(yv, color="#888", lw=.8, ls=":")
            ax3.text(t_hist[np.flatnonzero(np.isfinite(H[0]))[0]] - np.timedelta64(10, "D"), yv, lab, ha="right", va="center", fontsize=7.5, color="#555")
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("probCls1")
        ax3.set_title(hist_title, fontsize=10)

        d_fp = to_date(f.fp_julianDay)
        for ax in (ax1, ax2, ax3):
            ax.axvspan(to_date(f.fp_date_min), to_date(f.fp_date_max), color="#ef4444", alpha=.08,
                       label="range of member dates" if ax is ax1 else None)
            if np.isfinite(f.fp_onset):
                ax.axvline(to_date(f.fp_onset), color=RED, lw=1.5, label="onset" if ax is ax1 else None)
            ax.axvline(d_fp, color="#111827", ls="--", lw=1.1, label="footprint date" if ax is ax1 else None)
            ax.grid(alpha=.17)
        ax2.axvspan(d_fp, d_fp + np.timedelta64(window, "D"), color=BLUE, alpha=.08, label=f"{window}-day contrast window")
        ax1.legend(fontsize=7.5, ncol=3, loc="lower left")
        ax2.legend(fontsize=7.5, ncol=3, loc="upper left")
        ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax3.set_xlim(t_all[0] - np.timedelta64(5, "D"), t_all[-1] + np.timedelta64(5, "D"))
        info = (f"onset {to_date(f.fp_onset)}" if np.isfinite(f.fp_onset) else "no onset") + \
               f" · footprint date {d_fp} · member dates {to_date(f.fp_date_min)} to {to_date(f.fp_date_max)}"
        ax3.annotate(info, (0, -0.42), xycoords="axes fraction", fontsize=8.5, color="#333")

    fig.suptitle("BULC-D footprint diagnostics: acquisitions, expected NBR, departure, and probability of decrease", fontsize=12)
    fig.tight_layout(rect=(0, 0.01, 1, .97))
    out = os.path.join(a.run_dir, f"footprint_{'_'.join(map(str, a.fp_ids))}.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
