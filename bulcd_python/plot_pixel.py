"""Harmonic chart for one pixel of a finished run (replaces addHarmonicChartInspector).

    python plot_pixel.py outputs/tangier_target2023_exp2021-2022 --lat 35.733 --lon -5.712
    python plot_pixel.py outputs/tangier_target2023_exp2021-2022 --cell 67332202318367 --sub 4

Top: expectation years, observed NBR vs the harmonic fit.
Middle: target year, observed NBR vs the fit of the expectation years.
Bottom: BULC probability of decrease (probCls1) through the target year.
Saves a PNG in the run folder.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--cell", type=int)
    ap.add_argument("--sub", type=int, default=4)
    a = ap.parse_args()

    A = np.load(os.path.join(a.run_dir, "arrays.npz"))
    if a.cell is not None:
        hit = np.where((A["cell_id"] == a.cell) & (A["subpixel_id"] == a.sub))[0]
        if not hit.size:
            raise SystemExit("pixel not in this run")
        i = int(hit[0])
    elif a.lat is not None and a.lon is not None:
        from pyproj import Transformer
        x, y = Transformer.from_crs(4326, 26191, always_xy=True).transform(a.lon, a.lat)
        d = np.hypot(A["x"] - x, A["y"] - y)
        i = int(np.argmin(d))
        if d[i] > 10:
            raise SystemExit("point is outside this run's AOI")
    else:
        raise SystemExit("give --lat/--lon or --cell")

    res = pd.read_csv(os.path.join(a.run_dir, "pixels.csv"))
    row = res[(res.cell_id == A["cell_id"][i]) & (res.subpixel_id == A["subpixel_id"][i])].iloc[0]

    def day_axis(doy, yr):
        # The last bin's nominal DOY wraps to 0/1 (its end falls in the next
        # year), so plot each bin at its end day counted from the year's first bin.
        x = doy.astype(float).copy()
        for yv in np.unique(yr):
            m = np.where(yr == yv)[0]
            x[m] = np.maximum.accumulate(np.where(np.diff(np.r_[-1, doy[m]]) < 0, doy[m] + 365, doy[m]))
        return x

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for k, (lab, y, fit, doy, yr) in enumerate([
            ("Expectation years vs. fit of expectation", A["y_exp"][i], A["fit_exp"][i],
             day_axis(A["doy_exp"], A["year_exp"]), A["year_exp"]),
            ("Target year vs. fit of expectation", A["y_tgt"][i], A["fit_tgt"][i],
             day_axis(A["doy_tgt"], A["year_tgt"]), A["year_tgt"])]):
        for j, yv in enumerate(np.unique(yr)):
            m = yr == yv
            ax[k].scatter(doy[m], y[m], s=10, label=f"observed {yv}", color=f"C{j}")
        o = np.argsort(doy)
        ax[k].plot(doy[o], fit[o], color="k", lw=2, label="fitted")
        ax[k].set(title=lab, ylabel="NBR", ylim=(-1, 1))
        ax[k].legend(fontsize=8, loc="lower left")
    h = A["prob1_history"][i]
    ax[2].plot(np.r_[0, day_axis(A["doy_tgt"], A["year_tgt"])], h, color="C3")
    ax[2].axhline(0.5, ls="--", color="grey", lw=1)
    ax[2].axhline(0.3, ls=":", color="grey", lw=1)
    if row.candidate_changed:
        ax[2].axvline(row.julianDay, color="C3", ls="--", lw=1)
    ax[2].set(title="Probability of decrease (probCls1)", ylabel="probability", xlabel="Day of year (bin end)", ylim=(0, 1), xlim=(0, 368))
    fig.suptitle(f"cell {row.cell_id} sub {row.subpixel_id}  |  probCls1={row.probCls1:.2f}  "
                 f"changed={bool(row.changed)}  julianDay={row.julianDay}  amplitude={row.amplitude:.3f}", fontsize=9)
    fig.tight_layout()
    out = os.path.join(a.run_dir, f"pixel_{row.cell_id}_{row.subpixel_id}.png")
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
