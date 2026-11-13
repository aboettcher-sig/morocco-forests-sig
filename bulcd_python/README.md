# BULC-D in Python (port of `682-fao-disturbance/1.1-bulcdCaller`)

This folder runs the Earth Engine BULC-D disturbance workflow on your own computer, using the pixel time series in the FAO PostGIS database (`geo_db`) instead of Earth Engine collections.

## Run

```bash
conda activate claude_code          # any env with numpy, pandas, psycopg2, shapely, geopandas, matplotlib, pyproj
# connection settings: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER (and optionally
# POSTGRES_PASSWORD) in the environment or in a .env file here or at the repo root (never committed)
export FAO_DB_PASSWORD='...'        # or PGPASSWORD / POSTGRES_PASSWORD; if none is set, the script asks for it
python run_bulcd.py                 # the Tangier AOI and the parameters of the GEE script
python run_bulcd.py --aoi-id 12     # any study area from aoi_boundaries
python run_bulcd.py --bbox W S E N --target-year 2024 --expectation-years 2022 2023
python plot_pixel.py outputs/<run> --lat 35.733 --lon -5.712     # replaces the click-to-chart inspector
```

The first run of an AOI downloads its observations, about 3 minutes per year for the Tangier AOI (6,438 cells). The cache in `cache/` holds one file per batch of cells, per sensor, per calendar year. Any later run that needs a year already downloaded for the same AOI reads it from disk, whatever the combination of target and expectation years. Runs with other thresholds start in seconds. Use `--refresh` to download again.

The parameters (`numerator_factor`, `nbr_threshold`, `amp_threshold`, and so on) are in the `PARAMS` block at the top of `run_bulcd.py`, with the same names and values as the GEE caller.

## Outputs (`outputs/<run>/`)

| File | Contents |
|---|---|
| `change.gpkg` | Layer `change`: final changed pixels, the equivalent of the GEE export. Layer `candidates`: every pixel with probCls1 ≥ 0.5. Layer `aoi`. Polygons are 10 m pixels in EPSG:26191. |
| `pixels.csv` | Every 10 m pixel in the AOI: probCls1–3, `candidate_changed`, `changed`, `julianDay`, `month`, post-change NBR, amplitude, pre-change NBR, observation counts. |
| `arrays.npz` | Time series and fits, used by `plot_pixel.py`. |
| `params.json` | The parameters of the run. |

A pixel is `(cell_id, subpixel_id)`: the 9 Sentinel-2 sub-pixels of a 30 m grid cell, numbered row by row from the north-west corner. Landsat has one value per cell, which all 9 sub-pixels share.

## Files

| Python | Replaces (Earth Engine) |
|---|---|
| `run_bulcd.py` | `fpelletier/SIG:682-fao-disturbance/1.1-bulcdCaller` |
| `bulcd_core.get_probability_of_decrease` | `fpelletier/SIG:pc643/0.0-bulcdDependencies` → `getProbabilityOfDecrease` |
| `db_loader.build_series` | `flaviecpelletier/SIG:pc643/1001.06-515.gatherCollections27b` (S2, L8, L9, NBR) |
| `bulcd_core.organize_bulcd_inputs` | `flaviecpelletier/SIG:pc643/1002.03-…organizeBULCD_Inputs` |
| `bulcd_core.fit_harmonic` / `harmonic_design` | `alemlakes/CommonCode2:502.7-Harmonics/502.7-1h5-HarmonicFunctions` |
| `bulcd_core.z_to_bins` | `alemlakes/CommonCode:502.8-Binning/502.8-Binning-5` |
| `bulcd_core.run_bulc` | `alemlakes/r-2909-BULC-Releases:…/BULC-Minimal-Module-107` + `r-2903-Dev:…/6002.B2-BULCD-Module`, parameters from `6003.3c-BULC-AdvancedParameters` |
| `bulcd_core.first_evidence_of_change` | `flaviecpelletier/BULC-D:100-theFunction/104.10-timingDisturbance` |
| `bulcd_core.post_change_mean` | `flaviecpelletier/SIG:pc643/functions/104.20-afn_toDnbr` |
| `plot_pixel.py` | `addHarmonicChartInspector` |

`test_core.py` checks the date bins, Z-score bins, the Bayesian update, the harmonic fit and the timing against values computed by hand.

## How the method runs

1. **Gather:** for every year, one value per pixel per 2-day bin. This is the median NBR of all clear S2, L8 and L9 observations in the bin, with NBR > 0.8 masked.
2. **Expectation:** fit `NBR = b0 + b1·cos(2πt) + b2·sin(2πt)` per pixel on the expectation years, where t is the bin end's day of year / 365. The fit gives the amplitude √(b1² + b2²), the SD of its residuals, and the mean NBR.
3. **Target:** Z = 0.8 · (observed − fitted) / max(SD, 0.1), clamped to ±10. Z is then cut into 10 bins at −2, −1.5, …, 2.
4. **BULC:** start every pixel at P(decrease, stable, increase) = (0.1, 0.8, 0.1). Each bin updates the probabilities through the 10×3 transition matrix, damped by 0.7 (transitions) and 0.9 (posterior).
5. **Change:** candidate where P(decrease) ≥ 0.5 at the end of the year. `julianDay` = 2 × the event at which P(decrease) last rose above 0.3.
6. **Caller masks:** keep a candidate only if the mean NBR of the first 5 clear bins after the change is ≤ 0.4, the amplitude is ≤ 0.15, and the pre-change mean NBR is > 0.15.
7. **Minimum mapping unit (added in Python, not in the GEE script):** group the changed pixels into patches of touching pixels, where corners count as touching (8-connectivity), and drop patches smaller than 3 pixels. Change this with `mmu_pixels` / `mmu_connectivity` in `PARAMS`. To re-apply it to a finished run without downloading again, use `python postprocess.py outputs/<run> --min-pixels 5 --connectivity 4`. In `pixels.csv`, `changed` is the result after the MMU, `changed_before_mmu` the result before it, and `patch_id` / `patch_size` describe each pixel's patch. `change.gpkg` gets a `change_before_mmu` layer as well.

## Differences from the Earth Engine version

Where the data allow, these were chosen to stay as close to GEE as possible.

* **Sentinel-2 clouds:** GEE used Cloud Score+ (`cs ≥ 0.6`), which is not in the database. Here SCL ∈ {4, 5, 6} and `MSK_CLDPRB` < 10 is used instead, the same rule as `point_timeseries.py`.
* **Landsat scene filter:** GEE dropped Landsat scenes with `CLOUD_COVER ≥ 70`. Scene cloud cover is not stored in the database, so only the per-pixel QA mask is applied (bits 0–4 and `QA_RADSAT`). Landsat-heavy runs can therefore keep a few more observations.
* **Water mask:** GEE masked JRC and WorldCover water. Those layers aren't in the database. The annual mode skips the mask; the continuous mode uses Sentinel-2 SCL instead: a sub-pixel classed as water (SCL 6) in at least 10% of its clear observations (2021-2025) is never mapped, and is left out of the area-wide correction and of the footprint surroundings (`water_share` / `water` columns, `water` layer). The counts are cached in `cache/water/`.
* **Footprint links:** changed pixels join the same footprint only if they touch (diagonals included, link distance 14.2 m); no gaps are bridged.
* **Grid:** GEE worked on its own 10 m pixels. Here the pixels are the database's sub-pixels, and Landsat comes from the cell centroid, so values at pixel edges can differ slightly.
* **Median of 2 values:** numpy takes the mean of the two middle values. GEE's median reducer may differ slightly.
* **`probCls1` in the export:** the GEE script casts `probCls1` to `uint16`, which turns every probability into 0 (or 1). Here it stays a float.
* **`changeType`:** the GEE cast mentions a `changeType` band, but the caller never adds it (`getChangeType` is not called). It is not computed here either.

## GEE quirks reproduced on purpose

Change these only if you also change them in GEE, otherwise the two versions won't match.

* A year's last bin starts on Dec 31 (Dec 30 in leap years) and runs 2 days into the next year. Its `nominalDOY` is the 0-based day of year of the bin end, so it is 1 (or 0), not 366. An observation on Jan 1 can therefore count in the last bin of one year and the first bin of the next.
* `afn_toDnbr` compares the 0-based bin index with the 1-based event number, so the post-change mean starts one bin after the bin that triggered the change.
* `julianDay` is `2 × event number` (`seqImage`), not the true calendar date of the bin.

## Continuous mode (`bulcd_continuous.py`, not in GEE)

`run_bulcd.py` reproduces the GEE caller: one calendar year at a time, a fixed baseline, and the state on 31 December. `bulcd_continuous.py` runs BULC once over the whole record instead, so a change in late December is not split across two years and a disturbed pixel is not flagged again every year.

```bash
python bulcd_continuous.py --aoi-file ../box205_landsat_coverage/sentinelTest2.gpkg                            # map every event from 2022
python bulcd_continuous.py --aoi-file ../box205_landsat_coverage/sentinelTest2.gpkg --report-from 2025-01-01   # only events starting in 2025
python plot_footprint.py outputs/sentinelTest2_continuous_report2025-end 87
```

* **Moving baseline:** monitoring starts in January 2022, with 2021 as the first baseline. The harmonic curve is refitted every month from the pixel's previous 24 months (12 at the start). It is frozen while probCls1 ≥ 0.5, and bins seen while probCls1 ≥ 0.3 or with Z ≤ −1 never enter a baseline.
* **Area-wide correction:** in every 2-day bin, the median departure of the clean pixels is removed, so drought, a wet year or a hazy scene doesn't trigger alerts everywhere at once. Clean = not in an open event, not alerted (probCls1 ≥ 0.5) in the last 24 months, and not more than 0.15 below expected, so a large fire, a burn scar or its regrowth is not taken for weather. It is computed per 2 km block; a block with fewer than 100 clean pixels uses the 3 × 3 blocks around it, then the whole run area, then no correction.
* **Footprints with disturbed surroundings:** when fewer than 10 clean pixels lie within 300 m, the footprint is compared with the nearest clean pixels within 600, 1000 or 1500 m. If none are found it gets the status `review: no clean surroundings` and is not mapped as change (the distance used is in `fp_ring_m`).
* **Events:** an alert that holds 60 days opens an event dated to the alert start. After 12 months the new state is learned as the baseline and the probabilities reset, so a second change can be detected. An event is closed as `recovered`, and not mapped, only when NBR comes back near the baseline frozen at the alert start (smoothed departure < 0.05) for 120 days.
* **Outputs:** `events.csv` (every event, with status), plus the same `pixels.csv`, `change.gpkg` and `footprints.csv` as the yearly runs, for the report period. Footprint dates include the year (`fp_date`, `fp_onset_date`).
* **Settings:** `PARAMS` at the top of `bulcd_continuous.py`. The masks, the MMU and the footprints are the same as in the yearly runs.
