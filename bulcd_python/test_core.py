"""Checks of bulcd_core against hand-computed GEE behaviour.  Run: python -m pytest test_core.py"""
from datetime import date

import numpy as np

import bulcd_core as core


def test_calendar_matches_gee_sequence():
    c = core.bin_calendar(2023)               # non-leap
    assert c.n == 183
    assert c.starts[0] == date(2023, 1, 1) and c.starts[-1] == date(2023, 12, 31)
    assert c.nominal_doy[0] == 2              # end = Jan 3 -> 0-based DOY 2
    assert c.nominal_doy[-1] == 1             # end = Jan 2 2024
    leap = core.bin_calendar(2024)
    assert leap.n == 183 and leap.starts[-1] == date(2024, 12, 30)
    d = np.array(["2023-01-01", "2023-01-02", "2023-01-03", "2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    assert c.bin_index(d).tolist() == [0, 0, 1, 182, -1]


def test_z_bins_edges():
    z = np.array([-10, -2, -1.99, 0, 0.01, 2, 2.01, np.nan])
    assert core.z_to_bins(z).tolist() == [1, 1, 2, 5, 6, 9, 10, 0]


def test_bulc_single_update():
    probs, hist = core.run_bulc(np.array([[1]]))
    prior = np.array([0.1, 0.8, 0.1])
    like = np.array([0.83, 0.08, 0.08]) * 0.7 + 0.1
    post = prior * like / (prior @ like) * 0.9 + 0.1 / 3
    np.testing.assert_allclose(probs[0], post)
    np.testing.assert_allclose(hist[0], [0.1, post[0]])


def test_masked_event_leaves_probs_unchanged():
    probs, hist = core.run_bulc(np.array([[0, 0]]))
    np.testing.assert_allclose(probs[0], [0.1, 0.8, 0.1])


def _synthetic(drop_bin=None, n_pix=1, seed=0):
    rng = np.random.default_rng(seed)
    ce = [core.bin_calendar(2021), core.bin_calendar(2022)]
    ct = [core.bin_calendar(2023)]
    t_e = np.concatenate([c.frac_doy for c in ce]); t_t = ct[0].frac_doy
    season = lambda t: 0.5 + 0.1 * np.cos(2 * np.pi * t)
    ye = season(t_e) + rng.normal(0, 0.02, (n_pix, t_e.size))
    yt = season(t_t) + rng.normal(0, 0.02, (n_pix, t_t.size))
    ye[:, rng.random(t_e.size) < 0.6] = np.nan           # cloud gaps
    yt[:, rng.random(t_t.size) < 0.6] = np.nan
    if drop_bin is not None:
        yt[:, drop_bin:] -= 0.4                             # abrupt NBR drop
    return ye, ce, yt, ct


def test_stable_pixel_not_changed():
    ye, ce, yt, ct = _synthetic()
    r = core.get_probability_of_decrease(ye, ce, yt, ct, numerator_factor=0.8, denominator_factor=0.1)
    np.testing.assert_allclose(r["amplitude"], 0.1, atol=0.02)
    assert not r["was_changed"][0]


def test_drop_detected_and_timed():
    ye, ce, yt, ct = _synthetic(drop_bin=100)
    r = core.get_probability_of_decrease(ye, ce, yt, ct, numerator_factor=0.8, denominator_factor=0.1)
    assert r["was_changed"][0]
    h = r["prob1_history"][0]
    crossed = int(np.argmax(h > 0.3))          # history column where probCls1 first exceeds 0.3
    assert np.all(h[crossed:] > 0.3)
    assert crossed - 1 >= 100                  # the event (bin crossed-1) is after the drop
    assert r["first_change"][0] == crossed     # 1-based index of the last layer <= 0.3
    assert r["julian_day"][0] == 2 * crossed
    mean, _ = core.post_change_mean(yt, r["first_change"])
    after = yt[0, int(r["first_change"][0]):]
    np.testing.assert_allclose(mean[0], np.mean(after[~np.isnan(after)][:5]))


def test_harmonic_fit_recovers_coefficients():
    t = core.bin_calendar(2021).frac_doy
    X = core.harmonic_design(t, ["constant", "cos", "sin"])
    y = (X @ np.array([0.4, 0.05, -0.08]))[None, :]
    np.testing.assert_allclose(core.fit_harmonic(y, X)[0], [0.4, 0.05, -0.08], atol=1e-9)


def test_mmu_patches():
    import pandas as pd
    import postprocess
    # 5x5 grid of 10 m pixels; changed: an L of 3 (kept), a diagonal pair (2, dropped), a single (dropped)
    rows, cols = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    res = pd.DataFrame({"x_26191": cols.ravel() * 10.0 + 5, "y_26191": 100 - rows.ravel() * 10.0, "changed": False})
    on = {(0, 0), (1, 0), (1, 1),   # L-shape, 3 px
          (3, 3), (4, 4),           # diagonal pair
          (0, 4)}                   # single
    res["changed"] = [(r, c) in on for r, c in zip(rows.ravel(), cols.ravel())]
    out8 = postprocess.apply_mmu(res, 3, 8)
    kept = {(r, c) for r, c, k in zip(rows.ravel(), cols.ravel(), out8.changed) if k}
    assert kept == {(0, 0), (1, 0), (1, 1)}
    out8b = postprocess.apply_mmu(res, 2, 8)          # diagonal pair touches with 8-connectivity
    assert out8b.changed.sum() == 5
    out4 = postprocess.apply_mmu(res, 2, 4)           # ...but not with 4-connectivity
    assert out4.changed.sum() == 3
    again = postprocess.apply_mmu(out8, 2, 8)         # re-applying starts from changed_before_mmu
    assert again.changed.sum() == 5


def test_footprints_split_by_date_and_contrast():
    import pandas as pd
    import postprocess
    # 20x20 grid of 10 m pixels, target series of 183 bins with fit = 0.5 everywhere
    n = 20
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    res = pd.DataFrame({"x_26191": cols.ravel() * 10.0 + 5, "y_26191": 1000 - rows.ravel() * 10.0})
    fit = np.full((n * n, 183), 0.5)
    y = fit.copy()
    jd = np.full(n * n, np.nan)
    a = (rows.ravel() < 3) & (cols.ravel() < 3)                       # 9 px dropping in March
    b = (rows.ravel() < 3) & (cols.ravel() >= 3) & (cols.ravel() < 6)  # 9 touching px dropping in October
    jd[a], jd[b] = 60, 280
    y[a, 30:] -= 0.3
    y[b, 140:] -= 0.3
    res["changed"] = a | b
    res["julianDay"] = jd
    out, fps = postprocess.apply_footprints(res, y, fit)
    assert len(fps) == 2 and (fps.fp_status == "accepted").all()      # touching, but 7 months apart
    assert sorted(fps.fp_julianDay) == [60, 280]
    assert out.changed_footprint.sum() == 18
    # the whole area drops at the same time as footprint a -> not stronger than its surroundings
    y2 = y.copy()
    y2[~(a | b), 30:] -= 0.3
    _, fps2 = postprocess.apply_footprints(res, y2, fit)
    assert fps2.set_index("fp_julianDay").loc[60, "fp_status"] == "rejected: no stronger than surroundings"


def test_footprint_surroundings_burned_too():
    import pandas as pd
    import postprocess
    # a 3x3 footprint in the middle of a 21x21 area; every other pixel also dropped and is a BULC-D candidate
    n = 21
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    res = pd.DataFrame({"x_26191": cols.ravel() * 10.0 + 5, "y_26191": 1000 - rows.ravel() * 10.0})
    core = (abs(rows.ravel() - 10) <= 1) & (abs(cols.ravel() - 10) <= 1)
    fit = np.full((n * n, 183), 0.5); y = fit.copy(); y[:, 100:] -= 0.3
    res["changed"] = core
    res["candidate_changed"] = True
    res["julianDay"] = np.where(core, 200.0, np.nan)
    _, fps = postprocess.apply_footprints(res, y, fit)
    assert fps.fp_status.tolist() == ["review: no clean surroundings"]    # nothing clean within 1500 m
    _, fps_off = postprocess.apply_footprints(res, y, fit, exclude_ring_candidates=False)
    assert fps_off.fp_status.tolist() == ["rejected: no stronger than surroundings"]


def test_footprint_compares_further_out_when_surroundings_changed():
    import pandas as pd
    import postprocess
    # 3x3 footprint; everything within ~400 m is a candidate; undisturbed pixels further out
    n, s = 121, 10.0
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, yy = cols.ravel() * s, rows.ravel() * s
    res = pd.DataFrame({"x_26191": x, "y_26191": yy})
    c = (n // 2) * s
    dist = np.hypot(x - c, yy - c)
    core = (abs(rows.ravel() - n // 2) <= 1) & (abs(cols.ravel() - n // 2) <= 1)
    fit = np.full((n * n, 183), 0.5); y = fit.copy()
    y[dist < 400, 100:] -= 0.3                         # footprint and its neighbours dropped (a fire)...
    res["changed"] = core
    res["candidate_changed"] = dist < 400
    res["julianDay"] = np.where(core, 200.0, np.nan)
    _, fps = postprocess.apply_footprints(res, y, fit)
    assert fps.fp_status.tolist() == ["accepted"] and fps.fp_ring_m.tolist() == [600.0]
    y[:, 100:] = 0.2                                   # ...or the whole area dropped (weather): rejected
    _, fps2 = postprocess.apply_footprints(res, y, fit)
    assert fps2.fp_status.tolist() == ["rejected: no stronger than surroundings"]


def test_regional_signal_uses_clean_neighbours():
    import bulcd_continuous as bc
    # 3 x 1 blocks; the middle block has too few eligible pixels -> uses the 3 x 3 neighbourhood
    bx = np.repeat([0, 1, 2], 200); by = np.zeros(600, np.int64)
    ub, inv = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
    d = np.where(bx == 0, -0.1, np.where(bx == 2, 0.1, 0.5))
    elig = np.ones(600, bool); elig[(bx == 1)][:] = False
    elig[200:395] = False                              # 5 eligible pixels left in block 1
    r = bc.regional_signal(d, elig, (bx, by, ub, inv.ravel()), 100)
    assert np.allclose(r[bx == 0], -0.1) and np.allclose(r[bx == 2], 0.1)
    assert abs(r[300]) < 0.11                          # neighbours, not the 5 disturbed values (0.5)
    assert np.allclose(bc.regional_signal(d, np.zeros(600, bool), (bx, by, ub, inv.ravel()), 100), 0)


def test_first_sustained_alert():
    h = np.full((4, 184), 0.05)                     # column 0 = prior, then 183 events
    h[0, 6:12] = 0.7                                # short false alarm (6 events = 12 days)...
    h[0, 50:150] = 0.9                              # ...then a lasting alert from event 50 -> day 100
    h[1, 5:100] = 0.9; h[1, 100:130] = 0.2; h[1, 130:] = 0.9   # alert fades mid-year: first lasting alert is day 10
    h[2, 170:] = 0.9                                # alert still on at year end, shorter than 60 days -> counts
    d = core.first_sustained_alert(h, 0.5, 60)
    assert d[0] == 100 and d[1] == 10 and d[2] == 340 and np.isnan(d[3])
    # GEE timing (last rise above 0.3) puts pixel 1 at event 130 instead
    assert core.first_evidence_of_change(h[1:2], np.array([True]), 0.3)[0] == 130


def test_footprint_pixels_must_touch():
    import pandas as pd
    import postprocess
    # two 3x3 blocks with a one-pixel (10 m) gap between them, same date -> two footprints;
    # a diagonal neighbour still counts as touching
    n = 30
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r, c = rows.ravel(), cols.ravel()
    res = pd.DataFrame({"x_26191": c * 10.0 + 5, "y_26191": 1000 - r * 10.0})
    a = (abs(r - 15) <= 1) & (abs(c - 10) <= 1)
    b = (abs(r - 15) <= 1) & (abs(c - 14) <= 1)        # columns 13-15: gap at column 12
    d = (r == 17) & (c == 8)                            # touches a only diagonally (row 16, col 9)
    fit = np.full((n * n, 183), 0.5); y = fit.copy()
    y[a | b | d, 100:] -= 0.3
    res["changed"] = a | b | d
    res["julianDay"] = np.where(a | b | d, 200.0, np.nan)
    _, fps = postprocess.apply_footprints(res, y, fit)
    assert sorted(fps.fp_size) == [9, 10]
