"""BULC-D core algorithm, ported from the Earth Engine modules used by
users/fpelletier/SIG:682-fao-disturbance/1.1-bulcdCaller.

Everything here is plain numpy on arrays shaped (pixels, time bins); nothing
touches the database. Each function names the GEE function it replaces.

Array conventions
-----------------
* A "series" is a float array (P, N): one row per 10 m pixel, one column per
  2-day bin, NaN where the bin had no clear observation (= masked in GEE).
* Bin calendars come from `bin_calendar`, which reproduces the date logic of
  afn_gatherCollectionsAndReduce (515.gatherCollections27b).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Defaults, from 0.0-bulcdDependencies and 6003.3c-BULC-AdvancedParameters
# ---------------------------------------------------------------------------
BIN_CUTS = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], dtype=float)

# Rows = event bins 1..10 (Z-score low -> high); columns = classes
# 1 (decrease), 2 (stable), 3 (increase).
CUSTOM_TRANSITION_MATRIX = np.array([
    [0.83, 0.08, 0.08],
    [0.66, 0.24, 0.08],
    [0.53, 0.37, 0.08],
    [0.14, 0.76, 0.08],
    [0.08, 0.83, 0.08],
    [0.08, 0.83, 0.08],
    [0.08, 0.76, 0.14],
    [0.08, 0.37, 0.53],
    [0.08, 0.24, 0.66],
    [0.08, 0.08, 0.83],
])


@dataclass
class BULCParams:
    """getBULCParameterDictionary() in 6003.3c-BULC-AdvancedParameters."""
    initializing_leveler: float = 0.7
    transition_leveler: float = 0.7
    posterior_leveler: float = 0.9
    n_classes: int = 3
    base_class: int = 2  # "nothing has changed"
    transition_matrix: np.ndarray = field(default_factory=lambda: CUSTOM_TRANSITION_MATRIX.copy())


@dataclass
class Calendar:
    """One synthetic image per bin, as built by afn_nestedDay."""
    year: int
    starts: list[date]          # inclusive bin start
    step: int                   # dayStepSize

    @property
    def n(self) -> int:
        return len(self.starts)

    @property
    def ends(self) -> list[date]:
        return [s + timedelta(days=self.step) for s in self.starts]

    @property
    def nominal_doy(self) -> np.ndarray:
        # end.getRelative('day', 'year') -> 0-based day of year of the bin END
        return np.array([(e - date(e.year, 1, 1)).days for e in self.ends])

    @property
    def frac_doy(self) -> np.ndarray:
        return self.nominal_doy / 365.0

    def bin_index(self, obs_dates: np.ndarray) -> np.ndarray:
        """Bin number for each observation date (datetime64[D]); -1 if outside."""
        start0 = np.datetime64(self.starts[0], "D")
        k = (obs_dates - start0).astype(int) // self.step
        return np.where((obs_dates >= start0) & (k < self.n), k, -1)

    @property
    def fetch_range(self) -> tuple[date, date]:
        """[first day, day after last bin) needed from the database."""
        return self.starts[0], self.ends[-1]


def _doy_to_month_day_1970(doy: int) -> tuple[int, int]:
    d = date(1970, 1, 1) + timedelta(days=doy - 1)
    return d.month, d.day


def bin_calendar(year: int, first_doy: int = 1, last_doy: int = 365, step: int = 2) -> Calendar:
    """Bin start dates for one year (afn_nestedYear).

    GEE builds ee.List.sequence(startMillis, endMillis, step), so the last bin
    starts on (or before) the end date and runs `step` days past it, into the
    next year if lastDOY is 365. Month/day are resolved in 1970 (non-leap), as
    afn_DOYToMonth(…, 1970) does.
    """
    sm, sd = _doy_to_month_day_1970(first_doy)
    em, ed = _doy_to_month_day_1970(last_doy)
    start = date(year, sm, sd)
    end = date(year, em, ed)
    if sm > em or (sm == em and sd > ed):
        end = date(year + 1, em, ed)
    starts, s = [], start
    while s <= end:
        starts.append(s)
        s += timedelta(days=step)
    return Calendar(year=year, starts=starts, step=step)


# ---------------------------------------------------------------------------
# Harmonic regression (502.7-1h5-HarmonicFunctions)
# ---------------------------------------------------------------------------
def harmonic_independents(modality: dict) -> list[str]:
    """afn_determineHarmonicIndependentsViaModalityDictionary."""
    names = []
    if modality.get("constant"):
        names.append("constant")
    if modality.get("linear"):
        names.append("t")
    if modality.get("unimodal"):
        names += ["cos", "sin"]
    if modality.get("bimodal"):
        names += ["cos2", "sin2"]
    if modality.get("trimodal"):
        names += ["cos3", "sin3"]
    return names


def harmonic_design(frac_doy: np.ndarray, names: list[str]) -> np.ndarray:
    """afn_addHarmonicBandsViaModalityDictionary: one column per independent.

    t = fracDOY. Angular frequency = 2 / period, with period 2 (unimodal),
    1 (bimodal) and 0.667 (trimodal), so unimodal is cos(2*pi*t).
    """
    t = np.asarray(frac_doy, dtype=float)
    two_pi_f = {p: (2.0 / p) * 2 * np.pi for p in (2, 1, 0.667)}
    cols = {
        "constant": np.ones_like(t),
        "t": t,
        "cos": np.cos(t * two_pi_f[2]), "sin": np.sin(t * two_pi_f[2]),
        "cos2": np.cos(t * two_pi_f[1]), "sin2": np.sin(t * two_pi_f[1]),
        "cos3": np.cos(t * two_pi_f[0.667]), "sin3": np.sin(t * two_pi_f[0.667]),
    }
    return np.column_stack([cols[n] for n in names])


def fit_harmonic(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """afn_fitHarmonicToCollection: per-pixel OLS (ee.Reducer.linearRegression).

    y: (P, N) with NaN gaps; X: (N, K). Returns coefficients (P, K), NaN for
    pixels with fewer than K clear observations or a singular design.
    """
    valid = ~np.isnan(y)
    y0 = np.where(valid, y, 0.0)
    w = valid.astype(float)
    xtx = np.einsum("pn,ni,nj->pij", w, X, X)
    xty = np.einsum("pn,ni->pi", y0, X)
    k = X.shape[1]
    coefs = np.full((y.shape[0], k), np.nan)
    ok = valid.sum(axis=1) >= k
    ok &= np.abs(np.linalg.det(xtx)) > 1e-12
    if ok.any():
        coefs[ok] = np.linalg.solve(xtx[ok], xty[ok][..., None])[..., 0]
    return coefs


def apply_harmonic(coefs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """afn_applyHarmonicToCollection: 'fitted' band, shape (P, N)."""
    return coefs @ X.T


# ---------------------------------------------------------------------------
# afn_organizeBULCD_Inputs (1002.03-6002.A2b.3), optical-only stream
# ---------------------------------------------------------------------------
def organize_bulcd_inputs(y_exp, cal_exp, y_tgt, cal_tgt, modality, numerator_factor, denominator_factor):
    names = harmonic_independents(modality)
    X_exp = harmonic_design(np.concatenate([c.frac_doy for c in cal_exp]), names)
    X_tgt = harmonic_design(np.concatenate([c.frac_doy for c in cal_tgt]), names)

    coefs = fit_harmonic(y_exp, X_exp)
    exp_fit = apply_harmonic(coefs, X_exp)
    tgt_fit = apply_harmonic(coefs, X_tgt)

    # expectationPeriodSD = sampleStdDev of (observed - fitted) in the expectation years
    resid = y_exp - exp_fit
    n_res = np.sum(~np.isnan(resid), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.where(n_res >= 2, np.nanstd(resid, axis=1, ddof=1), np.nan)
        # targetLOFAsZScore = (obs - fitted) * num / max(SD, den), clamped to [-10, 10].
        # A masked SD stays masked in GEE (masked.max(c) is masked), hence fmax is NOT used.
        denom = np.where(np.isnan(sd), np.nan, np.maximum(sd, denominator_factor))
        z = np.clip((y_tgt - tgt_fit) * numerator_factor / denom[:, None], -10, 10)

    if "cos" in names and "sin" in names:
        amplitude = np.hypot(coefs[:, names.index("cos")], coefs[:, names.index("sin")])
    else:
        amplitude = np.full(y_exp.shape[0], np.nan)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels -> NaN
        pre_mean = np.nanmean(y_exp, axis=1)

    return {
        "independents": names,
        "coefficients": coefs,
        "expect_fit": exp_fit,            # expectCollectionFit 'fitted' band
        "target_fit": tgt_fit,            # targetCollectionFit 'fitted' band
        "expectation_sd": sd,             # expectationPeriodSD
        "target_z": z,                    # targetLOFAsZScore
        "amplitude": amplitude,           # nbr_amplitude
        "expectation_mean": pre_mean,     # expectationPeriodSummaryMean
    }


# ---------------------------------------------------------------------------
# Binning (502.8-Binning-5, afn_getOneZBin)
# ---------------------------------------------------------------------------
def z_to_bins(z: np.ndarray, cuts: np.ndarray = BIN_CUTS) -> np.ndarray:
    """Bin i (1..len(cuts)+1) where cuts[i-2] < z <= cuts[i-1]; 0 where z is NaN."""
    b = np.searchsorted(cuts, np.nan_to_num(z), side="left") + 1
    return np.where(np.isnan(z), 0, b).astype(np.int16)


# ---------------------------------------------------------------------------
# BULC (BULC-Minimal-Module-107, runBULCAlgorithm)
# ---------------------------------------------------------------------------
def run_bulc(events: np.ndarray, p: BULCParams = BULCParams()):
    """Iterative Bayesian update over event images.

    events: int (P, N), 1..10, 0 = masked (no update for that pixel).
    Returns final probabilities (P, 3) and the probCls1 history (P, N+1),
    whose column 0 is the prior (GEE records the initial layer too, because the
    Record* flags are ee objects and therefore always truthy in JS).
    """
    k = p.n_classes
    prior_min = (1 - p.initializing_leveler) / k
    prior = np.full(k, prior_min)
    prior[p.base_class - 1] += p.initializing_leveler

    trans_min = (1 - p.transition_leveler) / k
    table = np.vstack([np.ones(k), p.transition_matrix * p.transition_leveler + trans_min])  # dummy row 0

    post_min = (1 - p.posterior_leveler) / k
    P, N = events.shape
    probs = np.tile(prior, (P, 1))
    hist1 = np.empty((P, N + 1))
    hist1[:, 0] = probs[:, 0]
    for t in range(N):
        e = events[:, t]
        v = e > 0
        if v.any():
            like = table[e[v]]
            num = probs[v] * like
            post = num / num.sum(axis=1, keepdims=True)
            probs[v] = post * p.posterior_leveler + post_min
        hist1[:, t + 1] = probs[:, 0]
    return probs, hist1


# ---------------------------------------------------------------------------
# Timing (104.10-timingDisturbance, afn_firstEvidenceOfChange)
# ---------------------------------------------------------------------------
def first_evidence_of_change(prob1_hist: np.ndarray, changed: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """1-based index of the last probCls1 layer <= threshold (NaN if not changed).

    Layer 1 is the prior, so a value k means the probability rose above the
    threshold at event k and never fell back.
    """
    n = prob1_hist.shape[1]
    idx = np.arange(1, n + 1)
    last_low = np.max(np.where(prob1_hist <= threshold, idx, 0), axis=1).astype(float)
    last_low[(last_low + 1 > n) | (last_low == 0) | ~changed] = np.nan
    return last_low


def doy_to_month(doy: np.ndarray) -> np.ndarray:
    """Month expression in getProbabilityOfDecrease (non-leap thresholds)."""
    cuts = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    m = np.searchsorted(cuts, doy, side="left") + 1.0
    return np.where(np.isnan(doy), np.nan, m)


def first_sustained_alert(prob1_hist: np.ndarray, threshold: float = 0.5, hold_days: int = 60,
                          step: int = 2) -> np.ndarray:
    """Day of the first event at which probCls1 >= threshold and stays there for hold_days.

    Not in GEE. julianDay uses the LAST rise above 0.3, which moves a change to
    months later whenever the alert fades for a while (e.g. when the seasonal
    curve dips close to an already-low NBR). This gives the start of the first
    lasting alert instead. A run that is still above the threshold at the end of
    the year counts even if shorter than hold_days. Day = step x event number,
    the same scale as julianDay. NaN when no lasting alert.
    """
    above = prob1_hist[:, 1:] >= threshold          # column 0 is the prior
    n = above.shape[1]
    run = np.zeros(above.shape, np.int32)            # length of the run of True starting at each event
    nxt = np.zeros(above.shape[0], np.int32)
    for j in range(n - 1, -1, -1):
        nxt = np.where(above[:, j], nxt + 1, 0)
        run[:, j] = nxt
    hold = max(1, hold_days // step)
    ok = (run >= hold) | ((run > 0) & (run == n - np.arange(n)))
    first = np.argmax(ok, axis=1)
    return np.where(ok.any(axis=1), (first + 1) * step, np.nan).astype(float)


# ---------------------------------------------------------------------------
# Post-change mean (104.20-afn_toDnbr)
# ---------------------------------------------------------------------------
def post_change_mean(y_tgt: np.ndarray, first_change: np.ndarray, n_obs: int = 5):
    """Mean and median of the first `n_obs` clear target observations at bin
    index >= first_change (0-based bin index compared with the 1-based event
    number, exactly as the GEE code does)."""
    P, N = y_tgt.shape
    fc = np.where(np.isnan(first_change), np.inf, first_change)
    take = (np.arange(N)[None, :] >= fc[:, None]) & ~np.isnan(y_tgt)
    take &= np.cumsum(take, axis=1) <= n_obs
    cnt = take.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.where(cnt > 0, np.where(take, y_tgt, 0).sum(axis=1) / cnt, np.nan)
        median = np.nanmedian(np.where(take, y_tgt, np.nan), axis=1)
    return mean, median


# ---------------------------------------------------------------------------
# getProbabilityOfDecrease (0.0-bulcdDependencies)
# ---------------------------------------------------------------------------
def get_probability_of_decrease(y_exp, cal_exp, y_tgt, cal_tgt, *, first_doy=1,
                                modality=None, numerator_factor=1.0, denominator_factor=0.05,
                                bin_cuts=BIN_CUTS, bulc_params=BULCParams(), timing_threshold=0.3):
    modality = modality or {"bimodal": False, "constant": True, "linear": False,
                            "trimodal": False, "unimodal": True}
    inp = organize_bulcd_inputs(y_exp, cal_exp, y_tgt, cal_tgt, modality,
                                numerator_factor, denominator_factor)
    events = z_to_bins(inp["target_z"], bin_cuts)
    final_probs, hist1 = run_bulc(events, bulc_params)

    p1 = final_probs[:, 0]
    was_changed = p1 >= 0.5
    first_change = first_evidence_of_change(hist1, was_changed, timing_threshold)

    step = cal_tgt[0].step
    seq = first_change * step                                  # 'julianDay' band (seqImage)
    true_jd = np.mod(seq + first_doy - 1, 365) + 1
    month = doy_to_month(true_jd)

    return {
        **inp,
        "events": events,
        "final_probs": final_probs,        # finalBULCprobs (probCls1..3)
        "prob1_history": hist1,            # allProbs, probCls1 layers
        "was_changed": was_changed,        # wasChanged (probCls1 >= 0.5)
        "first_change": first_change,      # sequencer.firstChange
        "julian_day": seq,                 # seqImage
        "month": month,                    # monthChange
    }
