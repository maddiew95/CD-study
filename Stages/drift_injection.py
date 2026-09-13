"""
Stage 2 drift-injection module: Phase 3 (Vdc detrending) + Phase 4 (parametric
monotonic drift injection) from concept_drift_gan_methodology.md.

Design decisions baked into this module (flagged for review):
  1. Detrending scope: applies to Vdc, class-0 (Normal) rows ONLY, across that
     class's own train+val+test window. This is the only Normal data available
     in base_splits.pkl (the 1600-row model-ready window), which is much
     narrower than the full raw corpus used for the Phase 2 EDA audit. The
     resulting local trend is small (see manipulation-check output) -- this
     step is included for methodological completeness/consistency across
     stages, not because it removes a large confound in this particular slice.
  2. Drift-injection timeline: uses a LOCAL, per-class artificial progress
     variable tau in [0,1], defined as each row's position within its OWN
     class's train->val->test window (verified contiguous in Time). This was
     chosen over a single GLOBAL real-time timeline because each class occupies
     a narrow, non-overlapping slice of the full ~11s recording -- a
     global-time ramp would leave almost no tau variation *within* any single
     class's split (e.g. Normal's whole window covers tau ~ 0.267-0.284 of the
     global range), making the chunk-based manipulation check meaningless.
     The local design also directly mirrors Phase 2.2's own "chunk 1 vs chunk
     3" comparison and produces the classic "train on clean-ish data, test on
     drifted data" scenario the experiment wants.
  3. Injection scope: applied to ALL 8 classes (not Normal-only), since the
     drift represents a shared physical degradation process (panel/inverter
     aging) that would affect sensor readings regardless of which fault
     happens to be concurrently present. Flagged as a design choice -- easy to
     restrict to Normal-only via `scope="normal_only"` if the intended design
     differs.
  4. Calibration metric: severity is matched across features in Z-SCORE units
     (using the frozen Stage-1 scaler's per-feature std), not raw units. The
     roadmap's "3-5x the Vdc floor (W~=1.089)" is a raw-Vdc-volts number, and
     applying it literally in raw units to Ipv/Iabc (whose natural std is
     ~0.1 and ~0.003 respectively) would demand physically absurd shifts.
     Matching effect size in the same standardized space the classifiers
     actually see is the more defensible reading of "3-5x more severe than
     the natural floor."
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import ks_2samp, wasserstein_distance

DRIFT_FEATURES = ["Ipv", "Vpv", "Iabc"]       # Phase 4.1: moderate-response group (F2/F4-informed)
VDC_FLOOR_W_RAW = 1.089                        # Phase 2 EDA reference (Normal-only, full corpus, chunk1 vs chunk3)
VDC_FLOOR_KS_RAW = 0.553


def _deepcopy_splits(splits):
    import copy
    return copy.deepcopy(splits)


# ---------------------------------------------------------------------------
# Phase 3: Vdc detrending (Normal class only)
# ---------------------------------------------------------------------------

def detrend_vdc_normal(splits, normal_label=0):
    """Linear-detrend Vdc for the Normal class only, across its own
    train+val+test window (fit jointly, applied in original Time order).
    Fault-class rows are returned completely untouched.

    Returns
    -------
    splits2 : dict[int, ClassSplit]   -- new object, input is not mutated
    trend_info : dict with the raw/detrended/fitted-trend series (indexed
                 like the concatenated Normal partition) for Phase 3.4
                 verification plots/tests.
    """
    splits2 = _deepcopy_splits(splits)
    s = splits2[normal_label]

    parts = [("train", s.train), ("val", s.val), ("test", s.test)]
    combined = pd.concat([p[1].assign(_part=p[0]) for p in parts], axis=0).sort_values("Time")

    raw = combined["Vdc"].to_numpy()
    # scipy.signal.detrend fits/removes a best-fit line vs. sample index;
    # valid here since Time increments are ~uniform within a class.
    residual = signal.detrend(raw, type="linear")
    detrended = residual + raw.mean()            # Phase 3.2: re-center on original mean
    fitted_trend = raw - detrended                # reconstruction path (memory-documented formula)

    combined = combined.assign(Vdc=detrended)
    for part_name, df in parts:
        idx = combined.index[combined["_part"] == part_name]
        new_vdc = combined.loc[idx, "Vdc"]
        target_df = getattr(s, part_name)
        target_df.loc[new_vdc.index, "Vdc"] = new_vdc.values

    trend_info = {
        "time": combined["Time"].to_numpy(),
        "raw": raw,
        "detrended": detrended,
        "fitted_trend": fitted_trend,
    }
    return splits2, trend_info


# ---------------------------------------------------------------------------
# Phase 4: parametric monotonic drift injection
# ---------------------------------------------------------------------------

def _local_tau(df, tmin, tmax):
    span = tmax - tmin
    if span <= 0:
        return np.zeros(len(df))
    return ((df["Time"].to_numpy() - tmin) / span).clip(0.0, 1.0)


def inject_drift(splits, amplitudes, features=DRIFT_FEATURES, scope="all"):
    """Inject Delta_feature(tau) = -amplitude * tau into each targeted feature.

    tau is LOCAL to each class: 0 at that class's earliest train row, 1 at its
    latest test row (train->val->test verified contiguous in Time).

    Parameters
    ----------
    amplitudes : dict[str, float]   raw-unit amplitude per feature
    scope      : "all" (default) or "normal_only"
    """
    splits2 = _deepcopy_splits(splits)
    labels = sorted(splits2) if scope == "all" else [0]

    tau_lookup = {}
    for lbl in labels:
        s = splits2[lbl]
        tmin = s.train["Time"].min()
        tmax = s.test["Time"].max()
        for part in ("train", "val", "test"):
            df = getattr(s, part)
            tau = _local_tau(df, tmin, tmax)
            tau_lookup[(lbl, part)] = tau
            for feat in features:
                df.loc[:, feat] = df[feat].to_numpy() - amplitudes[feat] * tau

    return splits2, tau_lookup


def calibrate_amplitude(splits, feature, target_wasserstein, scope="all",
                         labels=None, tol=0.02, max_iter=25):
    """Binary-search an amplitude so that the MEAN per-class (train vs test)
    Wasserstein distance for `feature`, after injection, matches
    `target_wasserstein` (raw units) within `tol` relative error."""
    labels = labels if labels is not None else (sorted(splits) if scope == "all" else [0])

    def mean_w(amp):
        s2, _ = inject_drift(splits, {feature: amp}, features=[feature], scope=scope)
        ws = []
        for lbl in labels:
            tr = s2[lbl].train[feature].to_numpy()
            te = s2[lbl].test[feature].to_numpy()
            ws.append(wasserstein_distance(tr, te))
        return float(np.mean(ws))

    lo, hi = 0.0, max(1.0, target_wasserstein * 10)
    # expand hi until we bracket the target
    while mean_w(hi) < target_wasserstein and hi < 1e6:
        hi *= 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        w = mean_w(mid)
        if abs(w - target_wasserstein) / target_wasserstein < tol:
            return mid, w
        if w < target_wasserstein:
            lo = mid
        else:
            hi = mid
    return mid, w


# ---------------------------------------------------------------------------
# Manipulation check (Phase 4.4 / Phase 2.2-style)
# ---------------------------------------------------------------------------

def manipulation_check(splits_before, splits_after, cols, labels=None):
    """Per-class, per-feature KS + Wasserstein (train vs test), before vs
    after injection. Returns a tidy DataFrame."""
    labels = labels if labels is not None else sorted(splits_before)
    rows = []
    for lbl in labels:
        for feat in cols:
            for tag, s in (("before", splits_before), ("after", splits_after)):
                tr = s[lbl].train[feat].to_numpy()
                te = s[lbl].test[feat].to_numpy()
                ks = ks_2samp(tr, te).statistic
                w = wasserstein_distance(tr, te)
                rows.append({"class": lbl, "feature": feat, "condition": tag,
                             "ks": ks, "wasserstein": w})
    return pd.DataFrame(rows)
