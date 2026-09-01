"""
Base Data Split — build once, reuse across all 4 stages
=========================================================
GPVS-Faults dataset — contiguous, seeded, stratified train/val/test
windows per class (Normal + F1-F7), with Vdc detrending for the
Normal partition only.

Renamed from stage1_sampling_split.py: despite the earlier name, this
split is NOT specific to Stage 1 — it's the one shared foundation all
four stages (1: no GAN/no drift, 2: no GAN/drift, 3: GAN/no drift,
4: GAN/drift) sit on top of. Build it once, save it, and every stage
loads the same file instead of rebuilding it.

Design decisions encoded here (per project conversation history):
- One seeded contiguous window per class, in chronological order
  (train earliest, test latest). Chronological (not random) splitting
  avoids leakage from autocorrelated adjacent rows landing on both
  sides of a train/test boundary.
- The data-split seed is FIXED and INDEPENDENT of the 5 model-training
  seeds used downstream. This split runs once; only fault-injection
  (stages 2/4) and GAN augmentation of the TRAIN block (stages 3/4)
  vary between stages. Validation and test are read-only and identical
  in every stage.
- Windows are checked to never cross a class-label boundary, via both
  row-position contiguity and a Time-gap check (catches cases where the
  same label reappears in a different, non-adjacent experiment segment
  — confirmed NOT to occur in the actual GPVS-Faults CSV, where each
  class is exactly one contiguous block, but the check stays as a
  guard in case the input data source ever changes).
- Vdc detrending (Normal partition only) is fit on the TRAIN slice only
  and applied to val/test using that fitted trend model, to avoid
  leakage. Fault-class rows (F1-F7) are never detrended, in any stage.
- Optional purge gap: rows dropped between train/val and val/test
  boundaries, for extra insurance against leakage from autocorrelated
  rows sitting right at a split edge.
- After detrending, a KS/Wasserstein check re-runs on the result to
  confirm the drift actually dropped (Phase 3.4 in the methodology
  doc) rather than just assuming the fit worked.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Config — adjust column names / constants to match your actual dataframe
# ----------------------------------------------------------------------------

TIME_COL = "Time"
LABEL_COL = "Fault"
VDC_COL = "Vdc"
NORMAL_LABEL = 0

EXPECTED_COLUMNS = [
    "Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic",
    "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf", "Fault",
]

TRAIN_N = 1000
VAL_N = 300
TEST_N = 300

# Rows dropped between train/val and val/test boundaries (each gap applied
# once on each side). 0 = no gap (original design). Increase for extra
# insurance against autocorrelated rows leaking across a split edge.
PURGE_GAP_N = 0

DATA_SPLIT_SEED = 20260827  # fixed, independent of model-training seeds

# Optional: Time ranges to exclude per class label (e.g. the asymmetric
# min/max region flagged during detrending diagnostics on Vdc). Fill in
# once you've pinned down the exact Time range; leave empty to skip.
# Example: EXCLUDE_RANGES = {0: [(1234.5, 1250.0)]}
EXCLUDE_RANGES: dict[int, list[tuple[float, float]]] = {}

# Pre-detrend baseline, for comparison when verifying detrend results
# (Phase 2.2 finding, chunk 1 vs chunk 3 on raw Vdc):
PRE_DETREND_KS = 0.553
PRE_DETREND_WASSERSTEIN = 1.089


def compute_window_n(purge_gap: int = PURGE_GAP_N) -> int:
    """Total rows needed per class: train + val + test + 2 purge gaps."""
    return TRAIN_N + VAL_N + TEST_N + 2 * purge_gap


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_dataframe(path: str | Path, expected_columns: list[str] = EXPECTED_COLUMNS) -> pd.DataFrame:
    """
    Load the GPVS-Faults CSV with a schema check, so a stage script fails
    loudly and early if the input file's columns or types ever drift,
    instead of failing confusingly deep inside segment detection.
    """
    df = pd.read_csv(path)

    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"CSV at {path} is missing expected columns: {sorted(missing)}")

    extra = set(df.columns) - set(expected_columns)
    if extra:
        print(f"Warning: CSV at {path} has unexpected extra columns: {sorted(extra)}")

    non_numeric = [
        c for c in expected_columns
        if c != LABEL_COL and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Expected numeric dtype for columns: {non_numeric}")

    if not pd.api.types.is_integer_dtype(df[LABEL_COL]):
        print(
            f"Warning: '{LABEL_COL}' column has dtype {df[LABEL_COL].dtype}, "
            f"not integer — check label encoding before proceeding."
        )

    if df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        print(f"Warning: CSV at {path} contains NaN values in columns: {na_cols}")

    return df


# ----------------------------------------------------------------------------
# Segment detection
# ----------------------------------------------------------------------------

def find_contiguous_segments(
    df: pd.DataFrame,
    label: int,
    time_col: str = TIME_COL,
    label_col: str = LABEL_COL,
    max_gap: float | None = None,
) -> list[tuple[int, int]]:
    """
    Return (start_row_idx, end_row_idx) iloc-based ranges where label_col == label
    AND rows are contiguous in the dataframe's row order AND Time doesn't jump
    by more than max_gap. Guards against the same label reappearing in a
    separate, non-adjacent experiment block being treated as one window.

    max_gap defaults to 3x the median per-row Time delta for this class —
    robust to normal sampling jitter, but catches real experiment-boundary jumps.
    """
    mask = (df[label_col] == label).to_numpy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No rows found for label {label}")

    pos_breaks = np.flatnonzero(np.diff(idx) != 1)

    t = df[time_col].to_numpy()[idx]
    dt = np.diff(t)
    if max_gap is None:
        positive_dt = dt[dt > 0]
        max_gap = 3 * np.median(positive_dt) if positive_dt.size else np.inf
    time_breaks = np.flatnonzero(dt > max_gap)

    breaks = np.union1d(pos_breaks, time_breaks)
    segment_starts = np.concatenate(([0], breaks + 1))
    segment_ends = np.concatenate((breaks, [idx.size - 1]))

    return [(idx[s], idx[e]) for s, e in zip(segment_starts, segment_ends)]


def eligible_segments(segments: list[tuple[int, int]], min_len: int) -> list[tuple[int, int]]:
    """Keep only segments long enough to host a full window of min_len rows."""
    return [(s, e) for s, e in segments if (e - s + 1) >= min_len]


# ----------------------------------------------------------------------------
# Window selection
# ----------------------------------------------------------------------------

def select_window(
    df: pd.DataFrame,
    label: int,
    rng: np.random.Generator,
    exclude_ranges: list[tuple[float, float]] | None = None,
    window_n: int = compute_window_n(),
) -> pd.DataFrame:
    """
    Pick one seeded, contiguous window_n-row block for `label`, avoiding any
    excluded Time ranges. Returns the block sorted by Time with a fresh index.
    """
    all_segs = find_contiguous_segments(df, label)
    segs = eligible_segments(all_segs, min_len=window_n)
    if not segs:
        longest = max((e - s + 1) for s, e in all_segs)
        raise ValueError(
            f"No contiguous segment long enough ({window_n} rows) for label {label}. "
            f"Longest available: {longest}"
        )

    exclude_ranges = exclude_ranges or []

    # weight segment choice by length so longer segments are more likely picked
    lengths = np.array([e - s + 1 for s, e in segs], dtype=float)
    seg_choice = rng.choice(len(segs), p=lengths / lengths.sum())
    s, e = segs[seg_choice]
    max_start_offset = (e - s + 1) - window_n

    for _ in range(200):  # retry budget against exclusion zones
        offset = int(rng.integers(0, max_start_offset + 1))
        start = s + offset
        end = start + window_n - 1
        t_start, t_end = df[TIME_COL].iloc[start], df[TIME_COL].iloc[end]

        if not any(t_start <= hi and t_end >= lo for lo, hi in exclude_ranges):
            window = df.iloc[start : end + 1].sort_values(TIME_COL).reset_index(drop=True)
            assert (window[LABEL_COL] == label).all(), "Window crosses a label boundary — bug."
            assert len(window) == window_n
            return window

    raise RuntimeError(
        f"Could not find a {window_n}-row window for label {label} avoiding "
        f"excluded ranges after 200 attempts — segment may be too fragmented, "
        f"or exclusion ranges too broad relative to segment length."
    )


def split_window(window: pd.DataFrame, purge_gap: int = PURGE_GAP_N):
    """
    Chronological split with an optional purge gap dropped on each side of
    val: [train] [purge_gap] [val] [purge_gap] [test]. With purge_gap=0 this
    is a plain contiguous train -> val -> test cut (original behavior).
    """
    train_end = TRAIN_N
    val_start = train_end + purge_gap
    val_end = val_start + VAL_N
    test_start = val_end + purge_gap
    test_end = test_start + TEST_N

    train = window.iloc[:train_end].reset_index(drop=True)
    val = window.iloc[val_start:val_end].reset_index(drop=True)
    test = window.iloc[test_start:test_end].reset_index(drop=True)
    return train, val, test


# ----------------------------------------------------------------------------
# Vdc detrending (Normal partition only; fit on train, apply to val/test)
# ----------------------------------------------------------------------------

def fit_linear_trend(time: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Fit values ~ a*time + b on the TRAIN slice only; return (a, b)."""
    a, b = np.polyfit(time, values, deg=1)
    return a, b


def detrend_normal_vdc(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit a linear trend to Vdc using TRAIN only, subtract it from train/val/test,
    and re-center on the train mean. Fitting only on train avoids leaking
    val/test information into the trend model. Call this on the Normal-class
    split only — never on fault-class splits.

    NOTE: if the asymmetric min/max flagged earlier in the pipeline turns out
    to reflect a non-linear trend rather than a transient artifact, replace
    fit_linear_trend with a polynomial or LOESS fit here — the train-only-fit
    / apply-to-all discipline stays the same either way.
    """
    a, b = fit_linear_trend(train[TIME_COL].to_numpy(), train[VDC_COL].to_numpy())
    original_mean = train[VDC_COL].mean()

    out = []
    for part in (train, val, test):
        part = part.copy()
        trend = a * part[TIME_COL].to_numpy() + b
        part[VDC_COL] = part[VDC_COL].to_numpy() - trend + original_mean
        out.append(part)
    return tuple(out)  # train_dt, val_dt, test_dt


def verify_detrend(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, vdc_col: str = VDC_COL, chunk_frac: float = 1 / 3
) -> dict:
    """
    Re-run the Phase 2.2-style KS/Wasserstein check on the DETRENDED Vdc to
    confirm the drift actually dropped, rather than assuming the linear fit
    worked. Compares the first chunk_frac vs last chunk_frac of the
    concatenated, chronologically-ordered train+val+test Vdc series —
    mirroring the original chunk-1-vs-chunk-3 comparison.

    Only meaningful when called on the Normal-class split (the only one
    that gets detrended).

    For reference, the pre-detrend natural drift measured KS~=0.553,
    Wasserstein~=1.089 (module-level PRE_DETREND_KS / PRE_DETREND_WASSERSTEIN).
    A successful detrend should bring these down near the no-drift baseline
    seen in other stable features (KS in roughly 0.01-0.07).
    """
    combined = pd.concat([train, val, test], ignore_index=True)
    vdc = combined[vdc_col].to_numpy()
    n = len(vdc)
    chunk_n = int(n * chunk_frac)

    first_chunk = vdc[:chunk_n]
    last_chunk = vdc[-chunk_n:]

    ks_stat, ks_pvalue = ks_2samp(first_chunk, last_chunk)
    w_dist = wasserstein_distance(first_chunk, last_chunk)

    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein": float(w_dist),
        "pre_detrend_ks": PRE_DETREND_KS,
        "pre_detrend_wasserstein": PRE_DETREND_WASSERSTEIN,
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

@dataclass
class ClassSplit:
    label: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_base_splits(
    df: pd.DataFrame,
    class_labels=range(8),
    split_seed: int = DATA_SPLIT_SEED,
    exclude_ranges: dict[int, list[tuple[float, float]]] | None = None,
    purge_gap: int = PURGE_GAP_N,
    verify: bool = True,
) -> dict[int, ClassSplit]:
    """
    Build the fixed, seeded train/val/test split for every class label.

    Run this ONCE per project (not once per stage). Save the result with
    save_splits() and load it with load_splits() in every stage's script —
    only fault-injection (stages 2/4) and GAN augmentation of the train
    block (stages 3/4) should vary downstream, never this underlying split.

    verify=True re-runs the KS/Wasserstein check on the detrended Normal
    class and prints the result, so you have documented evidence the
    detrend worked rather than just an assumption.
    """
    exclude_ranges = exclude_ranges or EXCLUDE_RANGES
    window_n = compute_window_n(purge_gap)
    rng = np.random.default_rng(split_seed)

    results: dict[int, ClassSplit] = {}
    for label in class_labels:
        window = select_window(df, label, rng, exclude_ranges.get(label), window_n=window_n)
        train, val, test = split_window(window, purge_gap=purge_gap)

        if label == NORMAL_LABEL:
            train, val, test = detrend_normal_vdc(train, val, test)
            if verify:
                result = verify_detrend(train, val, test)
                print(
                    f"[detrend check] label {label}: "
                    f"KS={result['ks_stat']:.4f} (p={result['ks_pvalue']:.3g}), "
                    f"Wasserstein={result['wasserstein']:.4f}  "
                    f"[pre-detrend baseline: KS~={result['pre_detrend_ks']}, "
                    f"Wasserstein~={result['pre_detrend_wasserstein']}]"
                )

        results[label] = ClassSplit(label=label, train=train, val=val, test=test)

    return results


def summarize_splits(splits: dict[int, ClassSplit]) -> pd.DataFrame:
    rows = []
    for label, cs in splits.items():
        rows.append(
            {
                "label": label,
                "train_n": len(cs.train),
                "val_n": len(cs.val),
                "test_n": len(cs.test),
                "time_start": cs.train[TIME_COL].iloc[0],
                "time_end": cs.test[TIME_COL].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Persistence — build once, load everywhere else
# ----------------------------------------------------------------------------

DEFAULT_SPLITS_PATH = "base_splits.pkl"


def save_splits(splits: dict[int, ClassSplit], path: str | Path = DEFAULT_SPLITS_PATH) -> None:
    """Serialize the built splits to disk so later stages don't need to rebuild them."""
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved base splits for {len(splits)} classes to {path.resolve()}")


def load_splits(path: str | Path = DEFAULT_SPLITS_PATH) -> dict[int, ClassSplit]:
    """
    Load the previously-built splits. Use this at the top of every stage's
    script instead of calling build_base_splits again — guarantees byte-for-
    byte identical train/val/test across stages 1-4, with zero risk of the
    result drifting if the df-loading step upstream ever changes.

    Requires base_splits.py to be importable (defines the ClassSplit class
    that the pickle references) — keep this file alongside base_splits.pkl.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_base_splits(df) once and save_splits(splits) "
            f"before loading it in stage scripts."
        )
    with open(path, "rb") as f:
        splits = pickle.load(f)
    print(f"Loaded base splits for {len(splits)} classes from {path.resolve()}")
    return splits


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Run this once, e.g. in a "00_build_base_splits" notebook/script ---
    # df = load_dataframe("df.csv")
    # splits = build_base_splits(df)
    # print(summarize_splits(splits))
    # save_splits(splits)  # writes base_splits.pkl in the current directory
    #
    # --- Then in every stage's script (1, 2, 3, 4), just load it: ---
    # from base_splits import load_splits
    # splits = load_splits()
    #
    # Access e.g. Normal train set: splits[0].train
    # Access e.g. F3 test set:      splits[3].test
    pass
