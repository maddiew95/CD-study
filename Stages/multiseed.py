"""
Multi-seed model-training repeats + aggregation.

Terminology note: this is a repeated-restart / seed-variance study, not
Monte Carlo cross-validation. The train/val/test PARTITION is fixed
(DATA_SPLIT_SEED = 20260827) and identical across every seed and every
stage -- only the model's own stochastic elements change per seed (weight
init, minibatch shuffle order, XGBoost's random_state). This is deliberate:
it isolates model-training variance from data-sampling variance, unlike
Li et al.'s protocol where resampling across runs conflates the two. It
also means results across seeds are PAIRED on the same held-out test set,
enabling paired comparisons (see `paired_diff`) both within a stage (seed
vs seed) and across stages (Stage 1 vs Stage 2, same 5 seeds, same test
rows modulo the drift injection applied to them).
"""
import numpy as np
import pandas as pd

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_multi_seed(run_fn, seeds=DEFAULT_SEEDS, **kwargs):
    """Call run_fn once per seed (run_scenario or run_scenario_cnn_lstm),
    forcing verbose=False so 5 runs don't dump 5x the epoch logs, and print
    one summary line per seed instead. kwargs are forwarded as keyword args
    (pass scenario_idx=, dct=, cols=, device=, etc. by name)."""
    results = []
    for seed in seeds:
        r = run_fn(seed=seed, verbose=False, **kwargs)
        results.append(r)
        print(f"  seed={seed}  acc={r['accuracy']:.4f}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  "
              f"train_sec={r.get('train_sec', float('nan')):.1f}")
    return results


def aggregate_results(results):
    """Mean/std/min/max across seeds for accuracy/precision/recall/f1, plus
    a seed-averaged confusion matrix (both raw-count and row-normalized
    rate form) and the raw per-seed confusion stack for further analysis."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    stats = {}
    for m in metrics:
        vals = np.array([r[m] for r in results], dtype=float)
        stats[m] = {
            "mean": vals.mean(),
            "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            "min": vals.min(),
            "max": vals.max(),
        }

    cms = np.stack([r["confusion"] for r in results]).astype(float)  # (n_seeds, C, C)
    mean_cm = cms.mean(axis=0)
    row_sums = mean_cm.sum(axis=1, keepdims=True)
    mean_cm_rate = np.divide(mean_cm, row_sums,
                             out=np.zeros_like(mean_cm), where=row_sums != 0)

    return {
        "stats": stats,
        "mean_confusion": mean_cm,
        "mean_confusion_rate": mean_cm_rate,
        "confusion_stack": cms,
        "n_seeds": len(results),
        "model": results[0]["model"],
        "scenario": results[0]["scenario"],
        "raw_results": results,
    }


def summary_table(agg_dict):
    """agg_dict: {label: aggregate_results(...) output} -> tidy DataFrame,
    one row per label, mean/std columns per metric."""
    rows = []
    for label, agg in agg_dict.items():
        row = {"model": label, "n_seeds": agg["n_seeds"]}
        for m, s in agg["stats"].items():
            row[f"{m}_mean"] = s["mean"]
            row[f"{m}_std"] = s["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_accuracy(mean_confusion_rate, class_names=None):
    """Diagonal of the row-normalized mean confusion matrix = per-class
    recall, averaged across seeds."""
    n = mean_confusion_rate.shape[0]
    class_names = class_names if class_names is not None else list(range(n))
    return pd.Series(np.diag(mean_confusion_rate), index=class_names, name="accuracy")


def paired_diff(results_a, results_b, metric="accuracy", seeds=DEFAULT_SEEDS):
    """Seed-matched (results_a[i] vs results_b[i], same seed) difference in
    `metric`, plus a simple paired t-test. Meant for later cross-stage
    comparisons (e.g. Stage 1 vs Stage 2, same 5 seeds, same underlying
    test rows) once both stages' per-seed results are available together.
    """
    from scipy import stats as sps
    a = np.array([r[metric] for r in results_a], dtype=float)
    b = np.array([r[metric] for r in results_b], dtype=float)
    diffs = a - b
    t_stat, p_val = sps.ttest_rel(a, b) if len(a) > 1 else (np.nan, np.nan)
    return pd.DataFrame({
        "seed": list(seeds)[:len(a)], "a": a, "b": b, "diff": diffs
    }), {"mean_diff": diffs.mean(), "t_stat": t_stat, "p_value": p_val}
