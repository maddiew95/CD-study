# Stage 1 Data Foundation — Finalized Design & Reasoning
### GPVS-Faults dataset | Companion to `concept_drift_gan_methodology.md`

---

## Scope

This document records the finalized data-preparation decisions underlying Stage 1 (no GAN, no drift baseline) — which, despite the name, is not Stage-1-specific. It is the single shared data foundation reused, unmodified, across all four stages:

| Stage | GAN | Drift |
|---|---|---|
| 1 | No | No |
| 2 | No | Yes (injected) |
| 3 | Yes | No |
| 4 | Yes | Yes (injected) |

Everything below describes work already implemented and verified in `base_splits.py`, run against the actual GPVS-Faults CSV (1,070,068 rows × 15 columns).

---

## 1. Confirmed dataset structure

Before finalizing the sampling logic, the raw CSV's actual layout was verified directly (not assumed):

| Fault | Rows | Time range within block |
|---|---|---|
| F0 (Normal) | 141,014 | 0.00 – 14.10s |
| F1 | 139,014 | 0.00 – 13.90s |
| F2 | 144,015 | 0.00 – 14.40s |
| F3 | 69,967 | 0.0001 – 6.996s |
| F4 | 144,014 | 0.0001 – 14.40s |
| F5 | 144,014 | 0.0001 – 14.40s |
| F6 | 144,015 | 0.00 – 14.40s |
| F7 | 144,015 | 0.0001 – 14.40s |

**Finding:** the file is a straight concatenation of 8 experiment runs, one per class, in order F0→F7. Each class is exactly one positionally-contiguous block with monotonically increasing `Time` (verified: `Time` resets to ~0 exactly at each of the 7 label-boundary transitions, and nowhere else). No class's rows are split across multiple non-adjacent segments.

**Why this matters:** the segment-detection logic in `base_splits.py` (row-position + Time-gap checks) was built defensively to handle the case of a label reappearing in a separate, non-adjacent block. That case does not occur in this dataset — but the check remains in the script as a guard, in case the input data source ever changes (e.g. a re-export, a different concatenation order, or additional experiment runs appended later).

F3 is notably smaller and shorter (~7s vs ~14s for the rest) — consistent with Bakdi et al.'s description of F3 (grid anomaly / intermittent voltage sags) as inherently shorter-duration than a sustained fault condition.

---

## 2. Sample size: 1,000 train / 300 val / 300 test per class

| Parameter | Value | Reasoning |
|---|---|---|
| Train | 1,000 rows/class | Deliberately scarce, tying the journal's framing back to ITNAC's small-sample scarcity narrative and positioning GAN augmentation as practically motivated |
| Validation | 300 rows/class | Fixed, independent of training scarcity — kept identical across every training-size condition so eval precision doesn't confound the scarcity effect |
| Test | 300 rows/class | Same fixed-pool logic as validation; held out, never touched by tuning |

**Why not proportional to class size:** classes are close enough in size (69,967–144,015) that a fixed per-class N avoids the complexity of proportional allocation without materially privileging any class.

**Precedent check:** Li et al. (Transformer-LSTM-SVM, same GPVS-Faults dataset) used 200/class combined for validation+test, with an ambiguous split between the two roles. This project's 300/300 (explicitly disjoint) is deliberately more generous and more clearly specified, because this project's factorial design (4 stages × up to 2 classifiers × 5 seeds) rests on more pairwise comparisons than Li et al.'s single scarcity sweep, and needs more eval-set precision to keep those comparisons trustworthy.

**Benchmark anchor:** Li et al.'s Transformer-LSTM-SVM reached ~92% accuracy at ~500 train samples/class (their 11% scenario), no drift, no augmentation, same dataset. This is a useful sanity anchor for Stage 1 results once the classifier is trained — a large deviation either direction is worth understanding before proceeding to Stage 2.

---

## 3. Window selection: seeded, contiguous, per class

Each class's 1,600-row window (1,000 + 300 + 300) is selected via:

1. Detect all positionally-contiguous, Time-monotonic segments for the class (in this dataset, exactly one per class — see Section 1).
2. Filter to segments long enough to hold the full window.
3. Pick a segment (weighted by length) and a random start offset within it, using a **single fixed seed (`DATA_SPLIT_SEED = 20260827`), independent of the 5 model-training seeds** used downstream.

**Why the data-split seed is separate from model-training seeds:** conflating them would mix two different sources of variance — data-sampling noise and model-training noise — into one number, inflating the reported standard deviation and making it impossible to run paired comparisons across stages (Stage 1 seed 3 vs. Stage 3 seed 3 must differ *only* because of the GAN, not because the eval data changed too). This is a deliberate methodological improvement over Li et al., whose reported averaging over 10 runs appears to re-sample data each run rather than fixing it, which conflates these two variance sources.

---

## 4. Split ordering: chronological, not random

Within the selected 1,600-row window, the first 1,000 rows become train, the next 300 become validation, the last 300 become test — in that temporal order, not a random shuffle.

**Reasoning:** sensor data sampled at a fixed rate is heavily autocorrelated — adjacent rows are nearly identical. A random split risks placing near-duplicate rows on both sides of a train/test boundary, which inflates measured accuracy through leakage rather than reflecting real generalization. Chronological splitting also more realistically simulates deployment (train on past behavior, evaluate on later behavior) and is compatible with Phase 4's planned drift injection, which operates along this same timeline.

An optional purge gap (`PURGE_GAP_N`, default 0) is implemented but not currently enabled — it would drop a small buffer of rows at each split boundary for additional leakage insurance, borrowed from purged cross-validation practice in time-series ML. Left off by default since the dataset's autocorrelation at split boundaries hasn't been shown to be a practical problem; available if a reviewer raises the concern.

---

## 5. Reuse across all four stages

The split above is built **once** and reused unmodified across Stages 1–4:

- **Validation and test are read-only in every stage** — real data only, never touched by GAN augmentation or drift injection, identical across all four stages. This is what makes Stage 1 vs. Stage 3 (or Stage 2 vs. Stage 4) a clean measurement of the manipulated condition's effect, rather than a measurement partly contaminated by a different evaluation set.
- **Only the training partition changes stage to stage**: real-only (Stages 1–2), or real + GAN-synthetic (Stages 3–4). Drift injection (Stages 2, 4) is applied on top of this same fixed structure, along the same established timeline.
- The GAN itself (Stages 3–4) is trained only on the real train partition, treated as an unordered set of tabular samples (chronology isn't relevant to what the GAN learns, since it's modeling the feature distribution, not a sequence).

Persistence: the built split is saved once to `base_splits.pkl` via `save_splits()`, and every stage script loads it with `load_splits()` rather than rebuilding it — guaranteeing byte-for-byte identical data across stages with no risk of drift from re-running the pipeline.

---

## 6. Vdc detrending — Normal (F0) partition only

**What:** a linear trend is fit to `Vdc` vs. `Time` using the Normal class's **train** rows only, then subtracted from train, validation, and test alike, re-centered on the train mean.

**Why train-only fitting:** fitting on val/test (or the whole window) would leak future information into the trend model — the same leakage-avoidance logic as the train/val/test split itself.

**Why Normal only:** fault-class rows have no physical basis for a Normal-derived trend correction — applying one would corrupt the ground truth intended for later drift injection design. Fault rows remain raw in all four stages.

**Why detrend at all:** natural Vdc drift was confirmed as real in the original Phase 2 audit (KS ≈ 0.553, Wasserstein ≈ 1.089, full-dataset chunk comparison) but too narrow and causally ambiguous to serve as the study's primary drift phenomenon. Removing it from the Normal baseline ensures that when synthetic drift is injected in Stages 2/4, it isn't confounded with this pre-existing natural drift — and ensures Stage 1's "no drift" label is actually accurate, not just "no injected drift on top of an already-nonstationary baseline."

---

## 7. Detrend verification — implemented and run

A verification step (`verify_detrend()`) re-runs the KS/Wasserstein chunk comparison on the *detrended* Normal Vdc, mirroring the original Phase 2.2 audit, rather than assuming the linear fit worked.

**Result on the actual seeded window:**

| Metric | Pre-detrend (full dataset, Phase 2) | Post-detrend (this 1,600-row window) |
|---|---|---|
| KS statistic | 0.553 | 0.3265 (p = 1.58e-25) |
| Wasserstein distance | 1.089 | 0.1457 |

**Diagnosis:** the Wasserstein distance dropped substantially (as expected — the linear trend was genuinely removed). The KS statistic did not drop to the expected near-baseline range (~0.01–0.07). Investigation showed this is **not** residual drift:

- Chunk-by-chunk means (146.81 / 146.80 / 146.91) and standard deviations (0.61 / 0.63 / 0.57) across the window are nearly identical — no leftover monotonic trend.
- A quadratic (degree-2) fit on the same train data did not improve the KS statistic (0.3471, slightly worse than linear) — ruling out unresolved curvature as the cause.
- The actual driver is **tail asymmetry**: chunk minimums are 141.50 / 143.55 / 144.14 — the first third of the window has meaningfully deeper transient dips than the later two-thirds, even though the average level is flat.

**Conclusion:** detrending is working correctly for what it's designed to do (remove monotonic trend). The elevated KS statistic reflects a separate phenomenon — transient outliers — not a detrending failure, and is not fixable by a better trend fit.

---

## 8. Outlier characterization — investigated, left as-is

Full-train (not just chunked) descriptive statistics for detrended Normal Vdc:

```
count    1000.000000
mean      146.820703
std         0.628127
min       141.502950   (8.47 std below mean)
25%       146.496711
50%       146.785130
75%       147.072826
max       151.191340   (6.96 std above mean)
```

Mean and median are close (146.82 vs. 146.79) with a tight IQR (0.576), consistent with no meaningful drift in central tendency — supporting Stage 1's "no drift" premise for the bulk of the data.

However, the min and max are extreme relative to the standard deviation (8.47σ and 6.96σ respectively — well beyond typical 3σ outlier thresholds). Locating these rows:

- **9 of 1,000 rows** (0.9%) sit more than 5σ from the mean — 4 extreme lows, 5 extreme highs.
- They are **scattered across the window** (Time ≈ 3.999–4.087s), not confined to one contiguous region — ruling out a simple `EXCLUDE_RANGES` time-window fix.
- Two rows (indices 936, 937) are **adjacent** and show a sharp low-then-high swing (143.57 → 151.19) within roughly 0.0001s of Time — consistent with a genuine transient/switching glitch in the sensor signal rather than measurement error or drift.

**Decision:** leave these rows unfiltered in the finalized Stage 1 data. Two alternatives were considered and explicitly not taken:

- *Boxplot/IQR-based outlier removal* (the method Li et al. used in their preprocessing) — would have literature precedent, but was not applied, in order to keep Stage 1's baseline representative of real, unedited sensor behavior.
- *A non-linear (polynomial/LOESS) detrend* — tested via the quadratic comparison above and confirmed not to address this issue, since it isn't a trend problem.

This is documented here specifically so it reads as a considered decision rather than an unexamined artifact, should it come up in review.

---

## 9. Deliverables

| File | Purpose |
|---|---|
| `base_splits.py` | Core module: `load_dataframe`, `build_base_splits`, `verify_detrend`, `save_splits`, `load_splits`, and supporting segment-detection / windowing / detrending functions |
| `base_splits.pkl` | The actual built split — 8 classes × (1,000 train / 300 val / 300 test), Normal-Vdc detrended, generated once from the real CSV and reused across all stages |
| `diagnose_detrend.py` | Standalone diagnostic script used to investigate the Section 7/8 findings — chunk mean/std/min/max comparison, linear-vs-quadratic residual check |

**Recommended repository layout**, given `base_splits.pkl` is shared infrastructure rather than Stage-1-specific:

```
cd-study/
├── base_splits.py
├── base_splits.pkl
├── stage1_baseline/
├── stage2_drift/
├── stage3_gan/
└── stage4_gan_drift/
```

---

## 10. Open items not yet resolved

- **Purge gap** (`PURGE_GAP_N`) implemented but disabled (0) — available if leakage at split boundaries becomes a concern later.
- **`EXCLUDE_RANGES`** implemented but empty — available if a future window (different seed, different class) surfaces a contiguous problem region worth excluding, unlike the scattered outliers found here.
- **Drift injection design (Phase 4)** and **classifier selection** remain to be finalized, per the main methodology roadmap.
