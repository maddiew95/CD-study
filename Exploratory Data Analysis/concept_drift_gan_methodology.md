# Concept Drift Mitigation via GAN Augmentation — Methodology Roadmap
### GPVS-Faults dataset | Journal extension of ICSPCS 2024 & ITNAC 2026

---

## Overview

This roadmap sequences the work into five phases: (1) dataset characterization, (2) baseline drift audit, (3) data cleaning, (4) drift injection design, and (5) the GAN-mitigation experiment. Each step lists the objective, the method, and the tools used.

---

## Phase 1 — Dataset Characterization

**Goal:** Understand what GPVS-Faults actually contains before touching it — feature structure, fault semantics, and known confounds.

| Step | Action | Tool |
|---|---|---|
| 1.1 | Confirm feature set: 13 continuous features across 3 measurement zones (array: Ipv, Vpv; DC-link: Vdc; inverter/grid AC: ia,ib,ic,va,vb,vc,Iabc,If,Vabc,Vf) | Mendeley dataset page (`n76t439f65/1`) |
| 1.2 | Pull the physical fault-type key | Bakdi et al. 2021, *Electrical Power and Energy Systems* — Table 1 |
| 1.3 | Note known confounds: MPPT vs. IPPT operating mode; temperature/insolation variability during experiments; manual fault injection at experiment midpoint | Same source paper |

**Fault-type reference table (Bakdi et al., Table 1):**

| Fault | Type | Description | Severity tier |
|---|---|---|---|
| F1 | Inverter fault | Complete failure in one of six IGBTs | Catastrophic |
| F2 | Feedback sensor fault | One-phase sensor fault, 20% | Moderate |
| F3 | Grid anomaly | Intermittent voltage sags | Grid-external |
| F4 | PV array mismatch | 10–20% nonhomogeneous partial shading | Moderate |
| F5 | PV array mismatch | 15% open circuit in PV array | Catastrophic |
| F6 | MPPT/IPPT controller fault | −20% PI gain (boost converter) | Subtle |
| F7 | Boost converter controller fault | +20% PI time-constant | Subtle |

---

## Phase 2 — Exploratory & Baseline Drift Audit

**Goal:** (a) See which features separate Normal from each fault, and (b) confirm whether the *Normal* baseline itself is stationary before any injection.

### 2.1 Visual exploration — boxplot grid

| Step | Action | Tool |
|---|---|---|
| 2.1.1 | Boxplot grid: 13 features × 8 conditions (Normal + F1–F7), `showfliers=False` for readability at scale | Python: `pandas`, `matplotlib`, `seaborn` |
| 2.1.2 | Interpret per feature: location-shift vs. dispersion-shift vs. no effect | Manual + cross-reference against Phase 1 fault table |

**Findings from this study:**
- **Ipv, Vpv, Vdc** — catastrophic collapse under F1, F5 (location shift)
- **ia, ib, ic** — F1 shows dispersion shift (outlier spread), not location shift
- **F2, F4** — moderate, proportional reduction in Ipv/Vpv/Iabc (a "moderate sub-group")
- **va, vb, vc** — essentially flat across all conditions (grid-imposed, fault-irrelevant)
- **F6, F7** — visually indistinguishable from Normal; consistent with source paper's own PCA-based detector failing on these

### 2.2 Baseline stationarity check (Normal data only)

| Step | Action | Tool |
|---|---|---|
| 2.2.1 | Rolling mean/std over time, per feature | `pandas.rolling()`, `matplotlib` |
| 2.2.2 | Chunk-based KS test + Wasserstein distance (chunk 1 vs. chunk 3, plus adjacent 1v2/2v3 for shift shape) | `scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance` |
| 2.2.3 | Streaming drift detector (secondary check) | `river.drift.ADWIN` |

**Findings from this study:**

| Feature | KS(1v3) | Wasserstein(1v3) | Verdict |
|---|---|---|---|
| Vdc | 0.553 | 1.089 | **Real, substantial, continuous drift** |
| Vabc | 0.051 | 0.948 | Real but non-monotonic/oscillatory |
| Iabc, Vf, If, Ipv, ia | 0.014–0.067 | ≤0.021 | Statistically detectable, practically negligible (large-N artifact) |
| ib, ic, va, vb, vc | ≤0.007 | ≤0.27 | No meaningful shift |

⚠️ **ADWIN caveat**: raw AC waveforms (va, vb, vc, ia, ib, ic) triggered thousands of false "drift points" because ADWIN reacts to sinusoidal oscillation, not real distributional change. This contradicted the KS results on the same features — **do not report raw ADWIN counts on oscillating signals** without first transforming to a non-oscillating statistic (e.g., rolling RMS/envelope).

**Conclusion of Phase 2:** natural drift is real (Vdc) but narrow (1–2 features), modest in magnitude relative to fault-driven shifts, and causally ambiguous (likely environmental, not degradation-driven). Not sufficient as a primary study vehicle — proceed to controlled injection, but first clean the baseline.

---

## Phase 3 — Data Cleaning (Detrending)

**Goal:** Remove the confirmed natural Vdc (and possibly Vabc) drift from the Normal reference partition so injected drift isn't confounded with pre-existing environmental drift.

| Step | Action | Tool |
|---|---|---|
| 3.1 | Fit a trend model to Vdc over Time (Normal data only) | `scipy`/`statsmodels` (polynomial or LOESS) |
| 3.2 | Subtract fitted trend, re-center on original mean | `numpy`/`pandas` |
| 3.3 | Inspect Vabc's oscillatory pattern before deciding whether a simple detrend is appropriate, or whether it should be left alone | Rolling-window visualization |
| 3.4 | **Re-run Phase 2.2 tests on detrended Vdc** to confirm KS/Wasserstein drop to near-baseline (ib/ic/va/vb/vc level) | Same tools as 2.2 — this becomes your documented verification step |
| 3.5 | Apply detrending only to the **Normal reference partition** used for GAN training/baseline characterization — leave fault-condition rows untouched | — |

---

## Phase 4 — Drift Injection Design

**Goal:** Inject controlled, physically-grounded, gradual drift into features that are fault-relevant but not already catastrophic — modeling a *degrading-toward-failure* trajectory rather than reproducing an existing fault class.

| Step | Action | Tool / Reference |
|---|---|---|
| 4.1 | Select target features: the **moderate-response group** informed by F2/F4 (e.g., Ipv, Vpv, Iabc), **not** the already-catastrophic F1/F5 group | Phase 2.1 findings |
| 4.2 | Choose injection method: **parametric, monotonic, additive perturbation** (ramping magnitude over an artificial timeline) — not feature rotation, not label flipping | Method family review (Žliobaitė 2010 = rotation baseline; parametric/additive = main method) |
| 4.3 | Set drift shape: **gradual/incremental** (smooth ramp), consistent with IGBT wear-out physics — not sudden step-change | — |
| 4.4 | Calibrate magnitude: injected Wasserstein distance should clearly exceed the empirical natural-drift floor from Phase 2 (Vdc ≈1.09) — e.g., target 3–5× that floor | Manipulation check using same KS/Wasserstein tools |
| 4.5 | Validate injected drift visually | Webb et al. (2016) drift-magnitude mapping technique — same visualization style used for characterization, reused for validation |
| 4.6 | Optional: add a feature-rotation (Žliobaitė) condition as a naive, non-physically-grounded baseline to contrast against the main physically-grounded injection | `numpy`/custom script |

**What KS/Wasserstein can and cannot prove post-injection:**

| Claim | Provable via KS/Wasserstein? |
|---|---|
| Drift is broader (more features affected) | ✅ Yes — run same chunk test across all 13 features |
| Drift is bigger in magnitude | ✅ Yes — compare injected Wasserstein vs. natural floor |
| Drift is less causally ambiguous | ❌ No — this comes from controlled experimental design, not the statistical test itself; KS/Wasserstein only serves as a manipulation check |

---

## Phase 5 — GAN Mitigation Experiment

**Goal:** Test whether GAN-based augmentation mitigates classifier degradation under injected drift, using a factorial design that isolates drift damage from augmentation benefit.

### 5.1 Core 2×2×2 factorial matrix

| | No GAN | With GAN |
|---|---|---|
| **No Drift** | Cell 1 — control baseline | Cell 2 — pure augmentation benefit (replicates ITNAC) |
| **With Drift** | Cell 3 — pure drift damage | Cell 4 — **main research question**: does GAN mitigate drift? |

Crossed with a **classifier axis** (pick 2, spanning ITNAC's finding):
- One **hybrid** classifier (e.g., LSTM-XGB or Transformer-LSTM-SVM — showed largest augmentation benefit under scarcity)
- One **non-hybrid** classifier (e.g., CNN or CNN-LSTM — showed little benefit under scarcity)

→ 8 cells total minimum.

**Key comparisons and what each proves:**

| Comparison | Answers |
|---|---|
| Cell 1 vs. 3 | Drift damage alone |
| Cell 1 vs. 2 | GAN benefit alone (no drift) |
| Cell 3 vs. 4 | GAN's mitigation effect *under* drift |
| Cell 2 vs. 4 | Whether drift erodes GAN's normal benefit |

### 5.2 Train/test drift-severity sub-matrix (applies specifically to Cell 4)

Rather than a single matched train/test drift level, sweep severity to get a response curve (mirrors ITNAC's scarcity-scene design):

| Test condition | Purpose |
|---|---|
| No drift | Sanity check — does GAN-augmented-for-drift model still work on clean data |
| Mild drift (below training severity) | Under-severity generalization |
| Matched drift (= training severity) | Interpolation case |
| Severe drift (above training severity) | Over-severity generalization — the harder, more novel claim |
| Optional: different feature-subset drift than trained on | True out-of-distribution robustness |

### 5.3 Fixed parameters for first pass

- **GAN architecture**: one strong performer from ITNAC (WGAN-GP) — defer multi-GAN comparison to a later extension
- **Seeds**: 5 seeds per cell, matching ITNAC's protocol, for statistical credibility
- **Metric**: macro-F1 as primary; consider per-fault-class error breakdown (especially F2/F4/F6/F7-adjacent classes) to tie results back to the physically-grounded injection story

*(Point 8 — full experimental parameters, evaluation metrics, and run logistics — deferred per your request; to be resolved once Phase 4 injection function is finalized.)*

---

## Tools Summary

| Purpose | Tool |
|---|---|
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical distance tests | `scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance` |
| Streaming drift detection | `river.drift.ADWIN` (use cautiously — see Phase 2 caveat) |
| Trend fitting / detrending | `scipy`/`statsmodels` (polynomial or LOESS) |
| Drift injection | Custom parametric perturbation functions (`numpy`) |
| Drift-magnitude visualization | Webb et al. (2016, 2018) mapping technique, reproduced in-house |
| GAN architectures | WGAN-GP (primary), others per ITNAC pipeline if extended |
| Classifiers | Subset of ITNAC's five (hybrid + non-hybrid pair) |

---

## Key References

- Bakdi, A., Bounoua, W., Guichi, A., & Mekhilef, S. (2021). Real-time fault detection in PV systems under MPPT using PMU and high-frequency multi-sensor data through online PCA-KDE-based multivariate KL divergence. *Electrical Power and Energy Systems*, 125, 106457.
- Bakdi, A., Guichi, A., Mekhilef, S., & Bounoua, W. (2020). GPVS-Faults dataset. *Mendeley Data*. https://data.mendeley.com/datasets/n76t439f65/1
- Webb, G. I., Hyde, R., Cao, H., Nguyen, H. L., & Petitjean, F. (2016). Characterizing concept drift. *Data Mining and Knowledge Discovery*, 30(4), 964–994.
- Webb, G. I., Lee, L. K., Goethals, B., & Petitjean, F. (2018). Analyzing concept drift and shift from sample data. *Data Mining and Knowledge Discovery*, 32(5), 1179–1199.
- Žliobaitė, I. (2010). Change with delayed labeling: When is it detectable? *IEEE ICDM Workshops*.
- [ITNAC 2026 paper — user's own conference paper, comparative GAN study on GPVS-Faults]
- [ICSPCS 2024 paper — user's own conference paper, GAN augmentation for IoT anomaly detection]

---

*Document generated as a working roadmap — revise as Phase 4/5 design decisions are finalized.*
