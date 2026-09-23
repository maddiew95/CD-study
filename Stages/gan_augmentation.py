"""
GAN-augmentation orchestration for Stage 3 (GAN, no drift) and Stage 4
(GAN, with drift). Wraps wgans.py's train_class_gans/augment API and
handles the conversion between the project's `splits` (ClassSplit dict)
representation and the flat [Time, features..., Fault] array wgans.py
expects.

Design decisions (consistent with drift_injection.py's conventions):
  1. GAN training and generation happen entirely in RAW (unscaled) feature
     space, upstream of the frozen z-score scaler -- exactly the same slot
     Phase 3/4 drift injection occupies. The frozen scaler is always the
     LAST step before building tensors, applied identically to real and
     synthetic rows, fit once on Stage 1's raw Normal training data and
     never refit. This keeps one consistent architecture across all four
     stages: raw-space data-generating-process modifications first (drift
     and/or GAN), frozen-scaler transform last.
  2. Augmentation is applied to the TRAIN partition ONLY. val/test are
     copied through unchanged -- consistent with "val/test pools remain
     constant across all seeds and all stages."
  3. GAN training uses ONE FIXED seed regardless of how many classifier
     seeds are later evaluated against the resulting augmented set. This
     isolates classifier-training variance (the axis multiseed.py already
     sweeps) from GAN-training variance (a separate, not-yet-explored
     axis -- see the Stage 3 notebook's notes).
  4. wgans.py's `augment()` zero-fills every non-feature, non-label column
     for synthetic rows -- so synthetic rows get Time=0. This is harmless
     for the classifier (Time is excluded from `cols` and never fed to the
     model) but matters for Stage 4: drift injection must happen to the
     REAL data BEFORE GAN training, not after via a Time-based tau lookup,
     since synthetic rows have no meaningful Time. See `project_to_tau`
     for the one place a tau is needed for synthetic rows, where it is
     supplied explicitly rather than derived from Time.
"""
import copy
import numpy as np
import pandas as pd


def splits_train_to_array(splits):
    """Stack every class's TRAIN partition into one raw array, columns in
    the DataFrame's own natural order: [Time, <13 features>, Fault]. This
    matches wgans.py's default feat_slice=(1,14), label_col=-1 exactly, as
    long as `col_order` (returned alongside) is used consistently.
    """
    col_order = list(splits[0].train.columns)  # [Time, Ipv, ..., Vf, Fault]
    parts = [splits[lbl].train[col_order].to_numpy(dtype="float64")
             for lbl in sorted(splits)]
    return np.vstack(parts), col_order


def rebuild_splits_with_augmented_train(splits, aug_array, col_order):
    """Rebuild a NEW splits dict: .train replaced (per class) with the rows
    of `aug_array` belonging to that class; .val/.test are untouched deep
    copies of the input. `aug_array` may contain more rows per class than
    the original (real + synthetic) -- that's expected."""
    splits2 = copy.deepcopy(splits)
    label_col = col_order[-1]
    aug_df = pd.DataFrame(aug_array, columns=col_order)
    aug_df[label_col] = aug_df[label_col].round().astype(int)
    for lbl in sorted(splits2):
        class_df = aug_df.loc[aug_df[label_col] == lbl, col_order].reset_index(drop=True)
        splits2[lbl].train = class_df
    return splits2


def gan_augment_splits(splits, seed, device, ratio=1.0, gan_kw=None):
    """Train one WGAN-GP per class on `splits`' raw TRAIN partitions, then
    generate `ratio` synthetic rows per real row per class and append them.
    Returns (splits_augmented, gans, histories, n_real_rows).

    `n_real_rows` is the row count of the original stacked train array --
    aug_array[:n_real_rows] are the real rows, aug_array[n_real_rows:] are
    synthetic (this ordering is guaranteed by wgans.augment()'s
    `blocks = [train_array] + [...synthetic...]` construction).
    """
    from wgans import train_class_gans, augment
    gan_kw = gan_kw or {}
    train_array, col_order = splits_train_to_array(splits)
    n_classes = len(splits)
    gans, histories = train_class_gans(train_array, seed=seed, device=device,
                                       n_classes=n_classes, **gan_kw)
    aug_array = augment(train_array, gans, ratio=ratio, seed=seed, device=device)
    splits_aug = rebuild_splits_with_augmented_train(splits, aug_array, col_order)
    return splits_aug, gans, histories, len(train_array)


def project_to_tau(synthetic_df, amplitudes, features, tau, feature_cols_are_raw=True):
    """Stage 4b (domain-adaptation variant): push already-generated synthetic
    rows forward along the SAME parametric drift ramp used in Phase 4
    (value - amplitude * tau), using an EXPLICIT tau rather than one derived
    from Time (synthetic rows have no meaningful Time -- see module docstring
    point 4). `tau` may be a scalar or an array matching len(synthetic_df).

    This is what makes it "domain adaptation" rather than plain augmentation:
    the classifier sees labeled synthetic examples resembling the TARGET
    (test-time, high-tau) distribution during training, not just more
    examples of the training distribution it already has.
    """
    df = synthetic_df.copy()
    tau_arr = np.broadcast_to(np.asarray(tau, dtype="float64"), (len(df),))
    for feat in features:
        df[feat] = df[feat].to_numpy() - amplitudes[feat] * tau_arr
    return df


def adaptation_only_augment(splits, cols, features, amplitudes, ratio=1.0,
                            tau_range=(0.8125, 1.0), seed=0, label_col="Fault",
                            apply_projection=True):
    """Stage 4c -- adaptation-only, NO GAN. Bootstrap-resamples REAL training
    rows and (when `apply_projection=True`) projects them with the exact
    same known-formula shift used in 4b (`project_to_tau`), sampling tau
    from the same target range 4b uses. The only thing removed relative to
    4b is the WGAN-GP generation step -- rows being projected are resampled
    real rows, not generated ones.

    `apply_projection=False` gives the ROW-COUNT CONTROL: identical bootstrap
    duplication, identical final row count, but NO shift -- i.e. plain
    oversampling with zero target-domain information. This isolates sample
    count from shift content: comparing this control against Stage 2 (1000
    rows) tells you how much of 4c's improvement is "more rows" versus
    "rows placed where the target distribution actually is." Without this
    control, the Stage-2-vs-4c comparison confounds row count with
    adaptation content, since Stage 2 trains on 1000 rows and 4c on 2000.

    This isolates whether the generalization component (the GAN densifying/
    smoothing the training manifold) contributes anything beyond the
    adaptation component (the projection itself). If 4c matches or beats
    4b, the GAN was doing no useful work in 4b; if 4c meaningfully
    underperforms 4b, the GAN's richer coverage is earning its cost.

    Returns a NEW splits dict; val/test untouched, matching every other
    augmentation function in this module.
    """
    rng = np.random.default_rng(seed)
    splits2 = copy.deepcopy(splits)
    for lbl in sorted(splits2):
        real_df = splits[lbl].train.reset_index(drop=True)
        n_real = len(real_df)
        n_new = int(round(ratio * n_real))
        idx = rng.integers(0, n_real, size=n_new)  # bootstrap with replacement
        resampled = real_df.iloc[idx].reset_index(drop=True)
        if apply_projection:
            tau_sample = rng.uniform(tau_range[0], tau_range[1], size=n_new)
            resampled = project_to_tau(resampled, amplitudes, features, tau=tau_sample)
        resampled[label_col] = lbl
        combined = pd.concat([real_df, resampled], axis=0, ignore_index=True)
        splits2[lbl].train = combined
    return splits2


def build_full_spectrum_pool(splits, features, amplitudes, tau_range=(0.0, 1.0),
                             pool_multiplier=1.0, seed=0, label_col="Fault"):
    """Build an expanded per-class GAN-TRAINING pool by bootstrap-resampling
    real rows and shifting them across the FULL tau spread (default 0-1),
    not just the low-tau range real training data naturally occupies
    (0-0.625). Intended to give the GAN examples resembling the entire
    drift trajectory during its own training, rather than only ever seeing
    the low-severity slice.

    CAVEAT this function does NOT solve: the pool has no tau label
    attached to each row for the generator to condition on -- it's just a
    blended mixture. An unconditional GAN trained on it learns the merged
    marginal distribution, with no mechanism to selectively generate
    samples resembling any one tau region at generation time. This is
    expected to underperform project_to_tau's deterministic, targeted
    shift (4b/4c) for exactly that reason -- Stage 4f (below) tests this
    empirically rather than assuming it.
    """
    rng = np.random.default_rng(seed)
    splits2 = copy.deepcopy(splits)
    for lbl in sorted(splits2):
        real_df = splits[lbl].train.reset_index(drop=True)
        n_real = len(real_df)
        n_pool = int(round(pool_multiplier * n_real))
        idx = rng.integers(0, n_real, size=n_pool)
        resampled = real_df.iloc[idx].reset_index(drop=True)
        tau_sample = rng.uniform(tau_range[0], tau_range[1], size=n_pool)
        shifted = project_to_tau(resampled, amplitudes, features, tau=tau_sample)
        shifted[label_col] = lbl
        pool = pd.concat([real_df, shifted], axis=0, ignore_index=True)
        splits2[lbl].train = pool
    return splits2


def gan_augment_full_spectrum(splits, cols, features, amplitudes, seed, device,
                              pool_multiplier=1.0, ratio=1.0,
                              tau_range=(0.0, 1.0), gan_kw=None):
    """Stage 4f -- train the WGAN-GP on the full-tau-spectrum pool above
    (unconditional), then generate `ratio` synthetic rows per REAL row
    (matching every other Stage-4 variant's final row count of 2x real)
    and append them to the ORIGINAL real training rows only -- the pool
    itself is purely a GAN-training device, kept out of the final
    training set so the row count stays comparable to 4a/4b/4c/4d.

    Returns (splits_augmented, gans, histories).
    """
    from wgans import train_class_gans, generate
    gan_kw = gan_kw or {}
    pooled_splits = build_full_spectrum_pool(splits, features, amplitudes,
                                             tau_range=tau_range,
                                             pool_multiplier=pool_multiplier, seed=seed)
    pool_array, col_order = splits_train_to_array(pooled_splits)
    n_classes = len(splits)
    gans, histories = train_class_gans(pool_array, seed=seed, device=device,
                                       n_classes=n_classes, **gan_kw)

    splits2 = copy.deepcopy(splits)
    feat_cols = col_order[1:-1]  # the 13 features, matching wgans' feat_slice=(1,14)
    for lbl, (G, scaler, _n_pool) in gans.items():
        n_gen = int(round(ratio * len(splits[lbl].train)))
        gen = generate(G, scaler, n_gen, seed=seed, device=device,
                       latent_dim=gan_kw.get("latent_dim", 64))
        gen_df = pd.DataFrame(gen, columns=feat_cols)
        gen_df[col_order[0]] = 0.0    # Time, unused downstream (matches wgans.augment's convention)
        gen_df[col_order[-1]] = lbl
        gen_df = gen_df[col_order]
        real_df = splits[lbl].train.reset_index(drop=True)
        splits2[lbl].train = pd.concat([real_df, gen_df], axis=0, ignore_index=True)
    return splits2, gans, histories


# ---------------------------------------------------------------------------
# Stage 4g -- stratified multi-severity pool (pristine + full drift spectrum)
# ---------------------------------------------------------------------------

DEFAULT_TAU_BINS = (
    ("pristine",      0.000, 0.000),   # genuinely undrifted, pre-deployment baseline
    ("early",         0.000, 0.300),   # early-life mild degradation
    ("observed",      0.300, 0.625),   # the severity range real training data spans
    ("test_horizon",  0.625, 1.000),   # the unseen near-future (val+test region)
    ("beyond",        1.000, 1.500),   # extrapolated further degradation
)


def build_stratified_multiseverity_pool(splits_pristine, features, amplitudes,
                                        tau_bins=DEFAULT_TAU_BINS,
                                        bin_fractions=None, pool_multiplier=1.0,
                                        seed=0, label_col="Fault", verbose=True):
    """Build a per-class GAN-training pool stratified across drift severities,
    including a reserved tier of genuinely NON-drifted (pristine) rows.

    IMPORTANT -- absolute vs. additive tau. This function takes
    `splits_pristine` (the PRE-injection data, i.e. `splits_detrended`) and
    applies an ABSOLUTE tau shift per bin. That differs from
    `build_full_spectrum_pool` (4f) and `adaptation_only_augment` (4c/4d),
    which operate on the already-drifted `splits_stage2` and therefore apply
    their shift ON TOP of each row's existing local drift. Building from
    pristine data is what makes "this bin is severity X" actually true,
    which is the point of stratifying in the first place.

    Parameters
    ----------
    splits_pristine : dict[int, ClassSplit]
        PRE-drift-injection splits (use `splits_detrended`, not `splits_stage2`).
    tau_bins : sequence of (name, tau_lo, tau_hi)
        Severity strata. tau_lo == tau_hi gives an exact-severity tier
        (used for the pristine 0.0 tier). tau > 1.0 extrapolates beyond the
        test region -- deliberate, representing degradation further into
        deployment than anything observed.
    bin_fractions : sequence of float or None
        Proportion of the pool drawn from each bin. None -> equal split
        across all bins (with 5 default bins, that's the 20%-each design).
        Normalised internally, so it need not sum to exactly 1.

    Returns
    -------
    splits_pool : dict[int, ClassSplit]  -- .train holds the stratified pool;
                                            val/test carried through untouched.
    """
    n_bins = len(tau_bins)
    if bin_fractions is None:
        bin_fractions = [1.0 / n_bins] * n_bins
    if len(bin_fractions) != n_bins:
        raise ValueError(f"bin_fractions has {len(bin_fractions)} entries, tau_bins has {n_bins}")
    total = float(sum(bin_fractions))
    fracs = [f / total for f in bin_fractions]

    rng = np.random.default_rng(seed)
    splits2 = copy.deepcopy(splits_pristine)

    for lbl in sorted(splits2):
        real_df = splits_pristine[lbl].train.reset_index(drop=True)
        n_real = len(real_df)
        n_pool = int(round(pool_multiplier * n_real))

        blocks, report = [], []
        for (name, lo, hi), frac in zip(tau_bins, fracs):
            n_bin = int(round(frac * n_pool))
            if n_bin <= 0:
                continue
            idx = rng.integers(0, n_real, size=n_bin)
            rows = real_df.iloc[idx].reset_index(drop=True)
            if hi > lo:
                tau_sample = rng.uniform(lo, hi, size=n_bin)
            else:
                tau_sample = np.full(n_bin, lo)
            if np.any(tau_sample != 0.0):
                rows = project_to_tau(rows, amplitudes, features, tau=tau_sample)
            rows[label_col] = lbl
            blocks.append(rows)
            report.append(f"{name}[{lo:.3f}-{hi:.3f}]:{n_bin}")

        splits2[lbl].train = pd.concat(blocks, axis=0, ignore_index=True)
        if verbose and lbl == 0:
            print(f"  pool composition (per class, n={len(splits2[lbl].train)}): " + "  ".join(report))

    return splits2


def gan_augment_stratified(splits_pristine, splits_target, cols, features, amplitudes,
                           seed, device, tau_bins=DEFAULT_TAU_BINS, bin_fractions=None,
                           pool_multiplier=1.0, ratio=1.0, gan_kw=None):
    """Stage 4g -- train the WGAN-GP on the stratified multi-severity pool
    (pristine + graded drift severities), then append `ratio` synthetic rows
    per real row to `splits_target`'s real training rows.

    `splits_pristine` supplies the GAN's training pool (pre-drift data,
    shifted to controlled absolute severities). `splits_target` supplies the
    real rows the synthetic data is appended to -- normally `splits_stage2`,
    so the final training set is the same real drifted rows every other
    Stage-4 variant uses, keeping the comparison fair and the row count at
    2x real.

    Returns (splits_augmented, gans, histories, pool_splits).
    """
    from wgans import train_class_gans, generate
    gan_kw = gan_kw or {}

    pool_splits = build_stratified_multiseverity_pool(
        splits_pristine, features, amplitudes, tau_bins=tau_bins,
        bin_fractions=bin_fractions, pool_multiplier=pool_multiplier, seed=seed)

    pool_array, col_order = splits_train_to_array(pool_splits)
    n_classes = len(splits_target)
    gans, histories = train_class_gans(pool_array, seed=seed, device=device,
                                       n_classes=n_classes, **gan_kw)

    splits2 = copy.deepcopy(splits_target)
    feat_cols = col_order[1:-1]
    for lbl, (G, scaler, _n_pool) in gans.items():
        n_gen = int(round(ratio * len(splits_target[lbl].train)))
        gen = generate(G, scaler, n_gen, seed=seed, device=device,
                       latent_dim=gan_kw.get("latent_dim", 64))
        gen_df = pd.DataFrame(gen, columns=feat_cols)
        gen_df[col_order[0]] = 0.0
        gen_df[col_order[-1]] = lbl
        gen_df = gen_df[col_order]
        real_df = splits_target[lbl].train.reset_index(drop=True)
        splits2[lbl].train = pd.concat([real_df, gen_df], axis=0, ignore_index=True)

    return splits2, gans, histories, pool_splits
