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
