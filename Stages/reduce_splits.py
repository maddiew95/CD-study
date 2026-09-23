"""
Scarcity-study utility: reduce a ClassSplit's training partition to a
smaller, genuinely scarce sample size, while keeping val/test untouched.

Design choice: keep the EARLIEST n_train rows chronologically (by Time),
not a random subsample. This preserves the train -> val -> test time
contiguity every downstream drift/tau computation relies on, and it makes
the train/test tau gap LARGER (harder), not smaller -- train now covers
tau 0 to n_train/total instead of 0 to 1000/total, while val/test (and
their tau range) are completely unchanged, so test results stay directly
comparable to the original n=1000 study on the exact same held-out rows.
"""
import copy


def reduce_train_size(splits, n_train=600, label_col_index=-1):
    """Return a NEW splits dict with each class's train partition truncated
    to its earliest n_train rows (by Time). val and test are untouched --
    same rows, same size, as the original n=1000 study, so Stage-1(n=600)
    vs. Stage-1(n=1000) is a clean, directly comparable test-set evaluation.
    """
    splits2 = copy.deepcopy(splits)
    for lbl in sorted(splits2):
        s = splits2[lbl]
        if len(s.train) < n_train:
            raise ValueError(f"class {lbl} has only {len(s.train)} train rows, "
                             f"cannot reduce to {n_train}")
        s.train = s.train.sort_values("Time").iloc[:n_train].reset_index(drop=True)
    return splits2
