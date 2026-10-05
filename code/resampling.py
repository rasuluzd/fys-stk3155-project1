"""
Resampling: the bootstrap bias-variance decomposition and k-fold cross-validation
(lecture notes, Secs. 2.10-2.12; week 36 exercises).

LLM-assisted
------------
Code level 1. Written by the author; docstrings added with Claude and OpenAI Codex (October 2026).
Verification: tests/test_resampling.py checks the exact decomposition and that our k-fold code
reproduces scikit-learn's cross_val_score on the same folds.
"""

import numpy as np
from sklearn.base import clone

from regression import mse


def bootstrap_bias_variance(model, x_train, y_train, x_test, y_test, n_bootstraps=200,
                            seed=None, f_test=None):
    """Bootstrap estimate of test error, bias^2 and variance for one model.

    The model and its scaler are refitted on n_bootstraps resamples of the training set and
    evaluated on the fixed test set; error = bias2 + variance holds exactly. With f_test,
    bias2_true uses the known function instead of the noisy test targets.
    """
    rng = np.random.default_rng(seed)
    n = len(x_train)
    y_pred = np.empty((len(x_test), n_bootstraps))
    for b in range(n_bootstraps):
        idx = rng.integers(0, n, n)  # draw n points with replacement
        y_pred[:, b] = clone(model).fit(x_train[idx], y_train[idx]).predict(x_test)

    mean_pred = np.mean(y_pred, axis=1)
    result = {
        "error": np.mean(np.mean((y_test[:, None] - y_pred) ** 2, axis=1)),
        "bias2": np.mean((y_test - mean_pred) ** 2),
        "variance": np.mean(np.var(y_pred, axis=1)),
    }
    if f_test is not None:
        result["bias2_true"] = np.mean((f_test - mean_pred) ** 2)
        result["noise"] = np.mean((y_test - f_test) ** 2)
    return result


def kfold_indices(n, k, seed=None):
    """Our own fold assignment: shuffle 0..n-1 and cut into k (almost) equal folds."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    return [(np.concatenate(folds[:j] + folds[j + 1:]), folds[j]) for j in range(k)]


def kfold_cv_mse(model, x, y, folds):
    """MSE on every test fold. folds is a list of (train_idx, test_idx) pairs, e.g. from
    kfold_indices or list(KFold(...).split(x)). The model (including its scaling) is refitted on
    the training folds only, so nothing is learned from the test fold.
    """
    scores = []
    for train_idx, test_idx in folds:
        fitted = clone(model).fit(x[train_idx], y[train_idx])
        scores.append(mse(y[test_idx], fitted.predict(x[test_idx])))
    return np.array(scores)
