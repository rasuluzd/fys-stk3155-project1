"""Tests of the bootstrap and cross-validation code.

LLM-assisted (code level 4): generated with Claude (Anthropic, Claude Code), October 2026.

Codex correction, 5 October 2026: set the sklearn OLS reference cutoff to
1e-15 so dense least squares matches the custom pseudoinverse convention.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from regression import PolynomialRegression, make_data, runge
from resampling import bootstrap_bias_variance, kfold_cv_mse, kfold_indices

x, y = make_data(100, noise=0.1, seed=7)


def test_bootstrap_decomposition_is_exact():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    res = bootstrap_bias_variance(PolynomialRegression(5), x[:80], y[:80], x[80:], y[80:],
                                  n_bootstraps=50, seed=0, f_test=runge(x[80:]))
    assert np.isclose(res["error"], res["bias2"] + res["variance"], rtol=1e-12)


def test_own_kfold_reproduces_sklearn_ols():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    ours = kfold_cv_mse(PolynomialRegression(7), x, y, list(kf.split(x)))
    pipe = make_pipeline(PolynomialFeatures(7, include_bias=False), StandardScaler(),
                         LinearRegression(tol=1e-15))
    ref = -cross_val_score(pipe, x[:, None], y, cv=kf, scoring="neg_mean_squared_error")
    assert np.allclose(ours, ref, rtol=1e-8)


def test_own_kfold_reproduces_sklearn_ridge():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    lam, k = 1e-3, 10
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    n_train = len(x) * (k - 1) // k  # 90 points in every training fold
    ours = kfold_cv_mse(PolynomialRegression(9, "ridge", lam), x, y, list(kf.split(x)))
    pipe = make_pipeline(PolynomialFeatures(9, include_bias=False), StandardScaler(),
                         Ridge(alpha=n_train * lam))
    ref = -cross_val_score(pipe, x[:, None], y, cv=kf, scoring="neg_mean_squared_error")
    assert np.allclose(ours, ref, rtol=1e-8)


def test_own_fold_assignment_is_a_partition():
    """LLM-assisted: generated with Claude (Claude Code, October 2026)."""
    folds = kfold_indices(103, 5, seed=0)
    test_all = np.sort(np.concatenate([te for _, te in folds]))
    assert np.array_equal(test_all, np.arange(103))
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == 103
