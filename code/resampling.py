"""
Resampling tools: the bootstrap bias-variance decomposition and k-fold cross-validation.

The bootstrap part follows the code in Sections 2.10-2.11 of the lecture notes / the week-36
notebook: the model is refitted on B bootstrap samples of the training set and evaluated on one
fixed test set. With predictions ytilde_b(x_i), b = 1..B, i = 1..n_test,

    error    = mean_i mean_b (y_i - ytilde_b(x_i))^2
    bias^2   = mean_i (y_i - mean_b ytilde_b(x_i))^2
    variance = mean_i var_b ytilde_b(x_i)

and error = bias^2 + variance holds exactly (an algebraic identity with variance divisor B).
Replacing f_i by noisy y_i adds the noise variance to the measured bias only in expectation
over independent test noise. On one realized test set the difference also contains a cross
term. If f_test is supplied, we return the squared discrepancy from the known function as well.

Connection to the taught methods (course-source audit by Codex, 5 October 2026)
---------------------------------------------------------------------------
- Week 36 Exercise 3 / Lecturebook Section 2.10 (PDF pages 95-96): split first, resample paired
  training rows, refit, and store one prediction column per resample at a fixed test set. The
  course example uses sklearn.utils.resample; drawing integer row indices here implements
  the same sampling-with-replacement operation and keeps each x paired with its observed y.
- Week 36 Exercise 2 / Eq. (2.52): the population decomposition averages over training sets
  and independent test noise. The empirical noisy-target identity above is exactly algebraic;
  the printed '>=' in the classroom snippet does not change that calculation into an inequality.
- Week 36 Exercise 4 / Sections 2.12 and 3.14: fit on all but one fold, evaluate that fold,
  and repeat. kfold_cv_mse implements this loop without calling cross_val_score. Part d also
  calls sklearn's cross_val_score directly, as required by Project 1, and compares identical
  fold assignments. Cloning the estimator also isolates preprocessing state between fits.
The course's fixed-grid Gaussian-bump data are replaced by the Runge data requested in Project 1;
its numerical degree minima are therefore not reference answers for this dataset.

LLM-assisted
------------
Code level 2. The functions were written by the author, based on the lecture-note code. The
docstrings and the course references above were written with LLM assistance (Claude via
Claude Code and OpenAI Codex, October 2026).
Verification: tests/test_resampling.py checks the exact decomposition and that our k-fold code
reproduces Scikit-Learn's cross_val_score for the same folds.
"""

import numpy as np
from sklearn.base import clone

from regression import mse


def bootstrap_bias_variance(model, x_train, y_train, x_test, y_test, n_bootstraps=200,
                            seed=None, f_test=None):
    """Bootstrap estimate of test error, bias^2 and variance for one model.

    model : estimator with fit(x, y) and predict(x) (cloned, so the original is untouched)
    f_test : optional noise-free values f(x_test); gives an f-based squared discrepancy.

    Following Week 36 Exercise 3, y_pred has shape (n_test, n_bootstraps): axis 1 averages
    over fitted models at a fixed x, not over distinct input locations. Our targets are 1-D,
    so y_test[:, None] supplies the column dimension that the lecture example kept with
    keepdims=True. Refitting each cloned estimator refits its training-only scaler too.

    With mean prediction m_i and test noise eps_i=y_i-f_i, bias2-bias2_true equals
    mean(eps_i**2) + 2*mean(eps_i*(f_i-m_i)), not necessarily sigma**2 for this finite sample.
    The returned variance uses NumPy's ddof=0, making error=bias2+variance exact up to roundoff.
    A finite conditional bootstrap estimate need not equal population bias/variance; f_test
    removes target noise from this diagnostic, not the sampling error in the mean prediction.
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
    """Our own fold assignment: shuffle 0..n-1 and cut into k (almost) equal folds.

    This is the shuffle/partition step in Week 36 Exercise 4 and Section 2.12. Each index
    appears in a test fold once. Passing the same integer seed to sklearn does not necessarily
    reproduce this permutation: the random-number generators can differ. For a solver check,
    pass one explicit list of fold indices to both implementations, as part d does.
    """
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    return [(np.concatenate(folds[:j] + folds[j + 1:]), folds[j]) for j in range(k)]


def kfold_cv_mse(model, x, y, folds):
    """MSE on every test fold. folds is a list of (train_idx, test_idx) pairs, e.g. from
    kfold_indices or list(KFold(...).split(x)). The model (including its scaling) is refitted on
    the training folds only, so nothing is learned from the test fold.

    This implements Week 36 Exercise 4 item 1 without delegating the fold loop to sklearn.
    The returned array contains k fold MSEs; the caller summarizes them. With equal fold sizes,
    their unweighted mean equals the pooled held-out MSE of Section 3.14, Eq. (3.92). Unequal
    folds require size weights if that pooled quantity is intended. The report's N=100 with
    k=5 or 10 has equal folds. Fold losses are dependent because training samples overlap;
    std(scores, ddof=1)/sqrt(k) is a descriptive one-SE heuristic, not a calibrated interval.
    """
    scores = []
    for train_idx, test_idx in folds:
        fitted = clone(model).fit(x[train_idx], y[train_idx])
        scores.append(mse(y[test_idx], fitted.predict(x[test_idx])))
    return np.array(scores)
