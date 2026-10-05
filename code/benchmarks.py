"""
Benchmarks quoted in Table II of the report: our code against closed-form results and
Scikit-Learn. (The same checks, with tolerances, are in tests/.)

Writes results/benchmarks.json.

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.

Codex correction, 5 October 2026: set the sklearn OLS reference cutoff to
1e-15 so dense least squares matches the custom pseudoinverse convention.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from optimizers import (AUTOGRAD_GRADIENTS, ANALYTICAL_GRADIENTS, GD, fista, gradient_descent,
                        grad_ols, hessian, make_gradient, stochastic_gradient_descent)
from regression import PolynomialRegression, Scaler, polynomial_features, ridge_parameters, runge
from resampling import bootstrap_bias_variance, kfold_cv_mse
from settings import main_split, save_results

warnings.filterwarnings("ignore", category=ConvergenceWarning)
x_tr, x_te, y_tr, y_te = main_split()
x_grid = np.linspace(-1, 1, 501)
out = {}

# OLS against Scikit-Learn's LinearRegression (predictions on a grid)
diffs = []
for p in (3, 6, 10):
    ours = PolynomialRegression(p).fit(x_tr, y_tr).predict(x_grid)
    ref = make_pipeline(PolynomialFeatures(p, include_bias=False), LinearRegression(tol=1e-15)).fit(
        x_tr[:, None], y_tr).predict(x_grid[:, None])
    diffs.append(np.max(np.abs(ours - ref)))
out["ols_vs_sklearn_max_abs_prediction"] = max(diffs)

# Ridge against Scikit-Learn's Ridge with alpha = n * lambda
diffs = []
for p, lam in ((6, 1e-3), (10, 1e-5), (15, 1e-7)):
    X = polynomial_features(x_tr, p)
    sc = Scaler().fit(X, y_tr)
    Xs, yc = sc.transform(X), sc.center(y_tr)
    ref = Ridge(alpha=len(yc) * lam, fit_intercept=False, solver="svd").fit(Xs, yc).coef_
    diffs.append(np.max(np.abs(ridge_parameters(Xs, yc, lam) - ref)))
out["ridge_vs_sklearn_max_abs_theta"] = max(diffs)

# scaling: OLS predictions with and without standardisation
out["ols_scaled_vs_centred_max_abs_prediction_p10"] = np.max(np.abs(
    PolynomialRegression(10, scale=True).fit(x_tr, y_tr).predict(x_grid)
    - PolynomialRegression(10, scale=False).fit(x_tr, y_tr).predict(x_grid)))

# analytical vs automatic gradients (relative, random points, p = 6 and 15)
rng = np.random.default_rng(0)
rel = []
for p in (6, 15):
    X = polynomial_features(x_tr, p)
    sc = Scaler().fit(X, y_tr)
    Xs, yc = sc.transform(X), sc.center(y_tr)
    for method in ("ols", "ridge", "lasso"):
        for _ in range(50):
            th = rng.normal(size=p)
            ga = ANALYTICAL_GRADIENTS[method](th, Xs, yc, 1e-3)
            gb = AUTOGRAD_GRADIENTS[method](th, Xs, yc, 1e-3)
            rel.append(np.max(np.abs(ga - gb)) / np.max(np.abs(ga)))
out["gradient_autograd_vs_analytic_max_relative"] = max(rel)

# SGD with one mini-batch per epoch is exactly full-batch GD
X = polynomial_features(x_tr, 6)
sc = Scaler().fit(X, y_tr)
Xs, yc = sc.transform(X), sc.center(y_tr)
eta = 0.9 * 2 / np.linalg.eigvalsh(hessian(Xs)).max()
th_gd, _ = gradient_descent(make_gradient("ols", Xs, yc), np.zeros(6), GD(eta), 500)
th_sgd, _ = stochastic_gradient_descent(grad_ols, Xs, yc, 0.0, np.zeros(6), GD(eta), 500, len(yc), seed=3)
out["sgd_full_batch_vs_gd_max_abs"] = np.max(np.abs(th_gd - th_sgd))

# our FISTA Lasso against Scikit-Learn's coordinate descent (degree 6, three lambdas)
diffs, same = [], []
L = np.linalg.eigvalsh(hessian(Xs)).max()
for lam in (1e-1, 1e-2, 1e-3):
    ref = Lasso(alpha=lam / 2, fit_intercept=False, max_iter=2_000_000, tol=1e-12).fit(Xs, yc).coef_
    th, _ = fista(make_gradient("ols", Xs, yc), np.zeros(6), 1 / L, lam, 1_000_000, theta_ref=ref, tol=1e-10)
    diffs.append(np.max(np.abs(th - ref)))
    same.append(bool(np.all((th == 0) == (ref == 0))))
out["fista_vs_sklearn_lasso_max_abs_theta_p6"] = max(diffs)
out["fista_vs_sklearn_lasso_same_zeros_p6"] = all(same)

# own k-fold cross-validation against cross_val_score with the same folds
from regression import make_data
from settings import N_POINTS, NOISE, SEED
x, y = make_data(N_POINTS, NOISE, SEED)
rel = []
for k in (5, 10):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    for p in (4, 8, 12):
        ours = kfold_cv_mse(PolynomialRegression(p), x, y, list(kf.split(x)))
        ref = -cross_val_score(make_pipeline(PolynomialFeatures(p, include_bias=False), StandardScaler(),
                                             LinearRegression(tol=1e-15)), x[:, None], y, cv=kf,
                               scoring="neg_mean_squared_error")
        rel.append(np.max(np.abs(ours - ref) / ref))
out["own_kfold_vs_sklearn_max_relative"] = max(rel)

# bootstrap: error = bias^2 + variance exactly
res = bootstrap_bias_variance(PolynomialRegression(8), x_tr, y_tr, x_te, y_te, 200, seed=1,
                              f_test=runge(x_te))
out["bootstrap_identity_relative"] = abs(res["error"] - res["bias2"] - res["variance"]) / res["error"]

save_results("benchmarks", out)
for k, v in out.items():
    print(f"{k:50s} {v}")
