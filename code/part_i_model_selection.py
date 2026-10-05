"""Part i: minimum ten-fold CV selections at two noise levels, and
degree-15 known-function bias/variance from independent training sets.
Scaling is fitted within each fold. Lasso uses warm-start coordinate descent
with alpha=lambda/2; every candidate must pass dual-gap and stationarity checks.
All degrees are refitted once to preserve the documented grid/refit audit.
Writes results/part_i.json.
LLM-assisted (code level 2): OpenAI Codex added the Lasso convergence checks, removed
unreported one-SE selections, bootstrap repetition and plots and edited the docstrings,
5 October 2026.
"""

import numpy as np
from sklearn.model_selection import KFold

from regression import Scaler, checked_lasso_path, make_data, polynomial_features, runge
from settings import N_POINTS, NOISE, SEED, save_results

results = {}
degrees = np.arange(1, 21)
lam_ridge = np.logspace(-10, 0, 21)
lam_lasso = np.logspace(-1, -4, 7)            # descending, for warm starts along the path
K = 10
lasso_fits = {"total": 0, "unresolved": 0, "initial_hit_limit": 0, "refined": 0,
              "max_dual_gap": 0.0, "max_kkt_residual": 0.0, "max_sweeps": 0,
              "requested_solver_tol": 1e-6, "dual_gap_tolerance": 1e-7, "kkt_tolerance": 1e-7}


def fit_paths(x_tr, y_tr, p):
    """Scaler plus coefficient paths for OLS (1 x p), Ridge (L_r x p) and Lasso (L_l x p)."""
    X = polynomial_features(x_tr, p)
    sc = Scaler().fit(X, y_tr)
    Xs, yc = sc.transform(X), sc.center(y_tr)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    uty = U.T @ yc
    ols = ((uty / s) @ Vt)[None, :]
    ridge = ((s[None, :] / (s[None, :] ** 2 + len(yc) * lam_ridge[:, None])) * uty[None, :]) @ Vt
    coefs, checks = checked_lasso_path(Xs, yc, lam_lasso)
    for key in ("total", "unresolved", "initial_hit_limit", "refined"):
        lasso_fits[key] += checks[key]
    for key in ("max_dual_gap", "max_kkt_residual", "max_sweeps"):
        lasso_fits[key] = max(lasso_fits[key], checks[key])
    return sc, {"OLS": ols, "Ridge": ridge, "Lasso": coefs.T}


def predict(sc, thetas, x, p):
    """Predictions for every row of thetas, using the training scaler sc."""
    return sc.y_mean + sc.transform(polynomial_features(x, p)) @ thetas.T


def cross_validate(x, y, k=K, seed=SEED):
    """Ten-fold CV MSE and fold standard error for every method, degree and penalty."""
    folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(x))
    shapes = {"OLS": 1, "Ridge": len(lam_ridge), "Lasso": len(lam_lasso)}
    err = {m: np.empty((k, len(degrees), L)) for m, L in shapes.items()}
    for f, (tr, va) in enumerate(folds):
        for a, p in enumerate(degrees):
            sc, paths = fit_paths(x[tr], y[tr], int(p))
            for m, th in paths.items():
                err[m][f, a] = np.mean((y[va][:, None] - predict(sc, th, x[va], int(p))) ** 2, axis=0)
        print(f"CV fold {f + 1}/{k} checked", flush=True)
    return {m: (e.mean(0), e.std(0, ddof=1) / np.sqrt(k)) for m, e in err.items()}


def select(x, y, cv, x_new, y_new):
    """Minimum-CV choices for every method, refitted on all data and
    evaluated on fresh points.
    """
    out = {}
    lam_of = {"OLS": np.array([0.0]), "Ridge": lam_ridge, "Lasso": lam_lasso}
    full = {int(p): fit_paths(x, y, int(p)) for p in degrees}      # refits on all data, once
    for m, (mean, se) in cv.items():
        a, b = np.unravel_index(np.argmin(mean), mean.shape)
        p = int(degrees[a])
        sc, paths = full[p]
        th = paths[m][b]
        out[m] = {"min": {
            "degree": p, "lambda": lam_of[m][b], "cv_mse": mean[a, b], "cv_se": se[a, b],
            "true_test_mse": np.mean((y_new - predict(sc, th[None, :], x_new, p)[:, 0]) ** 2),
            "nonzero": int(np.sum(th != 0))}}
    return out


# ------------------------------------------------------------------------------------------
# 1) Model selection for sigma = 0.1 (main data set) and sigma = 0.3
# ------------------------------------------------------------------------------------------
for noise in (NOISE, 0.3):
    print(f"Model selection: sigma={noise}", flush=True)
    x, y = make_data(N_POINTS, noise, SEED)
    x_new, y_new = make_data(2000, noise, SEED + 1)
    cv = cross_validate(x, y)
    sel = select(x, y, cv, x_new, y_new)
    results[f"sigma={noise}"] = {"k10": sel,
                                 "cv_curves_k10": {m: {"best_over_lambda": mean.min(1),
                                                       "se_at_best": se[np.arange(len(degrees)),
                                                                        mean.argmin(1)]}
                                                   for m, (mean, se) in cv.items()},
                                 "cv_maps_k10": {m: mean for m, (mean, se) in cv.items()}}
results["grids"] = {"degrees": degrees, "lambda_ridge": lam_ridge, "lambda_lasso": lam_lasso}
results["lasso_convergence_cv"] = dict(lasso_fits)

# ------------------------------------------------------------------------------------------
# 2) Bias^2 and variance against lambda at degree 15: fresh training sets
# ------------------------------------------------------------------------------------------
P_BV, REPS = 15, 150
x_eval = np.linspace(-1, 1, 401)
f_eval = runge(x_eval)
for key in ("total", "unresolved", "initial_hit_limit", "refined",
            "max_dual_gap", "max_kkt_residual", "max_sweeps"):
    lasso_fits[key] = 0


def decompose(pred):            # pred: (n_eval, n_lambda, n_sets)
    """Squared bias (with the known f) and variance of a set of predictions."""
    mean = pred.mean(axis=2)
    return np.mean((f_eval[:, None] - mean) ** 2, axis=0), np.mean(pred.var(axis=2), axis=0)


fresh = {m: [] for m in ("OLS", "Ridge", "Lasso")}
for r in range(REPS):
    xf, yf = make_data(80, NOISE, seed=7_000_000 + r)
    sc, paths = fit_paths(xf, yf, P_BV)
    for m in fresh:
        fresh[m].append(predict(sc, paths[m], x_eval, P_BV))
    if (r + 1) % 25 == 0:
        print(f"Bias/variance: {r + 1}/{REPS} training sets checked", flush=True)
bv = {}
for m in fresh:
    b2, var = decompose(np.stack(fresh[m], axis=2))
    bv[m] = {"bias2_fresh": b2, "variance_fresh": var,
             "total_fresh": b2 + var + NOISE**2}
results["bias_variance_degree15"] = bv
results["lasso_convergence_bias_variance"] = dict(lasso_fits)

# ------------------------------------------------------------------------------------------
save_results("part_i", results)
print("lasso convergence:", results["lasso_convergence_cv"], results["lasso_convergence_bias_variance"])
print("Selected models:", {k: v["k10"] for k, v in results.items() if k.startswith("sigma=")})
