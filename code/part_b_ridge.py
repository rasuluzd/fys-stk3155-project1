"""Part b: Ridge degree/penalty grid and degree-15 singular-value diagnostics.

Writes results/part_b.json. ridge_supplement.py supplies the two retained plots
and the paired sample-size/noise comparisons.
LLM-assisted: original Claude implementation; Codex removed unreported exploration, 5 October 2026.
"""

import numpy as np

from regression import (PolynomialRegression, Scaler, mse, polynomial_features,
                        ridge_shrinkage_factors)
from settings import main_split, save_results

results = {}
x_train, x_test, y_train, y_test = main_split()
n = len(x_train)


def ridge_path_predictions(x_tr, y_tr, x_te, degree, lams):
    """Test predictions for all lambdas at once, using one SVD of the scaled design matrix.

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    X = polynomial_features(x_tr, degree)
    sc = Scaler().fit(X, y_tr)
    Xs, yc = sc.transform(X), sc.center(y_tr)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    uty = U.T @ yc
    factors = s[None, :] / (s[None, :] ** 2 + len(y_tr) * np.asarray(lams)[:, None])
    thetas = (factors * uty[None, :]) @ Vt                 # shape (n_lambda, degree)
    return sc.y_mean + sc.transform(polynomial_features(x_te, degree)) @ thetas.T, thetas


# ------------------------------------------------------------------------------------------
# 1) Main data set: test MSE against degree for a few lambdas
# ------------------------------------------------------------------------------------------
degrees = np.arange(1, 21)
lam_show = [1e-7, 1e-5, 1e-3, 1e-1]
lams = np.logspace(-10, 1, 45)
main = {"lambdas": lams, "degrees": degrees}
test_map = np.empty((len(degrees), len(lams)))
train_map = np.empty_like(test_map)
for k, p in enumerate(degrees):
    pred_te, _ = ridge_path_predictions(x_train, y_train, x_test, p, lams)
    pred_tr, _ = ridge_path_predictions(x_train, y_train, x_train, p, lams)
    test_map[k] = np.mean((y_test[:, None] - pred_te) ** 2, axis=0)
    train_map[k] = np.mean((y_train[:, None] - pred_tr) ** 2, axis=0)
ols_test = [mse(y_test, PolynomialRegression(int(p)).fit(x_train, y_train).predict(x_test))
            for p in degrees]
kb, lb = np.unravel_index(np.argmin(test_map), test_map.shape)
main.update({"test_mse": test_map, "train_mse": train_map, "ols_test_mse": ols_test,
             "best_degree": int(degrees[kb]), "best_lambda": lams[lb],
             "best_test_mse": test_map[kb, lb],
             "best_r2": 1 - test_map[kb, lb] / np.var(y_test),
             "ols_best_degree": int(degrees[np.argmin(ols_test)]),
             "ols_best_test_mse": float(np.min(ols_test))})
for lam in lam_show:
    j = np.argmin(np.abs(np.log(lams / lam)))
    main[f"test_mse_lambda_{lam:g}"] = test_map[:, j]
results["main"] = main

# ------------------------------------------------------------------------------------------
# 3) SVD picture at degree 15: signal vs noise in each singular mode, and the shrinkage
# ------------------------------------------------------------------------------------------
p_svd = 15
X = polynomial_features(x_train, p_svd)
sc = Scaler().fit(X, y_train)
Xs, yc = sc.transform(X), sc.center(y_train)
U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
uty = U.T @ yc
results["svd_degree15"] = {
    "singular_values": s, "abs_uty": np.abs(uty),
    "ols_mode_coefficients": np.abs(uty) / s,
    "hessian_eigenvalues": 2 * s**2 / n,
    "shrinkage": {f"{lam:g}": ridge_shrinkage_factors(Xs, lam) for lam in lam_show}}

save_results("part_b", results)
print("Main best (degree, penalty, test MSE):", main["best_degree"], main["best_lambda"], main["best_test_mse"])
print("Degree-15 singular values:", s)
