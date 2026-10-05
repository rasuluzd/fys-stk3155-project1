"""Part g: Autograd's subgradient convention and fixed-budget OLS/Ridge/Lasso
comparisons. Writes results/part_g.json. lasso_course_comparison.py compares
all five taught updates and supplies the report's Lasso table.
LLM-assisted (code level 2): OpenAI Codex removed optional proximal solvers and their
unreported experiments and edited the docstrings, 5 October 2026.
"""


import numpy as np
import autograd.numpy as anp
from autograd import grad

from optimizers import Adam, GD, gradient_descent, hessian, make_gradient
from regression import (Scaler, lasso_sklearn_parameters, mse, ols_parameters, polynomial_features,
                        ridge_parameters)
from settings import main_split, save_results

results = {}
x_train, x_test, y_train, y_test = main_split()
n = len(x_train)

# what does automatic differentiation say about d|t|/dt at t = 0?
d_abs = grad(lambda t: anp.abs(t))
results["autograd_abs_derivative"] = {"at_0": d_abs(0.0), "at_+0.3": d_abs(0.3), "at_-0.3": d_abs(-0.3)}


def scaled_problem(p):
    """Standardized design matrix, centered targets and fitted scaler for degree p."""
    X_raw = polynomial_features(x_train, p)
    sc = Scaler().fit(X_raw, y_train)
    return sc.transform(X_raw), sc.center(y_train), sc


P = 10
X, y, sc = scaled_problem(P)
X_test = sc.transform(polynomial_features(x_test, P))
L = np.linalg.eigvalsh(hessian(X)).max()          # Lipschitz constant of the MSE gradient
lam_max = (2.0 / n) * np.max(np.abs(X.T @ y))     # smallest lambda giving theta = 0
results["setup"] = {"degree": P, "lipschitz_L": L, "lambda_max": lam_max}

# ------------------------------------------------------------------------------------------
# OLS / Ridge / Lasso at degree 10 obtained with gradient methods vs. closed forms
# ------------------------------------------------------------------------------------------
lam = 1e-3
ref = lasso_sklearn_parameters(X, y, lam, max_iter=2_000_000, tol=1e-12)
results["reference_lasso_theta"] = ref
lam_r = 1e-4
targets = {"OLS": ("ols", 0.0, ols_parameters(X, y)),
           "Ridge": ("ridge", lam_r, ridge_parameters(X, y, lam_r)),
           "Lasso": ("lasso", lam, ref)}
compare = {}
for name, (kind, lam_k, theta_ref) in targets.items():
    row = {"reference_test_mse": mse(y_test, sc.y_mean + X_test @ theta_ref)}
    for label, opt in (("gd", GD(1.0 / L)), ("adam", Adam(0.01))):
        th, _ = gradient_descent(make_gradient(kind, X, y, lam_k), np.zeros(P), opt, 10_000)
        row[f"{label}_test_mse"] = mse(y_test, sc.y_mean + X_test @ th)
        row[f"{label}_relative_error"] = np.linalg.norm(th - theta_ref) / np.linalg.norm(theta_ref)
    compare[name] = row
results["methods_degree10"] = {"lambda_ridge": lam_r, "lambda_lasso": lam, "iterations": 10_000,
                               "rows": compare}

# ------------------------------------------------------------------------------------------
save_results("part_g", results)
print("Autograd derivative of absolute value:", results["autograd_abs_derivative"])
print("Degree-10 comparisons:", compare)
