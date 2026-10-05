"""Part e: analytical vs. Autograd gradients, their cost, and plain GD with a fixed learning
rate for OLS and Ridge at degree six, compared with the closed-form solutions.
Writes results/part_e.json.

LLM-assisted (code level 4): written with Claude (Claude Code, October 2026), simplified
on 5 October 2026.
"""

import time

import numpy as np

from optimizers import (ANALYTICAL_GRADIENTS, AUTOGRAD_GRADIENTS, COSTS, GD,
                        gradient_descent, hessian, make_gradient)
from plot_style import save_results
from regression import Scaler, main_split, ols_parameters, polynomial_features, ridge_parameters

results = {}
x_train, x_test, y_train, y_test = main_split()
TOL = 1e-6


def scaled_problem(p):
    """Standardized design matrix and centered targets for degree p."""
    X = polynomial_features(x_train, p)
    sc = Scaler().fit(X, y_train)
    return sc.transform(X), sc.center(y_train)


def spectrum(X, lam=0.0):
    """Largest and smallest Hessian eigenvalue and their ratio (the condition number)."""
    ev = np.linalg.eigvalsh(hessian(X, lam))
    return ev.max(), ev.min(), ev.max() / ev.min()


# ------------------------------------------------------------------------------------------
# 1) Analytical vs automatic gradients for OLS and Ridge
# ------------------------------------------------------------------------------------------
rng = np.random.default_rng(1)
grad_check = {}
for p in (6, 15):
    Xp, yp = scaled_problem(p)
    for method, lam in (("ols", 0.0), ("ridge", 1e-3)):
        rel = []
        for _ in range(100):
            theta = rng.normal(size=p)
            ga = ANALYTICAL_GRADIENTS[method](theta, Xp, yp, lam)
            gauto = AUTOGRAD_GRADIENTS[method](theta, Xp, yp, lam)
            rel.append(np.max(np.abs(ga - gauto)) / np.max(np.abs(ga)))
        grad_check[f"p={p},{method}"] = max(rel)
results["gradient_check_max_rel"] = grad_check


def best_time(fun, number, repeat=7):
    """Best average wall time of fun() over repeat timing loops of number calls."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fun()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


# cost of one AD gradient relative to one cost evaluation, for a small and a large data set
timing = {}
for n_big, number in ((80, 200), (10**6, 5)):
    Xb = rng.normal(size=(n_big, 6))
    yb = rng.normal(size=n_big)
    th = rng.normal(size=6)
    t_cost = best_time(lambda: COSTS["ols"](th, Xb, yb), number)
    t_auto = best_time(lambda: AUTOGRAD_GRADIENTS["ols"](th, Xb, yb, 0.0), number)
    timing[n_big] = {"cost_s": t_cost, "autograd_s": t_auto, "autograd_over_cost": t_auto / t_cost}
results["timing"] = timing

# ------------------------------------------------------------------------------------------
# 2) Plain GD at degree 6: optimal learning rate, both gradients, a few other rates
# ------------------------------------------------------------------------------------------
P = 6
X, y = scaled_problem(P)
theta_ols = ols_parameters(X, y)
lmax, lmin, kappa = spectrum(X)
eta_max, eta_opt = 2 / lmax, 2 / (lmax + lmin)
info6 = {"degree": P, "lambda_max": lmax, "lambda_min": lmin, "kappa": kappa,
         "eta_max": eta_max, "eta_opt": eta_opt,
         "estimate_kappa_half_log": 0.5 * (kappa + 1) * np.log(1 / TOL)}

thetas = {}
for name, auto in (("analytic", False), ("autograd", True)):
    th, inf = gradient_descent(make_gradient("ols", X, y, use_autograd=auto), np.zeros(P),
                               GD(eta_opt), 10**6, theta_ref=theta_ols, tol=TOL)
    thetas[name] = th
    info6[f"gd_{name}_iterations"] = inf["iterations"]
info6["max_theta_difference_analytic_vs_autograd"] = np.max(np.abs(thetas["analytic"] -
                                                                  thetas["autograd"]))

# fractions of the stability bound 2/h_max: slower below it, divergence above it
fractions = [0.25, 0.5, 0.9, 0.99, 1.01]
scan = {}
for frac in fractions:
    _, inf = gradient_descent(make_gradient("ols", X, y), np.zeros(P), GD(frac * eta_max), 10**6,
                              theta_ref=theta_ols, tol=TOL)
    scan[frac] = {"iterations": inf["iterations"], "converged": inf["converged"],
                  "diverged": inf["diverged"]}
info6["learning_rate_scan"] = scan

# Ridge: the penalty adds 2*lam to every Hessian eigenvalue and lowers the condition number
ridge_table = []
for lam in (1e-4, 1e-3, 1e-2, 1e-1):
    lM, lm, kp = spectrum(X, lam)
    e_opt = 2 / (lM + lm)
    _, inf = gradient_descent(make_gradient("ridge", X, y, lam), np.zeros(P), GD(e_opt), 10**6,
                              theta_ref=ridge_parameters(X, y, lam), tol=TOL)
    ridge_table.append({"lambda": lam, "kappa": kp, "eta_opt": e_opt,
                        "iterations": inf["iterations"]})
info6["ridge_table"] = ridge_table
results["degree6"] = info6

save_results("part_e", results)
print("max relative gradient difference:", grad_check)
print("AD gradient time / cost time:", {k: round(v["autograd_over_cost"], 2) for k, v in timing.items()})
print(f"degree 6: kappa {kappa:.0f}, eta_max {eta_max:.3f}, GD iterations {info6['gd_analytic_iterations']} "
      f"(estimate {info6['estimate_kappa_half_log']:.0f})")
print("learning-rate scan:", scan)
print("Ridge:", ridge_table)
