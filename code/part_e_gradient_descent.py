"""Part e: analytical/Autograd gradients, cost timings, GD learning-rate scan
and Ridge conditioning at degree six. Writes results/part_e.json.
verification_supplement.py also runs the analytical/Autograd Ridge comparison.
LLM-assisted (code level 2): OpenAI Codex removed unreported spectral early-stopping and
extended conditioning experiments and edited the docstrings, 5 October 2026.
"""

import time

import numpy as np

from optimizers import (ANALYTICAL_GRADIENTS, AUTOGRAD_GRADIENTS, COSTS, GD,
                        gradient_descent, hessian, make_gradient)
from regression import Scaler, ols_parameters, polynomial_features, ridge_parameters
from settings import main_split, save_results

results = {}
x_train, x_test, y_train, y_test = main_split()
n = len(x_train)
TOL = 1e-6


def scaled_problem(p):
    """Standardized design matrix, centered targets and fitted scaler for degree p."""
    X = polynomial_features(x_train, p)
    sc = Scaler().fit(X, y_train)
    return sc.transform(X), sc.center(y_train), sc


def spectrum(X, lam=0.0):
    """Largest and smallest Hessian eigenvalue and their ratio (the condition number)."""
    ev = np.linalg.eigvalsh(hessian(X, lam))
    return ev.max(), ev.min(), ev.max() / ev.min()


def theory_iterations(X, theta_star, eta, lam=0.0, tol=TOL, kmax=10**8):
    """Exact iteration count for GD from theta=0 on the quadratic cost: the error in eigen-
    direction i is multiplied by (1 - eta h_i) per step, so ||e_k||^2 = sum_i r_i^(2k) e0_i^2.
    """
    h, V = np.linalg.eigh(hessian(X, lam))
    e0 = V.T @ (-theta_star)
    r = np.abs(1.0 - eta * h)
    if r.max() >= 1.0:
        return np.inf
    target = (tol * np.linalg.norm(theta_star)) ** 2
    lo, hi = 0, 1
    while np.sum(r ** (2 * hi) * e0**2) > target:
        lo, hi = hi, hi * 2
        if hi > kmax:
            return np.inf
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if np.sum(r ** (2 * mid) * e0**2) > target:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------------------------------
# 1) Analytical vs automatic gradients
# ------------------------------------------------------------------------------------------
rng = np.random.default_rng(1)
grad_check = {}
for p in (6, 15):
    Xp, yp, _ = scaled_problem(p)
    for method, lam in (("ols", 0.0), ("ridge", 1e-3), ("lasso", 1e-3)):
        abs_d, rel_d = [], []
        for _ in range(100):
            theta = rng.normal(size=p)
            ga = ANALYTICAL_GRADIENTS[method](theta, Xp, yp, lam)
            gauto = AUTOGRAD_GRADIENTS[method](theta, Xp, yp, lam)
            abs_d.append(np.max(np.abs(ga - gauto)))
            rel_d.append(np.max(np.abs(ga - gauto)) / np.max(np.abs(ga)))
        grad_check[f"p={p},{method}"] = {"max_abs": max(abs_d), "max_rel": max(rel_d)}
grad_check["machine_epsilon"] = np.finfo(float).eps
results["gradient_check"] = grad_check


def best_time(fun, repeat=7, number=20):
    """Best average wall time of fun() over repeat timing loops of number calls."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fun()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


timing = {}
for n_big in (80, 10**4, 10**6):
    Xb = rng.normal(size=(n_big, 6))
    yb = rng.normal(size=n_big)
    th = rng.normal(size=6)
    number = 200 if n_big <= 10**4 else 5
    t_cost = best_time(lambda: COSTS["ols"](th, Xb, yb), number=number)
    t_auto = best_time(lambda: AUTOGRAD_GRADIENTS["ols"](th, Xb, yb, 0.0), number=number)
    t_ana = best_time(lambda: ANALYTICAL_GRADIENTS["ols"](th, Xb, yb), number=number)
    timing[n_big] = {"cost_s": t_cost, "autograd_s": t_auto, "analytic_s": t_ana,
                     "autograd_over_cost": t_auto / t_cost, "analytic_over_cost": t_ana / t_cost}
results["timing"] = timing

# ------------------------------------------------------------------------------------------
# 2) Learning rate at degree 6: convergence curves and iteration counts
# ------------------------------------------------------------------------------------------
P = 6
X, y, sc = scaled_problem(P)
theta_ols = ols_parameters(X, y)
lmax, lmin, kappa = spectrum(X)
eta_max, eta_opt = 2 / lmax, 2 / (lmax + lmin)
info6 = {"degree": P, "lambda_max": lmax, "lambda_min": lmin, "kappa": kappa,
         "eta_max": eta_max, "eta_opt": eta_opt}

# both gradients inside the same GD code
runs = {}
for name, auto in (("analytic", False), ("autograd", True)):
    t0 = time.perf_counter()
    th, inf = gradient_descent(make_gradient("ols", X, y, use_autograd=auto), np.zeros(P),
                               GD(eta_opt), 10**6, theta_ref=theta_ols, tol=TOL)
    runs[name] = {"theta": th, "iterations": inf["iterations"], "time_s": time.perf_counter() - t0}
info6["gd_analytic_iterations"] = runs["analytic"]["iterations"]
info6["gd_autograd_iterations"] = runs["autograd"]["iterations"]
info6["gd_analytic_time_s"] = runs["analytic"]["time_s"]
info6["gd_autograd_time_s"] = runs["autograd"]["time_s"]
info6["max_theta_difference_analytic_vs_autograd"] = np.max(np.abs(runs["analytic"]["theta"] -
                                                                  runs["autograd"]["theta"]))
info6["theory_iterations_eta_opt"] = theory_iterations(X, theta_ols, eta_opt)
info6["simple_estimate_kappa_half_log"] = 0.5 * (kappa + 1) * np.log(1 / TOL)

scan = np.concatenate([np.logspace(-2, np.log10(0.95), 30), np.linspace(0.96, 1.04, 17)])
measured, theory = [], []
for frac in scan:
    _, inf = gradient_descent(make_gradient("ols", X, y), np.zeros(P), GD(frac * eta_max), 2 * 10**6,
                              theta_ref=theta_ols, tol=TOL)
    measured.append(inf["iterations"] if inf["converged"] else np.nan)
    theory.append(theory_iterations(X, theta_ols, frac * eta_max))
info6.update({"scan_fraction": scan, "scan_measured": measured, "scan_theory": theory})


ridge_table = []
for lam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
    ts = ridge_parameters(X, y, lam) if lam > 0 else theta_ols
    lM, lm, kp = spectrum(X, lam)
    e_opt = 2 / (lM + lm)
    _, inf = gradient_descent(make_gradient("ridge", X, y, lam), np.zeros(P), GD(e_opt), 10**6,
                              theta_ref=ts, tol=TOL)
    ridge_table.append({"lambda": lam, "lambda_max": lM, "lambda_min": lm, "kappa": kp,
                        "eta_max": 2 / lM, "eta_opt": e_opt, "iterations": inf["iterations"],
                        "theory_iterations": theory_iterations(X, ts, e_opt, lam)})
info6["ridge_table"] = ridge_table
results["degree6"] = info6

save_results("part_e", results)
print("Gradient checks:", grad_check)
print("Degree-6 GD:", info6)
print("AD/objective timings:", timing)
