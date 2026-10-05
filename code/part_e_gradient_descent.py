"""
Part e): plain gradient descent with a fixed learning rate for OLS and Ridge, with the gradient
computed analytically and by automatic differentiation (autograd).

Produces
  figures/e_learning_rate.pdf  (a) error against iteration for several eta, (b) iterations to reach
                               a relative error 1e-6 against eta/eta_max (measured and theory)
  figures/e_kappa.pdf          (a) iterations against the condition number for degrees 2-9,
                               (b) GD iterates at degree 15 compared with Ridge, lambda = 1/(2 eta k)
  results/part_e.json

All runs use the main training set (80 points), standardised features and centred targets.

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from optimizers import (ANALYTICAL_GRADIENTS, AUTOGRAD_GRADIENTS, COSTS, GD, Momentum,
                        gradient_descent, hessian, make_gradient)
from plot_style import COLORS, GREY, INK, DOUBLE, METHOD_COLORS, ordered_colors, save, set_style
from regression import Scaler, ols_parameters, polynomial_features, ridge_parameters, runge
from settings import NOISE, main_split, save_results

set_style()
results = {}
x_train, x_test, y_train, y_test = main_split()
n = len(x_train)
TOL = 1e-6
x_dense = np.linspace(-1, 1, 2001)


def scaled_problem(p):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    X = polynomial_features(x_train, p)
    sc = Scaler().fit(X, y_train)
    return sc.transform(X), sc.center(y_train), sc


def spectrum(X, lam=0.0):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    ev = np.linalg.eigvalsh(hessian(X, lam))
    return ev.max(), ev.min(), ev.max() / ev.min()


def theory_iterations(X, theta_star, eta, lam=0.0, tol=TOL, kmax=10**8):
    """Exact iteration count for GD from theta=0 on the quadratic cost: the error in eigen-
    direction i is multiplied by (1 - eta h_i) per step, so ||e_k||^2 = sum_i r_i^(2k) e0_i^2.

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
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
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
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

fractions = [0.05, 0.25, 0.5, 0.99, 1.002]
curves = {}
for frac in fractions:
    _, inf = gradient_descent(make_gradient("ols", X, y), np.zeros(P), GD(frac * eta_max), 60_000,
                              theta_ref=theta_ols, tol=1e-12)
    curves[frac] = inf["errors"]
_, inf = gradient_descent(make_gradient("ols", X, y), np.zeros(P), GD(eta_opt), 60_000,
                          theta_ref=theta_ols, tol=1e-12)
curves["opt"] = inf["errors"]

scan = np.concatenate([np.logspace(-2, np.log10(0.95), 30), np.linspace(0.96, 1.04, 17)])
measured, theory = [], []
for frac in scan:
    _, inf = gradient_descent(make_gradient("ols", X, y), np.zeros(P), GD(frac * eta_max), 2 * 10**6,
                              theta_ref=theta_ols, tol=TOL)
    measured.append(inf["iterations"] if inf["converged"] else np.nan)
    theory.append(theory_iterations(X, theta_ols, frac * eta_max))
info6.update({"scan_fraction": scan, "scan_measured": measured, "scan_theory": theory})


def empirical_threshold(X, y, lam, theta_star, lo=0.9, hi=1.1, iters=50_000):
    """Bisection for the largest eta/eta_max for which the relative error after `iters`
    iterations is below 1 (i.e. GD does not blow up).

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    eta_m = 2 / spectrum(X, lam)[0]
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        _, inf = gradient_descent(make_gradient("ridge", X, y, lam), np.zeros(X.shape[1]),
                                  GD(mid * eta_m), iters, theta_ref=theta_star)
        if inf["diverged"] or inf["errors"][-1] > 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


ridge_table = []
for lam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
    ts = ridge_parameters(X, y, lam) if lam > 0 else theta_ols
    lM, lm, kp = spectrum(X, lam)
    e_opt = 2 / (lM + lm)
    _, inf = gradient_descent(make_gradient("ridge", X, y, lam), np.zeros(P), GD(e_opt), 10**6,
                              theta_ref=ts, tol=TOL)
    ridge_table.append({"lambda": lam, "lambda_max": lM, "lambda_min": lm, "kappa": kp,
                        "eta_max": 2 / lM, "eta_opt": e_opt, "iterations": inf["iterations"],
                        "theory_iterations": theory_iterations(X, ts, e_opt, lam),
                        "threshold_ratio": empirical_threshold(X, y, lam, ts)})
info6["ridge_table"] = ridge_table
results["degree6"] = info6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
cols = ordered_colors(4, 0.35, 1.0)
for c, frac in zip(cols, fractions[:4]):
    e = curves[frac]
    k = np.arange(1, len(e) + 1)
    ax1.loglog(k, e, color=c, lw=1.2, label=rf"$\eta={frac:g}\,\eta_{{\max}}$")
e = curves["opt"]
ax1.loglog(np.arange(1, len(e) + 1), e, color=COLORS[1], lw=1.4, ls="--",
           label=r"$\eta^*=2/(h_{\max}+h_{\min})$")
e = curves[1.002]
ax1.loglog(np.arange(1, len(e) + 1), e, color=COLORS[7], lw=1.2, ls=":", label=r"$\eta=1.002\,\eta_{\max}$")
ax1.axhline(TOL, color=GREY, lw=0.8, ls=":")
ax1.set_ylim(1e-9, 1e2)
ax1.set_xlim(1, 6e4)
ax1.set_xlabel("Iteration $k$")
ax1.set_ylabel(r"$\|\theta_k-\hat\theta_{\mathrm{OLS}}\|/\|\hat\theta_{\mathrm{OLS}}\|$")
ax1.legend(fontsize=5.6, loc="lower left")
ax2.loglog(scan, theory, color=INK, lw=1.0, label="Theory (exact for a quadratic cost)")
ax2.loglog(scan, measured, "o", ms=3, mfc="none", color=METHOD_COLORS["OLS"], label="Measured")
ax2.axvline(1.0, color=COLORS[7], lw=0.9, ls="--")
ax2.text(1.02, 2e5, "divergence\n" + r"$\eta>2/h_{\max}$", fontsize=6, color=COLORS[7])
ax2.set_xlim(8e-3, 1.6)
ax2.set_xlabel(r"$\eta/\eta_{\max}$,  $\eta_{\max}=2/h_{\max}$")
ax2.set_ylabel(r"Iterations to relative error $10^{-6}$")
ax2.legend(fontsize=6, loc="upper right")
ax1.text(0.03, 0.96, "(a)", transform=ax1.transAxes, va="top", fontweight="bold")
ax2.text(0.03, 0.04, "(b)", transform=ax2.transAxes, va="bottom", fontweight="bold")
fig.tight_layout()
save(fig, "e_learning_rate")

# ------------------------------------------------------------------------------------------
# 3) Iterations against the condition number, degrees 2-9
# ------------------------------------------------------------------------------------------
kappa_rows = []
for p in range(2, 10):
    Xp, yp, _ = scaled_problem(p)
    ts = ols_parameters(Xp, yp)
    lM, lm, kp = spectrum(Xp)
    e_opt = 2 / (lM + lm)
    _, gd_inf = gradient_descent(make_gradient("ols", Xp, yp), np.zeros(p), GD(e_opt), 4 * 10**6,
                                 theta_ref=ts, tol=TOL)
    _, m9_inf = gradient_descent(make_gradient("ols", Xp, yp), np.zeros(p), Momentum(e_opt, 0.9),
                                 4 * 10**6, theta_ref=ts, tol=TOL)
    eta_hb = 4 / (np.sqrt(lM) + np.sqrt(lm)) ** 2
    beta_hb = ((np.sqrt(kp) - 1) / (np.sqrt(kp) + 1)) ** 2
    _, hb_inf = gradient_descent(make_gradient("ols", Xp, yp), np.zeros(p), Momentum(eta_hb, beta_hb),
                                 4 * 10**6, theta_ref=ts, tol=TOL)
    kappa_rows.append({"degree": p, "kappa": kp, "gd": gd_inf["iterations"],
                       "gd_converged": gd_inf["converged"], "momentum09": m9_inf["iterations"],
                       "momentum_opt": hb_inf["iterations"], "beta_opt": beta_hb, "eta_opt_hb": eta_hb})
results["kappa_scaling"] = kappa_rows

# ------------------------------------------------------------------------------------------
# 4) Degree 15: GD iterates compared with Ridge at lambda_eff = 1/(2 eta k)
# ------------------------------------------------------------------------------------------
X15, y15, sc15 = scaled_problem(15)
lM15, lm15, kp15 = spectrum(X15)
eta15 = 1.0 / lM15
theta_ols15 = ols_parameters(X15, y15)
X15_dense = sc15.transform(polynomial_features(x_dense, 15))
f_dense = runge(x_dense)


def true_mse(theta):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    return np.mean((f_dense - (sc15.y_mean + X15_dense @ theta)) ** 2) + NOISE**2


K = 10**6
checkpoints = np.unique(np.logspace(0, 6, 61).astype(int))
grad15 = make_gradient("ols", X15, y15)
theta = np.zeros(15)
gd_mse, gd_k = [], []
c_idx = 0
for k in range(1, K + 1):
    theta = theta - eta15 * grad15(theta)
    if k == checkpoints[c_idx]:
        gd_mse.append(true_mse(theta))
        gd_k.append(k)
        c_idx += 1
        if c_idx == len(checkpoints):
            break
# check against the closed-form GD iterate theta_k = V diag(1 - (1 - eta h)^k) V^T theta_ols
h, V = np.linalg.eigh(hessian(X15))
theta_formula = V @ ((1 - (1 - eta15 * h) ** K) * (V.T @ theta_ols15))
gd_k = np.array(gd_k)
lam_eff = 1.0 / (2 * eta15 * gd_k)
ridge_mse = [true_mse(ridge_parameters(X15, y15, lam)) for lam in lam_eff]
lam_grid = np.logspace(-10, 1, 89)
ridge_curve = [true_mse(ridge_parameters(X15, y15, lam)) for lam in lam_grid]
results["early_stopping_degree15"] = {
    "kappa": kp15, "eta": eta15, "k": gd_k, "lambda_eff": lam_eff, "gd_true_mse": gd_mse,
    "ridge_true_mse_at_lambda_eff": ridge_mse, "ols_true_mse": true_mse(theta_ols15),
    "max_abs_gd_vs_formula": np.max(np.abs(theta - theta_formula)),
    "ridge_best_lambda_true": lam_grid[np.argmin(ridge_curve)], "ridge_best_true_mse": min(ridge_curve),
    "gd_best_true_mse": min(gd_mse), "gd_best_k": int(gd_k[np.argmin(gd_mse)]),
    "relative_error_after_1e6": np.linalg.norm(theta - theta_ols15) / np.linalg.norm(theta_ols15)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
kap = np.array([r["kappa"] for r in kappa_rows])
ax1.loglog(kap, [r["gd"] for r in kappa_rows], "o-", ms=3.5, color=COLORS[0], label=r"Plain GD, $\eta^*$")
ax1.loglog(kap, [r["momentum09"] for r in kappa_rows], "s-", ms=3.5, color=COLORS[1],
           label=r"Momentum $\beta=0.9$, $\eta^*$")
ax1.loglog(kap, [r["momentum_opt"] for r in kappa_rows], "^-", ms=3.5, color=COLORS[2],
           label=r"Momentum, optimal $(\eta,\beta)$")
kk = np.logspace(0, np.log10(kap.max()) + 0.2, 20)
ax1.loglog(kk, 0.5 * kk * np.log(1 / TOL), color=GREY, lw=0.8, ls="--", label=r"$\frac{\kappa}{2}\ln 10^{6}$")
ax1.loglog(kk, 0.5 * np.sqrt(kk) * np.log(1 / TOL), color=GREY, lw=0.8, ls=":",
           label=r"$\frac{\sqrt{\kappa}}{2}\ln 10^{6}$")
for r in kappa_rows:
    ax1.annotate(f"$p={r['degree']}$", (r["kappa"], r["gd"]), textcoords="offset points",
                 xytext=(-6, 5), fontsize=5.5, color=GREY)
ax1.set_xlabel(r"Condition number $\kappa=h_{\max}/h_{\min}$ of the Hessian")
ax1.set_ylabel(r"Iterations to relative error $10^{-6}$")
ax1.legend(fontsize=5.6, loc="upper left")
ax2.semilogx(lam_grid, ridge_curve, color=METHOD_COLORS["Ridge"], lw=1.4, label=r"Ridge, $\lambda$")
ax2.semilogx(lam_eff, gd_mse, "o", ms=3, mfc="none", color=METHOD_COLORS["OLS"],
             label=r"GD after $k$ steps, $\lambda_{\mathrm{eff}}=1/(2\eta k)$")
ax2.axhline(true_mse(theta_ols15), color=METHOD_COLORS["OLS"], lw=0.9, ls="--", label="OLS (closed form)")
ax2.axhline(NOISE**2, color=GREY, lw=0.8, ls=":")
ax2.set_yscale("log")
ax2.set_xlim(1e-10, 10)
ax2.invert_xaxis()
ax2.set_xlabel(r"$\lambda$  (decreasing: less regularization / more iterations)")
ax2.set_ylabel(r"True MSE $\langle (f-\tilde y)^2\rangle+\sigma^2$")
ax2.legend(fontsize=5.6, loc="upper right")
for ax, lab in zip((ax1, ax2), ("(a)", "(b)")):
    ax.text(0.97 if ax is ax1 else 0.03, 0.04, lab, transform=ax.transAxes, va="bottom",
            ha="right" if ax is ax1 else "left", fontweight="bold")
fig.tight_layout()
save(fig, "e_kappa")

save_results("part_e", results)
print("gradient check:", {k: (f"{v['max_abs']:.2e}", f"{v['max_rel']:.2e}") if isinstance(v, dict) else v
                          for k, v in grad_check.items()})
print("timing:", {k: (f"{v['autograd_over_cost']:.1f}", f"{v['analytic_over_cost']:.1f}") for k, v in timing.items()})
print("degree 6:", {k: v for k, v in info6.items() if not isinstance(v, (list, np.ndarray)) and k != "ridge_table"})
for r in ridge_table:
    print("  ridge", r)
for r in kappa_rows:
    print("  kappa", r)
es = results["early_stopping_degree15"]
print("early stopping: kappa15", f"{kp15:.3g}", "gd best", es["gd_best_true_mse"], "at k", es["gd_best_k"],
      "ridge best", es["ridge_best_true_mse"], "lam", es["ridge_best_lambda_true"], "ols", es["ols_true_mse"],
      "formula check", es["max_abs_gd_vs_formula"], "rel err after 1e6", es["relative_error_after_1e6"])
