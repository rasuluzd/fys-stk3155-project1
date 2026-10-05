"""
Part g): Lasso regression with our own gradient methods.

The l1 term is not differentiable at theta_j = 0. We compare
  (i)   subgradient descent, using sign(0) = 0 (this is also what autograd returns),
  (ii)  the same subgradient fed to Adam,
  (iii) proximal gradient descent (ISTA): a GD step on the MSE followed by soft thresholding,
  (iv)  its accelerated (momentum) version FISTA,
with Scikit-Learn's coordinate-descent Lasso (alpha = lambda/2) as the reference.

Produces
  figures/g_lasso.pdf   (a) Lasso path at degree 10 (Scikit-Learn lines, our FISTA markers),
                        (b) convergence of the four methods at lambda = 1e-3
  results/part_g.json

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import grad
from sklearn.exceptions import ConvergenceWarning

from optimizers import Adam, GD, cost_lasso, fista, gradient_descent, hessian, make_gradient, soft_threshold
from plot_style import COLORS, GREY, INK, DOUBLE, METHOD_COLORS, ordered_colors, save, set_style
from regression import (Scaler, lasso_sklearn_parameters, mse, ols_parameters, polynomial_features,
                        ridge_parameters)
from settings import main_split, save_results

set_style()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
results = {}
x_train, x_test, y_train, y_test = main_split()
n = len(x_train)

# what does automatic differentiation say about d|t|/dt at t = 0?
d_abs = grad(lambda t: anp.abs(t))
results["autograd_abs_derivative"] = {"at_0": d_abs(0.0), "at_+0.3": d_abs(0.3), "at_-0.3": d_abs(-0.3)}


def scaled_problem(p):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    X_raw = polynomial_features(x_train, p)
    sc = Scaler().fit(X_raw, y_train)
    return sc.transform(X_raw), sc.center(y_train), sc


P = 10
X, y, sc = scaled_problem(P)
X_test = sc.transform(polynomial_features(x_test, P))
L = np.linalg.eigvalsh(hessian(X)).max()          # Lipschitz constant of the MSE gradient
lam_max = (2.0 / n) * np.max(np.abs(X.T @ y))     # smallest lambda giving theta = 0
results["setup"] = {"degree": P, "lipschitz_L": L, "lambda_max": lam_max}
g_mse = make_gradient("ols", X, y)

# ------------------------------------------------------------------------------------------
# 1) Lasso path: Scikit-Learn (coordinate descent) and our FISTA
# ------------------------------------------------------------------------------------------
lams = lam_max * np.logspace(0, -3.5, 36)
path = np.array([lasso_sklearn_parameters(X, y, lam, max_iter=2_000_000, tol=1e-12) for lam in lams])
check_idx = np.arange(3, 36, 4)
ours = {}
for i in check_idx:
    th, inf = fista(g_mse, np.zeros(P), 1.0 / L, lams[i], 2_000_000, theta_ref=path[i], tol=1e-8)
    ours[int(i)] = {"lambda": lams[i], "theta": th, "iterations": inf["iterations"],
                    "converged": inf["converged"], "max_abs_diff": np.max(np.abs(th - path[i])),
                    "same_zero_pattern": bool(np.all((th == 0) == (path[i] == 0))),
                    "cost_difference": cost_lasso(th, X, y, lams[i]) - cost_lasso(path[i], X, y, lams[i])}
nonzero = path != 0
first_odd = lams[np.argmax(nonzero[:, 0::2].any(1))]
first_even = lams[np.argmax(nonzero[:, 1::2].any(1))]
results["path"] = {"lambdas": lams, "theta": path, "n_nonzero": nonzero.sum(1),
                   "n_nonzero_odd": nonzero[:, 0::2].sum(1), "n_nonzero_even": nonzero[:, 1::2].sum(1),
                   "lambda_first_even_enters": first_even, "lambda_first_odd_enters": first_odd,
                   "test_mse": [mse(y_test, sc.y_mean + X_test @ t) for t in path]}
results["fista_vs_sklearn"] = ours

# ------------------------------------------------------------------------------------------
# 2) Convergence at lambda = 1e-3: subgradient GD, subgradient Adam, ISTA, FISTA
# ------------------------------------------------------------------------------------------
lam = 1e-3
ref = lasso_sklearn_parameters(X, y, lam, max_iter=2_000_000, tol=1e-12)
ref_fista, _ = fista(g_mse, np.zeros(P), 1.0 / L, lam, 2_000_000, theta_ref=ref, tol=1e-12)
c_star = min(cost_lasso(ref, X, y, lam), cost_lasso(ref_fista, X, y, lam))
N = 40_000
eta = 1.0 / L
g_sub = make_gradient("lasso", X, y, lam)          # subgradient with sign(0) = 0


def track(update, n_iter=N, state=None):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    theta = np.zeros(P)
    excess = np.empty(n_iter)
    for k in range(n_iter):
        theta = update(theta)
        excess[k] = cost_lasso(theta, X, y, lam) - c_star
    return theta, excess


adam = Adam(0.01)
adam.reset(P)
fz = {"z": np.zeros(P), "t": 1.0, "prev": np.zeros(P)}


def fista_step(theta):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    new = soft_threshold(fz["z"] - eta * g_mse(fz["z"]), eta * lam)
    t_new = 0.5 * (1 + np.sqrt(1 + 4 * fz["t"] ** 2))
    fz["z"] = new + ((fz["t"] - 1) / t_new) * (new - fz["prev"])
    fz["t"], fz["prev"] = t_new, new
    return new


runs = {
    "Subgradient GD": track(lambda t: t - eta * g_sub(t)),
    "Subgradient Adam": track(lambda t: t - adam.update(g_sub(t), adam.eta)),
    "ISTA (proximal GD)": track(lambda t: soft_threshold(t - eta * g_mse(t), eta * lam)),
    "FISTA": track(fista_step),
}
inactive = ref == 0
conv = {"lambda": lam, "reference_theta": ref, "inactive": np.where(inactive)[0] + 1,
        "eta": eta, "eta_times_lambda": eta * lam,
        "fista_vs_sklearn_max_abs": np.max(np.abs(ref_fista - ref))}
for name, (th, ex) in runs.items():
    conv[name] = {"final_excess": ex[-1], "exact_zeros": int(np.sum(th == 0)),
                  "zero_positions": np.where(th == 0)[0] + 1,
                  "max_abs_inactive": np.max(np.abs(th[inactive])) if inactive.any() else 0.0,
                  "max_abs_diff_to_reference": np.max(np.abs(th - ref)),
                  "iterations_to_1e-10": int(np.argmax(ex < 1e-10)) + 1 if (ex < 1e-10).any() else None}
results["convergence"] = conv

# the subgradient 'rattle' in isolation: degree 6, lambda = 1e-2 (fast convergence, 3 exact zeros)
X6, y6, _ = scaled_problem(6)
L6 = np.linalg.eigvalsh(hessian(X6)).max()
lam6 = 1e-2
ref6 = lasso_sklearn_parameters(X6, y6, lam6, max_iter=2_000_000, tol=1e-12)
th_sub, _ = gradient_descent(make_gradient("lasso", X6, y6, lam6), np.zeros(6), GD(1 / L6), 100_000)
th_ista, inf_ista = gradient_descent(make_gradient("ols", X6, y6), np.zeros(6), GD(1 / L6), 100_000,
                                     prox_lam=lam6, theta_ref=ref6, tol=1e-12)
results["degree6_rattle"] = {"lambda": lam6, "eta": 1 / L6, "eta_times_lambda": lam6 / L6,
                             "zeros_reference": np.where(ref6 == 0)[0] + 1,
                             "subgradient_values_at_zeros": th_sub[ref6 == 0],
                             "ista_values_at_zeros": th_ista[ref6 == 0],
                             "ista_iterations": inf_ista["iterations"]}

# ------------------------------------------------------------------------------------------
# 3) OLS / Ridge / Lasso at degree 10 obtained with gradient methods vs. closed forms
# ------------------------------------------------------------------------------------------
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
# Figure
# ------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
even_cols = ordered_colors(5, 0.35, 1.0, base=METHOD_COLORS["Lasso"])
odd_cols = ordered_colors(5, 0.35, 1.0)
for j in range(P):
    power = j + 1
    col = even_cols[j // 2] if power % 2 == 0 else odd_cols[j // 2]
    ax1.semilogx(lams, path[:, j], "-" if power % 2 == 0 else "--", color=col, lw=1.1,
                 label=rf"$\theta_{{{power}}}$")
    ax1.semilogx([ours[i]["lambda"] for i in ours], [ours[i]["theta"][j] for i in ours], "o", ms=2.6,
                 mfc="none", color=col)
ax1.set_yscale("symlog", linthresh=0.1)
ax1.invert_xaxis()
ax1.set_xlabel(r"$\lambda$")
ax1.set_ylabel(r"Lasso coefficient $\theta_j$ (degree 10)")
ax1.legend(fontsize=5.3, ncol=2, loc="lower left", title="solid: even, dashed: odd powers",
           title_fontsize=5.5)
styles = {"Subgradient GD": (COLORS[0], "-"), "Subgradient Adam": (COLORS[4], "-."),
          "ISTA (proximal GD)": (COLORS[3], "--"), "FISTA": (COLORS[2], "-")}
for name, (th, ex) in runs.items():
    c, ls = styles[name]
    ax2.loglog(np.arange(1, N + 1), np.maximum(ex, 1e-17), color=c, ls=ls, lw=1.2, label=name)
ax2.set_xlabel("Iteration $k$")
ax2.set_ylabel(r"$C(\theta_k)-C(\hat\theta_{\mathrm{Lasso}})$")
ax2.set_ylim(1e-16, 1)
ax2.legend(fontsize=6, loc="lower left", title=rf"$\lambda=10^{{-3}}$, degree {P}", title_fontsize=6)
for ax, lab in zip((ax1, ax2), ("(a)", "(b)")):
    ax.text(0.97, 0.96, lab, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "g_lasso")

save_results("part_g", results)
print("autograd d|t|/dt:", results["autograd_abs_derivative"])
print("lambda_max", lam_max, "L", L, "first even enters", first_even, "first odd enters", first_odd)
for i, d in ours.items():
    print(f"  FISTA lambda {d['lambda']:.3g}: it {d['iterations']} conv {d['converged']} maxdiff {d['max_abs_diff']:.2e} "
          f"same zeros {d['same_zero_pattern']} dC {d['cost_difference']:.1e}")
for name in runs:
    print(name, conv[name])
print("ref zeros", conv["inactive"], "fista vs sklearn", conv["fista_vs_sklearn_max_abs"])
print("degree-6 rattle", results["degree6_rattle"])
print("degree-10 comparison", compare)
