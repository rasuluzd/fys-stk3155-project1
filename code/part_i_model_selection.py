"""
Part i): final model selection for OLS, Ridge and Lasso with 10-fold cross-validation, and the
bias-variance decomposition as a function of lambda for Ridge and Lasso.

For each method the CV error is computed on a (degree, lambda) grid with the same folds and with
the scaling refitted inside every training fold. The selected models (minimum CV error and the
one-standard-error rule) are refitted on all 100 points and evaluated on 2000 fresh points from the
same distribution, which we can do because the data are synthetic. The analysis is repeated for
sigma = 0.3.

Lasso fits use Scikit-Learn's coordinate descent along the lambda path with warm starts
(lasso_path, alpha = lambda/2); part g) showed that our own FISTA code reproduces these solutions.
For lambda < 1e-4 and degrees >= 10 coordinate descent does not converge in a reasonable number of
sweeps (the Hessian condition number exceeds 1e6), so the Lasso grid stops at 1e-4; the number of
fits that still hit the iteration limit is recorded.

Produces
  figures/i_model_selection.pdf  (a) best CV MSE against degree for the three methods,
                                 (b, c) bias^2 and variance against lambda at degree 15 for Ridge
                                 and Lasso (fresh training sets and bootstrap)
  results/part_i.json

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold

from plot_style import COLORS, GREY, INK, DOUBLE, METHOD_COLORS, save, set_style
from regression import Scaler, make_data, polynomial_features, runge
from settings import N_POINTS, NOISE, SEED, main_split, save_results

set_style()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
results = {}
degrees = np.arange(1, 21)
lam_ridge = np.logspace(-10, 0, 21)
lam_lasso = np.logspace(-1, -4, 7)            # descending, for warm starts along the path
K = 10
MAX_SWEEPS = 100_000
lasso_fits = {"total": 0, "hit_limit": 0}


def fit_paths(x_tr, y_tr, p):
    """Scaler plus coefficient paths for OLS (1 x p), Ridge (L_r x p) and Lasso (L_l x p).

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    X = polynomial_features(x_tr, p)
    sc = Scaler().fit(X, y_tr)
    Xs, yc = sc.transform(X), sc.center(y_tr)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    uty = U.T @ yc
    ols = ((uty / s) @ Vt)[None, :]
    ridge = ((s[None, :] / (s[None, :] ** 2 + len(yc) * lam_ridge[:, None])) * uty[None, :]) @ Vt
    _, coefs, _, n_iter = lasso_path(Xs, yc, alphas=lam_lasso / 2, max_iter=MAX_SWEEPS, tol=1e-6,
                                     return_n_iter=True)
    lasso_fits["total"] += len(n_iter)
    lasso_fits["hit_limit"] += int(np.sum(np.array(n_iter) >= MAX_SWEEPS))
    return sc, {"OLS": ols, "Ridge": ridge, "Lasso": coefs.T}


def predict(sc, thetas, x, p):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    return sc.y_mean + sc.transform(polynomial_features(x, p)) @ thetas.T


def cross_validate(x, y, k=K, seed=SEED):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    folds = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(x))
    shapes = {"OLS": 1, "Ridge": len(lam_ridge), "Lasso": len(lam_lasso)}
    err = {m: np.empty((k, len(degrees), L)) for m, L in shapes.items()}
    for f, (tr, va) in enumerate(folds):
        for a, p in enumerate(degrees):
            sc, paths = fit_paths(x[tr], y[tr], int(p))
            for m, th in paths.items():
                err[m][f, a] = np.mean((y[va][:, None] - predict(sc, th, x[va], int(p))) ** 2, axis=0)
    return {m: (e.mean(0), e.std(0, ddof=1) / np.sqrt(k)) for m, e in err.items()}


def effective_dof(x, p, method, lam_index):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    X = polynomial_features(x, p)
    sc = Scaler().fit(X, np.zeros(len(x)))
    s = np.linalg.svd(sc.transform(X), compute_uv=False)
    if method == "OLS":
        return float(p)
    if method == "Ridge":
        return float(np.sum(s**2 / (s**2 + len(x) * lam_ridge[lam_index])))
    return None


def select(x, y, cv, x_new, y_new):
    """Minimum-CV and one-standard-error choices for every method, refitted on all data and
    evaluated on fresh points.

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    out = {}
    lam_of = {"OLS": np.array([0.0]), "Ridge": lam_ridge, "Lasso": lam_lasso}
    full = {int(p): fit_paths(x, y, int(p)) for p in degrees}      # refits on all data, once
    dof = {int(p): [effective_dof(x, int(p), "Ridge", j) for j in range(len(lam_ridge))]
           for p in degrees}
    for m, (mean, se) in cv.items():
        a, b = np.unravel_index(np.argmin(mean), mean.shape)
        thr = mean[a, b] + se[a, b]
        cand = np.argwhere(mean <= thr)
        # simplest candidate: fewest effective parameters (degree for OLS, df for Ridge,
        # non-zero coefficients for Lasso); ties broken by the lower CV error
        best_simple, key_best = None, None
        for (i, j) in cand:
            p = int(degrees[i])
            if m == "Lasso":
                complexity = int(np.sum(full[p][1]["Lasso"][j] != 0))
            elif m == "Ridge":
                complexity = dof[p][j]
            else:
                complexity = float(p)
            key = (complexity, mean[i, j])
            if key_best is None or key < key_best:
                key_best, best_simple = key, (i, j)
        rows = {}
        for rule, (i, j) in (("min", (a, b)), ("one_se", best_simple)):
            p = int(degrees[i])
            sc, paths = full[p]
            th = paths[m][j]
            row = {"degree": p, "lambda": lam_of[m][j], "cv_mse": mean[i, j], "cv_se": se[i, j],
                   "true_test_mse": np.mean((y_new - predict(sc, th[None, :], x_new, p)[:, 0]) ** 2),
                   "nonzero": int(np.sum(th != 0))}
            if m == "Ridge":
                row["dof"] = effective_dof(x, p, m, j)
            if m == "Lasso":
                row["nonzero_odd"] = int(np.sum(th[0::2] != 0))
                row["nonzero_even"] = int(np.sum(th[1::2] != 0))
            rows[rule] = row
        out[m] = rows
    return out


# ------------------------------------------------------------------------------------------
# 1) Model selection for sigma = 0.1 (main data set) and sigma = 0.3
# ------------------------------------------------------------------------------------------
cv_store = {}
for noise in (NOISE, 0.3):
    x, y = make_data(N_POINTS, noise, SEED)
    x_new, y_new = make_data(2000, noise, SEED + 1)
    cv = cross_validate(x, y)
    cv_store[noise] = cv
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
# 2) Bias^2 and variance against lambda at degree 15: fresh training sets and bootstrap
# ------------------------------------------------------------------------------------------
P_BV, REPS, B = 15, 150, 150
x_eval = np.linspace(-1, 1, 401)
f_eval = runge(x_eval)
lasso_fits.update({"total": 0, "hit_limit": 0})


def decompose(pred):            # pred: (n_eval, n_lambda, n_sets)
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    mean = pred.mean(axis=2)
    return np.mean((f_eval[:, None] - mean) ** 2, axis=0), np.mean(pred.var(axis=2), axis=0)


fresh = {m: [] for m in ("OLS", "Ridge", "Lasso")}
for r in range(REPS):
    xf, yf = make_data(80, NOISE, seed=7_000_000 + r)
    sc, paths = fit_paths(xf, yf, P_BV)
    for m in fresh:
        fresh[m].append(predict(sc, paths[m], x_eval, P_BV))
x_tr80, _, y_tr80, _ = main_split()            # the main training set (80 points)
rng = np.random.default_rng(SEED)
boot = {m: [] for m in ("OLS", "Ridge", "Lasso")}
for b in range(B):
    idx = rng.integers(0, 80, 80)
    sc, paths = fit_paths(x_tr80[idx], y_tr80[idx], P_BV)
    for m in boot:
        boot[m].append(predict(sc, paths[m], x_eval, P_BV))
bv = {}
for m in fresh:
    b2, var = decompose(np.stack(fresh[m], axis=2))
    bb2, bvar = decompose(np.stack(boot[m], axis=2))
    bv[m] = {"bias2_fresh": b2, "variance_fresh": var, "bias2_boot": bb2, "variance_boot": bvar,
             "total_fresh": b2 + var + NOISE**2}
results["bias_variance_degree15"] = bv
results["lasso_convergence_bias_variance"] = dict(lasso_fits)

# ------------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45))
ax = axes[0]
for m, mk in (("OLS", "o"), ("Ridge", "s"), ("Lasso", "^")):
    mean, se = cv_store[NOISE][m]
    best = mean.min(1)
    se_b = se[np.arange(len(degrees)), mean.argmin(1)]
    ax.semilogy(degrees, best, "-" + mk, ms=2.6, color=METHOD_COLORS[m], label=m)
    ax.fill_between(degrees, best - se_b, best + se_b, color=METHOD_COLORS[m], alpha=0.15, lw=0)
ax.axhline(NOISE**2, color=GREY, lw=0.9, ls=":")
ax.set_ylim(7e-3, 0.3)
ax.set_xticks(degrees[1::3])
ax.set_xlabel("Polynomial degree $p$")
ax.set_ylabel(r"10-fold CV MSE (best $\lambda$)")
ax.legend(fontsize=6, loc="upper right")
for ax, m, lams in ((axes[1], "Ridge", lam_ridge), (axes[2], "Lasso", lam_lasso)):
    d = bv[m]
    ax.loglog(lams, d["bias2_fresh"], "-", color=COLORS[0], lw=1.3, label=r"Bias$^2$")
    ax.loglog(lams, d["variance_fresh"], "-", color=COLORS[1], lw=1.3, label="Variance")
    ax.loglog(lams, d["bias2_fresh"] + d["variance_fresh"], "-", color=INK, lw=1.0,
              label=r"Bias$^2$ + Variance")
    ax.loglog(lams, d["variance_boot"], "--", color=COLORS[1], lw=1.0, label="Variance (bootstrap)")
    ax.axhline(bv["OLS"]["variance_fresh"][0], color=METHOD_COLORS["OLS"], lw=0.9, ls=":",
               label="OLS variance")
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(f"{m}, degree {P_BV}", fontsize=8)
axes[1].set_ylabel("Contribution to test MSE")
axes[1].set_ylim(1e-5, 10)
axes[2].set_ylim(1e-5, 10)
axes[2].legend(fontsize=5.5, loc="lower left")
axes[0].text(0.03, 0.04, "(a)", transform=axes[0].transAxes, va="bottom", fontweight="bold")
for ax, lab in zip(axes[1:], ("(b)", "(c)")):
    ax.text(0.97, 0.96, lab, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "i_model_selection")

save_results("part_i", results)
print("lasso convergence:", results["lasso_convergence_cv"], results["lasso_convergence_bias_variance"])
for noise in (NOISE, 0.3):
    for k in ("k10",):
        for m, rows in results[f"sigma={noise}"][k].items():
            for rule, r in rows.items():
                print(f"sigma={noise} {k} {m:5s} {rule:6s}: p={r['degree']:2d} lam={r['lambda']:.1e} "
                      f"CV={r['cv_mse']:.4f}+-{r['cv_se']:.4f} true={r['true_test_mse']:.4f} "
                      f"nz={r['nonzero']} dof={r.get('dof', '')} odd/even={r.get('nonzero_odd', '')}/{r.get('nonzero_even', '')}")
for m in ("Ridge", "Lasso"):
    d = bv[m]
    lams = lam_ridge if m == "Ridge" else lam_lasso
    i = np.argmin(d["bias2_fresh"] + d["variance_fresh"])
    print(m, "best lambda (fresh)", lams[i], "bias2", d["bias2_fresh"][i], "var", d["variance_fresh"][i],
          "| OLS bias2", bv["OLS"]["bias2_fresh"][0], "var", bv["OLS"]["variance_fresh"][0])
