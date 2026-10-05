"""
Part d): k-fold cross-validation with Scikit-Learn (KFold + cross_val_score) for OLS and Ridge,
our own k-fold code as a check, and a comparison with the bootstrap of part c).

Produces
  figures/d_cross_validation.pdf  (a) OLS: CV MSE (k = 5, 10) against degree, bootstrap error and
                                  the true expected test error; (b) Ridge: 10-fold CV map over
                                  (degree, lambda)
  results/part_d.json

Run part_c_bias_variance.py first (its results are read for the comparison).

LLM-assisted (code level 2): plotting code generated with Claude (Claude Code, October 2026);
docstrings edited with OpenAI Codex, 5 October 2026.
"""

import json
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from plot_style import BLUES, COLORS, GREY, INK, DOUBLE, METHOD_COLORS, RESULTS_DIR, save, set_style
from regression import PolynomialRegression, make_data
from resampling import kfold_cv_mse
from settings import N_POINTS, NOISE, SEED, save_results

set_style()
results = {}
x, y = make_data(N_POINTS, NOISE, SEED)      # the full main data set (same points as before)
X1 = x[:, None]
degrees = np.arange(1, 21)


def ols_pipe(p):
    """Scikit-learn OLS pipeline (features, standardization, fit) with the pinv rank cutoff."""
    # Codex correction, 5 October 2026: sklearn 1.9 applies tol to dense
    # least-squares rank. Match np.linalg.pinv's relative cutoff explicitly.
    return make_pipeline(PolynomialFeatures(p, include_bias=False), StandardScaler(),
                         LinearRegression(tol=1e-15))


def ridge_pipe(p, alpha):
    """Scikit-learn Ridge pipeline; alpha is the training-fold size times lambda."""
    return make_pipeline(PolynomialFeatures(p, include_bias=False), StandardScaler(),
                         Ridge(alpha=alpha, solver="svd"))


def cv_scores(estimator, k, seed=SEED):
    """Fold MSEs from scikit-learn's KFold and cross_val_score."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return -cross_val_score(estimator, X1, y, cv=kf, scoring="neg_mean_squared_error")


# ------------------------------------------------------------------------------------------
# 1) OLS: CV MSE against degree for k = 5 and k = 10 (+ standard error over folds)
# ------------------------------------------------------------------------------------------
ols = {}
for k in (5, 10):
    scores = np.array([cv_scores(ols_pipe(int(p)), k) for p in degrees])   # (degrees, folds)
    mean, se = scores.mean(1), scores.std(1, ddof=1) / np.sqrt(k)
    i = np.argmin(mean)
    one_se = int(degrees[np.where(mean <= mean[i] + se[i])[0][0]])        # simplest within 1 SE
    ols[k] = {"mean": mean, "se": se, "best_degree": int(degrees[i]), "best_mse": mean[i],
              "best_se": se[i], "one_se_degree": one_se}

# own k-fold code with the same folds as Scikit-Learn
check = {}
for k in (5, 10):
    folds = list(KFold(n_splits=k, shuffle=True, random_state=SEED).split(x))
    diffs = []
    for p in (3, 8, 12, 20):
        ours = kfold_cv_mse(PolynomialRegression(p), x, y, folds)
        ref = cv_scores(ols_pipe(p), k)
        diffs.append(np.max(np.abs(ours - ref) / ref))
    check[k] = {"max_relative_difference_ols": max(diffs)}
    lam = 1e-4
    ours = kfold_cv_mse(PolynomialRegression(10, "ridge", lam), x, y, folds)
    ref = cv_scores(ridge_pipe(10, len(x) * (k - 1) // k * lam), k)
    check[k]["max_relative_difference_ridge"] = np.max(np.abs(ours - ref) / ref)

# variability of the selected degree when the folds are reshuffled
reshuffle = {}
for k in (5, 10):
    picks = []
    for s in range(20):
        means = [cv_scores(ols_pipe(int(p)), k, seed=s).mean() for p in degrees]
        picks.append(int(degrees[np.argmin(means)]))
    reshuffle[k] = {"selected_degrees": picks, "median": float(np.median(picks)),
                    "min": min(picks), "max": max(picks)}

# leakage check: standardise with statistics from all data before CV (wrong) vs inside folds
leak = {}
Xall = PolynomialFeatures(12, include_bias=False).fit_transform(X1)
Xall_std = StandardScaler().fit_transform(Xall)
kf10 = KFold(n_splits=10, shuffle=True, random_state=SEED)
inside_ols = cv_scores(ols_pipe(12), 10).mean()
outside_ols = -cross_val_score(LinearRegression(tol=1e-15), Xall_std, y, cv=kf10,
                               scoring="neg_mean_squared_error").mean()
leak["ols_degree12_inside"], leak["ols_degree12_outside"] = inside_ols, outside_ols
rel = []
for lam in np.logspace(-8, 0, 9):
    a = 90 * lam
    inside = cv_scores(ridge_pipe(12, a), 10).mean()
    outside = -cross_val_score(Ridge(alpha=a, solver="svd"), Xall_std, y, cv=kf10,
                               scoring="neg_mean_squared_error").mean()
    rel.append((outside - inside) / inside)
leak["ridge_degree12_relative_difference"] = rel

# ------------------------------------------------------------------------------------------
# 2) Ridge: 10-fold CV over (degree, lambda), alpha = n_train_fold * lambda
# ------------------------------------------------------------------------------------------
lams = np.logspace(-10, 0, 21)
t0 = time.perf_counter()
ridge = {}
for k in (5, 10):
    n_fold = len(x) * (k - 1) // k
    cv = np.empty((len(degrees), len(lams)))
    se = np.empty_like(cv)
    for a, p in enumerate(degrees):
        for b, lam in enumerate(lams):
            s = cv_scores(ridge_pipe(int(p), n_fold * lam), k)
            cv[a, b], se[a, b] = s.mean(), s.std(ddof=1) / np.sqrt(k)
    a, b = np.unravel_index(np.argmin(cv), cv.shape)
    ridge[k] = {"cv": cv, "se": se, "best_degree": int(degrees[a]), "best_lambda": lams[b],
                "best_mse": cv[a, b], "best_se": se[a, b]}
results["ridge_time_s"] = time.perf_counter() - t0
results.update({"degrees": degrees, "lambdas": lams, "ols": ols, "own_vs_sklearn": check,
                "reshuffle": reshuffle, "leakage": leak, "ridge": ridge,
                "ols_svd_relative_cutoff": 1e-15})

# ------------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------------
with open(RESULTS_DIR / "part_c.json", encoding="utf-8") as f:
    part_c = json.load(f)
deg_c = np.array(part_c["train_test"]["degrees"])
boot = np.array(part_c["n=100"]["error"])

# LLM-assisted (Claude, Claude Code, October 2026): plotting code for this figure.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1, 1.12]})
for k, c, mk in ((5, COLORS[0], "o"), (10, COLORS[2], "s")):
    m, s = ols[k]["mean"], ols[k]["se"]
    ax1.semilogy(degrees, m, "-" + mk, ms=2.6, color=c, label=f"{k}-fold CV")
    ax1.fill_between(degrees, np.maximum(m - s, 1e-4), m + s, color=c, alpha=0.15, lw=0)
sel = (deg_c >= 1) & (deg_c <= 20)
ax1.semilogy(deg_c[sel], boot[sel], "-^", ms=2.6, color=COLORS[1], label="Bootstrap (part c)")
ax1.semilogy(deg_c[sel], np.array(part_c["train_test"]["mean_test"])[sel], "--", color=INK, lw=1.0,
             label="Fresh sets (mean)")
ax1.semilogy(deg_c[sel], np.array(part_c["train_test"]["median_test"])[sel], ":", color=INK, lw=1.2,
             label="Fresh sets (median)")
ax1.axhline(NOISE**2, color=GREY, lw=0.9, ls=":")
ax1.set_ylim(7e-3, 1.0)
ax1.set_xlabel("Polynomial degree $p$")
ax1.set_ylabel("Estimated test MSE (OLS)")
ax1.set_title("Detail: errors above 1 are outside the axis", fontsize=7)
ax1.set_xticks(degrees[1::3])
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.2, bbox_to_anchor=(0.5, -0.01))
cv10 = ridge[10]["cv"]
im = ax2.pcolormesh(np.log10(lams), degrees, np.log10(cv10), cmap=BLUES, shading="nearest",
                    vmin=np.log10(cv10.min()), vmax=-0.8, rasterized=True)
ax2.plot(np.log10(ridge[10]["best_lambda"]), ridge[10]["best_degree"], marker="*", ms=9,
         color=METHOD_COLORS["Ridge"], mec="white", mew=0.6)
ax2.set_xlabel(r"$\log_{10}\lambda$")
ax2.set_ylabel("Polynomial degree $p$")
ax2.set_yticks(degrees[1::3])
ax2.grid(False)
cb = fig.colorbar(im, ax=ax2, pad=0.02)
cb.set_label(r"$\log_{10}$ 10-fold CV MSE (Ridge)")
cb.outline.set_linewidth(0.5)
ax1.text(0.03, 0.96, "(a)", transform=ax1.transAxes, va="top", fontweight="bold")
ax2.text(0.97, 0.96, "(b)", transform=ax2.transAxes, va="top", ha="right", fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "d_cross_validation")

save_results("part_d", results)
for k in (5, 10):
    print(f"OLS k={k}: best degree {ols[k]['best_degree']} CV {ols[k]['best_mse']:.4g} +- {ols[k]['best_se']:.2g}, "
          f"one-SE degree {ols[k]['one_se_degree']}; reshuffles {reshuffle[k]}")
    print(f"Ridge k={k}: best (p, lam) = ({ridge[k]['best_degree']}, {ridge[k]['best_lambda']:.2g}) "
          f"CV {ridge[k]['best_mse']:.4g} +- {ridge[k]['best_se']:.2g}")
print("own vs sklearn:", check)
print("leakage:", leak)
print("ridge grid time", results["ridge_time_s"])
