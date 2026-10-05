"""
Part b): Ridge regression for Runge's function, and the SVD picture of the shrinkage.

Produces
  figures/b_ridge_mse.pdf  test MSE against degree for several lambda (main data set) and a
                           (degree, lambda) map of the median test MSE over 100 data sets
  figures/b_ridge_svd.pdf  singular values and |u_j^T y| (noise floor), Ridge filter factors,
                           and ||theta|| against degree for several lambda
  results/part_b.json

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from plot_style import BLUES, GREY, INK, DOUBLE, METHOD_COLORS, ordered_colors, save, set_style
from regression import (PolynomialRegression, Scaler, make_data, mse, polynomial_features, r2,
                        ridge_shrinkage_factors)
from settings import NOISE, TEST_SIZE, main_split, save_results

set_style()
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
# 2) Ensemble: median test MSE over 100 data sets on the (degree, lambda) grid
# ------------------------------------------------------------------------------------------
R = 100
ens = np.empty((R, len(degrees), len(lams)))
ens_ols = np.empty((R, len(degrees)))
for r in range(R):
    x, y = make_data(100, NOISE, seed=20_000 + r)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=TEST_SIZE, random_state=r)
    for k, p in enumerate(degrees):
        pred, _ = ridge_path_predictions(xtr, ytr, xte, p, lams)
        ens[r, k] = np.mean((yte[:, None] - pred) ** 2, axis=0)
        ens_ols[r, k] = mse(yte, PolynomialRegression(int(p)).fit(xtr, ytr).predict(xte))
med = np.median(ens, axis=0)
ke, le = np.unravel_index(np.argmin(med), med.shape)
results["ensemble"] = {"median_test_mse": med, "best_degree": int(degrees[ke]),
                       "best_lambda": lams[le], "best_median_mse": med[ke, le],
                       "ols_median": np.median(ens_ols, 0),
                       "ols_best_degree": int(degrees[np.argmin(np.median(ens_ols, 0))]),
                       "ols_best_median_mse": np.min(np.median(ens_ols, 0)),
                       "ridge_best_lambda_per_degree": lams[np.argmin(med, axis=1)],
                       "ridge_best_mse_per_degree": med.min(axis=1)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
ax1.semilogy(degrees, ols_test, "--o", ms=2.5, color=METHOD_COLORS["OLS"], label="OLS")
for c, lam in zip(ordered_colors(len(lam_show), 0.3, 1.0, base=METHOD_COLORS["Ridge"]), lam_show):
    ax1.semilogy(degrees, main[f"test_mse_lambda_{lam:g}"], "-s", ms=2.5, color=c,
                 label=rf"Ridge, $\lambda=10^{{{int(np.log10(lam))}}}$")
ax1.axhline(NOISE**2, color=GREY, lw=0.9, ls=":")
ax1.set_ylim(8e-3, 2.0)
ax1.set_xticks(degrees[1::3])
ax1.set_xlabel("Polynomial degree $p$")
ax1.set_ylabel("Test MSE (main data set)")
ax1.legend(fontsize=5.5, ncol=2, loc="upper left", columnspacing=0.8)
im = ax2.pcolormesh(np.log10(lams), degrees, np.log10(med), cmap=BLUES, shading="nearest",
                    vmin=np.log10(med.min()), vmax=-0.8, rasterized=True)
ax2.plot(np.log10(lams[le]), degrees[ke], marker="*", ms=9, color="#e34948", mec="white", mew=0.6)
ax2.set_xlabel(r"$\log_{10}\lambda$")
ax2.set_ylabel("Polynomial degree $p$")
ax2.set_yticks(degrees[1::3])
ax2.grid(False)
cb = fig.colorbar(im, ax=ax2, pad=0.02)
cb.set_label(r"$\log_{10}$ median test MSE")
cb.outline.set_linewidth(0.5)
for ax, lab, col in zip((ax1, ax2), ("(a)", "(b)"), (INK, "white")):
    ax.text(0.03, 0.96, lab, transform=ax.transAxes, va="top", fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "b_ridge_mse")

# ------------------------------------------------------------------------------------------
# 3) SVD picture at degree 15: signal vs noise in each singular mode, and the shrinkage
# ------------------------------------------------------------------------------------------
p_svd = 15
X = polynomial_features(x_train, p_svd)
sc = Scaler().fit(X, y_train)
Xs, yc = sc.transform(X), sc.center(y_train)
U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
uty = U.T @ yc
j = np.arange(1, p_svd + 1)
lam_best15 = lams[np.argmin(test_map[degrees == p_svd][0])]
lam_med15 = lams[np.argmin(med[degrees == p_svd][0])]
df = {f"{lam:g}": np.sum(ridge_shrinkage_factors(Xs, lam)) for lam in [1e-7, 1e-5, 1e-3, 1e-1]}
results["svd_degree15"] = {"singular_values": s, "abs_uty": np.abs(uty),
                           "ols_mode_coefficients": np.abs(uty) / s,
                           "noise_level": NOISE,
                           "n_lambda_crossing": [n * lam for lam in lam_show],
                           "best_lambda_main": lam_best15, "best_lambda_ensemble": lam_med15,
                           "dof_at_best_lambda_ensemble": np.sum(ridge_shrinkage_factors(Xs, lam_med15)),
                           "dof": df,
                           "hessian_eigenvalues": 2 * s**2 / n}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DOUBLE, 2.25))
ax1.semilogy(j, s, "o-", ms=3, color=INK, label=r"$d_j$")
ax1.semilogy(j, np.abs(uty), "s", ms=3.2, color=METHOD_COLORS["OLS"], label=r"$|u_j^Ty|$")
ax1.semilogy(j, np.abs(uty) / s, "^", ms=3.2, color="#1baf7a", label=r"$|u_j^Ty|/d_j$")
ax1.axhline(NOISE, color=GREY, ls=":", lw=0.9)
ax1.text(4.3, NOISE * 0.55, r"noise level $\sigma$", color=GREY, fontsize=6)
ax1.set_xlabel("Singular mode $j$")
ax1.set_ylabel("Magnitude")
ax1.legend(fontsize=6, loc="lower left")
for c, lam in zip(ordered_colors(len(lam_show), 0.3, 1.0, base=METHOD_COLORS["Ridge"]), lam_show):
    f = s**2 / (s**2 + n * lam)
    ax2.plot(j, f, "-o", ms=2.5, color=c, label=rf"$\lambda=10^{{{int(np.log10(lam))}}}$")
ax2.set_xlabel("Singular mode $j$")
ax2.set_ylabel(r"Shrinkage $d_j^2/(d_j^2+n\lambda)$")
ax2.legend(fontsize=6, loc="lower left")
ols_norm = [np.linalg.norm(PolynomialRegression(int(p)).fit(x_train, y_train).theta_) for p in degrees]
ax3.semilogy(degrees, ols_norm, "--o", ms=2.5, color=METHOD_COLORS["OLS"], label="OLS")
for c, lam in zip(ordered_colors(len(lam_show), 0.3, 1.0, base=METHOD_COLORS["Ridge"]), lam_show):
    norms = [np.linalg.norm(PolynomialRegression(int(p), "ridge", lam).fit(x_train, y_train).theta_)
             for p in degrees]
    ax3.semilogy(degrees, norms, "-s", ms=2.5, color=c, label=rf"$\lambda=10^{{{int(np.log10(lam))}}}$")
ax3.set_xticks(degrees[1::3])
ax3.set_xlabel("Polynomial degree $p$")
ax3.set_ylabel(r"$\|\theta\|_2$")
ax3.legend(fontsize=5.5, loc="upper left")
for ax, lab in zip((ax1, ax2, ax3), ("(a)", "(b)", "(c)")):
    ax.text(0.97, 0.96, lab, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
ax1.set_xticks(j[::2])
ax2.set_xticks(j[::2])
fig.tight_layout()
save(fig, "b_ridge_svd")

save_results("part_b", results)
print("main: best ridge (p, lambda, mse):", main["best_degree"], main["best_lambda"], main["best_test_mse"],
      "| OLS best:", main["ols_best_degree"], main["ols_best_test_mse"])
print("ensemble: best ridge (p, lambda, median mse):", degrees[ke], lams[le], med[ke, le],
      "| OLS best:", results["ensemble"]["ols_best_degree"], results["ensemble"]["ols_best_median_mse"])
print("degree 15: best lambda main", lam_best15, "ensemble", lam_med15,
      "dof", results["svd_degree15"]["dof_at_best_lambda_ensemble"])
print("singular values", np.array2string(s, precision=3))
print("|u^T y|", np.array2string(np.abs(uty), precision=3))
print("dof", df)
