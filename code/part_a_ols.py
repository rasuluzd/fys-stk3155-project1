"""
Part a): ordinary least squares for Runge's function.

Produces
  figures/a_fits.pdf         data, true function and OLS fits of increasing degree
  figures/a_mse_r2.pdf       train/test MSE and R^2 against polynomial degree (main data set)
  figures/a_coefficients.pdf parameters theta_j against degree, and ||theta|| against degree
  figures/a_ensemble.pdf     median test MSE against degree for several n and noise levels
  results/part_a.json        all numbers quoted in the report

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from plot_style import COLORS, GREY, INK, DOUBLE, SINGLE, METHOD_COLORS, ordered_colors, save, set_style
from regression import PolynomialRegression, Scaler, make_data, mse, polynomial_features, r2, runge
from settings import MAX_DEGREE, NOISE, SEED, TEST_SIZE, main_split, save_results

set_style()
results = {}
x_train, x_test, y_train, y_test = main_split()
degrees = np.arange(0, MAX_DEGREE + 1)
x_grid = np.linspace(-1, 1, 801)

# ------------------------------------------------------------------------------------------
# 1) MSE and R^2 against degree for the main data set
# ------------------------------------------------------------------------------------------
mse_train, mse_test, r2_train, r2_test, thetas = [], [], [], [], []
for p in degrees:
    model = PolynomialRegression(int(p), "ols").fit(x_train, y_train)
    pred_train, pred_test = model.predict(x_train), model.predict(x_test)
    mse_train.append(mse(y_train, pred_train))
    mse_test.append(mse(y_test, pred_test))
    r2_train.append(r2(y_train, pred_train))
    r2_test.append(r2(y_test, pred_test))
    thetas.append(model.theta_)

best = int(degrees[np.argmin(mse_test)])
results["main"] = {"degrees": degrees, "mse_train": mse_train, "mse_test": mse_test,
                   "r2_train": r2_train, "r2_test": r2_test, "best_test_degree": best,
                   "n_train": len(x_train), "n_test": len(x_test),
                   "x_train_range": [x_train.min(), x_train.max()],
                   "x_test_range": [x_test.min(), x_test.max()]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
ax1.semilogy(degrees, mse_train, "-o", color=COLORS[0], label="Training")
ax1.semilogy(degrees, mse_test, "-s", color=COLORS[1], label="Test")
ax1.axhline(NOISE**2, color=GREY, lw=0.9, ls=":", label=r"$\sigma^2$")
ax1.set_xlabel("Polynomial degree $p$")
ax1.set_ylabel("MSE")
ax1.legend()
ax2.plot(degrees, r2_train, "-o", color=COLORS[0], label="Training")
ax2.plot(degrees, r2_test, "-s", color=COLORS[1], label="Test")
ax2.set_xlabel("Polynomial degree $p$")
ax2.set_ylabel("$R^2$")
ax2.set_ylim(-0.6, 1.0)
ax2.legend(loc="lower right")
for ax, lab in zip((ax1, ax2), ("(a)", "(b)")):
    ax.set_xticks(degrees[::3])
    ax.text(0.02, 0.96, lab, transform=ax.transAxes, va="top", fontweight="bold")
fig.tight_layout()
save(fig, "a_mse_r2")

# ------------------------------------------------------------------------------------------
# 2) Fits of increasing degree
# ------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
ax.plot(x_train, y_train, "o", ms=2.6, color=GREY, alpha=0.8, label="Training data")
ax.plot(x_test, y_test, "x", ms=3.2, color=INK, label="Test data")
ax.plot(x_grid, runge(x_grid), color=INK, lw=1.0, label="$f(x)$")
for p, c, ls in zip((2, 6, 15), (COLORS[0], COLORS[1], COLORS[2]), ("--", "-", "-.")):
    model = PolynomialRegression(p, "ols").fit(x_train, y_train)
    ax.plot(x_grid, model.predict(x_grid), ls, color=c, lw=1.2, label=f"OLS, $p={p}$")
ax.set_ylim(-0.5, 1.75)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(ncol=3, fontsize=5.5, loc="upper center", columnspacing=0.8, handlelength=1.6)
fig.tight_layout()
save(fig, "a_fits")

# ------------------------------------------------------------------------------------------
# 3) Parameters against degree
# ------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
for j in range(1, 7):
    vals = [thetas[p][j - 1] if p >= j else np.nan for p in degrees]
    ax1.plot(degrees, vals, "-" if j % 2 == 0 else "--", marker="o" if j % 2 == 0 else "^",
             ms=2.5, color=COLORS[j - 1], label=rf"$\theta_{j}$")
ax1.set_yscale("symlog", linthresh=0.1)
ax1.set_xlabel("Polynomial degree $p$")
ax1.set_ylabel(r"$\theta_j$ (standardized features)")
ax1.legend(ncol=3, fontsize=6)
norms = [np.linalg.norm(t) if len(t) else 0.0 for t in thetas]
odd_share = [np.linalg.norm(t[0::2]) / np.linalg.norm(t) if len(t) else np.nan for t in thetas]
ax2.semilogy(degrees[1:], norms[1:], "-o", color=METHOD_COLORS["OLS"], label=r"$\|\theta\|_2$ (all)")
ax2.semilogy(degrees[2:], [np.linalg.norm(t[1::2]) for t in thetas[2:]], "--s", ms=2.5,
             color=COLORS[1], label="even powers")
ax2.semilogy(degrees[1:], [np.linalg.norm(t[0::2]) for t in thetas[1:]], ":^", ms=2.5,
             color=COLORS[2], label="odd powers")
ax2.set_xlabel("Polynomial degree $p$")
ax2.set_ylabel(r"Norm of $\theta$")
ax2.legend(fontsize=6)
for ax, lab in zip((ax1, ax2), ("(a)", "(b)")):
    ax.set_xticks(degrees[::3])
    ax.text(0.02, 0.96, lab, transform=ax.transAxes, va="top", fontweight="bold")
fig.tight_layout()
save(fig, "a_coefficients")
results["coefficients"] = {"norm": norms, "theta": [t.tolist() for t in thetas],
                           "odd_norm_share": odd_share}

# original-basis coefficients for the quoted example (degree 15)
model15 = PolynomialRegression(15, "ols").fit(x_train, y_train)
beta0, beta = model15.scaler_.original_coefficients(model15.theta_)
results["coefficients"]["beta_degree15_max_abs"] = np.max(np.abs(beta))
results["coefficients"]["theta_degree15_max_abs"] = np.max(np.abs(model15.theta_))

# ------------------------------------------------------------------------------------------
# 4) Scaling: conditioning and invariance of OLS predictions
# ------------------------------------------------------------------------------------------
scaling = {}
for p in (5, 6, 10, 15):
    X = polynomial_features(x_train, p)
    X_raw = np.column_stack([np.ones(len(x_train)), X])          # raw with intercept column
    X_cen = X - X.mean(axis=0)                                     # centred
    X_std = Scaler().fit(X, y_train).transform(X)                  # standardised
    ols_s = PolynomialRegression(p, "ols", scale=True).fit(x_train, y_train)
    ols_c = PolynomialRegression(p, "ols", scale=False).fit(x_train, y_train)
    h_cen = np.linalg.eigvalsh(2 / len(x_train) * X_cen.T @ X_cen)
    h_std = np.linalg.eigvalsh(2 / len(x_train) * X_std.T @ X_std)
    scaling[p] = {"cond_raw": np.linalg.cond(X_raw), "cond_centred": np.linalg.cond(X_cen),
                  "cond_standardised": np.linalg.cond(X_std),
                  "hessian_centred_max_min": [h_cen.max(), h_cen.min()],
                  "hessian_standardised_max_min": [h_std.max(), h_std.min()],
                  "max_pred_difference_scaled_vs_centred":
                      np.max(np.abs(ols_s.predict(x_grid) - ols_c.predict(x_grid))),
                  "column_std_min": X.std(axis=0).min(), "column_std_max": X.std(axis=0).max()}
results["scaling"] = scaling

# ------------------------------------------------------------------------------------------
# 5) Ensemble: median test MSE against degree for several n and noise levels
# ------------------------------------------------------------------------------------------
R = 200
deg_ens = np.arange(1, 21)


def ensemble(n, noise, reps=R):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    test = np.empty((reps, len(deg_ens)))
    train = np.empty_like(test)
    for r in range(reps):
        x, y = make_data(n, noise, seed=10_000 + r)
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=TEST_SIZE, random_state=r)
        for k, p in enumerate(deg_ens):
            m = PolynomialRegression(int(p), "ols").fit(xtr, ytr)
            test[r, k] = mse(yte, m.predict(xte))
            train[r, k] = mse(ytr, m.predict(xtr))
    return train, test


ens = {}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
n_values = (50, 100, 500, 1000)
for c, n in zip(ordered_colors(len(n_values), 0.35, 1.0), n_values):
    tr, te = ensemble(n, NOISE)
    med, q1, q3 = np.median(te, 0), np.quantile(te, 0.25, 0), np.quantile(te, 0.75, 0)
    ax1.semilogy(deg_ens, med, "-o", ms=2.5, color=c, label=f"$n={n}$")
    ax1.fill_between(deg_ens, q1, q3, color=c, alpha=0.15, lw=0)
    ens[f"n={n},sigma={NOISE}"] = {"median": med, "q1": q1, "q3": q3,
                                   "best_degree_median": int(deg_ens[np.argmin(med)]),
                                   "mean": te.mean(0), "train_median": np.median(tr, 0)}
ax1.axhline(NOISE**2, color=GREY, lw=0.9, ls=":")
ax1.set_ylim(5e-3, 2.0)
ax1.set_xlabel("Polynomial degree $p$")
ax1.set_ylabel("Test MSE (median of 200 data sets)")
ax1.legend(title=rf"$\sigma={NOISE}$", fontsize=6, title_fontsize=6)

noise_values = (0.0, 0.05, 0.1, 0.3)
for c, s in zip(ordered_colors(len(noise_values), 0.35, 1.0), noise_values):
    tr, te = ensemble(100, s)
    med, q1, q3 = np.median(te, 0), np.quantile(te, 0.25, 0), np.quantile(te, 0.75, 0)
    ax2.semilogy(deg_ens, med, "-o", ms=2.5, color=c, label=rf"$\sigma={s}$")
    ax2.fill_between(deg_ens, q1, q3, color=c, alpha=0.15, lw=0)
    ens[f"n=100,sigma={s}"] = {"median": med, "q1": q1, "q3": q3,
                               "best_degree_median": int(deg_ens[np.argmin(med)]),
                               "mean": te.mean(0), "train_median": np.median(tr, 0)}
# Codex, 5 October 2026: keep this figure focused on the sample-size and
# noise comparisons requested in part a, without an external approximation-rate model.
ax2.set_xlabel("Polynomial degree $p$")
ax2.set_ylabel("Test MSE (median of 200 data sets)")
ax2.legend(title="$n=100$", fontsize=6, title_fontsize=6, ncol=1)
for ax, lab in zip((ax1, ax2), ("(a)", "(b)")):
    ax.set_xticks(deg_ens[1::3])
    ax.text(0.97, 0.96, lab, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "a_ensemble")
results["ensemble"] = ens

save_results("part_a", results)
print("best test degree (main data set):", best, "test MSE", mse_test[best])
print("scaling:", {p: {k: (f"{v:.3g}" if np.isscalar(v) else np.round(v, 5).tolist()) for k, v in d.items()}
                   for p, d in scaling.items()})
for key, d in ens.items():
    print(key, "best degree (median):", d["best_degree_median"], "min median MSE", f"{min(d['median']):.4g}")
