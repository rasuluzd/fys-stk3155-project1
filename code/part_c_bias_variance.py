"""
Part c): training vs. test error (cf. Hastie et al., Fig. 2.11) and the bias-variance
decomposition of OLS estimated with the bootstrap.

Produces
  figures/c_train_test.pdf     training and test MSE against degree for 100 independent data sets
  figures/c_bias_variance.pdf  bootstrap error / bias^2 / variance against degree for n = 100 and
                               n = 1000, compared with the 'true' values from fresh training sets
  results/part_c.json

Because we know how the data are generated, the expectation over training sets can also be done
exactly (up to Monte Carlo noise) by drawing many fresh training sets. This lets us check what the
bootstrap actually estimates.

LLM-assisted (code level 3): experiments written by the author; script structure and plotting
code by Claude (Claude Code, October 2026), which also removed unused output on 5 October.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from plot_style import COLORS, GREY, INK, DOUBLE, SINGLE, save, save_results, set_style
from regression import NOISE, SEED, TEST_SIZE, PolynomialRegression, make_data, mse, runge
from resampling import bootstrap_bias_variance

set_style()
results = {}
degrees = np.arange(0, 21)

# ------------------------------------------------------------------------------------------
# 1) Hastie-style figure: 100 training sets of 80 points, test error on 2000 fresh points
# ------------------------------------------------------------------------------------------
R, n_train = 100, 80
rng = np.random.default_rng(SEED)
x_big = rng.uniform(-1, 1, 2000)
y_big = runge(x_big) + NOISE * rng.standard_normal(2000)
train_curves = np.empty((R, len(degrees)))
test_curves = np.empty_like(train_curves)
for r in range(R):
    xtr, ytr = make_data(n_train, NOISE, seed=30_000 + r)
    for k, p in enumerate(degrees):
        m = PolynomialRegression(int(p)).fit(xtr, ytr)
        train_curves[r, k] = mse(ytr, m.predict(xtr))
        test_curves[r, k] = mse(y_big, m.predict(x_big))
mean_train, mean_test = train_curves.mean(0), test_curves.mean(0)
med_test = np.median(test_curves, 0)
results["train_test"] = {"degrees": degrees, "mean_train": mean_train, "mean_test": mean_test,
                         "median_test": med_test,
                         "best_degree_mean": int(degrees[np.argmin(mean_test)]),
                         "best_degree_median": int(degrees[np.argmin(med_test)])}

# LLM-assisted (Claude, Claude Code, October 2026): plotting code for this figure.
fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
for r in range(40):
    ax.semilogy(degrees, train_curves[r], color=COLORS[0], alpha=0.12, lw=0.6)
    ax.semilogy(degrees, test_curves[r], color=COLORS[1], alpha=0.12, lw=0.6)
ax.semilogy(degrees, mean_train, color=COLORS[0], lw=2.0, label="Training error (mean)")
ax.semilogy(degrees, mean_test, color=COLORS[1], lw=2.0, label="Test error (mean)")
ax.semilogy(degrees, med_test, color=COLORS[1], lw=1.2, ls="--", label="Test error (median)")
ax.axhline(NOISE**2, color=GREY, lw=0.9, ls=":")
ax.text(16.6, NOISE**2 * 1.08, r"$\sigma^2$", color=GREY, fontsize=7)
ax.set_ylim(1.5e-3, 0.4)  # room for the legend below the training curves
ax.set_xlabel("Model complexity (polynomial degree $p$)")
ax.set_ylabel("MSE")
ax.text(0.02, 0.97, "High bias\nLow variance", transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
ax.text(0.98, 0.97, "Low bias\nHigh variance", transform=ax.transAxes, va="top", ha="right",
        fontsize=6.5, color=INK)
ax.legend(fontsize=6, loc="lower left")
ax.set_xticks(degrees[::2])
fig.tight_layout()
save(fig, "c_train_test")

# ------------------------------------------------------------------------------------------
# 2) Bootstrap bias-variance for n = 100 and n = 1000, compared with fresh training sets
# ------------------------------------------------------------------------------------------
B, MC = 200, 400


def analyse(n):
    """Bootstrap and fresh-training-set bias-variance estimates against degree for n points."""
    x, y = make_data(n, NOISE, seed=SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)
    f_te = runge(x_te)
    out = {k: [] for k in ("error", "bias2", "variance", "bias2_true", "variance_mc",
                           "bias2_mc", "error_mc")}
    for p in degrees:
        res = bootstrap_bias_variance(PolynomialRegression(int(p)), x_tr, y_tr, x_te, y_te,
                                      n_bootstraps=B, seed=SEED + int(p), f_test=f_te)
        for key in ("error", "bias2", "variance", "bias2_true"):
            out[key].append(res[key])
        # 'truth': expectation over fresh training sets of the same size, same test points
        preds = np.empty((len(x_te), MC))
        for b in range(MC):
            xf, yf = make_data(len(x_tr), NOISE, seed=1_000_000 + 1000 * int(p) + b)
            preds[:, b] = PolynomialRegression(int(p)).fit(xf, yf).predict(x_te)
        out["variance_mc"].append(np.mean(np.var(preds, axis=1)))
        out["bias2_mc"].append(np.mean((f_te - preds.mean(axis=1)) ** 2))
        out["error_mc"].append(np.mean(np.mean((y_te[:, None] - preds) ** 2, axis=1)))
    out = {k: np.array(v) for k, v in out.items()}
    out["best_degree"] = int(degrees[np.argmin(out["error"])])
    out["best_degree_mc"] = int(degrees[np.argmin(out["error_mc"])])
    return out


# LLM-assisted (Claude, Claude Code, October 2026): plotting code for this figure.
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.75))
for ax, n, lab, deg_max, ylim in zip(axes, (100, 1000), ("(a)", "(b)"), (14, 20),
                                     ((4e-4, 30.0), (5e-5, 0.3))):
    out = analyse(n)
    results[f"n={n}"] = out
    sel = degrees <= deg_max
    d = degrees[sel]
    ax.semilogy(d, out["error"][sel], "-o", ms=2.5, color=INK, label="Error (bootstrap)")
    ax.semilogy(d, out["bias2"][sel], "-s", ms=2.5, color=COLORS[0],
                label=r"Bias$^2$ measured with $y$ ($\approx$ Bias$^2$ + $\sigma^2$)")
    ax.semilogy(d, out["bias2_true"][sel], "--", color=COLORS[0], label=r"Bias$^2$ measured with $f$")
    ax.semilogy(d, out["variance"][sel], "-^", ms=2.5, color=COLORS[1], label="Variance (bootstrap)")
    ax.semilogy(d, out["variance_mc"][sel], ":", lw=1.8, color=COLORS[1],
                label="Variance (400 fresh training sets)")
    ax.axhline(NOISE**2, color=GREY, lw=0.9, ls=":")
    ax.set_ylim(*ylim)
    ax.set_title(f"$n={n}$ ({int(n * (1 - TEST_SIZE))} training points)", fontsize=8)
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_xticks(d[::2])
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, va="top", fontweight="bold")
axes[0].set_ylabel("MSE contribution")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1))
save(fig, "c_bias_variance")

save_results("part_c", results)
print("Hastie: best degree mean", results["train_test"]["best_degree_mean"], "median",
      results["train_test"]["best_degree_median"])
for n in (100, 1000):
    o = results[f"n={n}"]
    print(f"n={n}: best degree bootstrap {o['best_degree']}, fresh sets {o['best_degree_mc']}")
