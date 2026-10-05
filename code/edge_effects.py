"""
Edge effects: how much of the small-sample errors comes from extrapolation near x = -1.

On the main split (seed 2026, 80/20) three test inputs lie below the smallest training input.
This script measures their share of (1) the part-a OLS test error and (2) the part-c bootstrap
variance, and (3) records the realized noise levels behind the part-i comparison of the
cross-validated and the independent test error. It also draws the OLS fits against Runge's
function. Every recomputed quantity is checked against the stored part-a, part-c and part-i
results before anything is written.

Produces
  figures/a_edge_fits.pdf   main-split data and OLS fits of degree 6, 12 and 15
  results/edge_effects.json

Run after part_a_ols.py, part_c_bias_variance.py and part_i_model_selection.py.

LLM-assisted
------------
Tool: Claude (Anthropic, Claude Opus 5.5 via Claude Code), 5 October 2026.
Role: wrote this diagnostic script during a review of the report (code level 4).
Verification: the recomputed test MSEs, bootstrap variances, CV MSE and independent test MSE
must equal the stored part-a, part-c and part-i values (asserted below).
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from plot_style import COLORS, GREY, INK, METHOD_COLORS, RESULTS_DIR, SINGLE, ordered_colors, save, set_style
from regression import PolynomialRegression, make_data, runge
from settings import N_POINTS, NOISE, SEED, main_split, save_results

set_style()
stored = {}
for name in ("part_a", "part_c", "part_i"):
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as f:
        stored[name] = json.load(f)
results = {}

x_train, x_test, y_train, y_test = main_split()
outside = (x_test < x_train.min()) | (x_test > x_train.max())
results["main_split"] = {"training_range": [x_train.min(), x_train.max()],
                         "two_leftmost_training_x": np.sort(x_train)[:2],
                         "outside_test_x": np.sort(x_test[outside]),
                         "n_outside": int(outside.sum()), "n_test": len(x_test)}

# ------------------------------------------------------------------------------------------
# 1) Part a: share of the OLS test error from the extrapolated test inputs
# ------------------------------------------------------------------------------------------
deg_a = np.arange(0, 16)
rows = {"test_mse": [], "share_outside": [], "mse_inside": []}
for p in deg_a:
    se = (y_test - PolynomialRegression(int(p)).fit(x_train, y_train).predict(x_test)) ** 2
    rows["test_mse"].append(se.mean())
    rows["share_outside"].append(se[outside].sum() / se.sum())
    rows["mse_inside"].append(se[~outside].mean())
assert np.allclose(rows["test_mse"], stored["part_a"]["main"]["mse_test"], rtol=1e-10, atol=0)
results["part_a"] = {"degrees": deg_a, **rows}

# ------------------------------------------------------------------------------------------
# 2) Part c (n = 100): share of the bootstrap variance from the same test inputs.
#    Same resampling as resampling.bootstrap_bias_variance: seed SEED + p, B = 200.
# ------------------------------------------------------------------------------------------
deg_c, B = np.arange(0, 15), 200
rows = {"variance": [], "share_outside": [], "variance_inside": [], "error": [], "error_inside": []}
for p in deg_c:
    rng = np.random.default_rng(SEED + int(p))
    pred = np.empty((len(x_test), B))
    for b in range(B):
        idx = rng.integers(0, len(x_train), len(x_train))
        pred[:, b] = PolynomialRegression(int(p)).fit(x_train[idx], y_train[idx]).predict(x_test)
    v = pred.var(axis=1)
    e = np.mean((y_test[:, None] - pred) ** 2, axis=1)
    rows["variance"].append(v.mean())
    rows["share_outside"].append(v[outside].sum() / v.sum())
    rows["variance_inside"].append(v[~outside].mean())
    rows["error"].append(e.mean())
    rows["error_inside"].append(e[~outside].mean())
assert np.allclose(rows["variance"], stored["part_c"]["n=100"]["variance"][:len(deg_c)], rtol=1e-8, atol=0)
assert np.allclose(rows["error"], stored["part_c"]["n=100"]["error"][:len(deg_c)], rtol=1e-8, atol=0)
results["part_c"] = {"degrees": deg_c, **rows,
                     "best_degree": int(deg_c[np.argmin(rows["error"])]),
                     "best_degree_inside": int(deg_c[np.argmin(rows["error_inside"])]),
                     "variance_fresh_sets": stored["part_c"]["n=100"]["variance_mc"][:len(deg_c)],
                     "probability_point_absent_from_resample": (1 - 1 / len(x_train)) ** len(x_train)}

# ------------------------------------------------------------------------------------------
# 3) Part i: realized noise behind the CV and independent test errors of the selected OLS fit
# ------------------------------------------------------------------------------------------
sel = stored["part_i"]["sigma=0.1"]["k10"]["OLS"]["min"]
p_sel = int(sel["degree"])
x, y = make_data(N_POINTS, NOISE, SEED)
x_new, y_new = make_data(2000, NOISE, SEED + 1)
cv_mse, cv_f = [], []
for tr, va in KFold(n_splits=10, shuffle=True, random_state=SEED).split(x):
    pred = PolynomialRegression(p_sel).fit(x[tr], y[tr]).predict(x[va])
    cv_mse.append(np.mean((y[va] - pred) ** 2))
    cv_f.append(np.mean((runge(x[va]) - pred) ** 2))
pred_new = PolynomialRegression(p_sel).fit(x, y).predict(x_new)
test_mse = np.mean((y_new - pred_new) ** 2)
assert np.isclose(np.mean(cv_mse), sel["cv_mse"], rtol=1e-8)
assert np.isclose(test_mse, sel["true_test_mse"], rtol=1e-8)
results["part_i"] = {"degree": p_sel, "cv_mse": np.mean(cv_mse), "cv_error_vs_f": np.mean(cv_f),
                     "noise_variance_in_sample": np.mean((y - runge(x)) ** 2),
                     "test_mse": test_mse, "test_error_vs_f": np.mean((runge(x_new) - pred_new) ** 2),
                     "noise_variance_test_set": np.mean((y_new - runge(x_new)) ** 2)}

# ------------------------------------------------------------------------------------------
# Figure: (a) the fits on the main split, (b) zoom on the left edge. Shading marks the
# regions outside the training range, where every fit extrapolates.
# ------------------------------------------------------------------------------------------
xg = np.linspace(-1, 1, 2001)
fits = {p: PolynomialRegression(p).fit(x_train, y_train).predict(xg) for p in (6, 12, 15)}
styles = list(zip((6, 12, 15), ordered_colors(3, light=0.45, base=METHOD_COLORS["OLS"]), (":", "--", "-")))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE, 3.6), gridspec_kw={"height_ratios": [1.55, 1]})
for ax, (xlo, xhi), (ylo, yhi) in ((ax1, (-1, 1), (-0.45, 1.6)), (ax2, (-1, -0.8), (-2.3, 0.6))):
    for a, b in ((-1.0, x_train.min()), (x_train.max(), 1.0)):
        ax.axvspan(a, b, color=GREY, alpha=0.18, lw=0)
    ax.plot(xg, runge(xg), color=INK, lw=1.3, label=r"$f(x)$")
    for p, c, ls in styles:
        ax.plot(xg, fits[p], color=c, ls=ls, lw=1.2, label=f"OLS, $p={p}$")
    ax.scatter(x_train, y_train, s=4, color=GREY, lw=0, label="Training data", zorder=3)
    ax.scatter(x_test[~outside], y_test[~outside], s=10, marker="s", facecolors="none",
               edgecolors=INK, lw=0.6, label="Test data", zorder=3)
    ax.scatter(x_test[outside], y_test[outside], s=14, marker="s", color=COLORS[7],
               label="Test, extrapolated", zorder=4)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("$y$")
ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
ax2.set_xticks([-1, -0.95, -0.9, -0.85, -0.8])
ax2.set_xlabel("$x$")
handles, labels = ax1.get_legend_handles_labels()
curves = ax1.legend(handles[:4], labels[:4], loc="upper left", fontsize=5.6, handlelength=1.6)
ax1.add_artist(curves)
ax1.legend(handles[4:], labels[4:], loc="upper right", fontsize=5.6, handlelength=1.0)
ax1.text(0.97, 0.04, "(a)", transform=ax1.transAxes, va="bottom", ha="right", fontweight="bold")
ax2.text(0.97, 0.06, "(b) left edge", transform=ax2.transAxes, va="bottom", ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "a_edge_fits")

save_results("edge_effects", results)
print("outside test x:", results["main_split"]["outside_test_x"], "training range", results["main_split"]["training_range"])
for k, p in enumerate(deg_a):
    print(f"a) p={p:2d} test MSE {results['part_a']['test_mse'][k]:.5f}, share outside "
          f"{results['part_a']['share_outside'][k]:.3f}, inside-only MSE {results['part_a']['mse_inside'][k]:.5f}")
for k, p in enumerate(deg_c):
    print(f"c) p={p:2d} bootstrap variance {results['part_c']['variance'][k]:.4g}, share outside "
          f"{results['part_c']['share_outside'][k]:.4f}, inside-only {results['part_c']['variance_inside'][k]:.4g}, "
          f"fresh sets {results['part_c']['variance_fresh_sets'][k]:.4g}")
print("i)", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in results["part_i"].items()})
