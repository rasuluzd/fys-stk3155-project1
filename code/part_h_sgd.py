"""
Part h): stochastic (mini-batch) gradient descent for OLS at degree 6.

We study the mini-batch size M, the number of epochs, constant vs. decaying learning rates
eta_t = eta0 * t1 / (t + t1), and the five update rules of part f). The learning rate is tuned
separately for every (method, M), as a fair comparison requires, and the curves shown are medians
over five shuffling seeds. The accuracy measure is the excess training cost C(theta) - C(theta_OLS);
sigma^2 p / n is a statistical reference scale, not an optimization stopping theorem.

Produces
  figures/h_sgd.pdf   (a) excess cost against epochs for several M, constant and decaying eta,
                      (b) the five update rules with M = 10
  results/part_h.json

LLM-assisted (code level 2): plotting code generated with Claude (Claude Code, October 2026);
docstrings edited with OpenAI Codex, 5 October 2026.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from optimizers import OPTIMIZERS, cost_ols, grad_ols, hessian, stochastic_gradient_descent
from plot_style import (GREY, INK, DOUBLE, OPTIMIZER_COLORS, OPTIMIZER_LABELS, ordered_colors, save,
                        set_style)
from regression import Scaler, ols_parameters, polynomial_features
from settings import NOISE, main_split, save_results

set_style()
results = {}
x_train, x_test, y_train, y_test = main_split()
P = 6
X_raw = polynomial_features(x_train, P)
sc = Scaler().fit(X_raw, y_train)
X, y = sc.transform(X_raw), sc.center(y_train)
n = len(y)
theta_ols = ols_parameters(X, y)
c_ols = cost_ols(theta_ols, X, y)
floor = NOISE**2 * P / n
ev = np.linalg.eigvalsh(hessian(X))
lmax, lmin = ev.max(), ev.min()
row_norm_max = np.max(np.sum(X**2, axis=1))
EPOCHS = 2000
SEEDS = range(5)
results["setup"] = {"degree": P, "n_train": n, "statistical_floor": floor, "lambda_max_full": lmax,
                    "eta_max_full_batch": 2 / lmax, "max_row_norm_sq": row_norm_max,
                    "eta_max_single_point": 1 / row_norm_max, "epochs": EPOCHS}


def excess(theta):
    """Excess training cost C(theta) - C(theta_OLS)."""
    return cost_ols(theta, X, y) - c_ols


def run(method, eta, M, schedule=None, epochs=EPOCHS, seed=0):
    """One SGD run for an update rule, batch size and schedule; records excess cost per epoch."""
    t0 = time.perf_counter()
    th, info = stochastic_gradient_descent(grad_ols, X, y, 0.0, np.zeros(P), OPTIMIZERS[method](eta),
                                           epochs, M, schedule=schedule, seed=seed, callback=excess,
                                           theta_ref=theta_ols)
    info["time"] = time.perf_counter() - t0
    return th, info


def score(info):
    """Tuning criterion: geometric mean of the excess cost over the last 10 % of the epochs."""
    h = info["history"]
    if info["diverged"] or len(h) < EPOCHS or not np.all(np.isfinite(h)):
        return np.inf
    return np.exp(np.mean(np.log(np.maximum(h[-EPOCHS // 10:], 1e-300))))


def median_over_seeds(method, eta, M, schedule=None):
    """Median excess-cost history, final error and run time over the shuffling seeds."""
    hist, rel, times = [], [], []
    for s in SEEDS:
        _, info = run(method, eta, M, schedule, seed=s)
        hist.append(info["history"])
        rel.append(info["errors"][-1])
        times.append(info["time"])
    hist = np.array(hist)
    return {"median": np.median(hist, axis=0), "q25": np.quantile(hist, 0.25, axis=0),
            "q75": np.quantile(hist, 0.75, axis=0), "final_relative_error": float(np.median(rel)),
            "time_s": float(np.median(times))}


def epochs_below(h, level):
    """First epoch at which the history h falls below level, or None."""
    idx = np.where(h < level)[0]
    return int(idx[0]) + 1 if len(idx) else None


def log_bins(h, n_bins=60):
    """Geometric mean of h over logarithmically spaced epoch bins (for readable log-log plots)."""
    edges = np.unique(np.logspace(0, np.log10(len(h)), n_bins).astype(int))
    centres, values = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        centres.append(np.sqrt(a * b))
        values.append(np.exp(np.mean(np.log(h[a - 1:b - 1]))))
    return np.array(centres), np.array(values)


# ------------------------------------------------------------------------------------------
# 1) Plain SGD: batch size, constant vs decaying learning rate
# ------------------------------------------------------------------------------------------
etas = np.logspace(-3.5, np.log10(0.25), 14)
batch = {}
for M in (1, 5, 20, 80):
    m = int(np.ceil(n / M))
    scores = [score(run("gd", eta, M)[1]) for eta in etas]
    eta_c = etas[int(np.argmin(scores))]
    best_d = (np.inf, None, None)
    for factor in (1.0, 2.0, 4.0):
        eta0 = min(factor * eta_c, 0.9 * 2 / lmax)
        for t1_epochs in (10, 50, 250):
            t1 = t1_epochs * m
            s = score(run("gd", eta0, M, schedule=lambda t, e=eta0, t1=t1: e * t1 / (t + t1))[1])
            if s < best_d[0]:
                best_d = (s, eta0, t1_epochs)
    eta0, t1 = best_d[1], best_d[2] * m
    const = median_over_seeds("gd", eta_c, M)
    decay = median_over_seeds("gd", eta0, M, schedule=lambda t: eta0 * t1 / (t + t1))
    batch[M] = {"updates_per_epoch": m, "eta_scan": etas, "scores": scores,
                "constant": {"eta": eta_c, **const, "epochs_to_floor": epochs_below(const["median"], floor)},
                "decaying": {"eta0": eta0, "t1_epochs": best_d[2], **decay,
                             "epochs_to_floor": epochs_below(decay["median"], floor)}}
results["batch_size"] = batch

# full-batch GD with the optimal fixed step: iterations (= epochs) to reach the floor
_, info_full = run("gd", 2 / (lmax + lmin), n)
results["full_batch_eta_opt"] = {"eta": 2 / (lmax + lmin), "final_excess": info_full["history"][-1],
                                 "epochs_to_floor": epochs_below(info_full["history"], floor),
                                 "time_s": info_full["time"]}

# ------------------------------------------------------------------------------------------
# 2) The five update rules with M = 10, learning rate tuned for each
# ------------------------------------------------------------------------------------------
M10 = 10
etas_m = np.logspace(-4, 0, 17)
methods = {}
for name in ("gd", "momentum", "adagrad", "rmsprop", "adam"):
    scores = [score(run(name, e, M10)[1]) for e in etas_m]
    eta = etas_m[int(np.argmin(scores))]
    med = median_over_seeds(name, eta, M10)
    methods[name] = {"eta": eta, **med, "epochs_to_floor": epochs_below(med["median"], floor),
                     "final_excess": med["median"][-1], "n_stable_etas": int(np.sum(np.isfinite(scores))),
                     "eta_scan": etas_m, "scores": scores}
results["methods_M10"] = methods

# ------------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------------
# LLM-assisted (Claude, Claude Code, October 2026): plotting code for this figure.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
cols = ordered_colors(4, 0.35, 1.0)
for c, M in zip(cols, (1, 5, 20, 80)):
    b = batch[M]
    xc, yc = log_bins(b["constant"]["median"])
    ax1.loglog(xc, yc, "-", color=c, lw=1.3, label=rf"$M={M}$")
    xd, yd = log_bins(b["decaying"]["median"])
    ax1.loglog(xd, yd, "--", color=c, lw=1.3)
ax1.axhline(floor, color=GREY, lw=0.9, ls=":")
ax1.text(1.15, floor * 0.62, r"reference scale $\sigma^2p/n$", fontsize=6, color=GREY)
ax1.plot([], [], "-", color=INK, lw=1.0, label=r"constant $\eta$")
ax1.plot([], [], "--", color=INK, lw=1.0, label=r"$\eta_t=\eta_0t_1/(t+t_1)$")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"$C(\theta)-C(\hat\theta_{\mathrm{OLS}})$ (training)")
ax1.set_ylim(1e-6, 0.1)
ax1.legend(fontsize=5.6, loc="lower left", ncol=2, columnspacing=0.8)
for name, d in methods.items():
    xm, ym = log_bins(d["median"])
    ax2.loglog(xm, ym, color=OPTIMIZER_COLORS[name], lw=1.3,
               label=OPTIMIZER_LABELS[name] + rf", $\eta={d['eta']:.2g}$")
ax2.axhline(floor, color=GREY, lw=0.9, ls=":")
ax2.set_xlabel("Epoch")
ax2.set_ylabel(r"$C(\theta)-C(\hat\theta_{\mathrm{OLS}})$ (training)")
ax2.set_ylim(1e-6, 0.1)
ax2.legend(fontsize=5.6, loc="lower left", title=r"$M=10$, constant $\eta$", title_fontsize=6)
for ax, lab in zip((ax1, ax2), ("(a)", "(b)")):
    ax.text(0.97, 0.96, lab, transform=ax.transAxes, va="top", ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "h_sgd")

save_results("part_h", results)
print("setup", results["setup"])
for M, b in batch.items():
    c, d = b["constant"], b["decaying"]
    print(f"M={M}: const eta {c['eta']:.3g} final {c['median'][-1]:.2e} to-floor {c['epochs_to_floor']} "
          f"rel {c['final_relative_error']:.2e} t={c['time_s']:.2f}s | decay eta0 {d['eta0']:.3g} t1 {d['t1_epochs']} ep "
          f"final {d['median'][-1]:.2e} to-floor {d['epochs_to_floor']} rel {d['final_relative_error']:.2e} t={d['time_s']:.2f}s")
print("full batch", results["full_batch_eta_opt"])
for name, d in methods.items():
    print(f"{name}: eta {d['eta']:.3g} final {d['final_excess']:.2e} to-floor {d['epochs_to_floor']} "
          f"rel {d['final_relative_error']:.2e} t {d['time_s']:.2f}s stable etas {d['n_stable_etas']}")
