"""
Part f): gradient descent with momentum, AdaGrad, RMSprop and Adam for OLS and Ridge
(full-batch), compared with plain GD and with the closed-form solutions.

Produces
  figures/f_optimizers.pdf  (a) iterations to reach relative error 1e-6 against the learning rate
                            for each method, (b) convergence at the best learning rate
  results/part_f.json

Degree 6 (kappa ~ 2e3) is the main test case; degree 10 (kappa ~ 3e6) shows where all
first-order methods stall.

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from optimizers import OPTIMIZERS, RMSprop, gradient_descent, hessian, make_gradient
from plot_style import (GREY, INK, DOUBLE, OPTIMIZER_COLORS, OPTIMIZER_LABELS, OPTIMIZER_MARKERS,
                        save, set_style)
from regression import Scaler, ols_parameters, polynomial_features, ridge_parameters
from settings import main_split, save_results

set_style()
results = {}
x_train, x_test, y_train, y_test = main_split()
TOL = 1e-6
MAX_IT = 100_000
METHODS = ["gd", "momentum", "adagrad", "rmsprop", "adam"]


def scaled_problem(p):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    X = polynomial_features(x_train, p)
    sc = Scaler().fit(X, y_train)
    return sc.transform(X), sc.center(y_train)


def run(method, eta, X, y, lam, theta_star, n_iter=MAX_IT, tol=TOL):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    kind = "ridge" if lam > 0 else "ols"
    th, inf = gradient_descent(make_gradient(kind, X, y, lam), np.zeros(X.shape[1]),
                               OPTIMIZERS[method](eta), n_iter, theta_ref=theta_star, tol=tol)
    return th, inf


# ------------------------------------------------------------------------------------------
# 1) Learning-rate scan at degree 6 (OLS)
# ------------------------------------------------------------------------------------------
X, y = scaled_problem(6)
theta_ols = ols_parameters(X, y)
lmax = np.linalg.eigvalsh(hessian(X)).max()
etas = np.logspace(-4, 0.5, 28)
scan = {}
t_start = time.perf_counter()
for m in METHODS:
    its, final = [], []
    for eta in etas:
        _, inf = run(m, eta, X, y, 0.0, theta_ols)
        its.append(inf["iterations"] if inf["converged"] else np.nan)
        final.append(np.nan if inf["diverged"] else inf["errors"][-1])
    its = np.array(its)
    ok = np.isfinite(its)
    best = int(np.nanargmin(its)) if ok.any() else None
    scan[m] = {"iterations": its, "final_error": final,
               "best_eta": etas[best] if best is not None else None,
               "best_iterations": its[best] if best is not None else None,
               "converging_eta_range": [etas[ok].min(), etas[ok].max()] if ok.any() else None,
               "n_converging": int(ok.sum())}
results["scan_time_s"] = time.perf_counter() - t_start
results["degree6"] = {"etas": etas, "scan": scan, "lambda_max": lmax, "eta_max_gd": 2 / lmax,
                      "eta_max_momentum": 2 * (1 + 0.9) / lmax}

# convergence curves at the best learning rate (RMSprop never reaches 1e-6 with a constant eta;
# it is shown at eta = 1e-2 to make its plateau visible)
curves = {}
for m in METHODS:
    eta = scan[m]["best_eta"] if scan[m]["best_eta"] is not None else 1e-2
    _, inf = run(m, eta, X, y, 0.0, theta_ols, n_iter=30_000, tol=1e-12)
    curves[m] = (eta, inf["errors"])

# RMSprop plateau against eta: with a constant eta the iterate ends in a limit cycle
plateau = {}
for eta in (1e-3, 3e-3, 1e-2, 3e-2):
    _, inf = run("rmsprop", eta, X, y, 0.0, theta_ols, n_iter=30_000, tol=None)
    plateau[eta] = inf["errors"][-1]
results["rmsprop_plateau"] = plateau

# RMSprop with a decaying learning rate eta_t = eta0 * t1 / (t + t1)
class DecayingRMSprop(RMSprop):
    """RMSprop with eta_t = eta0 * t1 / (t + t1)."""

    def __init__(self, eta, t1=100.0, **kw):
        """LLM-assisted: Claude generated the original implementation, as recorded
        in the module declaration. Codex added this function-level attribution
        on 5 October 2026; this tag does not certify the student's own review.
        """
        super().__init__(eta, **kw)
        self.t1 = t1

    def reset(self, n_params):
        """LLM-assisted: Claude generated the original implementation, as recorded
        in the module declaration. Codex added this function-level attribution
        on 5 October 2026; this tag does not certify the student's own review.
        """
        super().reset(n_params)
        self.t = 0

    def update(self, g, eta):
        """LLM-assisted: Claude generated the original implementation, as recorded
        in the module declaration. Codex added this function-level attribution
        on 5 October 2026; this tag does not certify the student's own review.
        """
        self.t += 1
        return super().update(g, eta * self.t1 / (self.t + self.t1))


eta_rms = 1e-2
_, inf = gradient_descent(make_gradient("ols", X, y), np.zeros(6), DecayingRMSprop(eta_rms, t1=100), 30_000,
                          theta_ref=theta_ols, tol=1e-12)
curves["rmsprop_decay"] = (eta_rms, inf["errors"])
results["rmsprop_decay"] = {"eta0": eta_rms, "t1": 100, "final_error": inf["errors"][-1]}

# Ridge (lambda = 1e-3) with the same learning rates
lam = 1e-3
theta_ridge = ridge_parameters(X, y, lam)
ridge = {}
for m in METHODS:
    eta = scan[m]["best_eta"] or curves[m][0]
    _, inf = run(m, eta, X, y, lam, theta_ridge)
    ridge[m] = {"eta": eta, "iterations": inf["iterations"] if inf["converged"] else None,
                "final_error": inf["errors"][-1]}
results["ridge_degree6"] = {"lambda": lam, "runs": ridge}

# ------------------------------------------------------------------------------------------
# 2) Degree 10: small scan, best relative error after 50 000 iterations
# ------------------------------------------------------------------------------------------
X10, y10 = scaled_problem(10)
theta10 = ols_parameters(X10, y10)
ev10 = np.linalg.eigvalsh(hessian(X10))
deg10 = {"kappa": ev10.max() / ev10.min()}
for m in METHODS:
    best = (np.inf, None)
    for eta in np.logspace(-3, 0, 7):
        _, inf = run(m, eta, X10, y10, 0.0, theta10, n_iter=50_000, tol=None)
        if not inf["diverged"] and inf["errors"][-1] < best[0]:
            best = (inf["errors"][-1], eta)
    deg10[m] = {"best_final_error": best[0], "eta": best[1]}
results["degree10"] = deg10

# ------------------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
for m in METHODS:
    ax1.loglog(etas, scan[m]["iterations"], "-" + OPTIMIZER_MARKERS[m], ms=3, color=OPTIMIZER_COLORS[m],
               label=OPTIMIZER_LABELS[m])
ax1.axvline(2 / lmax, color=OPTIMIZER_COLORS["gd"], lw=0.8, ls=":")
ax1.axvline(2 * 1.9 / lmax, color=OPTIMIZER_COLORS["momentum"], lw=0.8, ls=":")
ax1.text(2 / lmax * 0.93, 1.3e5, r"$2/h_{\max}$", fontsize=5.8, ha="right", color=GREY)
ax1.text(2 * 1.9 / lmax * 1.07, 1.3e5, r"$2(1+\beta)/h_{\max}$", fontsize=5.8, color=GREY)
ax1.text(1.2e-2, 1.1e2, r"RMSprop: no constant $\eta$ reaches $10^{-6}$", fontsize=5.8,
         color=OPTIMIZER_COLORS["rmsprop"])
ax1.set_ylim(50, 3e5)
ax1.set_xlabel(r"Learning rate $\eta$")
ax1.set_ylabel(r"Iterations to relative error $10^{-6}$")
ax1.legend(fontsize=6, loc="lower left")
for m in METHODS + ["rmsprop_decay"]:
    eta, e = curves[m]
    base = "rmsprop" if m == "rmsprop_decay" else m
    lab = OPTIMIZER_LABELS[base] + rf" ($\eta={eta:.2g}$)"
    if m == "rmsprop_decay":
        lab = r"RMSprop, $\eta_t=10^{-2}\cdot 100/(t+100)$"
    ax2.loglog(np.arange(1, len(e) + 1), e, color=OPTIMIZER_COLORS[base], lw=1.2,
               ls="--" if m == "rmsprop_decay" else "-", label=lab)
ax2.axhline(TOL, color=GREY, lw=0.8, ls=":")
ax2.set_ylim(1e-10, 3)
ax2.set_xlim(1, 3e4)
ax2.set_xlabel("Iteration $k$")
ax2.set_ylabel(r"$\|\theta_k-\hat\theta_{\mathrm{OLS}}\|/\|\hat\theta_{\mathrm{OLS}}\|$")
ax2.legend(fontsize=5.6, loc="lower left")
ax1.text(0.03, 0.96, "(a)", transform=ax1.transAxes, va="top", fontweight="bold")
ax2.text(0.97, 0.96, "(b)", transform=ax2.transAxes, va="top", ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "f_optimizers")

save_results("part_f", results)
for m in METHODS:
    s = scan[m]
    print(f"{m:9s} best eta {s['best_eta']}, iterations {s['best_iterations']}, range {s['converging_eta_range']}, "
          f"n_ok {s['n_converging']} | ridge {ridge[m]} | deg10 {deg10[m]}")
print("RMSprop decaying", results["rmsprop_decay"], "plateau", plateau, "kappa10", deg10["kappa"])
print("scan time", results["scan_time_s"])
