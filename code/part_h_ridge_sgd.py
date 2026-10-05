"""
Part h) for Ridge: stochastic gradient descent compared with the closed-form Ridge solution.

Same problem as part_h_sgd.py (degree 6, main split, standardized features), but with the Ridge
cost and lambda = 1e-3, the penalty used in parts e and f. For each update rule we choose the
learning rate that gives the lowest training cost after the last epoch, and then report how close
SGD gets to the closed-form Ridge solution of part b and how long it takes, compared with
full-batch gradient descent on the same number of passes over the data.

Writes results/part_h_ridge.json.

LLM-assisted (code level 4): generated with Claude Opus 5.5 (Claude Code), 5 October 2026, using
the mini-batch SGD loop of weeks 38-39 as implemented in optimizers.py; no module was changed.
"""

import time

import numpy as np

from optimizers import (GD, OPTIMIZERS, cost_ridge, gradient_descent, grad_ridge, hessian,
                        make_gradient, stochastic_gradient_descent)
from plot_style import save_results
from regression import Scaler, main_split, polynomial_features, ridge_parameters

LAM = 1e-3                        # same penalty as the Ridge runs in parts e and f
P = 6                             # same degree as part h
M = 10                            # batch size: 8 mini-batches per epoch
EPOCHS = 2000
SEEDS = range(3)                  # shuffling seeds; we report medians
ETAS = np.logspace(-4, 0, 9)      # learning rates tried for every update rule

# Data: the same split and training-only scaling as in the other parts.
x_train, _, y_train, _ = main_split()
X_raw = polynomial_features(x_train, P)
scaler = Scaler().fit(X_raw, y_train)
X, y = scaler.transform(X_raw), scaler.center(y_train)
theta_ridge = ridge_parameters(X, y, LAM)            # closed-form answer (part b)
cost_min = cost_ridge(theta_ridge, X, y, LAM)


def relative_error(theta):
    """Distance to the closed-form Ridge coefficients, relative to their size."""
    return np.linalg.norm(theta - theta_ridge) / np.linalg.norm(theta_ridge)


def excess_cost(theta):
    """Training cost above the minimum; infinite if the run diverged."""
    if not np.all(np.isfinite(theta)):
        return np.inf
    return float(cost_ridge(theta, X, y, LAM) - cost_min)


def run_sgd(method, eta, seed):
    """One SGD run from theta = 0; returns the final coefficients and the run time."""
    start = time.perf_counter()
    theta, _ = stochastic_gradient_descent(grad_ridge, X, y, LAM, np.zeros(P),
                                           OPTIMIZERS[method](eta), EPOCHS, M, seed=seed)
    return theta, time.perf_counter() - start


results = {"setup": {"degree": P, "lambda": LAM, "batch_size": M, "epochs": EPOCHS,
                     "seeds": len(SEEDS), "etas": ETAS}, "sgd": {}}

for method in ("gd", "momentum", "adagrad", "rmsprop", "adam"):
    best = None
    for eta in ETAS:
        runs = [run_sgd(method, eta, seed) for seed in SEEDS]
        cost = np.median([excess_cost(theta) for theta, _ in runs])   # training data only
        if best is None or cost < best[0]:
            best = (cost, eta, runs)
    cost, eta, runs = best
    results["sgd"][method] = {
        "eta": eta, "excess_cost": cost,
        "relative_error": float(np.median([relative_error(theta) for theta, _ in runs])),
        "time_s": float(np.median([seconds for _, seconds in runs]))}

# Full-batch gradient descent with the optimal fixed step 2/(h_max + h_min), Eq. (4.26) of the
# lecture notes. One full step costs as much as one SGD epoch, Eq. (4.38), so we use EPOCHS steps.
h, vectors = np.linalg.eigh(hessian(X, LAM))         # eigenvalues in increasing order
eta_opt = 2.0 / (h.max() + h.min())
start = time.perf_counter()
theta_gd, _ = gradient_descent(make_gradient("ridge", X, y, LAM), np.zeros(P), GD(eta_opt), EPOCHS)
results["full_batch_gd"] = {"eta": eta_opt, "excess_cost": excess_cost(theta_gd),
                            "relative_error": float(relative_error(theta_gd)),
                            "time_s": time.perf_counter() - start}

# The error left after k updates lies mainly along the flattest Hessian direction, which shrinks
# by the factor (1 - eta*h_min)^k, Eq. (4.19) of the lecture notes. Starting from theta = 0, that
# direction carries the share computed below of the initial error.
share = abs(vectors[:, 0] @ theta_ridge) / np.linalg.norm(theta_ridge)
updates = {"full_batch_gd": (eta_opt, EPOCHS),
           "plain_sgd": (results["sgd"]["gd"]["eta"], EPOCHS * int(np.ceil(len(y) / M)))}
results["slow_direction"] = {"h_min": h.min(), "share_of_initial_error": share}
for name, (eta, k) in updates.items():
    results["slow_direction"][name] = {"eta_times_updates": eta * k,
                                       "predicted_relative_error": share * (1 - eta * h.min()) ** k}

save_results("part_h_ridge", results)
for name, r in {**results["sgd"], "full GD": results["full_batch_gd"]}.items():
    print(f"{name:9s} eta {r['eta']:.3g}  relative error {r['relative_error']:.3g}  "
          f"excess cost {r['excess_cost']:.2e}  time {r['time_s']:.3f} s")
