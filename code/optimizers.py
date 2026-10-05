"""Gradient-based optimization for Project 1, parts e-h.

How to read this module
-----------------------
Read a cost, its gradient, GD.update, then gradient_descent. That is the
course's "cost -> gradient -> step -> loop" structure. The five small classes
only keep state between updates. They are an alternative to the state
dictionary in week38tuesday, Case 1, not additional optimization algorithms.
update(g, eta) returns a step; the driver performs theta = theta - step.

Conventions used in every experiment
------------------------------------
X has shape (n, p), and y and theta are one-dimensional arrays. The caller
supplies training-standardized columns and centered targets; these routines
do not fit a scaler or include an intercept. Costs use mean squared error:
    OLS:   ||y - X theta||^2 / n
    Ridge: OLS + lam * ||theta||_2^2
    Lasso: OLS + lam * ||theta||_1
The Ridge gradient therefore adds 2*lam*theta, including in a mini-batch.
This matches the week-37/38 cost convention; sklearn uses alpha=n*lam for
Ridge and alpha=lam/2 for Lasso. Eta here is gamma in the weekly exercises.

Course connections (concepts and checked equations, not a copying history)
-------------------------------------------------------------------------
* OLS/Ridge GD and Hessian: book 4.3-4.5; week37 Exercise 4a-c and Tuesday
  Case 1. Momentum: book 4.6; week37 Exercise 6 and Tuesday Case 2.
* Automatic differentiation: book 4.14, especially 4.14.7-4.14.8; week37
  Exercise 4d. This project uses the permitted autograd alternative to JAX.
* Adaptive updates: book 4.8-4.11; week38 Exercises 1-3, Tuesday Case 1.
  The defaults beta=.9, rho=.99, beta1=.9, beta2=.999, epsilon=1e-8 and
  placement of epsilon outside the square root match that Tuesday listing.
* Batches/epochs/schedules: book 4.7.8-4.7.9; week38 Exercise 4, Tuesday
  Case 2. This driver reshuffles without replacement each epoch.
* Lasso subgradients: book 3.9 and 4.14.7; Project 1 part g.
  Codex removed unused proximal solvers on 5 October 2026.

Differences to explain when comparing with the lecture listings
--------------------------------------------------------------
The full-batch benchmark stops on relative coefficient error to a supplied
reference, rather than the gradient-norm stopping rule in week37tuesday.
Histories store scalar errors/costs, not all coefficient vectors. SGD's
schedule sees t=0 on its first update; Adam has a separate counter starting
at 1 for its bias corrections. State resets once per driver call, not at
every epoch. Bounds for plain GD do not apply unchanged to adaptive rules.

LLM-assisted
------------
Code level 2. The numerical code was written by the author. The docstrings and
the course mapping above were written with LLM assistance (Claude via Claude
Code and OpenAI Codex, October 2026). Apart from removing the unused proximal
solvers, Codex did not change executable statements.
Tests check analytical/AD agreement, all five update rules on one small
Ridge case and selected GD/SGD properties.
The experiment scripts supply further comparisons; these checks do not
prove convergence of every optimizer for every objective or learning rate.
"""

import numpy as np
import autograd.numpy as anp
from autograd import grad as _autograd_grad


# ----------------------------------------------------------------------------------------------
# Cost functions (autograd.numpy, so they can be differentiated automatically)
# ----------------------------------------------------------------------------------------------
def cost_ols(theta, X, y, lam=0.0):
    """Return mean squared residuals; lam is ignored for a common calling signature.

    Book 4.3 / week37 Exercise 4d. autograd.numpy keeps this scalar cost
    differentiable by the same AD call used for the penalized costs.
    """
    return anp.mean((y - anp.dot(X, theta)) ** 2)


def cost_ridge(theta, X, y, lam):
    """Add lam times the squared slope norm to the mean squared residuals.

    Book 4.4 / week37 Exercise 4b. Centering is done by the caller, so the
    intercept is absent and cannot accidentally be penalized here.
    """
    return cost_ols(theta, X, y) + lam * anp.sum(theta**2)


def cost_lasso(theta, X, y, lam):
    """Add the L1 slope penalty to the mean squared residuals.

    Book 3.9 / Project 1 part g. abs is not differentiable at zero;
    autograd selects a subgradient there, as checked in part_g_lasso.py.
    """
    return cost_ols(theta, X, y) + lam * anp.sum(anp.abs(theta))


COSTS = {"ols": cost_ols, "ridge": cost_ridge, "lasso": cost_lasso}


# ----------------------------------------------------------------------------------------------
# Analytical gradients
# ----------------------------------------------------------------------------------------------
def grad_ols(theta, X, y, lam=0.0):
    """Return (2/n) X.T @ (X @ theta - y), using the supplied batch length.

    Book 4.3 / week37 Exercise 4a. X @ theta - y is the residual; X.T
    sums each feature's contribution. The unused lam keeps the shared API.
    """
    return (2.0 / X.shape[0]) * (X.T @ (X @ theta - y))


def grad_ridge(theta, X, y, lam):
    """Add 2*lam*theta to the OLS gradient (book 4.4, week37 Exercise 4b).

    Use the same lam in full and mini-batch gradients: it is not multiplied
    by n/batch_size because each data-fit gradient is already an average.
    """
    return grad_ols(theta, X, y) + 2.0 * lam * theta


def grad_lasso(theta, X, y, lam):
    # np.sign(0) = 0, which is a valid subgradient of |theta_j| at theta_j = 0
    """Return the MSE gradient plus lam*sign(theta), choosing sign(0)=0.

    Book 3.9 and 4.14.7 / Project 1 part g. Zero is a valid subgradient
    of abs at zero, not an ordinary derivative. A fixed subgradient step
    does not guarantee exactly zero fitted coefficients.
    """
    return grad_ols(theta, X, y) + lam * np.sign(theta)


ANALYTICAL_GRADIENTS = {"ols": grad_ols, "ridge": grad_ridge, "lasso": grad_lasso}

# Automatic differentiation: autograd traces the Python cost function and applies the chain
# rule in reverse mode. grad(f) differentiates with respect to the first argument (theta).
AUTOGRAD_GRADIENTS = {name: _autograd_grad(cost) for name, cost in COSTS.items()}


def autograd_gradient(method):
    """Select the AD-generated gradient with respect to theta.

    Book 4.14.8 / week37 Exercise 4d. AD applies the chain rule through
    the scalar cost; this does not use finite-difference perturbations.
    """
    return AUTOGRAD_GRADIENTS[method]


def make_gradient(method, X, y, lam=0.0, use_autograd=False):
    """Bind one training problem, leaving only theta as the gradient argument.

    The returned lambda is a Python closure: it remembers X, y and lam.
    This is a convenience for the course's interchangeable-gradient loop,
    not a new algorithm. use_autograd changes only the gradient source.
    """
    g = AUTOGRAD_GRADIENTS[method] if use_autograd else ANALYTICAL_GRADIENTS[method]
    return lambda theta: g(theta, X, y, lam)


# ----------------------------------------------------------------------------------------------
# Hessian and the learning-rate bound
# ----------------------------------------------------------------------------------------------
def hessian(X, lam=0.0):
    """Return H=(2/n) X.T @ X + 2*lam*I for the OLS/Ridge quadratic.

    Book 4.5 / week37 Exercise 4c. This is constant in theta and does
    not describe the nonsmooth Lasso penalty. X has no intercept column.
    """
    n, p = X.shape
    return (2.0 / n) * (X.T @ X) + 2.0 * lam * np.eye(p)


def max_learning_rate(X, lam=0.0):
    """Return the strict plain-GD upper bound 2/h_max (book 4.5).

    For positive definite H, 0 < eta < this bound contracts every error
    mode. Equality is not safe; a singular H leaves null directions
    undamped. This bound does not cover momentum or adaptive updates.
    """
    return 2.0 / np.linalg.eigvalsh(hessian(X, lam)).max()


# ----------------------------------------------------------------------------------------------
# Update rules. Each object returns the step that is subtracted from theta.
# ----------------------------------------------------------------------------------------------
class GD:
    """Plain gradient descent (book 4.3; week37 Exercise 4).

    """

    name = "Plain GD"

    def __init__(self, eta):
        """Store the base learning rate; drivers may supply a scheduled rate to update."""
        self.eta = eta

    def reset(self, n_params):
        """Do nothing: plain GD has no accumulated state to clear."""
        pass

    def update(self, g, eta):
        """Return eta times the gradient; the driver subtracts this step from theta."""
        return eta * g


class Momentum(GD):
    """Heavy-ball momentum: change <- gamma*change + eta*g.

    Book 4.6 / week37 Exercise 6. The class stores the same previous
    change as the lecture state["v"]. Here gamma is the momentum weight;
    eta, not gamma, is the learning rate. This is not Nesterov momentum.
    """

    name = "Momentum"

    def __init__(self, eta, gamma=0.9):
        """Store eta and the previous-step weight gamma (beta in the course notation)."""
        self.eta, self.gamma = eta, gamma

    def reset(self, n_params):
        """Start a new run with zero previous change in every coordinate."""
        self.change = np.zeros(n_params)

    def update(self, g, eta):
        """Add the current gradient step to gamma times the previous change."""
        self.change = eta * g + self.gamma * self.change
        return self.change


class AdaGrad(GD):
    """Coordinatewise accumulated-square scaling (book 4.9; week38 Exercise 2).

    G remembers every squared gradient in the current run. The denominator
    is sqrt(G)+delta, matching the Tuesday listing. It grows with this
    history; an exact 1/sqrt(k) rate need not hold when gradients change.
    """

    name = "AdaGrad"

    def __init__(self, eta, delta=1e-8):
        """Store the base step and the small denominator offset outside the square root."""
        self.eta, self.delta = eta, delta

    def reset(self, n_params):
        """Clear the sum of squared gradients once at the start of a run."""
        self.G = np.zeros(n_params)

    def update(self, g, eta):
        """Accumulate g squared coordinatewise, then divide the step by its square root."""
        self.G += g * g
        return eta * g / (self.delta + np.sqrt(self.G))


class RMSprop(GD):
    """Exponential squared-gradient memory (book 4.10; week38 Exercise 2).

    s forgets older gradients with weight rho. There is no bias correction.
    A fixed eta can give an error plateau; neither this class nor the
    lecture-style update guarantees a target precision for every eta.
    """

    name = "RMSprop"

    def __init__(self, eta, rho=0.99, delta=1e-8):
        """Store eta, the squared-gradient memory rho, and the denominator offset."""
        self.eta, self.rho, self.delta = eta, rho, delta

    def reset(self, n_params):
        """Clear the moving squared-gradient average for a new run."""
        self.s = np.zeros(n_params)

    def update(self, g, eta):
        """Update the moving average and return the coordinatewise scaled step."""
        self.s = self.rho * self.s + (1.0 - self.rho) * g * g
        return eta * g / (self.delta + np.sqrt(self.s))


class Adam(GD):
    """Bias-corrected first and second moments (book 4.11; week38 Exercise 2).

    m is the gradient average; v is the squared-gradient average. These
    state variables reproduce the lecture dictionary formulation. Observed
    success across a rate grid is not unconditional convergence.
    """

    name = "Adam"

    def __init__(self, eta, beta1=0.9, beta2=0.999, delta=1e-8):
        """Store the two moment weights, base step and denominator offset."""
        self.eta, self.beta1, self.beta2, self.delta = eta, beta1, beta2, delta

    def reset(self, n_params):
        """Clear both moments and reset the bias-correction update counter."""
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def update(self, g, eta):
        """Update both moments, correct their zero-initialization bias, then form a step.

        self.t becomes 1 before the first correction, avoiding a zero
        denominator. This counter is independent of SGD's schedule index.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return eta * m_hat / (np.sqrt(v_hat) + self.delta)


OPTIMIZERS = {"gd": GD, "momentum": Momentum, "adagrad": AdaGrad, "rmsprop": RMSprop, "adam": Adam}


# ----------------------------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------------------------
def _relative_error(theta, theta_ref):
    """Measure coefficient error relative to a supplied nonzero reference vector.

    This benchmark metric requires ||theta_ref|| > 0. It is neither a
    prediction error nor a stopping test available without a reference.
    """
    return np.linalg.norm(theta - theta_ref) / np.linalg.norm(theta_ref)


def gradient_descent(gradient, theta0, optimizer, n_iter, theta_ref=None, tol=None,
                     record_every=1):
    """Run the full-batch loop with a chosen gradient and update rule.

    Course connection: week37 Tuesday Case 1 and week38 Exercise 2.
    Each iteration computes the gradient, asks the optimizer for a step,
    and subtracts it. The optimizer is reset once for each call.

    Parameters
    ----------
    gradient : callable theta -> gradient on the full training set
    theta0 : initial parameters (copied, not modified in place)
    optimizer : GD, Momentum, AdaGrad, RMSprop or Adam; uses optimizer.eta
    n_iter : maximum number of updates
    theta_ref : optional nonzero reference coefficients used to measure error
    tol : stop at relative coefficient error < tol; requires theta_ref
    record_every : retain a relative-error sample every this many updates

    Returns
    -------
    theta, info, with errors, iterations, converged and diverged.
    converged means the supplied reference-error criterion was met, not
    that a general optimality test passed. Without a reference, the loop
    runs to its budget unless the finite-value/large-coefficient guard trips.
    Unlike the lecture example, this records errors rather than every theta;
    the last error is only stored if its iteration meets record_every.
    """
    theta = np.array(theta0, dtype=float)
    optimizer.reset(theta.size)
    errors = []
    converged = diverged = False
    k = 0
    for k in range(1, n_iter + 1):
        step = optimizer.update(gradient(theta), optimizer.eta)
        theta = theta - step
        if not np.all(np.isfinite(theta)) or np.abs(theta).max() > 1e12:
            diverged = True
            break
        if theta_ref is not None:
            err = _relative_error(theta, theta_ref)
            if k % record_every == 0:
                errors.append(err)
            if tol is not None and err < tol:
                converged = True
                break
    return theta, {"errors": np.array(errors), "iterations": k,
                   "converged": converged, "diverged": diverged}


def inverse_time_schedule(t0, t1):
    """Return eta(t)=t0/(t+t1), with t1>0 and update index t starting at zero.

    Book 4.7.8 / week38 Exercise 4. eta(0)=t0/t1; the experiments'
    eta0*t1/(t+t1) form is equivalent when t0=eta0*t1. This module's
    first schedule index differs by one from the Tuesday template.
    """
    return lambda t: t0 / (t + t1)


def stochastic_gradient_descent(grad_fn, X, y, lam, theta0, optimizer, n_epochs, batch_size,
                                schedule=None, seed=None, theta_ref=None,
                                callback=None):
    """Run the same update rules on reshuffled mini-batches, once per epoch.

    Course connection: book 4.7.8-4.7.9, week38 Exercise 4 and Tuesday Case 2.
    Each epoch uses every training observation exactly once; batches in an
    epoch are therefore not independent draws. With batch_size=n, one epoch
    reproduces one full-batch update up to summation rounding.

    grad_fn(theta, X_batch, y_batch, lam) normalizes by the actual batch
    length, including a possibly shorter final batch. Ridge adds 2*lam*theta
    on every update; the penalty is not rescaled by n/batch_size. The caller
    must supply already-scaled X/centered y and a positive batch_size.

    schedule : None uses optimizer.eta; otherwise schedule(t) sees update
               indices 0, 1, ... . To express decay in epochs, multiply its
               time scale by ceil(n/batch_size).
    theta_ref : optional nonzero reference; relative error is recorded per epoch
    callback : optional scalar diagnostic evaluated after every epoch

    State is reset only at entry and persists across epochs. This driver
    has a fixed epoch budget, not a tolerance-based stopping rule; the
    finite-value/large-coefficient guard is checked after each epoch.
    Returns theta and info with per-epoch errors/history, update count,
    and the divergence flag. Histories contain diagnostics, not all theta.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = int(np.ceil(n / batch_size))
    theta = np.array(theta0, dtype=float)
    optimizer.reset(theta.size)
    errors, history = [], []
    diverged = False
    t = 0
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for i in range(m):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            eta = optimizer.eta if schedule is None else schedule(t)
            g = grad_fn(theta, X[idx], y[idx], lam)
            theta = theta - optimizer.update(g, eta)
            t += 1
        if not np.all(np.isfinite(theta)) or np.abs(theta).max() > 1e12:
            diverged = True
            break
        if theta_ref is not None:
            errors.append(_relative_error(theta, theta_ref))
        if callback is not None:
            history.append(callback(theta))
    return theta, {"errors": np.array(errors), "history": np.array(history), "updates": t,
                   "diverged": diverged}
