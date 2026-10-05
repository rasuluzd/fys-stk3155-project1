"""Part b: Ridge regression on the main split over a grid of degrees and penalties, compared
with OLS, and the singular values of the degree-15 design matrix.

Writes results/part_b.json. ridge_supplement.py reads the grid, makes the Ridge figure of the
report and runs the paired sample-size/noise comparisons.

LLM-assisted (code level 4): written with Claude (Claude Code, October 2026), simplified
on 5 October 2026.
"""

import numpy as np

from plot_style import save_results
from regression import PolynomialRegression, Scaler, main_split, mse, polynomial_features

results = {}
x_train, x_test, y_train, y_test = main_split()

# ------------------------------------------------------------------------------------------
# 1) Train and test MSE over (degree, lambda), and the best OLS fit for comparison
# ------------------------------------------------------------------------------------------
degrees = np.arange(1, 21)
lams = np.logspace(-10, 1, 45)
test_map = np.empty((len(degrees), len(lams)))
train_map = np.empty_like(test_map)
for a, p in enumerate(degrees):
    for b, lam in enumerate(lams):
        model = PolynomialRegression(int(p), "ridge", lam).fit(x_train, y_train)
        test_map[a, b] = mse(y_test, model.predict(x_test))
        train_map[a, b] = mse(y_train, model.predict(x_train))
ols_test = [mse(y_test, PolynomialRegression(int(p)).fit(x_train, y_train).predict(x_test))
            for p in degrees]
a, b = np.unravel_index(np.argmin(test_map), test_map.shape)
results["main"] = {"degrees": degrees, "lambdas": lams, "test_mse": test_map, "train_mse": train_map,
                   "ols_test_mse": ols_test, "best_degree": int(degrees[a]), "best_lambda": lams[b],
                   "best_test_mse": test_map[a, b],
                   "ols_best_degree": int(degrees[np.argmin(ols_test)]),
                   "ols_best_test_mse": float(np.min(ols_test))}

# ------------------------------------------------------------------------------------------
# 2) Singular values at degree 15: Ridge scales mode j by d_j^2 / (d_j^2 + n lambda)
# ------------------------------------------------------------------------------------------
X = polynomial_features(x_train, 15)
Xs = Scaler().fit(X, y_train).transform(X)
results["singular_values_degree15"] = np.linalg.svd(Xs, compute_uv=False)

save_results("part_b", results)
print("best Ridge (degree, lambda, test MSE):", results["main"]["best_degree"], lams[b], test_map[a, b])
print("best OLS (degree, test MSE):", results["main"]["ols_best_degree"], results["main"]["ols_best_test_mse"])
print("degree-15 singular values:", results["singular_values_degree15"])
