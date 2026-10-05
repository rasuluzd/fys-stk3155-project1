# FYS-STK3155/4155 Project 1: Regression, resampling and gradient descent

Repository: https://github.com/rasuluzd/fys-stk3155-project1 (private).

This project fits Runge's function, `f(x) = 1 / (1 + 25*x**2)` on `[-1, 1]`, with OLS, Ridge and Lasso. It examines polynomial degree, sample size, noise, bootstrap bias and variance, cross-validation, analytical and automatic differentiation, and deterministic and mini-batch optimization.

Author: Rasul Ruslanovitsj Øzdber, Department of Physics, University of Oslo.

The shortened report covers parts a–i with nine figures, five tables and a compact LLM appendix. It retains the required derivation, comparisons and numerical limitations. Optional proximal-solver and spectral early-stopping discussions are kept out of the report; their earlier code and results remain available here. Text and code have extensive declared LLM contributions.

## Contents

| Directory | Contents |
|---|---|
| `report/` | `report.tex`, `references.bib` and the figures used by the report |
| `code/` | Numerical implementations, experiment scripts and tests in `code/tests/` |
| `figures/` | Generated scientific figures as PDFs |
| `results/` | Saved experiment arrays and numerical verification records as JSON |

| File | Purpose |
|---|---|
| `code/regression.py` | Data generation, polynomial features, scaling, SVD OLS/Ridge, estimator wrapper, MSE and R² |
| `code/optimizers.py` | Objectives, analytical/Autograd gradients, five update rules, full/mini-batch drivers and supplementary proximal solvers |
| `code/resampling.py` | Paired bootstrap decomposition and manual k-fold CV |
| `code/part_a_ols.py` | OLS versus degree, sample count and noise |
| `code/part_b_ridge.py` | Ridge penalty sweep and singular-value shrinkage |
| `code/part_c_bias_variance.py` | Training/test errors and bootstrap bias–variance analysis |
| `code/part_d_cross_validation.py` | Five- and ten-fold sklearn CV, plus the custom implementation check |
| `code/part_e_gradient_descent.py` | Analytical/AD GD, learning rates and conditioning |
| `code/part_f_adaptive.py` | Momentum, AdaGrad, RMSprop and Adam for OLS/Ridge |
| `code/part_g_lasso.py` | Lasso subgradient GD/Adam and additional ISTA/FISTA comparisons |
| `code/part_h_sgd.py` | Batch size, epochs, schedules and SGD update rules |
| `code/part_i_model_selection.py` | Final OLS/Ridge/Lasso selection and regularization bias–variance analysis |
| `code/benchmarks.py` | Reference comparisons summarized in the implementation section |
| `code/run_all.py` | Original a–i experiments and benchmarks; copies their figures into `report/figures/` |

Four supplements complete additional checks and comparisons:

| Script | Saved result | Purpose |
|---|---|---|
| `code/ridge_supplement.py` | `results/ridge_supplement.json` | Ridge MSE/R², coefficient paths and paired sample-size/noise experiments |
| `code/verification_supplement.py` | `results/verification_supplement.json` | Analytical/AD Ridge GD and Ridge SGD comparisons |
| `code/model_selection_check.py` | `results/model_selection_check.json` | Stricter solves for the original minimum-CV and one-SE Lasso selections; the shorter report retains the minimum-CV choices |
| `code/lasso_course_comparison.py` | `results/lasso_course_comparison.json` | All five taught update rules applied to the same degree-10 Lasso problem |

The last supplement supplies the report's five-method Lasso table. Each method starts at zero and runs 40,000 updates at each of eight fixed rates: seven logarithmically spaced rates from `1e-4` to `1e-1`, plus `1/L`. The rate is selected by mean training objective gap over the final 200 updates. The reported coefficients are the last iterate; test MSE is evaluated after selection. Its JSON records every candidate, the reference, selected rates, objective and coefficient errors, KKT residuals, exact zeros and timings. This is a fixed-budget comparison on one dataset, degree and penalty, not a universal optimizer ranking.

## Reproducing the calculations

Run the following from the repository root, preferably in a Python virtual environment:

```sh
python -m pip install -r requirements.txt
cd code
python -m pytest tests -q
python run_all.py
python ridge_supplement.py
python verification_supplement.py
python model_selection_check.py
python lasso_course_comparison.py
```

For a subset of the original experiments, use, for example, `python run_all.py a b`. Run c before d when regenerating the bootstrap/CV comparison. The four supplements are separate commands: `run_all.py` does not execute them. Scripts overwrite their corresponding result and figure files, so preserve an earlier output if comparing runs. Large Lasso searches take longer than the smaller checks.

Fixed seeds specify data, folds and shuffle order. Timings and small floating-point differences depend on the system and libraries. The audited environment is Python 3.13.15, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, Matplotlib 3.11.1, Autograd 1.9.1 and pytest 9.1.1. The sklearn version is pinned because the high-degree OLS comparison explicitly sets its relative rank cutoff to `1e-15`, matching the custom pseudoinverse.

On October 5, 2026, a fresh-process run in an empty output directory passed all 26 tests, reran the complete a–i pipeline and benchmarks, and ran all four supplements. All compared numerical values reproduced the saved records within relative tolerance `1e-7` and absolute tolerance `1e-10`; elapsed-time fields were excluded because timings vary. The report retains the original measured timings. See `results/reproduction_check.json` for commands, per-file comparisons and code hashes, and `results/verification_audit.json` for the wider audit. This rerun preserves the reported Lasso iteration-limit warnings; the stricter check verifies selected scores without reranking the entire grid.

## Building and reading the report

For Overleaf, upload the contents of `report/`, including its `figures/` subdirectory, and set `report.tex` as the main document. The source uses REVTeX 4.2 and BibTeX; select pdfLaTeX. A local build with a suitable TeX installation uses pdfLaTeX, BibTeX and two further pdfLaTeX passes. The LaTeX source and figure assets are the report materials in this repository; compilation and layout should be checked in the submission environment.

The report cites the relevant book sections and weekly exercises; core-module documentation gives additional source links. Source comments beside numerical paragraphs identify the corresponding result JSON fields. Those references distinguish the origin of the method from this project's measured results.

If this repository is private, the assessor needs access to the submitted repository link.

## Connection to the course

The central algorithms follow the course's polynomial design matrices, train-only scaling, SVD, resampling and gradient-update procedures. Core-module documentation identifies the relevant book sections and exercises. Runge data, seeds, grids and stopping budgets are stated experimental choices; the project does not claim to reproduce the numerical answers from different classroom datasets.

Small estimator and optimizer classes organize operations shown as functions and state dictionaries in the weekly notebooks. The mathematics uses mean squared loss: Ridge therefore has `n*lambda` in the linear system, while sklearn Lasso uses `alpha=lambda/2`. All penalties apply to standardized slopes and leave the intercept unpenalized. The report states the reference-based optimizer stopping criterion. The final model table uses minimum CV error; extra one-SE comparisons remain in the saved results.

The report follows the [assignment](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/Projects/2026/Project1/Project1.pdf), [writing guide](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/Projects/ProjectWriting/projectwriting.do.txt), [grading form](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/Projects/EvaluationGrading/EvaluationForm.md), [week 39 writing workshop](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/WeeklyMaterial/week39/week39tuesday.ipynb), and [week 40 project workshop](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/WeeklyMaterial/week40/week40tuesday.pdf), checked on October 5. It uses a 147-word abstract, a short introduction identifying custom/library work, results organized around findings, a compact final CV table, and a three-paragraph conclusion. Week 41 moves on to neural networks for Project 2 and adds no such model requirement to this report.

The extra ISTA/FISTA implementations, detailed GD/Ridge spectral-filter comparison and known-function early-stopping diagnostic are exploratory material in the code/results, outside the shortened report. The main report uses the taught gradient updates and the library coordinate-descent Lasso reference.

## Use of AI/LLM tools

The inherited files attribute the original numerical implementation and LaTeX structure to Claude through Claude Code in October 2026. The precise original model label has not been independently verified. On October 5, 2026, OpenAI Codex inspected the course material, mathematics, code and saved results; corrected the OLS reference cutoff; added a high-degree test and course-linked function documentation; generated and ran the four supplements; and generated or substantially revised the report prose, derivations, captions, interpretations and declaration.

The report appendix records substantial text and code contributions, including the distinction between Codex's executed checks and the author's own understanding. Function-level assistance tags identify generated or assisted code. Reading or retyping generated material does not change its origin. The submitted declaration should remain consistent with the version actually submitted and any further work the author performs.
