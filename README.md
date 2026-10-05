# FYS-STK3155/4155 Project 1: regression, resampling and gradient descent for the Runge function

Author: Rasul Ruslanovitsj Øzdber, Department of Physics, University of Oslo.
Repository: https://github.com/rasuluzd/fys-stk3155-project1

The report fits Runge's function from noisy data with OLS, Ridge and Lasso regression, estimates the test error with the bootstrap and k-fold cross-validation, and compares plain, momentum, adaptive and stochastic gradient descent with the closed-form solutions. It answers parts a–i of the assignment with ten figures and three tables.

## Contents

| Location | Purpose |
|---|---|
| `report/report.tex`, `report/references.bib` | Report source and references |
| `report/figures/` | The ten figures used in the report |
| `code/` | Numerical modules, one script per assignment part, supplementary checks and tests |
| `results/` | Saved results (JSON) behind every number quoted in the report |
| `requirements.txt` | Python dependencies |

`regression.py`, `resampling.py` and `optimizers.py` implement the numerical methods; `part_a_ols.py` to `part_i_model_selection.py` run the assignment parts. Supplementary scripts produce further evidence quoted in the report:

- `ridge_supplement.py`: Ridge scores, coefficients and paired simulations (part b);
- `lasso_course_comparison.py`: all five update rules on the part-g Lasso problem;
- `edge_effects.py`: how much of the small-sample errors comes from extrapolation at the interval edge, and the fit figure (parts a, c and i);
- `benchmarks.py`: comparisons with closed-form results and scikit-learn.

## Reproduce the results

From the repository root, preferably in a virtual environment:

```sh
python -m pip install -r requirements.txt
python -m pytest code/tests -q
python code/run_all.py
```

`run_all.py` runs every part and check in dependency order and overwrites the files in `results/` and `report/figures/`. To run a subset, use for example `python code/run_all.py a b ridge`; run b before ridge, c before d, and a, c and i before edge. Fixed seeds determine the data, folds and shuffling. The results were produced with Python 3.13.15, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0 (pinned for the explicit OLS rank cutoff), Matplotlib 3.11.1, Autograd 1.9.1 and pytest 9.1.1. Timings depend on the machine.

## Build the report

Upload the contents of `report/` to Overleaf together with the `figures/` folder, set `report.tex` as the main document and compile with pdfLaTeX and BibTeX (REVTeX 4.2).

## Conventions

Costs are mean losses: Ridge solves `(X^T X + n*lambda*I) theta = X^T y`, and the scikit-learn references use `alpha = n*lambda` for Ridge and `alpha = lambda/2` for Lasso. Scaling is fitted on training data only, also inside every bootstrap sample and CV fold. LaTeX comments in the report name the result file behind each numerical claim.

## Use of LLMs

The numerical code and the original report text were written by the author. Claude (through Claude Code) structured the LaTeX report and generated its tables, the plotting code, the plotting style module, the unit tests and `benchmarks.py`. OpenAI Codex (5 October 2026) added docstrings, comments and source references, removed unreported material, corrected the OLS reference cutoff and generated two supplementary scripts. Claude Opus 5.5 (5 October 2026) rewrote the abstract, introduction and conclusions, drafted paragraphs in the methods and results, added references, generated `edge_effects.py`, shortened the docstrings of the core modules and removed two of Codex's verification scripts. The report's appendix gives the level for every section and file, consistent with the `LLM-assisted` tags in the code.
