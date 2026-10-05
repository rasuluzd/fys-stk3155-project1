# FYS-STK3155/4155 Project 1

Author: Rasul Ruslanovitsj Øzdber, Department of Physics, University of Oslo.
Repository: https://github.com/rasuluzd/fys-stk3155-project1 (private; the assessor needs access).

The report studies noisy Runge data with OLS, Ridge and Lasso, resampling, automatic differentiation and gradient optimization. It covers assignment parts a–i with nine figures and five tables. Text and code contain extensive declared LLM contributions.

## Submission contents

| Location | Purpose |
|---|---|
| `report/report.tex`, `report/references.bib` | Report source and cited references |
| `report/figures/` | The nine figures used in the report |
| `code/` | Core modules, required experiment scripts and tests |
| `results/` | Experiment data, solver diagnostics and verification records |
| `requirements.txt` | Python dependencies |

`regression.py`, `resampling.py` and `optimizers.py` implement the numerical methods. Scripts `part_a_ols.py` through `part_i_model_selection.py` handle the assignment parts. Four further scripts produce evidence quoted in the report: `ridge_supplement.py` supplies Ridge scores, coefficients and paired simulations; `verification_supplement.py` checks Ridge derivatives and SGD; `lasso_course_comparison.py` compares all five taught Lasso updates; `model_selection_check.py` refines the two selected Lasso models. `benchmarks.py` supplies the implementation comparisons. These scripts are included in the main run command.

Unused proximal solvers, spectral early-stopping experiments, extra final one-SE selections, unreported plots and duplicate figure files have been removed. The one-SE OLS check discussed in part d remains.

## Reproduce the results

From the repository root, preferably in a Python virtual environment:

```sh
python -m pip install -r requirements.txt
python -m pytest code/tests -q
python code/run_all.py
```

The run command executes all parts and checks in dependency order and writes figures directly to `report/figures/`. It overwrites corresponding generated outputs. To run a subset, use `python code/run_all.py a b ridge`; run c before d, b before ridge, and i before check. Fixed seeds determine data, folds and shuffling.

The audited environment is Python 3.13.15, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, Matplotlib 3.11.1, Autograd 1.9.1 and pytest 9.1.1. The sklearn pin supports the explicitly matched OLS rank cutoff. See `results/reproduction_check.json` for the fresh-run checks and source hashes, and `results/verification_audit.json` for their scope. Report timings are the original measured timings and vary by system; they are excluded from numerical reproduction comparisons.

The Lasso candidate grid contains iteration-limit hits. Stricter checks validate the selected scores, but do not rerank the complete grid; this limitation is stated in the report.

## Build the report

Upload the contents of `report/` to Overleaf, preserving `figures/`. Set `report.tex` as the main document, select pdfLaTeX, and use BibTeX with REVTeX 4.2. Inspect the compiled PDF before submission. The built-in editor's compiler currently fails with a platform-directory error, so final report layout has not been verified here.

## Course sources and AI use

Methods follow the course's design matrices, train-only scaling, SVD, resampling and update rules. Module documentation identifies book sections and exercises. The grids, seeds and budgets are stated experimental choices. Ridge uses mean loss with `n*lambda` in its linear system; sklearn Lasso uses `alpha=lambda/2`. Source comments in the report identify the result records behind numerical claims.

The report follows the [assignment](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/Projects/2026/Project1/Project1.pdf), [writing guide](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/Projects/ProjectWriting/projectwriting.do.txt), [grading form](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/Projects/EvaluationGrading/EvaluationForm.md), [week 39 writing workshop](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/WeeklyMaterial/week39/week39tuesday.ipynb) and [week 40 project workshop](https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/WeeklyMaterial/week40/week40tuesday.pdf), checked on October 5, 2026.

Original code is attributed to Claude through Claude Code; its precise model label is unverified. OpenAI Codex checked course sources and results, added the four comparison scripts, corrected the OLS reference cutoff, drafted and shortened the report, and removed unused exploration. The appendix declares text at level 3 and code at level 4; functions carry assistance tags. Executed agent checks do not establish the author's personal understanding or verification.
