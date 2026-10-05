# FYS-STK4155 Project 1: Regression, resampling and gradient descent

Rasul Ruslanovitsj Øzdber, University of Oslo

I fit Runge's function f(x) = 1/(1 + 25x²) to noisy data with OLS, Ridge and Lasso, use the bootstrap and cross-validation to choose the model, and replace the closed-form solutions with gradient descent (plain, momentum, AdaGrad, RMSprop, Adam and SGD).

## What's in the repository

- `report/`: the report (`report.tex`, `references.bib`, `figures/`)
- `code/`: `regression.py`, `resampling.py` and `optimizers.py`, one script per part (`part_a_ols.py` to `part_i_model_selection.py`), a few extra scripts and the tests in `tests/`
- `results/`: the saved numbers (JSON) that the report quotes

## How to run

```sh
pip install -r requirements.txt
python -m pytest code/tests -q
cd code
for s in part_a_ols part_b_ridge ridge_supplement part_c_bias_variance part_d_cross_validation \
         part_e_gradient_descent part_f_adaptive part_g_lasso lasso_course_comparison part_h_sgd \
         part_h_ridge_sgd part_i_model_selection edge_effects benchmarks; do python $s.py; done
```

Keep this order, since some scripts read the results of earlier ones. The seeds are fixed, so the numbers come out the same every time (except the timings). The whole run takes about eight minutes on a laptop. Tested with Python 3.13, NumPy 2.5, scikit-learn 1.9, Matplotlib 3.11 and Autograd 1.9.

## Use of LLMs

Claude (through Claude Code) and OpenAI Codex helped with both the code and the report. `resampling.py` and `part_g_lasso.py` are my own code. In `optimizers.py` and parts a, c and d, I wrote the formulas and the experiments, and Claude wrote the structure, the plots and the extra checks. The remaining files were generated with Claude or Codex. The appendix of the report lists the details, and generated code is tagged `LLM-assisted`.
