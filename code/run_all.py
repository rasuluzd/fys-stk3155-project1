"""
Run every calculation needed by the submitted report, including its extra checks.

    python run_all.py            # all parts and checks
    python run_all.py a b c      # only some parts

All random numbers come from fixed seeds (see settings.py), so a clean run reproduces every
figure in report/figures/ and the numerical records quoted in the report.
Wall-clock timings vary between runs and machines.

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
Codex added the required comparison scripts to this command, 5 October 2026.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = {
    "a": "part_a_ols.py",
    "b": "part_b_ridge.py",
    "ridge": "ridge_supplement.py",     # uses the part-b grid
    "c": "part_c_bias_variance.py",
    "d": "part_d_cross_validation.py",   # uses results of part c
    "e": "part_e_gradient_descent.py",
    "f": "part_f_adaptive.py",
    "g": "part_g_lasso.py",
    "lasso": "lasso_course_comparison.py",
    "h": "part_h_sgd.py",
    "verify": "verification_supplement.py",
    "i": "part_i_model_selection.py",
    "check": "model_selection_check.py",  # uses the part-i selections
    "bench": "benchmarks.py",
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(PARTS)
    for key in chosen:
        t0 = time.perf_counter()
        print(f"--- part {key}: {PARTS[key]}", flush=True)
        subprocess.run([sys.executable, PARTS[key]], cwd=HERE, check=True)
        print(f"    done in {time.perf_counter() - t0:.0f} s", flush=True)
    print("Results saved in results/; report figures saved directly in report/figures/.")
