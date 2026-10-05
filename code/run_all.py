"""
Run every part of the project in order and copy the figures to report/figures.

    python run_all.py            # all parts (about 20-25 minutes on a laptop)
    python run_all.py a b c      # only some parts

All random numbers come from fixed seeds (see settings.py), so a clean run reproduces every
figure in figures/ and every number in results/*.json that is quoted in the report.

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = {
    "a": "part_a_ols.py",
    "b": "part_b_ridge.py",
    "c": "part_c_bias_variance.py",
    "d": "part_d_cross_validation.py",   # uses results of part c
    "e": "part_e_gradient_descent.py",
    "f": "part_f_adaptive.py",
    "g": "part_g_lasso.py",
    "h": "part_h_sgd.py",
    "i": "part_i_model_selection.py",
    "bench": "benchmarks.py",
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(PARTS)
    for key in chosen:
        t0 = time.perf_counter()
        print(f"--- part {key}: {PARTS[key]}", flush=True)
        subprocess.run([sys.executable, PARTS[key]], cwd=HERE, check=True)
        print(f"    done in {time.perf_counter() - t0:.0f} s", flush=True)
    target = HERE.parent / "report" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for pdf in (HERE.parent / "figures").glob("*.pdf"):
        shutil.copy(pdf, target / pdf.name)
    print(f"figures copied to {target}")
