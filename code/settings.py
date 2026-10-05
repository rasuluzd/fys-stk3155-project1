"""
Shared settings for all scripts: the main data set, seeds and a helper that stores numbers.

Every number quoted in the report is written by one of the part_*.py scripts to results/*.json.

LLM-assisted: written with Claude (Anthropic, Claude Code; original model label unverified), October 2026.
"""

import json

import numpy as np
from sklearn.model_selection import train_test_split

from plot_style import RESULTS_DIR
from regression import make_data

SEED = 2026          # seed used for the main data set and all splits
N_POINTS = 100       # number of data points in the main data set
NOISE = 0.1          # standard deviation sigma of the Gaussian noise
TEST_SIZE = 0.2      # 80/20 train/test split
MAX_DEGREE = 15


def main_split(n=N_POINTS, noise=NOISE, seed=SEED):
    """The main data set of the report: x ~ U[-1, 1], y = f(x) + N(0, noise^2), 80/20 split.

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    x, y = make_data(n, noise, seed)
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=seed)


def _to_builtin(obj):
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_results(name, results):
    """Write a dictionary of results to results/<name>.json (numpy types converted).

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(results), f, indent=2)
    return path
