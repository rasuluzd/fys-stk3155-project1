"""
Common figure style, colours and output folders, so that all figures in the report look alike.

Figures are made for a two-column REVTeX article: SINGLE = one column (3.4 in), DOUBLE = full
page width (7.0 in). Colours are a colour-blind-checked categorical palette used in a fixed order,
and a one-hue (blue) sequential map for heatmaps. Lines also get different markers/line styles so
that the figures can be read in black-and-white print.

LLM-assisted
------------
Tool: Claude (Anthropic, Claude Code; original model label unverified), October 2026. Wrote this style module.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "report" / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SINGLE = 3.4
DOUBLE = 7.0

# categorical slots, always used in this order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
GREY = "#898781"
INK = "#0b0b0b"

# the same colour for a method in every figure (OLS blue, Ridge red, Lasso yellow)
METHOD_COLORS = {"OLS": "#2a78d6", "Ridge": "#e34948", "Lasso": "#eda100"}
# optimisers: fixed slots 1-5
OPTIMIZER_COLORS = {"gd": COLORS[0], "momentum": COLORS[1], "adagrad": COLORS[2],
                    "rmsprop": COLORS[3], "adam": COLORS[4]}
OPTIMIZER_LABELS = {"gd": "Plain GD", "momentum": "Momentum", "adagrad": "AdaGrad",
                    "rmsprop": "RMSprop", "adam": "Adam"}
OPTIMIZER_MARKERS = {"gd": "o", "momentum": "s", "adagrad": "^", "rmsprop": "D", "adam": "v"}

# one-hue sequential ramp (light = small values, dark = large values)
BLUES = LinearSegmentedColormap.from_list(
    "blues_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def set_style():
    """LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    mpl.rcParams.update({
        "figure.figsize": (SINGLE, 2.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "mathtext.fontset": "dejavusans",
        "axes.prop_cycle": cycler(color=COLORS),
        "axes.edgecolor": "#52514e",
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
    })


def ordered_colors(k, light=0.25, dark=1.0, base=None):
    """k colours from a one-hue ramp, for ordered series (degrees, lambdas, batch sizes).

    Without base the blue ramp is used; with base (a hex colour) the ramp runs from a light
    tint of that colour to a darker shade of it, so e.g. Ridge curves stay 'red'.

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    import numpy as np
    from matplotlib.colors import to_rgb
    if base is None:
        return [BLUES(v) for v in np.linspace(light, dark, k)]
    c = np.array(to_rgb(base))
    tint, shade = 0.65 * np.ones(3) + 0.35 * c, 0.55 * c
    ramp = LinearSegmentedColormap.from_list("ramp", [tint, c, shade])
    return [ramp(v) for v in np.linspace(light, dark, k)]


def save(fig, name):
    """Save a figure as PDF in the report/figures/ folder and close it.

    If the environment variable PREVIEW_DIR is set, a PNG copy is written there as well
    (handy for a quick look without a PDF viewer).

    LLM-assisted: Claude generated the original implementation, as recorded
    in the module declaration. Codex added this function-level attribution
    on 5 October 2026; this tag does not certify the student's own review.
    """
    import os
    fig.savefig(FIG_DIR / f"{name}.pdf")
    preview = os.environ.get("PREVIEW_DIR")
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview) / f"{name}.png", dpi=170)
    plt.close(fig)
