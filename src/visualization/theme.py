import matplotlib as mpl
import matplotlib.pyplot as plt

# Publication Color Palette constants
COLOR_BLACK = "#000000"
COLOR_DARK_GRAY_BG = "#111111"
COLOR_GRID = "#222222"
COLOR_AXIS_EDGE = "#333333"
COLOR_TITLE_WHITE = "#ffffff"
COLOR_LABEL_LIGHT_GRAY = "#cccccc"
COLOR_TICK_GRAY = "#aaaaaa"

# Semantic Model Evaluation Palette
COLOR_GRAY = "#999999"         # Baseline / Neutral
COLOR_MID_GRAY = "#666666"     # SteerNet intermediate
COLOR_LIGHT_GRAY = "#cccccc"   # Demonstrations / Annotations
COLOR_VIVID_ORANGE = "#ff8000" # Full ARS / Highest Performer
COLOR_SOFT_ORANGE = "#ffaa44"  # Secondary ARS / Accent

def set_ars_publication_theme():
    """Applies high-contrast, pure-black publication styling matching ARS design specs."""
    mpl.rcParams.update({
        # Figure and axes background
        "figure.facecolor": COLOR_BLACK,
        "figure.edgecolor": COLOR_BLACK,
        "axes.facecolor": COLOR_BLACK,
        "axes.edgecolor": COLOR_AXIS_EDGE,
        "axes.linewidth": 1.0,

        # Grid
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,

        # Text & Titles
        "text.color": COLOR_TITLE_WHITE,
        "axes.titlecolor": COLOR_TITLE_WHITE,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,

        # Labels & Ticks
        "axes.labelcolor": COLOR_LABEL_LIGHT_GRAY,
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.labelpad": 8,
        "xtick.color": COLOR_TICK_GRAY,
        "ytick.color": COLOR_TICK_GRAY,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],

        # Legend
        "legend.facecolor": COLOR_DARK_GRAY_BG,
        "legend.edgecolor": COLOR_AXIS_EDGE,
        "legend.fontsize": 10,
        "legend.labelcolor": COLOR_TITLE_WHITE,
        "legend.framealpha": 0.9,

        # Layout & DPI
        "savefig.facecolor": COLOR_BLACK,
        "savefig.edgecolor": COLOR_BLACK,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
