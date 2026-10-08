import os
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.visualization.theme import (
    set_ars_publication_theme,
    COLOR_BLACK,
    COLOR_DARK_GRAY_BG,
    COLOR_GRID,
    COLOR_AXIS_EDGE,
    COLOR_TITLE_WHITE,
    COLOR_LABEL_LIGHT_GRAY,
    COLOR_TICK_GRAY,
    COLOR_GRAY,
    COLOR_MID_GRAY,
    COLOR_VIVID_ORANGE,
    COLOR_SOFT_ORANGE,
)

def plot_multishot_scaling(
    benchmark_data: Dict[str, Dict[str, float]],
    dataset_name: str = "GSM8K",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
) -> plt.Figure:
    """Plots publication-grade multi-shot performance comparison (0, 2, 4, 8 shots).
    
    Expected benchmark_data format:
    {
        "Baseline": {"0_shot": 37.0, "2_shot": 37.5, ...},
        "SteerNet Only": {"0_shot": 41.0, ...},
        "Full ARS": {"0_shot": 47.0, "2_shot": 53.33, ...}
    }
    """
    set_ars_publication_theme()
    fig, ax = plt.subplots(figsize=figsize)

    shots = ["0_shot", "2_shot", "4_shot", "8_shot"]
    shot_labels = ["0-Shot", "2-Shot", "4-Shot", "8-Shot"]

    # Filter available shots
    active_shots = [s for s in shots if any(s in v for v in benchmark_data.values())]
    if not active_shots:
        active_shots = shots[:2]
    active_labels = [s.replace("_shot", "-Shot") for s in active_shots]

    x = np.arange(len(active_shots))
    n_series = len(benchmark_data)
    total_width = 0.72
    width = total_width / max(1, n_series)

    # Style definitions
    series_styles = {
        "Baseline": {"color": COLOR_GRAY, "edgecolor": "#bbbbbb", "label": "Frozen Baseline"},
        "SteerNet Only": {"color": COLOR_MID_GRAY, "edgecolor": "#888888", "label": "SteerNet Only"},
        "Full ARS": {"color": COLOR_VIVID_ORANGE, "edgecolor": "#ffa033", "label": "Full ARS (Ours)"},
    }

    offsets = np.linspace(-total_width / 2 + width / 2, total_width / 2 - width / 2, n_series)

    for i, (series_name, values) in enumerate(benchmark_data.items()):
        style = series_styles.get(series_name, {
            "color": COLOR_SOFT_ORANGE if "ars" in series_name.lower() else COLOR_GRAY,
            "edgecolor": "#ffffff",
            "label": series_name,
        })
        y_vals = [values.get(s, 0.0) for s in active_shots]
        bars = ax.bar(
            x + offsets[i],
            y_vals,
            width=width * 0.90,
            color=style["color"],
            edgecolor=style["edgecolor"],
            linewidth=1.0,
            label=style["label"],
            zorder=3,
        )

        # Value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=COLOR_TITLE_WHITE if style["color"] == COLOR_VIVID_ORANGE else COLOR_LABEL_LIGHT_GRAY,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(active_labels, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(f"Multi-Shot Reasoning Accuracy on {dataset_name}", fontweight="bold", pad=16)

    max_val = max([max(v.values()) for v in benchmark_data.values() if v] + [60.0])
    ax.set_ylim(0, min(100.0, max_val + 12.0))

    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, facecolor=COLOR_BLACK)

    return fig


def plot_synergy_heatmap(
    synergy_matrix: np.ndarray,
    layer_labels: Optional[List[str]] = None,
    title: str = "Cross-Layer Cooperative Synergy Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
) -> plt.Figure:
    """Plots high-contrast publication heatmap for cross-layer synergy S_ij in [-1, 1]."""
    set_ars_publication_theme()
    fig, ax = plt.subplots(figsize=figsize)

    if synergy_matrix.ndim == 3:
        synergy_matrix = synergy_matrix[0]
    mat = np.array(synergy_matrix, dtype=float)
    N = mat.shape[0]

    if layer_labels is None:
        layer_labels = [f"L{i}" for i in range(N)]

    # Diverging colormap: Dark Gray/Black (-1) -> Pure Black (0) -> Vivid Orange (+1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ars_synergy",
        ["#222222", "#444444", "#111111", "#aa4400", COLOR_VIVID_ORANGE, "#ffaa44"],
        N=256,
    )

    im = ax.imshow(mat, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    # Grid cell boundaries
    ax.set_xticks(np.arange(N) - 0.5, minor=True)
    ax.set_yticks(np.arange(N) - 0.5, minor=True)
    ax.grid(which="minor", color="#000000", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(layer_labels, fontweight="bold")
    ax.set_yticklabels(layer_labels, fontweight="bold")

    # Add numeric labels in each cell
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            if i == j:
                txt = "0"
                col = "#555555"
            else:
                txt = f"{val:+.2f}"
                col = COLOR_TITLE_WHITE if abs(val) > 0.25 else COLOR_TICK_GRAY
            ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise Synergy Score $S_{ij}$", color=COLOR_LABEL_LIGHT_GRAY, fontweight="bold")
    cbar.ax.tick_params(color=COLOR_TICK_GRAY, labelcolor=COLOR_TICK_GRAY)
    cbar.outline.set_edgecolor(COLOR_AXIS_EDGE)

    ax.set_title(title, fontweight="bold", pad=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, facecolor=COLOR_BLACK)

    return fig


def plot_layer_selection_and_k(
    selection_rates: Dict[Union[int, str], float],
    k_distribution: Dict[Union[int, str], float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4.5),
) -> plt.Figure:
    """Plots dual-panel diagnostic chart: Layer Selection Rate (left) and Realized K (right)."""
    set_ars_publication_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Layer Selection Rates
    layers = sorted([int(k) for k in selection_rates.keys()])
    rates = [selection_rates.get(l, selection_rates.get(str(l), 0.0)) * 100 for l in layers]
    x_layers = [f"L{l}" for l in layers]

    bars1 = ax1.bar(
        x_layers,
        rates,
        color=COLOR_VIVID_ORANGE,
        edgecolor="#ffa033",
        linewidth=1.0,
        zorder=3,
        width=0.6,
    )
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLOR_TITLE_WHITE,
        )
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Selection Frequency (%)", fontweight="bold")
    ax1.set_title("Layer Selection Distribution", fontweight="bold", pad=12)

    # Panel 2: Realized K Distribution
    k_keys = sorted([int(k) for k in k_distribution.keys() if int(k) > 0])
    k_probs = [k_distribution.get(k, k_distribution.get(str(k), 0.0)) * 100 for k in k_keys]
    x_k = [f"K={k}" for k in k_keys]

    colors_k = [COLOR_VIVID_ORANGE if p == max(k_probs) else COLOR_GRAY for p in k_probs]
    bars2 = ax2.bar(
        x_k,
        k_probs,
        color=colors_k,
        edgecolor="#bbbbbb",
        linewidth=1.0,
        zorder=3,
        width=0.55,
    )
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLOR_TITLE_WHITE,
        )
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Probability (%)", fontweight="bold")
    ax2.set_title("Realized Active Layers (K)", fontweight="bold", pad=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, facecolor=COLOR_BLACK)

    return fig


def plot_training_progression(
    training_logs: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4.5),
) -> plt.Figure:
    """Plots multi-phase training validation accuracy and loss trajectory."""
    set_ars_publication_theme()
    fig, ax1 = plt.subplots(figsize=figsize)

    phases = list(training_logs.keys())
    val_losses = [training_logs[p].get("best_val", np.nan) for p in phases]
    val_accs = [training_logs[p].get("best_acc", np.nan) * 100 if "best_acc" in training_logs[p] else np.nan for p in phases]

    x = np.arange(len(phases))
    clean_phase_labels = [p.replace("phase", "Phase ").replace("_", ".").upper() for p in phases]

    ax1.plot(x, val_losses, color=COLOR_GRAY, marker="o", linewidth=2.0, label="Validation Loss", zorder=3)
    ax1.set_ylabel("Validation Loss", color=COLOR_LABEL_LIGHT_GRAY, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_phase_labels, rotation=15, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, val_accs, color=COLOR_VIVID_ORANGE, marker="s", linewidth=2.5, label="Validation Acc (%)", zorder=4)
    ax2.set_ylabel("Accuracy (%)", color=COLOR_VIVID_ORANGE, fontweight="bold")
    ax2.tick_params(colors=COLOR_VIVID_ORANGE)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Multi-Phase ARS Training Convergence", fontweight="bold", pad=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, facecolor=COLOR_BLACK)

    return fig


def generate_experiment_report_charts(
    experiment_data: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> List[str]:
    """Generates all publication figures from an experiment record dictionary and returns saved paths."""
    exp_id = experiment_data.get("experiment_id", "run")
    if output_dir is None:
        run_dir = experiment_data.get("artifacts", {}).get("run_dir", f"./experiments/runs/{exp_id}")
        output_dir = os.path.join(run_dir, "charts")
    os.makedirs(output_dir, exist_ok=True)

    generated_charts = []

    # 1. Multi-Shot Scaling Chart
    benchmarks = experiment_data.get("benchmark_results", {})
    if "gsm8k" in benchmarks:
        gsm_data = {}
        for stage, shots_dict in benchmarks["gsm8k"].items():
            gsm_data[stage] = {k: v.get("accuracy", 0.0) for k, v in shots_dict.items()}
        path_scaling = os.path.join(output_dir, "scaling_multishot_gsm8k.png")
        plot_multishot_scaling(gsm_data, dataset_name="GSM8K", save_path=path_scaling)
        plt.close()
        generated_charts.append(path_scaling)

    # 2. Layer Selection and K Chart
    diag = experiment_data.get("routing_diagnostics", {})
    sel_rates = diag.get("selection_rates", {})
    k_dist = diag.get("k_distribution", {})
    if sel_rates and k_dist:
        path_layers = os.path.join(output_dir, "layer_selection_and_k.png")
        plot_layer_selection_and_k(sel_rates, k_dist, save_path=path_layers)
        plt.close()
        generated_charts.append(path_layers)

    # 3. Training Progression Chart
    logs = experiment_data.get("training_logs", {})
    if logs:
        path_progression = os.path.join(output_dir, "training_progression.png")
        plot_training_progression(logs, save_path=path_progression)
        plt.close()
        generated_charts.append(path_progression)

    return generated_charts
