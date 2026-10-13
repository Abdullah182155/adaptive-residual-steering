"""Generate editable and presentation-ready diagrams for the current ARS design."""

import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


BG, PANEL = "#000000", "#121212"
ORANGE, AMBER, GRAY, LIGHT_GRAY, WHITE = "#ff8000", "#ffa64d", "#888888", "#cccccc", "#ffffff"


def ensure_diagrams_dir() -> str:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagrams"))
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, path: str) -> None:
    for attempt in range(6):
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=300, facecolor=BG)
            plt.close(fig)
            return
        except OSError:
            if attempt == 5:
                plt.close(fig)
                raise
            time.sleep(0.35)


def _canvas(width=12, height=7.5):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, title, detail="", edge=GRAY, accent=WHITE):
    ax.add_patch(patches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8, edgecolor=edge, facecolor=PANEL,
    ))
    if detail:
        ax.text(x, y + .11, title, color=accent, fontsize=9.3, fontweight="bold", ha="center", va="center")
        ax.text(x, y - .14, detail, color=LIGHT_GRAY, fontsize=7.7, ha="center", va="center")
    else:
        ax.text(x, y, title, color=accent, fontsize=9.3, fontweight="bold", ha="center", va="center")


def _arrow(ax, x1, y1, x2, y2, color=ORANGE, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=14))
    if label:
        ax.text((x1 + x2) / 2 + .1, (y1 + y2) / 2, label, color=LIGHT_GRAY, fontsize=7.2, va="center")


def _header(ax, x, title, subtitle):
    ax.text(x, 7.1, title, color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(x, 6.7, subtitle, color=AMBER, fontsize=9.5, ha="center")


def generate_steernet_png(path: str) -> None:
    fig, ax = _canvas()
    _header(ax, 6, "RSCSteerNet: K-Conditioned Bounded Low-Rank Residual MLP", r"Active routes use rank $r=16$, $\alpha=32$, decoupled magnitude, and $\|\delta\|\leq0.08\|h\|$")
    _box(ax, 3.25, 5.8, 3.2, .65, "Input Hidden State", r"$h\in\mathbb{R}^{B\times T\times2560}$", GRAY)
    _arrow(ax, 3.25, 5.47, 3.25, 4.9, GRAY)
    _box(ax, 3.25, 4.55, 3.2, .65, "LayerNorm", r"$\mathrm{LN}(h)$", GRAY)
    _arrow(ax, 3.25, 4.22, 3.25, 3.65)
    _box(ax, 3.25, 3.32, 3.5, .68, "Down Projection + GELU", r"$2560\rightarrow16$; dropout $=0.05$", ORANGE, AMBER)
    _box(ax, 8.7, 4.55, 3.0, .65, "Active Route Width", r"For an active route: $K\in\{1,2,3,4\}$", AMBER)
    _arrow(ax, 8.7, 4.22, 8.7, 3.65, AMBER)
    _box(ax, 8.7, 3.32, 3.0, .68, "K Conditioning", r"$W_K(K/4)$ in rank-16 space", AMBER)
    ax.plot([3.25, 3.25, 5.2], [2.98, 2.72, 2.72], color=ORANGE, lw=2)
    ax.plot([8.7, 8.7, 6.8], [2.98, 2.72, 2.72], color=AMBER, lw=2)
    _arrow(ax, 5.2, 2.72, 5.2, 2.38)
    _arrow(ax, 6.8, 2.72, 6.8, 2.38, AMBER)
    _box(ax, 6, 2.08, 4.5, .68, "Rank-16 Residual MLP", "conditioning addition, GELU, rank-level skip", ORANGE, AMBER)
    _arrow(ax, 6, 1.74, 6, 1.2)
    _box(ax, 6, .88, 5.8, .72, "Up Projection + Decoupled Magnitude + Relative Bound", r"$16\rightarrow2560$; output $\delta$ satisfies $\|\delta\|\leq0.08\|h\|$", AMBER)
    ax.text(6, .22, "K=0 bypasses SteerNet completely.", color=LIGHT_GRAY, fontsize=8.3, ha="center", style="italic")
    _save(fig, path)


def generate_gate_png(path: str) -> None:
    fig, ax = _canvas()
    _header(ax, 6, "RSCUsefulnessGate: Context-Aware Token Gate", r"One learned $\alpha_{l,t}\in(0,1)$ per token; health is evaluated through diversity and saturation")
    _box(ax, 1.7, 5.65, 2.5, .65, "Hidden State", r"$h\in\mathbb{R}^{B\times T\times d}$", GRAY)
    _box(ax, 5.0, 5.65, 2.5, .65, "SteerNet Delta", r"$\delta\in\mathbb{R}^{B\times T\times d}$", ORANGE, AMBER)
    _box(ax, 9.6, 5.65, 2.5, .65, "Router Context", r"$c_{ctx}\in\mathbb{R}^{B\times16}$", AMBER)
    _arrow(ax, 1.7, 5.32, 1.7, 4.76, GRAY)
    _box(ax, 1.7, 4.45, 2.2, .58, "Normalize h", r"$\mathrm{LN}(h)$", GRAY)
    ax.plot([5.0, 3.9], [5.32, 5.32], color=ORANGE, lw=2); ax.plot([5.0, 6.1], [5.32, 5.32], color=ORANGE, lw=2)
    _arrow(ax, 3.9, 5.32, 3.9, 4.76); _arrow(ax, 6.1, 5.32, 6.1, 4.76, AMBER)
    _box(ax, 3.9, 4.45, 2.0, .58, "Normalize delta", r"$\mathrm{LN}(\delta)$", ORANGE, AMBER)
    _box(ax, 6.1, 4.45, 2.1, .58, "Delta Features", r"$\log(1+\|\delta\|)$, $\cos(h,\delta)$", AMBER)
    _arrow(ax, 9.6, 5.32, 9.6, 4.76, AMBER)
    _box(ax, 9.6, 4.45, 2.25, .58, "Context Broadcast", r"across sequence tokens", AMBER)
    for x, color in ((1.7, GRAY), (3.9, ORANGE), (6.1, AMBER), (9.6, AMBER)):
        _arrow(ax, x, 4.15, x, 3.60, color)
    _box(ax, 6, 3.28, 9.8, .65, "Feature Concatenation", r"$[\mathrm{LN}(h),\mathrm{LN}(\delta),\log(1+\|\delta\|),\cos(h,\delta),c_{ctx}]$", WHITE)
    _arrow(ax, 6, 2.95, 6, 2.40)
    _box(ax, 6, 2.1, 5.9, .65, "Gate MLP and Causal Token Context", r"Linear$(2d+18\rightarrow16)$ + GELU + dropout, depthwise Conv1D $(k=3)$", ORANGE, AMBER)
    _arrow(ax, 6, 1.77, 6, 1.22, AMBER)
    _box(ax, 6, .92, 4.9, .65, "Sigmoid Token Gate", r"Linear$(16\rightarrow1)$ $\rightarrow\alpha_{l,t}\in(0,1)$", AMBER)
    ax.text(6, .25, "Monitor token std, sequence std, saturation %, and intervened-token count.", color=LIGHT_GRAY, fontsize=8, ha="center", style="italic")
    _save(fig, path)


def generate_router_png(path: str) -> None:
    fig, ax = _canvas()
    _header(ax, 6, "JointLayerRouter: Counterfactual Cooperative Layer Selection", "RLOO learns when and where to intervene; benchmark evaluation uses deterministic argmax routing")
    _box(ax, 6, 5.85, 4.8, .65, "Router Input at Candidate Layer 9", r"hidden states $H\in\mathbb{R}^{B\times T\times2560}$ and unshielded-token mask", GRAY)
    _arrow(ax, 6, 5.52, 6, 4.97)
    _box(ax, 6, 4.65, 5.9, .65, "Masked Pooling and Difficulty Features", "semantic summary, RMS statistics, sequence length, hidden-state dispersion", AMBER)
    _arrow(ax, 6, 4.32, 6, 3.77)
    _box(ax, 6, 3.45, 6.8, .65, "Eight Candidate Representations", r"depth + identity + pooled semantics for $\{9,11,13,15,18,20,22,24\}$", WHITE)
    _arrow(ax, 6, 3.12, 6, 2.57)
    _box(ax, 6, 2.25, 6.1, .65, "Two-Pass Joint Attention Scoring", "utility logits and learned K distribution; no layer dropout", ORANGE, AMBER)
    ax.plot([6, 6], [1.92, 1.62], color=ORANGE, lw=2); ax.plot([2.25, 9.75], [1.62, 1.62], color=ORANGE, lw=2)
    for x, color in ((2.25, AMBER), (6, ORANGE), (9.75, AMBER)):
        _arrow(ax, x, 1.62, x, 1.15, color)
    _box(ax, 2.25, .83, 3.2, .70, "K Distribution", r"$K\in\{0,1,2,3,4\}$; $K=0$ means no intervention", AMBER)
    _box(ax, 6, .83, 3.35, .70, "Deterministic Subset", "argmax K, greedy utility, bounded synergy", ORANGE)
    _box(ax, 9.75, .83, 3.45, .70, "Pairwise Synergy", "bounded context score + pair bias; validate counterfactually", AMBER)
    _save(fig, path)


def generate_ars_full_system_png(path: str) -> None:
    fig, ax = _canvas(13.5, 8)
    ax.text(6.75, 7.5, "Adaptive Residual Steering (ARS): Frozen-Backbone Intervention", color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(6.75, 7.1, "Prompt shielding, K=0 bypass, bounded SteerNet, token gate, and shared multi-layer route budget", color=AMBER, fontsize=9.5, ha="center")
    _box(ax, 2.25, 6.1, 3.2, .65, "Frozen Transformer Layer l", r"hidden state $h_l\in\mathbb{R}^{B\times T\times2560}$", GRAY)
    _arrow(ax, 3.85, 6.1, 4.9, 6.1, GRAY)
    _box(ax, 6.35, 6.1, 2.9, .65, "Forward Hook", r"intercepts residual output $h_l$", WHITE)
    ax.plot([7.8, 12.6], [6.1, 6.1], color=GRAY, lw=1.8); ax.plot([12.6, 12.6], [6.1, 2.35], color=GRAY, lw=1.8)
    _arrow(ax, 12.6, 2.35, 11.5, 2.35, GRAY); ax.text(12.7, 4.2, "baseline residual path", color=LIGHT_GRAY, fontsize=7.5, va="center")
    ax.plot([6.35, 6.35], [5.77, 5.3], color=ORANGE, lw=2); ax.plot([1.6, 10.6], [5.3, 5.3], color=ORANGE, lw=2)
    _box(ax, 1.6, 4.4, 2.35, .70, "Prompt Shield", r"$M_{shield}$ protects prompt and demos", AMBER)
    _arrow(ax, 4.4, 5.3, 4.4, 4.77)
    _box(ax, 4.4, 4.4, 2.65, .70, "Router", r"selects $K\in\{0,\dots,4\}$ and $m_l$", ORANGE, AMBER)
    _arrow(ax, 7.5, 5.3, 7.5, 4.77)
    _box(ax, 7.5, 4.4, 2.65, .70, "SteerNet", r"rank 16, bounded $S_l(h_l)$", ORANGE, AMBER)
    _arrow(ax, 10.6, 5.3, 10.6, 4.77, AMBER)
    _box(ax, 10.6, 4.4, 2.65, .70, "Token Gate", r"per-token $\alpha_{l,t}\in(0,1)$", AMBER)
    for x, color, label in ((1.6, AMBER, "shield"), (4.4, ORANGE, "$m_l,K$"), (7.5, ORANGE, "$S_l(h_l)$"), (10.6, AMBER, r"$\alpha_{l,t}$")):
        _arrow(ax, x, 4.03, x, 2.78, color, label)
    _box(ax, 6.3, 2.35, 10.1, .85, "Conditional Residual Addition", r"$K>0:\quad h'_l=h_l+(1-M_{shield})m_l\alpha_{l,t}S_l(h_l)/\sqrt{K}$", AMBER)
    ax.text(6.3, 1.75, r"$K=0:\quad h'_l=h_l$ (intervention path bypassed)", color=LIGHT_GRAY, fontsize=8.4, ha="center", style="italic")
    _arrow(ax, 6.3, 1.47, 6.3, .95)
    _box(ax, 6.3, .62, 4.8, .65, "Next Frozen Transformer Layer", "continues with the selected residual stream", WHITE)
    _save(fig, path)


def _style(color: str) -> str:
    return f"rounded=1;whiteSpace=wrap;html=1;fillColor=#121212;strokeColor={color};fontColor=#ffffff;strokeWidth=2;"


def _cell(cell_id, text, x, y, w, h, color=ORANGE) -> str:
    return f'<mxCell id="{cell_id}" value="{text}" style="{_style(color)}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'


def _edge(edge_id, source, target, color=ORANGE) -> str:
    return f'<mxCell id="{edge_id}" edge="1" source="{source}" target="{target}" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor={color};strokeWidth=2;endArrow=block;"/>'


def _drawio(path, name, title, nodes, links) -> None:
    cells = "\n        ".join(_cell(*node) for node in nodes)
    edges = "\n        ".join(_edge(*link) for link in links)
    xml = f'''<mxfile host="app.diagrams.net" version="22.1.0"><diagram id="{name}" name="{name}"><mxGraphModel dx="1000" dy="700" grid="1" page="1" pageWidth="1000" pageHeight="700" background="#000000" math="1"><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="title" value="&lt;b&gt;{title}&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1"><mxGeometry x="120" y="20" width="760" height="30" as="geometry"/></mxCell>{cells}{edges}</root></mxGraphModel></diagram></mxfile>'''
    with open(path, "w", encoding="utf-8") as output:
        output.write(xml)


def generate_steernet_drawio(path: str) -> None:
    nodes = [("h", "Input h&lt;br&gt;B×T×2560", 70, 90, 190, 50, GRAY), ("ln", "LayerNorm(h)", 70, 180, 190, 45, GRAY), ("down", "Down + GELU&lt;br&gt;2560 → 16; drop 0.05", 50, 270, 230, 55, ORANGE), ("k", "Active K&lt;br&gt;K ∈ {1,2,3,4}", 520, 180, 210, 45, AMBER), ("kp", "K conditioning W_K(K/4)", 520, 270, 210, 50, AMBER), ("mid", "Rank-16 residual MLP", 285, 375, 280, 55, ORANGE), ("out", "Up + decoupled magnitude&lt;br&gt;‖δ‖ ≤ 0.08‖h‖", 265, 485, 320, 55, AMBER), ("zero", "K=0 bypasses SteerNet", 630, 485, 190, 45, GRAY)]
    links = [("e1", "h", "ln", GRAY), ("e2", "ln", "down"), ("e3", "down", "mid"), ("e4", "k", "kp", AMBER), ("e5", "kp", "mid", AMBER), ("e6", "mid", "out", AMBER)]
    _drawio(path, "steernet", "RSCSteerNet: Bounded Low-Rank Residual MLP", nodes, links)


def generate_gate_drawio(path: str) -> None:
    nodes = [("h", "Hidden state h", 50, 90, 150, 45, GRAY), ("d", "Steering delta δ", 270, 90, 160, 45, ORANGE), ("c", "Router context c_ctx ∈ ℝ^16", 540, 90, 190, 45, AMBER), ("features", "LN(h), LN(δ), log(1+‖δ‖), cos(h,δ), c_ctx", 120, 185, 550, 55, WHITE), ("mlp", "Gate MLP&lt;br&gt;Linear(2d+18 → 16) + GELU", 220, 290, 360, 55, ORANGE), ("conv", "Causal depthwise Conv1D (k=3)", 240, 395, 320, 50, ORANGE), ("out", "Linear(16 → 1) + sigmoid&lt;br&gt;α_(l,t) ∈ (0,1)", 260, 495, 280, 55, AMBER)]
    links = [("e1", "h", "features", GRAY), ("e2", "d", "features"), ("e3", "c", "features", AMBER), ("e4", "features", "mlp"), ("e5", "mlp", "conv"), ("e6", "conv", "out", AMBER)]
    _drawio(path, "gate", "RSCUsefulnessGate: Context-Aware Token Gate", nodes, links)


def generate_router_drawio(path: str) -> None:
    nodes = [("h", "Layer-9 hidden states + unshielded mask", 230, 85, 340, 50, GRAY), ("pool", "Masked pooling + difficulty features", 200, 175, 400, 55, AMBER), ("cand", "Eight candidates: {9,11,13,15,18,20,22,24}", 175, 270, 450, 55, WHITE), ("score", "Two-pass joint scoring&lt;br&gt;utility logits + K head; no layer dropout", 155, 370, 490, 55, ORANGE), ("k", "K ∈ {0,1,2,3,4}&lt;br&gt;K=0: no intervention", 40, 490, 220, 55, AMBER), ("subset", "Deterministic cooperative subset", 300, 490, 250, 55, ORANGE), ("syn", "Bounded synergy&lt;br&gt;validated by counterfactuals", 590, 490, 220, 55, AMBER)]
    links = [("e1", "h", "pool"), ("e2", "pool", "cand", AMBER), ("e3", "cand", "score"), ("e4", "score", "k", AMBER), ("e5", "score", "subset"), ("e6", "score", "syn", AMBER)]
    _drawio(path, "router", "JointLayerRouter: Counterfactual Cooperative Selection", nodes, links)


def generate_ars_full_system_drawio(path: str) -> None:
    nodes = [("layer", "Frozen transformer layer l&lt;br&gt;hidden state h_l", 50, 90, 230, 50, GRAY), ("hook", "Forward hook", 350, 90, 180, 50, WHITE), ("shield", "Prompt shield M_shield", 40, 205, 220, 50, AMBER), ("router", "Router&lt;br&gt;K ∈ {0,…,4}, m_l", 280, 205, 220, 50, ORANGE), ("steer", "SteerNet&lt;br&gt;rank 16, bounded S_l(h_l)", 525, 205, 240, 50, ORANGE), ("gate", "Token gate α_(l,t)", 790, 205, 190, 50, AMBER), ("add", "K&gt;0: h'_l = h_l + (1-M_shield)m_l α_(l,t)S_l(h_l)/√K&lt;br&gt;&lt;b&gt;K=0: h'_l = h_l&lt;/b&gt;", 200, 330, 610, 65, AMBER), ("next", "Next frozen transformer layer", 350, 460, 300, 50, WHITE)]
    links = [("e1", "layer", "hook", GRAY), ("e2", "hook", "router"), ("e3", "hook", "steer"), ("e4", "hook", "gate", AMBER), ("e5", "shield", "add", AMBER), ("e6", "router", "add"), ("e7", "steer", "add"), ("e8", "gate", "add", AMBER), ("e9", "add", "next", AMBER)]
    _drawio(path, "ars-full-system", "Adaptive Residual Steering: Frozen-Backbone Intervention", nodes, links)


def generate_all_diagrams(out_dir: str = None) -> Dict[str, str]:
    out_dir = out_dir or ensure_diagrams_dir()
    os.makedirs(out_dir, exist_ok=True)
    paths = {"steernet_png": os.path.join(out_dir, "diagram_steernet.png"), "steernet_drawio": os.path.join(out_dir, "diagram_steernet.drawio"), "gate_png": os.path.join(out_dir, "diagram_gate.png"), "gate_drawio": os.path.join(out_dir, "diagram_gate.drawio"), "router_png": os.path.join(out_dir, "diagram_router.png"), "router_drawio": os.path.join(out_dir, "diagram_router.drawio"), "full_system_png": os.path.join(out_dir, "diagram_ars_full_system.png"), "full_system_drawio": os.path.join(out_dir, "diagram_ars_full_system.drawio")}
    generate_steernet_png(paths["steernet_png"]); generate_steernet_drawio(paths["steernet_drawio"])
    generate_gate_png(paths["gate_png"]); generate_gate_drawio(paths["gate_drawio"])
    generate_router_png(paths["router_png"]); generate_router_drawio(paths["router_drawio"])
    generate_ars_full_system_png(paths["full_system_png"]); generate_ars_full_system_drawio(paths["full_system_drawio"])
    return paths


if __name__ == "__main__":
    for key, value in generate_all_diagrams().items():
        print(f"{key}: {value}")
