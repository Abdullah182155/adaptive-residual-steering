import os
import hashlib
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# Publication Styling Constants - Unified Orange & White Palette
# ==============================================================================
BG_COLOR = "#000000"
BOX_BG = "#121212"
BOX_EDGE = "#444444"
ORANGE = "#ff8000"
LIGHT_ORANGE = "#ffa64d"
AMBER = "#ffaa33"
DARK_ORANGE = "#cc6600"
GRAY = "#888888"
LIGHT_GRAY = "#cccccc"
WHITE = "#ffffff"


def ensure_diagrams_dir() -> str:
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagrams"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def draw_box(ax, x, y, w, h, text, sub="", edge=BOX_EDGE, bg=BOX_BG, text_col=WHITE, title_col=WHITE):
    """Draws a clean rounded rectangular node with consistent typography."""
    rect = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8, edgecolor=edge, facecolor=bg
    )
    ax.add_patch(rect)
    if sub:
        ax.text(x, y + 0.11, text, color=title_col, fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(x, y - 0.14, sub, color=text_col, fontsize=8.0, ha="center", va="center")
    else:
        ax.text(x, y, text, color=title_col, fontsize=9.5, fontweight="bold", ha="center", va="center")


def draw_straight_arrow(ax, x1, y1, x2, y2, col=ORANGE, label=""):
    """Draws a strictly horizontal or vertical straight arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14))
    if label:
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, label, color=LIGHT_GRAY, fontsize=8, va="center")


def safe_savefig(fig_module, out_path: str, **kwargs):
    """Saves figure with exponential backoff retry to prevent transient Windows file-lock errors."""
    import time
    for attempt in range(6):
        try:
            fig_module.savefig(out_path, **kwargs)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(0.35)


# ==============================================================================
# 1. SteerNet Diagram (PNG + Draw.io)
# ==============================================================================

def generate_steernet_png(out_path: str):
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Header
    ax.text(6.0, 7.1, "RSCSteerNet: K-Conditioned Low-Rank Residual MLP",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(6.0, 6.7, r"Bounded Relative Perturbation: $\|\delta\| \leq \beta \|h\|$, $\delta_t = \frac{\alpha}{r} W_{\mathrm{up}}(\mathrm{MLP}(h) + W_k(K))$",
            color=LIGHT_ORANGE, fontsize=9.5, ha="center")

    # Left Column (Hidden State Feature Stream)
    draw_box(ax, 4.2, 5.9, 3.2, 0.60, "Input Hidden State", r"$h \in \mathbb{R}^{B \times T \times d}$", edge=GRAY)
    draw_straight_arrow(ax, 4.2, 5.60, 4.2, 5.10)

    draw_box(ax, 4.2, 4.8, 3.2, 0.60, "Layer Normalization", r"$\mathrm{LayerNorm}(h)$", edge=GRAY)
    draw_straight_arrow(ax, 4.2, 4.50, 4.2, 4.00)

    draw_box(ax, 4.2, 3.7, 3.2, 0.60, "Down Projection + GELU", r"$W_{\mathrm{down}} \in \mathbb{R}^{d \times r}$ ($r \ll d$)", edge=ORANGE, title_col=LIGHT_ORANGE)

    # Right Column (K-Conditioning Stream)
    draw_box(ax, 8.8, 4.8, 2.8, 0.60, "Active K Input", r"$K \in \{1,\dots,4\} \rightarrow K/4.0$", edge=LIGHT_ORANGE, title_col=LIGHT_ORANGE)
    draw_straight_arrow(ax, 8.8, 4.50, 8.8, 4.00)

    draw_box(ax, 8.8, 3.7, 2.8, 0.60, "K-Conditioning Proj", r"$W_k \in \mathbb{R}^{1 \times r}$", edge=LIGHT_ORANGE, title_col=LIGHT_ORANGE)

    # Fusion at Center (Row 4)
    # Down-Proj into Addition
    ax.plot([4.2, 4.2], [3.40, 3.05], color=ORANGE, lw=2.0)
    ax.plot([4.2, 5.4], [3.05, 3.05], color=ORANGE, lw=2.0)
    draw_straight_arrow(ax, 5.4, 3.05, 5.4, 2.90, col=ORANGE)

    # K-Proj into Addition
    ax.plot([8.8, 8.8], [3.40, 3.05], color=LIGHT_ORANGE, lw=2.0)
    ax.plot([8.8, 7.6], [3.05, 3.05], color=LIGHT_ORANGE, lw=2.0)
    ax.text(8.2, 3.20, "K context", color=LIGHT_GRAY, fontsize=8, ha="center")
    draw_straight_arrow(ax, 7.6, 3.05, 7.6, 2.90, col=LIGHT_ORANGE)

    draw_box(ax, 6.5, 2.6, 4.0, 0.60, "Conditioning Addition", r"$x_{\mathrm{fused}} = x + W_k(K/4)$", edge=ORANGE, title_col=WHITE)

    # Bottleneck MLP (Row 5)
    draw_straight_arrow(ax, 6.5, 2.30, 6.5, 1.80)
    draw_box(ax, 6.5, 1.5, 4.4, 0.60, "Bottleneck MLP + GELU & Skip", r"$W_{\mathrm{mid}} \in \mathbb{R}^{r \times r} + \mathrm{residual}$", edge=ORANGE, title_col=LIGHT_ORANGE)

    # Up Projection & Bounded Output (Row 6)
    draw_straight_arrow(ax, 6.5, 1.20, 6.5, 0.80)
    draw_box(ax, 6.5, 0.5, 5.2, 0.60, "Up Proj & Bounded Relative Norm", r"$\delta = \min(\|\delta\|, \beta\|h\|) \frac{\delta}{\|\delta\| + \epsilon} \in \mathbb{R}^{B \times T \times d}$", edge=LIGHT_ORANGE, title_col=WHITE)

    plt.tight_layout()
    safe_savefig(plt, out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_steernet_drawio(out_path: str):
    xml_content = """<mxfile host="app.diagrams.net" version="22.1.0">
  <diagram id="steernet-arch" name="SteerNet Architecture">
    <mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="650" background="#000000" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="title" value="&lt;b&gt;RSCSteerNet: K-Conditioned Low-Rank Residual MLP&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="150" y="20" width="550" height="30" as="geometry" />
        </mxCell>
        <mxCell id="in_h" value="Input Hidden State &lt;br&gt;&lt;i&gt;h ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="180" y="80" width="220" height="50" as="geometry" />
        </mxCell>
        <mxCell id="norm" value="LayerNorm(h)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="180" y="160" width="220" height="45" as="geometry" />
        </mxCell>
        <mxCell id="down" value="Down Projection + GELU &lt;br&gt;&lt;b&gt;W_down ∈ ℝ^(d×r)&lt;/b&gt; (r ≪ d)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="170" y="240" width="240" height="50" as="geometry" />
        </mxCell>
        <mxCell id="k_in" value="Active K Input &lt;br&gt;K ∈ {1..4} → K / 4.0" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="500" y="160" width="200" height="45" as="geometry" />
        </mxCell>
        <mxCell id="k_proj" value="K Projection Proj&lt;br&gt;&lt;b&gt;W_k ∈ ℝ^(1×r)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="500" y="240" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="add_k" value="Conditioning Addition &lt;br&gt;x = x + W_k(K/4)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ff8000;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="270" y="325" width="280" height="45" as="geometry" />
        </mxCell>
        <mxCell id="mid" value="Bottleneck MLP + Skip &lt;br&gt;&lt;b&gt;W_mid ∈ ℝ^(r×r)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="290" y="405" width="240" height="50" as="geometry" />
        </mxCell>
        <mxCell id="up" value="Up Projection &amp; Bounded Relative Norm &lt;br&gt;&lt;b&gt;δ = min(‖δ‖, β‖h‖) · (δ / ‖δ‖)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="260" y="490" width="300" height="55" as="geometry" />
        </mxCell>
        <mxCell id="e1" edge="1" source="in_h" target="norm" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#888888;strokeWidth=2;" />
        <mxCell id="e2" edge="1" source="norm" target="down" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e3" edge="1" source="down" target="add_k" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e4" edge="1" source="k_in" target="k_proj" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="e5" edge="1" source="k_proj" target="add_k" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="e6" edge="1" source="add_k" target="mid" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e7" edge="1" source="mid" target="up" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())


# ==============================================================================
# 2. Token Gate Diagram (PNG + Draw.io)
# ==============================================================================

def generate_gate_png(out_path: str):
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Header
    ax.text(6.0, 7.1, "RSCUsefulnessGate: Semantic Context-Aware Token Gate 2.0",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(6.0, 6.7, r"Causal Conv1d ($k=3$) + Multi-Feature MLP: $\alpha_t = \sigma(\mathrm{MLP}(\mathrm{Conv1d}([h, \delta, \mathrm{mag}, \cos, c_{\mathrm{ctx}}]))) \in (0, 1)$",
            color=LIGHT_ORANGE, fontsize=9.5, ha="center")

    # Row 1: 3 Input Blocks
    draw_box(ax, 2.3, 5.8, 2.6, 0.60, "Hidden State", r"$h \in \mathbb{R}^{B \times T \times d}$", edge=GRAY)
    draw_box(ax, 6.0, 5.8, 3.2, 0.60, "Steer Correction", r"$\delta \in \mathbb{R}^{B \times T \times d}$", edge=ORANGE, title_col=LIGHT_ORANGE)
    draw_box(ax, 9.7, 5.8, 2.6, 0.60, "Semantic Context", r"$c_{\mathrm{ctx}} \in \mathbb{R}^{B \times 16}$", edge=LIGHT_ORANGE, title_col=WHITE)

    # Row 2: Feature Extraction (strictly vertical alignment under parents)
    # Under h:
    draw_straight_arrow(ax, 2.3, 5.50, 2.3, 5.00)
    draw_box(ax, 2.3, 4.7, 2.0, 0.60, "LN(h)", r"LayerNorm", edge=GRAY)

    # Under delta: orthogonal split into LN(delta) and Magnitude
    ax.plot([6.0, 4.8], [5.50, 5.50], color=ORANGE, lw=2.0)
    draw_straight_arrow(ax, 4.8, 5.50, 4.8, 5.00, col=ORANGE)
    draw_box(ax, 4.8, 4.7, 2.0, 0.60, "LN(delta)", r"LayerNorm", edge=ORANGE, title_col=LIGHT_ORANGE)

    ax.plot([6.0, 7.2], [5.50, 5.50], color=ORANGE, lw=2.0)
    draw_straight_arrow(ax, 7.2, 5.50, 7.2, 5.00, col=ORANGE)
    draw_box(ax, 7.2, 4.7, 2.2, 0.60, "Magnitude", r"$\log(1 + \|\delta\|_2)$", edge=LIGHT_ORANGE, title_col=LIGHT_ORANGE)

    # Under context:
    draw_straight_arrow(ax, 9.7, 5.50, 9.7, 5.00, col=LIGHT_ORANGE)
    draw_box(ax, 9.7, 4.7, 2.2, 0.60, "Context Proj", r"$\mathrm{Linear}(16 \rightarrow 16)$", edge=LIGHT_ORANGE, title_col=WHITE)

    # Row 3: Concat Feature Fusion Bar
    draw_straight_arrow(ax, 2.3, 4.40, 2.3, 3.80, col=GRAY)
    draw_straight_arrow(ax, 4.8, 4.40, 4.8, 3.80, col=ORANGE)
    draw_straight_arrow(ax, 7.2, 4.40, 7.2, 3.80, col=LIGHT_ORANGE)
    draw_straight_arrow(ax, 9.7, 4.40, 9.7, 3.80, col=LIGHT_ORANGE)

    draw_box(ax, 6.0, 3.5, 9.6, 0.60, "Feature Fusion Concatenation",
             r"$[ \mathrm{LN}(h), \mathrm{LN}(\delta), \log(1+\|\delta\|), \cos(h, \delta), c_{\mathrm{ctx}} ] \in \mathbb{R}^{2d + 2 + 16}$",
             edge=WHITE, title_col=WHITE)

    # Row 4: Causal Conv1d + Dense Projection
    draw_straight_arrow(ax, 6.0, 3.20, 6.0, 2.65, col=ORANGE)
    draw_box(ax, 6.0, 2.35, 5.4, 0.60, "Causal Depthwise Conv1d (k=3) + GELU",
             r"Preceding token context ($W_{\mathrm{conv}} \cdot \mathrm{Dirac}$) $\rightarrow \mathrm{Linear}(2d+18 \rightarrow 64)$",
             edge=ORANGE, title_col=LIGHT_ORANGE)

    # Row 5: Projection Head & Output
    draw_straight_arrow(ax, 6.0, 2.05, 6.0, 1.45, col=ORANGE)
    draw_box(ax, 6.0, 1.15, 4.8, 0.60, "Dynamic Sigmoid Token Gate Output",
             r"$\mathrm{Linear}(64 \rightarrow 1) + \mathrm{InitBias} \rightarrow \alpha_t \in (0, 1)^{B \times T \times 1}$",
             edge=LIGHT_ORANGE, title_col=WHITE)

    plt.tight_layout()
    safe_savefig(plt, out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_gate_drawio(out_path: str):
    xml_content = """<mxfile host="app.diagrams.net" version="22.1.0">
  <diagram id="gate-arch" name="Token Gate Architecture">
    <mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="650" background="#000000" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="title" value="&lt;b&gt;RSCUsefulnessGate: Semantic Context-Aware Token Gate 2.0&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="150" y="20" width="550" height="30" as="geometry" />
        </mxCell>
        <mxCell id="in_h" value="Hidden State &lt;br&gt;&lt;i&gt;h ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="100" y="80" width="160" height="45" as="geometry" />
        </mxCell>
        <mxCell id="in_delta" value="Steer Correction &lt;br&gt;&lt;i&gt;δ ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="340" y="80" width="180" height="45" as="geometry" />
        </mxCell>
        <mxCell id="in_ctx" value="Semantic Context &lt;br&gt;&lt;i&gt;c_ctx ∈ ℝ^(B×16)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="600" y="80" width="160" height="45" as="geometry" />
        </mxCell>
        <mxCell id="ln_h" value="LN(h)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="120" y="160" width="120" height="40" as="geometry" />
        </mxCell>
        <mxCell id="ln_d" value="LN(δ)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="310" y="160" width="110" height="40" as="geometry" />
        </mxCell>
        <mxCell id="mag" value="log(1 + ‖δ‖₂)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="440" y="160" width="110" height="40" as="geometry" />
        </mxCell>
        <mxCell id="ctx_proj" value="Context Proj" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="620" y="160" width="120" height="40" as="geometry" />
        </mxCell>
        <mxCell id="concat" value="Feature Concatenation &lt;br&gt;&lt;b&gt;[ LN(h), LN(δ), mag, cos, c_ctx ] ∈ ℝ^(2d + 18)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="140" y="240" width="580" height="45" as="geometry" />
        </mxCell>
        <mxCell id="conv" value="Causal Depthwise Conv1d (k=3) &amp; MLP &lt;br&gt;&lt;b&gt;Temporal Context + Linear(2d+18 → 64) + GELU&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="250" y="325" width="360" height="50" as="geometry" />
        </mxCell>
        <mxCell id="out" value="Dynamic Sigmoid Token Gate &lt;br&gt;&lt;b&gt;α_t = σ(·) ∈ (0, 1)^(B×T×1)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="280" y="415" width="300" height="50" as="geometry" />
        </mxCell>
        <mxCell id="ge1" edge="1" source="in_h" target="ln_h" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#888888;strokeWidth=2;" />
        <mxCell id="ge2" edge="1" source="in_delta" target="ln_d" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge3" edge="1" source="in_delta" target="mag" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="ge4" edge="1" source="in_ctx" target="ctx_proj" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="ge5" edge="1" source="ln_h" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#888888;strokeWidth=1.5;" />
        <mxCell id="ge6" edge="1" source="ln_d" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=1.5;" />
        <mxCell id="ge7" edge="1" source="mag" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=1.5;" />
        <mxCell id="ge8" edge="1" source="ctx_proj" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=1.5;" />
        <mxCell id="ge9" edge="1" source="concat" target="conv" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge10" edge="1" source="conv" target="out" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())


# ==============================================================================
# 3. Router Diagram (PNG + Draw.io)
# ==============================================================================

def generate_router_png(out_path: str):
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Header
    ax.text(6.0, 7.1, "JointLayerRouter: Difficulty-Aware 2-Pass Router 2.0",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(6.0, 6.7, r"Difficulty-aware joint scoring, Stochastic Layer Dropout ($p=0.15$), learned $K$, and Synergy $S_{ij}$",
            color=LIGHT_ORANGE, fontsize=9.5, ha="center")

    # Central Vertical Pipeline
    draw_box(ax, 6.0, 5.9, 4.4, 0.60, "Early Hidden State Tokens", r"$H \in \mathbb{R}^{B \times T \times d}$ (Layer 9)", edge=GRAY)
    draw_straight_arrow(ax, 6.0, 5.60, 6.0, 5.10)

    draw_box(ax, 6.0, 4.8, 5.2, 0.60, "Semantic Summary & Difficulty Extraction",
             r"$\mathrm{LN} \rightarrow \mathrm{Linear}(d \rightarrow 16) \rightarrow \tanh$ + Difficulty $[\log(1+T), \mathrm{dispersion}]$",
             edge=LIGHT_ORANGE, title_col=WHITE)
    draw_straight_arrow(ax, 6.0, 4.50, 6.0, 4.00)

    draw_box(ax, 6.0, 3.7, 5.2, 0.60, "Candidate Embeddings",
             r"Depth + Difficulty ($T, \mathrm{disp}$) + Identity ($d_{\mathrm{model}}=32$) + Semantics",
             edge=WHITE, title_col=WHITE)
    draw_straight_arrow(ax, 6.0, 3.40, 6.0, 2.90)

    draw_box(ax, 6.0, 2.6, 5.2, 0.60, "Pass 1 & 2: Joint Attention + Layer Dropout",
             r"Self-Attention $\rightarrow u_1 \rightarrow$ Refine Head $u_2$ (Training Dropout $p=0.15$)",
             edge=ORANGE, title_col=LIGHT_ORANGE)

    # Clean 3-way fork with perfectly straight vertical arrows entering top of each box
    ax.plot([6.0, 6.0], [2.30, 1.70], color=ORANGE, lw=2.0)
    ax.plot([2.3, 9.7], [1.70, 1.70], color=ORANGE, lw=2.0)

    # Output 1 (Left): K-Distribution
    draw_straight_arrow(ax, 2.3, 1.70, 2.3, 1.25, col=LIGHT_ORANGE)
    draw_box(ax, 2.3, 0.9, 2.8, 0.65, "K-Distribution Head", r"$\mathrm{Softmax} \rightarrow K \in \{1..4\}$", edge=LIGHT_ORANGE, title_col=WHITE)

    # Output 2 (Center): Active Layer Selection
    draw_straight_arrow(ax, 6.0, 1.70, 6.0, 1.25, col=ORANGE)
    draw_box(ax, 6.0, 0.9, 3.2, 0.65, "Top-K Selection", r"$\mathrm{argtopk}(u_2, K) \rightarrow \{11, 15, 20\}$", edge=ORANGE, title_col=WHITE)

    # Output 3 (Right): Synergy Matrix
    draw_straight_arrow(ax, 9.7, 1.70, 9.7, 1.25, col=LIGHT_ORANGE)
    draw_box(ax, 9.7, 0.9, 3.0, 0.65, "Pairwise Synergy Matrix", r"$S_{ij} = \tanh\left(\frac{q_i^T k_j}{\sqrt{d}}\right) \in [-1, 1]$", edge=LIGHT_ORANGE, title_col=LIGHT_ORANGE)

    plt.tight_layout()
    safe_savefig(plt, out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_router_drawio(out_path: str):
    xml_content = """<mxfile host="app.diagrams.net" version="22.1.0">
  <diagram id="router-arch" name="Router Architecture">
    <mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="650" background="#000000" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="title" value="&lt;b&gt;JointLayerRouter: Difficulty-Aware 2-Pass Router 2.0&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="150" y="20" width="550" height="30" as="geometry" />
        </mxCell>
        <mxCell id="in_h" value="Early Hidden State Tokens &lt;br&gt;&lt;i&gt;H ∈ ℝ^(B×T×d)&lt;/i&gt; (Layer 9)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="270" y="70" width="280" height="45" as="geometry" />
        </mxCell>
        <mxCell id="sem_proj" value="Semantic Summary &amp; Difficulty Extraction &lt;br&gt;&lt;b&gt;LN → Linear(d → 16) → tanh + [log(1+T), disp]&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="230" y="145" width="360" height="45" as="geometry" />
        </mxCell>
        <mxCell id="cand_emb" value="Candidate Embeddings &lt;br&gt;&lt;i&gt;Depth + Difficulty + Identity (d_model=32) + Semantics&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="220" y="220" width="380" height="45" as="geometry" />
        </mxCell>
        <mxCell id="pass1_2" value="Pass 1 &amp; 2: Joint Attention &amp; Stochastic Layer Dropout &lt;br&gt;&lt;b&gt;Self-Attention → u₁ → Refine Head u₂ (Dropout p=0.15)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="200" y="295" width="420" height="50" as="geometry" />
        </mxCell>
        <mxCell id="k_head" value="K-Head (Softmax)&lt;br&gt;&lt;b&gt;P(K) ∈ ℝ^4 → K=3&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="80" y="380" width="180" height="50" as="geometry" />
        </mxCell>
        <mxCell id="topk" value="Top-K Layer Selection&lt;br&gt;&lt;b&gt;argtopk(u₂, K) → {11, 15, 20}&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="310" y="380" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="synergy" value="Pairwise Synergy Matrix&lt;br&gt;&lt;b&gt;S_ij = tanh(q_i^T k_j / √d)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="560" y="380" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="re1" edge="1" source="in_h" target="sem_proj" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re2" edge="1" source="sem_proj" target="cand_emb" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="re3" edge="1" source="cand_emb" target="pass1_2" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re4" edge="1" source="pass1_2" target="k_head" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="re5" edge="1" source="pass1_2" target="topk" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re6" edge="1" source="pass1_2" target="synergy" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())


# ==============================================================================
# 4. ARS Full System Integration Diagram (PNG + Draw.io)
# ==============================================================================

def generate_ars_full_system_png(out_path: str):
    fig, ax = plt.subplots(figsize=(13.5, 8.0), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # Header
    ax.text(6.75, 7.5, "Adaptive Residual Steering (ARS): Full Architecture Integration",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(6.75, 7.1, r"Forward hook modulation with Prompt Shielding, Difficulty Routing, and Bounded SteerNet",
            color=LIGHT_ORANGE, fontsize=9.5, ha="center")

    # Top Row: Backbone Representation Stream
    draw_box(ax, 2.5, 6.2, 3.2, 0.60, "Frozen Transformer Layer l", r"Hidden state output: $h_l \in \mathbb{R}^{B \times T \times d}$", edge=GRAY)
    draw_straight_arrow(ax, 4.1, 6.2, 5.0, 6.2, col=GRAY)

    draw_box(ax, 6.5, 6.2, 3.0, 0.60, "Forward Hook Tap", "Intercept hidden state $h_l$", edge=WHITE, title_col=WHITE)

    # Skip line connecting h_l directly to Adder via clear outer right path
    ax.plot([8.0, 12.6], [6.2, 6.2], color=GRAY, lw=1.8)
    ax.plot([12.6, 12.6], [6.2, 2.5], color=GRAY, lw=1.8)
    ax.text(12.7, 4.5, "Skip h_l", color=LIGHT_GRAY, fontsize=8, va="center")
    draw_straight_arrow(ax, 12.6, 2.5, 11.5, 2.5, col=GRAY)

    # Middle Row: 4 Parallel Functional Modules (Y = 4.4)
    # Bus bar from Hook Tap at Y=5.3
    ax.plot([6.5, 6.5], [5.90, 5.30], color=ORANGE, lw=2.0)
    ax.plot([4.4, 9.8], [5.30, 5.30], color=ORANGE, lw=2.0)

    # Module 1: Prompt Shielding Mask (Leftmost)
    draw_box(ax, 1.7, 4.4, 2.2, 0.65, "Prompt Shielding Mask", r"$M_{\mathrm{shield}} \in \{0, 1\}^{B \times T}$", edge=LIGHT_ORANGE, title_col=WHITE)

    # Module 2: Joint Router
    draw_straight_arrow(ax, 4.4, 5.30, 4.4, 4.75, col=ORANGE)
    draw_box(ax, 4.4, 4.4, 2.4, 0.65, "Joint Layer Router", r"Selected $r_l \in \{0, 1\}$, Budget $K$", edge=ORANGE, title_col=LIGHT_ORANGE)

    # Module 3: Bounded SteerNet
    draw_straight_arrow(ax, 7.1, 5.30, 7.1, 4.75, col=ORANGE)
    draw_box(ax, 7.1, 4.4, 2.4, 0.65, "RSCSteerNet 2.0", r"$\delta_l \in \mathbb{R}^{B \times T \times d}, \|\delta\| \leq \beta \|h\|$", edge=ORANGE, title_col=LIGHT_ORANGE)

    # Module 4: Token Gate
    draw_straight_arrow(ax, 9.8, 5.30, 9.8, 4.75, col=LIGHT_ORANGE)
    draw_box(ax, 9.8, 4.4, 2.4, 0.65, "RSCUsefulnessGate 2.0", r"$\alpha_l \in (0, 1)^{B \times T \times 1}$", edge=LIGHT_ORANGE, title_col=WHITE)

    # Bottom Row: Adaptive Residual Injection
    # 4 Parallel vertical downward arrows entering top of the formula block
    draw_straight_arrow(ax, 1.7, 4.05, 1.7, 2.90, col=LIGHT_ORANGE, label="Shield")
    draw_straight_arrow(ax, 4.4, 4.05, 4.4, 2.90, col=ORANGE, label="r_l")
    draw_straight_arrow(ax, 7.1, 4.05, 7.1, 2.90, col=ORANGE, label="delta")
    draw_straight_arrow(ax, 9.8, 4.05, 9.8, 2.90, col=LIGHT_ORANGE, label="alpha")

    draw_box(ax, 6.2, 2.5, 9.6, 0.75, "Adaptive Residual Addition",
             r"$h_l^{\mathrm{new}} = h_l + \frac{r_l \cdot \alpha_l \cdot \delta_l \cdot (1 - M_{\mathrm{shield}})}{\sqrt{K}}$",
             edge=LIGHT_ORANGE, title_col=WHITE)

    # Next Transformer Layer Input
    draw_straight_arrow(ax, 6.2, 2.12, 6.2, 1.45, col=ORANGE)
    draw_box(ax, 6.2, 1.05, 5.2, 0.65, "Transformer Layer l+1 Input", "Conditioned, stable steering representation", edge=WHITE, title_col=WHITE)

    plt.tight_layout()
    safe_savefig(plt, out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_ars_full_system_drawio(out_path: str):
    xml_content = """<mxfile host="app.diagrams.net" version="22.1.0">
  <diagram id="ars-full-arch" name="ARS Full Architecture">
    <mxGraphModel dx="1100" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="950" pageHeight="700" background="#000000" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="title" value="&lt;b&gt;Adaptive Residual Steering (ARS): Full Architecture Integration&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="180" y="20" width="580" height="30" as="geometry" />
        </mxCell>
        <mxCell id="shield" value="Prompt Shielding Mask &lt;br&gt;&lt;b&gt;M_shield ∈ {0, 1}^(B×T)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="40" y="160" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="bb_layer" value="Frozen Transformer Layer l &lt;br&gt;&lt;i&gt;Output hidden state h_l ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="150" y="70" width="280" height="50" as="geometry" />
        </mxCell>
        <mxCell id="hook" value="Forward Hook Tap &lt;br&gt;Intercept h_l" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="500" y="70" width="180" height="50" as="geometry" />
        </mxCell>
        <mxCell id="router" value="Joint Layer Router &lt;br&gt;&lt;b&gt;Selects r_l ∈ {0, 1}, Budget K&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="270" y="160" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="steernet" value="RSCSteerNet 2.0 &lt;br&gt;&lt;b&gt;δ_l = MLP(h_l, K), ‖δ‖ ≤ β‖h‖&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="495" y="160" width="210" height="50" as="geometry" />
        </mxCell>
        <mxCell id="gate" value="RSCUsefulnessGate 2.0 &lt;br&gt;&lt;b&gt;α_l = σ(h_l, δ_l, c)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="730" y="160" width="180" height="50" as="geometry" />
        </mxCell>
        <mxCell id="formula" value="Adaptive Residual Addition &lt;br&gt;&lt;b&gt;h_l^new = h_l + [ r_l · α_l · δ_l · (1 - M_shield) ] / √K&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="180" y="270" width="580" height="55" as="geometry" />
        </mxCell>
        <mxCell id="next_layer" value="Transformer Layer l+1 Input &lt;br&gt;Modulated Steering Representation" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="320" y="380" width="300" height="50" as="geometry" />
        </mxCell>
        <mxCell id="fe1" edge="1" source="bb_layer" target="hook" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#888888;strokeWidth=2;" />
        <mxCell id="fe2" edge="1" source="hook" target="router" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe3" edge="1" source="hook" target="steernet" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe4" edge="1" source="hook" target="gate" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="fe5" edge="1" source="hook" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#888888;strokeWidth=1.5;" />
        <mxCell id="fe6" edge="1" source="router" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe7" edge="1" source="steernet" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe8" edge="1" source="gate" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="fe9" edge="1" source="shield" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="fe10" edge="1" source="formula" target="next_layer" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())


# ==============================================================================
# 5. Master Generator Function
# ==============================================================================

def generate_all_diagrams(out_dir: str = None) -> Dict[str, str]:
    if out_dir is None:
        out_dir = ensure_diagrams_dir()
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "steernet_png": os.path.join(out_dir, "diagram_steernet.png"),
        "steernet_drawio": os.path.join(out_dir, "diagram_steernet.drawio"),
        "gate_png": os.path.join(out_dir, "diagram_gate.png"),
        "gate_drawio": os.path.join(out_dir, "diagram_gate.drawio"),
        "router_png": os.path.join(out_dir, "diagram_router.png"),
        "router_drawio": os.path.join(out_dir, "diagram_router.drawio"),
        "full_system_png": os.path.join(out_dir, "diagram_ars_full_system.png"),
        "full_system_drawio": os.path.join(out_dir, "diagram_ars_full_system.drawio"),
    }

    generate_steernet_png(paths["steernet_png"])
    generate_steernet_drawio(paths["steernet_drawio"])

    generate_gate_png(paths["gate_png"])
    generate_gate_drawio(paths["gate_drawio"])

    generate_router_png(paths["router_png"])
    generate_router_drawio(paths["router_drawio"])

    generate_ars_full_system_png(paths["full_system_png"])
    generate_ars_full_system_drawio(paths["full_system_drawio"])

    return paths


if __name__ == "__main__":
    generated = generate_all_diagrams()
    print("Generated all architecture diagrams successfully:")
    for k, v in generated.items():
        print(f"  {k}: {v}")
