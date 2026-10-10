import os
import hashlib
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# Publication Styling Constants
# ==============================================================================
BG_COLOR = "#000000"
BOX_BG = "#111111"
BOX_EDGE = "#444444"
ORANGE = "#ff8000"
LIGHT_ORANGE = "#ffa64d"
GRAY = "#888888"
LIGHT_GRAY = "#cccccc"
WHITE = "#ffffff"
CYAN = "#00c3ff"
GREEN = "#00e676"


def ensure_diagrams_dir() -> str:
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagrams"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ==============================================================================
# 1. SteerNet Diagram (PNG + Draw.io)
# ==============================================================================

def generate_steernet_png(out_path: str):
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(5.0, 8.4, "RSCSteerNet: K-Conditioned Low-Rank Residual MLP",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(5.0, 8.0, r"Bounded Relative Perturbation: $\|\delta\| \leq \beta \|h\|$, $\delta_t = \frac{\alpha}{r} W_{\mathrm{up}}(\mathrm{MLP}(h) + W_k(K))$",
            color=GRAY, fontsize=9.5, ha="center")

    def draw_box(x, y, w, h, text, sub="", edge=BOX_EDGE, bg=BOX_BG, text_col=WHITE):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=1.8, edgecolor=edge, facecolor=bg)
        ax.add_patch(rect)
        if sub:
            ax.text(x, y + 0.12, text, color=text_col, fontsize=10, fontweight="bold", ha="center", va="center")
            ax.text(x, y - 0.16, sub, color=LIGHT_GRAY, fontsize=8, ha="center", va="center")
        else:
            ax.text(x, y, text, color=text_col, fontsize=10, fontweight="bold", ha="center", va="center")

    def draw_arrow(x1, y1, x2, y2, col=ORANGE, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=15))
        if label:
            ax.text((x1 + x2)/2 + 0.15, (y1 + y2)/2, label, color=LIGHT_GRAY, fontsize=8)

    # Input h
    draw_box(3.5, 7.2, 2.8, 0.65, "Input Hidden State", r"$h \in \mathbb{R}^{B \times T \times d}$", edge=GRAY)
    draw_arrow(3.5, 6.85, 3.5, 6.25)

    # LayerNorm
    draw_box(3.5, 5.9, 2.8, 0.65, "Layer Normalization", r"$\mathrm{LayerNorm}(h)$", edge=GRAY)
    draw_arrow(3.5, 5.55, 3.5, 4.95)

    # Down Projection
    draw_box(3.5, 4.6, 3.0, 0.65, "Down Projection + GELU", r"$W_{\mathrm{down}} \in \mathbb{R}^{d \times r}$ (rank $r \ll d$)", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(3.5, 4.25, 3.5, 3.55)

    # Active K Conditioning
    draw_box(7.5, 4.6, 2.6, 0.75, "Active K Input", r"$K \in \{1,\dots,4\} \rightarrow K/4.0$", edge=CYAN, text_col=CYAN)
    draw_box(7.5, 3.4, 2.6, 0.65, "K-Conditioning Proj", r"$W_k \in \mathbb{R}^{1 \times r}$", edge=CYAN, text_col=CYAN)
    draw_arrow(7.5, 4.2, 7.5, 3.75, col=CYAN)

    # Sum / Fusion Node
    draw_box(3.5, 3.2, 2.2, 0.55, "Feature Addition", r"$x + W_k(K/4)$", edge=GRAY)
    draw_arrow(6.2, 3.4, 4.65, 3.25, col=CYAN, label="K context")
    draw_arrow(3.5, 2.9, 3.5, 2.25)

    # Mid Layer + Residual Bottleneck
    draw_box(3.5, 1.9, 3.0, 0.65, "Bottleneck MLP + GELU", r"$W_{\mathrm{mid}} \in \mathbb{R}^{r \times r} + \mathrm{residual}$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(3.5, 1.55, 3.5, 0.95)

    # Up Projection & Output
    draw_box(3.5, 0.6, 3.6, 0.65, "Up Proj & Alpha Scaling", r"$\delta = \frac{\alpha}{r} W_{\mathrm{up}}(x) \in \mathbb{R}^{B \times T \times d}$", edge=GREEN, text_col=GREEN)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
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
          <mxGeometry x="220" y="80" width="220" height="50" as="geometry" />
        </mxCell>
        <mxCell id="norm" value="LayerNorm(h)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="220" y="160" width="220" height="45" as="geometry" />
        </mxCell>
        <mxCell id="down" value="Down Projection + GELU &lt;br&gt;&lt;b&gt;W_down ∈ ℝ^(d×r)&lt;/b&gt; (r ≪ d)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="210" y="235" width="240" height="50" as="geometry" />
        </mxCell>
        <mxCell id="k_in" value="Active K Input &lt;br&gt;K ∈ {1..4} → K / 4.0" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="520" y="235" width="180" height="50" as="geometry" />
        </mxCell>
        <mxCell id="k_proj" value="K Projection Proj&lt;br&gt;&lt;b&gt;W_k ∈ ℝ^(1×r)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="520" y="315" width="180" height="45" as="geometry" />
        </mxCell>
        <mxCell id="add_k" value="Conditioning Addition &lt;br&gt;x = x + W_k(K/4)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="220" y="315" width="220" height="45" as="geometry" />
        </mxCell>
        <mxCell id="mid" value="Bottleneck MLP + Skip &lt;br&gt;&lt;b&gt;W_mid ∈ ℝ^(r×r)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="210" y="390" width="240" height="50" as="geometry" />
        </mxCell>
        <mxCell id="up" value="Up Projection &amp; Scaling &lt;br&gt;&lt;b&gt;δ = (α/r) · W_up(x) ∈ ℝ^(B×T×d)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#002411;strokeColor=#00e676;fontColor=#00e676;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="200" y="470" width="260" height="55" as="geometry" />
        </mxCell>
        <mxCell id="e1" edge="1" source="in_h" target="norm" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e2" edge="1" source="norm" target="down" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e3" edge="1" source="down" target="add_k" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e4" edge="1" source="k_in" target="k_proj" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#00c3ff;strokeWidth=2;" />
        <mxCell id="e5" edge="1" source="k_proj" target="add_k" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#00c3ff;strokeWidth=2;" />
        <mxCell id="e6" edge="1" source="add_k" target="mid" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="e7" edge="1" source="mid" target="up" parent="1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#00e676;strokeWidth=2;" />
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
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(5.0, 8.4, "RSCUsefulnessGate: Multi-Feature Dynamic Token Gate",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(5.0, 8.0, r"Per-token confidence scalar: $\alpha_t = \sigma( \mathrm{MLP}( [\mathrm{LN}(h), \mathrm{LN}(\delta), \mathrm{mag}, \mathrm{cos}, c_{\mathrm{ctx}}] ) ) \in (0, 1)$",
            color=GRAY, fontsize=10, ha="center")

    def draw_box(x, y, w, h, text, sub="", edge=BOX_EDGE, bg=BOX_BG, text_col=WHITE):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=1.8, edgecolor=edge, facecolor=bg)
        ax.add_patch(rect)
        if sub:
            ax.text(x, y + 0.12, text, color=text_col, fontsize=9.5, fontweight="bold", ha="center", va="center")
            ax.text(x, y - 0.16, sub, color=LIGHT_GRAY, fontsize=8, ha="center", va="center")
        else:
            ax.text(x, y, text, color=text_col, fontsize=9.5, fontweight="bold", ha="center", va="center")

    def draw_arrow(x1, y1, x2, y2, col=ORANGE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=15))

    # Input Branches
    draw_box(2.0, 7.0, 2.4, 0.65, "Hidden State", r"$h \in \mathbb{R}^{B \times T \times d}$", edge=GRAY)
    draw_box(5.0, 7.0, 2.4, 0.65, "Steer Correction", r"$\delta \in \mathbb{R}^{B \times T \times d}$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_box(8.0, 7.0, 2.4, 0.65, "Semantic Context", r"$c_{\mathrm{ctx}} \in \mathbb{R}^{B \times 16}$", edge=CYAN, text_col=CYAN)

    # Feature Extractors
    draw_box(1.5, 5.7, 2.0, 0.65, "LN(h)", r"LayerNorm", edge=GRAY)
    draw_box(3.8, 5.7, 2.0, 0.65, "LN(delta)", r"LayerNorm", edge=ORANGE)
    draw_box(6.2, 5.7, 2.0, 0.65, "Magnitude", r"$\log(1 + \|\delta\|_2)$", edge=LIGHT_ORANGE)
    draw_box(8.5, 5.7, 2.0, 0.65, "Cosine Sim", r"$\cos(h, \delta)$", edge=CYAN)

    draw_arrow(2.0, 6.65, 1.5, 6.05)
    draw_arrow(5.0, 6.65, 3.8, 6.05)
    draw_arrow(5.0, 6.65, 6.2, 6.05)
    draw_arrow(2.0, 6.65, 8.5, 6.05)
    draw_arrow(5.0, 6.65, 8.5, 6.05)

    # Concat Node
    draw_box(5.0, 4.3, 5.5, 0.65, "Feature Fusion Concat", r"$[ \mathrm{LN}(h), \mathrm{LN}(\delta), \mathrm{mag}, \mathrm{cos}, c_{\mathrm{ctx}} ] \in \mathbb{R}^{2d + 2 + 16}$", edge=WHITE)
    draw_arrow(1.5, 5.35, 3.5, 4.65)
    draw_arrow(3.8, 5.35, 4.5, 4.65)
    draw_arrow(6.2, 5.35, 5.5, 4.65)
    draw_arrow(8.5, 5.35, 6.5, 4.65)
    draw_arrow(8.0, 6.65, 7.2, 4.65, col=CYAN)

    # Gate MLP
    draw_box(5.0, 3.0, 4.0, 0.65, "GELU Dense Layer", r"$\mathrm{Linear}(2d+18 \rightarrow d_{\mathrm{gate}}=64) + \mathrm{Dropout}$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(5.0, 3.95, 5.0, 3.35)

    draw_box(5.0, 1.8, 3.6, 0.65, "Projection Head", r"$\mathrm{Linear}(64 \rightarrow 1) + \mathrm{InitBias}(\mathrm{logit}_0)$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(5.0, 2.65, 5.0, 2.15)

    # Sigmoid Output
    draw_box(5.0, 0.6, 3.8, 0.65, "Usefulness Gate Output", r"$\alpha_t = \sigma(\cdot) \in (0, 1)^{B \times T \times 1}$", edge=GREEN, text_col=GREEN)
    draw_arrow(5.0, 1.45, 5.0, 0.95, col=GREEN)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_gate_drawio(out_path: str):
    xml_content = """<mxfile host="app.diagrams.net" version="22.1.0">
  <diagram id="gate-arch" name="Token Gate Architecture">
    <mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="650" background="#000000" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="title" value="&lt;b&gt;RSCUsefulnessGate: Multi-Feature Dynamic Token Gate&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="150" y="20" width="550" height="30" as="geometry" />
        </mxCell>
        <mxCell id="in_h" value="Hidden State &lt;br&gt;&lt;i&gt;h ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="100" y="80" width="160" height="45" as="geometry" />
        </mxCell>
        <mxCell id="in_delta" value="Steer Correction &lt;br&gt;&lt;i&gt;δ ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="340" y="80" width="160" height="45" as="geometry" />
        </mxCell>
        <mxCell id="in_ctx" value="Semantic Context &lt;br&gt;&lt;i&gt;c_ctx ∈ ℝ^(B×16)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="580" y="80" width="160" height="45" as="geometry" />
        </mxCell>
        <mxCell id="ln_h" value="LN(h)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="80" y="170" width="110" height="40" as="geometry" />
        </mxCell>
        <mxCell id="ln_d" value="LN(δ)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="220" y="170" width="110" height="40" as="geometry" />
        </mxCell>
        <mxCell id="mag" value="log(1 + ‖δ‖₂)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="360" y="170" width="110" height="40" as="geometry" />
        </mxCell>
        <mxCell id="cos" value="cos(h, δ)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="500" y="170" width="110" height="40" as="geometry" />
        </mxCell>
        <mxCell id="concat" value="Feature Concatenation &lt;br&gt;&lt;b&gt;[ LN(h), LN(δ), mag, cos, c_ctx ] ∈ ℝ^(2d + 18)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="180" y="260" width="480" height="45" as="geometry" />
        </mxCell>
        <mxCell id="mlp1" value="Dense Projection &amp; Dropout &lt;br&gt;&lt;b&gt;Linear(2d+18 → 64) + GELU&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="280" y="345" width="280" height="50" as="geometry" />
        </mxCell>
        <mxCell id="mlp2" value="Projection Head &lt;br&gt;&lt;b&gt;Linear(64 → 1) + Init Bias&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="290" y="430" width="260" height="45" as="geometry" />
        </mxCell>
        <mxCell id="out" value="Dynamic Sigmoid Token Gate &lt;br&gt;&lt;b&gt;α_t = σ(·) ∈ (0, 1)^(B×T×1)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#002411;strokeColor=#00e676;fontColor=#00e676;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="270" y="515" width="300" height="50" as="geometry" />
        </mxCell>
        <mxCell id="ge1" edge="1" source="in_h" target="ln_h" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge2" edge="1" source="in_delta" target="ln_d" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge3" edge="1" source="in_delta" target="mag" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge4" edge="1" source="in_delta" target="cos" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00c3ff;strokeWidth=2;" />
        <mxCell id="ge5" edge="1" source="ln_h" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#888888;strokeWidth=1.5;" />
        <mxCell id="ge6" edge="1" source="ln_d" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#888888;strokeWidth=1.5;" />
        <mxCell id="ge7" edge="1" source="mag" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#888888;strokeWidth=1.5;" />
        <mxCell id="ge8" edge="1" source="cos" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#888888;strokeWidth=1.5;" />
        <mxCell id="ge9" edge="1" source="in_ctx" target="concat" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00c3ff;strokeWidth=2;" />
        <mxCell id="ge10" edge="1" source="concat" target="mlp1" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge11" edge="1" source="mlp1" target="mlp2" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="ge12" edge="1" source="mlp2" target="out" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00e676;strokeWidth=2;" />
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
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    ax.text(5.0, 9.0, "JointLayerRouter: Difficulty-Aware 2-Pass Router",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(5.0, 8.6, r"Difficulty-aware joint scoring, Layer Dropout, learned $K$, and Synergy $S_{ij}$",
            color=GRAY, fontsize=10, ha="center")

    def draw_box(x, y, w, h, text, sub="", edge=BOX_EDGE, bg=BOX_BG, text_col=WHITE):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=1.8, edgecolor=edge, facecolor=bg)
        ax.add_patch(rect)
        if sub:
            ax.text(x, y + 0.12, text, color=text_col, fontsize=9.5, fontweight="bold", ha="center", va="center")
            ax.text(x, y - 0.16, sub, color=LIGHT_GRAY, fontsize=8, ha="center", va="center")
        else:
            ax.text(x, y, text, color=text_col, fontsize=9.5, fontweight="bold", ha="center", va="center")

    def draw_arrow(x1, y1, x2, y2, col=ORANGE, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=15))
        if label:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, label, color=LIGHT_GRAY, fontsize=8)

    # Input Sequence
    draw_box(5.0, 7.8, 3.8, 0.65, "Early Hidden State Tokens", r"$H \in \mathbb{R}^{B \times T \times d}$ (Layer 9)", edge=GRAY)

    # Semantic Summary Projection
    draw_box(5.0, 6.7, 4.2, 0.65, "Semantic Summary Proj", r"$\mathrm{LayerNorm} \rightarrow \mathrm{Linear}(d \rightarrow 16) \rightarrow \tanh$", edge=CYAN, text_col=CYAN)
    draw_arrow(5.0, 7.45, 5.0, 7.05)

    # Candidate Embeddings
    draw_box(5.0, 5.5, 5.2, 0.65, "Candidate Embeddings", r"Depth + Difficulty ($T,\mathrm{var}$) + Identity + Semantics", edge=WHITE)
    draw_arrow(5.0, 6.35, 5.0, 5.85)

    # Pass 1: Transformer Encoder
    draw_box(5.0, 4.3, 4.2, 0.65, "Pass 1: Joint Self-Attention", r"$2\times \mathrm{Heads}, \mathrm{GELU} \rightarrow \mathrm{UtilityHead}(u_1)$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(5.0, 5.15, 5.0, 4.65)

    # Pass 2: Iterative Refinement
    draw_box(5.0, 3.1, 4.4, 0.65, "Pass 2: Iterative Refinement", r"Conditioned on $u_1 \rightarrow$ Refined Utilities $u_2$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(5.0, 3.95, 5.0, 3.45)

    # Split: K-Head vs Top-K Routing vs Synergy Matrix
    # K-Head (Left)
    draw_box(2.0, 1.7, 2.6, 0.75, "K-Distribution Head", r"$\mathrm{Softmax} \rightarrow K \in \{1..4\}$", edge=CYAN, text_col=CYAN)
    draw_arrow(3.6, 2.8, 2.2, 2.1, col=CYAN)

    # Active Layer Selection (Center)
    draw_box(5.0, 1.7, 2.8, 0.75, "Top-K Selection", r"$\mathrm{argtopk}(u_2, K) \rightarrow \{9, 15, 20\}$", edge=GREEN, text_col=GREEN)
    draw_arrow(5.0, 2.75, 5.0, 2.1, col=GREEN)

    # Synergy Matrix S_ij (Right)
    draw_box(8.0, 1.7, 2.8, 0.75, "Pairwise Synergy Matrix", r"$S_{ij} = \tanh\left(\frac{q_i^T k_j}{\sqrt{d}}\right) \in [-1, 1]$", edge=LIGHT_ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(6.4, 2.8, 7.8, 2.1, col=LIGHT_ORANGE)

    # Final Combined Output
    draw_box(5.0, 0.5, 4.8, 0.6, "Routing Mask & Synergy Alignment", r"Realized $K=3$ (88.4%) | Alliances: (9, 15, 20)", edge=WHITE)
    draw_arrow(2.0, 1.3, 4.0, 0.8, col=GRAY)
    draw_arrow(5.0, 1.3, 5.0, 0.8, col=GRAY)
    draw_arrow(8.0, 1.3, 6.0, 0.8, col=GRAY)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()


def generate_router_drawio(out_path: str):
    xml_content = """<mxfile host="app.diagrams.net" version="22.1.0">
  <diagram id="router-arch" name="Router Architecture">
    <mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="650" background="#000000" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="title" value="&lt;b&gt;JointLayerRouter: 2-Pass Attention &amp; Synergy Router&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=16;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="150" y="20" width="550" height="30" as="geometry" />
        </mxCell>
        <mxCell id="in_h" value="Early Hidden State Tokens &lt;br&gt;&lt;i&gt;H ∈ ℝ^(B×T×d)&lt;/i&gt; (Layer 9)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="270" y="70" width="280" height="45" as="geometry" />
        </mxCell>
        <mxCell id="sem_proj" value="Semantic Summary Projection &lt;br&gt;&lt;b&gt;LN → Linear(d → 16) → tanh&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="260" y="145" width="300" height="45" as="geometry" />
        </mxCell>
        <mxCell id="cand_emb" value="Candidate Embeddings &lt;br&gt;&lt;i&gt;Depth + Identity (d_model=32) + Candidate Semantics&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="230" y="220" width="360" height="45" as="geometry" />
        </mxCell>
        <mxCell id="pass1" value="Pass 1: Joint Self-Attention &lt;br&gt;&lt;b&gt;2 Heads, GELU → UtilityHead (u₁)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="250" y="295" width="320" height="45" as="geometry" />
        </mxCell>
        <mxCell id="pass2" value="Pass 2: Iterative Refinement &lt;br&gt;&lt;b&gt;Conditioned on u₁ → Refined Utilities u₂&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="245" y="370" width="330" height="45" as="geometry" />
        </mxCell>
        <mxCell id="k_head" value="K-Head (Softmax)&lt;br&gt;&lt;b&gt;P(K) ∈ ℝ^4 → K=3&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="100" y="460" width="180" height="50" as="geometry" />
        </mxCell>
        <mxCell id="topk" value="Top-K Layer Selection&lt;br&gt;&lt;b&gt;argtopk(u₂, K) → {9, 15, 20}&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#002411;strokeColor=#00e676;fontColor=#00e676;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="320" y="460" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="synergy" value="Pairwise Synergy Matrix&lt;br&gt;&lt;b&gt;S_ij = tanh(q_i^T k_j / √d)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="560" y="460" width="200" height="50" as="geometry" />
        </mxCell>
        <mxCell id="re1" edge="1" source="in_h" target="sem_proj" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re2" edge="1" source="sem_proj" target="cand_emb" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re3" edge="1" source="cand_emb" target="pass1" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re4" edge="1" source="pass1" target="pass2" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="re5" edge="1" source="pass2" target="k_head" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00c3ff;strokeWidth=2;" />
        <mxCell id="re6" edge="1" source="pass2" target="topk" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00e676;strokeWidth=2;" />
        <mxCell id="re7" edge="1" source="pass2" target="synergy" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ffa64d;strokeWidth=2;" />
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
    fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5.5, 9.5, "Adaptive Residual Steering (ARS): Full Architecture Integration",
            color=WHITE, fontsize=14, fontweight="bold", ha="center")
    ax.text(5.5, 9.1, r"End-to-end forward hook modulation with Prompt Shielding & Cooperative Routing",
            color=GRAY, fontsize=10, ha="center")

    def draw_box(x, y, w, h, text, sub="", edge=BOX_EDGE, bg=BOX_BG, text_col=WHITE):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=1.8, edgecolor=edge, facecolor=bg)
        ax.add_patch(rect)
        if sub:
            ax.text(x, y + 0.12, text, color=text_col, fontsize=9.5, fontweight="bold", ha="center", va="center")
            ax.text(x, y - 0.16, sub, color=LIGHT_GRAY, fontsize=8, ha="center", va="center")
        else:
            ax.text(x, y, text, color=text_col, fontsize=9.5, fontweight="bold", ha="center", va="center")

    def draw_arrow(x1, y1, x2, y2, col=ORANGE, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=15))
        if label:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, label, color=LIGHT_GRAY, fontsize=8)

    # Prompt Shielding Mask (Top Left)
    draw_box(2.2, 8.0, 3.4, 0.75, "Prompt Shielding Mask", r"$M_{\mathrm{shield}} \in \{0, 1\}^{B \times T}$ (Few-Shot)", edge=CYAN, text_col=CYAN)

    # Transformer Backbone Layer
    draw_box(6.8, 8.0, 3.6, 0.75, "Frozen Transformer Layer l", r"Hidden state output: $h_l \in \mathbb{R}^{B \times T \times d}$", edge=GRAY)

    # Forward Hook Tap
    draw_box(6.8, 6.7, 3.2, 0.65, "Forward Hook Intercept", "Tap output hidden state $h_l$", edge=WHITE)
    draw_arrow(6.8, 7.6, 6.8, 7.05)

    # Joint Router
    draw_box(2.2, 5.4, 3.4, 0.75, "Joint Attention Router", r"Selected $r_l \in \{0, 1\}$, Realized $K=3$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(6.8, 6.35, 3.5, 5.6, label="Early h (L9)")

    # SteerNet Block
    draw_box(5.5, 4.4, 2.8, 0.75, "RSCSteerNet", r"$\delta_l = \mathrm{MLP}(h_l, K) \in \mathbb{R}^{B \times T \times d}$", edge=ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(6.8, 6.35, 5.5, 4.8)
    draw_arrow(2.2, 5.0, 4.3, 4.5, col=CYAN, label="K ctx")

    # Token Gate Block
    draw_box(8.8, 4.4, 2.8, 0.75, "RSCUsefulnessGate", r"$\alpha_l = \sigma(h_l, \delta_l, c) \in (0, 1)$", edge=LIGHT_ORANGE, text_col=LIGHT_ORANGE)
    draw_arrow(6.8, 6.35, 8.8, 4.8)
    draw_arrow(5.5, 4.0, 7.5, 4.3, col=ORANGE, label="delta")

    # Steering Injection Formula
    draw_box(5.5, 2.5, 6.5, 0.9, "Adaptive Residual Addition",
             r"$h_l^{\mathrm{new}} = h_l + \frac{r_l \cdot \alpha_l \cdot \delta_l \cdot (1 - M_{\mathrm{shield}})}{\sqrt{K}}$",
             edge=GREEN, text_col=GREEN)

    draw_arrow(6.8, 6.35, 7.5, 2.95, label="Skip h_l")
    draw_arrow(5.5, 4.0, 5.5, 3.0, label="delta")
    draw_arrow(8.8, 4.0, 6.8, 2.95, label="alpha")
    draw_arrow(2.2, 7.6, 4.2, 2.95, col=CYAN, label="Shield")

    # Next Transformer Layer
    draw_box(5.5, 1.0, 4.2, 0.65, "Transformer Layer l+1 Input", "Cleanly conditioned modulated representation", edge=WHITE)
    draw_arrow(5.5, 2.05, 5.5, 1.35, col=GREEN)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
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
        <mxCell id="shield" value="Prompt Shielding Mask &lt;br&gt;&lt;b&gt;M_shield ∈ {0, 1}^(B×T)&lt;/b&gt; (Few-Shot)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#001a26;strokeColor=#00c3ff;fontColor=#00c3ff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="60" y="80" width="240" height="50" as="geometry" />
        </mxCell>
        <mxCell id="bb_layer" value="Frozen Transformer Layer l &lt;br&gt;&lt;i&gt;Output hidden state h_l ∈ ℝ^(B×T×d)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#888888;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="450" y="80" width="280" height="50" as="geometry" />
        </mxCell>
        <mxCell id="hook" value="Forward Hook Tap &lt;br&gt;Intercept h_l" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="490" y="170" width="200" height="40" as="geometry" />
        </mxCell>
        <mxCell id="router" value="Joint Attention Router &lt;br&gt;&lt;b&gt;Selects r_l ∈ {0, 1}, Realized K=3&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="60" y="250" width="240" height="55" as="geometry" />
        </mxCell>
        <mxCell id="steernet" value="RSCSteerNet &lt;br&gt;&lt;b&gt;δ_l = MLP(h_l, K)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ff8000;fontColor=#ff9933;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="370" y="250" width="200" height="55" as="geometry" />
        </mxCell>
        <mxCell id="gate" value="RSCUsefulnessGate &lt;br&gt;&lt;b&gt;α_l = σ(h_l, δ_l, c)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1100;strokeColor=#ffa64d;fontColor=#ffa64d;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="630" y="250" width="200" height="55" as="geometry" />
        </mxCell>
        <mxCell id="formula" value="Adaptive Residual Addition &lt;br&gt;&lt;b&gt;h_l^new = h_l + [ r_l · α_l · δ_l · (1 - M_shield) ] / √K&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#002411;strokeColor=#00e676;fontColor=#00e676;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="250" y="380" width="460" height="60" as="geometry" />
        </mxCell>
        <mxCell id="next_layer" value="Transformer Layer l+1 Input &lt;br&gt;Modulated Steering Representation" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#111111;strokeColor=#ffffff;fontColor=#ffffff;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="340" y="500" width="280" height="50" as="geometry" />
        </mxCell>
        <mxCell id="fe1" edge="1" source="bb_layer" target="hook" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#888888;strokeWidth=2;" />
        <mxCell id="fe2" edge="1" source="hook" target="router" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe3" edge="1" source="hook" target="steernet" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe4" edge="1" source="hook" target="gate" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="fe5" edge="1" source="router" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe6" edge="1" source="steernet" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ff8000;strokeWidth=2;" />
        <mxCell id="fe7" edge="1" source="gate" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#ffa64d;strokeWidth=2;" />
        <mxCell id="fe8" edge="1" source="shield" target="formula" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00c3ff;strokeWidth=2;" />
        <mxCell id="fe9" edge="1" source="formula" target="next_layer" parent="1" style="edgeStyle=orthogonalEdgeStyle;strokeColor=#00e676;strokeWidth=2;" />
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
    print("Generated all architecture diagrams:")
    for k, v in generated.items():
        print(f"  {k}: {v}")
