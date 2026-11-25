"""Generate editable Draw.io diagrams and presentation-ready PNGs.

Run from the repository root:
    python scripts/generate_study_diagrams.py

Each diagram is defined once and exported both as Draw.io XML and PNG. The
figures explain the current final-study path; they do not encode results that
are still pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_DIR = ROOT / "diagrams"
WIDTH, HEIGHT = 1600, 900
BLACK, ORANGE, WHITE = "#000000", "#FF7A00", "#FFFFFF"
FONT = "Calibri"
CALIBRI_REGULAR = Path("C:/Windows/Fonts/calibri.ttf")
CALIBRI_BOLD = Path("C:/Windows/Fonts/calibrib.ttf")


@dataclass(frozen=True)
class Node:
    key: str
    x: int
    y: int
    width: int
    height: int
    text: str
    emphasis: bool = False
    font_size: int = 21


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: str = ""
    source_anchor: str = "right"
    target_anchor: str = "left"


@dataclass(frozen=True)
class Diagram:
    stem: str
    title: str
    subtitle: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


DIAGRAMS = (
    Diagram(
        stem="diagram_ars_full_system",
        title="Adaptive Residual Steering",
        subtitle="Research workflow: verified data → fixed adapter → route bank and router → matched evaluation",
        nodes=(
            Node("data", 28, 265, 230, 370,
                 "1  ·  TRAINING DATA\nGSM8K official train\n7,473 source questions\n5,000 sampled for collection\nQuestion-level split\n0 / 2 / 4-shot contexts"),
            Node("rollouts", 290, 265, 230, 370,
                 "2  ·  VERIFIED ROLLOUTS\n2 initial attempts per question\nRetry only if zero strict positives\nUp to 2 targeted retries\nDeduplicate correct reasoning\nKeep up to 2 positives per question"),
            Node("fixed", 552, 265, 230, 370,
                 "3  ·  FIXED ADAPTER\nFrozen Phi-2 backbone\nRank 16 residual module\nRoute 12 / 15 / 18 / 21\nTrain on verified completions\np99 update bound"),
            Node("bank", 814, 265, 230, 370,
                 "4  ·  ROUTE BANK\n14 candidate layer locations\nRoutes with k = 1, 2, 3, 4\nMinimum layer gap: 3\nIndependent route experts\nScreen and select on held-out data", True),
            Node("router", 1076, 265, 230, 370,
                 "5  ·  PROMPT ROUTER\nPrompt-only features\nScore route × gain actions\nInclude no steering (k = 0)\nFit measured action scores\nSelect using task validation"),
            Node("eval", 1338, 265, 230, 370,
                 "6  ·  EVALUATION\nFour matched arms\n0-shot and 2-shot primary\nShot and transfer slices\nPaired deltas and uncertainty\nAnswer quality and runtime", True),
        ),
        edges=tuple(Edge(a, b) for a, b in zip(
            ("data", "rollouts", "fixed", "bank", "router"),
            ("rollouts", "fixed", "bank", "router", "eval"),
        )),
    ),
    Diagram(
        stem="diagram_rollout_collection",
        title="Building the Fixed-Adapter Training Set",
        subtitle="A strict-positive gate spends retries on questions that produced no verified correct answer initially",
        nodes=(
            Node("question", 50, 290, 255, 380,
                 "SOURCE QUESTION\nSampled from GSM8K train\nStable question ID\nAssign 0 / 2 / 4-shot context\nUse diversified prompt wording"),
            Node("initial", 360, 290, 255, 380,
                 "INITIAL SAMPLING\nGenerate 2 completions\nTemperature 0.7\ntop_p = 0.95\nRecord prompt, answer, and seed"),
            Node("verify", 670, 290, 255, 380,
                 "STRICT VERIFICATION\nCheck final answer and format\nKeep strict-positive attempts\nIf at least one is correct:\nno retry is needed", True),
            Node("retry", 980, 290, 255, 380,
                 "TARGETED RETRY\nOnly when initial strict positives = 0\nGenerate up to 2 more attempts\nTemperature 0.9\nKeep retry provenance"),
            Node("rows", 1290, 290, 255, 380,
                 "TRAINING ROWS\nDeduplicate by reasoning key\nKeep at most 2 distinct positives\nPrompt and padding excluded from loss\nSave shard and validation statistics", True),
        ),
        edges=tuple(Edge(a, b) for a, b in zip(
            ("question", "initial", "verify", "retry"),
            ("initial", "verify", "retry", "rows"),
        )),
    ),
    Diagram(
        stem="diagram_steernet",
        title="Fixed-Route Residual Adapter",
        subtitle="The same low-rank module is injected at layers 12, 15, 18, and 21; Phi-2 weights remain frozen",
        nodes=(
            Node("hidden", 32, 345, 200, 260,
                 "RESIDUAL STATE\nh_l\nper-token\nhidden vector"),
            Node("down", 258, 345, 200, 260,
                 "DOWN PROJECTION\nW_down\nd → rank 16\ntrainable"),
            Node("gelu", 484, 345, 200, 260,
                 "NONLINEARITY\nGELU\nacts in\nlow-rank space"),
            Node("up", 710, 345, 200, 260,
                 "UP PROJECTION\nW_up\nrank 16 → d\ntrainable"),
            Node("gain", 936, 345, 200, 260,
                 "CALIBRATED GAIN\nα from validation\nselected from a grid\nconstant for fixed arm"),
            Node("cap", 1162, 345, 200, 260,
                 "P99 NORM CAP\nρ_max = 0.10\nlayer-specific p99\nlimits update size", True),
            Node("add", 1388, 345, 180, 260,
                 "RESIDUAL ADD\nh'_l = h_l + Δ_l\ncontinue through\nfrozen backbone"),
        ),
        edges=tuple(Edge(a, b) for a, b in zip(
            ("hidden", "down", "gelu", "up", "gain", "cap"),
            ("down", "gelu", "up", "gain", "cap", "add"),
        )),
    ),
    Diagram(
        stem="diagram_route_bank",
        title="Independent Route Bank",
        subtitle="A staged search reduces the combinatorial space while keeping one separately trained expert per retained route",
        nodes=(
            Node("layers", 26, 285, 230, 390,
                 "CANDIDATE LAYERS\n8 · 11 · 12 · 13\n15 · 17 · 18 · 19\n20 · 21 · 22 · 23\n24 · 26\n14 locations from Stage A"),
            Node("spacing", 288, 285, 230, 390,
                 "ROUTE CONSTRAINTS\nRoute size k = 1…4\nMinimum layer gap = 3\nSeparate control gap = 2\nNo adjacent-layer pairs\nCaps bound retained routes"),
            Node("curriculum", 550, 285, 230, 390,
                 "CURRICULUM\nTrain all singletons\nScreen spaced pairs\nExpand selected subsets\nScreen triples, then quads\nKeep up to 8 per size", True),
            Node("expert", 812, 285, 230, 390,
                 "ROUTE EXPERTS\nInitialize larger routes\nfrom trained subsets\nCopy into route-specific weights\nFine-tune each expert separately\nNo shared expert weights"),
            Node("objective", 1074, 285, 230, 390,
                 "TRAINING SIGNAL\nToken-mean NLL on positives\nMatched same-prompt pairs\nadd softplus preference loss\nλ_pref = 0.25\nOnly use pairs when available"),
            Node("selection", 1336, 285, 230, 390,
                 "HELD-OUT SELECTION\nScreen each route on validation rows\nUse preference evidence when present\nChoose a static route and gain\nSeparate task-accuracy screen", True),
        ),
        edges=tuple(Edge(a, b) for a, b in zip(
            ("layers", "spacing", "curriculum", "expert", "objective"),
            ("spacing", "curriculum", "expert", "objective", "selection"),
        )),
    ),
    Diagram(
        stem="diagram_static_vs_dynamic",
        title="Three Steering Policies Answer Different Questions",
        subtitle="A fixed route is a reference; a static bank tests bank quality; the router tests prompt-conditioned action choice",
        nodes=(
            Node("fixed", 55, 240, 450, 470,
                 "FIXED STEERING\nOne trained route\nOne calibrated gain\nSame action for every question\n\nQuestion tested:\nDoes a learned residual update\nhelp compared with no steering?"),
            Node("static", 575, 240, 450, 470,
                 "STATIC BANK\nChoose one bank route and gain\non held-out data\nReuse that action for all questions\n\nQuestion tested:\nDoes the bank contain a better\ngeneral-purpose action?", True),
            Node("router", 1095, 240, 450, 470,
                 "PROMPT ROUTER\nRead each question's prompt features\nSelect route and gain per question\nMay choose no steering (k = 0)\n\nQuestion tested:\nCan prompt context identify which\naction helps this question?"),
        ),
        edges=(),
    ),
    Diagram(
        stem="diagram_router",
        title="Router: Separate Offline Learning from Test-Time Choice",
        subtitle="Teacher scoring and validation happen during training; inference sees only the prompt and selects a route–gain action",
        nodes=(
            Node("teacher", 40, 185, 350, 230,
                 "OFFLINE 1  ·  TEACHER SCORES\nScore route–gain actions\non router-training prompts\nAdd route cost: 0.01 × k"),
            Node("targets", 420, 185, 350, 230,
                 "OFFLINE 2  ·  TARGETS\nNormalize action scores per prompt\nRetain preference ordering\nStore prompt/action supervision"),
            Node("fit", 800, 185, 350, 230,
                 "OFFLINE 3  ·  FIT POLICY\nRouteValueRouter predicts action scores\nSmooth-L1 score regression\nplus KL teacher preference loss", True),
            Node("select", 1180, 185, 350, 230,
                 "OFFLINE 4  ·  SELECT CHECKPOINT\nUp to 3 epochs / 2,048 rows\nSeparate task validation: up to 84 rows\nRestore best selected checkpoint"),
            Node("prompt", 40, 525, 350, 230,
                 "INFERENCE 1  ·  INPUT\nQuestion plus demonstration context\nNo generated completion\nNo gold answer"),
            Node("features", 420, 525, 350, 230,
                 "INFERENCE 2  ·  FEATURES\nPooled valid prompt-token states\nNormalized shot count\nNormalized question-token count"),
            Node("catalog", 800, 525, 350, 230,
                 "INFERENCE 3  ·  SCORE CATALOG\nNo-op plus bank routes\nEach route paired with gain\nα = 1 / 1.25 / 1.5 / 1.75 / 2", True),
            Node("apply", 1180, 525, 350, 230,
                 "INFERENCE 4  ·  APPLY ACTION\nChoose one route and gain\nSet steering before generation\nGenerate answer with frozen Phi-2"),
        ),
        edges=tuple(Edge(a, b) for a, b in zip(
            ("teacher", "targets", "fit", "prompt", "features", "catalog"),
            ("targets", "fit", "select", "features", "catalog", "apply"),
        )),
    ),
    Diagram(
        stem="diagram_paired_evaluation",
        title="Matched Four-Arm Evaluation",
        subtitle="Every arm receives the same question IDs, shot condition, prompt schedule, and decoding settings",
        nodes=(
            Node("testset", 45, 350, 280, 230,
                 "SHARED TEST SET\nSame question IDs\nSame shot condition\nSame prompt schedule\nSame decoding settings"),
            Node("baseline", 500, 140, 330, 125,
                 "1  ·  BASELINE\nFrozen Phi-2, no intervention"),
            Node("fixed", 500, 300, 330, 125,
                 "2  ·  FIXED STEERING\nOne route and calibrated gain"),
            Node("static", 500, 460, 330, 125,
                 "3  ·  STATIC BANK\nOne held-out-selected action"),
            Node("router", 500, 620, 330, 125,
                 "4  ·  ROUTER\nPrompt-specific route and gain", True),
            Node("record", 1050, 310, 485, 390,
                 "PAIRED RESULT RECORD\nPer-question outcomes\nWins · losses · ties · Δ percentage points\nBootstrap 95% interval and McNemar test\nAnswer format and quality checks\nLatency, tokens, route usage", True),
        ),
        edges=(
            Edge("testset", "baseline"),
            Edge("testset", "fixed"),
            Edge("testset", "static"),
            Edge("testset", "router"),
            Edge("baseline", "record"),
            Edge("fixed", "record"),
            Edge("static", "record"),
            Edge("router", "record"),
        ),
    ),
)


def _anchor(node: Node, side: str) -> tuple[float, float]:
    if side == "left":
        return node.x, node.y + node.height / 2
    if side == "right":
        return node.x + node.width, node.y + node.height / 2
    if side == "top":
        return node.x + node.width / 2, node.y
    if side == "bottom":
        return node.x + node.width / 2, node.y + node.height
    raise ValueError(f"Unknown connector anchor: {side}")


def _split_heading_body(text: str) -> tuple[str, str]:
    parts = text.split("\n", 1)
    return parts[0], parts[1] if len(parts) == 2 else ""


def _draw_png(diagram: Diagram, path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
    draw = ImageDraw.Draw(image)
    regular_path = str(CALIBRI_REGULAR) if CALIBRI_REGULAR.is_file() else "C:/Windows/Fonts/arial.ttf"
    bold_path = str(CALIBRI_BOLD) if CALIBRI_BOLD.is_file() else "C:/Windows/Fonts/arialbd.ttf"
    def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(bold_path if bold else regular_path, size=size)

    def wrap_pixels(value: str, font_obj: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        result: list[str] = []
        for paragraph in value.splitlines():
            if not paragraph:
                result.append("")
                continue
            words = paragraph.split()
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if line and draw.textbbox((0, 0), candidate, font=font_obj)[2] > max_width:
                    result.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                result.append(line)
        return result

    def centered_lines(lines: list[str], center_x: float, start_y: float,
                       font_obj: ImageFont.FreeTypeFont, color: str,
                       line_gap: int) -> int:
        y = start_y
        for line in lines:
            if line:
                draw.text((center_x, y), line, fill=color, font=font_obj, anchor="ma")
            y += line_gap
        return y

    node_map = {node.key: node for node in diagram.nodes}
    for edge in diagram.edges:
        source, target = node_map[edge.source], node_map[edge.target]
        sx, sy = _anchor(source, edge.source_anchor)
        tx, ty = _anchor(target, edge.target_anchor)
        start = (int(sx), int(sy))
        end = (int(tx), int(ty))
        draw.line((start, end), fill=ORANGE, width=4)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        tip = (end[0] - int(11 * math.cos(angle)), end[1] - int(11 * math.sin(angle)))
        normal = angle + math.pi / 2
        points = [end,
                  (tip[0] + int(7 * math.cos(normal)), tip[1] + int(7 * math.sin(normal))),
                  (tip[0] - int(7 * math.cos(normal)), tip[1] - int(7 * math.sin(normal)))]
        draw.polygon(points, fill=ORANGE)
        if edge.label:
            label_font = font(16)
            draw.text(((sx + tx) / 2, (sy + ty) / 2 - 14), edge.label,
                      fill=ORANGE, font=label_font, anchor="mm")

    for node in diagram.nodes:
        fill_color = ORANGE if node.emphasis else BLACK
        text_color = BLACK if node.emphasis else WHITE
        box = (node.x, node.y, node.x + node.width, node.y + node.height)
        draw.rounded_rectangle(box, radius=18, outline=ORANGE, fill=fill_color, width=3)
        heading, body = _split_heading_body(node.text)
        heading_rows = heading.splitlines()
        body_rows = body.splitlines() if body else []
        heading_size = min(node.font_size + 4, 26)
        body_size = min(node.font_size + 1, 23)
        heading_font = font(heading_size, bold=True)
        body_font = font(body_size)
        content_width = node.width - 30
        heading_rows = [line.replace("·", "|").replace("→", "to")
                        for line in heading_rows]
        heading_rows = [wrapped for line in heading_rows
                        for wrapped in wrap_pixels(line, heading_font, content_width)]
        body_rows = [line.replace("·", "|").replace("→", "to") for line in body_rows]
        body_rows = [wrapped for line in body_rows
                     for wrapped in wrap_pixels(line, body_font, content_width)]
        heading_gap = heading_size + 7
        body_gap = body_size + 7
        heading_height = len(heading_rows) * heading_gap
        body_height = len(body_rows) * body_gap
        available = node.height - 36
        if heading_height + body_height + 10 > available and body_rows:
            body_size = max(13, int((available - heading_height - 10) / max(1, len(body_rows)) - 7))
            body_font = font(body_size)
            body_gap = body_size + 7
            body_rows = [wrapped for line in [line.replace("·", "|").replace("→", "to")
                                              for line in (body.splitlines() if body else [])]
                         for wrapped in wrap_pixels(line, body_font, content_width)]
            body_height = len(body_rows) * body_gap
        total_height = heading_height + (12 if body_rows else 0) + body_height
        y = node.y + max(18, (node.height - total_height) // 2)
        y = centered_lines(heading_rows, node.x + node.width / 2, y,
                           heading_font, text_color, heading_gap)
        if body_rows:
            y += 12
            centered_lines(body_rows, node.x + node.width / 2, y,
                           body_font, text_color, body_gap)

    title_font = font(34, bold=True)
    subtitle_font = font(19)
    draw.text((55, 60), diagram.title, fill=ORANGE, font=title_font, anchor="lm")
    draw.text((58, 112), diagram.subtitle, fill=WHITE, font=subtitle_font, anchor="lm")
    image.save(path)


def _draw_drawio(diagram: Diagram, path: Path) -> None:
    mxfile = Element("mxfile", {"host": "app.diagrams.net", "version": "24.7.17"})
    page = SubElement(mxfile, "diagram", {"id": diagram.stem, "name": diagram.title})
    graph = SubElement(page, "mxGraphModel", {
        "dx": str(WIDTH), "dy": str(HEIGHT), "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(WIDTH),
        "pageHeight": str(HEIGHT), "math": "0", "shadow": "0",
    })
    root = SubElement(graph, "root")
    SubElement(root, "mxCell", {"id": "0"})
    SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    def add_text(cell_id: str, text: str, x: int, y: int, w: int, h: int,
                 size: int, color: str, bold: bool = False) -> None:
        style = (f"text;html=1;strokeColor=none;fillColor=none;align=left;"
                 f"verticalAlign=middle;fontFamily={FONT};fontSize={size};"
                 f"fontColor={color};fontStyle={1 if bold else 0};")
        cell = SubElement(root, "mxCell", {
            "id": cell_id, "value": escape(text).replace("\n", "&lt;br&gt;"),
            "style": style, "vertex": "1", "parent": "1",
        })
        SubElement(cell, "mxGeometry", {
            "x": str(x), "y": str(y), "width": str(w), "height": str(h),
            "as": "geometry",
        })

    add_text("title", diagram.title, 55, 30, 1490, 58, 32, ORANGE, True)
    add_text("subtitle", diagram.subtitle, 58, 82, 1490, 42, 21, WHITE)

    node_ids: dict[str, str] = {}
    for index, node in enumerate(diagram.nodes, start=2):
        cell_id = str(index)
        node_ids[node.key] = cell_id
        fill_color = ORANGE if node.emphasis else BLACK
        font_color = BLACK if node.emphasis else WHITE
        heading, body = node.text.split("\n", 1) if "\n" in node.text else (node.text, "")
        value = f"<b>{escape(heading)}</b>" + ("<br><br>" + escape(body).replace("\n", "<br>") if body else "")
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill_color};"
                 f"strokeColor={ORANGE};fontColor={font_color};fontFamily={FONT};"
                 f"fontSize={node.font_size + 1};align=center;verticalAlign=middle;"
                 "spacing=12;strokeWidth=2;")
        cell = SubElement(root, "mxCell", {
            "id": cell_id, "value": value, "style": style,
            "vertex": "1", "parent": "1",
        })
        SubElement(cell, "mxGeometry", {
            "x": str(node.x), "y": str(node.y), "width": str(node.width),
            "height": str(node.height), "as": "geometry",
        })

    anchor_coords = {"left": (0, 0.5), "right": (1, 0.5), "top": (0.5, 0), "bottom": (0.5, 1)}
    next_id = len(diagram.nodes) + 2
    for edge_index, edge in enumerate(diagram.edges):
        sx, sy = anchor_coords[edge.source_anchor]
        tx, ty = anchor_coords[edge.target_anchor]
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
                 f"exitX={sx};exitY={sy};entryX={tx};entryY={ty};jettySize=auto;"
                 f"html=1;strokeColor={ORANGE};strokeWidth=2;endArrow=block;endFill=1;"
                 f"fontColor={WHITE};fontFamily={FONT};fontSize=16;")
        cell = SubElement(root, "mxCell", {
            "id": str(next_id + edge_index), "value": escape(edge.label),
            "style": style, "edge": "1", "parent": "1",
            "source": node_ids[edge.source], "target": node_ids[edge.target],
        })
        SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    path.write_bytes(tostring(mxfile, encoding="utf-8", xml_declaration=True))


def main() -> None:
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    for diagram in DIAGRAMS:
        _draw_png(diagram, DIAGRAM_DIR / f"{diagram.stem}.png")
        _draw_drawio(diagram, DIAGRAM_DIR / f"{diagram.stem}.drawio")
        print(f"Generated {diagram.stem}.drawio and .png")


if __name__ == "__main__":
    main()
