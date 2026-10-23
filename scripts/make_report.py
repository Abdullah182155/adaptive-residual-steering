"""Generate summary tables (Markdown + LaTeX) from experiment_record.json."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.stats import compare_to_control

ARM_ORDER  = ["baseline", "fixed", "router", "full_ars", "raw_ars"]
_LATEX_ESC = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def latex_escape(s) -> str:
    return "".join(_LATEX_ESC.get(c, c) for c in str(s))


def load_record(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_p(p) -> str:
    if p is None:
        return "-"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _ordered(arms: Dict[str, Any]) -> List[str]:
    return [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]


def ensure_stats(block: Dict[str, Any], n_boot: int = 10000, seed: int = 42, margin: float = 1.0):
    """Use stored paired stats; recompute from per-item bits if absent."""
    if block.get("stats"):
        return block["stats"]
    arms = block["arms"]
    if "baseline" not in arms:
        return {}
    stats = {}
    for metric, key in (("strict", "correct_strict"), ("lenient", "correct_lenient")):
        treat = {a: arms[a][key] for a in arms if a != "baseline"}
        stats[metric] = compare_to_control(
            arms["baseline"][key], treat, n_boot=n_boot, seed=seed, noninf_margin_pp=margin
        )
    return stats


def build_tables(block: Dict[str, Any], n_boot: int = 10000, seed: int = 42) -> Dict[str, Dict[str, Any]]:
    arms   = block["arms"]
    order  = _ordered(arms)
    stats  = ensure_stats(block, n_boot, seed)
    tables: Dict[str, Dict[str, Any]] = {}

    tables["accuracy"] = {
        "columns": ["Arm", "Strict acc (%)", "Lenient acc (%)"],
        "rows": [
            [arms[a]["label"], f"{arms[a]['accuracy_strict']:.2f}", f"{arms[a]['accuracy_lenient']:.2f}"]
            for a in order
        ],
    }

    for metric in ("strict", "lenient"):
        rows = []
        for a in order:
            if a == "baseline" or metric not in stats or a not in stats[metric]:
                continue
            s = stats[metric][a]
            rows.append([
                arms[a]["label"],
                f"{s['delta_pp']:+.2f}",
                f"[{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]",
                f"({s['b']}, {s['c']})",
                _fmt_p(s["p"]),
                _fmt_p(s["p_holm"]),
            ])
        tables[f"paired_{metric}"] = {
            "columns": ["Arm", "Delta vs baseline (pp)", "95% CI", "(b, c)", "McNemar p", "Holm p"],
            "rows": rows,
        }

    rrows = []
    for a in order:
        sm = arms[a]["summary"]
        kd = " ".join(f"{k}:{v:.2f}" for k, v in sm["k_dist"].items())
        lf = " ".join(f"L{k}:{v:.2f}" for k, v in sm["layer_freq"].items()) or "-"
        rrows.append([
            arms[a]["label"],
            arms[a]["gate_mode"],
            f"{sm['route_entropy_bits']:.3f}",
            f"{sm['abstain_rate']:.3f}",
            kd, lf,
            f"{sm.get('guard_trip_rate', 0.0):.4f}",
        ])
    tables["routing"] = {
        "columns": [
            "Arm", "Gate mode", "Route entropy (bits)", "Abstain rate",
            "K distribution", "Layer frequencies", "Guard trip rate",
        ],
        "rows": rrows,
    }
    return tables


def to_markdown(table: Dict[str, Any]) -> str:
    cols  = table["columns"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in table["rows"]:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def to_latex(table: Dict[str, Any], caption: str = "", label: str = "") -> str:
    cols = table["columns"]
    spec = "l" + "r" * (len(cols) - 1)
    out  = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for r in table["rows"]:
        out.append(" & ".join(latex_escape(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        out.append(rf"\caption{{{latex_escape(caption)}}}")
    if label:
        out.append(rf"\label{{{label}}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def write_report(record: Dict[str, Any], out_dir: str = "report") -> List[str]:
    tdir = os.path.join(out_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    cfg    = record.get("config", {})
    n_boot = int(cfg.get("bench_n_boot", 10000))
    seed   = int(cfg.get("seed", 42))
    margin = float(cfg.get("bench_noninf_margin_pp", 1.0))
    written: List[str] = []

    md = [
        "# ARS Benchmark Report", "",
        f"- created: {record.get('created', '-')} | git: {record.get('git_commit', '-')}",
        f"- model: {record.get('model_name', '-')}",
        "",
    ]
    c = record.get("contract", {})
    if c:
        md += [
            "## Deployment contract", "",
            f"- s*_fixed: `{c.get('s_fixed_star')}` | theta*_abs: `{c.get('theta_abs')}`",
            f"- gate adopted: `{c.get('gate_adopted')}` | envelope cap: `{c.get('rel_norm_cap')}`",
            "",
        ]

    for section, blocks in (("primary", record.get("shots", {})), ("secondary", record.get("secondary", {}))):
        for k, blk in sorted(blocks.items(), key=lambda kv: int(kv[0])):
            tables = build_tables(blk, n_boot, seed)
            md    += [f"## {section.capitalize()}: {k}-shot (n={blk['n']})", ""]
            stats  = ensure_stats(blk, n_boot, seed, margin)

            for name, tbl in tables.items():
                fname    = f"{section}_{k}shot_{name}"
                tex_path = os.path.join(tdir, fname + ".tex")
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(to_latex(
                        tbl,
                        caption=f"{k}-shot {name.replace('_', ' ')} (n={blk['n']})",
                        label=f"tab:{fname}",
                    ))
                written.append(tex_path)
                md += [f"### {name.replace('_', ' ')}", "", to_markdown(tbl), ""]

            if "strict" in stats:
                md.append(f"Non-inferiority margin: {margin} pp (CI lower bound > -margin).")
                for a, s in stats["strict"].items():
                    md.append(
                        f"- {blk['arms'][a]['label']}: delta {s['delta_pp']:+.2f} pp, "
                        f"non-inferior={s.get('noninferior')}, significant (Holm)={s.get('significant')}"
                    )
                md.append("")

    md_path = os.path.join(out_dir, "REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    written.append(md_path)
    return written


def main():
    p = argparse.ArgumentParser(description="Build ARS report from experiment_record.json")
    p.add_argument("--record",  default="experiment_record.json")
    p.add_argument("--out_dir", default="report")
    a = p.parse_args()
    for path in write_report(load_record(a.record), a.out_dir):
        print("wrote", path)


if __name__ == "__main__":
    main()
