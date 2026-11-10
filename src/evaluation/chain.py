"""Reasoning chain auditing, failure taxonomy, and statistical tests."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


STOP_RE = re.compile(
    r"(?:Final answer|The answer is|So the result is|####)\s*[:$-]?\s*\$?[-+]?[\d,]+(?:\.\d+)?[^\n]*(?:\n|$)"
)


def stop_pattern() -> re.Pattern:
    """Return the complete-number, newline/end-terminated stop regex."""
    return STOP_RE


def is_numeric_prefix(pred: Any, gold: Any) -> bool:
    """Return True if pred is a strict leading numeric prefix of gold (e.g. 1 vs 1250)."""
    if pred is None or gold is None:
        return False
    pred_str = str(pred).replace(",", "").replace("$", "").strip()
    gold_str = str(gold).replace(",", "").replace("$", "").strip()
    if pred_str.endswith(".0"):
        pred_str = pred_str[:-2]
    if gold_str.endswith(".0"):
        gold_str = gold_str[:-2]
    if not pred_str or not gold_str:
        return False
    return gold_str.startswith(pred_str) and len(pred_str) < len(gold_str)


class _SafeMathEvaluator(ast.NodeVisitor):
    """Safely evaluate arithmetic AST expressions without eval()."""

    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else float("nan"),
        ast.FloorDiv: lambda a, b: a // b if b != 0 else float("nan"),
        ast.Mod: lambda a, b: a % b if b != 0 else float("nan"),
        ast.Pow: lambda a, b: a ** b if abs(b) <= 10 else float("nan"),
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: a,
    }

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value}")

    def visit_Num(self, node: ast.Num):  # Python 3.7 compatibility
        return float(node.n)

    def visit_BinOp(self, node: ast.BinOp):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Disallowed operator: {op_type}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.ALLOWED_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Disallowed operator: {op_type}")
        operand = self.visit(node.operand)
        return self.ALLOWED_OPERATORS[op_type](operand)

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def safe_eval_expr(expr_str: str) -> Optional[float]:
    """Safely evaluate a mathematical expression string using AST."""
    cleaned = expr_str.replace("×", "*").replace("÷", "/").replace(",", "").replace("$", "")
    # Replace standalone 'x' or 'X' between numbers as multiplication
    cleaned = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", cleaned)
    # Handle percentage: e.g. 25% -> (25 / 100)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", cleaned)
    cleaned = cleaned.strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
        val = _SafeMathEvaluator().visit(tree)
        return float(val) if math.isfinite(val) else None
    except Exception:
        return None


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_EQ_LINE_RE = re.compile(
    r"([0-9\$\s\+\-\*\/\(\)\.\,\^×÷xX%]+)\s*=\s*(\$?-?\d[\d,]*(?:\.\d+)?%?)"
)


def audit_chain(text: str, question: str = "") -> Dict[str, Any]:
    """Audit arithmetic calculations in a reasoning chain.

    Returns:
        Dict containing:
        - eaa: Exact Arithmetic Accuracy (correct equations / total equations)
        - car: Clean Arithmetic Rate (1.0 if >=1 eq and all correct, else 0.0)
        - total_equations: int
        - correct_equations: int
        - first_bad_equation_idx: Optional[int]
        - equations: list of equation details
        - operands: list of all numbers mentioned
        - ungrounded_operands: list of operands not in question or previous results
    """
    lines = text.split("\n")
    parsed_equations = []
    correct_count = 0
    first_bad_idx = None

    # Track numbers seen in the prompt / question
    question_nums = {float(m.replace(",", "")) for m in _NUM_RE.findall(question)}
    common_constants = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 24.0, 30.0, 60.0, 100.0, 365.0}
    known_grounded = set(question_nums) | common_constants
    ungrounded = []

    for line in lines:
        match = _EQ_LINE_RE.search(line)
        if not match:
            continue
        raw_lhs, raw_rhs = match.group(1), match.group(2)
        # Verify lhs has at least one operator
        if not re.search(r"[\+\-\*\/×÷xX\^]", raw_lhs):
            continue
        eval_lhs = safe_eval_expr(raw_lhs)
        rhs_str = raw_rhs.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            rhs_val = float(rhs_str)
        except ValueError:
            continue

        if eval_lhs is None:
            continue

        is_correct = abs(eval_lhs - rhs_val) < 1e-3 or (
            rhs_val != 0 and abs((eval_lhs - rhs_val) / rhs_val) < 1e-4
        )
        eq_idx = len(parsed_equations)
        if is_correct:
            correct_count += 1
            known_grounded.add(rhs_val)
        elif first_bad_idx is None:
            first_bad_idx = eq_idx

        # Check operands in raw_lhs
        lhs_operands = [float(m.replace(",", "")) for m in _NUM_RE.findall(raw_lhs)]
        for op in lhs_operands:
            if op not in known_grounded:
                ungrounded.append(op)
            else:
                known_grounded.add(op)

        parsed_equations.append(
            {
                "raw": line.strip(),
                "lhs": raw_lhs.strip(),
                "rhs": rhs_val,
                "evaluated": eval_lhs,
                "correct": is_correct,
            }
        )

    total_eq = len(parsed_equations)
    eaa = (correct_count / total_eq) if total_eq > 0 else 1.0
    car = 1.0 if (total_eq > 0 and correct_count == total_eq) else 0.0

    return {
        "eaa": float(eaa),
        "car": float(car),
        "total_equations": total_eq,
        "correct_equations": correct_count,
        "first_bad_equation_idx": first_bad_idx,
        "equations": parsed_equations,
        "ungrounded_operands": ungrounded,
    }


def gold_step_values(answer_text: str) -> List[float]:
    """Extract gold intermediate calculation values from GSM8K <<...>> annotations."""
    values = []
    for match in re.finditer(r"<<(?:[^=>]*=)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*>>", answer_text):
        try:
            val = float(match.group(1).replace(",", ""))
            values.append(val)
        except ValueError:
            pass
    return values


def gold_step_recall(text: str, gold_values: List[float]) -> float:
    """Fraction of gold intermediate values found in generated text (GSR)."""
    if not gold_values:
        return 1.0
    # Restrict to values >= 10 to avoid small integer coincidence
    filtered = [v for v in gold_values if abs(v) >= 10.0]
    if not filtered:
        filtered = gold_values
    gen_nums = set()
    for m in _NUM_RE.findall(text):
        try:
            gen_nums.add(float(m.replace(",", "")))
        except ValueError:
            pass
    found = sum(1 for v in filtered if v in gen_nums)
    return float(found / len(filtered))


def classify_outcome(
    res_or_baseline: Dict[str, Any],
    steered_or_gold: Union[Dict[str, Any], float, str],
    gold: Optional[Union[float, str]] = None,
    question: str = "",
) -> str:
    """Classify generation outcome according to the mutually exclusive failure taxonomy.

    Precedence order:
    1. TRUNC_NUMBER: answer_stop fired and predicted number is strict leading prefix of gold.
    2. LOOP_OR_RUNON: stopped by max_new_tokens or repeat_stop.
    3. NO_ANSWER: no extractable number.
    4. EXTRACTION: gold appears as verified intermediate result but different final answer reported.
    5. ARITHMETIC: at least one expr = value is false.
    6. UNGROUNDED_OPERAND: arithmetic is valid but ungrounded operand used.
    7. PLAN_ERROR: valid, grounded arithmetic that answers the wrong question.
    """
    if gold is None:
        # Called as classify_outcome(gen_res, gold, question=...)
        res = res_or_baseline
        gold_val = float(str(steered_or_gold).replace(",", "").replace("$", "").strip())
    else:
        # Called as classify_outcome(baseline_res, steered_res, gold)
        # Classify the steered generation if wrong, or baseline generation if steered repaired it
        b_res = res_or_baseline
        s_res = steered_or_gold if isinstance(steered_or_gold, dict) else {}
        gold_val = float(str(gold).replace(",", "").replace("$", "").strip())
        res = s_res if not s_res.get("lenient", False) else b_res

    pred = res.get("pred_marker", res.get("pred", None))
    text = res.get("text", "")
    stop_reason = res.get("stop_reason", "")

    # Check if correct
    try:
        if pred is not None and abs(float(str(pred).replace(",", "").replace("$", "")) - gold_val) < 1e-4:
            return "CORRECT"
    except (ValueError, TypeError):
        pass

    # 1. TRUNC_NUMBER
    if stop_reason == "answer_stop" and is_numeric_prefix(pred, gold_val):
        return "TRUNC_NUMBER"

    # 2. LOOP_OR_RUNON — only when generation ran away AND no answer found
    no_answer = pred is None or (isinstance(pred, str) and not pred.strip())
    if stop_reason in {"max_new_tokens", "repeat_stop"} and no_answer:
        return "LOOP_OR_RUNON"

    # 3. NO_ANSWER
    if no_answer:
        return "NO_ANSWER"

    audit = audit_chain(text, question)

    # 4. EXTRACTION
    # Check if gold appears as a verified rhs/intermediate in equations
    intermediate_results = [eq["rhs"] for eq in audit["equations"] if eq["correct"]]
    if any(abs(v - gold_val) < 1e-4 for v in intermediate_results):
        return "EXTRACTION"

    # 5. ARITHMETIC
    if audit["total_equations"] > 0 and audit["first_bad_equation_idx"] is not None:
        return "ARITHMETIC"

    # 6. UNGROUNDED_OPERAND
    if audit["ungrounded_operands"]:
        return "UNGROUNDED_OPERAND"

    # 7. PLAN_ERROR
    return "PLAN_ERROR"


def paired_bootstrap_ci(
    baseline_flags: Sequence[bool],
    steered_flags: Sequence[bool],
    b: int = 10000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Compute point estimate of delta (pp) and 95% bootstrap confidence interval."""
    b_arr = np.array(baseline_flags, dtype=np.float64)
    s_arr = np.array(steered_flags, dtype=np.float64)
    n = len(b_arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    delta_pp = float(100.0 * (np.mean(s_arr) - np.mean(b_arr)))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(b, n))
    b_boot = b_arr[indices].mean(axis=1)
    s_boot = s_arr[indices].mean(axis=1)
    diffs = 100.0 * (s_boot - b_boot)

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    return delta_pp, ci_lo, ci_hi


def mcnemar_one_sided(wins: int, losses: int) -> float:
    """One-sided exact McNemar p-value (H0: wins <= losses vs H1: wins > losses)."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    # P(Binomial(n, 0.5) >= wins)
    tail = sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
    return min(1.0, float(tail / (2**discordant)))


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        mult = m - rank
        adj = min(1.0, mult * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted
