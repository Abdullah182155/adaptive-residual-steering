#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/verify_anchors.py -- TASK A0: Anchor Verification (ARS / RSC project)

Strictly READ-ONLY audit of the repository. It inspects, measures and reports;
it never modifies repository files and never aborts on findings.

Outputs (the only files written):
    docs/ANCHOR_REPORT.md
    docs/anchor_profile_dump.json

Usage:
    python scripts/verify_anchors.py --device cpu --tiny
    python scripts/verify_anchors.py --device cuda --ckpt checkpoints/steer_phase1_adaptive_v11.pt

Audit items:
    A0.1  Gate initialization audit
    A0.2  Operating point & steering-norm audit
    A0.3  Audit / preflight wiring analysis
    A0.4  Kaggle profile & config dump
    A0.5  Dropout in router counterfactuals
    A0.6  Fixed-arm topology verification
    A0.7  Config drift analysis
    A0.8  Evaluation splits & example counts
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the repo

import argparse
import ast
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAVE_TORCH = False

SCRIPT_VERSION = "A0.1.0"
SCRIPT_PATH = Path(__file__).resolve()
FENCE = "`" * 3

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "ENV", "node_modules", "build", "dist",
    ".idea", ".vscode", ".ipynb_checkpoints", "checkpoints", "wandb", "logs", "runs",
    "outputs", "results", "hf_cache", ".cache", "site-packages",
}
SCAN_EXTS = {".py", ".ipynb", ".md", ".json", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini"}

# Default policy for the "M0 replacement" column of A0.3 (see design notes in the task document).
M0_POLICY: Dict[str, Dict[str, str]] = {
    "audit_router_policy_health": {
        "replacement": "M0: run as read-only diagnostic; log + persist its health dict to the run report; "
                       "its verdict must NOT gate, skip or abort any phase.",
    },
    "skip_adaptive_phases_when_router_fails": {
        "replacement": "M0: flag is ignored / forced False -- Phases 2A-2D always run; router failure is recorded "
                       "as a warning in the report.",
    },
    "router_fail_fast_on_collapse": {
        "replacement": "M0: replace raise/abort with warning + counter; training continues; collapse metrics "
                       "are persisted.",
    },
    "continue_on_diagnostic_failure": {
        "replacement": "M0: hard-wire tolerant semantics (diagnostic failures are logged, never fatal); "
                       "keep field only for config backward-compat.",
    },
    "router_dead_layer_veto_frac": {
        "replacement": "M0: remove the checkpoint veto; compute dead-layer fraction as a reported diagnostic only "
                       "and still save accuracy-improving checkpoints. Declare the field in RSCConfig "
                       "(non-blocking default) to remove the AttributeError.",
    },
    "router_checkpoint_every_n_steps": {
        "replacement": "M0: declare in RSCConfig (default 0 = disabled) so the trainer never raises AttributeError.",
    },
    "router_k_cap_gap_summary": {
        "replacement": "M0: guarded by hasattr() today; either implement on the wrapper as read-only diagnostics "
                       "or leave as no-op.",
    },
}
PRIMARY_WIRING_SYMBOLS = [
    "audit_router_policy_health",
    "skip_adaptive_phases_when_router_fails",
    "router_fail_fast_on_collapse",
    "continue_on_diagnostic_failure",
    "router_dead_layer_veto_frac",
]
ADJACENT_WIRING_SYMBOLS = ["router_checkpoint_every_n_steps", "router_k_cap_gap_summary"]

SAMPLE_QA = [
    ("A baker makes 12 muffins per tray and bakes 5 trays. She sells 37 muffins. How many are left?",
     "The baker makes 12 * 5 = 60 muffins. After selling 37 she has 60 - 37 = 23 left.\n#### 23"),
    ("A train travels 45 miles each hour for 4 hours, then 30 miles more. How far did it travel in total?",
     "In 4 hours it travels 45 * 4 = 180 miles. Adding 30 gives 180 + 30 = 210 miles.\n#### 210"),
    ("Sam has $50. He buys 3 books at $9 each and a pen for $4. How much money does he have left?",
     "The books cost 3 * 9 = 27 dollars. Total spent is 27 + 4 = 31. Left: 50 - 31 = 19.\n#### 19"),
    ("There are 8 boxes with 6 pencils each. If 14 pencils are given away, how many pencils remain?",
     "There are 8 * 6 = 48 pencils. After giving away 14 there are 48 - 14 = 34.\n#### 34"),
]


# ----------------------------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[A0] {msg}", flush=True)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def md_cell(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(md_cell(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return out


def code_block(text: str, lang: str = "") -> List[str]:
    return [f"{FENCE}{lang}"] + str(text).rstrip().splitlines() + [FENCE]


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if x != 0.0 and (abs(x) < 10 ** (-nd) or abs(x) >= 1e6):
            return f"{x:.3e}"
        return f"{x:.{nd}f}"
    return str(x)


def jsonable(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else str(o)
    if isinstance(o, (set, frozenset)):
        return [jsonable(x) for x in sorted(o, key=str)]
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return jsonable(dataclasses.asdict(o))
    if HAVE_TORCH and torch.is_tensor(o):
        return jsonable(o.detach().cpu().tolist()) if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


@contextlib.contextmanager
def in_dir(path: Path):
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def origin_of(rel: str) -> str:
    if rel.startswith("reference_implementation/") or rel == "RSC_Training.py" or rel.endswith(".original.py"):
        return "reference"
    return "repo"


def iter_repo_files(root: Path, exts: Iterable[str], exclude: set) -> Iterable[Path]:
    exts = set(exts)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.suffix.lower() not in exts:
                continue
            try:
                if p.resolve() in exclude:
                    continue
            except OSError:
                continue
            yield p


# ----------------------------------------------------------------------------------------------
# Repo / AST helpers
# ----------------------------------------------------------------------------------------------
class PyFile:
    def __init__(self, path: Path, rel: str):
        self.path = path
        self.rel = rel
        self.text = path.read_text(encoding="utf-8", errors="replace")
        self.lines = self.text.splitlines()
        self.tree: Optional[ast.AST] = None
        self.error: Optional[str] = None
        self.parents: Dict[ast.AST, ast.AST] = {}
        try:
            self.tree = ast.parse(self.text)
        except SyntaxError as e:
            self.error = f"SyntaxError: {e}"
        if self.tree is not None:
            for parent in ast.walk(self.tree):
                for child in ast.iter_child_nodes(parent):
                    self.parents[child] = parent

    def segment(self, node: Optional[ast.AST], limit: int = 110) -> str:
        if node is None:
            return ""
        seg = ast.get_source_segment(self.text, node) or ""
        seg = " ".join(seg.split())
        return seg if len(seg) <= limit else seg[: limit - 3] + "..."

    def qualname_of(self, node: ast.AST) -> str:
        names: List[str] = []
        cur = self.parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(cur.name)
            cur = self.parents.get(cur)
        return ".".join(reversed(names)) or "<module>"

    def enclosing_func(self, node: ast.AST) -> Optional[ast.AST]:
        cur = self.parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = self.parents.get(cur)
        return None

    def gating(self, node: ast.AST) -> Optional[str]:
        child = node
        cur = self.parents.get(node)
        while cur is not None and not isinstance(
            cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)
        ):
            test = getattr(cur, "test", None)
            if isinstance(cur, (ast.If, ast.While, ast.IfExp, ast.Assert)) and child is test:
                effects: List[str] = []
                if isinstance(cur, ast.Assert):
                    effects.append("raises AssertionError")
                else:
                    body = getattr(cur, "body", [])
                    body = body if isinstance(body, list) else [body]
                    for st in body:
                        for sub in ast.walk(st):
                            if isinstance(sub, ast.Raise):
                                effects.append("raises")
                            elif isinstance(sub, ast.Return):
                                effects.append("returns early")
                            elif isinstance(sub, ast.Break):
                                effects.append("breaks loop")
                            elif isinstance(sub, ast.Continue):
                                effects.append("continues loop")
                            elif isinstance(sub, (ast.Assign, ast.AugAssign)):
                                tgts = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                                for t in tgts:
                                    if isinstance(t, ast.Name):
                                        effects.append(f"sets `{t.id}`")
                uniq: List[str] = []
                for e in effects:
                    if e not in uniq:
                        uniq.append(e)
                return (f"{type(cur).__name__.lower()}-gate `{self.segment(test, 90)}` -> "
                        + (", ".join(uniq) if uniq else "conditional branch"))
            child = cur
            cur = self.parents.get(cur)
        return None


class Repo:
    def __init__(self, root: Path, exclude: set):
        self.root = root
        self.exclude = exclude
        self._py: Dict[Path, PyFile] = {}
        self._files: Dict[Tuple[str, ...], List[Path]] = {}

    def rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    def files(self, exts: Iterable[str]) -> List[Path]:
        key = tuple(sorted(exts))
        if key not in self._files:
            self._files[key] = list(iter_repo_files(self.root, key, self.exclude))
        return self._files[key]

    def py(self, p: Path) -> PyFile:
        if p not in self._py:
            self._py[p] = PyFile(p, self.rel(p))
        return self._py[p]

    def text(self, p: Path) -> str:
        return p.read_text(encoding="utf-8", errors="replace")


def symbol_hits(repo: Repo, symbols: List[str], substr: bool = False, py_only: bool = False,
                rel_prefixes: Optional[Tuple[str, ...]] = None) -> List[Dict[str, Any]]:
    """Locate symbols in .py (AST-based) and other text files (regex-based)."""

    def match(s: str) -> Optional[str]:
        if substr:
            return next((x for x in symbols if x in s), None)
        return s if s in symbols else None

    hits: List[Dict[str, Any]] = []
    seen = set()
    for p in repo.files(SCAN_EXTS):
        rel = repo.rel(p)
        if rel_prefixes and not rel.startswith(rel_prefixes):
            continue
        handled = False
        if p.suffix == ".py":
            pf = repo.py(p)
            if pf.tree is not None:
                handled = True
                for node in ast.walk(pf.tree):
                    sym = None
                    kind = ""
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sym, kind = match(node.name), "def"
                    elif isinstance(node, ast.Name):
                        sym, kind = match(node.id), "name"
                        par = pf.parents.get(node)
                        if isinstance(par, ast.AnnAssign) and par.target is node:
                            kind = "field-def"
                        elif isinstance(par, ast.Assign) and node in par.targets:
                            kind = "assign"
                    elif isinstance(node, ast.Attribute):
                        sym, kind = match(node.attr), "attribute"
                        if sym and isinstance(getattr(node, "ctx", None), ast.Store):
                            kind = "attribute-store"
                    elif isinstance(node, ast.keyword) and node.arg:
                        sym, kind = match(node.arg), "kwarg"
                    elif isinstance(node, ast.arg):
                        sym, kind = match(node.arg), "param"
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 200:
                        sym, kind = match(node.value), "string"
                    if not sym:
                        continue
                    lineno = getattr(node, "lineno", None) or getattr(pf.parents.get(node), "lineno", 0) or 0
                    key = (rel, lineno, sym, kind)
                    if key in seen:
                        continue
                    seen.add(key)
                    func = pf.qualname_of(node)
                    if kind == "def":
                        func = (func + "." if func != "<module>" else "") + node.name  # type: ignore[attr-defined]
                    behavior = pf.gating(node) or {
                        "def": "definition", "field-def": "config field definition", "assign": "assignment",
                        "attribute": "attribute read", "attribute-store": "attribute write", "kwarg": "keyword argument",
                        "param": "function parameter", "string": "string key (getattr/dict/CLI)", "name": "name reference",
                    }.get(kind, kind)
                    hits.append({
                        "symbol": sym, "file": rel, "origin": origin_of(rel), "line": lineno, "kind": kind,
                        "func": func, "behavior": behavior,
                        "text": pf.lines[lineno - 1].strip()[:160] if 0 < lineno <= len(pf.lines) else "",
                    })
        if handled or py_only:
            continue
        text = repo.text(p)
        for i, line in enumerate(text.splitlines(), 1):
            sym = match(line)
            if sym:
                key = (rel, i, sym, "text-mention")
                if key in seen:
                    continue
                seen.add(key)
                hits.append({"symbol": sym, "file": rel, "origin": origin_of(rel), "line": i,
                             "kind": "text-mention", "func": "-", "behavior": "textual mention (non-AST)",
                             "text": line.strip()[:160]})
    hits.sort(key=lambda h: (h["symbol"], h["file"], h["line"]))
    return hits


def find_func(pf: PyFile, name: str) -> Optional[ast.AST]:
    if pf.tree is None:
        return None
    for node in ast.walk(pf.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def mode_calls(pf: PyFile, fn: ast.AST) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("train", "eval"):
            out.append((n.lineno, pf.segment(n, 60)))
        if isinstance(n, ast.With):
            for it in n.items:
                s = pf.segment(it.context_expr, 70)
                if "no_grad" in s or "inference_mode" in s:
                    out.append((n.lineno, "with " + s))
    return sorted(out)


# ----------------------------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class AuditResult:
    sid: str
    title: str
    status: str = "INFO"  # PASS | FINDING | WARN | INFO | N/A | ERROR
    headline: str = ""
    lines: List[str] = dataclasses.field(default_factory=list)
    data: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def add(self, *ls: str) -> None:
        self.lines.extend(ls)


class Ctx:
    pass


# ----------------------------------------------------------------------------------------------
# Context construction (config, model, checkpoint, batch)
# ----------------------------------------------------------------------------------------------
def build_tiny_backbone(hidden: int = 64, n_layers: int = 32, vocab: int = 128):
    class Block(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, h, *args, **kwargs):
            return (h + self.ff(self.ln(h)),)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([Block(hidden) for _ in range(n_layers)])

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=hidden, vocab_size=vocab)
            self.model = Inner()
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            h = self.model.embed(input_ids)
            for layer in self.model.layers:
                h = layer(h)[0]
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
            return types.SimpleNamespace(loss=loss, logits=logits)

    return TinyLM()


def discover_ckpt(ctx: Ctx) -> Optional[Path]:
    if ctx.args.ckpt:
        p = Path(ctx.args.ckpt)
        p = p if p.is_absolute() else (Path.cwd() / p)
        if p.exists():
            return p.resolve()
        alt = ctx.root / ctx.args.ckpt
        if alt.exists():
            return alt.resolve()
        ctx.notes.append(f"--ckpt path not found: {ctx.args.ckpt}")
        return None
    cands: List[Path] = []
    names = ["steer_phase1_adaptive_v11.pt"]
    if ctx.cfg is not None:
        names.insert(0, Path(ctx.cfg.lora_save_path).name)
    for d in (ctx.root / "checkpoints", ctx.root):
        for n in names:
            if (d / n).exists():
                cands.append(d / n)
        if d.exists():
            for pat in ("steer_phase1*.pt", "p1_best.pt", "p1_epoch*.pt"):
                cands.extend(sorted(d.glob(pat)))
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda c: c.stat().st_mtime).resolve()


def load_ckpt_state(path: Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "steer_state" in obj and isinstance(obj["steer_state"], dict):
        obj = obj["steer_state"]
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected checkpoint container type {type(obj).__name__}")
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def make_batch(ctx: Ctx):
    dev = ctx.device
    if ctx.tokenizer is None:
        g = torch.Generator().manual_seed(ctx.args.seed)
        B, T = 4, 24
        ids = torch.randint(1, ctx.vocab, (B, T), generator=g)
        mask = torch.ones(B, T, dtype=torch.long)
        mask[-1, -4:] = 0
        ids[mask == 0] = 0
        labels = ids.clone()
        labels[:, : T // 2] = -100
        labels[mask == 0] = -100
        return ids.to(dev), mask.to(dev), labels.to(dev)
    tok = ctx.tokenizer
    prompts = [f"Question: {q}\nAnswer: Let's think step by step.\n" for q, _ in SAMPLE_QA]
    fulls = [p + a for p, (_, a) in zip(prompts, SAMPLE_QA)]
    enc = tok(fulls, return_tensors="pt", padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    labels = ids.clone()
    for i, p in enumerate(prompts):
        plen = len(tok(p, add_special_tokens=False)["input_ids"])
        nz = mask[i].nonzero()
        start = int(nz[0]) if nz.numel() else 0
        labels[i, : start + plen] = -100
    labels[mask == 0] = -100
    return ids.to(dev), mask.to(dev), labels.to(dev)


def build_context(args: argparse.Namespace) -> Ctx:
    ctx = Ctx()
    ctx.args = args
    ctx.root = Path(args.repo_root).resolve()
    ctx.out_dir = Path(args.out_dir).resolve() if args.out_dir else ctx.root / "docs"
    ctx.notes = []
    ctx.tmp = Path(tempfile.mkdtemp(prefix="a0_verify_"))
    exclude = {SCRIPT_PATH, (ctx.out_dir / "ANCHOR_REPORT.md").resolve(),
               (ctx.out_dir / "anchor_profile_dump.json").resolve()}
    ctx.exclude = exclude
    ctx.repo = Repo(ctx.root, exclude)
    ctx.cfg = None
    ctx.RSCConfig = None
    ctx.model = None
    ctx.tokenizer = None
    ctx.batch = None
    ctx.tiny = bool(args.tiny)
    ctx.vocab = 128
    ctx.device = torch.device("cpu") if HAVE_TORCH else None
    ctx.ckpt_path = None
    ctx.ckpt_applied: Dict[str, Any] = {}
    ctx.steer_trained = False
    ctx.target_layers: List[int] = []
    ctx.hidden_dim = None
    ctx.n_backbone_layers = None

    if not HAVE_TORCH:
        ctx.notes.append("torch not importable: dynamic measurements are N/A; static audits still run.")
    if str(ctx.root) not in sys.path:
        sys.path.insert(0, str(ctx.root))
    try:
        from src.configs.config import RSCConfig  # type: ignore

        with in_dir(ctx.tmp):  # RSCConfig.__post_init__ calls os.makedirs -> keep it out of the repo
            ctx.cfg = RSCConfig()
        ctx.RSCConfig = RSCConfig
    except Exception as e:
        ctx.notes.append(f"Could not import src.configs.config.RSCConfig: {type(e).__name__}: {e}")

    if HAVE_TORCH and ctx.cfg is not None:
        try:
            build_model(ctx)
        except Exception as e:
            ctx.notes.append(f"Model construction failed: {type(e).__name__}: {e}")
            ctx.model = None
    if ctx.model is not None:
        ctx.batch = make_batch(ctx)
        ctx.ckpt_path = discover_ckpt(ctx)
        if ctx.ckpt_path is not None:
            try:
                apply_ckpt(ctx)
            except Exception as e:
                ctx.notes.append(f"Checkpoint {ctx.ckpt_path} could not be applied: {type(e).__name__}: {e}")
    else:
        ctx.ckpt_path = discover_ckpt(ctx) if HAVE_TORCH and ctx.cfg is not None else None
    return ctx


def build_model(ctx: Ctx) -> None:
    args = ctx.args
    from src.models.wrapper import Phi2WithRSC  # type: ignore

    if args.device == "cuda" and not torch.cuda.is_available():
        ctx.notes.append("--device cuda requested but CUDA unavailable: falling back to CPU.")
    ctx.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    base = None
    if not ctx.tiny:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = args.model_name or ctx.cfg.model_name
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=args.trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float16 if ctx.device.type == "cuda" else torch.float32
            base = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, trust_remote_code=args.trust_remote_code).to(ctx.device)
            ctx.tokenizer = tok
        except Exception as e:
            ctx.notes.append(f"Real backbone load failed ({type(e).__name__}: {e}); falling back to --tiny backbone.")
            ctx.tiny = True
            base = None
    if base is None:
        base = build_tiny_backbone(hidden=64, n_layers=32, vocab=ctx.vocab).to(ctx.device)
        ctx.tokenizer = None
    with in_dir(ctx.tmp):
        ctx.model = Phi2WithRSC(base, ctx.cfg, device=ctx.device)
    ctx.model.eval()
    ctx.target_layers = list(ctx.model.target_layers)
    ctx.hidden_dim = int(ctx.model.hidden_dim)
    ctx.n_backbone_layers = len(base.model.layers)


def apply_ckpt(ctx: Ctx) -> None:
    state = load_ckpt_state(ctx.ckpt_path)
    own = ctx.model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    skipped = [k for k in state if k not in ok]
    ctx.model.load_state_dict(ok, strict=False)
    n_steer = sum(1 for k in ok if "steer_nets" in k)
    ctx.ckpt_applied = {"path": str(ctx.ckpt_path), "tensors_in_ckpt": len(state), "applied": len(ok),
                        "skipped_shape_or_name": len(skipped), "steer_tensors_applied": n_steer}
    ctx.steer_trained = n_steer > 0
    if skipped and not ok:
        ctx.notes.append("Checkpoint tensors did not match the backbone dimensions (expected with --tiny); "
                         "SteerNet magnitudes below are therefore UNTRAINED/synthetic.")


# ----------------------------------------------------------------------------------------------
# A0.1 Gate initialization audit
# ----------------------------------------------------------------------------------------------
def audit_gate_init(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.1", "Gate Initialization Audit")
    cfg = ctx.cfg
    # ---- static: documentation / docstring claims vs. code
    claim_pats = [
        ("claims bias -6.0", r"-\s*6\.0"),
        ("claims alpha ~ 0.0025", r"0\.0025"),
        ("claims sigmoid(-6)", r"sigmoid\(\s*-6"),
        ("'earn activation' wording", r"earn(?:s|ed)?\s+activation"),
        ("init-intent comment", r"Preserve the already-trained"),
        ("gate_init_alpha reference", r"gate_init_alpha"),
    ]
    claim_rows: List[List[Any]] = []
    low_init_claim = False
    for p in ctx.repo.files({".md", ".py", ".ipynb"}):
        rel = ctx.repo.rel(p)
        if not (rel.endswith(".md") or rel in ("src/gating/gate.py", "src/configs/config.py") or rel.endswith(".ipynb")):
            continue
        for i, line in enumerate(ctx.repo.text(p).splitlines(), 1):
            for label, pat in claim_pats:
                if re.search(pat, line):
                    claim_rows.append([f"{rel}:{i}", label, line.strip()[:150]])
                    if label.startswith("claims") and (rel.endswith(".md") or rel.endswith("gate.py")):
                        low_init_claim = True
    r.add("### Documentation / docstring claims", "")
    r.add(*(md_table(["Location", "Pattern", "Line"], claim_rows) if claim_rows else ["_No init-related claims found._"]))
    r.add("")

    if cfg is None or not HAVE_TORCH:
        r.status, r.headline = "N/A", "Dynamic gate measurement unavailable (torch or RSCConfig missing)."
        return r
    try:
        from src.gating.gate import RSCUsefulnessGate  # type: ignore
    except ImportError:
        from src.steering.gate import RSCUsefulnessGate  # type: ignore

    torch.manual_seed(ctx.args.seed)
    d = 2560  # Phi-2 hidden size: gate is evaluated at the production width regardless of backbone
    gate = RSCUsefulnessGate(d, cfg).eval()
    h = torch.randn(4, 64, d)
    delta = torch.randn(4, 64, d) * 0.1
    gctx = torch.randn(4, 16)
    with torch.no_grad():
        a = gate(h, delta, global_context=gctx).float()
        a_scaled = gate(h * 50.0, delta * 5.0, global_context=gctx * 3.0).float()
    b2 = float(gate.fc2.bias.item())
    w2max = float(gate.fc2.weight.abs().max().item())
    expected = cfg.gate_init_alpha
    meas = {
        "gate_init_alpha_cfg": expected, "fc2_bias": b2, "fc2_bias_expected_logit": logit(expected),
        "sigmoid_bias": sigmoid(b2), "fc2_weight_absmax": w2max,
        "alpha_mean": float(a.mean()), "alpha_std": float(a.std()), "alpha_min": float(a.min()),
        "alpha_max": float(a.max()), "frac_alpha_gt_0.85": float((a > 0.85).float().mean()),
        "frac_alpha_gt_0.50": float((a > 0.50).float().mean()),
        "alpha_mean_under_10x_input_scale": float(a_scaled.mean()),
    }
    r.data["measured_init"] = meas
    r.add("### Fresh-initialised gate on random hidden states (d=2560, eval mode)", "")
    r.add(*md_table(["Quantity", "Value"], [[k, fmt(v, 5)] for k, v in meas.items()]))
    r.add("",
          f"* `cfg.gate_init_alpha = {expected}` -> `fc2.bias = logit({expected}) = {fmt(logit(expected), 4)}`; "
          f"`fc2.weight` is zero-initialised (|w|max = {fmt(w2max)}), so alpha is **input-independent at init** "
          f"(std across tokens = {fmt(meas['alpha_std'], 6)}).",
          f"* Fraction of tokens with alpha > 0.85 at init: **{100 * meas['frac_alpha_gt_0.85']:.1f}%** "
          "(note the contrastive/anti-saturation targets in `contrastive.py` / `trainer.py` use 0.85).")

    # ---- checkpoint inspection
    r.add("", "### Phase 1 checkpoint inspection", "")
    verdict = "NO_CHECKPOINT"
    if ctx.ckpt_path is None:
        r.add("_No Phase 1 checkpoint found (searched `--ckpt`, `cfg.lora_save_path`, `checkpoints/steer_phase1*.pt`, "
              "`p1_best.pt`). Gate state of a trained run cannot be verified here._")
    else:
        r.add(f"Checkpoint: `{ctx.ckpt_path}`")
        try:
            state = load_ckpt_state(ctx.ckpt_path)
            per_gate: Dict[int, Dict[str, Any]] = {}
            per_steer: Dict[int, Dict[str, Any]] = {}
            for k, v in state.items():
                m = re.search(r"(?:^|\.)gates\.(\d+)\.(.+)$", k)
                if m:
                    per_gate.setdefault(int(m.group(1)), {})[m.group(2)] = v
                m = re.search(r"(?:^|\.)steer_nets\.(\d+)\.(.+)$", k)
                if m:
                    per_steer.setdefault(int(m.group(1)), {})[m.group(2)] = v
            r.add(f"Tensors: {len(state)}; `gates.*` layers: {sorted(per_gate)}; `steer_nets.*` layers: {sorted(per_steer)}")
            rows = []
            states = []
            probe_rows = []
            for layer in sorted(per_gate):
                g = per_gate[layer]
                w, b = g.get("fc2.weight"), g.get("fc2.bias")
                if w is None or b is None:
                    rows.append([layer, "missing fc2", "", "", "INCOMPLETE"])
                    states.append("INCOMPLETE")
                    continue
                wmax = float(w.abs().max())
                bval = float(b.reshape(-1)[0])
                if wmax == 0.0 and abs(sigmoid(bval) - expected) < 1e-4:
                    st = "FRESH_INIT"
                elif wmax == 0.0:
                    st = "BIAS_ONLY_CHANGE"
                else:
                    st = "LOADED_TRAINED"
                states.append(st)
                rows.append([layer, fmt(wmax, 6), fmt(bval, 4), fmt(sigmoid(bval), 4), st])
                fc1 = g.get("fc1.weight")
                if fc1 is not None and fc1.shape[0] == cfg.gate_dim and len(probe_rows) < 8:
                    dd = (fc1.shape[1] - 2 - 16) // 2
                    try:
                        gp = RSCUsefulnessGate(dd, cfg)
                        gp.load_state_dict({k: v.float() for k, v in g.items()}, strict=False)
                        gp.eval()
                        with torch.no_grad():
                            ap = gp(torch.randn(2, 32, dd), torch.randn(2, 32, dd) * 0.1,
                                    global_context=torch.randn(2, 16)).float()
                        probe_rows.append([layer, dd, fmt(float(ap.mean()), 4), fmt(float(ap.std()), 4),
                                           f"{100 * float((ap > 0.85).float().mean()):.1f}%"])
                    except Exception as e:  # pragma: no cover
                        probe_rows.append([layer, dd, f"probe failed: {type(e).__name__}", "", ""])
            if rows:
                r.add(*md_table(["Gate layer", "|fc2.weight|max", "fc2.bias", "sigmoid(bias)", "Verdict"], rows))
                if all(s == "FRESH_INIT" for s in states):
                    verdict = "ALL_GATES_FRESH_INIT"
                elif any(s == "LOADED_TRAINED" for s in states):
                    verdict = "GATES_CONTAIN_TRAINED_WEIGHTS"
                else:
                    verdict = "MIXED_OR_BIAS_ONLY"
            else:
                verdict = "NO_GATE_KEYS"
                r.add("_Checkpoint contains no `gates.*` keys._")
            if probe_rows:
                r.add("", "Gate alpha on random hidden states, using the checkpoint's gate weights:", "")
                r.add(*md_table(["Layer", "d", "mean alpha", "std alpha", "% alpha>0.85"], probe_rows))
            if per_steer:
                srows = []
                for layer in sorted(per_steer):
                    s = per_steer[layer]
                    up = s.get("up.weight")
                    kp = s.get("k_proj.weight")
                    srows.append([layer, fmt(float(up.norm()), 4) if up is not None else "n/a",
                                  fmt(float(kp.abs().max()), 5) if kp is not None else "n/a"])
                r.add("", "SteerNet weight summary (trained SteerNets have larger ||up.weight||; k_proj starts at 0):", "")
                r.add(*md_table(["Layer", "||up.weight||_F", "|k_proj.weight|max"], srows))
            # explain the mechanism from the trainer source, if present
            tp = ctx.root / "src" / "training" / "trainer.py"
            if tp.exists():
                for i, line in enumerate(ctx.repo.text(tp).splitlines(), 1):
                    if '"steer_net" in k or "router" in k' in line:
                        r.add("", f"Mechanism (static): `trainer.py:{i}` stores the Phase-1 *best_state* WITHOUT `gate` keys, "
                                  "while `save_steer_weights` later persists every `steer_net|gate|router` tensor -> a Phase 1 "
                                  "file legitimately contains `gates.*` keys that are just the untouched constructor "
                                  "initialisation (gates are frozen at 1.0 during Phase 1).")
                        break
        except Exception as e:
            r.add(f"_Checkpoint inspection failed: {type(e).__name__}: {e}_")
            verdict = "CKPT_UNREADABLE"
    r.data["checkpoint_verdict"] = verdict
    r.data["doc_claims_low_init"] = low_init_claim

    mismatch = low_init_claim and meas["alpha_mean"] > 0.5
    r.add("", "### Verdict", "")
    r.add(f"* Measured init alpha = **{fmt(meas['alpha_mean'], 4)}** (config-driven: `sigmoid(logit(gate_init_alpha))`).")
    if mismatch:
        r.add("* **DOC/CODE MISMATCH:** documentation claims a conservative init (bias -6.0, alpha ~ 0.0025) but the code "
              f"initialises alpha at {expected}. Code behaviour is authoritative; fix the documentation (or the init).")
    r.add(f"* Phase 1 checkpoint gate state: **{verdict}**.")
    r.status = "FINDING" if mismatch else "PASS"
    r.headline = (f"alpha_init={fmt(meas['alpha_mean'], 3)} (>0.85: {100 * meas['frac_alpha_gt_0.85']:.0f}%); "
                  f"docs claim low-init: {low_init_claim}; ckpt gates: {verdict}")
    return r


# ----------------------------------------------------------------------------------------------
# A0.2 Operating point & steering norm
# ----------------------------------------------------------------------------------------------
def capture_hidden(ctx: Ctx, layers: List[int]) -> Dict[int, Any]:
    model = ctx.model
    ids, mask, _ = ctx.batch
    store: Dict[int, Any] = {}
    handles = []
    for idx in layers:
        def mk(i):
            def hook(module, args, output):
                hh = output[0] if isinstance(output, tuple) else output
                store[i] = hh.detach().float()
            return hook
        handles.append(model.base_model.model.layers[idx].register_forward_hook(mk(idx)))
    old = model._routing_override
    model.set_routing_override("none")  # hooks become identity -> raw backbone hidden states
    model.eval()
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
        model._routing_override = old
    return store


def tstats(x) -> Dict[str, float]:
    x = x.flatten().float().cpu()
    n = int(x.numel())
    if n == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan"), "n": 0}
    s = torch.sort(x).values
    return {"mean": float(x.mean()), "median": float(s[(n - 1) // 2]), "p95": float(s[int(0.95 * (n - 1))]),
            "max": float(s[-1]), "n": n}


def locate_scale_sites(ctx: Ctx) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    prefixes = ("src/", "scripts/", "configs/")
    steer_hits = symbol_hits(ctx.repo, ["steer_scale"], substr=True, rel_prefixes=prefixes)
    rows: List[List[Any]] = []
    for p in ctx.repo.files({".py"}):
        rel = ctx.repo.rel(p)
        if not rel.startswith(prefixes):
            continue
        pf = ctx.repo.py(p)
        if pf.tree is None:
            continue
        for node in ast.walk(pf.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and abs(node.value - 0.3) < 1e-12:
                line = pf.lines[node.lineno - 1].strip() if node.lineno - 1 < len(pf.lines) else ""
                role = "unclassified"
                for pat, lab in (("n_total", "candidate-layer window start fraction (NOT a steering scale)"),
                                 ("fewshot", "few-shot sampling probability (NOT a steering scale)"),
                                 ("layer_threshold", "gate anti-saturation threshold ramp (NOT a steering scale)"),
                                 ("lora|alpha|steer|delta", "POSSIBLE steering multiplier -- review")):
                    if re.search(pat, line):
                        role = lab
                        break
                rows.append([f"{rel}:{pf.qualname_of(node)}:{node.lineno}", line[:110], role])
    return rows, steer_hits


def audit_operating_point(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.2", "Operating Point & Steering Norm Audit")
    rows, steer_hits = locate_scale_sites(ctx)
    r.add("### Occurrences of `0.30` / `steer_scale` in `src/`, `scripts/`, `configs/`", "")
    r.add(*(md_table(["Site", "Line", "Role"], rows) if rows else ["_No `0.30` float literals found._"]))
    r.add("")
    if steer_hits:
        r.add(*md_table(["Site", "Kind", "Behavior", "Line"],
                        [[f"{h['file']}:{h['func']}:{h['line']}", h["kind"], h["behavior"], h["text"]] for h in steer_hits]))
    else:
        r.add("**`steer_scale` does not exist anywhere in the scanned tree.**")
    wp = ctx.root / "src" / "models" / "wrapper.py"
    if wp.exists():
        for i, line in enumerate(ctx.repo.text(wp).splitlines(), 1):
            if "h_new" in line and "alpha" in line and "delta" in line:
                r.add("", f"Injection site `wrapper.py:{i}`: `{line.strip()}` -> effective multiplier is "
                          "`route * alpha.detach() / sqrt(K)`; there is no separate global steering scale.")
                break
    if ctx.model is None:
        r.status, r.headline = "N/A", "No model available for norm measurement."
        return r

    args = ctx.args
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    ks = [int(x) for x in args.k_values.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    init_a = float(ctx.cfg.gate_init_alpha)
    all_alphas = sorted(set(alphas + [init_a]))
    cands = ctx.model.target_layers
    usable = [l for l in layers if l in cands]
    skipped = [l for l in layers if l not in cands]
    if skipped:
        r.add("", f"_Layers not in the candidate set {cands} were skipped: {skipped}_")
    store = capture_hidden(ctx, usable)
    vm = ctx.batch[1].bool()
    grid: Dict[str, Any] = {}
    rows2: List[List[Any]] = []
    worst_p95_at_1 = 0.0
    for l in usable:
        h = store[l]
        hn = h.norm(dim=-1).clamp_min(1e-6)
        sn = ctx.model.steer_nets[str(l)].eval()
        for K in ks:
            ak = torch.full((h.size(0),), K, device=h.device)
            with torch.no_grad():
                delta = sn(h, active_k=ak).float()
            cell: Dict[str, Any] = {}
            row = [l, K, fmt(float(hn[vm].mean()), 2)]
            for a in all_alphas:
                ratio = ((a * delta / math.sqrt(K)).norm(dim=-1) / hn)[vm]
                st = tstats(ratio)
                cell[f"alpha={a:g}"] = st
                row.append(f"{fmt(st['mean'], 5)} / {fmt(st['p95'], 5)}")
                if abs(a - 1.0) < 1e-9:
                    worst_p95_at_1 = max(worst_p95_at_1, st["p95"])
            grid[f"layer={l},K={K}"] = cell
            rows2.append(row)
    r.data["norm_grid"] = grid
    hdr = ["Layer", "K", "mean ||h||"] + [f"alpha={a:g}{' (init)' if abs(a - init_a) < 1e-9 and a not in alphas else ''}: mean / p95"
                                          for a in all_alphas]
    r.add("", "### Relative steering norm  ||alpha * delta / sqrt(K)|| / ||h||  (per valid token)", "")
    r.add(*md_table(hdr, rows2))
    untrained = not ctx.steer_trained
    r.add("")
    if untrained:
        r.add("> **SteerNets are UNTRAINED (no applicable checkpoint tensors).** The magnitudes above reflect the "
              "constructor init (xavier gain 0.02) and are NOT representative of a trained operating point. "
              "Re-run with `--ckpt <phase1.pt>` on the real backbone (`--device cuda`, without `--tiny`).")
    target = float(args.target_rel_norm)
    rec: Dict[str, Any] = {"target_rel_norm_p95": target, "worst_p95_at_alpha_1": worst_p95_at_1,
                           "steernets_trained": not untrained}
    if untrained or worst_p95_at_1 <= 0:
        rec["steer_scale_recommended"] = 0.30
        rec["basis"] = "PROVISIONAL (untrained SteerNets): reference operating point 0.30 retained"
    else:
        s_cap = min(1.0, max(0.05, target / worst_p95_at_1))
        s_cap_floor = math.floor(s_cap * 100) / 100.0
        rec["max_scale_within_budget"] = s_cap_floor
        rec["steer_scale_recommended"] = min(0.30, s_cap_floor)
        rec["p95_ratio_at_scale_0.30"] = 0.30 * worst_p95_at_1
        rec["basis"] = (f"largest scale s with s * worst_p95(alpha=1) <= {target}; recommended = min(0.30, s_max)")
    r.data["recommendation"] = rec
    r.add("### Recommended `steer_scale` operating point", "")
    r.add(*md_table(["Item", "Value"], [[k, fmt(v, 4)] for k, v in rec.items()]))
    r.add("", "Suggested wiring (for the implementing agent, NOT applied here): "
              "`h_new = h_fp + steer_scale * route * alpha.detach() * delta * mask / sqrt(K)`, with `steer_scale` "
              "added to `RSCConfig` and defaulting to the recommended value.")
    r.status = "WARN" if untrained else ("FINDING" if not steer_hits else "PASS")
    r.headline = (f"steer_scale knob present: {bool(steer_hits)}; recommended={fmt(rec['steer_scale_recommended'], 2)} "
                  f"({'provisional/untrained' if untrained else 'measured'}); worst p95 rel-norm@alpha=1: {fmt(worst_p95_at_1, 4)}")
    return r


# ----------------------------------------------------------------------------------------------
# A0.3 Audit & preflight wiring
# ----------------------------------------------------------------------------------------------
def audit_wiring(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.3", "Audit & Preflight Wiring Analysis")
    syms = PRIMARY_WIRING_SYMBOLS + ADJACENT_WIRING_SYMBOLS
    hits = symbol_hits(ctx.repo, syms)
    cfg_fields = parse_rsc_config(ctx)[0]
    rows: List[List[Any]] = []
    counts = {s: 0 for s in syms}
    for h in hits:
        counts[h["symbol"]] += 1
        pol = M0_POLICY.get(h["symbol"], {}).get("replacement", "M0: advisory/non-blocking")
        if h["kind"] == "field-def":
            pol = "M0: keep field for config compat with a non-blocking default."
        rows.append([f"{h['file']}:{h['func']}:{h['line']}" + (" (reference)" if h["origin"] == "reference" else ""),
                     h["symbol"], h["kind"], h["behavior"], pol])
    for s in syms:
        if counts[s] == 0:
            rows.append(["(none)", s, "-", "NOT FOUND in this snapshot (0 call sites)",
                         M0_POLICY.get(s, {}).get("replacement", "")])
    r.add("### Call-site table", "")
    r.add(*md_table(["Site (file:function:line)", "Symbol", "Kind", "Current behavior (AST-observed)", "M0 replacement"], rows))
    r.add("", "### Symbol summary", "")
    r.add(*md_table(["Symbol", "Sites", "Declared in RSCConfig"],
                    [[s, counts[s], s in cfg_fields] for s in syms]))
    undeclared_used = [s for s in syms if counts[s] > 0 and s not in cfg_fields
                       and any(h["symbol"] == s and h["kind"] in ("attribute", "string", "name", "kwarg") for h in hits)]
    if undeclared_used:
        r.add("", f"**Used but NOT declared in `RSCConfig`:** {undeclared_used} -> `AttributeError` when the guarded path "
                  "executes (hard `cfg.<field>` read).")
    r.data["call_sites"] = hits
    r.data["counts"] = counts
    r.data["m0_replacement_note"] = ("M0 replacement text is the A0 default policy (advisory, non-blocking); confirm vs "
                                     "IMPLEMENTATION_TASKS_SPEC.md section 3.")
    gating = [h for h in hits if "-gate" in h["behavior"] or h["kind"] == "text-mention" and False]
    r.status = "FINDING" if (gating or undeclared_used) else ("INFO" if hits else "WARN")
    r.headline = (f"{sum(1 for s in PRIMARY_WIRING_SYMBOLS if counts[s])} / {len(PRIMARY_WIRING_SYMBOLS)} primary symbols present; "
                  f"{len(gating)} flow-gating site(s); undeclared-but-used: {undeclared_used}")
    return r


# ----------------------------------------------------------------------------------------------
# A0.4 Profile & config dump
# ----------------------------------------------------------------------------------------------
def to_profile_dict(obj: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"value": repr(obj)}


def audit_profile_dump(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.4", "Kaggle Profile & Config Dump")
    names = ["ENGINE_V40_KAGGLE_PROFILE", "make_engine_v40_kaggle_config"]
    hits = symbol_hits(ctx.repo, names, py_only=False)
    defs = [h for h in hits if h["kind"] in ("def", "assign", "field-def")]
    r.add("### Profile symbol discovery", "")
    r.add(*(md_table(["Site", "Symbol", "Kind"], [[f"{h['file']}:{h['line']}", h["symbol"], h["kind"]] for h in hits])
            if hits else ["_Neither `ENGINE_V40_KAGGLE_PROFILE` nor `make_engine_v40_kaggle_config` exists in this snapshot._"]))
    discovery: Dict[str, Any] = {"status": "NOT_FOUND", "source": None, "error": None}
    profile: Dict[str, Any] = {}
    prof_obj = None
    ordered = sorted(defs, key=lambda h: (h["origin"] == "reference", h["file"]))
    for h in ordered:
        if not h["file"].endswith(".py"):
            continue
        mod_name = h["file"][:-3].replace("/", ".")
        try:
            importlib.invalidate_caches()
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, h["symbol"])
            if callable(obj) and h["kind"] == "def":
                with in_dir(ctx.tmp):
                    obj = obj()
            prof_obj = obj
            profile = to_profile_dict(obj)
            discovery = {"status": "LOADED_BY_IMPORT", "source": f"{h['file']}:{h['line']} ({h['symbol']})", "error": None}
            break
        except Exception as e:
            discovery = {"status": "IMPORT_FAILED", "source": f"{h['file']}:{h['line']}", "error": f"{type(e).__name__}: {e}"}
            try:  # AST literal fallback for a constant dict
                pf = ctx.repo.py(ctx.root / h["file"])
                for node in ast.walk(pf.tree):  # type: ignore[arg-type]
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == h["symbol"] for t in node.targets):
                        profile = ast.literal_eval(node.value)
                        discovery["status"] = "LOADED_BY_AST_LITERAL"
                        break
            except Exception:
                pass
            if profile:
                break
    fallback = False
    if not profile and ctx.cfg is not None:
        profile = dataclasses.asdict(ctx.cfg)
        prof_obj = ctx.cfg
        fallback = True
        discovery["fallback"] = "RSCConfig() defaults used as the profile (named profile absent/unloadable)"
    r.add("", f"Profile status: **{discovery['status']}**" + (" -> falling back to `RSCConfig()` defaults" if fallback else ""))
    if discovery.get("error"):
        r.add(f"Import error: `{discovery['error']}`")

    flat = flatten(jsonable(profile)) if profile else {}
    dropout = {k: v for k, v in flat.items() if "drop" in k.lower()}
    loss_w = {k: v for k, v in flat.items()
              if re.match(r"(.*\.)?(lambda_|beta_)", k) or k.lower().endswith("_weight") or k.lower().endswith("lambda_cost")}
    flags = {k: v for k, v in flat.items() if isinstance(v, bool) or k.endswith("_mode") or k in ("pg_mode",)}
    cand_layers: Dict[str, Any] = {}
    if ctx.model is not None:
        cand_layers["live_wrapper_target_layers"] = list(ctx.model.target_layers)
        cand_layers["backbone_layers"] = ctx.n_backbone_layers
        cand_layers["max_active_layers"] = ctx.cfg.max_active_layers
    if prof_obj is not None and hasattr(prof_obj, "rsc_layers_for"):
        for n in (32,):
            try:
                cand_layers[f"profile.rsc_layers_for({n})"] = prof_obj.rsc_layers_for(n)
            except Exception as e:
                cand_layers[f"profile.rsc_layers_for({n})"] = f"error: {e}"
    cand_layers.update({k: v for k, v in flat.items() if "layer" in k.lower()})
    mod_dropout: Dict[str, Any] = {}
    if ctx.model is not None:
        for name, m in ctx.model.named_modules():
            if isinstance(m, nn.Dropout):
                top = name.split(".")[0]
                mod_dropout.setdefault(top, []).append(float(m.p))
        mod_dropout = {k: {"n_modules": len(v), "p_values": sorted(set(v))} for k, v in mod_dropout.items()}
    print("\n[A0.4] Active candidate layers :", cand_layers)
    print("[A0.4] Dropout (config)        :", dropout)
    print("[A0.4] Dropout (live modules)  :", mod_dropout)
    print("[A0.4] Loss weights            :", loss_w)
    print("[A0.4] Flags                   :", flags, "\n")

    r.add("", "### Active candidate layers", "", *md_table(["Key", "Value"], [[k, v] for k, v in cand_layers.items()]))
    r.add("", "### Dropout (config fields)", "", *md_table(["Key", "Value"], [[k, v] for k, v in dropout.items()]))
    if mod_dropout:
        r.add("", "### Dropout (instantiated modules, by top-level container)", "",
              *md_table(["Container", "n Dropout modules", "p values"],
                        [[k, v["n_modules"], v["p_values"]] for k, v in mod_dropout.items()]))
    r.add("", "### Loss weights", "", *md_table(["Key", "Value"], [[k, v] for k, v in loss_w.items()]))
    r.add("", "### Flags", "", *md_table(["Key", "Value"], [[k, v] for k, v in flags.items()]))
    r.data = {
        "profile_discovery": discovery, "profile": jsonable(profile),
        "rsc_config_defaults": jsonable(dataclasses.asdict(ctx.cfg)) if ctx.cfg is not None else None,
        "active_candidate_layers": jsonable(cand_layers), "dropout_config": jsonable(dropout),
        "dropout_modules": jsonable(mod_dropout), "loss_weights": jsonable(loss_w), "flags": jsonable(flags),
    }
    r.status = "WARN" if discovery["status"] in ("NOT_FOUND", "IMPORT_FAILED") else "PASS"
    r.headline = f"profile {discovery['status']}; candidate layers={cand_layers.get('live_wrapper_target_layers')}"
    return r


# ----------------------------------------------------------------------------------------------
# A0.5 Dropout in router counterfactuals
# ----------------------------------------------------------------------------------------------
def audit_router_dropout(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.5", "Dropout in Router Counterfactuals")
    findings_train_mode = None
    # ---- static
    r.add("### Static analysis (train/eval mode around counterfactual forwards)", "")
    trows: List[List[Any]] = []
    spec = [("src/training/trainer.py", "train_router_bootstrap"), ("src/training/trainer.py", "run_staged_phase"),
            ("src/routing/rloo.py", "compute_router_rloo_loss")]
    info: Dict[str, Any] = {}
    for rel, fname in spec:
        p = ctx.root / rel
        if not p.exists():
            trows.append([rel, fname, "file missing", ""])
            continue
        pf = ctx.repo.py(p)
        fn = find_func(pf, fname)
        if fn is None:
            trows.append([rel, fname, "function missing", ""])
            continue
        calls = mode_calls(pf, fn)
        info[fname] = calls
        trows.append([rel, fname, "; ".join(f"L{ln}: {t}" for ln, t in calls) or "no train()/eval()/no_grad found", ""])
    r.add(*md_table(["File", "Function", "train()/eval()/no_grad sites", ""], trows))
    rloo_calls = info.get("compute_router_rloo_loss", [])
    boot_calls = info.get("train_router_bootstrap", [])
    has_eval_rloo = any(".eval" in t for _, t in rloo_calls)
    has_train_boot = any(".train" in t for _, t in boot_calls)
    if "compute_router_rloo_loss" in info:
        findings_train_mode = (not has_eval_rloo) and (has_train_boot or True)
        r.add("", "* `compute_router_rloo_loss` never calls `.eval()`; its counterfactual forwards run under `torch.no_grad()` "
                  "(which disables autograd, **not dropout**) in whatever mode the caller left the model. "
                  + ("`train_router_bootstrap` calls `model.train()` at every epoch start, so the counterfactual NLLs are "
                     "computed with dropout ACTIVE." if has_train_boot else ""))
    r.data["static"] = {"counterfactuals_in_train_mode": findings_train_mode,
                        "sites": {k: [[ln, t] for ln, t in v] for k, v in info.items()}}

    # ---- module inventory
    mod_rows: List[List[Any]] = []
    if ctx.model is not None:
        groups: Dict[str, List[float]] = {}
        for name, m in ctx.model.named_modules():
            if isinstance(m, nn.Dropout):
                groups.setdefault(name.split(".")[0], []).append(float(m.p))
        for k, v in sorted(groups.items()):
            mod_rows.append([k, len(v), sorted(set(v)), "ACTIVE in train()" if any(x > 0 for x in v) else "inactive (p=0)"])
        bc = getattr(ctx.model.base_model, "config", None)
        for key in ("attention_dropout", "resid_pdrop", "embd_pdrop", "hidden_dropout"):
            if bc is not None and hasattr(bc, key):
                mod_rows.append([f"base_model.config.{key}", "-", getattr(bc, key), "config value"])
        r.add("", "### Dropout inventory", "", *md_table(["Container", "n", "p", "Note"], mod_rows))

    # ---- dynamic noise measurement
    if ctx.model is None or ctx.batch is None:
        r.status = "FINDING" if findings_train_mode else "N/A"
        r.headline = f"static: counterfactuals in train mode = {findings_train_mode}; dynamic N/A"
        return r
    model = ctx.model
    ids, mask, labels = ctx.batch
    try:
        from src.routing.rloo import per_example_answer_nll  # type: ignore
    except Exception:
        def per_example_answer_nll(logits, labels):  # type: ignore
            sl = logits[:, :-1, :].float()
            tl = labels[:, 1:]
            nll = torch.nn.functional.cross_entropy(sl.transpose(1, 2), tl, ignore_index=-100, reduction="none")
            v = tl.ne(-100)
            return (nll * v).sum(1) / v.sum(1).clamp(min=1)
    ncand = len(model.target_layers)
    prompt_mask = mask.bool() & labels.eq(-100)
    A = torch.zeros(ids.size(0), ncand, dtype=torch.bool, device=ids.device)
    B = torch.zeros_like(A)
    A[:, 0::2] = True
    B[:, 1::2] = True
    saved_override, saved_training = model._routing_override, model.training
    saved_freeze = [(g, g._freeze_gate, g._freeze_gate_value) for g in model.gates.values()]
    amp_state = None
    if ctx.tiny and not ctx.steer_trained:
        amp_state = [(p, p.data.clone()) for sn in model.steer_nets.values() for p in sn.parameters()]
        for sn in model.steer_nets.values():
            for lin in (sn.down1, sn.mid, sn.up):
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
    reps = 8

    def run(sel, train_mode: bool):
        outs = []
        model.train(train_mode)
        model.set_routing_override("matrix", sel)
        for _ in range(reps):
            model.set_router_context_mask(prompt_mask)
            with torch.no_grad():
                outs.append(per_example_answer_nll(model(input_ids=ids, attention_mask=mask).logits, labels).float().cpu())
        return torch.stack(outs)

    try:
        model.set_gate_freeze(True, 1.0)  # same as train_router_bootstrap
        eval_runs = run(A, False)
        train_runs = run(A, True)
        eval_B = run(B, False)
        noise_eval = float(eval_runs.std(0).max())
        noise_train = float(train_runs.std(0).max())
        signal = float((eval_runs.mean(0) - eval_B.mean(0)).abs().max())
        dyn = {"noise_std_eval_mode": noise_eval, "noise_std_train_mode": noise_train,
               "counterfactual_signal_|nll_A-nll_B|(eval)": signal,
               "snr_train_mode": signal / noise_train if noise_train > 0 else float("inf"),
               "amplified_synthetic_steernets": amp_state is not None, "repeats": reps}
        r.data["dynamic"] = dyn
        r.add("", "### Empirical dropout noise in counterfactual NLL (fixed subset, identical input, 8 repeats)", "",
              *md_table(["Quantity", "Value"], [[k, fmt(v, 6)] for k, v in dyn.items()]))
        if amp_state is not None:
            r.add("", "_Tiny mode: SteerNet weights were temporarily re-initialised with gain 0.5 purely so the dropout effect "
                      "is observable; originals are restored afterwards. Magnitudes are synthetic; the eval-vs-train contrast is "
                      "what matters._")
        dyn_flag = noise_train > 0.0 and noise_eval == 0.0
        r.add("", f"* eval-mode repeat std = {fmt(noise_eval, 8)} (deterministic); train-mode repeat std = {fmt(noise_train, 8)}.")
    except Exception as e:
        dyn_flag = False
        r.add(f"_Dynamic noise test failed: {type(e).__name__}: {e}_")
    finally:
        if amp_state is not None:
            for p, d0 in amp_state:
                p.data.copy_(d0)
        for g, fz, fv in saved_freeze:
            g._freeze_gate, g._freeze_gate_value = fz, fv
        model._routing_override = saved_override
        model.train(saved_training)
    r.status = "FINDING" if (findings_train_mode or dyn_flag) else "PASS"
    r.headline = (f"counterfactuals run in train mode: {findings_train_mode}; measured dropout noise std="
                  f"{fmt(r.data.get('dynamic', {}).get('noise_std_train_mode'), 6)} vs eval "
                  f"{fmt(r.data.get('dynamic', {}).get('noise_std_eval_mode'), 6)}")
    return r


# ----------------------------------------------------------------------------------------------
# A0.6 Fixed-arm topology
# ----------------------------------------------------------------------------------------------
def audit_fixed_arm(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.6", "Fixed-Arm Topology Verification")
    issues = 0
    # ---- allowed modes parsed from the wrapper
    allowed = {"only", "exclude", "none", "force", "matrix"}
    wp = ctx.root / "src" / "models" / "wrapper.py"
    if wp.exists():
        pf = ctx.repo.py(wp)
        fn = find_func(pf, "set_routing_override")
        if fn is not None:
            for n in ast.walk(fn):
                if isinstance(n, ast.Compare) and any(isinstance(o, ast.NotIn) for o in n.ops):
                    for c in n.comparators:
                        if isinstance(c, ast.Set):
                            try:
                                allowed = {ast.literal_eval(e) for e in c.elts}
                            except Exception:
                                pass
    r.add(f"`set_routing_override` accepts modes: `{sorted(allowed)}`", "")

    # ---- call sites with literal modes (all text files)
    pat = re.compile(r"set_routing_override\(\s*(?:mode\s*=\s*)?([\"'])(\w+)\1")
    crow: List[List[Any]] = []
    for p in ctx.repo.files({".py", ".ipynb", ".md"}):
        rel = ctx.repo.rel(p)
        for i, line in enumerate(ctx.repo.text(p).splitlines(), 1):
            m = pat.search(line)
            if m:
                bad = m.group(2) not in allowed
                issues += int(bad and origin_of(rel) == "repo")
                crow.append([f"{rel}:{i}", m.group(2), "**WILL RAISE ValueError**" if bad else "ok", line.strip()[:100]])
    r.add("### Literal `set_routing_override(...)` call sites", "")
    r.add(*(md_table(["Site", "Mode", "Validity vs wrapper", "Line"], crow) if crow else ["_none_"]))

    # ---- fixed_steernet in multi_benchmark_runner.py
    runners = [p for p in ctx.repo.files({".py"}) if p.name == "multi_benchmark_runner.py"]
    r.add("", "### `fixed_steernet` arm in `multi_benchmark_runner.py`", "")
    if not runners:
        r.add("**`multi_benchmark_runner.py` is not present in this snapshot**; the arm definition could not be inspected "
              "statically. Dynamic topology of every override mode is measured below instead.")
    for p in runners:
        pf = ctx.repo.py(p)
        rel = ctx.repo.rel(p)
        hh = symbol_hits(ctx.repo, ["fixed_steernet"], substr=True, rel_prefixes=(rel,))
        r.add(f"`{rel}`: {len(hh)} reference(s) to `fixed_steernet`")
        for h in hh[:12]:
            r.add(f"* `{h['file']}:{h['line']}` ({h['func']}, {h['kind']}): `{h['text']}`")
            if pf.tree is None:
                continue
            node = next((n for n in ast.walk(pf.tree) if getattr(n, "lineno", -1) == h["line"]), None)
            fn = pf.enclosing_func(node) if node is not None else None
            if fn is not None:
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
                            "set_routing_override", "set_gate_freeze", "set_router_warmup", "set_router_inference_mode"):
                        r.add(f"  * in `{pf.qualname_of(n)}` L{n.lineno}: `{pf.segment(n, 100)}`")
                    if isinstance(n, (ast.Assign, ast.AugAssign)):
                        tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                        for t in tg:
                            if isinstance(t, ast.Attribute) and t.attr == "_desired_k":
                                r.add(f"  * direct `_desired_k` write L{n.lineno}: `{pf.segment(n, 100)}`")
    dk_writes = [h for h in symbol_hits(ctx.repo, ["_desired_k"], py_only=True)
                 if h["kind"] == "attribute-store" and not h["file"].endswith("models/wrapper.py")]
    if dk_writes:
        r.add("", "`_desired_k` is written outside the wrapper at: " + ", ".join(f"{h['file']}:{h['line']}" for h in dk_writes))

    # ---- steernet K normalisation constant
    sp = ctx.root / "src" / "steering" / "steernet.py"
    if sp.exists():
        for i, line in enumerate(ctx.repo.text(sp).splitlines(), 1):
            if re.search(r"active_k.*/\s*4\.0", line):
                r.add("", f"SteerNet normalises `active_k` by a hard-coded constant: `steernet.py:{i}` `{line.strip()}` "
                          "(K-conditioning is only in-range for K <= 4).")
                break

    # ---- dynamic scenarios
    if ctx.model is None or ctx.batch is None:
        r.status = "FINDING" if issues else "N/A"
        r.headline = f"static only; invalid literal override call sites={issues}"
        return r
    model = ctx.model
    cands = model.target_layers
    Kmax = ctx.cfg.max_active_layers
    ids, mask, _ = ctx.batch
    B = min(2, ids.size(0))
    ids, mask = ids[:B], mask[:B]
    five = [l for l in (13, 15, 18, 20, 22) if l in cands] or cands[:5]
    mat = torch.zeros(B, len(cands), dtype=torch.bool, device=ids.device)
    mat[:, : min(5, len(cands))] = True
    scen: List[Tuple[str, Optional[str], Any]] = [
        ("none (Stage A baseline)", "none", set()),
        (f"only {{{cands[0]}}}", "only", {cands[0]}),
        (f"only {five} (5 layers)", "only", set(five)),
        (f"only ALL {len(cands)} candidates", "only", set(cands)),
        (f"force {{{cands[0]}}}", "force", {cands[0]}),
        (f"exclude {{{cands[0]}}}", "exclude", {cands[0]}),
        ("matrix: first 5 candidates", "matrix", mat),
        ("mode 'all' (as used by notebook/run_experiment)", "all", set()),
        ("learned router (override=None, argmax)", None, None),
    ]
    saved = (model._routing_override, model.training, model._router_inference_mode)
    model.eval()
    rows: List[List[Any]] = []
    data_rows: List[Dict[str, Any]] = []
    for name, mode, layers in scen:
        row: Dict[str, Any] = {"scenario": name}
        try:
            if mode is None:
                model.set_routing_override(None)
            else:
                model.set_routing_override(mode, layers)
            model.set_router_context_mask(None)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
            routes = model._precomputed_routes
            sel = [cands[i] for i in range(len(cands)) if bool(routes[0, i])]
            dk = int(model._desired_k[0])
            ak = int(model._active_k_realized[0])
            row.update({"realized_layers": sel, "n_realized": len(sel), "desired_k": dk, "active_k_realized": ak,
                        "k_scale_sqrt": math.sqrt(ak), "steernet_k_norm": ak / 4.0})
            flags = []
            if len(sel) > Kmax:
                flags.append(f"EXCEEDS K_max={Kmax}")
                issues += 1
            if sel and dk != len(sel):
                flags.append(f"desired_k({dk}) != realized({len(sel)})")
                issues += 1
            if sel and ak / 4.0 > 1.0:
                flags.append("K-conditioning out of [0,1] range")
            row["flags"] = flags
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if mode == "all":
                issues += 1
        data_rows.append(row)
        rows.append([name, row.get("realized_layers", "-"), row.get("n_realized", "-"), row.get("desired_k", "-"),
                     row.get("active_k_realized", "-"), fmt(row.get("k_scale_sqrt"), 3) if "k_scale_sqrt" in row else "-",
                     row.get("error") or "; ".join(row.get("flags", [])) or "ok"])
    model._routing_override = saved[0]
    model.train(saved[1])
    model._router_inference_mode = saved[2]
    r.data["scenarios"] = data_rows
    r.add("", f"### Measured topology per override mode (candidates={cands}, `max_active_layers`={Kmax})", "")
    r.add(*md_table(["Scenario", "Realized layers", "n", "_desired_k", "_active_k_realized", "sqrt(K) used", "Flags / error"], rows))
    r.add("", "Interpretation: `_desired_k` is clamped to `max_active_layers` while `_active_k_realized` (used for `1/sqrt(K)` "
              "scaling and SteerNet K-conditioning) is the *unclamped* realized count for `only` / `matrix` / `force`; only "
              "`exclude` goes through `_finalize_selection`. A fixed arm with more than K_max layers therefore gets a different "
              "scale and an out-of-range K-embedding than anything seen in training.")
    r.status = "FINDING" if issues else "PASS"
    r.headline = f"{issues} topology issue(s) (invalid override modes, K_max violations, desired_k mismatches); runner present: {bool(runners)}"
    return r


# ----------------------------------------------------------------------------------------------
# A0.7 Config drift
# ----------------------------------------------------------------------------------------------
CFG_NAMES = {"cfg", "CFG"}


def is_cfg_expr(n: ast.AST) -> bool:
    return (isinstance(n, ast.Name) and n.id in CFG_NAMES) or (isinstance(n, ast.Attribute) and n.attr == "cfg")


def parse_rsc_config(ctx: Ctx) -> Tuple[Dict[str, Any], set]:
    fields: Dict[str, Any] = {}
    members: set = set()
    p = ctx.root / "src" / "configs" / "config.py"
    if p.exists():
        pf = ctx.repo.py(p)
        if pf.tree is not None:
            for node in ast.walk(pf.tree):
                if isinstance(node, ast.ClassDef) and node.name == "RSCConfig":
                    for st in node.body:
                        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                            fields[st.target.id] = pf.segment(st.value, 60) if st.value is not None else None
                            members.add(st.target.id)
                        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            members.add(st.name)
    if ctx.RSCConfig is not None:
        members |= {n for n in dir(ctx.RSCConfig) if not n.startswith("__")}
        try:
            members |= {f.name for f in dataclasses.fields(ctx.RSCConfig)}
        except TypeError:
            pass
    return fields, members


def collect_cfg_accesses(pf: PyFile) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if pf.tree is None:
        return out
    for node in ast.walk(pf.tree):
        if isinstance(node, ast.Attribute) and is_cfg_expr(node.value):
            mode = "set" if isinstance(getattr(node, "ctx", None), ast.Store) else "hard"
            out.append({"field": node.attr, "mode": mode, "line": node.lineno, "func": pf.qualname_of(node), "file": pf.rel})
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr")
              and len(node.args) >= 2 and is_cfg_expr(node.args[0])
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            soft = node.func.id == "getattr" and len(node.args) >= 3
            out.append({"field": node.args[1].value, "mode": "soft(default)" if soft else
                        ("probe(hasattr)" if node.func.id == "hasattr" else "hard(getattr)"),
                        "line": node.lineno, "func": pf.qualname_of(node), "file": pf.rel,
                        "default": pf.segment(node.args[2], 30) if soft else None})
    return out


def audit_config_drift(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.7", "Config Drift Analysis")
    fields, members = parse_rsc_config(ctx)
    if not members:
        r.status, r.headline = "N/A", "Could not parse RSCConfig."
        return r
    primary: List[Dict[str, Any]] = []
    secondary: List[Dict[str, Any]] = []
    for p in ctx.repo.files({".py"}):
        rel = ctx.repo.rel(p)
        if origin_of(rel) == "reference" or not (rel.startswith("src/") or rel.startswith("scripts/")):
            continue
        acc = collect_cfg_accesses(ctx.repo.py(p))
        (primary if rel.startswith("src/training/") else secondary).extend(acc)
    nb_sets: Dict[str, set] = {}
    for p in ctx.repo.files({".ipynb"}):
        for m in re.finditer(r"\bC(?:FG|fg)\.(\w+)\s*=(?!=)", ctx.repo.text(p)):
            nb_sets.setdefault(m.group(1), set()).add(ctx.repo.rel(p))
    for a in secondary:
        if a["mode"] == "set":
            nb_sets.setdefault(a["field"], set()).add(a["file"])

    def missing_rows(accs: List[Dict[str, Any]]) -> List[List[Any]]:
        by_field: Dict[str, List[Dict[str, Any]]] = {}
        for a in accs:
            if a["field"] not in members and a["mode"] != "set":
                by_field.setdefault(a["field"], []).append(a)
        rows = []
        for f, lst in sorted(by_field.items()):
            first = lst[0]
            modes = sorted({x["mode"] for x in lst})
            rows.append([f, "/".join(modes), len(lst), f"{first['file']}:{first['func']}:{first['line']}",
                         ", ".join(sorted(nb_sets.get(f, []))) or "-"])
        return rows

    p_rows = missing_rows(primary)
    s_rows = missing_rows(secondary)
    hdr = ["Field", "Access mode", "#uses", "First use (file:function:line)", "Assigned dynamically in"]
    r.add(f"`RSCConfig` declares {len(fields)} dataclass fields (+ {len(members) - len(fields)} other members).", "")
    r.add("### Missing fields accessed in `src/training/` (primary scope)", "")
    r.add(*(md_table(hdr, p_rows) if p_rows else ["_None._"]))
    r.add("", "### Missing fields accessed elsewhere in `src/` / `scripts/` (informational)", "")
    r.add(*(md_table(hdr, s_rows) if s_rows else ["_None._"]))
    hard_primary = [row for row in p_rows if "hard" in row[1]]
    r.add("")
    if hard_primary:
        r.add("**Hard (`cfg.<field>`) reads of undeclared fields raise `AttributeError` unless the field was assigned "
              "dynamically beforehand (the Kaggle notebook assigns only a subset).** Affected runtime path: "
              + ", ".join(sorted({row[3].split(":")[1] for row in hard_primary})) + ".")
    all_acc = primary + secondary
    used = {a["field"] for a in all_acc}
    for p in ctx.repo.files({".py"}):
        rel = ctx.repo.rel(p)
        if rel.startswith("src/") and origin_of(rel) == "repo":
            for m in re.finditer(r"\b(?:cfg|CFG|self\.cfg)\.(\w+)", ctx.repo.text(p)):
                used.add(m.group(1))
    unused = sorted(f for f in fields if f not in used)
    r.add("", "### Declared but never read (informational)", "", ", ".join(f"`{u}`" for u in unused) or "_none_")
    r.data = {"declared_fields": sorted(fields), "missing_primary": [row[0] for row in p_rows],
              "missing_primary_hard": [row[0] for row in hard_primary], "missing_secondary": [row[0] for row in s_rows],
              "notebook_or_script_assignments": {k: sorted(v) for k, v in nb_sets.items()}, "declared_unused": unused}
    r.status = "FINDING" if (p_rows or s_rows) else "PASS"
    r.headline = (f"{len(p_rows)} missing field(s) in src/training ({len(hard_primary)} hard reads), "
                  f"{len(s_rows)} elsewhere; {len(unused)} declared-but-unused")
    return r


# ----------------------------------------------------------------------------------------------
# A0.8 Evaluation splits & example counts
# ----------------------------------------------------------------------------------------------
def audit_eval_splits(ctx: Ctx) -> AuditResult:
    r = AuditResult("A0.8", "Evaluation Splits & Example Counts")
    cfg = ctx.cfg
    if cfg is None:
        r.status, r.headline = "N/A", "RSCConfig unavailable."
        return r
    n_total = int(cfg.n_samples)
    gsm_train_full = 7473
    n_total_eff = min(n_total, gsm_train_full)
    test_size = 0.08  # literal in src/data/dataset.py: ds.train_test_split(test_size=0.08, ...)
    n_eval = math.ceil(test_size * n_total_eff)
    n_train = n_total_eff - n_eval
    source = "analytic (HF train_test_split formula)"
    buckets: Optional[Dict[str, int]] = None
    gsm_test_len: Optional[int] = None
    if not ctx.tiny and ctx.tokenizer is not None:
        try:
            from src.data.dataset import gsm8k_to_cot  # type: ignore

            with in_dir(ctx.tmp):
                tr, ev = gsm8k_to_cot(ctx.tokenizer, cfg)
            n_train, n_eval, source = len(tr), len(ev), "measured (gsm8k_to_cot)"
            buckets = {"easy": 0, "medium": 0, "hard": 0}
            for ex in ev:
                n = len(ex["steps_list"])
                buckets["easy" if n <= 1 else "medium" if n <= 3 else "hard"] += 1
            from datasets import load_dataset

            gsm_test_len = len(load_dataset("openai/gsm8k", "main", split="test"))
        except Exception as e:
            r.add(f"_Real dataset load failed ({type(e).__name__}: {e}); analytic estimates used._", "")
    acc_n = int(cfg.accuracy_eval_n)
    if buckets:
        per_bucket = min(acc_n // 3, *buckets.values())
        sel_n = min(acc_n, n_eval)
        r.data["bucket_sizes"] = buckets
    else:
        per_bucket, sel_n = acc_n // 3, min(acc_n, n_eval)
    steps_per_epoch = n_train // cfg.effective_batch_size
    pool = min(cfg.router_bootstrap_n_examples, n_train)

    rows = [
        ["GSM8K train pool used (cfg.n_samples)", n_total_eff, "src/data/dataset.py: shuffle(seed).select(n_samples)"],
        ["train split", n_train, source],
        ["eval_data (validation: evaluate_loss uses ALL of it)", n_eval, source],
        ["checkpoint-selection subset (epoch_end_accuracy, cfg.accuracy_eval_n)", sel_n,
         f"stratified easy/medium/hard, {per_bucket} per bucket + remainder; fixed seed={cfg.seed} -> SAME subset every call"],
        ["optimizer steps / epoch (len(train)//effective_batch)", steps_per_epoch, f"effective batch = {cfg.effective_batch_size}"],
        ["router bootstrap pool", pool, f"min(router_bootstrap_n_examples={cfg.router_bootstrap_n_examples}, train)"],
    ]
    if gsm_test_len is not None:
        rows.append(["GSM8K test split (benchmark source)", gsm_test_len, "measured"])
    else:
        rows.append(["GSM8K test split (benchmark source)", 1319, "known size of openai/gsm8k test (not measured)"])
    if buckets:
        rows.append(["eval_data difficulty buckets (easy/medium/hard)", f"{buckets['easy']}/{buckets['medium']}/{buckets['hard']}", "measured"])
    r.add("### Dataset / split sizes", "", *md_table(["Quantity", "Count", "Source / note"], rows))

    # benchmark n_test usage
    nrows: List[List[Any]] = []
    for p in ctx.repo.files({".py", ".ipynb"}):
        rel = ctx.repo.rel(p)
        if origin_of(rel) == "reference":
            continue
        for i, line in enumerate(ctx.repo.text(p).splitlines(), 1):
            if re.search(r"\bn_test\s*[:=]\s*\d+|eval_samples.*default\s*=\s*\d+", line):
                nrows.append([f"{rel}:{i}", line.strip()[:110]])
    r.add("", "### Benchmark sample-size settings found", "", *(md_table(["Site", "Line"], nrows) if nrows else ["_none_"]))

    def half_ci(p: float, n: int) -> float:
        return 1.96 * math.sqrt(p * (1 - p) / max(1, n))

    stat_rows = []
    for label, n in (("checkpoint selection", sel_n), ("benchmark n=50 (notebook/CLI default)", 50),
                     ("benchmark n=100 (function default)", 100), ("full eval_data", n_eval)):
        stat_rows.append([label, n, fmt(1.0 / max(1, n), 4), f"+/-{100 * half_ci(0.5, n):.1f} pp", f"+/-{100 * half_ci(0.25, n):.1f} pp"])
    r.add("", "### Statistical resolution (binomial, 95% CI half-width)", "",
          *md_table(["Evaluation", "n", "1-example step", "CI @ p=0.50", "CI @ p=0.25"], stat_rows))
    improve_thr = 0.005
    r.add("", f"* Checkpoint-selection improvement threshold is `acc > best + {improve_thr}` while one example moves accuracy by "
              f"{fmt(1.0 / max(1, sel_n), 4)} -> **any single extra correct example counts as an improvement** "
              "(selection is noise-dominated; the same fixed subset is reused, so selection can overfit it).",
          "* `evaluate_loss` (validation CE) uses the full eval split, but checkpoint choice in `_evaluate_and_maybe_checkpoint` "
          "is driven by the small accuracy subset.")
    r.data.update({"n_total": n_total_eff, "n_train": n_train, "n_eval": n_eval, "selection_n": sel_n,
                   "source": source, "steps_per_epoch": steps_per_epoch, "router_pool": pool})
    r.status = "FINDING" if sel_n < 200 else "PASS"
    r.headline = f"train={n_train}, eval={n_eval} ({source}); selection subset n={sel_n} (1 example = {100.0 / max(1, sel_n):.1f} pp)"
    return r


# ----------------------------------------------------------------------------------------------
# Orchestration, report writing, integrity
# ----------------------------------------------------------------------------------------------
def hash_manifest(root: Path, exclude: set) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in iter_repo_files(root, SCAN_EXTS, exclude):
        try:
            if p.stat().st_size > 5_000_000:
                continue
            out[p.resolve().relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        except (OSError, ValueError):
            continue
    return out


def run_audit(fn, ctx: Ctx, sid: str, title: str) -> AuditResult:
    log(f"running {sid}: {title}")
    try:
        res = fn(ctx)
    except Exception:
        res = AuditResult(sid, title, "ERROR", "audit crashed (non-fatal)")
        res.add(*code_block(traceback.format_exc(), "text"))
    if not res.headline:
        res.headline = res.status
    log(f"  -> {res.status}: {res.headline}")
    return res


def write_outputs(ctx: Ctx, results: List[AuditResult], integrity: Dict[str, Any]) -> Tuple[Path, Path]:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    md: List[str] = ["# ANCHOR REPORT (Task A0)", "",
                     f"_Generated by `scripts/verify_anchors.py` v{SCRIPT_VERSION} at {now}. Read-only audit; findings never abort._", "",
                     "## Run configuration", ""]
    md += md_table(["Setting", "Value"], [
        ["repo root", ctx.root], ["device", str(ctx.device)], ["--tiny (effective)", ctx.tiny],
        ["torch available", HAVE_TORCH], ["backbone layers", ctx.n_backbone_layers],
        ["candidate layers", ctx.target_layers], ["checkpoint", ctx.ckpt_path or "none found"],
        ["checkpoint applied", ctx.ckpt_applied or "no"], ["seed", ctx.args.seed]])
    if ctx.notes:
        md += ["", "**Notes:**"] + [f"* {n}" for n in ctx.notes]
    md += ["", "## Summary", ""]
    md += md_table(["ID", "Audit", "Status", "Headline"], [[x.sid, x.title, f"**{x.status}**", x.headline] for x in results])
    md += ["", "Status legend: PASS = verified OK; FINDING = discrepancy/risk detected; WARN = partial/provisional; "
               "INFO = informational; N/A = could not be measured; ERROR = audit crashed (traceback inside section)."]
    for x in results:
        md += ["", f"## {x.sid} -- {x.title}", "", f"**Status:** {x.status} -- {x.headline}", ""] + x.lines
    md += ["", "## Repo integrity (read-only guarantee)", ""]
    md += md_table(["Check", "Result"], [[k, v] for k, v in integrity.items()])
    md += ["", "## Assumptions & limitations", "",
           "* The M0 replacement column (A0.3) is the A0 default policy (advisory, non-blocking); confirm against `IMPLEMENTATION_TASKS_SPEC.md` section 3.",
           "* `--tiny` magnitudes (hidden-state norms, SteerNet deltas) are synthetic; only structural findings transfer to the real backbone.",
           "* Steering-norm recommendation is PROVISIONAL unless SteerNets were loaded from a checkpoint on the real backbone.",
           "* `--target-rel-norm` (default 0.10, p95 per-token ||steer||/||h||) is an assumption, not a measured optimum."]
    report_path = ctx.out_dir / "ANCHOR_REPORT.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    a04 = next((x for x in results if x.sid == "A0.4"), None)
    dump: Dict[str, Any] = {
        "generated_at": now, "script_version": SCRIPT_VERSION, "repo_root": str(ctx.root), "device": str(ctx.device),
        "tiny": ctx.tiny, "notes": ctx.notes,
        "audit_summary": {x.sid: {"title": x.title, "status": x.status, "headline": x.headline} for x in results},
        "recommendation_steer_scale": next((x.data.get("recommendation") for x in results if x.sid == "A0.2"), None),
        "repo_integrity": integrity,
    }
    if a04 is not None:
        dump.update(a04.data)
    json_path = ctx.out_dir / "anchor_profile_dump.json"
    json_path.write_text(json.dumps(jsonable(dump), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return report_path, json_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TASK A0: read-only anchor verification for the ARS/RSC repo.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="execution device (default: cpu)")
    ap.add_argument("--tiny", action="store_true", help="use a synthetic tiny backbone (offline, fast, CPU-friendly)")
    ap.add_argument("--ckpt", default=None, help="path to a Phase 1 checkpoint (default: auto-discover)")
    ap.add_argument("--repo-root", default=str(SCRIPT_PATH.parents[1]), help="repository root (default: parent of scripts/)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: <repo-root>/docs)")
    ap.add_argument("--model-name", default=None, help="HF model id for the real backbone (default: cfg.model_name)")
    ap.add_argument("--trust-remote-code", action="store_true", help="pass trust_remote_code=True when loading the backbone")
    ap.add_argument("--layers", default="13,15,18,20,22", help="layers for the steering-norm audit")
    ap.add_argument("--k-values", default="1,2,3", help="K values for the steering-norm audit")
    ap.add_argument("--alphas", default="0.30,1.0", help="alpha values for the steering-norm audit")
    ap.add_argument("--target-rel-norm", type=float, default=0.10,
                    help="assumed p95 relative steering norm budget used for the steer_scale recommendation")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    if not root.exists():
        print(f"[A0] repo root does not exist: {root}", file=sys.stderr)
        return 2
    ctx = None
    try:
        ctx = build_context(args)
        before_hash = hash_manifest(ctx.root, ctx.exclude)
        before_top = set(os.listdir(ctx.root))
        audits = [
            (audit_gate_init, "A0.1", "Gate Initialization Audit"),
            (audit_operating_point, "A0.2", "Operating Point & Steering Norm Audit"),
            (audit_wiring, "A0.3", "Audit & Preflight Wiring Analysis"),
            (audit_profile_dump, "A0.4", "Kaggle Profile & Config Dump"),
            (audit_router_dropout, "A0.5", "Dropout in Router Counterfactuals"),
            (audit_fixed_arm, "A0.6", "Fixed-Arm Topology Verification"),
            (audit_config_drift, "A0.7", "Config Drift Analysis"),
            (audit_eval_splits, "A0.8", "Evaluation Splits & Example Counts"),
        ]
        results = [run_audit(fn, ctx, sid, title) for fn, sid, title in audits]
        after_hash = hash_manifest(ctx.root, ctx.exclude)
        after_top = set(os.listdir(ctx.root))
        try:
            out_top = ctx.out_dir.relative_to(ctx.root).parts[0]
        except ValueError:
            out_top = None
        changed = sorted(k for k in set(before_hash) | set(after_hash) if before_hash.get(k) != after_hash.get(k))
        new_top = sorted((after_top - before_top) - ({out_top} if out_top else set()))
        integrity = {
            "files hashed (text/py/ipynb/json/md)": len(before_hash),
            "repo files modified/added/removed": changed or "none",
            "unexpected new top-level entries": new_top or "none",
            "output files written": "docs/ANCHOR_REPORT.md, docs/anchor_profile_dump.json (only)",
            "READ-ONLY GUARANTEE HELD": (not changed) and (not new_top),
        }
        report_path, json_path = write_outputs(ctx, results, integrity)
        print("\n" + "=" * 100)
        print(f"{'ID':6} {'STATUS':8} HEADLINE")
        for x in results:
            print(f"{x.sid:6} {x.status:8} {x.headline}")
        print("=" * 100)
        print(f"Report : {report_path}\nProfile: {json_path}\nRead-only guarantee held: {integrity['READ-ONLY GUARANTEE HELD']}")
    finally:
        if ctx is not None:
            shutil.rmtree(ctx.tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
