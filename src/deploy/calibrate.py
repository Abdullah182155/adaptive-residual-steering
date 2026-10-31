"""S5 deployment calibration -> immutable ``deployment_contract.json``.

Wrapper API assumed (M1-M7):
  model.counterfactual_mode()              context manager
  model.set_gate_mode(mode, scale=None)    mode in {"const", "learned"}
  model.set_routing_override(mode, layers) "only" / "none" / None ...
  model.set_theta_abs(theta)               catalog-router abstention threshold
  model.set_guards(enabled=, k_max=, rel_norm_cap=, finite_check=, revert_to_k0=)
  model.guard_state["rel_norms"]           {layer: 1-D tensor of ||delta_injected||/||h||}
  model.tokenizer                          set by build_model()

Optional (called only if present):
  set_router_policy, set_router_inference_mode, set_k_scale_mode,
  reset_guard_state, reset_routing_diagnostics.
"""

import hashlib
import json
import os
import random
import subprocess
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.routing.rloo import per_example_answer_nll

SCHEMA_VERSION = 1
THETA_NO_ABSTAIN = 1.01
NOMINAL_REL_NORM_CAP = 0.02
ARMS = ("fixed", "router", "full_ars", "raw_ars")
AUTO_FIELDS = ("schema_version", "created", "git_commit")
REQUIRED_FIELDS = (
    "schema_version", "created", "git_commit",
    "steer_scale", "k_scale_mode", "steernet_k_conditioning",
    "catalog", "s_fixed_star", "fixed_evidence",
    "theta_abs", "theta_table",
    "router", "gate", "envelope", "guards", "steernet",
    "selection_split", "manifest_path",
)
NESTED_REQUIRED = {
    "catalog":         ("layers", "max_k"),
    "router":          ("policy", "inference_mode", "ckpt", "sha256"),
    "gate":            ("mode", "adopted", "evidence", "ckpt", "sha256"),
    "envelope":        ("rel_norm_cap", "quantile", "p50", "p99", "max", "per_layer"),
    "guards":          ("k_max", "finite_check", "revert_to_k0"),
    "steernet":        ("ckpt", "sha256"),
    "selection_split": ("n", "ids_hash"),
}

# --------------------------------------------------------------------------- helpers

def _opt(obj, name, *args, **kwargs):
    fn = getattr(obj, name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _calib_bs(cfg) -> int:
    return max(1, int(getattr(cfg, "calib_batch_size", 4)))


def _scale(cfg) -> float:
    return float(getattr(cfg, "steer_scale", 1.0))


def _gate_const(model, cfg):
    model.set_gate_mode("const", scale=_scale(cfg))


def _stack(chunk, key, device):
    return torch.stack([torch.as_tensor(r[key]) for r in chunk]).to(device)


def _forward_chunk(model, chunk, device):
    ids  = _stack(chunk, "input_ids", device)
    mask = _stack(chunk, "attention_mask", device)
    lbl  = _stack(chunk, "labels", device)
    model._attn_mask_cache = mask
    if any("question_span_mask" in r for r in chunk):
        qm = _stack(chunk, "question_span_mask", device)
        _opt(model, "set_router_context_mask", qm.bool())
    elif any("prompt_shield_mask" in r for r in chunk):
        sh = _stack(chunk, "prompt_shield_mask", device)
        _opt(model, "set_router_context_mask", mask.bool() & (~sh.bool()))
    else:
        _opt(model, "set_router_context_mask", mask.bool() & lbl.eq(-100))
    if any("n_shot" in r or "shot" in r for r in chunk):
        shots = [r.get("n_shot", r.get("shot", 0)) for r in chunk]
        _opt(model, "set_shot_bucket", _opt(model, "shot_to_bucket", shots))
    with torch.amp.autocast("cuda", enabled=(torch.device(device).type == "cuda")):
        out = model(input_ids=ids, attention_mask=mask)
    return out.logits, lbl


@torch.no_grad()
def _row_nll(model, rows, cfg, device) -> np.ndarray:
    """Per-row answer NLL under the model's *current* routing/gate state."""
    model.eval()
    bs  = _calib_bs(cfg)
    out = []
    for s in range(0, len(rows), bs):
        logits, lbl = _forward_chunk(model, rows[s:s + bs], device)
        out.append(per_example_answer_nll(logits, lbl).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def paired_greedy_eval(model, rows, cfg, device) -> List[bool]:
    """Greedy-decode every selection row under the current state; per-row correctness."""
    from src.evaluation.benchmark import generate_batch
    from src.evaluation.verifier import extract_final_number
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        raise RuntimeError("paired_greedy_eval needs model.tokenizer (use build_model or set it)")
    prompts, golds = [], []
    for r in rows:
        ids, lbl = torch.as_tensor(r["input_ids"]), torch.as_tensor(r["labels"])
        qm = (lbl == -100) & (ids != tok.pad_token_id)
        prompts.append(tok.decode(ids[qm], skip_special_tokens=True))
        golds.append(extract_final_number(tok.decode(ids[lbl != -100], skip_special_tokens=True)))
    texts = generate_batch(
        model, tok, prompts, device,
        max_new_tokens=int(getattr(cfg, "calib_max_new_tokens", 256)),
        batch_size=int(getattr(cfg, "calib_gen_batch", 8)),
    )
    ok = []
    for t, g in zip(texts, golds):
        p = extract_final_number(t)
        ok.append(bool(g is not None and p is not None and abs(p - g) < 0.01))
    return ok


def _catalog_subsets(catalog) -> List[Tuple[int, ...]]:
    seq = getattr(catalog, "subsets", None)
    if seq is None and isinstance(catalog, dict):
        seq = catalog.get("subsets")
    if seq is None:
        seq = catalog
    out: List[Tuple[int, ...]] = []
    for s in seq:
        t = tuple(sorted(int(x) for x in s))
        if t and t not in out:
            out.append(t)
    return out


def default_catalog(target_layers: Sequence[int], cfg):
    """Use the M5 Catalog.build if present; otherwise singles/pairs/triples from cfg flags."""
    try:
        from src.routing.catalog import Catalog
        layers = getattr(cfg, "catalog_layers", None) or tuple(target_layers)
        max_k = getattr(cfg, "catalog_max_k", 3)
        return Catalog.build(tuple(layers), max_k)
    except Exception:
        pass
    ls   = list(target_layers)
    subs = [(l,) for l in ls]
    if getattr(cfg, "catalog_include_pairs", True):
        subs += [(ls[i], ls[i + 1]) for i in range(len(ls) - 1)]
    if getattr(cfg, "catalog_consecutive_triples", True):
        subs += [(ls[i], ls[i + 1], ls[i + 2]) for i in range(len(ls) - 2)]
    return subs


def select_rows(eval_data, cfg) -> list:
    """Stratified difficulty selection split (n = cfg.calib_sel_n) drawn from eval_data.
    
    Samples balanced across difficulty levels (2-step, 3-step, 5+ step reasoning)
    to make gate adoption decisions statistically robust.
    """
    n   = min(int(cfg.calib_sel_n), len(eval_data))
    rng = random.Random(int(cfg.seed) + 8080)
    
    # Categorize by number of reasoning steps
    buckets = {"easy": [], "medium": [], "hard": []}
    for idx, ex in enumerate(eval_data):
        steps = len(ex.get("steps_list", [])) if isinstance(ex, dict) else 0
        bucket = "easy" if steps <= 2 else "medium" if steps <= 4 else "hard"
        buckets[bucket].append(idx)
    
    # Sample proportionally from each bucket (at least 1 each)
    per_bucket = max(1, n // max(1, len(buckets)))
    idxs = []
    for values in buckets.values():
        idxs.extend(rng.sample(values, min(per_bucket, len(values))))
    
    # Fill remainder from remaining examples
    if len(idxs) < n:
        remaining = sorted(set(range(len(eval_data))) - set(idxs))
        rng.shuffle(remaining)
        idxs.extend(remaining[: n - len(idxs)])
    
    return [eval_data[i] for i in sorted(idxs[:n])]


def _ids_hash(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        q = r.get("question") if hasattr(r, "get") else None
        if q is None:
            q = json.dumps(torch.as_tensor(r["input_ids"]).tolist())
        h.update(str(q).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _abstain_rate(model) -> Optional[float]:
    try:
        return float(model.routing_k_distribution().get(0, 0.0))
    except Exception:
        return None


def _router_state(model, cfg, theta: float):
    model.set_routing_override(None)
    _opt(model, "set_router_policy", "catalog")
    _opt(model, "set_router_inference_mode", "argmax")
    model.set_theta_abs(float(theta))

# --------------------------------------------------------------------------- (a)/(b) fixed subset

def choose_fixed_subset(model, sel_rows, catalog, cfg, device) -> Tuple[Tuple[int, ...], dict]:
    subsets = _catalog_subsets(catalog)
    if not subsets:
        raise ValueError("catalog has no non-empty subsets")
    eps   = float(getattr(cfg, "calib_nll_tie_eps", 0.003))
    table: List[Dict[str, Any]] = []
    with model.counterfactual_mode():
        _gate_const(model, cfg)
        for s in subsets:
            model.set_routing_override("only", tuple(s))
            nll = _row_nll(model, sel_rows, cfg, device)
            table.append({"subset": list(s), "k": len(s), "mean_nll": float(np.mean(nll))})
        best  = min(e["mean_nll"] for e in table)
        cands = [e for e in table if e["mean_nll"] <= best + eps]
        for e in cands:
            model.set_routing_override("only", tuple(e["subset"]))
            ok      = paired_greedy_eval(model, sel_rows, cfg, device)
            e["acc"] = float(np.mean(ok)) if ok else 0.0
        model.set_routing_override(None)
    winner = sorted(cands, key=lambda e: (-e["acc"], e["k"], e["mean_nll"], e["subset"]))[0]
    evidence = {
        "n":         len(sel_rows),
        "tie_eps":   eps,
        "best_nll":  best,
        "ranking":   sorted(table, key=lambda e: (e["mean_nll"], e["subset"])),
        "candidates": cands,
        "winner":    winner["subset"],
        "rule":      "min paired NLL within eps -> max greedy acc -> smaller K -> lower NLL",
    }
    return tuple(winner["subset"]), evidence

# --------------------------------------------------------------------------- theta search

def search_theta_abs(model, sel_rows, cfg, device) -> Tuple[float, List[dict]]:
    grid  = [float(t) for t in cfg.calib_theta_grid]
    table: List[Dict[str, Any]] = []
    with model.counterfactual_mode():
        _gate_const(model, cfg)
        model.set_routing_override("none")
        base = _row_nll(model, sel_rows, cfg, device)
        model.set_routing_override(None)
        _opt(model, "set_router_policy", "catalog")
        _opt(model, "set_router_inference_mode", "argmax")
        for th in grid:
            model.set_theta_abs(th)
            _opt(model, "reset_routing_diagnostics")
            nll = _row_nll(model, sel_rows, cfg, device)
            table.append({
                "theta":        th,
                "nll_gain":     float(np.mean(base - nll)),
                "mean_nll":     float(np.mean(nll)),
                "abstain_rate": _abstain_rate(model),
                "acc":          None,
            })
    best_gain = max(e["nll_gain"] for e in table)
    cands     = [e for e in table if e["nll_gain"] >= best_gain - 1e-9]
    for e in cands:
        model.set_theta_abs(e["theta"])
        ok      = paired_greedy_eval(model, sel_rows, cfg, device)
        e["acc"] = float(np.mean(ok)) if ok else 0.0
    winner = max(cands, key=lambda e: (e["acc"], e["theta"]))
    return float(winner["theta"]), table

# --------------------------------------------------------------------------- envelope

def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else float("nan")


def _summ(x: np.ndarray, q: float) -> Dict[str, Any]:
    return {
        "p50": _q(x, 0.5),
        "p99": _q(x, 0.99),
        "q":   _q(x, q),
        "max": float(x.max()) if x.size else float("nan"),
        "n":   int(x.size),
    }


@torch.no_grad()
def measure_envelope(model, rows, cfg, device) -> dict:
    """Distribution of ||delta_injected|| / ||h|| under the model's current learned routes + gates."""
    model.eval()
    q  = float(cfg.calib_envelope_quantile)
    bs = _calib_bs(cfg)
    per_layer: Dict[int, List[np.ndarray]] = {}
    _opt(model, "reset_guard_state")
    model._collect_guard_rel_norms = True
    try:
      with model.counterfactual_mode():
        for s in range(0, len(rows), bs):
            _forward_chunk(model, rows[s:s + bs], device)
            gs = getattr(model, "guard_state", None) or {}
            for layer, vals in (gs.get("rel_norms") or {}).items():
                arr = (
                    vals.detach().float().cpu().numpy()
                    if torch.is_tensor(vals)
                    else np.asarray(vals, dtype=float)
                ).ravel()
                per_layer.setdefault(int(layer), []).append(arr[np.isfinite(arr)])
    finally:
      model._collect_guard_rel_norms = False
    _opt(model, "reset_guard_state")
    layer_arrays = {l: (np.concatenate(v) if v else np.zeros(0)) for l, v in per_layer.items()}
    allv = np.concatenate(list(layer_arrays.values())) if layer_arrays else np.zeros(0)
    g    = _summ(allv, q)
    warns: List[str] = []
    cap = g["q"]
    if not np.isfinite(cap) or allv.size == 0:
        cap = NOMINAL_REL_NORM_CAP
        msg = "W_ENVELOPE_MISMATCH: no injected-norm samples were recorded; using nominal cap 0.02"
        warns.append(msg)
        warnings.warn(msg)
    else:
        ratio = cap / NOMINAL_REL_NORM_CAP
        if ratio > 2.0 or ratio < 0.5:
            msg = (
                f"W_ENVELOPE_MISMATCH: measured q{q} cap {cap:.5f} differs from "
                f"{NOMINAL_REL_NORM_CAP} by {ratio:.2f}x (> 2x)"
            )
            warns.append(msg)
            warnings.warn(msg)
    return {
        "rel_norm_cap": float(cap),
        "quantile":     q,
        "p50":          g["p50"],
        "p99":          g["p99"],
        "max":          g["max"],
        "n_samples":    g["n"],
        "nominal_cap":  NOMINAL_REL_NORM_CAP,
        "per_layer":    {str(l): _summ(a, q) for l, a in sorted(layer_arrays.items())},
        "warnings":     warns,
    }

# --------------------------------------------------------------------------- gate adoption

def decide_gate_adoption(nll_paired: dict, acc_paired: dict, tol: int = 2) -> Tuple[str, dict]:
    n     = max(1, int(acc_paired.get("n") or nll_paired.get("n") or 1))
    diff  = float(nll_paired["mean_nll_diff"])
    acc_g = float(acc_paired["acc_gated"])
    acc_u = float(acc_paired["acc_ungated"])
    floor  = acc_u - max(1, tol) * (1.0 / n)
    # Primary criterion: task NLL reduction with greedy accuracy non-inferiority
    nll_ok = diff < 0
    acc_ok = acc_g >= floor - 1e-12
    decision = "learned" if (nll_ok and acc_ok) else "passthrough"
    return decision, {
        "mean_nll_diff": diff,
        "acc_gated":     acc_g,
        "acc_ungated":   acc_u,
        "n":             n,
        "tol":           tol,
        "acc_floor":     floor,
        "nll_ok":        bool(nll_ok),
        "acc_ok":        bool(acc_ok),
        "decision":      decision,
    }


def _gate_comparison(model, rows, cfg, device) -> Tuple[dict, dict]:
    with model.counterfactual_mode():
        _gate_const(model, cfg)
        nll_u = _row_nll(model, rows, cfg, device)
        acc_u = paired_greedy_eval(model, rows, cfg, device)
        model.set_gate_mode("learned")
        nll_g = _row_nll(model, rows, cfg, device)
        acc_g = paired_greedy_eval(model, rows, cfg, device)
    diff = nll_g - nll_u
    au, ag = np.asarray(acc_u, bool), np.asarray(acc_g, bool)
    nll_paired = {
        "mean_nll_diff":       float(diff.mean()),
        "mean_nll_gated":      float(nll_g.mean()),
        "mean_nll_ungated":    float(nll_u.mean()),
        "frac_rows_improved":  float((diff < 0).mean()),
        "n":                   len(rows),
    }
    acc_paired = {
        "acc_gated":             float(ag.mean()),
        "acc_ungated":           float(au.mean()),
        "n":                     len(rows),
        "gated_only_correct":    int((ag & ~au).sum()),
        "ungated_only_correct":  int((au & ~ag).sum()),
    }
    return nll_paired, acc_paired


def decide_gate_adoption_per_shot(nll_paired_by_shot: dict, acc_paired_by_shot: dict, tol: int = 2) -> Tuple[str, dict]:
    """Evaluate gate adoption separately for each shot condition.
    
    Gate is adopted if it improves NLL and maintains accuracy in ANY shot condition.
    This allows shot-specific gate policy where the gate helps in some conditions
    but not others.
    """
    decisions = {}
    for shot in [0, 2, 4, 8]:
        if shot not in nll_paired_by_shot or shot not in acc_paired_by_shot:
            continue
        nll_p = nll_paired_by_shot[shot]
        acc_p = acc_paired_by_shot[shot]
        decision, _ = decide_gate_adoption(nll_p, acc_p, tol)
        decisions[shot] = decision
    
    # Overall decision: adopt if beneficial in any shot
    overall_decision = "learned" if "learned" in decisions.values() else "passthrough"
    
    return overall_decision, {
        "overall": overall_decision,
        "by_shot": decisions,
    }

# --------------------------------------------------------------------------- contract I/O

def _validate_contract(c: dict):
    missing = [k for k in REQUIRED_FIELDS if k not in c]
    if missing:
        raise ValueError(f"deployment contract missing fields: {missing}")
    for parent, subs in NESTED_REQUIRED.items():
        if not isinstance(c[parent], dict):
            raise ValueError(f"contract field '{parent}' must be an object")
        sm = [s for s in subs if s not in c[parent]]
        if sm:
            raise ValueError(f"contract field '{parent}' missing keys: {sm}")


def write_contract(path: str, **fields) -> dict:
    contract = dict(fields)
    contract.setdefault("schema_version", SCHEMA_VERSION)
    contract.setdefault("created",        datetime.now(timezone.utc).isoformat())
    contract.setdefault("git_commit",     git_commit())
    _validate_contract(contract)
    text = json.dumps(contract, indent=2, default=_default)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return json.loads(text)


def load_contract(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        c = json.load(f)
    _validate_contract(c)
    if int(c["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported contract schema_version {c['schema_version']}")
    return c


def load_contract_weights(model, contract: dict, device=None, verify: bool = True):
    from src.training.trainer import load_steer_weights
    from src.routing.catalog import Catalog
    if "catalog" in contract:
        cat_info = contract["catalog"]
        layers = tuple(int(l) for l in cat_info["layers"])
        max_k = int(cat_info.get("max_k", 3))
        # Use the exact subsets stored in the contract so the catalog_head
        # output dimension matches the router checkpoint (e.g. 26 not 163).
        # _catalog_subsets() strips the empty set () when writing the contract,
        # so we prepend it back here to restore the original 26-entry catalog.
        if "subsets" in cat_info:
            non_empty = tuple(tuple(int(x) for x in s) for s in cat_info["subsets"])
            subsets = ((),) + non_empty
            catalog = Catalog(layers=layers, max_k=max_k, subsets=subsets)
        else:
            catalog = Catalog.build(layers, max_k)
        _opt(model, "set_catalog", catalog)
    seen = set()
    for key in ("steernet", "router", "gate"):
        ent  = contract[key]
        path = ent["ckpt"]
        if path in seen:
            continue
        seen.add(path)
        # Skip missing checkpoints (Phase 1: SteerNet-only has no router/gate)
        if not os.path.exists(path):
            print(f"[load_contract_weights] Skipping {key} checkpoint (not found): {path}")
            continue
        if verify:
            actual = sha256_file(path)
            if actual != ent["sha256"]:
                raise ValueError(f"sha256 mismatch for {key} checkpoint {path}")
        load_steer_weights(model, path, device=device)

# --------------------------------------------------------------------------- arm switching

def apply_contract(model, contract: dict, which: str):
    if which not in ARMS:
        raise ValueError(f"which must be one of {ARMS}, got {which!r}")
    scale = float(contract["steer_scale"])
    g     = contract["guards"]
    cap   = float(contract["envelope"]["rel_norm_cap"])
    _opt(model, "set_k_scale_mode", contract.get("k_scale_mode"))

    def guards(enabled: bool):
        model.set_guards(
            enabled=enabled,
            k_max=int(g["k_max"]),
            rel_norm_cap=cap,
            finite_check=bool(g["finite_check"]),
            revert_to_k0=bool(g["revert_to_k0"]),
        )

    if which == "fixed":
        model.set_routing_override("only", tuple(int(l) for l in contract["s_fixed_star"]))
        model.set_gate_mode("const", scale=scale)
        guards(True)
        return

    model.set_routing_override(None)
    _opt(model, "set_router_policy",         contract["router"]["policy"])
    _opt(model, "set_router_inference_mode", contract["router"]["inference_mode"])

    if which == "raw_ars":
        model.set_theta_abs(THETA_NO_ABSTAIN)
        model.set_gate_mode("learned")
        guards(False)
        return

    model.set_theta_abs(float(contract["theta_abs"]))
    if which == "router" or contract["gate"]["adopted"] != "learned":
        model.set_gate_mode("const", scale=scale)
    else:
        model.set_gate_mode("learned")
    guards(True)

# --------------------------------------------------------------------------- model loading

def build_model(cfg, device):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from src.models.wrapper import Phi2WithRSC
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
        model_config.pad_token_id = tok.pad_token_id or getattr(model_config, "eos_token_id", 50256)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=model_config,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)
    model           = Phi2WithRSC(base, cfg, device=device)
    model.tokenizer = tok
    return model, tok

# --------------------------------------------------------------------------- orchestration

def run_deployment_calibration(
    model,
    tokenizer,
    eval_data,
    cfg,
    device,
    catalog=None,
    steernet_ckpt: Optional[str] = None,
    router_ckpt:   Optional[str] = None,
    gate_ckpt:     Optional[str] = None,
    manifest_path: str           = "",
    out_path:      Optional[str] = None,
    load_weights:  bool          = True,
) -> dict:
    from src.training.trainer import load_steer_weights
    model.tokenizer = tokenizer

    steernet_ckpt = steernet_ckpt or getattr(cfg, "s1_save_path", None) or getattr(cfg, "lora_save_path", None) or os.path.join(cfg.checkpoint_dir, "s1_steernet.pt")
    router_ckpt   = router_ckpt   or getattr(cfg, "router_catalog_save_path", None) or getattr(cfg, "router_bootstrap_save_path", None) or os.path.join(cfg.checkpoint_dir, "router_catalog_v5.pt")
    gate_ckpt     = gate_ckpt     or getattr(cfg, "gate_v5_save_path", None) or getattr(cfg, "phase2_save_path", None) or os.path.join(cfg.checkpoint_dir, "gate_v5.pt")

    # SteerNet is always required; router and gate are optional for phased deployment.
    # Phase 1 (SteerNet-only) skips router/gate; Phase 2 adds router; Phase 3 adds gate.
    has_router = os.path.exists(router_ckpt)
    has_gate   = os.path.exists(gate_ckpt)
    if not os.path.exists(steernet_ckpt):
        raise FileNotFoundError(f"SteerNet checkpoint required for deployment contract: {steernet_ckpt}")
    if not has_router:
        print(f"[S5] Router checkpoint not found ({router_ckpt}); running SteerNet-only (Phase 1)")
    if not has_gate:
        print(f"[S5] Gate checkpoint not found ({gate_ckpt}); gate will be passthrough")

    if catalog is None:
        catalog = default_catalog(model.target_layers, cfg)
    _opt(model, "set_catalog", catalog)

    if load_weights:
        seen = set()
        # Always load SteerNet
        load_steer_weights(model, steernet_ckpt, device=device)
        seen.add(steernet_ckpt)
        # Load router and gate only if they exist
        for p in (router_ckpt, gate_ckpt):
            if p not in seen and os.path.exists(p):
                seen.add(p)
                load_steer_weights(model, p, device=device)

    rows = select_rows(eval_data, cfg)
    print(f"[S5] selection split n={len(rows)}")

    s_fixed, fixed_ev = choose_fixed_subset(model, rows, catalog, cfg, device)
    print(f"[S5] s*_fixed = {s_fixed}")

    theta, theta_table = search_theta_abs(model, rows, cfg, device)
    print(f"[S5] theta*_abs = {theta}")

    _router_state(model, cfg, theta)
    # Skip gate comparison if no gate checkpoint exists (Phase 1: SteerNet-only)
    if has_gate:
        nll_p, acc_p = _gate_comparison(model, rows, cfg, device)
        use_shot_aware = True
        shots = [r.get("n_shot", r.get("shot", 0)) for r in rows]
        if len(set(shots)) > 1:
            nll_by_shot = {}
            acc_by_shot = {}
            for shot in set(shots):
                shot_rows = [r for r, s in zip(rows, shots) if s == shot]
                if len(shot_rows) > 0:
                    nll_sp, acc_sp = _gate_comparison(model, shot_rows, cfg, device)
                    nll_by_shot[shot] = nll_sp
                    acc_by_shot[shot] = acc_sp
            if nll_by_shot and acc_by_shot:
                adopted, gate_ev = decide_gate_adoption_per_shot(
                    nll_by_shot, acc_by_shot, tol=int(getattr(cfg, "calib_gate_acc_tolerance", 1)))
            else:
                use_shot_aware = False
                adopted, gate_ev = decide_gate_adoption(
                    nll_p, acc_p, tol=int(getattr(cfg, "calib_gate_acc_tolerance", 1)))
        else:
            use_shot_aware = False
            adopted, gate_ev = decide_gate_adoption(
                nll_p, acc_p, tol=int(getattr(cfg, "calib_gate_acc_tolerance", 1)))
    else:
        # Phase 1: no gate checkpoint — skip gate comparison, adopt passthrough
        use_shot_aware = False
        adopted = "passthrough"
        nll_p = {"mean_nll_diff": 0.0, "n": len(rows)}
        acc_p = {"acc_gated": 0.0, "acc_ungated": 0.0, "n": len(rows)}
        gate_ev = {
            "mean_nll_diff": 0.0, "acc_gated": 0.0, "acc_ungated": 0.0,
            "n": len(rows), "decision": "passthrough", "reason": "no_gate_checkpoint",
        }
    
    gate_ev = {**gate_ev, "nll_paired": nll_p, "acc_paired": acc_p}
    print(f"[S5] gate adoption -> {adopted} (shot_aware={use_shot_aware})")

    _router_state(model, cfg, theta)
    if adopted == "learned":
        model.set_gate_mode("learned")
    else:
        _gate_const(model, cfg)

    envelope = measure_envelope(model, rows, cfg, device)
    print(f"[S5] envelope cap={envelope['rel_norm_cap']:.5f} p99={envelope['p99']:.5f}")

    contract = write_contract(
        out_path or cfg.contract_path,
        steer_scale=_scale(cfg),
        k_scale_mode=getattr(cfg, "k_scale_mode", "sqrt"),
        steernet_k_conditioning=bool(getattr(cfg, "steernet_k_conditioning", True)),
        catalog={
            "layers":  list(catalog.layers),
            "max_k":   int(cfg.max_active_layers),
            "subsets": [list(s) for s in _catalog_subsets(catalog)],
        },
        s_fixed_star=list(s_fixed),
        fixed_evidence=fixed_ev,
        theta_abs=theta,
        theta_table=theta_table,
        router={
            "policy":         "catalog",
            "inference_mode": "argmax",
            "ckpt":           router_ckpt if has_router else "",
            "sha256":         sha256_file(router_ckpt) if has_router else "",
        },
        gate={
            "mode":     "learned",
            "adopted":  adopted,
            "evidence": gate_ev,
            "ckpt":     gate_ckpt if has_gate else "",
            "sha256":   sha256_file(gate_ckpt) if has_gate else "",
        },
        envelope=envelope,
        guards={
            "k_max":        int(getattr(cfg, "guard_k_max", cfg.max_active_layers)),
            "finite_check": True,
            "revert_to_k0": True,
        },
        steernet={
            "ckpt":   steernet_ckpt,
            "sha256": sha256_file(steernet_ckpt),
        },
        selection_split={"n": len(rows), "ids_hash": _ids_hash(rows)},
        manifest_path=manifest_path,
    )
    print(f"[S5] wrote {out_path or cfg.contract_path}")
    return contract
