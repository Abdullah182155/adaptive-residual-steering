"""GSM8K evaluation: legacy single-arm benchmark + 5-arm production runner with output cache."""

import contextlib
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from src.evaluation.answers import cut_continuation
from src.deploy.calibrate import apply_contract, git_commit
from src.evaluation.output_cache import (
    EMPTY_ROUTE_KEY,
    OutputCache,
    canonical_route_key,
    key_to_str,
)
from src.evaluation.stats import compare_to_control
from src.evaluation.verifier import (
    build_prompt,
    evaluate_response,
    extract_final_answer,
    generate_response,
)

# ----------------------------------------------------------------------------- legacy

def eval_gsm8k_benchmark(
    model,
    tokenizer,
    device: torch.device,
    stage_name: str,
    n_test: int = 100,
    n_shot: int = 0,
    display_examples: int = 3,
) -> Dict[str, Any]:
    """Single-arm reproducible GSM8K test evaluation (Stage A / B / C protocol)."""
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
    ds = ds.select(range(min(n_test, len(ds))))
    correct, results, total_time, total_tokens = 0, [], 0.0, 0
    print(f"\n{'=' * 80}\n GSM8K Benchmark: {stage_name} (n={len(ds)}, {n_shot}-shot)\n{'=' * 80}")
    for idx, ex in enumerate(ds):
        q    = ex["question"]
        gp   = ex["answer"].split("####")
        gold = float(gp[1].replace(",", "").strip()) if len(gp) > 1 else None
        prompt = build_prompt(q, n_shot=n_shot)
        t0     = time.time()
        resp   = generate_response(model, tokenizer, prompt, device, max_new_tokens=512)
        elapsed = time.time() - t0
        total_time += elapsed
        er      = evaluate_response(resp, gold, min_steps=1)
        correct += int(er["correct"])
        n_tok   = len(tokenizer.encode(resp, add_special_tokens=False))
        total_tokens += n_tok
        results.append({
            "idx": idx, "question": q, "gold": gold, "response": resp,
            "extracted": er["extracted"], "correct": er["correct"],
            "steps": er["steps"], "has_answer": er["has_answer"],
            "time_sec": elapsed, "tokens": n_tok,
        })
        if idx < display_examples:
            icon = "✅" if er["correct"] else "❌"
            print(f"[{idx + 1}/{len(ds)}] {icon} Expected: {gold} | Got: {er['extracted']} ({elapsed:.2f}s)")
    acc     = 100.0 * correct / max(1, len(results))
    summary = {
        "stage":          stage_name,
        "n_examples":     len(results),
        "n_shot":         n_shot,
        "correct":        correct,
        "accuracy":       acc,
        "answer_present": 100.0 * sum(r["has_answer"] for r in results) / max(1, len(results)),
        "step_coverage":  100.0 * sum(r["steps"] > 0 for r in results) / max(1, len(results)),
        "avg_time_sec":   total_time / max(1, len(results)),
        "tokens_per_sec": total_tokens / max(1e-5, total_time),
    }
    print(
        f"\n{'-' * 80}\nSummary [{stage_name} - {n_shot} shot]: "
        f"Accuracy {acc:.2f}% ({correct}/{len(results)}) "
        f"Tokens/sec {summary['tokens_per_sec']:.2f}\n{'-' * 80}\n"
    )
    return {"summary": summary, "results": results}

# ----------------------------------------------------------------------------- arms

ARM_SPEC: List[Tuple[str, str, Optional[str]]] = [
    ("baseline", "Baseline",      None),
    ("fixed",    "Fixed-control", "fixed"),
    ("router",   "Router-only",   "router"),
    ("full_ars", "Full ARS",      "full_ars"),
    ("raw_ars",  "Raw ARS",       "raw_ars"),
]


def _opt(obj, name, *a, **k):
    fn = getattr(obj, name, None)
    return fn(*a, **k) if callable(fn) else None


def apply_baseline(model):
    model.set_routing_override("none")
    model.set_guards(enabled=False)


def _arm_gate(contract: dict, which: Optional[str]):
    scale = float(contract["steer_scale"])
    if which is None:
        return "none", None
    if which in ("fixed", "router"):
        return "const", scale
    if which == "raw_ars":
        return "learned", None
    if contract["gate"]["adopted"] == "learned":
        return "learned", None
    return "const", scale


def _arm_guards_enabled(which: Optional[str]) -> bool:
    return which in ("fixed", "router", "full_ars")


def _guard_monitor(model, contract: dict, enabled: bool):
    """Benchmark generation only *flags* trips; the runner substitutes the cached K=0 text."""
    g = contract["guards"]
    model.set_guards(
        enabled=enabled,
        k_max=int(g["k_max"]),
        rel_norm_cap=float(contract["envelope"]["rel_norm_cap"]),
        finite_check=bool(g["finite_check"]),
        revert_to_k0=False,
    )


def batch_size_for(cfg, shots: int) -> int:
    return int(dict(cfg.bench_batch_size).get(shots, 4))

# ----------------------------------------------------------------------------- extraction

_STRICT_PATTERNS = (
    re.compile(r"####\s*\$?(-?\d[\d,]*\.?\d*)"),
    re.compile(r"[Ff]inal\s+[Aa]nswer(?:\s+is)?\s*[:\-]?\s*\$?(-?\d[\d,]*\.?\d*)"),
)
_TRIM_PATTERNS = (
    r"####\s*-?\d[\d,]*\.?\d*",
    r"[Ff]inal\s+[Aa]nswer\s*[:\-]?\s*-?\d[\d,]*\.?\d*",
    r"[Tt]herefore[^.\n]*\.",
    r"[Tt]he answer is[^.\n]*\.",
)


def extract_strict(text: str) -> Optional[float]:
    """Answer only from the explicit '####' / 'Final answer' markers (no fallbacks)."""
    for pat in _STRICT_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", "").rstrip("."))
            except ValueError:
                return None
    return None


def _trim_response(raw: str) -> str:
    for pat in _TRIM_PATTERNS:
        m = re.search(pat, raw)
        if m:
            return raw[: m.end()].strip()
    return raw


def _is_correct(pred: Optional[float], gold: Optional[float]) -> bool:
    return pred is not None and gold is not None and abs(pred - gold) <= 0.01

# ----------------------------------------------------------------------------- generation

@contextlib.contextmanager
def _left_padding(tokenizer):
    old = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = old


class _AllDoneCriteria(StoppingCriteria):
    """Stop once every sequence in the batch has emitted a complete answer marker."""
    PAT = re.compile(r"(?:####|[Ff]inal\s+[Aa]nswer\s*[:\-]?)\s*\$?-?\d[\d,]*\.?\d*[^\d,.]")

    def __init__(self, tokenizer, prompt_len: int, every: int = 4):
        self.tok, self.prompt_len, self.every = tokenizer, prompt_len, every

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - self.prompt_len
        if n <= 0 or n % self.every:
            return False
        texts = self.tok.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return all(self.PAT.search(t) is not None for t in texts)


def _read_tripped(model, n: int) -> List[bool]:
    gs = getattr(model, "guard_state", None)
    t  = gs.get("tripped") if isinstance(gs, dict) else None
    if t is None:
        return [False] * n
    if torch.is_tensor(t):
        t = t.detach().cpu().tolist()
    t = [bool(x) for x in t]
    return (t + [False] * n)[:n]


def _generate_batch_full(
    model,
    tokenizer,
    prompts: List[str],
    device,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    desc: str = "gen",
) -> Tuple[List[str], List[bool]]:
    n = len(prompts)
    texts, tripped = [""] * n, [False] * n
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    model.eval()
    with _left_padding(tokenizer):
        for s in tqdm(range(0, n, batch_size), desc=desc, leave=False):
            idx = order[s:s + batch_size]
            enc = tokenizer(
                [prompts[i] for i in idx],
                return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(device)
            plen = enc["input_ids"].shape[1]
            _opt(model, "reset_guard_state")
            stop = StoppingCriteriaList([_AllDoneCriteria(tokenizer, plen)])
            # Multi-shot prompt shielding: protect few-shot demonstrations from steering representation drift
            if hasattr(model, "set_prompt_shield_mask"):
                q_toks = tokenizer.encode("Question:", add_special_tokens=False)
                B, T_prompt = enc["input_ids"].shape
                shield_mask = torch.zeros((B, T_prompt), dtype=torch.bool, device=device)
                has_shield = False
                for b_i in range(B):
                    inp_list = enc["input_ids"][b_i].tolist()
                    last_match = -1
                    for k in range(len(inp_list) - len(q_toks) + 1):
                        if inp_list[k : k + len(q_toks)] == q_toks:
                            last_match = k
                    if last_match > 0:
                        shield_mask[b_i, :last_match] = True
                        has_shield = True
                if has_shield:
                    model.set_prompt_shield_mask(shield_mask)
                else:
                    model.set_prompt_shield_mask(None)

            try:
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stop,
                        do_sample=False,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=6,
                        use_cache=True,
                    )
            finally:
                if hasattr(model, "set_prompt_shield_mask"):
                    model.set_prompt_shield_mask(None)

            trip = _read_tripped(model, len(idx))
            for j, i in enumerate(idx):
                raw       = tokenizer.decode(out[j][plen:], skip_special_tokens=True).strip()
                cleaned   = cut_continuation(raw)
                texts[i]   = _trim_response(cleaned)
                tripped[i] = trip[j]
    return texts, tripped


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device,
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> List[str]:
    return _generate_batch_full(model, tokenizer, prompts, device, max_new_tokens, batch_size)[0]


@torch.no_grad()
def predict_routes(
    model,
    tokenizer,
    prompts: List[str],
    device,
    bs: int,
    n_shot: int = 0,
) -> List[Tuple[int, ...]]:
    """Prefill-time route prediction per prompt under the model's current arm state."""
    fn = getattr(model, "predict_routes", None)
    if fn is None:
        raise NotImplementedError(
            "Phi2WithRSC.predict_routes(input_ids, attention_mask) is required for router arms"
        )
    out: List[Any] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    model.eval()

    # Pass shot-bucket if supported
    shot_bucket = None
    if hasattr(model, "shot_to_bucket"):
        shot_bucket = model.shot_to_bucket(n_shot)
    elif hasattr(model, "router") and hasattr(model.router, "shot_to_bucket"):
        shot_bucket = model.router.shot_to_bucket(n_shot)
    elif n_shot is not None:
        shot_bucket = 0 if n_shot == 0 else (2 if n_shot == 4 else 3)

    with _left_padding(tokenizer):
        for s in range(0, len(order), bs):
            idx   = order[s:s + bs]
            sub_prompts = [prompts[i] for i in idx]
            enc   = tokenizer(
                sub_prompts,
                return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(device)

            # Build question-span mask for router feature pooling
            q_mask = None
            try:
                from src.data.shots import question_span_mask_from_prompts
                q_mask = question_span_mask_from_prompts(tokenizer, sub_prompts, padding_side="left", max_len=2048).to(device)
            except Exception:
                q_mask = None

            try:
                preds = fn(enc["input_ids"], enc["attention_mask"], question_span_mask=q_mask, shot_bucket=shot_bucket)
            except TypeError:
                preds = fn(enc["input_ids"], enc["attention_mask"])

            for i, p in zip(idx, preds):
                route = p["route"] if isinstance(p, dict) and "route" in p else p
                out[i] = tuple(sorted(int(x) for x in route))
    return out

# ----------------------------------------------------------------------------- one arm

def run_arm(
    model,
    tokenizer,
    prompts: List[str],
    contract: dict,
    cfg,
    device,
    which: Optional[str],
    cache: OutputCache,
    shots: int = 0,
    route_key_fn=None,
) -> Dict[str, Any]:
    """Generate (or fetch from cache) every prompt under one arm.

    Tripped items (guards on) are replaced by the cached K=0 generation;
    the raw arm has guards off, so its steered text is always kept.
    """
    n       = len(prompts)
    bs      = batch_size_for(cfg, shots)
    max_new = int(cfg.bench_max_new_tokens)
    guards_on = _arm_guards_enabled(which)

    if which is None:
        apply_baseline(model)
        routes: List[Tuple[int, ...]] = [()] * n
    else:
        apply_contract(model, contract, which)
        _guard_monitor(model, contract, guards_on)
        if which == "fixed":
            routes = [tuple(sorted(int(l) for l in contract["s_fixed_star"]))] * n
        else:
            routes = predict_routes(model, tokenizer, prompts, device, bs, n_shot=shots)

    gate_mode, scale = _arm_gate(contract, which)
    kfn  = route_key_fn or canonical_route_key
    keys = [kfn(r, gate_mode, scale) for r in routes]
    ns   = "guards_on" if guards_on else "guards_off"

    entries: List[Optional[dict]] = [cache.get(prompts[i], keys[i], ns) for i in range(n)]
    miss = [i for i, e in enumerate(entries) if e is None]
    if miss:
        texts, trip = _generate_batch_full(
            model, tokenizer, [prompts[i] for i in miss],
            device, max_new, bs, desc=f"{which or 'baseline'} {shots}-shot",
        )
        for j, i in enumerate(miss):
            e = {"text": texts[j], "tripped": bool(trip[j])}
            cache.put(prompts[i], keys[i], ns, e)
            entries[i] = e

    revert_on = which is not None and guards_on and bool(contract["guards"]["revert_to_k0"])
    tripped   = [bool(e["tripped"]) for e in entries]  # type: ignore[index]

    base_text: Dict[int, str] = {}
    if revert_on:
        pending = []
        for i in range(n):
            if tripped[i]:
                b = cache.get(prompts[i], EMPTY_ROUTE_KEY, ns)
                if b is None:
                    pending.append(i)
                else:
                    base_text[i] = b["text"]
        if pending:
            apply_baseline(model)
            texts, _ = _generate_batch_full(
                model, tokenizer, [prompts[i] for i in pending],
                device, max_new, bs, desc="k0-revert",
            )
            for j, i in enumerate(pending):
                cache.put(prompts[i], EMPTY_ROUTE_KEY, ns, {"text": texts[j], "tripped": False})
                base_text[i] = texts[j]

    final, reverted = [], []
    for i in range(n):
        if revert_on and tripped[i]:
            final.append(base_text[i])
            reverted.append(True)
        else:
            final.append(entries[i]["text"])  # type: ignore[index]
            reverted.append(False)

    return {
        "texts":    final,
        "routes":   [list(r) for r in routes],
        "tripped":  tripped,
        "reverted": reverted,
    }

# ----------------------------------------------------------------------------- summaries

def route_summary(routes: Sequence[Sequence[int]], max_k: int) -> Dict[str, Any]:
    n      = max(1, len(routes))
    counts = Counter(tuple(r) for r in routes)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    top_k   = max([max_k] + [len(r) for r in routes])
    k_dist  = {str(k): sum(1 for r in routes if len(r) == k) / n for k in range(top_k + 1)}
    lf      = Counter(l for r in routes for l in set(r))
    return {
        "abstain_rate":       k_dist.get("0", 0.0),
        "route_entropy_bits": float(entropy),
        "k_dist":             k_dist,
        "layer_freq":         {str(l): c / n for l, c in sorted(lf.items())},
        "n_distinct_routes":  len(counts),
    }


def assemble_shot_block(
    shots: int,
    golds: List[Optional[float]],
    arm_results: Dict[str, dict],
    arm_meta:    Dict[str, dict],
    n_boot:           int   = 10000,
    seed:             int   = 42,
    noninf_margin_pp: float = 1.0,
    max_k:            int   = 4,
) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    bits: Dict[str, Dict[str, List[bool]]] = {"strict": {}, "lenient": {}}

    for key, res in arm_results.items():
        strict, lenient, extracted = [], [], []
        for t, g in zip(res["texts"], golds):
            strict.append(_is_correct(extract_strict(t), g))
            lenient.append(bool(evaluate_response(t, g, min_steps=1)["correct"]))
            extracted.append(extract_final_answer(t))
        bits["strict"][key], bits["lenient"][key] = strict, lenient
        n    = max(1, len(strict))
        summ = route_summary(res["routes"], max_k)
        summ["guard_trip_rate"] = sum(res["tripped"]) / n
        summ["reverted_rate"]   = sum(res["reverted"]) / n
        meta = arm_meta[key]
        arms[key] = {
            "label":             meta["label"],
            "gate_mode":         meta["gate_mode"],
            "guards":            meta["guards"],
            "accuracy_strict":   100.0 * sum(strict) / n,
            "accuracy_lenient":  100.0 * sum(lenient) / n,
            "summary":           summ,
            "correct_strict":    [int(x) for x in strict],
            "correct_lenient":   [int(x) for x in lenient],
            "routes":            res["routes"],
            "tripped":           [int(x) for x in res["tripped"]],
            "reverted":          [int(x) for x in res["reverted"]],
            "responses":         res["texts"],
            "extracted":         extracted,
        }

    stats: Dict[str, Any] = {}
    if "baseline" in arm_results:
        for metric in ("strict", "lenient"):
            treat = {k: v for k, v in bits[metric].items() if k != "baseline"}
            stats[metric] = compare_to_control(
                bits[metric]["baseline"], treat,
                n_boot=n_boot, seed=seed, noninf_margin_pp=noninf_margin_pp,
            )
    return {"shots": shots, "n": len(golds), "gold": golds, "arms": arms, "stats": stats}

# ----------------------------------------------------------------------------- full runner

def _load_gsm8k_test(n: Optional[int], seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    qs, golds = [], []
    for ex in ds:
        qs.append(ex["question"])
        parts = ex["answer"].split("####")
        golds.append(float(parts[1].replace(",", "").strip()) if len(parts) > 1 else None)
    return qs, golds


def cache_fingerprint(contract: dict, cfg) -> str:
    raw = json.dumps([
        contract["steernet"]["sha256"],
        contract["router"]["sha256"],
        contract["gate"]["sha256"],
        contract["steer_scale"],
        contract.get("k_scale_mode"),
        cfg.model_name,
        int(cfg.bench_max_new_tokens),
        "rp1.1-nr6",
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_shot(
    model,
    tokenizer,
    contract: dict,
    cfg,
    device,
    shots: int,
    n_test: Optional[int],
    cache: OutputCache,
    arms: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    qs, golds = _load_gsm8k_test(n_test, seed=42)
    prompts   = [build_prompt(q, n_shot=shots) for q in qs]
    wanted    = set(arms) if arms else {k for k, _, _ in ARM_SPEC}
    wanted.add("baseline")  # needed as control and as the K=0 revert source

    arm_results, arm_meta = {}, {}
    for key, label, which in ARM_SPEC:
        if key not in wanted:
            continue
        print(f"[bench] {shots}-shot arm={label} n={len(prompts)}")
        arm_results[key] = run_arm(
            model, tokenizer, prompts, contract, cfg, device, which, cache, shots
        )
        gm, _ = _arm_gate(contract, which)
        arm_meta[key] = {"label": label, "gate_mode": gm, "guards": _arm_guards_enabled(which)}

    apply_baseline(model)
    block = assemble_shot_block(
        shots, golds, arm_results, arm_meta,
        n_boot=int(cfg.bench_n_boot),
        seed=int(cfg.seed),
        noninf_margin_pp=float(cfg.bench_noninf_margin_pp),
        max_k=int(cfg.max_active_layers),
    )
    block["cache"] = cache.stats()
    return block


def run_full_benchmark(
    model,
    tokenizer,
    contract: dict,
    cfg,
    device,
    shots:     Optional[Sequence[int]] = None,
    secondary: bool                    = False,
    n_test:    Optional[int]           = None,
    cache_dir: Optional[str]           = None,
    arms:      Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    cache  = OutputCache(cache_dir or cfg.bench_cache_dir, cache_fingerprint(contract, cfg))
    record: Dict[str, Any] = {
        "schema_version": 1,
        "created":    datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_name": cfg.model_name,
        "config": {
            "bench_n_boot":          int(cfg.bench_n_boot),
            "bench_max_new_tokens":  int(cfg.bench_max_new_tokens),
            "bench_noninf_margin_pp": float(cfg.bench_noninf_margin_pp),
            "seed":                  int(cfg.seed),
            "max_active_layers":     int(cfg.max_active_layers),
        },
        "contract": {
            "s_fixed_star":       contract["s_fixed_star"],
            "theta_abs":          contract["theta_abs"],
            "gate_adopted":       contract["gate"]["adopted"],
            "rel_norm_cap":       contract["envelope"]["rel_norm_cap"],
            "steer_scale":        contract["steer_scale"],
            "git_commit":         contract["git_commit"],
            "created":            contract["created"],
            "selection_ids_hash": contract["selection_split"]["ids_hash"],
            "steernet_sha256":    contract["steernet"]["sha256"],
            "router_sha256":      contract["router"]["sha256"],
            "gate_sha256":        contract["gate"]["sha256"],
        },
        "shots":     {},
        "secondary": {},
    }
    for k in (shots if shots is not None else cfg.bench_shots):
        record["shots"][str(k)] = run_shot(
            model, tokenizer, contract, cfg, device, int(k), n_test, cache, arms
        )
    if secondary:
        for k in cfg.bench_secondary_shots:
            record["secondary"][str(k)] = run_shot(
                model, tokenizer, contract, cfg, device, int(k),
                int(cfg.bench_secondary_n), cache, arms,
            )
    return record


def write_record(record: Dict[str, Any], path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)
