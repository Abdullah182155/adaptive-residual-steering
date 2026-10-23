from __future__ import annotations

import csv
import contextlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from src.evaluation.answers import cut_continuation, extract_numeric_answer
from src.evaluation.verifier import build_prompt, extract_and_verify_equations

# Marker whose number has at least begun (used for stopping; grace tokens finish the number).
_ANSWER_START_RE = re.compile(r"####\s*\$?-?\d|[Ff]inal\s+answer\s*[:\-]?\s*\$?-?\d")
# Marker with the complete number (used for truncation).
_ANSWER_FULL_RE = re.compile(r"(?:####|[Ff]inal\s+answer\s*[:\-]?)\s*\$?-?\d[\d,]*(?:\.\d+)?")
_RUNAWAY = "\nQuestion:"
_TOL = 0.01


@dataclass
class OnPolicyChain:
    qid: int
    sample_idx: int
    shot: int
    prompt: str
    text: str
    pred_strict: Optional[float]
    pred_lenient: Optional[float]
    gold: float
    correct: bool
    n_tokens: int
    div_line: Optional[int]  # index of first line that differs from a correct sibling (pairs only)


class BatchAnswerStopping(StoppingCriteria):
    """Per-row batch stopping criteria for HuggingFace generate.

    Tracks each row independently: a row is finished on EOS, on an answer marker
    ('#### <num>', 'Final answer: <num>') once `grace_tokens` more tokens have been
    produced (so the number can complete), or immediately on runaway continuation
    ('\\nQuestion:'). Returns True only when every row in the batch is finished.
    """

    def __init__(self, tokenizer, prompt_len: int, grace_tokens: int = 12, window: int = 32):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.grace_tokens = int(grace_tokens)
        self.window = int(window)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.row_done: List[bool] = []
        self._marker_at: List[Optional[int]] = []

    def _init_rows(self, n: int) -> None:
        if len(self.row_done) != n:
            self.row_done = [False] * n
            self._marker_at = [None] * n

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        n = input_ids.shape[0]
        self._init_rows(n)
        cur_len = input_ids.shape[1] - self.prompt_len
        if cur_len <= 0:
            return False
        new_ids = input_ids[:, self.prompt_len:]
        for b in range(n):
            if self.row_done[b]:
                continue
            row = new_ids[b]
            if self.eos_id is not None and bool((row == self.eos_id).any()):
                self.row_done[b] = True
                continue
            if self._marker_at[b] is None:
                text = self.tokenizer.decode(row[-self.window:], skip_special_tokens=True)
                if _RUNAWAY in text:
                    self.row_done[b] = True
                    continue
                if _ANSWER_START_RE.search(text):
                    self._marker_at[b] = cur_len
            if self._marker_at[b] is not None and cur_len >= self._marker_at[b] + self.grace_tokens:
                self.row_done[b] = True
        return all(self.row_done)


# --------------------------------------------------------------------------- helpers

def _enc(tokenizer, text: str) -> List[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def _first_diff_line(good: str, bad: str) -> Optional[int]:
    gl, bl = good.split("\n"), bad.split("\n")
    for i, (a, b) in enumerate(zip(gl, bl)):
        if a.strip() != b.strip():
            return i
    if len(gl) != len(bl):
        return min(min(len(gl), len(bl)), len(bl) - 1)
    return None


def _first_failing_equation_line(text: str) -> Optional[int]:
    for i, line in enumerate(text.split("\n")):
        total, correct, _ = extract_and_verify_equations(line)
        if total > correct:
            return i
    return None


def _div_line(good: str, bad: str) -> Optional[int]:
    """First line of `bad` differing from `good`; falls back to first failing-equation line."""
    d = _first_diff_line(good, bad)
    return d if d is not None else _first_failing_equation_line(bad)


def _truncate_chain(text: str) -> str:
    idx = text.find(_RUNAWAY)
    if idx >= 0:
        text = text[:idx]
    text = cut_continuation(text)
    m = _ANSWER_FULL_RE.search(text)
    if m:
        text = text[: m.end()]
    return text.strip()


def _parse_num(v: Any) -> float:
    return float(str(v).replace(",", "").replace("$", "").strip())


def _gold_of(q: Mapping[str, Any]) -> float:
    if q.get("gold") is not None:
        return _parse_num(q["gold"])
    if q.get("final_num") not in (None, ""):
        return _parse_num(q["final_num"])
    if "answer" in q:
        return _parse_num(str(q["answer"]).split("####")[-1])
    raise KeyError("question mapping needs one of: gold, final_num, answer")


@contextlib.contextmanager
def _seeded(seed: int) -> Iterator[None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _append_jsonl(path: str, chains: Sequence[OnPolicyChain]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in chains:
            f.write(json.dumps(asdict(c)) + "\n")
        f.flush()


_TRUE = {"1", "true", "t", "yes", "y", "correct"}
_FALSE = {"0", "false", "f", "no", "n", "incorrect"}


# --------------------------------------------------------------------------- store

class OnPolicyStore:
    def __init__(self, chains: Optional[List[OnPolicyChain]] = None):
        self.chains: List[OnPolicyChain] = chains if chains is not None else []

    def save(self, path: str) -> None:
        """Save store as JSONL to path (creates parent dir if needed)."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for c in self.chains:
                f.write(json.dumps(asdict(c)) + "\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "OnPolicyStore":
        """Load store from JSONL file (a truncated trailing line from a crash is skipped)."""
        names = {f.name for f in fields(OnPolicyChain)}
        chains: List[OnPolicyChain] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chains.append(OnPolicyChain(**{k: v for k, v in d.items() if k in names}))
        return cls(chains)

    def correct_chains(self) -> List[OnPolicyChain]:
        return [c for c in self.chains if c.correct]

    def pairs(self) -> List[Tuple[OnPolicyChain, OnPolicyChain]]:
        """(correct, incorrect) pairs; first correct chain of a question is paired with each
        incorrect chain, and the returned incorrect copy carries div_line."""
        by_q: Dict[int, List[OnPolicyChain]] = {}
        for c in self.chains:
            by_q.setdefault(c.qid, []).append(c)
        out: List[Tuple[OnPolicyChain, OnPolicyChain]] = []
        for qid, cs in by_q.items():
            good = [c for c in cs if c.correct]
            bad = [c for c in cs if not c.correct]
            if not good or not bad:
                continue
            g0 = good[0]
            for b in bad:
                out.append((g0, replace(b, div_line=_div_line(g0.text, b.text))))
        return out

    def stats(self) -> Dict[str, Any]:
        n_total = len(self.chains)
        qids = {c.qid for c in self.chains}
        per_shot_tot: Dict[int, int] = {}
        per_shot_ok: Dict[int, int] = {}
        by_q: Dict[int, List[bool]] = {}
        for c in self.chains:
            per_shot_tot[c.shot] = per_shot_tot.get(c.shot, 0) + 1
            per_shot_ok[c.shot] = per_shot_ok.get(c.shot, 0) + int(c.correct)
            by_q.setdefault(c.qid, []).append(c.correct)
        return {
            "n_total": n_total,
            "n_correct": sum(1 for c in self.chains if c.correct),
            "n_questions": len(qids),
            "n_questions_with_pair": sum(1 for v in by_q.values() if any(v) and not all(v)),
            "pass_rate_per_shot": {s: per_shot_ok[s] / per_shot_tot[s] for s in sorted(per_shot_tot)},
        }

    _SPOT_COLS = ["qid", "sample_idx", "shot", "prompt", "text", "pred_strict",
                  "pred_lenient", "gold", "correct", "human_pred"]

    def export_spotcheck(self, n: int, path: str, seed: int = 42) -> None:
        rng = random.Random(seed)
        picked = rng.sample(self.chains, min(n, len(self.chains)))
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._SPOT_COLS)
            w.writeheader()
            for c in picked:
                w.writerow({
                    "qid": c.qid, "sample_idx": c.sample_idx, "shot": c.shot,
                    "prompt": c.prompt, "text": c.text,
                    "pred_strict": "" if c.pred_strict is None else c.pred_strict,
                    "pred_lenient": "" if c.pred_lenient is None else c.pred_lenient,
                    "gold": c.gold, "correct": c.correct, "human_pred": "",
                })

    @staticmethod
    def spotcheck_agreement(path: str) -> float:
        """Fraction of filled rows where human_pred (true/false) equals `correct`."""
        agree, total = 0, 0
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                hp = (row.get("human_pred") or "").strip().lower()
                if not hp:
                    continue
                if hp in _TRUE:
                    h = True
                elif hp in _FALSE:
                    h = False
                else:
                    raise ValueError(f"unrecognised human_pred value: {hp!r}")
                c = (row["correct"] or "").strip().lower() in _TRUE
                agree += int(h == c)
                total += 1
        if total == 0:
            raise ValueError("no human_pred values filled in")
        return agree / total

    def build_training_rows(
        self,
        tokenizer,
        cfg,
        gold_rows: Sequence[dict],
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[dict], List[dict]]:
        """Build (train_rows, val_rows). Prompt tokens (incl. few-shot exemplars) get label -100."""
        max_len = int(getattr(cfg, "onpolicy_max_tok_len", 1280))
        gold_mix = float(getattr(cfg, "onpolicy_gold_mix", 0.0))
        rng = random.Random(seed)

        def pack(chain: OnPolicyChain) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, List[int]]]:
            p_ids = _enc(tokenizer, chain.prompt)
            t_ids = _enc(tokenizer, chain.text)
            if not t_ids or len(p_ids) + len(t_ids) > max_len:
                return None
            ids = torch.tensor(p_ids + t_ids, dtype=torch.long)
            lbl = torch.tensor([-100] * len(p_ids) + t_ids, dtype=torch.long)
            return ids, lbl, len(p_ids), t_ids

        pairs = self.pairs()
        paired_pos = {(g.qid, g.sample_idx) for g, _ in pairs}
        on_rows: List[dict] = []

        for good, bad in pairs:
            pos, neg = pack(good), pack(bad)
            if pos is None or neg is None:
                continue
            ids, lbl, _, _ = pos
            n_ids, n_lbl, n_plen, _ = neg
            mask = torch.zeros(len(n_ids), dtype=torch.bool)
            if bad.div_line is not None:
                lines = bad.text.split("\n")
                if 0 <= bad.div_line < len(lines):
                    start = sum(len(x) + 1 for x in lines[: bad.div_line])
                    end = start + len(lines[bad.div_line])
                    a = n_plen + len(_enc(tokenizer, bad.text[:start]))
                    b = n_plen + len(_enc(tokenizer, bad.text[:end]))
                    a, b = min(a, len(n_ids)), min(b, len(n_ids))
                    mask[a:b] = True
                    mask &= n_lbl != -100
            on_rows.append({
                "input_ids": ids, "labels": lbl, "is_onpolicy": True,
                "shot": good.shot, "qid": good.qid,
                "neg_input_ids": n_ids, "neg_labels": n_lbl, "div_token_mask": mask,
            })

        for c in self.correct_chains():
            if (c.qid, c.sample_idx) in paired_pos:
                continue
            packed = pack(c)
            if packed is None:
                continue
            ids, lbl, _, _ = packed
            on_rows.append({"input_ids": ids, "labels": lbl, "is_onpolicy": True,
                            "shot": c.shot, "qid": c.qid})

        # gold mixing: gold share == gold_mix of (on-policy + gold)
        gold_pool = list(gold_rows)
        if not on_rows or not gold_pool or gold_mix <= 0:
            n_gold = 0
        elif gold_mix >= 1:
            n_gold = len(gold_pool)
        else:
            n_gold = min(len(gold_pool), int(round(len(on_rows) * gold_mix / (1.0 - gold_mix))))
        rng.shuffle(gold_pool)
        gold_out: List[dict] = []
        for row in gold_pool[:n_gold]:
            ids = torch.as_tensor(row["input_ids"]).long()
            lbl = torch.as_tensor(row["labels"]).long()
            if "attention_mask" in row:
                keep = torch.as_tensor(row["attention_mask"]).bool()
                ids, lbl = ids[keep], lbl[keep]
            gold_out.append({"input_ids": ids, "labels": lbl, "is_onpolicy": False,
                             "shot": 0, "qid": int(row.get("qid", -1))})

        # split: on-policy by question id (no leakage between siblings), gold by row
        qids = sorted({r["qid"] for r in on_rows})
        rng.shuffle(qids)
        n_val_q = int(round(len(qids) * val_fraction))
        if val_fraction > 0 and len(qids) >= 2:
            n_val_q = max(1, n_val_q)
        val_q = set(qids[:n_val_q])
        n_val_g = int(round(len(gold_out) * val_fraction))
        train = [r for r in on_rows if r["qid"] not in val_q] + gold_out[n_val_g:]
        val = [r for r in on_rows if r["qid"] in val_q] + gold_out[:n_val_g]
        rng.shuffle(train)
        rng.shuffle(val)
        return train, val


# --------------------------------------------------------------------------- sampling

def _generate_batch(model, tokenizer, prompts: List[str], cfg, device: torch.device) -> List[str]:
    orig_padding_side = tokenizer.padding_side
    was_training = model.training
    try:
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        stopping = StoppingCriteriaList([BatchAnswerStopping(tokenizer, prompt_len)])
        model.eval()
        with torch.no_grad(), model.hooks_disabled():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=cfg.onpolicy_temperature,
                max_new_tokens=cfg.onpolicy_max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        return [tokenizer.decode(out[b, prompt_len:], skip_special_tokens=True) for b in range(out.shape[0])]
    finally:
        tokenizer.padding_side = orig_padding_side
        if was_training:
            model.train()


def sample_chains(
    model,
    tokenizer,
    questions: Sequence[Mapping[str, Any]],
    cfg,
    device: torch.device,
    *,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> OnPolicyStore:
    """Sample reasoning chains from the un-steered model (see module spec).

    Each question mapping needs 'question' plus one of 'gold' / 'final_num' / 'answer';
    optional 'qid' (defaults to position). Shot count is shots_mix[qid % len(shots_mix)].
    """
    shots = tuple(cfg.onpolicy_shots_mix) or (0,)
    n0, n1 = cfg.onpolicy_samples
    bs = max(1, int(cfg.onpolicy_batch_size))
    qs = list(questions)
    limit = getattr(cfg, "onpolicy_n_questions", 0)
    if limit and limit > 0:
        qs = qs[:limit]

    records: List[Dict[str, Any]] = []
    for pos, q in enumerate(qs):
        qid = int(q.get("qid", pos))
        shot = int(shots[qid % len(shots)])
        records.append({
            "qid": qid, "shot": shot, "gold": _gold_of(q),
            "prompt": build_prompt(str(q["question"]), n_shot=shot),
        })

    store = OnPolicyStore()
    if out_path and os.path.exists(out_path):
        store = OnPolicyStore.load(out_path)
    done = {(c.qid, c.sample_idx) for c in store.chains}

    def run(pending: List[Tuple[Dict[str, Any], int]], round_no: int) -> None:
        pending.sort(key=lambda it: it[0]["shot"])  # stable: batch similar prompt lengths
        for b_idx, start in enumerate(range(0, len(pending), bs)):
            batch = pending[start:start + bs]
            batch_seed = (seed * 1_000_003 + round_no * 10_007 + b_idx) % (2 ** 31 - 1)
            with _seeded(batch_seed):
                raws = _generate_batch(model, tokenizer, [r["prompt"] for r, _ in batch], cfg, device)
            new: List[OnPolicyChain] = []
            for (rec, idx), raw in zip(batch, raws):
                text = _truncate_chain(raw)
                strict = extract_numeric_answer(text, lenient=False)
                lenient = extract_numeric_answer(text, lenient=True)
                ok = any(p is not None and abs(p - rec["gold"]) <= _TOL for p in (strict, lenient))
                new.append(OnPolicyChain(
                    qid=rec["qid"], sample_idx=idx, shot=rec["shot"], prompt=rec["prompt"],
                    text=text, pred_strict=strict, pred_lenient=lenient, gold=rec["gold"],
                    correct=bool(ok), n_tokens=len(_enc(tokenizer, text)), div_line=None,
                ))
            store.chains.extend(new)
            done.update((c.qid, c.sample_idx) for c in new)
            if out_path:
                _append_jsonl(out_path, new)

    # round 1: n0 samples per question
    run([(r, i) for r in records for i in range(n0) if (r["qid"], i) not in done], 1)

    # round 2: +n1 samples for questions with no correct chain so far
    has_correct = {c.qid for c in store.chains if c.correct}
    run([(r, i) for r in records if r["qid"] not in has_correct
         for i in range(n0, n0 + n1) if (r["qid"], i) not in done], 2)

    return store
