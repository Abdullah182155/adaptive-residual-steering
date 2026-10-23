"""Counterfactual (CF) table builder for the S0 / S0b audit.

For every example and every catalog subset S (index 0 == empty set) we run a teacher-forced
forward with the routing matrix forced to S and record the per-token answer NLL.  All C subsets
of one example are batched as replicated rows of a single forward (rows_per_forward at a time).

Sign conventions
    ans_nll   : mean NLL over target tokens (lower is better)
    rewards() : -ans_nll - c * |S|
    gain()    : ans_nll[:, 0] - ans_nll[:, c]   (> 0 means subset c helps vs. the base model)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.configs.config import RSCConfig
from src.data.shots import build_shot_example, from_dataset_row
from src.routing.catalog import Catalog
from src.utils.failure_policy import FatalContractError, get_manifest

_MAX_AUDIT_LEN = 1536
_ROW_FIELDS = ("qid", "split", "n_shot", "ans_nll", "token_nll", "token_mask", "token_ids", "pooled", "scalars")


# --------------------------------------------------------------------------------------
# small io helpers
# --------------------------------------------------------------------------------------
def _atomic_savez(path: str, **arrays) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:           # file handle => numpy does not append ".npz"
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _pad_len(a: np.ndarray, L: int, fill) -> np.ndarray:
    if a.shape[-1] == L:
        return a
    pad = [(0, 0)] * (a.ndim - 1) + [(0, L - a.shape[-1])]
    return np.pad(a, pad, constant_values=fill)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# CFTable
# --------------------------------------------------------------------------------------
@dataclass
class CFTable:
    catalog: Catalog
    qid: np.ndarray            # (N,) str
    split: np.ndarray          # (N,) str
    n_shot: np.ndarray         # (N,) int64
    steer_scale: float
    ans_nll: np.ndarray        # (N, C) float32, mean NLL over target tokens
    token_nll: np.ndarray      # (N, C, L) float32, zero where token_mask is False
    token_mask: np.ndarray     # (N, L) bool, True on real target tokens
    token_ids: np.ndarray      # (N, L) int64, -1 on padding
    pooled: np.ndarray         # (N, d) float32, router._last_pooled (question span)
    scalars: np.ndarray        # (N, 3) float32, router._last_scalars
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.qid.shape[0])

    def rewards(self, c: float = 1e-3) -> np.ndarray:
        """R(S) = -NLL_answer(S) - c * |S|, shape (N, C)."""
        sizes = self.catalog.sizes().numpy().astype(np.float32)
        return (-self.ans_nll - np.float32(c) * sizes[None, :]).astype(np.float32)

    def gain(self) -> np.ndarray:
        """NLL[empty] - NLL[S], shape (N, C). Positive == steering with S helps."""
        return (self.ans_nll[:, :1] - self.ans_nll).astype(np.float32)

    def select(self, mask) -> "CFTable":
        idx = np.asarray(mask)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        rows = {f: getattr(self, f)[idx] for f in _ROW_FIELDS}
        return CFTable(catalog=self.catalog, steer_scale=self.steer_scale, meta=dict(self.meta), **rows)

    def save(self, path: str) -> None:
        _atomic_savez(
            path,
            **{f: getattr(self, f) for f in _ROW_FIELDS},
            catalog_layers=np.asarray(self.catalog.layers, dtype=np.int64),
            catalog_max_k=np.asarray(self.catalog.max_k, dtype=np.int64),
            steer_scale=np.asarray(self.steer_scale, dtype=np.float64),
            meta_json=np.asarray(json.dumps(self.meta, sort_keys=True, default=str)),
        )

    @classmethod
    def load(cls, path: str) -> "CFTable":
        with np.load(path, allow_pickle=False) as z:
            catalog = Catalog.build([int(x) for x in z["catalog_layers"]], int(z["catalog_max_k"]))
            rows = {f: z[f] for f in _ROW_FIELDS}
            return cls(
                catalog=catalog,
                steer_scale=float(z["steer_scale"]),
                meta=json.loads(str(z["meta_json"])),
                **rows,
            )


# --------------------------------------------------------------------------------------
# checkpoint loading (SteerNets only)
# --------------------------------------------------------------------------------------
def load_steernets_for_audit(model, path: str) -> dict:
    """Load ONLY ``steer_nets.*`` tensors from ``path`` into ``model``.

    Admitted layers == model.cfg.catalog_layers.  A missing admitted layer (or any missing tensor
    of one) is a contract violation -> FatalContractError("E_CKPT_CORRUPT").  Missing non-admitted
    candidate layers only produce a manifest warning (W_LAYER_MISSING).
    """
    manifest = get_manifest()
    if not os.path.isfile(path):
        raise FatalContractError("E_CKPT_CORRUPT", f"checkpoint not found: {path}")
    sha = _sha256_file(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # truncated / not a torch archive / unsafe pickle
        raise FatalContractError("E_CKPT_CORRUPT", f"cannot read checkpoint {path}: {exc}") from exc
    if isinstance(raw, dict) and "steer_state" in raw:
        raw = raw["steer_state"]
    if not isinstance(raw, dict):
        raise FatalContractError("E_CKPT_CORRUPT", f"checkpoint {path} does not hold a state_dict")

    steer = {k: v for k, v in raw.items() if k.startswith("steer_nets.") and torch.is_tensor(v)}
    target = [int(l) for l in model.target_layers]
    admitted = {int(l) for l in getattr(model.cfg, "catalog_layers", ())}
    stray = sorted(admitted - set(target))
    if stray:
        raise FatalContractError("E_CKPT_CORRUPT", f"admitted layers {stray} are not model.target_layers {target}")

    model_keys = set(model.state_dict().keys())
    to_load: Dict[str, torch.Tensor] = {}
    loaded_layers: List[int] = []
    missing_layers: List[int] = []
    for l in target:
        prefix = f"steer_nets.{l}."
        expected = {k for k in model_keys if k.startswith(prefix)}
        got = {k for k in steer if k.startswith(prefix)}
        absent = sorted(expected - got)
        if absent:
            if l in admitted:
                raise FatalContractError(
                    "E_CKPT_CORRUPT",
                    f"admitted layer {l} is missing {len(absent)} SteerNet tensors in {path} (e.g. {absent[0]})",
                )
            missing_layers.append(l)
            manifest.warn("W_LAYER_MISSING", f"non-admitted candidate layer {l} has no SteerNet weights in {path}")
            continue
        for k in expected:
            to_load[k] = steer[k]
        loaded_layers.append(l)

    bad = [k for k, v in to_load.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise FatalContractError("E_CKPT_CORRUPT", f"non-finite SteerNet tensors in {path} (e.g. {bad[0]})")
    try:
        model.load_state_dict(to_load, strict=False)
    except RuntimeError as exc:  # shape mismatch etc.
        raise FatalContractError("E_CKPT_CORRUPT", f"SteerNet tensors in {path} do not fit the model: {exc}") from exc

    meta = {
        "path": path,
        "sha256": sha,
        "n_tensors": len(to_load),
        "layers_loaded": sorted(loaded_layers),
        "layers_missing": sorted(missing_layers),
        "admitted_layers": sorted(admitted),
    }
    model._audit_ckpt_meta = dict(meta)   # picked up by build_cf_table for shard fingerprints / meta
    return meta


# --------------------------------------------------------------------------------------
# example sampling
# --------------------------------------------------------------------------------------
def _make_example(row, tokenizer, n_shot: int, qid: str, split: str) -> Optional[dict]:
    try:
        q, steps, fin = row["question"], row["steps_list"], row["final_num"]
        ex = build_shot_example(tokenizer, q, [str(s) for s in steps], str(fin), n_shot)
    except KeyError:
        if n_shot != 0:
            return None
        ex = from_dataset_row(row)
        ex["n_shot"] = 0
    ex["qid"] = qid
    ex["split"] = split
    return ex


def sample_audit_examples(train_data, eval_data, tokenizer, cfg: RSCConfig, seed: int = 0) -> List[dict]:
    """Audit rows per ``cfg.cf_audit_plan`` from train_data plus ``cfg.cf_heldout_n`` 0-shot rows from eval_data.

    All shot counts walk the SAME shuffled order of train indices, so the 4-/8-shot rows are (up to
    over-length skips) a paired subset of the 0-shot rows.  qid is ``train:<idx>`` / ``eval:<idx>``;
    (qid, n_shot) is unique.
    """
    manifest = get_manifest()
    out: List[dict] = []

    perm = list(range(len(train_data)))
    random.Random(seed).shuffle(perm)
    for n_shot, n_rows in cfg.cf_audit_plan:
        got, skipped = 0, 0
        for idx in perm:
            if got >= n_rows:
                break
            ex = _make_example(train_data[idx], tokenizer, int(n_shot), f"train:{idx}", "train")
            if ex is None or ex["input_ids"].numel() > _MAX_AUDIT_LEN:
                skipped += 1
                continue
            out.append(ex)
            got += 1
        if got < n_rows:
            manifest.warn("W_AUDIT_SHORT", f"{n_shot}-shot: wanted {n_rows} rows, got {got} (skipped {skipped})")

    if cfg.cf_heldout_n > 0 and eval_data is not None:
        eperm = list(range(len(eval_data)))
        random.Random(seed + 1).shuffle(eperm)
        got = 0
        for idx in eperm:
            if got >= cfg.cf_heldout_n:
                break
            ex = _make_example(eval_data[idx], tokenizer, 0, f"eval:{idx}", "heldout")
            if ex is None or ex["input_ids"].numel() > _MAX_AUDIT_LEN:
                continue
            out.append(ex)
            got += 1
        if got < cfg.cf_heldout_n:
            manifest.warn("W_AUDIT_SHORT", f"heldout: wanted {cfg.cf_heldout_n} rows, got {got}")
    return out


# --------------------------------------------------------------------------------------
# core evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _eval_example(model, ex: dict, catalog: Catalog, rows_per_forward: int, device: torch.device) -> dict:
    ids = ex["input_ids"].to(device).long()
    labels = ex["labels"].to(device).long()
    qm = ex["question_span_mask"].to(device).bool()
    T = ids.numel()

    pos = torch.arange(T, device=device)
    tgt_pos = torch.nonzero((labels != -100) & (pos >= 1)).squeeze(-1)
    if tgt_pos.numel() == 0:
        raise ValueError(f"example {ex.get('qid')} has no target tokens")
    tgt_ids = ids[tgt_pos]
    L, C = int(tgt_pos.numel()), len(catalog)

    tok = torch.zeros(C, L, dtype=torch.float32)
    pooled = scalars = None
    for lo in range(0, C, rows_per_forward):
        cls = torch.arange(lo, min(lo + rows_per_forward, C), dtype=torch.long)
        R = int(cls.numel())
        model.set_routing_override("matrix", catalog.to_candidate_matrix(cls, model.target_layers))
        model.set_router_context_mask(qm.unsqueeze(0).expand(R, -1).contiguous())
        ids_b = ids.unsqueeze(0).expand(R, -1).contiguous()
        am = torch.ones_like(ids_b)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(input_ids=ids_b, attention_mask=am).logits
        if pooled is None:   # router input is computed pre-steering at the first candidate layer => subset independent
            pooled = model.router._last_pooled[0].detach().float().cpu().numpy()
            scalars = model.router._last_scalars[0].detach().float().cpu().numpy()
        for r in range(R):
            lg = logits[r, tgt_pos - 1, :].float()
            tok[lo + r] = F.cross_entropy(lg, tgt_ids, reduction="none").cpu()
        del logits

    if not bool(torch.isfinite(tok).all()):
        get_manifest().warn("W_NONFINITE_NLL", f"non-finite token NLL for {ex.get('qid')} (n_shot={ex.get('n_shot')})")
    return {
        "qid": str(ex.get("qid", "")),
        "split": str(ex.get("split", "")),
        "n_shot": int(ex.get("n_shot", 0)),
        "tok": tok.numpy(),
        "ans": tok.mean(dim=1).numpy(),
        "ids": tgt_ids.cpu().numpy().astype(np.int64),
        "pooled": pooled.astype(np.float32),
        "scalars": scalars.astype(np.float32),
    }


def _stack_rows(rows: List[dict]) -> Dict[str, np.ndarray]:
    L = max(r["tok"].shape[1] for r in rows)
    M, C = len(rows), rows[0]["tok"].shape[0]
    token_nll = np.zeros((M, C, L), np.float32)
    token_mask = np.zeros((M, L), bool)
    token_ids = np.full((M, L), -1, np.int64)
    for i, r in enumerate(rows):
        n = r["tok"].shape[1]
        token_nll[i, :, :n] = r["tok"]
        token_mask[i, :n] = True
        token_ids[i, :n] = r["ids"]
    return {
        "qid": np.asarray([r["qid"] for r in rows]),
        "split": np.asarray([r["split"] for r in rows]),
        "n_shot": np.asarray([r["n_shot"] for r in rows], dtype=np.int64),
        "ans_nll": np.stack([r["ans"] for r in rows]).astype(np.float32),
        "token_nll": token_nll,
        "token_mask": token_mask,
        "token_ids": token_ids,
        "pooled": np.stack([r["pooled"] for r in rows]),
        "scalars": np.stack([r["scalars"] for r in rows]),
    }


def _concat_shards(shards: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    L = max(s["token_mask"].shape[-1] for s in shards)
    fills = {"token_nll": 0.0, "token_mask": False, "token_ids": -1}
    out = {}
    for f in _ROW_FIELDS:
        parts = [_pad_len(s[f], L, fills[f]) if f in fills else s[f] for s in shards]
        out[f] = np.concatenate(parts, axis=0)
    return out


def _shard_fingerprint(chunk: List[dict], catalog: Catalog, steer_scale: float, seed: int, ckpt_sha: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({
        "layers": list(catalog.layers), "max_k": catalog.max_k,
        "scale": repr(float(steer_scale)), "seed": int(seed), "ckpt": ckpt_sha,
    }, sort_keys=True).encode())
    for ex in chunk:
        h.update(str(ex.get("qid")).encode())
        h.update(str(int(ex.get("n_shot", 0))).encode())
        for key in ("input_ids", "labels", "question_span_mask"):
            h.update(ex[key].cpu().numpy().tobytes())
    return h.hexdigest()


def _try_load_shard(path: str, fp: str, n_expected: int, n_classes: int) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.isfile(path):
        return None
    manifest = get_manifest()
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) != fp:
                manifest.warn("W_CF_SHARD_STALE", f"{os.path.basename(path)} fingerprint mismatch; recomputing")
                return None
            data = {f: z[f] for f in _ROW_FIELDS}
    except Exception as exc:
        manifest.warn("W_CF_SHARD_UNREADABLE", f"{os.path.basename(path)} unreadable ({exc}); recomputing")
        return None
    if data["ans_nll"].shape != (n_expected, n_classes):
        manifest.warn("W_CF_SHARD_STALE", f"{os.path.basename(path)} has unexpected shape; recomputing")
        return None
    return data


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------
def build_cf_table(model, tokenizer, examples: List[dict], catalog: Catalog, cfg: RSCConfig,
                   device: torch.device, steer_scale: float, out_dir: str, seed: int = 0) -> CFTable:
    """Build (resumably) the counterfactual table for ``examples`` x ``catalog``.

    Shards ``cf_table_part_{k:04d}.npz`` are written every ``cfg.cf_shard_size`` examples; a shard is
    reused only if its fingerprint (catalog, steer_scale, seed, SteerNet ckpt sha, example contents)
    matches.  The combined table is saved to ``<out_dir>/cf_table.npz``.
    """
    if not examples:
        raise ValueError("build_cf_table: no examples")
    manifest = get_manifest()
    missing = [l for l in catalog.layers if l not in set(model.target_layers)]
    if missing:
        raise FatalContractError("E_LAYER_MISSING", f"catalog layers {missing} not in model.target_layers {list(model.target_layers)}")

    os.makedirs(out_dir, exist_ok=True)
    rpf_table = {int(a): int(b) for a, b in cfg.cf_rows_per_forward}
    warned_shots = set()

    def rows_per_forward(n_shot: int) -> int:
        if n_shot in rpf_table:
            return rpf_table[n_shot]
        if n_shot not in warned_shots:
            warned_shots.add(n_shot)
            manifest.warn("W_CF_RPF_DEFAULT", f"no rows_per_forward for n_shot={n_shot}; using the smallest configured value")
        larger = [v for k, v in rpf_table.items() if k >= n_shot]
        return min(larger) if larger else min(rpf_table.values())

    ckpt_sha = (getattr(model, "_audit_ckpt_meta", None) or {}).get("sha256")
    was_training = model.training
    prev_override = getattr(model, "_routing_override", None)
    prev_scale = getattr(model, "steer_scale", None)

    model.eval()
    not_eval = [n for n, m in model.named_modules() if m.training]
    if not_eval:
        raise FatalContractError("E_NOT_EVAL", f"submodules still in training mode after eval(): {not_eval[:3]}")
    torch.manual_seed(seed)

    S = max(1, int(cfg.cf_shard_size))
    n_shards = math.ceil(len(examples) / S)
    shards: List[Dict[str, np.ndarray]] = []
    try:
        with model.counterfactual_mode():
            model.set_steer_scale(steer_scale)
            for k in range(n_shards):
                chunk = examples[k * S:(k + 1) * S]
                path = os.path.join(out_dir, f"cf_table_part_{k:04d}.npz")
                fp = _shard_fingerprint(chunk, catalog, steer_scale, seed, ckpt_sha)
                shard = _try_load_shard(path, fp, len(chunk), len(catalog))
                if shard is None:
                    rows = [
                        _eval_example(model, ex, catalog, rows_per_forward(int(ex.get("n_shot", 0))), device)
                        for ex in tqdm(chunk, desc=f"cf shard {k + 1}/{n_shards}", leave=False)
                    ]
                    shard = _stack_rows(rows)
                    _atomic_savez(path, **shard, fingerprint=np.asarray(fp))
                shards.append(shard)
    finally:
        if prev_override is None:
            model.set_routing_override(None)
        else:
            model.set_routing_override(prev_override[0], prev_override[1])
        if prev_scale is not None:
            try:
                model.set_steer_scale(prev_scale)
            except Exception:
                pass
        model.train(was_training)

    rows = _concat_shards(shards)
    n_shot_counts = {int(s): int((rows["n_shot"] == s).sum()) for s in np.unique(rows["n_shot"])}
    meta = {
        "seed": int(seed),
        "steer_scale": float(steer_scale),
        "n_examples": int(len(rows["qid"])),
        "n_shot_counts": n_shot_counts,
        "catalog_layers": list(catalog.layers),
        "catalog_max_k": int(catalog.max_k),
        "shard_size": S,
        "ckpt_sha256": ckpt_sha,
        "model_name": getattr(cfg, "model_name", None),
    }
    table = CFTable(catalog=catalog, steer_scale=float(steer_scale), meta=meta, **rows)
    table.save(os.path.join(out_dir, "cf_table.npz"))
    return table


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
def pair_synergy(table: CFTable) -> Dict[Tuple[Tuple[int, ...], int], np.ndarray]:
    """For every non-empty S and layer j not in S with |S|+1 <= max_k:

        synergy[(S, j)] = NLL[S u {j}] - NLL[S] - (NLL[{j}] - NLL[empty])      shape (N,)

    Negative == layers combine better than additively (synergy); positive == interference.
    """
    cat = table.catalog
    nll = table.ans_nll.astype(np.float64)
    index = {s: i for i, s in enumerate(cat.subsets)}
    base = nll[:, index[()]]
    out: Dict[Tuple[Tuple[int, ...], int], np.ndarray] = {}
    for S in cat.subsets:
        if not S or len(S) + 1 > cat.max_k:
            continue
        for j in cat.layers:
            if j in S:
                continue
            union = tuple(sorted(S + (j,)))
            out[(S, j)] = nll[:, index[union]] - nll[:, index[S]] - (nll[:, index[(j,)]] - base)
    return out
