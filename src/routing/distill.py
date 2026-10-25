"""26-subset catalog router distillation engine (M5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis.cf_table import CFTable
from src.configs.config import RSCConfig
from src.routing.catalog import Catalog
from src.utils.failure_policy import RunManifest, get_manifest

_NEG = -1e9          # finite "minus infinity" used to mask disallowed classes during training
_TIE_TOL = 1e-6      # float tolerance for the near-optimal set


# --------------------------------------------------------------------------- targets & losses
def build_soft_targets(
    rewards: torch.Tensor,
    eps: float = 0.01,
    tau: float = 0.01,
    sizes: Optional[torch.Tensor] = None,
    k_free: int = 2,
    c_size: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute soft distillation target distribution q and near-optimal mask.

    rewards: (N, C) float tensor (e.g. NLL gains from CFTable); -inf entries are never near-optimal.
    eps: near-tie threshold
    tau: temperature for softmax over near-optimal set E_n
    sizes: (C,) integer or float tensor of subset sizes |S| for each class
    k_free: number of layers exempt from size penalty (default 2: K<=2 free)
    c_size: penalty per layer exceeding k_free

    Returns:
        q: (N, C) target probabilities
        near_opt: (N, C) bool mask of E_n
    """
    rewards = rewards.float()
    if sizes is not None and c_size > 0.0:
        sizes_t = sizes.to(rewards.device).float()
        excess_k = torch.clamp(sizes_t - float(k_free), min=0.0)
        rewards = rewards - float(c_size) * excess_k.unsqueeze(0)
    best = rewards.max(dim=1, keepdim=True).values
    near_opt = rewards >= (best - eps - _TIE_TOL)
    logits = (rewards - best) / tau                       # <= 0, numerically stable
    logits = logits.masked_fill(~near_opt, float("-inf"))
    q = torch.softmax(logits, dim=1)
    return q, near_opt


def kl_q_p(q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """KL divergence KL(q || p) = sum_c q * (log(q) - log_p).
    Returns mean over batch dimension N.
    """
    if q.shape[0] == 0:
        return q.new_zeros(())
    log_q = torch.log(q.clamp_min(1e-30))
    term = torch.where(q > 0, q * (log_q - log_p), torch.zeros_like(q))
    return term.sum(dim=-1).mean()


def balance_loss(p: torch.Tensor, multi_opt_mask: torch.Tensor) -> torch.Tensor:
    """Switch Transformer load balance loss: C * sum_c (f_c * P_c).
    Computed only over rows with |E_n| >= 2 (multi_opt_mask == True).
    If no row has |E_n| >= 2, returns scalar 0.0.
    """
    if multi_opt_mask is None or p.shape[0] == 0 or not bool(multi_opt_mask.any()):
        return p.new_zeros(())
    ps = p[multi_opt_mask.bool()]
    C = ps.shape[1]
    f = F.one_hot(ps.argmax(dim=-1), C).float().mean(dim=0).detach()   # hard routing fractions
    P = ps.mean(dim=0)                                                  # mean router probability
    return C * (f * P).sum()


def consistency_loss(p_ref: torch.Tensor, p_other: torch.Tensor) -> torch.Tensor:
    """Consistency loss between shot-paired representations of the same question:
    KL(p_ref.detach() || p_other), where p_ref is the 0-shot prediction (teacher).
    """
    if p_ref.shape[0] == 0:
        return p_other.new_zeros(())
    pr = p_ref.detach()
    kl = pr * (torch.log(pr.clamp_min(1e-12)) - torch.log(p_other.clamp_min(1e-12)))
    return kl.sum(dim=-1).mean()


# --------------------------------------------------------------------------- helpers
@dataclass
class DistillReport:
    train_loss: float
    val_loss: float
    top1_agreement: float
    mean_regret: float
    regret_best_fixed: float
    regret_ratio: float
    route_entropy_bits_by_shot: Dict[int, float]
    abstain_rate_by_shot: Dict[int, float]
    class_hist: List[int]


def _to_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_manifest(manifest: Optional[RunManifest]):
    if manifest is not None:
        return manifest
    try:
        return get_manifest()
    except Exception:
        return None


def _advise(manifest, key: str, value: Any, interpretation: str):
    if manifest is not None:
        manifest.advisory(key, value, interpretation)


def _warn(manifest, code: str, message: str):
    if manifest is not None:
        manifest.warn(code, message)


def _table_arrays(table: CFTable):
    rewards = _to_np(table.gain()).astype(np.float32)
    pooled = _to_np(table.pooled).astype(np.float32)
    scalars = _to_np(table.scalars).astype(np.float32)
    n_shot = _to_np(table.n_shot).astype(np.int64)
    raw_qid = _to_np(table.qid)
    try:
        qid = raw_qid.astype(np.int64)
    except (ValueError, TypeError):
        qid = raw_qid.astype(str)
    return rewards, pooled, scalars, n_shot, qid


def _allowed_mask(C: int, class_mask: Optional[Sequence[int]]) -> torch.Tensor:
    if class_mask is None:
        return torch.ones(C, dtype=torch.bool)
    idx = sorted(set(int(c) for c in class_mask))
    if not idx or idx[0] < 0 or idx[-1] >= C:
        raise ValueError(f"class_mask must be a non-empty subset of [0, {C - 1}], got {list(class_mask)}")
    m = torch.zeros(C, dtype=torch.bool)
    m[idx] = True
    return m


def split_by_qid(qid: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split row indices into (train, val) so every qid lives entirely on one side."""
    uniq = np.unique(qid)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    n_val = int(round(val_frac * len(uniq))) if len(uniq) >= 2 else 0
    n_val = min(max(n_val, 1 if (val_frac > 0 and len(uniq) >= 2) else 0), max(len(uniq) - 1, 0))
    val_q = set(uniq[perm[:n_val]].tolist())
    is_val = np.array([q in val_q for q in qid.tolist()], dtype=bool)
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def _reference_rows(qid: np.ndarray, n_shot: np.ndarray) -> np.ndarray:
    """For each row, the index of its qid-group's lowest-shot row (teacher); -1 for the teacher itself."""
    q_list = qid.tolist()
    s_list = n_shot.tolist()
    best: Dict[Any, int] = {}
    for i, (q, s) in enumerate(zip(q_list, s_list)):
        if q not in best or s < s_list[best[q]]:
            best[q] = i
    ref = np.full(len(q_list), -1, dtype=np.int64)
    for i, q in enumerate(q_list):
        if best[q] != i:
            ref[i] = best[q]
    return ref


def _log_probs(router, pooled, scalars, bucket, allowed: torch.Tensor) -> torch.Tensor:
    logits = router.catalog_logits_from_features(pooled, scalars, bucket)
    logits = logits.masked_fill(~allowed.unsqueeze(0), _NEG)
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def predict_classes(router, table: CFTable, indices: Sequence[int],
                    class_mask: Optional[Sequence[int]] = None, chunk: int = 2048) -> np.ndarray:
    """Argmax catalog class for the given table rows (disallowed classes are masked to -inf)."""
    _, pooled, scalars, n_shot, _ = _table_arrays(table)
    idx = np.asarray(indices, dtype=np.int64)
    C = router.catalog_head.out_features
    device = next(router.parameters()).device
    allowed = _allowed_mask(C, class_mask).to(device)
    buckets = router.shot_to_bucket(n_shot)
    was_training = router.training
    router.eval()
    out = []
    for s in range(0, len(idx), chunk):
        rows = torch.from_numpy(idx[s:s + chunk])
        logits = router.catalog_logits_from_features(
            torch.from_numpy(pooled[rows.numpy()]).to(device),
            torch.from_numpy(scalars[rows.numpy()]).to(device),
            buckets[rows].to(device),
        )
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        out.append(logits.argmax(dim=-1).cpu().numpy())
    if was_training:
        router.train()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def router_eval_metrics(router, table: CFTable, indices: Sequence[int], cfg, class_mask: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Evaluate router metrics on a subset of table rows:
    top1 agreement with table argmax, mean regret R* - R[chosen], regret vs best fixed,
    route entropy in bits per shot, abstain rate per shot.

    R* is the best reward among allowed classes. Top-1 agreement is tie-aware: the router is counted
    correct if its choice attains R* exactly. `cfg` is accepted for interface uniformity.
    """
    rewards, _, _, n_shot, _ = _table_arrays(table)
    idx = np.asarray(indices, dtype=np.int64)
    C = rewards.shape[1]
    allowed = _allowed_mask(C, class_mask).numpy()
    n = len(idx)
    if n == 0:
        return {
            "n": 0, "top1_agreement": 0.0, "mean_regret": 0.0, "regret_best_fixed": 0.0,
            "regret_ratio": 0.0, "best_fixed_class": 0,
            "route_entropy_bits_by_shot": {}, "abstain_rate_by_shot": {}, "class_hist": [0] * C,
        }

    chosen = predict_classes(router, table, idx, class_mask)
    R = rewards[idx].astype(np.float64)
    r_star = np.where(allowed[None, :], R, -np.inf).max(axis=1)
    got = R[np.arange(n), chosen]
    regret = r_star - got
    top1 = float((got >= r_star - 1e-9).mean())
    mean_regret = float(regret.mean())

    allowed_idx = np.nonzero(allowed)[0]
    fixed_regrets = (r_star[:, None] - R[:, allowed_idx]).mean(axis=0)
    j = int(fixed_regrets.argmin())
    regret_best_fixed = float(fixed_regrets[j])
    if regret_best_fixed <= 1e-12:
        regret_ratio = 0.0 if mean_regret <= 1e-12 else float("inf")
    else:
        regret_ratio = mean_regret / regret_best_fixed

    shots = n_shot[idx]
    entropy_by_shot: Dict[int, float] = {}
    abstain_by_shot: Dict[int, float] = {}
    for s in sorted(set(shots.tolist())):
        sel = shots == s
        counts = np.bincount(chosen[sel], minlength=C).astype(np.float64)
        pr = counts / counts.sum()
        nz = pr[pr > 0]
        entropy_by_shot[int(s)] = float(-(nz * np.log2(nz)).sum())
        abstain_by_shot[int(s)] = float((chosen[sel] == 0).mean())

    return {
        "n": int(n),
        "top1_agreement": top1,
        "mean_regret": mean_regret,
        "regret_best_fixed": regret_best_fixed,
        "regret_ratio": float(regret_ratio),
        "best_fixed_class": int(allowed_idx[j]),
        "route_entropy_bits_by_shot": entropy_by_shot,
        "abstain_rate_by_shot": abstain_by_shot,
        "class_hist": np.bincount(chosen, minlength=C).astype(int).tolist(),
    }


# --------------------------------------------------------------------------- distillation
def distill_router(
    router,
    table: CFTable,
    cfg: RSCConfig,
    device: torch.device,
    manifest: Optional[RunManifest] = None,
    *,
    class_mask: Optional[Sequence[int]] = None,
    seed: int = 0,
    patience: int = 12,
    batch_qids: int = 64,
    verbose: bool = False,
) -> DistillReport:
    """Distill table rewards into router catalog_head without LLM forwards.

    1. Splits table rows into train / val BY qid (ensuring shot-paired copies of the
       same question stay together in either train or val, no leakage).
    2. Constructs soft targets q via build_soft_targets.
    3. Optimizes router.catalog_head (and router representation) using AdamW with:
       Loss = KL(q || p) + lambda_bal * balance_loss + lambda_cons * consistency_loss.
    4. If class_mask is given (e.g. [0, s_star] for DECISION 2 fixed_abstain mode),
       masks non-allowed classes to -inf.
    5. Early stops on validation KL loss; saves best router weights to cfg.router_catalog_save_path.
    6. Logs advisories to manifest (regret_ratio, etc.).
    """
    manifest = _resolve_manifest(manifest)
    if getattr(router, "catalog_head", None) is None:
        raise ValueError("router has no catalog_head; construct JointLayerRouter(..., n_catalog=C)")

    rewards_np, pooled_np, scalars_np, n_shot_np, qid_np = _table_arrays(table)
    N, C = rewards_np.shape
    if router.catalog_head.out_features != C:
        raise ValueError(f"router.catalog_head has {router.catalog_head.out_features} classes but table has {C}")

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    router.to(device)

    allowed = _allowed_mask(C, class_mask).to(device)
    allowed_idx = allowed.nonzero().squeeze(-1)

    train_idx, val_idx = split_by_qid(qid_np, float(cfg.router_distill_val_frac), seed)
    if len(train_idx) == 0:
        raise ValueError("empty training split")
    if len(val_idx) == 0:
        _warn(manifest, "ROUTER_DISTILL_NO_VAL", "validation split is empty; monitoring on training rows")
        val_idx = train_idx

    pooled_t = torch.from_numpy(pooled_np).to(device)
    scalars_t = torch.from_numpy(scalars_np).to(device)
    bucket_t = router.shot_to_bucket(n_shot_np).to(device)
    rewards_t = torch.from_numpy(rewards_np).to(device)
    rewards_t = rewards_t.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    sizes_t = None
    if hasattr(table, "catalog") and table.catalog is not None and hasattr(table.catalog, "sizes"):
        sizes_t = table.catalog.sizes().to(device)

    c_size = float(getattr(cfg, "router_c_size", 0.0))
    k_free = int(getattr(cfg, "router_k_free", 2))
    q_all, near_all = build_soft_targets(
        rewards_t,
        eps=float(cfg.router_eps_tie),
        tau=float(cfg.router_tau),
        sizes=sizes_t,
        k_free=k_free,
        c_size=c_size,
    )
    multi_all = near_all.sum(dim=1) >= 2
    ref_np = _reference_rows(qid_np, n_shot_np)

    # Standardization statistics come from the training rows only.
    tr_t = torch.from_numpy(train_idx).to(device)
    router.fit_bucket_stats(pooled_t[tr_t], bucket_t[tr_t])

    groups: Dict[Any, List[int]] = {}
    for i in train_idx.tolist():
        groups.setdefault(qid_np[i], []).append(i)
    group_rows = [np.asarray(v, dtype=np.int64) for v in groups.values()]

    for p in router.parameters():
        p.requires_grad = True
    params = [p for p in router.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.router_distill_lr), weight_decay=float(cfg.router_distill_wd))
    lam_bal = float(cfg.router_lambda_bal)
    lam_cons = float(cfg.router_lambda_cons)
    pos = np.full(N, -1, dtype=np.int64)

    def batch_loss(rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        rows_t = torch.from_numpy(rows).to(device)
        log_p = _log_probs(router, pooled_t[rows_t], scalars_t[rows_t], bucket_t[rows_t], allowed)
        p = log_p.exp()
        kl = kl_q_p(q_all[rows_t], log_p)
        bal = balance_loss(p[:, allowed_idx], multi_all[rows_t])
        pos[rows] = np.arange(len(rows))
        refs = ref_np[rows]
        sel = np.nonzero(refs >= 0)[0]
        if len(sel):
            sel = sel[pos[refs[sel]] >= 0]
        if len(sel):
            cons = consistency_loss(p[torch.from_numpy(pos[refs[sel]]).to(device)],
                                    p[torch.from_numpy(sel).to(device)])
        else:
            cons = p.new_zeros(())
        pos[rows] = -1
        return kl + lam_bal * bal + lam_cons * cons, kl

    @torch.no_grad()
    def val_kl(rows_np: np.ndarray, chunk: int = 2048) -> float:
        router.eval()
        total, count = 0.0, 0
        for s in range(0, len(rows_np), chunk):
            rows_t = torch.from_numpy(rows_np[s:s + chunk]).to(device)
            log_p = _log_probs(router, pooled_t[rows_t], scalars_t[rows_t], bucket_t[rows_t], allowed)
            total += float(kl_q_p(q_all[rows_t], log_p)) * len(rows_t)
            count += len(rows_t)
        return total / max(1, count)

    best_val, best_train, best_state, wait = float("inf"), float("inf"), None, 0
    for epoch in range(int(cfg.router_distill_epochs)):
        router.train()
        order = rng.permutation(len(group_rows))
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch_qids):
            rows = np.concatenate([group_rows[g] for g in order[s:s + batch_qids]])
            loss, _ = batch_loss(rows)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        train_loss = tot / max(1, nb)
        v = val_kl(val_idx)
        if verbose and (epoch % 10 == 0 or epoch == int(cfg.router_distill_epochs) - 1):
            print(f"  [distill] epoch {epoch:3d}  train={train_loss:.4f}  val_kl={v:.4f}")
        if v < best_val - 1e-5:
            best_val, best_train, wait = v, train_loss, 0
            best_state = {k: t.detach().clone() for k, t in router.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        router.load_state_dict(best_state)
    router.eval()

    save_path = cfg.router_catalog_save_path
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save({k: t.detach().cpu() for k, t in router.state_dict().items()}, save_path)

    m = router_eval_metrics(router, table, val_idx, cfg, class_mask)
    report = DistillReport(
        train_loss=float(best_train),
        val_loss=float(best_val),
        top1_agreement=m["top1_agreement"],
        mean_regret=m["mean_regret"],
        regret_best_fixed=m["regret_best_fixed"],
        regret_ratio=m["regret_ratio"],
        route_entropy_bits_by_shot=m["route_entropy_bits_by_shot"],
        abstain_rate_by_shot=m["abstain_rate_by_shot"],
        class_hist=m["class_hist"],
    )

    _advise(manifest, "router_distill_regret_ratio", report.regret_ratio,
            "held-out mean regret / best-fixed-subset regret; < 1 means the router beats the best fixed subset")
    _advise(manifest, "router_distill_top1_agreement", report.top1_agreement,
            "fraction of held-out rows where the router picks a table-optimal subset")
    _advise(manifest, "router_distill_abstain_rate_by_shot", report.abstain_rate_by_shot,
            "share of held-out rows routed to the empty subset, per shot count")
    _advise(manifest, "router_distill_class_hist", report.class_hist,
            "held-out routing histogram over catalog classes")
    if report.regret_ratio >= 1.0:
        _warn(manifest, "ROUTER_NO_BETTER_THAN_FIXED",
              f"distilled router regret ratio {report.regret_ratio:.3f} >= 1 on held-out rows")
    return report


def pair_synergy_diagnostics(table: CFTable) -> dict:
    """Re-export of cf_table.pair_synergy for convenient diagnostics."""
    from src.analysis.cf_table import pair_synergy
    return pair_synergy(table)
