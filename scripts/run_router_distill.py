"""CLI: distill a CFTable into the 26-subset catalog router (M5)."""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Distill counterfactual-table rewards into the catalog router (no LLM forwards).")
    p.add_argument("--table_path", type=str, required=True, help="Path to the CFTable .npz built by M1/S0")
    p.add_argument("--out_path", type=str, default=None, help="Output checkpoint (default: cfg.router_catalog_save_path)")
    p.add_argument("--epochs", type=int, default=None, help="Override cfg.router_distill_epochs")
    p.add_argument("--lr", type=float, default=None, help="Override cfg.router_distill_lr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed_abstain", action="store_true",
                   help="DECISION 2 mode: restrict the router to {abstain, best fixed subset s*}")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated candidate layers (only needed if the table does not expose them)")
    p.add_argument("--n_layers", type=int, default=32, help="Backbone depth (Phi-2 = 32)")
    return p.parse_args()


def _load_table(path):
    from src.analysis.cf_table import CFTable
    for name in ("load", "from_npz", "from_file", "read"):
        fn = getattr(CFTable, name, None)
        if callable(fn):
            return fn(path)
    raise RuntimeError("CFTable exposes no load / from_npz / from_file / read classmethod")


def _resolve_layers(table, args):
    if args.layers:
        return [int(x) for x in args.layers.split(",") if x.strip()]
    for owner in (table, getattr(table, "catalog", None)):
        for attr in ("target_layers", "layers"):
            v = getattr(owner, attr, None) if owner is not None else None
            if v is not None:
                return [int(x) for x in v]
    raise RuntimeError("Could not infer candidate layers from the table; pass --layers 13,15,18,20,22")


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import numpy as np
    import torch
    from src.configs.config import RSCConfig
    from src.routing.router import JointLayerRouter
    from src.routing.distill import distill_router
    from src.utils.failure_policy import get_manifest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RSCConfig()
    if args.epochs is not None:
        cfg.router_distill_epochs = args.epochs
    if args.lr is not None:
        cfg.router_distill_lr = args.lr
    if args.out_path is not None:
        cfg.router_catalog_save_path = args.out_path

    print(f"[*] Router distillation | device={device} | table={args.table_path}")
    table = _load_table(args.table_path)
    n_catalog = len(table.catalog.subsets) if hasattr(table, "catalog") and hasattr(table.catalog, "subsets") else len(table.catalog)
    pooled = table.pooled
    hidden_dim = int(pooled.shape[1])
    layers = _resolve_layers(table, args)

    router = JointLayerRouter(
        layers, hidden_dim, args.n_layers, cfg.max_active_layers, n_catalog=n_catalog,
    ).to(device)

    class_mask = None
    if args.fixed_abstain:
        gain = table.gain()
        gain = gain.detach().cpu().numpy() if torch.is_tensor(gain) else np.asarray(gain)
        s_star = 1 + int(np.argmax(gain.mean(axis=0)[1:]))
        class_mask = [0, s_star]
        print(f"[*] fixed_abstain: restricting router to classes {class_mask}")

    manifest = get_manifest()
    report = distill_router(router, table, cfg, device, manifest, class_mask=class_mask, seed=args.seed, verbose=True)

    print("\n[*] Distillation report")
    print(f"    train_loss       : {report.train_loss:.4f}")
    print(f"    val_loss (KL)    : {report.val_loss:.4f}")
    print(f"    top1_agreement   : {report.top1_agreement:.3f}")
    print(f"    mean_regret      : {report.mean_regret:.4f}")
    print(f"    regret_best_fixed: {report.regret_best_fixed:.4f}")
    print(f"    regret_ratio     : {report.regret_ratio:.3f}")
    print(f"    entropy bits/shot: {report.route_entropy_bits_by_shot}")
    print(f"    abstain rate/shot: {report.abstain_rate_by_shot}")
    print(f"    class_hist       : {report.class_hist}")
    print(f"[*] Saved best router weights to {cfg.router_catalog_save_path}")


if __name__ == "__main__":
    main()
