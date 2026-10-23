import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ARS Stage-0 signal audit (Decision 1 engine)")
    p.add_argument("--ckpt", type=str, required=True, help="SteerNet checkpoint to audit")
    p.add_argument("--out_dir", type=str, default="./reports/s0_audit")
    p.add_argument("--steer_scale", type=float, default=None, help="Defaults to the config value")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--resume", action="store_true", help="Resume a partially built CF table")
    return p.parse_args(argv)


def main():
    args = parse_args()

    # Heavy imports deferred so --help stays cheap.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.configs.config import RSCConfig
    from src.models.wrapper import Phi2WithRSC
    from src.models.contracts import run_contract_checks
    from src.data.dataset import gsm8k_to_cot
    from src.analysis.cf_table import build_cf_table, load_steernets_for_audit, sample_audit_examples
    from src.routing.catalog import Catalog
    from src.analysis.signal_audit import compute_signal_audit, write_audit
    from src.utils.failure_policy import get_manifest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RSCConfig()
    cfg.audit_n_boot = args.n_boot
    steer_scale = args.steer_scale if args.steer_scale is not None else getattr(cfg, "steer_scale", 1.0)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = get_manifest()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)
    model = Phi2WithRSC(base_model, cfg, device=device)

    load_steernets_for_audit(model, args.ckpt)
    run_contract_checks(model, tokenizer, device)

    train_data, eval_data = gsm8k_to_cot(tokenizer, cfg)
    examples = sample_audit_examples(train_data, eval_data, tokenizer, cfg)
    catalog = Catalog.build(cfg.catalog_layers, cfg.catalog_max_k)

    kwargs = dict(steer_scale=steer_scale, out_dir=args.out_dir)
    table = build_cf_table(model, tokenizer, examples, catalog, cfg, device, **kwargs)

    result = compute_signal_audit(table, cfg)
    write_audit(
        result, manifest,
        os.path.join(args.out_dir, "audit_s0.json"),
        os.path.join(args.out_dir, "AUDIT_REPORT_S0.md"),
    )
    print(result.decision_1)


if __name__ == "__main__":
    main()
