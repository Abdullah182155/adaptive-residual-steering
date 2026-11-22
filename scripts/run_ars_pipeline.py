"""Compatibility CLI for fixed-route, optional routing/gating, and evaluation stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Any

_verbosity = "verbose" if "--verbose" in sys.argv else "quiet"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.runtime import configure_pipeline_runtime
configure_pipeline_runtime(verbose=_verbosity == "verbose")

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.configs.config import RSCConfig, apply_overrides
from src.data.dataset import gsm8k_to_cot
from src.data.benchmark import load_gsm8k_test_rows, materialize_dataset_split
from src.deploy.calibrate import (
    apply_contract,
    load_contract,
    measure_envelope,
    write_contract,
)
from src.evaluation.benchmark import run_benchmark, write_record, write_report
from src.models.wrapper import Phi2WithRSC
from src.models.loading import build_frozen_phi2
from src.training.trainer import load_steer_weights, train_phase1_steernet
from src.training.contrastive import train_phase1c_contrastive
from src.training.star import train_phase1r_star
from src.training.router_trainer import (
    train_phase2_router,
    save_router_weights,
    load_router_weights,
)
from src.training.gate_trainer import train_phase2b_gate, load_gate_weights


LEGACY_NOTEBOOK_OVERRIDES = {"catalog_max_k", "s1_k_choices", "s1_onpolicy_fraction"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ARS fixed-route Phase 1 pipeline")
    parser.add_argument(
        "--stage",
        choices=("S1", "S1C", "S1R", "S2", "S2B", "S5", "bench", "all", "train", "calibrate", "router", "gate"),
        default="all",
    )
    parser.add_argument(
        "--steering_architecture",
        "--steernet_arch",
        dest="steernet_arch",
        choices=(
            "legacy_feature_residual",
            "v5",
            "fixed_route_low_rank_residual",
            "v2",
            "prompt_conditioned_residual",
        ),
        default="v5",
        help=(
            "fixed steering architecture; use fixed_route_low_rank_residual for "
            "the rank-16 calibrated adapter or prompt_conditioned_residual for "
            "the rank-32 prompt-conditioned adapter; legacy aliases remain supported"
        ),
    )
    parser.add_argument("--objective", choices=("nll",), default="nll")
    parser.add_argument("--s1_epochs", type=int)
    parser.add_argument("--s1_lr", type=float)
    parser.add_argument("--s1_train_n", type=int)
    parser.add_argument("--s1_val_n", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--steer_scale", type=float)
    parser.add_argument("--s_fixed")
    parser.add_argument("--calib_sel_n", type=int)
    parser.add_argument("--contract")
    parser.add_argument("--n_test", type=int)
    parser.add_argument("--test_seed", type=int, default=42)
    parser.add_argument("--shots", type=int, nargs="+")
    parser.add_argument("--record", default="experiment_record.json")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--arms",
        default="baseline,fixed",
        help=(
            "Comma-separated benchmark arms; for prompt-conditioned steering, "
            "strength_fixed:1.0 is the fixed-strength reference and steer_only "
            "uses the learned prompt-conditioned strength"
        ),
    )
    parser.add_argument("--ood", default="")
    parser.add_argument("--ood_n", type=int, default=300)
    parser.add_argument("--ood_shots", type=int, nargs="+", default=[0])
    parser.add_argument("--preface", choices=("on", "off"), default="on")
    parser.add_argument("--live_samples", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable full HuggingFace / transformers logging output (weight bars, downloads, etc.)",
    )
    args = parser.parse_args(argv)
    architecture_aliases = {
        "legacy_feature_residual": "v5",
        "fixed_route_low_rank_residual": "v2",
    }
    requested_architecture = args.steernet_arch
    args.architecture_id = {
        "legacy_feature_residual": "ars.legacy_feature_residual",
        "v5": "ars.legacy_feature_residual",
        "fixed_route_low_rank_residual": "ars.fixed_route_low_rank_residual",
        "v2": "ars.fixed_route_low_rank_residual",
        "prompt_conditioned_residual": "ars.prompt_conditioned_residual",
    }[requested_architecture]
    args.steernet_arch = architecture_aliases.get(requested_architecture, requested_architecture)
    return args


def build_config(args) -> RSCConfig:
    cfg = RSCConfig(steernet_arch=args.steernet_arch, objective=args.objective)
    if args.steernet_arch in {"v2", "prompt_conditioned_residual"}:
        cfg.fixed_layers = [12, 15, 18, 21]
        cfg.candidate_layers = sorted(set(cfg.candidate_layers) | set(cfg.fixed_layers))
        cfg.lora_rank = 16 if args.steernet_arch == "v2" else 32
        cfg.rho_max = 0.10
        cfg.steernet_norm_stat = "p99_norm"
        cfg.steernet_use_mag_head = False
        cfg.steernet_use_conv = False
    values = {
        "phase1_epochs": args.s1_epochs,
        "phase1_lr": args.s1_lr,
        "batch_size": args.batch_size,
        "steer_scale": args.steer_scale,
    }
    for name, value in values.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.s_fixed is not None:
        cfg.fixed_layers = [int(layer.strip()) for layer in args.s_fixed.split(",")]
    if args.contract is not None:
        cfg.contract_path = args.contract
    cfg.record_path = args.record
    accepted = []
    for override in args.overrides:
        key = override.split("=", 1)[0]
        if key in LEGACY_NOTEBOOK_OVERRIDES:
            print(f"Ignoring retired Phase 2 notebook override: {override}")
        else:
            accepted.append(override)
    apply_overrides(cfg, accepted)
    cfg.validate()
    return cfg


def build_model(
    cfg: RSCConfig, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    """Compatibility entry point for constructing the frozen Phi-2 model."""
    return build_frozen_phi2(
        cfg,
        device,
        tokenizer_loader=AutoTokenizer,
        config_loader=AutoConfig,
        model_loader=AutoModelForCausalLM,
        wrapper_factory=Phi2WithRSC,
    )


def load_benchmark_rows(limit: int | None, seed: int = 42) -> list[dict[str, object]]:
    """Compatibility entry point for deterministic GSM8K test sampling."""
    return load_gsm8k_test_rows(limit, seed=seed, dataset_loader=load_dataset)


def prepare_dataset_split(
    dataset: Any, limit: int | None = None
) -> list[dict[str, object]]:
    """Compatibility entry point for ordered dataset materialization."""
    return materialize_dataset_split(dataset, limit=limit)


def _load_arm_weights(model, arm: str, cfg: RSCConfig, device: torch.device) -> None:
    """Load the checkpoint belonging to a benchmark arm before it is evaluated."""
    if arm in {"baseline", "identity"}:
        return

    if (
        arm in {"fixed", "steer_only", "random", "router"}
        or arm.startswith(("steer_gain:", "strength_fixed:"))
    ):
        checkpoint = Path(cfg.lora_save_path)
    elif arm in {"gate", "full"}:
        checkpoint = Path(cfg.steernet_gated_save_path)
    else:
        raise ValueError(f"unsupported benchmark arm: {arm}")

    if not checkpoint.exists():
        raise FileNotFoundError(f"SteerNet checkpoint for arm '{arm}' not found: {checkpoint}")
    load_steer_weights(model, str(checkpoint), cfg, device)


def run(args) -> int:
    cfg = build_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"ARS fixed-route steering | architecture={args.architecture_id} "
        f"device={device} | route={cfg.fixed_layers}"
    )
    model, tokenizer = build_model(cfg, device)
    if stage_requires_checkpoint(args.stage):
        checkpoint = Path(cfg.lora_save_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"S1 checkpoint is required for {args.stage}: {checkpoint}"
            )
        load_steer_weights(model, str(checkpoint), cfg, device)
    stage = {"train": "S1", "calibrate": "S5", "router": "S2", "gate": "S2B"}.get(args.stage, args.stage)

    train_data = validation_data = []
    if stage in {"S1", "S1C", "S1R", "S2", "S2B", "S5", "all"}:
        raw_train, raw_val = gsm8k_to_cot(tokenizer, cfg)
        train_data = prepare_dataset_split(raw_train, args.s1_train_n)
        validation_data = prepare_dataset_split(raw_val, args.s1_val_n)

    if stage in {"S1", "all"}:
        print("Running S1 fixed-route SteerNet training")
        train_phase1_steernet(model, train_data, validation_data, cfg, device, tokenizer=tokenizer)
    if stage in {"S1C"}:
        print("Running S1C contrastive residual training")
        train_phase1c_contrastive(model, train_data, validation_data, cfg, device, tokenizer=tokenizer)
    if stage in {"S1R"}:
        print("Running S1R Self-Taught Reasoning (STaR)")
        train_phase1r_star(model, train_data, validation_data, cfg, device, tokenizer=tokenizer)
    if stage == "S2":
        print("Running S2 Phase 2 Router training")
        # Load S1 checkpoint when running S2 standalone.
        checkpoint = Path(cfg.lora_save_path)
        if checkpoint.exists():
            load_steer_weights(model, str(checkpoint), cfg, device)
        else:
            print(f"[warning] No S1 checkpoint found at {checkpoint}, training router from scratch")
        router_result = train_phase2_router(
            model, train_data, validation_data, cfg, device, tokenizer=tokenizer
        )
        save_router_weights(
            router_result["router"], router_result["gate"], cfg, cfg.router_save_path
        )
        print(f"Wrote router checkpoint: {cfg.router_save_path}")

    if stage == "S2B":
        print("Running S2B Phase 2B Gate training")
        checkpoint = Path(cfg.lora_save_path)
        if checkpoint.exists():
            load_steer_weights(model, str(checkpoint), cfg, device)
        else:
            print(f"[warning] No S1 checkpoint found at {checkpoint}, using current weights")
        # Enable gate on the model before training
        cfg.use_gate = True
        gate_result = train_phase2b_gate(
            model, train_data, validation_data, cfg, device, tokenizer=tokenizer
        )
        if gate_result:
            best_imp = gate_result.get("best_val_improvement", "N/A")
            if isinstance(best_imp, float):
                print(f"[S2B] best_val_improvement={best_imp:+.4f}")
            else:
                print(f"[S2B] best_val_improvement={best_imp}")

    if stage in {"S5", "all"}:
        print("Running S5 deployment-envelope calibration")
        rows = (
            validation_data[: args.calib_sel_n] if args.calib_sel_n else validation_data
        )
        contract = write_contract(
            cfg.contract_path,
            cfg=cfg,
            envelope=measure_envelope(model, rows, cfg, device),
        )
        print(f"Wrote deployment contract: {cfg.contract_path}")
    if stage in {"bench", "all"}:
        print("Running paired GSM8K benchmark")
        contract = load_contract(cfg.contract_path)
        apply_contract(model, contract)
        shots = args.shots or cfg.bench_shots
        arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]

        # Gate modules are only needed for explicitly requested gate arms.
        if any(arm in {"gate", "full"} for arm in arms):
            import torch.nn as nn_mod
            from src.gating.gate import RSCUsefulnessGate as _Gate
            gate_ckpt_path = Path(getattr(cfg, "gate_save_path", "checkpoints/gate.pt"))
            if gate_ckpt_path.exists():
                hidden_size = model.base_model.config.hidden_size
                model.gates = nn_mod.ModuleDict({
                    str(layer): _Gate(hidden_size, cfg).to(device)
                    for layer in model.target_layers
                })
                load_gate_weights(model, str(gate_ckpt_path), device)
                target_layers = model.target_layers
                gates_ref = model.gates
                def _gate_provider(position, hidden, delta):
                    return gates_ref[str(target_layers[position])](hidden, delta)
                model.set_gate_provider(_gate_provider)
                print(f"Loaded gate: {gate_ckpt_path}")
            else:
                raise FileNotFoundError(f"Gate checkpoint for requested gate arm not found: {gate_ckpt_path}")

        # Load router checkpoint if "router" arm requested
        router_prefill_fn = None
        if "router" in arms:
            router_ckpt = Path(cfg.router_save_path)
            if router_ckpt.exists():
                _, _, router_prefill_fn = load_router_weights(model, cfg, str(router_ckpt), device)
                print(f"Loaded router checkpoint: {router_ckpt}")
            else:
                print(f"[warning] Router arm requested but no checkpoint found at {router_ckpt}; skipping")
                arms = [a for a in arms if a != "router"]
        record = run_benchmark(
            model,
            tokenizer,
            load_benchmark_rows(args.n_test, seed=args.test_seed),
            cfg,
            device,
            shots,
            arms,
            contract,
            router_prefill_fn=router_prefill_fn,
            arm_setup=lambda arm: _load_arm_weights(model, arm, cfg, device),
            test_seed=args.test_seed,
        )
        write_record(record, cfg.record_path)
        if args.report:
            write_report(record, cfg.report_dir)
        print(f"Wrote benchmark record: {cfg.record_path}")
        if args.ood:
            from src.evaluation.ood import load_ood_rows

            for dataset_name in args.ood.split(","):
                dname = dataset_name.strip()
                rows = load_ood_rows(dname, args.ood_n)
                ood_record = run_benchmark(
                    model, tokenizer, rows, cfg, device, args.ood_shots, arms, contract, suite=dname
                )
                rec_path = f"experiment_record_{dname}.json"
                write_record(ood_record, rec_path)
                print(f"Wrote OOD benchmark record: {rec_path}")
                if args.report:
                    write_report(ood_record, f"{cfg.report_dir}/ood_{dname}")
    return 0


def stage_requires_checkpoint(stage: str) -> bool:
    """Separate S5 and benchmark invocations reload the S1 adapter checkpoint."""
    return stage in {"S5", "bench", "calibrate", "S1C", "S1R", "S2", "S2B", "router", "gate"}


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception:
        print("\nARS pipeline failed. Full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        raise
