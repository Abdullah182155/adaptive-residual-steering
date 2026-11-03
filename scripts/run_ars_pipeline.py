"""Kaggle entry point for calibration, S1 training, S5, and GSM8K evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs.config import RSCConfig, apply_overrides
from src.data.dataset import gsm8k_to_cot
from src.deploy.calibrate import (
    apply_contract,
    load_contract,
    measure_envelope,
    write_contract,
)
from src.evaluation.benchmark import run_benchmark, write_record, write_report
from src.models.wrapper import Phi2WithRSC
from src.training.trainer import load_steer_weights, train_phase1_steernet


LEGACY_NOTEBOOK_OVERRIDES = {"catalog_max_k", "s1_k_choices", "s1_onpolicy_fraction"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ARS fixed-route Phase 1 pipeline")
    parser.add_argument(
        "--stage",
        choices=("S1", "S5", "bench", "all", "train", "calibrate"),
        default="all",
    )
    parser.add_argument("--steernet_arch", choices=("v5",), default="v5")
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
    parser.add_argument("--shots", type=int, nargs="+")
    parser.add_argument("--record", default="experiment_record.json")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--arms", default="baseline,fixed")
    parser.add_argument("--ood", default="")
    parser.add_argument("--ood_n", type=int, default=300)
    parser.add_argument("--ood_shots", type=int, nargs="+", default=[0])
    parser.add_argument("--preface", choices=("on", "off"), default="on")
    parser.add_argument("--live_samples", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args(argv)


def build_config(args) -> RSCConfig:
    cfg = RSCConfig(steernet_arch=args.steernet_arch, objective=args.objective)
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


def build_model(cfg, device):
    """Load Phi-2 with an explicit padding ID for recent Transformers releases."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("the tokenizer does not provide an EOS or padding token ID")
    model_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    model_config.pad_token_id = tokenizer.pad_token_id
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=model_config,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model = Phi2WithRSC(backbone, cfg, device)
    return model, tokenizer


def load_benchmark_rows(limit):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    rows = []
    for example in dataset:
        answer = example["answer"].split("####")[-1].strip()
        rows.append({"question": example["question"], "answer": answer})
    return rows


def prepare_dataset_split(dataset, limit: int | None = None):
    """Materialize a dataset split into a list of row dicts."""
    total = len(dataset)
    n = min(limit, total) if limit is not None else total
    if hasattr(dataset, "select"):
        selected = dataset.select(range(n))
        return [selected[i] for i in range(n)]
    return list(dataset[:n])


def run(args) -> int:
    cfg = build_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ARS Phase 1 | device={device} | route={cfg.fixed_layers}")
    model, tokenizer = build_model(cfg, device)
    if stage_requires_checkpoint(args.stage):
        checkpoint = Path(cfg.lora_save_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"S1 checkpoint is required for {args.stage}: {checkpoint}"
            )
        load_steer_weights(model, str(checkpoint), cfg, device)
    stage = {"train": "S1", "calibrate": "S5"}.get(args.stage, args.stage)

    train_data = validation_data = []
    if stage in {"S1", "S5", "all"}:
        raw_train, raw_val = gsm8k_to_cot(tokenizer, cfg)
        train_data = prepare_dataset_split(raw_train, args.s1_train_n)
        validation_data = prepare_dataset_split(raw_val, args.s1_val_n)

    if stage in {"S1", "all"}:
        print("Running S1 fixed-route SteerNet training")
        train_phase1_steernet(model, train_data, validation_data, cfg, device)
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
        record = run_benchmark(
            model,
            tokenizer,
            load_benchmark_rows(args.n_test),
            cfg,
            device,
            shots,
            arms,
            contract,
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
    return stage in {"S5", "bench", "calibrate"}


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception:
        print("\nARS pipeline failed. Full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        raise
