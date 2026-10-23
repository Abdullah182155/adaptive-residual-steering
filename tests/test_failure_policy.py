import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.utils.failure_policy import (
    FatalContractError,
    RunManifest,
    get_manifest,
    set_manifest,
    to_jsonable,
)


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_atomic_and_roundtrip(tmp_path):
    path = str(tmp_path / "m.json")
    m = RunManifest(path=path, policy="warn", run_id="rt1")
    assert _load(path)["schema_version"] == 1

    m.set("config.steer_scale", 0.30)
    m.warn("W_DEAD_LAYERS", "some dead", stage="S3", ratio=torch.tensor(0.5), arr=np.array([1, 2]))
    m.advisory("gate_gap", {"v": np.float32(2.0)}, "informational")
    with m.stage("S1_steernet"):
        m.record_metrics("S1_steernet", loss=0.5)
    with pytest.raises(ValueError):
        with m.stage("S2_synergy"):
            raise ValueError("boom")

    d = _load(path)
    assert d["run_id"] == "rt1"
    assert d["failure_policy"] == "warn"
    assert d["config"]["steer_scale"] == pytest.approx(0.30)
    w = d["warnings"][0]
    assert w["code"] == "W_DEAD_LAYERS" and w["stage"] == "S3"
    assert w["data"]["ratio"] == pytest.approx(0.5) and w["data"]["arr"] == [1, 2]
    assert d["advisories"]["gate_gap"]["value"] == {"v": 2.0}
    assert d["advisories"]["gate_gap"]["interpretation"] == "informational"
    s1 = d["stages"]["S1_steernet"]
    assert s1["status"] == "completed" and s1["t_end"] is not None and s1["metrics"]["loss"] == 0.5
    assert d["stages"]["S2_synergy"]["status"] == "failed"
    # atomic write leaves no temp files behind
    assert sorted(os.listdir(tmp_path)) == ["m.json"]
    assert to_jsonable({"p": tmp_path, "s": {3}}) == {"p": str(tmp_path), "s": [3]}


def test_warn_policy_never_raises(tmp_path):
    m = RunManifest(path=str(tmp_path / "w.json"), policy="warn")
    assert m.diagnostic("W_ROUTER_COLLAPSE", ok=True, message="fine") is True
    assert m.diagnostic("W_ROUTER_COLLAPSE", ok=False, message="collapsed", stage="x", failures=["a"]) is False
    d = _load(str(tmp_path / "w.json"))
    assert [w["code"] for w in d["warnings"]] == ["W_ROUTER_COLLAPSE"]

    s = RunManifest(path=str(tmp_path / "s.json"), policy="strict")
    assert s.diagnostic("W_ROUTER_COLLAPSE", ok=True, message="fine") is True
    with pytest.raises(FatalContractError) as ei:
        s.diagnostic("W_ROUTER_COLLAPSE", ok=False, message="collapsed")
    assert ei.value.code == "W_ROUTER_COLLAPSE"
    assert _load(str(tmp_path / "s.json"))["warnings"][0]["code"] == "W_ROUTER_COLLAPSE"


def test_dead_layer_veto_is_warn_only(tmp_path, monkeypatch):
    from src.training import trainer

    manifest = RunManifest(path=str(tmp_path / "dead.json"), policy="warn")
    set_manifest(manifest)

    class _FakeModel:
        def reset_routing_diagnostics(self):
            pass

        def routing_diagnostics(self):
            return {
                l: {"selection_rate_ema": 0.0, "logit_ema": 0.0, "selected_forwards": 0, "seen_forwards": 10}
                for l in (3, 4, 5, 6)
            }

        def state_dict(self):
            return {"router.w": torch.ones(2), "steer_nets.0.w": torch.zeros(2), "backbone.w": torch.zeros(1)}

    monkeypatch.setattr(trainer, "evaluate_loss", lambda *a, **k: 1.0)
    monkeypatch.setattr(trainer, "epoch_end_accuracy", lambda *a, **k: 0.8)
    cfg = SimpleNamespace(
        failure_policy="warn", router_dead_layer_veto_frac=1.0, accuracy_eval_n=4, seed=0
    )
    save_path = tmp_path / "ckpt" / "best.pt"

    best_acc, best_val, best_state, pctr, improved, val_loss, acc = trainer._evaluate_and_maybe_checkpoint(
        _FakeModel(), None, [], cfg, torch.device("cpu"), str(save_path),
        0.0, float("inf"), 0, True, "unit",
    )
    assert improved is True
    assert save_path.exists()
    assert best_acc == pytest.approx(0.8) and best_state is not None and pctr == 0
    codes = [w["code"] for w in _load(str(tmp_path / "dead.json"))["warnings"]]
    assert "W_DEAD_LAYERS" in codes


def test_train_rsc_completes_under_collapse(tmp_path, tiny_cfg, tiny_model, monkeypatch):
    from src.training import trainer

    manifest = RunManifest(path=tiny_cfg.manifest_path, policy="warn")
    set_manifest(manifest)
    calls = []

    def fake_p1(model, train, ev, cfg, device):
        calls.append("p1")
        return None, 0.5, {"p1_train": [0.5], "p1_val": [0.5]}

    def fake_boot(model, train, ev, cfg, device):
        calls.append("boot")
        return None, 0.1

    def fake_staged(*args, **kwargs):
        calls.append(kwargs["stage_name"])
        return kwargs.get("global_step", 0) + 1, 0.4, 0.3

    def fake_audit(model, cfg):
        return {"passed": False, "reliability_waived": False, "failures": ["collapsed"]}

    monkeypatch.setattr(trainer, "train_phase1_steernet", fake_p1)
    monkeypatch.setattr(trainer, "train_router_bootstrap", fake_boot)
    monkeypatch.setattr(trainer, "run_staged_phase", fake_staged)
    monkeypatch.setattr(trainer, "audit_router_policy_health", fake_audit)

    logs, best_p1, best_p2 = trainer.train_rsc(tiny_model, None, [], [], tiny_cfg, torch.device("cpu"))

    assert logs["manifest_path"] == tiny_cfg.manifest_path
    # Phase 2 (2A, 2B, 2C, 2D) executed despite the collapsed router audit
    assert calls[0] == "p1" and calls[1] == "boot"
    assert len([c for c in calls if c.startswith("PHASE 2")]) == 4
    d = _load(tiny_cfg.manifest_path)
    expected = {"S1_steernet", "S2_synergy", "S1_steernet_finetune", "S3_router", "S4_token_gate", "S3_gate_retune"}
    assert set(d["stages"]) == expected
    assert all(s["status"] == "completed" for s in d["stages"].values())
    assert "W_ROUTER_COLLAPSE" in [w["code"] for w in d["warnings"]]


def test_nonfinite_streak_fatal(tmp_path):
    from src.training.trainer import NonfiniteLossTracker

    manifest = RunManifest(path=str(tmp_path / "nf.json"), policy="warn")
    set_manifest(manifest)
    assert get_manifest() is manifest

    # a finite loss resets the streak
    tracker = NonfiniteLossTracker(limit=20, stage_name="S1_steernet")
    for _ in range(19):
        tracker.observe(bool(torch.isfinite(torch.tensor(float("nan")))))
    tracker.observe(True)
    assert tracker.count == 0

    with pytest.raises(FatalContractError) as ei:
        with manifest.stage("S1_steernet"):
            for _ in range(20):
                tracker.observe(bool(torch.isfinite(torch.tensor(float("nan")))))
    assert ei.value.code == "E_NONFINITE_LOSS_STREAK"
    d = _load(str(tmp_path / "nf.json"))
    assert d["stages"]["S1_steernet"]["status"] == "failed"
    assert d["warnings"][-1]["code"] == "E_NONFINITE_LOSS_STREAK"
