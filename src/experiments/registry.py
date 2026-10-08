import os
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn

def _get_git_commit() -> str:
    """Safely retrieves current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"

def _get_hardware_info() -> Dict[str, Any]:
    """Retrieves hardware and GPU execution environment telemetry."""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        })
    else:
        info["device_name"] = "CPU"
    return info

def _serialize_config(cfg: Any) -> Dict[str, Any]:
    """Safely converts RSCConfig dataclass or dictionary to JSON-serializable dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        raw = cfg
    elif hasattr(cfg, "__dict__"):
        raw = vars(cfg)
    else:
        return {"repr": str(cfg)}

    clean = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = [
                elem if isinstance(elem, (int, float, str, bool, type(None))) else str(elem)
                for elem in v
            ]
        elif isinstance(v, dict):
            clean[k] = {str(dk): str(dv) for dk, dv in v.items()}
        else:
            clean[k] = str(v)
    return clean


class ExperimentRun:
    """Manages tracking, logging, and artifact persistence for a single experiment."""

    def __init__(self, record_path: str, data: Dict[str, Any]):
        self.record_path = record_path
        self.data = data
        self.run_dir = os.path.dirname(record_path)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

    @property
    def experiment_id(self) -> str:
        return self.data["experiment_id"]

    @property
    def tag(self) -> str:
        return self.data.get("tag", "")

    def log_config(self, cfg: Any):
        """Records hyperparameter configuration."""
        self.data["config"] = _serialize_config(cfg)
        self.save()

    def log_training_phase(self, phase_name: str, metrics: Dict[str, Any]):
        """Records phase training metrics (loss, accuracy, duration)."""
        if "training_logs" not in self.data:
            self.data["training_logs"] = {}
        clean_metrics = {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in metrics.items()
        }
        self.data["training_logs"][phase_name] = clean_metrics
        self.save()

    def log_routing_diagnostics(self, diagnostics: Dict[str, Any]):
        """Records router telemetry (layer selection, K distribution, synergy)."""
        if "routing_diagnostics" not in self.data:
            self.data["routing_diagnostics"] = {}
        for k, v in diagnostics.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                self.data["routing_diagnostics"][k] = v
            else:
                self.data["routing_diagnostics"][k] = str(v)
        self.save()

    def log_benchmark_result(
        self,
        stage_name: str,
        accuracy: float,
        n_shot: int = 0,
        dataset: str = "gsm8k",
        tokens_per_sec: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Records evaluation benchmark performance under a standardized hierarchy."""
        if "benchmark_results" not in self.data:
            self.data["benchmark_results"] = {}
        if dataset not in self.data["benchmark_results"]:
            self.data["benchmark_results"][dataset] = {}
        if stage_name not in self.data["benchmark_results"][dataset]:
            self.data["benchmark_results"][dataset][stage_name] = {}

        shot_key = f"{n_shot}_shot"
        entry: Dict[str, Any] = {
            "accuracy": float(accuracy),
            "n_shot": n_shot,
            "tokens_per_sec": float(tokens_per_sec) if tokens_per_sec is not None else None,
        }
        if extra_metrics:
            entry.update({k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in extra_metrics.items()})

        self.data["benchmark_results"][dataset][stage_name][shot_key] = entry
        self.save()

    def save_weights(self, model: nn.Module, filename: str = "final_ars_weights.pt") -> str:
        """Saves trainable ARS steering/gate/router weights inside the isolated run directory."""
        dest_path = os.path.join(self.run_dir, "checkpoints", filename)
        state = {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if "steer_net" in k or "gate" in k or "router" in k
        }
        torch.save(state, dest_path)
        self.data["artifacts"]["weights_path"] = dest_path
        self.save()
        return dest_path

    def load_weights(
        self,
        model: nn.Module,
        filename: str = "final_ars_weights.pt",
        device: Optional[torch.device] = None,
    ) -> bool:
        """Restores ARS weights from this specific experiment run."""
        weights_path = os.path.join(self.run_dir, "checkpoints", filename)
        if not os.path.exists(weights_path):
            weights_path = self.data.get("artifacts", {}).get("weights_path", "")
        if not os.path.exists(weights_path):
            return False

        map_loc = device if device is not None else torch.device("cpu")
        state = torch.load(weights_path, map_location=map_loc)
        if isinstance(state, dict) and "steer_state" in state:
            state = state["steer_state"]
        model.load_state_dict(state, strict=False)
        return True

    def register_chart(self, chart_name: str, source_path: Optional[str] = None) -> str:
        """Registers a generated publication chart inside the experiment artifacts."""
        target_path = os.path.join(self.run_dir, "charts", f"{chart_name}.png")
        if source_path and os.path.exists(source_path) and os.path.abspath(source_path) != os.path.abspath(target_path):
            shutil.copy2(source_path, target_path)

        if "charts" not in self.data["artifacts"]:
            self.data["artifacts"]["charts"] = []
        if target_path not in self.data["artifacts"]["charts"]:
            self.data["artifacts"]["charts"].append(target_path)
        self.save()
        return target_path

    def finalize(self, status: str = "completed", summary_notes: str = ""):
        """Marks experiment completed and refreshes hardware telemetry."""
        self.data["status"] = status
        self.data["completed_at"] = datetime.now(timezone.utc).isoformat()
        if summary_notes:
            self.data["summary_notes"] = summary_notes
        self.data["hardware"] = _get_hardware_info()
        self.save()

    def save(self):
        """Persists the experiment record JSON to disk."""
        with open(self.record_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


class ExperimentRegistry:
    """Global experiment catalog managing run creation, indexing, and querying."""

    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = os.path.abspath(base_dir)
        self.runs_dir = os.path.join(self.base_dir, "runs")
        self.index_path = os.path.join(self.base_dir, "registry.json")
        os.makedirs(self.runs_dir, exist_ok=True)
        self._ensure_index()

    def _ensure_index(self):
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"experiments": []}, f, indent=2)

    def _read_index(self) -> List[Dict[str, Any]]:
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("experiments", [])
        except Exception:
            return []

    def _write_index(self, index: List[Dict[str, Any]]):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump({"experiments": index}, f, indent=2)

    def create_experiment(
        self,
        tag: str = "ars_run",
        config: Any = None,
        notes: str = "",
        custom_id: Optional[str] = None,
    ) -> ExperimentRun:
        """Creates a new isolated experiment run with unique ID and registered index entry."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        exp_id = custom_id if custom_id else f"exp_{timestamp}_{tag}"
        run_dir = os.path.join(self.runs_dir, exp_id)
        os.makedirs(run_dir, exist_ok=True)

        record_path = os.path.join(run_dir, "experiment_record.json")
        data: Dict[str, Any] = {
            "experiment_id": exp_id,
            "tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "status": "running",
            "notes": notes,
            "hardware": _get_hardware_info(),
            "config": _serialize_config(config),
            "training_logs": {},
            "routing_diagnostics": {},
            "benchmark_results": {},
            "artifacts": {
                "run_dir": run_dir,
                "record_path": record_path,
                "weights_path": None,
                "charts": [],
            },
        }

        run = ExperimentRun(record_path, data)
        run.save()

        # Update registry global index
        index = self._read_index()
        # Avoid duplicate entries
        index = [e for e in index if e.get("experiment_id") != exp_id]
        index.append({
            "experiment_id": exp_id,
            "tag": tag,
            "created_at": data["created_at"],
            "status": "running",
            "git_commit": data["git_commit"],
            "record_path": record_path,
        })
        self._write_index(index)
        return run

    def get_experiment(self, exp_id: str) -> Optional[ExperimentRun]:
        """Loads an existing experiment run by its ID."""
        run_dir = os.path.join(self.runs_dir, exp_id)
        record_path = os.path.join(run_dir, "experiment_record.json")
        if not os.path.exists(record_path):
            return None
        with open(record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentRun(record_path, data)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Returns summaries of all recorded experiments sorted chronologically."""
        return sorted(self._read_index(), key=lambda x: x.get("created_at", ""), reverse=True)

    def get_latest_experiment(self) -> Optional[ExperimentRun]:
        """Retrieves the most recently created experiment run."""
        experiments = self.list_experiments()
        if not experiments:
            return None
        return self.get_experiment(experiments[0]["experiment_id"])


# Module-level default singleton registry
_default_registry = ExperimentRegistry()

def create_experiment(tag: str = "ars_run", config: Any = None, notes: str = "", custom_id: Optional[str] = None) -> ExperimentRun:
    return _default_registry.create_experiment(tag=tag, config=config, notes=notes, custom_id=custom_id)

def get_experiment(exp_id: str) -> Optional[ExperimentRun]:
    return _default_registry.get_experiment(exp_id)

def list_experiments() -> List[Dict[str, Any]]:
    return _default_registry.list_experiments()

def get_latest_experiment() -> Optional[ExperimentRun]:
    return _default_registry.get_latest_experiment()
