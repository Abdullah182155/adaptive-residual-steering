"""Configuration for the fixed-route Phase 1 ARS pipeline."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_SHOT_AWARE_SCALE = {0: 1.0, 2: 0.50, 4: 0.35}

@dataclass
class RSCConfig:
    model_name: str = "microsoft/phi-2"
    candidate_layers: List[int] = field(default_factory=lambda: [13, 15, 18, 20, 22])
    fixed_layers: List[int] = field(default_factory=lambda: [13, 20, 22])
    steernet_arch: str = "v5"
    lora_rank: int = 64
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    steer_kappa: float = 0.08
    layer_norms_calibration: str = "norms_calibration.json"
    steer_scale: float = 1.0
    shot_aware_deploy: bool = True
    prompt_shielding: bool = True
    batch_size: int = 1
    grad_accum: int = 4
    phase1_epochs: int = 2
    phase1_lr: float = 1e-3
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    max_tok_len: int = 384
    n_samples: int = 7000
    val_split: float = 0.08
    diversify_prompts: bool = False
    checkpoint_dir: str = "checkpoints"
    lora_save_path: str = "checkpoints/steernet_v5.pt"
    contract_path: str = "deployment_contract.json"
    seed: int = 42
    bench_shots: List[int] = field(default_factory=lambda: [0, 2, 4])
    generation_max_new_tokens: int = 256

def shot_aware_scale_map(cfg: RSCConfig) -> Optional[Dict[int, float]]:
    return dict(DEFAULT_SHOT_AWARE_SCALE) if cfg.shot_aware_deploy else None

def decode_only_for_shot(_: RSCConfig, __: int) -> bool:
    return False
