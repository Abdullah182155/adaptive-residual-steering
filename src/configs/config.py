import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class RSCConfig:
    """Hyperparameter configuration for Adaptive Residual Steering (ARS / RSC)."""
    model_name: str = "microsoft/phi-2"
    # Layer positions are intentionally discovered from the backbone depth.
    # The router decides which candidates to use for each forward pass and selects at most max_active_layers.
    max_active_layers: int = 4
    use_router: bool = True  # If False, router is bypassed
    fixed_layers: Optional[List[int]] = None  # If set, forces these exact layers to steer
    # Catalog router: distribution over subsets. Empty is a legal action.
    catalog_include_pairs: bool = True
    catalog_consecutive_triples: bool = True
    router_catalog_samples: int = 2  # extra subset forwards besides empty
    router_max_tok_len: int = 512
    router_shot_mix: Tuple[int, ...] = (0, 2, 4)

    # SteerNet
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    steer_magnitude_mode: str = "direct"  # 'direct' or 'decoupled' (direction normalized + bounded scale)
    steer_max_magnitude: float = 2.0

    # Data
    n_samples: int = 7000
    max_tok_len: int = 384
    fewshot_train_prob: float = 0.30
    fewshot_train_max_demos: int = 2

    # Phase 1: SteerNet only
    phase1_lr: float = 5e-3
    phase1_epochs: int = 12
    phase1_warmup_ratio: float = 0.06
    phase1_patience: int = 4

    # Phase 2 staged parameters
    phase2_warmup_ratio: float = 0.03
    phase2_patience: int = 5

    # Gate architecture
    gate_dim: int = 16
    gate_dropout: float = 0.05
    gate_init_alpha: float = 0.80
    gating_mode: str = "learned"  # 'learned' (MLP), 'constant' (fixed scalar), or 'step'
    gate_constant_value: float = 0.80

    # Gate loss regularisation
    lambda_pg: float = 2.0
    lambda_l1_gate: float = 0.001
    lambda_gate_usefulness: float = 0.80
    lambda_gate_antisat: float = 0.25
    lambda_gate_diversity: float = 3.0
    gate_diversity_floor: float = 0.05
    pg_n_samples: int = 4
    pg_baseline_momentum: float = 0.8
    pg_every_n_steps: int = 8

    # Regularisation & Hardware optimization
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    batch_size: int = 1
    grad_accum: int = 16
    seed: int = 42

    # Monitoring
    verifier_every_n: int = 10
    log_gpu_memory: bool = True

    # Checkpoint selection
    use_accuracy_checkpoint: bool = True
    accuracy_eval_n: int = 100
    accuracy_eval_every_n: int = 1

    # Checkpoint paths (relative by default, portable for local and Kaggle)
    lora_save_path: str = "./checkpoints/steer_phase1_adaptive_v11.pt"
    phase2_save_path: str = "./checkpoints/rsc_phase2_catalog_v14.pt"
    checkpoint_dir: str = "./checkpoints"

    # Router bootstrap settings
    router_bootstrap_epochs: int = 3
    router_bootstrap_n_examples: int = 2000
    router_bootstrap_lr: float = 1e-3
    router_bootstrap_patience: int = 2
    router_bootstrap_save_path: str = "./checkpoints/router_catalog_v14.pt"
    router_probe_every_n_steps: int = 1
    router_checkpoint_every_n_steps: int = 0
    router_dead_layer_veto_frac: float = 0.50

    # Cooperative Subset Router & RLOO settings
    lambda_cost: float = 0.005
    beta_entropy_k: float = 0.15
    beta_entropy_subset: float = 0.15
    router_balance_weight: float = 0.10
    lambda_synergy: float = 0.10

    # Prompt Shielding (Preserves pristine representations in multi-shot contexts)
    prompt_shielding: bool = True

    # Phase 2A: SteerNet fine-tuning under learned routing
    steer_finetune_epochs: int = 2
    steer_finetune_lr: float = 1e-4
    steer_finetune_save_path: str = "./checkpoints/steer_finetune.pt"

    # Phase 2B: Token Gate training
    gate_only_epochs: int = 3
    gate_only_lr: float = 1e-3
    gate_only_save_path: str = "./checkpoints/gate_only.pt"

    # Phase 2C: Light joint fine-tuning
    joint_finetune_epochs: int = 2
    joint_finetune_lr_scale: float = 0.10
    joint_finetune_router_lr_scale: float = 0.05

    # Phase 2D: Gate re-tuning against final router
    gate_retune_epochs: int = 1
    gate_retune_lr: float = 5e-4

    checkpoint_every_n: int = 3
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 3
    lr_reduce_min_lr: float = 1e-6
    early_stop_delta: float = 1e-4
    use_gradient_monitor: bool = True
    grad_warn_threshold: float = 2.0

    # Prompt Diversification & Invariance
    diversify_prompts: bool = True
    lambda_invariance: float = 0.5
    invariance_every_n_steps: int = 4
    pg_mode: str = "teacher_forced"  # 'teacher_forced' or 'generation'

    # Curriculum mode
    curriculum_mode: str = "full"  # 'full' (6 phases) or 'streamlined' (2 stages)
    streamlined_warmup_epochs: int = 3
    streamlined_joint_epochs: int = 3
    streamlined_steer_lr: float = 1e-4
    streamlined_gate_lr: float = 1e-3

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Ensure parent dirs exist for paths
        for path in [
            self.lora_save_path,
            self.phase2_save_path,
            self.router_bootstrap_save_path,
            self.steer_finetune_save_path,
            self.gate_only_save_path,
        ]:
            pdir = os.path.dirname(path)
            if pdir:
                os.makedirs(pdir, exist_ok=True)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def max_length(self) -> int:
        return self.max_tok_len

    @max_length.setter
    def max_length(self, value: int):
        self.max_tok_len = value

    @property
    def val_split(self) -> float:
        return 0.08

    @val_split.setter
    def val_split(self, value: float):
        pass

    def rsc_layers_for(self, n_total: int) -> List[int]:
        if self.fixed_layers is not None:
            return sorted(set(max(1, min(int(idx), n_total - 2)) for idx in self.fixed_layers))
        if not 1 <= self.max_active_layers <= 4:
            raise ValueError("max_active_layers must be in [1, 4]")
        start, stop = int(0.30 * n_total), int(0.80 * n_total)
        count = min(8, max(1, stop - start))
        indices = torch.linspace(start, stop - 1, steps=count).round().long().tolist()
        return sorted(set(max(1, min(idx, n_total - 2)) for idx in indices))
