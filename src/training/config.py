from dataclasses import dataclass, field


@dataclass
class TrainingCfg:
    batch_size: int = 1
    lr: float = 6e-4
    weight_decay: float = 0.1
    max_steps: int = 200_000
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    save_every: int = 2000
    mixed_precision: bool = True
    val_every_n_steps: int = 1000
    viz_every_n_steps: int = 500
    gen_max_ctx: int = 50
    gen_max_steps: int = 150
    eos_weight: float = 1.0
    tri_neighbor_weight: float = 1.0
    masking: list = field(default_factory=list)
    target_builder: dict = field(default_factory=dict)
    ordering: dict = field(default_factory=lambda: {"strategies": [{"type": "canonical", "prob": 1.0}]})
    pc_cond_prob: float = 1.0
    save_last: bool = True
