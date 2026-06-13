import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainingCfg


def build_optimizer(model: nn.Module, cfg: TrainingCfg) -> dict:
    opt = AdamW(
        model.decoder.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    warmup = cfg.warmup_steps

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return min(step / warmup, (warmup / step) ** 0.5)

    sched = LambdaLR(opt, lr_lambda)
    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
