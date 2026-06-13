from dataclasses import dataclass

import torch
from jaxtyping import Bool
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseMasking


@dataclass
class BidirectionalMaskingCfg:
    context_fraction_min: float = 0.5
    context_fraction_max: float = 0.9
    exclusive_prob: float = 0.2


class BidirectionalMasking(BaseMasking):
    """Bidirectional context masking: a contiguous prefix of the ordered sequence is context,
    the rest is masked. Context faces attend to each other fully (no causal constraint)."""

    def __init__(self, cfg: BidirectionalMaskingCfg):
        self.cfg = cfg

    def select(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        faces   = batch["faces"]
        lengths = batch["lengths"]
        B, N    = faces.shape[:2]
        device  = faces.device

        r = torch.empty(B, device=device).uniform_(self.cfg.context_fraction_min, self.cfg.context_fraction_max)
        K = (r * lengths.to(device=device, dtype=r.dtype)).round().long().clamp(min=1)  # (B,)

        # mask[b, q, k] = True if k >= K[b]  (same for every query row)
        col_idx = torch.arange(N, device=device).view(1, 1, N)  # (1, 1, N)
        return (col_idx >= K.view(B, 1, 1)).expand(B, N, N).contiguous()  # (B, N, N)
