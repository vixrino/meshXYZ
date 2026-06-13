from dataclasses import dataclass

import torch
from jaxtyping import Bool
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseMasking


@dataclass
class CausalMaskingCfg:
    pass


class CausalMasking(BaseMasking):
    """Deterministic causal mask following the dataloader face ordering.

    Face k is masked from query q when k > q — each face can only attend
    to faces that appear earlier in the sequence (ZYX lexicographic order).
    """

    def __init__(self, cfg: CausalMaskingCfg):
        self.cfg = cfg

    def select(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        faces = batch["faces"]
        B, N, _ = faces.shape
        device = faces.device

        idx = torch.arange(N, device=device)
        mask = idx.unsqueeze(0) > idx.unsqueeze(1)  # (N, N)
        return mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
