from dataclasses import dataclass

import torch
from jaxtyping import Bool
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseMasking


@dataclass
class AdjacentMaskingCfg:
    prob: float = 1.0
    prob_mask_adjacent: float = 0.5


class AdjacentMasking(BaseMasking):
    def __init__(self, cfg: AdjacentMaskingCfg):
        self.cfg = cfg

    def select(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        face_neighbors = batch["face_neighbors"]                       # (B, N, 3), -1=boundary
        B, N, _ = face_neighbors.shape
        device = face_neighbors.device

        mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        if torch.rand(1).item() >= self.cfg.prob:
            return mask

        safe = face_neighbors.clamp(min=0)                             # (B, N, 3)
        valid = face_neighbors >= 0                                    # (B, N, 3)
        rand = torch.rand(B, N, 3, device=device) < self.cfg.prob_mask_adjacent
        to_mask = rand & valid                                         # (B, N, 3)

        # Single scatter over all 3 slots at once — slots never alias on triangle meshes.
        mask.scatter_(2, safe, to_mask)

        return mask
