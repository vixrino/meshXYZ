from dataclasses import dataclass

import torch
from jaxtyping import Bool
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseMasking


@dataclass
class AxisMaskingCfg:
    pass


class AxisMasking(BaseMasking):
    """For each query face, masks all faces on one side of a randomly sampled axis plane."""

    def __init__(self, cfg: AxisMaskingCfg):
        self.cfg = cfg

    def select(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        faces = batch["faces"]  # (B, N, 9) — layout: V0x V0y V0z V1x V1y V1z V2x V2y V2z
        B, N, _ = faces.shape
        device = faces.device

        # Face centroid per axis: (B, N, 3)
        centroids = faces.float().reshape(B, N, 3, 3).mean(dim=2)

        # Per query face: sample a random axis (0=x, 1=y, 2=z) and direction (True=mask greater)
        axis      = torch.randint(0, 3, (B, N), device=device)         # (B, N)
        direction = torch.randint(0, 2, (B, N), device=device).bool()  # (B, N)

        # Query centroid value on its sampled axis: (B, N)
        query_val = centroids.gather(2, axis.unsqueeze(2)).squeeze(2)

        # Key centroid values on each query's axis: (B, N, N)
        # key_vals[b, q, k] = centroids[b, k, axis[b, q]]
        axis_for_keys = axis.unsqueeze(2).expand(B, N, N)
        key_vals = (
            centroids.unsqueeze(1)
                     .expand(B, N, N, 3)
                     .gather(3, axis_for_keys.unsqueeze(3))
                     .squeeze(3)
        )

        greater_mask = key_vals > query_val.unsqueeze(2)  # (B, N, N)
        lower_mask   = key_vals < query_val.unsqueeze(2)  # (B, N, N)

        return torch.where(direction.unsqueeze(2), greater_mask, lower_mask)
