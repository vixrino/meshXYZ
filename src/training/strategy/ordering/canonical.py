import torch
from torch import Tensor

from ....constants import QUANT_MAX
from .base import BaseOrdering


class CanonicalOrdering(BaseOrdering):
    """Sort faces by ZYX key of their first (already vertex-rotated) vertex."""

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, _ = faces.shape
        device = faces.device
        max_val = int(QUANT_MAX) + 1

        v = faces[:, :, :3]  # first vertex x, y, z  — (B, N, 3)
        keys = (
            v[:, :, 2].long() * max_val ** 2
            + v[:, :, 1].long() * max_val
            + v[:, :, 0].long()
        ).float()  # (B, N)

        # push padding to the end
        for b in range(B):
            keys[b, lengths[b]:] = float("inf")

        return keys.argsort(dim=1)
