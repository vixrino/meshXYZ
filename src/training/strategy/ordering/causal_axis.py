import random

import torch
from torch import Tensor

from .base import BaseOrdering

_AXIS_ORDERINGS: list[tuple[int, int, int]] = [
    (0, 1, 2),  # xyz
    (0, 2, 1),  # xzy
    (1, 0, 2),  # yxz
    (1, 2, 0),  # yzx
    (2, 0, 1),  # zxy
    (2, 1, 0),  # zyx
]


class CausalAxisOrdering(BaseOrdering):
    """Sort faces lexicographically by a randomly sampled axis ordering and per-axis direction.

    Axis ordering and directions are sampled independently per sample in the batch.
    """

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, _ = faces.shape
        device = faces.device

        centroids = faces.double().reshape(B, N, 3, 3).mean(dim=2)  # (B, N, 3)
        scale = 256.0
        keys = torch.zeros(B, N, device=device)

        for b_idx in range(B):
            a, b, c = random.choice(_AXIS_ORDERINGS)
            sa, sb, sc = [random.choice([-1.0, 1.0]) for _ in range(3)]

            ca = (sa * centroids[b_idx, :, a] + 127.0) * scale ** 2
            cb = (sb * centroids[b_idx, :, b] + 127.0) * scale
            cc = (sc * centroids[b_idx, :, c] + 127.0)
            keys[b_idx] = (ca + cb + cc).float()
            keys[b_idx, lengths[b_idx]:] = float("inf")

        return keys.argsort(dim=1)
