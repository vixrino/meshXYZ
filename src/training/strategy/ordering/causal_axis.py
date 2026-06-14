import random

import torch
from torch import Tensor

from ....constants import TRI_PAD
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
        B, N, T = faces.shape
        device = faces.device

        if T == 9:
            # Triangle-only (9 tokens = 3 verts × 3 coords): straightforward mean.
            centroids = faces.double().reshape(B, N, 3, 3).mean(dim=2)  # (B, N, 3)
        else:
            # Unified 12-token layout: 4 vertex slots per face, but triangle faces
            # carry TRI_PAD (129) at positions 0-2, marking those as non-real vertices.
            # Average only over the valid (non-padded) vertices.
            verts  = faces.double().reshape(B, N, 4, 3)                  # (B, N, 4, 3)
            is_pad = (faces.reshape(B, N, 4, 3) == TRI_PAD).all(dim=-1)  # (B, N, 4) bool
            valid  = (~is_pad).double().unsqueeze(-1)                     # (B, N, 4, 1)
            centroids = (verts * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)  # (B,N,3)

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
