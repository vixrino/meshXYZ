import torch
from torch import Tensor

from ....constants import QUANT_MAX, TRI_PAD
from .base import BaseOrdering


class CanonicalOrdering(BaseOrdering):
    """Sort faces by ZYX key of their first real vertex.

    For 9-token triangles the first vertex is at tokens 0-2.
    For 12-token unified faces the first *real* vertex depends on face type:
      - quad:     tokens 0-2 (no padding)
      - triangle: tokens 3-5 (tokens 0-2 are TRI_PAD sentinels)
    Detecting by TRI_PAD avoids placing all triangles after all quads
    (TRI_PAD=129 > QUANT_MAX=127 so a naïve face[:3] key would sort
    all padded triangles to the very end, breaking spatial interleaving).
    """

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, T = faces.shape
        device = faces.device
        max_val = int(QUANT_MAX) + 1

        if T == 9:
            # Triangle-only: first vertex at tokens 0-2.
            v = faces[:, :, :3]
        else:
            # Unified 12-token layout: detect triangle faces by TRI_PAD sentinel.
            is_tri = faces[:, :, 0] == TRI_PAD  # (B, N) bool
            v_quad = faces[:, :, 0:3]            # quad first vertex
            v_tri  = faces[:, :, 3:6]            # triangle first real vertex
            v = torch.where(is_tri.unsqueeze(-1), v_tri, v_quad)  # (B, N, 3)

        keys = (
            v[:, :, 2].long() * max_val ** 2
            + v[:, :, 1].long() * max_val
            + v[:, :, 0].long()
        ).float()  # (B, N)

        # push padding to the end
        for b in range(B):
            keys[b, lengths[b]:] = float("inf")

        return keys.argsort(dim=1)
