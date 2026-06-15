import torch
from torch import Tensor

from ....constants import QUANT_MAX
from .base import BaseOrdering


class CanonicalOrdering(BaseOrdering):
    """Sort faces by ZYX key of their first real vertex.

    The first real vertex is always at tokens 0-2 — for 9-token triangles, and
    for 12-token unified faces alike (TRI_PAD padding now lives at the END of the
    block, positions 9-11, so it never collides with the sort key).
    """

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, T = faces.shape
        device = faces.device
        max_val = int(QUANT_MAX) + 1

        # First real vertex at tokens 0-2 for both 9- and 12-token layouts.
        v = faces[:, :, :3]

        keys = (
            v[:, :, 2].long() * max_val ** 2
            + v[:, :, 1].long() * max_val
            + v[:, :, 0].long()
        ).float()  # (B, N)

        # push padding to the end
        for b in range(B):
            keys[b, lengths[b]:] = float("inf")

        return keys.argsort(dim=1)
