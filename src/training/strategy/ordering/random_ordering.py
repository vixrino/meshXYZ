import torch
from torch import Tensor

from .base import BaseOrdering


class RandomOrdering(BaseOrdering):
    """Random permutation of valid faces per sample."""

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, _ = faces.shape
        device = faces.device
        perm = self._base_perm(B, N, lengths, device)
        for b in range(B):
            L = int(lengths[b].item())
            perm[b, :L] = torch.randperm(L, device=device)
        return perm
