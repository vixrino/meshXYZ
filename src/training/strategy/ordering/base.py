from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseOrdering(ABC):
    @abstractmethod
    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        """Compute permutation indices for face reordering.

        faces:          (B, N, T)  T=9  (triangle-only) or T=12 (unified quad/tri)
        face_neighbors: (B, N, S)  S=3  (triangle-only) or S=4  (quad/mixed);
                                   -1 = no neighbor
        lengths:        (B,) number of valid (non-padding) faces per sample

        Returns perm (B, N) where:
          perm[b, :lengths[b]] is a permutation of [0, lengths[b])
          perm[b, lengths[b]:] = arange(lengths[b], N)  — padding unchanged
        """

    def _base_perm(self, B: int, N: int, lengths: Tensor, device: torch.device) -> Tensor:
        """Identity permutation — subclasses fill in the valid-face region."""
        return torch.arange(N, device=device).unsqueeze(0).expand(B, N).clone()
