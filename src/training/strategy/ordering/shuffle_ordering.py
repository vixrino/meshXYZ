import random
from dataclasses import dataclass

import torch
from torch import Tensor

from .base import BaseOrdering


@dataclass
class ShuffleCfg:
    prob: float = 1.0
    max_fraction: float = 0.5


class ShuffleOrdering(BaseOrdering):
    """Shuffle a random subset of valid faces.

    Per sample:
      1. Skip entirely with probability (1 - cfg.prob).
      2. Sample count k ~ Uniform[0, max_fraction * L].
      3. Sample k indices uniformly at random (without replacement) from [0, L).
      4. Randomly permute the faces at those k positions; everything else unchanged.
    """

    def __init__(self, cfg: ShuffleCfg):
        self.cfg = cfg

    def permute(self, faces: Tensor, face_neighbors: Tensor, lengths: Tensor) -> Tensor:
        B, N, _ = faces.shape
        device = faces.device
        perm = self._base_perm(B, N, lengths, device)

        for b in range(B):
            if random.random() >= self.cfg.prob:
                continue
            L = int(lengths[b].item())
            k = random.randint(0, int(self.cfg.max_fraction * L))
            if k < 2:
                continue
            chosen = torch.randperm(L, device=device)[:k]
            perm[b, chosen] = chosen[torch.randperm(k, device=device)]

        return perm
