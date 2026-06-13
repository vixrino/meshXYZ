import random
from dataclasses import dataclass

import torch
from jaxtyping import Bool
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseMasking


@dataclass
class CausalAxisMaskingCfg:
    prob: float = 0.5


# All six permutations of (x=0, y=1, z=2) as (primary, secondary, tertiary) index tuples
_AXIS_ORDERINGS: list[tuple[int, int, int]] = [
    (0, 1, 2),  # xyz
    (0, 2, 1),  # xzy
    (1, 0, 2),  # yxz
    (1, 2, 0),  # yzx
    (2, 0, 1),  # zxy
    (2, 1, 0),  # zyx
]


class CausalAxisMasking(BaseMasking):
    """Causal mask along a random lexicographic axis ordering with random per-axis direction.

    Each call:
      1. Samples one of 6 axis orderings (primary → secondary → tertiary).
      2. Independently samples ascending/descending for each of the 3 axis positions.
      3. Ranks faces lexicographically and masks face k from query q when rank[k] > rank[q].

    Args:
        cfg.prob: probability of applying the mask at all (default 0.5).
    """

    def __init__(self, cfg: CausalAxisMaskingCfg):
        self.cfg = cfg

    def select(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        faces = batch["faces"]
        B, N, _ = faces.shape
        device = faces.device

        if random.random() >= self.cfg.prob:
            return torch.zeros(B, N, N, dtype=torch.bool, device=device)

        # centroid per face: (B, N, 3)
        centroids = faces.float().reshape(B, N, 3, 3).mean(dim=2)

        # sample axis ordering and per-position direction (+1=ascending, -1=descending)
        a, b, c = random.choice(_AXIS_ORDERINGS)
        s_a, s_b, s_c = [random.choice([-1.0, 1.0]) for _ in range(3)]

        # signed centroid values per axis position: (B, N)
        ca = s_a * centroids[:, :, a]
        cb = s_b * centroids[:, :, b]
        cc = s_c * centroids[:, :, c]

        # Direct lexicographic comparison via broadcasting — no argsort needed.
        ca_q, ca_k = ca.unsqueeze(2), ca.unsqueeze(1)  # (B, N, 1) / (B, 1, N)
        cb_q, cb_k = cb.unsqueeze(2), cb.unsqueeze(1)
        cc_q, cc_k = cc.unsqueeze(2), cc.unsqueeze(1)

        gt_a = ca_k > ca_q
        eq_a = ca_k == ca_q
        gt_b = cb_k > cb_q
        eq_b = cb_k == cb_q
        gt_c = cc_k > cc_q

        return gt_a | (eq_a & gt_b) | (eq_a & eq_b & gt_c)  # (B, N, N)
