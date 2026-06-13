import random
from dataclasses import dataclass

import torch
from jaxtyping import Bool
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseMasking


@dataclass
class RandomMaskingCfg:
    mask_ratio: float = 0.75
    prob: float = 1.0


class RandomMasking(BaseMasking):
    """Randomly mask a fraction of key faces, uniformly across all queries.

    For each batch item the actual ratio is sampled from [0, mask_ratio],
    then that many keys are chosen at random and hidden from every query.

    Args:
        cfg.mask_ratio: maximum fraction of tokens to mask (default 0.75).
        cfg.prob: probability of applying any masking per call (default 1.0).
    """

    def __init__(self, cfg: RandomMaskingCfg):
        self.cfg = cfg

    def select(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        faces = batch["faces"]
        B, N, _ = faces.shape
        device = faces.device

        if random.random() >= self.cfg.prob:
            return torch.zeros(B, N, N, dtype=torch.bool, device=device)

        # Random scores for every (batch, face) position, all on GPU.
        rand_vals = torch.rand(B, N, device=device)

        if "lengths" in batch:
            lengths = batch["lengths"].to(device)
            valid   = torch.arange(N, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            rand_vals = torch.where(valid, rand_vals, torch.ones(1, device=device) * 2.0)
        else:
            lengths = torch.full((B,), N, device=device, dtype=torch.long)

        # Per-sample mask ratio sampled on GPU.
        num_masked = (lengths * (torch.rand(B, device=device) * self.cfg.mask_ratio)).long()

        # argsort gives cheapest-to-mask indices first; select first num_masked per row.
        sorted_idx  = torch.argsort(rand_vals, dim=1)          # (B, N)
        rank        = torch.arange(N, device=device).unsqueeze(0)
        key_mask    = torch.zeros(B, N, dtype=torch.bool, device=device)
        key_mask.scatter_(1, sorted_idx, rank < num_masked.unsqueeze(1))

        # Same masked keys hidden from every query → broadcast across query dim.
        return key_mask.unsqueeze(1).expand(B, N, N)
