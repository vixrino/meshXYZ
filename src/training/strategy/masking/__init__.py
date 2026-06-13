from __future__ import annotations

import dacite
import torch
from jaxtyping import Bool
from torch import Tensor

from .adjacent import AdjacentMasking, AdjacentMaskingCfg
from .axis import AxisMasking, AxisMaskingCfg
from .base import BaseMasking
from .bidirectional import BidirectionalMasking, BidirectionalMaskingCfg
from .causal import CausalMasking, CausalMaskingCfg
from .causal_axis import CausalAxisMasking, CausalAxisMaskingCfg
from .random import RandomMasking, RandomMaskingCfg
from ....dataset.types import Batch

__all__ = [
    "BaseMasking", "AdjacentMasking", "AxisMasking", "BidirectionalMasking",
    "CausalMasking", "CausalAxisMasking", "RandomMasking", "MaskingPipeline",
]

_REGISTRY: dict[str, tuple[type[BaseMasking], type]] = {
    "adjacent":      (AdjacentMasking,      AdjacentMaskingCfg),
    "axis":          (AxisMasking,          AxisMaskingCfg),
    "bidirectional": (BidirectionalMasking, BidirectionalMaskingCfg),
    "causal":        (CausalMasking,        CausalMaskingCfg),
    "causal_axis":   (CausalAxisMasking,    CausalAxisMaskingCfg),
    "random":        (RandomMasking,        RandomMaskingCfg),
}


class MaskingPipeline:
    def __init__(self, additive: list[BaseMasking], exclusive: BidirectionalMasking | None = None):
        self.additive  = additive
        self.exclusive = exclusive

    def compute_mask(self, batch: Batch) -> Bool[Tensor, "batch query_faces key_faces"]:
        faces = batch["faces"]
        B, N, _ = faces.shape
        device = faces.device

        # decide whether to use the exclusive (bidirectional) strategy this step
        if self.exclusive is not None and torch.rand(1).item() < self.exclusive.cfg.exclusive_prob:
            token_mask = self.exclusive.select(batch)
        else:
            token_mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)
            for s in self.additive:
                token_mask = token_mask | s.select(batch)

        lengths = batch.get("lengths")
        if lengths is not None:
            idx = torch.arange(N, device=device)
            pad = idx.unsqueeze(0) >= lengths.to(device).unsqueeze(1)  # (B, N) True=padded
            token_mask |= pad.unsqueeze(1)   # padded keys hidden from all queries
            token_mask |= pad.unsqueeze(2)   # padded queries hidden from all keys

        # queries should never mask themselves
        token_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        return token_mask

    @classmethod
    def from_cfg(cls, cfg_list: list[dict]) -> "MaskingPipeline":
        additive:  list[BaseMasking]          = []
        exclusive: BidirectionalMasking | None = None

        for entry in cfg_list:
            entry = dict(entry)
            key = entry.pop("type")
            strategy_cls, cfg_cls = _REGISTRY[key]
            cfg = dacite.from_dict(cfg_cls, entry)
            strategy = strategy_cls(cfg)

            if isinstance(strategy, BidirectionalMasking):
                exclusive = strategy
            else:
                additive.append(strategy)

        return cls(additive, exclusive)
