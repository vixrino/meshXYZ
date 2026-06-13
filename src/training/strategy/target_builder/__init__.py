from __future__ import annotations

import dacite
from jaxtyping import Bool, Int
from torch import Tensor

from .adjacent import AdjacentTargetBuilder, AdjacentTargetBuilderCfg
from .base import BaseTargetBuilder
from .causal import CausalTargetBuilder, CausalTargetBuilderCfg
from ....dataset.types import Batch

__all__ = ["BaseTargetBuilder", "AdjacentTargetBuilder", "CausalTargetBuilder", "TargetBuilder"]

_REGISTRY: dict[str, tuple[type[BaseTargetBuilder], type]] = {
    "adjacent": (AdjacentTargetBuilder, AdjacentTargetBuilderCfg),
    "causal":   (CausalTargetBuilder,   CausalTargetBuilderCfg),
}


class TargetBuilder:
    def __init__(self, strategy: BaseTargetBuilder):
        self.strategy = strategy

    def compute_targets(
        self,
        batch: Batch,
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ):
        return self.strategy.compute_targets(batch, target_mask, use_edge_cond=use_edge_cond)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "TargetBuilder":
        cfg = dict(cfg)
        key = cfg.pop("type")
        strategy_cls, cfg_cls = _REGISTRY[key]
        return cls(strategy_cls(dacite.from_dict(cfg_cls, cfg)))
