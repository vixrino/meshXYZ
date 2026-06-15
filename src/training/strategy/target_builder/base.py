from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from jaxtyping import Bool, Int
from torch import Tensor

if TYPE_CHECKING:
    from ....dataset.types import Batch


class BaseTargetBuilder(ABC):
    @abstractmethod
    def compute_targets(
        self,
        batch: "Batch",
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ) -> "tuple[Int[Tensor, 'batch faces T'], Int[Tensor, 'batch faces 6'] | None]":
        """Return (targets, query_edges).  T is 9 (triangle-only) or 12 (unified).

        use_edge_cond=True:
            targets[..., :6]   = PAD_TARGET  (edge positions, excluded from loss)
            9-token : targets[..., 6:9]  = new-vertex coords (or EOS_RESIDUAL = no target)
            12-token: targets[..., 6:9]  = v1 (or EOS_RESIDUAL = stop edge)
                      targets[..., 9:12] = v2 (quad) / EOS_RESIDUAL (triangle) / PAD
            query_edges        = (B, N, 6) two-vertex edge being extended
        use_edge_cond=False:
            targets            = full canonical T-coord face (original behaviour)
            query_edges        = None
        """
