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
    ) -> "tuple[Int[Tensor, 'batch faces 9'], Int[Tensor, 'batch faces 6'] | None]":
        """Return (targets_9, query_edges).

        use_edge_cond=True:
            targets_9[..., :6] = PAD_TARGET  (edge positions, excluded from loss)
            targets_9[..., 6:] = new-vertex absolute coords (or EOS_RESIDUAL when no target)
            query_edges        = (B, N, 6) two-vertex edge being extended
        use_edge_cond=False:
            targets_9          = full canonical 9-coord face (original behaviour)
            query_edges        = None
        """
