from dataclasses import dataclass

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from ....constants import EOS_RESIDUAL
from ....dataset.types import Batch
from .base import BaseTargetBuilder


@dataclass
class CausalTargetBuilderCfg:
    pass


class CausalTargetBuilder(BaseTargetBuilder):
    """For each query face q, predict face q+1 (the immediately next face in causal order); EOS if none.

    Edge conditioning (use_edge_cond) is ignored here, the concept of "extending an edge"
    only applies to the adjacent strategy.  Always returns full 9-coord targets and query_edges=None.
    """

    def __init__(self, cfg: CausalTargetBuilderCfg):
        self.cfg = cfg

    def compute_targets(
        self,
        batch: Batch,
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ):
        faces  = batch["faces"]   # (B, N, 9)
        B, N   = faces.shape[:2]
        device = faces.device

        next_idx  = torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, -1)
        has_next  = next_idx < N
        safe_next = next_idx.clamp(0, N - 1)

        n_coords   = faces.shape[-1]   # 9 (tri-only) or 12 (unified quad/tri)
        tgt_coords = faces.gather(1, safe_next.unsqueeze(-1).expand(B, N, n_coords))
        eos        = torch.full((B, N, n_coords), EOS_RESIDUAL, dtype=faces.dtype, device=device)
        return torch.where(has_next.unsqueeze(-1), tgt_coords, eos), None
