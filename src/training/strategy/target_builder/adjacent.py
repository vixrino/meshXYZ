from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ....constants import EOS_RESIDUAL, PAD_TARGET, QUANT_MAX
from ....dataset.types import Batch
from .base import BaseTargetBuilder


@dataclass
class AdjacentTargetBuilderCfg:
    pass


class AdjacentTargetBuilder(BaseTargetBuilder):
    """For each query face, predict its first open-edge neighbor (edge 0→1→2); EOS if none."""

    def __init__(self, cfg: AdjacentTargetBuilderCfg):
        self.cfg = cfg

    def _first_open_edge(
        self,
        target_mask: Bool[Tensor, "batch faces faces"],
        face_neighbors: Int[Tensor, "batch faces 3"],
    ) -> tuple[Bool[Tensor, "batch faces"], Int[Tensor, "batch faces"], Int[Tensor, "batch faces 3"]]:
        """Return per-face: has_target flag, index of first masked edge, and clamped neighbors."""
        safe = face_neighbors.clamp(min=0)
        masked_per_edge = target_mask.gather(-1, safe) & (face_neighbors >= 0)
        has_target = masked_per_edge.any(dim=-1)
        open_edge = masked_per_edge.int().argmax(dim=-1)
        return has_target, open_edge, safe

    def _query_edge(
        self,
        faces: Float[Tensor, "batch faces 9"],
        open_edge: Int[Tensor, "batch faces"],
    ) -> tuple[Float[Tensor, "BN 3"], Float[Tensor, "BN 3"], Float[Tensor, "batch faces 6"]]:
        """Extract and lex-sort the two vertices of the open edge.
        Edge mapping: 0→(v0,v1), 1→(v1,v2), 2→(v2,v0)."""
        B, N = faces.shape[:2]
        BN = B * N
        faces_flat = faces.reshape(BN, 9)
        edge_flat  = open_edge.reshape(BN)
        idx        = torch.arange(BN, device=faces.device)

        # each vertex occupies 3 consecutive coords; wrap with % 3 to handle edge 2 → v0
        v0_start = edge_flat * 3
        v1_start = ((edge_flat + 1) % 3) * 3
        ev0 = torch.stack([faces_flat[idx, v0_start + i] for i in range(3)], dim=-1)
        ev1 = torch.stack([faces_flat[idx, v1_start + i] for i in range(3)], dim=-1)

        # lex-sort via integer key encoding, matching build_edge_adjacency convention
        B_ = QUANT_MAX + 1
        key = lambda v: v[:, 0] * B_ ** 2 + v[:, 1] * B_ + v[:, 2]
        swap = (key(ev0) > key(ev1)).unsqueeze(-1)
        ev0_s, ev1_s = torch.where(swap, ev1, ev0), torch.where(swap, ev0, ev1)
        return ev0_s, ev1_s, torch.cat([ev0_s, ev1_s], dim=-1).reshape(B, N, 6)

    def _unique_vertex(
        self,
        tgt_coords: Float[Tensor, "batch faces 9"],
        ev0: Float[Tensor, "BN 3"],
        ev1: Float[Tensor, "BN 3"],
    ) -> Float[Tensor, "BN 3"]:
        """Find the vertex in the target face that is not part of the shared edge."""
        BN = tgt_coords.shape[0] * tgt_coords.shape[1]
        idx       = torch.arange(BN, device=tgt_coords.device)
        tgt_verts = tgt_coords.reshape(BN, 3, 3)
        is_shared = (tgt_verts == ev0.unsqueeze(1)).all(-1) | (tgt_verts == ev1.unsqueeze(1)).all(-1)
        return tgt_verts[idx, (~is_shared).float().argmax(dim=-1)]

    def compute_targets(
        self,
        batch: Batch,
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ):
        """Build prediction targets for each query face given the target mask."""
        faces, face_neighbors = batch["faces"], batch["face_neighbors"]
        B, N = target_mask.shape[:2]

        # find the first masked neighbor and gather its full 9-coord face
        has_target, open_edge, safe_neighbors = self._first_open_edge(target_mask, face_neighbors)
        tgt_idx    = safe_neighbors.gather(-1, open_edge.unsqueeze(-1))
        tgt_coords = faces.gather(1, tgt_idx.expand(B, N, 9))

        if not use_edge_cond:
            eos = torch.full((B, N, 9), EOS_RESIDUAL, dtype=faces.dtype, device=faces.device)
            return torch.where(has_target.unsqueeze(-1), tgt_coords, eos), None

        # extract the shared edge and find the unique vertex in the target face
        ev0, ev1, query_edges = self._query_edge(faces, open_edge)
        unique_vertex = self._unique_vertex(tgt_coords, ev0, ev1)

        # assemble output: PAD at 0-5 (edge, not predicted), unique vertex or EOS at 6-8
        tgt_coords = torch.full((B*N, 9), PAD_TARGET, dtype=faces.dtype, device=faces.device)
        is_eos = (~has_target).reshape(B*N).unsqueeze(-1)
        tgt_coords[:, 6:] = torch.where(
            is_eos,
            torch.full((B*N, 3), EOS_RESIDUAL, dtype=faces.dtype, device=faces.device),
            unique_vertex,
        )
        return tgt_coords.reshape(B, N, 9), query_edges
