from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ....constants import EOS_RESIDUAL, PAD_TARGET, QUANT_MAX, TRI_NEIGHBOR, TRI_PAD
from ....dataset.types import Batch
from .base import BaseTargetBuilder


@dataclass
class AdjacentTargetBuilderCfg:
    pass


class AdjacentTargetBuilder(BaseTargetBuilder):
    """For each query face, predict its first open-edge neighbor; EOS if none.

    Two face layouts are supported and dispatched on faces.shape[-1]:

    9-token (triangle-only, original behaviour, untouched)
        A triangle neighbor shares an edge (2 verts) and contributes 1 unique
        vertex.  Target: positions 0-5 = PAD (edge), positions 6-8 = unique vertex
        or EOS_RESIDUAL when there is no neighbor.

    12-token (unified quad/tri, Option-A hierarchical EOS)
        A quad neighbor contributes 2 unique vertices, a triangle neighbor 1.
        Target layout: positions 0-5 = PAD (edge), positions 6-8 = v1, 9-11 = v2.
            no neighbor    → 6-8 = EOS_RESIDUAL, 9-11 = PAD          ("STOP this edge")
            triangle nbr   → 6-8 = unique vertex, 9-11 = TRI_NEIGHBOR ("no v2")
            quad neighbor  → 6-8 = v1, 9-11 = v2
        Read order at generation: slot1 EOS_RESIDUAL ⇒ stop; else slot2
        TRI_NEIGHBOR ⇒ triangle; else quad.

        Slot 1 and slot 2 use DISTINCT sentinels (STOP = EOS_RESIDUAL,
        TRI_NEIGHBOR for "triangle neighbor").  They were once both EOS_RESIDUAL,
        which let the slot-1 stop signal bleed into slot 2 and collapse quads to
        triangles — see constants.TRI_NEIGHBOR.  Do not merge them.

        Vertex ordering (winding).  The two unique quad vertices are ordered so
        that [ev0, ev1, v1, v2] is a valid 4-cycle regardless of how the shared
        edge was lex-sorted:
            v1 = the unique vertex cyclically adjacent to ev1
            v2 = the unique vertex cyclically adjacent to ev0
        canonical_face_12 (rotation only) then normalizes the start vertex.
    """

    def __init__(self, cfg: AdjacentTargetBuilderCfg):
        self.cfg = cfg

    # ── shared ────────────────────────────────────────────────────────────────

    def _first_open_edge(
        self,
        target_mask: Bool[Tensor, "batch faces faces"],
        face_neighbors: Int[Tensor, "batch faces slots"],
    ) -> tuple[Bool[Tensor, "batch faces"], Int[Tensor, "batch faces"], Int[Tensor, "batch faces slots"]]:
        """Per-face: has_target flag, index of first masked edge, clamped neighbors.

        Works for any number of edge slots (3 for triangles, 4 for the unified
        layout); the (face_neighbors >= 0) guard treats the -1 padding slot of a
        triangle in the 4-slot layout as 'no edge'.
        """
        safe = face_neighbors.clamp(min=0)
        masked_per_edge = target_mask.gather(-1, safe) & (face_neighbors >= 0)
        has_target = masked_per_edge.any(dim=-1)
        open_edge = masked_per_edge.int().argmax(dim=-1)
        return has_target, open_edge, safe

    def compute_targets(
        self,
        batch: Batch,
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ):
        faces = batch["faces"]
        if faces.shape[-1] == 12:
            return self._compute_targets_12(batch, target_mask, use_edge_cond)
        return self._compute_targets_9(batch, target_mask, use_edge_cond)

    # ── 9-token (triangle-only) — original behaviour, unchanged ────────────────

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

        v0_start = edge_flat * 3
        v1_start = ((edge_flat + 1) % 3) * 3
        ev0 = torch.stack([faces_flat[idx, v0_start + i] for i in range(3)], dim=-1)
        ev1 = torch.stack([faces_flat[idx, v1_start + i] for i in range(3)], dim=-1)

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

    def _compute_targets_9(
        self,
        batch: Batch,
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ):
        faces, face_neighbors = batch["faces"], batch["face_neighbors"]
        B, N = target_mask.shape[:2]

        has_target, open_edge, safe_neighbors = self._first_open_edge(target_mask, face_neighbors)
        tgt_idx    = safe_neighbors.gather(-1, open_edge.unsqueeze(-1))
        tgt_coords = faces.gather(1, tgt_idx.expand(B, N, 9))

        if not use_edge_cond:
            eos = torch.full((B, N, 9), EOS_RESIDUAL, dtype=faces.dtype, device=faces.device)
            return torch.where(has_target.unsqueeze(-1), tgt_coords, eos), None

        ev0, ev1, query_edges = self._query_edge(faces, open_edge)
        unique_vertex = self._unique_vertex(tgt_coords, ev0, ev1)

        tgt_coords = torch.full((B*N, 9), PAD_TARGET, dtype=faces.dtype, device=faces.device)
        is_eos = (~has_target).reshape(B*N).unsqueeze(-1)
        tgt_coords[:, 6:] = torch.where(
            is_eos,
            torch.full((B*N, 3), EOS_RESIDUAL, dtype=faces.dtype, device=faces.device),
            unique_vertex,
        )
        return tgt_coords.reshape(B, N, 9), query_edges

    # ── 12-token (unified quad/tri) ────────────────────────────────────────────

    def _query_edge_12(
        self,
        faces: Int[Tensor, "batch faces 12"],
        open_edge: Int[Tensor, "batch faces"],
    ) -> tuple[Int[Tensor, "BN 3"], Int[Tensor, "BN 3"], Int[Tensor, "batch faces 6"]]:
        """Extract and lex-sort the two vertices of the query face's open edge.

        TRI_PAD-aware: a triangle's real vertices start at position 3 and form a
        3-cycle (edges 0/1/2); a quad uses positions 0-11 and a 4-cycle (edges
        0/1/2/3).  Lex-sorting matches the (min,max) convention used by generate().
        """
        B, N = faces.shape[:2]
        BN = B * N
        ff   = faces.reshape(BN, 12)
        e    = open_edge.reshape(BN)
        is_tri = ff[:, 0] == TRI_PAD                      # (BN,)
        nv     = torch.where(is_tri, torch.full_like(e, 3), torch.full_like(e, 4))
        base   = torch.where(is_tri, torch.full_like(e, 3), torch.full_like(e, 0))

        a_start = base + e * 3
        b_start = base + ((e + 1) % nv) * 3
        off     = torch.arange(3, device=faces.device)
        ev_a = ff.gather(1, (a_start.unsqueeze(-1) + off))   # (BN,3)
        ev_b = ff.gather(1, (b_start.unsqueeze(-1) + off))

        B_ = QUANT_MAX + 1
        key = lambda v: v[:, 0] * B_ ** 2 + v[:, 1] * B_ + v[:, 2]
        swap = (key(ev_a) > key(ev_b)).unsqueeze(-1)
        ev0 = torch.where(swap, ev_b, ev_a)
        ev1 = torch.where(swap, ev_a, ev_b)
        return ev0, ev1, torch.cat([ev0, ev1], dim=-1).reshape(B, N, 6)

    def _ordered_uniques_12(
        self,
        tgt_coords: Int[Tensor, "batch faces 12"],
        ev0: Int[Tensor, "BN 3"],
        ev1: Int[Tensor, "BN 3"],
    ) -> tuple[Int[Tensor, "BN 3"], Int[Tensor, "BN 3"], Bool[Tensor, "BN"]]:
        """Ordered unique vertices of the neighbor face + neighbor-is-triangle flag.

        Quad neighbor: returns (v1, v2) with v1 cyclically adjacent to ev1 and v2
        cyclically adjacent to ev0, so [ev0, ev1, v1, v2] is a valid 4-cycle for
        any lex-sort direction of the shared edge.
        Triangle neighbor: returns (v1, *) where v1 is the single unique vertex;
        v2 is unused (the caller writes EOS_RESIDUAL into slot 2).
        """
        BN = tgt_coords.shape[0] * tgt_coords.shape[1]
        nb = tgt_coords.reshape(BN, 4, 3)                       # pos 0 == TRI_PAD for triangles
        nb_is_tri = tgt_coords.reshape(BN, 12)[:, 0] == TRI_PAD
        idx = torch.arange(BN, device=tgt_coords.device)

        is_ev0 = (nb == ev0.reshape(BN, 1, 3)).all(-1)         # (BN,4)
        is_ev1 = (nb == ev1.reshape(BN, 1, 3)).all(-1)
        is_pad = (nb == TRI_PAD).all(-1)                        # TRI_PAD slot of a triangle
        is_unique = (~is_ev0) & (~is_ev1) & (~is_pad)          # (BN,4)

        # Quad: order the two uniques by cyclic adjacency to the edge endpoints.
        pos_ev0 = is_ev0.float().argmax(-1)
        pos_ev1 = is_ev1.float().argmax(-1)
        c1a, c1b = (pos_ev1 + 1) % 4, (pos_ev1 - 1) % 4
        v1_pos = torch.where(is_unique.gather(1, c1a.unsqueeze(-1)).squeeze(-1), c1a, c1b)
        c0a, c0b = (pos_ev0 + 1) % 4, (pos_ev0 - 1) % 4
        v2_pos = torch.where(is_unique.gather(1, c0a.unsqueeze(-1)).squeeze(-1), c0a, c0b)
        v1_q = nb[idx, v1_pos]
        v2_q = nb[idx, v2_pos]

        # Triangle: the single unique vertex.
        v1_t = nb[idx, is_unique.float().argmax(-1)]

        v1 = torch.where(nb_is_tri.unsqueeze(-1), v1_t, v1_q)
        return v1, v2_q, nb_is_tri

    def _compute_targets_12(
        self,
        batch: Batch,
        target_mask: Bool[Tensor, "batch query_faces key_faces"],
        use_edge_cond: bool = True,
    ):
        faces, face_neighbors = batch["faces"], batch["face_neighbors"]
        B, N = target_mask.shape[:2]
        dev, dt = faces.device, faces.dtype

        has_target, open_edge, safe_neighbors = self._first_open_edge(target_mask, face_neighbors)
        tgt_idx    = safe_neighbors.gather(-1, open_edge.unsqueeze(-1))
        tgt_coords = faces.gather(1, tgt_idx.expand(B, N, 12))

        if not use_edge_cond:
            eos = torch.full((B, N, 12), EOS_RESIDUAL, dtype=dt, device=dev)
            return torch.where(has_target.unsqueeze(-1), tgt_coords, eos), None

        ev0, ev1, query_edges = self._query_edge_12(faces, open_edge)
        v1, v2, nb_is_tri = self._ordered_uniques_12(tgt_coords, ev0, ev1)

        BN    = B * N
        has_t = has_target.reshape(BN)
        stop3 = torch.full((BN, 3), EOS_RESIDUAL, dtype=dt, device=dev)  # slot-1 "STOP"
        tri3  = torch.full((BN, 3), TRI_NEIGHBOR, dtype=dt, device=dev)  # slot-2 "triangle nbr"
        pad3  = torch.full((BN, 3), PAD_TARGET,   dtype=dt, device=dev)

        out = torch.full((BN, 12), PAD_TARGET, dtype=dt, device=dev)
        # slot 1 (6-8): unique vertex v1 when there is a neighbor, else STOP (EOS_RESIDUAL).
        out[:, 6:9] = torch.where(has_t.unsqueeze(-1), v1, stop3)
        # slot 2 (9-11): triangle neighbor → TRI_NEIGHBOR marker; quad neighbor → v2;
        #                no neighbor → PAD (ignored).  Distinct from the slot-1 STOP
        #                token so the two decisions never share a softmax class.
        slot2 = torch.where(nb_is_tri.unsqueeze(-1), tri3, v2)
        slot2 = torch.where(has_t.unsqueeze(-1), slot2, pad3)
        out[:, 9:12] = slot2
        return out.reshape(B, N, 12), query_edges
