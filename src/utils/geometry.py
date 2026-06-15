import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..constants import QUANT_MAX, TRI_PAD

_SQRT3 = math.sqrt(3)


def face_cartesian_to_spherical(
    face_norm: Float[Tensor, "... C"],
    is_tri: "Bool[Tensor, '...'] | None" = None,
) -> Float[Tensor, "... C"]:
    """Spherical embedding of a normalized face, preserving the channel count.

    The vertex anchor v0 is kept absolute; every other vertex is encoded as the
    spherical coords (r, θ, φ) of its vector relative to v0, each normalized to
    [0, 1].  Coordinate order within each vertex is (z, y, x); θ is the polar
    angle from the z-axis and φ the azimuth in the (y, x) plane.

    The layout is selected by the last-dim size C:

    C == 9 — triangle-only (original format, unchanged):
        input  [v0, v1, v2]
        output [v0, sph(v1-v0), sph(v2-v0)]

    C == 12 — unified quad/tri block:

        quad     input  [v0, v1, v2, v3]
                 output [v0, sph(v1-v0), sph(v2-v0), sph(v3-v0)]

        triangle input  [v0, v1, v2, PAD, PAD, PAD]
                 output [v0, sph(v1-v0), sph(v2-v0), PAD, PAD, PAD]

        The TRI_PAD pad (now at the END) is passed through untouched so the
        face-type signal survives; positions 0-8 are bit-identical to the C==9
        result for the same triangle, and quad/tri share the v0 anchor at 0-2.

    Face-type detection (C == 12 only)
        is_tri : optional bool tensor with the same leading shape as face_norm
        (face_norm.shape[:-1]).  True marks a triangle row.  Pass it from the
        caller computed on the RAW integer tokens (face_input[..., 9] == TRI_PAD)
        so detection is an exact integer comparison, never a float threshold.

        When is_tri is None a threshold fallback (face_norm[..., 9] > 1.0) is used.
        That is only safe because real coords ≤ 127/128 = 0.992, EOS_COORD =
        128/128 = 1.0 and TRI_PAD = 129/128 = 1.0078 are all exactly representable;
        the explicit mask is preferred in production to avoid boundary fragility
        (EOS_COORD padding rows land exactly on 1.0).
    """

    def to_spherical(d: Tensor) -> Tensor:
        # d: (..., 3) with order (dz, dy, dx)
        r = d.norm(dim=-1, keepdim=True)                              # (..., 1)
        safe_r = r.clamp(min=1e-8)
        theta = torch.acos((d[..., :1] / safe_r).clamp(-1, 1))       # polar from z
        phi   = torch.atan2(d[..., 1:2], d[..., 2:3])                # azimuthal in (y,x)
        r_n     = r     / _SQRT3                                       # [0, 1]
        theta_n = theta / math.pi                                      # [0, 1]
        phi_n   = (phi  + math.pi) / (2 * math.pi)                    # [0, 1]
        return torch.cat([r_n, theta_n, phi_n], dim=-1)

    C = face_norm.shape[-1]

    if C == 9:
        v0 = face_norm[..., 0:3]   # (z0, y0, x0)
        v1 = face_norm[..., 3:6]
        v2 = face_norm[..., 6:9]
        return torch.cat([v0, to_spherical(v1 - v0), to_spherical(v2 - v0)], dim=-1)

    if C == 12:
        # v0 anchor is always at positions 0-2 (tri and quad alike); positions 3-8
        # are sph(v1-v0), sph(v2-v0) — identical for both face types.  They differ
        # only at positions 9-11: a quad encodes sph(v3-v0); a triangle keeps its
        # trailing TRI_PAD pad untouched so the face-type signal survives.
        # Prefer the caller-supplied exact mask; fall back to the float threshold
        # (pad at position 9) only when no mask is given.
        if is_tri is not None:
            is_tri_col = is_tri.unsqueeze(-1)      # (..., 1) bool
        else:
            is_tri_col = face_norm[..., 9:10] > 1.0

        v0 = face_norm[..., 0:3]
        shared = torch.cat([
            v0,
            to_spherical(face_norm[..., 3:6] - v0),
            to_spherical(face_norm[..., 6:9] - v0),
        ], dim=-1)

        quad_tail = to_spherical(face_norm[..., 9:12] - v0)   # sph(v3 - v0)
        tri_tail  = face_norm[..., 9:12]                       # trailing TRI_PAD, untouched
        tail = torch.where(is_tri_col, tri_tail, quad_tail)

        return torch.cat([shared, tail], dim=-1)

    raise ValueError(
        f"face_cartesian_to_spherical: expected last dim 9 or 12, got {C}."
    )


def canonical_face(verts: Int[Tensor, "3 3"]) -> Int[Tensor, "9"]:
    """Roll vertices so the ZYX-lexicographic minimum is first, then flatten."""
    keys = [(int(v[2]), int(v[1]), int(v[0])) for v in verts]
    min_idx = keys.index(min(keys))
    return torch.roll(verts, -min_idx, dims=0).reshape(9)


def canonical_face_12(face12: Int[Tensor, "12"]) -> Int[Tensor, "12"]:
    """Canonicalize a 12-token unified face (triangle or quad) for use in generate().

    Face-type oracle: token[9] (the trailing pad slot; padding moved to the end).
        token[9] > QUANT_MAX (127)  →  triangle  (TRI_PAD was predicted)
        token[9] ≤ QUANT_MAX (127)  →  quad       (real coordinate predicted)

    Triangle canonicalization
        Positions 0-8 are clamped to [0, QUANT_MAX] and the three vertices are
        rotated so the ZYX-lexicographic minimum vertex is first.  Positions 9-11
        are forced to TRI_PAD regardless of what was predicted there.

    Quad canonicalization
        All 12 positions are clamped to [0, QUANT_MAX] and the four vertices are
        rotated so the ZYX-lexicographic minimum vertex is first.

    Inconsistent TRI_PAD positions
        token[9] is the sole oracle.  Example: token[9]=coord → treated as quad;
        any rogue TRI_PAD elsewhere is clamped to QUANT_MAX (127).
        Example: token[9]=TRI_PAD → treated as triangle; positions 9-11 are
        forced to TRI_PAD.

    Winding order is preserved (cyclic rotation only, no flip).
    """
    if face12[9].item() > QUANT_MAX:                       # triangle
        coord_tokens = face12[:9].clamp(0, QUANT_MAX).reshape(3, 3)
        keys    = [(int(v[2]), int(v[1]), int(v[0])) for v in coord_tokens]
        min_idx = keys.index(min(keys))
        rotated = torch.roll(coord_tokens, -min_idx, dims=0).reshape(9)
        pad     = face12.new_full((3,), TRI_PAD)
        return torch.cat([rotated, pad])
    else:                                                   # quad
        coord_tokens = face12.clamp(0, QUANT_MAX).reshape(4, 3)
        keys    = [(int(v[2]), int(v[1]), int(v[0])) for v in coord_tokens]
        min_idx = keys.index(min(keys))
        return torch.roll(coord_tokens, -min_idx, dims=0).reshape(12)
