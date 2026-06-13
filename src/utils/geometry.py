import math

import torch
from jaxtyping import Float, Int
from torch import Tensor

from ..constants import QUANT_MAX, TRI_PAD

_SQRT3 = math.sqrt(3)


def face_cartesian_to_spherical(face_norm: Float[Tensor, "... 9"]) -> Float[Tensor, "... 9"]:
    """
    Convert normalized face coords [v0, v1, v2] (all in [0,1)) to
    [v0_global, r1, θ1, φ1, r2, θ2, φ2] where (r, θ, φ) are spherical
    coords of the relative vectors (v1-v0) and (v2-v0), all normalized to [0, 1].

    Coordinate order within each vertex: (z, y, x).
    Spherical convention: θ = polar angle from z-axis, φ = azimuthal in (y, x) plane.
    """
    v0 = face_norm[..., :3]   # (z0, y0, x0)
    v1 = face_norm[..., 3:6]
    v2 = face_norm[..., 6:9]

    def to_spherical(d: Tensor):
        # d: (..., 3) with order (dz, dy, dx)
        r = d.norm(dim=-1, keepdim=True)                              # (..., 1)
        safe_r = r.clamp(min=1e-8)
        theta = torch.acos((d[..., :1] / safe_r).clamp(-1, 1))       # polar from z
        phi   = torch.atan2(d[..., 1:2], d[..., 2:3])                # azimuthal in (y,x)
        r_n     = r     / _SQRT3                                       # [0, 1]
        theta_n = theta / math.pi                                      # [0, 1]
        phi_n   = (phi  + math.pi) / (2 * math.pi)                    # [0, 1]
        return torch.cat([r_n, theta_n, phi_n], dim=-1)

    sph1 = to_spherical(v1 - v0)
    sph2 = to_spherical(v2 - v0)
    return torch.cat([v0, sph1, sph2], dim=-1)


def canonical_face(verts: Int[Tensor, "3 3"]) -> Int[Tensor, "9"]:
    """Roll vertices so the ZYX-lexicographic minimum is first, then flatten."""
    keys = [(int(v[2]), int(v[1]), int(v[0])) for v in verts]
    min_idx = keys.index(min(keys))
    return torch.roll(verts, -min_idx, dims=0).reshape(9)


def canonical_face_12(face12: Int[Tensor, "12"]) -> Int[Tensor, "12"]:
    """Canonicalize a 12-token unified face (triangle or quad) for use in generate().

    Face-type oracle: token[0].
        token[0] > QUANT_MAX (127)  →  triangle  (TRI_PAD was predicted)
        token[0] ≤ QUANT_MAX (127)  →  quad       (real coordinate predicted)

    Triangle canonicalization
        Positions 0-2 are forced to TRI_PAD regardless of what was predicted there.
        Positions 3-11 are clamped to [0, QUANT_MAX] and the three vertices are
        rotated so the ZYX-lexicographic minimum vertex is first.

    Quad canonicalization
        All 12 positions are clamped to [0, QUANT_MAX] and the four vertices are
        rotated so the ZYX-lexicographic minimum vertex is first.

    Inconsistent TRI_PAD positions
        token[0] is the sole oracle.  Example: token[0]=coord, token[1]=TRI_PAD
        → treated as quad; the rogue TRI_PAD is clamped to QUANT_MAX (127).
        Example: token[0]=TRI_PAD, token[2]=coord → treated as triangle;
        positions 0-2 are forced to TRI_PAD.

    Winding order is preserved (cyclic rotation only, no flip).
    """
    if face12[0].item() > QUANT_MAX:                       # triangle
        coord_tokens = face12[3:].clamp(0, QUANT_MAX).reshape(3, 3)
        keys    = [(int(v[2]), int(v[1]), int(v[0])) for v in coord_tokens]
        min_idx = keys.index(min(keys))
        rotated = torch.roll(coord_tokens, -min_idx, dims=0).reshape(9)
        pad     = face12.new_full((3,), TRI_PAD)
        return torch.cat([pad, rotated])
    else:                                                   # quad
        coord_tokens = face12.clamp(0, QUANT_MAX).reshape(4, 3)
        keys    = [(int(v[2]), int(v[1]), int(v[0])) for v in coord_tokens]
        min_idx = keys.index(min(keys))
        return torch.roll(coord_tokens, -min_idx, dims=0).reshape(12)
