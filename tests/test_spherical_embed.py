"""Tests for face_cartesian_to_spherical with the unified 12-token quad/tri layout.

Covers:
  (a) a 12-coord quad,
  (b) a 12-coord triangle with a trailing TRI_PAD pad (positions 9-11),
  (c) a pure 9-coord triangle (backward compatibility),
plus the consistency guarantee that the triangle's spherical content is identical
in the 9-coord and 12-coord (trailing-pad) layouts, a mixed batch, and the
error path for an unsupported channel count.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.constants import QUANT_MAX, TRI_PAD
from src.utils.geometry import face_cartesian_to_spherical

_NORM = QUANT_MAX + 1  # 128 — the divisor used in Decoder.embed_faces


def _norm(int_tokens):
    """Mimic embed_faces: integer tokens → float in [0, ~1.008]."""
    return torch.tensor(int_tokens, dtype=torch.float64) / _NORM


def _is_unit_interval(t):
    return bool((t >= -1e-6).all() and (t <= 1 + 1e-6).all())


def test_quad_12_shape_and_anchor():
    # Real quad: four vertices, all coords in [0, QUANT_MAX].
    quad = _norm([10, 20, 30,  10, 20, 90,  10, 80, 90,  10, 80, 30])
    out = face_cartesian_to_spherical(quad)

    assert out.shape == (12,)
    # v0 anchor preserved untouched.
    assert torch.allclose(out[0:3], quad[0:3])
    # The three spherical triplets are normalized to [0, 1].
    assert _is_unit_interval(out[3:12])


def test_triangle_12_preserves_tri_pad_suffix():
    pad = TRI_PAD
    tri12 = _norm([10, 20, 30,  10, 20, 90,  10, 80, 90,  pad, pad, pad])
    out = face_cartesian_to_spherical(tri12)

    assert out.shape == (12,)
    # TRI_PAD suffix passes through untouched (face-type signal survives).
    assert torch.allclose(out[9:12], tri12[9:12])
    # Suffix is still detectably out-of-range (normalized TRI_PAD = 129/128 > 1).
    assert bool((out[9:12] > 1.0).all())
    assert _is_unit_interval(out[3:9])


def test_pure_9_coord_backward_compatible():
    tri9 = _norm([10, 20, 30,  10, 20, 90,  10, 80, 90])
    out = face_cartesian_to_spherical(tri9)

    assert out.shape == (9,)
    # v0 anchor preserved.
    assert torch.allclose(out[0:3], tri9[0:3])
    assert _is_unit_interval(out[3:9])


def test_tri12_matches_tri9_on_spherical_part():
    """The triangle's spherical content must be identical across both layouts."""
    coords = [10, 20, 30,  10, 20, 90,  10, 80, 90]
    tri9  = _norm(coords)
    tri12 = _norm(coords + [TRI_PAD, TRI_PAD, TRI_PAD])

    out9  = face_cartesian_to_spherical(tri9)
    out12 = face_cartesian_to_spherical(tri12)

    # Positions 0-8 of the 12-token output == the 9-token output.
    assert torch.allclose(out12[0:9], out9, atol=1e-9)


def test_known_spherical_value_along_z_axis():
    # v0 at origin, v1 displaced purely along +z by 64 quant steps.
    quad = _norm([0, 0, 0,  64, 0, 0,  0, 64, 0,  0, 0, 64])
    out = face_cartesian_to_spherical(quad)

    # sph(v1 - v0): d = (dz=64/128, 0, 0) → r = 0.5 / sqrt(3); theta = 0; phi = 0.5
    r, theta, phi = out[3], out[4], out[5]
    assert math.isclose(float(r),     (64 / _NORM) / math.sqrt(3), abs_tol=1e-9)
    assert math.isclose(float(theta), 0.0,                          abs_tol=1e-9)
    assert math.isclose(float(phi),   0.5,                          abs_tol=1e-9)


def test_mixed_batch_tri_and_quad():
    """A batch containing both a TRI_PAD triangle and a quad is handled per-row."""
    tri12 = [10, 20, 30, 10, 20, 90, 10, 80, 90, TRI_PAD, TRI_PAD, TRI_PAD]
    quad  = [10, 20, 30, 10, 20, 90, 10, 80, 90, 10, 80, 30]
    batch = _norm([tri12, quad])              # (2, 12)
    out = face_cartesian_to_spherical(batch)

    assert out.shape == (2, 12)
    # Row 0 (triangle): equals the standalone triangle result.
    assert torch.allclose(out[0], face_cartesian_to_spherical(_norm(tri12)))
    # Row 1 (quad): equals the standalone quad result.
    assert torch.allclose(out[1], face_cartesian_to_spherical(_norm(quad)))


def test_explicit_mask_matches_threshold_on_real_data():
    """For real data, the explicit is_tri mask gives the same result as the fallback."""
    tri12 = [10, 20, 30, 10, 20, 90, 10, 80, 90, TRI_PAD, TRI_PAD, TRI_PAD]
    quad  = [10, 20, 30, 10, 20, 90, 10, 80, 90, 10, 80, 30]
    batch = _norm([tri12, quad])                 # (2, 12)

    mask = torch.tensor([True, False])           # exact mask from raw tokens
    out_mask      = face_cartesian_to_spherical(batch, is_tri=mask)
    out_threshold = face_cartesian_to_spherical(batch)   # fallback path

    assert torch.allclose(out_mask, out_threshold)


def test_mask_takes_precedence_over_values():
    """The mask overrides the value-based heuristic: an all-coord row forced to
    is_tri=True must be encoded with the triangle interpretation, proving the
    function relies on the mask rather than the float threshold."""
    all_coords = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    vec = _norm(all_coords)

    forced_tri  = face_cartesian_to_spherical(vec, is_tri=torch.tensor(True))
    forced_quad = face_cartesian_to_spherical(vec, is_tri=torch.tensor(False))

    # Triangle interpretation: v0 at 0-2, the 9-coord transform on positions 0-8,
    # and the trailing pad at 9-11 passed through untouched; quad uses all 4 verts.
    assert torch.allclose(forced_tri[0:3], vec[0:3])      # v0 anchor preserved
    assert torch.allclose(forced_tri[9:12], vec[9:12])    # trailing tail untouched
    # The two interpretations must differ (mask actually changes the encoding).
    assert not torch.allclose(forced_tri, forced_quad)
    # forced_tri's first 9 positions equal the pure 9-coord result of tokens 0-8.
    assert torch.allclose(forced_tri[0:9], face_cartesian_to_spherical(vec[0:9]))


def test_unsupported_channel_count_raises():
    with pytest.raises(ValueError):
        face_cartesian_to_spherical(torch.zeros(6, dtype=torch.float64))
