"""Unit tests for canonical_face_12 — the unified 12-token face canonicalization.

Run with:
    python3.11 -m pytest tests/test_phase3_geometry.py -v
"""
import pytest
torch = pytest.importorskip("torch")


def _import():
    from src.utils.geometry import canonical_face_12
    from src.constants import TRI_PAD, QUANT_MAX
    return canonical_face_12, TRI_PAD, QUANT_MAX


def test_canonical_face_12_triangle_basic():
    """Triangle: TRI_PAD forced at 0-2; coord vertices rotated to put minimum first.

    Input vertices (in ZYX storage):
        v0 = (10, 20, 30),  v1 = (5, 5, 5),  v2 = (50, 50, 50)
    Comparison key = (v[2], v[1], v[0]):
        v0 key = (30, 20, 10),  v1 key = (5, 5, 5),  v2 key = (50, 50, 50)
    min is v1 → roll by -1 → order becomes [v1, v2, v0]
    Expected output: [TRI_PAD, TRI_PAD, TRI_PAD, 5,5,5, 50,50,50, 10,20,30]
    """
    canonical_face_12, TRI_PAD, _ = _import()

    face = torch.tensor(
        [TRI_PAD, TRI_PAD, TRI_PAD, 10, 20, 30, 5, 5, 5, 50, 50, 50],
        dtype=torch.long,
    )
    result = canonical_face_12(face)

    assert result[:3].tolist()  == [TRI_PAD, TRI_PAD, TRI_PAD], "pad prefix must be TRI_PAD"
    assert result[3:6].tolist() == [5, 5, 5],     "min vertex must be first"
    assert result[6:9].tolist() == [50, 50, 50]
    assert result[9:12].tolist() == [10, 20, 30]
    assert len(result) == 12


def test_canonical_face_12_quad_basic():
    """Quad: all 12 coords; minimum vertex first, ring order preserved.

    Input vertices: v0=(100,50,80), v1=(10,10,10), v2=(90,30,60), v3=(70,20,40)
    Keys (v[2],v[1],v[0]):
        v0=(80,50,100), v1=(10,10,10), v2=(60,30,90), v3=(40,20,70)
    min is v1 → roll by -1 → [v1, v2, v3, v0]
    """
    canonical_face_12, _, _ = _import()

    face = torch.tensor(
        [100, 50, 80,  10, 10, 10,  90, 30, 60,  70, 20, 40],
        dtype=torch.long,
    )
    result = canonical_face_12(face)

    assert result[:3].tolist()  == [10, 10, 10],       "min vertex must be first"
    assert result[3:6].tolist() == [90, 30, 60],       "ring order: v1 → v2"
    assert result[6:9].tolist() == [70, 20, 40],       "ring order: v2 → v3"
    assert result[9:12].tolist() == [100, 50, 80],     "ring order: v3 → v0"
    assert result.max().item() <= 127


def test_canonical_face_12_inconsistent_pad_positions():
    """Policy: token[0] is the sole oracle for face type.

    Case: token[0] is coord-valid (≤ QUANT_MAX = 127) but token[1] = TRI_PAD (129).
    Interpretation: face type = QUAD (token[0] rules).
    Behaviour: TRI_PAD at position 1 is clamped to QUANT_MAX (127).
    No TRI_PAD value should appear anywhere in the output.

    Input layout (quad):
        v0 = [10, TRI_PAD=129, 20]  →  after clamp → [10, 127, 20]
        v1 = [5, 5, 5]
        v2 = [50, 50, 50]
        v3 = [80, 80, 80]
    Keys (v[2],v[1],v[0]):
        v0 clamped = (20, 127, 10)
        v1 = (5, 5, 5)   ← minimum
        v2 = (50, 50, 50)
        v3 = (80, 80, 80)
    min is v1 → roll by -1 → [v1, v2, v3, v0_clamped]
    """
    canonical_face_12, TRI_PAD, QUANT_MAX = _import()

    face = torch.tensor(
        [10, TRI_PAD, 20,  5, 5, 5,  50, 50, 50,  80, 80, 80],
        dtype=torch.long,
    )
    result = canonical_face_12(face)

    assert result.max().item() <= QUANT_MAX, \
        f"No TRI_PAD (={TRI_PAD}) may appear in a quad result; got max={result.max().item()}"
    assert result.min().item() >= 0
    assert len(result) == 12
    assert result[:3].tolist() == [5, 5, 5], "Minimum vertex (v1) must be first"
    assert result[3:6].tolist() == [50, 50, 50]
    assert result[6:9].tolist() == [80, 80, 80]
    # v0 clamped: position 1 was TRI_PAD=129 → clamped to 127
    assert result[9:12].tolist() == [10, 127, 20], \
        "Rogue TRI_PAD at token[1] must be clamped to QUANT_MAX (127)"
