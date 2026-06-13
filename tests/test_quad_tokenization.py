"""Tests for Phase 2 quad tokenization — imports from src/dataset/mesh_ops.py (pure numpy).

torch-dependent tests (collate_fn, MeshDataset) use pytest.importorskip so they
are skipped in a numpy-only environment and run automatically when torch is present
(e.g., in the Colab training environment).

Test categories
---------------
1. TRI_PAD prefix  — triangle faces in unified 12-token layout have TRI_PAD at [0:3]
2. Quad no-prefix  — quad faces occupy all 12 positions with coordinate tokens
3. Canonical ordering — smallest ZYX vertex is rotated to position 0 (tri and quad)
4. Round-trip      — tokenize → detokenize → geometry matches the original
5. Adjacency: two adjacent quads sharing one edge
6. Adjacency: realistic tri-quad shared edge (exactly one edge = 2 vertices shared)
7. Adjacency regression — triangle-only path gives same result as original function
8. collate_fn      — produces (B, F_max, 12) faces and (B, F_max, 4) neighbors (torch)
9. process_mesh integration — correct output shapes for both tri and quad modes
"""

import textwrap

import numpy as np
import pytest

from src.constants import EOS_COORD, QUANT_MAX, TRI_PAD
from src.dataset.mesh_ops import (
    _normalize_face_vertices,
    _normalize_quad_vertices,
    _to_unified_12_tokens,
    build_edge_adjacency,
    build_edge_adjacency_unified,
    process_mesh,
    quantize_vertices,
)
from src.dataset.obj_parser import parse_obj

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_B = QUANT_MAX + 1  # 128


def _vkey(v: np.ndarray) -> int:
    """Scalar ZYX vertex key matching build_edge_adjacency's hash."""
    return int(v[2]) * _B * _B + int(v[1]) * _B + int(v[0])


def _write_obj(tmp_path, content: str, name: str = "mesh.obj") -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. TRI_PAD prefix
# ---------------------------------------------------------------------------

def test_tri_pad_at_positions_0_1_2():
    """All triangle rows in the unified sequence must start with [TRI_PAD, TRI_PAD, TRI_PAD]."""
    verts_q = np.array([
        [ 0,  0,  0],
        [10,  0,  0],
        [ 0, 10,  0],
        [ 0,  0, 10],
    ], dtype=np.int64)
    faces_tri = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

    seq_tri  = _normalize_face_vertices(faces_tri, verts_q)          # (2, 9)
    seq_quad = np.empty((0, 12), dtype=np.int64)
    face_seq_12, is_quad = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)

    assert face_seq_12.shape == (2, 12)
    assert not is_quad.any()

    assert (face_seq_12[:, 0] == TRI_PAD).all()
    assert (face_seq_12[:, 1] == TRI_PAD).all()
    assert (face_seq_12[:, 2] == TRI_PAD).all()
    assert (face_seq_12[:, 3:] <= QUANT_MAX).all()
    assert (face_seq_12[:, 3:] >= 0).all()


# ---------------------------------------------------------------------------
# 2. Quad — no TRI_PAD, all 12 positions are coordinates
# ---------------------------------------------------------------------------

def test_quad_has_no_tri_pad():
    """Quad faces must not start with TRI_PAD; all 12 positions are coord tokens."""
    verts_q = np.array([
        [ 0,  0,  0],
        [20,  0,  0],
        [20, 20,  0],
        [ 0, 20,  0],
    ], dtype=np.int64)
    faces_quad = np.array([[0, 1, 2, 3]], dtype=np.int64)

    seq_quad = _normalize_quad_vertices(faces_quad, verts_q)
    seq_tri  = np.empty((0, 9), dtype=np.int64)
    face_seq_12, is_quad = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)

    assert face_seq_12.shape == (1, 12)
    assert is_quad.all()
    assert face_seq_12[0, 0] != TRI_PAD
    assert (face_seq_12[0] <= QUANT_MAX).all()
    assert (face_seq_12[0] >= 0).all()


# ---------------------------------------------------------------------------
# 3. Canonical vertex ordering
# ---------------------------------------------------------------------------

def test_tri_canonical_smallest_first():
    """_normalize_face_vertices must rotate so the ZYX-smallest vertex is first."""
    # v0 stored as [coord0,coord1,coord2]; key = coord[2]*B^2 + coord[1]*B + coord[0]
    # v0=(10,30,50) key=50*128²+30*128+10=823050
    # v1=(20,50,10) key=10*128²+50*128+20=170420  ← smallest
    # v2=(90,10,80) key=80*128²+10*128+90≈1313370
    verts_q = np.array([
        [10, 30, 50],
        [20, 50, 10],
        [90, 10, 80],
    ], dtype=np.int64)
    faces_tri = np.array([[0, 1, 2]], dtype=np.int64)

    seq = _normalize_face_vertices(faces_tri, verts_q)   # (1, 9)
    first = seq[0, :3]
    key_first = int(first[2]) * _B * _B + int(first[1]) * _B + int(first[0])
    keys = [int(verts_q[i, 2]) * _B * _B + int(verts_q[i, 1]) * _B + int(verts_q[i, 0])
            for i in range(3)]
    assert key_first == min(keys)


def test_quad_canonical_smallest_first():
    """_normalize_quad_vertices must rotate so the ZYX-smallest vertex is first."""
    # Create quad with v0 at ring position 1 (not first)
    # v0=(0,0,0) key=0 ← smallest;  feed quad as [v1, v0, v2, v3]
    verts_q = np.array([
        [ 0,  0,  0],   # v0  key=0
        [10,  0,  0],   # v1  key=10
        [10, 10,  0],   # v2  key=10*128+10=1290
        [ 0, 10,  0],   # v3  key=10*128=1280
    ], dtype=np.int64)
    faces_quad = np.array([[1, 0, 3, 2]], dtype=np.int64)   # v0 is at ring position 1

    seq = _normalize_quad_vertices(faces_quad, verts_q)   # (1, 12)
    first = seq[0, :3]
    key_first = int(first[2]) * _B * _B + int(first[1]) * _B + int(first[0])
    keys = [int(verts_q[i, 2]) * _B * _B + int(verts_q[i, 1]) * _B + int(verts_q[i, 0])
            for i in range(4)]
    assert key_first == min(keys)


def test_quad_canonical_preserves_ring_order():
    """Canonicalization must be a cyclic rotation, never a reversal."""
    # Quad ring [v0,v1,v2,v3] with v2 as smallest.  After rotation: [v2,v3,v0,v1].
    # Reversed ring would be [v2,v1,v0,v3], which is different.
    verts_q = np.array([
        [30,  5,  7],   # v0 key > v2
        [15, 10,  3],   # v1 key > v2
        [ 2,  1,  0],   # v2 key=0*128²+1*128+2=130  ← smallest
        [50,  8,  9],   # v3 key > v2
    ], dtype=np.int64)
    faces_quad = np.array([[0, 1, 2, 3]], dtype=np.int64)

    seq = _normalize_quad_vertices(faces_quad, verts_q)   # (1, 12)
    verts_out = seq[0].reshape(4, 3)

    # After rotation, ring should be [v2, v3, v0, v1]
    expected_order = [2, 3, 0, 1]
    for out_pos, orig_idx in enumerate(expected_order):
        assert np.array_equal(verts_out[out_pos], verts_q[orig_idx]), (
            f"Ring position {out_pos}: got {verts_out[out_pos]}, "
            f"expected {verts_q[orig_idx]} (v{orig_idx})"
        )


# ---------------------------------------------------------------------------
# 4. Round-trip: tokenize → read back coords → matches original geometry
# ---------------------------------------------------------------------------

def test_tri_round_trip():
    """Triangle coords survive tokenization: the 9 coord tokens equal quantized geometry."""
    verts_q = np.array([[ 5, 12, 33], [77,  4,  9], [20, 60,  1]], dtype=np.int64)
    faces_tri = np.array([[0, 1, 2]], dtype=np.int64)

    seq9 = _normalize_face_vertices(faces_tri, verts_q)
    seq12 = np.concatenate([np.full((1, 3), TRI_PAD, dtype=np.int64), seq9], axis=1)
    coords = seq12[0, 3:].reshape(3, 3)

    for row in coords:
        assert any(np.array_equal(row, verts_q[i]) for i in range(3)), (
            f"Detokenized vertex {row.tolist()} not in original verts_q"
        )


def test_quad_round_trip():
    """Quad coords survive tokenization: all 12 tokens equal quantized geometry."""
    verts_q = np.array([
        [ 0,  0,  0], [10,  0,  0], [10, 10,  0], [ 0, 10,  0],
    ], dtype=np.int64)
    faces_quad = np.array([[0, 1, 2, 3]], dtype=np.int64)

    seq12 = _normalize_quad_vertices(faces_quad, verts_q)
    coords = seq12[0].reshape(4, 3)

    for row in coords:
        assert any(np.array_equal(row, verts_q[i]) for i in range(4)), (
            f"Detokenized vertex {row.tolist()} not in original verts_q"
        )


def test_mixed_round_trip(tmp_path):
    """Mixed OBJ: both triangle and quad vertices survive through the tokenization pipeline."""
    path = _write_obj(tmp_path, """
        v 0  0 0
        v 10 0 0
        v 10 10 0
        v 0  10 0
        v 5  0  5
        f 1 2 3 4
        f 1 2 5
    """)
    result = parse_obj(path)
    verts_q = quantize_vertices(result.vertices)

    seq_tri  = _normalize_face_vertices(result.faces_tri,  verts_q)
    seq_quad = _normalize_quad_vertices(result.faces_quad, verts_q)
    face_seq_12, is_quad = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)

    assert face_seq_12.shape == (2, 12)
    assert (face_seq_12[~is_quad, 0] == TRI_PAD).all()
    assert (face_seq_12[is_quad,  0] != TRI_PAD).all()


# ---------------------------------------------------------------------------
# 5. Adjacency: two adjacent quads sharing one edge
# ---------------------------------------------------------------------------

def test_adjacency_two_adjacent_quads():
    """Two quads sharing one edge must have matching neighbor slots."""
    # Quad 0: v0=(0,0,0), v1=(0,0,10), v2=(0,10,10), v3=(0,10,0)
    # Quad 1: v1=(0,0,10), v4=(0,0,20), v5=(0,10,20), v2=(0,10,10)
    # Shared edge: undirected {v1, v2} — one edge, two vertices shared.
    verts_q = np.array([
        [0,  0,  0],   # v0
        [0,  0, 10],   # v1
        [0, 10, 10],   # v2
        [0, 10,  0],   # v3
        [0,  0, 20],   # v4
        [0, 10, 20],   # v5
    ], dtype=np.int64)
    faces_quad = np.array([
        [0, 1, 2, 3],   # quad 0
        [1, 4, 5, 2],   # quad 1
    ], dtype=np.int64)

    seq_quad = _normalize_quad_vertices(faces_quad, verts_q)
    face_seq_12, _ = _to_unified_12_tokens(np.empty((0, 9), dtype=np.int64), seq_quad, TRI_PAD)
    adj = build_edge_adjacency_unified(face_seq_12)

    assert adj.shape == (2, 4)

    v1_key = _vkey(verts_q[1])
    v2_key = _vkey(verts_q[2])
    shared = {v1_key, v2_key}

    def find_shared_slot(seq12_row: np.ndarray) -> int:
        vk = seq12_row.reshape(4, 3)
        for s in range(4):
            ka = _vkey(vk[s])
            kb = _vkey(vk[(s + 1) % 4])
            if {ka, kb} == shared:
                return s
        return -1

    slot0 = find_shared_slot(seq_quad[0])
    slot1 = find_shared_slot(seq_quad[1])

    assert slot0 != -1, "shared edge not found in canonical quad 0"
    assert slot1 != -1, "shared edge not found in canonical quad 1"
    assert adj[0, slot0] == 1
    assert adj[1, slot1] == 0
    for s in range(4):
        if s != slot0:
            assert adj[0, s] == -1
        if s != slot1:
            assert adj[1, s] == -1


# ---------------------------------------------------------------------------
# 6. Adjacency: realistic tri-quad shared edge (exactly one edge = 2 vertices)
# ---------------------------------------------------------------------------

def test_adjacency_tri_quad_one_shared_edge():
    """Quad and triangle sharing exactly one edge — two vertices shared, none others.

    Topology:
        Quad  [v0,v1,v2,v3]: v0=(0,0, 0), v1=(0,0,10), v2=(0,10,10), v3=(0,10, 0)
        Tri   [v1,v4,v2]:    v1=(0,0,10), v4=(0,0, 20), v2=(0,10,10)

        Shared edge: {v1, v2}.  v4 unique to triangle; v0, v3 unique to quad.
    """
    verts_q = np.array([
        [ 0,  0,  0],   # v0  key=0
        [ 0,  0, 10],   # v1  key=10
        [ 0, 10, 10],   # v2  key=10*128+10=1290
        [ 0, 10,  0],   # v3  key=10*128=1280
        [ 0,  0, 20],   # v4  key=20
    ], dtype=np.int64)

    faces_tri  = np.array([[1, 4, 2]], dtype=np.int64)    # tri:  v1, v4, v2
    faces_quad = np.array([[0, 1, 2, 3]], dtype=np.int64) # quad: v0, v1, v2, v3

    seq_tri  = _normalize_face_vertices(faces_tri,  verts_q)   # (1, 9)
    seq_quad = _normalize_quad_vertices(faces_quad, verts_q)   # (1, 12)

    # Triangles come first in _to_unified_12_tokens → face index 0
    # Quads come second → face index 1
    face_seq_12, is_quad = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)
    adj = build_edge_adjacency_unified(face_seq_12)

    assert adj.shape == (2, 4)

    v1_key = _vkey(verts_q[1])
    v2_key = _vkey(verts_q[2])
    shared = {v1_key, v2_key}

    # Locate shared edge slot in triangle (face 0)
    tri_vkeys = seq_tri[0].reshape(3, 3)
    tri_slot = next(
        s for s in range(3)
        if {_vkey(tri_vkeys[s]), _vkey(tri_vkeys[(s + 1) % 3])} == shared
    )

    # Locate shared edge slot in quad (face 1)
    quad_vkeys = seq_quad[0].reshape(4, 3)
    quad_slot = next(
        s for s in range(4)
        if {_vkey(quad_vkeys[s]), _vkey(quad_vkeys[(s + 1) % 4])} == shared
    )

    # Triangle is face 0, quad is face 1
    assert adj[0, tri_slot]  == 1, f"tri slot {tri_slot} → expected 1 (quad), got {adj[0, tri_slot]}"
    assert adj[1, quad_slot] == 0, f"quad slot {quad_slot} → expected 0 (tri), got {adj[1, quad_slot]}"

    # Triangle's slot 3 must always be -1
    assert adj[0, 3] == -1

    # All non-shared slots are boundary
    for s in range(3):
        if s != tri_slot:
            assert adj[0, s] == -1
    for s in range(4):
        if s != quad_slot:
            assert adj[1, s] == -1


# ---------------------------------------------------------------------------
# 7. Regression: triangle-only path is unchanged
# ---------------------------------------------------------------------------

def test_triangle_only_adjacency_matches_original():
    """build_edge_adjacency (old, (F,9)) and build_edge_adjacency_unified must agree."""
    verts_q = np.array([
        [ 0,  0,  0], [10,  0,  0], [20,  0,  0],
        [ 0, 10,  0], [10, 10,  0], [20, 10,  0],
    ], dtype=np.int64)
    faces_tri = np.array([
        [0, 1, 3], [1, 4, 3], [1, 2, 4],
    ], dtype=np.int64)

    seq9 = _normalize_face_vertices(faces_tri, verts_q)   # (3, 9) — old format
    adj3 = build_edge_adjacency(seq9)                       # (3, 3) — old function

    face_seq_12, _ = _to_unified_12_tokens(seq9, np.empty((0, 12), dtype=np.int64), TRI_PAD)
    adj4 = build_edge_adjacency_unified(face_seq_12)       # (3, 4)

    np.testing.assert_array_equal(
        adj3, adj4[:, :3],
        err_msg="Unified adjacency must match original for triangle-only input",
    )
    assert (adj4[:, 3] == -1).all(), "Slot 3 must be -1 for all triangle faces"


# ---------------------------------------------------------------------------
# 8. collate_fn (requires torch)
# ---------------------------------------------------------------------------

def test_collate_fn_quad_shapes():
    """collate_fn must produce (B, F_max, 12) faces and (B, F_max, 4) neighbors."""
    torch = pytest.importorskip("torch")
    from src.dataset.collate import collate_fn

    item_a = {
        "pc":             torch.zeros(2048, 3),
        "faces":          torch.randint(0, 127, (3, 12)).long(),
        "face_neighbors": torch.full((3, 4), -1, dtype=torch.long),
    }
    item_b = {
        "pc":             torch.zeros(2048, 3),
        "faces":          torch.randint(0, 127, (5, 12)).long(),
        "face_neighbors": torch.full((5, 4), -1, dtype=torch.long),
    }
    batch = collate_fn([item_a, item_b])

    assert batch["faces"].shape          == (2, 5, 12)
    assert batch["face_neighbors"].shape == (2, 5, 4)
    assert batch["lengths"].tolist()     == [3, 5]
    assert (batch["faces"][0, 3:] == EOS_COORD).all()
    assert (batch["face_neighbors"][0, 3:] == -1).all()


def test_collate_fn_tri_shapes_unchanged():
    """collate_fn must still produce (B, F_max, 9)/(B, F_max, 3) for tri mode."""
    torch = pytest.importorskip("torch")
    from src.dataset.collate import collate_fn

    item = {
        "pc":             torch.zeros(2048, 3),
        "faces":          torch.randint(0, 127, (4, 9)).long(),
        "face_neighbors": torch.full((4, 3), -1, dtype=torch.long),
    }
    batch = collate_fn([item, item])
    assert batch["faces"].shape          == (2, 4, 9)
    assert batch["face_neighbors"].shape == (2, 4, 3)


# ---------------------------------------------------------------------------
# 9. process_mesh integration (pure numpy, no torch needed)
# ---------------------------------------------------------------------------

def test_process_mesh_quad_output_shapes():
    """process_mesh in quad mode must return (F,12) faces and (F,4) neighbors."""
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], [0.5, 0.0, 1.0],
    ], dtype=np.float64)
    faces_tri  = np.array([[0, 1, 4]], dtype=np.int64)
    faces_quad = np.array([[0, 1, 2, 3]], dtype=np.int64)

    pc, face_seq, neighbors = process_mesh(
        verts, faces_tri, num_points=128, face_layout="quad", faces_quad=faces_quad,
    )

    assert pc.shape        == (128, 3)
    assert face_seq.shape  == (2, 12)
    assert neighbors.shape == (2, 4)
    # Tri face (index 0): TRI_PAD at positions 0-2
    assert face_seq[0, 0] == TRI_PAD
    assert face_seq[0, 1] == TRI_PAD
    assert face_seq[0, 2] == TRI_PAD
    # Quad face (index 1): no TRI_PAD
    assert face_seq[1, 0] != TRI_PAD


def test_process_mesh_tri_mode_unchanged():
    """process_mesh in tri mode must return (F,9) and (F,3) — original behavior."""
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    faces_tri = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

    pc, face_seq, neighbors = process_mesh(
        verts, faces_tri, num_points=64, face_layout="tri",
    )

    assert face_seq.shape  == (2, 9)
    assert neighbors.shape == (2, 3)
    assert (face_seq != TRI_PAD).all()


def test_process_mesh_quad_only(tmp_path):
    """Mesh with only quads (no triangles) must succeed and produce (Q,12) faces."""
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    # Top and bottom faces of a cube (quads only)
    faces_quad = np.array([
        [0, 1, 2, 3],   # bottom
        [4, 5, 6, 7],   # top
    ], dtype=np.int64)
    faces_tri = np.empty((0, 3), dtype=np.int64)

    pc, face_seq, neighbors = process_mesh(
        verts, faces_tri, num_points=64, face_layout="quad", faces_quad=faces_quad,
    )

    assert face_seq.shape  == (2, 12)
    assert neighbors.shape == (2, 4)
    # No TRI_PAD — all quad faces
    assert (face_seq[:, 0] != TRI_PAD).all()
