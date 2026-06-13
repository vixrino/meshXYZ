"""Bit-identical regression check: triangle processing pipeline.

Verifies that process_mesh(face_layout='tri') in the new code produces
byte-identical (face_seq, face_neighbors, pc) output compared to the original
single-function implementation.

We reproduce the OLD implementation inline from the git history rather than
importing it, because the original code lived in mesh_dataset.py which has
been refactored.  All leaf functions (_normalize_face_vertices,
build_edge_adjacency, etc.) are unchanged and still live in mesh_ops.py;
the old process_mesh was just the same leaf calls in the same order.
The new triangle path is:

    if face_layout == "tri":
        pc             = normalize_point_cloud(_sample_surface(verts, faces_tri, num_points))
        verts_q        = quantize_vertices(verts)
        face_seq       = _normalize_face_vertices(faces_tri, verts_q)
        face_neighbors = build_edge_adjacency(face_seq)
        return pc, face_seq, face_neighbors

which is provably identical to the original.  The seed-fixed test below
confirms this at runtime on a concrete mesh.

Usage
-----
    python scripts/regression_check_tri.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.dataset.mesh_ops import (
    build_edge_adjacency,
    normalize_point_cloud,
    _normalize_face_vertices,
    _sample_surface,
    quantize_vertices,
    process_mesh,
)

# ── Reference: original process_mesh reproduced verbatim ──────────────────
def _reference_process_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact copy of the original mesh_dataset.process_mesh (triangle-only)."""
    pc = normalize_point_cloud(_sample_surface(verts, faces, num_points))
    verts_q        = quantize_vertices(verts)
    face_seq       = _normalize_face_vertices(faces, verts_q)
    face_neighbors = build_edge_adjacency(face_seq)
    return pc, face_seq, face_neighbors


# ── Test meshes ────────────────────────────────────────────────────────────
def _make_test_mesh(n_verts: int, n_faces: int, seed: int):
    rng = np.random.default_rng(seed)
    verts = rng.uniform(-1.0, 1.0, (n_verts, 3))
    # Random triangle faces — indices may repeat but that's fine for the test
    faces = rng.integers(0, n_verts, (n_faces, 3)).astype(np.int64)
    return verts, faces


MESHES = [
    ("small",  20, 30, 0),
    ("medium", 64, 120, 1),
    ("large",  256, 512, 2),
]
NUM_POINTS = 512
SEED = 99999          # numpy random seed for surface sampling

all_ok = True
print(f"{'Mesh':>8}  {'face_seq':^18}  {'face_nbrs':^18}  {'pc':^18}")
print("-" * 75)

for name, n_v, n_f, mesh_seed in MESHES:
    verts, faces = _make_test_mesh(n_v, n_f, mesh_seed)

    np.random.seed(SEED)
    pc_ref, seq_ref, adj_ref = _reference_process_mesh(verts, faces, NUM_POINTS)

    np.random.seed(SEED)
    pc_new, seq_new, adj_new = process_mesh(
        verts, faces, NUM_POINTS, face_layout="tri"
    )

    ok_seq = np.array_equal(seq_ref, seq_new)
    ok_adj = np.array_equal(adj_ref, adj_new)
    ok_pc  = np.array_equal(pc_ref,  pc_new)

    status = "PASS" if (ok_seq and ok_adj and ok_pc) else "FAIL"
    seq_str = f"{'✓' if ok_seq else '✗'} {seq_ref.shape}"
    adj_str = f"{'✓' if ok_adj else '✗'} {adj_ref.shape}"
    pc_str  = f"{'✓' if ok_pc  else '✗'} {pc_ref.shape}"
    print(f"{name:>8}  {seq_str:^18}  {adj_str:^18}  {pc_str:^18}  [{status}]")

    if not (ok_seq and ok_adj and ok_pc):
        all_ok = False
        if not ok_seq:
            diff = np.where(seq_ref != seq_new)
            print(f"         face_seq mismatch at indices: {diff}")
        if not ok_adj:
            diff = np.where(adj_ref != adj_new)
            print(f"         face_neighbors mismatch at indices: {diff}")
        if not ok_pc:
            diff = np.where(pc_ref != pc_new)
            print(f"         pc mismatch at indices: {diff}")

print()
if all_ok:
    print("✓  ALL PASS — triangle pipeline is bit-identical across all test meshes.")
    sys.exit(0)
else:
    print("✗  REGRESSION DETECTED — triangle pipeline has diverged.")
    sys.exit(1)
