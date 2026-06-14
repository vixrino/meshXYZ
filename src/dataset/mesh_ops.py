"""Pure-numpy mesh operations for tokenization.

All functions in this module operate on numpy arrays and have no dependency on
torch, trimesh, or any training framework.  This makes them importable in unit
tests without a GPU environment and usable in offline data-preparation scripts.

mesh_dataset.py imports from here and adds the Dataset / DataModule wrappers.
"""

import numpy as np

from ..constants import QUANT_MAX, TRI_PAD

# 90° rotation matrices for X, Y, Z axes (column-vector convention)
_ROT90: np.ndarray = np.array([
    [[1,  0,  0], [0,  0, -1], [0,  1,  0]],  # X-axis
    [[0,  0,  1], [0,  1,  0], [-1, 0,  0]],  # Y-axis
    [[0, -1,  0], [1,  0,  0], [0,  0,  1]],  # Z-axis
], dtype=np.float64)


def random_90_rotation_matrix() -> np.ndarray:
    """50% identity; otherwise a random rotation composed of k×90° per axis, k ∈ {1,2,3}."""
    if np.random.randint(0, 2) == 0:
        return np.eye(3)
    R = np.eye(3)
    for axis in range(3):
        k = np.random.randint(1, 4)
        R = np.linalg.matrix_power(_ROT90[axis], k) @ R
    return R


def random_reflection_matrix() -> np.ndarray:
    """Reflect along one randomly chosen axis (x, y, or z), or no reflection (25% each)."""
    R = np.eye(3)
    axis = np.random.randint(0, 4)  # 0=none, 1=x, 2=y, 3=z
    if axis > 0:
        R[axis - 1, axis - 1] = -1.0
    return R


def quantize_vertices(verts: np.ndarray) -> np.ndarray:
    """Min-max quantize vertices to [0, QUANT_MAX] preserving aspect ratio."""
    v_min = verts.min(axis=0, keepdims=True)
    v_max = verts.max(axis=0, keepdims=True)
    scale = (v_max - v_min).max() + 1e-8
    verts_norm = (verts - v_min) / scale
    return np.round(verts_norm * QUANT_MAX).clip(0, QUANT_MAX).astype(np.int64)


def normalize_point_cloud(pc: np.ndarray) -> np.ndarray:
    center = pc.mean(axis=0, keepdims=True)
    pc = pc - center
    scale = np.abs(pc).max() + 1e-8
    return pc / scale


def _normalize_face_vertices(faces: np.ndarray, verts_q: np.ndarray) -> np.ndarray:
    """Rotate each triangle's vertex ring so the ZYX-lex-smallest vertex comes first.

    Returns (F, 9) int64 — quantized face coords in normalized vertex order.
    ZYX sort key: coord[2]*128² + coord[1]*128 + coord[0].
    """
    F = len(faces)
    face_verts = verts_q[faces]  # (F, 3, 3)

    max_val = int(QUANT_MAX) + 1
    keys = (face_verts[:, :, 2].astype(np.int64) * (max_val ** 2) +
            face_verts[:, :, 1].astype(np.int64) * max_val +
            face_verts[:, :, 0].astype(np.int64))  # (F, 3)

    min_idx   = np.argmin(keys, axis=1)                          # (F,)
    shift_idx = (np.arange(3)[None, :] + min_idx[:, None]) % 3  # (F, 3)
    row_idx   = np.arange(F)[:, None]

    return face_verts[row_idx, shift_idx].reshape(F, 9).astype(np.int64)


def _normalize_quad_vertices(faces: np.ndarray, verts_q: np.ndarray) -> np.ndarray:
    """Rotate each quad's vertex ring so the ZYX-lex-smallest vertex comes first.

    Identical logic to _normalize_face_vertices but for 4-vertex faces.
    Only cyclic rotations are applied (never flips) so face orientation is preserved.

    Returns (Q, 12) int64 — quantized quad coords in normalized vertex order.
    """
    Q = len(faces)
    face_verts = verts_q[faces]  # (Q, 4, 3)

    max_val = int(QUANT_MAX) + 1
    keys = (face_verts[:, :, 2].astype(np.int64) * (max_val ** 2) +
            face_verts[:, :, 1].astype(np.int64) * max_val +
            face_verts[:, :, 0].astype(np.int64))  # (Q, 4)

    min_idx   = np.argmin(keys, axis=1)                          # (Q,)
    shift_idx = (np.arange(4)[None, :] + min_idx[:, None]) % 4  # (Q, 4)
    row_idx   = np.arange(Q)[:, None]

    return face_verts[row_idx, shift_idx].reshape(Q, 12).astype(np.int64)


def _to_unified_12_tokens(
    face_seq_tri:  np.ndarray,   # (T, 9)  int64
    face_seq_quad: np.ndarray,   # (Q, 12) int64
    tri_pad_token: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine triangle and quad face sequences into a unified (F, 12) block format.

    Triangle faces are padded to 12 tokens by prepending 3 *tri_pad_token* values:
        [TRI_PAD, TRI_PAD, TRI_PAD, v0_c0, v0_c1, v0_c2, v1_c0, …, v2_c2]

    Quad faces are unchanged (already 12 tokens):
        [v0_c0, v0_c1, v0_c2, v1_c0, …, v3_c2]

    Returns
    -------
    face_seq_12 : (T+Q, 12) int64
        Unified face sequence — triangles first, then quads.
    is_quad : (T+Q,) bool
        True for quad faces, False for triangle faces.
    """
    parts:      list[np.ndarray] = []
    is_q_parts: list[np.ndarray] = []

    T = len(face_seq_tri)
    Q = len(face_seq_quad)

    if T > 0:
        pad = np.full((T, 3), tri_pad_token, dtype=np.int64)
        parts.append(np.concatenate([pad, face_seq_tri], axis=1))  # (T, 12)
        is_q_parts.append(np.zeros(T, dtype=bool))

    if Q > 0:
        parts.append(face_seq_quad)                                 # (Q, 12)
        is_q_parts.append(np.ones(Q, dtype=bool))

    if not parts:
        return np.empty((0, 12), dtype=np.int64), np.empty(0, dtype=bool)

    return np.concatenate(parts, axis=0), np.concatenate(is_q_parts)


def build_edge_adjacency(face_seq_q: np.ndarray) -> np.ndarray:
    """Compute face adjacency using quantized vertex positions as vertex identifiers.

    Uses quantized coordinates instead of vertex indices so that meshes with
    duplicated vertices at seams (UV seams, hard edges) are handled correctly.

    face_seq_q: (F, 9) array of quantized face coords [v0c0,v0c1,v0c2, v1c0,…, v2c2]

    Returns (F, 3) int64 — slot k holds the face index of the neighbor sharing
    edge k, or -1 for boundary edges:
        slot 0: edge (v0, v1)
        slot 1: edge (v1, v2)
        slot 2: edge (v2, v0)
    """
    F = len(face_seq_q)
    B = int(QUANT_MAX) + 1  # 128

    vk = (face_seq_q[:, 0::3].astype(np.int64) * B * B +
          face_seq_q[:, 1::3].astype(np.int64) * B +
          face_seq_q[:, 2::3].astype(np.int64))              # (F, 3)

    all_edges = np.vstack([vk[:, [0, 1]], vk[:, [1, 2]], vk[:, [2, 0]]])
    all_edges = np.sort(all_edges, axis=1)

    edge_keys = all_edges[:, 0] * (B ** 3) + all_edges[:, 1]
    face_ids  = np.tile(np.arange(F), 3)
    slot_ids  = np.repeat(np.arange(3), F)

    sort_perm   = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[sort_perm]

    matches = sorted_keys[:-1] == sorted_keys[1:]
    idx1 = sort_perm[:-1][matches]
    idx2 = sort_perm[1:][matches]

    adj = np.full((F, 3), -1, dtype=np.int64)
    adj[face_ids[idx1], slot_ids[idx1]] = face_ids[idx2]
    adj[face_ids[idx2], slot_ids[idx2]] = face_ids[idx1]
    return adj


def build_edge_adjacency_unified(face_seq_12: np.ndarray) -> np.ndarray:
    """Compute face adjacency for a mixed (F, 12) unified face sequence.

    Detects face type from the first token:
        face_seq_12[:, 0] == TRI_PAD (129) → triangle (coords at positions 3–11, 3 edges)
        face_seq_12[:, 0] in [0, QUANT_MAX] → quad (coords at positions 0–11, 4 edges)

    Edges are identified by quantized vertex key pairs — same as build_edge_adjacency —
    so tri-quad shared edges are detected correctly even across face types.

    Returns (F, 4) int64 — slot k holds the neighbor face index or -1 for boundary:
        triangles: slots 0/1/2 are (v0,v1)/(v1,v2)/(v2,v0); slot 3 is always -1
        quads:     slots 0/1/2/3 are (v0,v1)/(v1,v2)/(v2,v3)/(v3,v0)

    Example — one quad (face 0) adjacent to one triangle (face 1) sharing one edge:

        Quad  [v0,v1,v2,v3]: v0=(0,0, 0), v1=(0,0,10), v2=(0,10,10), v3=(0,10, 0)
        Triangle [v1,v4,v2]: v1=(0,0,10),              v2=(0,10,10), v4=(0, 0,20)

        Shared edge: undirected (v1, v2).  Two vertices shared, no other overlap.
        v4 is unique to the triangle; v0 and v3 are unique to the quad.

        Quad slot 1  → (v1,v2) → neighbor = face 1 (triangle)
        Triangle slot 2 → (v2,v1) sorted = (v1,v2) → neighbor = face 0 (quad)

        face_neighbors[0, 1] = 1;  face_neighbors[0, 0/2/3] = -1
        face_neighbors[1, 2] = 0;  face_neighbors[1, 0/1/3] = -1
    """
    F = len(face_seq_12)
    B = int(QUANT_MAX) + 1  # 128

    tri_mask = face_seq_12[:, 0] == TRI_PAD   # (F,) bool
    tri_idx  = np.where( tri_mask)[0]
    quad_idx = np.where(~tri_mask)[0]

    # --- vertex key extraction -------------------------------------------------

    if len(tri_idx) > 0:
        fs_t = face_seq_12[tri_idx, 3:].astype(np.int64)       # (F_T, 9)
        vk_t = (fs_t[:, 0::3] * B * B + fs_t[:, 1::3] * B + fs_t[:, 2::3])  # (F_T, 3)
    else:
        vk_t = np.empty((0, 3), dtype=np.int64)

    if len(quad_idx) > 0:
        fs_q = face_seq_12[quad_idx].astype(np.int64)           # (F_Q, 12)
        vk_q = (fs_q[:, 0::3] * B * B + fs_q[:, 1::3] * B + fs_q[:, 2::3])  # (F_Q, 4)
    else:
        vk_q = np.empty((0, 4), dtype=np.int64)

    # --- edge list construction ------------------------------------------------

    if len(tri_idx) > 0:
        tri_edges = np.vstack([
            vk_t[:, [0, 1]],
            vk_t[:, [1, 2]],
            vk_t[:, [2, 0]],
        ])
        tri_edges    = np.sort(tri_edges, axis=1)
        tri_face_ids = np.tile(tri_idx, 3)
        tri_slot_ids = np.repeat(np.arange(3), len(tri_idx))
    else:
        tri_edges    = np.empty((0, 2), dtype=np.int64)
        tri_face_ids = np.empty(0,      dtype=np.int64)
        tri_slot_ids = np.empty(0,      dtype=np.int64)

    if len(quad_idx) > 0:
        quad_edges = np.vstack([
            vk_q[:, [0, 1]],
            vk_q[:, [1, 2]],
            vk_q[:, [2, 3]],
            vk_q[:, [3, 0]],
        ])
        quad_edges    = np.sort(quad_edges, axis=1)
        quad_face_ids = np.tile(quad_idx, 4)
        quad_slot_ids = np.repeat(np.arange(4), len(quad_idx))
    else:
        quad_edges    = np.empty((0, 2), dtype=np.int64)
        quad_face_ids = np.empty(0,      dtype=np.int64)
        quad_slot_ids = np.empty(0,      dtype=np.int64)

    # --- combine and find matches ----------------------------------------------

    all_edges    = np.vstack([tri_edges, quad_edges])
    all_face_ids = np.concatenate([tri_face_ids, quad_face_ids])
    all_slot_ids = np.concatenate([tri_slot_ids, quad_slot_ids])

    edge_keys = all_edges[:, 0] * (B ** 3) + all_edges[:, 1]

    sort_perm   = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[sort_perm]

    matches = sorted_keys[:-1] == sorted_keys[1:]
    idx1 = sort_perm[:-1][matches]
    idx2 = sort_perm[1:][matches]

    adj = np.full((F, 4), -1, dtype=np.int64)
    adj[all_face_ids[idx1], all_slot_ids[idx1]] = all_face_ids[idx2]
    adj[all_face_ids[idx2], all_slot_ids[idx2]] = all_face_ids[idx1]
    return adj


def _sample_surface(verts: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """Weighted random surface sampling without constructing a Trimesh object."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / (areas.sum() + 1e-8)
    # Defensive normalization: ensure probs sum to exactly 1 (avoids floating point drift)
    probs = np.asarray(probs, dtype=np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        # All-zero areas (degenerate mesh) — fall back to uniform sampling
        probs = np.ones(len(faces), dtype=np.float64) / len(faces)
    else:
        probs = probs / probs_sum  # exact renormalization
    # Defensive normalization: ensure probs sum to exactly 1 (avoids floating point drift)
    probs = np.asarray(probs, dtype=np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        # All-zero areas (degenerate mesh) — fall back to uniform sampling
        probs = np.ones(len(faces), dtype=np.float64) / len(faces)
    else:
        probs = probs / probs_sum  # exact renormalization
    # Defensive normalization: ensure probs sum to exactly 1 (avoids floating point drift)
    probs = np.asarray(probs, dtype=np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        # All-zero areas (degenerate mesh) — fall back to uniform sampling
        probs = np.ones(len(faces), dtype=np.float64) / len(faces)
    else:
        probs = probs / probs_sum  # exact renormalization
    # Defensive normalization: ensure probs sum to exactly 1 (avoids floating point drift)
    probs = np.asarray(probs, dtype=np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        # All-zero areas (degenerate mesh) — fall back to uniform sampling
        probs = np.ones(len(faces), dtype=np.float64) / len(faces)
    else:
        probs = probs / probs_sum  # exact renormalization
    chosen = np.random.choice(len(faces), size=num_points, p=probs)
    r1 = np.random.rand(num_points, 1)
    r2 = np.random.rand(num_points, 1)
    sqrt_r1 = np.sqrt(r1)
    pts = (1 - sqrt_r1) * v0[chosen] + sqrt_r1 * (1 - r2) * v1[chosen] + sqrt_r1 * r2 * v2[chosen]
    return pts.astype(np.float32)


def _quads_to_tris_for_sampling(faces_quad: np.ndarray) -> np.ndarray:
    """Fan-triangulate quad faces into triangles for surface-area sampling only.

    Each quad [v0,v1,v2,v3] becomes two triangles [v0,v1,v2] and [v0,v2,v3].
    The result is used only for _sample_surface; it is NOT stored or tokenized.
    """
    if len(faces_quad) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.vstack([
        faces_quad[:, [0, 1, 2]],
        faces_quad[:, [0, 2, 3]],
    ])


def process_mesh(
    verts:       np.ndarray,
    faces_tri:   np.ndarray,
    num_points:  int,
    augment:     bool = False,
    face_layout: str  = "tri",
    faces_quad:  "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    # Returns: (pc (points,3) float32, face_seq (F,9|12) int64, face_neighbors (F,3|4) int64)
    """Apply augmentation, quantize, and compute adjacency.

    Parameters
    ----------
    verts, faces_tri :
        Vertex positions and triangle faces from load_mesh_raw.
    num_points :
        Number of surface points to sample for the point cloud.
    augment :
        If True, apply random 90° rotation + random reflection.
    face_layout :
        "tri"  — triangle-only path (original behavior, returns (F,9) and (F,3)).
        "quad" — unified 12-token path, may include triangles with TRI_PAD prefix.
    faces_quad :
        (Q, 4) int64 array of quad faces.  Required when face_layout="quad";
        may be empty if the mesh has no quads.
    """
    if augment:
        R_3x3 = random_reflection_matrix() @ random_90_rotation_matrix()
        verts = verts @ R_3x3.T

    _fq = faces_quad if faces_quad is not None else np.empty((0, 4), dtype=np.int64)

    # ---- TRIANGLE-ONLY PATH (original, unchanged) ----------------------------
    if face_layout == "tri":
        pc = normalize_point_cloud(_sample_surface(verts, faces_tri, num_points))
        verts_q        = quantize_vertices(verts)
        face_seq       = _normalize_face_vertices(faces_tri, verts_q)
        face_neighbors = build_edge_adjacency(face_seq)
        return pc, face_seq, face_neighbors

    # ---- QUAD / MIXED PATH ---------------------------------------------------
    tris_for_sampling = _quads_to_tris_for_sampling(_fq)
    all_tris = (
        np.vstack([faces_tri, tris_for_sampling])
        if len(faces_tri) > 0 else tris_for_sampling
    )
    if len(all_tris) == 0:
        raise ValueError(
            "process_mesh: mesh has no faces to sample from "
            "(faces_tri and faces_quad are both empty)."
        )

    pc      = normalize_point_cloud(_sample_surface(verts, all_tris, num_points))
    verts_q = quantize_vertices(verts)

    seq_tri  = (
        _normalize_face_vertices(faces_tri, verts_q)
        if len(faces_tri) > 0 else np.empty((0, 9),  dtype=np.int64)
    )
    seq_quad = (
        _normalize_quad_vertices(_fq, verts_q)
        if len(_fq)       > 0 else np.empty((0, 12), dtype=np.int64)
    )

    face_seq_12, _  = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)

    # Global ZYX sort: order all faces by their first real vertex's ZYX key.
    # Triangle first real vertex is at tokens [3,4,5] = [x,y,z]; quad at [0,1,2].
    # ZYX key = z*B² + y*B + x, matching the intra-face canonicalization key.
    # This interleaves tri and quad faces spatially (instead of a rigid tri-then-quad
    # block), producing diverse quad→tri and tri→quad context transitions for the
    # autoregressive decoder to learn face-type distinction.
    if len(face_seq_12) > 1:
        _B     = int(QUANT_MAX) + 1
        _is_t  = face_seq_12[:, 0] == TRI_PAD
        _x     = np.where(_is_t, face_seq_12[:, 3], face_seq_12[:, 0])
        _y     = np.where(_is_t, face_seq_12[:, 4], face_seq_12[:, 1])
        _z     = np.where(_is_t, face_seq_12[:, 5], face_seq_12[:, 2])
        _keys  = _z.astype(np.int64) * _B * _B + _y.astype(np.int64) * _B + _x.astype(np.int64)
        face_seq_12 = face_seq_12[np.argsort(_keys, kind="stable")]

    face_neighbors  = build_edge_adjacency_unified(face_seq_12)

    return pc, face_seq_12, face_neighbors
