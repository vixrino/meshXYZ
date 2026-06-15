"""End-to-end quad smoke test: sphere.obj → unified 12-token pipeline.

Generates a UV sphere OBJ with mixed topology (polar triangle caps + quad
latitude bands), runs the full Phase 2 pipeline, and prints the first 5
faces as human-readable text so we can visually verify the token layout
before feeding anything to the model.

Usage
-----
    python scripts/smoke_test_quad.py [--nLat N] [--nLon N] [--save path]

Defaults: --nLat 5  --nLon 8  (gives 16 tri + 24 quad = 40 faces)
"""
import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.constants import TRI_PAD, QUANT_MAX
from src.dataset.obj_parser import parse_obj
from src.dataset.mesh_ops import (
    quantize_vertices,
    _normalize_face_vertices,
    _normalize_quad_vertices,
    _to_unified_12_tokens,
    build_edge_adjacency_unified,
)


# ── UV sphere generator ────────────────────────────────────────────────────

def make_uv_sphere_obj(nLat: int = 5, nLon: int = 8) -> str:
    """Return an OBJ string for a UV sphere.

    Topology:
        • polar caps: nLon triangles each (top + bottom)
        • latitude bands: (nLat-1) bands × nLon quads

    Parameters
    ----------
    nLat : int  Number of latitude divisions (≥2 for at least one quad band)
    nLon : int  Number of longitude divisions (≥3)
    """
    lines: list[str] = [f"# UV sphere  nLat={nLat}  nLon={nLon}"]

    # ── vertices ──
    verts: list[tuple[float, float, float]] = []
    verts.append((0.0, 1.0, 0.0))   # top pole, index 0

    ring_starts: list[int] = []
    for i in range(1, nLat):
        phi = math.pi * i / nLat
        ring_starts.append(len(verts))
        for j in range(nLon):
            theta = 2 * math.pi * j / nLon
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.append((x, y, z))

    bot_pole_idx = len(verts)
    verts.append((0.0, -1.0, 0.0))   # bottom pole

    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    lines.append("")

    # ── top cap: triangles (pole → first ring) ──
    lines.append("# top cap — triangles")
    top_ring = ring_starts[0]
    for j in range(nLon):
        a = 1                              # top pole (1-based)
        b = top_ring + j + 1
        c = top_ring + (j + 1) % nLon + 1
        lines.append(f"f {a} {b} {c}")

    lines.append("")

    # ── latitude bands: quads ──
    lines.append("# latitude bands — quads")
    for i in range(len(ring_starts) - 1):
        r0 = ring_starts[i]
        r1 = ring_starts[i + 1]
        for j in range(nLon):
            v0 = r0 + j + 1
            v1 = r0 + (j + 1) % nLon + 1
            v2 = r1 + (j + 1) % nLon + 1
            v3 = r1 + j + 1
            lines.append(f"f {v0} {v1} {v2} {v3}")

    lines.append("")

    # ── bottom cap: triangles (last ring → pole) ──
    lines.append("# bottom cap — triangles")
    bot_ring = ring_starts[-1]
    for j in range(nLon):
        a = bot_ring + j + 1
        b = bot_pole_idx + 1             # bottom pole (1-based)
        c = bot_ring + (j + 1) % nLon + 1
        lines.append(f"f {a} {b} {c}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nLat", type=int, default=5)
    parser.add_argument("--nLon", type=int, default=8)
    parser.add_argument("--save", type=str, default=None,
                        help="Save the generated OBJ to this path.")
    args = parser.parse_args()

    # ── generate OBJ ──
    obj_text = make_uv_sphere_obj(nLat=args.nLat, nLon=args.nLon)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".obj", prefix="sphere_", delete=False
    ) as fh:
        fh.write(obj_text)
        obj_path = fh.name

    if args.save:
        with open(args.save, "w") as fh:
            fh.write(obj_text)
        print(f"Saved sphere OBJ → {args.save}")

    # ── parse ──
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    result = parse_obj(obj_path)
    os.unlink(obj_path)

    n_v = len(result.vertices)
    n_t = len(result.faces_tri)
    n_q = len(result.faces_quad)
    print(f"\nParsed:  {n_v} verts | {n_t} triangles | {n_q} quads")

    # ── tokenize ──
    verts_q  = quantize_vertices(result.vertices)
    seq_tri  = (_normalize_face_vertices(result.faces_tri, verts_q)
                if n_t > 0 else np.empty((0, 9),  dtype=np.int64))
    seq_quad = (_normalize_quad_vertices(result.faces_quad, verts_q)
                if n_q > 0 else np.empty((0, 12), dtype=np.int64))

    face_seq_12, is_quad = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)
    adj = build_edge_adjacency_unified(face_seq_12)

    F = len(face_seq_12)
    n_tri_out  = int((~is_quad).sum())
    n_quad_out = int(is_quad.sum())
    print(f"Unified: {F} faces total  ({n_tri_out} triangles + {n_quad_out} quads)")
    print(f"         face_seq shape = {face_seq_12.shape}   neighbors shape = {adj.shape}")

    # ── print first 5 faces ──
    SHOW = min(5, F)
    print(f"\nFirst {SHOW} faces (decoded):")
    print()
    print(f"  {'#':>3}  {'Type':>8}  {'Vertex tokens (ZYX per vertex)':^58}  Neighbors")
    print("  " + "-" * 100)

    for i in range(SHOW):
        face  = face_seq_12[i]
        iq    = bool(is_quad[i])
        nbrs  = adj[i].tolist()

        if iq:
            label     = "QUAD"
            verts_tok = face.reshape(4, 3)
            v_str = "  ".join(f"({r[0]:3d},{r[1]:3d},{r[2]:3d})" for r in verts_tok)
            tok_display = f"{v_str}"
        else:
            label     = "TRI"
            pad_vals  = face[9:].tolist()   # should all be TRI_PAD=129 (pad at end)
            verts_tok = face[:9].reshape(3, 3)
            v_str = "  ".join(f"({r[0]:3d},{r[1]:3d},{r[2]:3d})" for r in verts_tok)
            tok_display = f"{v_str} [pad={pad_vals[0]}×3]"

        nbr_fmt = str(nbrs)
        print(f"  {i:>3}  {label:>8}  {tok_display:<58}  {nbr_fmt}")

    print()
    print("Raw 12-token vectors (first 5 faces):")
    print()
    for i in range(SHOW):
        face  = face_seq_12[i]
        iq    = bool(is_quad[i])
        label = "QUAD" if iq else "TRI "
        raw   = " ".join(f"{t:3d}" for t in face)
        print(f"  [{label}] [{raw}]")

    print()
    # ── sanity checks ──
    tri_rows  = face_seq_12[~is_quad]
    quad_rows = face_seq_12[is_quad]

    checks = [
        ("All tri  rows: tokens[9:12] == TRI_PAD",
         (tri_rows[:, 9:] == TRI_PAD).all() if len(tri_rows) > 0 else True),
        ("All tri  rows: tokens[0:9] in [0,QUANT_MAX]",
         ((tri_rows[:, :9] >= 0) & (tri_rows[:, :9] <= QUANT_MAX)).all() if len(tri_rows) > 0 else True),
        ("All quad rows: tokens[9] != TRI_PAD",
         (quad_rows[:, 9] != TRI_PAD).all() if len(quad_rows) > 0 else True),
        ("All quad rows: tokens[0:12] in [0,QUANT_MAX]",
         ((quad_rows >= 0) & (quad_rows <= QUANT_MAX)).all() if len(quad_rows) > 0 else True),
        ("Neighbors shape == (F, 4)",
         adj.shape == (F, 4)),
        ("Tri face slot-3 neighbors == -1",
         (adj[~is_quad, 3] == -1).all() if len(tri_rows) > 0 else True),
    ]

    print("Sanity checks:")
    all_ok = True
    for desc, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status}  {desc}")
        if not result:
            all_ok = False

    print()
    if all_ok:
        print("✓  All sanity checks passed.")
    else:
        print("✗  Some sanity checks FAILED — inspect output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
