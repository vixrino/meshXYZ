"""Wavefront OBJ parser that preserves triangle and quad faces separately.

Unlike trimesh (which always triangulates on load), this parser routes each face
by its per-face vertex count:

    3 vertices  → triangle     stored in ObjParseResult.faces_tri  (T, 3)
    4 vertices  → quad         stored in ObjParseResult.faces_quad (Q, 4)
    5+ vertices → n-gon        fan-triangulated into triangles, counted in .n_ngon

Supported OBJ features
-----------------------
* Vertex positions:    ``v x y z [w]``  (w ignored)
* Face vertices:       ``f v``, ``f v/vt``, ``f v/vt/vn``, ``f v//vn``
                       Non-standard extra fields (``v/vt/vn/extra``) are silently ignored.
* Negative indices:    ``f -1 -2 -3`` (relative to the current vertex list end, per spec §1.2.1)
* Inline comments:     ``#`` anywhere on a line starts a comment; everything after is ignored.
* Line continuation:   a trailing ``\\`` merges the next line before parsing.

Directives not listed above (``vt``, ``vn``, ``o``, ``g``, ``usemtl``, etc.) are silently
skipped; they carry no geometry needed for mesh tokenization.

Usage
-----
::

    from src.dataset.obj_parser import parse_obj

    result = parse_obj("path/to/mesh.obj")
    print(result.vertices.shape)    # (V, 3)  float64, original XYZ order
    print(result.faces_tri.shape)   # (T, 3)  int64,   0-based vertex indices
    print(result.faces_quad.shape)  # (Q, 4)  int64,   0-based vertex indices
    print(result.n_ngon)            # int, number of n-gon input faces
"""

import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ObjParseResult:
    vertices: np.ndarray    # (V, 3) float64 — XYZ order, as written in the file
    faces_tri: np.ndarray   # (T, 3) int64   — 0-based vertex indices (includes fan-expanded n-gons)
    faces_quad: np.ndarray  # (Q, 4) int64   — 0-based vertex indices
    n_ngon: int             # number of n-gon input faces (≥5 verts) before fan expansion


def _vertex_index(token: str, n_verts: int) -> int:
    """Parse one face-vertex token and return a 0-based vertex index.

    Accepted forms (OBJ spec + common non-standard extensions):
        ``1``         plain vertex index (1-based)
        ``1/2``       vertex / texcoord
        ``1/2/3``     vertex / texcoord / normal
        ``1//3``      vertex / (no texcoord) / normal
        ``-1``        negative: counts backward from current vertex list end
        ``1/2/3/4``   non-standard extra field — silently ignored beyond the third slash

    Raises ``ValueError`` if the vertex part is not an integer.
    """
    raw = token.split("/")[0]   # take only the vertex-index part
    idx = int(raw)
    if idx < 0:
        idx = n_verts + idx     # e.g. -1 → n_verts - 1 (last vertex)
    else:
        idx -= 1                # OBJ is 1-based → convert to 0-based
    return idx


def parse_obj(path: str) -> ObjParseResult:
    """Parse a Wavefront OBJ file, preserving quad and triangle faces separately.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.obj`` file.

    Returns
    -------
    ObjParseResult
        ``vertices``   — (V, 3) float64 array of vertex positions (XYZ order).
        ``faces_tri``  — (T, 3) int64 array of triangle faces (0-based indices).
                         Includes triangles originating from n-gon fan expansion.
        ``faces_quad`` — (Q, 4) int64 array of quad faces (0-based indices).
        ``n_ngon``     — count of n-gon input faces (≥5 vertices) encountered.

    Logging
    -------
    Always logs one INFO line with per-file face-type counts.
    Logs a WARNING if any n-gons are present, reporting the count and percentage
    so callers can decide on the fan-triangulation policy for their dataset.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # Merge backslash line-continuation *before* splitting into lines.
    # "foo \\\nbar" becomes "foo  bar" (one space from the newline removal).
    text = text.replace("\\\n", " ")

    vertices:   list[list[float]] = []
    faces_tri:  list[list[int]]   = []
    faces_quad: list[list[int]]   = []

    # Track original input face counts separately from the fan-expanded output.
    n_tri_input  = 0
    n_quad_input = 0
    n_ngon       = 0
    n_tri_from_ngon = 0   # extra triangles created by fan expansion

    for raw_line in text.splitlines():
        # Strip inline comment: everything from the first '#' onward is ignored.
        comment_pos = raw_line.find("#")
        if comment_pos >= 0:
            raw_line = raw_line[:comment_pos]
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        directive = parts[0]

        if directive == "v":
            # v x y z  [w]   — w (homogeneous weight) is optional and ignored
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

        elif directive == "f":
            n = len(vertices)   # current count for negative-index resolution
            idxs = [_vertex_index(p, n) for p in parts[1:]]
            k = len(idxs)

            if k == 3:
                n_tri_input += 1
                faces_tri.append(idxs)
            elif k == 4:
                n_quad_input += 1
                faces_quad.append(idxs)
            elif k >= 5:
                # Fan triangulation: anchor at idxs[0], walk pairs (i, i+1).
                n_ngon += 1
                fan_count = k - 2
                n_tri_from_ngon += fan_count
                for i in range(1, k - 1):
                    faces_tri.append([idxs[0], idxs[i], idxs[i + 1]])
        # All other directives (vt, vn, o, g, s, mtllib, usemtl, …) are skipped.

    fname = os.path.basename(path)
    n_tri_output = n_tri_input + n_tri_from_ngon

    log.info(
        "%s: %d verts | %d tri faces | %d quad faces | %d n-gon(s) "
        "→ %d extra fan-tris",
        fname,
        len(vertices),
        n_tri_output,
        n_quad_input,
        n_ngon,
        n_tri_from_ngon,
    )

    if n_ngon > 0:
        total_input_faces = n_tri_input + n_quad_input + n_ngon
        pct = 100.0 * n_ngon / max(1, total_input_faces)
        log.warning(
            "%s: %d n-gon face(s) (≥5 verts) found — %.1f%% of input faces. "
            "Each was fan-triangulated. Review dataset if prevalence is high; "
            "QuadGPT tokenization only models tris and quads natively.",
            fname,
            n_ngon,
            pct,
        )

    V = (np.array(vertices, dtype=np.float64)
         if vertices else np.empty((0, 3), dtype=np.float64))
    T = (np.array(faces_tri, dtype=np.int64)
         if faces_tri else np.empty((0, 3), dtype=np.int64))
    Q = (np.array(faces_quad, dtype=np.int64)
         if faces_quad else np.empty((0, 4), dtype=np.int64))

    return ObjParseResult(vertices=V, faces_tri=T, faces_quad=Q, n_ngon=n_ngon)
