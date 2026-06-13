"""Tests for src/dataset/obj_parser.py.

Each test writes a minimal OBJ fixture to a tmp_path directory,
then verifies the shape and content of the parsed result.

Fixtures covered:
    1. Negative vertex indices
    2. Full-line and trailing inline comments
    3. v/vt/vn slash syntax (triangle)
    4. Non-standard v/vt/vn/extra (4-slash) syntax on a quad
    5. Backslash line-continuation (\\)
    6. Mixed tri + quad in the same file
    7. N-gon fan triangulation + WARNING log
    8. Empty-texcoord slash syntax (v//vn)
"""
import logging
import textwrap

import numpy as np
import pytest

from src.dataset.obj_parser import ObjParseResult, parse_obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, content: str, name: str = "test.obj") -> str:
    """Dedent *content* and write it to *tmp_path/name*. Returns the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. Negative indices
# ---------------------------------------------------------------------------

def test_negative_indices_quad(tmp_path):
    """f -4 -3 -2 -1 should resolve to vertices 0 1 2 3 (a quad)."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        f -4 -3 -2 -1
    """)
    r = parse_obj(path)
    assert r.vertices.shape == (4, 3)
    assert r.faces_quad.shape == (1, 4)
    assert list(r.faces_quad[0]) == [0, 1, 2, 3]
    assert r.faces_tri.shape == (0, 3)
    assert r.n_ngon == 0


def test_negative_indices_tri(tmp_path):
    """f -3 -2 -1 should resolve to the last three vertices (a triangle)."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        v 99 99 99
        f -3 -2 -1
    """)
    r = parse_obj(path)
    # -3 → index 4-3=1, -2 → 4-2=2, -1 → 4-1=3 (v1, v2, v3 in 0-based)
    assert r.faces_tri.shape == (1, 3)
    assert list(r.faces_tri[0]) == [1, 2, 3]
    assert r.faces_quad.shape == (0, 4)


# ---------------------------------------------------------------------------
# 2. Comments (full-line and trailing inline)
# ---------------------------------------------------------------------------

def test_full_line_comment_ignored(tmp_path):
    """Lines starting with # must be completely ignored."""
    path = _write(tmp_path, """
        # this is a comment — vertex below
        v 0 0 0
        v 1 0 0
        v 0 1 0
        # f 1 2 3   <-- this commented face must NOT be loaded
        f 1 2 3
    """)
    r = parse_obj(path)
    assert r.faces_tri.shape == (1, 3)
    assert r.faces_quad.shape == (0, 4)


def test_inline_comment_stripped(tmp_path):
    """A trailing # comment on a data line must be stripped before parsing."""
    path = _write(tmp_path, """
        v 0 0 0  # vertex A
        v 1 0 0  # vertex B
        v 0 1 0  # vertex C
        v 0 0 1  # vertex D
        f 1 2 3 4  # this is a quad
    """)
    r = parse_obj(path)
    assert r.faces_quad.shape == (1, 4)
    assert list(r.faces_quad[0]) == [0, 1, 2, 3]
    assert r.vertices.shape == (4, 3)


# ---------------------------------------------------------------------------
# 3. v/vt/vn slash syntax — triangle
# ---------------------------------------------------------------------------

def test_slash_syntax_triangle(tmp_path):
    """f 1/1/1 2/2/2 3/3/3 — only the vertex index (first field) should be used."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        vt 0.0 0.0
        vt 1.0 0.0
        vt 0.0 1.0
        vn 0 0 1
        f 1/1/1 2/2/1 3/3/1
    """)
    r = parse_obj(path)
    assert r.faces_tri.shape == (1, 3)
    assert list(r.faces_tri[0]) == [0, 1, 2]
    assert r.faces_quad.shape == (0, 4)


def test_slash_no_texcoord(tmp_path):
    """f 1//1 2//1 3//1 — empty texcoord field (v//vn) must be handled."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        vn 0 0 1
        f 1//1 2//1 3//1
    """)
    r = parse_obj(path)
    assert r.faces_tri.shape == (1, 3)
    assert list(r.faces_tri[0]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 4. Non-standard v/vt/vn/extra (4-field) slash syntax — quad
# ---------------------------------------------------------------------------

def test_nonstandard_4field_slash_quad(tmp_path):
    """f 1/1/1/1 2/2/2/2 3/3/3/3 4/4/4/4 — extra 4th slash field is silently ignored."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        f 1/1/1/1 2/2/2/2 3/3/3/3 4/4/4/4
    """)
    r = parse_obj(path)
    assert r.faces_quad.shape == (1, 4)
    assert list(r.faces_quad[0]) == [0, 1, 2, 3]
    assert r.faces_tri.shape == (0, 3)


# ---------------------------------------------------------------------------
# 5. Backslash line-continuation
# ---------------------------------------------------------------------------

def test_continuation_line_quad(tmp_path):
    """A trailing \\ should merge the continuation line before parsing."""
    # Write without textwrap.dedent so backslash is at exact column
    content = (
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "f 1 2 \\\n"
        "3 4\n"
    )
    p = tmp_path / "cont.obj"
    p.write_text(content, encoding="utf-8")
    r = parse_obj(str(p))
    assert r.faces_quad.shape == (1, 4)
    assert list(r.faces_quad[0]) == [0, 1, 2, 3]


def test_continuation_line_triangle(tmp_path):
    """Continuation across three lines should still produce one triangle."""
    content = (
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 \\\n"
        "2 \\\n"
        "3\n"
    )
    p = tmp_path / "cont_tri.obj"
    p.write_text(content, encoding="utf-8")
    r = parse_obj(str(p))
    assert r.faces_tri.shape == (1, 3)
    assert list(r.faces_tri[0]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 6. Mixed tri + quad in the same file
# ---------------------------------------------------------------------------

def test_mixed_tri_quad(tmp_path):
    """A file with both f a b c and f a b c d lines must route them correctly."""
    path = _write(tmp_path, """
        v 0   0 0
        v 1   0 0
        v 1   1 0
        v 0   1 0
        v 0.5 0 1
        f 1 2 3 4
        f 1 2 5
    """)
    r = parse_obj(path)
    assert r.faces_quad.shape == (1, 4), "expected 1 quad"
    assert r.faces_tri.shape  == (1, 3), "expected 1 triangle"
    assert list(r.faces_quad[0]) == [0, 1, 2, 3]
    assert list(r.faces_tri[0])  == [0, 1, 4]
    assert r.n_ngon == 0


def test_mixed_ordering_interleaved(tmp_path):
    """Interleaved tri/quad/tri must all be parsed correctly regardless of order."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        v 0.5 2 0
        f 1 2 3        # tri
        f 1 2 3 4      # quad
        f 1 3 5        # tri
    """)
    r = parse_obj(path)
    assert r.faces_tri.shape  == (2, 3)
    assert r.faces_quad.shape == (1, 4)


# ---------------------------------------------------------------------------
# 7. N-gon fan triangulation + WARNING log
# ---------------------------------------------------------------------------

def test_ngon_pentagon_fan(tmp_path):
    """A pentagon (5 verts) must produce 3 triangles and n_ngon=1."""
    path = _write(tmp_path, """
        v  1  0 0
        v  0  1 0
        v -1  0 0
        v -1 -1 0
        v  1 -1 0
        f 1 2 3 4 5
    """)
    r = parse_obj(path)
    assert r.n_ngon == 1
    assert r.faces_tri.shape == (3, 3), "pentagon → 3 fan triangles"
    assert r.faces_quad.shape == (0, 4)
    # Fan: [0,1,2], [0,2,3], [0,3,4]
    expected = [[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    assert r.faces_tri.tolist() == expected


def test_ngon_emits_warning(tmp_path, caplog):
    """An n-gon must trigger a WARNING-level log entry mentioning 'n-gon'."""
    path = _write(tmp_path, """
        v 1  0 0
        v 0  1 0
        v -1 0 0
        v -1 -1 0
        v 1 -1 0
        f 1 2 3 4 5
    """)
    with caplog.at_level(logging.WARNING, logger="src.dataset.obj_parser"):
        parse_obj(path)

    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_msgs, "expected at least one WARNING"
    assert any("n-gon" in m.lower() for m in warning_msgs)


def test_no_ngon_no_warning(tmp_path, caplog):
    """A pure tri+quad file must not emit any WARNING."""
    path = _write(tmp_path, """
        v 0 0 0
        v 1 0 0
        v 1 1 0
        v 0 1 0
        f 1 2 3
        f 1 2 3 4
    """)
    with caplog.at_level(logging.WARNING, logger="src.dataset.obj_parser"):
        parse_obj(path)

    warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_msgs, f"unexpected warnings: {[r.message for r in warning_msgs]}"


# ---------------------------------------------------------------------------
# 8. Vertex geometry round-trip
# ---------------------------------------------------------------------------

def test_vertex_coordinates_preserved(tmp_path):
    """Vertex XYZ values must survive the parse exactly."""
    path = _write(tmp_path, """
        v 1.5  2.75 -3.0
        v 0.0  0.0   0.0
        v 10.0 5.0   0.5
        f 1 2 3
    """)
    r = parse_obj(path)
    expected = np.array([[1.5, 2.75, -3.0], [0.0, 0.0, 0.0], [10.0, 5.0, 0.5]])
    np.testing.assert_array_almost_equal(r.vertices, expected)


def test_empty_file(tmp_path):
    """An OBJ file with no geometry must return empty arrays without error."""
    path = _write(tmp_path, """
        # just comments
        # no vertices or faces
        mtllib dummy.mtl
    """)
    r = parse_obj(path)
    assert r.vertices.shape  == (0, 3)
    assert r.faces_tri.shape  == (0, 3)
    assert r.faces_quad.shape == (0, 4)
    assert r.n_ngon == 0
