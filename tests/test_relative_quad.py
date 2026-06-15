"""Tests for relative mode with the unified 12-token quad/tri layout.

Relative mode predicts coordinate residuals against an anchor v0 and then remaps
them to absolute-coordinate logits in MLP._rel_to_abs_logits.  Two things must
hold for the 12-token layout:

  1. The anchor is the *real* first vertex: positions 0-2 for a quad, positions
     3-5 for a TRI_PAD-prefixed triangle (Decoder._relative_anchor).
  2. TRI_PAD=129 sits in the remapping dead-zone [128, 254] that is forced to
     -inf, so the triangle prefix positions (0-2) must keep their raw absolute
     logits — otherwise the model could never predict TRI_PAD.

Backward compatibility: 9-token triangle-only mode (is_tri=None) is unchanged.
"""

import pytest

torch = pytest.importorskip("torch")

from src.constants import EOS_RESIDUAL, QUANT_MAX, TRI_PAD
from src.model.decoder import MLP, Decoder, DecoderCfg

VOCAB = 257
N_REL = 2 * QUANT_MAX + 1   # 255 — first residual slot index that is NOT remapped


def _quad_cfg(**kw):
    base = dict(
        d_latent=8, d_hidden=32, n_layers=1, n_heads=2, max_faces=16,
        vocab_size=VOCAB, n_face_tokens=12, use_pos_embed=False,
        use_spherical_embed=False, relative=True, use_edge_cond=False,
    )
    base.update(kw)
    return DecoderCfg(**base)


# ─────────────────────────── _relative_anchor ────────────────────────────────

def test_anchor_quad_uses_positions_0_2():
    dec = Decoder(_quad_cfg())
    # quad row: positions 0-2 are the real v0
    quad = torch.tensor([[[5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18]]])
    anchor = dec._relative_anchor(quad)
    assert anchor.shape == (1, 1, 3)
    assert anchor[0, 0].tolist() == [5, 6, 7]


def test_anchor_triangle_skips_tri_pad_prefix():
    dec = Decoder(_quad_cfg())
    # triangle row: positions 0-2 = TRI_PAD, real v0 at positions 3-5
    tri = torch.tensor([[[TRI_PAD, TRI_PAD, TRI_PAD, 20, 21, 22, 30, 31, 32, 40, 41, 42]]])
    anchor = dec._relative_anchor(tri)
    assert anchor[0, 0].tolist() == [20, 21, 22]   # not [129,129,129]


def test_anchor_9token_unchanged():
    dec = Decoder(_quad_cfg(n_face_tokens=9))
    tri9 = torch.tensor([[[3, 4, 5, 6, 7, 8, 9, 10, 11]]])
    anchor = dec._relative_anchor(tri9)
    assert anchor[0, 0].tolist() == [3, 4, 5]


# ─────────────────────── _rel_to_abs_logits position-aware ────────────────────

def test_tri_prefix_keeps_raw_logits_so_tri_pad_is_predictable():
    """For a triangle row, prefix positions 0-2 must keep raw logits (TRI_PAD
    finite); coordinate positions 3-11 are remapped (TRI_PAD dead-zoned)."""
    mlp = MLP(d_hidden=32, vocab_size=VOCAB, use_edge_cond=False,
              relative=True, n_face_tokens=12)
    rel = torch.randn(2, 12, VOCAB)
    anchor = torch.tensor([[10, 20, 30], [40, 50, 60]])
    is_tri = torch.tensor([True, False])

    out = mlp._rel_to_abs_logits(rel, anchor, is_tri=is_tri)

    # Triangle row 0: prefix positions are the raw logits verbatim.
    assert torch.equal(out[0, 0:3, :], rel[0, 0:3, :])
    assert torch.isfinite(out[0, 0:3, TRI_PAD]).all()          # TRI_PAD predictable
    # Triangle row 0: coordinate positions 3-11 are remapped → TRI_PAD dead-zoned.
    assert torch.isinf(out[0, 3:12, TRI_PAD]).all()


def test_quad_row_has_no_tri_pad_passthrough():
    """A quad row is remapped at every position → TRI_PAD is dead-zoned everywhere."""
    mlp = MLP(d_hidden=32, vocab_size=VOCAB, use_edge_cond=False,
              relative=True, n_face_tokens=12)
    rel = torch.randn(2, 12, VOCAB)
    anchor = torch.tensor([[10, 20, 30], [40, 50, 60]])
    is_tri = torch.tensor([True, False])

    out = mlp._rel_to_abs_logits(rel, anchor, is_tri=is_tri)

    assert torch.isinf(out[1, :, TRI_PAD]).all()               # quad: no TRI_PAD anywhere
    # EOS_RESIDUAL is passthrough in both rows (predictable).
    assert torch.isfinite(out[:, :, EOS_RESIDUAL]).all()


def test_backward_compat_is_tri_none_dead_zones_prefix():
    """Without a mask (9-token tri-only / explicit None) behaviour is the legacy
    one: TRI_PAD is dead-zoned at every position, including 0-2."""
    mlp = MLP(d_hidden=32, vocab_size=VOCAB, use_edge_cond=False,
              relative=True, n_face_tokens=12)
    rel = torch.randn(1, 12, VOCAB)
    anchor = torch.tensor([[10, 20, 30]])

    out = mlp._rel_to_abs_logits(rel, anchor, is_tri=None)
    assert torch.isinf(out[0, :, TRI_PAD]).all()
    # absolute coord slots [0,127] are populated, dead-zone [128,254] is -inf.
    assert torch.isinf(out[0, :, QUANT_MAX + 1:N_REL]).all()


def test_remap_anchored_at_v0_predicts_v0_at_residual_zero():
    """Sanity: with rel logit peaked at residual 0 (index QUANT_MAX), the absolute
    argmax equals the anchor coordinate."""
    mlp = MLP(d_hidden=32, vocab_size=VOCAB, use_edge_cond=False,
              relative=True, n_face_tokens=9)
    rel = torch.full((1, 9, VOCAB), -10.0)
    rel[:, :, QUANT_MAX] = 10.0                      # residual 0 → absolute == anchor
    anchor = torch.tensor([[5, 17, 99]])
    out = mlp._rel_to_abs_logits(rel, anchor)        # is_tri None (legacy)
    abs_pred = out[0].argmax(-1)                      # (9,)
    assert abs_pred.tolist() == [5, 17, 99, 5, 17, 99, 5, 17, 99]


# ─────────────────────────── Decoder.forward integration ─────────────────────

def test_forward_quad_relative_shapes_and_finiteness():
    dec = Decoder(_quad_cfg())
    B, N = 2, 3
    C = torch.randn(B, 4, 8)
    # row 0 triangle, rows 1-2 quad
    faces = torch.randint(0, QUANT_MAX + 1, (B, N, 12))
    faces[:, 0, 0:3] = TRI_PAD
    logits = dec(C, faces)
    assert logits.shape == (B, N, 12, VOCAB)
    # Triangle prefix (pos 0-2) must allow TRI_PAD; quad rows must not.
    assert torch.isfinite(logits[:, 0, 0:3, TRI_PAD]).all()
    assert torch.isinf(logits[:, 1:, :, TRI_PAD]).all()


def test_forward_tri9_relative_unchanged():
    """9-token relative path stays the legacy one: TRI_PAD dead-zoned everywhere."""
    dec = Decoder(_quad_cfg(n_face_tokens=9, vocab_size=256))
    B, N = 1, 4
    C = torch.randn(B, 4, 8)
    faces = torch.randint(0, QUANT_MAX + 1, (B, N, 9))
    logits = dec(C, faces)
    assert logits.shape == (B, N, 9, 256)
    assert torch.isinf(logits[..., TRI_PAD]).all()


def test_vocab_too_small_raises():
    with pytest.raises(AssertionError):
        Decoder(_quad_cfg(vocab_size=200))   # <= EOS_RESIDUAL (255)
