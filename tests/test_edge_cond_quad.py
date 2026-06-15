"""Tests for edge-conditioned target building & decoding with the 12-token layout.

Option A (hierarchical EOS): for each query edge the model predicts up to two new
vertices.  Target slot layout (positions): 0-5 = PAD (edge), 6-8 = v1, 9-11 = v2.
    no neighbor    → 6-8 = EOS_RESIDUAL, 9-11 = PAD          ("stop this edge")
    triangle nbr   → 6-8 = unique vertex, 9-11 = EOS_RESIDUAL ("no v2")
    quad neighbor  → 6-8 = v1, 9-11 = v2

Vertex ordering: v1 is cyclically adjacent to ev1, v2 to ev0, so [ev0,ev1,v1,v2]
is a valid 4-cycle for any lex-sort direction of the shared edge.

The 9-token triangle-only path must be unchanged.
"""

import pytest

torch = pytest.importorskip("torch")

from src.constants import EOS_RESIDUAL, PAD_TARGET, QUANT_MAX, TRI_NEIGHBOR, TRI_PAD
from src.model.decoder import MLP, Decoder, DecoderCfg
from src.training.loss import edge_face_type_acc, edge_face_type_recall
from src.training.strategy.target_builder.adjacent import (
    AdjacentTargetBuilder,
    AdjacentTargetBuilderCfg,
)

# quad Q0 = [A,B,C,D] cyclic; Q1 = [B,C,E,F] shares edge (B,C)
A = [0, 0, 0]; B = [0, 0, 10]; C = [0, 10, 10]; D = [0, 10, 0]
E = [5, 10, 10]; F = [5, 0, 10]


def _builder():
    return AdjacentTargetBuilder(AdjacentTargetBuilderCfg())


def _batch(faces, neighbors):
    return {
        "faces": torch.tensor(faces, dtype=torch.long).unsqueeze(0),
        "face_neighbors": torch.tensor(neighbors, dtype=torch.long).unsqueeze(0),
    }


def _mask(n, edges):
    m = torch.zeros(1, n, n, dtype=torch.bool)
    for q, k in edges:
        m[0, q, k] = True
    return m


# ─────────────────────────── target builder: 12-token ────────────────────────

def test_quad_neighbor_target_and_winding():
    faces = [A + B + C + D, B + C + E + F]
    neighbors = [[-1, 1, -1, -1], [0, -1, -1, -1]]
    tgt, qe = _builder().compute_targets(_batch(faces, neighbors), _mask(2, [(0, 1)]))

    assert tgt.shape == (1, 2, 12)
    assert tgt[0, 0, 0:6].tolist() == [PAD_TARGET] * 6           # edge positions
    # edge (B,C) lex-sorted → ev0=B, ev1=C; v1 adj C = E, v2 adj B = F
    assert qe[0, 0].tolist() == B + C
    assert tgt[0, 0, 6:9].tolist()  == E                          # v1 (adjacent to ev1)
    assert tgt[0, 0, 9:12].tolist() == F                          # v2 (adjacent to ev0)


def test_triangle_neighbor_target():
    faces = [A + B + C + D, B + C + E + [TRI_PAD] * 3]            # neighbor is a triangle (pad at end)
    neighbors = [[-1, 1, -1, -1], [0, -1, -1, -1]]
    tgt, _ = _builder().compute_targets(_batch(faces, neighbors), _mask(2, [(0, 1)]))

    assert tgt[0, 0, 6:9].tolist()  == E                          # single unique vertex
    # slot-2 "triangle neighbor" marker — distinct from the slot-1 STOP (EOS_RESIDUAL).
    assert tgt[0, 0, 9:12].tolist() == [TRI_NEIGHBOR] * 3
    assert (tgt[0, 0, 9:12] != EOS_RESIDUAL).all()               # never the STOP token


def test_no_neighbor_is_stop_eos():
    faces = [A + B + C + D]
    neighbors = [[-1, -1, -1, -1]]
    tgt, _ = _builder().compute_targets(_batch(faces, neighbors), _mask(1, []))

    assert tgt[0, 0, 6:9].tolist()  == [EOS_RESIDUAL] * 3         # stop
    assert tgt[0, 0, 9:12].tolist() == [PAD_TARGET] * 3           # slot2 ignored


def test_reversed_edge_still_valid_quad():
    """Edge (D,A) lex-sorts to (A,D) (reversed vs winding); reconstruction must
    still cover the quad's four undirected edges."""
    faces = [A + B + C + D, D + A + E + F]                        # Q1 shares edge (D,A)
    neighbors = [[-1, -1, -1, 1], [0, -1, -1, -1]]               # Q0 edge3=(D,A)
    tgt, qe = _builder().compute_targets(_batch(faces, neighbors), _mask(2, [(0, 1)]))

    ev = qe[0, 0].tolist()
    v1 = tgt[0, 0, 6:9].tolist()
    v2 = tgt[0, 0, 9:12].tolist()
    recon = [tuple(ev[0:3]), tuple(ev[3:6]), tuple(v1), tuple(v2)]
    assert len(set(recon)) == 4                                   # valid 4-cycle
    got = {frozenset((recon[i], recon[(i + 1) % 4])) for i in range(4)}
    want = {frozenset((tuple(x), tuple(y))) for x, y in
            [(D, A), (A, E), (E, F), (F, D)]}
    assert got == want


# ─────────────────────────── target builder: 9-token unchanged ───────────────

def test_9token_path_unchanged():
    P = [0, 0, 0]; Q = [0, 0, 10]; R = [0, 10, 0]; S = [0, 10, 10]
    faces = [P + Q + R, Q + R + S]                               # share edge (Q,R)
    neighbors = [[-1, 1, -1], [0, -1, -1]]
    tgt, qe = _builder().compute_targets(_batch(faces, neighbors), _mask(2, [(0, 1)]))

    assert tgt.shape == (1, 2, 9)
    assert qe[0, 0].tolist() == Q + R                            # lex-sorted edge
    assert tgt[0, 0, 0:6].tolist() == [PAD_TARGET] * 6
    assert tgt[0, 0, 6:9].tolist() == S                         # the unique vertex
    # Retro-compat: TRI_NEIGHBOR is a 12-token-only sentinel; it must never appear
    # anywhere in the 9-token triangle path.
    assert (tgt != TRI_NEIGHBOR).all()


# ─────────────────────────── MLP / Decoder shapes ────────────────────────────

def test_mlp_nout_scales_with_layout():
    m9  = MLP(32, 257, use_edge_cond=True, relative=False, n_face_tokens=9)
    m12 = MLP(32, 257, use_edge_cond=True, relative=False, n_face_tokens=12)
    assert m9.net[-1].out_features  == 3 * 257                   # 1 new vertex
    assert m12.net[-1].out_features == 6 * 257                   # up to 2 new vertices


def _cfg(**kw):
    base = dict(d_latent=8, d_hidden=32, n_layers=1, n_heads=2, max_faces=16,
                vocab_size=257, n_face_tokens=12, use_pos_embed=False,
                use_spherical_embed=False, relative=False, use_edge_cond=True)
    base.update(kw)
    return DecoderCfg(**base)


@pytest.mark.parametrize("relative", [False, True])
def test_decoder_forward_edge_cond_12(relative):
    dec = Decoder(_cfg(relative=relative))
    B, N = 1, 2
    Cl = torch.randn(B, 4, 8)
    faces = torch.randint(0, QUANT_MAX + 1, (B, N, 12))
    qe = torch.randint(0, QUANT_MAX + 1, (B, N, 6))
    logits = dec(Cl, faces, query_edges=qe)
    assert logits.shape == (B, N, 12, 257)
    # positions 0-5 are zero-padded (shared edge is not predicted)
    assert torch.equal(logits[:, :, 0:6, :], torch.zeros_like(logits[:, :, 0:6, :]))


def test_edge_cond_guard_rejects_bad_token_count():
    with pytest.raises(AssertionError):
        Decoder(_cfg(n_face_tokens=10))


# ─────────────────────────── monitoring metric ───────────────────────────────

def test_edge_face_type_acc_perfect_classification():
    vocab = 257
    targets = torch.full((1, 2, 12), PAD_TARGET, dtype=torch.long)
    # face0 = quad neighbor (slot2 real coords); face1 = triangle neighbor (slot2 TRI_NEIGHBOR)
    targets[0, 0, 6:9]  = torch.tensor([5, 6, 7])
    targets[0, 0, 9:12] = torch.tensor([8, 9, 10])
    targets[0, 1, 6:9]  = torch.tensor([11, 12, 13])
    targets[0, 1, 9:12] = TRI_NEIGHBOR

    logits = torch.full((1, 2, 12, vocab), -10.0)
    logits[0, 0, 9, 5] = 10.0                  # quad: pos9 argmax = coord (not TRI_NEIGHBOR)
    logits[0, 1, 9, TRI_NEIGHBOR] = 10.0       # tri:  pos9 argmax = TRI_NEIGHBOR
    assert torch.isclose(edge_face_type_acc(logits, targets), torch.tensor(1.0))

    # flip the quad prediction to TRI_NEIGHBOR → quad collapses to triangle → 0.5
    logits[0, 0, 9, 5] = -10.0
    logits[0, 0, 9, TRI_NEIGHBOR] = 10.0
    assert torch.isclose(edge_face_type_acc(logits, targets), torch.tensor(0.5))

    # the recall split must localise it: quad_recall→0, tri_recall→1
    rec = edge_face_type_recall(logits, targets)
    assert torch.isclose(rec["quad_recall"], torch.tensor(0.0))
    assert torch.isclose(rec["tri_recall"],  torch.tensor(1.0))
    assert torch.isclose(rec["perc_tri_nb"], torch.tensor(0.5))


def test_edge_face_type_acc_noop_on_9token():
    logits = torch.randn(1, 2, 9, 256)
    targets = torch.zeros(1, 2, 9, dtype=torch.long)
    assert torch.isclose(edge_face_type_acc(logits, targets), torch.tensor(1.0))
