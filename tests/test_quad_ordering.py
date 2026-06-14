"""Quad-path coverage for the ordering layer.

Tests the three fixes that had no existing coverage:
  1. _apply_perm with T=12 faces and S=4 neighbor slots
  2. CanonicalOrdering with TRI_PAD-aware key (12-token mode)
  3. CausalAxisOrdering with T=12 (would crash before the fix)

All tests require torch and are skipped automatically in environments
without it (e.g. local numpy-only CI).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip whole module if torch missing


from src.training.strategy.ordering import _apply_perm
from src.training.strategy.ordering.canonical import CanonicalOrdering
from src.training.strategy.ordering.causal_axis import CausalAxisOrdering
from src.constants import TRI_PAD, QUANT_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tri_face_12(x: int, y: int, z: int) -> list[int]:
    """12-token triangle face with first real vertex at (x,y,z)."""
    return [TRI_PAD, TRI_PAD, TRI_PAD, x, y, z, x + 1, y, z, x, y + 1, z]


def _quad_face_12(x: int, y: int, z: int) -> list[int]:
    """12-token quad face with first vertex at (x,y,z)."""
    return [x, y, z, x + 1, y, z, x + 1, y + 1, z, x, y + 1, z]


# ---------------------------------------------------------------------------
# 1. _apply_perm — T=12 faces, S=4 neighbor slots
# ---------------------------------------------------------------------------

class TestApplyPermQuad:
    """_apply_perm must reorder all 12 face tokens and all 4 neighbor slots."""

    def _make_batch(self):
        """Batch of 1 sample, 3 faces, 12 tokens each, 4 neighbor slots."""
        B, N, T, S = 1, 3, 12, 4
        # Face tokens: face i has all tokens = i (easy to verify after permutation)
        faces = torch.stack([
            torch.full((T,), i, dtype=torch.long) for i in range(N)
        ]).unsqueeze(0)  # (1, 3, 12)

        # Neighbors: face i's 4 neighbor slots = [i-1, i+1, -1, -1] (clipped)
        def nbrs(i):
            return [i - 1 if i > 0 else -1,
                    i + 1 if i < N - 1 else -1,
                    -1, -1]
        neighbors = torch.tensor([[nbrs(i) for i in range(N)]],
                                 dtype=torch.long)  # (1, 3, 4)

        # Permutation: [2, 0, 1] (rotate)
        perm = torch.tensor([[2, 0, 1]], dtype=torch.long)
        lengths = torch.tensor([N])
        return faces, neighbors, perm, lengths

    def test_face_tokens_reordered(self):
        faces, neighbors, perm, _ = self._make_batch()
        new_faces, _ = _apply_perm(faces, neighbors, perm)
        # perm [2,0,1]: new[0]=old[2], new[1]=old[0], new[2]=old[1]
        assert new_faces[0, 0].unique().item() == 2, "new face 0 should be old face 2"
        assert new_faces[0, 1].unique().item() == 0, "new face 1 should be old face 0"
        assert new_faces[0, 2].unique().item() == 1, "new face 2 should be old face 1"

    def test_all_12_token_slots_preserved(self):
        faces, neighbors, perm, _ = self._make_batch()
        new_faces, _ = _apply_perm(faces, neighbors, perm)
        assert new_faces.shape == (1, 3, 12), f"Expected (1,3,12), got {new_faces.shape}"

    def test_neighbor_4th_slot_preserved(self):
        """The 4th neighbor slot (index 3) must survive remapping — was silently
        truncated before the fix (Colab had expand(B,N,3) instead of expand(B,N,S))."""
        faces, neighbors, perm, _ = self._make_batch()
        _, new_nbrs = _apply_perm(faces, neighbors, perm)
        assert new_nbrs.shape == (1, 3, 4), f"Expected (1,3,4), got {new_nbrs.shape}"
        # 4th slot must stay -1 (was -1 in the original; truncation would corrupt it)
        assert (new_nbrs[0, :, 3] == -1).all(), "4th neighbor slot corrupted"

    def test_neighbor_indices_remapped(self):
        """After permutation, neighbor indices must point to new (not old) positions."""
        faces, neighbors, perm, _ = self._make_batch()
        _, new_nbrs = _apply_perm(faces, neighbors, perm)
        # Original face 0 had neighbor 1 in slot 1.
        # Under perm [2,0,1]: face 0 moves to new position 1.
        # new face 1 = old face 0; its slot-1 neighbor was old face 1, now at new position 2.
        new_face_idx_of_old_0 = 1   # new position of old face 0
        slot0_of_new_face1 = int(new_nbrs[0, new_face_idx_of_old_0, 0].item())
        slot1_of_new_face1 = int(new_nbrs[0, new_face_idx_of_old_0, 1].item())
        # old face 0 has no left neighbor (slot 0 = -1) and right neighbor = old 1 (new pos 2)
        assert slot0_of_new_face1 == -1, "left neighbor of new-face-1 should be -1"
        assert slot1_of_new_face1 == 2,  "right neighbor of new-face-1 should be new pos 2"

    def test_tri_mode_unchanged(self):
        """_apply_perm must still work with T=9, S=3 (triangle-only mode)."""
        B, N, T, S = 1, 3, 9, 3
        faces     = torch.arange(N * T).reshape(B, N, T)
        neighbors = torch.full((B, N, S), -1, dtype=torch.long)
        perm      = torch.tensor([[1, 2, 0]], dtype=torch.long)
        new_faces, new_nbrs = _apply_perm(faces, neighbors, perm)
        assert new_faces.shape  == (B, N, T)
        assert new_nbrs.shape   == (B, N, S)


# ---------------------------------------------------------------------------
# 2. CanonicalOrdering — TRI_PAD-aware key in 12-token mode
# ---------------------------------------------------------------------------

class TestCanonicalOrderingQuad:
    """CanonicalOrdering must use the first *real* vertex as the sort key."""

    def test_tri_face_sorts_by_real_vertex_not_tri_pad(self):
        """A triangle face with real vertex near origin must sort before a quad
        face with first vertex at (10,10,10).  Before the fix, TRI_PAD (=129)
        at positions 0-2 gave the triangle a huge key, placing it after the quad."""
        B, N = 1, 2
        # Face 0: triangle, real first vertex = (0, 0, 0) → key = 0
        # Face 1: quad, first vertex = (10, 10, 10) → key = 10*128²+10*128+10 = 165898
        f0 = torch.tensor(_tri_face_12(0, 0, 0), dtype=torch.long)
        f1 = torch.tensor(_quad_face_12(10, 10, 10), dtype=torch.long)
        faces     = torch.stack([f0, f1]).unsqueeze(0)          # (1, 2, 12)
        neighbors = torch.full((B, N, 4), -1, dtype=torch.long)
        lengths   = torch.tensor([N])

        ordering = CanonicalOrdering()
        perm = ordering.permute(faces, neighbors, lengths)

        # Face 0 (key=0) must sort first → perm[0,0] = 0
        assert perm[0, 0].item() == 0, (
            f"Triangle with key=0 should sort first, but perm={perm.tolist()}. "
            "Likely TRI_PAD (129) is being used as key instead of the real vertex."
        )

    def test_quad_sorts_before_high_key_tri(self):
        """A quad with a small first-vertex key must sort before a triangle whose
        real vertex is at a higher coordinate."""
        B, N = 1, 2
        # Face 0: quad, first vertex = (1, 0, 0) → key = 1
        # Face 1: triangle, real first vertex = (5, 0, 0) → key = 5
        f0 = torch.tensor(_quad_face_12(1, 0, 0), dtype=torch.long)
        f1 = torch.tensor(_tri_face_12(5, 0, 0), dtype=torch.long)
        faces     = torch.stack([f0, f1]).unsqueeze(0)
        neighbors = torch.full((B, N, 4), -1, dtype=torch.long)
        lengths   = torch.tensor([N])

        ordering = CanonicalOrdering()
        perm = ordering.permute(faces, neighbors, lengths)
        assert perm[0, 0].item() == 0, f"Quad with key=1 should sort first; got perm={perm.tolist()}"

    def test_tri_only_mode_unchanged(self):
        """9-token triangle-only mode must still produce the correct ZYX sort."""
        B, N, T = 1, 3, 9
        # Three faces, first vertices at z=10, z=5, z=1 → expected sort: [2, 1, 0]
        def tri9(z):
            return [0, 0, z, 0, 1, z, 1, 0, z]
        faces = torch.tensor([[tri9(10), tri9(5), tri9(1)]], dtype=torch.long)  # (1,3,9)
        neighbors = torch.full((B, N, 3), -1, dtype=torch.long)
        lengths   = torch.tensor([N])

        perm = CanonicalOrdering().permute(faces, neighbors, lengths)
        assert perm[0].tolist() == [2, 1, 0], f"Expected [2,1,0], got {perm[0].tolist()}"


# ---------------------------------------------------------------------------
# 3. CausalAxisOrdering — T=12 and T=9 (crash regression)
# ---------------------------------------------------------------------------

class TestCausalAxisOrdering:
    """CausalAxisOrdering must run without error for both face layouts."""

    def _quad_batch(self, B: int = 2, N: int = 5):
        """Random 12-token batch with a mix of tri and quad faces."""
        T, S = 12, 4
        faces = torch.randint(0, QUANT_MAX, (B, N, T), dtype=torch.long)
        # Make half the faces triangles (set positions 0-2 to TRI_PAD)
        faces[:, ::2, :3] = TRI_PAD
        neighbors = torch.full((B, N, S), -1, dtype=torch.long)
        lengths   = torch.tensor([N] * B)
        return faces, neighbors, lengths

    def _tri_batch(self, B: int = 2, N: int = 5):
        """Random 9-token triangle-only batch."""
        T, S = 9, 3
        faces     = torch.randint(0, QUANT_MAX, (B, N, T), dtype=torch.long)
        neighbors = torch.full((B, N, S), -1, dtype=torch.long)
        lengths   = torch.tensor([N] * B)
        return faces, neighbors, lengths

    def test_quad_mode_no_crash(self):
        """CausalAxisOrdering must not crash on (B, N, 12) faces.
        Before the fix, reshape(B, N, 4, 3) was applied unconditionally,
        crashing for T=9 batches."""
        faces, nbrs, lengths = self._quad_batch()
        ordering = CausalAxisOrdering()
        perm = ordering.permute(faces, nbrs, lengths)
        B, N, _ = faces.shape
        assert perm.shape == (B, N), f"Expected ({B},{N}), got {perm.shape}"

    def test_quad_perm_is_valid(self):
        """Permutation indices for valid faces must be a proper permutation of [0, N)."""
        faces, nbrs, lengths = self._quad_batch(B=1, N=6)
        perm = CausalAxisOrdering().permute(faces, nbrs, lengths)
        L = int(lengths[0].item())
        valid_perm = perm[0, :L].sort().values
        expected   = torch.arange(L)
        assert (valid_perm == expected).all(), f"Not a valid permutation: {perm[0].tolist()}"

    def test_tri_mode_no_crash(self):
        """CausalAxisOrdering must still work on (B, N, 9) tri-only batches.
        Before the fix, the T=9 branch was removed, causing reshape(B,N,4,3) crash."""
        faces, nbrs, lengths = self._tri_batch()
        ordering = CausalAxisOrdering()
        perm = ordering.permute(faces, nbrs, lengths)
        B, N, _ = faces.shape
        assert perm.shape == (B, N)

    def test_tri_perm_is_valid(self):
        faces, nbrs, lengths = self._tri_batch(B=1, N=4)
        perm = CausalAxisOrdering().permute(faces, nbrs, lengths)
        L = int(lengths[0].item())
        valid_perm = perm[0, :L].sort().values
        assert (valid_perm == torch.arange(L)).all()
