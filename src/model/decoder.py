from dataclasses import dataclass, field

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..constants import EOS_RESIDUAL, QUANT_MAX, TRI_NEIGHBOR, TRI_PAD
from ..utils.geometry import face_cartesian_to_spherical


@dataclass
class DecoderCfg:
    d_latent: int = 64
    d_hidden: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    max_faces: int = 2048
    vocab_size: int = 256
    n_face_tokens: int = 9    # 9 for triangle-only mode; 12 for unified quad/tri mode
    use_pos_embed: bool = True
    use_spherical_embed: bool = False
    relative: bool = True
    use_edge_cond: bool = True


class MLP(nn.Module):
    """MLP head.

    use_edge_cond=True : takes a query edge (2 verts, 6 coords) and predicts the
                         remaining vertices of the neighbor face.  The edge always
                         occupies 6 coord slots, so the head predicts
                         n_face_tokens - 6 coords:
                           n_face_tokens=9  → 3 (1 new vertex; triangle neighbor)
                           n_face_tokens=12 → 6 (up to 2 new vertices; quad neighbor,
                                                 or 1 vertex + EOS marker for a tri).
    use_edge_cond=False: predicts all n_face_tokens coordinate slots of the face.
                         Works for n_face_tokens=9 (triangle) or 12 (unified quad/tri).
    """

    def __init__(
        self,
        d_hidden: int,
        vocab_size: int,
        use_edge_cond: bool = True,
        relative: bool = True,
        n_face_tokens: int = 9,
    ):
        super().__init__()
        self.vocab_size    = vocab_size
        self.use_edge_cond = use_edge_cond
        self.relative      = relative
        self.n_face_tokens = n_face_tokens
        self.h_proj        = nn.Linear(d_hidden, d_hidden)
        if use_edge_cond:
            self.edge_encoder = nn.Sequential(
                nn.Linear(6, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_hidden),
            )
        n_out = (n_face_tokens - 6) if use_edge_cond else n_face_tokens
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_out * vocab_size),
        )

    def _rel_to_abs_logits(
        self,
        rel_logits: Float[Tensor, "batch_faces n_coords vocab"],
        anchor: Int[Tensor, "batch_faces 3"],
        is_tri: "Bool[Tensor, 'batch_faces'] | None" = None,
    ) -> Float[Tensor, "batch_faces n_coords vocab"]:
        """Convert relative-residual logits to absolute-coordinate logits.

        anchor is repeated to cover all n_coords (works for n_coords=3, 9 or 12).

        is_tri (12-token unified layout only): bool mask, True for triangle faces.
        A triangle's positions 0-2 hold TRI_PAD=129, which falls in the remapping
        dead-zone [n_abs, n_rel_coords) that is forced to -inf below — so it could
        never be predicted.  For those rows the prefix positions are restored to
        the raw (un-remapped) absolute-vocab logits, leaving TRI_PAD predictable;
        coordinate positions 3-11 are still remapped relative to the anchor.
        When is_tri is None (9-token tri-only / quad coord positions) the behaviour
        is unchanged.
        """
        BN, n_coords, _ = rel_logits.shape
        n_abs        = QUANT_MAX + 1       # absolute coordinate slots: [0, QUANT_MAX]
        n_rel_coords = 2 * QUANT_MAX + 1   # coordinate residual slots in relative vocab
        base = anchor.repeat(1, n_coords // 3).unsqueeze(-1)   # (BN, n_coords, 1)

        abs_coords  = torch.arange(n_abs, device=rel_logits.device)  # (n_abs,)
        rel_indices = abs_coords - base + QUANT_MAX                   # (BN, n_coords, n_abs)

        valid = (rel_indices >= 0) & (rel_indices < n_rel_coords)

        abs_logits = rel_logits.gather(2, rel_indices.clamp(0, n_rel_coords - 1))
        abs_logits = abs_logits.masked_fill(~valid, float('-inf'))

        out = rel_logits.new_full(rel_logits.shape, float('-inf'))
        out[..., :n_abs]        = abs_logits
        out[..., n_rel_coords:] = rel_logits[..., n_rel_coords:]

        if is_tri is not None and n_coords == 12:
            prefix = torch.zeros(n_coords, dtype=torch.bool, device=rel_logits.device)
            prefix[:3] = True   # TRI_PAD positions of a triangle face
            keep_raw = is_tri.view(BN, 1, 1) & prefix.view(1, n_coords, 1)
            out = torch.where(keep_raw, rel_logits, out)

        return out

    def forward(
        self,
        h: Float[Tensor, "batch_faces d_hidden"],
        query_edges: "Int[Tensor, 'batch_faces 6'] | None" = None,
        face_v0: "Int[Tensor, 'batch_faces 3'] | None" = None,
        is_tri: "Bool[Tensor, 'batch_faces'] | None" = None,
    ) -> Float[Tensor, "batch_faces n_face_tokens vocab"]:
        """Returns (BN, n_face_tokens, vocab).

        When use_edge_cond=True the first 6 positions are zero-padded (the shared
        edge is not predicted); the remaining n_face_tokens-6 positions carry real
        logits (3 for n_face_tokens=9, 6 for n_face_tokens=12).
        """
        if self.use_edge_cond:
            assert query_edges is not None, (
                "use_edge_cond=True requires query_edges to be provided."
            )

        fused = self.h_proj(h)
        if self.use_edge_cond:
            if self.relative:
                edge_input = (query_edges - face_v0.repeat(1, 2)).float() / QUANT_MAX
            else:
                edge_input = query_edges.float() / (QUANT_MAX + 1)
            fused = fused + self.edge_encoder(edge_input)

        n_coords = (self.n_face_tokens - 6) if self.use_edge_cond else self.n_face_tokens
        logits = self.net(fused).reshape(-1, n_coords, self.vocab_size)

        if self.relative:
            logits = self._rel_to_abs_logits(logits, face_v0, is_tri=is_tri)

        if self.use_edge_cond:
            zeros_6 = torch.zeros(
                logits.shape[0], 6, self.vocab_size,
                dtype=logits.dtype, device=logits.device,
            )
            logits = torch.cat([zeros_6, logits], dim=1)

        return logits


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderCfg):
        super().__init__()
        self.cfg = cfg

        # ── Safety guards ────────────────────────────────────────────────────
        # relative=True with n_face_tokens=12 is supported: the anchor is the real
        # first vertex (TRI_PAD-aware, see _relative_anchor) and the TRI_PAD prefix
        # of triangle faces keeps raw absolute-vocab logits (see _rel_to_abs_logits)
        # instead of being remapped into the residual dead-zone.  All it requires is
        # a vocab large enough to hold the TRI_PAD and EOS_RESIDUAL classes.
        if cfg.relative and cfg.n_face_tokens == 12:
            # Highest target class in 12-token edge-cond mode is TRI_NEIGHBOR (256),
            # the slot-2 triangle marker (distinct from the slot-1 STOP=EOS_RESIDUAL).
            # vocab_size=257 already covers it (index 256), so no output-dim change.
            assert cfg.vocab_size > TRI_NEIGHBOR, (
                "relative=True with n_face_tokens=12 needs vocab_size > TRI_NEIGHBOR "
                f"({TRI_NEIGHBOR}) so TRI_PAD/EOS_RESIDUAL/TRI_NEIGHBOR classes are "
                f"representable; got vocab_size={cfg.vocab_size}."
            )
        # use_edge_cond=True pads positions 0-5 (shared edge, 2×3 coords) and predicts
        # the remaining slots: 3 for n_face_tokens=9 (1 new vertex of a triangle
        # neighbor) or 6 for n_face_tokens=12 (up to 2 new vertices of a quad
        # neighbor; a triangle neighbor uses 1 vertex + an EOS marker in slot 2).
        assert not (cfg.use_edge_cond and cfg.n_face_tokens not in (9, 12)), (
            "use_edge_cond=True supports n_face_tokens=9 (6 edge pad + 3 predicted) "
            "or 12 (6 edge pad + 6 predicted). "
            f"Got n_face_tokens={cfg.n_face_tokens}."
        )

        self.latent_proj = nn.Linear(cfg.d_latent, cfg.d_hidden)
        # face_embed input size = n_face_tokens (9 for tri, 12 for quad)
        # TRI_PAD (129) is normalized as 129/128 ≈ 1.0078 — slightly outside [0,1)
        # but distinct from any real coord value (max coord/128 = 127/128 ≈ 0.992).
        # A Linear layer can learn to detect this signal without special treatment.
        self.face_embed = nn.Sequential(
            nn.Linear(cfg.n_face_tokens, cfg.d_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
        )
        self.pos_embed = nn.Embedding(cfg.max_faces + 1, cfg.d_hidden)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.d_hidden, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_hidden * 4,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            ),
            num_layers=cfg.n_layers,
        )
        self.norm = nn.LayerNorm(cfg.d_hidden)
        self.mlp = MLP(
            d_hidden=cfg.d_hidden,
            vocab_size=cfg.vocab_size,
            use_edge_cond=cfg.use_edge_cond,
            relative=cfg.relative,
            n_face_tokens=cfg.n_face_tokens,
        )

    def _relative_anchor(
        self,
        face_input: Int[Tensor, "batch faces n_face_tokens"],
    ) -> Int[Tensor, "batch faces 3"]:
        """Real first vertex v0 used as the relative-coordinate anchor.

        - 9-token (tri-only) or a 12-token quad: v0 = positions 0-2.
        - 12-token triangle: positions 0-2 are TRI_PAD, the real v0 = positions 3-5.

        Detection uses the exact integer oracle (face_input[..., 0] == TRI_PAD),
        the same one used by the spherical embedding, never a float threshold.
        """
        if self.cfg.n_face_tokens == 12:
            is_tri = (face_input[..., 0] == TRI_PAD).unsqueeze(-1)
            return torch.where(is_tri, face_input[..., 3:6], face_input[..., 0:3])
        return face_input[..., 0:3]

    def embed_faces(
        self,
        face_input: Int[Tensor, "batch faces n_face_tokens"],
    ) -> Float[Tensor, "batch faces d_hidden"]:
        N     = face_input.shape[1]
        dtype = self.face_embed[0].weight.dtype
        # Divide by (QUANT_MAX+1)=128 to normalise coords into [0, 0.992].
        # TRI_PAD=129 maps to 129/128=1.0078 — intentionally out-of-range so the
        # linear layer can distinguish pad positions from real coordinates.
        f = face_input.to(dtype=dtype) / (QUANT_MAX + 1)
        if self.cfg.use_spherical_embed:
            # Detect the triangle prefix on the RAW integer tokens (exact
            # == TRI_PAD), not on the normalized float, so face-type detection
            # never depends on a float threshold.  Only meaningful for the
            # 12-token unified layout; 9-token tri-only passes is_tri=None.
            is_tri = face_input[..., 0] == TRI_PAD if face_input.shape[-1] == 12 else None
            f = face_cartesian_to_spherical(f, is_tri=is_tri)
        emb = self.face_embed(f)
        if self.cfg.use_pos_embed:
            emb = emb + self.pos_embed(torch.arange(N, device=face_input.device))
        return emb

    def forward(
        self,
        C: Float[Tensor, "batch slots d_latent"],
        face_input: Int[Tensor, "batch faces n_face_tokens"],
        token_mask: "torch.Tensor | None" = None,
        query_edges: "Int[Tensor, 'batch faces 6'] | None" = None,
    ) -> Float[Tensor, "batch faces n_face_tokens vocab"]:
        B, N, device = *face_input.shape[:2], face_input.device

        latents  = self.latent_proj(C)
        face_emb = self.embed_faces(face_input)

        attn_mask = None
        if token_mask is not None:
            attn_mask = token_mask.repeat_interleave(self.cfg.n_heads, dim=0)

        x = self.transformer(face_emb, latents, tgt_mask=attn_mask)
        x = self.norm(x)

        qe_flat = query_edges.reshape(B * N, 6) if query_edges is not None else None
        # face_v0 / is_tri are used only when relative=True.  The anchor is the real
        # first vertex (TRI_PAD-aware for the 12-token layout); is_tri lets the MLP
        # keep raw logits at the TRI_PAD prefix so TRI_PAD stays predictable.
        face_v0 = self._relative_anchor(face_input).reshape(B * N, 3)
        is_tri_flat = None
        if self.cfg.relative and self.cfg.n_face_tokens == 12:
            is_tri_flat = (face_input[..., 0] == TRI_PAD).reshape(B * N)
        logits  = self.mlp(
            x.reshape(B * N, self.cfg.d_hidden),
            query_edges=qe_flat, face_v0=face_v0, is_tri=is_tri_flat,
        )
        return logits.reshape(B, N, self.cfg.n_face_tokens, self.cfg.vocab_size)
