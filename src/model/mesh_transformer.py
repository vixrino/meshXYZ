from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .decoder import Decoder, DecoderCfg
from .encoder import ENCODER_ARCH, EncoderCfg, KLAutoEncoder
from ..constants import EOS_COORD, EOS_RESIDUAL, QUANT_MAX
from ..utils.geometry import canonical_face, canonical_face_12


def _face_tokens_to_verts_edges(
    face_tokens: list[int],
    n_face_tokens: int,
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Extract vertex lists and edge index pairs from a flat face token list.

    n_face_tokens == 9  (original triangle, no TRI_PAD):
        verts  = [tokens[0:3], tokens[3:6], tokens[6:9]]
        edges  = [(0,1), (1,2), (2,0)]

    n_face_tokens == 12 (unified block):
        token[0] > QUANT_MAX  → triangle (TRI_PAD prefix):
            verts  = [tokens[3:6], tokens[6:9], tokens[9:12]]
            edges  = [(0,1), (1,2), (2,0)]
        token[0] ≤ QUANT_MAX  → quad:
            verts  = [tokens[0:3], tokens[3:6], tokens[6:9], tokens[9:12]]
            edges  = [(0,1), (1,2), (2,3), (3,0)]
    """
    if n_face_tokens == 9:
        verts = [face_tokens[0:3], face_tokens[3:6], face_tokens[6:9]]
        return verts, [(0, 1), (1, 2), (2, 0)]
    # 12-token unified block
    if face_tokens[0] > QUANT_MAX:          # TRI_PAD at position 0 → triangle
        verts = [face_tokens[3:6], face_tokens[6:9], face_tokens[9:12]]
        return verts, [(0, 1), (1, 2), (2, 0)]
    # quad
    verts = [face_tokens[0:3], face_tokens[3:6], face_tokens[6:9], face_tokens[9:12]]
    return verts, [(0, 1), (1, 2), (2, 3), (3, 0)]


@dataclass
class MeshTransformerCfg:
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    decoder: DecoderCfg = field(default_factory=DecoderCfg)


class MeshTransformer(nn.Module):
    """
    Autoregressive AutoEncoder for triangle mesh generation.

    Encoder: KLAutoEncoder from 3DShape2VecSet — maps point clouds to VecSet latents.
    Decoder: Decoder — generates faces autoregressively from latents.
    """

    def __init__(self, cfg: MeshTransformerCfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = KLAutoEncoder(**ENCODER_ARCH, latent_dim=cfg.encoder.latent_dim)
        self.decoder = Decoder(cfg.decoder)
        self.null_latent = nn.Parameter(
            torch.zeros(1, ENCODER_ARCH["num_latents"], cfg.encoder.latent_dim)
        )

    def forward(
        self,
        points: "Float[Tensor, 'batch points 3'] | None",
        faces: Int[Tensor, "batch faces n_face_tokens"],
        token_mask: "torch.Tensor | None" = None,
        query_edges: "Int[Tensor, 'batch faces 6'] | None" = None,
    ) -> Float[Tensor, "batch faces n_face_tokens vocab"]:
        if points is None:
            B = faces.shape[0]
            C = self.null_latent.expand(B, -1, -1)
        else:
            _, C = self.encoder.encode(points)

        return self.decoder(C, faces, token_mask=token_mask, query_edges=query_edges)

    @torch.no_grad()
    def generate(
        self,
        faces: Int[Tensor, "batch faces n_face_tokens"],
        pc: "Float[Tensor, 'batch points 3'] | None" = None,
        max_steps: int = 500,
        return_intermediates: bool = False,
        confidence_threshold: float = 0.98,
    ):
        if pc is not None:
            _, latents = self.encoder.encode(pc)  # (B, 512, d_latent)
        else:
            latents = None

        results = []
        all_intermediates      = []
        all_eos_snapshots      = []
        all_step_probs         = []
        all_boundary_snapshots = []
        all_query_snapshots    = []

        for b in range(faces.shape[0]):
            valid = faces[b, :, 0] != EOS_COORD
            curr_faces = faces[b][valid]  # (real_N, 9)
            if latents is not None:
                latent = latents[b : b + 1]
            else:
                latent = self.null_latent.expand(1, -1, -1)
            face_set: set = {tuple(f.tolist()) for f in curr_faces}
            n_face_tokens       = curr_faces.shape[-1]   # 9 (tri) or 12 (unified)
            intermediates       = [curr_faces.cpu().numpy().copy()] if return_intermediates else None
            eos_snapshots       = [frozenset()]                      if return_intermediates else None
            boundary_snapshots  = []   # list of boundary edge key lists, one per step
            query_snapshots     = []   # list of query edge_key (or None), one per step
            attempts   = 0
            eos_count  = 0
            step_probs: list[np.ndarray] = []  # argmax probs per step, shape (n_face_tokens,) or (3,)

            # edge_map: edge_key -> face_idx (only open/boundary edges)
            edge_map: dict = {}
            for i, face_tokens in enumerate(curr_faces.cpu().tolist()):
                verts_i, edges_i = _face_tokens_to_verts_edges(face_tokens, n_face_tokens)
                for a, b_idx in edges_i:
                    key = (
                        min(tuple(verts_i[a]), tuple(verts_i[b_idx])),
                        max(tuple(verts_i[a]), tuple(verts_i[b_idx])),
                    )
                    if key in edge_map:
                        edge_map.pop(key)
                    else:
                        edge_map[key] = i

            if return_intermediates:
                boundary_snapshots.append(list(edge_map.keys()))
                query_snapshots.append(None)

            use_edge_cond = self.cfg.decoder.use_edge_cond
            # Per-edge tracking (used when use_edge_cond=True)
            eos_edges: set          = set()
            visited_edges: set      = set()
            skip_count: dict        = {}   # edge_key -> times skipped due to low confidence
            # Per-face tracking (used when use_edge_cond=False, legacy)
            eos_faces: set = set()
            visited:   set = set()

            while attempts < max_steps:
                if use_edge_cond:
                    open_items = [
                        (key, idx) for key, idx in edge_map.items()
                        if key not in eos_edges and key not in visited_edges
                    ]
                    if not open_items:
                        break
                    edge_key, query_idx = open_items[0]
                else:
                    open_faces = list(dict.fromkeys(
                        i for i in edge_map.values() if i not in visited and i not in eos_faces
                    ))
                    if not open_faces:
                        break
                    query_idx = open_faces[0]

                attempts += 1

                if use_edge_cond:
                    # Build query_edges tensor: zeros everywhere, fill at query_idx
                    N_curr = curr_faces.shape[0]
                    query_edges_t = torch.zeros(1, N_curr, 6, dtype=curr_faces.dtype, device=curr_faces.device)
                    ev0_list, ev1_list = edge_key  # each is a 3-tuple of ints (sorted)
                    ev0 = torch.tensor(list(ev0_list), dtype=curr_faces.dtype, device=curr_faces.device)
                    ev1 = torch.tensor(list(ev1_list), dtype=curr_faces.dtype, device=curr_faces.device)
                    query_edges_t[0, query_idx] = torch.cat([ev0, ev1])

                    logits = self.decoder(latent, curr_faces.unsqueeze(0), token_mask=None, query_edges=query_edges_t)
                    coord_logits = logits[0, query_idx, 6:]          # (3, vocab)
                    pred_v2 = coord_logits.argmax(-1)                 # (3,)

                    if (pred_v2 == EOS_RESIDUAL).any():
                        eos_count += 1
                        eos_edges.add(edge_key)
                        continue

                    new_face = canonical_face(torch.stack([ev0, ev1, pred_v2.clamp(0, QUANT_MAX)]))
                    face_key = tuple(new_face.tolist())

                    if face_key in face_set:
                        visited_edges.add(edge_key)
                        continue

                    probs = coord_logits.softmax(-1)
                    argmax_probs = probs[torch.arange(3), pred_v2]
                    step_probs.append(argmax_probs.cpu().numpy())

                    # low-confidence: re-queue at the back, or drop after 2 skips
                    if argmax_probs.min().item() < confidence_threshold:
                        skip_count[edge_key] = skip_count.get(edge_key, 0) + 1
                        if skip_count[edge_key] > 2:
                            visited_edges.add(edge_key)
                        else:
                            face_idx = edge_map.pop(edge_key)
                            edge_map[edge_key] = face_idx  # re-insert at back of queue
                        continue
                else:
                    logits = self.decoder(latent, curr_faces.unsqueeze(0), token_mask=None)
                    coord_logits = logits[0, query_idx]               # (n_face_tokens, vocab)
                    pred = coord_logits.argmax(-1)                    # (n_face_tokens,)

                    if (pred == EOS_RESIDUAL).any():
                        eos_count += 1
                        eos_faces.add(query_idx)
                        continue

                    if n_face_tokens == 12:
                        new_face = canonical_face_12(pred)
                    else:
                        new_face = canonical_face(pred.clamp(0, QUANT_MAX).reshape(3, 3))
                    face_key = tuple(new_face.tolist())

                    if face_key in face_set:
                        visited.add(query_idx)
                        continue

                    probs = coord_logits.softmax(-1)
                    # TODO: for quad mode, filter to coord-only positions
                    #       (skip TRI_PAD slots 0-2 for triangle faces)
                    step_probs.append(probs[torch.arange(n_face_tokens), pred].cpu().numpy())

                new_idx = curr_faces.shape[0]
                face_set.add(face_key)
                curr_faces = torch.cat([curr_faces, new_face.unsqueeze(0)], dim=0)
                new_verts, new_edges = _face_tokens_to_verts_edges(
                    new_face.cpu().tolist(), n_face_tokens
                )
                for a, b_idx in new_edges:
                    key = (
                        min(tuple(new_verts[a]), tuple(new_verts[b_idx])),
                        max(tuple(new_verts[a]), tuple(new_verts[b_idx])),
                    )
                    if key in edge_map:
                        j = edge_map.pop(key)
                        if use_edge_cond:
                            visited_edges.discard(key)
                        else:
                            visited.discard(j)
                    else:
                        edge_map[key] = new_idx

                if return_intermediates:
                    intermediates.append(curr_faces.cpu().numpy().copy())
                    eos_snapshots.append(frozenset(eos_edges if use_edge_cond else eos_faces))
                    boundary_snapshots.append(list(edge_map.keys()))
                    query_snapshots.append(edge_key if use_edge_cond else None)

            print(f"[generate] done b={b} final_faces={curr_faces.shape[0]} attempts={attempts} eos={eos_count}")
            results.append(curr_faces)
            if return_intermediates:
                all_intermediates.append(intermediates)
                all_eos_snapshots.append(eos_snapshots)
                n_step = 3 if use_edge_cond else n_face_tokens
                all_step_probs.append(np.stack(step_probs) if step_probs else np.empty((0, n_step)))
                all_boundary_snapshots.append(boundary_snapshots)
                all_query_snapshots.append(query_snapshots)

        if return_intermediates:
            return results, all_intermediates, all_eos_snapshots, all_step_probs, all_boundary_snapshots, all_query_snapshots
        return results

