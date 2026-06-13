from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .decoder import Decoder, DecoderCfg
from .encoder import ENCODER_ARCH, EncoderCfg, KLAutoEncoder
from ..constants import EOS_COORD, EOS_RESIDUAL, QUANT_MAX
from ..utils.geometry import canonical_face


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
        faces: Int[Tensor, "batch faces 9"],
        token_mask: "torch.Tensor | None" = None,
        query_edges: "Int[Tensor, 'batch faces 6'] | None" = None,
    ) -> Float[Tensor, "batch faces 9 vocab"]:
        if points is None:
            B = faces.shape[0]
            C = self.null_latent.expand(B, -1, -1)
        else:
            _, C = self.encoder.encode(points)

        return self.decoder(C, faces, token_mask=token_mask, query_edges=query_edges)

    @torch.no_grad()
    def generate(
        self,
        faces: Int[Tensor, "batch faces 9"],
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
            intermediates       = [curr_faces.cpu().numpy().copy()] if return_intermediates else None
            eos_snapshots       = [frozenset()]                      if return_intermediates else None
            boundary_snapshots  = []   # list of boundary edge key lists, one per step
            query_snapshots     = []   # list of query edge_key (or None), one per step
            attempts   = 0
            eos_count  = 0
            step_probs: list[np.ndarray] = []  # argmax probs per successful step, shape (3,)

            # edge_map: edge_key -> face_idx (only open/boundary edges)
            faces_cpu = curr_faces.cpu().reshape(-1, 3, 3).tolist()
            edge_map: dict = {}
            for i, fv in enumerate(faces_cpu):
                for a, b_idx in [(0, 1), (1, 2), (2, 0)]:
                    key = (min(tuple(fv[a]), tuple(fv[b_idx])), max(tuple(fv[a]), tuple(fv[b_idx])))
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
                    coord_logits = logits[0, query_idx]               # (9, vocab)
                    pred = coord_logits.argmax(-1)                    # (9,)

                    if (pred == EOS_RESIDUAL).any():
                        eos_count += 1
                        eos_faces.add(query_idx)
                        continue

                    new_face = canonical_face(pred.clamp(0, QUANT_MAX).reshape(3, 3))
                    face_key = tuple(new_face.tolist())

                    if face_key in face_set:
                        visited.add(query_idx)
                        continue

                    probs = coord_logits.softmax(-1)
                    step_probs.append(probs[torch.arange(9), pred].cpu().numpy())

                new_idx = curr_faces.shape[0]
                face_set.add(face_key)
                curr_faces = torch.cat([curr_faces, new_face.unsqueeze(0)], dim=0)
                new_fv = new_face.cpu().reshape(3, 3).tolist()
                for a, b_idx in [(0, 1), (1, 2), (2, 0)]:
                    key = (min(tuple(new_fv[a]), tuple(new_fv[b_idx])), max(tuple(new_fv[a]), tuple(new_fv[b_idx])))
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
                all_step_probs.append(np.stack(step_probs) if step_probs else np.empty((0, 3)))
                all_boundary_snapshots.append(boundary_snapshots)
                all_query_snapshots.append(query_snapshots)

        if return_intermediates:
            return results, all_intermediates, all_eos_snapshots, all_step_probs, all_boundary_snapshots, all_query_snapshots
        return results

