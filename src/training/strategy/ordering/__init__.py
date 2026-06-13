from __future__ import annotations

import random

import dacite
import torch
from torch import Tensor

from ....dataset.types import Batch
from .base import BaseOrdering
from .bfs_ordering import BFSOrdering
from .canonical import CanonicalOrdering
from .causal_axis import CausalAxisOrdering
from .shuffle_ordering import ShuffleCfg, ShuffleOrdering
from .dfs_ordering import DFSOrdering
from .random_ordering import RandomOrdering

__all__ = [
    "BaseOrdering", "CanonicalOrdering", "RandomOrdering", "CausalAxisOrdering",
    "BFSOrdering", "DFSOrdering", "ShuffleOrdering", "OrderingPipeline",
]

_REGISTRY: dict[str, tuple[type[BaseOrdering], type | None]] = {
    "canonical":   (CanonicalOrdering,  None),
    "random":      (RandomOrdering,     None),
    "causal_axis": (CausalAxisOrdering, None),
    "bfs":         (BFSOrdering,        None),
    "dfs":         (DFSOrdering,        None),
    "shuffle":     (ShuffleOrdering,    ShuffleCfg),
}


def _build_ordering(entry: dict) -> BaseOrdering:
    entry = dict(entry)
    key = entry.pop("type")
    strategy_cls, cfg_cls = _REGISTRY[key]
    return strategy_cls(dacite.from_dict(cfg_cls, entry)) if cfg_cls else strategy_cls()


def _apply_perm(faces: Tensor, face_neighbors: Tensor, perm: Tensor) -> tuple[Tensor, Tensor]:
    """Apply a batch permutation to faces and remap face_neighbors accordingly.

    faces:          (B, N, 9)
    face_neighbors: (B, N, 3)  — indices into original sequence, -1 = no neighbor
    perm:           (B, N)     — new[i] = old[perm[i]]

    Returns reordered (faces, face_neighbors).
    """
    B, N, _ = faces.shape
    device = faces.device

    new_faces = faces.gather(1, perm.unsqueeze(-1).expand(B, N, 9))

    # Reorder neighbor rows to match new face positions
    new_neighbors = face_neighbors.gather(1, perm.unsqueeze(-1).expand(B, N, 3))

    # Build inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm.scatter_(1, perm, torch.arange(N, device=device).unsqueeze(0).expand(B, N))

    # Remap neighbor indices: old position k → new position inv_perm[b, k]
    valid = new_neighbors >= 0
    safe = new_neighbors.clamp(min=0)
    remapped = inv_perm.gather(1, safe.reshape(B, -1)).reshape(B, N, 3)
    new_neighbors = torch.where(valid, remapped, torch.full_like(remapped, -1))

    return new_faces, new_neighbors


class OrderingPipeline:
    """Two-stage ordering applied each training step.

    Stage 1 — pick exactly one strategy from ``strategies`` according to prob weights.
    Stage 2 — apply each entry in ``post_processing`` independently (each fires with its
               own per-sample prob baked into the strategy's cfg, e.g. ContiguousShuffleCfg.prob).

    """

    def __init__(
        self,
        strategies: list[BaseOrdering],
        probs: list[float],
        post_processors: list[BaseOrdering],
    ):
        total = sum(probs)
        self.strategies = strategies
        self.probs = [p / total for p in probs]
        self.post_processors = post_processors

    def apply_to_batch(self, batch: Batch) -> Batch:
        # Stage 1: pick one main ordering
        strategy = random.choices(self.strategies, weights=self.probs, k=1)[0]
        perm = strategy.permute(batch["faces"], batch["face_neighbors"], batch["lengths"])
        faces, neighbors = _apply_perm(batch["faces"], batch["face_neighbors"], perm)
        batch = {**batch, "faces": faces, "face_neighbors": neighbors}

        # Stage 2: post-processors fire independently
        for proc in self.post_processors:
            perm = proc.permute(batch["faces"], batch["face_neighbors"], batch["lengths"])
            faces, neighbors = _apply_perm(batch["faces"], batch["face_neighbors"], perm)
            batch = {**batch, "faces": faces, "face_neighbors": neighbors}

        return batch

    @classmethod
    def from_cfg(cls, cfg: dict) -> "OrderingPipeline":
        strategies: list[BaseOrdering] = []
        probs: list[float] = []
        for entry in cfg.get("strategies", [{"type": "canonical", "prob": 1.0}]):
            entry = dict(entry)
            prob = float(entry.pop("prob", 1.0))
            strategies.append(_build_ordering(entry))
            probs.append(prob)

        post_processors = [
            _build_ordering(dict(entry))
            for entry in cfg.get("post_processing", [])
        ]

        return cls(strategies, probs, post_processors)
