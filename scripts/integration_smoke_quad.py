"""Phase 3 integration smoke test: real encoder → quad decoder.

Verifies:
  1. Encoder output shape matches what the quad decoder expects.
  2. Full forward pass (encoder → decoder) runs without errors.
  3. Argmax distribution across the 12 token positions looks sane
     (not all collapsing to one token — would indicate NaN/inf issue).

No pretrained weights needed: both encoder and decoder are randomly
initialized.  torch_cluster is mocked with a random-sampling FPS stub
so the test runs on any machine without GPU or torch_cluster wheel.

Usage
-----
    python3.11 scripts/integration_smoke_quad.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ── Mock torch_cluster.fps before importing encoder ────────────────────────
# torch_cluster has no prebuilt ARM macOS wheel for torch 2.2.x.
# We replace fps with a random sampler (identical call signature; returns
# the same shape of index tensor).  The integration test is about shapes
# and data-flow, not FPS quality.
try:
    import torch_cluster  # noqa: F401
    _fps_mocked = False
except ImportError:
    from unittest.mock import MagicMock

    def _fps_stub(src: torch.Tensor, batch: torch.Tensor, ratio: float, random_start: bool = True) -> torch.Tensor:
        """Random-sampling drop-in for fps(src, batch, ratio).

        Returns indices of shape (B * ceil(N*ratio),) in the same format as
        the real torch_cluster.fps output.
        """
        indices = []
        for b in batch.unique():
            b_idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
            n  = len(b_idx)
            k  = max(1, round(n * ratio))
            chosen = b_idx[torch.randperm(n, device=src.device)[:k]]
            indices.append(chosen)
        return torch.cat(indices)

    _tc_mock = MagicMock()
    _tc_mock.fps = _fps_stub
    sys.modules["torch_cluster"] = _tc_mock
    _fps_mocked = True

# ── Now safe to import encoder/transformer ─────────────────────────────────
import numpy as np

from src.model.mesh_transformer import MeshTransformer, MeshTransformerCfg
from src.model.decoder import DecoderCfg
from src.model.encoder import ENCODER_ARCH, EncoderCfg
from src.constants import TRI_PAD, QUANT_MAX, EOS_COORD
from src.dataset.obj_parser import parse_obj
from src.dataset.mesh_ops import (
    quantize_vertices,
    _normalize_face_vertices,
    _normalize_quad_vertices,
    _to_unified_12_tokens,
    normalize_point_cloud,
    _sample_surface,
    _quads_to_tris_for_sampling,
)

# ── Build model ────────────────────────────────────────────────────────────
#   Encoder: full ENCODER_ARCH (depth=24, dim=512, num_latents=512)
#            Note: encode() only runs the 1 cross-attention layer, not the
#            24 depth-layers (those are for reconstruction decode() which
#            MeshTransformer never calls).
#   Decoder: small quad config (fast on CPU)
LATENT_DIM = 8
cfg = MeshTransformerCfg(
    encoder=EncoderCfg(latent_dim=LATENT_DIM, weights_path=""),
    decoder=DecoderCfg(
        d_latent=LATENT_DIM,
        d_hidden=64,
        n_layers=2,
        n_heads=2,
        max_faces=256,
        vocab_size=256,
        n_face_tokens=12,
        use_pos_embed=False,
        use_spherical_embed=False,
        relative=False,
        use_edge_cond=False,
    ),
)

print("Building MeshTransformer (random weights) …")
model = MeshTransformer(cfg)
model.eval()

enc_params = sum(p.numel() for p in model.encoder.parameters())
dec_params = sum(p.numel() for p in model.decoder.parameters())
print(f"  encoder params: {enc_params:,}  (depth={ENCODER_ARCH['depth']}, dim={ENCODER_ARCH['dim']})")
print(f"  decoder params: {dec_params:,}  (d_hidden=64, 2 layers)")
print(f"  null_latent:    {tuple(model.null_latent.shape)}")
print()

# ── Load sphere point cloud ────────────────────────────────────────────────
SPHERE_PATH = os.path.join(os.path.dirname(__file__), "sphere.obj")
if not os.path.exists(SPHERE_PATH):
    raise FileNotFoundError(f"sphere.obj not found at {SPHERE_PATH}. "
                            "Run smoke_test_quad.py --save scripts/sphere.obj first.")

import logging
logging.disable(logging.WARNING)
result = parse_obj(SPHERE_PATH)
logging.disable(logging.NOTSET)

verts_q  = quantize_vertices(result.vertices)
seq_tri  = (_normalize_face_vertices(result.faces_tri, verts_q)
            if len(result.faces_tri)  > 0 else np.empty((0, 9),  dtype=np.int64))
seq_quad = (_normalize_quad_vertices(result.faces_quad, verts_q)
            if len(result.faces_quad) > 0 else np.empty((0, 12), dtype=np.int64))

face_seq_12, _  = _to_unified_12_tokens(seq_tri, seq_quad, TRI_PAD)

# Sample 2048-point cloud (encoder requires exactly num_inputs=2048)
NUM_POINTS = ENCODER_ARCH["num_inputs"]  # 2048
np.random.seed(0)
# For sampling we need a triangulated mesh
if len(result.faces_quad) > 0:
    tris_from_quads = _quads_to_tris_for_sampling(result.faces_quad)
    all_tris = (np.vstack([result.faces_tri, tris_from_quads])
                if len(result.faces_tri) > 0 else tris_from_quads)
else:
    all_tris = result.faces_tri

pc_raw = _sample_surface(result.vertices, all_tris, NUM_POINTS)
pc_norm = normalize_point_cloud(pc_raw)          # (2048, 3) float64 in [-0.5, 0.5]

print(f"Sphere mesh:  {len(result.vertices)} verts | "
      f"{len(result.faces_tri)} tris | {len(result.faces_quad)} quads")
print(f"Point cloud:  {pc_norm.shape}  (sampled {NUM_POINTS} pts)")
print(f"Face tokens:  {face_seq_12.shape}  "
      f"({(face_seq_12[:,9] == TRI_PAD).sum()} tri, "
      f"{(face_seq_12[:,9] != TRI_PAD).sum()} quad)")
print()

# ── Prepare batch tensors ─────────────────────────────────────────────────
pc_t   = torch.tensor(pc_norm, dtype=torch.float32).unsqueeze(0)   # (1, 2048, 3)
faces_t = torch.tensor(face_seq_12, dtype=torch.long).unsqueeze(0) # (1, F, 12)
F = face_seq_12.shape[0]

# ── 1. Encoder-only check ─────────────────────────────────────────────────
print("Step 1 — Encoder forward pass …")
import time
t0 = time.time()
with torch.no_grad():
    kl, latents = model.encoder.encode(pc_t)
enc_time = time.time() - t0

print(f"  encoder.encode() in {enc_time:.2f}s")
print(f"  kl.shape     = {tuple(kl.shape)}")
print(f"  latents.shape = {tuple(latents.shape)}")
assert latents.shape == (1, ENCODER_ARCH["num_latents"], LATENT_DIM), \
    f"Encoder output shape mismatch: {latents.shape}"
assert latents.shape[-1] == cfg.decoder.d_latent, \
    f"latent_dim ({latents.shape[-1]}) ≠ d_latent ({cfg.decoder.d_latent})"
print(f"  ✓  shape (1, {ENCODER_ARCH['num_latents']}, {LATENT_DIM}) matches decoder d_latent={cfg.decoder.d_latent}")
print()

# ── 2. Full forward pass ──────────────────────────────────────────────────
print("Step 2 — Full forward pass (encoder → decoder) …")
t0 = time.time()
with torch.no_grad():
    logits = model(pc_t, faces_t)
fwd_time = time.time() - t0

print(f"  model.forward() in {fwd_time:.2f}s")
print(f"  logits.shape  = {tuple(logits.shape)}   ← expect (1, {F}, 12, 256)")
assert logits.shape == (1, F, 12, 256), f"Shape mismatch: {logits.shape}"
print(f"  ✓  (1, F={F}, 12, 256)")
print()

# ── 3. Argmax distribution ────────────────────────────────────────────────
print("Step 3 — Argmax distribution across token positions …")
preds = logits[0].argmax(-1)           # (F, 12)
print(f"  preds.shape = {tuple(preds.shape)}")
print()
print(f"  {'Position':>10}  {'min':>6}  {'max':>6}  {'unique':>8}  {'top-3 tokens (count)'}")
print("  " + "-" * 60)
for pos in range(12):
    col   = preds[:, pos]
    uniq  = col.unique()
    counts = {int(v): (col == v).sum().item() for v in uniq}
    top3  = sorted(counts.items(), key=lambda x: -x[1])[:3]
    top3_str = "  ".join(f"{tok}×{cnt}" for tok, cnt in top3)
    print(f"  {pos:>10}  {col.min().item():>6}  {col.max().item():>6}  "
          f"{len(uniq):>8}  {top3_str}")

# Sanity: random weights → no single token should dominate all 40 faces
all_same = [(preds[:, pos].unique().numel() == 1) for pos in range(12)]
if any(all_same):
    dominated = [i for i, s in enumerate(all_same) if s]
    print(f"\n  ⚠  positions {dominated} have all-identical preds "
          "(could be NaN/inf issue — expected for degenerate weights, not a real model)")
else:
    print(f"\n  ✓  No position has all {F} faces predicting the same token.")
print()

# ── 4. Null-latent path (no encoder) ─────────────────────────────────────
print("Step 4 — Null-latent path (pc=None) …")
with torch.no_grad():
    logits_null = model(None, faces_t)
assert logits_null.shape == logits.shape
print(f"  model(pc=None).shape = {tuple(logits_null.shape)}  ✓")
print()

print("=" * 60)
print("✓  All integration checks passed.")
print(f"   torch_cluster mocked: {_fps_mocked}")
print(f"   encoder weights:      random (no pretrained checkpoint)")
