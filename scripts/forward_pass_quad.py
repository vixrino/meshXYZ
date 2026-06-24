"""Phase 3 forward-pass smoke test.

Instantiates a Decoder (quad config) with random weights, runs one forward
pass with synthetic 12-token faces, and verifies the output shape is
(B, F, 12, 256).  No training, no encoder, no gradient.

Usage
-----
    python3.11 scripts/forward_pass_quad.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.decoder import Decoder, DecoderCfg
from src.constants import TRI_PAD, QUANT_MAX

# ── Small quad-mode config (fast to instantiate) ──────────────────────────
cfg = DecoderCfg(
    d_latent=8,
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
)

decoder = Decoder(cfg)
decoder.eval()

# ── Synthetic mixed batch: 6 triangles + 4 quads ──────────────────────────
B, F = 2, 10
faces = torch.randint(0, QUANT_MAX, (B, F, 12))   # all quads initially
faces[:, :6, 9:] = TRI_PAD                        # mark first 6 rows as triangles (pad at end)

# Fake latents (as if from encoder with d_latent=8, 16 latent slots)
C = torch.randn(B, 16, cfg.d_latent)

with torch.no_grad():
    logits = decoder(C, faces)

# ── Shape assertion ────────────────────────────────────────────────────────
expected = (B, F, 12, 256)
assert logits.shape == expected, f"Shape mismatch: got {tuple(logits.shape)}, expected {expected}"

print("=" * 60)
print("Phase 3 forward-pass smoke test")
print("=" * 60)
print(f"  faces.shape   = {tuple(faces.shape)}")
print(f"  C.shape       = {tuple(C.shape)}")
print(f"  logits.shape  = {tuple(logits.shape)}  ✓")
print()

# ── Decode argmax for a couple of faces ────────────────────────────────────
for b_idx, f_idx, label in [(0, 0, "batch 0, face 0  [TRI input]"),
                              (0, 7, "batch 0, face 7  [QUAD input]")]:
    pred   = logits[b_idx, f_idx].argmax(-1).tolist()   # (12,)
    inp    = faces[b_idx, f_idx].tolist()
    is_tri = inp[9] == TRI_PAD
    print(f"  {label}")
    print(f"    input  tokens: {inp}")
    print(f"    argmax tokens: {pred}")
    print(f"    pred type:     {'TRI' if pred[9] > QUANT_MAX else 'QUAD'}"
          f"  (based on pred[9]={pred[9]})")
    print()

print("✓  All assertions passed.")
