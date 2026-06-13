# meshXYZ — Quad-Extended Autoregressive Mesh Transformer

An extension of [bralani/mesh_genai](https://github.com/bralani/mesh_genai) that adds
support for **quadrilateral faces** using a unified 12-token-per-face block representation,
following the approach of [QuadGPT](https://arxiv.org/abs/2406.01238).

The original triangle pipeline is preserved bit-identically. Quad support is opt-in via
`config/config-quad.yaml` and a `face_layout: "quad"` data flag.

---

## Status

| Phase | Description | Status |
|---|---|---|
| **1** | TRI_PAD constant, quad config skeleton | ✅ Done |
| **2** | Custom OBJ parser, unified tokenization, adjacency | ✅ Done |
| **3** | Decoder generalized to n_face_tokens, generate() update | ✅ Done |
| **4** | Loss decomposition, face_type_acc metric, ZYX sort | ✅ Done |
| **5 prep** | Colab notebook, Objaverse data prep, Drive sync | ✅ Done |
| **5 run** | Full training on Objaverse quad meshes | 🔲 Pending |

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Encoder weights

Download the pretrained KLAutoEncoder weights and place them at the path specified in
your config (`encoder.weights_path`). A set of weights trained on the triangle baseline
is available from the project's Google Drive folder.

### Run tests

```bash
# All numpy-only tests (no GPU required)
pytest tests/ -v

# Full suite including torch-dependent tests (requires torch)
pytest tests/ -v  # 35/35 pass with torch installed
```

### Train — triangle baseline

```bash
python -m src.train \
  --config config/config-small-shuffle.yaml \
  --train_dir dataset/ \
  --val_dir dataset/ \
  --output_dir runs/tri-baseline
```

### Train — quad mode (Colab recommended)

Open `colab_quad_training.ipynb` on Google Colab. The notebook handles:
- GPU detection, Drive mount, dependency install
- Objaverse dataset download via `scripts/prep_objaverse.py`
- Wandb logging, checkpoint resume, and Drive sync

For local runs:

```bash
python -m src.train \
  --config config/config-quad.yaml \
  --train_dir data/objaverse_quad/ \
  --val_dir data/objaverse_quad/ \
  --output_dir runs/quad
```

---

## Architecture — Quad Extension

### Token vocabulary

| Token | Value | Meaning |
|---|---|---|
| `QUANT_MAX` | 127 | Max quantized coordinate |
| `EOS_COORD` | 128 | Face sequence end marker |
| `TRI_PAD` | 129 | Triangle prefix in 12-token block |
| `EOS_RESIDUAL` | 255 | No-neighbor sentinel (residual target) |

Vocab size expands from 256 → **257** to accommodate `TRI_PAD`.

### Unified 12-token face block

Both triangles and quads are encoded into a fixed 12-token sequence:

```
Triangle: [TRI_PAD, TRI_PAD, TRI_PAD, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z]
Quad:     [v0x,     v0y,     v0z,     v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]
```

`TRI_PAD = 129 > QUANT_MAX = 127`, so it cannot be confused with a coordinate token.
Dividing by 128 gives `1.0078`, a value distinct from all coordinate inputs — the linear
layers learn to treat this as a face-type signal.

### Face ordering

Faces are sorted globally by **ZYX key** of their first real vertex
(`z·128² + y·128 + x`). This spatially interleaves triangles and quads instead of
grouping them in rigid blocks, forcing the autoregressive decoder to learn face-type
distinction from context rather than position.

### Adjacency

`build_edge_adjacency_unified` returns `(F, 4)` neighbors: slot 3 is `-1` for triangles
(which have only 3 edges). The decoder uses `n_face_tokens` instead of a hardcoded 9
throughout — `face_embed`, `MLP` output, and logit reshape all scale automatically.

### Loss decomposition (Phase 4)

`decompose_loss()` splits the cross-entropy loss into three components:

| Component | Targets |
|---|---|
| `loss_tri_pad` | positions 0–2 of triangle faces (`TRI_PAD` tokens) |
| `loss_coord` | coordinate tokens in valid face positions |
| `loss_eos` | `EOS_RESIDUAL` targets (no-neighbor signal) |

`face_type_acc` measures whether the model correctly predicts `TRI_PAD` vs a coord token
at position 0 — the primary sanity metric for topology distinction.

### Phase 4 validation results (toy dataset)

| Metric | Step 0 | Step 150 |
|---|---|---|
| Total loss | 5.52 | 1.70 |
| `face_type_acc` | ~0.50 | ~0.97 |

---

## Repository Layout

```
config/
  config-small-shuffle.yaml   # triangle training config
  config-quad.yaml            # quad training config (Phase 5 tuned)
src/
  constants.py                # token vocabulary
  dataset/
    obj_parser.py             # OBJ parser preserving tri+quad topology
    mesh_ops.py               # pure-numpy geometry (tokenization, adjacency)
    mesh_dataset.py           # Lightning DataModule
    collate.py                # collate_fn (torch-only, no lightning dep)
    types.py                  # batch type annotations
  model/
    encoder.py                # KLAutoEncoder (frozen, pretrained)
    decoder.py                # autoregressive Transformer decoder
    mesh_transformer.py       # encoder → decoder pipeline + generate()
  training/
    loss.py                   # reconstruction loss + decompose_loss + face_type_acc
    module.py                 # Lightning training module
    callbacks/
      drive_checkpoint.py     # Colab: periodic checkpoint sync to Google Drive
  utils/
    geometry.py               # canonical_face_12, coordinate utilities
scripts/
  prep_objaverse.py           # download + filter Objaverse meshes
  train_quad_mini.py          # 150-step standalone validation run
  regression_check_tri.py     # bit-identical triangle regression check
  smoke_test_quad.py          # end-to-end quad tokenization smoke test
  forward_pass_quad.py        # decoder-only shape smoke test
  integration_smoke_quad.py   # full encoder→decoder integration smoke test
tests/
  test_obj_parser.py          # 16 OBJ parser tests
  test_quad_tokenization.py   # 14 tokenization + adjacency tests
  test_phase3_geometry.py     # 3 canonical_face_12 unit tests
colab_quad_training.ipynb     # end-to-end Colab training notebook
```

---

## Credits

- Original codebase: [bralani/mesh_genai](https://github.com/bralani/mesh_genai)
  (forked at commit `9522db2`)
- QuadGPT paper: [arxiv.org/abs/2406.01238](https://arxiv.org/abs/2406.01238)
- Quad extension (Phases 1–5): [@vixrino](https://github.com/vixrino)
