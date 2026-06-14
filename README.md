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
| **5 prep** | Colab notebook, Objaverse data prep, quadrangulation, Drive sync | ✅ Done |
| **5 run** | Full training on Objaverse quad meshes | 🟡 In progress |

A 100-step dry run on a Colab T4 (100 meshes, balanced tri/quad, `batch_size=4`)
validates the full pipeline end-to-end: loss drops `5.5 → 4.05`,
`face_type_acc` reaches **0.97**, and `generate()` produces coherent meshes
(13–37 faces) without errors. The full 50 000-step run is the remaining step.

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
# numpy-only tests run anywhere; torch-dependent tests auto-skip without torch
pytest tests/ -v
```

The suite has **47 tests** across four files:

| File | Tests | Coverage |
|---|---|---|
| `test_obj_parser.py` | 16 | OBJ parsing: negative indices, `a/b/c/d` syntax, mixed tri+quad |
| `test_quad_tokenization.py` | 16 | 12-token tokenization, adjacency, bit-identical tri regression |
| `test_phase3_geometry.py` | 3 | `canonical_face_12` edge cases |
| `test_quad_ordering.py` | 12 | quad-aware `_apply_perm`, `CanonicalOrdering`, `CausalAxisOrdering` |

Tests requiring torch (`test_quad_ordering.py`, parts of tokenization) skip
automatically when torch is absent, so the numpy core can be validated locally.

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

### Dataset preparation — balanced tri/quad

Objaverse exports are all triangulated (GLB/GLTF, no native quads).
`scripts/prep_objaverse.py` downloads, filters, and **quadrangulates** meshes
into a balanced tri/quad dataset using pymeshlab's smart triangle pairing:

```bash
# Balanced ~50% quad / ~50% triangle (default)
python scripts/prep_objaverse.py --n 500 --out_dir data/train --mix_ratio 0.5

# All quadrangulated  |  all triangle-only
python scripts/prep_objaverse.py --n 500 --out_dir data/train --mix_ratio 1.0
python scripts/prep_objaverse.py --n 500 --out_dir data/train --mix_ratio 0.0
```

`--mix_ratio` controls the fraction of meshes that get quadrangulated; the rest
are kept as triangles. The per-mesh decision is seeded (`seed ^ hash(uid)`) so
the split is reproducible regardless of download order. **All outputs are saved
as `.obj`** so the format stays homogeneous — triangle-only meshes are encoded
into 12-token blocks with `TRI_PAD` by the dataset, same as quads.

A balanced dataset matters: quadrangulating everything produces ~100%-quad
meshes, which starves `loss_tri_pad` and `face_type_acc` of triangle signal.
The script prints the final composition (tri-only / mixed / quad-dominant
mesh counts, plus the global face-level tri/quad fraction).

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

### EOS weighting (`eos_weight`)

**What it is.** The total training loss (`reconstruction_loss`) is a single
unweighted cross-entropy over *all* token positions. But `EOS_RESIDUAL` (the
"end of mesh" target) appears only once per mesh — under 1% of tokens. With so
little gradient pressure, the optimizer trades EOS away to sharpen coordinate
predictions, and the shared softmax pushes the EOS logit down everywhere. The
symptom is `loss_eos` *increasing* during training instead of being merely
noisy. This is class imbalance, which **more data does not fix** (the EOS:coord
ratio is invariant to dataset size).

`training.eos_weight` applies a per-class weight to the `EOS_RESIDUAL` index in
the cross-entropy, counteracting the imbalance. Because all 12 token positions
of an EOS face are `EOS_RESIDUAL` (the causal target builder fills the whole
row), weighting the single class index cleanly upweights every EOS position.

**Default `1.0`.** At `1.0` the weight tensor is all-ones, so the loss is
bit-identical to the unweighted version — triangle runs and existing configs
are unaffected.

**How to tune.** Start cautious and raise gradually:

| `eos_weight` | Effect |
|---|---|
| `1.0` | No upweighting (default, backward-compatible) |
| `10` | Cautious first test — current `config-quad.yaml` value |
| `20–30` | Stronger; use only if `loss_eos` still rises at 10 |

Watch these Wandb metrics together:
- `train/loss_eos` — should stop rising / start dropping.
- `train/eos_acc` — should climb.
- **`train/gen_final_faces`** — the guard-rail. Upweighting EOS too much makes
  the model emit EOS too early, collapsing generated meshes to a handful of
  faces. If `gen_final_faces` drops toward 1–5, `eos_weight` is too high —
  step back down. A healthy value stays in the tens-to-low-hundreds range.

### Phase 4 validation results (toy dataset)

| Metric | Step 0 | Step 150 |
|---|---|---|
| Total loss | 5.52 | 1.70 |
| `face_type_acc` | ~0.50 | ~0.97 |

### Robustness (Colab-validated)

Fixes hardened during real Colab runs, all covered by tests or regression checks:

- **`torch_cluster` fallback** — `encoder.py` falls back to a pure-PyTorch
  Farthest Point Sampling (`fps_fallback.py`) when no precompiled wheel exists
  for the runtime's torch/CUDA version.
- **Degenerate-mesh sampling** — `_sample_surface` normalizes face-area
  probabilities defensively (uniform fallback on zero/NaN areas) so
  `np.random.choice` never raises on broken Objaverse meshes.
- **Ordering generalization** — `_apply_perm`, `CanonicalOrdering`, and
  `CausalAxisOrdering` handle both 9-token (tri) and 12-token (quad) layouts,
  with TRI_PAD-aware sort keys so padded triangles interleave correctly.
- **Config propagation** — `face_layout: "quad"` reaches `MeshDataset` reliably
  (the `DataCfg` field is `str`, not `Literal`, to survive older `dacite`).
- **Visualization** — `viz.py` decodes 9- and 12-token faces generically
  instead of assuming 3 vertices per face.

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
    encoder.py                # KLAutoEncoder (frozen, pretrained) + fps fallback hook
    fps_fallback.py           # pure-PyTorch Farthest Point Sampling (no torch_cluster)
    decoder.py                # autoregressive Transformer decoder
    mesh_transformer.py       # encoder → decoder pipeline + generate()
  training/
    loss.py                   # reconstruction loss + decompose_loss + face_type_acc
    module.py                 # Lightning training module
    strategy/ordering/        # face ordering strategies (quad-aware)
    callbacks/
      drive_checkpoint.py     # Colab: periodic checkpoint sync to Google Drive
  utils/
    geometry.py               # canonical_face_12, coordinate utilities
    viz.py                    # generic 9-/12-token face rendering
scripts/
  prep_objaverse.py           # download + filter + quadrangulate (--mix_ratio)
  train_quad_mini.py          # 150-step standalone validation run
  regression_check_tri.py     # bit-identical triangle regression check
  smoke_test_quad.py          # end-to-end quad tokenization smoke test
  forward_pass_quad.py        # decoder-only shape smoke test
  integration_smoke_quad.py   # full encoder→decoder integration smoke test
tests/
  test_obj_parser.py          # 16 OBJ parser tests
  test_quad_tokenization.py   # 16 tokenization + adjacency tests
  test_phase3_geometry.py     # 3 canonical_face_12 unit tests
  test_quad_ordering.py       # 12 quad-aware ordering tests
colab_quad_training.ipynb     # end-to-end Colab training notebook
```

---

## Credits

- Original codebase: [bralani/mesh_genai](https://github.com/bralani/mesh_genai)
  (forked at commit `9522db2`)
- QuadGPT paper: [arxiv.org/abs/2406.01238](https://arxiv.org/abs/2406.01238)
- Quad extension (Phases 1–5): [@vixrino](https://github.com/vixrino)
