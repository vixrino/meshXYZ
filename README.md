# mesh_genai

Autoregressive mesh generation via a Transformer decoder conditioned on a point cloud encoder. Each triangle face is predicted as a set of 9 quantized coordinates relative to the first vertex of the query face.

## Setup

```bash
pip install -r requirements.txt
```

Download the encoder weights from [Google Drive](https://drive.google.com/file/d/1qX_YTMAE2tLFppJps3vKAWFbFZgE9CtQ/view?usp=drive_link) and set the path in your config under `encoder.weights_path`.

## Dataset

The dataset should be a directory of mesh files. Supported formats: `.obj`, `.ply`, `.glb`, `.gltf`, `.stl`, `.off`.

```
dataset/
  mesh1.obj
  mesh2.ply
  ...
```

## Training

```bash
python -m src.train --train_dir ./dataset --val_dir ./dataset --config config/config-small-shuffle.yaml
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--train_dir` | required | Directory of training meshes |
| `--val_dir` | required | Directory of validation meshes |
| `--config` | `config/config.yaml` | Path to config file |
| `--output_dir` | `runs` | Directory for checkpoints and logs |

## Outputs

Training outputs are saved under `--output_dir` (default `runs/`):

- `step-N.ckpt` / `last.ckpt` — model checkpoints
- `train_images/` — visualization of predictions vs ground truth at each viz step
- `attention_heatmaps/` — per-layer self-attention heatmaps
- `loss_curve.png` — training loss plot
