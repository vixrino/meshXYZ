"""Standalone inference script for mesh generation.

Usage:
    python -m src.infer \
        --config config/config.yaml \
        --ckpt   runs/last.ckpt \
        --mesh   path/to/mesh.obj \
        --out    output/gen.mp4 \
        --ctx    50 \
        --steps  300 \
        [--use-pc]   # add this only when trained with pc_cond_prob > 0
"""

import argparse
import os
import time

import imageio
import numpy as np
import torch
import trimesh

from .dataset.mesh_dataset import load_mesh_raw, process_mesh
from .model.mesh_transformer import MeshTransformer
from .training.module import MeshTransformerModule
from .utils.cfg import load_cfg
from .utils.viz import _bounding_sphere_scale, _render_gen_frame, _rotation_matrix


def _fix_normals(faces_t: torch.Tensor) -> torch.Tensor:
    """Flip face winding so all normals point outward, using trimesh."""
    faces_np = faces_t.cpu().numpy().reshape(-1, 3, 3)   # (N, 3, 3) int coords
    verts_flat = faces_np.reshape(-1, 3)
    unique_verts, inverse = np.unique(verts_flat, axis=0, return_inverse=True)
    face_indices = inverse.reshape(-1, 3)

    mesh = trimesh.Trimesh(vertices=unique_verts.astype(float), faces=face_indices, process=False)
    trimesh.repair.fix_normals(mesh, multibody=True)

    corrected = unique_verts[mesh.faces].reshape(-1, 9).astype(np.int64)
    return torch.from_numpy(corrected).to(faces_t.device)


def _save_mesh(faces_t: torch.Tensor, out_path: str) -> None:
    """Export generated mesh as OBJ (quantized integer coordinates)."""
    faces_np = faces_t.cpu().numpy().reshape(-1, 3, 3)
    verts_flat = faces_np.reshape(-1, 3)
    unique_verts, inverse = np.unique(verts_flat, axis=0, return_inverse=True)
    face_indices = inverse.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=unique_verts.astype(float), faces=face_indices, process=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    mesh.export(out_path)
    print(f"Saved mesh ({len(face_indices)} faces) → {out_path}")


def _bfs_order(face_neighbors: np.ndarray) -> np.ndarray:
    """Return BFS face ordering starting from face 0, handling disconnected components."""
    from collections import deque
    N = len(face_neighbors)
    visited = np.zeros(N, dtype=bool)
    order = []
    seeds = [0] + list(range(N))
    for seed in seeds:
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in face_neighbors[node]:
                if nb >= 0 and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
    return np.array(order, dtype=np.int64)


def _build_context(
    mesh_path: str,
    num_points: int,
    device: torch.device,
    mask_ratio: float = 0.75,
    max_ctx: int | None = None,
):
    """Load a mesh, BFS-order it, then take the first n_keep faces as context."""
    verts, faces = load_mesh_raw(mesh_path)
    pc_np, face_seq_np, face_neighbors_np = process_mesh(verts, faces, num_points, augment=False)

    bfs_perm   = _bfs_order(face_neighbors_np)
    face_seq_np = face_seq_np[bfs_perm]                         # reorder by BFS

    N      = len(face_seq_np)
    r      = np.random.uniform(0.0, mask_ratio)
    n_keep = max(1, N - int(N * r))
    if max_ctx is not None:
        n_keep = min(n_keep, max_ctx)

    ctx  = torch.from_numpy(face_seq_np[:n_keep]).long().unsqueeze(0).to(device)  # (1, n_keep, 9)
    pc_t = torch.from_numpy(pc_np).to(device).unsqueeze(0)                        # (1, N_pts, 3)

    print(f"Mesh: {N} faces  |  context: {n_keep} kept ({100 * n_keep / N:.0f}%)")
    return ctx, pc_t, face_seq_np


def _save_video(
    intermediates: list[np.ndarray],
    out_path: str,
    fps: int = 10,
    elevation_deg: float = 30.0,
    n_rotations: float = 1.0,
    eos_snapshots: "list | None" = None,
    boundary_snapshots: "list | None" = None,
    query_snapshots: "list | None" = None,
) -> None:
    if not intermediates:
        return
    final_verts = intermediates[-1].reshape(-1, 3).astype(np.float32)
    centroid, scale = _bounding_sphere_scale(final_verts)
    n = len(intermediates)
    azimuths = np.linspace(0.0, 360.0 * n_rotations, n, endpoint=False)
    frames = []
    for i, faces_np in enumerate(intermediates):
        R = _rotation_matrix(azimuth_deg=float(azimuths[i]), elevation_deg=elevation_deg)
        proj = final_verts @ R.T
        sx_min = float(proj[:, 0].mean()) - scale / 2
        sy_min = float(proj[:, 1].mean()) - scale / 2
        eos      = eos_snapshots[i]      if eos_snapshots      is not None else None
        boundary = boundary_snapshots[i] if boundary_snapshots is not None else None
        query    = query_snapshots[i]    if query_snapshots    is not None else None
        frames.append(np.array(_render_gen_frame(
            faces_np, sx_min, sy_min, scale, R=R,
            highlight_newest=(i > 0), eos_faces=eos,
            boundary_edges=boundary, query_edge=query,
        )))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved video ({len(frames)} frames) → {out_path}")


def _print_confidence(step_probs: np.ndarray) -> None:
    """Print quantiles of argmax probability across all generation steps."""
    if step_probs.shape[0] == 0:
        return
    flat = step_probs.flatten()
    for q in [10, 25, 50, 75, 90]:
        print(f"  p{q:2d}: {np.percentile(flat, q):.3f}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, data_cfg, training_cfg, _, _ = load_cfg(args.config)

    module = MeshTransformerModule.load_from_checkpoint(
        args.ckpt,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        map_location=device,
        strict=False,
    )
    model: MeshTransformer = module.model.eval().to(device)

    use_pc   = args.use_pc
    max_ctx  = args.ctx if args.ctx is not None else training_cfg.gen_max_ctx

    ctx, pc_t, face_seq_np = _build_context(
        args.mesh, data_cfg.num_points, device,
        mask_ratio=args.mask_ratio,
        max_ctx=max_ctx,
    )

    print(f"Conditioning: {'pc' if use_pc else 'null_latent'}")

    with torch.no_grad():
        t0 = time.perf_counter()
        result, intermediates, eos_snapshots, step_probs, boundary_snapshots, query_snapshots = model.generate(
            ctx,
            pc=pc_t if use_pc else None,
            max_steps=args.steps,
            return_intermediates=True,
            confidence_threshold=args.confidence_threshold,
        )
        gen_time = time.perf_counter() - t0

    print(f"Generated {result[0].shape[0]} faces total in {gen_time:.2f}s.")

    fixed = _fix_normals(result[0])
    print(f"Normals fixed.")

    if args.out_mesh:
        _save_mesh(fixed, args.out_mesh)

    intermediates[0][-1] = fixed.cpu().numpy()
    _save_video(
        intermediates[0], args.out, fps=args.fps,
        eos_snapshots=eos_snapshots[0],
        boundary_snapshots=boundary_snapshots[0],
        query_snapshots=query_snapshots[0],
    )

    print("Argmax probability quantiles:")
    _print_confidence(step_probs[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh generation inference")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--ckpt",   required=True,                help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--mesh",   required=True,                help="Input mesh file (.obj/.ply/etc.)")
    parser.add_argument("--out",    default="output/gen.mp4",     help="Output video path")
    parser.add_argument("--ctx",        type=int,   default=None,  help="Max context faces (default: gen_max_ctx from config)")
    parser.add_argument("--mask-ratio", type=float, default=0.0,   dest="mask_ratio",
                        help="Fraction of faces to randomly mask; context = remaining faces (default: 0.75)")
    parser.add_argument("--steps",      type=int,   default=1000,   help="Max generation steps")
    parser.add_argument("--fps",        type=int,   default=10,    help="Video FPS")
    parser.add_argument("--out-mesh",   dest="out_mesh", default=None,
                        help="Also save the final mesh as a file (e.g. output/gen.obj)")
    parser.add_argument("--use-pc",     dest="use_pc", action="store_true", default=True,
                        help="Condition on point cloud (only if trained with pc_cond_prob > 0)")
    parser.add_argument("--confidence-threshold", type=float, default=0.98, dest="confidence_threshold",
                        help="Min argmax probability to accept a prediction; lower → re-queue (default: 0.98)")
    main(parser.parse_args())
