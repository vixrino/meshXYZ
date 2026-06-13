import os
import tempfile
import threading
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

IMG_SIZE = 512
PAD = 40
N_VIZ = 8
ZOOM_FRACTION = 0.25  # fraction of the global mesh extent shown around the query token
EOS_RESIDUAL = 255    # matches constants.EOS_RESIDUAL
COLOR_THRESHOLD = 16.0  # error (in quant steps) at which prediction colour saturates to blue


def _rotation_matrix(azimuth_deg: float = 45.0, elevation_deg: float = 30.0) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    Ry = np.array([[ np.cos(az), 0, np.sin(az)],
                   [          0, 1,           0],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1,          0,           0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    return Rx @ Ry


_R = _rotation_matrix()



def _screen_bounds(
    all_verts: np.ndarray,
    R: "np.ndarray | None" = None,
) -> tuple[float, float, float]:
    if R is None:
        R = _R
    rot = all_verts @ R.T
    sx_min, sy_min = rot[:, 0].min(), rot[:, 1].min()
    sx_max, sy_max = rot[:, 0].max(), rot[:, 1].max()
    scale = max(sx_max - sx_min, sy_max - sy_min, 1e-3)
    return float(sx_min), float(sy_min), float(scale)


def _bounding_sphere_scale(all_verts: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (centroid, diameter) — rotation-invariant scale for stable turntable video."""
    centroid = all_verts.mean(axis=0)
    radius = float(np.linalg.norm(all_verts - centroid, axis=1).max())
    return centroid, max(2.0 * radius, 1e-3)


def _to_pixels(
    face: np.ndarray,
    sx_min: float, sy_min: float, scale: float,
    R: "np.ndarray | None" = None,
) -> tuple[list[tuple[int, int]], float]:
    if R is None:
        R = _R
    rot   = face @ R.T
    depth = float(rot[:, 2].mean())
    draw_w = IMG_SIZE - 2 * PAD
    pts = [
        (int((rot[i, 0] - sx_min) / scale * draw_w + PAD),
         int((1.0 - (rot[i, 1] - sy_min) / scale) * draw_w + PAD))
        for i in range(3)
    ]
    return pts, depth



def _render_one(
    t: int,
    preds: np.ndarray,
    tgts: np.ndarray,
    gt_targets: np.ndarray,
    valid_pos: np.ndarray,
    token_mask_t: np.ndarray,
) -> "Image.Image | None":
    """Render one prediction and return a PIL Image, or None if nothing to show."""
    real_pos = valid_pos[(tgts[valid_pos] < 128).any(axis=1)]
    if len(real_pos) == 0:
        return None
    _, _, global_scale = _screen_bounds(tgts[real_pos].reshape(-1, 3).astype(np.float32))

    q_rot  = tgts[t].reshape(3, 3).astype(np.float32) @ _R.T
    q_cx   = float(q_rot[:, 0].mean())
    q_cy   = float(q_rot[:, 1].mean())
    scale  = ZOOM_FRACTION * global_scale
    sx_min = q_cx - scale / 2
    sy_min = q_cy - scale / 2

    img  = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    ctx: list[tuple[float, list]] = []
    for pos in valid_pos[(valid_pos != t) & ~token_mask_t[valid_pos]]:
        pts, depth = _to_pixels(tgts[pos].reshape(3, 3).astype(np.float32), sx_min, sy_min, scale)
        ctx.append((depth, pts))
    ctx.sort(key=lambda x: x[0])

    for _, pts in ctx:
        draw.line([pts[0], pts[1], pts[2], pts[0]], fill=(130, 130, 130, 180), width=1)

    gt_pts, _ = _to_pixels(tgts[t].reshape(3, 3).astype(np.float32), sx_min, sy_min, scale)
    draw.polygon(gt_pts, fill=(44, 160, 44, 230), outline=(26, 122, 26, 255))

    pred_face = preds[t].reshape(3, 3).astype(np.float32)
    gt_face   = gt_targets[t].reshape(3, 3).astype(np.float32)

    gt_tgt_pts, _ = _to_pixels(gt_face, sx_min, sy_min, scale)
    draw.polygon(gt_tgt_pts, fill=None, outline=(200, 180, 0, 220))

    err = float(np.abs(preds[t].astype(np.float32) - gt_targets[t].astype(np.float32)).mean())
    t_  = min(err / COLOR_THRESHOLD, 1.0)
    r   = (int(214 * (1 - t_) + 31  * t_), int(39  * (1 - t_) + 119 * t_), int(40  * (1 - t_) + 180 * t_), 200)
    ro  = (int(160 * (1 - t_) + 10  * t_), int(16  * (1 - t_) + 80  * t_), int(16  * (1 - t_) + 140 * t_), 255)
    pred_pts, _ = _to_pixels(pred_face, sx_min, sy_min, scale)
    draw.polygon(pred_pts, fill=r, outline=ro)

    bg = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def _edge_to_pixels(
    ev0: tuple, ev1: tuple,
    sx_min: float, sy_min: float, scale: float,
    R: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    verts = np.array([ev0, ev1], dtype=np.float32)
    rot = verts @ R.T
    draw_w = IMG_SIZE - 2 * PAD
    def proj(r):
        return (int((r[0] - sx_min) / scale * draw_w + PAD),
                int((1.0 - (r[1] - sy_min) / scale) * draw_w + PAD))
    return proj(rot[0]), proj(rot[1])


def _render_gen_frame(
    faces_np: np.ndarray,
    sx_min: float,
    sy_min: float,
    scale: float,
    R: "np.ndarray | None" = None,
    highlight_newest: bool = True,
    eos_faces: "frozenset[int] | None" = None,
    boundary_edges: "list | None" = None,
    query_edge: "tuple | None" = None,
) -> Image.Image:
    """Render one intermediate generation state. Newest face highlighted in orange.
    Boundary edges drawn in red, query edge in green."""
    if R is None:
        R = _R
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    N = faces_np.shape[0]
    if N == 0:
        return img

    faces_3d = faces_np.reshape(N, 3, 3).astype(np.float32)
    items = []
    for idx, face in enumerate(faces_3d):
        rot = face @ R.T
        depth = float(rot[:, 2].mean())
        pts, _ = _to_pixels(face, sx_min, sy_min, scale, R=R)
        items.append((depth, pts, idx, highlight_newest and idx == N - 1))

    items.sort(key=lambda x: x[0])

    depths = [d for d, _, _, _ in items]
    dmin, dmax = min(depths), max(depths)
    drange = max(dmax - dmin, 1e-3)

    for depth, pts, idx, is_newest in items:
        t = (depth - dmin) / drange
        brightness = 0.45 + 0.55 * t
        if is_newest:
            fill    = (int(220 * brightness), int(100 * brightness), int(30  * brightness))
            outline = (int(160 * brightness), int( 60 * brightness), int(10  * brightness))
        elif eos_faces is not None and idx in eos_faces:
            fill    = (int(255 * brightness), int(220 * brightness), int(  0 * brightness))
            outline = (int(180 * brightness), int(160 * brightness), int(  0 * brightness))
        else:
            fill    = (int( 70 * brightness), int(130 * brightness), int(180 * brightness))
            outline = (int( 30 * brightness), int( 80 * brightness), int(120 * brightness))
        draw.polygon(pts, fill=fill, outline=outline)

    # draw boundary edges in red
    if boundary_edges:
        for ev0, ev1 in boundary_edges:
            p0, p1 = _edge_to_pixels(ev0, ev1, sx_min, sy_min, scale, R)
            draw.line([p0, p1], fill=(220, 40, 40), width=2)

    # draw query edge in green (on top)
    if query_edge is not None:
        ev0, ev1 = query_edge
        p0, p1 = _edge_to_pixels(ev0, ev1, sx_min, sy_min, scale, R)
        draw.line([p0, p1], fill=(30, 200, 30), width=4)

    return img


def _render_generation_video(
    intermediates: "list[np.ndarray]",
    out_dir: "str | None",
    wandb_run: "Any | None",
    fps: int = 10,
    elevation_deg: float = 30.0,
    n_rotations: float = 1.0,
    eos_snapshots: "list | None" = None,
    boundary_snapshots: "list | None" = None,
    query_snapshots: "list | None" = None,
) -> None:
    if not intermediates:
        return

    import imageio
    import wandb

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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)

    if wandb_run is not None:
        wandb_run.log({"viz/generation": wandb.Video(path, fps=fps, format="mp4")})
    elif out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        os.replace(path, os.path.join(out_dir, "generation_latest.mp4"))
        return

    os.remove(path)


def save_generation_video(
    intermediates: "list[np.ndarray]",
    out_dir: "str | None",
    wandb_run: "Any | None" = None,
    fps: int = 10,
    eos_snapshots: "list | None" = None,
    boundary_snapshots: "list | None" = None,
    query_snapshots: "list | None" = None,
) -> None:
    threading.Thread(
        target=_render_generation_video,
        args=(intermediates, out_dir, wandb_run, fps),
        kwargs={
            "eos_snapshots":      eos_snapshots,
            "boundary_snapshots": boundary_snapshots,
            "query_snapshots":    query_snapshots,
        },
        daemon=True,
    ).start()


def _canonical_face_np(verts: np.ndarray) -> np.ndarray:
    """Return (9,) canonical face from (3, 3) int vertices (ZYX lex-min first)."""
    keys = [(int(v[2]), int(v[1]), int(v[0])) for v in verts]
    min_idx = keys.index(min(keys))
    return np.roll(verts, -min_idx, axis=0).reshape(9)


def _reconstruct_faces(
    raw_preds: np.ndarray,    # (N, 9) argmax logits; positions 0-5 are zeros, 6-8 are pred vertex
    raw_gt: np.ndarray,       # (N, 9) targets; positions 0-5 are PAD=-1, 6-8 are new vertex or EOS
    query_edges: np.ndarray,  # (N, 6) ev0, ev1
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct full canonical (N, 9) faces for viz from edge + predicted/gt vertex."""
    N = query_edges.shape[0]
    pred_faces = np.zeros((N, 9), dtype=raw_preds.dtype)
    gt_faces   = raw_gt.copy()  # keep EOS/PAD structure for EOS detection; overwrite non-EOS below

    for i in range(N):
        gt_v = raw_gt[i, 6:]
        if (gt_v == EOS_RESIDUAL).any():
            # EOS face — mark pred as EOS too so EOS detection works
            pred_faces[i, 6:] = EOS_RESIDUAL
            continue
        ev0, ev1 = query_edges[i, :3], query_edges[i, 3:]
        pred_faces[i] = _canonical_face_np(np.stack([ev0, ev1, raw_preds[i, 6:]]))
        gt_faces[i]   = _canonical_face_np(np.stack([ev0, ev1, gt_v]))

    return pred_faces, gt_faces


def _render_all(
    preds: np.ndarray,
    tgts: np.ndarray,
    gt_targets: np.ndarray,
    valid_pos: np.ndarray,
    token_mask: np.ndarray,
    out_dir: "str | None",
    global_step: int,
    n_viz: int,
    wandb_run: "Any | None",
) -> None:
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # Use .any() so EOS detection works for both modes:
    #   - use_edge_cond=False: all 9 coords are EOS_RESIDUAL → any() = True ✓
    #   - use_edge_cond=True:  positions 6-8 are EOS_RESIDUAL (after reconstruction) → any() = True ✓
    masked_pos = valid_pos[~(gt_targets[valid_pos] == EOS_RESIDUAL).any(axis=1)]
    if len(masked_pos) == 0:
        return

    # Error only over non-PAD positions (positions with target != -1).
    p    = preds[masked_pos].astype(np.float32)
    g    = gt_targets[masked_pos].astype(np.float32)
    pad  = gt_targets[masked_pos] == -1
    diff = np.abs(p - g)
    diff[pad] = 0.0
    n_valid = (~pad).sum(axis=1).clip(min=1)
    errors  = diff.sum(axis=1) / n_valid

    top_idx = np.argsort(errors)[::-1][:n_viz]

    wandb_images = []
    for rank, i in enumerate(top_idx):
        t   = masked_pos[i]
        err = float(errors[i])
        pil = _render_one(t, preds, tgts, gt_targets, valid_pos, token_mask[t])
        if pil is None:
            continue
        if wandb_run is not None:
            import wandb
            wandb_images.append(wandb.Image(pil, caption=f"rank{rank} t={t} err={err:.1f}"))
        else:
            fname = f"step_{global_step:07d}_rank{rank:02d}_err{err:.1f}_t{t:04d}.png"
            pil.save(os.path.join(out_dir, fname))

    if wandb_images:
        wandb_run.log({"viz/predictions": wandb_images}, step=global_step)


def save_prediction_grid(
    logits: torch.Tensor,
    faces: torch.Tensor,
    gt_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    token_mask: torch.Tensor,
    out_dir: "str | None",
    global_step: int,
    wandb_run: "Any | None" = None,
    n_viz: int = N_VIZ,
    query_edges: "torch.Tensor | None" = None,
) -> None:
    raw_preds = logits[0].argmax(dim=-1).cpu().numpy()   # (N, 9)
    tgts      = faces[0].cpu().numpy()
    raw_gt    = gt_targets[0].cpu().numpy()               # (N, 9)
    valid_pos = np.where(valid_mask[0].cpu().numpy())[0]
    tmask     = token_mask[0].cpu().numpy()
    if len(valid_pos) == 0:
        return

    if query_edges is not None:
        # edge_cond mode: positions 0-5 of preds/gt are dummy — reconstruct full canonical faces.
        qe = query_edges[0].cpu().numpy()   # (N, 6)
        preds, gt_tgts = _reconstruct_faces(raw_preds, raw_gt, qe)
    else:
        preds, gt_tgts = raw_preds, raw_gt

    threading.Thread(
        target=_render_all,
        args=(preds, tgts, gt_tgts, valid_pos, tmask, out_dir, global_step, n_viz, wandb_run),
        daemon=True,
    ).start()
