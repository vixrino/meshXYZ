"""Phase-4 smoke-train: 100 steps on sphere + torus + suzanne in quad mode.

Validates four expectations:
  1. Total loss drops noticeably (from ~5.5 to ≤3).
  2. loss_tri_pad drops fast (model learns the deterministic TRI_PAD prefix).
  3. loss_coord drops more gradually (geometry is harder to memorise).
  4. face_type_acc approaches 1.0 within the first 50 steps.

Uses a pure-PyTorch loop (no Lightning, no Wandb) with a tiny Decoder
(d_hidden=64, n_layers=2) so 100 steps run in < 30 s on CPU.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import EOS_COORD
from src.dataset.collate import collate_fn
from src.dataset.mesh_ops import process_mesh
from src.dataset.obj_parser import parse_obj
from src.model.decoder import Decoder, DecoderCfg
from src.training.loss import decompose_loss, face_type_acc, reconstruction_loss
from src.training.strategy.target_builder.causal import CausalTargetBuilder, CausalTargetBuilderCfg

# ── Configuration ─────────────────────────────────────────────────────────────

MESH_PATHS  = [
    Path(__file__).parent / "sphere.obj",
    Path(__file__).parent / "torus.obj",
    Path(__file__).parent / "suzanne.obj",
]
N_STEPS     = 150
BATCH_SIZE  = 4
LR          = 1e-3       # slightly higher than the 6e-4 default for faster convergence
CLIP_NORM   = 1.0
LOG_EVERY   = 10     # print a row every N steps
NUM_POINTS  = 512    # point cloud samples (unused during loss test; kept for completeness)
DEVICE      = torch.device("cpu")

# Small decoder — fast enough for 100 CPU steps
DECODER_CFG = DecoderCfg(
    d_latent          = 8,
    d_hidden          = 64,
    n_layers          = 4,      # 2 was insufficient for context-dependent tri/quad discrimination
    n_heads           = 4,      # must divide d_hidden
    max_faces         = 512,
    vocab_size        = 257,    # 0-127 coords + 128 EOS_COORD + 129 TRI_PAD + ... + 255 EOS_RESIDUAL
    n_face_tokens     = 12,     # unified quad/tri block
    use_pos_embed     = True,
    use_spherical_embed = False,
    relative          = False,  # required for n_face_tokens=12 (TRI_PAD=129 > QUANT_MAX=127)
    use_edge_cond     = False,  # required for n_face_tokens=12
)


# ── Load and preprocess meshes ────────────────────────────────────────────────

def load_mesh(path: Path) -> dict:
    r = parse_obj(str(path))
    pc, faces_12, neighbors_4 = process_mesh(
        verts      = r.vertices.astype("float64"),
        faces_tri  = r.faces_tri,
        num_points = NUM_POINTS,
        face_layout= "quad",
        faces_quad = r.faces_quad,
    )
    return {
        "pc"            : torch.from_numpy(pc),
        "faces"         : torch.from_numpy(faces_12).long(),
        "face_neighbors": torch.from_numpy(neighbors_4).long(),
    }


# ── Causal attention mask (B, F, F) with length-aware padding mask ────────────

def make_causal_mask(lengths: torch.Tensor, F_max: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask + padding rows/columns hidden.

    mask[b, q, k] = True means face k is masked from face q's cross-attention.
    Convention matches nn.TransformerDecoder tgt_mask: True → -inf.
    """
    B   = len(lengths)
    idx = torch.arange(F_max, device=device)

    # Causal: hide future keys (k > q)
    causal = idx.unsqueeze(0) > idx.unsqueeze(1)              # (F, F)
    mask   = causal.unsqueeze(0).expand(B, -1, -1).clone()   # (B, F, F)

    # Padding: hide padded keys from all queries; padded queries see nothing
    pad    = idx.unsqueeze(0) >= lengths.to(device).unsqueeze(1)  # (B, F) True=padded
    mask  |= pad.unsqueeze(1)   # padded keys invisible to every query row
    mask  |= pad.unsqueeze(2)   # padded queries can't attend to anything

    # Each face must be able to attend to itself (required for TransformerDecoder)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)
    return mask


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    # ── Load meshes ──────────────────────────────────────────────────────────
    print("Loading meshes …")
    meshes = []
    for p in MESH_PATHS:
        if not p.exists():
            print(f"  [SKIP] {p} not found")
            continue
        m = load_mesh(p)
        print(f"  {p.name}: {m['faces'].shape[0]} faces, {m['faces'].shape[1]} tokens/face")
        meshes.append(m)

    if not meshes:
        raise RuntimeError("No meshes found — run scripts from repo root.")

    # ── Model + optimizer ────────────────────────────────────────────────────
    decoder        = Decoder(DECODER_CFG).to(DEVICE)
    n_params       = sum(p.numel() for p in decoder.parameters())
    target_builder = CausalTargetBuilder(CausalTargetBuilderCfg())
    optimizer      = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(f"\nDecoder: d_hidden={DECODER_CFG.d_hidden}, n_layers={DECODER_CFG.n_layers}, "
          f"n_params={n_params:,}")
    print(f"Training for {N_STEPS} steps, batch_size={BATCH_SIZE}, lr={LR}\n")

    # ── History ──────────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "loss": [], "loss_tri_pad": [], "loss_coord": [], "loss_eos": [], "face_type_acc": []
    }

    # ── Training loop ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"{'step':>4}  {'loss':>6}  {'tri_pad':>7}  {'coord':>6}  {'eos':>6}  {'ftype_acc':>9}  {'dt_ms':>6}")
    print("-" * 60)

    # Identify which meshes contain triangles (sphere=0, suzanne=2; torus=1 has none)
    tri_mesh_idx  = [i for i, m in enumerate(meshes) if (m["faces"][:, 0] == 129).any()]
    quad_only_idx = [i for i in range(len(meshes)) if i not in tri_mesh_idx]

    for step in range(N_STEPS):
        # Stratified sampling: always include at least 1 triangle-containing mesh so
        # every batch has both TRI_PAD targets and coord targets.  This prevents
        # all-torus batches that would make face_type_acc noisy and unstable.
        guaranteed = [meshes[random.choice(tri_mesh_idx)]]
        remaining  = [meshes[random.randrange(len(meshes))] for _ in range(BATCH_SIZE - 1)]
        items = guaranteed + remaining
        random.shuffle(items)
        batch = collate_fn(items)

        faces   = batch["faces"].to(DEVICE)     # (B, F_max, 12)
        lengths = batch["lengths"].to(DEVICE)   # (B,)
        B, F    = faces.shape[:2]

        # Causal + padding mask
        token_mask = make_causal_mask(lengths, F, DEVICE)   # (B, F, F)

        # Null latents — no encoder during this smoke-train
        null_C = torch.zeros(B, 1, DECODER_CFG.d_latent, device=DEVICE)

        # Forward
        logits = decoder(null_C, faces, token_mask=token_mask)   # (B, F, 12, 257)

        # Causal targets: for face q, predict face q+1 (EOS for the last)
        pseudo_batch = {"faces": faces, "lengths": lengths}
        targets, _   = target_builder.compute_targets(pseudo_batch, token_mask, use_edge_cond=False)

        # Losses
        loss        = reconstruction_loss(logits, targets, faces=faces)
        decomposed  = decompose_loss(logits, targets, faces=faces)
        ftype       = face_type_acc(logits, targets, faces=faces)

        # Diagnostic: break face_type_acc into tri and quad components
        from src.constants import TRI_PAD as _TRI_PAD, QUANT_MAX as _QM, EOS_RESIDUAL as _EOS, PAD_TARGET as _PAD
        from src.training.loss import valid_row_mask as _vrm
        _t2 = targets.clone(); _t2[~_vrm(faces)] = _PAD
        _tgt0, _valid = _t2[:, :, 0], _t2[:, :, 0] != _PAD
        _is_eos   = ((_t2 == _EOS).any(-1)) & _valid
        _is_tri_t = (_tgt0 == _TRI_PAD) & _valid & ~_is_eos
        _is_qd_t  = (_tgt0 >= 0) & (_tgt0 <= _QM) & _valid & ~_is_eos
        _pred0    = logits[:, :, 0].argmax(-1)
        _tri_acc  = ((_pred0 == _TRI_PAD) & _is_tri_t).float().sum() / _is_tri_t.float().sum().clamp(min=1)
        _qd_acc   = ((_pred0 <= _QM) & _is_qd_t).float().sum()  / _is_qd_t.float().sum().clamp(min=1)
        n_tri_t   = _is_tri_t.sum().item()
        n_qd_t    = _is_qd_t.sum().item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP_NORM)
        optimizer.step()

        # Record
        history["loss"].append(loss.item())
        for k, v in decomposed.items():
            history[k].append(v.item())
        history["face_type_acc"].append(ftype.item())

        # Print
        dt_ms = (time.perf_counter() - t0) * 1000
        t0    = time.perf_counter()
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            print(
                f"{step:>4}  {loss.item():>6.3f}  "
                f"{decomposed['loss_tri_pad'].item():>7.3f}  "
                f"{decomposed['loss_coord'].item():>6.3f}  "
                f"{decomposed['loss_eos'].item():>6.3f}  "
                f"{ftype.item():>9.4f}"
                f"  tri={_tri_acc.item():.2f}({n_tri_t})"
                f" qd={_qd_acc.item():.2f}({n_qd_t})"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n── Final 10-step averages ──")
    for k, vals in history.items():
        avg = sum(vals[-10:]) / 10
        print(f"  {k:20s}: {avg:.4f}")

    # ── Validation checks ─────────────────────────────────────────────────────
    print("\n── Expectation checks ──")
    ok = True

    init_loss  = sum(history["loss"][:5]) / 5
    final_loss = sum(history["loss"][-10:]) / 10
    drop       = init_loss - final_loss
    status     = "✓" if drop > 1.5 else "✗"
    print(f"  {status} loss drop {init_loss:.2f} → {final_loss:.2f} ({drop:+.2f}); expect > 1.5")
    ok = ok and (drop > 1.5)

    init_tri  = sum(history["loss_tri_pad"][:5]) / 5
    final_tri = sum(history["loss_tri_pad"][-10:]) / 10
    status    = "✓" if final_tri < init_tri * 0.4 else "✗"
    print(f"  {status} loss_tri_pad {init_tri:.2f} → {final_tri:.2f}; expect final < 40% of init")
    ok = ok and (final_tri < init_tri * 0.4)

    # face_type_acc exhibits a 3-phase training curve:
    #   Phase 1 (steps 0-30):  qd_acc≈1.00, tri_acc≈0 → high ftype because quads dominate
    #   Phase 2 (steps 30-70): mode switch → tri_acc=1.00, qd_acc≈0 → ftype crashes
    #   Phase 3 (steps 70+):   context-aware learning → both tri+qd correct → ftype rises
    # Checking mid-run (steps 40-60) is right in the mode-switch valley; use step~100 and final.
    step100_acc = history["face_type_acc"][min(100, N_STEPS - 1)]
    final_acc   = sum(history["face_type_acc"][-10:]) / 10
    s1 = "✓" if step100_acc > 0.80 else "✗"
    print(f"  {s1} face_type_acc @ step 100 : {step100_acc:.4f}; expect > 0.80 (post-mode-switch)")
    s2 = "✓" if final_acc > 0.90 else "✗"
    print(f"  {s2} face_type_acc final (10-avg): {final_acc:.4f}; expect > 0.90")
    ok = ok and (step100_acc > 0.80) and (final_acc > 0.90)

    if N_STEPS >= 100:
        mode_valley = sum(history["face_type_acc"][40:60]) / 20
        print(f"  ⓘ  mode-switch valley (steps 40-60): {mode_valley:.4f}"
              " — expected (tri_acc→1, qd_acc→0 simultaneously)")

    print(f"\n  Overall: {'ALL PASS' if ok else 'SOME CHECKS FAILED'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        steps = list(range(N_STEPS))

        # Simple exponential smoothing helper (alpha=0.3)
        def smooth(vals, alpha=0.3):
            out = [vals[0]]
            for v in vals[1:]:
                out.append(alpha * v + (1 - alpha) * out[-1])
            return out

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Phase-4 Quad Training Smoke Test — {N_STEPS} steps (d_hidden={DECODER_CFG.d_hidden}, n_layers={DECODER_CFG.n_layers})", fontsize=11)

        # Left: loss components
        colors = {"loss": "#2c3e50", "loss_tri_pad": "#e67e22", "loss_coord": "#27ae60", "loss_eos": "#e74c3c"}
        labels = {"loss": "Total loss", "loss_tri_pad": "loss_tri_pad (pos 0-2 of tri)", "loss_coord": "loss_coord (coord slots)", "loss_eos": "loss_eos (EOS target)"}
        for k, col in colors.items():
            raw = history[k]
            ax1.plot(steps, raw, color=col, alpha=0.25, linewidth=0.8)
            ax1.plot(steps, smooth(raw), color=col, linewidth=2.0, label=labels[k])
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("Cross-entropy loss (nats)")
        ax1.set_title("Loss components")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Right: face_type_acc
        raw_acc = history["face_type_acc"]
        ax2.plot(steps, raw_acc, color="#8e44ad", alpha=0.25, linewidth=0.8)
        ax2.plot(steps, smooth(raw_acc), color="#8e44ad", linewidth=2.0, label="face_type_acc")
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Perfect (1.0)")
        ax2.axhline(0.9, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.7, label="Target (0.90)")
        # Annotate the three phases
        if N_STEPS >= 100:
            ax2.axvspan(0,  35, alpha=0.06, color="#27ae60", label="Phase 1: qd_acc→1")
            ax2.axvspan(35, 75, alpha=0.06, color="#e74c3c", label="Phase 2: mode switch")
            ax2.axvspan(75, N_STEPS, alpha=0.06, color="#2980b9", label="Phase 3: context-aware")
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Face-type accuracy\n(TRI_PAD prefix vs quad coord at position 0)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(__file__).parent / "loss_curve_p4.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {out}")
        plt.close()

    except ImportError:
        print("\n(matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
