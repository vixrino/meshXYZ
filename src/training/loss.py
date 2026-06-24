import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..constants import EOS_RESIDUAL, PAD_TARGET, QUANT_MAX, TRI_PAD


def valid_row_mask(
    faces: Int[Tensor, "batch faces n_face_tokens"],
    pad_value: int = 128,
) -> Bool[Tensor, "batch faces"]:
    """Returns True for real face rows, False for collate_fn padding rows.

    The collate_fn fills padding rows with EOS_COORD (128).  Since TRI_PAD=129
    and all real coordinates are in [0, QUANT_MAX=127], no real face row can
    ever have all tokens equal to 128.  Works for both 9-token and 12-token faces.
    """
    return ~(faces == pad_value).all(dim=-1)


def reconstruction_loss(
    logits: Float[Tensor, "batch faces n_coords vocab"],
    targets: Int[Tensor, "batch faces n_coords"],
    faces: Int[Tensor, "batch faces 9"] | None = None,
    pad_value: int = -1,
    eos_weight: float = 1.0,
) -> Float[Tensor, ""]:
    """Cross-entropy loss over coordinate predictions.

    targets == pad_value (-1) → face is fully excluded from the loss.
    targets == EOS_RESIDUAL (255) → face is trained to predict EOS.

    eos_weight : float
        Per-class weight applied to the EOS_RESIDUAL token in the cross-entropy.
        EOS targets are extremely rare (~1 face per mesh → <1% of tokens), so in a
        shared softmax the optimizer trades EOS away to sharpen coord predictions
        and loss_eos drifts upward.  eos_weight > 1.0 upweights the EOS class to
        counteract this.  Default 1.0 reproduces the unweighted loss exactly
        (the weight tensor is all-ones), so triangle runs are bit-identical.
    """
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value
    vocab_size   = logits.shape[-1]
    logits_flat  = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Only build a weight tensor when needed; eos_weight=1.0 keeps the exact
    # same code path (weight=None) as before for backward compatibility.
    weight = None
    if eos_weight != 1.0:
        weight = torch.ones(vocab_size, device=logits.device, dtype=logits.dtype)
        weight[EOS_RESIDUAL] = eos_weight

    # ignore_index handles empty valid sets inside the CUDA kernel — no CPU-GPU sync.
    loss    = F.cross_entropy(logits_flat, targets_flat, weight=weight,
                              ignore_index=pad_value, reduction="sum")
    n_valid = (targets_flat != pad_value).sum().clamp(min=1)
    return loss / n_valid


def compute_metrics(
    logits: Float[Tensor, "batch faces n_coords vocab"],
    targets: Int[Tensor, "batch faces n_coords"],
    faces: Int[Tensor, "batch faces 9"] | None = None,
    pad_value: int = -1,
) -> dict[str, Float[Tensor, ""]]:
    """
    Returns:
        coord_acc:  fraction of individual non-PAD coordinates predicted exactly right (EOS excluded)
        coord_mae:  mean absolute error in quantization steps (EOS faces excluded)
        face_acc:   fraction of non-EOS faces where all non-PAD coords are correct
        eos_acc:    fraction of EOS faces where model correctly predicts EOS on non-PAD coords
    Padded faces (all targets == pad_value) and PAD_TARGET positions within a face are excluded.

    Compatible with both modes:
      - use_edge_cond=False: all 9 target positions are real coords or EOS_RESIDUAL.
      - use_edge_cond=True:  positions 0-5 are pad_value (edge, not predicted);
                             positions 6-8 are the new-vertex target or EOS_RESIDUAL.
    """
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value

    # Face-level validity: a face is valid if not ALL positions are pad_value.
    valid   = ~(targets == pad_value).all(dim=-1)          # (B, F)
    n_valid = valid.float().sum().clamp(min=1)

    preds = logits.argmax(-1)                              # (B, F, n_coords)

    # EOS detection: any non-PAD coord equals EOS_RESIDUAL (works for both modes).
    non_pad = targets != pad_value                         # (B, F, n_coords)
    eos     = (non_pad & (targets == EOS_RESIDUAL)).any(dim=-1)  # (B, F)

    coord_valid  = valid & ~eos                            # (B, F) — real coordinate rows
    n_coord      = coord_valid.float().sum().clamp(min=1)
    n_eos        = (valid & eos).float().sum().clamp(min=1)

    # Per-position mask: valid face, not EOS, and not a PAD position.
    coord_pos_valid = coord_valid.unsqueeze(-1) & non_pad  # (B, F, n_coords)
    n_coord_pos     = coord_pos_valid.float().sum().clamp(min=1)

    coord_correct = (preds == targets) & coord_pos_valid   # (B, F, n_coords)
    coord_acc     = coord_correct.float().sum() / n_coord_pos

    mae_mask  = coord_pos_valid & (preds != EOS_RESIDUAL) & (targets != EOS_RESIDUAL)
    n_mae     = mae_mask.float().sum().clamp(min=1)
    coord_mae = ((preds - targets).abs().float() * mae_mask).sum() / n_mae

    # face_acc: all non-PAD coords correct (PAD positions treated as "don't care").
    face_acc  = (coord_correct | ~coord_pos_valid).all(dim=-1)
    face_acc  = (face_acc & coord_valid).float().sum() / n_coord

    eos_acc   = ((preds == EOS_RESIDUAL).any(dim=-1) & valid & eos).float().sum() / n_eos
    perc_eos  = (valid & eos).float().sum() / n_valid

    return {"coord_acc": coord_acc, "coord_mae": coord_mae, "face_acc": face_acc, "eos_acc": eos_acc, "perc_eos": perc_eos}


def decompose_loss(
    logits: Float[Tensor, "batch faces n_face_tokens vocab"],
    targets: Int[Tensor, "batch faces n_face_tokens"],
    faces: "Int[Tensor, 'batch faces n_face_tokens'] | None" = None,
    pad_value: int = PAD_TARGET,
) -> dict[str, "Float[Tensor, '']"]:
    """Split reconstruction_loss into three disjoint components by target token value.

    Decomposition (ranges are disjoint: TRI_PAD=129 > QUANT_MAX=127, EOS_RESIDUAL=255
    > TRI_PAD, and the three masks are mutually exclusive):

        target == TRI_PAD (129)          → loss_tri_pad   (whole-face trailing pad AND
                                                            edge-cond slot-2 triangle marker)
        target in [0, QUANT_MAX] (0-127) → loss_coord     (coord slots of tri and quad faces)
        target == EOS_RESIDUAL (255)     → loss_eos       (slot-1 STOP: edge has no neighbor)
        target == PAD_TARGET (-1)        → ignored        (collate padding rows or masked positions)

    loss_tri_pad now also captures the edge-cond slot-2 triangle marker (TRI_PAD): a
    rising loss_tri_pad in edge-cond mode is the direct signal that the model is
    failing to mark triangle neighbors (which would collapse quads to triangles).

    Weighted by their token counts, the three components sum to reconstruction_loss.
    Each component uses mean-over-valid-tokens reduction, so the values are
    comparable in magnitude even when one category has very few positions.

    Expected training dynamics:
        loss_tri_pad: drops fast in whole-face mode (deterministic trailing pad); in
                      edge-cond mode it tracks how well triangle neighbors are marked.
        loss_coord:   drops more slowly as the model learns geometry
        loss_eos:     typically low and stable (few EOS positions)
    """
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value

    vocab_size  = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    tgt_flat    = targets.reshape(-1)

    def _ce(mask: Bool[Tensor, "n"]) -> "Float[Tensor, '']":
        """Cross-entropy restricted to positions where mask is True."""
        t = tgt_flat.clone()
        t[~mask] = pad_value
        return (
            F.cross_entropy(logits_flat, t, ignore_index=pad_value, reduction="sum")
            / mask.sum().clamp(min=1)
        )

    return {
        "loss_tri_pad": _ce(tgt_flat == TRI_PAD),
        "loss_coord":   _ce((tgt_flat >= 0) & (tgt_flat <= QUANT_MAX)),
        "loss_eos":     _ce(tgt_flat == EOS_RESIDUAL),
    }


def face_type_acc(
    logits: Float[Tensor, "batch faces n_face_tokens vocab"],
    targets: Int[Tensor, "batch faces n_face_tokens"],
    faces: "Int[Tensor, 'batch faces n_face_tokens'] | None" = None,
    pad_value: int = PAD_TARGET,
) -> "Float[Tensor, '']":
    """Fraction of real non-EOS faces where the model correctly predicts the face type.

    Face type is determined from position 9 of the *target* (the trailing pad slot;
    padding moved to the END of the block):
        target[9] == TRI_PAD (129)       → triangle target; correct if argmax(logits[9]) == TRI_PAD
        target[9] in [0, QUANT_MAX=127]  → quad target;    correct if argmax(logits[9]) <= QUANT_MAX

    EOS faces (any target position is EOS_RESIDUAL) and padding rows are excluded.

    This is the primary sanity check that the model has learned the topology
    distinction: TRI_PAD at position 9 means "the next face is a triangle; fill
    positions 9-11 with pad tokens after the 3 vertices".
    """
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value

    tgt_type = targets[:, :, 9]                                  # (B, F) — trailing pad slot
    valid    = tgt_type != pad_value                            # non-padding rows

    is_eos         = (targets == EOS_RESIDUAL).any(dim=-1) & valid
    is_tri_target  = (tgt_type == TRI_PAD)                    & valid & ~is_eos
    is_quad_target = (tgt_type >= 0) & (tgt_type <= QUANT_MAX) & valid & ~is_eos

    pred_type = logits[:, :, 9].argmax(-1)                      # (B, F) argmax at position 9

    tri_correct  = (pred_type == TRI_PAD)   & is_tri_target
    quad_correct = (pred_type <= QUANT_MAX) & is_quad_target

    n_typed   = (is_tri_target | is_quad_target).float().sum().clamp(min=1)
    n_correct = (tri_correct   | quad_correct  ).float().sum()
    return n_correct / n_typed


def edge_face_type_acc(
    logits: Float[Tensor, "batch faces n_face_tokens vocab"],
    targets: Int[Tensor, "batch faces n_face_tokens"],
    faces: "Int[Tensor, 'batch faces n_face_tokens'] | None" = None,
    pad_value: int = PAD_TARGET,
) -> "Float[Tensor, '']":
    """Edge-cond (12-token) monitor: triangle-neighbor vs quad-neighbor.

    In edge-cond mode the neighbor topology is signalled by slot 2 (positions
    9-11) of the target:
        slot2 == TRI_PAD (129)         → triangle neighbor (2nd vertex is padding)
        slot2 in [0, QUANT_MAX]        → quad neighbor (real 2nd vertex)
    Only faces that have a real neighbor are counted (slot 1, positions 6-8, is a
    real vertex, not the EOS_RESIDUAL 'STOP' marker and not padding).

    EOS_RESIDUAL is only ever the slot-1 STOP; a triangle neighbor is detected by
    TRI_PAD in slot 2 — never by EOS_RESIDUAL.

    The metric is the fraction of those faces where the model predicts the correct
    topology at position 9 (argmax == TRI_PAD ⇔ triangle).  This is the primary
    guard-rail: if the model drifts toward emitting TRI_PAD in slot 2, quads
    silently collapse to triangles and this metric drops well before the meshes
    look obviously wrong.

    Returns a no-op 1.0 when the target layout is not 12-token (e.g. tri-only).
    """
    if targets.shape[-1] != 12:
        return logits.new_tensor(1.0)
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value

    slot1 = targets[:, :, 6:9]                                   # (B, F, 3)
    slot2 = targets[:, :, 9:12]
    has_nb = (slot1 != pad_value).all(-1) & ~(slot1 == EOS_RESIDUAL).any(-1)
    is_tri_nb  = has_nb & (slot2 == TRI_PAD).any(-1)
    is_quad_nb = has_nb & (slot2 >= 0).all(-1) & (slot2 <= QUANT_MAX).all(-1)

    pred_tri = logits[:, :, 9].argmax(-1) == TRI_PAD            # (B, F)
    correct  = (pred_tri & is_tri_nb) | (~pred_tri & is_quad_nb)

    n = (is_tri_nb | is_quad_nb).float().sum().clamp(min=1)
    return correct.float().sum() / n


def edge_face_type_recall(
    logits: Float[Tensor, "batch faces n_face_tokens vocab"],
    targets: Int[Tensor, "batch faces n_face_tokens"],
    faces: "Int[Tensor, 'batch faces n_face_tokens'] | None" = None,
    pad_value: int = PAD_TARGET,
) -> "dict[str, Float[Tensor, '']]":
    """Per-class breakdown of edge_face_type_acc (diagnostic, no behaviour change).

    Kept ALONGSIDE edge_face_type_acc so a drop can be attributed to the right
    failure mode (quad_recall should sit near ~1.0; a falling tri_recall means
    triangle neighbors are being missed):

        tri_recall  : among triangle neighbours (slot2 == TRI_PAD), fraction the
                      model predicts as triangle (argmax position 9 == TRI_PAD).
        quad_recall : among quad neighbours (slot2 == real coord), fraction the
                      model predicts as quad (argmax position 9 != TRI_PAD).
        perc_tri_nb : fraction of typed neighbours that are triangles — the live
                      class balance (tiny on native-quad meshes ⇒ tri_recall is
                      high-variance).

    Returns a no-op (1.0 / 1.0 / 0.0) for non-12-token layouts.
    """
    if targets.shape[-1] != 12:
        one, zero = logits.new_tensor(1.0), logits.new_tensor(0.0)
        return {"tri_recall": one, "quad_recall": one, "perc_tri_nb": zero}
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value

    slot1 = targets[:, :, 6:9]
    slot2 = targets[:, :, 9:12]
    has_nb = (slot1 != pad_value).all(-1) & ~(slot1 == EOS_RESIDUAL).any(-1)
    is_tri_nb  = has_nb & (slot2 == TRI_PAD).any(-1)
    is_quad_nb = has_nb & (slot2 >= 0).all(-1) & (slot2 <= QUANT_MAX).all(-1)

    pred_tri = logits[:, :, 9].argmax(-1) == TRI_PAD

    n_tri  = is_tri_nb.float().sum()
    n_quad = is_quad_nb.float().sum()
    n_typed = (n_tri + n_quad).clamp(min=1)

    tri_recall  = (pred_tri & is_tri_nb).float().sum()  / n_tri.clamp(min=1)
    quad_recall = (~pred_tri & is_quad_nb).float().sum() / n_quad.clamp(min=1)
    perc_tri_nb = n_tri / n_typed
    return {"tri_recall": tri_recall, "quad_recall": quad_recall, "perc_tri_nb": perc_tri_nb}
