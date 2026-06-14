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

    Decomposition (ranges are disjoint because TRI_PAD=129 > QUANT_MAX=127 and
    EOS_RESIDUAL=255 > TRI_PAD, so the three masks never overlap):

        target == TRI_PAD (129)          → loss_tri_pad   (positions 0-2 of tri-target faces)
        target in [0, QUANT_MAX] (0-127) → loss_coord     (coord slots of both tri and quad faces)
        target == EOS_RESIDUAL (255)     → loss_eos       (faces without a next neighbor)
        target == PAD_TARGET (-1)        → ignored        (collate padding rows or masked positions)

    Weighted by their token counts, the three components sum to reconstruction_loss.
    Each component uses mean-over-valid-tokens reduction, so the values are
    comparable in magnitude even when one category has very few positions.

    Expected training dynamics:
        loss_tri_pad: drops fast (model learns the deterministic TRI_PAD prefix in ~20 steps)
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

    Face type is determined from position 0 of the *target* (the face to predict):
        target[0] == TRI_PAD (129)       → triangle target; correct if argmax(logits[0]) == TRI_PAD
        target[0] in [0, QUANT_MAX=127]  → quad target;    correct if argmax(logits[0]) <= QUANT_MAX

    EOS faces (any target position is EOS_RESIDUAL) and padding rows are excluded.

    This is the primary sanity check that the model has learned the topology
    distinction: TRI_PAD at position 0 means "the next face is a triangle; fill
    positions 0-2 with pad tokens before predicting the 3 vertices".
    """
    if faces is not None:
        targets = targets.clone()
        targets[~valid_row_mask(faces)] = pad_value

    tgt0  = targets[:, :, 0]                                    # (B, F)
    valid = tgt0 != pad_value                                    # non-padding rows

    is_eos         = (targets == EOS_RESIDUAL).any(dim=-1) & valid
    is_tri_target  = (tgt0 == TRI_PAD)                & valid & ~is_eos
    is_quad_target = (tgt0 >= 0) & (tgt0 <= QUANT_MAX) & valid & ~is_eos

    pred0 = logits[:, :, 0].argmax(-1)                          # (B, F) argmax at position 0

    tri_correct  = (pred0 == TRI_PAD)   & is_tri_target
    quad_correct = (pred0 <= QUANT_MAX) & is_quad_target

    n_typed   = (is_tri_target | is_quad_target).float().sum().clamp(min=1)
    n_correct = (tri_correct   | quad_correct  ).float().sum()
    return n_correct / n_typed
