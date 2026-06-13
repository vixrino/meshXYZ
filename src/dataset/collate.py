"""collate_fn for MeshDataset batches.

Depends on torch only — no trimesh, no lightning.  Kept separate so it can be
imported in unit tests without pulling in the full training environment.
"""
import torch
from torch.nn.utils.rnn import pad_sequence

from ..constants import EOS_COORD


def collate_fn(batch: list[dict]) -> dict:
    """Pad a list of per-mesh dicts into batched tensors.

    Input dict keys per item:
        "pc"             : (num_points, 3) float tensor
        "faces"          : (F, coords) long tensor — coords=9 (tri) or 12 (quad)
        "face_neighbors" : (F, slots)  long tensor — slots=3 (tri) or 4 (quad)

    Padding:
        faces          — padded to F_max with EOS_COORD (128); padding rows are
                         all-EOS_COORD and detected by valid_row_mask().
        face_neighbors — padded to F_max with -1.

    Returns a dict with keys:
        "pc"             : (B, num_points, 3)
        "faces"          : (B, F_max, coords)
        "lengths"        : (B,) long — true face count per item
        "face_neighbors" : (B, F_max, slots)
    """
    pcs        = torch.stack([b["pc"] for b in batch], dim=0)
    faces_list = [b["faces"] for b in batch]

    padded_faces = pad_sequence(faces_list, batch_first=True, padding_value=EOS_COORD)
    lengths      = torch.tensor([f.shape[0] for f in faces_list], dtype=torch.long)

    face_neighbors_list = [b["face_neighbors"] for b in batch]
    max_F   = max(s.shape[0] for s in face_neighbors_list)
    B       = len(batch)
    n_slots = face_neighbors_list[0].shape[-1]   # 3 for tri-only, 4 for quad/mixed
    padded_face_neighbors = torch.full((B, max_F, n_slots), -1, dtype=torch.long)
    for i, s in enumerate(face_neighbors_list):
        padded_face_neighbors[i, : s.shape[0]] = s

    return {
        "pc":             pcs,
        "faces":          padded_faces,
        "lengths":        lengths,
        "face_neighbors": padded_face_neighbors,
    }
