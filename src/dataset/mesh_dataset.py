import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from jaxtyping import Float, Int
from lightning.pytorch import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..constants import EOS_COORD, QUANT_MAX

MESH_EXTENSIONS = ("*.obj", "*.ply", "*.glb", "*.gltf", "*.stl", "*.off")

# 90° rotation matrices for X, Y, Z axes (column-vector convention)
_ROT90: np.ndarray = np.array([
    [[1,  0,  0], [0,  0, -1], [0,  1,  0]],  # X-axis
    [[0,  0,  1], [0,  1,  0], [-1, 0,  0]],  # Y-axis
    [[0, -1,  0], [1,  0,  0], [0,  0,  1]],  # Z-axis
], dtype=np.float64)


def random_90_rotation_matrix() -> np.ndarray:
    """50% identity; otherwise a random rotation composed of k×90° per axis, k ∈ {1,2,3}."""
    if np.random.randint(0, 2) == 0:
        return np.eye(3)
    R = np.eye(3)
    for axis in range(3):
        k = np.random.randint(1, 4)
        R = np.linalg.matrix_power(_ROT90[axis], k) @ R
    return R


def random_reflection_matrix() -> np.ndarray:
    """Reflect along one randomly chosen axis (x, y, or z), or no reflection (25% each)."""
    R = np.eye(3)
    axis = np.random.randint(0, 4)  # 0=none, 1=x, 2=y, 3=z
    if axis > 0:
        R[axis - 1, axis - 1] = -1.0
    return R


@dataclass
class DataCfg:
    num_points: int = 2048
    num_workers: int = 4
    augment: bool = False


def quantize_vertices(verts: np.ndarray) -> np.ndarray:
    v_min = verts.min(axis=0, keepdims=True)
    v_max = verts.max(axis=0, keepdims=True)
    scale = (v_max - v_min).max() + 1e-8
    verts_norm = (verts - v_min) / scale
    return np.round(verts_norm * QUANT_MAX).clip(0, QUANT_MAX).astype(np.int64)


def _normalize_face_vertices(faces: np.ndarray, verts_q: np.ndarray) -> np.ndarray:
    """Rotate each face's vertex sequence so the lexicographically smallest vertex comes first.

    Returns (F, 9) array of quantized face coords with normalized vertex order.
    Face sequence order is preserved — sorting is handled by ordering strategies.
    """
    F = len(faces)
    face_verts = verts_q[faces]  # (F, 3, 3)

    max_val = int(QUANT_MAX) + 1
    keys = (face_verts[:, :, 2].astype(np.int64) * (max_val ** 2) +
            face_verts[:, :, 1].astype(np.int64) * max_val +
            face_verts[:, :, 0].astype(np.int64))  # (F, 3)

    min_idx   = np.argmin(keys, axis=1)                          # (F,)
    shift_idx = (np.arange(3)[None, :] + min_idx[:, None]) % 3  # (F, 3)
    row_idx   = np.arange(F)[:, None]

    return face_verts[row_idx, shift_idx].reshape(F, 9).astype(np.int64)


def build_edge_adjacency(face_seq_q: np.ndarray) -> np.ndarray:
    """Compute face adjacency using quantized vertex positions as vertex identifiers.

    Uses quantized coordinates instead of vertex indices so that meshes with
    duplicated vertices at seams (UV seams, hard edges) are handled correctly.

    face_seq_q: (F, 9) array of quantized face coords [v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z]
    """
    F = len(face_seq_q)
    B = int(QUANT_MAX) + 1  # 128

    # Encode each quantized vertex position as a single int64 key
    vk = (face_seq_q[:, 0::3].astype(np.int64) * B * B +
          face_seq_q[:, 1::3].astype(np.int64) * B +
          face_seq_q[:, 2::3].astype(np.int64))              # (F, 3)

    # All three directed edges per face, slots 0/1/2: (V0,V1), (V1,V2), (V2,V0)
    all_edges = np.vstack([vk[:, [0, 1]], vk[:, [1, 2]], vk[:, [2, 0]]])
    all_edges = np.sort(all_edges, axis=1)                   # make undirected

    edge_keys = all_edges[:, 0] * (B ** 3) + all_edges[:, 1]

    face_ids = np.tile(np.arange(F), 3)
    slot_ids = np.repeat(np.arange(3), F)

    sort_perm   = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[sort_perm]

    matches = sorted_keys[:-1] == sorted_keys[1:]
    idx1 = sort_perm[:-1][matches]
    idx2 = sort_perm[1:][matches]

    adj = np.full((F, 3), -1, dtype=np.int64)
    adj[face_ids[idx1], slot_ids[idx1]] = face_ids[idx2]
    adj[face_ids[idx2], slot_ids[idx2]] = face_ids[idx1]
    return adj


def normalize_point_cloud(pc: np.ndarray) -> np.ndarray:
    center = pc.mean(axis=0, keepdims=True)
    pc = pc - center
    scale = np.abs(pc).max() + 1e-8
    return pc / scale


def load_mesh_raw(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh file and return raw (vertices, faces). Called once per mesh at init."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())
    return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int64)


def _sample_surface(verts: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """Weighted random surface sampling without constructing a Trimesh object."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / (areas.sum() + 1e-8)
    chosen = np.random.choice(len(faces), size=num_points, p=probs)
    r1 = np.random.rand(num_points, 1)
    r2 = np.random.rand(num_points, 1)
    sqrt_r1 = np.sqrt(r1)
    pts = (1 - sqrt_r1) * v0[chosen] + sqrt_r1 * (1 - r2) * v1[chosen] + sqrt_r1 * r2 * v2[chosen]
    return pts.astype(np.float32)


def process_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    augment: bool = False,
) -> tuple[
    Float[np.ndarray, "points 3"],
    Int[np.ndarray, "faces 9"],
    Int[np.ndarray, "faces 3"],
]:
    """Apply augmentation, quantize, and compute adjacency. Uses cached raw mesh data."""
    if augment:
        R_3x3 = random_reflection_matrix() @ random_90_rotation_matrix()
        verts = verts @ R_3x3.T

    pc = normalize_point_cloud(_sample_surface(verts, faces, num_points))

    verts_q = quantize_vertices(verts)
    face_seq = _normalize_face_vertices(faces, verts_q)
    face_neighbors = build_edge_adjacency(face_seq)

    return pc, face_seq, face_neighbors


def collate_fn(batch: list[dict]) -> dict:
    pcs = torch.stack([b["pc"] for b in batch], dim=0)
    faces_list = [b["faces"] for b in batch]

    padded_faces = pad_sequence(faces_list, batch_first=True, padding_value=EOS_COORD)
    lengths = torch.tensor([f.shape[0] for f in faces_list], dtype=torch.long)

    face_neighbors_list = [b["face_neighbors"] for b in batch]
    max_F = max(s.shape[0] for s in face_neighbors_list)
    B = len(batch)
    padded_face_neighbors = torch.full((B, max_F, 3), -1, dtype=torch.long)
    for i, s in enumerate(face_neighbors_list):
        padded_face_neighbors[i, : s.shape[0]] = s

    return {
        "pc": pcs,
        "faces": padded_faces,
        "lengths": lengths,
        "face_neighbors": padded_face_neighbors,
    }


class MeshDataset(Dataset):
    def __init__(
        self,
        mesh_paths: list[str],
        num_points: int = 8192,
        augment: bool = False,
        virtual_size: int | None = None,
    ):
        self.num_points    = num_points
        self.augment       = augment
        self._virtual_size = virtual_size if virtual_size is not None else len(mesh_paths)
        self._raw = [load_mesh_raw(p) for p in mesh_paths]

    def __len__(self) -> int:
        return self._virtual_size

    def __getitem__(self, idx: int) -> dict:
        verts, faces = self._raw[idx % len(self._raw)]
        pc, face_seq, face_neighbors = process_mesh(
            verts, faces, self.num_points, self.augment
        )
        return {
            "pc": torch.from_numpy(pc),
            "faces": torch.from_numpy(face_seq).long(),
            "face_neighbors": torch.from_numpy(face_neighbors).long(),
        }


class MeshDataModule(LightningDataModule):
    def __init__(self, cfg: DataCfg, batch_size: int, train_dir: str, val_dir: str):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir

    def _collect_paths(self, directory: str) -> list[str]:
        paths: list[str] = []
        for ext in MESH_EXTENSIONS:
            paths += glob.glob(os.path.join(directory, "**", ext), recursive=True)
        if not paths:
            raise RuntimeError(f"No mesh files found under {directory}")
        return paths

    def setup(self, stage: str = "") -> None:
        train_paths = self._collect_paths(self.train_dir)
        virtual_size = max(len(train_paths), self.batch_size)
        self.train_dataset = MeshDataset(
            train_paths, self.cfg.num_points,
            augment=self.cfg.augment,
            virtual_size=virtual_size,
        )
        self.val_dataset = MeshDataset(
            self._collect_paths(self.val_dir), self.cfg.num_points,
            augment=False,
        )

    def _dataloader(self, dataset: MeshDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)
