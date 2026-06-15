import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ..constants import EOS_COORD
from .collate import collate_fn  # noqa: F401 — re-exported for convenience
from .mesh_ops import process_mesh
from .obj_parser import parse_obj

MESH_EXTENSIONS = ("*.obj", "*.ply", "*.glb", "*.gltf", "*.stl", "*.off")


@dataclass
class DataCfg:
    num_points: int = 2048
    num_workers: int = 4
    augment: bool = False
    face_layout: str = "tri"
    # "tri": triangle-only mode — existing (F, 9) faces, (F, 3) neighbors. Default; fully
    #        backward-compatible with all existing triangle configs and checkpoints.
    # "quad": QuadGPT unified 12-token block — (F, 12) faces, (F, 4) neighbors.
    #         Triangles are padded with TRI_PAD at positions 0-2; quads use all 12 positions.
    #         Requires vocab_size=257 in decoder config. relative=true is supported
    #         (TRI_PAD-aware anchor + raw prefix logits); use_edge_cond=false for now.
    # Note: str not Literal["tri","quad"] — dacite silently drops Literal-typed fields
    # when the installed dacite version predates full Literal support, causing face_layout
    # to remain "tri" even when config says "quad".


def load_mesh_raw(
    path: str,
    face_layout: str = "tri",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a mesh file and return (vertices, faces_tri, faces_quad).

    Parameters
    ----------
    path : str
        Path to the mesh file.
    face_layout : {"tri", "quad"}
        "tri" — use trimesh with force="mesh" (triangulates everything).
                faces_quad is always an empty (0, 4) array.
        "quad" — for .obj files, parse natively to preserve quad faces.
                Other formats fall back to triangle-only (trimesh path).

    Returns
    -------
    vertices : (V, 3) float64
    faces_tri : (T, 3) int64   — 0-based vertex indices
    faces_quad : (Q, 4) int64  — 0-based vertex indices; empty for triangle mode
    """
    _empty_quad = np.empty((0, 4), dtype=np.int64)

    if face_layout == "quad" and path.lower().endswith(".obj"):
        result = parse_obj(path)
        return result.vertices, result.faces_tri, result.faces_quad

    # Triangle path — original behavior preserved exactly.
    # trimesh is imported lazily so collate_fn / MeshDataset can be imported
    # in test environments that only have torch (not trimesh).
    import trimesh  # noqa: PLC0415
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())
    return (
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces,    dtype=np.int64),
        _empty_quad,
    )


class MeshDataset(Dataset):
    def __init__(
        self,
        mesh_paths:   list[str],
        num_points:   int = 8192,
        augment:      bool = False,
        virtual_size: "int | None" = None,
        face_layout:  str = "tri",
    ):
        self.num_points    = num_points
        self.augment       = augment
        self.face_layout   = face_layout
        self._virtual_size = virtual_size if virtual_size is not None else len(mesh_paths)
        self._raw = [load_mesh_raw(p, face_layout=face_layout) for p in mesh_paths]

    def __len__(self) -> int:
        return self._virtual_size

    def __getitem__(self, idx: int) -> dict:
        verts, faces_tri, faces_quad = self._raw[idx % len(self._raw)]
        pc, face_seq, face_neighbors = process_mesh(
            verts, faces_tri, self.num_points, self.augment,
            face_layout=self.face_layout, faces_quad=faces_quad,
        )
        return {
            "pc":             torch.from_numpy(pc),
            "faces":          torch.from_numpy(face_seq).long(),
            "face_neighbors": torch.from_numpy(face_neighbors).long(),
        }


class MeshDataModule(LightningDataModule):
    def __init__(self, cfg: DataCfg, batch_size: int, train_dir: str, val_dir: str):
        super().__init__()
        self.cfg        = cfg
        self.batch_size = batch_size
        self.train_dir  = train_dir
        self.val_dir    = val_dir

    def _collect_paths(self, directory: str) -> list[str]:
        paths: list[str] = []
        for ext in MESH_EXTENSIONS:
            paths += glob.glob(os.path.join(directory, "**", ext), recursive=True)
        if not paths:
            raise RuntimeError(f"No mesh files found under {directory}")
        return paths

    def setup(self, stage: str = "") -> None:
        train_paths  = self._collect_paths(self.train_dir)
        virtual_size = max(len(train_paths), self.batch_size)
        self.train_dataset = MeshDataset(
            train_paths, self.cfg.num_points,
            augment=self.cfg.augment,
            virtual_size=virtual_size,
            face_layout=self.cfg.face_layout,
        )
        self.val_dataset = MeshDataset(
            self._collect_paths(self.val_dir), self.cfg.num_points,
            augment=False,
            face_layout=self.cfg.face_layout,
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
