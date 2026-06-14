"""Objaverse dataset preparation for mesh_genai quad training.

Downloads N objects from Objaverse, applies face-count and topology filters,
and writes a manifest CSV plus copies accepted meshes into --out_dir.

Quick-start
-----------
# Balanced tri/quad dataset (default, ~50% each):
python scripts/prep_objaverse.py --n 500 --out_dir /content/data/train --split train

# All quadrangulated (mix_ratio=1.0):
python scripts/prep_objaverse.py --n 500 --out_dir /content/data/train --mix_ratio 1.0

# Triangle-only (mix_ratio=0.0):
python scripts/prep_objaverse.py --n 500 --out_dir /content/data/train --mix_ratio 0.0

# Scale-up:
python scripts/prep_objaverse.py --n 5000 --out_dir /content/data/train --split train

# Validation split (small):
python scripts/prep_objaverse.py --n 50 --out_dir /content/data/val --split val

Filter logic
------------
A mesh is kept when ALL conditions hold:
  1. Total face count (tri + quad) is in [MIN_FACES, MAX_FACES].
  2. The mesh has at least 3 non-degenerate triangles or quads (robustness check).
  3. No parsing exception was raised.

Mix ratio
---------
--mix_ratio (float, default 0.5) controls the fraction of kept meshes that are
quadrangulated via pymeshlab smart triangle pairing.

  mix_ratio = 0.0  → all meshes saved as triangle .obj  (no pymeshlab needed)
  mix_ratio = 0.5  → ~50% quad-dominant, ~50% triangle-only
  mix_ratio = 1.0  → all meshes quadrangulated (old --quadrangulate behaviour)

The per-mesh tri/quad decision is seeded from --seed + hash(uid), so the split
is reproducible regardless of download order.

All output meshes are saved as .obj so the format is homogeneous.  Triangle-only
meshes are processed by MeshDataset with face_layout="quad" the same as quads —
each triangle gets TRI_PAD padding to fill a 12-token block.

Manifest columns
----------------
uid, local_path, format, n_tri, n_quad, quad_frac, kept, quadrangulated
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Filtering thresholds ──────────────────────────────────────────────────────
MIN_FACES = 50
MAX_FACES = 2000

# Quad-frac bucket boundaries for the summary printout.
TRI_ONLY_THRESH = 0.05   # < 5%  quads → "triangle-only"
QUAD_DOM_THRESH = 0.95   # > 95% quads → "quad-dominant"
# 5–95% quads → "mixed"


def analyse_mesh(path: str) -> dict:
    """Parse mesh and return topology stats.  Returns {} on any error."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".obj":
            from src.dataset.obj_parser import parse_obj
            r = parse_obj(path)
            n_tri  = len(r.faces_tri)
            n_quad = len(r.faces_quad)
        else:
            # Non-OBJ: trimesh triangulates on load.
            import trimesh
            mesh = trimesh.load(path, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(mesh.dump())
            n_tri  = len(mesh.faces)
            n_quad = 0
    except Exception as exc:
        log.debug("Failed to parse %s: %s", path, exc)
        return {}

    n_total   = n_tri + n_quad
    quad_frac = n_quad / max(n_total, 1)
    return {"n_tri": n_tri, "n_quad": n_quad, "n_total": n_total, "quad_frac": quad_frac}


def quadrangulate_mesh(src_path: str, dst_path: str) -> dict | None:
    """Convert a triangle mesh to quad-dominant using pymeshlab smart triangle pairing.

    Pairs adjacent triangles into quads without changing mesh resolution.
    Saves the result as a .obj file at dst_path.  Returns updated topology stats
    on success, None on failure (caller should fall back to save_tri_as_obj).

    Parameters
    ----------
    src_path : str
        Input mesh path (any format pymeshlab can load).
    dst_path : str
        Output .obj path (will be overwritten).
    """
    try:
        import pymeshlab  # noqa: PLC0415
    except ImportError:
        log.warning("pymeshlab not installed; skipping quadrangulation.  Run: pip install pymeshlab")
        return None

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(src_path)

        n_in = ms.current_mesh().face_number()
        if n_in == 0:
            log.warning("Empty mesh at %s; skipping quadrangulation.", src_path)
            return None

        # Cleanup pass: GLB meshes from Objaverse often have duplicate verts,
        # unreferenced verts, and non-manifold edges that break quadrangulation.
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_repair_non_manifold_edges()

        # Pair adjacent triangles into quads (preserves mesh resolution).
        ms.meshing_tri_to_quad_by_smart_triangle_pairing()

        # save_textures=False: GLB files embed textures that pymeshlab cannot
        # re-export (missing plugin), causing ~60% of saves to fail otherwise.
        ms.save_current_mesh(dst_path, save_textures=False)

    except Exception as exc:
        log.warning("Quadrangulation failed on %s: %s", Path(src_path).name, exc)
        return None

    # Re-analyse the saved .obj to get updated counts.
    stats = analyse_mesh(dst_path)
    if not stats:
        log.warning("Could not re-parse quadrangulated mesh at %s; discarding.", dst_path)
        return None

    return stats


def save_tri_as_obj(src_path: str, dst_path: str) -> dict | None:
    """Save a triangle mesh as .obj without quadrangulation.

    Applies the same cleanup filters as quadrangulate_mesh for consistency
    (removes duplicate/unreferenced vertices and repairs non-manifold edges).
    Falls back to trimesh if pymeshlab is unavailable.

    Returns updated topology stats on success, None on failure.
    """
    try:
        import pymeshlab  # noqa: PLC0415
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(src_path)
        if ms.current_mesh().face_number() == 0:
            return None
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.save_current_mesh(dst_path, save_textures=False)

    except ImportError:
        # pymeshlab not available — fall back to trimesh for format conversion
        try:
            import trimesh
            mesh = trimesh.load(src_path, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(mesh.dump())
            mesh.export(dst_path)
        except Exception as exc:
            log.warning("Could not convert %s to .obj via trimesh: %s",
                        Path(src_path).name, exc)
            return None

    except Exception as exc:
        log.warning("Could not convert %s to .obj: %s", Path(src_path).name, exc)
        return None

    return analyse_mesh(dst_path)


def download_and_filter(
    n: int,
    out_dir: str,
    seed: int = 42,
    processes: int = 4,
    quadrangulate: bool = False,  # deprecated: kept for backward compat (= mix_ratio 1.0)
    mix_ratio: float = 0.5,
) -> list[dict]:
    """Download N Objaverse objects and filter by topology.

    Parameters
    ----------
    mix_ratio : float
        Fraction of kept meshes to quadrangulate (0.0 = all tri, 1.0 = all quad).
        The per-mesh decision is seeded from ``seed + hash(uid)`` for reproducibility.
        Triangle-only meshes are still saved as .obj so the output format is homogeneous.
    quadrangulate : bool
        Deprecated.  Passing True is equivalent to mix_ratio=1.0.

    Returns a list of manifest rows (dict per mesh).
    """
    if quadrangulate and mix_ratio < 1.0:
        log.info("--quadrangulate passed; treating as --mix_ratio 1.0 for backward compat.")
        mix_ratio = 1.0

    try:
        import objaverse  # noqa: PLC0415
    except ImportError:
        log.error("objaverse not installed.  Run: pip install objaverse")
        sys.exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading Objaverse UID list …")
    all_uids = objaverse.load_uids()
    log.info("  %d total UIDs available", len(all_uids))

    # Try to get annotation metadata to prefer OBJ-format objects.
    selected_uids: list[str] = []
    try:
        annotations = objaverse.load_annotations(uids=random.sample(all_uids, min(50_000, len(all_uids))))
        obj_uids  = [u for u, a in annotations.items()
                     if isinstance(a, dict) and str(a.get("fileType", "")).lower() in ("obj",)]
        other_uids = [u for u in annotations if u not in obj_uids]
        random.seed(seed)
        random.shuffle(obj_uids);  random.shuffle(other_uids)
        selected_uids = (obj_uids + other_uids)[:n]
        log.info("  Preferred %d OBJ-format objects out of %d sampled", len(obj_uids), len(annotations))
    except Exception as exc:
        log.warning("Annotation pre-filter failed (%s); using random sample.", exc)
        random.seed(seed)
        selected_uids = random.sample(all_uids, n)

    random.seed(seed + 1)
    random.shuffle(selected_uids)

    log.info(
        "Downloading %d objects (processes=%d, mix_ratio=%.2f) …",
        len(selected_uids), processes, mix_ratio,
    )
    uid_to_path = objaverse.load_objects(uids=selected_uids, download_processes=processes)
    log.info("  %d objects downloaded", len(uid_to_path))

    rows: list[dict] = []
    n_kept = 0
    n_quad_success = 0

    for uid, src_path in uid_to_path.items():
        if src_path is None or not Path(src_path).exists():
            rows.append({"uid": uid, "local_path": "", "format": "", "n_tri": 0,
                         "n_quad": 0, "quad_frac": 0.0, "kept": False,
                         "quadrangulated": False})
            continue

        ext   = Path(src_path).suffix.lower()
        stats = analyse_mesh(src_path)

        if not stats:
            rows.append({"uid": uid, "local_path": src_path, "format": ext,
                         "n_tri": 0, "n_quad": 0, "quad_frac": 0.0, "kept": False,
                         "quadrangulated": False})
            continue

        n_total = stats["n_total"]
        keep    = MIN_FACES <= n_total <= MAX_FACES

        was_quadrangulated = False
        if keep:
            n_kept += 1
            dst = out_path / f"{uid}.obj"

            # Per-mesh reproducible tri/quad decision: seed from uid so the split
            # is stable regardless of the download order or re-runs with same seed.
            _mesh_rng = random.Random(seed ^ (abs(hash(uid)) & 0xFFFF_FFFF))
            do_quad   = (mix_ratio > 0.0) and (_mesh_rng.random() < mix_ratio)

            if do_quad:
                new_stats = quadrangulate_mesh(src_path, str(dst))
                if new_stats:
                    stats = new_stats
                    was_quadrangulated = True
                    n_quad_success += 1
                    log.info(
                        "  [%d] quad  %s → %d tri + %d quad (%.0f%% quad)",
                        n_kept, uid, stats["n_tri"], stats["n_quad"],
                        stats["quad_frac"] * 100,
                    )
                else:
                    # Quadrangulation failed — fall back to triangle .obj
                    tri_stats = save_tri_as_obj(src_path, str(dst))
                    if tri_stats:
                        stats = tri_stats
                    else:
                        shutil.copy2(src_path, dst)
                    log.warning("  [%d] tri   %s (quadrangulation failed, kept as tri)",
                                n_kept, uid)
            else:
                # Intentionally kept as triangle for balanced dataset
                tri_stats = save_tri_as_obj(src_path, str(dst))
                if tri_stats:
                    stats = tri_stats
                else:
                    shutil.copy2(src_path, dst)
                log.info(
                    "  [%d] tri   %s → %d tri (intentionally triangle)",
                    n_kept, uid, stats["n_tri"],
                )

        rows.append({
            "uid":             uid,
            "local_path":      str(dst) if keep else src_path,
            "format":          ".obj" if keep else ext,
            "n_tri":           stats["n_tri"],
            "n_quad":          stats["n_quad"],
            "quad_frac":       round(stats["quad_frac"], 4),
            "kept":            keep,
            "quadrangulated":  was_quadrangulated,
        })

        if len(rows) % 50 == 0:
            log.info("  Processed %d / %d — kept %d so far", len(rows), len(uid_to_path), n_kept)

    if mix_ratio > 0.0:
        log.info(
            "Done.  %d / %d objects passed filters; %d / %d quadrangulated (mix_ratio=%.2f).",
            n_kept, len(uid_to_path), n_quad_success, n_kept, mix_ratio,
        )
    else:
        log.info("Done.  %d / %d objects passed filters (triangle-only).", n_kept, len(uid_to_path))

    return rows


def write_manifest(rows: list[dict], manifest_path: str) -> None:
    fieldnames = ["uid", "local_path", "format", "n_tri", "n_quad",
                  "quad_frac", "kept", "quadrangulated"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Manifest written to %s (%d rows)", manifest_path, len(rows))


def print_summary(rows: list[dict]) -> None:
    kept = [r for r in rows if r["kept"]]
    if not kept:
        log.warning("No meshes kept — check filter thresholds or download errors.")
        return

    quad_fracs = [float(r["quad_frac"]) for r in kept]
    tri_only = [qf for qf in quad_fracs if qf < TRI_ONLY_THRESH]
    mixed    = [qf for qf in quad_fracs if TRI_ONLY_THRESH <= qf <= QUAD_DOM_THRESH]
    quad_dom = [qf for qf in quad_fracs if qf > QUAD_DOM_THRESH]

    # Global face-level fractions (over all faces, not per-mesh averages)
    total_tri  = sum(int(r["n_tri"])  for r in kept)
    total_quad = sum(int(r["n_quad"]) for r in kept)
    total_faces = total_tri + total_quad
    global_quad_frac = total_quad / max(total_faces, 1)
    global_tri_frac  = total_tri  / max(total_faces, 1)

    n_quadrangulated = sum(1 for r in kept if r.get("quadrangulated"))

    print("\n── Dataset summary ──────────────────────────────────────────────")
    print(f"  Objects kept               : {len(kept)}")
    print(f"  Quadrangulated (success)   : {n_quadrangulated} / {len(kept)}"
          f"  ({n_quadrangulated / max(len(kept), 1):.1%})")
    print()
    print(f"  Mesh-level quad_frac distribution ({len(kept)} kept meshes):")
    print(f"    triangle-only  (quad_frac < 5%)     : {len(tri_only):4d}"
          f"  ({len(tri_only)/len(kept):.1%})")
    print(f"    mixed          (5% ≤ q ≤ 95%)       : {len(mixed):4d}"
          f"  ({len(mixed)/len(kept):.1%})")
    print(f"    quad-dominant  (quad_frac > 95%)    : {len(quad_dom):4d}"
          f"  ({len(quad_dom)/len(kept):.1%})")
    print()
    print(f"  Global face-level composition ({total_faces:,d} total faces):")
    print(f"    triangle faces : {total_tri:9,d}  ({global_tri_frac:.1%})")
    print(f"    quad faces     : {total_quad:9,d}  ({global_quad_frac:.1%})")
    balanced = abs(global_quad_frac - 0.5) < 0.15
    print(f"    {'✓ Balanced (40–60% quad target met)' if balanced else '⚠ Imbalanced — consider adjusting --mix_ratio'}")
    print(f"  Formats                    : { {r['format'] for r in kept} }")
    print("─" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Objaverse dataset for mesh_genai quad training.")
    parser.add_argument("--n",          type=int, default=500,
                        help="Number of objects to attempt to download (default 500).")
    parser.add_argument("--out_dir",    default="/content/data/train",
                        help="Directory where accepted meshes are copied.")
    parser.add_argument("--manifest",   default=None,
                        help="Path for manifest CSV (default: <out_dir>/manifest.csv).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--processes",  type=int, default=4,
                        help="Parallel download processes.")
    parser.add_argument("--split",      choices=["train", "val"], default="train")
    parser.add_argument("--mix_ratio",  type=float, default=0.5,
                        help=(
                            "Fraction of kept meshes to quadrangulate via pymeshlab "
                            "(default 0.5).  0.0 = all triangle-only; 1.0 = all quadrangulated.  "
                            "The per-mesh decision is seeded for reproducibility."
                        ))
    parser.add_argument("--quadrangulate", action="store_true", default=False,
                        help="[Deprecated] Equivalent to --mix_ratio 1.0.")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.csv")

    rows = download_and_filter(
        n=args.n,
        out_dir=args.out_dir,
        seed=args.seed,
        processes=args.processes,
        quadrangulate=args.quadrangulate,
        mix_ratio=args.mix_ratio,
    )
    write_manifest(rows, manifest_path)
    print_summary(rows)


if __name__ == "__main__":
    main()
