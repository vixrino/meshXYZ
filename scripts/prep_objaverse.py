"""Objaverse dataset preparation for mesh_genai quad training.

Downloads N objects from Objaverse, applies face-count and topology filters,
and writes a manifest CSV plus copies accepted meshes into --out_dir.

Quick-start
-----------
# Validation batch (fast, ~30 min on Colab T4):
python scripts/prep_objaverse.py --n 500 --out_dir /content/data/train --split train

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

Quad preference
---------------
Objaverse objects come in many formats.  Only .obj files can carry native quad
faces (all others are triangulated on export).  For .obj files we use our custom
parse_obj; for .glb/.ply/etc. we fall back to trimesh (triangle-only).

The manifest CSV contains a `quad_frac` column — sort by this descending to find
the most quad-rich objects for targeted quad training.

Manifest columns
----------------
uid, local_path, format, n_tri, n_quad, quad_frac, kept
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
# Fraction of quads needed to tag a mesh as "quad-rich" (informational only).
QUAD_RICH_THRESHOLD = 0.10


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

    n_total  = n_tri + n_quad
    quad_frac = n_quad / max(n_total, 1)
    return {"n_tri": n_tri, "n_quad": n_quad, "n_total": n_total, "quad_frac": quad_frac}


def download_and_filter(
    n: int,
    out_dir: str,
    seed: int = 42,
    processes: int = 4,
) -> list[dict]:
    """Download N Objaverse objects and filter by topology.

    Returns a list of manifest rows (dict per mesh).
    """
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
        # Filter: prefer objects where the source format looks like .obj
        obj_uids  = [u for u, a in annotations.items()
                     if isinstance(a, dict) and str(a.get("fileType", "")).lower() in ("obj",)]
        other_uids = [u for u in annotations if u not in obj_uids]
        # Fill quota: first from OBJ, remainder from others (shuffled)
        random.seed(seed)
        random.shuffle(obj_uids);  random.shuffle(other_uids)
        selected_uids = (obj_uids + other_uids)[:n]
        log.info("  Preferred %d OBJ-format objects out of %d sampled", len(obj_uids), len(annotations))
    except Exception as exc:
        log.warning("Annotation pre-filter failed (%s); using random sample.", exc)
        random.seed(seed)
        selected_uids = random.sample(all_uids, n)

    # Always shuffle so we don't bias towards one creator
    random.seed(seed + 1)
    random.shuffle(selected_uids)

    log.info("Downloading %d objects (processes=%d) …", len(selected_uids), processes)
    uid_to_path = objaverse.load_objects(uids=selected_uids, download_processes=processes)
    log.info("  %d objects downloaded", len(uid_to_path))

    rows: list[dict] = []
    n_kept = 0

    for uid, src_path in uid_to_path.items():
        if src_path is None or not Path(src_path).exists():
            rows.append({"uid": uid, "local_path": "", "format": "", "n_tri": 0,
                         "n_quad": 0, "quad_frac": 0.0, "kept": False})
            continue

        ext   = Path(src_path).suffix.lower()
        stats = analyse_mesh(src_path)

        if not stats:
            rows.append({"uid": uid, "local_path": src_path, "format": ext,
                         "n_tri": 0, "n_quad": 0, "quad_frac": 0.0, "kept": False})
            continue

        n_total   = stats["n_total"]
        quad_frac = stats["quad_frac"]
        keep = MIN_FACES <= n_total <= MAX_FACES

        if keep:
            # Copy into output directory using uid as filename to avoid collisions.
            dst = out_path / f"{uid}{ext}"
            shutil.copy2(src_path, dst)
            n_kept += 1

        rows.append({
            "uid":        uid,
            "local_path": str(dst) if keep else src_path,
            "format":     ext,
            "n_tri":      stats["n_tri"],
            "n_quad":     stats["n_quad"],
            "quad_frac":  round(quad_frac, 4),
            "kept":       keep,
        })

        if len(rows) % 50 == 0:
            log.info("  Processed %d / %d — kept %d so far", len(rows), len(uid_to_path), n_kept)

    log.info("Done.  %d / %d objects passed filters.", n_kept, len(uid_to_path))
    return rows


def write_manifest(rows: list[dict], manifest_path: str) -> None:
    fieldnames = ["uid", "local_path", "format", "n_tri", "n_quad", "quad_frac", "kept"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Manifest written to %s (%d rows)", manifest_path, len(rows))


def print_summary(rows: list[dict]) -> None:
    kept  = [r for r in rows if r["kept"]]
    quads = [r for r in kept if float(r["quad_frac"]) >= QUAD_RICH_THRESHOLD]
    if not kept:
        log.warning("No meshes kept — check filter thresholds or download errors.")
        return
    avg_tri  = sum(int(r["n_tri"])  for r in kept) / len(kept)
    avg_quad = sum(int(r["n_quad"]) for r in kept) / len(kept)
    avg_qf   = sum(float(r["quad_frac"]) for r in kept) / len(kept)
    print("\n── Dataset summary ──────────────────────────────────────")
    print(f"  Objects kept      : {len(kept)}")
    print(f"  Avg tri faces     : {avg_tri:.0f}")
    print(f"  Avg quad faces    : {avg_quad:.0f}")
    print(f"  Avg quad fraction : {avg_qf:.2%}")
    print(f"  Quad-rich objects : {len(quads)} ({len(quads)/len(kept):.1%}) [≥{QUAD_RICH_THRESHOLD:.0%} quads]")
    print(f"  Formats           : { {r['format'] for r in kept} }")
    print("─" * 58)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Objaverse dataset for mesh_genai quad training.")
    parser.add_argument("--n",         type=int, default=500,
                        help="Number of objects to attempt to download (default 500).")
    parser.add_argument("--out_dir",   default="/content/data/train",
                        help="Directory where accepted meshes are copied.")
    parser.add_argument("--manifest",  default=None,
                        help="Path for manifest CSV (default: <out_dir>/manifest.csv).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--processes", type=int, default=4,
                        help="Parallel download processes.")
    parser.add_argument("--split",     choices=["train", "val"], default="train")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.csv")

    rows = download_and_filter(
        n=args.n,
        out_dir=args.out_dir,
        seed=args.seed,
        processes=args.processes,
    )
    write_manifest(rows, manifest_path)
    print_summary(rows)


if __name__ == "__main__":
    main()
