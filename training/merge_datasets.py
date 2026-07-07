"""Merge multiple Roboflow-exported YOLOv8 datasets into one unified training set.

Produces `training/data/merged/` with the following canonical 7-class taxonomy:

    0: spaghetti
    1: stringing
    2: blob
    3: crack
    4: detachment
    5: over_extrusion
    6: under_extrusion

The per-source class remap collapses noisy / typo'd labels in the flaws dataset
into the canonical set. Provenance (LICENSE + SOURCES) is written alongside
`data.yaml` so `prepare_dataset.py` doesn't refuse to run.

Run from repo root:
    python training/merge_datasets.py

Idempotent-ish: rewrites the merged dir each run (deletes and recreates).
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("merge")

CANONICAL_CLASSES = [
    "spaghetti",
    "stringing",
    "blob",
    "crack",
    "detachment",
    "over_extrusion",
    "under_extrusion",
]

CANONICAL_IDX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}


@dataclass(frozen=True)
class SourceSpec:
    """One Roboflow-exported dataset with its per-source class remap."""

    root: Path
    slug: str  # short prefix used to disambiguate filenames
    # source class id (int from its data.yaml `names`) -> canonical class id
    # If a key is missing, that class's labels are DROPPED from the merge.
    remap: dict[int, int]


def _resolve_source_from_yaml(root: Path, remap_by_name: dict[str, str]) -> dict[int, int]:
    """Load `data.yaml` at root, translate a name→canonical-name map to id→id."""
    d = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = d.get("names") or []
    remap: dict[int, int] = {}
    for src_idx, src_name in enumerate(names):
        canonical_name = remap_by_name.get(src_name)
        if canonical_name is None:
            logger.warning("%s: no remap for source class %r — labels will be dropped", root.name, src_name)
            continue
        if canonical_name not in CANONICAL_IDX:
            raise ValueError(f"{root.name}: canonical target {canonical_name!r} not in canonical taxonomy")
        remap[src_idx] = CANONICAL_IDX[canonical_name]
    return remap


def build_sources(data_root: Path) -> list[SourceSpec]:
    """Build the SourceSpec list, resolving each dataset's name-based remap
    against its data.yaml so we don't have to hand-count indices.
    """
    sources: list[SourceSpec] = []

    # ---- spaghetti-3d (2 classes: spaghetti, stringing) -----------------
    sources.append(SourceSpec(
        root=data_root / "spaghetti-3d",
        slug="s3d",
        remap=_resolve_source_from_yaml(data_root / "spaghetti-3d", {
            "spaghetti": "spaghetti",
            "stringing": "stringing",
        }),
    ))

    # ---- 3d-printing-flaws (9 noisy classes -> 5 canonical) -------------
    flaws_remap_by_name = {
        "Blobs": "blob",
        "Crack": "crack",
        "Spagatti": "spaghetti",     # typo
        "Spaghetti": "spaghetti",    # capitalized dup
        "Stringing": "stringing",
        "bed adhesion failure": "detachment",
        "fail": "blob",              # ambiguous — best-effort merge as blob
        "poor initial layer bed adhesion faiure": "detachment",  # typo
        "spaghetti": "spaghetti",
    }
    sources.append(SourceSpec(
        root=data_root / "3d-printing-flaws",
        slug="flw",
        remap=_resolve_source_from_yaml(data_root / "3d-printing-flaws", flaws_remap_by_name),
    ))

    # ---- syLucauc/3d-printing-failure-detection (6 clean classes) -------
    sylucauc_remap_by_name = {
        "blobs": "blob",
        "cracks": "crack",
        "over_extrusion": "over_extrusion",
        "spaghetti": "spaghetti",
        "stringing": "stringing",
        "under_extrusion": "under_extrusion",
    }
    sources.append(SourceSpec(
        root=data_root / "sylucauc-3dpfd",
        slug="syl",
        remap=_resolve_source_from_yaml(data_root / "sylucauc-3dpfd", sylucauc_remap_by_name),
    ))
    return sources


def _rewrite_label_file(src: Path, dst: Path, remap: dict[int, int]) -> int:
    """Rewrite a YOLO label file with class IDs remapped. Returns kept-line count."""
    lines_kept = 0
    out_lines: list[str] = []
    try:
        raw = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            src_cls = int(parts[0])
        except (ValueError, IndexError):
            continue
        canonical = remap.get(src_cls)
        if canonical is None:
            # class not in the remap -> drop this box
            continue
        out_lines.append(f"{canonical} {' '.join(parts[1:])}")
        lines_kept += 1
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return lines_kept


def _copy_split(source: SourceSpec, split: str, merged_root: Path) -> tuple[int, int, int]:
    """Copy one split (train/valid/test) from a source into the merged dir.

    Returns (images_copied, labels_written, boxes_kept).
    """
    src_img_dir = source.root / split / "images"
    src_lbl_dir = source.root / split / "labels"
    if not src_img_dir.is_dir():
        return 0, 0, 0
    dst_img_dir = merged_root / split / "images"
    dst_lbl_dir = merged_root / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs_copied = 0
    lbls_written = 0
    boxes_kept = 0
    for img in src_img_dir.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem = img.stem
        # Prefix filenames with source slug so collisions across sources don't clobber.
        new_stem = f"{source.slug}_{stem}"
        dst_img = dst_img_dir / (new_stem + img.suffix.lower())
        dst_lbl = dst_lbl_dir / (new_stem + ".txt")
        if dst_img.exists():
            continue  # already merged in a previous run; skip
        # Hardlink to save disk. Fall back to copy across drives.
        try:
            dst_img.hardlink_to(img)
        except (OSError, NotImplementedError):
            shutil.copy2(img, dst_img)
        imgs_copied += 1
        src_lbl = src_lbl_dir / (stem + ".txt")
        if src_lbl.exists():
            kept = _rewrite_label_file(src_lbl, dst_lbl, source.remap)
            lbls_written += 1
            boxes_kept += kept
        else:
            # No label file = negative example (no failures visible).
            # YOLO wants an empty .txt to signal this.
            dst_lbl.write_text("", encoding="utf-8")
            lbls_written += 1
    return imgs_copied, lbls_written, boxes_kept


def write_provenance(merged_root: Path, sources: list[SourceSpec]) -> None:
    """Emit LICENSE.txt + SOURCES.md so prepare_dataset.py's provenance check passes."""
    lic = merged_root / "LICENSE.txt"
    lic.write_text(
        "All three source datasets are licensed under CC BY 4.0 (Roboflow Universe).\n"
        "Attribution required for any distributed weights. See SOURCES.md.\n",
        encoding="utf-8",
    )
    src_md = merged_root / "SOURCES.md"
    lines = [
        "# Dataset sources",
        "",
        "This directory is a merge of three Roboflow-Universe-hosted datasets under CC BY 4.0.",
        "",
    ]
    for s in sources:
        lines.append(f"## {s.slug} ({s.root.name})")
        lines.append(f"- root: `{s.root}`")
        lines.append(f"- class remap (source id -> canonical id): `{s.remap}`")
        lines.append("")
    src_md.write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml(merged_root: Path) -> None:
    body = {
        "path": str(merged_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }
    (merged_root / "data.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "training" / "data"
    merged_root = data_root / "merged"

    if merged_root.exists():
        logger.info("removing existing %s", merged_root)
        shutil.rmtree(merged_root)
    merged_root.mkdir(parents=True)

    sources = build_sources(data_root)
    for s in sources:
        logger.info("source %s: remap=%s", s.slug, s.remap)

    totals = {"images": 0, "labels": 0, "boxes": 0}
    for s in sources:
        for split in ("train", "valid", "test"):
            imgs, lbls, boxes = _copy_split(s, split, merged_root)
            if imgs:
                logger.info("%s/%s: images=%d labels=%d boxes=%d", s.slug, split, imgs, lbls, boxes)
                totals["images"] += imgs
                totals["labels"] += lbls
                totals["boxes"] += boxes
    write_provenance(merged_root, sources)
    write_data_yaml(merged_root)
    logger.info("MERGED: %d images, %d label files, %d boxes across %d classes",
                totals["images"], totals["labels"], totals["boxes"], len(CANONICAL_CLASSES))
    logger.info("data.yaml written to %s", merged_root / "data.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
