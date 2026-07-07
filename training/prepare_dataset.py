"""Wire labeled datasets into the Ultralytics YOLO format.

Brief §7.1. This script is a *scaffold*: it validates the directory layout,
emits a `data.yaml`, and reports class-balance statistics. It does not
download data, relabel, or augment — that's a manual / per-dataset step
documented in `datasets.md`.

Usage:
    py prepare_dataset.py --root training/data --classes spaghetti detachment blob failure

Exits non-zero if license / source files are missing — provenance is a
hard requirement (see `datasets.md`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("prepare_dataset")

REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)
PROVENANCE_FILES = ("LICENSE.txt", "SOURCES.md")


@dataclass
class SplitStats:
    split: str
    image_count: int
    label_count: int
    class_counts: Counter

    def __str__(self) -> str:
        cls_str = ", ".join(f"{c}:{n}" for c, n in sorted(self.class_counts.items()))
        return f"{self.split}: {self.image_count} images, {self.label_count} label files, {cls_str}"


def _scan_split(root: Path, split: str) -> SplitStats:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"split {split!r} missing images/ or labels/ at {root}")

    images = sorted(p for p in img_dir.glob("*.jpg")) + sorted(p for p in img_dir.glob("*.jpeg"))
    labels = sorted(p for p in lbl_dir.glob("*.txt"))

    cls_counts: Counter = Counter()
    for lbl in labels:
        try:
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                cls_counts[cls_id] += 1
        except (UnicodeDecodeError, ValueError):
            logger.warning("could not parse %s; skipping", lbl)
            continue

    return SplitStats(split=split, image_count=len(images), label_count=len(labels), class_counts=cls_counts)


def _check_provenance(root: Path) -> list[str]:
    missing = []
    for name in PROVENANCE_FILES:
        if not (root / name).exists():
            missing.append(str(root / name))
    return missing


def write_data_yaml(root: Path, class_names: list[str], out_path: Path | None = None) -> Path:
    out_path = out_path or (root / "data.yaml")
    body = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    out_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="dataset root (contains train/ val/ ...)")
    p.add_argument(
        "--classes",
        nargs="+",
        default=["spaghetti", "detachment", "blob", "failure"],
    )
    p.add_argument("--ignore-provenance", action="store_true", help="DO NOT use for shipped models")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    root: Path = args.root
    if not root.is_dir():
        logger.error("dataset root not found: %s", root)
        return 2

    missing = _check_provenance(root)
    if missing and not args.ignore_provenance:
        logger.error("missing provenance files: %s", missing)
        logger.error("training without LICENSE.txt + SOURCES.md is blocked. See training/datasets.md.")
        return 3

    stats = []
    for split in REQUIRED_SPLITS:
        stats.append(_scan_split(root, split))
    for split in OPTIONAL_SPLITS:
        if (root / split).is_dir():
            stats.append(_scan_split(root, split))

    for s in stats:
        logger.info(str(s))

    data_yaml = write_data_yaml(root, args.classes)
    logger.info("wrote %s", data_yaml)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
