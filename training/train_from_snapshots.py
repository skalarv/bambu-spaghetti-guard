"""Retrain the spaghetti detector from operator-reviewed failure snapshots.

Brief §7.3 retraining loop. The guard writes `trigger-*.jpg` files under
`failure_snapshots/` every time it fires. This script:

1. Enumerates snapshots and their operator-provided YOLO labels
   (`trigger-*.txt` sitting next to the jpeg).
2. Categorises each snapshot:
   - **labelled true-positive** — has a matching label file with >=1 box
     -> promoted to the training set as fresh chamber-specific data
   - **labelled false-positive** — has an empty label file (0 boxes)
     -> promoted as a *negative* example (empty label = "no failure here")
   - **unreviewed** — no label file present -> printed to the operator to
     label with an annotation tool (`labelImg`, Roboflow web UI, etc.)
3. Copies the reviewed set into a per-batch source directory under
   `training/data/chamber-<batch>/` with a matching YOLO layout, then invokes
   `merge_datasets.py` so the next training run consumes it alongside the
   public Roboflow datasets.
4. Optionally kicks off `train.py` on the merged dataset.

Snapshot filename convention (from `guard._save_snapshot`):
    trigger-YYYYMMDD-HHMMSS-<class>-<conf>.jpg

Companion label file (operator-produced):
    trigger-YYYYMMDD-HHMMSS-<class>-<conf>.txt   # YOLO format lines

Empty label file = "no failure — this was a false positive; use as negative".

Run:
    python training/train_from_snapshots.py --review        # just report status
    python training/train_from_snapshots.py --promote       # copy into dataset
    python training/train_from_snapshots.py --promote --train
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("retrain")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "failure_snapshots"
DEFAULT_DATA_ROOT = REPO_ROOT / "training" / "data"


@dataclass
class ReviewedSnapshot:
    jpeg: Path
    label: Path | None
    boxes: int  # 0 = false-positive negative example, >0 = true positive


def _count_yolo_boxes(label_path: Path) -> int:
    try:
        text = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def enumerate_snapshots(snapshot_dir: Path) -> tuple[list[ReviewedSnapshot], list[Path]]:
    """Return (reviewed, unreviewed). `reviewed` includes both true-positives
    (>=1 box) and labelled negatives (0 boxes)."""
    reviewed: list[ReviewedSnapshot] = []
    unreviewed: list[Path] = []
    if not snapshot_dir.is_dir():
        return reviewed, unreviewed
    for jpeg in sorted(snapshot_dir.glob("trigger-*.jpg")):
        label = jpeg.with_suffix(".txt")
        if label.exists():
            reviewed.append(ReviewedSnapshot(jpeg=jpeg, label=label, boxes=_count_yolo_boxes(label)))
        else:
            unreviewed.append(jpeg)
    return reviewed, unreviewed


def _emit_review_report(reviewed: list[ReviewedSnapshot], unreviewed: list[Path]) -> None:
    tp = sum(1 for r in reviewed if r.boxes > 0)
    fp = sum(1 for r in reviewed if r.boxes == 0)
    print(f"snapshots: {len(reviewed) + len(unreviewed)} total")
    print(f"  reviewed true-positives (>=1 box): {tp}")
    print(f"  reviewed false-positives (empty label): {fp}")
    print(f"  unreviewed:                          {len(unreviewed)}")
    if unreviewed:
        print()
        print("To label the unreviewed snapshots, open the folder in labelImg / Roboflow /")
        print("your annotator of choice and save YOLO-format .txt files next to each .jpg.")
        print("An EMPTY .txt file is meaningful — it marks the snapshot as a false positive.")
        print()
        for p in unreviewed[:20]:
            print(f"  unlabelled: {p.name}")
        if len(unreviewed) > 20:
            print(f"  ... and {len(unreviewed) - 20} more")


def promote_to_dataset(
    reviewed: list[ReviewedSnapshot],
    data_root: Path,
    *,
    batch_tag: str,
    val_fraction: float = 0.15,
) -> Path:
    """Copy reviewed snapshots into `data_root/chamber-<batch_tag>/` in the
    Roboflow YOLOv8 layout (train/valid + images/labels). Returns the batch root."""
    batch_root = data_root / f"chamber-{batch_tag}"
    if batch_root.exists():
        raise FileExistsError(f"batch dir already exists: {batch_root}")
    (batch_root / "train" / "images").mkdir(parents=True)
    (batch_root / "train" / "labels").mkdir(parents=True)
    (batch_root / "valid" / "images").mkdir(parents=True)
    (batch_root / "valid" / "labels").mkdir(parents=True)

    # Deterministic split: every ceil(1/val_fraction)th item goes to valid.
    stride = max(2, int(round(1 / val_fraction)))
    for i, r in enumerate(reviewed):
        split = "valid" if i % stride == 0 else "train"
        img_dst = batch_root / split / "images" / r.jpeg.name
        lbl_dst = batch_root / split / "labels" / (r.jpeg.stem + ".txt")
        shutil.copy2(r.jpeg, img_dst)
        if r.label is not None:
            shutil.copy2(r.label, lbl_dst)
        else:
            lbl_dst.write_text("", encoding="utf-8")

    # Provenance so prepare_dataset.py doesn't refuse.
    (batch_root / "LICENSE.txt").write_text(
        "Chamber footage captured by the local Bambu spaghetti guard; operator-annotated.\n"
        "Not licensed for external redistribution without operator permission.\n",
        encoding="utf-8",
    )
    (batch_root / "SOURCES.md").write_text(
        f"# chamber-{batch_tag}\n\n"
        f"Captured from failure_snapshots/ on {time.strftime('%Y-%m-%d')}.\n"
        f"Reviewed and labelled by the operator using labelImg / Roboflow.\n",
        encoding="utf-8",
    )
    # Minimal data.yaml — the merge script rewrites everything anyway; this
    # lets the batch stand alone for smoke-testing if desired.
    (batch_root / "data.yaml").write_text(
        "path: {}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 7\n"
        "names: [spaghetti, stringing, blob, crack, detachment, over_extrusion, under_extrusion]\n".format(
            batch_root.resolve()
        ),
        encoding="utf-8",
    )
    return batch_root


def _invoke_merge(data_root: Path) -> int:
    """Re-run merge_datasets.py to fold the new batch into the training set."""
    cmd = [sys.executable, str(REPO_ROOT / "training" / "merge_datasets.py")]
    logger.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _invoke_train(data_yaml: Path, base: str, epochs: int, batch: int, name: str) -> int:
    cmd = [
        sys.executable, str(REPO_ROOT / "training" / "train.py"),
        "--data", str(data_yaml),
        "--base", base,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--name", name,
        "--device", "0",
    ]
    logger.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--review", action="store_true", help="just print status, no side effects")
    p.add_argument("--promote", action="store_true", help="copy reviewed snapshots into training/data/chamber-*/")
    p.add_argument("--train", action="store_true", help="after promote, re-merge + re-train")
    p.add_argument("--base", default="yolo26n.pt", help="base weights for retraining")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batch-tag", default=time.strftime("%Y%m%d"), help="tag for chamber-<tag> dir")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    reviewed, unreviewed = enumerate_snapshots(args.snapshots)
    _emit_review_report(reviewed, unreviewed)

    if args.review or not (args.promote or args.train):
        return 0

    if not reviewed:
        logger.error("no reviewed snapshots to promote")
        return 2

    if args.promote:
        batch_root = promote_to_dataset(reviewed, args.data_root, batch_tag=args.batch_tag)
        logger.info("promoted %d snapshots into %s", len(reviewed), batch_root)
        # NOTE: merge_datasets.py currently doesn't auto-pick chamber-* dirs.
        # Add them to `build_sources()` (or extend the script to auto-scan).
        # For now, print a nudge so the operator does it once.
        print()
        print(f"NEXT: add chamber-{args.batch_tag} to build_sources() in merge_datasets.py,")
        print("then re-run merge + train:")
        print(f"  python training/merge_datasets.py")
        print(f"  python training/train.py --data training/data/merged/data.yaml --base {args.base} \\")
        print(f"      --epochs {args.epochs} --batch {args.batch} --name chamber-{args.batch_tag}")
        return 0

    if args.train:
        rc = _invoke_merge(args.data_root)
        if rc != 0:
            return rc
        merged_yaml = args.data_root / "merged" / "data.yaml"
        return _invoke_train(merged_yaml, args.base, args.epochs, args.batch, f"chamber-{args.batch_tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
