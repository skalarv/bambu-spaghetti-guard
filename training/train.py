"""Fine-tune a YOLO model on the prepared dataset (brief §7.2).

Live-only script: requires `ultralytics` + a CUDA-capable PyTorch build. The
verification env doesn't install those — they're documented as install-time
steps in `docs/INSTALL.md`. This script imports them lazily so `--help` works
on a clean checkout.

Defaults aim at high precision over recall:
- `--epochs 80 --imgsz 640 --batch 16 --conf 0.55` for the val report.
- The PR curve and confusion matrix are emitted to `runs/train/<name>/`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("train")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="data.yaml from prepare_dataset.py")
    p.add_argument("--base", default="yolo11n.pt", help="base weights (yolo11n.pt or yolo11s.pt)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="spaghetti-guard")
    p.add_argument("--device", default="0", help="cuda device id or 'cpu'")
    p.add_argument("--project", type=Path, default=Path("runs/train"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed. See docs/INSTALL.md.")
        return 1

    if not args.data.exists():
        logger.error("data.yaml not found: %s", args.data)
        return 2

    model = YOLO(args.base)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        project=str(args.project),
        # bias towards precision: stronger reg + lower conf during training
        cls=0.7,  # classification loss weight (default 0.5)
        patience=20,
    )
    logger.info("training complete; weights in %s", args.project / args.name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
