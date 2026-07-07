"""Validate fine-tuned weights and emit the brief's required reports (§7.2).

Outputs:
* Per-class PR curve + confusion matrix (Ultralytics defaults).
* JSON summary with precision / recall / mAP per class.
* FP-per-print-hour estimate computed by replaying the held-out clean clips
  through `verification.metrics`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("validate")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--clean-clips", type=Path, nargs="*", default=[],
                   help="optional held-out clean-clip folders for FP/hour estimate")
    p.add_argument("--summary-out", type=Path, default=Path("runs/validate/summary.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed. See docs/INSTALL.md.")
        return 1

    model = YOLO(str(args.weights))
    results = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    # Ultralytics val() returns DetMetrics: the numbers live under results.box
    # (mp/mr/map50/map). Fail loudly on anything else — an all-zero summary
    # that looks like a catastrophically bad model is worse than a crash.
    box = getattr(results, "box", None)
    if box is None:
        logger.error(
            "unexpected val() result shape %s (no .box metrics) — "
            "Ultralytics API change? Refusing to write a zeroed summary.",
            type(results).__name__,
        )
        return 2
    summary: dict = {
        "weights": str(args.weights),
        "data": str(args.data),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }

    if args.clean_clips:
        # Lazy import so the verification module isn't required for `validate.py --help`.
        from verification.metrics import aggregate, evaluate_clip, ClipLabels
        from verification.replay_harness import _build_yolo_detector, iter_jpegs_from_folder, replay

        detector = _build_yolo_detector(
            args.weights, failure_classes=["spaghetti", "detachment", "blob", "failure"], conf_threshold=0.55
        )
        clip_metrics = []
        for clip_dir in args.clean_clips:
            labels_path = clip_dir / "labels.json"
            frames = list(iter_jpegs_from_folder(clip_dir))
            rep = replay(frames, detector, debounce_window=6, clip_label=str(clip_dir))
            labels = ClipLabels.load(labels_path)
            clip_metrics.append(evaluate_clip(rep, labels))
        agg = aggregate(clip_metrics)
        summary["fp_per_print_hour"] = agg.fp_per_print_hour

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.summary_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
