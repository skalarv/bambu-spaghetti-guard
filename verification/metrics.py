"""Detection-quality metrics for labeled chamber clips.

Brief §6.3. Computes:

* Detection latency (frames AND seconds) from the true failure onset to the
  frame where the debouncer would have fired.
* Frame-level precision / recall over the failure class.
* False-positive rate per print-hour on labeled clean footage — the
  operational metric that decides whether a guard is shippable.
* Threshold sweep that proposes an operating point.

Labels for a clip come from a sidecar JSON next to the frames folder, schema:

    {
        "kind": "spaghetti" | "clean",
        "failure_onset_frame": int | null,
        "fps": float,
        "frames": [{"index": int, "is_failure": bool}, ...]  # per-frame labels
    }

Per-frame `is_failure` is optional; if absent, every frame from
`failure_onset_frame` onward is treated as failure (clean clips have no
failure frames at all).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable

from spaghetti_guard.detector import FailureDetector

from .replay_harness import (
    ReplayReport,
    _build_marker_detector,
    iter_jpegs_from_folder,
    replay,
)

logger = logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------


@dataclass
class ClipLabels:
    kind: str  # "spaghetti" | "clean"
    failure_onset_frame: int | None
    fps: float
    per_frame: dict[int, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ClipLabels:
        data = json.loads(path.read_text(encoding="utf-8"))
        per_frame = {
            int(row["index"]): bool(row["is_failure"]) for row in data.get("frames", [])
        }
        return cls(
            kind=data["kind"],
            failure_onset_frame=data.get("failure_onset_frame"),
            fps=float(data["fps"]),
            per_frame=per_frame,
        )

    def is_failure_at(self, index: int) -> bool:
        if index in self.per_frame:
            return self.per_frame[index]
        if self.kind == "clean":
            return False
        if self.failure_onset_frame is None:
            return False
        return index >= self.failure_onset_frame


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


@dataclass
class ClipMetrics:
    clip: str
    kind: str
    frame_count: int
    fps: float
    confusion: ConfusionMatrix
    latency_frames: int | None
    latency_s: float | None
    false_alerts: int
    # Footage where a fire would count as a false alert: the whole clip for
    # clean clips, the pre-onset segment for spaghetti clips. This is the
    # denominator footage for fp_per_print_hour.
    non_failure_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "clip": self.clip,
            "kind": self.kind,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "confusion": self.confusion.__dict__,
            "precision": self.confusion.precision,
            "recall": self.confusion.recall,
            "f1": self.confusion.f1,
            "latency_frames": self.latency_frames,
            "latency_s": self.latency_s,
            "false_alerts": self.false_alerts,
            "non_failure_seconds": self.non_failure_seconds,
        }


@dataclass
class AggregateMetrics:
    clips: list[ClipMetrics]
    fp_per_print_hour: float
    avg_latency_s: float | None

    def to_dict(self) -> dict:
        return {
            "fp_per_print_hour": self.fp_per_print_hour,
            "avg_latency_s": self.avg_latency_s,
            "clips": [c.to_dict() for c in self.clips],
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def evaluate_clip(report: ReplayReport, labels: ClipLabels) -> ClipMetrics:
    cm = ConfusionMatrix()
    for row in report.rows:
        truth = labels.is_failure_at(row.index)
        pred = row.hit
        if pred and truth:
            cm.tp += 1
        elif pred and not truth:
            cm.fp += 1
        elif not pred and truth:
            cm.fn += 1
        else:
            cm.tn += 1

    latency_frames: int | None = None
    latency_s: float | None = None
    if labels.kind == "spaghetti" and labels.failure_onset_frame is not None:
        for idx in report.fired_indices:
            if idx >= labels.failure_onset_frame:
                latency_frames = idx - labels.failure_onset_frame
                latency_s = latency_frames / labels.fps if labels.fps > 0 else None
                break

    # False alerts on clean clips: every fire is a false alert; on spaghetti
    # clips, fires before the onset are false alerts.
    false_alerts = 0
    for idx in report.fired_indices:
        if labels.kind == "clean" or labels.failure_onset_frame is not None and idx < labels.failure_onset_frame:
            false_alerts += 1

    # Footage on which the false alerts above were counted.
    if labels.fps > 0:
        if labels.kind == "clean":
            non_failure_seconds = report.frame_count / labels.fps
        elif labels.failure_onset_frame is not None:
            non_failure_seconds = labels.failure_onset_frame / labels.fps
        else:
            non_failure_seconds = 0.0
    else:
        non_failure_seconds = 0.0

    return ClipMetrics(
        clip=report.clip,
        kind=labels.kind,
        frame_count=report.frame_count,
        fps=labels.fps,
        confusion=cm,
        latency_frames=latency_frames,
        latency_s=latency_s,
        false_alerts=false_alerts,
        non_failure_seconds=non_failure_seconds,
    )


def aggregate(clip_metrics: Iterable[ClipMetrics]) -> AggregateMetrics:
    clips = list(clip_metrics)
    total_false_alerts = sum(c.false_alerts for c in clips)
    # Numerator and denominator cover the same footage: everything where a
    # fire would have been a false alert (clean clips + pre-onset segments).
    total_non_failure_seconds = sum(c.non_failure_seconds for c in clips)
    fp_per_hour = (
        (total_false_alerts / total_non_failure_seconds * 3600.0)
        if total_non_failure_seconds
        else 0.0
    )

    latencies = [c.latency_s for c in clips if c.latency_s is not None]
    avg_lat = sum(latencies) / len(latencies) if latencies else None

    return AggregateMetrics(
        clips=clips,
        fp_per_print_hour=fp_per_hour,
        avg_latency_s=avg_lat,
    )


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    conf_threshold: float
    window: int
    aggregate: dict


def threshold_sweep(
    detector_factory,
    clips: list[tuple[Path, Path]],  # (frames_dir, labels_json)
    *,
    thresholds: list[float],
    windows: list[int],
) -> list[SweepPoint]:
    """For each (threshold, window) combination, run replay + evaluate."""
    points: list[SweepPoint] = []
    for thr in thresholds:
        for w in windows:
            detector = detector_factory(thr)
            clip_metrics: list[ClipMetrics] = []
            for frames_dir, labels_json in clips:
                frames = list(iter_jpegs_from_folder(frames_dir))
                report = replay(frames, detector, debounce_window=w, clip_label=str(frames_dir))
                labels = ClipLabels.load(labels_json)
                clip_metrics.append(evaluate_clip(report, labels))
            agg = aggregate(clip_metrics)
            points.append(
                SweepPoint(conf_threshold=thr, window=w, aggregate=agg.to_dict())
            )
    return points


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detection-quality metrics over labeled clips.")
    p.add_argument(
        "clip_dirs",
        type=Path,
        nargs="+",
        help="One or more clip folders; each must contain a sibling 'labels.json'.",
    )
    p.add_argument("--conf", type=float, default=0.55)
    p.add_argument("--window", type=int, default=6)
    p.add_argument(
        "--classes",
        nargs="+",
        default=["spaghetti", "detachment", "blob", "failure"],
    )
    p.add_argument("--sweep", action="store_true", help="run a small threshold sweep")
    p.add_argument("--json-out", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    clips: list[tuple[Path, Path]] = []
    for clip_dir in args.clip_dirs:
        labels_path = clip_dir / "labels.json"
        if not labels_path.exists():
            logger.error("missing labels at %s", labels_path)
            return 2
        clips.append((clip_dir, labels_path))

    def factory(thr: float) -> FailureDetector:
        return _build_marker_detector(
            failure_classes=args.classes, conf_threshold=thr
        )

    if args.sweep:
        sweep = threshold_sweep(
            factory,
            clips,
            thresholds=[0.4, 0.5, 0.6, 0.7, 0.8],
            windows=[3, 6, 9],
        )
        result = {"sweep": [p.__dict__ for p in sweep]}
    else:
        clip_metrics: list[ClipMetrics] = []
        detector = factory(args.conf)
        for frames_dir, labels_json in clips:
            frames = list(iter_jpegs_from_folder(frames_dir))
            report = replay(frames, detector, debounce_window=args.window, clip_label=str(frames_dir))
            labels = ClipLabels.load(labels_json)
            clip_metrics.append(evaluate_clip(report, labels))
        agg = aggregate(clip_metrics)
        result = agg.to_dict()

    print(json.dumps(result, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
