"""Failure detector + N-of-N debounce (brief §5.4).

The model is injected at construction time so tests run without
`ultralytics` / `cv2` / GPU. A live caller uses :func:`load_yolo_model` to
get a real Ultralytics YOLO instance.

The debouncer is the primary false-positive defense per brief §3.3: a single
qualifying frame must never trigger; only N consecutive hits do.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols (so tests can inject fakes without importing ultralytics)
# ---------------------------------------------------------------------------


class DetectionBox(Protocol):
    cls_name: str
    conf: float


class YoloLike(Protocol):
    def predict(self, image: np.ndarray, **kwargs) -> list[Any]: ...


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_cv2():
    import cv2  # type: ignore

    return cv2


def load_yolo_model(model_path: str | Path) -> YoloLike:
    """Live-only entrypoint. Imports Ultralytics on demand.

    Tests should not call this; pass a fake YoloLike to ``FailureDetector``.
    """
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(model_path))


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    """Decode a JPEG payload to a BGR ndarray (live path uses cv2)."""
    cv2 = _import_cv2()
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None — payload not a valid JPEG")
    return img


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameResult:
    hit: bool
    conf: float
    best_class: str | None


class FailureDetector:
    """Wraps a YOLO-like model with a class+threshold filter."""

    def __init__(
        self,
        model: YoloLike,
        *,
        failure_classes: Iterable[str],
        conf_threshold: float,
        decoder=decode_jpeg,
    ) -> None:
        self._model = model
        self._failure_classes = frozenset(failure_classes)
        self._conf_threshold = conf_threshold
        self._decode = decoder

    def is_failure_frame(self, jpeg: bytes) -> FrameResult:
        image = self._decode(jpeg)
        predictions = self._model.predict(image, verbose=False)
        # Ultralytics returns a list of Results; each has a boxes object exposing
        # .cls (tensor of class indices) and .conf (tensor of confidences) and
        # .names mapping. To stay decoupled from that API, the model adapter
        # (or the test fake) is expected to surface a flat list of DetectionBox.
        boxes = _flatten_boxes(predictions)
        best_hit_conf = 0.0
        best_hit_class: str | None = None
        for box in boxes:
            if box.cls_name not in self._failure_classes:
                continue
            if box.conf < self._conf_threshold:
                continue
            if box.conf > best_hit_conf:
                best_hit_conf = box.conf
                best_hit_class = box.cls_name
        if best_hit_class is None:
            return FrameResult(hit=False, conf=0.0, best_class=None)
        return FrameResult(hit=True, conf=best_hit_conf, best_class=best_hit_class)


@dataclass(frozen=True)
class _FlatBox:
    cls_name: str
    conf: float


def flatten_ultralytics_results(results) -> list[DetectionBox]:
    """Flatten an Ultralytics ``Results`` list into DetectionBox-shaped objects.

    Each Results item exposes ``.names`` (class-index -> name) and ``.boxes``
    with indexable ``.cls`` / ``.conf`` tensors.
    """
    out: list[DetectionBox] = []
    for r in results:
        names = getattr(r, "names", None) or {}
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for i in range(len(boxes)):
            try:
                cls_idx = int(boxes.cls[i])
                conf = float(boxes.conf[i])
            except (IndexError, TypeError, ValueError):
                continue
            out.append(_FlatBox(names.get(cls_idx, str(cls_idx)), conf))
    return out


def _flatten_boxes(predictions) -> list[DetectionBox]:
    """Accept the test-fake shape (flat DetectionBox-like objects) or a real
    Ultralytics Results list; anything else raises.

    Raising matters: a prediction shape we can't read must never degrade to
    "no detections" — a blind guard that reports healthy is the worst
    failure mode this module can have.
    """
    if not predictions:
        return []
    first = predictions[0]
    if hasattr(first, "cls_name") and hasattr(first, "conf"):
        return list(predictions)
    if hasattr(first, "boxes"):
        return flatten_ultralytics_results(predictions)
    raise ValueError(
        f"unrecognized prediction shape: {type(first).__name__} — "
        f"expected DetectionBox-like objects or Ultralytics Results"
    )


# ---------------------------------------------------------------------------
# Debouncer
# ---------------------------------------------------------------------------


class Debouncer:
    """N-of-N rolling debounce.

    `update(hit)` appends a frame outcome. `confirmed()` is True iff the
    last N updates were all hits. Any miss inside the window resets immediately
    so a single clean frame cancels the alert.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("debounce window must be >= 1")
        self._window = window
        self._buf: deque[bool] = deque(maxlen=window)

    @property
    def window(self) -> int:
        return self._window

    def reset(self) -> None:
        self._buf.clear()

    def update(self, hit: bool) -> None:
        if not hit:
            self._buf.clear()
            return
        self._buf.append(True)

    def confirmed(self) -> bool:
        return len(self._buf) >= self._window and all(self._buf)

    def streak(self) -> int:
        return len(self._buf)
