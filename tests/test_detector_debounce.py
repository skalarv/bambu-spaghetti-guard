"""Brief §6.4: detector class/threshold filter, N-of-N debouncer behavior."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from spaghetti_guard.detector import (
    Debouncer,
    FailureDetector,
    _flatten_boxes,
    decode_jpeg,
    load_yolo_model,
)


@dataclass
class FakeBox:
    cls_name: str
    conf: float


class FakeYolo:
    def __init__(self, boxes_per_call):
        # boxes_per_call: list of (list of FakeBox) — one entry per predict() call
        self._boxes_per_call = list(boxes_per_call)
        self._idx = 0

    def predict(self, image, **kwargs):
        out = self._boxes_per_call[self._idx]
        self._idx += 1
        return out


def _identity_decoder(jpeg):
    return jpeg  # detector doesn't touch the image bytes in tests


def _detector(boxes_per_call, *, threshold=0.5, classes=("spaghetti", "blob")):
    return FailureDetector(
        FakeYolo(boxes_per_call),
        failure_classes=classes,
        conf_threshold=threshold,
        decoder=_identity_decoder,
    )


# ---- Ultralytics Results handled natively ---------------------------------
# The live path (FailureDetector(load_yolo_model(...))) must work exactly as
# the module docstring promises — no separate adapter, no silent no-op.


class _UltraBoxes:
    """Shape of ultralytics Results.boxes: indexable .cls / .conf tensors."""

    def __init__(self, cls, conf):
        self.cls = cls
        self.conf = conf

    def __len__(self):
        return len(self.cls)


class _UltraResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def test_detector_flattens_ultralytics_results_natively():
    result = _UltraResult(
        names={0: "spaghetti", 1: "blob"},
        boxes=_UltraBoxes(cls=[0, 1], conf=[0.9, 0.4]),
    )
    det = _detector([[result]])
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is True
    assert r.best_class == "spaghetti"
    assert r.conf == 0.9


def test_detector_ultralytics_result_without_boxes_is_clean_frame():
    result = _UltraResult(names={}, boxes=None)
    det = _detector([[result]])
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is False


def test_unknown_prediction_shape_raises():
    """A prediction shape the detector can't read must raise, not silently
    return 'no detections' — a blind safety guard that reports healthy is the
    worst failure mode."""

    class _Weird:
        pass

    det = _detector([[_Weird()]])
    with pytest.raises(ValueError, match="prediction shape"):
        det.is_failure_frame(b"jpeg")


# ---- single-frame detector -----------------------------------------------


def test_hit_when_class_matches_and_above_threshold():
    det = _detector([[FakeBox("spaghetti", 0.9)]])
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is True
    assert r.best_class == "spaghetti"
    assert r.conf == pytest.approx(0.9)


def test_no_hit_when_below_threshold():
    det = _detector([[FakeBox("spaghetti", 0.49)]], threshold=0.5)
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is False
    assert r.conf == 0.0


def test_class_not_in_failure_list_is_ignored():
    det = _detector([[FakeBox("printer_part", 0.99)]])
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is False


def test_best_among_multiple_failure_boxes_wins():
    det = _detector([[FakeBox("blob", 0.6), FakeBox("spaghetti", 0.85), FakeBox("blob", 0.7)]])
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is True
    assert r.best_class == "spaghetti"
    assert r.conf == pytest.approx(0.85)


def test_empty_predictions_is_clean():
    det = _detector([[]])
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is False


# ---- debouncer -----------------------------------------------------------


def test_fires_exactly_at_n_not_earlier():
    d = Debouncer(window=4)
    for i in range(3):
        d.update(True)
        assert not d.confirmed(), f"premature fire after {i+1} hits"
    d.update(True)
    assert d.confirmed(), "should fire on the 4th consecutive hit"


def test_single_miss_resets():
    d = Debouncer(window=3)
    d.update(True)
    d.update(True)
    d.update(False)  # reset
    d.update(True)
    d.update(True)
    assert not d.confirmed()
    d.update(True)
    assert d.confirmed()


def test_misses_alone_never_fire():
    d = Debouncer(window=2)
    for _ in range(10):
        d.update(False)
        assert not d.confirmed()


def test_window_1_fires_immediately():
    d = Debouncer(window=1)
    d.update(True)
    assert d.confirmed()


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        Debouncer(window=0)


def test_reset_clears_state():
    d = Debouncer(window=2)
    d.update(True)
    d.update(True)
    assert d.confirmed()
    d.reset()
    assert not d.confirmed()
    d.update(True)
    assert not d.confirmed()


def test_streak_reports_current_run():
    d = Debouncer(window=5)
    assert d.streak() == 0
    d.update(True)
    d.update(True)
    assert d.streak() == 2
    d.update(False)
    assert d.streak() == 0


# ---- _flatten_boxes unit behavior ------------------------------------------


def test_flatten_boxes_empty_predictions_is_empty_list():
    assert _flatten_boxes([]) == []


def test_flatten_boxes_unknown_shape_raises():
    """No cls_name / boxes attribute -> must raise, never silently return []."""

    class _X:
        pass

    with pytest.raises(ValueError):
        _flatten_boxes([_X()])


# ---- live-path helpers (cv2 / ultralytics mocked) ---------------------------


def test_decode_jpeg_invalid_payload_raises(monkeypatch):
    class FakeCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imdecode(arr, flag):
            return None  # simulate "not a jpeg"

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    with pytest.raises(ValueError):
        decode_jpeg(b"not-a-jpeg")


def test_decode_jpeg_returns_decoded_image(monkeypatch):
    import numpy as np

    class FakeCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imdecode(arr, flag):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    img = decode_jpeg(b"fake")
    assert img.shape == (10, 10, 3)


def test_load_yolo_model_imports_ultralytics_lazily(monkeypatch):
    """load_yolo_model is the only place ultralytics is imported live."""

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

    fake_module = MagicMock()
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    m = load_yolo_model("x.pt")
    assert isinstance(m, FakeYOLO)
    assert m.path == "x.pt"
