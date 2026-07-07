"""training/validate.py must read real metrics, never silently report zeros.

Ultralytics `model.val()` returns a DetMetrics whose numbers live under
`results.box.mp/.mr/.map50/.map` — not directly on the object. A defensive
getattr(results, "mp", 0.0) silently writes an all-zero summary.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from training.validate import main as validate_main


class _Box:
    mp = 0.86
    mr = 0.68
    map50 = 0.72
    map = 0.42


def _install_fake_ultralytics(monkeypatch, results_obj):
    class _YOLO:
        def __init__(self, path):
            self.path = path

        def val(self, **kwargs):
            return results_obj

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def test_validate_reads_metrics_from_detmetrics_box(tmp_path, monkeypatch):
    results = types.SimpleNamespace(box=_Box())
    _install_fake_ultralytics(monkeypatch, results)
    out = tmp_path / "summary.json"

    rc = validate_main(
        ["--weights", "w.pt", "--data", "d.yaml", "--summary-out", str(out)]
    )
    assert rc == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["precision"] == pytest.approx(0.86)
    assert summary["recall"] == pytest.approx(0.68)
    assert summary["map50"] == pytest.approx(0.72)
    assert summary["map50_95"] == pytest.approx(0.42)


def test_validate_gate_fails_below_floor(tmp_path, monkeypatch):
    """--min-* floors turn validate.py into a model regression gate: a
    retrained model that lost recall must fail CI, not ship silently."""
    results = types.SimpleNamespace(box=_Box())  # recall = 0.68
    _install_fake_ultralytics(monkeypatch, results)
    out = tmp_path / "summary.json"

    rc = validate_main(
        [
            "--weights", "w.pt", "--data", "d.yaml", "--summary-out", str(out),
            "--min-recall", "0.70",  # floor above the model's 0.68
        ]
    )
    assert rc == 6
    # The summary is still written so the failure is diagnosable.
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["recall"] == pytest.approx(0.68)
    assert summary["gate"]["passed"] is False


def test_validate_gate_passes_at_or_above_floor(tmp_path, monkeypatch):
    results = types.SimpleNamespace(box=_Box())
    _install_fake_ultralytics(monkeypatch, results)
    out = tmp_path / "summary.json"

    rc = validate_main(
        [
            "--weights", "w.pt", "--data", "d.yaml", "--summary-out", str(out),
            "--min-precision", "0.80",
            "--min-recall", "0.60",
            "--min-map50", "0.65",
        ]
    )
    assert rc == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["gate"]["passed"] is True


def test_validate_fails_loudly_on_unexpected_results_shape(tmp_path, monkeypatch):
    """No .box on the result → error out; an all-zero summary that looks like
    a catastrophically bad model is worse than a crash."""
    results = types.SimpleNamespace()  # no .box
    _install_fake_ultralytics(monkeypatch, results)
    out = tmp_path / "summary.json"

    rc = validate_main(
        ["--weights", "w.pt", "--data", "d.yaml", "--summary-out", str(out)]
    )
    assert rc != 0
    assert not out.exists()  # no misleading summary written
