"""Smoke + regression for `verification.replay_harness` and `verification.metrics`.

The marker-based detector is deterministic, so we can assert exact frame
indices and fire/no-fire counts on synthesised clips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verification.metrics import (
    ClipLabels,
    aggregate,
    evaluate_clip,
    threshold_sweep,
)
from verification.mock_printer import MARKER_CLEAN, MARKER_FAILURE, make_jpeg
from verification.replay_harness import (
    _build_marker_detector,
    iter_jpegs_from_folder,
    replay,
)


# ---- fixture builders ----------------------------------------------------


def _write_jpeg(folder: Path, idx: int, marker: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"frame-{idx:05d}.jpg"
    path.write_bytes(make_jpeg(marker))
    return path


def _make_clip(
    folder: Path,
    *,
    kind: str,
    n_frames: int,
    onset: int | None = None,
    fps: float = 1.0,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_frames):
        is_fail = kind == "spaghetti" and onset is not None and i >= onset
        marker = MARKER_FAILURE if is_fail else MARKER_CLEAN
        _write_jpeg(folder, i, marker)
        rows.append({"index": i, "is_failure": is_fail})
    labels_path = folder / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": kind,
                "failure_onset_frame": onset,
                "fps": fps,
                "frames": rows,
            }
        ),
        encoding="utf-8",
    )
    return folder


# ---- replay --------------------------------------------------------------


def test_replay_fires_after_window_on_spaghetti(tmp_path):
    clip = _make_clip(tmp_path / "spag", kind="spaghetti", n_frames=20, onset=10)
    det = _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    rep = replay(iter_jpegs_from_folder(clip), det, debounce_window=3, clip_label="t")
    # First fire happens at onset + (window - 1) — onset=10, window=3 -> idx 12.
    assert rep.fired_indices == [12]
    # Streak crosses through 1,2,3 on frames 10,11,12
    assert rep.rows[10].streak_after == 1
    assert rep.rows[11].streak_after == 2
    assert rep.rows[12].would_fire is True


def test_replay_silent_on_clean(tmp_path):
    clip = _make_clip(tmp_path / "clean", kind="clean", n_frames=30, onset=None)
    det = _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    rep = replay(iter_jpegs_from_folder(clip), det, debounce_window=3, clip_label="t")
    assert rep.fired_indices == []
    assert all(not r.hit for r in rep.rows)


def test_replay_one_off_failure_does_not_fire(tmp_path):
    """A single failure-marker frame in a clean clip must not fire (debounce works)."""
    clip = tmp_path / "blip"
    clip.mkdir()
    for i in range(15):
        marker = MARKER_FAILURE if i == 7 else MARKER_CLEAN
        _write_jpeg(clip, i, marker)
    (clip / "labels.json").write_text(
        json.dumps({"kind": "clean", "failure_onset_frame": None, "fps": 1.0, "frames": []})
    )
    det = _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    rep = replay(iter_jpegs_from_folder(clip), det, debounce_window=3, clip_label="t")
    assert rep.fired_indices == []


# ---- metrics -------------------------------------------------------------


def test_evaluate_spaghetti_clip(tmp_path):
    clip = _make_clip(tmp_path / "spag", kind="spaghetti", n_frames=20, onset=10, fps=2.0)
    det = _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    rep = replay(iter_jpegs_from_folder(clip), det, debounce_window=3, clip_label="s1")
    labels = ClipLabels.load(clip / "labels.json")
    m = evaluate_clip(rep, labels)
    # 10 clean frames + 10 failure frames; detector mirrors truth perfectly
    assert m.confusion.tp == 10
    assert m.confusion.tn == 10
    assert m.confusion.fp == 0
    assert m.confusion.fn == 0
    assert m.confusion.precision == pytest.approx(1.0)
    assert m.confusion.recall == pytest.approx(1.0)
    # Fire at frame 12 (onset 10 + window 3 - 1) -> latency 2 frames, 1.0 s
    assert m.latency_frames == 2
    assert m.latency_s == pytest.approx(1.0)
    assert m.false_alerts == 0


def test_evaluate_clean_clip_with_no_fires(tmp_path):
    clip = _make_clip(tmp_path / "clean", kind="clean", n_frames=40, onset=None, fps=1.0)
    det = _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    rep = replay(iter_jpegs_from_folder(clip), det, debounce_window=3, clip_label="c1")
    labels = ClipLabels.load(clip / "labels.json")
    m = evaluate_clip(rep, labels)
    assert m.false_alerts == 0
    assert m.latency_frames is None
    assert m.confusion.tn == 40


def test_aggregate_fp_per_hour(tmp_path):
    """One clean clip with one engineered false alert should land at a sensible rate."""
    # Build a clean clip where 3 consecutive frames are mislabeled as failure
    # markers — that creates a single fire in a clean-labeled clip.
    clip = tmp_path / "mostly-clean"
    clip.mkdir()
    n = 60  # 60 frames at 1 fps = 60 s = 1/60 hour
    for i in range(n):
        marker = MARKER_FAILURE if 20 <= i < 23 else MARKER_CLEAN
        _write_jpeg(clip, i, marker)
    (clip / "labels.json").write_text(
        json.dumps({"kind": "clean", "failure_onset_frame": None, "fps": 1.0, "frames": []})
    )
    det = _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    rep = replay(iter_jpegs_from_folder(clip), det, debounce_window=3, clip_label="mc")
    labels = ClipLabels.load(clip / "labels.json")
    m = evaluate_clip(rep, labels)
    agg = aggregate([m])
    # 1 false alert in 60 s = 60 per hour
    assert agg.fp_per_print_hour == pytest.approx(60.0)


def test_threshold_sweep_smoke(tmp_path):
    spag = _make_clip(tmp_path / "s", kind="spaghetti", n_frames=20, onset=10, fps=1.0)
    clean = _make_clip(tmp_path / "c", kind="clean", n_frames=20, onset=None, fps=1.0)
    points = threshold_sweep(
        lambda thr: _build_marker_detector(failure_classes=("spaghetti",), conf_threshold=thr),
        [(spag, spag / "labels.json"), (clean, clean / "labels.json")],
        thresholds=[0.5, 0.7],
        windows=[3, 6],
    )
    assert len(points) == 4  # 2 thresholds x 2 windows
    for pt in points:
        assert "fp_per_print_hour" in pt.aggregate
        assert isinstance(pt.aggregate["clips"], list)


# ---- CLI smoke -----------------------------------------------------------


def test_replay_cli_writes_json(tmp_path):
    from verification.replay_harness import main

    clip = _make_clip(tmp_path / "spag", kind="spaghetti", n_frames=15, onset=5)
    out = tmp_path / "report.json"
    rc = main([str(clip), "--quiet", "--json-out", str(out), "--window", "3"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["frame_count"] == 15
    assert data["fired_indices"] == [7]  # onset=5 + (window=3 - 1)


def test_metrics_cli_writes_json(tmp_path):
    from verification.metrics import main

    spag = _make_clip(tmp_path / "s", kind="spaghetti", n_frames=15, onset=5, fps=1.0)
    clean = _make_clip(tmp_path / "c", kind="clean", n_frames=20, onset=None, fps=1.0)
    out = tmp_path / "metrics.json"
    rc = main([str(spag), str(clean), "--window", "3", "--json-out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "fp_per_print_hour" in data
    assert len(data["clips"]) == 2
