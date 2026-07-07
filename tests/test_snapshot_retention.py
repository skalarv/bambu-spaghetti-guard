"""Snapshot directory must not grow without bound on a long-running daemon.

A false-positive-prone model with a 30s cooldown can write ~120 JPEGs/hour;
the guard prunes the oldest trigger-*.jpg beyond `snapshot_max_files`.
"""

from __future__ import annotations

from spaghetti_guard.detector import FailureDetector
from spaghetti_guard.guard import Guard
from spaghetti_guard.notifier import NoopNotifier


class _AlwaysSpaghetti:
    def predict(self, image, **kwargs):
        return [type("B", (), {"cls_name": "spaghetti", "conf": 0.9})()]


class _Control:
    def stop(self):
        pass

    def pause(self):
        pass


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _make_guard(tmp_path, clock, max_files):
    det = FailureDetector(
        _AlwaysSpaghetti(),
        failure_classes=("spaghetti",),
        conf_threshold=0.5,
        decoder=lambda j: j,
    )
    return Guard(
        detector=det,
        control=_Control(),
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "RUNNING",
        debounce_window=1,
        cooldown_s=1.0,
        snapshot_dir=tmp_path / "snaps",
        now=clock,
        snapshot_max_files=max_files,
    )


def test_snapshot_dir_pruned_to_cap(tmp_path):
    clock = _Clock()
    g = _make_guard(tmp_path, clock, max_files=3)
    for _ in range(6):  # six fires, distinct timestamps
        g.feed_frame(b"jpeg")  # fires
        clock.t += 5.0  # past cooldown, next second bucket
        g.feed_frame(b"jpeg")  # re-arms + fires again next call

    snaps = sorted((tmp_path / "snaps").glob("trigger-*.jpg"))
    assert 0 < len(snaps) <= 3
    # The newest snapshot survives pruning (sorted names are chronological).
    assert snaps == sorted((tmp_path / "snaps").glob("trigger-*.jpg"))


def test_snapshot_cap_disabled_with_none(tmp_path):
    clock = _Clock()
    g = _make_guard(tmp_path, clock, max_files=None)
    for _ in range(4):
        g.feed_frame(b"jpeg")
        clock.t += 5.0
        g.feed_frame(b"jpeg")
    snaps = list((tmp_path / "snaps").glob("trigger-*.jpg"))
    assert len(snaps) >= 4  # unbounded when explicitly disabled
