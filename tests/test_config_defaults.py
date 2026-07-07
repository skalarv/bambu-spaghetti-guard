"""Defaults must match what config.yaml and the docs promise.

config.yaml documents that only spaghetti + detachment fire the guard; a
code default that silently adds blob/failure makes a config with the key
deleted MORE trigger-happy than documented.
"""

from __future__ import annotations

from spaghetti_guard.config import DetectorConfig, SnapshotConfig


def test_failure_classes_default_matches_documented_set():
    cfg = DetectorConfig(model_path="weights.pt")
    assert cfg.failure_classes == ["spaghetti", "detachment"]


def test_snapshot_max_files_default_is_bounded():
    cfg = SnapshotConfig()
    assert cfg.max_files == 500
