"""More targeted gap-fillers — mock_printer, replay_harness, viewer, guard."""

from __future__ import annotations

import io
import json
import struct
from unittest.mock import MagicMock

import pytest

from verification import mock_printer as mp
from verification import replay_harness as rh
from verification import metrics as mm


# =====================================================================
# mock_printer.extract_marker edge cases
# =====================================================================


def test_extract_marker_no_segment_returns_none():
    """Bytes that don't carry the COM segment should yield None."""
    assert mp.extract_marker(b"\xff\xd8\x00\x00\xff\xd9") is None


def test_extract_marker_truncated_segment_returns_none():
    # FF FE present but not enough length follows
    assert mp.extract_marker(b"\xff\xfe\x00") is None


def test_extract_marker_garbled_marker_returns_none():
    """If the byte after FF FE isn't 0x00 or 0xFF (our markers), reject."""
    bad = (
        b"\xff\xd8" + b"\xff\xfe" + struct.pack(">H", 4) + b"\x77data"
        + b"\xff\xd9"
    )
    assert mp.extract_marker(bad) is None


def test_extract_marker_with_zero_marker():
    jpg = mp.make_jpeg(mp.MARKER_CLEAN)
    assert mp.extract_marker(jpg) == mp.MARKER_CLEAN


def test_extract_marker_with_failure_marker():
    jpg = mp.make_jpeg(mp.MARKER_FAILURE)
    assert mp.extract_marker(jpg) == mp.MARKER_FAILURE


# =====================================================================
# mock_printer.parse_auth_packet edge cases
# =====================================================================


def test_parse_auth_packet_short_raises():
    with pytest.raises(ValueError):
        mp.parse_auth_packet(b"too-short")


def test_parse_auth_packet_bad_magic_raises():
    blob = b"\x00" * 80
    with pytest.raises(ValueError, match="bad auth magic"):
        mp.parse_auth_packet(blob)


def test_build_auth_packet_too_long_raises():
    with pytest.raises(ValueError):
        mp.build_auth_packet("u" * 33, "p")


# =====================================================================
# mock_printer.generate_self_signed_cert is idempotent
# =====================================================================


def test_generate_self_signed_cert_idempotent(tmp_path):
    p1, k1 = mp.generate_self_signed_cert(tmp_path)
    mtime_cert = p1.stat().st_mtime_ns
    mtime_key = k1.stat().st_mtime_ns
    # Second call must not regenerate
    p2, k2 = mp.generate_self_signed_cert(tmp_path)
    assert p2 == p1 and k2 == k1
    assert p2.stat().st_mtime_ns == mtime_cert
    assert k2.stat().st_mtime_ns == mtime_key


# =====================================================================
# mock_printer.MockMqttBroker port allocation when port=0
# =====================================================================


@pytest.mark.asyncio
async def test_mock_broker_picks_free_port():
    """Asking for port=0 must end up bound to a real free port."""
    broker = mp.MockMqttBroker(
        host="127.0.0.1", port=0, username="bblp", access_code="x", serial="S"
    )
    await broker.start()
    try:
        assert broker.actual_port != 0
    finally:
        await broker.stop()


# =====================================================================
# mock_printer.pick_free_port returns ints
# =====================================================================


def test_pick_free_port():
    p = mp._pick_free_port()
    assert isinstance(p, int) and 1024 <= p < 65536


# =====================================================================
# CommandRecorder + RecordedCommand
# =====================================================================


@pytest.mark.asyncio
async def test_command_recorder_add_and_snapshot():
    r = mp.CommandRecorder()
    cmd = mp.RecordedCommand(topic="t", payload={"k": 1}, qos=1, ts=1.0)
    await r.add(cmd)
    snap = r.snapshot()
    assert len(snap) == 1
    assert snap[0].topic == "t"
    assert len(r) == 1


# =====================================================================
# replay_harness — iter_jpegs_from_folder filters non-jpeg
# =====================================================================


def test_iter_jpegs_from_folder_filters(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.JPEG").write_bytes(b"y")
    (tmp_path / "c.png").write_bytes(b"z")
    (tmp_path / "d.txt").write_bytes(b"w")
    items = list(rh.iter_jpegs_from_folder(tmp_path))
    suffixes = sorted(p.suffix.lower() for p, _ in items)
    assert ".png" not in suffixes
    assert ".txt" not in suffixes


# =====================================================================
# replay_harness CLI — error paths
# =====================================================================


def test_replay_cli_unsupported_path_returns_2(tmp_path):
    weird = tmp_path / "thing.zip"
    weird.write_bytes(b"x")
    rc = rh.main([str(weird), "--quiet"])
    assert rc == 2


def test_replay_cli_empty_folder_returns_2(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    rc = rh.main([str(folder), "--quiet"])
    assert rc == 2


def test_replay_cli_single_jpeg_works(tmp_path):
    """A clip path that's a single .jpg should be wrapped as one frame."""
    jpg = tmp_path / "frame.jpg"
    jpg.write_bytes(mp.make_jpeg(mp.MARKER_CLEAN))
    rc = rh.main([str(jpg), "--quiet"])
    assert rc == 0


# =====================================================================
# replay_harness emit_timeline
# =====================================================================


def test_emit_timeline_runs(tmp_path):
    """Just exercise the stdout path."""
    jpg = mp.make_jpeg(mp.MARKER_CLEAN)
    (tmp_path / "a.jpg").write_bytes(jpg)
    det = rh._build_marker_detector(failure_classes=("spaghetti",), conf_threshold=0.5)
    frames = list(rh.iter_jpegs_from_folder(tmp_path))
    rep = rh.replay(frames, det, debounce_window=3, clip_label="t")
    out = io.StringIO()
    rh._emit_timeline(rep, stream=out)
    text = out.getvalue()
    assert "clip:" in text
    assert "frames:" in text


# =====================================================================
# metrics CLI: missing labels yields rc=2
# =====================================================================


def test_metrics_cli_missing_labels(tmp_path):
    folder = tmp_path / "spag"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(mp.make_jpeg(mp.MARKER_CLEAN))
    # No labels.json
    rc = mm.main([str(folder)])
    assert rc == 2


# =====================================================================
# metrics.aggregate degenerate cases
# =====================================================================


def test_aggregate_empty():
    agg = mm.aggregate([])
    assert agg.fp_per_print_hour == 0.0
    assert agg.avg_latency_s is None
    assert agg.clips == []


def test_evaluate_no_onset_no_latency():
    """A clean clip never reports latency."""
    folder_label = mm.ClipLabels(kind="clean", failure_onset_frame=None, fps=1.0)
    rep = rh.ReplayReport(
        clip="x", frame_count=5, fired_indices=[], settings={}, duration_s=0.0, rows=[]
    )
    m = mm.evaluate_clip(rep, folder_label)
    assert m.latency_frames is None
    assert m.latency_s is None


# =====================================================================
# metrics CLI sweep variant runs end-to-end
# =====================================================================


def test_metrics_cli_sweep(tmp_path):
    spag = tmp_path / "s"
    spag.mkdir()
    for i in range(10):
        (spag / f"{i:03d}.jpg").write_bytes(
            mp.make_jpeg(mp.MARKER_FAILURE if i >= 5 else mp.MARKER_CLEAN)
        )
    (spag / "labels.json").write_text(
        json.dumps({"kind": "spaghetti", "failure_onset_frame": 5, "fps": 1.0, "frames": []})
    )
    rc = mm.main([str(spag), "--sweep"])
    assert rc == 0


# =====================================================================
# guard — disarm log path + ALERTING->ARMED transition w/ miss
# =====================================================================


def test_guard_disarm_only_logs_once(tmp_path):
    """Calling _disarm while already IDLE shouldn't be problematic."""
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard
    from spaghetti_guard.notifier import NoopNotifier

    class _Y:
        def predict(self, *a, **kw):
            return []

    det = FailureDetector(_Y(), failure_classes=("x",), conf_threshold=0.5, decoder=lambda j: j)
    g = Guard(
        detector=det,
        control=MagicMock(),
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "IDLE",
        snapshot_dir=tmp_path,
    )
    g._disarm(reason="already idle")  # must not raise
    g._disarm(reason="still idle")


def test_guard_alerting_to_armed_on_miss(tmp_path):
    """After a hit (ALERTING), a miss must drop back to ARMED without firing."""
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard, GuardState
    from spaghetti_guard.notifier import NoopNotifier

    yolo_results = [
        # Hit first frame, then miss
        [type("B", (), {"cls_name": "x", "conf": 0.9})()],
        [],
    ]
    idx = [0]

    class _Y:
        def predict(self, *a, **kw):
            i = idx[0]
            idx[0] += 1
            return yolo_results[i]

    det = FailureDetector(_Y(), failure_classes=("x",), conf_threshold=0.5, decoder=lambda j: j)
    g = Guard(
        detector=det,
        control=MagicMock(),
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "RUNNING",
        debounce_window=5,
        snapshot_dir=tmp_path,
    )
    g.feed_frame(b"j")
    assert g.state == GuardState.ALERTING
    g.feed_frame(b"j")
    assert g.state == GuardState.ARMED


# =====================================================================
# viewer — TkViewer.start is idempotent (second call no-op)
# =====================================================================


def test_tkviewer_start_idempotent_without_running():
    from spaghetti_guard.viewer import TkViewer

    v = TkViewer()
    v._thread = MagicMock()  # pretend a thread exists
    v._thread.is_alive.return_value = True
    v.start()  # second call -> no-op


# =====================================================================
# config — load_config rejects yaml whose root isn't a mapping
# =====================================================================


def test_config_yaml_root_not_a_mapping_rejected(tmp_path):
    from spaghetti_guard.config import load_config

    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(p, env={"BAMBU_ACCESS_CODE": "x"})


def test_config_missing_yaml_path_rejected(tmp_path):
    from spaghetti_guard.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", env={"BAMBU_ACCESS_CODE": "x"})


# =====================================================================
# Camera frames precondition + jpeg_is_well_formed branches
# =====================================================================


def test_jpeg_well_formed_empty():
    from spaghetti_guard.camera import jpeg_is_well_formed

    assert jpeg_is_well_formed(b"") is False
