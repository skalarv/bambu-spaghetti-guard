"""Brief §6.4: payload, topic, QoS exactness; dry-run doesn't publish."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from spaghetti_guard.control import (
    REPORT_TOPIC_FMT,
    REQUEST_TOPIC_FMT,
    STOP_QOS,
    CommandPublishError,
    PrinterControl,
    PrinterState,
)


class FakePahoClient:
    """Minimal stand-in for paho.mqtt.client.Client (CallbackAPIVersion.VERSION2)."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.username = None
        self.password = None
        self.published: list[tuple[str, bytes | str, int]] = []
        self.subscribed: list[tuple[str, int]] = []
        self.tls_set_called = False
        self.tls_insecure = False
        self.connected_host: tuple[str, int] | None = None
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None

    def username_pw_set(self, username, password):
        self.username = username
        self.password = password

    def tls_set(self, **kwargs):
        self.tls_set_called = True

    def tls_insecure_set(self, v):
        self.tls_insecure = v

    def connect(self, host, port, keepalive=60):
        self.connected_host = (host, port)

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, qos=0):
        self.subscribed.append((topic, qos))

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload, qos))
        m = MagicMock()
        m.rc = 0
        return m

    def reconnect(self):
        pass

    def reconnect_delay_set(self, min_delay=1, max_delay=120):
        self.reconnect_delay = (min_delay, max_delay)


def _build_control(serial="ABC123", dry_run=False) -> tuple[PrinterControl, FakePahoClient]:
    fake = FakePahoClient("test-client")
    pc = PrinterControl(
        host="127.0.0.1",
        serial=serial,
        access_code="secret",
        dry_run=dry_run,
        client_factory=lambda cid: fake,
    )
    return pc, fake


def _decode_only_published(fake: FakePahoClient) -> tuple[str, dict, int]:
    assert len(fake.published) == 1, fake.published
    topic, payload, qos = fake.published[0]
    return topic, json.loads(payload), qos


# ---- payload exactness ---------------------------------------------------


def test_stop_payload_exact():
    pc, fake = _build_control(serial="SER1")
    pc.stop()
    topic, decoded, qos = _decode_only_published(fake)
    assert topic == "device/SER1/request"
    assert decoded == {"print": {"command": "stop", "sequence_id": "0"}}
    assert qos == STOP_QOS == 1


def test_pause_payload_exact():
    pc, fake = _build_control(serial="SER2")
    pc.pause()
    topic, decoded, qos = _decode_only_published(fake)
    assert topic == "device/SER2/request"
    assert decoded == {"print": {"command": "pause", "sequence_id": "0"}}
    assert qos == 1


def test_resume_payload_exact():
    pc, fake = _build_control(serial="SER3")
    pc.resume()
    topic, decoded, qos = _decode_only_published(fake)
    assert decoded == {"print": {"command": "resume", "sequence_id": "0"}}
    assert qos == 1


# ---- dry-run ------------------------------------------------------------


def test_dry_run_never_publishes():
    pc, fake = _build_control(dry_run=True)
    assert pc.stop() is None
    assert pc.pause() is None
    assert pc.resume() is None
    assert fake.published == []


# ---- publish delivery verification ---------------------------------------
# A stop/pause that the broker never accepted must raise, not silently
# succeed — the guard treats the exception as "action failed" and stays
# TRIGGERED instead of moving to COOLDOWN believing it acted.


def test_publish_rejected_rc_raises():
    pc, fake = _build_control()

    def failing_publish(topic, payload=None, qos=0, retain=False):
        m = MagicMock()
        m.rc = 4  # paho MQTT_ERR_NO_CONN
        return m

    fake.publish = failing_publish
    with pytest.raises(CommandPublishError):
        pc.pause()


def test_publish_unacked_within_timeout_raises():
    pc, fake = _build_control()

    def unacked_publish(topic, payload=None, qos=0, retain=False):
        m = MagicMock()
        m.rc = 0
        m.is_published.return_value = False  # PUBACK never arrives
        return m

    fake.publish = unacked_publish
    with pytest.raises(CommandPublishError):
        pc.stop()


def test_publish_success_waits_for_ack():
    pc, fake = _build_control()
    acked = MagicMock()
    acked.rc = 0
    acked.is_published.return_value = True

    def ok_publish(topic, payload=None, qos=0, retain=False):
        fake.published.append((topic, payload, qos))
        return acked

    fake.publish = ok_publish
    info = pc.pause()
    assert info is acked
    acked.wait_for_publish.assert_called_once()
    # the wait must be bounded — a hung broker must not hang the guard forever
    _, kwargs = acked.wait_for_publish.call_args
    assert kwargs.get("timeout", 0) > 0


def test_publish_wait_raising_maps_to_command_error():
    """paho raises RuntimeError from wait_for_publish if the message was
    dropped from the out-queue; that must surface as CommandPublishError."""
    pc, fake = _build_control()

    def dropped_publish(topic, payload=None, qos=0, retain=False):
        m = MagicMock()
        m.rc = 0
        m.wait_for_publish.side_effect = RuntimeError("message not queued")
        return m

    fake.publish = dropped_publish
    with pytest.raises(CommandPublishError):
        pc.stop()


# ---- topics ------------------------------------------------------------


def test_topic_formats_use_serial():
    pc, _ = _build_control(serial="XYZ")
    assert pc.request_topic == REQUEST_TOPIC_FMT.format(serial="XYZ") == "device/XYZ/request"
    assert pc.report_topic == REPORT_TOPIC_FMT.format(serial="XYZ") == "device/XYZ/report"


# ---- credentials -------------------------------------------------------


def test_credentials_passed_to_client():
    pc, fake = _build_control()
    assert fake.username == "bblp"
    assert fake.password == "secret"
    assert fake.tls_set_called
    assert fake.tls_insecure is True


# ---- state mirror ------------------------------------------------------


def test_printer_state_updates_from_report():
    s = PrinterState()
    s.update_from_report({"print": {"gcode_state": "RUNNING", "layer_num": 42}})
    state, layer, ts = s.snapshot()
    assert state == "RUNNING"
    assert layer == 42
    assert ts > 0


def test_printer_state_ignores_unknown_keys():
    s = PrinterState()
    s.update_from_report({"print": {"gcode_state": "PAUSE"}})
    s.update_from_report({"unrelated": {"foo": "bar"}})
    state, layer, _ = s.snapshot()
    assert state == "PAUSE"  # not overwritten by the second message
    assert layer == 0


def test_control_state_updates_on_report_message(monkeypatch):
    pc, fake = _build_control(serial="SER")
    # Simulate paho delivering a report message
    msg = MagicMock()
    msg.topic = "device/SER/report"
    msg.payload = json.dumps({"print": {"gcode_state": "RUNNING", "layer_num": 5}}).encode()
    pc._on_message(fake, None, msg)
    state, layer, _ = pc.state.snapshot()
    assert state == "RUNNING"
    assert layer == 5


def test_control_ignores_unrelated_topic():
    pc, fake = _build_control(serial="SER")
    msg = MagicMock()
    msg.topic = "device/OTHER/report"
    msg.payload = json.dumps({"print": {"gcode_state": "RUNNING"}}).encode()
    pc._on_message(fake, None, msg)
    state, _, _ = pc.state.snapshot()
    assert state == "IDLE"  # untouched


def test_control_handles_malformed_report():
    pc, fake = _build_control(serial="SER")
    msg = MagicMock()
    msg.topic = "device/SER/report"
    msg.payload = b"not-json"
    pc._on_message(fake, None, msg)  # must not raise
    state, _, _ = pc.state.snapshot()
    assert state == "IDLE"
