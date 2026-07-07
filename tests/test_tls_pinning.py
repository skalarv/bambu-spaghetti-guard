"""Optional TLS certificate pinning for the camera + MQTT channels.

The P1S serves self-signed certs, so classic CA verification is off — but
the operator can pin the printer's certificate (SHA-256 of the DER form) so
a LAN MITM can't intercept the access code or feed fake frames.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from spaghetti_guard import camera as camera_mod
from spaghetti_guard import control as control_mod
from spaghetti_guard.camera import CameraError, RawSocketBackend, sha256_fingerprint

DER = b"fake-der-certificate-bytes"
GOOD_FP = hashlib.sha256(DER).hexdigest()
BAD_FP = "00" * 32


def test_sha256_fingerprint_hex():
    assert sha256_fingerprint(DER) == GOOD_FP


# ---- camera ----------------------------------------------------------------


def _fake_camera_env(monkeypatch):
    tls_sock = MagicMock()
    tls_sock.getpeercert.return_value = DER
    ctx = MagicMock()
    ctx.wrap_socket.return_value = tls_sock
    monkeypatch.setattr(
        camera_mod.socket, "create_connection", lambda *a, **kw: MagicMock()
    )
    return tls_sock, ctx


def test_camera_pin_mismatch_refuses_connection(monkeypatch):
    tls_sock, ctx = _fake_camera_env(monkeypatch)
    b = RawSocketBackend(
        host="1.2.3.4", access_code="tok", ssl_context=ctx, tls_fingerprint=BAD_FP
    )
    with pytest.raises(CameraError, match="fingerprint"):
        b.connect()
    # The auth packet (which carries the access code) must never be sent
    # over an unverified channel.
    tls_sock.sendall.assert_not_called()
    tls_sock.close.assert_called()


def test_camera_pin_match_connects_and_authenticates(monkeypatch):
    tls_sock, ctx = _fake_camera_env(monkeypatch)
    # Colons + uppercase (openssl x509 -fingerprint format) must be accepted.
    pretty = ":".join(GOOD_FP[i : i + 2] for i in range(0, 64, 2)).upper()
    b = RawSocketBackend(
        host="1.2.3.4", access_code="tok", ssl_context=ctx, tls_fingerprint=pretty
    )
    b.connect()
    tls_sock.sendall.assert_called_once()


def test_camera_no_pin_keeps_current_behavior(monkeypatch):
    tls_sock, ctx = _fake_camera_env(monkeypatch)
    b = RawSocketBackend(host="1.2.3.4", access_code="tok", ssl_context=ctx)
    b.connect()
    tls_sock.sendall.assert_called_once()


# ---- MQTT -------------------------------------------------------------------


class _FakePahoTls:
    def __init__(self, der=DER):
        self._der = der
        self.subscribed = []
        self.disconnect_calls = 0

    # constructor-time calls
    def username_pw_set(self, u, p):
        pass

    def tls_set(self, **kw):
        pass

    def tls_insecure_set(self, v):
        pass

    def reconnect_delay_set(self, min_delay=1, max_delay=120):
        pass

    # runtime
    def socket(self):
        s = MagicMock()
        s.getpeercert.return_value = self._der
        return s

    def subscribe(self, topic, qos=0):
        self.subscribed.append((topic, qos))

    def disconnect(self):
        self.disconnect_calls += 1


def _build_control(fingerprint):
    fake = _FakePahoTls()
    pc = control_mod.PrinterControl(
        host="1.2.3.4",
        serial="S",
        access_code="tok",
        client_factory=lambda cid: fake,
        tls_fingerprint=fingerprint,
    )
    return pc, fake


def test_mqtt_pin_mismatch_disconnects_without_subscribing():
    pc, fake = _build_control(BAD_FP)
    pc._on_connect(fake, None, {}, 0, None)
    assert fake.subscribed == []
    assert fake.disconnect_calls == 1
    assert pc.is_connected is False


def test_mqtt_pin_match_subscribes():
    pc, fake = _build_control(GOOD_FP)
    pc._on_connect(fake, None, {}, 0, None)
    assert fake.subscribed  # report topic subscribed
    assert pc.is_connected is True
