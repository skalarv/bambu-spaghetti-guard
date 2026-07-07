"""MQTT control + report-state tracking (brief §5.3).

paho-mqtt 2.x callback API. TLS to port 8883, username 'bblp', password is the
LAN access code. Stop/pause/resume at QoS 1 on `device/{serial}/request`. The
report topic feeds a thread-safe PrinterState.
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import paho.mqtt.client as mqtt

from .camera import normalize_fingerprint, sha256_fingerprint

logger = logging.getLogger(__name__)

MQTT_USERNAME = "bblp"
MQTT_DEFAULT_PORT = 8883
REQUEST_TOPIC_FMT = "device/{serial}/request"
REPORT_TOPIC_FMT = "device/{serial}/report"
STOP_QOS = 1


class CommandPublishError(RuntimeError):
    """A stop/pause/resume publish was rejected or never acknowledged.

    Raised instead of returning, so Guard._fire's exception path engages:
    the guard stays TRIGGERED and alerts the operator rather than entering
    COOLDOWN believing the printer received the command.
    """


@dataclass
class PrinterState:
    """Mirror of the printer's last-known state.

    Thread-safe: every read goes through `snapshot()`, every write through
    `update_from_report()`.
    """

    gcode_state: str = "IDLE"
    layer_num: int = 0
    last_update_ts: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def snapshot(self) -> tuple[str, int, float]:
        with self._lock:
            return self.gcode_state, self.layer_num, self.last_update_ts

    def update_from_report(self, payload: dict) -> None:
        """Apply a `device/{serial}/report` payload to the state.

        Bambu reports nest interesting fields under `print`. We only read
        `gcode_state` and `layer_num`, the two fields the guard makes decisions
        from (brief §2).
        """
        print_block = payload.get("print") or {}
        with self._lock:
            if "gcode_state" in print_block:
                self.gcode_state = str(print_block["gcode_state"])
            if "layer_num" in print_block:
                try:
                    self.layer_num = int(print_block["layer_num"])
                except (TypeError, ValueError):
                    pass
            self.last_update_ts = time.time()


def _build_command_payload(command: str) -> dict[str, Any]:
    return {"print": {"command": command, "sequence_id": "0"}}


class PrinterControl:
    """Wraps a paho 2.x client; exposes stop/pause/resume + state mirror.

    The paho client is created via `client_factory` so tests can inject a fake.
    """

    def __init__(
        self,
        *,
        host: str,
        serial: str,
        access_code: str,
        port: int = MQTT_DEFAULT_PORT,
        dry_run: bool = False,
        use_tls: bool = True,
        client_id: str = "spaghetti-guard",
        client_factory: Callable[..., mqtt.Client] | None = None,
        reconnect_backoff_max_s: float = 30.0,
        publish_timeout_s: float = 5.0,
        tls_fingerprint: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._serial = serial
        self._access_code = access_code
        self._dry_run = dry_run
        self._reconnect_backoff_max_s = reconnect_backoff_max_s
        self._publish_timeout_s = publish_timeout_s
        # Optional SHA-256 pin of the broker's DER certificate (the P1S is
        # self-signed, so this is the only MITM defense on this channel).
        self._tls_fingerprint = normalize_fingerprint(tls_fingerprint) if tls_fingerprint else None
        self._state = PrinterState()
        self._connected = threading.Event()
        self._stop_loop = threading.Event()

        factory = client_factory or (
            lambda cid: mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=cid,
                protocol=mqtt.MQTTv311,
            )
        )
        self._client = factory(client_id)
        self._client.username_pw_set(MQTT_USERNAME, access_code)
        # Reconnection is paho's job: loop_start() auto-reconnects on
        # disconnect using these delays. No second hand-rolled reconnect
        # thread — two mechanisms racing on one client can wedge it.
        self._client.reconnect_delay_set(min_delay=1, max_delay=int(reconnect_backoff_max_s))
        # The real P1S serves a self-signed TLS broker; the integration mock
        # uses a plain-MQTT amqtt broker. Skip TLS setup when use_tls=False.
        if use_tls:
            try:
                self._client.tls_set(cert_reqs=ssl.CERT_NONE)
            except (AttributeError, ValueError):  # already configured by fake / not applicable
                pass
            try:
                self._client.tls_insecure_set(True)
            except AttributeError:
                pass

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    # ---- properties -----------------------------------------------------
    @property
    def state(self) -> PrinterState:
        return self._state

    @property
    def request_topic(self) -> str:
        return REQUEST_TOPIC_FMT.format(serial=self._serial)

    @property
    def report_topic(self) -> str:
        return REPORT_TOPIC_FMT.format(serial=self._serial)

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ---- lifecycle ------------------------------------------------------
    def connect(self, *, wait_s: float = 5.0) -> bool:
        self._client.connect(self._host, self._port, keepalive=60)
        self._client.loop_start()
        return self._connected.wait(timeout=wait_s)

    def close(self) -> None:
        self._stop_loop.set()
        try:
            self._client.loop_stop()
        except Exception:
            pass
        try:
            self._client.disconnect()
        except Exception:
            pass

    # ---- commands -------------------------------------------------------
    def stop(self) -> mqtt.MQTTMessageInfo | None:
        return self._publish_command("stop")

    def pause(self) -> mqtt.MQTTMessageInfo | None:
        return self._publish_command("pause")

    def resume(self) -> mqtt.MQTTMessageInfo | None:
        return self._publish_command("resume")

    def _publish_command(self, command: str) -> mqtt.MQTTMessageInfo | None:
        payload = json.dumps(_build_command_payload(command), separators=(",", ":"))
        topic = self.request_topic
        if self._dry_run:
            logger.warning("[dry-run] would publish %s -> %s", topic, payload)
            return None
        logger.info("publishing %s -> %s", topic, payload)
        info = self._client.publish(topic, payload=payload, qos=STOP_QOS)
        rc = getattr(info, "rc", mqtt.MQTT_ERR_SUCCESS)
        if rc != mqtt.MQTT_ERR_SUCCESS:
            raise CommandPublishError(f"{command} publish rejected by client (rc={rc})")
        try:
            info.wait_for_publish(timeout=self._publish_timeout_s)
        except Exception as e:
            raise CommandPublishError(f"{command} publish could not be confirmed: {e}") from e
        if not info.is_published():
            raise CommandPublishError(
                f"{command} not acknowledged by broker within {self._publish_timeout_s:.1f}s"
            )
        return info

    # ---- callbacks ------------------------------------------------------
    def _peer_cert_matches(self, client) -> bool:
        try:
            der = client.socket().getpeercert(binary_form=True)
        except Exception:
            logger.exception("could not read peer certificate for pinning")
            return False
        return bool(der) and sha256_fingerprint(der) == self._tls_fingerprint

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None):
        # paho 2.x VERSION2 signature: (client, userdata, connect_flags, reason_code, properties)
        rc = getattr(reason_code, "value", reason_code)
        if rc == 0 or rc == "Success":
            if self._tls_fingerprint and not self._peer_cert_matches(client):
                logger.critical(
                    "MQTT TLS certificate fingerprint mismatch — possible MITM; "
                    "disconnecting and refusing to send credentials-bearing traffic"
                )
                client.disconnect()
                return
            self._connected.set()
            client.subscribe(self.report_topic, qos=0)
            logger.info("MQTT connected; subscribed to %s", self.report_topic)
        else:
            logger.error("MQTT connect failed: %s", reason_code)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self._connected.clear()
        if self._stop_loop.is_set():
            return
        logger.warning(
            "MQTT disconnected (%s); paho will auto-reconnect with backoff", reason_code
        )

    def _on_message(self, client, userdata, msg):
        if msg.topic != self.report_topic:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.debug("malformed report payload on %s", msg.topic)
            return
        self._state.update_from_report(payload)
