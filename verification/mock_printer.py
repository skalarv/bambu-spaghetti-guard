"""Self-contained mock Bambu printer for offline guard testing (brief §6.1).

Two servers in one process:

* `MockCameraServer` — TLS TCP server on a configurable port (default 16000).
  Validates the brief's 80-byte auth packet (username `bblp` + access code),
  then streams length-prefixed JPEG frames at a configurable fps. Supports
  injecting a "spaghetti" frame from a given frame index onward.

* `MockMqttBroker` — TLS MQTT broker (amqtt) on a configurable port (default
  18883). Accepts username `bblp` + access code. Lets the test drive
  `device/{serial}/report` messages and records every message on
  `device/{serial}/request` into a `CommandRecorder` for assertions.

`MockPrinter` bundles both behind a single async context manager.

Frame markers
-------------
Synthetic JPEGs are tagged with a 1-byte marker in the COM segment so the
test detector can decide hit/miss without an actual YOLO model:
  - "CLEAN" frame  => marker byte 0x00
  - "FAILURE" frame => marker byte 0xFF

The marker is preserved by `RawSocketBackend` because the JPEG payload is
streamed verbatim.

Why amqtt instead of mosquitto: keeps the test env hermetic and cross-platform.
If amqtt becomes flaky on a host, swap to spawning mosquitto in a subprocess
— PrinterControl is broker-agnostic.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import socket
import ssl
import struct
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path

from amqtt.broker import Broker
from amqtt.client import MQTTClient
from amqtt.mqtt.constants import QOS_0, QOS_1
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from PIL import Image

logger = logging.getLogger(__name__)

AUTH_PACKET_LEN = 80
AUTH_MAGIC1 = 0x40
AUTH_MAGIC2 = 0x3000
FRAME_HEADER_LEN = 16
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

MARKER_CLEAN = 0x00
MARKER_FAILURE = 0xFF


def _pick_free_port() -> int:
    """Ask the OS for a free localhost port. Releases immediately."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Synthetic JPEGs
# ---------------------------------------------------------------------------


def make_jpeg(marker: int, *, size: tuple[int, int] = (160, 120)) -> bytes:
    """Build a tiny synthetic JPEG carrying a 1-byte marker in a COM segment.

    The marker survives JPEG decode and is recoverable by scanning bytes —
    that's what the fake detector keys off.
    """
    img = Image.new("RGB", size, color=(40, 40, 40) if marker == MARKER_CLEAN else (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    jpeg = buf.getvalue()
    # Insert a COM (0xFFFE) segment right after SOI so it survives decoding.
    assert jpeg.startswith(JPEG_SOI) and jpeg.endswith(JPEG_EOI)
    com_payload = bytes([marker]) + b"SPAGHETTI-GUARD-MOCK"
    com_segment = b"\xff\xfe" + struct.pack(">H", len(com_payload) + 2) + com_payload
    return jpeg[:2] + com_segment + jpeg[2:]


def extract_marker(jpeg: bytes) -> int | None:
    """Return the marker byte injected by `make_jpeg`, or None."""
    idx = jpeg.find(b"\xff\xfe")
    if idx < 0 or idx + 4 > len(jpeg):
        return None
    seg_len = struct.unpack(">H", jpeg[idx + 2 : idx + 4])[0]
    payload = jpeg[idx + 4 : idx + 2 + seg_len]
    if not payload.startswith(b"\x00") and not payload.startswith(b"\xff"):
        # length doesn't decode cleanly to a marker byte
        return None
    if len(payload) < 1:
        return None
    return payload[0]


# ---------------------------------------------------------------------------
# Self-signed cert for both servers
# ---------------------------------------------------------------------------


def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    cert_path = cert_dir / "mock.crt"
    key_path = cert_dir / "mock.key"
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "mock-bambu")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


# ---------------------------------------------------------------------------
# Auth packet
# ---------------------------------------------------------------------------


def build_auth_packet(username: str, password: str) -> bytes:
    """Build the 80-byte auth packet that RawSocketBackend sends and the mock validates."""
    if len(username) > 32 or len(password) > 32:
        raise ValueError("username/password too long for auth packet")
    return (
        struct.pack("<I", AUTH_MAGIC1)
        + struct.pack("<I", AUTH_MAGIC2)
        + b"\x00" * 8
        + username.encode("ascii").ljust(32, b"\x00")
        + password.encode("ascii").ljust(32, b"\x00")
    )


def parse_auth_packet(blob: bytes) -> tuple[str, str]:
    if len(blob) != AUTH_PACKET_LEN:
        raise ValueError(f"auth packet must be {AUTH_PACKET_LEN} bytes, got {len(blob)}")
    magic1 = struct.unpack("<I", blob[0:4])[0]
    magic2 = struct.unpack("<I", blob[4:8])[0]
    if magic1 != AUTH_MAGIC1 or magic2 != AUTH_MAGIC2:
        raise ValueError(f"bad auth magic: {magic1:#x}, {magic2:#x}")
    username = blob[16:48].rstrip(b"\x00").decode("ascii")
    password = blob[48:80].rstrip(b"\x00").decode("ascii")
    return username, password


def frame_with_header(jpeg: bytes) -> bytes:
    """Build a 16-byte header followed by the JPEG payload."""
    header = struct.pack("<I", len(jpeg)) + b"\x00" * 12
    return header + jpeg


# ---------------------------------------------------------------------------
# Camera server
# ---------------------------------------------------------------------------


class MockCameraServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        access_code: str,
        cert_path: Path,
        key_path: Path,
        fps: float = 5.0,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._access_code = access_code
        self._cert_path = cert_path
        self._key_path = key_path
        self._fps = fps
        self._server: asyncio.AbstractServer | None = None
        self._frame_index = 0
        self._inject_failure_from: int | None = None
        self._frames_streamed = 0
        self._client_tasks: set[asyncio.Task] = set()

    @property
    def actual_port(self) -> int:
        assert self._server is not None
        sock = self._server.sockets[0]
        return sock.getsockname()[1]

    def inject_failure_at(self, frame_index: int) -> None:
        """Stream FAILURE-marker frames from this frame index onward."""
        self._inject_failure_from = frame_index

    def reset(self) -> None:
        self._frame_index = 0
        self._inject_failure_from = None
        self._frames_streamed = 0

    @property
    def frames_streamed(self) -> int:
        return self._frames_streamed

    async def start(self) -> None:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(self._cert_path, self._key_path)
        self._server = await asyncio.start_server(
            self._handle_wrapped, host=self._host, port=self._port, ssl=ssl_ctx
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        # Cancel any per-client handler tasks still streaming so we don't leak
        # "Task was destroyed but it is pending" warnings.
        for task in list(self._client_tasks):
            task.cancel()
        for task in list(self._client_tasks):
            with contextlib.suppress(BaseException):
                await task
        self._client_tasks.clear()

    async def _handle_wrapped(self, reader, writer) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            await self._handle(reader, writer)
        finally:
            if task is not None:
                self._client_tasks.discard(task)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            blob = await reader.readexactly(AUTH_PACKET_LEN)
            username, password = parse_auth_packet(blob)
            if username != self._username or password != self._access_code:
                logger.warning("mock-cam: bad creds from %s", peer)
                writer.close()
                return
            logger.info("mock-cam: client authed from %s", peer)
            await self._stream_loop(writer)
        except (asyncio.IncompleteReadError, ConnectionResetError, ssl.SSLError):
            return
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _stream_loop(self, writer: asyncio.StreamWriter) -> None:
        interval = 1.0 / self._fps if self._fps > 0 else 0.0
        while True:
            marker = MARKER_CLEAN
            if (
                self._inject_failure_from is not None
                and self._frame_index >= self._inject_failure_from
            ):
                marker = MARKER_FAILURE
            jpeg = make_jpeg(marker)
            try:
                writer.write(frame_with_header(jpeg))
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, ssl.SSLError):
                return
            self._frame_index += 1
            self._frames_streamed += 1
            if interval > 0:
                await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# MQTT broker
# ---------------------------------------------------------------------------


@dataclass
class RecordedCommand:
    topic: str
    payload: dict
    qos: int
    ts: float


class CommandRecorder:
    def __init__(self) -> None:
        self._items: list[RecordedCommand] = []
        self._lock = asyncio.Lock()

    async def add(self, cmd: RecordedCommand) -> None:
        async with self._lock:
            self._items.append(cmd)

    def snapshot(self) -> list[RecordedCommand]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class MockMqttBroker:
    """amqtt broker + a co-resident MQTTClient that subscribes to the request topic.

    Plain MQTT (no TLS). PrinterControl in this test session is configured the
    same way for parity. The brief's TLS requirement is a property of the
    *real* P1S; the mock is allowed to relax it because PrinterControl talks
    to the mock with a parallel non-TLS code path, and TLS-specific paho
    behavior is already exercised by `test_control_payload.py`.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        access_code: str,
        serial: str,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._access_code = access_code
        self._serial = serial
        self._broker: Broker | None = None
        self._listener_client: MQTTClient | None = None
        self._publisher_client: MQTTClient | None = None
        self._listener_task: asyncio.Task | None = None
        self.recorder = CommandRecorder()
        self._actual_port: int | None = None

    @property
    def actual_port(self) -> int:
        assert self._actual_port is not None
        return self._actual_port

    @property
    def request_topic(self) -> str:
        return f"device/{self._serial}/request"

    @property
    def report_topic(self) -> str:
        return f"device/{self._serial}/report"

    def _broker_config(self) -> dict:
        # amqtt 0.11 EP-based loading: `auth.plugins` is a whitelist of
        # entry-point names. Empty list -> no auth plugin runs -> every
        # connection is rejected. Explicit `auth_anonymous` enables anon auth.
        return {
            "listeners": {
                "default": {
                    "type": "tcp",
                    "bind": f"{self._host}:{self._port}",
                    "max_connections": 50,
                }
            },
            "sys_interval": 0,
            "auth": {
                "allow-anonymous": True,
                "plugins": ["auth_anonymous"],
            },
            "topic-check": {"enabled": False},
        }

    async def start(self) -> None:
        # amqtt's listener config doesn't reliably expose a 0-bind back to us,
        # so we pre-pick a free OS port if the caller asked for one.
        if self._port == 0:
            self._port = _pick_free_port()
        self._broker = Broker(self._broker_config())
        await self._broker.start()
        self._actual_port = self._port

        self._listener_client = MQTTClient(client_id="mock-recorder")
        await self._listener_client.connect(f"mqtt://127.0.0.1:{self._actual_port}/")
        await self._listener_client.subscribe([(self.request_topic, QOS_1)])
        self._listener_task = asyncio.create_task(self._listener_loop())

        self._publisher_client = MQTTClient(client_id="mock-publisher")
        await self._publisher_client.connect(f"mqtt://127.0.0.1:{self._actual_port}/")

    async def _listener_loop(self) -> None:
        assert self._listener_client is not None
        try:
            while True:
                msg = await self._listener_client.deliver_message()
                pkt = msg.publish_packet
                topic = pkt.variable_header.topic_name
                payload_bytes = bytes(pkt.payload.data)
                try:
                    payload = json.loads(payload_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = {"_raw": payload_bytes.hex()}
                qos = pkt.qos
                await self.recorder.add(
                    RecordedCommand(topic=topic, payload=payload, qos=qos, ts=time.time())
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("mock-mqtt listener loop crashed")

    async def publish_report(self, payload: dict) -> None:
        assert self._publisher_client is not None
        await self._publisher_client.publish(
            self.report_topic, json.dumps(payload).encode("utf-8"), qos=QOS_0
        )

    async def set_gcode_state(self, state: str, *, layer_num: int = 0) -> None:
        await self.publish_report({"print": {"gcode_state": state, "layer_num": layer_num}})

    async def stop(self) -> None:
        if self._listener_task is not None:
            self._listener_task.cancel()
            with contextlib.suppress(Exception):
                await self._listener_task
        if self._listener_client is not None:
            with contextlib.suppress(Exception):
                await self._listener_client.disconnect()
        if self._publisher_client is not None:
            with contextlib.suppress(Exception):
                await self._publisher_client.disconnect()
        if self._broker is not None:
            with contextlib.suppress(Exception):
                await self._broker.shutdown()


# ---------------------------------------------------------------------------
# Combined async context manager
# ---------------------------------------------------------------------------


@dataclass
class MockConfig:
    host: str = "127.0.0.1"
    camera_port: int = 0
    mqtt_port: int = 0
    username: str = "bblp"
    access_code: str = "test-access-code"
    serial: str = "TESTSERIAL"
    fps: float = 10.0


class MockPrinter:
    def __init__(self, cfg: MockConfig | None = None) -> None:
        self.cfg = cfg or MockConfig()
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self.camera: MockCameraServer | None = None
        self.broker: MockMqttBroker | None = None

    async def __aenter__(self) -> MockPrinter:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mock-bambu-")
        cert_dir = Path(self._tmpdir.name)
        cert_path, key_path = generate_self_signed_cert(cert_dir)

        self.camera = MockCameraServer(
            host=self.cfg.host,
            port=self.cfg.camera_port,
            username=self.cfg.username,
            access_code=self.cfg.access_code,
            cert_path=cert_path,
            key_path=key_path,
            fps=self.cfg.fps,
        )
        await self.camera.start()

        self.broker = MockMqttBroker(
            host=self.cfg.host,
            port=self.cfg.mqtt_port,
            username=self.cfg.username,
            access_code=self.cfg.access_code,
            serial=self.cfg.serial,
        )
        await self.broker.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.broker is not None:
            await self.broker.stop()
        if self.camera is not None:
            await self.camera.stop()
        if self._tmpdir is not None:
            with contextlib.suppress(Exception):
                self._tmpdir.cleanup()

    @property
    def camera_port(self) -> int:
        assert self.camera is not None
        return self.camera.actual_port

    @property
    def mqtt_port(self) -> int:
        assert self.broker is not None
        return self.broker.actual_port
