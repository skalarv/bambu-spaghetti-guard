"""Camera backends for the Bambu chamber stream (brief §5.2).

Two backends behind a single ABC:

* `RawSocketBackend` — TLS socket to port 6000, 80-byte auth packet, then a
  loop of 16-byte headers followed by JPEG payloads (first 4 bytes of header
  are the payload length, little-endian). Malformed frames are dropped, EOF
  raises `CameraStreamClosed`.

* `LibBackend` — stub that delegates to the third-party `bambulabs_api` package
  for sites where the raw handshake misbehaves on a given firmware. Not wired
  in this session per the plan; raises NotImplementedError.

The wire format here matches `verification.mock_printer` byte-for-byte so the
raw backend is exercised by the integration test loop.
"""

from __future__ import annotations

import logging
import socket
import ssl
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterator

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_PORT = 6000
AUTH_PACKET_LEN = 80
AUTH_MAGIC1 = 0x40
AUTH_MAGIC2 = 0x3000
FRAME_HEADER_LEN = 16
MAX_FRAME_BYTES = 8 * 1024 * 1024  # 8 MiB sanity bound — real frames are ~50-200 KiB
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


class CameraError(Exception):
    """Base for camera-side errors."""


class CameraStreamClosed(CameraError):
    """Stream ended cleanly or was closed by the printer."""


class CameraAuthError(CameraError):
    """Authentication packet rejected by the printer."""


class CameraBackend(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def frames(self) -> Iterator[bytes]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "CameraBackend":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Wire helpers (pure functions, easy to unit-test)
# ---------------------------------------------------------------------------


def build_auth_packet(username: str, password: str) -> bytes:
    if len(username) > 32 or len(password) > 32:
        raise ValueError("username/password too long for auth packet")
    return (
        struct.pack("<I", AUTH_MAGIC1)
        + struct.pack("<I", AUTH_MAGIC2)
        + b"\x00" * 8
        + username.encode("ascii").ljust(32, b"\x00")
        + password.encode("ascii").ljust(32, b"\x00")
    )


def parse_frame_header(header: bytes) -> int:
    """Return the JPEG payload length encoded in a 16-byte frame header."""
    if len(header) != FRAME_HEADER_LEN:
        raise ValueError(f"frame header must be {FRAME_HEADER_LEN} bytes")
    (length,) = struct.unpack("<I", header[:4])
    if length <= 0 or length > MAX_FRAME_BYTES:
        raise ValueError(f"implausible frame length: {length}")
    return length


def jpeg_is_well_formed(payload: bytes) -> bool:
    """Heuristic: payload starts with FF D8 SOI and ends with FF D9 EOI."""
    return payload.startswith(JPEG_SOI) and payload.endswith(JPEG_EOI)


def iter_frames_from_stream(read_exact, *, drop_malformed: bool = True) -> Iterator[bytes]:
    """Pull length-prefixed JPEG frames from any 'read N bytes or raise' callable.

    Factored out so tests can synthesize buffers without touching sockets.
    """
    while True:
        try:
            header = read_exact(FRAME_HEADER_LEN)
        except EOFError:
            # Clean EOS between frames -- not an error. Real printer may close
            # the camera channel cleanly when the print finishes; guard's outer
            # loop will reconnect if needed.
            return
        try:
            length = parse_frame_header(header)
        except ValueError as e:
            if drop_malformed:
                logger.warning("dropping frame with bad header: %s", e)
                continue
            raise
        try:
            payload = read_exact(length)
        except EOFError:
            raise CameraStreamClosed("stream ended mid-payload")
        if not jpeg_is_well_formed(payload):
            if drop_malformed:
                logger.warning("dropping malformed JPEG (no SOI/EOI markers)")
                continue
            raise CameraError("malformed JPEG payload")
        yield payload


# ---------------------------------------------------------------------------
# RawSocketBackend
# ---------------------------------------------------------------------------


class RawSocketBackend(CameraBackend):
    def __init__(
        self,
        *,
        host: str,
        port: int = DEFAULT_CAMERA_PORT,
        username: str = "bblp",
        access_code: str,
        connect_timeout_s: float = 10.0,
        recv_timeout_s: float = 30.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._access_code = access_code
        self._connect_timeout_s = connect_timeout_s
        self._recv_timeout_s = recv_timeout_s
        self._sock: ssl.SSLSocket | None = None
        if ssl_context is None:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        self._ssl_context = ssl_context

    def connect(self) -> None:
        raw = socket.create_connection((self._host, self._port), timeout=self._connect_timeout_s)
        self._sock = self._ssl_context.wrap_socket(raw, server_hostname=self._host)
        self._sock.settimeout(self._recv_timeout_s)
        self._sock.sendall(build_auth_packet(self._username, self._access_code))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None

    def frames(self) -> Iterator[bytes]:
        if self._sock is None:
            raise CameraError("backend not connected")
        return iter_frames_from_stream(self._recv_exact)

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise EOFError("camera socket closed")
            buf.extend(chunk)
        return bytes(buf)


# ---------------------------------------------------------------------------
# LibBackend stub
# ---------------------------------------------------------------------------


class LibBackend(CameraBackend):
    """Stub for `bambulabs_api`-backed camera. Intentionally not wired this session.

    Per brief §5.2 the lib backend exists as a swap-in when the raw handshake
    breaks on a given firmware. Wiring it requires the operator to install the
    third-party package and to map its frame-source API onto `frames()`.
    """

    def __init__(self, **_kwargs) -> None:
        raise NotImplementedError(
            "LibBackend not wired yet; use camera.backend='raw'. "
            "See brief §5.2."
        )

    def connect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def frames(self) -> Iterator[bytes]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError
