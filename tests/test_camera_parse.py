"""Brief §6.4: header parse, split recvs, malformed frame handling."""

from __future__ import annotations

import io
import struct

import pytest

from spaghetti_guard.camera import (
    FRAME_HEADER_LEN,
    JPEG_EOI,
    JPEG_SOI,
    CameraError,
    CameraStreamClosed,
    LibBackend,
    build_auth_packet,
    iter_frames_from_stream,
    jpeg_is_well_formed,
    parse_frame_header,
)


# ---- helpers ------------------------------------------------------------


def make_jpeg(body: bytes = b"\x00\x00\x00") -> bytes:
    return JPEG_SOI + body + JPEG_EOI


def make_frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + b"\x00" * 12 + payload


def buffer_reader(blob: bytes):
    """Return a recv_exact callable that pulls from a bytes buffer."""
    buf = io.BytesIO(blob)

    def read_exact(n: int) -> bytes:
        chunk = buf.read(n)
        if len(chunk) < n:
            raise EOFError("short read")
        return chunk

    return read_exact


def chunked_reader(blob: bytes, chunk_size: int):
    """Return a recv_exact that simulates recv() boundaries by serving max chunk_size bytes."""
    pos = 0

    def read_exact(n: int) -> bytes:
        nonlocal pos
        out = bytearray()
        while len(out) < n:
            remaining = n - len(out)
            take = min(remaining, chunk_size)
            chunk = blob[pos : pos + take]
            if not chunk:
                raise EOFError("EOF")
            out.extend(chunk)
            pos += take
        return bytes(out)

    return read_exact


# ---- pure helpers --------------------------------------------------------


def test_auth_packet_layout():
    pkt = build_auth_packet("bblp", "secret")
    assert len(pkt) == 80
    assert struct.unpack("<I", pkt[0:4])[0] == 0x40
    assert struct.unpack("<I", pkt[4:8])[0] == 0x3000
    assert pkt[16:48].rstrip(b"\x00") == b"bblp"
    assert pkt[48:80].rstrip(b"\x00") == b"secret"


def test_auth_packet_rejects_long_user():
    with pytest.raises(ValueError):
        build_auth_packet("x" * 33, "y")


def test_parse_frame_header_round_trip():
    header = struct.pack("<I", 1234) + b"\x00" * 12
    assert parse_frame_header(header) == 1234


def test_parse_frame_header_rejects_bad_length():
    bad = struct.pack("<I", 0) + b"\x00" * 12
    with pytest.raises(ValueError):
        parse_frame_header(bad)


def test_jpeg_well_formed_checks_both_markers():
    assert jpeg_is_well_formed(make_jpeg())
    assert not jpeg_is_well_formed(b"\x00" * 4 + JPEG_EOI)
    assert not jpeg_is_well_formed(JPEG_SOI + b"\x00" * 4)


# ---- stream iterator -----------------------------------------------------


def test_yields_two_clean_frames():
    j1 = make_jpeg(b"\x10")
    j2 = make_jpeg(b"\x20")
    blob = make_frame(j1) + make_frame(j2)
    frames = list(iter_frames_from_stream(buffer_reader(blob)))
    assert frames == [j1, j2]


def test_handles_split_recvs():
    """Force a read boundary every 7 bytes — exercises recv_exact accumulation."""
    j1 = make_jpeg(b"\x55" * 40)
    j2 = make_jpeg(b"\xaa" * 60)
    blob = make_frame(j1) + make_frame(j2)
    frames = []
    gen = iter_frames_from_stream(chunked_reader(blob, chunk_size=7))
    for f in gen:
        frames.append(f)
        if len(frames) == 2:
            break
    assert frames == [j1, j2]


def test_drops_malformed_jpeg_and_continues():
    bad = make_frame(b"\x00\x00not-a-jpeg\x00\x00")
    good = make_jpeg(b"\xff")
    blob = bad + make_frame(good)
    frames = list(iter_frames_from_stream(buffer_reader(blob)))
    # Only the good one survives
    assert frames == [good]


def test_eof_at_header_is_clean_end_of_stream():
    """EOF at a frame boundary terminates the generator without raising."""
    assert list(iter_frames_from_stream(buffer_reader(b""))) == []


def test_eof_mid_payload_raises_stream_closed():
    j1 = make_jpeg(b"\x01" * 100)
    full = make_frame(j1)
    truncated = full[: FRAME_HEADER_LEN + 10]  # header + 10 payload bytes only
    with pytest.raises(CameraStreamClosed):
        list(iter_frames_from_stream(buffer_reader(truncated)))


def test_drop_malformed_false_raises():
    bad = make_frame(b"\x00not-a-jpeg")
    with pytest.raises(CameraError):
        list(iter_frames_from_stream(buffer_reader(bad), drop_malformed=False))


# ---- LibBackend stub -----------------------------------------------------


def test_lib_backend_not_wired():
    with pytest.raises(NotImplementedError, match="raw"):
        LibBackend()
