# Protocol reference

This is the wire format the guard speaks. The handshake bytes for the camera
channel are community-reverse-engineered and **firmware-sensitive** — before
relying on them in production, verify them against
[OpenBambuAPI](https://github.com/Doridian/OpenBambuAPI) for your specific
firmware version.

## Camera channel (TCP 6000, TLS)

| Item | Value |
|---|---|
| Port | 6000 (A1 / P1 series). X1 / H2D use port 322 — not covered. |
| Transport | TLS over TCP. Bambu serves a self-signed cert; client uses `CERT_NONE`. |
| Auth | 80-byte fixed-layout packet sent immediately after TLS handshake. |
| Frames | 16-byte header + length-prefixed JPEG. Roughly 1 fps. |

### Auth packet (80 bytes total)

| Offset | Length | Field | Notes |
|---|---|---|---|
| 0 | 4 | `magic1` | Little-endian `0x40`. |
| 4 | 4 | `magic2` | Little-endian `0x3000`. |
| 8 | 8 | reserved | All zeroes. |
| 16 | 32 | `username` | ASCII, `bblp`, NUL-padded. |
| 48 | 32 | `password` | ASCII, the **LAN access code**, NUL-padded. |

Both ends in this repo (`RawSocketBackend.build_auth_packet` and
`mock_printer.parse_auth_packet`) speak this exact layout, so the
integration test exercises the wire format end-to-end.

### Frame header (16 bytes)

| Offset | Length | Field | Notes |
|---|---|---|---|
| 0 | 4 | `payload_length` | Little-endian uint32. Bytes of the JPEG that follow. |
| 4 | 12 | reserved | Bambu fills with zeroes. |

After the header, exactly `payload_length` bytes of JPEG. The JPEG begins
`FF D8` (SOI) and ends `FF D9` (EOI). The reader drops a frame whose payload
fails that check.

### Reconnect

On EOF at a frame boundary the iterator terminates cleanly — that's how the
printer signals end-of-stream. EOF mid-payload raises `CameraStreamClosed`,
which the guard catches and treats as a transient outage worthy of a
watchdog alert.

## MQTT channel (TCP 8883, TLS)

| Item | Value |
|---|---|
| Port | 8883 |
| Transport | TLS. Self-signed broker cert; client uses `CERT_NONE`, `tls_insecure_set(True)`. |
| paho version | 2.x callback API (`CallbackAPIVersion.VERSION2`). |
| Username | `bblp` |
| Password | LAN access code |
| Protocol | MQTT 3.1.1 (`MQTTv311`) |
| Keepalive | 60 s |

### Topics

| Direction | Topic | Format |
|---|---|---|
| Subscribe | `device/{serial}/report` | JSON; printer pushes state updates. |
| Publish | `device/{serial}/request` | JSON; guard issues commands. |

### Commands

All commands share the same envelope:

```json
{"print": {"command": "<cmd>", "sequence_id": "0"}}
```

| Command | Effect |
|---|---|
| `stop` | Abort the current print. |
| `pause` | Pause the print; M-button on the printer can resume. |
| `resume` | Resume from pause. |

Commands are published at **QoS 1** (at-least-once). The integration test
asserts both the payload bytes and the QoS for the live wire.

### Report fields read

The guard consumes only two fields from the report:

| Field | Purpose |
|---|---|
| `print.gcode_state` | Drives arming. `RUNNING` arms; `FINISH` / `FAILED` / `IDLE` / `PAUSE` disarms. |
| `print.layer_num` | Logged for context; not used in decisions. |

Other fields (filament, fan speed, temperatures) are ignored. Adding a new
input means changing `PrinterState.update_from_report` and the state machine.

### Firmware authorization

Post-2024-05 firmware locks local MQTT, camera, and FTP behind **LAN Mode**
(or Developer Mode). Without it, `publish` returns "MQTT command
verification failed" and the printer takes no action. Enable LAN Mode on the
touchscreen and record the access code before installing the guard.
