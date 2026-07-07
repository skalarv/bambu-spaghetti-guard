# Architecture

## Data flow

```
                 device/{serial}/report (MQTT 8883, TLS)
                          |  gcode_state, layer_num
                          v
   port 6000  +----------------------+   stop/pause   port 8883
   JPEG ~1fps |       GUARD          |--------------->  printer
  ----------> |   (state machine)    |   MQTT request
   camera     +----------------------+
                  |            |
                  v            v
              detector     notifier
            (YOLO+debounce) (ntfy/TG/HA)
```

Camera frames flow in at roughly 1 fps. Each frame is decoded, scored by the
detector, and pushed into the debouncer. State updates from the printer's
report topic shape the guard's posture (armed vs. disarmed). Only the guard
issues control commands, and only via the same MQTT broker the report stream
arrives on.

## Modules and responsibilities

| Module | File | Responsibility |
|---|---|---|
| Config | `src/spaghetti_guard/config.py` | Pydantic-validated config; secrets env-only. |
| Camera | `src/spaghetti_guard/camera.py` | `CameraBackend` ABC, `RawSocketBackend`, `LibBackend` stub. |
| Control | `src/spaghetti_guard/control.py` | paho-mqtt 2.x wrapper, exact payloads, state mirror, reconnect. |
| Detector | `src/spaghetti_guard/detector.py` | YOLO call adapter, threshold + class filter, N-of-N debouncer. |
| Notifier | `src/spaghetti_guard/notifier.py` | ntfy / Telegram / Home Assistant; failures suppressed. |
| Guard | `src/spaghetti_guard/guard.py` | State machine, watchdog, snapshot logging, run loop. |
| CLI | `src/spaghetti_guard/cli.py` | argparse front door for run / verify / replay / train / validate. |
| Mock printer | `verification/mock_printer.py` | Fake camera + fake MQTT broker for offline integration. |
| Replay | `verification/replay_harness.py` | Tune thresholds against recorded clips. |
| Metrics | `verification/metrics.py` | Precision / recall / latency / FP-per-print-hour. |

## State machine

```
IDLE ──(gcode_state RUNNING)──> ARMED
ARMED ──(frame hit)──> ALERTING (accumulating consecutive hits)
ALERTING ──(miss)──> ARMED            (debounce resets)
ALERTING ──(N consecutive hits)──> TRIGGERED (send action, notify)
TRIGGERED ──(action sent)──> COOLDOWN
any ──(gcode_state FINISH/FAILED/IDLE)──> IDLE   (disarm)
```

Two invariants:

1. **Detection runs only while `gcode_state == RUNNING`.** Without this guard,
   idle frames (bed clear, filament load, hot-end purge) generate trivial
   false positives.
2. **Camera loss never sends a stop.** The absence of evidence is not
   evidence of failure. The watchdog notifies once per outage and tries to
   reconnect; it never publishes a control command.

## Threading model

The live process spawns three concurrent paths:

* The main thread runs the guard loop (blocking camera iterator + per-frame
  processing).
* `paho.mqtt.Client.loop_start()` runs a paho-owned thread that delivers
  report messages into the thread-safe `PrinterState`.
* The watchdog runs interleaved on the main thread between frames; if a
  frame source blocks longer than `camera_timeout_s`, the camera socket's
  `recv_timeout_s` raises and the run loop's exception handler keeps the
  process alive while the watchdog flags the outage.

Tests collapse this to a single asyncio loop for the mock printer plus one
background thread for the guard.

## Failure modes & how the design contains them

| Failure | Containment |
|---|---|
| YOLO mis-classification on a single bad frame | N-of-N debounce. Default 6 consecutive hits required. |
| Camera disconnect mid-print | Watchdog notifies; control never sees a stop. Outer loop reconnects. |
| MQTT disconnect | paho reconnect with exponential backoff; report state stale but `last_update_ts` reveals it. |
| Bad config (typo in IP, missing secret) | Pydantic validation rejects at startup, before any socket opens. |
| Notifier API down | All notifiers wrap exceptions and log; the guard keeps going. |
| Operator forgot LAN Mode | MQTT connect succeeds but publish returns "verification failed"; logged loudly, no silent failure. |
