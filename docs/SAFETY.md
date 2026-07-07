# Safety posture

## What this guard is for

A reasonable, automated last line of defence against runaway spaghetti
prints. It does *not* replace a smoke alarm, a fire-rated cabinet, or
occupant presence during a long print.

## What it cannot do

* It cannot see a hot-end fire, smoke, or thermal runaway. Those need
  hardware sensors and a real fire alarm.
* It cannot recover after a triggered stop — the print is over.
* It cannot detect failures the camera can't see (under-the-print
  delamination, internal layer shift on big parts).
* It will never be perfectly accurate. The whole design assumes both
  false positives and false negatives happen, and tries to minimise the
  worse one.

## Pause vs. stop

`action.mode` defaults to `pause`. The trade:

| Mode | Pros | Cons |
|---|---|---|
| `pause` | Filament saved; operator inspects before deciding. | Requires a person to be available to act. Hot-end stays warm. |
| `stop` | Frees the printer immediately; minimal continued risk from a live nozzle. | Discards the print; if the trigger was a false positive, the whole print is lost. |

Start in `pause` while you build trust in the model and the thresholds.
Promote to `stop` only after you've watched several real triggers behave
correctly and have a sense of the false-positive rate from `metrics.py`.

## Debounce rationale

`detector.consecutive_hits` (default 6) is the primary false-positive
defence. At ~1 fps, that's six seconds of continuous evidence before any
action. A single misclassified frame — flickering shadow, sudden filament
swap, momentary camera glitch — can never trigger the guard.

Six is a *floor*. If your model trends towards false positives, raise it
before lowering the threshold. The latency cost is trivial compared with
the alternative.

## Camera-loss policy

If the camera goes silent for `camera.timeout_s` while ARMED, the guard:

1. Logs a warning at WARNING level.
2. Sends **one** notification per outage via the configured notifier.
3. Does **not** publish a stop or pause.

Rationale: a missing frame is not evidence of failure. Treating it as one
would let any router glitch end every long print. The guard re-arms
silently when frames resume.

If you want a stop on extended camera loss, build that policy outside the
guard — e.g. a Home Assistant automation that watches the notifier and
issues its own command after a longer timeout.

## Dry-run is real

`--dry-run` and `tasks.ps1 run-dry` run the full live pipeline. Real
camera. Real detector. Real MQTT report parsing. Only the publish step is
skipped. Tests assert this; see `test_control_payload.py::test_dry_run_never_publishes`.

Never edit the dry-run path to "just this once" — if you need a real
publish, set `action.dry_run: false` and live with the consequences.

## Kill switch

`SIGINT` (Ctrl-C in interactive use; `systemctl stop` in service mode)
shuts the guard down cleanly. The printer keeps printing. There is no
on-purpose action published during shutdown.

If you need to stop the *printer* manually, do it from the touchscreen or
Bambu's app — not by killing the guard.

## Fire risk

A spaghetti print is not on its own a fire hazard, but it can become one
if the molten extrusion drapes onto the heated bed or wraps the hot-end.
The probability is low; the consequence isn't. The guard reduces the
window during which a runaway can develop, but it is not a substitute for:

* A smoke detector wired to a real alarm.
* Locating the printer in a non-flammable area (ideally a printed parts
  enclosure with a real fire-rated cabinet around it for unattended runs).
* Keeping the printer's filter clean (PLA dust + hot air is its own
  problem).

If you can't be near the printer for a long unattended print, prefer
`action.mode = stop` over `pause`. A paused printer with the hot-end
still hot is, marginally, worse than a stopped one.
