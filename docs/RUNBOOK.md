# Operator runbook

## Day-to-day check (15 seconds)

1. Is the guard process alive? (`Get-Process spaghetti-guard` or `systemctl status`.)
2. Is the most recent log line within the last minute?
3. Is `failure_snapshots/` empty since the last review? If not, review them.

## Tuning thresholds

Symptom: **trips during normal prints**.
- Raise `detector.conf_threshold` by 0.05 and re-run `tasks.ps1 replay <clip>`
  against the clip the false trigger came from.
- Increase `detector.consecutive_hits` by 1 or 2. The cost is a 1-2 frame
  longer latency, which at ~1 fps is acceptable.
- If the false hits all come from the same visual feature (e.g. a particular
  filament colour), capture that footage and fold it into training as
  negative examples. See "Retraining loop".

Symptom: **doesn't trip on a real spaghetti**.
- Replay the clip with `--conf 0.45 --window 3` and find the first frame the
  detector actually scores above threshold. If the model sees nothing, the
  problem is the model, not the threshold — retrain.
- If the detector scores in the 0.4–0.5 band but never crosses 0.55, lower
  the threshold to 0.50 *and* keep the debounce window high so single
  borderline frames don't fire.

Symptom: **camera-loss alert keeps firing**.
- Increase `camera.timeout_s`. The default 15 s assumes ~1 fps. If your
  printer streams slower, raise it; if you've changed firmware and frames
  arrive in bursts, raise it.
- Confirm the printer's static IP hasn't drifted.

## Reviewing snapshots

Every TRIGGERED event writes the JPEG that fired it into
`snapshots.dir/trigger-<timestamp>-<class>-<conf>.jpg`. Once a week:

1. Open each snapshot. Was it a real failure?
2. If yes — keep, mark for the next training round.
3. If no — keep, mark as a false-positive sample for retraining.
4. If you have both kinds piling up, the model needs more domain-specific
   data. See "Retraining loop".

## Retraining loop

Full guide: **`docs/RETRAINING.md`**. Short form:

The stock P1S chamber angle and lighting differ from public-dataset footage,
so the shipped model will trend towards over-triggering on this specific
chamber until adapted. Plan to retrain after the first 50 hours of guard
runtime, or after ~10-20 reviewed snapshots:

```powershell
# 1. See what's unlabelled
python training\train_from_snapshots.py --review

# 2. Label the snapshots in labelImg / Roboflow (YOLO format .txt next to .jpg).
#    An EMPTY .txt = "false positive; use as a negative example" — DO save these.

# 3. Promote to training set, rebuild merged/, retrain
python training\train_from_snapshots.py --promote --batch-tag $(Get-Date -f yyyyMMdd)
# One-time: add the new chamber-YYYYMMDD source to build_sources() in merge_datasets.py.
python training\merge_datasets.py
python training\train.py --data training\data\merged\data.yaml --base models\yolo11n-spaghetti.pt \
    --epochs 40 --batch 32 --name chamber-YYYYMMDD

# 4. Safely deploy the new weights: keep the old file, dry-run the new one first,
#    only overwrite yolo11n-spaghetti.pt once you've watched it behave for a print.
copy runs\detect\runs\train\chamber-YYYYMMDD\weights\best.pt models\chamber-YYYYMMDD.pt
# edit config.yaml -> detector.model_path: models\chamber-YYYYMMDD.pt, then --dry-run.
```

Validate with `tasks.ps1 validate` — the PR curve and `fp_per_print_hour`
are the numbers that matter. Don't promote a model whose FP/print-hour went
*up* even if precision went up.

## Common failures

| Symptom | Diagnosis | Fix |
|---|---|---|
| `MQTT command verification failed` in logs | LAN Mode disabled, or wrong access code. | Re-enable LAN Mode on touchscreen; re-copy access code into `.env`. |
| `socket.gaierror` on startup | `BAMBU_IP` wrong or printer offline. | Confirm the static IP; ping the printer. |
| Guard armed but no frames | Camera channel not authorised. | Check the auth packet bytes against `docs/PROTOCOL.md` for your firmware. |
| `ultralytics` not installed | Live mode without the optional GPU deps. | `pip install ultralytics opencv-python torch --index-url ...` per `docs/INSTALL.md`. |
| Notifier silent | API token expired, network down. | Tail the log — every notify failure logs an exception line. |
| Guard pauses then nothing happens | Pause-mode without a person nearby. | Set `action.mode = stop` once you trust detection. |

## Stop / start / restart

Linux: `sudo systemctl restart spaghetti-guard`.

Windows (NSSM): `nssm restart SpaghettiGuard`.

Windows (interactive dev): close the PowerShell window; the guard cleans up
on SIGINT.

## Dry-run for a new firmware

Whenever the P1S firmware updates, the camera handshake may change. Run
`tasks.ps1 run-dry` for one full print before re-enabling live mode. The
dry-run validates the wire format end-to-end without risking a live stop.
