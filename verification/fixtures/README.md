# Verification fixtures

Synthetic JPEG clips for `replay_harness.py` and `metrics.py` smoke / regression tests.

Generated on demand by `tests.test_replay_metrics._make_clip` (no binaries
checked into git). Each clip is a folder of sequentially named JPEGs plus a
`labels.json` sidecar.

The fixtures use the same marker-based JPEG format as
`verification.mock_printer.make_jpeg`, so the marker detector recognises
them out-of-the-box.
