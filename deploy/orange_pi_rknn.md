# Orange Pi 3B / RK3566 edge inference

Optional always-on inference target. The RK3566 NPU is roughly 1 TOPS — fine
for yolo11n at ~1 fps. Use this when you want the guard running without a
GPU workstation always on.

## What you need

* Orange Pi 3B with Joshua-Riek's Ubuntu 24.04 (RKNN toolkit is awkward on
  vendor BSPs).
* A USB-attached storage device for the venv (microSD wears out fast).
* Rockchip's `rknn-toolkit2` (host side, for the conversion) and
  `rknn-runtime` / `rknn-toolkit-lite2` (target side).
* The trained YOLO `.pt` from `training/` and a calibration image set.

## Workflow

### 1. Train as usual on the workstation

```powershell
.\tasks.ps1 train --data training\data\merged\data.yaml
.\tasks.ps1 validate --weights models\yolo11n-spaghetti.pt --data training\data\merged\data.yaml
```

You want the PR curve and `fp_per_print_hour` before bothering with the
conversion.

### 2. Export to ONNX

On the workstation:

```powershell
.venv\Scripts\python -c "from ultralytics import YOLO; YOLO('models/yolo11n-spaghetti.pt').export(format='onnx', imgsz=640, opset=12)"
```

### 3. Convert ONNX → RKNN

On a Linux host with `rknn-toolkit2`, follow Rockchip's model-conversion
examples (there is no conversion script in this repo yet — the snippet below
is the shape of what you'll write, based on the `rknn-toolkit2` API):

```python
# convert_rknn.py — sketch; consult Rockchip's yolo examples for the details
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform="rk3566")
rknn.load_onnx("models/yolo11n-spaghetti.onnx")
rknn.build(do_quantization=True, dataset="calib_images.txt")
rknn.export_rknn("models/yolo11n-spaghetti.rknn")
```

This step is hardware-vendor land; expect to read Rockchip's docs and to
tune their quantisation knobs. The conversion is the only Rockchip-specific
piece — everything else is normal Python.

### 4. Run on the Pi

```bash
ssh orangepi
cd /opt/bambu-spaghetti-guard
./.venv/bin/spaghetti-guard run --config config.yaml
```

You'll need a small swap-in for `detector.load_yolo_model` that returns an
RKNN runner instead of an Ultralytics YOLO. Keep the swap behind a
config flag; do not break the workstation/live path.

Reference path: `SENTINEL-B1` (separate project) — same RKNN conversion
recipe.

## Performance budget

| Stage | Target |
|---|---|
| TLS handshake + auth | < 1 s on Wi-Fi |
| Per-frame decode | < 100 ms (cv2 software) |
| YOLO inference (RKNN int8 yolo11n) | < 200 ms |
| Headroom | At 1 fps you have ~700 ms of slack per frame. |

If you can't hit those numbers, the per-frame pipeline will fall behind and
detection will lag. Drop input resolution before changing the model.

## What doesn't move to the Pi

* The MQTT control path. paho-mqtt runs fine on aarch64.
* The notifier. ntfy / Telegram / HA over HTTPS all work.
* The unit / integration test suite. Run those on the workstation; on the
  Pi run only `spaghetti-guard run --dry-run` for a full print before going
  live.
