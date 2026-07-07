# Model weights

`config.yaml` expects the active detector at `models/yolo11n-spaghetti.pt`.

## Provenance

| Weights file | Trained from | Dataset | License | Val (best.pt) | Notes |
|---|---|---|---|---|---|
| `yolo11n-spaghetti.pt` | `yolo26n.pt` (Ultralytics base) | Merged CC BY 4.0 (24,098 imgs, 7 classes): spaghetti-3d + 3d-printing-flaws-remapped + syLucauc — see `training/datasets.md` | CC BY 4.0 attribution required if redistributed | P=0.86, R=0.68, mAP50=0.72, mAP50-95=0.42 (peak epoch 44: mAP50=0.734) | 60 epochs, imgsz=640, batch=32, RTX 5070, 8h14m (2026-07-06). Trained by the merge+train pipeline in `training/`. Filename kept as `yolo11n-spaghetti.pt` for config compatibility even though the architecture is yolo26n. |
| `yolov8s.pt` | (kept as demo baseline) | COCO 80-class | AGPL-3.0 (Ultralytics) | — | Only useful for pipeline-sanity demos (waving a hand). Not for real spaghetti detection. |

## Retraining

New weights land in `runs/detect/runs/train/<name>/weights/best.pt`. To
promote:

```powershell
copy runs\detect\runs\train\<name>\weights\best.pt models\<name>.pt
# Validate under --dry-run first:
#   edit config.yaml -> detector.model_path: models\<name>.pt
python -m spaghetti_guard run --dry-run --viewer
# When trusted, promote to the canonical name:
copy models\<name>.pt models\yolo11n-spaghetti.pt
```

## Compliance

Per brief §1: **no Chinese-origin models**. Approved sources:

- Ultralytics-published YOLO base weights (`yolo11n.pt`, `yolo11s.pt`, `yolo26n.pt`)
- Self-trained derivatives from those bases
- CC-BY / MIT Roboflow Universe datasets from clearly identified maintainers

## Why this folder is gitignored

`.gitignore` excludes `*.pt`, `*.onnx`, and `*.rknn` so weights stay out of
the repo. Distribute via a separate channel (a release artifact, a private
bucket, or an out-of-band file). `README.md` in this folder is checked in.
