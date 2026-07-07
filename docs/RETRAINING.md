# Retraining loop

The guard's public-dataset model (currently `models/yolo11n-spaghetti.pt`) is
trained on generic 3D-print imagery from Roboflow Universe. For your specific
P1S chamber — camera angle, lighting, filament colours, print bed reflectivity
— the model will over- and under-trigger until you fold in chamber-specific
data. This is the loop for doing that (brief §7.3).

## The short version

```powershell
# 1. See what needs review
python training\train_from_snapshots.py --review

# 2. Open failure_snapshots\ in labelImg (or your annotator of choice).
#    - Draw boxes on real failures, save as .txt next to the .jpg.
#    - For false positives (guard fired but nothing was wrong), save an EMPTY
#      .txt file — the retrain script treats it as a negative example.

# 3. Promote your reviewed snapshots into the training set
python training\train_from_snapshots.py --promote --batch-tag $(Get-Date -f yyyyMMdd)

# 4. Fold the batch into the merged dataset and retrain
#    (one-time: add the new chamber-YYYYMMDD dir to build_sources() in merge_datasets.py)
python training\merge_datasets.py
python training\train.py --data training\data\merged\data.yaml --base yolo26n.pt --epochs 40 --batch 32 --name chamber-YYYYMMDD

# 5. Promote the new weights
copy runs\detect\runs\train\chamber-YYYYMMDD\weights\best.pt models\yolo11n-spaghetti.pt
```

## What snapshots are, and where they come from

Every time the guard's state machine transitions to `TRIGGERED`, it writes a
JPEG under `failure_snapshots/` named:

```
trigger-YYYYMMDD-HHMMSS-<class>-<confidence>.jpg
```

Example: `trigger-20260706-133205-spaghetti-0.68.jpg` — a `spaghetti` detection
at 0.68 confidence, on 2026-07-06 at 13:32:05 UTC.

These files are the raw material for retraining. The class name embedded in
the filename is what the *current* model thought it was; the retraining loop's
job is to have you (the operator) confirm or correct that.

## Reviewing snapshots

You need an annotation tool that emits YOLO-format text files. Two good
options:

**labelImg** — free, local, one binary. Install with `pip install labelImg`
into a *separate* venv (don't pollute the guard venv). Point it at
`failure_snapshots/` and:

- Set **Save Format** to `YOLO`
- Set **PascalVOC → YOLO** conversion off
- Draw one box per visible failure, pick the class from the dropdown
- Save. It writes `<same-name>.txt` next to the JPG.

**Roboflow web UI** — better if you already have a Roboflow account. Upload
the snapshots as a new dataset version, label in the browser, export as
YOLOv8, and unpack the resulting `.txt` files back next to the JPGs.

### The false-positive rule

If the guard fired but nothing was actually wrong, **save an empty `.txt`
file** for that snapshot. `train_from_snapshots.py` treats an empty label as
"this frame contains no failure — use it as a negative example." Negatives
train the model to shut up on things that look like spaghetti but aren't
(strings of filament from unrelated parts, reflections, purge waste, etc.).

**Do not delete false-positive snapshots.** They're the most valuable data
you have — the model already got them right on the true positives, but it's
guessing on the false positives, and negatives are how you fix that.

## Promoting to the training set

`train_from_snapshots.py --promote` copies your reviewed snapshots into
`training/data/chamber-<batch-tag>/` in the same YOLOv8 layout as the
Roboflow-hosted datasets. Each batch gets its own dir so you can prune old
ones later (older chamber conditions may not match current camera / filament
setup).

The 7 canonical classes are:

```
0 spaghetti
1 stringing
2 blob
3 crack
4 detachment
5 over_extrusion
6 under_extrusion
```

Match whichever your annotator produces to those IDs.

## Merging into the training dataset

After promoting a batch, add it as a source in
`training/merge_datasets.py` (`build_sources()`). One entry per chamber batch:

```python
sources.append(SourceSpec(
    root=data_root / "chamber-20260706",
    slug="cham0706",
    remap={i: i for i in range(7)},  # canonical -> canonical, no remap needed
))
```

Then re-run:

```powershell
python training\merge_datasets.py
```

That regenerates `training/data/merged/` with all sources folded in.

## Retraining

Once the merged set is refreshed, re-run `train.py` on the same base you used
originally:

```powershell
python training\train.py --data training\data\merged\data.yaml \
    --base yolo26n.pt --epochs 40 --batch 32 --name chamber-20260706
```

**Why 40 epochs and not 60?** Once you have chamber-specific data mixed into
the training set, you're fine-tuning from the previous best rather than
retraining from scratch. 40 epochs is a good balance between "long enough to
adapt" and "short enough not to overfit the new batch."

If you want to warm-start from your last model instead of the public base:

```powershell
python training\train.py --data training\data\merged\data.yaml \
    --base models\yolo11n-spaghetti.pt --epochs 30 --batch 32 --name chamber-20260706
```

## Safe deployment

Never overwrite `models/yolo11n-spaghetti.pt` with the fresh weights until
you've validated them. Path:

```powershell
# 1. Copy the new weights under a versioned name
copy runs\detect\runs\train\chamber-20260706\weights\best.pt models\chamber-20260706.pt

# 2. Point the config at the new weights (or use --dry-run)
#    Edit config.yaml: detector.model_path: models/chamber-20260706.pt
python -m spaghetti_guard run --dry-run --viewer --config config.yaml

# 3. Watch it run a full print. If the new model behaves, promote it.
copy models\chamber-20260706.pt models\yolo11n-spaghetti.pt
```

The dry-run flag means the guard runs the full pipeline (camera + detection +
snapshotting) but never publishes stop/pause — perfect for validating a new
model against a real print without risking the print itself.

## How often to retrain

There's no hard rule. Practical heuristics:

- **Every 10-20 reviewed snapshots**: usually enough to move the needle on a
  specific failure mode the model was consistently getting wrong.
- **After a filament / lighting / camera-angle change**: chamber-specific
  data captured under old conditions might actively mislead the new setup.
- **Before a critical print**: if you've been accumulating snapshots for a
  while and haven't refreshed the model, do it now.

## What NOT to include in training

- Snapshots from prints you know were **already failed before the guard fired**
  (i.e., the guard was late — the failure was obvious minutes earlier).
  Those teach the model that the failure state is normal.
- Snapshots where the chamber was in a weird state (door open, purging,
  bed clearing). Those aren't representative of "watching a print in progress."
- Snapshots older than ~6 months if your setup has meaningfully changed.

## Regression gate (mandatory before promoting new weights)

Never overwrite `models/yolo11n-spaghetti.pt` without running the gate:

```powershell
.\tasks.ps1 model-gate            # or: make model-gate
```

This validates the active weights against `training/data/merged` and fails
(exit 6) if any metric drops below the floors: **precision >= 0.80,
recall >= 0.60, mAP50 >= 0.65** (the 2026-07-07 baseline is P=0.863,
R=0.684, mAP50=0.721). To gate candidate weights before promotion:

```powershell
python training\validate.py --weights runs\detect\<name>\weights\best.pt `
  --data training\data\merged\data.yaml `
  --min-precision 0.80 --min-recall 0.60 --min-map50 0.65
```

The summary (with per-floor pass/fail detail) lands at
`runs/validate/summary.json`.

Known weak spot as of 2026-07-07: the `detachment` class has near-zero
recall (only 9 instances in the validation split). Since detachment is one
of the two classes that FIRE the guard, prioritize collecting and labeling
detachment snapshots in the retraining loop.
