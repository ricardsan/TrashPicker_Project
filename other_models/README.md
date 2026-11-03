# Mask R-CNN and RF-DETR, with run artifacts that match the YOLO ones

Trains non-YOLO detectors on the same data, with the same hyperparameters, and writes the same run
directory as the Ultralytics YOLO runs in `runs/`, so they can be compared panel for panel in a
paper. Two architecture families are registered:

| family | models | what it is |
|---|---|---|
| `torchvision` | `maskrcnn`, `maskrcnn_v2`, `fasterrcnn_v2` | ResNet-50 + FPN, region proposals, per-RoI box and mask heads |
| `rfdetr` | `rfdetr_seg_small` | DINOv2-small transformer + deformable-DETR decoder + mask head |

Everything below assumes the project venv is active (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS), so `python` is the venv one with torch.

---

## Quick start

```bash
# 1. build the COCO annotation files (once, ~4 s)
python other_models/prepare_data.py --all

# 2. look at what you are about to train on
python other_models/visualise_masks.py --source dataset_seg --labels coco --n 6   # the labels themselves
python other_models/visualise_masks.py --source dataset_seg --n 6                 # + check the conversion

# 3. train both Mask R-CNN generations with the 26-original-finetune-nms recipe
python other_models/train.py --model maskrcnn maskrcnn_v2 --preset FINETUNE_NMS

# 4. the same recipe for the transformer. Note the RFDETR_ prefix: RF-DETR cannot run at
#    imgsz 640 at all, so it has its own presets -- see "RF-DETR" below for why.
python other_models/train.py --model rfdetr_seg_small --preset RFDETR_FINETUNE_NMS
```

## The four files you will want to edit

| file | what you change there |
|---|---|
| `config.py` | hyperparameters. Presets mirroring the YOLO YAMLs, field names copied verbatim from them |
| `datasets.py` | which dataset to train on (`ACTIVE_SOURCE`, or `--source`) |
| `augment.py` | the augmentation pipeline itself |
| `models.py` | which network (`MODELS` registry — one entry per model) |

The rest (`prepare_data.py`, `metrics.py`, `artifacts.py`, `engine.py`, `train.py`,
`rfdetr_adapter.py`) should not need touching to run a different experiment.

---

## The datasets

| source | images | masks | use |
|---|---|---|---|
| `dataset_seg` | 611 train / 152 val | real polygons | **the default.** Box *and* Mask metrics |
| `dataset_v2` | 537 train / 135 val | none in the current export | Box metrics only, see below |
| `taco` | 706 / 208 / 100 | real polygons | pretraining |

`prepare_data.py` turns each of them into COCO JSON plus a `provenance.json` sidecar recording
input hashes, counts and anything skipped. The COCO files live next to the dataset they describe,
so each dataset is self-contained and portable:

```
datasets/dataset_seg/annotations/{train,val}.json            <- converted from datasets/dataset_seg/labels/*.txt
dataset_v2/annotations/{train,val}.json
datasets/taco_yolo/annotations/{train,val,test}.json
```

They are ordinary COCO: `images` with `file_name`/`width`/`height`, `annotations` with
`segmentation` (polygon lists), `bbox` (x, y, w, h), `area`, `iscrowd`, and `categories` with
0-based ids matching `data_v2.yaml` (glass 0, metal 1, paper 2, plastic 3). Any COCO reader can
use them, not just this code.

**`dataset_v2` has no masks yet.** The 2026-10-05 Label Studio export
(`coco-project-2-at-2026-10-05-13-26-ffd7b114/result.json`) has `segmentation: []` on all 1848
annotations — it is boxes only. Training on it still works: each mask becomes the filled
bounding box, `has_real_masks` is `False`, and the four `metrics/*(M)` columns are written as
`NaN` rather than scoring rectangles as if they were segmentations. Drop in an export that has
polygons, re-run `prepare_data.py --source dataset_v2`, and the mask path lights up with no code
change. (`test_the_current_export_has_no_polygons` is designed to fail when that happens, as the
signal.)

The export's `file_name` fields point at another machine
(`../../home/rica/.local/share/label-studio/media/upload/2/…`); they are rewritten to the
basename and matched against `dataset_v2/images/{train,val}`. 672 of 763 resolve; the other 91 are
`IMG_17xx.jpeg` photos that only exist in `dataset_seg`, and they are listed in the sidecar rather
than dropped silently.

**TACO is built from the reduced class mapping.** `taco_class_mapping_taco_yolo_reduced.json`
drops 25 of TACO's 60 categories (foil, bottle caps, lids, broken glass, films and wrappers,
scrap metal, styrofoam …); those annotations are not written at all. Result: 35 categories, 1014
images, 1985 annotations, split 706/208/100 as `taco_to_yolo.py` assigned them.

> Note: `datasets/taco_yolo/labels/` on disk is the older **full**-mapping build (1453
> images / 3553 annotations), which is what the YOLO TACO pretrain actually trained on. So the
> Mask R-CNN TACO pretrain is not label-identical to the YOLO one. It only affects the pretraining
> stage, not the `dataset_seg` fine-tune the comparison rests on. To make them match, regenerate
> the YOLO labels from `taco_yolo/source_index.json` with `datasets/taco_to_yolo.py`
> and re-run the YOLO pretrain; nothing in `other_models/` changes.

### EXIF: the one thing to not get wrong

**OpenCV applies EXIF rotation.** `cv2.imread` and `cv2.imdecode` both return the *upright*
image, not the stored one, and Ultralytics loads through `cv2.imdecode` — so the YOLO runs trained
on upright pixels, and so does this code. These photos are stored 4032×3024 with EXIF orientation
6, which means everything here sees 3024×4032.

All three annotation sets are already in that upright frame, so nothing is rotated anywhere. The
one trap: `result.json`'s `width`/`height` fields report the *stored* size, which contradicts its
own coordinates — `prepare_data.py` ignores them and measures the images instead. Measured, out of
its 1512 resolved annotations: 1512 fit the upright frame, 951 fit the raw one, and 20 collapse to
nothing entirely if you clip to raw.

---

## Training

```bash
# the two fine-tune recipes, both Mask R-CNN generations
python other_models/train.py --model maskrcnn maskrcnn_v2 --preset FINETUNE_NMS
python other_models/train.py --model maskrcnn maskrcnn_v2 --preset FINETUNE_AUGMENT

# pretrain on TACO, then fine-tune on dataset_seg from that checkpoint
python other_models/train.py --model maskrcnn_v2 --source taco --preset TACO_PRETRAIN
python other_models/train.py --model maskrcnn_v2 --preset FINETUNE_AUGMENT \
    --weights runs/segment/maskrcnn-taco/taco-pretrain_v/weights/best.pt

# quick check that the artifacts come out right
python other_models/train.py --model maskrcnn_v2 --epochs 3 --name smoke_v
```

Every `config.py` field is also a flag: `--epochs 3 --batch 8 --lr0 0.0005 --imgsz 512`.

### RF-DETR

```bash
# the two fine-tunes
python other_models/train.py --model rfdetr_seg_small --preset RFDETR_FINETUNE_NMS
python other_models/train.py --model rfdetr_seg_small --preset RFDETR_FINETUNE_AUGMENT

# the TACO chain
python other_models/train.py --model rfdetr_seg_small --source taco \
    --preset RFDETR_TACO_PRETRAIN
python other_models/train.py --model rfdetr_seg_small --preset RFDETR_FINETUNE_AUGMENT \
    --weights runs/segment/rfdetr-taco/taco-pretrain_v/weights/best.pt
```

`rfdetr` is a declared requirement but the **base** package only, not `rfdetr[train]`:
`engine.py` runs the loop itself, so all that is taken from the library is the `nn.Module`, its
`SetCriterion` and the COCO-pretrained weights. `rfdetr_adapter.py` wraps those in one module that
answers torchvision's interface — `model(images, targets)` returning named losses in `train()`,
`model(images)` returning boxes/scores/labels/masks in `eval()` — which is why nothing in
`datasets.py`, `augment.py`, `metrics.py` or `engine.py` needed an RF-DETR branch. The adapter is
also where three convention differences are converted, each of which would otherwise be a silent
accuracy loss rather than an error: ImageNet pixel normalisation, xyxy pixels → normalised cxcywh,
and 1-based labels → the 0-based COCO `category_id` RF-DETR indexes by.

**Why RF-DETR has its own presets.** Its resolution must satisfy
`resolution % (patch_size * num_windows) == 0`, which is 24 for the segmentation variants — and
`640 % 24 == 832 % 24 == 16`. The YAML values are not suboptimal, they are *invalid*, so the presets
run at the nearest valid values instead. Everything else is inherited from the Mask R-CNN presets,
so `test_presets_mirror_the_yolo_yamls` still diffs the whole recipe against the YAMLs.

Five declared deviations, all written up in `config.py` section 4:

| key | YAML | RF-DETR | why |
|---|---|---|---|
| `imgsz` | 640 / 832 | **648 / 840** | the divisibility rule above; nearest valid values, +1.25% and +0.96% |
| `batch` | 16 | **4 / 2** | 16 does not fit in 16 GB. At 648 the patch grid is 54×54 = 2916 tokens against the checkpoint's native 32×32 = 1024, i.e. 2.85× the activations (840 → 4900, 4.8×). `nbs` stays 64, so accumulation holds the **effective batch at 64** exactly as the YAML intends. Measured below |
| `patience` | 100 | **10** (fine-tunes only) | the longest job in the comparison. A *stopping* rule, not a hyperparameter: every epoch that runs uses the YAML recipe unchanged, and `final_eval` still writes all 24 files from `best.pt` afterwards, so only `results.csv` is shorter. The TACO pretrain keeps its YAML's `patience: 15` |
| `warmup_bias_lr` | 0.1 | **0.001** (fine-tunes only) | **the largest deviation here.** The YAML value trains this architecture to mAP 0.002; 0.001 reaches 0.881. Measured below. The TACO pretrain keeps its YAML's 0.01 |
| `iou` | 0.7 | recorded, meaningless | a DETR is NMS-free — set prediction has no duplicate-suppression step for a threshold to apply to. What bounds its detection count is `num_queries × class columns` = 500, comfortably above `max_det: 300`, and `args.yaml` records the resolved figure as `max_det_effective` |

A reader comparing epoch counts across the table will notice the RF-DETR fine-tune ran fewer than
150 epochs; that is the `patience` row, not a crash.

**What `batch` actually costs**, measured on the 16 GB RTX 3080 laptop, one epoch of `dataset_seg`
at imgsz 648:

| `batch` | peak VRAM | s/epoch | mAP50-95(M) after 1 epoch |
|---|---|---|---|
| **4** (the preset) | 7.2 GB | 84.7 | 0.381 |
| 8 | 14.6 GB | 79.6 | 0.212 |

Mask R-CNN takes 84 s/epoch at imgsz 640 batch 16, so the transformer costs essentially nothing in
wall clock despite the smaller batch — a 150-epoch fine-tune is ~3.5 h either way. 8 is not the
default because it buys 6% for 89% of the card (1.8 GB of headroom on a GPU that also drives the
display, across a 3.5-hour run) and, at a fixed `nbs` of 64, halves the optimiser steps per epoch —
which is what the last column shows. Raise it with `--batch` if the card is dedicated.

One operational note: `rfdetr` resolves a bare checkpoint filename to `~/.roboflow/models`
(override with `$RF_HOME`), but pydantic does not validate *default* field values, so left alone it
downloads 134 MB into the current working directory. `rfdetr_adapter.model_config` re-states the
default explicitly to make that validator fire; a test guards it.

### Why `warmup_bias_lr` is 0.001 and not the YAML's 0.1 — measured

This is the one place where the presets do not mirror the YAMLs for an architectural reason, but
because the YAML value **does not train RF-DETR at all**. Three runs of `RFDETR_FINETUNE_AUGMENT`
on `dataset_seg`:

| run | box_loss | mask_loss | aux_loss | mAP50(B) | mAP50-95(B) | mAP50-95(M) |
|---|---|---|---|---|---|---|
| 1 epoch, warmup inactive¹ | 0.96 | 3.10 | 26.6 | 0.473 | 0.354 | 0.381 |
| 3 epochs, `warmup_bias_lr: 0.1` (the YAML) | 2.24 | 7.55 | 46.2 | **0.002** | **0.000** | **0.002** |
| 3 epochs, `warmup_bias_lr: 0.001` (the preset) | 0.20 | 0.86 | 10.1 | **0.881** | **0.748** | **0.743** |

¹ `engine.py` clamps `warmup_epochs` to `epochs - 1`, so a 1-epoch run has no warmup at all —
which is why the shortest run outscores the YAML-faithful one. That is the clue that pointed at
warmup rather than at `lr0`.

A learning rate of 0.1 on every bias parameter wipes out the COCO weights inside the first epoch,
and three epochs is not enough to climb back. YOLO's loss and architecture tolerate it; a DETR's
do not — rfdetr's own defaults are `lr` 1e-4 and `lr_encoder` 1.5e-4, an order of magnitude below
the YAML's `lr0: 0.001`, with no bias spike at all. At 0.001, three epochs reach
0.748 + 0.743 = **1.49 fitness**, already approaching the 1.766 the fully-trained YOLO run this is
compared against recorded over 150.

Unlike the other four deviations, this one is **a judgement, not a constraint**: 648 and batch 4 are
forced by the architecture and the card, whereas 0.1 would run fine — it would just produce a
number worth nothing. Reporting a 0.002 mAP as "RF-DETR under the YOLO recipe" would be the less
honest choice. To reproduce the YAML value anyway:

```bash
python other_models/train.py --model rfdetr_seg_small \
    --preset RFDETR_FINETUNE_AUGMENT --warmup_bias_lr 0.1
```

Both values are recorded in `args.yaml`, so any run says which it used. The Mask R-CNN presets are
untouched at 0.1, since the YOLO runs they are compared against used it.

Also deliberately *not* done, though it is the DETR convention: a separate lower learning rate for
the backbone (rfdetr's own defaults are `lr_encoder` 1.5e-4 against `lr` 1e-4, plus
`lr_vit_layer_decay` and `lr_component_decay`). Ultralytics' three parameter groups are the recipe
being mirrored, and a fourth group would both depart from it and overflow `results.csv`'s three
`lr/pg*` columns. How much of the transformer trains is controlled by `trainable_encoder_blocks`
instead — 1 of the 12 DINOv2 blocks for the fine-tunes, 7 for the pretrain, which is the same
fraction of the backbone that `freeze: 10` and `freeze: 5` give Mask R-CNN. The table in
`config.py` sets all three knobs side by side.

One happy coincidence worth knowing: RF-DETR's `mask_downsample_ratio` is 4 and
`metrics.MASK_RATIO` is 4, so asking `PostProcess` not to upsample returns masks at exactly the
resolution the evaluator scores at. That skips a round trip *and* removes the per-image mask cost
that forces Mask R-CNN to validate in smaller batches — `val_batch()` gates that cap on the family
for precisely this reason.

### The TACO chain

The two commands above are the pair `taco-pretrain/26-taco-pretrain.yaml` and
`taco-pretrain/26-taco-finetune.yaml` describe. The fine-tune stage has no preset of its own
because it does not need one: `26-taco-finetune.yaml` is hyperparameter-identical to
`26-original-finetune-augment.yaml` — only `data`, `model`, `name` and `project` differ — so
`FINETUNE_AUGMENT` is already it, and a test diffs the preset against *both* YAMLs.

The two stages differ in more than their data: the pretrain runs at `imgsz 832` with SGD and
`trainable_backbone_layers=3`, the fine-tune at 640 with Adam and `1`. That asymmetry is the point
— the pretrain is where the backbone is supposed to move.

TACO and `dataset_seg` carry the same four classes, so the pretrained box and mask heads transfer
unchanged and nothing like YOLO's `cls_remap` is needed. The fine-tune prints how many tensors it
loaded from the checkpoint, and records the path in its `args.yaml` as `pretrained_from`; a
checkpoint trained on a different class count is refused with an explanation rather than a wall of
`size mismatch` lines.

### What is reproduced from Ultralytics, and what is not

Reproduced, because otherwise "the same hyperparameters" would not be true:

* three optimiser param groups (decayed weights / norm weights / biases) → the real `lr/pg0..2`
* the linear schedule `lf(e) = max(1 - e/epochs, 0) * (1 - lrf) + lrf`, or `one_cycle` for `cos_lr`
* warmup over `warmup_epochs × batches` iterations: bias LR falls from `warmup_bias_lr`, the rest
  rise from 0, momentum rises from `warmup_momentum`, accumulation ramps to `nbs / batch`
* gradient accumulation to the nominal batch size `nbs`, and gradient clipping at `max_norm=10`
* `close_mosaic`: mosaic, copy-paste, mixup and cutmix are switched off for the final N epochs —
  which rebuilds the dataloader, because persistent workers would otherwise never see the change
* `copy_paste` (used by the TACO pretrain at 0.25): each object's mirror image is pasted back into
  the picture, but only where it would cover less than 30% of anything already there, taking the
  least intrusive candidates first until `round(p × eligible)` have landed. `p` is a fraction of
  objects, not a probability
* `box_candidates`: instances cropped to under 1% of their pre-warp area are dropped, not kept
  as slivers still labelled as whole objects
* mAP, P/R and the curves come from `ultralytics.utils.metrics` itself, so "mAP50-95" means
  exactly what it means in the YOLO runs
* `best.pt` is chosen by `fitness = mAP50-95(M) + mAP50-95(B)`, Ultralytics' `SegmentMetrics` rule

Not reproduced, listed with reasons at the bottom of `config.py`: YOLO's per-term loss gains
(`box`, `cls`, `dfl`, …), `nms`, `overlap_mask`, `mask_ratio`, `bgr`, `copy_paste_mode` (only
`flip` is implemented, which is what all our configs use — the other mode raises rather than
silently substituting), and `erasing` (which Ultralytics applies to classification only, so the
YOLO segment runs never used it either despite it appearing in their YAML).

`freeze` → `trainable_backbone_layers` is a judgement call rather than a translation; the chosen
correspondence is written out as a table in `config.py`. Note that yolo26's backbone is layers
0–10 of 24, so the fine-tune configs' `freeze: 10` freezes 10 of 11 backbone layers — hence
`trainable_backbone_layers=1` — while the TACO pretrain's `freeze: 5` gives `3`. Nothing derives
one from the other; set both together.

Masks are scored at `imgsz/4`, which is the resolution Ultralytics' segment validator compares at.
That also keeps predicted masks from costing gigabytes: at full resolution one batch of 16 images
with `max_det=300` would be ~2 GB of masks.

---

## The run directory

A run writes the same set as
`runs/segment/runs/segment/original-yolo26-seg/original-finetune-nms_v/` -- 24 top-level files
plus `weights/`:

```
args.yaml  results.csv  results.png  labels.jpg
BoxP_curve.png  BoxR_curve.png  BoxF1_curve.png  BoxPR_curve.png
MaskP_curve.png MaskR_curve.png MaskF1_curve.png MaskPR_curve.png
confusion_matrix.png  confusion_matrix_normalized.png
train_batch{0,1,2}.jpg  + the first three batches after mosaic closes
val_batch{0,1,2}_labels.jpg  val_batch{0,1,2}_pred.jpg
weights/best.pt  weights/last.pt
```

`results.csv` has the same 23 columns in the same order as the YOLO one; only the five loss names
differ, because the architectures have different losses. `artifacts.csv_columns()` builds the header
from whichever five the model has (`ModelSpec.loss_columns`):

| YOLO | Mask R-CNN | RF-DETR |
|---|---|---|
| `box_loss` | `box_loss` (RoI box regression) | `box_loss` (L1) |
| `seg_loss` | `mask_loss` | `mask_loss` (point-sampled BCE + Dice) |
| `cls_loss` | `cls_loss` | `cls_loss` (IA-BCE) |
| `l1_loss` | `rpn_box_loss` | `giou_loss` |
| `sem_loss` | `rpn_obj_loss` | `aux_loss` |

`aux_loss` is why RF-DETR gets five columns rather than three. A DETR computes its loss at every
decoder layer and at the encoder output, so its criterion returns 25 weighted terms of which only
5 belong to the final layer. The other 20 are a large share of the gradient, and the engine's
`if k in totals` filter would have dropped them from `results.csv` while still backpropagating them
— a loss curve that is not the quantity being minimised. Summed into their own column, the five
columns add up to exactly what the optimiser sees, which a test asserts.

**Read the RF-DETR loss panels with care.** `aux_loss` is about 79% of the total (measured: 58.1 of
73.2), so it sets the y-scale and the `box`/`mask`/`cls`/`giou` panels sit near the axis next to the
YOLO panels they are placed beside. Only the *total* is comparable across architectures; the
individual panels are comparable within a family, not across one. The same caveat already applies
to Mask R-CNN (whose `rpn_*` terms have no YOLO counterpart), but it bites harder here.

Ten `*loss*` columns and eight `*metric*` columns is what makes `plot_results` lay out the same
**2 × 9** grid, with the eight metric panels in identical positions. The `metrics/*` names are also
the ones `collect_results.py` reads, so it works on these runs unchanged.

Validation runs in smaller batches than training (4 at the default imgsz 640, printed at startup).
A mask head pastes all `max_det=300` detections back to full resolution, so one validated image
costs ~0.9 GB and the training batch of 16 would OOM. Metrics are unaffected — the evaluator
accumulates over the whole split either way — but `val_batch*.jpg` shows that many images per
file rather than Ultralytics' 16.

Curves, confusion matrices and `val_batch*` images are written at the end from `best.pt`, which is
what Ultralytics' `final_eval` does — so they describe the best checkpoint, and they still get
written if training stops early on `patience`.

Checkpoints hold the model, not the optimiser state. Keeping Adam's two moments would triple the
file (180 MB → 550 MB) and nothing here resumes from it; Ultralytics also strips the optimiser
from its final `best.pt`/`last.pt`.

---

## Tests

```bash
python -m pytest other_models/tests -q                  # everything, 218 tests, ~4 min
python -m pytest other_models/tests -q -m "not slow"    # skip the real training runs, ~45 s
```

The slow tests train a randomly-initialised Mask R-CNN — and, in `test_rfdetr.py`, a randomly
initialised RF-DETR — for a few epochs on a synthetic six-image dataset, on the CPU. No download,
no GPU: RF-DETR forces `load_dinov2_weights=False` for every `patch_size != 14` variant and builds
its encoder from a bundled config JSON, so `pretrain_weights=None` is enough to stay offline. Each
family runs the pretrain → fine-tune chain in miniature and checks that the second stage really
starts from the first's `best.pt`, with a negative control proving the assertion is not vacuous.
Tests that need the real datasets skip themselves if those are absent; `test_rfdetr.py` skips
itself entirely if `rfdetr` is not installed.

Three guard the things that have actually gone wrong here. `test_presets_mirror_the_yolo_yamls`
diffs every preset against its YAML key by key (the YAMLs are the reference; the presets only
mirror them) — and its per-preset allowance table is itself guarded, because a *global* allowance
for `imgsz`/`batch`/`patience` would let the RF-DETR deviations switch the check off for Mask R-CNN
too. `test_imports.py` fails if any file still refers to the package by its old name, and also
asserts in a subprocess that no module imports `rfdetr` at module scope — it costs ~10 s, and a
Mask R-CNN run should not pay for it.

The RF-DETR conversions are the one place in this package where a mistake produces a model that
trains, converges and simply scores worse, with nothing in the logs to say so. So each is pinned in
both directions: pixel normalisation against the exact ImageNet constants, the box format both
round-tripped and spelled out (a self-consistent wrong formula round-trips perfectly), the label
base at both boundaries, and the five loss columns asserted to sum to exactly the weighted total
the optimiser sees — with the criterion's unweighted diagnostics given absurd values (42, 99) so
that admitting even one of them cannot pass.

Worth knowing: a *perfect* detector scores **0.995**, not 1.0. Ultralytics' `compute_ap` appends a
sentinel point at recall 1.0 with precision 0, and one of its 101 interpolation points lands on
it. That number is asserted rather than worked around — matching it is the point.

---

## Licence

This code imports `ultralytics.utils.metrics` and `ultralytics.utils.plotting` so that the figures
and the mAP definition are produced by the same code as the YOLO runs. Ultralytics is AGPL-3.0, so
this places `other_models/` under the same obligation the repository already carries by training with
Ultralytics at all. torchvision itself is BSD-3-Clause, and `rfdetr` is Apache-2.0 — note that only
the variants in the base package are: the Atto/Femto/Pico/XLarge/2XLarge sizes live in
`rfdetr[plus]` under Roboflow's own PML 1.0 licence, which is why none of them is registered here.
