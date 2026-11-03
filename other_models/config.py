"""Hyperparameters for Mask R-CNN training, named exactly like the YOLO ones.

EDIT POINT 1 of 4. The other three are datasets.py (which data), augment.py (how it is
augmented) and models.py (which network).

Every field below carries the *same name* as the corresponding key in the Ultralytics YAMLs in
yolo_train_configs/, so you can put the two files side by side and see that the recipe matches.
Three presets mirror the three YAMLs we compare against:

    FINETUNE_NMS      <- yolo_train_configs/26-original-finetune-nms.yaml
    FINETUNE_AUGMENT  <- yolo_train_configs/26-original-finetune-augment.yaml
    TACO_PRETRAIN     <- yolo_train_configs/taco-pretrain/26-taco-pretrain.yaml

To change a hyperparameter: edit the preset. To add a preset: copy one, rename it, and add it to
PRESETS at the bottom. `train.py --preset NAME` picks one, and any field can be overridden on the
command line (`--epochs 3`, `--lr0 0.0005`, ...).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


# =============================================================================
# 1. THE HYPERPARAMETERS
# =============================================================================
@dataclass
class Hyp:
    """One training recipe. Defaults are Ultralytics' defaults where one exists."""

    # --- what to train on / for how long -------------------------------------
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    workers: int = 8
    patience: int = 100          # early stop after N epochs with no fitness improvement
    seed: int = 0
    deterministic: bool = True
    amp: bool = True             # mixed precision
    cache: bool = True           # keep decoded annotations in RAM

    # --- optimiser -----------------------------------------------------------
    optimizer: str = "Adam"      # Adam | AdamW | SGD
    lr0: float = 0.001           # initial learning rate
    lrf: float = 0.01            # final LR fraction: final LR = lr0 * lrf
    momentum: float = 0.937      # SGD momentum / Adam beta1
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False         # cosine schedule instead of linear
    nbs: int = 64                # nominal batch size; gradients accumulate to reach it

    # --- how much of the backbone to train -----------------------------------
    # YOLO's `freeze: N` freezes model.0 .. model.{N-1}. yolo26 has 24 indexed layers, of which
    # layers 0-10 are the backbone and 11-23 the head (counted from ultralytics' yolo26.yaml).
    # The two architectures here take different knobs:
    #
    #   Mask R-CNN  `trainable_backbone_layers`: how many ResNet stages to train, counted
    #               from the deepest, 0-5.
    #   RF-DETR     `trainable_encoder_blocks`: how many DINOv2 transformer blocks to train,
    #               counted from the last, 0-12. (RF-DETR's own knob is the all-or-nothing
    #               `freeze_encoder`, which cannot express the table below, so
    #               rfdetr_adapter.freeze_encoder_blocks does it block by block.)
    #
    # Three different knobs, so the correspondence is a judgement call and is written out
    # rather than hidden in a formula. The second column is the fraction of the 11 *backbone*
    # layers that `freeze: N` actually freezes; the other two columns apply that same
    # fraction to each architecture's own depth:
    #
    #     YOLO freeze | backbone frozen | trainable_backbone_layers | trainable_encoder_blocks
    #                 |                 | (ResNet stages, of 5)     | (DINOv2 blocks, of 12)
    #     ------------+-----------------+---------------------------+-------------------------
    #          0      |   0/11    0%    | 5  (everything trains)    | 12 (everything trains)
    #          3      |   3/11   27%    | 4                         |  9
    #          5      |   5/11   45%    | 3                         |  7  <- the TACO pretrain
    #          8      |   8/11   73%    | 2                         |  3
    #         10      |  10/11   91%    | 1  <- the two fine-tunes  |  1  <- the two fine-tunes
    #         11      |  11/11  100%    | 0  (only FPN + heads)     |  0  (only decoder + heads)
    #
    # Set them together: nothing derives one from another, and `freeze` itself is only a
    # record of the YOLO setting being mirrored -- the other two are what act. Only the one
    # belonging to the model being trained has any effect, so the inactive one is simply
    # carried into args.yaml.
    freeze: int = 10
    trainable_backbone_layers: int = 1
    trainable_encoder_blocks: int = 1

    # --- augmentation (consumed by augment.py) -------------------------------
    hsv_h: float = 0.015         # hue       shift, fraction
    hsv_s: float = 0.7           # saturation gain, fraction
    hsv_v: float = 0.4           # value/brightness gain, fraction
    degrees: float = 0.0         # rotation, +/- deg
    translate: float = 0.1       # translation, +/- fraction of imgsz
    scale: float = 0.5           # scale gain, +/-
    shear: float = 0.0           # shear, +/- deg
    perspective: float = 0.0     # perspective, 0-0.001 is the useful range
    flipud: float = 0.0          # probability of a vertical flip
    fliplr: float = 0.5          # probability of a horizontal flip
    mosaic: float = 1.0          # probability of stitching 4 images into one
    mixup: float = 0.0           # probability of blending two images
    cutmix: float = 0.0          # probability of pasting a patch of one image into another
    copy_paste: float = 0.0      # fraction of eligible objects duplicated elsewhere in the image
    copy_paste_mode: str = "flip"  # only "flip" is implemented -- see augment.copy_paste
    erasing: float = 0.4         # recorded to match the YAML, but NOT applied -- see section 3
    close_mosaic: int = 10       # turn mosaic/copy_paste/mixup/cutmix off for the last N epochs

    # --- validation ----------------------------------------------------------
    conf: float = 0.001          # keep detections above this score when scoring
    iou: float = 0.7             # NMS IoU threshold inside the model
    max_det: int = 300           # cap on detections per image
    plots: bool = True

    # --- where the run goes --------------------------------------------------
    project: str = "runs/segment/maskrcnn"
    name: str = "train"
    exist_ok: bool = False       # False -> append -2, -3, ... instead of overwriting
    save_period: int = -1        # also save epoch<N>.pt every N epochs; <1 disables

    def replace(self, **kwargs) -> "Hyp":
        """Return a copy with some fields changed. Unknown field names raise."""
        known = {f.name for f in fields(self)}
        unknown = set(kwargs) - known
        if unknown:
            raise KeyError(f"unknown hyperparameter(s): {sorted(unknown)}")
        return Hyp(**{**asdict(self), **kwargs})


# =============================================================================
# 2. THE PRESETS
# =============================================================================
# 26-original-finetune-nms.yaml: Ultralytics' default augmentation, 70 epochs, mosaic on.
FINETUNE_NMS = Hyp(
    epochs=70,
    batch=16,
    imgsz=640,
    optimizer="Adam",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.001,
    momentum=0.937,
    warmup_epochs=3.0,
    freeze=10,
    trainable_backbone_layers=1,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    erasing=0.4,
    close_mosaic=10,
    name="finetune-nms_v",
)

# 26-original-finetune-augment.yaml: much more aggressive, 150 epochs, mosaic off but
# mixup/cutmix on, full 90 deg rotation and vertical flips.
FINETUNE_AUGMENT = FINETUNE_NMS.replace(
    epochs=150,
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.7,
    degrees=90.0, translate=0.2, scale=0.5, shear=2.5, perspective=0.0001,
    flipud=0.5, fliplr=0.5,
    mosaic=0.0, mixup=0.5, cutmix=0.3,
    name="finetune-augment_v",
)

# taco-pretrain/26-taco-pretrain.yaml: pretraining on TACO. Bigger images, SGD, less freezing,
# and its own augmentation recipe -- it is not FINETUNE_NMS with a few knobs moved, so every
# field it sets differently is spelled out here.
TACO_PRETRAIN = FINETUNE_NMS.replace(
    epochs=40,
    imgsz=832,
    patience=15,
    optimizer="SGD",
    cache=False,
    # freeze: 5 freezes 5 of yolo26's 11 backbone layers, so rather more of the backbone trains
    # here than in the fine-tunes -- which is the point of a pretrain. See the table above.
    freeze=5,
    trainable_backbone_layers=3,
    lr0=0.005,             # an SGD-scale learning rate, unlike the Adam 0.001 of the fine-tunes
    lrf=0.0001,            # final LR = lr0 * lrf = 5e-7, i.e. decayed to nothing by the last epoch
    momentum=0.937,
    weight_decay=0.0008,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.01,
    hsv_h=0.02, hsv_s=0.3, hsv_v=0.5,
    # Mild scaling (0.15, against 0.5 in both fine-tunes) but stronger rotation and shear: this
    # recipe perturbs orientation far more than size.
    degrees=20.0, translate=0.1, scale=0.15, shear=10.0, perspective=0.0001,
    mosaic=0.5, mixup=0.0, cutmix=0.25, copy_paste=0.25,
    project="runs/segment/maskrcnn-taco",
    name="taco-pretrain_v",
)

# -----------------------------------------------------------------------------
#  The same three recipes for RF-DETR
# -----------------------------------------------------------------------------
# Derived from the three above rather than rewritten, so that every hyperparameter the
# YAMLs specify -- epochs, optimizer, lr0, lrf, momentum, weight_decay, all three warmup_*,
# nbs, close_mosaic and the whole augmentation recipe -- is inherited and still diffed
# against them by test_presets_mirror_the_yolo_yamls. Only what *has* to differ differs, and
# each one is spelled out with its reason. Section 3 below records the same list as the
# honest account of what did not carry over.
#
#   imgsz   RF-DETR requires resolution % (patch_size * num_windows) == 0, which is 24 for
#           the segmentation variants. 640 % 24 == 16 and 832 % 24 == 16, so the YAML values
#           are not merely suboptimal, they are invalid. 648 and 840 are the nearest valid
#           values (+1.25% and +0.96%).
#   batch   16 does not fit in 16 GB here. At imgsz 648 the DINOv2 patch grid is 54x54 =
#           2916 tokens against the published checkpoint's native 384/12 = 32x32 = 1024, so
#           2.85x the activations (840 gives 70x70 = 4900, 4.8x). `nbs` is left at 64, so
#           gradient accumulation holds the effective batch at 64 exactly as the YAMLs
#           intend -- only the physical batch shrinks.
#
#           Measured on the 16 GB RTX 3080 laptop this was developed on, one epoch of
#           dataset_seg at imgsz 648 (611 train / 152 val images):
#
#               batch | peak VRAM | s/epoch | mAP50-95(M) after 1 epoch
#               ------+-----------+---------+--------------------------
#                 4   |  7.2 GB   |  84.7   |  0.381
#                 8   | 14.6 GB   |  79.6   |  0.212
#
#           4 is the default because 8 buys 6% speed for 89% of the card -- 1.8 GB of
#           headroom on a GPU that also drives the display, over a 3.5-hour run -- and at
#           a fixed nbs of 64 it also halves the number of optimiser steps per epoch,
#           which is what the third column shows. Raise it with --batch if the card is
#           dedicated. For reference, Mask R-CNN takes 84 s/epoch at imgsz 640 batch 16,
#           so this costs essentially nothing in wall clock.
#   patience  the two fine-tunes only; see RFDETR_FINETUNE_NMS.
#   warmup_bias_lr  the two fine-tunes only. The YAMLs' 0.1 trains this architecture to
#           mAP 0.002; 0.001 reaches 0.881. The one deviation here that is a judgement
#           rather than an architectural requirement -- see section 4.
RFDETR_FINETUNE_NMS = FINETUNE_NMS.replace(
    imgsz=648,
    batch=4,
    trainable_encoder_blocks=1,     # freeze 10 of 11 -> 1 of 12 blocks. See the table above.
    # The one stopping-rule deviation in this package. Both fine-tune YAMLs pair
    # `patience: 100` with `epochs: 150`, i.e. effectively no early stop, and the Mask R-CNN
    # presets mirror that. At batch 4 with accumulate 16 this is the longest job in the
    # comparison, and the one most able to spend hours past its best checkpoint, so it stops
    # after 10 epochs without improvement. Nothing is lost by it: engine.final_eval runs
    # after the loop and writes every plot from best.pt, so an early-stopped run still
    # produces the full 24-file directory describing its best weights -- only results.csv is
    # shorter. Note this is a *stopping* deviation, not a hyperparameter one: every epoch
    # that does run uses the YAML recipe exactly.
    patience=10,
    # The YAMLs' 0.1 destroys the pretrained transformer -- a 0.1 learning rate on every
    # bias wipes out the COCO weights inside the first epoch. Measured on dataset_seg over
    # three epochs of this recipe, mAP50(B) / mAP50-95(M): 0.002 / 0.002 at 0.1, against
    # 0.881 / 0.743 at 0.001. This is the largest deviation from the YOLO recipe in the
    # package and section 4 spells out the numbers; it is here rather than left to a flag
    # because a 150-epoch run is too long to lose to a forgotten override.
    warmup_bias_lr=0.001,
    project="runs/segment/rfdetr",
    name="finetune-nms_v",
    nbs=16,
)

RFDETR_FINETUNE_AUGMENT = FINETUNE_AUGMENT.replace(
    imgsz=648,
    batch=4,
    trainable_encoder_blocks=1,
    patience=10,
    warmup_bias_lr=0.001,           # see RFDETR_FINETUNE_NMS and section 4
    project="runs/segment/rfdetr",
    name="finetune-augment_v",
    nbs=16,
)

RFDETR_TACO_PRETRAIN = TACO_PRETRAIN.replace(
    imgsz=840,
    batch=2,
    trainable_encoder_blocks=7,     # freeze 5 of 11 -> 7 of 12 blocks: the backbone moves
    # patience is left at TACO_PRETRAIN's 15, which is what 26-taco-pretrain.yaml sets. The
    # early stop above is for the 150-epoch fine-tunes; this stage already has a short fuse
    # of its own and shortening it further would not be mirroring anything.
    project="runs/segment/rfdetr-taco",
    name="taco-pretrain_v",
    nbs=16,
)

PRESETS = {
    "FINETUNE_NMS": FINETUNE_NMS,
    "FINETUNE_AUGMENT": FINETUNE_AUGMENT,
    "TACO_PRETRAIN": TACO_PRETRAIN,
    "RFDETR_FINETUNE_NMS": RFDETR_FINETUNE_NMS,
    "RFDETR_FINETUNE_AUGMENT": RFDETR_FINETUNE_AUGMENT,
    "RFDETR_TACO_PRETRAIN": RFDETR_TACO_PRETRAIN,
}


# =============================================================================
# 3. YOLO KEYS WITH NO MASK R-CNN EQUIVALENT, AND WHY
# =============================================================================
# These appear in the YAMLs but are deliberately absent above. Listed so that a paper
# comparing the two can state honestly what did and did not carry over.
#
#   box, cls, dfl, sem, dis   Loss-term weights for YOLO's own loss. Mask R-CNN's five losses
#                             (RPN objectness, RPN box, classifier, box, mask) are summed with
#                             weight 1.0 each, which is torchvision's design; re-weighting them
#                             would be a different experiment, not the same recipe.
#   nms                       YOLO26 can train an NMS-free head. Mask R-CNN always runs NMS in
#                             roi_heads, so there is nothing to switch. `iou` above is its
#                             NMS threshold.
#   overlap_mask              YOLO can rasterise all of an image's instance masks into one map.
#                             Mask R-CNN predicts a separate 28x28 mask per RoI, so there is
#                             nothing to overlap.
#   mask_ratio                Not a training knob here, but its value (4) is used: metrics.py
#                             scores masks at imgsz/4, which is the resolution ultralytics'
#                             segment validator compares at. See metrics.MASK_RATIO.
#   bgr                       Channel-swap augmentation. torchvision's transform expects RGB
#                             and normalises internally; swapping channels would fight the
#                             pretrained weights.
#   copy_paste_mode           Only "flip" is implemented, which is what all our configs use.
#                             augment.copy_paste raises on "mixup" rather than silently
#                             substituting the other mode. (copy_paste itself IS applied.)
#   erasing                   Ultralytics applies random erasing to *classification* training
#                             only, so the YOLO segment runs we compare against never used it
#                             despite `erasing: 0.4` sitting in their YAML. We keep the field so
#                             args.yaml looks like theirs, and likewise do not apply it.
#   auto_augment, dropout     Classification-only keys.
#   multi_scale, rect         YOLO batching strategies. Mask R-CNN accepts variable sizes
#                             natively; we letterbox to a fixed imgsz to match YOLO's geometry.
#   cls_remap, pretrained     Handled in models.py: COCO-pretrained weights are loaded and both
#                             output heads are replaced for our 4 classes.
#
#
# =============================================================================
# 4. WHERE THE RF-DETR PRESETS DEPART FROM THE YAMLS, AND WHY
# =============================================================================
# Everything in section 3 applies to RF-DETR too (its loss terms are its own, it is
# NMS-free, and so on). These four are additional, and unlike section 3 they are keys the
# RFDETR_* presets set to a *different value* rather than keys that do not apply -- so they
# are the deviations a paper has to declare.
#
#   imgsz         648 instead of 640, and 840 instead of 832. Not a preference: RF-DETR's
#                 resolution must satisfy resolution % (patch_size * num_windows) == 0,
#                 which is 24 for the segmentation variants, and 640 % 24 == 832 % 24 == 16.
#                 648 and 840 are the nearest valid values, +1.25% and +0.96%. The position
#                 embeddings are bicubic-interpolated onto the resulting patch grid by
#                 rfdetr.models.weights.interpolate_position_embeddings, so this is a
#                 geometry change, not a correctness problem.
#   batch         4 instead of 16 for the fine-tunes, 2 for the pretrain, because 16 does
#                 not fit in 16 GB at these resolutions. `nbs` stays at 64, so the engine's
#                 gradient accumulation holds the effective batch at exactly the 64 the
#                 YAMLs intend; what changes is the number of images in flight, not the
#                 optimiser step.
#   patience      10 instead of 100 on the two fine-tunes (the pretrain keeps its YAML's
#                 15). A stopping rule, not a hyperparameter: every epoch that runs uses the
#                 YAML recipe unchanged, and engine.final_eval still writes all 24 files
#                 from best.pt afterwards. It is here because this is the longest job in the
#                 comparison. A reader comparing epoch counts across the table will see the
#                 RF-DETR fine-tune ran fewer than 150 epochs, which is why it is declared.
#   iou           Recorded but meaningless. yolo26 can train an NMS-free head and RF-DETR
#                 always is one -- a DETR's set prediction has no duplicate-suppression step
#                 for a threshold to apply to. What bounds its detection count instead is
#                 the query count: RFDETRSegSmall has 100 queries against ultralytics'
#                 max_det of 300, so max_det is clamped and the effective value is written
#                 to args.yaml as `max_det_effective`.
#
#   warmup_bias_lr  0.001 instead of the YAML's 0.1 on the two fine-tunes. **The largest
#                 deviation in this package, and the one to quote first if anyone asks what
#                 did not carry over.** The YAML value destroys the pretrained transformer.
#                 Three RFDETR_FINETUNE_AUGMENT runs on dataset_seg, mAP50(B) / mAP50-95(M):
#
#                     1 epoch, warmup inactive (engine clamps it to epochs - 1)  0.473 / 0.381
#                     3 epochs, warmup_bias_lr 0.1, as the YAML specifies        0.002 / 0.002
#                     3 epochs, warmup_bias_lr 0.001                            0.881 / 0.743
#
#                 A 0.1 learning rate on every bias wipes out the COCO weights before
#                 training properly begins -- YOLO's loss tolerates it, a DETR's does not
#                 (rfdetr's own defaults are lr 1e-4 and lr_encoder 1.5e-4, an order of
#                 magnitude under the YAML's lr0 0.001, with no bias spike at all). At 0.001
#                 three epochs reach 0.748 + 0.743 = 1.49 fitness, against the 1.766 the
#                 YOLO run being compared with took 150 epochs to reach.
#
#                 Unlike the four above, this one is not forced by the architecture -- it is
#                 a judgement that a recipe which trains to mAP 0.002 is not a meaningful
#                 comparison to report. `--warmup_bias_lr 0.1` restores the YAML value if
#                 you want to show that result. RFDETR_TACO_PRETRAIN is left alone: it
#                 inherits 26-taco-pretrain.yaml's own 0.01, which is already 10x lower, and
#                 nothing has been measured at that value.
#
# Deliberately *not* done, though it is the DETR convention: a separate, lower learning rate
# for the backbone (rfdetr's own TrainConfig defaults to lr_encoder 1.5e-4 alongside lr 1e-4,
# plus lr_vit_layer_decay 0.8 and lr_component_decay 0.7). Ultralytics' three parameter
# groups -- decayed weights, norm weights, biases -- are the recipe being mirrored here, and
# adding a fourth group would both depart from it and overflow results.csv's three lr/pg*
# columns. The transformer is therefore trained on the YAML's single lr0, and how much of it
# trains at all is controlled by trainable_encoder_blocks instead.
