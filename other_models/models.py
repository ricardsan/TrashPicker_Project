"""Which network to train.

EDIT POINT 4 of 4.

Two model families live here, and `ModelSpec.family` says which one an entry belongs to.

**torchvision** (`family="torchvision"`, the default).
Mask R-CNN = backbone (ResNet-50 + FPN) -> region proposal network -> two heads per region:
    box head   "which class is this region, and where exactly is its box"
    mask head  "which pixels inside the box belong to the object"
The COCO-pretrained heads predict 91 classes, so build() replaces both with fresh ones sized for
our 4 + background. Everything else keeps its pretrained weights.

To add one, add a line to MODELS with:
    builder   the torchvision factory function
    weights   its pretrained weights enum, or None to train from scratch
    kind      "instance_seg" if it predicts masks, "detect" if boxes only
`head_swap` only has to be written if the model is not an R-CNN; the default handles every
torchvision *_rcnn model, including Faster R-CNN (whose mask_predictor is simply absent).

**RF-DETR** (`family="rfdetr"`).
A DINOv2 vision transformer + deformable-DETR decoder. It shares none of torchvision's
constructor arguments, which is why `build` dispatches on the family rather than passing
everything through `extra`: `trainable_backbone_layers`, `box_nms_thresh`, `min_size` and the
rest simply do not exist on it. An entry needs:
    family        "rfdetr"
    config_class  the *name* of the rfdetr.config.ModelConfig subclass, as a string
    loss_columns  its five results.csv loss names (RF-DETR's losses are not Mask R-CNN's)
    kind          "instance_seg" for a *Seg* variant, "detect" otherwise
and `builder`/`weights`/`head_swap` are unused -- other_models/rfdetr_adapter.py does the
building, the head resizing and the wrapping that makes it answer torchvision's interface.

`config_class` is a string rather than the class itself so that this module stays importable
without paying for rfdetr: `import rfdetr.config` costs ~10 s because it pulls in
transformers, and nothing that trains a Mask R-CNN -- or merely collects the test suite --
should wait for that. rfdetr_config_class() resolves it on first use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from other_models.artifacts import RFDETR_LOSS_COLUMNS

tvd = torchvision.models.detection

# The state_dict key whose first dimension is "number of classes, background included".
# Used only to turn a shape mismatch into a sentence a human can act on; see
# load_checkpoint.
CLASS_HEAD_KEYS = {
    "torchvision": "roi_heads.box_predictor.cls_score.weight",
    "rfdetr": "class_embed.weight",
}


def swap_rcnn_heads(model, num_classes: int) -> None:
    """Give an R-CNN fresh output heads for num_classes (background included)."""
    in_features = model.roi_heads.box_predictor.cls_score.in_features          # 1024
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if getattr(model.roi_heads, "mask_predictor", None) is not None:
        in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels    # 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, num_classes)


@dataclass(frozen=True, eq=False)
class ModelSpec:
    builder: Callable | None = None
    weights: object = None
    kind: str = "instance_seg"                       # "instance_seg" | "detect"
    family: str = "torchvision"                      # "torchvision" | "rfdetr"
    head_swap: Callable = field(default=swap_rcnn_heads)
    extra: dict = field(default_factory=dict)        # passed straight to the builder/config
    # rfdetr only: the *name* of the rfdetr.config.ModelConfig subclass (resolved lazily by
    # rfdetr_config_class), and the five loss names its criterion produces. loss_columns of
    # None means artifacts.LOSS_COLUMNS (torchvision's) applies.
    config_class: str | None = None
    loss_columns: dict | None = None


def rfdetr_config_class(spec: ModelSpec):
    """spec.config_class resolved to the real class, importing rfdetr on first use.

    Lazy on purpose -- see the note at the top of this module about the 10 s import.
    """
    import rfdetr.config

    try:
        return getattr(rfdetr.config, spec.config_class)
    except AttributeError as error:
        raise AttributeError(
            f"rfdetr.config has no {spec.config_class!r}. The variant classes are named "
            f"like 'RFDETRSegSmallConfig'; check the spelling against the installed rfdetr "
            f"version."
        ) from error


# =============================================================================
#  MODEL REGISTRY -- add an entry to train a different model
# =============================================================================
MODELS = {
    # The original 2017 Mask R-CNN recipe.
    "maskrcnn": ModelSpec(
        builder=tvd.maskrcnn_resnet50_fpn,
        weights=tvd.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        kind="instance_seg",
    ),
    # The 2021 refresh: FPN v2 blocks, a deeper box head, better-tuned training recipe.
    "maskrcnn_v2": ModelSpec(
        builder=tvd.maskrcnn_resnet50_fpn_v2,
        weights=tvd.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        kind="instance_seg",
    ),
    # Box-only example, to show what adding a non-mask model looks like. Runs fine; the engine
    # sees kind="detect" and writes NaN in the Mask columns instead of inventing them.
    "fasterrcnn_v2": ModelSpec(
        builder=tvd.fasterrcnn_resnet50_fpn_v2,
        weights=tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        kind="detect",
    ),
    # The transformer. DINOv2-small encoder (12 blocks) + 4 deformable-decoder layers + a
    # mask head, 33.7 M parameters against Mask R-CNN R50-FPN's ~44 M, which is why this
    # size was picked for the comparison. Its resolution must be a multiple of
    # patch_size * num_windows = 12 * 2 = 24, so it runs at the RFDETR_* presets' 648/840
    # rather than the YAMLs' 640/832 -- see config.py section 3.
    "rfdetr_seg_small": ModelSpec(
        family="rfdetr",
        kind="instance_seg",
        config_class="RFDETRSegSmallConfig",
        loss_columns=RFDETR_LOSS_COLUMNS,
    ),
}
# =============================================================================


def build(name: str, num_classes: int, hyp, weights_path: str | None = None):
    """A model ready to train on `num_classes` foreground classes.

    num_classes excludes background; the +1 happens once, inside the family's builder.
    `weights_path` loads a checkpoint from an earlier run (that is how the TACO pretrain feeds
    the fine-tune) and takes the place of the COCO weights.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    spec = MODELS[name]

    if spec.family == "rfdetr":
        # Imported here, not at module scope: see the note at the top about the 10 s cost.
        from other_models import rfdetr_adapter

        model, _config = rfdetr_adapter.build(spec, num_classes, hyp, weights_path)
    elif spec.family == "torchvision":
        model = _build_torchvision(spec, num_classes, hyp)
    else:
        raise KeyError(f"unknown model family {spec.family!r} on {name!r}")

    if weights_path:
        load_checkpoint(model, weights_path, num_classes, spec)
    return model, spec


def _build_torchvision(spec: ModelSpec, num_classes: int, hyp):
    """A torchvision detector, with every constructor argument hyp has an opinion about.

    None of these keyword names exists on RF-DETR, which is why build() dispatches on the
    family rather than routing everything through spec.extra.
    """
    model = spec.builder(
        # Always build from the COCO weights, even when a checkpoint is about to be loaded over
        # them. torchvision derives `is_trained = weights is not None or weights_backbone is not
        # None` and, when that is False, silently overrides trainable_backbone_layers to 5 and
        # (for Mask R-CNN v1) swaps FrozenBatchNorm2d for trainable BatchNorm2d. Passing
        # weights=None here would therefore change both the freezing and the normalisation of
        # every --weights run.
        weights=spec.weights,
        trainable_backbone_layers=hyp.trainable_backbone_layers,
        box_detections_per_img=hyp.max_det,
        box_nms_thresh=hyp.iou,
        # Without this torchvision drops everything below 0.05 inside roi_heads, so hyp.conf
        # (0.001, ultralytics' validation default) could never take effect and the low-confidence
        # tail of the PR curve would be missing -- which would depress mAP relative to the YOLO
        # runs for no reason other than a default.
        box_score_thresh=hyp.conf,
        # Every torchvision detector wraps itself in a GeneralizedRCNNTransform that rescales
        # inputs to min_size=800, max_size=1333 by default. Our images are already letterboxed
        # to a square imgsz, so pinning both ends makes that transform a no-op and `imgsz: 640`
        # mean the same 640 pixels it means in the YOLO configs.
        min_size=hyp.imgsz,
        max_size=hyp.imgsz,
        **spec.extra,
    )
    spec.head_swap(model, num_classes + 1)  # +1 for background, which torchvision reserves as 0
    return model


def load_checkpoint(model, weights_path: str, num_classes: int,
                    spec: ModelSpec | None = None) -> None:
    """Load an earlier run's best.pt over a freshly built model.

    This is the whole mechanism of the TACO pretrain -> fine-tune chain, so it reports what it did
    rather than succeeding silently: a pretrain that quietly failed to transfer would look exactly
    like one that worked, only with worse numbers.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    family = spec.family if spec is not None else "torchvision"
    if family == "rfdetr":
        # A DETR's position embeddings are sized by imgsz, and the TACO chain changes imgsz
        # between its two stages (840 -> 648), so they have to be resampled onto this
        # model's grid before the shapes will match. See align_checkpoint_resolution.
        from other_models import rfdetr_adapter

        resized = rfdetr_adapter.align_checkpoint_resolution(state, model)
        if resized:
            print(f"interpolated position embeddings from a {resized[0]}x{resized[0]} "
                  f"patch grid to {resized[1]}x{resized[1]} (imgsz differs between stages)")

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        # strict=False forgives absent and surplus keys but NOT shapes, and the shape that
        # realistically differs is the output head: a checkpoint trained on a different number of
        # classes. torch's own message lists every mismatched tensor without saying why.
        #
        # The key to look at differs by family -- both heads carry one row per class with
        # background included, they are just called different things. The adapter's
        # state_dict prefixes RF-DETR's with "net.", since the adapter is the nn.Module.
        head_key = CLASS_HEAD_KEYS[family]
        theirs = state.get(head_key, state.get(f"net.{head_key}"))
        # The head shape must actually disagree before blaming the class count. Checking
        # only for "size mismatch" in the message made every shape error -- notably a
        # resolution-dependent position embedding -- report itself as "trained on 4 classes,
        # but this run has 4", which is both wrong and unactionable.
        if (theirs is None or "size mismatch" not in str(error)
                or theirs.shape[0] - 1 == num_classes):
            raise
        raise RuntimeError(
            f"{weights_path} was trained on {theirs.shape[0] - 1} classes, but this run has "
            f"{num_classes}. Both stages of a pretrain -> fine-tune chain must use the same "
            f"class list (datasets.CLASS_NAMES); YOLO's cls_remap, which matches head rows up by "
            f"class name, has no equivalent here.\n\n{error}"
        ) from error

    loaded = len(state) - len(unexpected)
    print(f"loaded {loaded}/{len(model.state_dict())} tensors from {weights_path}"
          + (f" ({len(missing)} missing, {len(unexpected)} unexpected)"
             if missing or unexpected else ""))


def predicts_masks(spec: ModelSpec) -> bool:
    return spec.kind == "instance_seg"
