"""RF-DETR behind torchvision's detection interface.

This is the only file that knows RF-DETR exists. Everything else in the package -- the
dataset, the augmentation pipeline, the metrics, the run directory, the training loop --
already speaks exactly one contract, torchvision's:

    model.train();  model(images, targets)  ->  dict[str, Tensor] of named losses
    model.eval();   model(images)           ->  list of per-image dicts with
                                                boxes (xyxy pixels), scores,
                                                labels (1-based, 0 = background), masks

RF-DETR is organised differently, and in a way that is actually easier to wrap: its
`LWDETR.forward(samples)` returns raw tensors and deliberately does *not* compute the loss
(its own `targets` argument is documented as "Unused; kept for call compatibility"), the
loss lives in `SetCriterion`, and decoding lives in `PostProcess`. So one nn.Module can
hold all three and present the contract above, which leaves datasets.py, augment.py,
metrics.py and engine.py untouched.

The alternative -- teaching datasets.py to emit normalised cxcywh boxes, 0-based labels and
ImageNet-normalised pixels -- would fork the dataset contract that Mask R-CNN depends on
and invalidate test_datasets.py. The conversions belong here, at the one boundary where the
two conventions actually meet.

Three conventions differ, and all three are converted in this file:

                     | the harness (torchvision)      | RF-DETR
    -----------------+--------------------------------+----------------------------------
    pixels           | RGB 0..1, model normalises     | ImageNet mean/std, caller normalises
    boxes            | xyxy, absolute pixels          | cxcywh, normalised to 0..1
    labels           | 1..N, 0 reserved for background| 0..N-1 == the COCO category_id,
                     |                                | column N is the spare/background
"""
from __future__ import annotations

import math

import torch
from torch import nn

from rfdetr.models import (
    PostProcess,
    build_criterion_from_config,
    build_model_from_config,
    load_pretrain_weights,
)
from rfdetr.models.weights import interpolate_position_embeddings

from other_models.artifacts import RFDETR_LOSS_COLUMNS

# The state_dict key suffix of DINOv2's positional embedding table, which is the one
# parameter whose shape depends on imgsz. rfdetr's own _PE_KEY_SUFFIX, repeated because it
# is private there.
PE_KEY_SUFFIX = "embeddings.position_embeddings"

# RF-DETR normalises with the ImageNet statistics -- see rfdetr/datasets/transforms.py,
# whose Normalize defaults are exactly these. Our dataset hands over RGB in 0..1
# (datasets.CocoInstanceDataset.__getitem__), so this is the whole of the difference.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# How many transformer blocks the DINOv2-small encoder every RF-DETR variant we register
# uses has. Asserted against the model at build time rather than trusted, because
# config.py's freeze table is written in terms of it.
DINOV2_SMALL_BLOCKS = 12

# The five results.csv loss columns are RFDETR_LOSS_COLUMNS, defined in artifacts.py beside
# Mask R-CNN's five (models.py needs to name it without paying for the rfdetr import).
# Keeping it to five is not cosmetic: ten `*loss*` columns and eight `*metric*` columns are
# what make ultralytics' plot_results lay out the same 2x9 grid as the YOLO runs.
#
# `aux_loss` is the reason there are five rather than four. A DETR computes its loss at
# every decoder layer and at the encoder output, so the criterion returns 25 weighted terms
# (5 base x 5 layers) of which the 5 unsuffixed ones are the final layer. The auxiliary
# terms are a large share of the gradient, and engine.train_epoch's `if k in totals` filter
# would drop them from results.csv while still backpropagating them -- a loss curve that
# does not add up to the loss being optimised. Summing them into their own column keeps the
# CSV honest: the five columns add up to exactly what the optimiser sees.

# Which criterion keys fold into which of the five columns. `loss_mask` is the sum of the
# two mask terms because RF-DETR splits its mask loss into a point-sampled BCE and a Dice
# term, and Mask R-CNN's single `loss_mask` is what the column is being compared against.
_FINAL_LAYER_TERMS = {
    "loss_bbox": ("loss_bbox",),
    "loss_mask": ("loss_mask_ce", "loss_mask_dice"),
    "loss_ce": ("loss_ce",),
    "loss_giou": ("loss_giou",),
}


def aggregate_losses(raw: dict, weight_dict: dict) -> dict[str, torch.Tensor]:
    """The criterion's 31 outputs as the five columns results.csv has, already weighted.

    Exhaustive by construction, which is what makes `sum(result.values())` the true
    training objective -- engine.train_epoch backpropagates exactly that sum. Every key the
    criterion returns either lands in one of the five columns or is not a loss at all:

      * the 25 keys in `weight_dict` are 5 terms x 5 layers (the final layer unsuffixed,
        then `_0`, `_1`, ... per auxiliary decoder layer, then `_enc`). The unsuffixed five
        go to their own columns; every suffixed one is summed into `loss_aux`.
      * `class_error` and `cardinality_error*` are diagnostics, carry no weight, and are
        absent from `weight_dict` -- so keying off `weight_dict` drops them for free rather
        than needing a deny-list that could go stale.

    A term the criterion did not produce at all is *omitted* rather than returned as zero.
    That matters for a hypothetical box-only variant, which has no mask terms: the engine
    and artifacts turn an absent key into NaN in results.csv, which is what `fasterrcnn_v2`
    already does and what metrics.py argues for -- a flat zero line would read as "it tried
    and scored nothing", a different and misleading claim. The column *count* is unaffected
    either way, since csv_columns is built from RFDETR_LOSS_COLUMNS rather than from
    whatever this returns.
    """
    direct = {key: column for column, keys in _FINAL_LAYER_TERMS.items() for key in keys}
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        weight = weight_dict.get(key)
        if weight is None:
            continue  # class_error / cardinality_error: reported by the criterion, not optimised
        column = direct.get(key, "loss_aux")
        out[column] = out[column] + value * weight if column in out else value * weight
    # Insertion order follows the criterion's; return them in the CSV's order instead, so
    # that printing the dict reads like the row it becomes.
    return {name: out[name] for name in RFDETR_LOSS_COLUMNS if name in out}


def align_checkpoint_resolution(state: dict, model) -> tuple[int, int] | None:
    """Interpolate a checkpoint's position embeddings onto this model's patch grid.

    This is what makes the TACO chain possible at all. DINOv2's `position_embeddings` is the
    one parameter whose shape depends on `imgsz` -- it is
    `(1, (imgsz / patch_size)^2 + 1, dim)` -- and the two stages deliberately run at
    different resolutions: the pretrain at 840 (a 70 x 70 grid, 4901 rows), the fine-tune at
    648 (54 x 54, 2917). `load_state_dict(strict=False)` forgives missing and surplus keys
    but *not* mismatched shapes, so without this the second stage cannot start from the
    first's best.pt.

    rfdetr does the same bicubic interpolation inside `load_pretrain_weights` when loading
    its own published COCO checkpoint at a non-native resolution. That path is skipped when
    `--weights` is given (the checkpoint replaces the COCO weights entirely), so the
    interpolation has to happen here instead.

    Returns (source grid, target grid) when it interpolated, None when there was nothing to
    do, so the caller can report it -- a silently skipped transfer is the failure mode the
    whole pretrain -> fine-tune chain is most exposed to.
    """
    target = next((int(math.isqrt(t.shape[1] - 1))
                   for k, t in model.state_dict().items() if k.endswith(PE_KEY_SUFFIX)), None)
    source = next((int(math.isqrt(t.shape[1] - 1))
                   for k, t in state.items() if k.endswith(PE_KEY_SUFFIX)), None)
    if target is None or source is None or source == target:
        return None

    interpolate_position_embeddings(state, target)
    return source, target


def freeze_encoder_blocks(net, trainable_blocks: int) -> int:
    """Train only the last `trainable_blocks` of the DINOv2 encoder. Returns how many.

    RF-DETR's own knob is all-or-nothing (`ModelConfig.freeze_encoder` sets requires_grad
    False on every encoder parameter) or LoRA, neither of which mirrors the YOLO `freeze`
    table that the rest of this package is built around. The encoder's blocks are a plain
    nn.ModuleList, though, so the stage-wise equivalent of Mask R-CNN's
    trainable_backbone_layers is a few lines -- see the table in config.py for which number
    corresponds to which YOLO `freeze`.

    The patch and position embeddings are frozen whenever anything is, which matches
    torchvision: resnet_fpn_backbone's trainable_layers never unfreezes the stem either.

    engine.build_optimizer skips parameters with requires_grad False, so calling this
    before the optimizer is built is all that is needed -- there is no engine change.
    """
    blocks = net._get_backbone_encoder_layers()
    if blocks is None:
        raise RuntimeError(
            "could not find the encoder's transformer blocks on this RF-DETR backbone, so "
            "`trainable_encoder_blocks` cannot be honoured. Rather than silently training "
            "the whole encoder (which would make the run a different experiment from the "
            "one args.yaml records), this is a hard error. See "
            "rfdetr.models.lwdetr.LWDETR._get_backbone_encoder_layers for the shapes it "
            "knows about."
        )
    if not 0 <= trainable_blocks <= len(blocks):
        raise ValueError(
            f"trainable_encoder_blocks={trainable_blocks} is out of range for an encoder "
            f"with {len(blocks)} blocks"
        )

    frozen_upto = len(blocks) - trainable_blocks
    for block in blocks[:frozen_upto]:
        for param in block.parameters():
            param.requires_grad = False

    if trainable_blocks < len(blocks):
        # The stem: cls_token, mask_token, position_embeddings and the patch projection, all
        # named `encoder.embeddings.*`. The transformer blocks are `encoder.layer.N.*`, so
        # matching on "embeddings" cannot catch them. `encoder.layernorm` is deliberately
        # left trainable -- it feeds the projector, and torchvision leaves the FPN trainable
        # for the same reason.
        frozen = 0
        for name, param in net.backbone[0].encoder.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
                frozen += 1
        if not frozen:
            raise RuntimeError(
                "found no `embeddings` parameters to freeze on this RF-DETR encoder. The "
                "names this expects are `encoder.embeddings.*` (cls_token, mask_token, "
                "position_embeddings, patch_embeddings.projection); if they have been "
                "renamed upstream, the stem is silently training and every "
                "`trainable_encoder_blocks` run is a different experiment from the one "
                "args.yaml records."
            )
    return len(blocks)


class RFDETRAdapter(nn.Module):
    """RF-DETR presenting torchvision's detection interface. See the module docstring."""

    def __init__(self, net, criterion, postprocess, num_classes: int, imgsz: int):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.postprocess = postprocess
        # The index of the spare output column. rfdetr's build_model allocates
        # num_classes + 1 columns (the "max category_id + 1" convention: for COCO's ids
        # 1..90 you pass 90, and column 0 is never a real class). Our category ids are
        # 0-based, 0..3, so our classes occupy columns 0..3 and column 4 is the spare --
        # which PostProcess's top-k will happily select, so predictions carrying it have to
        # be dropped. See _to_torchvision.
        self.background_index = num_classes
        # Output columns, i.e. our classes plus the spare. PostProcess's top-k scores every
        # (query, column) pair, so this is a factor in how many detections one image can
        # yield -- see engine.effective_max_det.
        self.class_columns = num_classes + 1
        self.imgsz = imgsz
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    # --- the torchvision contract -------------------------------------------
    def forward(self, images, targets=None):
        """Losses when training with targets, predictions otherwise.

        The two branches are exactly the two calls engine.py makes. Note that
        engine.val_losses also lands in the first branch: it puts the model in train() mode
        under no_grad precisely so that a model returns losses, which is what makes the
        val/* columns possible.
        """
        batch = torch.stack([image for image in images])
        normalised = (batch - self.pixel_mean) / self.pixel_std

        if self.training and targets is not None:
            outputs = self.net(normalised)
            converted = [self._to_detr_target(t) for t in targets]
            raw = self.criterion(outputs, converted)
            return aggregate_losses(raw, self.criterion.weight_dict)

        outputs = self.net(normalised)
        height, width = batch.shape[-2:]
        sizes = torch.full((len(batch), 2), 0, dtype=torch.long, device=batch.device)
        sizes[:, 0], sizes[:, 1] = height, width
        return [self._to_torchvision(p) for p in self.postprocess(outputs, sizes)]

    # --- conversions ---------------------------------------------------------
    def _to_detr_target(self, target: dict) -> dict:
        """One image's target in RF-DETR's conventions.

        Masks are passed at full input resolution and deliberately not resized:
        SetCriterion.loss_masks point-samples the ground truth at *normalised* coordinates
        (`_sample_target_masks_at_points`), so the target resolution is free, and handing
        over the dataset's own masks avoids a resample that could only lose detail.
        """
        return {
            "boxes": xyxy_to_cxcywh_normalised(target["boxes"], self.imgsz),
            # 1-based (0 = background) back to the 0-based category_id RF-DETR indexes by
            "labels": target["labels"] - 1,
            "masks": target["masks"].float(),
        }

    def _to_torchvision(self, prediction: dict) -> dict:
        """One image's predictions in torchvision's conventions.

        Boxes already come back as absolute pixels from PostProcess, so the only work is
        dropping the spare column's detections and shifting labels up by one.
        """
        keep = prediction["labels"] != self.background_index
        out = {
            "boxes": prediction["boxes"][keep],
            "scores": prediction["scores"][keep],
            "labels": prediction["labels"][keep] + 1,
        }
        masks = prediction.get("masks")
        if masks is not None:
            # engine._to_numpy indexes channel 0 and thresholds at 0.5, so keep the
            # [n, 1, h, w] rank torchvision's mask head produces. PostProcess returns bool,
            # which compares fine against 0.5 but is cast here so the two models' masks are
            # the same dtype wherever they meet.
            out["masks"] = masks[keep].float()
        return out


def xyxy_to_cxcywh_normalised(boxes: torch.Tensor, imgsz: int) -> torch.Tensor:
    """Absolute xyxy pixels -> centre-based cxcywh normalised to 0..1.

    Dividing by a scalar `imgsz` rather than by per-image width and height is correct here
    and only here: augment.letterbox pads every image to a square imgsz x imgsz before it
    reaches the model, so the two are the same number. datasets.py's `reshape(-1, 4)` keeps
    the rank right for an image with no instances, and so does this.
    """
    boxes = boxes.reshape(-1, 4)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([centres, sizes], dim=1) / imgsz


def cxcywh_normalised_to_xyxy(boxes: torch.Tensor, imgsz: int) -> torch.Tensor:
    """The inverse of xyxy_to_cxcywh_normalised. Used by the tests, not the training path."""
    boxes = boxes.reshape(-1, 4) * imgsz
    centres, sizes = boxes[:, :2], boxes[:, 2:]
    return torch.cat([centres - sizes / 2, centres + sizes / 2], dim=1)


# =============================================================================
#  BUILDING ONE
# =============================================================================
def model_config(spec, num_classes: int, hyp, pretrained: bool):
    """The RF-DETR ModelConfig for one of our runs.

    `num_classes` is RF-DETR's "max category_id + 1" (rfdetr/detr.py's
    _align_num_classes_from_dataset computes exactly that from an annotation file). Our
    category ids are 0-based 0..3, so len(CLASS_NAMES) == 4 is already the right number,
    and rfdetr's build_model adds its own +1 for the spare column -- the same single +1
    models.build performs for torchvision's background.

    `resolution` must be divisible by patch_size * num_windows, which is 24 for the
    segmentation variants. That is why the presets run at 648/840 rather than the YAMLs'
    640/832; it is checked here so a bad --imgsz fails immediately with the arithmetic
    rather than deep inside the backbone.
    """
    from other_models.models import rfdetr_config_class

    config_class = rfdetr_config_class(spec)
    divisor = (config_class.model_fields["patch_size"].default
               * config_class.model_fields["num_windows"].default)
    if hyp.imgsz % divisor:
        nearest = (hyp.imgsz // divisor) * divisor
        raise ValueError(
            f"imgsz={hyp.imgsz} is not valid for this RF-DETR variant: its resolution must "
            f"be divisible by patch_size * num_windows = {divisor}, and "
            f"{hyp.imgsz} % {divisor} = {hyp.imgsz % divisor}. The nearest valid values are "
            f"{nearest} and {nearest + divisor}."
        )

    kwargs = dict(
        resolution=hyp.imgsz,
        num_classes=num_classes,
        device="cpu",  # engine.Trainer moves the model itself
        # Passed explicitly even though it is the class default, because pydantic does not
        # run field validators on defaults -- and ModelConfig.expand_path is the validator
        # that turns a bare filename into a path under the model cache directory
        # (~/.roboflow/models, or $RF_HOME). Left to the default, the bare
        # "rf-detr-seg-small.pt" stays relative and rfdetr downloads 134 MB into the
        # *current working directory*: the repository root, once per directory you ever run
        # from. Re-stating the default is what makes the validator fire.
        pretrain_weights=config_class.model_fields["pretrain_weights"].default,
    )
    if not pretrained:
        # Explicitly no checkpoint: the backbone is then built from rfdetr's own bundled
        # config JSON with random weights and nothing is downloaded, which is what lets the
        # tests construct a real model offline.
        kwargs["pretrain_weights"] = None
    return config_class(**{**kwargs, **spec.extra})


def build(spec, num_classes: int, hyp, weights_path: str | None = None):
    """An RF-DETR ready to train, wrapped in the adapter. Called by models.build.

    Unlike torchvision, there is no reason to load the COCO checkpoint when a run's own
    weights are about to replace it: rfdetr derives nothing from "were weights supplied"
    the way torchvision derives `is_trained`, so the pretrain -> fine-tune chain skips the
    download entirely and models.load_checkpoint supplies the weights.
    """
    from rfdetr.config import SegmentationTrainConfig, TrainConfig

    pretrained = weights_path is None
    config = model_config(spec, num_classes, hyp, pretrained)

    # Only reached for the loss weights and the matcher costs; the dataset_dir/output_dir it
    # insists on are never used, because engine.py owns the data and the run directory.
    train_config_class = SegmentationTrainConfig if config.segmentation_head else TrainConfig
    train_config = train_config_class(dataset_dir=".", output_dir=".")

    # spec.extra reaches the ModelConfig, so it could override num_classes and leave
    # RFDETRAdapter.background_index pointing at one of our real classes -- which would
    # silently drop every prediction of it. Cheaper to refuse than to debug.
    if config.num_classes != num_classes:
        raise ValueError(
            f"spec.extra set num_classes={config.num_classes}, but this run has "
            f"{num_classes} classes (datasets.CLASS_NAMES). The adapter derives which output "
            f"column is the spare one from the class count, so the two must agree."
        )

    net = build_model_from_config(config, train_config)
    if pretrained:
        # Resizes the class head for our 4 classes and bicubic-interpolates the position
        # embeddings onto our resolution's patch grid (648/12 = 54 against the published
        # checkpoint's 384/12 = 32), which is what makes running above native resolution
        # work at all.
        load_pretrain_weights(net, config, trust=True)

    blocks = freeze_encoder_blocks(net, hyp.trainable_encoder_blocks)
    if blocks != DINOV2_SMALL_BLOCKS:
        raise RuntimeError(
            f"config.py's freeze table is written for a {DINOV2_SMALL_BLOCKS}-block "
            f"encoder, but this one has {blocks}. Update the table rather than letting "
            f"`trainable_encoder_blocks` mean a different fraction than it documents."
        )

    criterion, _ = build_criterion_from_config(config, train_config)
    adapter = RFDETRAdapter(
        net,
        criterion,
        PostProcess(
            # hyp.max_det verbatim, which is also rfdetr's own default for num_select.
            #
            # It is tempting to clamp this to config.num_queries (100 for RFDETRSegSmall,
            # against ultralytics' 300) on the grounds that a DETR cannot return more
            # detections than it has queries. That is false for this implementation:
            # PostProcess._select_topk flattens to (batch, num_queries * num_columns)
            # before taking the top k, so one query can contribute up to one detection per
            # class column and the real ceiling is num_queries * (num_classes + 1) = 500.
            # Measured: num_select=300 really does return 300 detections spread over all
            # 100 queries. Clamping to 100 would have cut the detection budget to a third
            # of what the YOLO and Mask R-CNN runs get, truncating the low-confidence tail
            # of the PR curve and depressing mAP for no reason other than a wrong
            # assumption -- the same mistake box_score_thresh exists to avoid on the
            # torchvision side.
            num_select=hyp.max_det,
            # Masks at head resolution, which is imgsz // mask_downsample_ratio. RF-DETR's
            # ratio is 4 and metrics.MASK_RATIO is 4, so these arrive at exactly the
            # resolution the evaluator scores at -- no upsample-then-downsample round trip,
            # and none of the per-image mask cost that forces engine.val_batch to shrink
            # Mask R-CNN's validation batch.
            upsample_masks_to_image_size=False,
        ),
        num_classes=num_classes,
        imgsz=hyp.imgsz,
    )
    return adapter, config
