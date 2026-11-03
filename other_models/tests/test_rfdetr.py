"""RF-DETR: the conversions, the five loss columns, the freezing, and one real training run.

The conversions are the whole risk surface of rfdetr_adapter.py. Three conventions differ
between the harness and RF-DETR (pixel normalisation, box format, label base), and a
mistake in any of them produces a model that trains without erroring and scores badly --
the failure mode that is most expensive to notice. So each one is pinned here, in both
directions, with the boundary it crosses named in the docstring.

Everything except the three `slow` tests avoids building a model, because
`import rfdetr.config` alone costs ~10 s.
"""
from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("rfdetr", reason="rfdetr is not installed; see requirements.txt")

from other_models.datasets import ROOT  # noqa: E402

from other_models.artifacts import RFDETR_LOSS_COLUMNS, csv_columns  # noqa: E402
from other_models.config import (  # noqa: E402
    FINETUNE_AUGMENT,
    FINETUNE_NMS,
    PRESETS,
    RFDETR_FINETUNE_AUGMENT,
    RFDETR_FINETUNE_NMS,
    RFDETR_TACO_PRETRAIN,
    TACO_PRETRAIN,
)
from other_models.metrics import MASK_RATIO  # noqa: E402
from other_models.rfdetr_adapter import (  # noqa: E402
    DINOV2_SMALL_BLOCKS,
    RFDETRAdapter,
    aggregate_losses,
    cxcywh_normalised_to_xyxy,
    freeze_encoder_blocks,
    xyxy_to_cxcywh_normalised,
)

def divisor_of(config_class_name: str) -> int:
    """patch_size * num_windows, read from rfdetr rather than hardcoded.

    24 for the segmentation variants and 12 for SegNano (which uses one window) at the time
    of writing -- but read from the installed package, so an upstream change surfaces as a
    failure in the tests that assert the arithmetic rather than leaving them asserting
    numbers that are no longer true.
    """
    import rfdetr.config

    fields = getattr(rfdetr.config, config_class_name).model_fields
    return fields["patch_size"].default * fields["num_windows"].default


SEG_DIVISOR = divisor_of("RFDETRSegSmallConfig")
SEG_NANO_DIVISOR = divisor_of("RFDETRSegNanoConfig")


# =============================================================================
# the resolution constraint -- why the presets do not use the YAML's imgsz
# =============================================================================
def test_the_seg_variants_cannot_run_at_the_yaml_resolutions():
    """RF-DETR's imgsz deviation is forced, not chosen.

    `resolution % (patch_size * num_windows) == 0` is a hard architectural requirement, and
    both YAML values fail it by the same 16 pixels. This is the executable record of that
    arithmetic: it fails if someone "corrects" the presets back to 640/832, which would
    otherwise only show up as a backbone error deep into a run.
    """
    assert SEG_DIVISOR == 24, f"the arithmetic below is written for 24, not {SEG_DIVISOR}"
    assert 640 % SEG_DIVISOR == 16, "640 is not a valid RF-DETR seg resolution"
    assert 832 % SEG_DIVISOR == 16, "832 is not a valid RF-DETR seg resolution"
    assert RFDETR_FINETUNE_NMS.imgsz % SEG_DIVISOR == 0
    assert RFDETR_FINETUNE_AUGMENT.imgsz % SEG_DIVISOR == 0
    assert RFDETR_TACO_PRETRAIN.imgsz % SEG_DIVISOR == 0
    # and they are the *nearest* valid values, not an arbitrary round number
    assert RFDETR_FINETUNE_NMS.imgsz == 648 and 648 - 640 < SEG_DIVISOR
    assert RFDETR_TACO_PRETRAIN.imgsz == 840 and 840 - 832 < SEG_DIVISOR


def test_the_conftest_image_size_is_valid_for_a_seg_nano():
    """The slow tests below depend on this, so it is stated rather than discovered.

    RFDETRSegNano uses num_windows=1, so its divisor is 12 rather than 24, and the existing
    96 px synthetic fixture happens to satisfy it. Without that the CPU smoke tests would
    need their own larger dataset.
    """
    from other_models.tests.conftest import IMAGE_SIZE

    assert IMAGE_SIZE % SEG_NANO_DIVISOR == 0, IMAGE_SIZE


def test_the_checkpoint_is_cached_outside_the_repository(monkeypatch):
    """A 134 MB download must not land in the working directory.

    rfdetr's ModelConfig.expand_path resolves a bare filename to the model cache dir, but
    pydantic does not run field validators on *defaults* -- so leaving pretrain_weights
    alone downloads "rf-detr-seg-small.pt" relative to CWD, i.e. into the repository root,
    once per directory anyone ever trains from. model_config re-states the default
    explicitly so the validator fires; this is what notices if that line is removed as
    redundant.
    """
    from other_models.models import MODELS, ModelSpec
    from other_models.rfdetr_adapter import model_config

    spec = ModelSpec(family="rfdetr", config_class="RFDETRSegSmallConfig")
    monkeypatch.setitem(MODELS, "probe", spec)
    config = model_config(spec, 4, FINETUNE_NMS.replace(imgsz=648), pretrained=True)

    weights = str(config.pretrain_weights)
    assert os.path.isabs(weights), f"relative path downloads into CWD: {weights}"
    assert str(ROOT) not in weights, f"the checkpoint would land in the repository: {weights}"


def test_an_invalid_imgsz_is_refused_with_the_arithmetic(monkeypatch):
    """Not a bare backbone shape error 40 layers deep."""
    from other_models.models import MODELS, ModelSpec
    from other_models.rfdetr_adapter import model_config

    spec = ModelSpec(family="rfdetr", config_class="RFDETRSegSmallConfig")
    monkeypatch.setitem(MODELS, "probe", spec)
    with pytest.raises(ValueError, match=r"divisible by patch_size \* num_windows = 24"):
        model_config(spec, 4, FINETUNE_NMS.replace(imgsz=640), pretrained=False)
    # the message has to name the way out, not just the problem
    with pytest.raises(ValueError, match="624 and 648"):
        model_config(spec, 4, FINETUNE_NMS.replace(imgsz=640), pretrained=False)


# =============================================================================
# the presets
# =============================================================================
# The expected *changed* key set per preset. Two things are set in the presets but do not
# appear here, because they are set to the value already inherited and so are not changes:
#   name                      the run keeps its parent's name; only `project` differs, which
#                             is what puts RF-DETR's runs in their own tree alongside the
#                             Mask R-CNN and YOLO ones rather than renaming the experiment.
#   trainable_encoder_blocks  Hyp's default is 1, the fine-tune value, exactly as
#                             trainable_backbone_layers' default is. The fine-tune presets
#                             still spell it out for readability; only the pretrain's 7 is a
#                             real change.
PARENTS = [
    ("RFDETR_FINETUNE_NMS", RFDETR_FINETUNE_NMS, FINETUNE_NMS,
     {"imgsz", "batch", "patience", "warmup_bias_lr", "project"}),
    ("RFDETR_FINETUNE_AUGMENT", RFDETR_FINETUNE_AUGMENT, FINETUNE_AUGMENT,
     {"imgsz", "batch", "patience", "warmup_bias_lr", "project"}),
    ("RFDETR_TACO_PRETRAIN", RFDETR_TACO_PRETRAIN, TACO_PRETRAIN,
     {"imgsz", "batch", "trainable_encoder_blocks", "project"}),
]


@pytest.mark.parametrize("name,preset,parent,allowed", PARENTS, ids=[p[0] for p in PARENTS])
def test_presets_change_only_what_they_are_allowed_to(name, preset, parent, allowed):
    """Every other hyperparameter is inherited, so the YAML diff still covers it.

    The RF-DETR presets are derived from the Mask R-CNN ones rather than rewritten,
    precisely so that `epochs`, `optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`, the
    three `warmup_*`, `nbs`, `close_mosaic` and the whole augmentation recipe keep being
    compared against the YAMLs by test_presets_mirror_the_yolo_yamls. This is what notices
    if one of them is edited here instead of in its YAML.
    """
    from dataclasses import asdict

    mine, theirs = asdict(preset), asdict(parent)
    changed = {k for k in mine if mine[k] != theirs[k]}
    assert changed == allowed, f"{name}: unexpected {changed - allowed}, missing {allowed - changed}"


def test_only_the_rfdetr_finetunes_stop_early():
    """The early stop is deliberately asymmetric; this is what stops it leaking.

    `patience: 10` applies to the two RF-DETR fine-tunes and nothing else. The negative half
    is the load-bearing half: if it ever reached the Mask R-CNN presets, their runs would
    stop early too and would no longer be the like-for-like comparison against the YOLO
    runs that the whole package exists to provide.
    """
    assert RFDETR_FINETUNE_NMS.patience == 10
    assert RFDETR_FINETUNE_AUGMENT.patience == 10
    # the pretrain keeps 26-taco-pretrain.yaml's own short fuse, not the fine-tunes'
    assert RFDETR_TACO_PRETRAIN.patience == TACO_PRETRAIN.patience == 15
    # and the models the paper compares against are untouched
    assert FINETUNE_NMS.patience == FINETUNE_AUGMENT.patience == 100


def test_the_fine_tunes_lower_the_bias_warmup_and_the_pretrain_does_not():
    """The YAMLs' warmup_bias_lr 0.1 trains RF-DETR to mAP 0.002; 0.001 reaches 0.881.

    Measured over three epochs of RFDETR_FINETUNE_AUGMENT on dataset_seg -- a 0.1 learning
    rate on every bias wipes out the COCO weights before training properly begins. This is
    the largest deviation from the YOLO recipe in the package, so it is asserted rather than
    left to a comment, and the negative halves matter just as much: the Mask R-CNN presets
    keep 0.1 because YOLO's own runs used it, and RFDETR_TACO_PRETRAIN keeps its YAML's
    0.01, which is already 10x lower and has not been measured at anything else.
    """
    assert RFDETR_FINETUNE_NMS.warmup_bias_lr == 0.001
    assert RFDETR_FINETUNE_AUGMENT.warmup_bias_lr == 0.001
    assert RFDETR_TACO_PRETRAIN.warmup_bias_lr == TACO_PRETRAIN.warmup_bias_lr == 0.01
    assert FINETUNE_NMS.warmup_bias_lr == FINETUNE_AUGMENT.warmup_bias_lr == 0.1


def test_the_freeze_table_is_applied_consistently():
    """config.py's table made executable, as its Mask R-CNN counterpart already is.

    Same YOLO `freeze`, so the same fraction of each backbone: 10 of 11 frozen gives 1 of 5
    ResNet stages and 1 of 12 DINOv2 blocks; 5 of 11 gives 3 and 7.
    """
    for preset in (RFDETR_FINETUNE_NMS, RFDETR_FINETUNE_AUGMENT):
        assert (preset.freeze, preset.trainable_encoder_blocks) == (10, 1)
    assert (RFDETR_TACO_PRETRAIN.freeze, RFDETR_TACO_PRETRAIN.trainable_encoder_blocks) == (5, 7)
    # the pretrain trains more of the backbone than the fine-tunes -- the point of a pretrain
    assert (RFDETR_TACO_PRETRAIN.trainable_encoder_blocks
            > RFDETR_FINETUNE_NMS.trainable_encoder_blocks)


def test_effective_batch_is_still_the_yaml_nominal_batch():
    """`batch` is lowered for VRAM, but `nbs` is not, so the optimiser step is unchanged.

    This is the difference between "a smaller batch" and "a different recipe": the engine
    accumulates nbs/batch micro-batches before stepping, so an effective 64 is preserved.
    """
    for preset, parent in [(RFDETR_FINETUNE_NMS, FINETUNE_NMS),
                           (RFDETR_TACO_PRETRAIN, TACO_PRETRAIN)]:
        assert preset.nbs == parent.nbs == 64
        assert preset.nbs % preset.batch == 0, "accumulation would not reach nbs exactly"
        assert round(preset.nbs / preset.batch) == preset.nbs // preset.batch


def test_the_rfdetr_presets_are_registered():
    for name in ("RFDETR_FINETUNE_NMS", "RFDETR_FINETUNE_AUGMENT", "RFDETR_TACO_PRETRAIN"):
        assert name in PRESETS, f"{name} missing from PRESETS, so --preset cannot select it"


def test_crossing_a_preset_with_the_wrong_family_is_refused():
    """`--model maskrcnn_v2 --preset RFDETR_FINETUNE_NMS` would otherwise train Mask R-CNN
    at 648/4 and file it under runs/segment/rfdetr/ -- no longer the like-for-like
    comparison the package exists for, and nothing in the run would say so.

    The other direction already fails on the divisibility arithmetic; this covers the quiet
    one.
    """
    from other_models.train import check_preset_suits_the_model

    with pytest.raises(SystemExit, match="is for the rfdetr family"):
        check_preset_suits_the_model("RFDETR_FINETUNE_NMS", "maskrcnn_v2")
    with pytest.raises(SystemExit, match="is for the torchvision family"):
        check_preset_suits_the_model("FINETUNE_NMS", "rfdetr_seg_small")

    # the matching pairs are silent
    check_preset_suits_the_model("RFDETR_TACO_PRETRAIN", "rfdetr_seg_small")
    check_preset_suits_the_model("TACO_PRETRAIN", "maskrcnn_v2")


# =============================================================================
# results.csv
# =============================================================================
def test_the_rfdetr_columns_keep_the_two_by_nine_grid():
    """Different loss names, identical layout -- which is what makes the figures comparable.

    ultralytics' plot_results splits columns on the substrings "loss" and "metric" and lays
    out (2, n/2), so ten loss columns and eight metric columns give the same 2x9 grid as the
    YOLO segment run regardless of what the five losses are called.
    """
    columns = csv_columns(RFDETR_LOSS_COLUMNS)
    assert len(columns) == 23
    assert len([c for c in columns if "loss" in c]) == 10
    assert len([c for c in columns if "metric" in c]) == 8
    assert columns[:2] == ["epoch", "time"]
    assert columns[-3:] == ["lr/pg0", "lr/pg1", "lr/pg2"]


def test_the_rfdetr_loss_names_say_what_they_are():
    """box/mask/cls are shared with Mask R-CNN; giou and aux are RF-DETR's own."""
    assert list(RFDETR_LOSS_COLUMNS.values()) == [
        "box_loss", "mask_loss", "cls_loss", "giou_loss", "aux_loss"]


# =============================================================================
# the loss aggregation
# =============================================================================
def fake_criterion_output():
    """What SetCriterion actually returns, in miniature.

    Shape taken from a real RFDETRSegNano run: 25 weighted terms (5 base x the final layer,
    three auxiliary decoder layers and the encoder) plus 6 unweighted diagnostics.
    """
    raw, weights = {}, {}
    base = {"loss_bbox": 5.0, "loss_giou": 2.0, "loss_ce": 1.0,
            "loss_mask_ce": 5.0, "loss_mask_dice": 5.0}
    for suffix in ("", "_0", "_1", "_2", "_enc"):
        for key, weight in base.items():
            raw[key + suffix] = torch.tensor(0.5)
            weights[key + suffix] = weight
    # reported by the criterion, carries no weight, must never reach the CSV or the backward
    raw["class_error"] = torch.tensor(42.0)
    for suffix in ("", "_0", "_1", "_2", "_enc"):
        raw["cardinality_error" + suffix] = torch.tensor(99.0)
    return raw, weights


def test_the_five_columns_sum_to_exactly_the_optimised_total():
    """engine.train_epoch backpropagates sum(losses.values()), so this must be exhaustive.

    If any weighted term were dropped, the loss curve in results.png would be a different
    quantity from the one being minimised -- a plot that silently lies. If any unweighted
    diagnostic were included (cardinality_error is ~99), the total would be dominated by a
    number that carries no gradient.
    """
    raw, weights = fake_criterion_output()
    aggregated = aggregate_losses(raw, weights)

    assert set(aggregated) == set(RFDETR_LOSS_COLUMNS)
    expected = sum(float(raw[k]) * w for k, w in weights.items())
    assert float(sum(aggregated.values())) == pytest.approx(expected)


def test_diagnostics_are_excluded_rather_than_deny_listed():
    """Keying off weight_dict means a new diagnostic cannot leak in later.

    The weighted total is exactly 45: five layers x (5 + 2 + 1 + 5 + 5) weight x 0.5 each.
    class_error is 42 and each cardinality_error is 99, so admitting even one of the six
    would overshoot -- which is why they are given values nothing legitimate can produce.
    """
    raw, weights = fake_criterion_output()
    aggregated = aggregate_losses(raw, weights)

    assert float(sum(aggregated.values())) == pytest.approx(45.0)
    # and the diagnostics really were in the input, so this is not a vacuous pass
    assert {"class_error", "cardinality_error"} <= set(raw)
    assert not any(k in weights for k in raw if "error" in k)


def test_aux_losses_are_not_silently_dropped():
    """The reason there is a fifth column at all.

    Four of the five layers RF-DETR computes losses at are auxiliary, so aux_loss should
    carry 4/5 of the total. The engine's `if k in totals` filter would have discarded all of
    it from results.csv while still training on it.
    """
    raw, weights = fake_criterion_output()
    aggregated = aggregate_losses(raw, weights)
    total = float(sum(aggregated.values()))
    assert float(aggregated["loss_aux"]) == pytest.approx(total * 4 / 5)
    assert float(aggregated["loss_aux"]) > 0


def test_the_mask_column_is_both_mask_terms():
    """RF-DETR splits its mask loss in two; Mask R-CNN's single mask_loss is the comparison."""
    raw, weights = fake_criterion_output()
    aggregated = aggregate_losses(raw, weights)
    expected = float(raw["loss_mask_ce"]) * 5.0 + float(raw["loss_mask_dice"]) * 5.0
    assert float(aggregated["loss_mask"]) == pytest.approx(expected)


def test_a_box_only_criterion_omits_the_mask_column_rather_than_zeroing_it():
    """A detection variant has no mask terms, so results.csv must show NaN, not 0.

    `fasterrcnn_v2` already behaves this way (its absent `loss_mask` becomes NaN via
    log_epoch's `.get(k, nan)`), and metrics.py makes the argument explicitly: a zero would
    read as "it tried and scored nothing", which is a different and misleading claim. The
    column count is unaffected -- csv_columns is built from RFDETR_LOSS_COLUMNS, not from
    this dict.
    """
    raw, weights = fake_criterion_output()
    raw = {k: v for k, v in raw.items() if "mask" not in k}
    weights = {k: v for k, v in weights.items() if "mask" not in k}
    aggregated = aggregate_losses(raw, weights)

    assert "loss_mask" not in aggregated, "a zero here would plot as a real flat line"
    assert set(aggregated) == set(RFDETR_LOSS_COLUMNS) - {"loss_mask"}
    # and the CSV still has its 23 columns and 2x9 grid
    assert len(csv_columns(RFDETR_LOSS_COLUMNS)) == 23


def test_the_columns_come_back_in_the_csvs_own_order():
    """So that the dict reads like the row it becomes, whatever order the criterion used."""
    raw, weights = fake_criterion_output()
    assert list(aggregate_losses(raw, weights)) == list(RFDETR_LOSS_COLUMNS)


# =============================================================================
# the box conversion
# =============================================================================
def test_box_conversion_round_trips():
    """xyxy pixels <-> cxcywh normalised, the format RF-DETR's criterion wants."""
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 96.0, 96.0]])
    converted = xyxy_to_cxcywh_normalised(boxes, 96)
    assert torch.allclose(cxcywh_normalised_to_xyxy(converted, 96), boxes, atol=1e-5)


def test_box_conversion_is_normalised_and_centre_based():
    """Spelled out rather than only round-tripped: a self-consistent wrong formula would
    round-trip perfectly."""
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    cx, cy, w, h = xyxy_to_cxcywh_normalised(boxes, 100)[0].tolist()
    assert (cx, cy) == pytest.approx((0.30, 0.50))   # centres, not corners
    assert (w, h) == pytest.approx((0.40, 0.60))     # extents, not corners
    assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))


def test_box_conversion_survives_an_image_with_no_instances():
    """datasets.py reshapes for this case; so must every consumer of it."""
    empty = torch.zeros(0, 4)
    assert xyxy_to_cxcywh_normalised(empty, 96).shape == (0, 4)
    assert cxcywh_normalised_to_xyxy(empty, 96).shape == (0, 4)


# =============================================================================
# the adapter's two interfaces, without building a real model
# =============================================================================
class StubNet(torch.nn.Module):
    """Just enough LWDETR to exercise the adapter's conversions."""

    def __init__(self, num_classes=4, queries=6, mask_size=24):
        super().__init__()
        self.num_classes, self.queries, self.mask_size = num_classes, queries, mask_size
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.seen = None

    def forward(self, x):
        self.seen = x
        batch = len(x)
        return {
            "pred_logits": torch.zeros(batch, self.queries, self.num_classes + 1),
            "pred_boxes": torch.full((batch, self.queries, 4), 0.5),
        }


def stub_adapter(num_classes=4, imgsz=96, num_select=6):
    from rfdetr.models import PostProcess

    net = StubNet(num_classes=num_classes)
    adapter = RFDETRAdapter(
        net,
        criterion=None,
        postprocess=PostProcess(num_select=num_select, upsample_masks_to_image_size=False),
        num_classes=num_classes,
        imgsz=imgsz,
    )
    return adapter


def test_the_adapter_normalises_with_the_imagenet_statistics():
    """The dataset hands over RGB 0..1 and does not normalise; RF-DETR expects it done.

    Getting this wrong is the quietest possible bug: the model trains, converges to
    something, and simply scores worse than it should.
    """
    adapter = stub_adapter()
    adapter.eval()
    with torch.no_grad():
        adapter([torch.zeros(3, 96, 96)])
    seen = adapter.net.seen[0, :, 0, 0]
    expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(seen, expected, atol=1e-6), "input was not ImageNet-normalised"


def test_the_adapter_returns_the_torchvision_contract():
    """engine.validate and engine._to_numpy depend on exactly these keys and ranks."""
    adapter = stub_adapter()
    adapter.eval()
    with torch.no_grad():
        predictions = adapter([torch.rand(3, 96, 96), torch.rand(3, 96, 96)])

    assert isinstance(predictions, list) and len(predictions) == 2
    for prediction in predictions:
        assert set(prediction) >= {"boxes", "scores", "labels"}
        assert prediction["boxes"].ndim == 2 and prediction["boxes"].shape[1] == 4
        assert prediction["scores"].ndim == 1
        assert prediction["labels"].dtype == torch.int64


def test_predicted_labels_are_one_based_and_never_background():
    """PostProcess's top-k will select the spare column; the harness has no name for it.

    rfdetr allocates num_classes + 1 output columns, our four classes occupy 0..3, and
    column 4 is the spare. Left in, `labels + 1` would be 5 -- past the end of
    datasets.CLASS_NAMES, which metrics.Evaluator and ultralytics' plot_images both index
    by. Verified against a model that really does emit it.
    """
    adapter = stub_adapter(num_select=6)
    # make the spare column win on half the queries
    adapter.net.forward = lambda x: {
        "pred_logits": torch.tensor([[[0.0, 0, 0, 0, 9.0]] * 3 + [[9.0, 0, 0, 0, 0.0]] * 3]),
        "pred_boxes": torch.full((1, 6, 4), 0.5),
    }
    adapter.eval()
    with torch.no_grad():
        labels = adapter([torch.rand(3, 96, 96)])[0]["labels"]

    assert len(labels) == 3, f"the background column was not dropped: {labels.tolist()}"
    assert labels.min() >= 1 and labels.max() <= 4, labels.tolist()


def test_target_labels_are_converted_to_the_zero_based_category_id():
    """The other direction: datasets.py emits 1..4, SetCriterion indexes by category_id.

    rfdetr derives num_classes as max(category_id) + 1, so with our 0-based COCO ids the
    label index *is* the category id.
    """
    adapter = stub_adapter()
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1]),                      # glass, 1-based
        "masks": torch.zeros(1, 96, 96, dtype=torch.uint8),
    }
    converted = adapter._to_detr_target(target)
    assert converted["labels"].tolist() == [0], "glass must reach the criterion as 0"
    assert converted["masks"].dtype == torch.float32


def test_target_masks_are_passed_at_full_resolution():
    """SetCriterion.loss_masks point-samples at normalised coordinates, so no resize is
    needed -- and resizing could only lose detail."""
    adapter = stub_adapter()
    target = {
        "boxes": torch.zeros(1, 4),
        "labels": torch.tensor([1]),
        "masks": torch.zeros(1, 96, 96, dtype=torch.uint8),
    }
    assert adapter._to_detr_target(target)["masks"].shape == (1, 96, 96)


# =============================================================================
# the mask resolution that makes the full validation batch affordable
# =============================================================================
def test_the_two_mask_ratios_agree():
    """RF-DETR's mask_downsample_ratio and metrics.MASK_RATIO are both 4, which is the
    coincidence that makes the next test's resolution the right one to ask for."""
    from other_models.models import MODELS, ModelSpec
    from other_models.rfdetr_adapter import model_config

    spec = ModelSpec(family="rfdetr", config_class="RFDETRSegSmallConfig")
    config = model_config(spec, 4, FINETUNE_NMS.replace(imgsz=648), pretrained=False)
    assert config.mask_downsample_ratio == MASK_RATIO == 4


def test_the_adapter_does_not_let_postprocess_upsample_the_masks():
    """Masks must arrive at imgsz/4, the resolution metrics.MASK_RATIO scores at.

    Asserted on the adapter rather than only on the config, because the config says nothing
    about it: the earlier version of this test checked `mask_downsample_ratio == 4` and
    would have stayed green if `upsample_masks_to_image_size=False` were deleted -- at which
    point masks come back at the full 648 squared, engine._to_numpy pays a 16x downsample,
    and the per-image mask cost that val_batch's cap exists to avoid comes straight back
    while val_batch still hands out the full batch for this family.
    """
    adapter = stub_adapter()
    assert adapter.postprocess.upsample_masks_to_image_size is False


def test_val_batch_is_not_shrunk_for_rfdetr():
    """engine.val_batch's cap is a torchvision concern; gating it on `kind` would halve
    RF-DETR's validation batch for a cost it does not pay."""
    from other_models.engine import Trainer
    from other_models.models import ModelSpec

    trainer = Trainer.__new__(Trainer)
    trainer.hyp = RFDETR_FINETUNE_NMS
    trainer.spec = ModelSpec(family="rfdetr", kind="instance_seg",
                             config_class="RFDETRSegSmallConfig")
    assert trainer.val_batch() == RFDETR_FINETUNE_NMS.batch

    # the Mask R-CNN behaviour is unchanged by the same edit
    trainer.spec = ModelSpec(builder=None, weights=None, kind="instance_seg")
    trainer.hyp = FINETUNE_NMS
    assert trainer.val_batch() == 4


# =============================================================================
# freezing (needs a real model, so these are the first slow-ish ones)
# =============================================================================
# A SegNano-shaped entry with the COCO checkpoint explicitly switched off. Two reasons:
# the published rf-detr-seg-nano.pt is a 128 MB download, and RF-DETR forces
# load_dinov2_weights=False for every patch_size != 14 variant and builds the encoder from
# its own bundled config JSON -- so with pretrain_weights=None a real model is constructed
# entirely offline. `extra` is the ModelSpec escape hatch that reaches the ModelConfig.
TINY_SPEC_KWARGS = dict(
    family="rfdetr",
    kind="instance_seg",
    config_class="RFDETRSegNanoConfig",
    loss_columns=RFDETR_LOSS_COLUMNS,
    extra={"pretrain_weights": None},
)


@pytest.fixture(scope="module")
def seg_nano():
    """A real, randomly-initialised RFDETRSegNano at 96 px. Built once; nothing downloads."""
    from other_models import rfdetr_adapter
    from other_models.models import ModelSpec

    spec = ModelSpec(**TINY_SPEC_KWARGS)
    hyp = FINETUNE_NMS.replace(imgsz=96, trainable_encoder_blocks=DINOV2_SMALL_BLOCKS)
    adapter, _ = rfdetr_adapter.build(spec, 4, hyp)
    return adapter


def trainable_block_indexes(net) -> list[int]:
    blocks = net._get_backbone_encoder_layers()
    return [i for i, b in enumerate(blocks) if any(p.requires_grad for p in b.parameters())]


@pytest.mark.slow
@pytest.mark.parametrize("trainable", [0, 1, 7, 12])
def test_freezing_matches_the_table(seg_nano, trainable):
    """config.py's table is a judgement call, which is exactly why it is asserted.

    The trainable blocks must be the *last* N: a transformer's late blocks are the
    task-specific ones, the same reasoning behind torchvision counting ResNet stages from
    the deepest.
    """
    import copy

    net = copy.deepcopy(seg_nano.net)
    for param in net.parameters():
        param.requires_grad = True

    count = freeze_encoder_blocks(net, trainable)
    assert count == DINOV2_SMALL_BLOCKS
    assert trainable_block_indexes(net) == list(range(12 - trainable, 12))


# The stem parameters, named so the test fails loudly if they are ever renamed upstream
# rather than quietly matching nothing. These are what `freeze_encoder_blocks` must catch.
STEM_PARAMS = [
    "encoder.embeddings.cls_token",
    "encoder.embeddings.mask_token",
    "encoder.embeddings.position_embeddings",
    "encoder.embeddings.patch_embeddings.projection.weight",
    "encoder.embeddings.patch_embeddings.projection.bias",
]


@pytest.mark.slow
def test_freezing_also_freezes_the_stem(seg_nano):
    """torchvision's trainable_layers never unfreezes the ResNet stem either.

    The first version of this freeze filtered on `"embeddings" in name and "encoder" not in
    name`, which matched nothing at all -- every parameter here is `encoder.embeddings.*` --
    so the stem trained in every run while args.yaml claimed one block was trainable. Hence
    the explicit name list rather than a repeat of the implementation's own predicate.
    """
    import copy

    net = copy.deepcopy(seg_nano.net)
    for param in net.parameters():
        param.requires_grad = True
    freeze_encoder_blocks(net, 1)

    named = dict(net.backbone[0].encoder.named_parameters())
    for name in STEM_PARAMS:
        assert name in named, f"{name} has been renamed upstream; update freeze_encoder_blocks"
        assert not named[name].requires_grad, f"{name} is still training"

    # the encoder's final layernorm is deliberately NOT frozen: it feeds the projector
    assert named["encoder.layernorm.weight"].requires_grad


@pytest.mark.slow
def test_freezing_rejects_an_out_of_range_count(seg_nano):
    import copy

    with pytest.raises(ValueError, match="out of range"):
        freeze_encoder_blocks(copy.deepcopy(seg_nano.net), 13)


@pytest.mark.slow
def test_the_frozen_blocks_do_not_reach_the_optimizer(seg_nano):
    """Freezing is only real if build_optimizer honours it, and it does so implicitly (it
    skips requires_grad=False), which is the kind of coupling worth pinning."""
    import copy

    from other_models.engine import build_optimizer

    everything = copy.deepcopy(seg_nano.net)
    for param in everything.parameters():
        param.requires_grad = True
    frozen = copy.deepcopy(everything)
    freeze_encoder_blocks(frozen, 1)

    def count(model):
        optimizer = build_optimizer(model, RFDETR_FINETUNE_NMS)
        groups = [g["param_group"] for g in optimizer.param_groups]
        assert groups == ["weight", "bn", "bias"], groups    # still the three lr/pg* columns
        return sum(p.numel() for g in optimizer.param_groups for p in g["params"])

    # The exact delta, not merely "fewer": freezing one scalar would satisfy an inequality.
    withheld = sum(p.numel() for p in everything.parameters() if p.requires_grad) - \
        sum(p.numel() for p in frozen.parameters() if p.requires_grad)
    assert withheld > 0
    assert count(everything) - count(frozen) == withheld


@pytest.mark.slow
def test_max_det_is_not_clamped_to_the_query_count(seg_nano):
    """hyp.max_det reaches PostProcess verbatim, and really does yield that many.

    The tempting-but-wrong belief is that a DETR cannot return more detections than it has
    queries, so num_select should be min(max_det, num_queries) = 100. PostProcess's top-k
    is taken over (query, class column) pairs, not over queries, so one query can
    contribute several detections and the real ceiling is num_queries * columns = 500.
    Clamping would have given RF-DETR a third of the detection budget the YOLO and Mask
    R-CNN runs get, truncating the low-confidence tail of the PR curve and depressing mAP.
    Measured here rather than argued.
    """
    assert seg_nano.postprocess.num_select == FINETUNE_NMS.max_det == 300
    assert seg_nano.net.num_queries == 100, "the premise of the test: fewer queries than 300"

    seg_nano.eval()
    with torch.no_grad():
        prediction = seg_nano([torch.rand(3, 96, 96)])[0]
    # 300 minus however many landed on the spare column -- certainly far more than 100
    assert len(prediction["scores"]) > 100, len(prediction["scores"])


@pytest.mark.slow
def test_real_masks_come_back_at_mask_ratio_resolution_and_rank(seg_nano):
    """engine._to_numpy does `masks[keep][:, 0] > 0.5`, so both the rank and the size matter.

    The stub tests above cannot cover this -- StubNet emits no pred_masks -- so this is the
    only check that the `[n, 1, imgsz/4, imgsz/4]` contract actually holds against a real
    segmentation head.
    """
    seg_nano.eval()
    with torch.no_grad():
        prediction = seg_nano([torch.rand(3, 96, 96)])[0]

    masks = prediction["masks"]
    assert masks.ndim == 4 and masks.shape[1] == 1, tuple(masks.shape)
    assert masks.shape[-2:] == (96 // MASK_RATIO, 96 // MASK_RATIO), tuple(masks.shape)
    assert len(masks) == len(prediction["scores"]) == len(prediction["labels"])


# =============================================================================
# the real thing
# =============================================================================
def register_tiny_rfdetr(monkeypatch):
    """An RFDETRSegNano with random weights, so the slow tests never hit the network."""
    from other_models.models import MODELS, ModelSpec

    monkeypatch.setitem(MODELS, "tiny_rfdetr", ModelSpec(**TINY_SPEC_KWARGS))


def rfdetr_hyp(hyp, **overrides):
    return hyp.replace(trainable_encoder_blocks=1, **overrides)


@pytest.mark.slow
def test_two_epochs_produce_the_full_artifact_set(tmp_path, source, hyp, monkeypatch):
    """The same 24-file run directory as Mask R-CNN, from a completely different architecture.

    That is the whole point of the package, so it is checked for RF-DETR exactly as it is
    for Mask R-CNN -- including all four Mask*_curve.png, which only appear because this is
    a segmentation variant on a dataset with real polygons.
    """
    from other_models.engine import Trainer

    register_tiny_rfdetr(monkeypatch)
    config = rfdetr_hyp(hyp, project=str(tmp_path), name="smoke", epochs=2, batch=2,
                        workers=0, plots=True, close_mosaic=0, mosaic=0.0)
    result = Trainer("tiny_rfdetr", source, config, device="cpu").train()

    run_dir = tmp_path / "smoke"
    expected = {
        "args.yaml", "results.csv", "results.png", "labels.jpg",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        *[f"{p}{c}_curve.png" for p in ("Box", "Mask") for c in ("P", "R", "F1", "PR")],
        "train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    }
    produced = {p.name for p in run_dir.iterdir()}
    assert expected <= produced, expected - produced

    rows = run_dir.joinpath("results.csv").read_text().strip().splitlines()
    assert len(rows) == 3                                    # header + 2 epochs
    assert rows[0].split(",") == csv_columns(RFDETR_LOSS_COLUMNS)
    assert (run_dir / "weights" / "best.pt").exists()
    assert result["best_fitness"] is not None


@pytest.mark.slow
def test_args_yaml_records_the_rfdetr_specific_resolutions(tmp_path, source, hyp, monkeypatch):
    """args.yaml records both the recipe's max_det and what the model can actually return.

    Here they agree: RFDETRSegNano's top-k runs over (query, class column) pairs, so its
    ceiling is 100 x 5 = 500 and the YAML's 300 is reachable in full. The pair of fields
    exists so that a variant where they *disagree* says so rather than letting a reader
    assume 300 detections were scored when fewer were possible.
    """
    import yaml

    from other_models.engine import Trainer

    register_tiny_rfdetr(monkeypatch)
    config = rfdetr_hyp(hyp, project=str(tmp_path), name="args", epochs=1, batch=2,
                        workers=0, plots=False, close_mosaic=0, mosaic=0.0)
    trainer = Trainer("tiny_rfdetr", source, config, device="cpu")
    written = yaml.safe_load((trainer.run.dir / "args.yaml").read_text())

    assert written["family"] == "rfdetr"
    assert written["max_det"] == config.max_det == 300
    assert written["max_det_effective"] == 300, "the full budget is reachable; see above"
    assert trainer.model.net.num_queries * trainer.model.class_columns == 500 > 300
    assert written["trainable_encoder_blocks"] == 1
    assert written["task"] == "segment"


@pytest.mark.slow
def test_training_reduces_the_loss(tmp_path, source, hyp, monkeypatch):
    """A few epochs on six easy images should move the loss down, not just run without erroring."""
    import csv

    from other_models.engine import Trainer

    register_tiny_rfdetr(monkeypatch)
    config = rfdetr_hyp(hyp, project=str(tmp_path), name="loss", epochs=4, batch=2,
                        workers=0, plots=False, close_mosaic=0, mosaic=0.0,
                        lr0=0.0005, warmup_epochs=0.0)
    Trainer("tiny_rfdetr", source, config, device="cpu").train()

    rows = list(csv.DictReader((tmp_path / "loss" / "results.csv").read_text().splitlines()))
    totals = [sum(float(r[f"train/{c}"]) for c in RFDETR_LOSS_COLUMNS.values()) for r in rows]
    assert totals[-1] < totals[0], totals


# =============================================================================
# the pretrain -> fine-tune chain
# =============================================================================
CLASS_HEAD = "net.class_embed.weight"


@pytest.mark.slow
def test_a_pretrain_checkpoint_carries_into_the_fine_tune(tmp_path, source, hyp, monkeypatch):
    """Stage 2 of the TACO chain really starts from stage 1's best.pt.

    The step that would be invisible if it broke: a fine-tune that silently discarded the
    pretrain looks exactly like one that used it, only with worse numbers. Note the key is
    prefixed `net.` because the adapter, not the LWDETR, is the nn.Module being saved.
    """
    from pathlib import Path

    import yaml

    from other_models.engine import Trainer

    register_tiny_rfdetr(monkeypatch)
    common = dict(project=str(tmp_path), epochs=1, batch=2, workers=0, plots=False,
                  close_mosaic=0, mosaic=0.0)

    pretrain = Trainer("tiny_rfdetr", source, rfdetr_hyp(hyp, name="pretrain", **common),
                       device="cpu").train()
    best = Path(pretrain["run_dir"]) / "weights" / "best.pt"
    trained = torch.load(best, map_location="cpu", weights_only=False)["model"][CLASS_HEAD]

    finetune = Trainer("tiny_rfdetr", source, rfdetr_hyp(hyp, name="finetune", **common),
                       weights_path=str(best), device="cpu")
    assert torch.equal(finetune.model.state_dict()[CLASS_HEAD], trained)

    # The same seed builds the same random model every time, so without this the assertion
    # above could pass on a fine-tune that had loaded nothing at all.
    scratch = Trainer("tiny_rfdetr", source, rfdetr_hyp(hyp, name="scratch", **common),
                      device="cpu")
    assert not torch.equal(scratch.model.state_dict()[CLASS_HEAD], trained)

    finetune.train()
    args = yaml.safe_load((finetune.run.dir / "args.yaml").read_text())
    assert args["pretrained_from"] == str(best)


@pytest.mark.slow
def test_the_chain_survives_the_two_stages_having_different_resolutions(
        tmp_path, source, hyp, monkeypatch):
    """The real TACO chain pretrains at imgsz 840 and fine-tunes at 648.

    DINOv2's position_embeddings is sized (1, (imgsz/patch)^2 + 1, dim), and
    load_state_dict does not forgive mismatched shapes -- so without
    align_checkpoint_resolution the documented chain simply cannot load. The first version
    of this suite ran both stages at the same imgsz and missed it entirely, and the failure
    additionally mis-reported itself as a class-count mismatch.

    96 and 192 here are the two smallest multiples of RFDETRSegNano's divisor that keep the
    test fast; the 2x ratio is what matters, not the values.
    """
    from pathlib import Path

    from other_models.engine import Trainer
    from other_models.rfdetr_adapter import PE_KEY_SUFFIX

    register_tiny_rfdetr(monkeypatch)
    common = dict(project=str(tmp_path), epochs=1, batch=2, workers=0, plots=False,
                  close_mosaic=0, mosaic=0.0)

    pretrain = Trainer("tiny_rfdetr", source,
                       rfdetr_hyp(hyp, name="pretrain_big", imgsz=192, **common),
                       device="cpu").train()
    best = Path(pretrain["run_dir"]) / "weights" / "best.pt"

    # the checkpoint really is at the other resolution, or this test proves nothing
    saved = torch.load(best, map_location="cpu", weights_only=False)["model"]
    pe_key = next(k for k in saved if k.endswith(PE_KEY_SUFFIX))
    assert saved[pe_key].shape[1] == (192 // 12) ** 2 + 1 == 257

    finetune = Trainer("tiny_rfdetr", source,
                       rfdetr_hyp(hyp, name="finetune_small", imgsz=96, **common),
                       weights_path=str(best), device="cpu")
    loaded = finetune.model.state_dict()[f"net.{pe_key.split('net.')[-1]}"]
    assert loaded.shape[1] == (96 // 12) ** 2 + 1 == 65, "PEs were not resampled onto this grid"


@pytest.mark.slow
def test_a_checkpoint_from_a_different_class_count_says_so(tmp_path, source, hyp, monkeypatch):
    """Not a bare torch 'size mismatch for net.class_embed...' wall of text.

    The per-family head key is what makes this message possible for RF-DETR; the
    torchvision key would simply not be found and the raw error would be re-raised.
    """
    from other_models.engine import Trainer

    register_tiny_rfdetr(monkeypatch)
    config = rfdetr_hyp(hyp, project=str(tmp_path), name="mismatch", epochs=1, batch=2,
                        workers=0, plots=False, close_mosaic=0, mosaic=0.0)
    trainer = Trainer("tiny_rfdetr", source, config, device="cpu")

    state = trainer.model.state_dict()
    # an 8-class checkpoint: 8 classes + the spare column
    state[CLASS_HEAD] = torch.zeros(9, state[CLASS_HEAD].shape[1])
    state["net.class_embed.bias"] = torch.zeros(9)
    checkpoint = tmp_path / "eight_classes.pt"
    torch.save({"model": state}, checkpoint)

    with pytest.raises(RuntimeError, match="trained on 8 classes, but this run has 4"):
        Trainer("tiny_rfdetr", source, config.replace(name="refused"),
                weights_path=str(checkpoint), device="cpu")
