"""Models, the learning-rate schedule, and one real (tiny) training run."""
from __future__ import annotations

import pytest
import torch

from other_models.config import FINETUNE_AUGMENT, FINETUNE_NMS, PRESETS, TACO_PRETRAIN, Hyp
from other_models.datasets import CLASS_NAMES
from other_models.engine import Trainer, build_optimizer, lr_schedule
from other_models.models import MODELS, build, predicts_masks, swap_rcnn_heads


# =============================================================================
# config
# =============================================================================
PRESET_YAMLS = [
    ("FINETUNE_NMS", "yolo_train_configs/26-original-finetune-nms.yaml"),
    ("FINETUNE_AUGMENT", "yolo_train_configs/26-original-finetune-augment.yaml"),
    ("TACO_PRETRAIN", "yolo_train_configs/taco-pretrain/26-taco-pretrain.yaml"),
    # The second stage of the TACO chain. Its YAML is hyperparameter-identical to
    # 26-original-finetune-augment.yaml -- only data, model, name and project differ -- so one
    # preset serves both, and this entry is what will notice if that stops being true.
    ("FINETUNE_AUGMENT", "yolo_train_configs/taco-pretrain/26-taco-finetune.yaml"),
    # The RF-DETR presets mirror the same YAMLs. They are derived from the Mask R-CNN ones
    # rather than rewritten, so every key except their own documented deviations (below) is
    # inherited and gets diffed here too.
    ("RFDETR_FINETUNE_NMS", "yolo_train_configs/26-original-finetune-nms.yaml"),
    ("RFDETR_FINETUNE_AUGMENT", "yolo_train_configs/26-original-finetune-augment.yaml"),
    ("RFDETR_TACO_PRETRAIN", "yolo_train_configs/taco-pretrain/26-taco-pretrain.yaml"),
    ("RFDETR_FINETUNE_AUGMENT", "yolo_train_configs/taco-pretrain/26-taco-finetune.yaml"),
]

# Keys every preset is allowed to set differently from its YAML, with the reason. Anything
# not listed here (or in PRESET_ALLOWANCES) must match the YAML exactly.
ALLOWED_DIFFERENCES = {
    "conf": "the YAML leaves this null; ultralytics resolves null to 0.001 when validating",
    "project": "each model family's runs go in their own tree, as siblings of the YOLO ones",
    "name": "ditto",
}

# Per-preset additions, deliberately NOT folded into ALLOWED_DIFFERENCES: a single global
# list permitting imgsz/batch/patience would stop the Mask R-CNN diff from checking them,
# which is the exact failure mode this test's docstring is about. Each entry is one of the
# RF-DETR deviations written up in config.py section 4.
PRESET_ALLOWANCES = {
    "RFDETR_FINETUNE_NMS": {
        "imgsz": "RF-DETR needs resolution % 24 == 0; 640 % 24 == 16, so 648",
        "batch": "16 does not fit in 16 GB at 648; nbs still holds the effective batch at 64",
        "patience": "the one early stop in this package -- see config.py section 4",
        "warmup_bias_lr": "the YAML's 0.1 trains this architecture to mAP 0.002; "
                          "0.001 reaches 0.881. The largest deviation here, measured",
    },
    "RFDETR_FINETUNE_AUGMENT": {
        "imgsz": "as above",
        "batch": "as above",
        "patience": "as above",
        "warmup_bias_lr": "as above",
    },
    "RFDETR_TACO_PRETRAIN": {
        "imgsz": "832 % 24 == 16, so 840",
        "batch": "imgsz 840 is 4.8x the tokens of the native 384; 2 is what fits",
        # note: no `patience` -- the pretrain keeps its YAML's 15
    },
}


@pytest.mark.parametrize("preset_name,relative_path", PRESET_YAMLS,
                         ids=[f"{n}-{p.rsplit('/', 1)[-1]}" for n, p in PRESET_YAMLS])
def test_presets_mirror_the_yolo_yamls(preset_name, relative_path):
    """Diff each preset against its YAML key by key, rather than spot-checking a few fields.

    The spot-check version of this test passed for months while TACO_PRETRAIN silently carried
    FINETUNE_NMS's entire augmentation recipe instead of its own. It has since caught two real
    edits to 26-taco-pretrain.yaml, which is what it is for -- the YAMLs are the reference, and
    the presets exist only to mirror them.
    """
    import yaml
    from dataclasses import asdict
    from other_models.datasets import ROOT

    yaml_path = ROOT / relative_path
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path} not in this checkout")
    reference = yaml.safe_load(yaml_path.read_text())
    preset = asdict(PRESETS[preset_name])

    allowed = {**ALLOWED_DIFFERENCES, **PRESET_ALLOWANCES.get(preset_name, {})}
    shared = [k for k in preset if k in reference and k not in allowed]
    assert len(shared) > 25, f"only {len(shared)} comparable keys -- the diff has stopped biting"
    mismatched = {k: (reference[k], preset[k]) for k in shared if reference[k] != preset[k]}
    assert not mismatched, f"{preset_name} differs from {yaml_path.name}: {mismatched}"


def test_only_the_rfdetr_presets_are_allowed_to_deviate():
    """Guards the guard: PRESET_ALLOWANCES must not grow to cover the Mask R-CNN presets.

    Those are the like-for-like comparison against the YOLO runs. The moment one of them is
    permitted its own imgsz or batch, `test_presets_mirror_the_yolo_yamls` stops being able
    to say that the recipe matched.
    """
    assert set(PRESET_ALLOWANCES) == {
        "RFDETR_FINETUNE_NMS", "RFDETR_FINETUNE_AUGMENT", "RFDETR_TACO_PRETRAIN"}
    # The closed list of things RF-DETR is allowed to differ on. Adding to it is a decision
    # that belongs in config.py section 4, which is why widening it fails here first.
    for name, keys in PRESET_ALLOWANCES.items():
        assert set(keys) <= {"imgsz", "batch", "patience", "warmup_bias_lr"}, \
            f"{name}: {sorted(keys)}"


def test_every_yaml_augmentation_key_is_actually_compared():
    """Guards the guard: if Hyp ever loses a field, the diff above silently stops checking it."""
    import yaml
    from dataclasses import asdict
    from other_models.datasets import ROOT

    yaml_path = ROOT / next(p for n, p in PRESET_YAMLS if n == "TACO_PRETRAIN")
    if not yaml_path.exists():
        pytest.skip("YAML not in this checkout")
    reference = yaml.safe_load(yaml_path.read_text())
    preset = asdict(TACO_PRETRAIN)
    for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste",
                "copy_paste_mode",
                "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum"):
        assert key in preset and key in reference, key


def test_taco_pretrain_matches_the_new_yaml():
    """The headline numbers of the 2026-10-16 rewrite of 26-taco-pretrain.yaml, spelled out.

    The key-by-key diff above is the real guard; this exists so that a reader can see at a glance
    which recipe the preset is supposed to be, without running yaml. The previous recipe (lr0 0.05,
    scale 0.7, mixup 0.25, 70 epochs) is still on disk as taco-pretrain/best-yolo26.yaml.
    """
    assert (TACO_PRETRAIN.epochs, TACO_PRETRAIN.patience) == (40, 15)
    assert (TACO_PRETRAIN.lr0, TACO_PRETRAIN.lrf) == (0.005, 0.0001)
    assert (TACO_PRETRAIN.scale, TACO_PRETRAIN.shear) == (0.15, 10.0)
    assert (TACO_PRETRAIN.mixup, TACO_PRETRAIN.copy_paste) == (0.0, 0.25)


def test_taco_pretrain_is_not_just_finetune_nms():
    """The specific regression: TACO has its own augmentation recipe, much stronger than NMS's."""
    assert (TACO_PRETRAIN.degrees, TACO_PRETRAIN.shear) == (20.0, 10.0)
    assert (TACO_PRETRAIN.mosaic, TACO_PRETRAIN.cutmix) == (0.5, 0.25)
    assert TACO_PRETRAIN.lrf != FINETUNE_NMS.lrf
    assert TACO_PRETRAIN.optimizer != FINETUNE_NMS.optimizer
    assert TACO_PRETRAIN.copy_paste and not FINETUNE_NMS.copy_paste


def test_the_pretrain_trains_more_of_the_backbone_than_the_fine_tunes():
    """The pretrain is where the backbone is supposed to move; the fine-tunes mostly ride on it.

    This is the `freeze` arithmetic in config.py's table made executable: freeze 5 of yolo26's 11
    backbone layers -> 3 trainable ResNet stages, freeze 10 of 11 -> 1. Nothing derives one from
    the other, which is exactly why it is worth asserting.
    """
    assert (TACO_PRETRAIN.freeze, TACO_PRETRAIN.trainable_backbone_layers) == (5, 3)
    for preset in (FINETUNE_NMS, FINETUNE_AUGMENT):
        assert (preset.freeze, preset.trainable_backbone_layers) == (10, 1)




def test_finetune_presets_differ_in_the_way_their_yamls_do():
    assert (FINETUNE_NMS.epochs, FINETUNE_NMS.mosaic, FINETUNE_NMS.mixup) == (70, 1.0, 0.0)
    assert (FINETUNE_AUGMENT.epochs, FINETUNE_AUGMENT.mosaic) == (150, 0.0)
    assert (FINETUNE_AUGMENT.mixup, FINETUNE_AUGMENT.cutmix) == (0.5, 0.3)
    assert FINETUNE_AUGMENT.degrees == 90.0 and FINETUNE_AUGMENT.flipud == 0.5


def test_replace_rejects_a_typo():
    with pytest.raises(KeyError):
        FINETUNE_NMS.replace(learning_rate=0.1)
    assert FINETUNE_NMS.replace(epochs=5).epochs == 5
    assert FINETUNE_NMS.epochs == 70       # the original is untouched


# =============================================================================
# the learning rate schedule
# =============================================================================
def test_linear_schedule_runs_from_one_down_to_lrf():
    hyp = Hyp(epochs=100, lrf=0.01, cos_lr=False)
    lf = lr_schedule(hyp)
    assert lf(0) == pytest.approx(1.0)
    assert lf(50) == pytest.approx(0.505)
    assert lf(100) == pytest.approx(0.01)


def test_cosine_schedule_runs_from_one_down_to_lrf():
    lf = lr_schedule(Hyp(epochs=100, lrf=0.01, cos_lr=True))
    assert lf(0) == pytest.approx(1.0)
    assert lf(100) == pytest.approx(0.01)
    assert lf(50) == pytest.approx(0.505, abs=1e-6)


def test_optimizer_has_the_three_groups_results_csv_reports():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    optimizer = build_optimizer(model, FINETUNE_NMS)
    groups = [g["param_group"] for g in optimizer.param_groups]
    assert groups == ["weight", "bn", "bias"]
    assert optimizer.param_groups[0]["weight_decay"] == FINETUNE_NMS.weight_decay
    assert optimizer.param_groups[1]["weight_decay"] == 0.0   # norm weights never decay
    assert optimizer.param_groups[2]["weight_decay"] == 0.0   # nor do biases


def test_sgd_preset_builds_too():
    model = torch.nn.Linear(4, 4)
    optimizer = build_optimizer(model, TACO_PRETRAIN)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == TACO_PRETRAIN.momentum


# =============================================================================
# models
# =============================================================================
TORCHVISION_MODELS = sorted(n for n, s in MODELS.items() if s.family == "torchvision")
RFDETR_MODELS = sorted(n for n, s in MODELS.items() if s.family == "rfdetr")


@pytest.mark.parametrize("name", TORCHVISION_MODELS)
def test_every_registered_model_builds_with_our_class_count(name):
    """No pretrained download: weights=None keeps this offline and fast.

    Parametrised over the torchvision family only -- `roi_heads` is a torchvision concept,
    and the RF-DETR entries are covered by their own equivalent below. Both lists come from
    MODELS, so a new entry in either family is picked up automatically and a new *family*
    fails test_every_model_family_is_covered rather than silently going untested.
    """
    spec = MODELS[name]
    model = spec.builder(weights=None, weights_backbone=None,
                         trainable_backbone_layers=3, num_classes=91)
    spec.head_swap(model, len(CLASS_NAMES) + 1)
    assert model.roi_heads.box_predictor.cls_score.out_features == 5   # 4 classes + background
    if predicts_masks(spec):
        assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 5
    else:
        assert getattr(model.roi_heads, "mask_predictor", None) is None


@pytest.mark.slow
@pytest.mark.parametrize("name", RFDETR_MODELS)
def test_every_registered_rfdetr_model_builds_with_our_class_count(name):
    """The RF-DETR half of the above. Marked slow because building one is not instant.

    `class_embed` carries one row per class plus the spare column rfdetr reserves, which is
    the same 4 + 1 the torchvision assertion checks -- the two families agree on the number,
    they just call the head different things.
    """
    pytest.importorskip("rfdetr")
    from dataclasses import replace

    from other_models import rfdetr_adapter

    spec = MODELS[name]
    # no download: see test_rfdetr.TINY_SPEC_KWARGS for why this is enough to stay offline
    offline = replace(spec, extra={**spec.extra, "pretrain_weights": None})
    hyp = FINETUNE_NMS.replace(imgsz=648, trainable_encoder_blocks=1)
    adapter, config = rfdetr_adapter.build(offline, len(CLASS_NAMES), hyp)

    assert adapter.net.class_embed.out_features == 5    # 4 classes + the spare column
    assert config.num_classes == len(CLASS_NAMES)
    assert config.segmentation_head is predicts_masks(spec)


def test_a_torchvision_model_reports_max_det_verbatim():
    """The counterpart to the RF-DETR assertion in test_rfdetr.py.

    torchvision passes hyp.max_det straight to box_detections_per_img, so effective and
    nominal always agree -- and the `getattr(self.model, "postprocess", None)` fallback that
    makes that true is easy to break while editing the RF-DETR path.
    """
    from other_models.engine import Trainer
    from other_models.models import MODELS

    trainer = Trainer.__new__(Trainer)
    trainer.hyp = FINETUNE_NMS
    trainer.spec = MODELS["maskrcnn_v2"]
    trainer.model = MODELS["maskrcnn_v2"].builder(
        weights=None, weights_backbone=None, num_classes=91, trainable_backbone_layers=5)
    assert not hasattr(trainer.model, "postprocess")
    assert trainer.effective_max_det() == FINETUNE_NMS.max_det == 300


def test_every_model_family_is_covered():
    """A third family would otherwise be registered with nothing testing that it builds."""
    assert {s.family for s in MODELS.values()} == {"torchvision", "rfdetr"}
    assert set(TORCHVISION_MODELS) | set(RFDETR_MODELS) == set(MODELS)


def test_build_rejects_an_unknown_model():
    with pytest.raises(KeyError):
        build("resnet18", 4, FINETUNE_NMS)


# =============================================================================
# validation batch size
# =============================================================================
def val_batch_for(hyp, kind="instance_seg"):
    from other_models.engine import Trainer
    from other_models.models import MODELS, ModelSpec
    trainer = Trainer.__new__(Trainer)
    trainer.hyp = hyp
    trainer.spec = ModelSpec(builder=MODELS["maskrcnn"].builder, weights=None, kind=kind)
    return trainer.val_batch()


def test_default_preset_validates_in_batches_that_fit_in_memory():
    """Measured: one validated image costs ~0.92 GB at max_det=300, imgsz=640.

    Validating the training batch of 16 therefore OOMs on a 16 GB card even though training the
    same batch peaks at 5.3 GB, so validation uses a smaller batch.
    """
    assert val_batch_for(FINETUNE_NMS) == 4
    assert val_batch_for(TACO_PRETRAIN) == 2                 # imgsz 832 is dearer still


def test_val_batch_never_exceeds_the_training_batch_or_drops_below_one():
    assert val_batch_for(FINETUNE_NMS.replace(imgsz=320)) == FINETUNE_NMS.batch   # cheap: no cap
    assert val_batch_for(FINETUNE_NMS.replace(imgsz=320, batch=2)) == 2
    assert val_batch_for(FINETUNE_NMS.replace(imgsz=4096)) == 1                   # never 0


def test_box_only_models_validate_at_the_full_batch():
    """The cap exists for pasted masks; a detector without a mask head does not pay for them."""
    assert val_batch_for(FINETUNE_NMS, kind="detect") == FINETUNE_NMS.batch


def capture_build_kwargs(monkeypatch, hyp, weights_path=None):
    """Run build() against a stub builder and return the kwargs it was handed.

    Lets the contract with torchvision be tested without downloading any weights.
    """
    from other_models.models import MODELS, ModelSpec
    seen = {}

    def stub(**kwargs):
        seen.update(kwargs)
        return MODELS["maskrcnn"].builder(weights=None, weights_backbone=None, num_classes=91,
                                          trainable_backbone_layers=5)

    monkeypatch.setitem(MODELS, "stub",
                        ModelSpec(builder=stub, weights="SENTINEL", kind="instance_seg"))
    build("stub", len(CLASS_NAMES), hyp, weights_path)
    return seen


def test_build_always_passes_pretrained_weights(monkeypatch, tmp_path):
    """Even with --weights, the builder must see real weights.

    torchvision computes `is_trained` from this argument; when it is None it silently overrides
    trainable_backbone_layers to 5 and (for Mask R-CNN v1) swaps FrozenBatchNorm2d for trainable
    BatchNorm2d. Passing None would make every --weights run train the whole backbone.
    """
    hyp = FINETUNE_NMS.replace(trainable_backbone_layers=1)
    assert capture_build_kwargs(monkeypatch, hyp)["weights"] == "SENTINEL"

    checkpoint = tmp_path / "pretrain.pt"
    torch.save({"model": {}}, checkpoint)
    seen = capture_build_kwargs(monkeypatch, hyp, weights_path=str(checkpoint))
    assert seen["weights"] == "SENTINEL", "weights=None would reset trainable_backbone_layers to 5"
    assert seen["trainable_backbone_layers"] == 1


def test_build_wires_the_hyperparameters_torchvision_would_otherwise_default(monkeypatch):
    """box_score_thresh in particular: its 0.05 default would make hyp.conf=0.001 unreachable."""
    hyp = FINETUNE_NMS.replace(conf=0.001, iou=0.7, max_det=300, imgsz=640)
    seen = capture_build_kwargs(monkeypatch, hyp)
    assert seen["box_score_thresh"] == 0.001
    assert seen["box_nms_thresh"] == 0.7
    assert seen["box_detections_per_img"] == 300
    # imgsz must pin both ends, or the model's own transform rescales to its 800/1333 defaults
    assert seen["min_size"] == seen["max_size"] == 640


@pytest.mark.parametrize("preset_name,stages", [("TACO_PRETRAIN", 3), ("FINETUNE_NMS", 1)])
def test_trainable_backbone_layers_reaches_the_model(monkeypatch, preset_name, stages):
    """The preset's number arrives at torchvision, and freezes that many ResNet stages.

    Two halves because the honest version of the second half would need the pretrained weights
    downloaded: torchvision forces trainable_backbone_layers to 5 unless it believes weights are in
    play, so the detector builders cannot be asked offline. resnet_fpn_backbone has no such
    condition, and it is the function the detector builders delegate the freezing to.
    """
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    assert capture_build_kwargs(monkeypatch, PRESETS[preset_name])[
        "trainable_backbone_layers"] == stages

    body = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                               trainable_layers=stages).body
    trainable = {name.split(".")[0] for name, p in body.named_parameters() if p.requires_grad}
    assert len(trainable) == stages, sorted(trainable)


# =============================================================================
# the real thing
# =============================================================================
def register_untrained_model(monkeypatch):
    """A Mask R-CNN with random weights, so the slow tests never hit the network."""
    from other_models.models import ModelSpec

    def builder(weights=None, **kwargs):
        return MODELS["maskrcnn"].builder(weights=None, weights_backbone=None,
                                          num_classes=91, **kwargs)

    monkeypatch.setitem(MODELS, "tiny", ModelSpec(builder=builder, weights=None,
                                                  kind="instance_seg"))


@pytest.mark.slow
def test_two_epochs_produce_the_full_artifact_set(tmp_path, source, hyp, monkeypatch):
    """Trains for real on the synthetic dataset, on the CPU, from scratch (no download).

    Checks what the plan promised end to end: every file the YOLO run has, a results.csv that
    grows one row per epoch, and a best.pt that reloads.
    """
    register_untrained_model(monkeypatch)
    config = hyp.replace(project=str(tmp_path), name="smoke", epochs=2, batch=2,
                         workers=0, plots=True, close_mosaic=0, mosaic=0.0)
    result = Trainer("tiny", source, config, device="cpu").train()

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
    assert len(rows) == 3                                   # header + 2 epochs
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert result["best_fitness"] is not None

    ckpt = torch.load(run_dir / "weights" / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["train_args"]["model"] == "tiny"


@pytest.mark.slow
def test_training_reduces_the_loss(tmp_path, source, hyp, monkeypatch):
    """A few epochs on six easy images should move the loss down, not just run without erroring."""
    register_untrained_model(monkeypatch)
    config = hyp.replace(project=str(tmp_path), name="loss", epochs=4, batch=2, workers=0,
                         plots=False, close_mosaic=0, mosaic=0.0, lr0=0.005, warmup_epochs=0.0)
    Trainer("tiny", source, config, device="cpu").train()

    import csv
    rows = list(csv.DictReader((tmp_path / "loss" / "results.csv").read_text().splitlines()))
    totals = [sum(float(r[f"train/{c}"]) for c in
                  ("box_loss", "mask_loss", "cls_loss", "rpn_box_loss", "rpn_obj_loss"))
              for r in rows]
    assert totals[-1] < totals[0], totals


# =============================================================================
# the pretrain -> fine-tune chain
# =============================================================================
MASK_HEAD = "roi_heads.mask_predictor.mask_fcn_logits.weight"


@pytest.mark.slow
def test_a_pretrain_checkpoint_carries_into_the_fine_tune(tmp_path, source, hyp, monkeypatch):
    """Stage 2 of the TACO chain really starts from stage 1's best.pt.

    This is the step that would be invisible if it broke: a fine-tune that silently discarded the
    pretrain looks exactly like one that used it, only with worse numbers at the end.
    """
    import yaml
    from pathlib import Path

    register_untrained_model(monkeypatch)
    common = dict(project=str(tmp_path), epochs=1, batch=2, workers=0, plots=False,
                  close_mosaic=0, mosaic=0.0)

    pretrain = Trainer("tiny", source, hyp.replace(name="pretrain", **common),
                       device="cpu").train()
    best = Path(pretrain["run_dir"]) / "weights" / "best.pt"
    trained = torch.load(best, map_location="cpu", weights_only=False)["model"][MASK_HEAD]

    finetune = Trainer("tiny", source, hyp.replace(name="finetune", **common),
                       weights_path=str(best), device="cpu")
    assert torch.equal(finetune.model.state_dict()[MASK_HEAD], trained)

    # The same seed builds the same random model every time, so without this the assertion above
    # could pass on a fine-tune that had loaded nothing at all.
    scratch = Trainer("tiny", source, hyp.replace(name="scratch", **common), device="cpu")
    assert not torch.equal(scratch.model.state_dict()[MASK_HEAD], trained)

    finetune.train()
    args = yaml.safe_load((finetune.run.dir / "args.yaml").read_text())
    assert args["pretrained_from"] == str(best)


def test_a_checkpoint_from_a_different_class_count_says_so(tmp_path, monkeypatch):
    """Not a bare torch 'size mismatch for roi_heads.box_predictor...' wall of text."""
    model = MODELS["maskrcnn"].builder(weights=None, weights_backbone=None, num_classes=91,
                                       trainable_backbone_layers=5)
    swap_rcnn_heads(model, 8 + 1)                      # an 8-class checkpoint
    checkpoint = tmp_path / "eight_classes.pt"
    torch.save({"model": model.state_dict()}, checkpoint)

    with pytest.raises(RuntimeError, match="trained on 8 classes, but this run has 4"):
        capture_build_kwargs(monkeypatch, FINETUNE_NMS, weights_path=str(checkpoint))
