"""Which checkpoint a command line selects, and the labels that come with it."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from camera.segmentation_realsense import (
    ARCHITECTURES,
    DATASET_LABELS,
    MODEL_LABELS,
    MODEL_RUNS,
    ROOT,
    architecture_from_name,
    build_parser,
    check_weights,
    dataset_label,
    infer_architecture,
    is_lfs_pointer,
    list_models,
    main,
    read_run_args,
    resolve_custom_weights,
    resolve_registry,
    run_dir_of,
    run_out_dir,
    weights_of,
)

SCRIPT = ROOT / "camera" / "segmentation_realsense.py"


# --- the registry ------------------------------------------------------------
@pytest.mark.parametrize("key", list(MODEL_RUNS))
def test_every_entry_points_at_a_checkpoint(key):
    """The run folder exists and holds a best.pt. A stub counts: fetching is not this test's job."""
    entry = MODEL_RUNS[key]
    run_dir = ROOT / entry.run_dir
    if not run_dir.exists():
        pytest.skip(f"{entry.run_dir} is not in this working tree")
    assert weights_of(entry.run_dir).exists(), f"no weights/best.pt under {entry.run_dir}"


@pytest.mark.parametrize("key", list(MODEL_RUNS))
def test_every_entry_has_sane_labels(key):
    model, _ = key
    assert model in ARCHITECTURES
    assert MODEL_RUNS[key].dataset in DATASET_LABELS


def test_resolve_registry_returns_the_table_entry():
    weights, architecture, dataset = resolve_registry("yolo26", "taco-finetune-augment")
    assert architecture == "yolo26"
    assert dataset == "custom"
    assert weights == ROOT.joinpath(
        "runs/segment/runs/segment/taco-yolo26-seg/taco-finetune-nms_augment_v",
        "weights", "best.pt")


def test_yolo26_taco_pretrain_is_pinned_to_v11():
    """One of twelve numbered runs is the good one; the rest must stay unreachable."""
    weights, _, dataset = resolve_registry("yolo26", "taco-pretrain-nms-augment")
    assert weights.parent.parent.name == "taco-pretrain-nms-augment_v-11"
    assert dataset == "taco"


@pytest.mark.parametrize("unwanted", [
    "taco-pretrain-nms-augment_v-2", "taco-pretrain-nms-augment_v-3",
    "taco-pretrain-nms-augment_v-4", "taco-pretrain-nms-augment_v-4-fullretrain",
    "taco-pretrain-nms-augment_v-5", "taco-pretrain-nms-augment_v-6",
    "taco-pretrain-nms-augment_v-7", "taco-pretrain-nms-augment_v-8",
    "taco-pretrain-nms-augment_v-9", "taco-pretrain-nms-augment_v-10",
    "taco-pretrain-nms-augment_v-12",
    # Dropped on request: the non-NMS yolo8 TACO pretrain.
    "taco-seg/taco-pretrain-augment_v",
])
def test_scratch_runs_are_not_selectable(unwanted):
    assert not any(unwanted in entry.run_dir for entry in MODEL_RUNS.values())


@pytest.mark.parametrize("unwanted", [
    "runs/segment/rfdetr/finetune_augment_v",
    "runs/segment/rfdetr/finetune_augment_v-2",
    "runs/segment/rfdetr/taco/taco_pretrain_augment_v",
    "runs/segment/rfdetr/taco/taco_finetune_augment_v",
    "runs/segment/rfdetr/smoke_v",
])
def test_rfdetr_scratch_runs_are_not_selectable(unwanted):
    """Each RF-DETR variant was run more than once; only the last of each is the good one.

    Exact match rather than the substring test above, because the unnumbered first run of each
    variant is a prefix of the numbered one that *is* selected.
    """
    assert unwanted not in {entry.run_dir for entry in MODEL_RUNS.values()}


def test_no_duplicate_run_directories():
    """Two names for the same run would make two differently-labelled logs of one model."""
    directories = [entry.run_dir for entry in MODEL_RUNS.values()]
    assert len(directories) == len(set(directories))


def test_unknown_model_raises_with_the_choices():
    with pytest.raises(SystemExit) as error:
        resolve_registry("yolov9", "whatever")
    assert "yolo8" in str(error.value) and "maskrcnn" in str(error.value)


def test_unknown_variant_lists_what_is_available():
    with pytest.raises(SystemExit) as error:
        resolve_registry("maskrcnn", "original-finetune")
    message = str(error.value)
    assert "taco-finetune-augment" in message and "finetune-augment" in message


def test_variant_of_another_architecture_is_rejected():
    """`richard-train-seg` is a yolo8 run; asking for it as yolo26 must not silently work."""
    with pytest.raises(SystemExit):
        resolve_registry("yolo26", "richard-train-seg")


def test_list_models_covers_every_entry():
    text = list_models()
    for (model, variant), entry in MODEL_RUNS.items():
        assert f"{model} {variant}" in text
        assert entry.run_dir in text


# --- the LFS guard -----------------------------------------------------------
def test_pointer_is_recognised(pointer_weights):
    assert is_lfs_pointer(pointer_weights)


def test_real_weights_are_not_a_pointer(run_dir):
    assert not is_lfs_pointer(run_dir("real") / "weights" / "best.pt")


def test_a_small_non_pointer_file_is_not_flagged(tmp_path):
    """The size test alone would misreport any small file; the magic line settles it."""
    path = tmp_path / "best.pt"
    path.write_bytes(b"\x80\x05tiny but genuine")
    assert not is_lfs_pointer(path)


def test_check_weights_passes_a_real_file(run_dir):
    weights = run_dir("real") / "weights" / "best.pt"
    assert check_weights(weights) == weights


def test_check_weights_explains_an_lfs_stub(pointer_weights):
    with pytest.raises(RuntimeError) as error:
        check_weights(pointer_weights)
    message = str(error.value)
    assert "git lfs pull --include=" in message
    assert "best.pt" in message


def test_check_weights_names_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        check_weights(tmp_path / "nope" / "best.pt")
    assert "nope" in str(error.value)


def test_check_weights_never_shells_out(monkeypatch, pointer_weights):
    """It reports; it does not fetch. A hundreds-of-MB download must never be implicit."""
    def explode(*args, **kwargs):
        raise AssertionError("check_weights must not shell out")

    for module, name in ((subprocess, "run"), (subprocess, "check_call"),
                         (subprocess, "check_output"), (subprocess, "Popen"),
                         (os, "system"), (os, "popen")):
        monkeypatch.setattr(module, name, explode, raising=False)

    with pytest.raises(RuntimeError):
        check_weights(pointer_weights)


# --- run directories and args.yaml -------------------------------------------
def test_run_dir_of_a_checkpoint():
    assert run_dir_of(Path("/a/run/weights/best.pt")) == Path("/a/run")


def test_run_dir_of_a_loose_checkpoint():
    assert run_dir_of(Path("/a/run/best.pt")) == Path("/a/run")


def test_read_run_args_of_a_run_without_one(run_dir):
    assert read_run_args(run_dir("bare")) == {}


# --- architecture inference --------------------------------------------------
@pytest.mark.parametrize("recorded,expected", [
    ("yolov8s-seg.pt", "yolo8"),
    ("yolov8n-seg.pt", "yolo8"),
    ("yolo8s-seg", "yolo8"),
    ("yolo26s-seg.pt", "yolo26"),
    ("yolo26s-seg", "yolo26"),
    ("maskrcnn", "maskrcnn"),
    ("maskrcnn_v2", "maskrcnn"),
    ("rfdetr_seg_small", "rfdetr"),
    ("rf-detr-seg-small.pt", "rfdetr"),
    ("fasterrcnn_v2", None),
    ("some/path/best.pt", None),
])
def test_architecture_from_name(recorded, expected):
    assert architecture_from_name(recorded) == expected


def test_infer_architecture_from_args_yaml(run_dir):
    assert infer_architecture(run_dir("y8", {"model": "yolov8s-seg.pt"})) == "yolo8"
    assert infer_architecture(run_dir("y26", {"model": "yolo26s-seg.pt"})) == "yolo26"
    assert infer_architecture(run_dir("mr", {"model": "maskrcnn"})) == "maskrcnn"
    # Both RF-DETR stages record the spec name, not the checkpoint they started from, so
    # even the TACO fine-tune resolves without following a path.
    assert infer_architecture(run_dir("rf", {"model": "rfdetr_seg_small"})) == "rfdetr"


def test_infer_architecture_follows_a_pretrain_path(run_dir):
    """A fine-tune run records the checkpoint it started from, not an architecture name."""
    pretrain = run_dir("pretrain", {"model": "yolo26s-seg"})
    finetune = run_dir("finetune", {"model": str(pretrain / "weights" / "best.pt")})
    assert infer_architecture(finetune) == "yolo26"


def test_infer_architecture_does_not_chase_a_path_loop(run_dir):
    """One hop only: a run whose parent also records a path must give up, not recurse."""
    a = run_dir("a", {"model": "x/weights/best.pt"})
    b = run_dir("b", {"model": str(a / "weights" / "best.pt")})
    with pytest.raises(SystemExit):
        infer_architecture(b)


def test_infer_architecture_raises_without_args_yaml(run_dir):
    with pytest.raises(SystemExit) as error:
        infer_architecture(run_dir("bare"))
    message = str(error.value)
    assert "--model" in message and "bare" in message


def test_infer_architecture_raises_on_an_unknown_model_field(run_dir):
    with pytest.raises(SystemExit) as error:
        infer_architecture(run_dir("odd", {"model": "rtdetr-l.pt"}))
    message = str(error.value)
    assert "rtdetr-l.pt" in message and "--model" in message


# --- dataset labels ----------------------------------------------------------
@pytest.mark.parametrize("args,expected", [
    ({"data": "additional_datasets/taco_yolo/dataset.yaml"}, "taco"),
    ({"source": "taco"}, "taco"),
    ({"data": "dataset_seg/data.yaml"}, "custom"),
    ({"source": "dataset_seg"}, "custom"),
    ({"data": "dataset_seg_v2/data.yaml"}, "custom"),
    ({"data": "dataset_v2/data.yaml"}, "custom"),
    ({"data": "C:/Users/jager/Documents/dataset700/other/data.yaml"}, "richard"),
    ({"data": "something_else/data.yaml"}, "richard"),
    ({}, "richard"),
])
def test_dataset_label_from_args_yaml(run_dir, args, expected):
    assert dataset_label(run_dir("r", args)) == expected


def test_dataset_label_handles_windows_paths(run_dir):
    """Richard's runs record backslash paths; the label must not depend on the separator."""
    assert dataset_label(run_dir("w", {"data": r"C:\data\dataset_seg\data.yaml"})) == "custom"


@pytest.mark.parametrize("key", [k for k in MODEL_RUNS if k != ("yolo8", "richard-train-seg")])
def test_registry_labels_agree_with_the_run_metadata(key):
    """The table is hand-written, so cross-check it against what each run says it trained on."""
    entry = MODEL_RUNS[key]
    run_dir = ROOT / entry.run_dir
    if not (run_dir / "args.yaml").exists():
        pytest.skip(f"{entry.run_dir} has no args.yaml in this working tree")
    assert dataset_label(run_dir) == entry.dataset


def test_richard_train_seg_is_a_deliberate_override():
    """Its args.yaml points at a dataset_seg copy, but the run itself is Richard's own."""
    entry = MODEL_RUNS[("yolo8", "richard-train-seg")]
    run_dir = ROOT / entry.run_dir
    if not (run_dir / "args.yaml").exists():
        pytest.skip("runs/segment/train_seg has no args.yaml in this working tree")
    assert entry.dataset == "richard"
    assert dataset_label(run_dir) == "custom"       # which is why the table overrides it


def test_model_labels_cover_every_architecture():
    assert set(MODEL_LABELS) == set(ARCHITECTURES)
    assert MODEL_LABELS["yolo8"] == "yolov8"        # the report's spelling, not the CLI's


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_no_model_label_contains_a_dash(architecture):
    """benchmark.labels_from reads a log stem as four dash-separated fields.

    A label spelled "rf-detr" would make that fallback split `…-rf-detr-custom` on the wrong
    two, and every figure rebuilt from an older CSV would be titled "detr / custom".
    """
    assert "-" not in MODEL_LABELS[architecture]


# --- --weights ---------------------------------------------------------------
def test_resolve_custom_weights_from_a_run_dir(run_dir):
    directory = run_dir("r", {"model": "maskrcnn"})
    assert resolve_custom_weights(directory) == directory / "weights" / "best.pt"


def test_resolve_custom_weights_from_a_weights_dir(run_dir):
    directory = run_dir("r", {"model": "maskrcnn"})
    assert resolve_custom_weights(directory / "weights") == directory / "weights" / "best.pt"


def test_resolve_custom_weights_from_a_checkpoint(run_dir):
    weights = run_dir("r") / "weights" / "last.pt"
    weights.write_bytes(b"\x80\x05" + b"\x00" * 4096)
    assert resolve_custom_weights(weights) == weights


def test_resolve_custom_weights_names_the_conventional_path_when_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    assert resolve_custom_weights(tmp_path / "empty").name == "best.pt"


# --- the command line --------------------------------------------------------
def no_model_may_load(monkeypatch):
    """Make any attempt to resolve or load a checkpoint fail the test loudly."""
    def explode(*args, **kwargs):
        raise AssertionError("an invalid command line must be rejected before anything loads")

    monkeypatch.setattr("camera.segmentation_realsense.resolve_weights", explode)
    monkeypatch.setattr("camera.segmentation_realsense.build_detector", explode)


def test_weights_and_variant_are_rejected_before_anything_loads(monkeypatch, run_dir):
    """Previously checked after build_detector, so the error cost a torch.load first."""
    no_model_may_load(monkeypatch)
    directory = run_dir("r", {"model": "maskrcnn"})
    with pytest.raises(SystemExit):
        main(["--weights", str(directory), "--variant", "finetune-augment"])


def test_plot_and_no_log_are_rejected_before_anything_loads(monkeypatch):
    """--plot with nothing to plot is a mistake, and must not cost a model load to discover.

    This test used to monkeypatch build_detector to a no-op, which is precisely what hid the
    ordering bug: the patch made the expensive call free, so nothing noticed it ran first.
    """
    no_model_may_load(monkeypatch)
    with pytest.raises(SystemExit):
        main(["--plot", "--no-log"])


def test_an_invalid_combination_loads_no_checkpoint_as_a_subprocess():
    """The same property end to end, where nothing is patched at all."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", "maskrcnn", "--plot", "--no-log"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "not allowed with argument" in combined
    # The two things a checkpoint load would print.
    assert "loading" not in result.stdout
    assert "building" not in result.stdout


def test_missing_plot_from_log_is_reported(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--plot-from", str(tmp_path / "nope.csv")])
    assert "nope.csv" in str(error.value)


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.model is None and args.variant is None      # filled in by resolve_weights
    assert args.conf == 0.25
    assert args.plot is False and args.no_log is False


def test_list_models_as_a_subprocess():
    """The whole script must be importable and runnable without a camera attached."""
    result = subprocess.run([sys.executable, str(SCRIPT), "--list-models"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "maskrcnn taco-finetune-augment" in result.stdout


def test_plot_from_as_a_subprocess(tmp_path):
    from camera.benchmark import RunLogger

    csv_path = tmp_path / "2026-10-20-143512-yolo26-taco.csv"
    with RunLogger(csv_path, "yolo26", "taco") as logger:
        for i in range(25):
            logger.log(i % 3, 0.02, 0.05)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--plot-from", str(csv_path), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "2026-10-20-143512-yolo26-taco.png").exists()


def test_the_module_does_not_need_pyrealsense2(tmp_path):
    """rs is imported inside run_live, so plotting and listing work without the runtime."""
    script = tmp_path / "blocked.py"
    script.write_text(
        "import builtins, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name == 'pyrealsense2':\n"
        "        raise ImportError('pyrealsense2 is blocked for this test')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "from camera.segmentation_realsense import main\n"
        "sys.exit(main(['--list-models']))\n"
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_importing_the_module_runs_nothing(tmp_path):
    """No argparse, no camera, no model load at import time.

    `not hasattr(m, 'rs')` used to stand in for this and could never fail -- rs is a local inside
    run_live. These assertions can: an argv that argparse would reject, and a torch/ultralytics
    stack that must not have been touched.
    """
    script = tmp_path / "import_only.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "sys.argv = ['x', '--this-would-exit-2-if-parsed']\n"
        "import camera.segmentation_realsense as m\n"
        "assert callable(m.main)\n"
        "assert m.MODEL_RUNS, 'the registry should be built at import'\n"
        "# Importing must not drag in a model or open a camera.\n"
        "for heavy in ('pyrealsense2', 'ultralytics.nn.tasks', 'torch', 'rfdetr',\n"
        "              'transformers', 'other_models.models'):\n"
        "    assert heavy not in sys.modules, heavy\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# =============================================================================
# WHERE A RUN'S FILES GO
# =============================================================================
# A run writes up to four files now, and the repository root is a demonstration of what a flat
# layout turns into after a dozen of them.


def test_a_run_gets_a_folder_named_after_itself():
    assert run_out_dir("runs/camera", "2026-10-27-143512-yolo26-custom") == \
        Path("runs/camera/2026-10-27-143512-yolo26-custom")


def test_flat_is_the_layout_runs_used_to_have():
    """A script that globs --out-dir/*.csv must still have a way to work."""
    assert run_out_dir("runs/camera", "2026-10-27-143512-yolo26-custom", flat=True) == \
        Path("runs/camera")


def test_the_parent_can_be_the_current_directory():
    """The default for a camera run."""
    assert run_out_dir(".", "stem") == Path("stem")


def test_two_runs_a_minute_apart_get_different_folders():
    """run_log_name carries the seconds, which is what makes this true."""
    first = run_out_dir("out", "2026-10-27-143512-yolo26-custom")
    second = run_out_dir("out", "2026-10-27-143548-yolo26-custom")
    assert first != second


def test_the_stem_is_not_mangled_by_the_folder():
    """benchmark.labels_from falls back to splitting the stem on dashes, so it must survive."""
    stem = "2026-10-27-143512-yolo26-custom"
    assert run_out_dir("out", stem).name == stem


@pytest.mark.parametrize("flag", ["--centre-depth", "--no-track"])
def test_the_extras_flags_that_need_objects_say_so(flag):
    """Silently ignoring one would leave a user believing they got something they did not."""
    with pytest.raises(SystemExit, match="only means something with --objects"):
        main([flag])


def test_show_ids_implies_objects():
    """There is no id to draw without the tracker that makes them."""
    from camera.segmentation_realsense import check_extras

    args = build_parser().parse_args(["--show-ids"])
    assert not args.objects
    check_extras(args)
    assert args.objects


def test_show_axis_implies_objects():
    """The axis comes out of the mask pass that writes the per-object log."""
    from camera.segmentation_realsense import check_extras

    args = build_parser().parse_args(["--show-axis"])
    assert not args.objects
    check_extras(args)
    assert args.objects


def test_show_axis_and_centre_depth_are_refused():
    """--centre-depth is the mode that never looks at a mask, so there is no axis to draw.

    Rejected rather than quietly implied away: the run would otherwise come up with an empty
    overlay and a blank orientation column, which is an obedient answer to an impossible request.
    """
    with pytest.raises(SystemExit, match="--show-axis needs the object masks"):
        main(["--show-axis", "--centre-depth"])


def test_show_ids_still_works_with_centre_depth():
    """Only --show-axis conflicts with it; the ids do not come from the masks."""
    from camera.segmentation_realsense import check_extras

    args = build_parser().parse_args(["--show-ids", "--centre-depth"])
    check_extras(args)
    assert args.objects and args.centre_depth


def test_check_extras_leaves_a_plain_run_alone():
    from camera.segmentation_realsense import check_extras

    args = build_parser().parse_args([])
    check_extras(args)
    assert not args.objects and not args.show_ids and not args.show_axis


def test_save_video_takes_an_optional_path():
    """Bare --save-video means "the default name"; with a path it means that path."""
    assert build_parser().parse_args(["--save-video"]).save_video is True
    assert build_parser().parse_args(["--save-video", "clip.mp4"]).save_video == "clip.mp4"
    assert build_parser().parse_args([]).save_video is None


def test_the_video_defaults_to_the_run_stem_in_the_run_folder():
    from camera.segmentation_realsense import build_recorder

    args = build_parser().parse_args(["--save-video"])
    recorder = build_recorder(args, "stem", Path("out/stem"), 15.0)
    try:
        assert recorder.path == Path("out/stem/stem.mp4")
        assert recorder.fps == 15.0
    finally:
        recorder.close()


def test_an_explicit_video_path_is_not_prefixed(tmp_path):
    from camera.segmentation_realsense import build_recorder

    args = build_parser().parse_args(["--save-video", str(tmp_path / "clip.mp4")])
    recorder = build_recorder(args, "stem", Path("out/stem"), 15.0)
    try:
        assert recorder.path == tmp_path / "clip.mp4"
    finally:
        recorder.close()


def test_video_fps_overrides_the_streams_rate():
    from camera.segmentation_realsense import build_recorder

    args = build_parser().parse_args(["--save-video", "--video-fps", "5"])
    recorder = build_recorder(args, "stem", Path("out/stem"), 15.0)
    try:
        assert recorder.fps == 5.0
    finally:
        recorder.close()


def test_no_recorder_without_the_flag():
    from camera.segmentation_realsense import build_recorder

    assert build_recorder(build_parser().parse_args([]), "stem", Path("out"), 15.0) is None
