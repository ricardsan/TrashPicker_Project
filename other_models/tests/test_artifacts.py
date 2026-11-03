"""The run directory, checked against the real YOLO run it has to sit next to."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch

from other_models.artifacts import CSV_COLUMNS, LOSS_COLUMNS, RunDirectory, xyxy_to_xywh
from other_models.datasets import NAMES, ROOT

YOLO_RUN = ROOT / "runs/segment/runs/segment/original-yolo26-seg/original-finetune-nms_v"


@pytest.fixture
def run(tmp_path, hyp):
    return RunDirectory(hyp.replace(project=str(tmp_path), name="run"), NAMES)


def losses(**overrides):
    base = {k: 1.0 for k in LOSS_COLUMNS}
    return {**base, **overrides}


def metrics(box=0.5, mask=0.4):
    return {
        "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7,
        "metrics/mAP50(B)": 0.75, "metrics/mAP50-95(B)": box,
        "metrics/precision(M)": 0.78, "metrics/recall(M)": 0.68,
        "metrics/mAP50(M)": 0.72, "metrics/mAP50-95(M)": mask,
    }


# =============================================================================
# results.csv
# =============================================================================
def test_csv_columns_line_up_with_the_yolo_run():
    """Same 23 columns in the same order; only the five loss names differ.

    That is what makes ultralytics' plot_results draw the same 2x9 grid, and what keeps
    collect_results.py working against both runs.
    """
    if not (YOLO_RUN / "results.csv").exists():
        pytest.skip("the reference YOLO run is not in this checkout")
    reference = (YOLO_RUN / "results.csv").read_text().splitlines()[0].split(",")

    assert len(CSV_COLUMNS) == len(reference) == 23
    for ours, theirs in zip(CSV_COLUMNS, reference):
        if "loss" in theirs:
            assert "loss" in ours                       # a loss slot stays a loss slot
            assert ours.split("/")[0] == theirs.split("/")[0]   # train/ vs val/ must agree
        else:
            assert ours == theirs                       # metrics and lr names are identical


def test_every_model_familys_columns_line_up_with_each_other():
    """The five loss names are per-architecture; the 23 slots are not.

    csv_columns() is what each family's RunDirectory builds its header from, so a second
    family cannot quietly change the column count or order and still be diffable against the
    YOLO run panel for panel.
    """
    from other_models.artifacts import RFDETR_LOSS_COLUMNS, csv_columns

    for loss_columns in (LOSS_COLUMNS, RFDETR_LOSS_COLUMNS):
        columns = csv_columns(loss_columns)
        assert len(columns) == len(CSV_COLUMNS) == 23
        # only the loss names may differ; every other slot is identical
        for ours, theirs in zip(columns, CSV_COLUMNS):
            if "loss" in theirs:
                assert "loss" in ours
                assert ours.split("/")[0] == theirs.split("/")[0]
            else:
                assert ours == theirs
    assert csv_columns() == CSV_COLUMNS, "the default must stay torchvision's"


@pytest.mark.parametrize("family", ["torchvision", "rfdetr"])
def test_plot_results_would_draw_a_two_by_nine_grid(family):
    """Reproduces ultralytics plot_results' own column-splitting logic.

    Run for both families, because the grid is what makes the figures comparable and it
    depends on the column *counts* rather than the names.
    """
    from other_models.artifacts import RFDETR_LOSS_COLUMNS, csv_columns

    names = LOSS_COLUMNS if family == "torchvision" else RFDETR_LOSS_COLUMNS
    header = csv_columns(names)
    loss = [c for c in header if "loss" in c]
    metric = [c for c in header if "metric" in c]
    assert (len(loss), len(metric)) == (10, 8)

    half_loss, half_metric = len(loss) // 2, (len(metric) + 1) // 2
    columns = loss[:half_loss] + metric[:half_metric] + loss[half_loss:] + metric[half_metric:]
    assert len(columns) == 18
    assert (len(columns) + 1) // 2 == 9
    # the four box metrics end the top row, the four mask metrics end the bottom row
    assert columns[5:9] == [c for c in metric if c.endswith("(B)")]
    assert columns[14:18] == [c for c in metric if c.endswith("(M)")]


def test_log_epoch_writes_a_header_once_then_one_row_per_epoch(run):
    for epoch in range(3):
        run.log_epoch(epoch, losses(), metrics(), losses(), [0.01, 0.01, 0.1])
    rows = list(csv.reader(run.csv.read_text().splitlines()))
    assert rows[0] == CSV_COLUMNS
    assert len(rows) == 4
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]       # epochs are 1-based like YOLO's


def test_log_epoch_records_nan_for_a_loss_the_model_does_not_have(run):
    """A box-only model has no mask loss; the column must stay, holding NaN."""
    without_mask = {k: 1.0 for k in LOSS_COLUMNS if k != "loss_mask"}
    run.log_epoch(0, without_mask, metrics(), without_mask, [0.01, 0.01, 0.1])
    row = dict(zip(CSV_COLUMNS, run.csv.read_text().splitlines()[1].split(",")))
    assert math.isnan(float(row["train/mask_loss"]))
    assert float(row["train/box_loss"]) == 1.0


def test_log_epoch_pads_missing_learning_rate_groups(run):
    run.log_epoch(0, losses(), metrics(), losses(), [0.01])
    row = dict(zip(CSV_COLUMNS, run.csv.read_text().splitlines()[1].split(",")))
    assert float(row["lr/pg0"]) == 0.01
    assert math.isnan(float(row["lr/pg2"]))


# =============================================================================
# checkpoints
# =============================================================================
def test_best_is_only_rewritten_when_fitness_improves(run):
    model = torch.nn.Linear(2, 2)
    args = {"model": "test"}

    assert run.save_checkpoint(model, 0, fitness=0.5, metrics=metrics(), train_args=args)
    run.log_epoch(0, losses(), metrics(), losses(), [0.01, 0.01, 0.1])
    best_after_peak = (run.weights / "best.pt").read_bytes()

    assert not run.save_checkpoint(model, 1, fitness=0.2, metrics=metrics(), train_args=args)
    assert run.best_fitness == 0.5
    assert (run.weights / "best.pt").read_bytes() == best_after_peak
    assert (run.weights / "last.pt").exists()
    # best.pt froze at the peak, last.pt moved on
    assert (run.weights / "last.pt").read_bytes() != best_after_peak


def test_checkpoint_carries_the_metadata_needed_to_reload_it(run):
    model = torch.nn.Linear(2, 2)
    run.log_epoch(0, losses(), metrics(), losses(), [0.01, 0.01, 0.1])
    run.save_checkpoint(model, 0, 0.5, metrics(), {"model": "maskrcnn_v2"})

    ckpt = torch.load(run.weights / "best.pt", map_location="cpu", weights_only=False)
    assert set(ckpt) >= {"model", "epoch", "best_fitness", "train_args",
                         "train_metrics", "train_results", "date"}
    assert ckpt["train_args"]["model"] == "maskrcnn_v2"
    assert ckpt["train_metrics"]["fitness"] == 0.5
    assert ckpt["train_results"]["epoch"] == [1.0]
    torch.nn.Linear(2, 2).load_state_dict(ckpt["model"])


def test_checkpoint_does_not_carry_optimizer_state(run):
    """Deliberate: it would triple the file and nothing here resumes from it."""
    run.save_checkpoint(torch.nn.Linear(2, 2), 0, 0.5, metrics(), {})
    ckpt = torch.load(run.weights / "last.pt", map_location="cpu", weights_only=False)
    assert "optimizer" not in ckpt


# =============================================================================
# the run directory itself
# =============================================================================
def test_args_yaml_records_the_resolved_configuration(run, hyp):
    import yaml
    args = yaml.safe_load((run.dir / "args.yaml").read_text())
    assert args["epochs"] == hyp.epochs
    assert args["lr0"] == hyp.lr0
    assert args["save_dir"] == str(run.dir)


def test_run_directories_increment_instead_of_overwriting(tmp_path, hyp):
    config = hyp.replace(project=str(tmp_path), name="run", exist_ok=False)
    first = RunDirectory(config, NAMES)
    second = RunDirectory(config, NAMES)
    assert first.dir != second.dir
    assert second.dir.name.startswith("run")


def test_xyxy_to_xywh():
    boxes = np.array([[10.0, 20.0, 30.0, 60.0]], np.float32)
    assert xyxy_to_xywh(boxes).tolist() == [[20.0, 40.0, 20.0, 40.0]]
    assert xyxy_to_xywh(np.zeros((0, 4), np.float32)).shape == (0, 4)


def test_plot_dataset_labels_writes_labels_jpg(run, source):
    from other_models.datasets import CocoInstanceDataset
    dataset = CocoInstanceDataset(source, "train", 96, cache=False)
    run.plot_dataset_labels(dataset)
    assert (run.dir / "labels.jpg").exists()


def test_plot_metrics_writes_results_png(run):
    for epoch in range(3):
        run.log_epoch(epoch, losses(), metrics(box=0.4 + epoch / 10), losses(),
                      [0.01, 0.01, 0.1])
    run.plot_metrics()
    assert (run.dir / "results.png").exists()
