"""The report's overlay latency figures: discovery, labelling and the per-run statistics.

Everything here is synthetic -- the module works on CSVs, so none of these need runs/camera to be
present. The load-bearing part is the folder -> label mapping: it decides which numbers get
published under which model's name, and a wrong answer there is a mislabelled figure in a report
rather than a crash.
"""
from __future__ import annotations

import csv
import statistics

import pytest

from camera.benchmark import CSV_COLUMNS
from camera.latency_figures import (FOLDER_DATASETS, FOLDER_MODELS,
                                    SCENE_ORDER, discover_runs, group_runs,
                                    labels_for, plot_latency, scene_label,
                                    table_rows)

SCENES = {
    "clutter_med": "20261016_231841_clutter_med",
    "clutter_high2": "20261016_232523_clutter_high2",
    "stacked": "20261016_233733_stacked",
}


def write_log(directory, latencies, model="yolov8", start=3.4):
    """A per-frame log of the shape RunLogger writes, warm-up frame first."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "2026-10-27-143655-yolov8-custom.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, ms in enumerate(latencies):
            writer.writerow({
                "frame": i,
                "timestamp": f"2026-10-27T14:36:{55 + i:02d}.000",
                "elapsed_s": round(start + i * 0.066, 4),
                "n_detections": 2,
                "inference_ms": ms,
                "loop_ms": ms + 50,
                "fps": round(1000.0 / (ms + 50), 3),
                "model_fps": round(1000.0 / ms, 3),
                "model": model,
                "dataset": "custom",
                "stream_fps": 15.0,
                "extras": "",
            })
    return path


def make_tree(root, model="yolo8", dataset="dataset_seg", csv_model="yolov8",
              scenes=tuple(SCENES), latencies=None):
    """A runs/camera-shaped tree: <model>/<dataset>/<scene>/<run>.csv."""
    latencies = latencies or [3186.0] + [7.0 + 0.1 * i for i in range(20)]
    for scene in scenes:
        write_log(root / model / dataset / SCENES[scene], latencies, model=csv_model)
    return root


# --- labelling -------------------------------------------------------------------------------

def test_scene_label_strips_the_recording_timestamp():
    assert scene_label("20261016_232523_clutter_high2") == "clutter_high2"
    assert scene_label("clutter_high2") == "clutter_high2"


@pytest.mark.parametrize("folder,expected", sorted(FOLDER_MODELS.items()))
def test_every_model_folder_maps_to_its_report_name(tmp_path, folder, expected):
    """The yolo8/yolo26/maskrcnn/rfdetr the report names, not the yolov8 the CSV logs."""
    path = tmp_path / folder / "dataset_seg" / SCENES["stacked"] / "r.csv"
    assert labels_for(path)[0] == expected


@pytest.mark.parametrize("folder,expected", sorted(FOLDER_DATASETS.items()))
def test_every_dataset_folder_maps_to_its_report_name(tmp_path, folder, expected):
    """Both yolo8's `taco_finetune` and everything else's `taco` land on `taco`."""
    path = tmp_path / "yolo8" / folder / SCENES["stacked"] / "r.csv"
    assert labels_for(path)[1] == expected


@pytest.mark.parametrize("bad", ["yolo9", "rfdetr", "yolov8"])
def test_an_unknown_model_folder_raises_rather_than_inventing_a_filename(tmp_path, bad):
    with pytest.raises(KeyError, match="unknown model folder"):
        labels_for(tmp_path / bad / "dataset_seg" / SCENES["stacked"] / "r.csv")


def test_an_unknown_dataset_folder_raises(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset folder"):
        labels_for(tmp_path / "yolo8" / "taco_pretrain" / SCENES["stacked"] / "r.csv")


def test_an_unknown_scene_folder_raises(tmp_path):
    """Two unknown scenes would otherwise be two indistinguishable grey lines on one panel."""
    with pytest.raises(KeyError, match="unknown scene folder"):
        labels_for(tmp_path / "yolo8" / "dataset_seg" / "20261016_231841_mystery" / "r.csv")


# --- the misfiling check ---------------------------------------------------------------------

def test_a_log_filed_under_the_wrong_model_raises(tmp_path):
    """The failure this module cannot otherwise see: a well-formed figure, another model's ms.

    The tree is assembled by hand -- segmentation_realsense writes runs/camera/<bag>/<stem>/ --
    so a file in the wrong folder is a real possibility, and it would publish a 34 ms Mask R-CNN
    number as yolo8 without a word.
    """
    make_tree(tmp_path, model="yolo8", csv_model="maskrcnn")
    with pytest.raises(ValueError, match="filed under"):
        discover_runs(tmp_path)


def test_the_logs_own_model_column_agreeing_is_not_an_error(tmp_path):
    make_tree(tmp_path, model="yolo8", csv_model="yolov8")   # yolov8 is yolo8's logged spelling
    runs, _ = discover_runs(tmp_path)
    assert [r.model for r in runs] == ["yolo8"] * 3


def test_a_log_with_no_model_column_still_reads(tmp_path):
    """Append-only columns: a log predating the column must keep working."""
    directory = tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"]
    path = write_log(directory, [3186.0] + [7.0] * 20)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c in CSV_COLUMNS if c != "model"])
        writer.writeheader()
        for row in rows:
            row.pop("model")
            writer.writerow(row)
    runs, _ = discover_runs(tmp_path)
    assert [r.model for r in runs] == ["yolo8"]


# --- discovery -------------------------------------------------------------------------------

def test_videos_is_skipped_entirely(tmp_path):
    """runs/camera/videos/<scene>/<run>/ matches the same glob the runs do."""
    make_tree(tmp_path)
    write_log(tmp_path / "videos" / SCENES["stacked"] / "anything", [3186.0] + [7.0] * 20)
    runs, skipped = discover_runs(tmp_path)
    assert len(runs) == 3
    assert not any("videos" in line for line in skipped)


def test_runs_are_sorted_by_scene_difficulty_not_alphabetically(tmp_path):
    make_tree(tmp_path)
    runs, _ = discover_runs(tmp_path)
    assert [r.scene for r in runs] == SCENE_ORDER


def test_a_header_only_log_is_skipped_and_said_so(tmp_path):
    """A silently dropped scene is a short table with nothing pointing at the cause."""
    make_tree(tmp_path, scenes=("clutter_med", "clutter_high2"))
    write_log(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], [])
    runs, skipped = discover_runs(tmp_path)
    assert len(runs) == 2
    assert len(skipped) == 1 and "no rows logged" in skipped[0]


def test_a_run_of_only_a_warmup_frame_is_dropped_not_reported_as_a_result(tmp_path):
    """Otherwise a 3000 ms cold start lands in a table captioned 'warm-up excluded'."""
    write_log(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], [3186.0])
    runs, skipped = discover_runs(tmp_path)
    assert runs == []
    assert "all warm-up" in skipped[0]


# --- statistics and output -------------------------------------------------------------------

def test_the_warmup_frame_is_excluded_from_the_series_and_the_statistics(tmp_path):
    latencies = [3186.0] + [7.0, 8.0, 9.0, 10.0]
    write_log(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], latencies)
    (run,), _ = discover_runs(tmp_path)
    assert run.frames == 4
    assert run.mean_ms == pytest.approx(8.5)
    assert run.latency.max() == 10.0          # not 3186
    assert run.elapsed[0] == 0.0              # re-zeroed on the first *measured* frame


def test_the_standard_deviation_is_the_sample_one(tmp_path):
    """ddof=1, matching what the report's table claims. Distinguishable at n=2."""
    write_log(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], [3186.0, 6.0, 8.0])
    (run,), _ = discover_runs(tmp_path)
    assert run.sd_ms == pytest.approx(statistics.stdev([6.0, 8.0]))   # 1.414, not 1.0


def test_one_figure_per_model_dataset_pair_named_for_the_report(tmp_path):
    make_tree(tmp_path, model="yolo8", dataset="taco_finetune")
    make_tree(tmp_path, model="rf-detr", dataset="dataset_seg", csv_model="rfdetr")
    runs, _ = discover_runs(tmp_path)
    out = tmp_path / "out"
    written = {plot_latency(group, out / f"{model}-{dataset}.png").name
               for (model, dataset), group in group_runs(runs).items()}
    assert written == {"yolo8-taco.png", "rfdetr-dataset_seg.png"}
    assert all((out / name).stat().st_size > 0 for name in written)


def test_the_table_carries_one_row_per_run_plus_a_header(tmp_path):
    make_tree(tmp_path)
    runs, _ = discover_runs(tmp_path)
    rows = table_rows(runs)
    assert rows[0] == ["model", "dataset", "scene", "frames", "mean_ms", "sd_ms"]
    assert len(rows) == 1 + 3
    assert [r[2] for r in rows[1:]] == SCENE_ORDER


def test_the_table_and_the_figures_describe_the_same_runs(tmp_path):
    """Both come from one discover_runs pass, so they cannot drift. Pin that."""
    make_tree(tmp_path)
    runs, _ = discover_runs(tmp_path)
    from_table = {(r[0], r[1], r[2]) for r in table_rows(runs)[1:]}
    from_groups = {(run.model, run.dataset, run.scene)
                   for group in group_runs(runs).values() for run in group}
    assert from_table == from_groups
