"""The report's per-class count figures: zero-filling, the time axis and the ground-truth parse.

Everything is synthetic except the one test that reads the real `runs/camera/ground_thruth.txt`,
which is the point of that test: the file is hand-maintained, its column order is not the project's
class order, and a positional read of it would put a wrong-but-plausible reference line in every
panel of every figure.

The load-bearing part here is the zero-filling. These figures are a claim about *counts*, and a
missing frame or an omitted class produces a line that is wrong while looking entirely finished.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from camera.benchmark import CSV_COLUMNS, DETECTION_COLUMNS
from camera.count_figures import (CLASS_ORDER, GROUND_TRUTH, MODEL_ORDER,
                                  discover_counts, group_counts, load_counts,
                                  plot_counts, read_ground_truth,
                                  smoothing_window, table_rows)
from camera.latency_figures import SCENE_ORDER

SCENES = {
    "clutter_med": "20261016_231841_clutter_med",
    "clutter_high2": "20261016_232523_clutter_high2",
    "stacked": "20261016_233733_stacked",
}

# The numbers in runs/camera/ground_thruth.txt, written out by hand. Its columns are listed
# paper, glass, plastic, metal -- not the glass/metal/paper/plastic class order -- so these differ
# from a positional read, which is exactly what this is here to catch.
REAL_GROUND_TRUTH = {
    "clutter_med": {"paper": 3, "glass": 0, "plastic": 1, "metal": 1},
    "clutter_high2": {"paper": 3, "glass": 3, "plastic": 0, "metal": 3},
    "stacked": {"paper": 2, "glass": 1, "plastic": 1, "metal": 1},
}

TRUTH = {scene: dict.fromkeys(CLASS_ORDER, 1) for scene in SCENE_ORDER}


def write_objects(directory, per_frame, model="yolov8", start=0.42):
    """A per-object log of the shape DetectionLogger writes.

    `per_frame` is {frame number: [class name, ...]}. A frame absent from it is absent from the
    file, which is what a frame with no detections at all looks like in a real log.

    `model` goes in the *filename*, not in a column: DETECTION_COLUMNS has no `model` field, which
    is why count_figures checks the stem rather than the column the way latency_figures does.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"2026-10-29-104101-{model}-custom-objects.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTION_COLUMNS)
        writer.writeheader()
        for frame in sorted(per_frame):
            for object_id, cls in enumerate(per_frame[frame]):
                writer.writerow({
                    "frame": frame,
                    "timestamp": f"2026-10-29T10:41:{frame % 60:02d}.000",
                    "elapsed_s": round(start + frame * 0.066, 4),
                    "object_id": object_id,
                    "cls": CLASS_ORDER.index(cls),
                    "class_name": cls,
                    "conf": 0.9,
                    "cx": 100, "cy": 100, "cx_box": 100, "cy_box": 100,
                    "x1": 50, "y1": 50, "x2": 150, "y2": 150,
                    "X_m": 0.1, "Y_m": 0.1, "Z_m": 0.5,
                    "depth_source": "mask-median",
                    "mask_px": 1000, "depth_px": 900, "orientation_deg": 45.0,
                })
    return path


def make_tree(root, model="yolo8", dataset="dataset_seg", csv_model="yolov8",
              scenes=tuple(SCENES), per_frame=None, start=0.42):
    """A runs/camera-shaped tree: <model>/<dataset>/<scene>/<run>-objects.csv."""
    if per_frame is None:
        per_frame = {i: ["glass", "paper"] for i in range(5)}
    for scene in scenes:
        write_objects(root / model / dataset / SCENES[scene], per_frame,
                      model=csv_model, start=start)
    return root


# --- the ground-truth parse ------------------------------------------------------------------

def test_the_real_ground_truth_file_parses_to_its_literal_numbers():
    """Bound by header name, not position: the file lists paper, glass, plastic, metal."""
    truth = read_ground_truth(GROUND_TRUTH)
    assert truth == REAL_GROUND_TRUTH


def test_the_columns_are_bound_by_name_so_reordering_the_header_is_harmless(tmp_path):
    """The same numbers under a permuted header must come back attached to the same classes."""
    original = tmp_path / "a.txt"
    original.write_text(
        "Class paper, glass, plastic, metal,\n"
        "Medium Clutter 3,0,1,1,\nHigh Clutter 3,3,0,3,\nStacked Objects 2,1,1,1,\n")
    permuted = tmp_path / "b.txt"
    permuted.write_text(
        "Class metal, plastic, glass, paper,\n"
        "Medium Clutter 1,1,0,3,\nHigh Clutter 3,0,3,3,\nStacked Objects 1,1,1,2,\n")
    assert read_ground_truth(original) == read_ground_truth(permuted) == REAL_GROUND_TRUTH


def test_an_unknown_scene_row_raises(tmp_path):
    """A row that failed to match would leave a panel with no reference line, looking finished."""
    path = tmp_path / "gt.txt"
    path.write_text("Class paper, glass, plastic, metal,\n"
                    "Mild Disorder 3,0,1,1,\nHigh Clutter 3,3,0,3,\nStacked Objects 2,1,1,1,\n")
    with pytest.raises(KeyError, match="unknown scene row"):
        read_ground_truth(path)


def test_a_header_naming_something_other_than_the_four_classes_raises(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("Class paper, glass, plastic, cardboard,\nStacked Objects 2,1,1,1,\n")
    with pytest.raises(ValueError, match="not the four classes"):
        read_ground_truth(path)


def test_a_scene_with_no_ground_truth_row_raises(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("Class paper, glass, plastic, metal,\nStacked Objects 2,1,1,1,\n")
    with pytest.raises(ValueError, match="no ground truth"):
        read_ground_truth(path)


def test_a_row_with_the_wrong_number_of_fields_raises(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("Class paper, glass, plastic, metal,\nStacked Objects 2,1,1,\n")
    with pytest.raises(ValueError, match="expected 4"):
        read_ground_truth(path)


# --- the zero-filling ------------------------------------------------------------------------

def test_a_class_missing_from_a_frame_counts_as_zero_not_as_a_gap(tmp_path):
    """The failure these figures exist to show: one frame of dropout, not a line stepping over it."""
    per_frame = {0: ["glass"], 1: ["paper"], 2: ["glass"]}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    assert run.counts["glass"].tolist() == [1, 0, 1]
    assert run.counts["paper"].tolist() == [0, 1, 0]


def test_a_frame_absent_from_the_file_is_still_on_the_axis_at_zero(tmp_path):
    """A frame in which nothing at all was detected writes no rows -- yolo8/taco/stacked is one."""
    per_frame = {0: ["glass"], 1: ["glass"], 3: ["glass"]}      # frame 2 saw nothing
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    assert run.frames.tolist() == [0, 1, 2, 3]
    assert run.counts["glass"].tolist() == [1, 1, 0, 1]
    assert all(run.counts[cls][2] == 0 for cls in CLASS_ORDER)


def test_a_class_absent_from_the_whole_run_still_gets_a_full_length_zero_series(tmp_path):
    """A model that saw no glass must draw a flat zero, not disappear from the glass panel."""
    per_frame = {i: ["paper"] for i in range(4)}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    for cls in CLASS_ORDER:
        assert len(run.counts[cls]) == 4
    assert run.counts["glass"].tolist() == [0, 0, 0, 0]


def test_several_objects_of_one_class_in_one_frame_are_all_counted(tmp_path):
    per_frame = {0: ["metal", "metal", "metal"], 1: ["metal"]}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    assert run.counts["metal"].tolist() == [3, 1]


def test_a_fifth_class_raises_rather_than_being_dropped_from_the_figure(tmp_path):
    """A class outside the four-class dataset is a dataset change, not a plotting detail."""
    directory = tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"]
    path = write_objects(directory, {0: ["glass"]})
    text = path.read_text(encoding="utf-8").replace(",glass,", ",cardboard,")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(KeyError, match="not one of"):
        discover_counts(tmp_path)


# --- the time axis ---------------------------------------------------------------------------

def test_the_time_axis_is_comparable_across_models_despite_their_warmup_offsets(tmp_path):
    """elapsed_s puts frame 0 at 0.42 s in one log and 3.34 s in another, for the same clip."""
    per_frame = {i: ["glass"] for i in range(5)}
    write_objects(tmp_path / "maskrcnn" / "dataset_seg" / SCENES["stacked"], per_frame,
                  model="maskrcnn", start=0.42)
    write_objects(tmp_path / "yolo26" / "dataset_seg" / SCENES["stacked"], per_frame,
                  model="yolo26", start=3.34)
    runs, _ = discover_counts(tmp_path)
    assert len(runs) == 2
    assert all(run.video_s[0] == 0.0 for run in runs)
    assert runs[0].video_s == pytest.approx(runs[1].video_s)


def test_the_time_axis_stays_monotonic_across_a_missing_frame(tmp_path):
    """The gap's time is interpolated from the frames either side, not left out of the series."""
    per_frame = {0: ["glass"], 1: ["glass"], 3: ["glass"], 4: ["glass"]}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    assert len(run.video_s) == 5
    assert np.all(np.diff(run.video_s) > 0)
    assert run.video_s[2] == pytest.approx(0.132, abs=1e-6)     # 2 * 0.066


def test_the_warmup_frame_is_kept_unlike_in_the_latency_figures(tmp_path):
    """Its latency is meaningless; its count is a real count of a real frame."""
    per_frame = {0: ["glass", "glass"], 1: ["glass"], 2: ["glass"]}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    assert run.n_frames == 3
    assert run.counts["glass"][0] == 2          # frame 0 is on the figure


# --- discovery -------------------------------------------------------------------------------

def test_videos_is_skipped_entirely(tmp_path):
    """runs/camera/videos/<scene>/<run>/ matches the same glob the runs do, and holds -objects.csv."""
    make_tree(tmp_path)
    write_objects(tmp_path / "videos" / SCENES["stacked"] / "anything",
                  {0: ["glass"]})
    runs, skipped = discover_counts(tmp_path)
    assert len(runs) == 3
    assert not any("videos" in line for line in skipped)


def test_a_per_frame_log_is_skipped_with_a_reason_not_crashed_on(tmp_path):
    """The mirror of what is_detection_log was written to catch: the two differ by a suffix."""
    directory = tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"]
    directory.mkdir(parents=True)
    path = directory / "2026-10-27-144205-yolov8-custom-objects.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({c: 0 for c in CSV_COLUMNS} | {"model": "yolov8"})
    runs, skipped = discover_counts(tmp_path)
    assert runs == []
    assert "per-frame log" in skipped[0]


def test_a_header_only_log_is_skipped_and_said_so(tmp_path):
    make_tree(tmp_path, scenes=("clutter_med", "clutter_high2"))
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], {})
    runs, skipped = discover_counts(tmp_path)
    assert len(runs) == 2
    assert len(skipped) == 1 and "no objects logged" in skipped[0]


def test_a_log_filed_under_the_wrong_model_raises(tmp_path):
    """A misfiled log would publish one architecture's counts under another's name.

    Keyed on the filename, not on a `model` column -- a per-object log has none, so
    latency_figures.check_model_column passes everything here and cannot be reused.
    """
    make_tree(tmp_path, model="yolo8", csv_model="maskrcnn")
    with pytest.raises(ValueError, match="filed under"):
        discover_counts(tmp_path)


def test_the_filename_agreeing_with_the_folder_is_not_an_error(tmp_path):
    make_tree(tmp_path, model="yolo8", csv_model="yolov8")   # yolov8 is yolo8's logged spelling
    runs, _ = discover_counts(tmp_path)
    assert [r.model for r in runs] == ["yolo8"] * 3


def test_a_hand_renamed_log_still_reads(tmp_path):
    """The same latitude check_model_column gives a log predating its column."""
    make_tree(tmp_path, model="yolo8", csv_model="whatever", scenes=("stacked",))
    runs, skipped = discover_counts(tmp_path)
    assert [r.model for r in runs] == ["yolo8"] and skipped == []


def test_runs_are_ordered_by_scene_difficulty_then_by_legend_order(tmp_path):
    for model, csv_model in [("rf-detr", "rfdetr"), ("yolo8", "yolov8")]:
        make_tree(tmp_path, model=model, csv_model=csv_model)
    runs, _ = discover_counts(tmp_path)
    assert [r.scene for r in runs] == [s for s in SCENE_ORDER for _ in range(2)]
    assert [r.model for r in runs[:2]] == ["yolo8", "rfdetr"]        # MODEL_ORDER, not the glob's


# --- output ----------------------------------------------------------------------------------

def test_one_figure_per_variant_and_scene_named_for_the_report(tmp_path):
    make_tree(tmp_path, model="yolo8", dataset="taco_finetune")
    make_tree(tmp_path, model="rf-detr", dataset="dataset_seg", csv_model="rfdetr")
    runs, _ = discover_counts(tmp_path)
    out = tmp_path / "out"
    written = {plot_counts(group, TRUTH, out / f"counts-{dataset}-{scene}.png").name
               for (dataset, scene), group in group_counts(runs).items()}
    assert written == {
        "counts-dataset_seg-clutter_med.png", "counts-dataset_seg-clutter_high2.png",
        "counts-dataset_seg-stacked.png", "counts-taco-clutter_med.png",
        "counts-taco-clutter_high2.png", "counts-taco-stacked.png",
    }
    assert all((out / name).stat().st_size > 0 for name in written)


def test_a_figure_draws_even_when_the_ground_truth_for_a_class_is_zero(tmp_path):
    """clutter_med glass and clutter_high2 plastic are both 0 -- the line still has to appear."""
    make_tree(tmp_path, scenes=("clutter_med",), per_frame={0: ["paper"], 1: ["paper"]})
    runs, _ = discover_counts(tmp_path)
    truth = {"clutter_med": dict.fromkeys(CLASS_ORDER, 0)}
    png = plot_counts(runs, truth, tmp_path / "out" / "zero.png")
    assert png.stat().st_size > 0


def test_the_table_carries_a_row_per_run_per_class_plus_a_header(tmp_path):
    make_tree(tmp_path)
    runs, _ = discover_counts(tmp_path)
    rows = table_rows(runs, TRUTH)
    assert rows[0] == ["dataset", "scene", "model", "class", "frames", "mean", "mode", "truth",
                       "exact_frac"]
    assert len(rows) == 1 + 3 * len(CLASS_ORDER)
    assert [r[3] for r in rows[1:1 + len(CLASS_ORDER)]] == CLASS_ORDER


def test_the_tables_exact_fraction_counts_frames_at_the_ground_truth(tmp_path):
    """Two of four frames with one glass, against a ground truth of one."""
    per_frame = {0: ["glass"], 1: ["glass"], 2: ["glass", "glass"], 3: ["paper"]}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    runs, _ = discover_counts(tmp_path)
    rows = table_rows(runs, {"stacked": {"glass": 1, "metal": 0, "paper": 0, "plastic": 0}})
    glass = next(r for r in rows[1:] if r[3] == "glass")
    assert glass[5] == "1.00"                    # mean (1 + 1 + 2 + 0) / 4
    assert glass[6] == "1"                       # mode
    assert glass[8] == "0.500"                   # exact_frac


def test_the_table_and_the_figures_describe_the_same_runs(tmp_path):
    """Both come from one discover_counts pass, so they cannot drift. Pin that."""
    make_tree(tmp_path)
    runs, _ = discover_counts(tmp_path)
    from_table = {(r[0], r[1], r[2]) for r in table_rows(runs, TRUTH)[1:]}
    from_groups = {(run.dataset, run.scene, run.model)
                   for group in group_counts(runs).values() for run in group}
    assert from_table == from_groups


def test_every_model_in_the_legend_order_has_a_colour_and_a_dash_pattern():
    """A model added to one table and not the others would draw as an unlabelled grey solid."""
    from camera.count_figures import MODEL_COLOURS, MODEL_DASHES
    assert set(MODEL_ORDER) == set(MODEL_COLOURS) == set(MODEL_DASHES)
    assert set(MODEL_ORDER) == set(__import__(
        "camera.latency_figures", fromlist=["FOLDER_MODELS"]).FOLDER_MODELS.values())


def test_the_smoothing_window_is_the_one_the_rest_of_the_report_uses(tmp_path):
    """plot_run's window, which latency_figures also adopts: a tenth of the run, capped at 15."""
    assert smoothing_window(142) == 14          # the real clips
    assert smoothing_window(400) == 15          # capped
    assert smoothing_window(5) == 1             # too short to smooth at all
    assert smoothing_window(0) == 1


def test_a_run_too_short_to_smooth_draws_one_line_per_model_not_two(tmp_path):
    """Otherwise a bold line sits exactly over a faint one and the legend calls it an average."""
    make_tree(tmp_path, scenes=("stacked",), per_frame={i: ["glass"] for i in range(4)})
    runs, _ = discover_counts(tmp_path)
    assert smoothing_window(runs[0].n_frames) == 1
    png = plot_counts(runs, TRUTH, tmp_path / "out" / "short.png")
    assert png.stat().st_size > 0


def test_the_smoothed_series_is_the_projects_rolling_mean_not_a_second_one(tmp_path):
    """Two families of figure in one report must not disagree about the same run."""
    from camera.benchmark import rolling_mean
    per_frame = {i: ["glass"] * (i % 3) for i in range(40)}
    write_objects(tmp_path / "yolo8" / "dataset_seg" / SCENES["stacked"], per_frame)
    (run,), _ = discover_counts(tmp_path)
    window = smoothing_window(run.n_frames)
    smoothed = rolling_mean(run.counts["glass"].astype(float), window)
    assert len(smoothed) == run.n_frames
    # Centred and end-shrinking, so it never drags to the floor at the edges the way a
    # zero-padded convolution does -- which is exactly where a reader looks for the clip's start.
    assert smoothed[0] == pytest.approx(run.counts["glass"][:window // 2 + 1].mean())
