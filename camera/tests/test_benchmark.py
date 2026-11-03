"""The CSV log, the summary statistics and the figure."""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from camera.benchmark import (
    CSV_COLUMNS,
    DETECTION_COLUMNS,
    WARMUP_FRAMES,
    DetectionLogger,
    RunLogger,
    axis_top,
    class_column,
    class_columns,
    harmonic_fps,
    is_detection_log,
    labels_from,
    mean_per_class_of,
    per_class_columns_of,
    plot_run,
    plotted_rows,
    read_log,
    run_log_name,
    summarise,
)

CLASS_NAMES = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}

NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}-(yolov8|yolo26|maskrcnn)-"
                          r"(custom|taco|richard)$")


# --- the filename ------------------------------------------------------------
@pytest.mark.parametrize("model", ["yolov8", "yolo26", "maskrcnn"])
@pytest.mark.parametrize("dataset", ["custom", "taco", "richard"])
def test_run_log_name_shape(model, dataset):
    name = run_log_name(model, dataset, datetime(2026, 10, 20, 14, 35, 12))
    assert name == f"2026-10-20-143512-{model}-{dataset}"
    assert NAME_PATTERN.match(name)


def test_run_log_name_includes_seconds():
    """Two runs a few seconds apart must not collide: benchmarking repeats the same model."""
    first = run_log_name("yolo26", "taco", datetime(2026, 10, 20, 14, 35, 12))
    second = run_log_name("yolo26", "taco", datetime(2026, 10, 20, 14, 35, 48))
    assert first != second


def test_run_log_name_defaults_to_now():
    assert NAME_PATTERN.match(run_log_name("maskrcnn", "custom"))


# --- RunLogger ---------------------------------------------------------------
class FakeClock:
    """A perf_counter stand-in, so elapsed_s is exact instead of whatever the test took."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_logger_writes_header_and_rows(tmp_path):
    clock = FakeClock()
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco", clock=clock) as logger:
        for i in range(3):
            clock.t += 0.05
            logger.log(n_detections=i, inference_s=0.02, loop_s=0.05)

    text = (tmp_path / "run.csv").read_text()
    assert text.splitlines()[0] == ",".join(CSV_COLUMNS)
    assert len(text.strip().splitlines()) == 4          # header + 3 frames

    rows = read_log(tmp_path / "run.csv")
    assert [int(r["frame"]) for r in rows] == [0, 1, 2]
    assert [int(r["n_detections"]) for r in rows] == [0, 1, 2]
    assert [float(r["elapsed_s"]) for r in rows] == [0.05, 0.1, 0.15]
    assert all(float(r["inference_ms"]) == 20.0 for r in rows)
    assert all(float(r["loop_ms"]) == 50.0 for r in rows)
    # Both rates are derived from the two *_ms columns, so they cannot drift apart from them.
    assert all(float(r["fps"]) == 20.0 for r in rows)
    assert all(float(r["model_fps"]) == 50.0 for r in rows)
    assert {r["model"] for r in rows} == {"yolo26"}
    assert {r["dataset"] for r in rows} == {"taco"}


def test_the_two_rates_are_consistent_with_their_columns(tmp_path):
    """fps and model_fps are conveniences; a reader must be able to trust them."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        logger.log(1, inference_s=0.008, loop_s=0.033)

    row, = read_log(tmp_path / "run.csv")
    assert float(row["fps"]) == pytest.approx(1000.0 / float(row["loop_ms"]), rel=1e-3)
    assert float(row["model_fps"]) == pytest.approx(1000.0 / float(row["inference_ms"]),
                                                    rel=1e-3)
    # The model could do ~125 FPS; the loop achieved ~30. That gap is the whole point.
    assert float(row["model_fps"]) > float(row["fps"])


def test_logger_flushes_every_row(tmp_path):
    """A run killed with Ctrl-C must leave the frames it already managed."""
    with RunLogger(tmp_path / "run.csv", "yolov8", "custom") as logger:
        logger.log(2, 0.01, 0.03)
        mid_run = read_log(tmp_path / "run.csv")          # read while still open
        assert len(mid_run) == 1
        logger.log(3, 0.01, 0.03)
        assert len(read_log(tmp_path / "run.csv")) == 2


def test_logger_creates_missing_out_dir(tmp_path):
    logger = RunLogger(tmp_path / "nested" / "deeper" / "run.csv", "yolo26", "custom")
    logger.log(0, 0.01, 0.02)
    logger.close()
    assert (tmp_path / "nested" / "deeper" / "run.csv").exists()


def test_logger_survives_a_zero_length_frame(tmp_path):
    """1/0 would be inf and would flatten the FPS axis; 0 is reported instead."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        row = logger.log(1, 0.0, 0.0)
    assert row["fps"] == 0.0


def test_logger_rows_match_the_file(tmp_path):
    with RunLogger(tmp_path / "run.csv", "maskrcnn", "taco") as logger:
        for _ in range(4):
            logger.log(1, 0.1, 0.2)
    on_disk = read_log(tmp_path / "run.csv")
    assert len(logger.rows) == len(on_disk)
    assert [str(r["frame"]) for r in logger.rows] == [r["frame"] for r in on_disk]


# --- summarise ---------------------------------------------------------------
def rows_for(latencies_ms, loop_ms=None, detections=None):
    """Synthetic log rows. elapsed_s accumulates the loop times, as a real run's would."""
    loop_ms = loop_ms if loop_ms is not None else [100.0] * len(latencies_ms)
    detections = detections if detections is not None else [1] * len(latencies_ms)

    rows, elapsed = [], 0.0
    for i, (ms, lp, n) in enumerate(zip(latencies_ms, loop_ms, detections)):
        elapsed += lp / 1000.0
        rows.append({
            "frame": i, "elapsed_s": round(elapsed, 6), "inference_ms": ms, "loop_ms": lp,
            "fps": 1000.0 / lp if lp else 0.0,
            "model_fps": 1000.0 / ms if ms else 0.0,
            "n_detections": n, "model": "yolo26", "dataset": "taco",
        })
    return rows


def test_summarise_excludes_the_warmup_frame():
    """The first call pays for the CUDA context; it is not the number a report means."""
    rows = rows_for([500.0, 20.0, 20.0, 20.0, 20.0])
    summary = summarise(rows)
    assert summary.frames == 5
    assert summary.measured == 5 - WARMUP_FRAMES
    assert summary.mean_ms == pytest.approx(20.0)
    assert summary.median_ms == pytest.approx(20.0)


def test_summarise_statistics():
    rows = rows_for([999.0, 10.0, 20.0, 30.0, 40.0],
                    loop_ms=[1000.0, 20.0, 25.0, 33.3, 50.0],
                    detections=[0, 2, 2, 4, 4])
    summary = summarise(rows)
    assert summary.mean_ms == pytest.approx(25.0)
    assert summary.median_ms == pytest.approx(25.0)
    assert summary.p95_ms == pytest.approx(38.5)          # numpy's linear interpolation
    assert summary.mean_detections == pytest.approx(3.0)
    # 4 measured frames over 20 + 25 + 33.3 + 50 = 128.3 ms.
    assert summary.mean_fps == pytest.approx(4 / 0.1283, rel=1e-3)
    # ...and the model alone did 4 frames of inference in 10 + 20 + 30 + 40 = 100 ms.
    assert summary.mean_model_fps == pytest.approx(40.0)


def test_mean_fps_is_throughput_not_the_mean_of_rates():
    """Frame times of 10 and 100 ms average to 55 FPS the naive way; throughput is 18.2."""
    rows = rows_for([1.0, 10.0, 100.0], loop_ms=[1.0, 10.0, 100.0])
    summary = summarise(rows)
    assert summary.mean_fps == pytest.approx(2 / 0.110, rel=1e-3)   # 18.2, not 55.0
    assert summary.mean_fps < 20


def test_harmonic_fps_directly():
    assert harmonic_fps(np.array([10.0, 100.0])) == pytest.approx(18.1818, rel=1e-4)
    assert harmonic_fps(np.array([50.0, 50.0])) == pytest.approx(20.0)
    assert harmonic_fps(np.array([])) == 0.0
    assert harmonic_fps(np.array([0.0, 0.0])) == 0.0


def test_wall_clock_spans_the_measured_frames_only():
    """A multi-second warm-up next to a warm-up-free frame count made the title contradict itself."""
    rows = rows_for([3500.0] + [33.0] * 10, loop_ms=[3600.0] + [33.0] * 10)
    summary = summarise(rows)

    assert summary.measured == 10
    # 10 frames at 33 ms is 0.33 s, not the 3.9 s the whole log spans.
    assert summary.wall_s == pytest.approx(0.33, abs=0.01)
    # The two numbers the title puts side by side now agree with each other.
    assert summary.measured / summary.wall_s == pytest.approx(summary.mean_fps, rel=1e-2)


def test_summarise_empty_log_is_none():
    assert summarise([]) is None


def test_summarise_keeps_the_warmup_when_it_is_all_there_is():
    """A one-frame run should still produce numbers rather than crashing the end of the run."""
    summary = summarise(rows_for([33.0]))
    assert summary.frames == 1
    assert summary.measured == 1
    assert summary.mean_ms == pytest.approx(33.0)


def test_summary_as_lines_mentions_the_key_numbers():
    text = "\n".join(summarise(rows_for([10.0, 20.0, 30.0])).as_lines())
    assert "inference" in text and "FPS" in text and "detections" in text


# --- labels ------------------------------------------------------------------
def test_labels_come_from_the_columns():
    rows = rows_for([10.0])
    assert labels_from("whatever.csv", rows) == ("yolo26", "taco")


def test_labels_fall_back_to_the_filename():
    assert labels_from("2026-10-20-143512-maskrcnn-custom.csv", []) == ("maskrcnn", "custom")


def test_labels_last_resort():
    assert labels_from("x.csv", []) == ("unknown", "unknown")


# --- the figure --------------------------------------------------------------
def write_log(path: Path, n: int) -> Path:
    with RunLogger(path, "yolo26", "taco") as logger:
        for i in range(n):
            logger.log(i % 3, 0.02 + 0.001 * (i % 5), 0.05)
    return path


def test_plot_run_writes_a_real_png(tmp_path):
    csv_path = write_log(tmp_path / "2026-10-20-143512-yolo26-taco.csv", 60)
    png = plot_run(csv_path)
    assert png == csv_path.with_suffix(".png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.stat().st_size > 10_000          # a real figure, not a blank canvas


def test_plot_run_honours_an_explicit_destination(tmp_path):
    csv_path = write_log(tmp_path / "run.csv", 20)
    png = plot_run(csv_path, tmp_path / "figures" / "speed.png")
    assert png.exists() and png.parent.name == "figures"


def test_plotted_rows_drop_the_warmup():
    """A 1000 ms first frame must not squash an 85 ms run into the bottom tenth of the panel."""
    rows = rows_for([1000.0] + [85.0] * 40)
    shown = plotted_rows(rows)
    assert len(shown) == 40
    assert max(float(r["inference_ms"]) for r in shown) == 85.0
    # And the y-axis follows the frames that are actually drawn.
    assert axis_top(np.array([float(r["inference_ms"]) for r in shown])) < 120


def test_plotted_rows_match_what_summarise_describes():
    """Figure and legend must agree: the same frames feed both."""
    rows = rows_for([999.0, 10.0, 20.0, 30.0])
    assert len(plotted_rows(rows)) == summarise(rows).measured


def test_plotted_rows_keep_a_one_frame_run():
    assert len(plotted_rows(rows_for([42.0]))) == 1


def test_axis_top_never_collapses():
    """A perfectly flat series would otherwise get a zero-height axis."""
    assert axis_top(np.zeros(5)) > 0
    assert axis_top(np.array([])) > 0
    assert axis_top(np.array([100.0])) > 100


def test_plot_run_on_a_single_frame(tmp_path):
    csv_path = write_log(tmp_path / "one.csv", 1)
    assert plot_run(csv_path).exists()


def test_plot_run_on_a_header_only_log(tmp_path):
    """A camera that never delivered a frame should not break the last step of the run."""
    csv_path = tmp_path / "empty.csv"
    RunLogger(csv_path, "yolo26", "taco").close()
    assert plot_run(csv_path).exists()


def test_plot_run_needs_no_display(tmp_path):
    """plot_run must force Agg itself: it runs where cv2 owns the display, or over ssh.

    Simply dropping DISPLAY proves nothing here -- no Qt bindings are installed, so matplotlib
    would fall back to agg on its own and this would pass with the use("Agg") call deleted.
    tkinter *is* present, so MPLBACKEND=TkAgg asks for a GUI backend explicitly: the only thing
    that can save the child then is plot_run overriding it.
    """
    pytest.importorskip("tkinter", reason="needs a GUI backend available to be a real test")

    csv_path = write_log(tmp_path / "run.csv", 30)
    script = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
        "from camera.benchmark import plot_run;"
        "import matplotlib;"
        f"print(plot_run({str(csv_path)!r}));"
        "print('backend:', matplotlib.get_backend())"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["MPLBACKEND"] = "TkAgg"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env=env)
    assert result.returncode == 0, result.stderr
    assert "backend: agg" in result.stdout.lower(), result.stdout
    assert csv_path.with_suffix(".png").exists()


def test_csv_is_plain_enough_for_a_spreadsheet(tmp_path):
    """No comment lines or metadata headers: the first line is the column names."""
    csv_path = write_log(tmp_path / "run.csv", 5)
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == CSV_COLUMNS


# =============================================================================
# PER-CLASS COUNTS
# =============================================================================
# The log used to carry one total. "two objects" and "two plastic bottles" are different claims,
# and only the second one is of any use in a report about sorting waste.


def test_the_header_is_unchanged_when_no_class_names_are_given():
    """The columns are append-only, so the default log must stay exactly what it was."""
    assert CSV_COLUMNS[-1] == "extras"
    assert not any(column.startswith("n_") and column != "n_detections"
                   for column in CSV_COLUMNS)


def test_class_names_append_their_columns_in_class_index_order(tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=CLASS_NAMES) as logger:
        assert logger.columns == CSV_COLUMNS + ["n_glass", "n_metal", "n_paper", "n_plastic"]
    header = (tmp_path / "run.csv").read_text().splitlines()[0]
    assert header.endswith("n_glass,n_metal,n_paper,n_plastic")


def test_class_names_out_of_order_still_come_out_in_index_order():
    """detector.names is a dict, and a dict's insertion order is not its class order."""
    assert class_columns({2: "paper", 0: "glass", 3: "plastic", 1: "metal"}) == \
        ["n_glass", "n_metal", "n_paper", "n_plastic"]


def test_a_class_name_with_punctuation_cannot_corrupt_the_header():
    """The names come out of a checkpoint, so a comma in one would break the CSV itself."""
    assert class_column("glass, clear") == "n_glass_clear"
    assert class_column("Plastic Bottle") == "n_plastic_bottle"
    assert class_column("  ") == "n_unnamed"


def test_the_per_class_counts_sum_to_the_total(tmp_path):
    from collections import Counter

    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=CLASS_NAMES) as logger:
        for detections in ([0, 0, 3], [1], [], [2, 2, 2, 3]):
            logger.log(len(detections), 0.02, 0.05, per_class=Counter(detections))

    for row in read_log(tmp_path / "run.csv"):
        per_class = sum(int(row[column]) for column in per_class_columns_of([row]))
        assert per_class == int(row["n_detections"]), row


def test_a_class_that_was_never_seen_is_written_as_zero_not_blank(tmp_path):
    """A blank would read as missing data rather than as "none of these"."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=CLASS_NAMES) as logger:
        logger.log(1, 0.02, 0.05, per_class={0: 1})
    row = read_log(tmp_path / "run.csv")[0]
    assert row["n_glass"] == "1"
    assert row["n_metal"] == row["n_paper"] == row["n_plastic"] == "0"


def test_per_class_counts_are_ignored_without_class_names(tmp_path):
    """Passing them to a logger that has no columns for them must not invent any."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        logger.log(2, 0.02, 0.05, per_class={0: 1, 3: 1})
    assert read_log(tmp_path / "run.csv")[0].keys() == {*CSV_COLUMNS}


def test_the_summary_breaks_the_detections_out_per_class(tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=CLASS_NAMES) as logger:
        logger.log(0, 0.02, 0.05, per_class={})                 # the warm-up, excluded
        logger.log(2, 0.02, 0.05, per_class={0: 2})
        logger.log(2, 0.02, 0.05, per_class={0: 1, 3: 1})

    summary = summarise(logger.rows)
    assert summary.mean_per_class == {"glass": 1.5, "metal": 0.0, "paper": 0.0, "plastic": 0.5}
    text = "\n".join(summary.as_lines())
    assert "glass 1.50" in text and "plastic 0.50" in text


def test_the_summary_says_nothing_per_class_for_a_log_without_the_columns(tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        logger.log(1, 0.02, 0.05)
        logger.log(1, 0.02, 0.05)
    summary = summarise(logger.rows)
    assert summary.mean_per_class == {}
    assert "--" not in [line for line in summary.as_lines() if line.startswith("detections")][0]


def test_mean_per_class_tolerates_a_blank_cell():
    """A hand-edited log should not be able to fail the end-of-run summary."""
    rows = [{"n_glass": "2"}, {"n_glass": ""}, {"n_glass": "4"}]
    assert mean_per_class_of(rows) == {"glass": 3.0}


def test_a_log_with_per_class_columns_still_plots(tmp_path):
    """plot_run reads columns by name, so the extra ones must be invisible to it."""
    path = tmp_path / "run.csv"
    with RunLogger(path, "yolo26", "custom", class_names=CLASS_NAMES) as logger:
        for i in range(20):
            logger.log(i % 3, 0.02, 0.05, per_class={i % 4: i % 3})
    assert plot_run(path).exists()


# =============================================================================
# THE EXTRAS COLUMN
# =============================================================================


def test_extras_is_empty_by_default(tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        logger.log(1, 0.02, 0.05)
    assert read_log(tmp_path / "run.csv")[0]["extras"] == ""


def test_extras_is_repeated_on_every_row(tmp_path):
    """Self-describing, like model and stream_fps: the file answers "is this comparable"."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", extras="objects+video") as logger:
        for _ in range(3):
            logger.log(1, 0.02, 0.05)
    assert [row["extras"] for row in read_log(tmp_path / "run.csv")] == ["objects+video"] * 3


def test_the_extras_label_spells_the_combinations():
    from camera.segmentation_realsense import extras_label

    assert extras_label(False, False) == ""
    assert extras_label(True, False) == "objects"
    assert extras_label(False, True) == "video"
    assert extras_label(True, True) == "objects+video"


# =============================================================================
# THE PER-OBJECT LOG
# =============================================================================


class FakeObservation:
    """What DetectionLogger reads off an Observation. Duck-typed on purpose, as the logger is."""

    def __init__(self, object_id=0, cls=3, conf=0.9, xyxy=(10, 20, 110, 120),
                 centroid=(60, 70), xyz=(0.1, 0.2, 0.55), depth_source="mask-median",
                 mask_px=4000, depth_px=3800, box_centre=None, orientation_deg=37.456):
        self.object_id = object_id
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy
        self.box_centre = box_centre or ((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)
        self.centroid = centroid
        self.xyz = xyz
        self.depth_source = depth_source
        self.mask_px = mask_px
        self.depth_px = depth_px
        self.orientation_deg = orientation_deg


def test_the_per_object_header_is_the_documented_one(tmp_path):
    DetectionLogger(tmp_path / "objects.csv").close()
    with open(tmp_path / "objects.csv", newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == DETECTION_COLUMNS


def test_the_request_is_satisfied_by_the_columns():
    """object id, object class, timestamp, z height, frame number, centroid location."""
    for column in ("object_id", "class_name", "timestamp", "Z_m", "frame", "cx", "cy"):
        assert column in DETECTION_COLUMNS


def test_one_row_per_detection_per_frame(tmp_path):
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(object_id=0), FakeObservation(object_id=1)], CLASS_NAMES)
        logger.log(1, [], CLASS_NAMES)
        logger.log(2, [FakeObservation(object_id=0)], CLASS_NAMES)

    rows = read_log(tmp_path / "objects.csv")
    assert len(rows) == 3
    assert [row["frame"] for row in rows] == ["0", "0", "2"]
    assert logger.rows_written == 3


def test_a_row_carries_the_class_name_as_well_as_the_index(tmp_path):
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(7, [FakeObservation(cls=2)], CLASS_NAMES)
    row = read_log(tmp_path / "objects.csv")[0]
    assert row["cls"] == "2"
    assert row["class_name"] == "paper"


def test_an_unnamed_class_falls_back_to_its_index(tmp_path):
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(cls=9)], CLASS_NAMES)
    assert read_log(tmp_path / "objects.csv")[0]["class_name"] == "9"


def test_a_missing_depth_is_blank_and_never_zero(tmp_path):
    """The whole point of the depth_source column.

    Z16 uses 0 for "nothing came back", which is the normal case on glass and clear plastic. A
    0.0 in this column would be a confident wrong number sitting next to the real ones.
    """
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(xyz=None, depth_source="none", depth_px=0)], CLASS_NAMES)

    row = read_log(tmp_path / "objects.csv")[0]
    assert row["X_m"] == row["Y_m"] == row["Z_m"] == ""
    assert row["depth_source"] == "none"
    assert row["depth_px"] == "0"


def test_the_orientation_is_recorded_to_two_decimals(tmp_path):
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(orientation_deg=37.456)], CLASS_NAMES)
    assert read_log(tmp_path / "objects.csv")[0]["orientation_deg"] == "37.46"


def test_rounding_cannot_push_the_orientation_out_of_range(tmp_path):
    """179.996 is inside [0, 180) and rounds to 180.0, which is not.

    The angle arrives already wrapped, so it is the rounding here that re-opens the range -- and
    a reader binning this column into 0..179, as the column's documentation suggests, would get a
    bucket that should not exist. 179.99 must survive untouched; only the overflow folds to 0.
    """
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(orientation_deg=179.996)], CLASS_NAMES)
        logger.log(1, [FakeObservation(orientation_deg=179.994)], CLASS_NAMES)

    rows = read_log(tmp_path / "objects.csv")
    assert rows[0]["orientation_deg"] == "0.0"
    assert rows[1]["orientation_deg"] == "179.99"


def test_a_missing_orientation_is_blank_and_never_zero(tmp_path):
    """0.0 is a real value in this column -- it means "lying horizontally".

    Writing it for an object that had no mask to take an axis over would be a claim about the
    object rather than about the absence of one, exactly as a depth of 0.0 would be.
    """
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(orientation_deg=None, mask_px=0)], CLASS_NAMES)
        logger.log(1, [FakeObservation(orientation_deg=0.0)], CLASS_NAMES)

    rows = read_log(tmp_path / "objects.csv")
    assert rows[0]["orientation_deg"] == ""
    assert rows[1]["orientation_deg"] == "0.0", "a real horizontal must survive as a number"


def test_the_box_and_the_centroid_are_both_recorded(tmp_path):
    """The box centre is what earlier runs measured at; the centroid is where this one did."""
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation(xyxy=(10, 20, 110, 120), centroid=(42, 99))], CLASS_NAMES)

    row = read_log(tmp_path / "objects.csv")[0]
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == ("10", "20", "110", "120")
    assert (row["cx"], row["cy"]) == ("42", "99")
    assert (row["cx_box"], row["cy_box"]) == ("60", "70")


def test_the_two_logs_share_a_clock_so_elapsed_s_is_joinable(tmp_path):
    """Two independent zeros would make the files impossible to line up in time."""
    ticks = iter([100.0, 100.5, 101.0, 101.5, 102.0])
    clock = lambda: next(ticks)      # noqa: E731

    frames = RunLogger(tmp_path / "run.csv", "yolo26", "custom", clock=clock)
    objects = DetectionLogger(tmp_path / "objects.csv", clock=clock, start=frames.start)
    assert objects.start == frames.start == 100.0

    frames.log(1, 0.02, 0.05)                           # elapsed 0.5
    objects.log(0, [FakeObservation()], CLASS_NAMES)     # elapsed 1.0
    frames.close()
    objects.close()

    assert read_log(tmp_path / "run.csv")[0]["elapsed_s"] == "0.5"
    assert read_log(tmp_path / "objects.csv")[0]["elapsed_s"] == "1.0"


def test_a_per_object_log_without_a_run_logger_starts_its_own_clock(tmp_path):
    logger = DetectionLogger(tmp_path / "objects.csv")
    assert logger.start is not None
    logger.close()


def test_the_per_object_log_creates_its_parent(tmp_path):
    logger = DetectionLogger(tmp_path / "deeper" / "still" / "objects.csv")
    logger.log(0, [FakeObservation()], CLASS_NAMES)
    logger.close()
    assert (tmp_path / "deeper" / "still" / "objects.csv").exists()


def test_rows_are_flushed_so_a_killed_run_leaves_a_usable_file(tmp_path):
    logger = DetectionLogger(tmp_path / "objects.csv")
    logger.log(0, [FakeObservation(), FakeObservation(object_id=1)], CLASS_NAMES)
    # No close(): the file must already be readable.
    assert len(read_log(tmp_path / "objects.csv")) == 2
    logger.close()


# =============================================================================
# TELLING THE TWO LOGS APART
# =============================================================================


def test_a_per_object_log_is_recognised(tmp_path):
    with DetectionLogger(tmp_path / "objects.csv") as logger:
        logger.log(0, [FakeObservation()], CLASS_NAMES)
    assert is_detection_log(read_log(tmp_path / "objects.csv"))


def test_a_per_frame_log_is_not_mistaken_for_one(tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=CLASS_NAMES) as logger:
        logger.log(1, 0.02, 0.05, per_class={0: 1})
    assert not is_detection_log(read_log(tmp_path / "run.csv"))


def test_an_empty_log_is_not_a_per_object_log():
    assert not is_detection_log([])


def test_two_class_names_that_slug_alike_do_not_share_a_column():
    """A duplicated column name would make the row dict lose one class's count silently.

    Unreachable with this project's four classes, but class_names comes from whatever
    checkpoint was loaded.
    """
    assert class_columns({0: "plastic bag", 1: "plastic-bag"}) == ["n_plastic_bag", "n_cls1"]


def test_a_collision_still_records_both_counts(tmp_path):
    names = {0: "plastic bag", 1: "plastic-bag"}
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=names) as logger:
        logger.log(3, 0.02, 0.05, per_class={0: 1, 1: 2})

    row = read_log(tmp_path / "run.csv")[0]
    assert row["n_plastic_bag"] == "1"
    assert row["n_cls1"] == "2"
    assert int(row["n_plastic_bag"]) + int(row["n_cls1"]) == int(row["n_detections"])


def test_the_header_and_the_rows_cannot_disagree_about_the_columns(tmp_path):
    names = {0: "glass, clear", 1: "glass clear", 2: "metal"}
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", class_names=names) as logger:
        logger.log(0, 0.02, 0.05, per_class={})
    header = (tmp_path / "run.csv").read_text().splitlines()[0].split(",")
    assert len(header) == len(set(header)), f"a column name is duplicated: {header}"
    assert header[len(CSV_COLUMNS):] == logger.class_columns
