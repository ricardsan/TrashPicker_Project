"""Per-frame speed logging for a live run, and the figure made from it.

Deliberately free of any camera, torch or ultralytics import: everything here works on a CSV, so
the figure can be rebuilt long after the run -- from a different machine, with no RealSense
plugged in -- and so the tests need none of that stack.

The CSV is the primary artefact. One row per frame:

    frame,timestamp,elapsed_s,n_detections,inference_ms,loop_ms,fps,model_fps,model,dataset,
    stream_fps,extras[,n_<class> for each class]

There are two speeds here and they answer different questions:

    inference_ms / model_fps   the model call alone -- what the architecture costs, and the only
                               number that distinguishes a fast model from a faster one.
    loop_ms / fps              the whole loop period, top to top. Bounded below by the frame
                               source: the RealSense streams at 30 Hz and the loop blocks
                               waiting for it, so any model quicker than ~33 ms reports ~30 FPS
                               here however quick it really is. It is the rate the pipeline
                               actually achieved, not the rate the model could sustain.

Quoting only the second would make every fast model look identical, so plot_run draws both.

`fps` and `model_fps` are derived from the two *_ms columns at write time rather than measured
separately, so no two columns in a row can disagree. `model`, `dataset` and `stream_fps` repeat
on every row, which is redundant but makes the CSV self-describing: plot_run then needs nothing
but the file. `stream_fps` is what caps `fps`, and it is per-run rather than a constant because a
replayed rosbag arrives at whatever rate it was recorded at -- 15 Hz for the clips in rosbags/.
Logs written before that column existed fall back to CAMERA_FPS, which is what they were recorded
at. Every row is flushed, so a run killed with Ctrl-C still leaves a usable log.

`extras` is the same idea applied to the last thing that changes what `loop_ms` means: which of
segmentation_realsense's optional per-frame work was switched on. Empty for the default fast path,
`objects` when the per-object measuring was on, `video` when a recording was being written, or
`objects+video`. Two runs are comparable on `loop_ms` when their `extras` agree; `inference_ms` is
unaffected either way. Reading it off the file beats having to remember which flags a run used.

The `n_<class>` columns are optional and appear only when RunLogger is told the class names. They
break out `n_detections` per class, in class-index order, so the log can answer "how much plastic
did it see" and not only "how many objects". Columns are append-only -- these go on the end, and a
log predating them still reads.

Two CSVs come out of a run. This module writes both: RunLogger for the per-frame one above, and
DetectionLogger for the per-object one described on that class.
"""
from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

CSV_COLUMNS = [
    "frame",
    "timestamp",
    "elapsed_s",
    "n_detections",
    "inference_ms",
    "loop_ms",
    "fps",
    "model_fps",
    "model",
    "dataset",
    "stream_fps",
    "extras",
]

# What the RealSense streams at, drawn on the FPS panel so the ceiling the achieved rate runs
# into is visible rather than mistaken for a property of the model. It is the *default* only:
# the rate is per-run and travels in the CSV's stream_fps column, because a replayed rosbag
# streams at whatever it was recorded at -- 15 Hz for the clips in rosbags/ -- and a cap line
# drawn at 30 there would be simply wrong.
CAMERA_FPS = 30.0

# The first model call pays for the CUDA context, cuDNN's autotuning and any lazy kernel
# compilation -- routinely 20x the steady-state latency. It is real, but it is not the number a
# report means by "inference speed", so the summary drops it and says so in the legend.
WARMUP_FRAMES = 1


def run_log_name(model_label: str, dataset_label: str, when: datetime | None = None) -> str:
    """The stem shared by a run's .csv and .png: date-time-model-dataset.

    Seconds are included because two runs of the same model a minute apart are a normal thing to
    do when benchmarking, and they must not overwrite each other.
    """
    when = when or datetime.now()
    return f"{when.strftime('%Y-%m-%d-%H%M%S')}-{model_label}-{dataset_label}"


def class_column(name: str) -> str:
    """The per-class count column for a class name: "glass" -> "n_glass".

    Slugged because the name comes from a checkpoint's own `names` dict, and a comma or a space in
    one would corrupt the header rather than just reading oddly.
    """
    slug = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
    return f"n_{slug or 'unnamed'}"


def class_columns(class_names) -> list[str]:
    """The per-class count columns for a {class index: name} mapping, in class-index order.

    Collisions are broken by falling back to the class index. Two names that slug to the same
    thing -- "plastic bag" and "plastic-bag" -- would otherwise produce a duplicated column name,
    and since the row is built as a dict the second class's count would overwrite the first's,
    losing it with nothing to say so. Unreachable with this project's four classes, but the names
    come out of whatever checkpoint was loaded.
    """
    if not class_names:
        return []

    columns, seen = [], set()
    for index in sorted(class_names):
        column = class_column(class_names[index])
        if column in seen:
            column = f"n_cls{index}"
        columns.append(column)
        seen.add(column)
    return columns


class RunLogger:
    """Writes one CSV row per frame, and keeps them in memory for the end-of-run summary."""

    def __init__(self, path, model_label: str, dataset_label: str, clock=time.perf_counter,
                 stream_fps: float = CAMERA_FPS, class_names=None, extras: str = ""):
        self.path = Path(path)
        self.model_label = model_label
        self.dataset_label = dataset_label
        self.stream_fps = float(stream_fps)
        self.extras = extras
        # {class index: name}, or None for a log that does not break the count out per class.
        self.class_names = dict(class_names) if class_names else None
        self.class_columns = class_columns(self.class_names)
        # The mapping log() writes through, so a collision resolved by class_columns is resolved
        # once rather than recomputed -- and the header and the rows cannot disagree.
        self.column_of_class = dict(zip(sorted(self.class_names or {}), self.class_columns))
        self.columns = CSV_COLUMNS + self.class_columns
        self.rows: list[dict] = []
        # Injectable so the tests can drive elapsed_s deterministically instead of sleeping.
        self._clock = clock
        # Public, so the per-object log can share this zero and the two CSVs' elapsed_s agree.
        self.start = clock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
        self._writer.writeheader()
        self._file.flush()

    def log(self, n_detections: int, inference_s: float, loop_s: float, per_class=None) -> dict:
        """Append one frame. Returns the row, mostly so callers can print it.

        `loop_s` is the whole loop period, `inference_s` the model call inside it. `per_class` is
        a {class index: count} mapping -- a collections.Counter over the frame's detections will
        do -- and is ignored unless the logger was given class names.
        """
        row = {
            "frame": len(self.rows),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(self._clock() - self.start, 4),
            "n_detections": int(n_detections),
            "inference_ms": round(inference_s * 1000, 3),
            "loop_ms": round(loop_s * 1000, 3),
            # A frame that somehow took no measurable time would divide by zero; report 0 rather
            # than inf, which would wreck the y-axis of the FPS panel.
            "fps": round(1.0 / loop_s, 3) if loop_s > 0 else 0.0,
            "model_fps": round(1.0 / inference_s, 3) if inference_s > 0 else 0.0,
            "model": self.model_label,
            "dataset": self.dataset_label,
            "stream_fps": self.stream_fps,
            "extras": self.extras,
        }
        if self.class_names is not None:
            counts = per_class or {}
            for index, column in self.column_of_class.items():
                row[column] = int(counts.get(index, 0))
        self.rows.append(row)
        self._writer.writerow(row)
        self._file.flush()
        return row

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


DETECTION_COLUMNS = [
    "frame",            # the same frame number as the per-frame log, which is how the two join
    "timestamp",
    "elapsed_s",
    "object_id",        # follows one object across frames; see camera/tracking.py
    "cls",
    "class_name",
    "conf",
    "cx", "cy",         # the pixel the position was deprojected from, in frame coordinates
    "cx_box", "cy_box",  # the box centre, which is what every run before this measured at
    "x1", "y1", "x2", "y2",
    "X_m", "Y_m", "Z_m",
    "depth_source",     # mask-median | centre-pixel | none
    "mask_px",          # the mask's area in pixels, 0 for a detection with no mask
    "depth_px",         # how many of those pixels had a depth reading
    # The mask's principal axis, in degrees: 0 is parallel to the bottom of the frame and the
    # angle grows anticlockwise as seen on screen, wrapped into [0, 180) because an axis is a
    # line and not an arrow. Blank when mask_px is 0 or 1. Appended last, as every new column
    # must be -- see the module docstring on logs predating a column.
    "orientation_deg",
]


class DetectionLogger:
    """Writes one CSV row per detection per frame: what was seen, where, and how far away.

    The companion to RunLogger. That one answers "how fast was the model and how much did it see";
    this one answers "what were the individual objects" -- which is what a report needs to say
    anything about the scene rather than about the model.

    **Z_m is depth along the camera's optical axis, not height above the table.** It is the Z of
    rs2_deproject_pixel_to_point in the camera's own frame, so it grows as an object moves *away*
    from the lens, and it says nothing about how tall the object is. Converting it to a height
    would need the camera-to-base transform, which lives in camera/calibrate_T_base_cam.py and is
    not applied here.

    **An empty Z_m means there was no depth reading, which is not the same as a reading of zero.**
    A Z16 depth frame uses 0 for "nothing came back", and that is the normal case on exactly the
    materials this project sorts -- glass, clear plastic, anything specular. Writing 0.0 for those
    would put a confident wrong number next to the real ones; the field is left blank instead and
    `depth_source` says `none`. `depth_px` tells you how much of the mask did return something,
    so a reading taken from three pixels is distinguishable from one taken from three thousand.

    **An empty orientation_deg means the same kind of thing: there was no mask to take an axis
    over.** A run with --centre-depth never looks at the masks, and a box-only checkpoint has none
    to look at, so the column is blank wherever mask_px is 0 -- and also where it is 1, since one
    pixel has no direction. A 0.0 there would read as "lying horizontally", which is a claim about
    the object rather than about the absence of one.

    Rows are flushed once per frame rather than once per row: a frame with a dozen detections
    would otherwise pay a dozen flushes inside the loop.
    """

    def __init__(self, path, clock=time.perf_counter, start: float | None = None):
        self.path = Path(path)
        self.rows_written = 0
        self._clock = clock
        # Shared with the per-frame log where there is one, so elapsed_s means the same thing in
        # both files and they can be joined on more than the frame number.
        self.start = clock() if start is None else start
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=DETECTION_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, frame: int, observations, names=None) -> int:
        """Append this frame's observations. Returns how many rows were written.

        `observations` duck-type segmentation_realsense.Observation: the attributes read are
        object_id, cls, conf, xyxy, box_centre, centroid, xyz, depth_source, mask_px, depth_px
        and orientation_deg. Nothing is imported from that module -- this one stays free of cv2,
        torch and ultralytics so the CSV tooling runs anywhere. The orientation arrives as a
        plain float for that reason: the PCA that produced it is measure()'s job, not this one's.
        """
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        elapsed = round(self._clock() - self.start, 4)
        names = names or {}

        for observation in observations:
            x1, y1, x2, y2 = observation.xyxy
            cx, cy = observation.centroid
            cx_box, cy_box = observation.box_centre
            xyz = observation.xyz
            orientation = observation.orientation_deg
            self._writer.writerow({
                "frame": frame,
                "timestamp": timestamp,
                "elapsed_s": elapsed,
                "object_id": observation.object_id,
                "cls": int(observation.cls),
                "class_name": names.get(observation.cls, str(observation.cls)),
                "conf": round(float(observation.conf), 4),
                "cx": int(round(cx)), "cy": int(round(cy)),
                "cx_box": int(cx_box), "cy_box": int(cy_box),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                # Blank, not 0.0, when there was nothing to measure -- see the class docstring.
                "X_m": "" if xyz is None else round(float(xyz[0]), 4),
                "Y_m": "" if xyz is None else round(float(xyz[1]), 4),
                "Z_m": "" if xyz is None else round(float(xyz[2]), 4),
                "depth_source": observation.depth_source,
                "mask_px": int(observation.mask_px),
                "depth_px": int(observation.depth_px),
                # Blank, not 0.0, when there was no mask to take an axis over -- see the class
                # docstring. 0.0 is a real value here: it means "lying horizontally".
                #
                # The second modulo is not redundant with the one the angle already went
                # through: rounding re-opens the range. 179.996 is inside [0, 180) and rounds to
                # 180.0, which is not, and a reader binning this column into 0..179 would get a
                # bucket that should not exist. 180 and 0 are the same orientation anyway.
                "orientation_deg": ("" if orientation is None
                                    else round(float(orientation), 2) % 180.0),
            })
            self.rows_written += 1

        self._file.flush()
        return len(observations)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class Summary:
    """What the figure legend and the end-of-run printout quote."""

    frames: int          # rows in the log, warm-up included
    measured: int        # rows the statistics below are computed from
    mean_ms: float
    median_ms: float
    p95_ms: float
    mean_fps: float          # achieved loop rate -- camera-capped
    mean_model_fps: float    # what the model alone sustains
    wall_s: float            # over the measured frames, matching `measured`
    mean_detections: float
    # The rate the frames arrived at, which caps mean_fps. Defaulted so a Summary built by
    # hand -- or from a log written before the column existed -- still works.
    stream_fps: float = CAMERA_FPS
    # {class name: mean detections per frame}, empty for a log with no per-class columns.
    mean_per_class: dict[str, float] = field(default_factory=dict)

    def as_lines(self) -> list[str]:
        detections = f"detections: {self.mean_detections:.2f} per frame"
        if self.mean_per_class:
            breakdown = ", ".join(f"{name} {mean:.2f}"
                                  for name, mean in self.mean_per_class.items())
            detections += f" -- {breakdown}"
        return [
            f"frames: {self.measured} measured over {self.wall_s:.1f} s "
            f"({self.frames} logged, including the warm-up)",
            f"inference: mean {self.mean_ms:.1f} ms, median {self.median_ms:.1f} ms, "
            f"p95 {self.p95_ms:.1f} ms",
            f"throughput: {self.mean_model_fps:.1f} FPS from the model, "
            f"{self.mean_fps:.1f} FPS achieved (the {self.stream_fps:.0f} Hz stream caps this)",
            detections,
        ]


def harmonic_fps(durations_ms: np.ndarray) -> float:
    """Frames divided by the time they took.

    NOT the mean of the per-frame rates, which is a different and much larger number: frame times
    of 10 ms and 100 ms average to 55 FPS that way, against a true throughput of 18.2. Rates do
    not average arithmetically -- the long frames have to carry their weight in the total.
    """
    total_s = float(np.sum(durations_ms)) / 1000.0
    return len(durations_ms) / total_s if total_s > 0 else 0.0


def stream_fps_of(rows) -> float:
    """The rate the frames arrived at, off the log's own column.

    Logs written before the column existed -- everything in runs/camera/ up to now -- fall back
    to CAMERA_FPS, which is what they were in fact recorded at.
    """
    if not rows:
        return CAMERA_FPS
    recorded = rows[0].get("stream_fps")
    try:
        value = float(recorded)
    except (TypeError, ValueError):
        return CAMERA_FPS
    return value if value > 0 else CAMERA_FPS


def per_class_columns_of(rows) -> list[str]:
    """The `n_<class>` columns a log actually has, in the order they were written.

    Read off the header rather than from a class list, so the statistics describe whatever the log
    contains -- including a log from a checkpoint with different classes, and a log from before
    these columns existed, which has none.
    """
    if not rows:
        return []
    return [key for key in rows[0]
            if isinstance(key, str) and key.startswith("n_") and key != "n_detections"]


def mean_per_class_of(rows) -> dict[str, float]:
    """{class name: mean detections per frame} over these rows, by the n_<class> columns."""
    means = {}
    for column in per_class_columns_of(rows):
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                continue        # a blank or malformed cell: skip it rather than fail the summary
        if values:
            means[column[len("n_"):]] = float(np.mean(values))
    return means


def summarise(rows) -> Summary | None:
    """Statistics over the log, excluding the warm-up frame. None for an empty log.

    A run of one or two frames has nothing left after the warm-up is removed, so in that case the
    warm-up is kept rather than returning nothing -- a figure of a 2-frame run is not useful, but
    it should not crash the end of a run either.
    """
    rows = list(rows)
    if not rows:
        return None

    measured = rows[WARMUP_FRAMES:] or rows
    latency = np.asarray([float(r["inference_ms"]) for r in measured], dtype=float)
    loop = np.asarray([float(r["loop_ms"]) for r in measured], dtype=float)
    detections = np.asarray([float(r["n_detections"]) for r in measured], dtype=float)

    # Span the same frames the rest of the statistics describe. Measuring from frame 0 would fold
    # a multi-second warm-up into a wall clock quoted next to a warm-up-free frame count, and the
    # two would contradict each other.
    wall_s = float(measured[-1]["elapsed_s"]) - float(measured[0]["elapsed_s"]) \
        + float(measured[0]["loop_ms"]) / 1000.0

    return Summary(
        stream_fps=stream_fps_of(rows),
        mean_per_class=mean_per_class_of(measured),
        frames=len(rows),
        measured=len(measured),
        mean_ms=float(latency.mean()),
        median_ms=float(np.median(latency)),
        p95_ms=float(np.percentile(latency, 95)),
        mean_fps=harmonic_fps(loop),
        mean_model_fps=harmonic_fps(latency),
        wall_s=wall_s,
        mean_detections=float(detections.mean()),
    )


def read_log(csv_path) -> list[dict]:
    """The rows of a run log, as written by RunLogger or by DetectionLogger."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_detection_log(rows) -> bool:
    """True for a per-object log rather than a per-frame one.

    The two share a stem and differ by an `-objects` suffix, so pointing the figure at the wrong
    one is an easy mistake to make. Worth catching: the per-object log has no inference_ms, so
    plot_run would otherwise fail somewhere less informative.
    """
    return bool(rows) and "object_id" in rows[0]


def labels_from(csv_path, rows) -> tuple[str, str]:
    """(model, dataset) for the figure title.

    Preferred source is the columns, which is why they are repeated on every row. The filename is
    the fallback, for a log that was hand-trimmed or produced by an older version.
    """
    if rows and rows[0].get("model") and rows[0].get("dataset"):
        return rows[0]["model"], rows[0]["dataset"]

    # <date>-<time>-<model>-<dataset>: the last two of four dash-separated fields.
    parts = Path(csv_path).stem.split("-")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown", "unknown"


def plotted_rows(rows) -> list:
    """The rows that go on the figure: the same ones the statistics describe.

    The warm-up frame is routinely 10-20x the steady-state latency, so leaving it in costs the
    whole y-axis -- the part a reader cares about gets squashed into the bottom tenth. It is
    annotated on the figure instead, so it is stated rather than hidden.
    """
    rows = list(rows)
    return rows[WARMUP_FRAMES:] if len(rows) > WARMUP_FRAMES else rows


def axis_top(values: np.ndarray) -> float:
    """Upper y-limit: the data plus a little headroom, never zero-height for a flat series."""
    return float(np.max(values)) * 1.15 + 1 if len(values) else 1.0


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, with the window shrinking at the ends.

    np.convolve would be shorter but pads the ends with zeros, which drags the line to the floor
    exactly where a reader looks for the start and end of a run.

    Public, rather than the private it used to be, because latency_figures draws the report's
    overlay figures from the same logs: it has to smooth the way plot_run smooths, or the two
    families of figure in one report would disagree about the same run.
    """
    if window <= 1 or len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    half = window // 2
    for i in range(len(values)):
        out[i] = values[max(0, i - half):i + half + 1].mean()
    return out


def plot_run(csv_path, png_path=None) -> Path:
    """The report figure: latency over time on top, FPS underneath. Returns the PNG path.

    The FPS panel carries both series described in the module docstring -- the model alone and
    the achieved loop rate -- with the camera's stream rate drawn in. Agg is forced because this
    is called both at the end of a live run, where cv2 owns the display, and from --plot-from on
    a machine with no display at all.
    """
    csv_path = Path(csv_path)
    png_path = Path(png_path) if png_path else csv_path.with_suffix(".png")

    rows = read_log(csv_path)
    model_label, dataset_label = labels_from(csv_path, rows)
    summary = summarise(rows)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_ms, ax_fps) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    if summary is None:
        # A header-only log. Still write a figure, so a scripted run does not fail at the last
        # step, but say plainly that there is nothing in it.
        ax_ms.text(0.5, 0.5, "no frames logged", ha="center", va="center",
                   transform=ax_ms.transAxes)
    else:
        shown = plotted_rows(rows)
        elapsed = np.asarray([float(r["elapsed_s"]) for r in shown], dtype=float)
        latency = np.asarray([float(r["inference_ms"]) for r in shown], dtype=float)
        fps = np.asarray([float(r["fps"]) for r in shown], dtype=float)
        model_fps = np.asarray([float(r["model_fps"]) for r in shown], dtype=float)
        # One tenth of the run, capped: long enough to calm the frame-to-frame noise, short
        # enough that a real slowdown is still visible rather than averaged away.
        window = min(15, max(1, len(shown) // 10))

        ax_ms.plot(elapsed, latency, linewidth=0.8, alpha=0.45, color="tab:blue",
                   label="per frame")
        if window > 1:
            ax_ms.plot(elapsed, rolling_mean(latency, window), linewidth=1.8,
                       color="tab:blue", label=f"rolling mean ({window} frames)")
        ax_ms.axhline(summary.mean_ms, linestyle="--", linewidth=1.0, color="tab:red",
                      label=f"mean {summary.mean_ms:.1f} ms")
        ax_ms.axhline(summary.p95_ms, linestyle=":", linewidth=1.0, color="tab:orange",
                      label=f"p95 {summary.p95_ms:.1f} ms")
        ax_ms.set_ylabel("inference latency (ms)")
        ax_ms.set_ylim(0, axis_top(latency))

        if len(rows) > WARMUP_FRAMES:
            warmup_ms = float(rows[0]["inference_ms"])
            ax_ms.annotate(f"warm-up frame: {warmup_ms:.0f} ms (off scale, excluded)",
                           xy=(0.012, 0.9), xycoords="axes fraction", fontsize=8,
                           color="0.35")

        # Two series, because they answer different questions and the second one alone would
        # hide the first: the camera caps the achieved rate at CAMERA_FPS, so every model
        # quicker than 1/CAMERA_FPS would draw the same flat line there.
        ax_fps.plot(elapsed, model_fps, linewidth=0.8, alpha=0.35, color="tab:purple")
        ax_fps.plot(elapsed, rolling_mean(model_fps, window), linewidth=1.8,
                    color="tab:purple",
                    label=f"model alone: {summary.mean_model_fps:.1f} FPS mean")
        ax_fps.plot(elapsed, fps, linewidth=0.8, alpha=0.35, color="tab:green")
        ax_fps.plot(elapsed, rolling_mean(fps, window), linewidth=1.8, color="tab:green",
                    label=f"achieved: {summary.mean_fps:.1f} FPS mean")
        ax_fps.axhline(summary.stream_fps, linestyle="--", linewidth=1.0, color="0.4",
                       label=f"stream: {summary.stream_fps:.0f} Hz")
        ax_fps.set_ylabel("throughput (FPS)")
        ax_fps.set_ylim(0, axis_top(np.concatenate([fps, model_fps, [summary.stream_fps]])))

        for ax in (ax_ms, ax_fps):
            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.legend(fontsize=8, loc="upper right")

    ax_fps.set_xlabel("time since start of run (s)")

    title = f"{model_label} / {dataset_label}"
    if summary is not None:
        title += (f"  --  {summary.measured} frames over {summary.wall_s:.1f} s,  "
                  f"mean {summary.mean_ms:.1f} ms  "
                  f"({summary.mean_model_fps:.1f} FPS model, "
                  f"{summary.mean_fps:.1f} achieved)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path
