"""The report's per-class count figures: one panel per class, the four models overlaid.

`latency_figures` publishes how *fast* the models were. This publishes what they *counted*, which
is the other half of the comparison and the half a reader can check against the scene: the three
rosbag clips hold a known, fixed set of objects, recorded in `runs/camera/ground_thruth.txt`, so
"did the model see the right number of things, and did it keep seeing them for the whole clip" is
answerable by counting rows in the per-object logs and drawing a constant next to them.

One figure per (variant, scene) -- six in all -- as a 2x2 grid of class panels. Each panel carries
one line per model, with the scene's ground-truth count for that class in the panel title. X is
time into the clip, y is the number of detections in that frame.

Per model there are two series: the raw per-frame count, thin and faint, and a bold rolling mean
over it. That is how latency_figures draws its panels and for the same reason -- the frame-to-frame
behaviour is part of what is being judged, but four raw traces on one panel are unreadable without
the means drawn over them. The smoothing is benchmark.rolling_mean at plot_run's window, so every
smoothed series in the report is smoothed identically.

Like `latency_figures` and `benchmark`, this module touches no camera, no checkpoint and no rosbag:
it works on the CSVs, so the figures rebuild on any machine.

`runs/camera` is laid out as

    runs/camera/<model>/<variant>/<scene>/<date>-<time>-<model>-<dataset>-objects.csv

-- the same tree `latency_figures` reads, whose per-*frame* siblings are a different, earlier run.
The folder -> report-name mapping, the scene order and the misfiling check are imported from there
rather than restated, so the two families of figure in one report cannot disagree about which run
belongs to which model.

    python camera/count_figures.py            # write the six figures
    python camera/count_figures.py --table    # ... and print the per-class statistics
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.benchmark import (is_detection_log, read_log,  # noqa: E402
                              rolling_mean)
from camera.latency_figures import (CSV_MODEL_ALIASES,  # noqa: E402
                                    FOLDER_DATASETS, FOLDER_MODELS, NOT_RUNS,
                                    SCENE_ORDER, labels_for, scene_label)

RUN_DIR = "runs/camera"
OUT_DIR = "plots_for_text/counts"
GROUND_TRUTH = "runs/camera/ground_thruth.txt"

# The dataset's own class order -- glass 0, metal 1, paper 2, plastic 3 -- which is the order the
# panels appear in, reading across then down. Deliberately *not* the order the ground-truth file
# lists its columns in; see read_ground_truth.
CLASS_ORDER = ["glass", "metal", "paper", "plastic"]

# Legend order, and the order lines are drawn in. The two YOLOs adjacent so they can be read
# against each other, then the two non-YOLO architectures.
MODEL_ORDER = ["yolo8", "yolo26", "maskrcnn", "rfdetr"]

# Keyed on the report names FOLDER_MODELS produces, so a model is the same colour in all six
# figures and in any figure added later.
MODEL_COLOURS = {
    "yolo8": "tab:blue",
    "yolo26": "tab:orange",
    "maskrcnn": "tab:green",
    "rfdetr": "tab:purple",
}

# Counts are small integers, so two models that agree draw the same line and the one plotted second
# hides the first completely. Distinct dash patterns keep both readable where they coincide --
# colour alone is not enough when the lines are identical rather than merely close.
MODEL_DASHES = {
    "yolo8": (None, None),         # solid
    "yolo26": (6, 2),
    "maskrcnn": (2, 2),
    "rfdetr": (6, 2, 1, 2),
}

# The row labels in ground_thruth.txt -> the scene names SCENE_ORDER uses. An explicit table for
# the same reason FOLDER_MODELS is one: the file is hand-written prose ("Medium Clutter"), and a
# slug function that silently failed to match would publish a panel with no reference line, which
# looks like a finished figure.
GROUND_TRUTH_SCENES = {
    "medium clutter": "clutter_med",
    "high clutter": "clutter_high2",
    "stacked objects": "stacked",
}


@dataclass
class CountRun:
    """One per-object log, reduced to a count per class per frame."""
    csv_path: Path
    model: str
    dataset: str
    scene: str
    frames: np.ndarray               # 0 .. max(frame), contiguous, including frames with no rows
    video_s: np.ndarray              # seconds since frame 0 of this run
    counts: dict[str, np.ndarray]    # class name -> count per frame, one entry per CLASS_ORDER

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def read_ground_truth(path=GROUND_TRUTH) -> dict[str, dict[str, int]]:
    """{scene: {class: count}} from runs/camera/ground_thruth.txt.

    The file is hand-maintained and its shape invites two mistakes, so both are handled explicitly
    rather than assumed away:

    - **Its columns are `paper, glass, plastic, metal`, which is not the class-index order** the
      rest of the project uses. Reading them positionally against CLASS_ORDER gives a wrong but
      entirely plausible number in every panel of every figure, so the columns are bound by their
      header names.
    - The rows are `Medium Clutter 3,0,1,1,` -- the label and its *first* value are separated by
      whitespace, not a comma, and every line carries a trailing one.

    An unknown row label or class name raises. A ground truth that quietly failed to parse is the
    one error these figures cannot show: the panel still draws, just without the line a reader
    would be comparing against.
    """
    path = Path(path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty")

    def fields(line: str) -> list[str]:
        return [part.strip() for part in line.split(",") if part.strip()]

    header = fields(lines[0])
    if not header or not header[0].lower().startswith("class"):
        raise ValueError(
            f"{path}: expected the first line to be the class header, got {lines[0]!r}")
    # `Class paper` -> `paper`: the word `Class` labels the column of row names, and the first
    # class name shares the field with it exactly as each row's first count shares a field with
    # its scene label.
    _, _, first_column = header[0].partition(" ")
    columns = [c.strip().lower() for c in [first_column, *header[1:]]]
    if sorted(columns) != sorted(CLASS_ORDER):
        raise ValueError(
            f"{path}: header names {columns} are not the four classes {sorted(CLASS_ORDER)}")

    truth: dict[str, dict[str, int]] = {}
    for line in lines[1:]:
        parts = fields(line)
        if len(parts) != len(columns):
            raise ValueError(
                f"{path}: {line!r} has {len(parts)} fields, expected {len(columns)}")
        # `Medium Clutter 3` -> ("Medium Clutter", "3").
        label, _, first = parts[0].rpartition(" ")
        scene = GROUND_TRUTH_SCENES.get(label.strip().lower())
        if scene is None:
            raise KeyError(
                f"{path}: unknown scene row {label.strip()!r}. Add it to GROUND_TRUTH_SCENES "
                f"(known: {sorted(GROUND_TRUTH_SCENES)})."
            )
        values = [first] + parts[1:]
        truth[scene] = {cls: int(value) for cls, value in zip(columns, values)}

    missing = [s for s in SCENE_ORDER if s not in truth]
    if missing:
        raise ValueError(f"{path}: no ground truth for {missing}")
    return truth


def check_model_filename(csv_path: Path, model: str) -> None:
    """Raise if the log's filename disagrees with the model folder it was found in.

    The counterpart of `latency_figures.check_model_column`, and it has to be a different check:
    DETECTION_COLUMNS carries no `model` column, so that function is a silent no-op on a per-object
    log -- it would pass a Mask R-CNN run filed under yolo8 and publish its counts as yolo8's,
    which is exactly the failure it exists to catch. The `runs/camera/<model>/<variant>/<scene>/`
    tree is assembled by hand (segmentation_realsense writes `runs/camera/<bag>/<stem>/`), so a
    misfiled log is a real possibility rather than a theoretical one.

    The stem is `<date>-<time>-<model>-<dataset>-objects`, so the model is the second-last field
    once the suffix is off. `benchmark.labels_from` is the wrong tool here despite doing almost
    this: its split would read the `-objects` suffix as the dataset and `custom` as the model.

    Silent when the stem does not parse or names something outside the alias table -- a
    hand-renamed file still reads, which is the same latitude check_model_column allows a log that
    predates its column.
    """
    stem = csv_path.stem
    if stem.endswith("-objects"):
        stem = stem[:-len("-objects")]
    parts = stem.split("-")
    if len(parts) < 2:
        return
    expected = CSV_MODEL_ALIASES.get(parts[-2].strip().lower())
    if expected is not None and expected != model:
        raise ValueError(
            f"{csv_path}: filename says model={parts[-2]!r} ({expected}) but it is filed under "
            f"{model!r}. One of the two is wrong -- a misfiled log would publish one "
            f"architecture's counts under another's name."
        )


def count_series(rows, frames: np.ndarray) -> dict[str, np.ndarray]:
    """{class: count per frame} over `frames`, zero where the class has no row in that frame.

    The zero-filling is the whole point. A per-object log has a row per *detection*, so a class the
    model did not see in a frame contributes nothing at all -- and a frame in which it saw nothing
    of any class is absent from the file entirely. Plotting the rows as they come would draw a line
    that steps straight over a dropout instead of dipping to zero, which is the opposite of what
    these figures are for.

    Every class in CLASS_ORDER gets a full-length series even if it never appears, so a model that
    detected no glass all clip draws a visible flat zero rather than vanishing from the panel.
    """
    index = {frame: i for i, frame in enumerate(frames.tolist())}
    counts = {cls: np.zeros(len(frames), dtype=int) for cls in CLASS_ORDER}
    for row in rows:
        name = (row.get("class_name") or "").strip()
        series = counts.get(name)
        if series is None:
            raise KeyError(
                f"class_name {name!r} is not one of {CLASS_ORDER}. The four-class dataset is "
                f"what every figure in the report is keyed on, so a fifth class is a dataset "
                f"change, not a plotting detail."
            )
        series[index[int(row["frame"])]] += 1
    return counts


def frame_times(rows, frames: np.ndarray) -> np.ndarray:
    """Seconds since frame 0, for every frame in `frames`.

    Two things to get right:

    - `elapsed_s` counts from the start of the *run*, and absorbs the warm-up frame -- frame 0 sits
      at 0.42 s in the Mask R-CNN logs and 3.34 s in the yolo26 ones. Plotted raw, four models that
      replayed the identical clip would be spread three seconds apart on one panel. Re-zeroing on
      frame 0 is what makes the x-axis mean "time into the clip", which is what the panel claims.
    - A frame absent from the log has no `elapsed_s` of its own, so its time is interpolated from
      the frames either side. np.interp rather than frame/stream_fps: the logs are a replay at
      whatever rate the model managed, so the nominal rate is not the time axis.
    """
    seen: dict[int, float] = {}
    for row in rows:
        seen.setdefault(int(row["frame"]), float(row["elapsed_s"]))
    observed = np.asarray(sorted(seen), dtype=float)
    elapsed = np.asarray([seen[int(f)] for f in observed], dtype=float)
    times = np.interp(frames.astype(float), observed, elapsed)
    return times - times[0]


def load_counts(csv_path: Path) -> CountRun | str:
    """One CountRun from a per-object log, or a string saying why there is nothing plottable.

    The reason comes back rather than a bare None, as in `latency_figures.load_run`: a model
    quietly missing from a panel is a figure with three lines where four were expected and nothing
    pointing at the cause.
    """
    rows = read_log(csv_path)
    if not rows:
        return "no objects logged"
    # The mirror of the mistake is_detection_log was written to catch. The per-frame log has no
    # class_name, so without this the failure would surface as a KeyError several frames deep.
    if not is_detection_log(rows):
        return "per-frame log, no class_name column"

    frame_numbers = [int(r["frame"]) for r in rows]
    # 0 .. max, not the frames present: see count_series on why the gaps have to be on the axis.
    frames = np.arange(0, max(frame_numbers) + 1)
    model, dataset, scene = labels_for(csv_path)
    check_model_filename(csv_path, model)

    # No plotted_rows/WARMUP_FRAMES here, deliberately, and it is the one place this module parts
    # company with latency_figures. The warm-up frame's *latency* is meaningless and is excluded
    # there; its *count* is a real count of a real frame, and dropping it would make one model's
    # line start a frame later than the others on a shared axis.
    return CountRun(
        csv_path=csv_path,
        model=model,
        dataset=dataset,
        scene=scene,
        frames=frames,
        video_s=frame_times(rows, frames),
        counts=count_series(rows, frames),
    )


def discover_counts(run_dir=RUN_DIR) -> tuple[list[CountRun], list[str]]:
    """Every per-object log under runs/camera, plus a line per file that was skipped.

    Sorted variant, then scene difficulty, then MODEL_ORDER -- the order the figures, their legends
    and the table share.
    """
    runs, skipped = [], []
    for csv_path in sorted(Path(run_dir).glob("*/*/*/*-objects.csv")):
        if csv_path.parts[-4] in NOT_RUNS:
            continue
        run = load_counts(csv_path)
        if isinstance(run, CountRun):
            runs.append(run)
        else:
            skipped.append(f"{csv_path}: {run}")

    def key(run: CountRun):
        return (run.dataset, SCENE_ORDER.index(run.scene), MODEL_ORDER.index(run.model))

    return sorted(runs, key=key), skipped


def group_counts(runs) -> dict[tuple[str, str], list[CountRun]]:
    """Runs keyed by (dataset, scene) -- one figure each. Insertion order is discover_counts's."""
    grouped: dict[tuple[str, str], list[CountRun]] = {}
    for run in runs:
        grouped.setdefault((run.dataset, run.scene), []).append(run)
    return grouped


def smoothing_window(n_frames: int) -> int:
    """How many frames the rolling mean averages over.

    One tenth of the run, capped at 15 frames -- plot_run's window, which latency_figures also
    adopts so that every smoothed series in the report is smoothed the same way. At ~142 frames
    that is 14 frames, a little under a second of clip.
    """
    return min(15, max(1, n_frames // 10))


def plot_counts(runs, ground_truth, png_path) -> Path:
    """The figure for one variant/scene: four class panels, four model lines each.

    Per model a thin raw trace plus a bold rolling mean, the way latency_figures draws its panels.
    Both are needed: the mean is the only thing readable when four models are near each other at
    small integer counts, but a class dropping out for a single frame is what these figures exist
    to show and a mean alone turns it into a dip of a tenth of an object.

    The ground truth is in each panel's title rather than drawn across it. As a line it sat at the
    height a correct model's series sits at, so it ran along the data through most of every panel.

    Agg is forced before pyplot is imported, as everywhere else in the project: this is run
    headless as often as not.
    """
    runs = list(runs)
    png_path = Path(png_path)
    dataset, scene = runs[0].dataset, runs[0].scene
    truth = ground_truth[scene]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)

    # Shared across the figure so the four panels can be read against each other: a count of 5 has
    # to be the same height in the paper panel as in the glass one.
    ceiling = max([int(run.counts[c].max()) for run in runs for c in CLASS_ORDER]
                  + list(truth.values()))
    window = smoothing_window(max(run.n_frames for run in runs))

    handles: dict[str, object] = {}
    for ax, cls in zip(axes.flat, CLASS_ORDER):
        for run in runs:
            colour = MODEL_COLOURS.get(run.model, "tab:gray")
            dashes = MODEL_DASHES.get(run.model, (None, None))
            series = run.counts[cls].astype(float)
            # On a run too short to smooth, the rolling mean *is* the raw series: drawing both
            # would put a bold line exactly over a faint one and call it an average in the legend.
            if window > 1:
                ax.plot(run.video_s, series, drawstyle="steps-post", linewidth=0.7, alpha=0.25,
                        color=colour, zorder=1.5)
            line, = ax.plot(
                run.video_s, rolling_mean(series, window),
                linewidth=1.8,
                color=colour,
                dashes=dashes if dashes[0] is not None else (None, None),
                zorder=2.5,
                label=run.model,
            )
            handles.setdefault(run.model, line)

        ax.set_title(f"{cls} -- ground truth {truth[cls]}", fontsize=10)
        ax.set_ylim(-0.3, ceiling + 1)          # -0.3 so a flat zero series is not on the axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linewidth=0.3, alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("time into the clip (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("detections in frame")

    order = [m for m in MODEL_ORDER if m in handles]
    fig.legend([handles[k] for k in order], order, loc="lower center",
               ncol=len(order), fontsize=9, frameon=False)

    frames = ", ".join(f"{run.model} {run.n_frames}f" for run in runs)
    smoothing = f", rolling mean over {window} frames" if window > 1 else ""
    fig.suptitle(f"detections per frame -- {dataset} / {scene}   ({frames}{smoothing})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def modal_count(series: np.ndarray) -> int:
    """The most common per-frame count, ties going to the smaller value for a stable table."""
    counts = Counter(series.tolist())
    return min(c for c, n in counts.items() if n == max(counts.values()))


def table_rows(runs, ground_truth) -> list[list[str]]:
    """The per-class count table: a header plus a row per run per class, in discover_counts's order.

    `exact_frac` -- the share of frames in which the model counted exactly the right number -- is
    the one number the figure cannot be read off precisely, and the one a write-up wants.
    """
    rows = [["dataset", "scene", "model", "class", "frames", "mean", "mode", "truth",
             "exact_frac"]]
    for run in runs:
        truth = ground_truth[run.scene]
        for cls in CLASS_ORDER:
            series = run.counts[cls]
            rows.append([
                run.dataset, run.scene, run.model, cls, str(run.n_frames),
                f"{series.mean():.2f}",
                str(modal_count(series)),
                str(truth[cls]),
                f"{float((series == truth[cls]).mean()):.3f}",
            ])
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=RUN_DIR, help="where the run folders live")
    parser.add_argument("--out-dir", default=OUT_DIR, help="where the figures go")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH,
                        help="the per-scene object inventory")
    parser.add_argument("--table", action="store_true",
                        help="also print the per-class count table as CSV")
    args = parser.parse_args(argv)

    ground_truth = read_ground_truth(args.ground_truth)
    runs, skipped = discover_counts(args.run_dir)
    for line in skipped:
        print(f"skipped {line}", file=sys.stderr)
    if not runs:
        print(f"no per-object logs under {args.run_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    for (dataset, scene), group in group_counts(runs).items():
        png = plot_counts(group, ground_truth, out_dir / f"counts-{dataset}-{scene}.png")
        models = ", ".join(r.model for r in group)
        print(f"{png}  ({models})")

    if args.table:
        print()
        for row in table_rows(runs, ground_truth):
            print(",".join(row))

    note = f"\n{len(runs)} runs, raw per-frame counts, warm-up frame included"
    if skipped:
        note += f", {len(skipped)} file(s) skipped"
    print(note, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
