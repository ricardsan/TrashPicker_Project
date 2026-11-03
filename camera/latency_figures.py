"""The report's latency figures: one panel per model/dataset, the three rosbag scenes overlaid.

`benchmark.plot_run` draws a two-panel figure per *run* -- latency on top, throughput underneath --
which is the right thing at the end of a run but not what the write-up needs. The write-up wants
latency alone, and it wants one figure per model/dataset pair rather than per run, so the three
scenes a pair was evaluated on can be read against each other instead of across three pages.

So this module re-renders the top panel from the CSVs rather than cropping the existing PNGs. The
CSVs are self-describing (see benchmark's docstring), which is what makes that cheap: nothing here
touches a camera, a checkpoint or a rosbag, and the figures rebuild on any machine.

`runs/camera` is laid out as

    runs/camera/<model>/<variant>/<scene>/<date>-<time>-<model>-<dataset>.csv

where `<variant>` is which fine-tune produced the checkpoint and `<scene>` is which clip from
rosbags/ it was replayed over. The report calls the variant layer the dataset, which is also what
the output filenames are keyed on -- see FOLDER_MODELS / FOLDER_DATASETS for why those names
cannot come out of the CSV.

    python camera/latency_figures.py            # write the eight figures
    python camera/latency_figures.py --table    # ... and print the per-run statistics
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.benchmark import (WARMUP_FRAMES, axis_top,  # noqa: E402
                              is_detection_log, plotted_rows, read_log,
                              rolling_mean, summarise)

RUN_DIR = "runs/camera"
OUT_DIR = "plots_for_text/latency"

# Directories under runs/camera that are not runs. `videos/` holds the rendered clips, one
# subdirectory per scene, so it matches the same */*/* glob the runs do.
NOT_RUNS = {"videos"}

# Folder name -> the name the report uses. Explicit tables rather than a slug function for the
# same reason segmentation_realsense.MODEL_RUNS is explicit: the folders are not consistent enough
# for a formula (`rf-detr` vs the `rfdetr` everything else writes, `taco_finetune` under yolo8
# against `taco` under the other three), and a formula that silently produced `rf-detr-taco.png`
# would mislabel a figure in the report rather than fail. An unmapped folder raises instead.
FOLDER_MODELS = {
    "maskrcnn": "maskrcnn",
    "rf-detr": "rfdetr",
    "yolo8": "yolo8",
    "yolo26": "yolo26",
}
FOLDER_DATASETS = {
    "dataset_seg": "dataset_seg",
    "taco": "taco",
    "taco_finetune": "taco",
}

# What a CSV's own `model` column says, for the folder it should be sitting in. The column cannot
# name the output file -- it says `yolov8` where the report says `yolo8`, and its `dataset` is
# always `custom` -- but it can say whether the file is in the right folder, and that is the check
# worth having: segmentation_realsense writes runs to runs/camera/<bag>/<stem>/, so the
# <model>/<dataset>/<scene>/ tree these are read from was assembled by hand afterwards. A file
# moved into the wrong folder would otherwise be published under the wrong model's name with the
# wrong model's latency, silently and with a plausible-looking number.
CSV_MODEL_ALIASES = {
    "yolov8": "yolo8",
    "yolo8": "yolo8",
    "yolo26": "yolo26",
    "maskrcnn": "maskrcnn",
    "rfdetr": "rfdetr",
}

# The three clips in rosbags/, in the order they should appear in a legend: increasing difficulty
# rather than the alphabetical order the filesystem hands back. A scene outside this table raises
# for the same reason an unknown model folder does -- it is the name that becomes the legend, and
# two unknown scenes would otherwise land on one panel as two indistinguishable grey lines.
SCENE_ORDER = ["clutter_med", "clutter_high2", "stacked"]

SCENE_COLOURS = {
    "clutter_med": "tab:blue",
    "clutter_high2": "tab:orange",
    "stacked": "tab:green",
}


@dataclass
class Run:
    """One run log, with the labels the figure and the table need."""
    csv_path: Path
    model: str
    dataset: str
    scene: str
    elapsed: np.ndarray      # seconds since this run's own first plotted frame
    latency: np.ndarray      # inference_ms, warm-up frame excluded
    mean_ms: float
    sd_ms: float

    @property
    def frames(self) -> int:
        return len(self.latency)


def scene_label(folder_name: str) -> str:
    """`20261016_232523_clutter_high2` -> `clutter_high2`.

    The date-time prefix identifies the recording, not the scene, and repeating it in every legend
    entry would push the numbers that matter off the edge of the panel.
    """
    parts = folder_name.split("_")
    if len(parts) > 2 and parts[0].isdigit() and parts[1].isdigit():
        return "_".join(parts[2:])
    return folder_name


def labels_for(csv_path: Path) -> tuple[str, str, str]:
    """(model, dataset, scene) from the path. Raises on a folder not in the tables.

    The path is the only source for these. Every CSV in runs/camera carries `dataset=custom`,
    because that is what the run was invoked with, and the `model` column says `yolov8` where the
    report says `yolo8` -- so benchmark.labels_from, right for a figure title, gives neither of
    the two names an output filename is keyed on.
    """
    model_dir, dataset_dir, scene_dir = csv_path.parts[-4:-1]
    if model_dir not in FOLDER_MODELS:
        raise KeyError(
            f"{csv_path}: unknown model folder {model_dir!r}. "
            f"Add it to FOLDER_MODELS (known: {sorted(FOLDER_MODELS)})."
        )
    if dataset_dir not in FOLDER_DATASETS:
        raise KeyError(
            f"{csv_path}: unknown dataset folder {dataset_dir!r}. "
            f"Add it to FOLDER_DATASETS (known: {sorted(FOLDER_DATASETS)})."
        )
    scene = scene_label(scene_dir)
    if scene not in SCENE_ORDER:
        raise KeyError(
            f"{csv_path}: unknown scene folder {scene_dir!r} (read as {scene!r}). "
            f"Add it to SCENE_ORDER and SCENE_COLOURS (known: {SCENE_ORDER})."
        )
    return FOLDER_MODELS[model_dir], FOLDER_DATASETS[dataset_dir], scene


def check_model_column(csv_path: Path, rows, model: str) -> None:
    """Raise if the log's own `model` column disagrees with the folder it was found in.

    Catches a run filed under the wrong model -- the one failure this module cannot otherwise
    see, because it would produce a perfectly well-formed figure and table row carrying another
    architecture's latency under this one's name. Silence when the column is absent or carries a
    name outside the alias table: an older log that predates the column still reads, which is the
    append-only contract the CSV columns are under.
    """
    logged = (rows[0].get("model") or "").strip() if rows else ""
    expected = CSV_MODEL_ALIASES.get(logged)
    if expected is not None and expected != model:
        raise ValueError(
            f"{csv_path}: log says model={logged!r} ({expected}) but it is filed under "
            f"{model!r}. One of the two is wrong -- a misfiled log would be published under the "
            f"wrong model's name."
        )


def load_run(csv_path: Path) -> Run | str:
    """One Run from a per-frame log, or a string saying why there is nothing plottable in it.

    The warm-up frame goes before anything is measured or drawn, via the same plotted_rows the
    per-run figures use, so these numbers and plot_run's agree by construction rather than by
    both happening to slice the same way.

    The reason comes back rather than a bare None so the caller can say which file it dropped and
    why: a scene quietly missing from a figure is a 23-row table where 24 were expected, with
    nothing pointing at the cause.
    """
    rows = read_log(csv_path)
    if is_detection_log(rows):
        return "per-object log, no inference_ms column"
    # Only its None-ness is wanted -- the statistics below are computed over the same frames, but
    # going through summarise means "is there anything here" is decided the way the rest of the
    # project decides it, rather than by a second empty-check that could drift from it.
    if summarise(rows) is None:
        return "no rows logged"
    # summarise deliberately keeps the warm-up frame when a run is too short to have anything
    # else -- better a figure of a 2-frame run than a crash at the end of one. Here that would
    # put a 3000 ms cold-start number in a report table under a heading saying the warm-up was
    # excluded, so a run with nothing measured is dropped instead.
    if len(rows) <= WARMUP_FRAMES:
        return f"only {len(rows)} frame(s), all warm-up"

    shown = plotted_rows(rows)
    elapsed = np.asarray([float(r["elapsed_s"]) for r in shown], dtype=float)
    latency = np.asarray([float(r["inference_ms"]) for r in shown], dtype=float)
    model, dataset, scene = labels_for(csv_path)
    check_model_column(csv_path, rows, model)

    return Run(
        csv_path=csv_path,
        model=model,
        dataset=dataset,
        scene=scene,
        # Re-zeroed per run, so three scenes can share an x-axis: the raw elapsed_s counts from
        # the start of the *run*, and a slow warm-up frame pushes the first measured frame out to
        # 3.5 s in some logs and 0.4 s in others. Within one figure the spread is much smaller --
        # under 0.15 s for all eight groups, since a model's warm-up costs it about the same every
        # time -- so this mostly buys a legible axis starting at zero rather than rescuing a
        # badly staggered one. It is still the correct zero: x is then time into the measured
        # part of the run, which is what the series actually show.
        elapsed=elapsed - elapsed[0],
        latency=latency,
        mean_ms=float(latency.mean()),
        # ddof=1: these are samples of a run's latency, not a population, and it keeps the figure
        # legend and the report's table on the same convention.
        sd_ms=float(latency.std(ddof=1)) if len(latency) > 1 else 0.0,
    )


def discover_runs(run_dir=RUN_DIR) -> tuple[list[Run], list[str]]:
    """Every per-frame run log under runs/camera, plus a line per file that was skipped.

    Sorted model, dataset, then scene difficulty -- the order the figures and the table share.
    """
    runs, skipped = [], []
    for csv_path in sorted(Path(run_dir).glob("*/*/*/*.csv")):
        if csv_path.parts[-4] in NOT_RUNS:
            continue
        run = load_run(csv_path)
        if isinstance(run, Run):
            runs.append(run)
        else:
            skipped.append(f"{csv_path}: {run}")

    def key(run: Run):
        return (run.model, run.dataset, SCENE_ORDER.index(run.scene))

    return sorted(runs, key=key), skipped


def group_runs(runs) -> dict[tuple[str, str], list[Run]]:
    """Runs keyed by (model, dataset) -- one figure each. Insertion order is discover_runs's."""
    grouped: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.model, run.dataset), []).append(run)
    return grouped


def plot_latency(runs, png_path) -> Path:
    """The figure for one model/dataset: its scenes' latency over time, on one panel.

    Per scene a thin per-frame trace plus a bold rolling mean, because the frame-to-frame noise is
    part of what a reader is judging -- a flat 45 ms and a spiky 45 ms are not the same model --
    but three raw traces on one panel are unreadable without the means drawn over them.
    """
    runs = list(runs)
    png_path = Path(png_path)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))

    pooled = np.concatenate([r.latency for r in runs])
    pooled_mean = float(pooled.mean())
    pooled_p95 = float(np.percentile(pooled, 95))

    for run in runs:
        colour = SCENE_COLOURS.get(run.scene, "tab:gray")
        # One tenth of the run, capped -- plot_run's window, so the smoothing on these figures
        # matches the per-run ones they are drawn from.
        window = min(15, max(1, run.frames // 10))
        label = f"{run.scene} -- {run.mean_ms:.1f} $\\pm$ {run.sd_ms:.1f} ms"
        # On a run too short to smooth, the rolling mean is the raw series: drawing both would
        # put a bold line exactly over a faint one and claim in the legend to be an average.
        if window > 1:
            ax.plot(run.elapsed, run.latency, linewidth=0.8, alpha=0.35, color=colour)
            ax.plot(run.elapsed, rolling_mean(run.latency, window), linewidth=1.8, color=colour,
                    label=label)
        else:
            ax.plot(run.elapsed, run.latency, linewidth=1.8, color=colour, label=label)

    ax.axhline(pooled_mean, linestyle="--", linewidth=1.0, color="tab:red",
               label=f"mean {pooled_mean:.1f} ms")
    ax.axhline(pooled_p95, linestyle=":", linewidth=1.0, color="0.35",
               label=f"p95 {pooled_p95:.1f} ms")

    ax.set_ylabel("inference latency (ms)")
    # Not "since start of run", which is what plot_run's x-axis means and what elapsed_s counts:
    # these series are re-zeroed on the first *measured* frame, which in some logs is seconds
    # into the run. The same words on both figures would mean two different things in one report.
    ax.set_xlabel("time since first measured frame (s)")
    # Headroom above axis_top's 15%, which on a flat series -- rfdetr sits within a millisecond of
    # 45 ms all run -- leaves the p95 line running underneath the legend box.
    ax.set_ylim(0, axis_top(pooled) * 1.12)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)

    model, dataset = runs[0].model, runs[0].dataset
    frames = sum(r.frames for r in runs)
    ax.set_title(f"{model} / {dataset}  --  {len(runs)} scenes, {frames} frames, "
                 f"mean {pooled_mean:.1f} ms", fontsize=11)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def table_rows(runs) -> list[list[str]]:
    """The per-run latency table: a header plus one row per run, in discover_runs's order."""
    rows = [["model", "dataset", "scene", "frames", "mean_ms", "sd_ms"]]
    for run in runs:
        rows.append([run.model, run.dataset, run.scene, str(run.frames),
                     f"{run.mean_ms:.2f}", f"{run.sd_ms:.2f}"])
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=RUN_DIR, help="where the run folders live")
    parser.add_argument("--out-dir", default=OUT_DIR, help="where the figures go")
    parser.add_argument("--table", action="store_true",
                        help="also print the per-run latency table as CSV")
    args = parser.parse_args(argv)

    runs, skipped = discover_runs(args.run_dir)
    for line in skipped:
        print(f"skipped {line}", file=sys.stderr)
    if not runs:
        print(f"no run logs under {args.run_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    for (model, dataset), group in group_runs(runs).items():
        png = plot_latency(group, out_dir / f"{model}-{dataset}.png")
        scenes = ", ".join(r.scene for r in group)
        print(f"{png}  ({scenes})")

    if args.table:
        print()
        for row in table_rows(runs):
            print(",".join(row))

    note = f"\n{len(runs)} runs, warm-up frame excluded (WARMUP_FRAMES={WARMUP_FRAMES})"
    if skipped:
        note += f", {len(skipped)} file(s) skipped"
    print(note, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
