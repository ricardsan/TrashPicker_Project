"""The results of a set of experiments: tables and figures from the folders experiment.py writes.

    python robot/experiment_summary.py                           # every run in experiments/
    python robot/experiment_summary.py --experiments experiments --out experiments/summary

Writes to --out:

    per_experiment.csv   one row per run: targets found, ground-truth objects matched (detection
                         recall), picks attempted, sequences run, verdicts, success rate
    per_class.csv        the same pooled by class, plus position and axis errors where ground truth
                         was given: mean, median, 95th percentile, max
    success_by_class.png
    position_error_by_class.png   (only with ground truth)

Success rate is success / (success + fail): 'skipped' (operator could not tell) and 'not run' (the
sequence was refused before the gripper closed) are reported, not counted as failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load(root) -> list[dict]:
    """Every experiment folder under `root`: its meta, targets and picks."""
    runs = []
    for folder in sorted(Path(root).iterdir() if Path(root).is_dir() else []):
        meta_path = folder / "meta.json"
        if not folder.is_dir() or not meta_path.exists():
            continue
        runs.append({"name": folder.name,
                     "meta": json.loads(meta_path.read_text(encoding="utf-8")),
                     "targets": _read_csv(folder / "targets.csv"),
                     "picks": _read_csv(folder / "picks.csv")})
    return runs


def _rate(success: int, fail: int):
    return round(success / (success + fail), 3) if success + fail else ""


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": "", "median": "", "p95": "", "max": ""}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"mean": round(statistics.fmean(values), 1), "median": round(statistics.median(values), 1),
            "p95": round(p95, 1), "max": round(max(values), 1)}


def per_experiment(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        verdicts = [pick["verdict"] for pick in run["picks"]]
        success, fail = verdicts.count("success"), verdicts.count("fail")
        meta = run["meta"]
        n_gt = meta.get("n_ground_truth", 0) or 0
        rows.append({
            "experiment": run["name"], "source": meta.get("source", ""),
            "n_targets": len(run["targets"]), "n_ground_truth": n_gt,
            "n_matched": meta.get("n_matched", ""),
            "detection_recall": round(meta.get("n_matched", 0) / n_gt, 3) if n_gt else "",
            "n_picks": len(run["picks"]),
            "n_ran": sum(int(pick["sequence_ran"]) for pick in run["picks"]),
            "success": success, "fail": fail, "skipped": verdicts.count("skipped"),
            "not_run": verdicts.count("not run"), "success_rate": _rate(success, fail)})
    return rows


def per_class(runs: list[dict]) -> tuple[list[dict], dict[str, list[float]]]:
    counts = defaultdict(lambda: defaultdict(int))
    xy_errors, axis_errors = defaultdict(list), defaultdict(list)
    for run in runs:
        for target in run["targets"]:
            counts[target["class"]]["n_targets"] += 1
        for pick in run["picks"]:
            cls = pick["class"]
            counts[cls]["n_picks"] += 1
            counts[cls][pick["verdict"] or "unjudged"] += 1
            if (error := _float(pick.get("err_xy_mm"))) is not None:
                xy_errors[cls].append(error)
            if (error := _float(pick.get("err_axis_deg"))) is not None:
                axis_errors[cls].append(error)
    rows = []
    for cls in sorted(counts):
        count = counts[cls]
        xy, axis = _stats(xy_errors[cls]), _stats(axis_errors[cls])
        rows.append({"class": cls, "n_targets": count["n_targets"], "n_picks": count["n_picks"],
                     "success": count["success"], "fail": count["fail"],
                     "skipped": count["skipped"], "not_run": count["not run"],
                     "success_rate": _rate(count["success"], count["fail"]),
                     "n_with_ground_truth": len(xy_errors[cls]),
                     **{f"err_xy_{key}_mm": value for key, value in xy.items()},
                     **{f"err_axis_{key}_deg": value for key, value in axis.items()}})
    return rows, dict(xy_errors)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(out: Path, class_rows: list[dict], xy_errors: dict[str, list[float]]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    judged = [row for row in class_rows if row["success_rate"] != ""]
    if judged:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = [row["class"] for row in judged]
        rates = [row["success_rate"] * 100 for row in judged]
        bars = ax.bar(names, rates, color="#4C78A8")
        for bar, row in zip(bars, judged):
            ax.annotate(f"{row['success']}/{row['success'] + row['fail']}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("grasp success (%)")
        ax.set_title("Grasp success by class")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        written.append(out / "success_by_class.png")
        fig.savefig(written[-1], dpi=200)
        plt.close(fig)
    with_errors = {cls: values for cls, values in xy_errors.items() if values}
    if with_errors:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.boxplot(list(with_errors.values()), tick_labels=list(with_errors))
        ax.set_ylabel("position error (mm)")
        ax.set_title("Grasp point error against ground truth")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        written.append(out / "position_error_by_class.png")
        fig.savefig(written[-1], dpi=200)
        plt.close(fig)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarise experiment folders into results.")
    parser.add_argument("--experiments", default=DEFAULT_DIR, metavar="DIR",
                        help="the folder holding the experiment folders (default: experiments/)")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="where to write the tables and figures (default: <experiments>/summary)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    runs = load(args.experiments)
    if not runs:
        raise SystemExit(f"no experiments in {args.experiments}: run robot_trash_picker.py "
                         f"with --experiment NAME first")
    out = Path(args.out) if args.out else Path(args.experiments) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    run_rows = per_experiment(runs)
    class_rows, xy_errors = per_class(runs)
    write_rows(out / "per_experiment.csv", run_rows)
    write_rows(out / "per_class.csv", class_rows)
    written = [] if args.no_figures else figures(out, class_rows, xy_errors)

    print(f"{len(runs)} experiment(s)\n")
    print(f"{'class':<10}{'targets':>8}{'picks':>7}{'ok':>5}{'fail':>6}{'rate':>7}"
          f"{'err mean':>10}{'err p95':>9}  (mm)")
    for row in class_rows:
        rate = "" if row["success_rate"] == "" else f"{row['success_rate'] * 100:.0f}%"
        print(f"{row['class']:<10}{row['n_targets']:>8}{row['n_picks']:>7}{row['success']:>5}"
              f"{row['fail']:>6}{rate:>7}{row['err_xy_mean_mm']!s:>10}{row['err_xy_p95_mm']!s:>9}")
    print(f"\nwrote {out / 'per_experiment.csv'}, {out / 'per_class.csv'}"
          + "".join(f", {path.name}" for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
