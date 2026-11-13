"""One experiment on disk: the scene the robot worked from, every planned and attempted pick, the result.

    python robot/robot_trash_picker.py --conf 0.4 --experiment sep3_a
    python robot/robot_trash_picker.py --conf 0.4 --experiment sep3_a --ground-truth experiments/gt_sep3_a.csv

Each run with --experiment writes experiments/<date>_<time>_<name>/:

    meta.json        what produced it: source (camera or bag + frame), model, conf, calibration and
                     the T_base_cam itself, host, ground-truth file
    snapshot.png     the annotated frame the robot worked from
    targets.csv      every target on that frame, in pick order, with where the algorithm put it
    picks.csv        one row per target the robot was sent to: commanded x/y/grasp z/rz, whether the
                     sequence ran, the operator's verdict, and -- with ground truth -- the errors
    scene_cloud.ply  the frame as a coloured point cloud in the robot base frame, metres
    scene_cloud.pts  the same as ASCII .pts, for RobotStudio's point-cloud import

After each pick the operator presses y (grasp good), n (grasp failed) or x (cannot tell / skip). In
RobotStudio the real object is not in the station, so "good" means the TCP came down on the object
in the live view, with the fingers across it.

Ground truth
------------
A CSV with a header `class,x_mm,y_mm,axis_deg`: where each object really was, in the robot base
frame, and its long axis in degrees from base +x (empty for round objects). `class` is the class
name. Each target is matched to the nearest ground-truth object of the same class within
MATCH_RADIUS_MM, and the errors go into picks.csv and targets.csv. With the board calibration,
positions measured on the sheet relative to the board convert to base mm through --board-at and
--board-yaw.

experiment_summary.py turns a folder of these into the results tables and figures.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot.pick_sequence import grasp_z_for, gripper_rz_deg
from robot.scene_cloud import cloud_in_base, write_ply, write_pts

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "experiments"
MATCH_RADIUS_MM = 60.0
VERDICT_KEYS = {"y": "success", "n": "fail", "x": "skipped"}

TARGET_FIELDS = ["rank", "class", "conf", "u_px", "v_px", "z_cam_m", "z_source",
                 "x_mm", "y_mm", "z_mm", "axis_deg", "rz_deg", "grasp_z_mm",
                 "gt_x_mm", "gt_y_mm", "gt_axis_deg", "err_xy_mm", "err_axis_deg"]
PICK_FIELDS = ["experiment", "rank", "class", "conf", "x_mm", "y_mm", "grasp_z_mm", "rz_deg",
               "sequence_ran", "verdict", "duration_s",
               "gt_x_mm", "gt_y_mm", "gt_axis_deg", "err_xy_mm", "err_axis_deg"]


@dataclass(frozen=True)
class GroundTruth:
    cls: str
    x_mm: float
    y_mm: float
    axis_deg: float | None


def read_ground_truth(path) -> list[GroundTruth]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no ground-truth file at {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"class", "x_mm", "y_mm"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: needs the columns class,x_mm,y_mm[,axis_deg]; "
                             f"missing {', '.join(sorted(missing))}")
        for row in reader:
            axis = (row.get("axis_deg") or "").strip()
            rows.append(GroundTruth(row["class"].strip(), float(row["x_mm"]), float(row["y_mm"]),
                                    float(axis) if axis else None))
    return rows


def axis_error_deg(measured: float | None, truth: float | None) -> float | None:
    """Smallest angle between two undirected axes, in [0, 90]."""
    if measured is None or truth is None:
        return None
    difference = abs(measured - truth) % 180.0
    return min(difference, 180.0 - difference)


def match(target, truths: list[GroundTruth], used: set[int]) -> tuple[int, GroundTruth] | None:
    """The nearest unused ground truth of the same class within MATCH_RADIUS_MM."""
    x, y, _ = target.xyz_base_mm
    best = None
    for index, truth in enumerate(truths):
        if index in used or truth.cls != target.name:
            continue
        distance = math.hypot(x - truth.x_mm, y - truth.y_mm)
        if distance <= MATCH_RADIUS_MM and (best is None or distance < best[0]):
            best = (distance, index, truth)
    return None if best is None else (best[1], best[2])


def _round(value, digits=2):
    return "" if value is None else round(float(value), digits)


class ExperimentLog:
    """Writes one experiment folder. Call snapshot() once, then pick() after each target."""

    def __init__(self, name: str, meta: dict, root=DEFAULT_DIR, ground_truth=None):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "run"
        self.name = f"{datetime.now():%Y%m%d_%H%M%S}_{safe}"
        self.dir = Path(root) / self.name
        self.dir.mkdir(parents=True, exist_ok=False)
        self.truths = read_ground_truth(ground_truth) if ground_truth else []
        self.meta = dict(meta, experiment=self.name, ground_truth=str(ground_truth or ""),
                         started=datetime.now().isoformat(timespec="seconds"))
        self._write_meta()
        self._matches: dict[int, GroundTruth] = {}
        self._pick_started = None
        with open(self.dir / "picks.csv", "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(PICK_FIELDS)

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def _errors(self, rank: int, target) -> list:
        truth = self._matches.get(rank)
        if truth is None:
            return ["", "", "", "", ""]
        x, y, _ = target.xyz_base_mm
        return [truth.x_mm, truth.y_mm, _round(truth.axis_deg, 1),
                _round(math.hypot(x - truth.x_mm, y - truth.y_mm)),
                _round(axis_error_deg(target.base_axis_deg, truth.axis_deg))]

    def snapshot(self, snapshot, T_base_cam, *, cloud: bool = True) -> None:
        """The frame the robot works from: picture, targets with ground-truth matches, point cloud."""
        cv2.imwrite(str(self.dir / "snapshot.png"), snapshot.annotated)
        used: set[int] = set()
        for rank, target in enumerate(snapshot.targets, 1):
            found = match(target, self.truths, used)
            if found is not None:
                used.add(found[0])
                self._matches[rank] = found[1]
        with open(self.dir / "targets.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(TARGET_FIELDS)
            for rank, target in enumerate(snapshot.targets, 1):
                x, y, z = target.xyz_base_mm
                u, v = target.centroid_px
                writer.writerow([rank, target.name, _round(target.conf, 3), _round(u, 1), _round(v, 1),
                                 _round(target.z_cam_m, 4), target.z_source,
                                 _round(x, 1), _round(y, 1), _round(z, 1),
                                 _round(target.base_axis_deg, 1),
                                 _round(gripper_rz_deg(target.base_axis_deg), 1),
                                 _round(grasp_z_for(target), 1)] + self._errors(rank, target))
        self.meta.update(frame_index=snapshot.frame_index, n_targets=len(snapshot.targets),
                         n_ground_truth=len(self.truths), n_matched=len(self._matches),
                         unmatched_ground_truth=[t.cls for i, t in enumerate(self.truths)
                                                 if i not in used],
                         dropped=list(snapshot.dropped))
        self._write_meta()
        if cloud and snapshot.image is not None and snapshot.metres is not None:
            points, colours = cloud_in_base(snapshot.image, snapshot.metres, snapshot.intrinsics,
                                            T_base_cam)
            write_ply(self.dir / "scene_cloud.ply", points, colours)
            write_pts(self.dir / "scene_cloud.pts", points, colours)
        self._pick_started = time.monotonic()

    def pick(self, rank: int, target, ran: bool, verdict: str) -> None:
        """One row of picks.csv. `verdict` is success, fail, skipped or not run."""
        now = time.monotonic()
        duration = None if self._pick_started is None else now - self._pick_started
        self._pick_started = now
        x, y, _ = target.xyz_base_mm
        row = [self.name, rank, target.name, _round(target.conf, 3), _round(x, 1), _round(y, 1),
               _round(grasp_z_for(target), 1), _round(gripper_rz_deg(target.base_axis_deg), 1),
               int(bool(ran)), verdict, _round(duration, 1)] + self._errors(rank, target)
        with open(self.dir / "picks.csv", "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)

    def finish(self, completed: int) -> None:
        self.meta.update(finished=datetime.now().isoformat(timespec="seconds"),
                         completed_sequences=completed)
        self._write_meta()


def ask_verdict(keys) -> str:
    """y / n / x from the operator, as success / fail / skipped."""
    print("Grasp good? Press 'y' (yes), 'n' (no) or 'x' (cannot tell).")
    return VERDICT_KEYS[keys.wait_for("ynx")]


def calibration_meta(T_base_cam: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in np.asarray(T_base_cam)]
