"""robot/experiment.py and robot/experiment_summary.py: what an experiment writes, and the results."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from robot import experiment as ex
from robot import experiment_summary as summary
from robot.frame_selection import Snapshot
from robot.pick_sequence import grasp_z_for, gripper_rz_deg
from synthetic import Intrinsics, ScriptedKeys, make_target

INTR = Intrinsics(width=32, height=24, fx=30.0, fy=31.0, ppx=15.5, ppy=11.5)


def write_gt(path, rows) -> str:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "x_mm", "y_mm", "axis_deg"])
        writer.writerows(rows)
    return str(path)


def snapshot(targets, with_frame=True) -> Snapshot:
    image = np.full((24, 32, 3), 128, np.uint8)
    metres = np.full((24, 32), 0.8) if with_frame else None
    return Snapshot(42, image.copy(), targets, ["paper: no depth"],
                    image=image if with_frame else None, metres=metres,
                    intrinsics=INTR if with_frame else None)


def read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# =============================================================================
# ground truth
# =============================================================================
def test_ground_truth_reads_with_and_without_an_axis(tmp_path):
    rows = ex.read_ground_truth(write_gt(tmp_path / "gt.csv", [["metal", 400, 100, 30],
                                                                ["glass", 500, -50, ""]]))
    assert rows == [ex.GroundTruth("metal", 400.0, 100.0, 30.0),
                    ex.GroundTruth("glass", 500.0, -50.0, None)]


def test_a_ground_truth_file_without_positions_is_refused(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text("class,x\nmetal,1\n")
    with pytest.raises(SystemExit, match="missing x_mm, y_mm"):
        ex.read_ground_truth(path)


@pytest.mark.parametrize("measured, truth, error", [(10, 20, 10), (170, 10, 20), (0, 90, 90),
                                                    (None, 10, None), (10, None, None)])
def test_axis_error_is_between_undirected_axes(measured, truth, error):
    assert ex.axis_error_deg(measured, truth) == (None if error is None else pytest.approx(error))


def test_a_target_matches_the_nearest_ground_truth_of_its_own_class():
    truths = [ex.GroundTruth("glass", 400, 100, None), ex.GroundTruth("metal", 450, 100, None),
              ex.GroundTruth("metal", 405, 100, None)]
    found = ex.match(make_target(cls=1, base=(400, 100, 50)), truths, set())
    assert found == (2, truths[2])
    assert ex.match(make_target(cls=1, base=(400, 100, 50)), truths, {1, 2}) is None
    assert ex.match(make_target(cls=1, base=(900, 100, 50)), truths, set()) is None


# =============================================================================
# one experiment on disk
# =============================================================================
def test_an_experiment_writes_the_snapshot_targets_picks_and_cloud(tmp_path):
    gt = write_gt(tmp_path / "gt.csv", [["metal", 410, 100, 40], ["plastic", 600, 0, ""]])
    targets = [make_target(cls=1, base=(400, 100, 50), base_axis_deg=30.0),
               make_target(cls=0, base=(300, 0, 60))]
    log = ex.ExperimentLog("sep2 test", {"source": "camera"}, root=tmp_path / "exp", ground_truth=gt)
    assert log.dir.name.endswith("_sep2_test")
    log.snapshot(snapshot(targets), np.eye(4))
    log.pick(1, targets[0], True, "success")
    log.pick(2, targets[1], False, "not run")
    log.finish(1)

    files = {path.name for path in log.dir.iterdir()}
    assert files == {"meta.json", "snapshot.png", "targets.csv", "picks.csv",
                     "scene_cloud.ply", "scene_cloud.pts"}

    rows = read(log.dir / "targets.csv")
    assert [row["class"] for row in rows] == ["metal", "glass"]
    assert float(rows[0]["err_xy_mm"]) == pytest.approx(10.0)
    assert float(rows[0]["err_axis_deg"]) == pytest.approx(10.0)
    assert rows[1]["err_xy_mm"] == ""                                  # no glass in the ground truth
    assert float(rows[0]["grasp_z_mm"]) == grasp_z_for(targets[0])
    assert float(rows[0]["rz_deg"]) == pytest.approx(gripper_rz_deg(30.0))

    picks = read(log.dir / "picks.csv")
    assert [(p["rank"], p["sequence_ran"], p["verdict"]) for p in picks] == \
        [("1", "1", "success"), ("2", "0", "not run")]
    assert float(picks[0]["err_xy_mm"]) == pytest.approx(10.0)

    meta = json.loads((log.dir / "meta.json").read_text())
    assert meta["source"] == "camera" and meta["frame_index"] == 42
    assert meta["n_ground_truth"] == 2 and meta["n_matched"] == 1
    assert meta["unmatched_ground_truth"] == ["plastic"]
    assert meta["completed_sequences"] == 1


def test_no_frame_means_no_cloud(tmp_path):
    log = ex.ExperimentLog("x", {}, root=tmp_path)
    log.snapshot(snapshot([make_target()], with_frame=False), np.eye(4))
    assert not (log.dir / "scene_cloud.ply").exists()


def test_the_verdict_keys():
    assert ex.ask_verdict(ScriptedKeys(waited=["y"])) == "success"
    assert ex.ask_verdict(ScriptedKeys(waited=["n"])) == "fail"
    keys = ScriptedKeys(waited=["x"])
    assert ex.ask_verdict(keys) == "skipped" and keys.asked == ["ynx"]


# =============================================================================
# the summary
# =============================================================================
def run_experiment(root, name, picks, gt_rows=()):
    gt = write_gt(root / f"gt_{name}.csv", gt_rows) if gt_rows else None
    targets = [target for target, _ in picks]
    log = ex.ExperimentLog(name, {"source": "camera"}, root=root / "exp", ground_truth=gt)
    log.snapshot(snapshot(targets, with_frame=False), np.eye(4))
    for rank, (target, verdict) in enumerate(picks, 1):
        log.pick(rank, target, verdict != "not run", verdict)
    log.finish(sum(verdict != "not run" for _, verdict in picks))


def test_the_summary_pools_runs_by_class_and_counts_only_judged_picks(tmp_path):
    metal = lambda x: make_target(cls=1, base=(x, 100, 50))           # noqa: E731
    run_experiment(tmp_path, "a", [(metal(400), "success"), (metal(500), "fail")],
                   gt_rows=[["metal", 403, 104, ""], ["metal", 500, 100, ""], ["glass", 0, 0, ""]])
    run_experiment(tmp_path, "b", [(metal(400), "success"), (make_target(cls=3), "skipped"),
                                   (make_target(cls=0), "not run")])
    runs = summary.load(tmp_path / "exp")
    assert len(runs) == 2

    by_run = {row["experiment"].split("_", 2)[2]: row for row in summary.per_experiment(runs)}
    assert by_run["a"]["success_rate"] == 0.5
    assert by_run["a"]["detection_recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert by_run["b"]["success_rate"] == 1.0 and by_run["b"]["not_run"] == 1

    rows, errors = summary.per_class(runs)
    by_class = {row["class"]: row for row in rows}
    assert by_class["metal"]["success"] == 2 and by_class["metal"]["fail"] == 1
    assert by_class["metal"]["success_rate"] == pytest.approx(0.667, abs=1e-3)
    assert by_class["plastic"]["success_rate"] == ""                  # only a skip: nothing judged
    assert sorted(errors["metal"]) == [0.0, 5.0]
    assert by_class["metal"]["err_xy_max_mm"] == 5.0


def test_the_summary_writes_tables_and_figures(tmp_path):
    run_experiment(tmp_path, "a", [(make_target(cls=1, base=(400, 100, 50)), "success")],
                   gt_rows=[["metal", 403, 104, ""]])
    out = tmp_path / "summary"
    assert summary.main(["--experiments", str(tmp_path / "exp"), "--out", str(out)]) == 0
    assert {path.name for path in out.iterdir()} == {
        "per_experiment.csv", "per_class.csv", "success_by_class.png", "position_error_by_class.png"}


def test_the_summary_refuses_an_empty_folder(tmp_path):
    with pytest.raises(SystemExit, match="no experiments"):
        summary.main(["--experiments", str(tmp_path)])


# =============================================================================
# the hooks in run_picks and robot_trash_picker
# =============================================================================
class StubRobot:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


def test_run_picks_reports_every_target_to_on_picked():
    from robot.pick_sequence import run_picks
    seen = []
    targets = [make_target(cls=1, base=(400, 100, 50)), make_target(cls=1, base=(450, 100, 50))]
    completed = run_picks(StubRobot(), targets, ScriptedKeys(waited=["c"]),
                          on_picked=lambda rank, target, ran: seen.append((rank, target, ran)))
    assert completed == 2
    assert [(rank, ran) for rank, _, ran in seen] == [(1, True), (2, True)]
    assert seen[1][1] is targets[1]


@pytest.mark.parametrize("argv, message", [
    (["--ground-truth", "gt.csv"], "need --experiment"),
    (["--mock", "--experiment", "a"], "--experiment means nothing with --mock"),
    (["--mock", "--viz"], "--viz means nothing with --mock"),
])
def test_experiment_flags_that_mean_nothing_are_refused(argv, message, capsys):
    from robot.robot_trash_picker import build_parser, check_args
    parser = build_parser()
    with pytest.raises(SystemExit):
        check_args(parser, parser.parse_args(argv))
    assert message in capsys.readouterr().err
