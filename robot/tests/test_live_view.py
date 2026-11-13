"""robot/live_view.py against a recording stand-in for rerun: what gets logged, and the grasp drawing."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from robot import live_view
from robot.pick_sequence import APPROACH_Z_MM, grasp_z_for
from synthetic import FakeDepth, Intrinsics, make_target


class Archetype:
    def __init__(self, kind, *args, **kwargs):
        self.kind, self.args, self.kwargs = kind, args, kwargs


def fake_rerun():
    """Enough of rerun's API to run LiveView, recording every log call."""
    rr = types.SimpleNamespace(logged=[], times=[], inits=[])
    rr.init = lambda app, spawn: rr.inits.append((app, spawn))
    rr.log = lambda path, item, static=False: rr.logged.append((path, item, static))
    rr.set_time = lambda timeline, sequence: rr.times.append(sequence)
    rr.ViewCoordinates = types.SimpleNamespace(RIGHT_HAND_Z_UP="z-up", RDF="rdf")
    for kind in ("Transform3D", "Pinhole", "Image", "Points3D", "LineStrips3D", "Clear"):
        setattr(rr, kind, lambda *a, _kind=kind, **k: Archetype(_kind, *a, **k))
    return rr


def paths(rr) -> list[str]:
    return [path for path, _, _ in rr.logged]


def looking_down() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.4, 0.1, 0.8]
    return T


def test_the_camera_and_axes_are_set_up_once():
    rr = fake_rerun()
    live_view.LiveView(looking_down(), rr=rr, spawn=False)
    assert rr.inits == [(live_view.APP_ID, False)]
    camera = dict((path, item) for path, item, _ in rr.logged)["world/camera"]
    assert camera.kwargs["translation"] == pytest.approx([400.0, 100.0, 800.0])


def test_a_frame_logs_image_cloud_and_targets_and_only_every_nth_frame_is_logged():
    rr = fake_rerun()
    view = live_view.LiveView(looking_down(), rr=rr, spawn=False, every=2, stride=8)
    intr = Intrinsics(width=64, height=48, fx=60.0, fy=61.0, ppx=31.5, ppy=23.5)
    image = np.zeros((48, 64, 3), np.uint8)
    depth = FakeDepth(np.full((48, 64), 0.8))
    target = make_target(cls=1, base=(400, 100, 30))

    view.frame(1, image, depth, intr, [target], image)
    assert rr.times == []                                              # odd frame: skipped
    view.frame(2, image, depth, intr, [target], image)
    assert rr.times == [2]
    logged = paths(rr)
    for path in ("world/camera/image", "image/annotated", "world/cloud", "world/targets"):
        assert path in logged
    cloud = [item for path, item, _ in rr.logged if path == "world/cloud"][0]
    assert np.allclose(np.asarray(cloud.args[0])[:, 2], 0.0)           # the floor, level at z = 0

    view.frame(4, image, depth, intr, [], image)
    assert [item.kind for path, item, _ in rr.logged if path == "world/targets"][-1] == "Clear"


def test_the_grasp_drawing_closes_the_fingers_across_the_long_axis():
    target = make_target(cls=1, base=(400, 100, 30), base_axis_deg=30.0)
    geometry = live_view.grasp_geometry(target)
    fingers = geometry["fingers"][1] - geometry["fingers"][0]
    axis = geometry["axis"][1] - geometry["axis"][0]
    assert abs(fingers @ axis) < 1e-6                                   # perpendicular
    assert geometry["approach"][0][2] == APPROACH_Z_MM
    assert geometry["grasp"][2] == grasp_z_for(target)
    assert math.isclose(np.linalg.norm(fingers), 2 * live_view.FINGER_HALF_MM)


def test_a_target_with_no_axis_gets_no_finger_line():
    geometry = live_view.grasp_geometry(make_target(base_axis_deg=None))
    assert "fingers" not in geometry and "axis" not in geometry


def test_snapshot_and_results_draw_every_target():
    rr = fake_rerun()
    view = live_view.LiveView(looking_down(), rr=rr, spawn=False)
    targets = [make_target(cls=1, base_axis_deg=0.0), make_target(cls=0, base_axis_deg=None)]
    view.snapshot(10, targets)
    view.pick_result(1, targets[0], "success")
    logged = paths(rr)
    assert {"world/grasp/1/fingers", "world/grasp/1/axis", "world/grasp/2/approach",
            "world/grasp/1/result"} <= set(logged)
    assert "world/grasp/2/fingers" not in logged


def test_missing_rerun_says_how_to_install_it(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_rerun(name, *args, **kwargs):
        if name == "rerun":
            raise ImportError("no rerun")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rerun)
    with pytest.raises(SystemExit, match="pip install rerun-sdk"):
        live_view.import_rerun()
