"""robot/board_demo_calibration.py: board placement, composition, agreement checks and the file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import robot.board_demo_calibration as demo
from camera.charuco import BOARDS
from robot.art_camera_client import Z_TABLE_MM
from robot.extrinsics import camera_to_base_mm, is_synthetic, load_T_base_cam

SPEC = BOARDS["home40"]


def rotation_x(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def cam_board(tilt_deg: float = 10.0, distance_m: float = 0.78) -> np.ndarray:
    """T_cam_board for a board whose centre is distance_m straight along the optical axis."""
    R = rotation_x(tilt_deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([0.0, 0.0, distance_m]) - R @ SPEC.centre_m
    return T


def measured(T=None) -> demo.Measured:
    return demo.Measured(cam_board() if T is None else T, 20, 25, 0.1, 0.01, 0.3)


def test_the_board_centre_lands_where_it_was_put_and_the_board_lies_flat():
    T = demo.board_in_base(SPEC, (450.0, 100.0), 30.0)
    centre = T @ np.append(SPEC.centre_m, 1.0)
    assert centre[:3] * 1000 == pytest.approx([450.0, 100.0, Z_TABLE_MM])
    corner = T @ np.array([0.0, 0.0, 0.0, 1.0])
    assert corner[2] * 1000 == pytest.approx(Z_TABLE_MM)
    assert (T[:3, :3] @ [0.0, 0.0, 1.0])[2] == pytest.approx(-1.0)     # into the board is down


def test_composed_the_camera_sees_the_board_centre_at_its_base_position_from_above():
    T = demo.compose(demo.board_in_base(SPEC, (450.0, 100.0), 0.0), cam_board(10.0, 0.78))
    x, y, z = camera_to_base_mm(T, (0.0, 0.0, 0.78))
    assert (x, y, z) == pytest.approx((450.0, 100.0, Z_TABLE_MM), abs=1e-6)
    assert T[2, 2] < 0                                                  # looking down
    assert T[2, 3] * 1000 == pytest.approx(Z_TABLE_MM + 780.0 * np.cos(np.radians(10.0)))


def test_height_and_tilt():
    height, tilt = demo.height_and_tilt(cam_board(12.0, 0.80))
    assert height == pytest.approx(800.0 * np.cos(np.radians(12.0)))
    assert tilt == pytest.approx(12.0)


def test_identical_poses_combine_to_themselves():
    T, spread_mm, spread_deg = demo.combine_poses([cam_board()] * 5)
    assert T == pytest.approx(cam_board())
    assert spread_mm == pytest.approx(0.0) and spread_deg == pytest.approx(0.0, abs=1e-4)


def test_a_board_that_moved_is_refused():
    moved = cam_board()
    moved[0, 3] += 0.005
    with pytest.raises(SystemExit, match="moves between frames"):
        demo.combine_poses([cam_board()] * 4 + [moved])
    with pytest.raises(SystemExit, match="moves between frames"):
        demo.combine_poses([cam_board(10.0)] * 4 + [cam_board(13.0)])


def test_too_few_frames_are_refused():
    with pytest.raises(SystemExit, match="only 2 usable frames"):
        demo.combine_poses([cam_board()] * 2)


def test_the_file_is_synthetic_and_loads_back_as_what_was_computed(tmp_path):
    T = demo.compose(demo.board_in_base(SPEC, (300.0, -50.0), 15.0), cam_board())
    path = tmp_path / "home.txt"
    path.write_text(demo.format_calibration(T, measured(), "home40", Path("x_board_only.db3"),
                                            (300.0, -50.0), 15.0), encoding="utf-8")
    assert is_synthetic(path)
    assert load_T_base_cam(path) == pytest.approx(T, abs=1e-6)


def test_main_writes_the_file_from_the_measured_pose(tmp_path, monkeypatch):
    bag = tmp_path / "20261107_120000_board_only.db3"
    bag.write_bytes(b"")
    monkeypatch.setattr(demo, "measure_bag", lambda bag, spec, step: measured())
    out = tmp_path / "out" / "home.txt"
    assert demo.main(["--rosbag", str(bag), "--board", "home40", "--board-at", "450,100",
                      "--output", str(out)]) == 0
    T = load_T_base_cam(out)
    assert camera_to_base_mm(T, (0.0, 0.0, 0.78)) == pytest.approx((450.0, 100.0, Z_TABLE_MM),
                                                                     abs=1e-3)
    assert "20261107_120000_board_only" in out.read_text(encoding="utf-8")


def test_a_missing_bag_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no such rosbag"):
        demo.main(["--rosbag", str(tmp_path / "missing.db3"), "--board", "home50"])
