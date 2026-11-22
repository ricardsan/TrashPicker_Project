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


# =============================================================================
# the board placed by three tape-measured corners
# =============================================================================
def corners_on_floor(T_base_board) -> list[tuple[float, float]]:
    """P0, P1, P2 where a board at T_base_board has them, in mm: what the tape would read."""
    from camera.charuco import reference_points
    points = (T_base_board[:3, :3] @ reference_points(SPEC).T).T + T_base_board[:3, 3]
    return [(float(x), float(y)) for x, y in points[:, :2] * 1000.0]


@pytest.mark.parametrize("at, yaw", [((527.0, -67.0), -90.0), ((450.0, 100.0), 0.0),
                                     ((300.0, 250.0), 37.0), ((600.0, -200.0), 180.0)])
def test_three_measured_corners_put_the_board_back_where_it_was(at, yaw):
    T = demo.board_in_base(SPEC, at, yaw)
    fit = demo.board_from_points(SPEC, corners_on_floor(T))
    assert fit.T_base_board == pytest.approx(T, abs=1e-9)
    assert max(fit.errors_mm) == pytest.approx(0.0, abs=1e-6)
    assert fit.corner_deg == pytest.approx(90.0)
    assert fit.distances_mm == pytest.approx(fit.expected_mm)


def test_tape_error_of_a_few_mm_is_accepted_and_reported():
    T = demo.board_in_base(SPEC, (527.0, -67.0), -90.0)
    (x0, y0), p1, (x2, y2) = corners_on_floor(T)
    fit = demo.board_from_points(SPEC, [(x0 + 3.0, y0), p1, (x2, y2 - 2.0)])
    assert 0.5 < max(fit.errors_mm) < 4.0
    centre = fit.T_base_board @ np.append(SPEC.centre_m, 1.0)
    assert np.hypot(*(centre[:2] * 1000 - (527.0, -67.0))) < 3.0


def test_swapped_corners_or_a_flipped_y_line_are_refused():
    p0, p1, p2 = corners_on_floor(demo.board_in_base(SPEC, (500.0, 0.0), 0.0))
    with pytest.raises(SystemExit, match="mirrored"):
        demo.board_from_points(SPEC, [p0, p2, p1])
    with pytest.raises(SystemExit, match="mirrored"):
        demo.board_from_points(SPEC, [(x, -y) for x, y in (p0, p1, p2)])


def test_points_that_are_not_the_board_are_refused():
    p0, p1, p2 = corners_on_floor(demo.board_in_base(SPEC, (500.0, 0.0), 0.0))
    with pytest.raises(SystemExit, match="do not fit the board"):
        demo.board_from_points(SPEC, [p0, (p1[0] + 40.0, p1[1]), p2])


def test_main_with_points_writes_the_measured_placement(tmp_path, monkeypatch, capsys):
    bag = tmp_path / "20261120_120000_board_only.db3"
    bag.write_bytes(b"")
    monkeypatch.setattr(demo, "measure_bag", lambda bag, spec, step: measured())
    T_board = demo.board_in_base(SPEC, (527.0, -67.0), -90.0)
    points = [f"{x:.2f},{y:.2f}" for x, y in corners_on_floor(T_board)]
    out = tmp_path / "home.txt"
    expected = demo.compose(T_board, cam_board())
    camera = expected[:3, 3] * 1000
    assert demo.main(["--rosbag", str(bag), "--board", "home40", "--points", *points,
                      "--camera-at", f"{camera[0]:.0f},{camera[1]:.0f},{camera[2] + 30:.0f}",
                      "--output", str(out)]) == 0
    assert load_T_base_cam(out) == pytest.approx(expected, abs=1e-4)
    assert is_synthetic(out) and "Measured with a tape" in out.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "off by" in printed and "(30 mm)" in printed


@pytest.mark.parametrize("extra, message", [
    (["--points", "1,1", "2,2", "3,3", "--board-at", "450,100"], "mean nothing with it"),
    (["--points", "1,1", "2,2", "3,3", "--board-yaw", "90"], "mean nothing with it"),
    (["--camera-at", "1,2,3"], "needs --points"),
])
def test_flags_that_contradict_the_points_are_refused(tmp_path, extra, message, capsys):
    with pytest.raises(SystemExit):
        demo.main(["--rosbag", str(tmp_path / "x.db3"), "--board", "home40", *extra])
    assert message in capsys.readouterr().err
