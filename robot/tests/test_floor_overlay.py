"""The floor picture for RobotStudio: geometry, drawing, files, and the hook in a run."""
from __future__ import annotations

import types

import cv2
import numpy as np
import pytest

from robot import floor_overlay as fo
from synthetic import Intrinsics

INTR = Intrinsics()
# Straight down from 1.0 m over (600, 0): image right = base -y, image down = base -x.
T_DOWN = np.array([[0.0, -1.0, 0.0, 0.6],
                   [-1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0]])


def pixel_of(x_mm, y_mm):
    """Where the straight-down camera sees floor point (x, y)."""
    cam = np.linalg.inv(T_DOWN) @ np.array([x_mm / 1000, y_mm / 1000, 0.0, 1.0])
    return (INTR.fx * cam[0] / cam[2] + INTR.ppx, INTR.fy * cam[1] / cam[2] + INTR.ppy)


def image_with_dot(x_mm, y_mm):
    image = np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)
    u, v = pixel_of(x_mm, y_mm)
    cv2.circle(image, (int(round(u)), int(round(v))), 6, (255, 255, 255), -1)
    return image


def bright_centre(picture):
    rows, columns = np.nonzero(picture[..., 0] > 128)
    return columns.mean(), rows.mean()


def test_the_picture_covers_the_fixed_rectangle_at_one_mm_a_pixel():
    picture = fo.lay_on_floor(np.zeros((INTR.height, INTR.width, 3), np.uint8), T_DOWN, INTR)
    assert picture.shape[:2] == (1420, 770)
    assert fo.for_robotstudio(picture).shape[:2] == (770, 1420)


@pytest.mark.parametrize("x, y", [(600, 0), (700, 150), (450, -200)])
def test_a_floor_point_lands_on_its_own_floor_pixel(x, y):
    picture = fo.lay_on_floor(image_with_dot(x, y), T_DOWN, INTR)
    column, row = bright_centre(picture)
    expected = fo.floor_px(x, y)
    assert abs(column - expected[0]) < 3 and abs(row - expected[1]) < 3


def test_floor_px_puts_far_x_at_the_top_and_plus_y_on_the_left():
    assert fo.floor_px(fo.FLOOR_X_MM[1], fo.FLOOR_Y_MM[1]) == (0, 0)
    far_left = fo.floor_px(1000, 300)
    near_right = fo.floor_px(200, -300)
    assert far_left[1] < near_right[1] and far_left[0] < near_right[0]


def test_floor_the_camera_does_not_see_is_grey():
    picture = fo.lay_on_floor(np.full((INTR.height, INTR.width, 3), 255, np.uint8), T_DOWN, INTR)
    assert tuple(picture[fo.floor_px(1350, 350)[::-1]]) == fo.UNSEEN_BGR       # far corner
    assert tuple(picture[fo.floor_px(600, 0)[::-1]]) == (255, 255, 255)        # under the camera


def test_the_turn_for_robotstudio_matches_the_checked_board_overlay():
    picture = np.zeros((4, 2, 3), np.uint8)
    picture[0, 0] = 255                                     # far X, +Y: top-left of the unturned
    turned = fo.for_robotstudio(picture)
    assert turned.shape[:2] == (2, 4) and turned[0, -1, 0] == 255


def target(x, y, axis):
    return types.SimpleNamespace(xyz_base_mm=(x, y, 30.0), base_axis_deg=axis, name="metal")


def test_the_cross_is_drawn_where_the_tcp_is_sent():
    rows, columns = fo.floor_size_px()
    picture = fo.draw_targets_on_floor(np.zeros((rows, columns, 3), np.uint8), [target(700, 100, 0)])
    column, row = fo.floor_px(700, 100)
    assert tuple(picture[row, column]) == fo.CROSS_BGR


def test_the_fingers_close_across_the_long_axis():
    rows, columns = fo.floor_size_px()
    picture = fo.draw_targets_on_floor(np.zeros((rows, columns, 3), np.uint8), [target(700, 0, 0)])
    green = np.all(picture == fo.FINGERS_BGR, axis=-1)
    ys, xs = np.nonzero(green)
    # a long axis along x: the fingers' line runs along y, so it spreads over columns, not rows
    assert np.ptp(xs) > 2 * fo.FINGERS_HALF_MM * 0.9 and np.ptp(ys) < 30


def test_write_puts_both_pictures_and_their_history_in_the_folder(tmp_path):
    floor = fo.write_floor_overlay(image_with_dot(600, 0), T_DOWN, INTR, [target(600, 0, 30)],
                                   out_dir=tmp_path)
    assert floor == tmp_path / "floor.png"
    assert cv2.imread(str(floor)).shape[:2] == (770, 1420)
    assert (tmp_path / "floor_check.png").exists()
    assert len(list((tmp_path / "history").glob("*.png"))) == 2


def test_a_run_writes_the_picture_and_a_broken_one_only_warns(monkeypatch, capsys,
                                                             floor_pictures_in_tmp):
    import robot.robot_trash_picker as cli
    snapshot = types.SimpleNamespace(annotated=image_with_dot(600, 0), intrinsics=INTR,
                                     targets=[target(600, 0, 0)])
    cli.show_in_station(snapshot, T_DOWN)
    assert (floor_pictures_in_tmp / "floor.png").exists()

    cli.show_in_station(types.SimpleNamespace(annotated=None, intrinsics=INTR, targets=[]), T_DOWN)
    assert "WARNING: no floor picture" in capsys.readouterr().out


def test_a_calibration_point_is_marked_at_its_floor_pixel(tmp_path):
    rows, columns = fo.floor_size_px()
    picture = fo.draw_points_on_floor(np.zeros((rows, columns, 3), np.uint8),
                                      [("P0", 482.0, -0.5, (255, 0, 255))])
    column, row = fo.floor_px(482.0, -0.5)
    assert tuple(picture[row, column]) == (255, 0, 255)
    fo.write_floor_overlay(image_with_dot(600, 0), T_DOWN, INTR, points=[("P0", 600, 0, (0, 255, 0))],
                           out_dir=tmp_path, label="calibration")
    assert len(list((tmp_path / "history").glob("*_calibration_floor.png"))) == 1
