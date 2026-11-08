"""camera/charuco.py: the presets, P0/P1/P2, and the pose recovered from a rendered board."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from camera.charuco import (BOARDS, camera_matrix, detect_pose, make_board,
                            reference_distances_mm, reference_points)

# The bags' colour stream: 1280 x 720, no distortion.
K = camera_matrix(907.5, 906.9, 638.6, 377.1)
SIZE = (1280, 720)
PX_PER_M = 2000     # about 1.7x the camera's 1160 px/m at 0.78 m, so the warp barely aliases


def rotation_x(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def render(name: str, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The board seen by K from pose (R, t): its generated image warped by the plane's homography."""
    spec = BOARDS[name]
    board, _ = make_board(spec)
    width_m, height_m = spec.size_m
    w, h = int(round(width_m * PX_PER_M)), int(round(height_m * PX_PER_M))
    flat = board.generateImage((w, h), marginSize=0, borderBits=1)
    corners_m = np.array([[0, 0, 0], [width_m, 0, 0], [width_m, height_m, 0], [0, height_m, 0]],
                         dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    seen, _ = cv2.projectPoints(corners_m, rvec, t, K, np.zeros(5))
    # Pixel i of the generated image covers [i, i+1), but warpPerspective puts pixel i's *centre* at
    # i. So the board's outer edge is at -0.5 and w - 0.5, not 0 and w: without this the board comes
    # out one pixel too big, which reads as ~0.2% too close.
    edges = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) - 0.5
    H = cv2.getPerspectiveTransform(edges, np.float32(seen.reshape(-1, 2)))
    return cv2.warpPerspective(flat, H, SIZE, flags=cv2.INTER_LINEAR, borderValue=255)


def test_the_a2_reference_points_are_the_240_and_300_mm_the_lab_script_always_checked():
    assert reference_distances_mm(BOARDS["a2"]) == pytest.approx((240.0, 300.0))
    p = reference_points(BOARDS["a2"])
    assert p[1, 1] == p[0, 1] and p[2, 0] == p[0, 0]      # P1 along x, P2 along y


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_every_preset_builds_a_board(name):
    board, detector = make_board(BOARDS[name])
    assert board.getChessboardSize() == (BOARDS[name].squares_x, BOARDS[name].squares_y)
    assert detector is not None


@pytest.mark.parametrize("name, tilt", [("a2", 0.0), ("a2", 12.0), ("home40", 8.0), ("home50", 5.0)])
def test_the_pose_of_a_rendered_board_is_recovered(name, tilt):
    spec = BOARDS[name]
    R = rotation_x(tilt)
    # Board centre straight ahead of the camera, 0.78 m away.
    t = np.array([0.0, 0.0, 0.78]) - R @ spec.centre_m
    board, detector = make_board(spec)
    pose = detect_pose(render(name, R, t), K, np.zeros(5), board, detector)

    assert pose is not None
    assert pose.rms_px < 1.0
    # One rendered image of a board this small recovers to 0.5-1.7 mm and 0.1-0.6 degrees (measured;
    # the 12-corner home50 is the worst). These limits are that, with room: they are here to catch a
    # wrong unit, axis or sign, which is off by centimetres or tens of degrees, not to grade PnP.
    assert np.linalg.norm(pose.T_cam_board[:3, 3] - t) * 1000 < 3.0
    cosine = (np.trace(pose.T_cam_board[:3, :3].T @ R) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 1.0


def test_no_board_is_no_pose():
    board, detector = make_board(BOARDS["a2"])
    blank = np.full((SIZE[1], SIZE[0]), 255, np.uint8)
    assert detect_pose(blank, K, np.zeros(5), board, detector) is None
