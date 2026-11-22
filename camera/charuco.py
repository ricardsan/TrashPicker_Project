"""The ChArUco boards this project uses, and the camera -> board pose from one image.

Shared by camera/calibrate_T_base_cam.py (lab: camera -> robot, with the TCP on three corners) and
robot/board_demo_calibration.py (home: camera -> board from a recorded bag, no robot).

Board frame
-----------
OpenCV's: the origin is the board's outer corner, x along the columns, y along the rows, both in
metres, and z = x cross y, which points *into* the board, away from a camera that sees it.

Presets
-------
    a2       the lab board: 6 x 7 squares of 60 mm, 43 mm markers. Confirmed with find_board.py.
    home40   home board with 40 mm squares: 6 x 5, DICT_4X4_50, ids 0-14. Grid confirmed with
             find_board.py on 2026-11-08 (15 markers, 20 corners). With 5 rows, legacy and new
             patterns are the same board. Printed 0.8% large: 242 mm across 6 squares (ruler,
             2026-11-20), so 40.33 mm squares and 30.25 mm markers.
    home50   home board with 50 mm squares: 5 x 4, DICT_4X4_50, ids 0-9, 37 mm markers (ruler).
             Confirmed with find_board.py on 2026-11-08 (10 markers, 12 corners). With 4 rows the
             legacy flag matters, and only legacy=True matched. Printed 3.2% large: 258 mm across
             5 squares (ruler, 2026-11-22), so 51.6 mm squares; the RealSense depth to the board
             agrees with that to 0.25%, and is 3.5% off with 50 mm.

To confirm a board (or its marker size), with it in view run

    python camera/find_board.py --square 40        (or --square 50)

and copy the BOARDS line it prints over the guess below.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_CORNERS = 6


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int          # columns
    squares_y: int          # rows
    square_m: float
    marker_m: float
    dictionary: int = cv2.aruco.DICT_4X4_50
    legacy: bool = True

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    @property
    def centre_m(self) -> np.ndarray:
        width, height = self.size_m
        return np.array([width / 2.0, height / 2.0, 0.0])


BOARDS: dict[str, BoardSpec] = {
    "a2": BoardSpec(6, 7, 0.060, 0.043),
    # printed 0.8% large: 242 mm across 6 squares by ruler, not 240. The marker is scaled to match.
    "home40": BoardSpec(6, 5, 0.242 / 6, 0.030 * 242 / 240),
    # printed 3.2% large: 258 mm across 5 squares by ruler, not 250; the depth camera agrees to 0.25%.
    # The 37 mm marker was measured on the print itself, so it stays.
    "home50": BoardSpec(5, 4, 0.258 / 5, 0.037),
}


def make_board(spec: BoardSpec):
    """(cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector) for `spec`."""
    dictionary = cv2.aruco.getPredefinedDictionary(spec.dictionary)
    board = cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_m, spec.marker_m,
                                   dictionary)
    board.setLegacyPattern(spec.legacy)
    return board, cv2.aruco.CharucoDetector(board)


def camera_matrix(fx: float, fy: float, ppx: float, ppy: float) -> np.ndarray:
    return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class BoardPose:
    T_cam_board: np.ndarray     # 4 x 4, metres: a board point -> the camera's frame
    n_corners: int
    rms_px: float               # reprojection error of the corners used
    rvec: np.ndarray
    tvec: np.ndarray


def detect_pose(gray: np.ndarray, K: np.ndarray, dist, board, detector,
                min_corners: int = MIN_CORNERS) -> BoardPose | None:
    """The board's pose in one greyscale image, or None if too few corners are seen."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < min_corners:
        return None
    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    dist = np.asarray(dist if dist is not None else np.zeros(5), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    rms = float(np.sqrt(((projected.reshape(-1, 2) - img_pts) ** 2).sum(axis=1).mean()))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return BoardPose(T, len(obj_pts), rms, rvec, tvec)


def reference_points(spec: BoardSpec) -> np.ndarray:
    """P0, P1, P2 in the board frame (metres): inner chessboard corners, as calibrate_T_base_cam.py
    has always used them. P1 is along x from P0, P2 along y."""
    s = spec.square_m
    return np.array([[s, s, 0.0],
                     [(spec.squares_x - 1) * s, s, 0.0],
                     [s, (spec.squares_y - 1) * s, 0.0]], dtype=np.float64)


def reference_distances_mm(spec: BoardSpec) -> tuple[float, float]:
    """(|P0 P1|, |P0 P2|) in mm: what the TCP readings must reproduce. A2 gives (240, 300)."""
    p = reference_points(spec)
    return (float(np.linalg.norm(p[1] - p[0]) * 1000.0), float(np.linalg.norm(p[2] - p[0]) * 1000.0))
