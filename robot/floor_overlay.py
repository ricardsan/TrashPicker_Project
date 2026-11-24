"""The camera picture laid flat on the floor, for the textured surface in the RobotStudio station.

Each floor point (x, y, 0) of the base frame is projected into the colour image with T_base_cam and
the camera's intrinsics, and its colour copied. The result is the picture where the robot sees it, so
a TCP sent to an object's x, y should sit on that object in the station.

The rectangle is fixed (FLOOR_X_MM, FLOOR_Y_MM, 1 px = 1 mm), so the surface in the station is set up
once and only its texture is reloaded. Floor the camera does not see stays grey.

What a run writes, on 'n' (robot_trash_picker.py), and a calibration once it is saved
(board_demo_calibration.py, camera/calibrate_T_base_cam.py), with its P0 P1 P2 marked:

    robot/robotstudio/live/floor.png            for RobotStudio: already turned for its texture mapping
    robot/robotstudio/live/floor_check.png      for people: +X up, +Y left, a 100 mm grid
    robot/robotstudio/live/history/...          a copy of both with a timestamp

On top of the camera picture, each target gets, at its xyz_base_mm -- where the TCP is sent:
a cross and its rank, the object's long axis (thin) and the line the fingers close along (thick).

The picture is laid on z = 0. An object's top is higher, so a tall object leans away from the camera
in it, by height x (horizontal distance from the camera / camera height). The cross does not lean:
it is the x, y the robot gets. For a can lying down that is about 20 mm; for a bottle standing up,
much more. Lay objects flat for the x, y check.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from robot.pick_sequence import FINGER_CLOSE_AXIS_AT_RZ0_DEG, gripper_rz_deg

FLOOR_X_MM = (-40.0, 1380.0)            # the surface in the station: X from the robot to the far end
FLOOR_Y_MM = (-390.0, 380.0)
MM_PER_PX = 1.0
UNSEEN_BGR = (60, 60, 60)

LIVE_DIR = Path(__file__).resolve().parent / "robotstudio" / "live"

CROSS_BGR = (0, 0, 255)
AXIS_BGR = (255, 255, 0)
FINGERS_BGR = (0, 255, 0)
AXIS_HALF_MM = 80.0
FINGERS_HALF_MM = 60.0


def floor_size_px() -> tuple[int, int]:
    """(rows, columns) of the unturned picture: rows along X, columns along Y."""
    rows = int(round((FLOOR_X_MM[1] - FLOOR_X_MM[0]) / MM_PER_PX))
    columns = int(round((FLOOR_Y_MM[1] - FLOOR_Y_MM[0]) / MM_PER_PX))
    return rows, columns


def floor_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    """(column, row) of a floor point in the unturned picture, for cv2's drawing functions.

    Row 0 is the largest X, far from the robot; column 0 the largest Y. Seen from above with +X up,
    +Y is to the left, so the picture is the floor as seen from above.
    """
    return (int(round((FLOOR_Y_MM[1] - y_mm) / MM_PER_PX)),
            int(round((FLOOR_X_MM[1] - x_mm) / MM_PER_PX)))


def camera_matrix(intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """K and the distortion coefficients out of a RealSense intrinsics (or anything with its fields)."""
    K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                  [0.0, intrinsics.fy, intrinsics.ppy],
                  [0.0, 0.0, 1.0]])
    dist = np.asarray(getattr(intrinsics, "coeffs", None) or np.zeros(5), dtype=np.float64)
    return K, dist


def lay_on_floor(image: np.ndarray, T_base_cam: np.ndarray, intrinsics) -> np.ndarray:
    """The unturned floor picture: each floor pixel takes the colour the camera sees there.

    T_base_cam is in metres, as everywhere (robot/extrinsics.py).
    """
    K, dist = camera_matrix(intrinsics)
    rows, columns = floor_size_px()
    xs = FLOOR_X_MM[1] - (np.arange(rows) + 0.5) * MM_PER_PX
    ys = FLOOR_Y_MM[1] - (np.arange(columns) + 0.5) * MM_PER_PX
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    base_m = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3) / 1000.0

    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    cam_m = (base_m - t) @ R                                # R^T (p - t), row by row
    in_front = cam_m[:, 2] > 1e-6
    uv = np.full((cam_m.shape[0], 2), -1.0, dtype=np.float32)
    if in_front.any():
        projected, _ = cv2.projectPoints(cam_m[in_front], np.zeros(3), np.zeros(3), K, dist)
        uv[in_front] = projected.reshape(-1, 2)
    uv = uv.reshape(rows, columns, 2)
    return cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=UNSEEN_BGR)


def _line_through(picture, x, y, angle_deg, half_mm, colour, thickness):
    """A line on the floor through (x, y), along angle_deg from +x towards +y."""
    dx, dy = np.cos(np.radians(angle_deg)) * half_mm, np.sin(np.radians(angle_deg)) * half_mm
    cv2.line(picture, floor_px(x - dx, y - dy), floor_px(x + dx, y + dy), colour, thickness,
             cv2.LINE_AA)
    return (x - dx, y - dy), (x + dx, y + dy)


def draw_targets_on_floor(picture: np.ndarray, targets) -> np.ndarray:
    """Each target where the TCP is sent: cross, rank, long axis and the fingers' closing line."""
    for rank, target in enumerate(targets, 1):
        x, y, _ = target.xyz_base_mm
        if target.base_axis_deg is not None:
            _line_through(picture, x, y, target.base_axis_deg, AXIS_HALF_MM, AXIS_BGR, 2)
            rz = gripper_rz_deg(target.base_axis_deg)
            ends = _line_through(picture, x, y, FINGER_CLOSE_AXIS_AT_RZ0_DEG + rz, FINGERS_HALF_MM,
                                 FINGERS_BGR, 4)
            for end in ends:                                # the two fingers
                cv2.circle(picture, floor_px(*end), 9, FINGERS_BGR, -1, cv2.LINE_AA)
        centre = floor_px(x, y)
        cv2.drawMarker(picture, centre, CROSS_BGR, cv2.MARKER_CROSS, 40, 3, cv2.LINE_AA)
        cv2.putText(picture, f"{rank} {target.name}", (centre[0] + 14, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CROSS_BGR, 2, cv2.LINE_AA)
    return picture


def draw_points_on_floor(picture: np.ndarray, points) -> np.ndarray:
    """Labelled points on the floor: (label, x_mm, y_mm, bgr) each, e.g. a calibration's P0 P1 P2."""
    for label, x, y, colour in points:
        centre = floor_px(x, y)
        cv2.circle(picture, centre, 8, colour, 2, cv2.LINE_AA)
        cv2.drawMarker(picture, centre, colour, cv2.MARKER_CROSS, 10, 1)
        cv2.putText(picture, label, (centre[0] + 10, centre[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2, cv2.LINE_AA)
    return picture


def for_robotstudio(picture: np.ndarray) -> np.ndarray:
    """Turned 90 degrees clockwise: how the station's surface maps its texture (checked by eye)."""
    return np.ascontiguousarray(np.rot90(picture, 3))


def with_grid(picture: np.ndarray) -> np.ndarray:
    """A copy for people: a 100 mm grid and labels on the unturned picture."""
    check = picture.copy()
    rows, _ = floor_size_px()
    for x in range(int(np.ceil(FLOOR_X_MM[0] / 100) * 100), int(FLOOR_X_MM[1]) + 1, 100):
        cv2.line(check, floor_px(x, FLOOR_Y_MM[1]), floor_px(x, FLOOR_Y_MM[0]), (200, 200, 200), 1)
        cv2.putText(check, f"x={x}", (5, floor_px(x, 0)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
    for y in range(int(np.ceil(FLOOR_Y_MM[0] / 100) * 100), int(FLOOR_Y_MM[1]) + 1, 100):
        cv2.line(check, floor_px(FLOOR_X_MM[1], y), floor_px(FLOOR_X_MM[0], y), (200, 200, 200), 1)
        cv2.putText(check, f"y={y}", (floor_px(0, y)[0] + 4, rows - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
    cv2.drawMarker(check, floor_px(0, 0), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
    cv2.putText(check, "+X up, +Y left, seen from above. Red cross = where the TCP is sent.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return check


def write_floor_overlay(image: np.ndarray, T_base_cam: np.ndarray, intrinsics, targets=(),
                        out_dir: Path | None = None, label: str = "detection",
                        points=()) -> Path:
    """Write floor.png (for RobotStudio), floor_check.png and timestamped copies. Returns floor.png.

    `targets` are a run's (crosses, axes, fingers); `points` are labelled marks, as a calibration's.
    """
    picture = lay_on_floor(image, T_base_cam, intrinsics)
    picture = draw_targets_on_floor(draw_points_on_floor(picture, points), targets)
    out_dir = Path(LIVE_DIR if out_dir is None else out_dir)
    history = out_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    station, check = for_robotstudio(picture), with_grid(picture)
    floor = out_dir / "floor.png"
    for path, content in ((floor, station), (out_dir / "floor_check.png", check),
                          (history / f"{stamp}_{label}_floor.png", station),
                          (history / f"{stamp}_{label}_floor_check.png", check)):
        if not cv2.imwrite(str(path), content):
            raise OSError(f"could not write {path}")
    return floor
