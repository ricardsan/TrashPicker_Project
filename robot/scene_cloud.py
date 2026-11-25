"""The scene as a coloured point cloud in the robot's base frame, and files RobotStudio can import.

One frame's aligned depth map, deprojected through the colour intrinsics, then moved into the base
frame with T_base_cam -- the same chain perception.py uses for a single target, applied to every
pixel. So the cloud and the targets agree by construction: a target sits on its object in the cloud
exactly as far as the calibration lets it.

Deprojection here is the pinhole model and ignores distortion. The RealSense colour stream reports
zero coefficients in every clip in rosbags/, and align_to_recording.sheet_points makes the same
assumption; a frame with non-zero coefficients is refused rather than drawn slightly wrong.

Files
-----
    write_ply   binary PLY, metres. Opens in CloudCompare, MeshLab and Open3D; CloudCompare can
                convert it to .e57 for RobotStudio's point-cloud import if .pts is not accepted.
    write_pts   ASCII .pts: a count line, then "x y z intensity r g b" per point, in metres.
                RobotStudio (Home > Import Geometry > point cloud) reads this format.

Both are in the robot BASE frame, so imported at the station origin -- the base -- the cloud lands
where the robot will reach for it. If the station's robot base is not at its origin, attach the
imported cloud to the robot's base frame instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MM_PER_M = 1000.0
DEFAULT_STRIDE = 2          # every 2nd pixel each way: ~58k points at 640x360, plenty for a picture
MAX_RANGE_M = 3.0           # past this, a D435i reading at this baseline is mostly noise


def _intrinsic(intrinsics, name: str) -> float:
    return float(intrinsics[name] if isinstance(intrinsics, dict) else getattr(intrinsics, name))


def require_pinhole(intrinsics) -> None:
    """ValueError unless the distortion coefficients are all zero, so the pinhole model is exact.

    Also what keeps the floor picture and the cloud in step (station_feed.py): both are built with
    the pinhole model, so neither may be used on a frame where it is not the camera's.
    """
    coeffs = intrinsics["coeffs"] if isinstance(intrinsics, dict) else getattr(intrinsics, "coeffs", [])
    if any(coeffs):
        raise ValueError(f"distortion coefficients {list(coeffs)}: the pinhole deprojection here "
                         f"would be wrong for them")


def cloud_in_camera(metres: np.ndarray, intrinsics, stride: int = DEFAULT_STRIDE,
                    max_range_m: float = MAX_RANGE_M) -> tuple[np.ndarray, np.ndarray]:
    """(N x 3 points in the camera frame in metres, N x 2 their (row, column)), valid depth only."""
    require_pinhole(intrinsics)
    fx, fy = _intrinsic(intrinsics, "fx"), _intrinsic(intrinsics, "fy")
    ppx, ppy = _intrinsic(intrinsics, "ppx"), _intrinsic(intrinsics, "ppy")
    stride = max(1, int(stride))
    rows, columns = np.mgrid[0:metres.shape[0]:stride, 0:metres.shape[1]:stride]
    depth = metres[rows, columns]
    valid = (depth > 0) & (depth < max_range_m)
    rows, columns, depth = rows[valid], columns[valid], depth[valid]
    points = np.column_stack([(columns - ppx) / fx * depth, (rows - ppy) / fy * depth, depth])
    return points, np.column_stack([rows, columns])


def cloud_in_base(image_bgr: np.ndarray, metres: np.ndarray, intrinsics, T_base_cam: np.ndarray,
                  stride: int = DEFAULT_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """(N x 3 points in the base frame in MILLIMETRES, N x 3 RGB uint8 colours)."""
    if image_bgr.shape[:2] != metres.shape[:2]:
        raise ValueError(f"image {image_bgr.shape[:2]} and depth {metres.shape[:2]} are not aligned")
    points_cam, pixels = cloud_in_camera(metres, intrinsics, stride)
    T = np.asarray(T_base_cam, dtype=np.float64)
    points_base = (points_cam @ T[:3, :3].T + T[:3, 3]) * MM_PER_M
    colours = image_bgr[pixels[:, 0], pixels[:, 1]][:, ::-1].astype(np.uint8)
    return points_base, colours


def write_ply(path, points_mm: np.ndarray, colours: np.ndarray) -> Path:
    """Binary little-endian PLY, in metres."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = np.empty(len(points_mm), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    metres = np.asarray(points_mm, dtype=np.float64) / MM_PER_M
    records["x"], records["y"], records["z"] = metres[:, 0], metres[:, 1], metres[:, 2]
    records["red"], records["green"], records["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment TrashPicker scene, robot base frame, metres\n"
              f"element vertex {len(records)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(records.tobytes())
    return path


def write_pts(path, points_mm: np.ndarray, colours: np.ndarray) -> Path:
    """ASCII .pts: count, then "x y z intensity r g b" per line, in metres."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metres = np.asarray(points_mm, dtype=np.float64) / MM_PER_M
    intensity = np.zeros((len(metres), 1))
    table = np.hstack([metres, intensity, colours.astype(np.float64)])
    with open(path, "w", encoding="ascii") as handle:
        handle.write(f"{len(table)}\n")
        np.savetxt(handle, table, fmt=["%.5f", "%.5f", "%.5f", "%d", "%d", "%d", "%d"])
    return path
