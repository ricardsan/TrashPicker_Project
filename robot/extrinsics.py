"""The T_base_cam a run loads by default, calibration.txt -> T_base_cam, and what applies it.

T_base_cam takes a point in the camera's frame to the robot's base frame. It is **in metres**:
camera/calibrate_T_base_cam.py solves it from board corners and TCP readings both converted to
metres, and the translation in calibration.txt is the camera's position, (-0.1627, 0.3009, 1.3917).
RAPID wants millimetres. So the order is fixed:

    p_base_mm = (T_base_cam @ [X_m, Y_m, Z_m, 1])[:3] * 1000

Converting to millimetres *before* applying T looks equivalent and is not: the rotation does not
care, but the translation is then added in the wrong unit, and the point lands about 1.4 m from
where it should. camera/live_grasp.py's pixel_to_base has that order, for a T in millimetres.

The frame is the **colour** camera's optical frame: the calibration solved the board pose with the
colour stream's intrinsics. Every depth map in this project is aligned to colour before it gets here,
so a deprojected point is already in that frame.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "calibration.txt"

MM_PER_M = 1000.0

# #################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>  HARDCODED EXTRINSICS -- WHAT THE PROGRAM LOADS  <<<<<<<<<<<<<<<<<<<<<<<<
#
# robot_trash_picker.py runs with these unless --calibration says otherwise. They are set by hand for
# the simulation and describe no measured rig, so the client refuses them for any host but this machine.
#
#   position   the camera at x = -390 mm, y = -390 mm: chosen, not measured.
#   tilt       13.46 degrees from straight down, from the sheet in 20261016_231841_clutter_med.
#   height     789.8 mm, which puts the surface at z = 0. It is the camera's height above the sheet at
#              the clip's top-left corner: the median of the top-left 21 x 21 pixels at 640 x 360, over
#              its 128 settled frames. A plane fitted to the whole sheet gives 789.5 mm.
#   heading    calibration.txt's, which a clip cannot measure: the smallest rotation gives that tilt.
#
# RAPID's Z_TABLE and the z guard's floor still assume a table at 70 mm, not this surface at 0.
HARDCODED_T_BASE_CAM = np.array([[-0.019634, -0.972549, 0.231866, -0.3900],
                                 [-0.999495, 0.024892, 0.019770, -0.3900],
                                 [-0.024999, -0.231361, -0.972547, 0.7898],
                                 [0.0, 0.0, 0.0, 1.0]])
HARDCODED_NAME = "the hardcoded extrinsics in robot/extrinsics.py"
# #################################################################################################

# calibration.txt carries four decimals, so R'R misses the identity by around 1e-4.
ORTHONORMAL_TOLERANCE = 1e-3
# The camera is bolted up in a room, so a translation beyond this means the file is in millimetres.
MAX_TRANSLATION_M = 5.0

# The first word of a file that says it is not a real calibration: calibration_rosbag_demo.txt.
SYNTHETIC_MARKER = "SYNTHETIC"

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_ROW = re.compile(rf"\[\s*({_NUMBER}(?:[\s,]+{_NUMBER}){{3}})\s*\]")


def parse_calibration_text(text: str, source="calibration") -> np.ndarray:
    """The 4x4 matrix out of calibration.txt, which is prose around a printed numpy array."""
    rows = [[float(value) for value in re.split(r"[\s,]+", match.group(1).strip())]
            for match in _ROW.finditer(text)]
    if len(rows) != 4:
        raise SystemExit(f"{source}: expected the four rows of T_base_cam, found {len(rows)}")
    return np.array(rows, dtype=np.float64)


def check_transform(T: np.ndarray, source="calibration") -> np.ndarray:
    """T itself, or a complaint. A wrong transform would otherwise surface as a robot move."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise SystemExit(f"{source}: T_base_cam must be 4x4, not {T.shape}")
    if not np.isfinite(T).all():
        raise SystemExit(f"{source}: T_base_cam holds a value that is not a number")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
        raise SystemExit(f"{source}: the last row of T_base_cam must be 0 0 0 1, not {T[3]}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=ORTHONORMAL_TOLERANCE):
        raise SystemExit(f"{source}: the rotation part of T_base_cam is not orthonormal")
    if np.linalg.det(R) < 0:
        raise SystemExit(f"{source}: the rotation part of T_base_cam is a reflection")
    if np.abs(T[:3, 3]).max() > MAX_TRANSLATION_M:
        raise SystemExit(f"{source}: translation {T[:3, 3]} is too large for metres; T_base_cam "
                         f"must be in metres, not millimetres")
    return T


def load_T_base_cam(path=DEFAULT_CALIBRATION) -> np.ndarray:
    """T_base_cam from a calibration.txt, or from the .npy camera/calibrate_T_base_cam.py writes."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no calibration file at {path}")
    if path.suffix == ".npy":
        T = np.load(path)
    else:
        T = parse_calibration_text(path.read_text(encoding="utf-8", errors="replace"), path)
    return check_transform(T, path)


def is_synthetic(path) -> bool:
    """True for a calibration file that says it does not describe anything in the room."""
    path = Path(path)
    if path.suffix == ".npy":
        return False
    return path.read_text(encoding="utf-8", errors="replace").lstrip().startswith(SYNTHETIC_MARKER)


def camera_to_base_mm(T_base_cam: np.ndarray, xyz_cam_m) -> tuple[float, float, float]:
    """A point in the camera's frame, in metres, in the robot's base frame, in millimetres."""
    X, Y, Z = xyz_cam_m
    base_m = T_base_cam @ np.array([X, Y, Z, 1.0], dtype=np.float64)
    return (float(base_m[0] * MM_PER_M), float(base_m[1] * MM_PER_M), float(base_m[2] * MM_PER_M))


def rays_at_height_mm(T_base_cam: np.ndarray, rays_cam, z_base_mm: float) -> np.ndarray | None:
    """Where rays from the camera cross the horizontal plane z_base_mm of the base: N x 3, in mm.

    `rays_cam` is N x 3, directions in the camera's frame. None if any of them runs along the plane or
    points away from it. A point at distance s along a ray has base z = R[2] @ (s * ray) + t[2], so s
    comes from T's third row alone. The order is the same as camera_to_base_mm's: T, then millimetres.
    """
    rays = np.asarray(rays_cam, dtype=np.float64).reshape(-1, 3)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    climb = rays @ R[2]
    if np.any(np.abs(climb) < 1e-9):
        return None
    distance = (z_base_mm / MM_PER_M - t[2]) / climb
    if np.any(distance <= 0):
        return None
    return ((distance[:, None] * rays) @ R.T + t) * MM_PER_M
