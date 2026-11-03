"""Write calibration_rosbag_demo.txt: calibration.txt, levelled to the sheet in the recorded clips.

    python robot/align_to_recording.py

The clips in rosbags/ were not filmed where calibration.txt says the camera is. Through the real
calibration their sheet reads about z = 628 mm, tilted, and above the approach plane, so going to it
climbs. A clip can measure the camera's relation to the sheet and nothing else: how far above it the
camera is, and which way is up. So this corrects exactly that.

    measured     the height and the tilt. The sheet comes out level, at the table's z, and objects
                 keep their real heights above it and their real distances from each other.
    not measured the heading is kept from calibration.txt, and x and y are chosen so that the objects
                 land inside the workspace. No clip holds a robot or a board to tie them to.

So the result is still synthetic, its positions correspond to nothing in the room, and the client
refuses it for any host other than this machine. It is for showing a whole pick in RobotStudio.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.segmentation_realsense import depth_map_metres  # noqa: E402
from robot.art_camera_client import Z_TABLE_MM  # noqa: E402
from robot.extrinsics import (DEFAULT_CALIBRATION, MM_PER_M, SYNTHETIC_MARKER,  # noqa: E402
                              load_T_base_cam)
from robot.frame_selection import ROSBAG_DIR, open_recording, started  # noqa: E402

DEFAULT_OUTPUT = DEFAULT_CALIBRATION.parent / "calibration_rosbag_demo.txt"

# Where the optical axis meets the sheet, chosen and not measured. It is what puts most of each clip's
# objects inside the workspace: 16 of 24 detections on frame 20 pass both x/y and the z guard.
IMAGE_CENTRE_AT_MM = (450.0, 100.0)

# Past the depth's first second (see frame_selection.SETTLE_S), and spread out through the clip.
SETTLED_FRAMES = (20, 60, 100, 140)
SAMPLE_POINTS = 30000
INLIER_M = 0.010
RANSAC_ROUNDS = 600
REFINE_ROUNDS = 4
SEED = 0

# One transform has to fit every clip. These are far above what they agree to, which is 0.25 degrees
# and 0.9 mm, and far below a camera that was actually moved between them.
MAX_TILT_SPREAD_DEG = 1.0
MAX_HEIGHT_SPREAD_M = 0.010


@dataclass(frozen=True)
class Plane:
    """A plane seen by the camera, in the camera's frame, which looks along +z."""

    up: np.ndarray          # unit normal, pointing towards the camera
    height_m: float         # how far above the plane the camera is
    on_plane: float         # the fraction of the points that belong to it
    rms_m: float            # how far those points are from it


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _least_squares(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a point on the plane, its unit normal), with the normal's sign arbitrary."""
    centre = points.mean(axis=0)
    return centre, np.linalg.svd(points - centre, full_matrices=False)[2][2]


def fit_plane(points: np.ndarray, rng: np.random.Generator) -> Plane:
    """The plane most of `points` lie on, past whatever stands on it: RANSAC, then least squares."""
    if len(points) < 3:
        raise SystemExit(f"cannot fit a plane to {len(points)} points")
    best = None
    for _ in range(RANSAC_ROUNDS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        on_plane = np.abs((points - a) @ (normal / length)) < INLIER_M
        if best is None or on_plane.sum() > best.sum():
            best = on_plane
    if best is None:
        raise SystemExit("cannot fit a plane: the points are all on one line")

    for _ in range(REFINE_ROUNDS):
        centre, normal = _least_squares(points[best])
        best = np.abs((points - centre) @ normal) < INLIER_M
    centre, normal = _least_squares(points[best])
    # The camera looks along +z, so anything pointing back at it has a negative z.
    up = -normal if normal[2] > 0 else normal
    return Plane(up, float(-(centre @ up)), float(best.mean()),
                 float(((points[best] - centre) @ up).std()))


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """The closest true rotation. calibration.txt's four decimals leave its R slightly off one."""
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def level_transform(T_base_cam: np.ndarray, up_cam: np.ndarray, height_m: float, z_table_m: float,
                    image_centre_mm: tuple[float, float]) -> np.ndarray:
    """T_base_cam, turned as little as possible so that the plane (up_cam, height_m) is level at z_table_m.

    The smallest rotation is what keeps the calibrated heading. Afterwards T's third row is up_cam, so
    a point on the plane, where up_cam @ p = -height_m, has base z = -height_m + t_z. That fixes t_z.
    """
    R = _nearest_rotation(T_base_cam[:3, :3])
    up_cam = np.asarray(up_cam, dtype=np.float64) / np.linalg.norm(up_cam)
    seen = R @ up_cam
    plus_z = np.array([0.0, 0.0, 1.0])
    cosine = float(seen @ plus_z)
    if cosine <= 0.0:
        raise SystemExit(f"T_base_cam has the plane's up pointing at {seen} in the base, which is not "
                         f"a tilt to correct: the two do not describe the same kind of view")
    axis = np.cross(seen, plus_z)
    sine = float(np.linalg.norm(axis))
    if sine < 1e-12:
        correction = np.eye(3)
    else:
        x, y, z = axis / sine
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        correction = np.eye(3) + sine * K + (1.0 - cosine) * K @ K
    level = _nearest_rotation(correction @ R)

    # The optical axis reaches the plane at this point, measured from the camera.
    centre = level @ np.array([0.0, 0.0, height_m / -up_cam[2]])
    T = np.eye(4)
    T[:3, :3] = level
    T[:3, 3] = (image_centre_mm[0] / MM_PER_M - centre[0], image_centre_mm[1] / MM_PER_M - centre[1],
                z_table_m + height_m)
    return T


def sheet_points(metres: np.ndarray, intrinsics) -> np.ndarray:
    """Every pixel with a depth, in the camera's frame, in metres.

    This is rs2_deproject_pixel_to_point only when the distortion coefficients are zero, as they are in
    all three clips, and it is what makes tens of thousands of points cheap.
    """
    if any(intrinsics.coeffs):
        raise SystemExit(f"this clip has distortion coefficients {list(intrinsics.coeffs)}, and "
                         f"sheet_points assumes none")
    rows, columns = np.nonzero(metres > 0)
    depth = metres[rows, columns]
    return np.column_stack([(columns - intrinsics.ppx) / intrinsics.fx * depth,
                            (rows - intrinsics.ppy) / intrinsics.fy * depth, depth])


def measure_clip(name: str, rng: np.random.Generator) -> list[Plane]:
    """The sheet on each of SETTLED_FRAMES of one clip."""
    source = open_recording(name)
    planes = []
    index = 0
    with started(source):
        while index <= max(SETTLED_FRAMES):
            try:
                grabbed = source.grab()
            except StopIteration:
                raise SystemExit(f"{source.describe()} ran out before frame "
                                 f"{max(SETTLED_FRAMES)}") from None
            if grabbed is None:
                continue
            if index in SETTLED_FRAMES:
                _, depth, intrinsics = grabbed
                points = sheet_points(depth_map_metres(depth), intrinsics)
                sample = points[rng.choice(len(points), min(SAMPLE_POINTS, len(points)), replace=False)]
                planes.append(fit_plane(sample, rng))
            index += 1
    return planes


def combine(planes: list[Plane]) -> tuple[np.ndarray, float]:
    """(up, height) for all of them together, or a refusal if they are not one view."""
    ups = np.array([plane.up for plane in planes])
    heights = np.array([plane.height_m for plane in planes])
    up = ups.mean(axis=0)
    up /= np.linalg.norm(up)
    height = float(heights.mean())
    tilt_spread = max(angle_between_deg(other, up) for other in ups)
    height_spread = float(np.abs(heights - height).max())
    if tilt_spread > MAX_TILT_SPREAD_DEG or height_spread > MAX_HEIGHT_SPREAD_M:
        raise SystemExit(f"the clips do not agree on one camera position: up to {tilt_spread:.2f} "
                         f"degrees and {height_spread * MM_PER_M:.1f} mm apart, where one transform "
                         f"allows {MAX_TILT_SPREAD_DEG} degrees and "
                         f"{MAX_HEIGHT_SPREAD_M * MM_PER_M:.0f} mm")
    return up, height


def format_calibration(T: np.ndarray, T_from: np.ndarray, up_cam: np.ndarray, height_m: float,
                       clips, image_centre_mm: tuple[float, float]) -> str:
    """The file: says what it is first, because is_synthetic reads its first word."""
    tilt = angle_between_deg(up_cam, np.array([0.0, 0.0, -1.0]))
    # The angle of the rotation taking one to the other, from its trace.
    relative = T[:3, :3] @ T_from[:3, :3].T
    turned = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
    rows = "\n ".join("[" + " ".join(f"{value: .6f}" for value in row) + "]" for row in T[:3])
    return (
        f"{SYNTHETIC_MARKER}. NOT A CALIBRATION. NEVER USE THIS ON THE REAL ROBOT.\n"
        f"\n"
        f"Written by robot/align_to_recording.py. To change it, change that and run it again.\n"
        f"This file exists only so that a clip can show a whole pick in RobotStudio. The client refuses\n"
        f"it for any host other than this machine.\n"
        f"\n"
        f"Measured from {', '.join(clips)}:\n"
        f"  the camera is {height_m * MM_PER_M:.1f} mm above the sheet, tilted {tilt:.2f} degrees from\n"
        f"  straight down. calibration.txt is turned by {turned:.2f} degrees, and placed in height, so\n"
        f"  that the sheet comes out level at z = {Z_TABLE_MM:.0f} mm.\n"
        f"Not measured, because no clip holds anything that ties it to the robot:\n"
        f"  the heading is calibration.txt's, and x and y are chosen so that the optical axis meets the\n"
        f"  sheet at x = {image_centre_mm[0]:.0f} mm, y = {image_centre_mm[1]:.0f} mm.\n"
        f"So heights above the sheet and distances between objects are real. Where they are in the room\n"
        f"is not.\n"
        f"\n"
        f"T_base_cam:\n"
        f"[{rows}\n"
        f" [ 0 0 0 1]]\n")


def available_clips() -> list[str]:
    clips = sorted(path.name for path in ROSBAG_DIR.glob("*.db3"))
    if not clips:
        raise SystemExit(f"no clips in {ROSBAG_DIR}")
    return clips


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the demo calibration from the recorded clips.")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION, metavar="FILE",
                        help=f"the calibration to level (default: {DEFAULT_CALIBRATION.name})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, metavar="FILE",
                        help=f"where to write it (default: {DEFAULT_OUTPUT.name})")
    args = parser.parse_args(argv)

    T_from = load_T_base_cam(args.calibration)
    rng = np.random.default_rng(SEED)
    clips, planes = available_clips(), []
    for clip in clips:
        for plane in measure_clip(clip, rng):
            print(f"{clip}: {plane.height_m * MM_PER_M:.1f} mm above the sheet, "
                  f"{plane.on_plane * 100:.1f}% of the points on it, rms "
                  f"{plane.rms_m * MM_PER_M:.1f} mm")
            planes.append(plane)

    up, height = combine(planes)
    T = level_transform(T_from, up, height, Z_TABLE_MM / MM_PER_M, IMAGE_CENTRE_AT_MM)
    text = format_calibration(T, T_from, up, height, [Path(clip).stem for clip in clips],
                              IMAGE_CENTRE_AT_MM)
    Path(args.output).write_text(text, encoding="utf-8")
    # Reading it back runs the same checks a robot run would.
    load_T_base_cam(args.output)
    print(f"\nwrote {args.output}\n{T}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
