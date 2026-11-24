"""A demo calibration from a ChArUco board in a recorded clip, for a camera with no robot beside it.

    python robot/board_demo_calibration.py --rosbag rosbags/home_2026-11/<..>_board_only.db3 --board home40   # or home50
    python robot/board_demo_calibration.py --rosbag <bag> --board a2 --board-at 450,100 --board-yaw 0

camera/calibrate_T_base_cam.py is the real calibration, and needs the robot: the TCP goes onto three
board corners. At home there is no robot, but the board still gives the camera's whole pose relative
to it -- height, tilt and heading -- which robot/align_to_recording.py's plane fit cannot (it measures
height and tilt only). So: say where the board lies in the robot's base frame (--board-at, --board-yaw;
it lies flat on the table at Z_TABLE_MM), and

    T_base_cam = T_base_board @ inverse(T_cam_board)

Every object then lands where it really was *relative to the board*. Where the board is relative to
the robot is chosen, not measured, so the file is SYNTHETIC: robot_trash_picker.py refuses it for any
host but this machine. It is for RobotStudio. Record the board in its own clip and take it away for
the object clips, which must be filmed without moving the camera.

Measured instead of chosen: mark a cross on the floor where the robot base is, an x line and a y
line from it, and measure with a tape where the board's corners P0, P1 and P2 are (see
camera/charuco.py reference_points), in mm from the cross:

    python robot/board_demo_calibration.py --rosbag <bag> --board home40 --points 520,40 520,-121 400,40 --camera-at 318,-141,796

The board's place then comes from those three points. --camera-at, the camera's colour lens measured
the same way, is only compared with where the board puts the camera.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.charuco import (BOARDS, BoardPose, BoardSpec, camera_matrix, detect_pose,  # noqa: E402
                            make_board, reference_distances_mm, reference_points)
from camera.rosbag_frames import default_export_dir, export_bag, read_meta  # noqa: E402
from robot.art_camera_client import Z_TABLE_MM  # noqa: E402
from robot.extrinsics import MM_PER_M, SYNTHETIC_MARKER, load_T_base_cam  # noqa: E402
from robot.frame_selection import EXPORT_DIR, SETTLE_S, resolve_bag  # noqa: E402

DEFAULT_OUTPUT = ROOT / "robot" / "calibration_home_board.txt"
DEFAULT_BOARD_AT_MM = (450.0, 100.0)    # as align_to_recording.IMAGE_CENTRE_AT_MM
DEFAULT_STEP = 5                        # every 5th settled frame

# How far one frame's pose may sit from the combined one. A single image of a home board pins the
# pose to about 0.5-1.7 mm and 0.1-0.6 degrees (camera/tests/test_charuco.py, rendered at 0.78 m), so
# sensor noise alone moves frames by that much; a camera or board that moved does by far more.
MAX_SPREAD_MM = 3.0
MAX_SPREAD_DEG = 1.0
MAX_RMS_PX = 1.0
MIN_FRAMES = 3


def board_in_base(spec: BoardSpec, at_mm, yaw_deg: float, z_mm: float = Z_TABLE_MM) -> np.ndarray:
    """T_base_board for a board lying flat, its centre at (at_mm, z_mm), its x turned yaw_deg from base +x.

    The board's +z points into the board, so on a table it points down: the rotation is a turn about
    base z after flipping y and z.
    """
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([at_mm[0], at_mm[1], z_mm]) / MM_PER_M - R @ spec.centre_m
    return T


# How far a tape-measured corner may sit from where the fitted board puts it. A tape read to a
# couple of mm, from a cross a few mm wide, gives about this much; more is a misread or a wrong corner.
MAX_POINT_ERROR_MM = 15.0


@dataclass(frozen=True)
class PointFit:
    T_base_board: np.ndarray
    errors_mm: tuple[float, float, float]       # each measured point to the fitted board's corner
    distances_mm: tuple[float, float]           # |P0 P1|, |P0 P2| as measured
    expected_mm: tuple[float, float]            # the same on the board
    corner_deg: float                           # the angle at P0 as measured; 90 on the board
    yaw_deg: float                              # the board's x axis from base +x


def board_from_points(spec: BoardSpec, points_mm, z_mm: float = Z_TABLE_MM) -> PointFit:
    """T_base_board from corners P0, P1, P2 measured on the floor, in mm from the cross.

    The board lies flat and face up, so only its heading and place are fitted: a 2-D turn and shift of
    the three corners, least squares, never a mirror image. Face up, P2 is clockwise from P1 seen from
    above; measured the other way round, the y line on the floor or the order of the points is wrong.
    """
    measured = np.array([[float(x), float(y)] for x, y in points_mm])
    if measured.shape != (3, 2) or not np.isfinite(measured).all():
        raise SystemExit("--points needs three X,Y pairs: P0, P1 and P2")
    flat = reference_points(spec)[:, :2] * MM_PER_M * np.array([1.0, -1.0])   # board y is base -y, turned
    a, b = measured[1] - measured[0], measured[2] - measured[0]
    if a[0] * b[1] - a[1] * b[0] >= 0.0:
        raise SystemExit("the points are mirrored: seen from above, P2 must be clockwise from P1, as on "
                         "a board lying face up. Either P1 and P2 are swapped, or the y line on the "
                         "floor points the wrong way: +y is +x turned 90 degrees to the left.")
    h = (flat - flat.mean(axis=0)).T @ (measured - measured.mean(axis=0))
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    turn = np.array([[c, -s], [s, c]])
    shift = measured.mean(axis=0) - turn @ flat.mean(axis=0)
    errors = np.linalg.norm(flat @ turn.T + shift - measured, axis=1)

    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = np.array([shift[0], shift[1], z_mm]) / MM_PER_M
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    fit = PointFit(T, tuple(float(e) for e in errors),
                   (float(np.linalg.norm(a)), float(np.linalg.norm(b))), reference_distances_mm(spec),
                   math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), math.degrees(yaw))
    if max(fit.errors_mm) > MAX_POINT_ERROR_MM:
        raise SystemExit(f"the three points do not fit the board: they sit "
                         f"{', '.join(f'{e:.0f}' for e in fit.errors_mm)} mm from its corners, more than "
                         f"{MAX_POINT_ERROR_MM:.0f}. Measured |P0 P1| = {fit.distances_mm[0]:.0f} and "
                         f"|P0 P2| = {fit.distances_mm[1]:.0f} mm, where the board has "
                         f"{fit.expected_mm[0]:.0f} and {fit.expected_mm[1]:.0f}, and the angle at P0 is "
                         f"{fit.corner_deg:.0f} degrees, not 90. Measure again.")
    return fit


def compose(T_base_board: np.ndarray, T_cam_board: np.ndarray) -> np.ndarray:
    return T_base_board @ np.linalg.inv(T_cam_board)


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    cosine = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def combine_poses(poses: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """(one T_cam_board for all of them, translation spread in mm, rotation spread in degrees).

    Translation is the per-axis median, rotation the nearest rotation to the mean matrix. Refuses a set
    that does not describe one still board.
    """
    if len(poses) < MIN_FRAMES:
        raise SystemExit(f"the board was found on only {len(poses)} usable frames; at least "
                         f"{MIN_FRAMES} are needed. Is it fully in view, flat and well lit?")
    stack = np.array(poses)
    t = np.median(stack[:, :3, 3], axis=0)
    R = _nearest_rotation(stack[:, :3, :3].mean(axis=0))
    spread_mm = float(np.linalg.norm(stack[:, :3, 3] - t, axis=1).max() * MM_PER_M)
    spread_deg = max(_angle_deg(R, other) for other in stack[:, :3, :3])
    if spread_mm > MAX_SPREAD_MM or spread_deg > MAX_SPREAD_DEG:
        raise SystemExit(f"the board's pose moves between frames by up to {spread_mm:.1f} mm and "
                         f"{spread_deg:.2f} degrees, where a still camera allows {MAX_SPREAD_MM} mm and "
                         f"{MAX_SPREAD_DEG} degrees. Did the camera or the board move?")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, spread_mm, spread_deg


def height_and_tilt(T_cam_board: np.ndarray) -> tuple[float, float]:
    """(camera height above the board in mm, degrees between the optical axis and straight down)."""
    into_board = T_cam_board[:3, 2]
    height = abs(float(into_board @ T_cam_board[:3, 3])) * MM_PER_M
    tilt = float(np.degrees(np.arccos(np.clip(into_board[2], -1.0, 1.0))))
    return height, tilt


@dataclass(frozen=True)
class Measured:
    T_cam_board: np.ndarray
    frames_used: int
    frames_tried: int
    spread_mm: float
    spread_deg: float
    worst_rms_px: float


def measure_bag(bag: Path, spec: BoardSpec, step: int = DEFAULT_STEP) -> Measured:
    """The board's pose over the settled frames of one clip, from its native colour frames."""
    export_dir = default_export_dir(bag, EXPORT_DIR)
    export_bag(bag, export_dir, clouds=False)
    meta = read_meta(export_dir)
    intr = meta["color_intrinsics"]
    K = camera_matrix(intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
    dist = np.array(intr["coeffs"], dtype=np.float64)
    board, detector = make_board(spec)

    first = math.ceil(SETTLE_S * float(meta["stream_fps"]))
    records = meta["frames"][first::max(1, step)]
    poses: list[BoardPose] = []
    for record in records:
        image = cv2.imread(str(export_dir / record["color"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        pose = detect_pose(image, K, dist, board, detector)
        if pose is not None and pose.rms_px <= MAX_RMS_PX:
            poses.append(pose)
    T, spread_mm, spread_deg = combine_poses([pose.T_cam_board for pose in poses])
    return Measured(T, len(poses), len(records), spread_mm, spread_deg,
                    max(pose.rms_px for pose in poses))


def format_calibration(T: np.ndarray, measured: Measured, board_name: str, bag: Path, at_mm=None,
                       yaw_deg: float = 0.0, fit: PointFit | None = None) -> str:
    """The file. Its first word is the synthetic marker, which is what is_synthetic reads.

    No square brackets in the prose: parse_calibration_text takes any bracketed four numbers for a row.
    """
    height, tilt = height_and_tilt(measured.T_cam_board)
    rows = "\n ".join("[" + " ".join(f"{value: .6f}" for value in row) + "]" for row in T[:3])
    if fit is None:
        placement = (
            f"Chosen, not measured:\n"
            f"  the board's centre is at x = {at_mm[0]:.0f} mm, y = {at_mm[1]:.0f} mm, z = {Z_TABLE_MM:.0f} mm,\n"
            f"  its x axis turned {yaw_deg:.1f} degrees from the base's +x.\n"
            f"So positions relative to the board are real. Where the board is relative to the robot is not.\n")
    else:
        placement = (
            f"Measured with a tape from a cross on the floor, which stands for the robot base:\n"
            f"  corners P0 P1 P2 fit to {', '.join(f'{e:.1f}' for e in fit.errors_mm)} mm;\n"
            f"  P0 to P1 {fit.distances_mm[0]:.1f} mm, on the board {fit.expected_mm[0]:.1f};\n"
            f"  P0 to P2 {fit.distances_mm[1]:.1f} mm, on the board {fit.expected_mm[1]:.1f};\n"
            f"  angle at P0 {fit.corner_deg:.1f} degrees; board x axis {fit.yaw_deg:.1f} degrees from base +x.\n"
            f"No robot touched the board, so this is still not a robot calibration.\n")
    return (
        f"{SYNTHETIC_MARKER}. NOT A CALIBRATION. NEVER USE THIS ON THE REAL ROBOT.\n"
        f"\n"
        f"Written by robot/board_demo_calibration.py from the ChArUco board in {bag.stem}.\n"
        f"The client refuses it for any host other than this machine. It is for RobotStudio.\n"
        f"\n"
        f"Measured, board {board_name}, {measured.frames_used} of {measured.frames_tried} frames used:\n"
        f"  the camera is {height:.1f} mm above the board, tilted {tilt:.2f} degrees from straight down.\n"
        f"  Frames agree to {measured.spread_mm:.2f} mm and {measured.spread_deg:.3f} degrees; worst\n"
        f"  reprojection {measured.worst_rms_px:.2f} px.\n"
        f"{placement}"
        f"\n"
        f"T_base_cam:\n"
        f"[{rows}\n"
        f" [ 0 0 0 1]]\n")


BOARD_POINT_BGR = (255, 0, 255)          # where the fitted board puts P0 P1 P2
TAPE_POINT_BGR = (0, 255, 0)             # where the tape put them


def show_in_station(bag: Path, spec: BoardSpec, T_base_cam: np.ndarray, T_base_board: np.ndarray,
                    tape_points=None) -> None:
    """The clip's last frame laid on the floor for RobotStudio, with P0 P1 P2 marked.

    A picture that cannot be written costs a warning, never the calibration.
    """
    from robot.floor_overlay import write_floor_overlay      # it imports the pick code; only here

    try:
        export_dir = default_export_dir(bag, EXPORT_DIR)
        meta = read_meta(export_dir)
        image = cv2.imread(str(export_dir / meta["frames"][-1]["color"]))
        if image is None:
            raise OSError("the clip's last colour frame did not load")
        intr = meta["color_intrinsics"]
        intrinsics = SimpleNamespace(fx=intr["fx"], fy=intr["fy"], ppx=intr["ppx"], ppy=intr["ppy"],
                                     coeffs=intr["coeffs"])
        on_board = (T_base_board @ np.hstack([reference_points(spec), np.ones((3, 1))]).T).T
        points = [(f"P{i}", p[0] * MM_PER_M, p[1] * MM_PER_M, BOARD_POINT_BGR)
                  for i, p in enumerate(on_board)]
        if tape_points is not None:
            points += [(f"P{i} tape", x, y, TAPE_POINT_BGR) for i, (x, y) in enumerate(tape_points)]
        floor = write_floor_overlay(image, T_base_cam, intrinsics, points=points,
                                    label="calibration")
    except (Exception, SystemExit) as failure:     # noqa: BLE001 - only a picture; read_meta exits
        print(f"WARNING: no floor picture for RobotStudio: {failure}")
        return
    print(f"floor picture for RobotStudio: {floor}  (P0 P1 P2 in magenta where the board is, "
          f"green where the tape put them)")


def _xy(text: str) -> tuple[float, float]:
    try:
        x, y = (float(value) for value in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y in mm, for example 450,100; got {text!r}") from None
    return x, y


def _xyz(text: str) -> tuple[float, float, float]:
    try:
        x, y, z = (float(value) for value in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y,Z in mm, for example 318,-141,796; got {text!r}") from None
    return x, y, z


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write a demo calibration from a ChArUco board clip.")
    parser.add_argument("--rosbag", required=True, metavar="BAG.db3",
                        help="a clip of the board alone, filmed from where the object clips were")
    parser.add_argument("--board", choices=sorted(BOARDS), required=True,
                        help="which board is in the clip: home40, home50 or a2")
    parser.add_argument("--board-at", type=_xy, default=None, metavar="X,Y",
                        help="the board's centre in the robot base frame, mm (default: 450,100)")
    parser.add_argument("--board-yaw", type=float, default=None, metavar="DEG",
                        help="the board's x axis, degrees from the base's +x (default: 0)")
    parser.add_argument("--points", type=_xy, nargs=3, default=None, metavar="X,Y",
                        help="instead of --board-at and --board-yaw: corners P0 P1 P2 measured with a "
                             "tape, mm from the cross on the floor")
    parser.add_argument("--camera-at", type=_xyz, default=None, metavar="X,Y,Z",
                        help="with --points: the camera's colour lens measured the same way, to compare")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help=f"use every Nth settled frame (default: {DEFAULT_STEP})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, metavar="FILE",
                        help=f"where to write it (default: robot/{DEFAULT_OUTPUT.name})")
    args = parser.parse_args(argv)
    if args.points is not None and (args.board_at is not None or args.board_yaw is not None):
        parser.error("--points places the board: --board-at and --board-yaw mean nothing with it")
    if args.camera_at is not None and args.points is None:
        parser.error("--camera-at is compared with a measured board, so it needs --points")
    at_mm = DEFAULT_BOARD_AT_MM if args.board_at is None else args.board_at
    yaw_deg = 0.0 if args.board_yaw is None else args.board_yaw

    bag = resolve_bag(args.rosbag)
    if not bag.exists():
        raise SystemExit(f"no such rosbag: {bag}")
    spec = BOARDS[args.board]
    fit = None if args.points is None else board_from_points(spec, args.points)   # before the slow part
    measured = measure_bag(bag, spec, args.step)
    T_base_board = board_in_base(spec, at_mm, yaw_deg) if fit is None else fit.T_base_board
    T = compose(T_base_board, measured.T_cam_board)
    if T[2, 2] >= 0.0:
        raise SystemExit("the camera would be looking up in the base frame: the board pose is not a "
                         "view from above")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_calibration(T, measured, args.board, bag, at_mm, yaw_deg, fit),
                      encoding="utf-8")
    load_T_base_cam(output)     # the same checks a robot run makes
    height, tilt = height_and_tilt(measured.T_cam_board)
    print(f"{bag.name}: board {args.board} on {measured.frames_used}/{measured.frames_tried} frames, "
          f"camera {height:.1f} mm above it, tilt {tilt:.2f} deg")
    print(f"wrote {output}\n{T}")
    if fit is not None:
        print(f"the points fit the board to {', '.join(f'{e:.1f}' for e in fit.errors_mm)} mm: P0-P1 "
              f"{fit.distances_mm[0]:.1f} mm (board {fit.expected_mm[0]:.1f}), P0-P2 "
              f"{fit.distances_mm[1]:.1f} mm (board {fit.expected_mm[1]:.1f}), angle at P0 "
              f"{fit.corner_deg:.1f} degrees (board 90)")
    camera = T[:3, 3] * MM_PER_M
    print(f"in RobotStudio, the camera goes at x = {camera[0]:.0f}, y = {camera[1]:.0f}, "
          f"z = {camera[2]:.0f} mm")
    if args.camera_at is not None:
        off = camera - np.array(args.camera_at)
        print(f"the tape put the camera at {args.camera_at[0]:.0f}, {args.camera_at[1]:.0f}, "
              f"{args.camera_at[2]:.0f}: off by {off[0]:+.0f}, {off[1]:+.0f}, {off[2]:+.0f} mm "
              f"({np.linalg.norm(off):.0f} mm). More than about 20 mm means a measurement is wrong.")
    show_in_station(bag, spec, T, T_base_board, args.points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
