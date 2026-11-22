"""Send the TCP to points on the floor, one by one, to see that the robot reaches them.

    python robot/touch_points.py --points 520,40 520,-121 400,40              # z = 0: on the floor
    python robot/touch_points.py --points 520,40 520,-121 400,40 --z 20       # 20 mm above it

The same three corners P0 P1 P2 that went into board_demo_calibration.py --points, measured from the
cross on the floor. For each: over it at home's height, straight down to z, wait for Enter, back up.
Then home. In RobotStudio, watch that the TCP lands on the corner of the board in the station, and
that no point is out of reach: the controller stops with a reach error, or rejects the move.

z = 0 is the floor itself, so the TCP touches it. That is fine in the simulator and is refused for any
controller but this machine: on a real robot, use --z with a safe height.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.art_camera_client import (DEFAULT_HOST, DEFAULT_PORT, ArtCameraClient,  # noqa: E402
                                     MoveRejected, RobotError)
from robot.board_demo_calibration import _xy  # noqa: E402
from robot.pick_sequence import (APPROACH_Z_MM, GRASP_Z_MIN_MM, REACH_MAX_MM,  # noqa: E402
                                 REACH_MIN_MM)
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")     # as robot_trash_picker.py, without its imports
MOVE_CLASS = 0          # a move needs an object class; any non-negative one is only a label


def touch(robot, points, z_mm: float, wait=input) -> list[str]:
    """Visit each point; one line per point saying how it went. Ends at home."""
    results = []
    robot.home()
    for number, (x, y) in enumerate(points):
        name = f"P{number} ({x:.0f}, {y:.0f})"
        reach = math.hypot(x, y)
        note = ("" if REACH_MIN_MM <= reach <= REACH_MAX_MM else
                f", {reach:.0f} mm from the base: outside the {REACH_MIN_MM:.0f}..{REACH_MAX_MM:.0f} mm "
                f"the pick sequence uses")
        print(f"\n{name}: over it, then down to z = {z_mm:.0f}{note}")
        try:
            robot.move_to(MOVE_CLASS, x, y, APPROACH_Z_MM)
            robot.move_to(MOVE_CLASS, x, y, z_mm)
        except MoveRejected as rejection:
            print(f"  REJECTED: {rejection}")
            results.append(f"{name}: rejected by the controller")
            robot.home()
            continue
        wait(f"  at {name}, z = {z_mm:.0f}. Check it in RobotStudio, then press Enter ")
        robot.move_to(MOVE_CLASS, x, y, APPROACH_Z_MM)
        results.append(f"{name}: reached{note}")
    robot.home()
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Touch floor points with the TCP to check reach.")
    parser.add_argument("--points", type=_xy, nargs="+", required=True, metavar="X,Y",
                        help="points in mm from the cross on the floor (the robot base)")
    parser.add_argument("--z", type=float, default=0.0,
                        help="the TCP's height at each point, mm (default: 0, the floor)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if args.z < GRASP_Z_MIN_MM and args.host not in LOCAL_HOSTS:
        parser.error(f"--z {args.z:.0f} is at or under the floor's margin ({GRASP_Z_MIN_MM:.0f} mm): "
                     f"only for the simulator on this machine, not {args.host}")
    if not 0.0 <= args.z < APPROACH_Z_MM:
        parser.error(f"--z must be from 0 (the floor) up to under {APPROACH_Z_MM:.0f}")

    try:
        with ArtCameraClient(args.host, args.port) as robot:
            results = touch(robot, args.points, args.z)
    except RobotError as failure:
        print(f"\nSTOPPED: {failure}\nA point out of reach stops RAPID with an error in RobotStudio; "
              f"look there for which one.", file=sys.stderr)
        return 1
    print("\n" + "\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
