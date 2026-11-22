"""What the robot does with a target: over it, down, grip, up, drop, home, and ask.

Nothing here knows about cameras. A Target comes in with its position already in the base frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.art_camera_client import (HOME_Z_MM, X_MAX_MM, X_MIN_MM, Y_MAX_MM,  # noqa: E402
                                     Y_MIN_MM, Z_TABLE_MM, ArtCameraClient, MoveRejected)
from robot.perception import Target  # noqa: E402

# #################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>  DUMMY DROP LOCATIONS -- EDIT THESE BY HAND  <<<<<<<<<<<<<<<<<<<<<<<<
#
# Where each class is released: (x, y, z) in mm, in the robot's base frame, keyed by class id.
# These are placeholders, copied from pDrop0..pDrop3 in rapid/ArtCamera.modx. They are NOT measured
# against any real bin. check_drop_locations() refuses at startup anything the controller would
# reject, and a class with no entry here is skipped before the robot moves.
DROP_LOCATIONS_MM: dict[int, tuple[float, float, float]] = {
    0: (165.0, -675.0, 120.0),      # glass - orange
    1: (450.0, -675.0, 120.0),      # metal
    2: (165.0, -380.0, 120.0),      # paper - blue
    3: (450.0, -380.0, 120.0),      # plastic - yellow
}
# #################################################################################################

# The plane the arm travels in: home's z, so the first move over the object does not change height.
APPROACH_Z_MM = HOME_Z_MM

# What a grasp z may be. The controller has no check on z (a TODO in InWorkspace), so this is the
# only thing between a bad depth reading and the table. The base's +z is up, as checked in
# RobotStudio, so the upper limit is the approach plane: a z above it would turn the descent into a
# climb, and only comes from a calibration that does not match the scene. A clip through the real
# calibration.txt reads about 628 mm, which is exactly that.
#
# The margin is because Z_TABLE_MM is the table's surface, so a TCP sent exactly there is touching
# it, and z is not that exact. Settled depth was measured at 9 mm (p95) at 0.75 m, which grows with
# the square of the range to an estimated 30 mm at the rig's 1.4 m, before any calibration error.
# The camera now hangs about 0.79 m over the floor, so 10 mm covers that noise. Anything whose top
# reads under 10 mm above the floor is skipped. Raise this if the camera goes higher.
GRASP_Z_MARGIN_MM = 10.0
GRASP_Z_MIN_MM = Z_TABLE_MM + GRASP_Z_MARGIN_MM
GRASP_Z_MAX_MM = APPROACH_Z_MM

# The line the RG6's fingers close along at rz = 0, which is pHome's orientation: degrees from the
# base's +x, anticlockwise seen from above. 0 is along the base's x, read off the RobotStudio station.
FINGER_CLOSE_AXIS_AT_RZ0_DEG = 0.0


EXTRINSICS_HINT = ("This position comes from the extrinsics in use, named at the start of the run. "
                   "--calibration changes them.")


class PickAborted(RuntimeError):
    """The robot is holding something and cannot carry on. It has been left where it is."""


def check_drop_locations(locations=None) -> None:
    """Refuse to start with a drop location the controller would reject mid-pick, gripper closed."""
    for cls, (x, y, z) in (DROP_LOCATIONS_MM if locations is None else locations).items():
        if not (X_MIN_MM <= x <= X_MAX_MM and Y_MIN_MM <= y <= Y_MAX_MM):
            raise SystemExit(f"DROP_LOCATIONS_MM[{cls}] = ({x}, {y}, {z}) is outside the workspace "
                             f"(x {X_MIN_MM:.0f}..{X_MAX_MM:.0f}, y {Y_MIN_MM:.0f}..{Y_MAX_MM:.0f})")
        if not Z_TABLE_MM <= z <= APPROACH_Z_MM:
            raise SystemExit(f"DROP_LOCATIONS_MM[{cls}] = ({x}, {y}, {z}): z must be between the "
                             f"table and the approach plane ({Z_TABLE_MM:.0f}.."
                             f"{APPROACH_Z_MM:.0f}), and the controller does not check it")


def gripper_rz_deg(base_axis_deg: float | None) -> float | None:
    """The rz that closes the fingers across an object's long axis, or None if it has no axis.

    rz turns the tool about the base's +z, so at rz the fingers close along
    FINGER_CLOSE_AXIS_AT_RZ0_DEG + rz. Two fingers look the same turned by 180 degrees, so of the two
    answers this takes the one in [-90, 90), which is the smaller turn away from pHome.
    """
    if base_axis_deg is None:
        return None
    return (base_axis_deg + 90.0 - FINGER_CLOSE_AXIS_AT_RZ0_DEG + 90.0) % 180.0 - 90.0


def grasp_z_for(target: Target) -> float:
    """The z the TCP goes down to for `target`. One place, so the log and the viewer agree with the move."""
    perceived_z = target.xyz_base_mm[2]
    # #############################################################################################
    # >>> THE GRASP Z: STILL TO BE TUNED <<<
    # This is the z of the point the camera saw, moved into the base frame, which is the object's
    # top surface at its centroid. The TCP stops exactly there, so the fingertips reach the surface
    # and do not straddle the object. Going below the surface by some depth, or a per-class height,
    # goes here. It used to be -perceived_z, which sent the TCP under the floor. GRASP_Z_MIN_MM
    # still keeps whatever this returns above the floor.
    return perceived_z
    # #############################################################################################


def pick_and_place(robot: ArtCameraClient, target: Target) -> bool:
    """One whole pick, starting and ending at home. True if the whole sequence ran.

    True does not mean an object was held: RAPID's GripClose answers DONE even when it times out.

    A target that cannot be attempted returns False with the robot at home, and the operator
    decides what next. Once the gripper has closed, a refused move raises PickAborted instead:
    opening or going home from there would drop whatever it holds.
    """
    cls = target.cls
    x, y, _ = target.xyz_base_mm
    grasp_z = grasp_z_for(target)

    # None means a one-pixel mask, which keeps pHome's orientation. On a round object the angle is
    # arbitrary, and the turn is harmless. The drop move carries no rz, so the tool comes back to
    # pHome's orientation on the way there.
    rz = gripper_rz_deg(target.base_axis_deg)
    turned = "" if rz is None else f", rz {rz:.1f}"

    drop = DROP_LOCATIONS_MM.get(cls)
    if drop is None:
        print(f"  SKIPPED: no drop location for class {cls} ({target.name}) in DROP_LOCATIONS_MM.")
        return False
    if not GRASP_Z_MIN_MM <= grasp_z <= GRASP_Z_MAX_MM:
        print(f"  SKIPPED: z = {grasp_z:.1f} mm is outside {GRASP_Z_MIN_MM:.0f}.."
              f"{GRASP_Z_MAX_MM:.0f} mm, so the robot did not move.")
        # A z under the floor is a flat object or a bad reading, which says nothing about --calibration.
        if grasp_z > GRASP_Z_MAX_MM:
            print(f"  {EXTRINSICS_HINT}")
        return False

    try:
        print(f"  moving over the object: ({x:.1f}, {y:.1f}, {APPROACH_Z_MM:.1f}{turned})")
        robot.move_to(cls, x, y, APPROACH_Z_MM, rz)
        print(f"  moving down: ({x:.1f}, {y:.1f}, {grasp_z:.1f}{turned})")
        robot.move_to(cls, x, y, grasp_z, rz)
    except MoveRejected as rejection:
        print(f"  SKIPPED: {rejection}. {EXTRINSICS_HINT}")
        print("  going home")
        robot.home()
        return False

    print("  closing the gripper")
    robot.grip_close()

    try:
        print(f"  moving back up: ({x:.1f}, {y:.1f}, {APPROACH_Z_MM:.1f}{turned})")
        robot.move_to(cls, x, y, APPROACH_Z_MM, rz)
        print(f"  moving to the drop location for {target.name}: {drop}")
        robot.move_to(cls, *drop)
    except MoveRejected as rejection:
        raise PickAborted(f"{rejection}, with the gripper closed. The robot has been left where "
                          f"it is, still holding the object.") from rejection

    print("  opening the gripper")
    robot.grip_open()
    print("  going home")
    robot.home()
    return True


def run_picks(robot: ArtCameraClient, targets, keys, on_picked=None) -> int:
    """Work down the list, asking between picks. Returns how many whole sequences ran.

    `on_picked(position, target, ran)`, when given, is called after each target with whether its
    whole sequence ran: the experiment log and the live view hang off it. It runs with the robot at
    home and may ask the operator for a key.
    """
    completed = 0
    for position, target in enumerate(targets, 1):
        x, y, z = target.xyz_base_mm
        print(f"\n--- Target {position} of {len(targets)}: {target.name}, base "
              f"({x:.1f}, {y:.1f}, {z:.1f}) mm, {target.z_cam_m:.3f} m from the camera")
        ran = pick_and_place(robot, target)
        completed += ran
        if on_picked is not None:
            on_picked(position, target, ran)

        if position == len(targets):
            print("\nThat was the last detection in the list.")
            break
        print(f"\nRobot is at home. Press 'c' to continue with target {position + 1} of "
              f"{len(targets)}, or 's' to stop.")
        print("The key works in the image window or in this terminal.")
        if keys.wait_for("cs") == "s":
            print("Stopping.")
            break
    return completed
