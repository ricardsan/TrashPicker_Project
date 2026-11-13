"""A live 3D view of a run, beside RobotStudio: RGB, masks, the point cloud and the planned grasps.

    python robot/robot_trash_picker.py --conf 0.4 --viz

RobotStudio shows the robot and cannot show what the camera sees; the image window shows what the
camera sees and not where the robot is going. This puts both into one 3D scene in the robot's base
frame, in millimetres, so a screen recording of it next to RobotStudio shows the whole experiment:

    world/camera          the camera where T_base_cam puts it, with the RGB image on its frustum
    world/cloud           the scene, deprojected and moved into the base frame (scene_cloud.py)
    world/targets         each detection's measured point, numbered in pick order
    world/grasp/<n>       for the snapshot: approach line, finger closing line, the object's axis
    image/annotated       the same picture the OpenCV window shows

It uses Rerun (https://rerun.io), which is optional: `pip install rerun-sdk`. The viewer it opens can
save the whole run as an .rrd file (File > Save recording), which replays it frame by frame later --
the record of an experiment, not just a video of it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.segmentation_realsense import depth_map_metres  # noqa: E402
from robot.pick_sequence import (APPROACH_Z_MM, FINGER_CLOSE_AXIS_AT_RZ0_DEG,  # noqa: E402
                                 grasp_z_for, gripper_rz_deg)
from robot.scene_cloud import cloud_in_base  # noqa: E402

APP_ID = "trashpicker"
DEFAULT_EVERY = 2           # log every 2nd frame: the cloud is the expensive part
DEFAULT_STRIDE = 3
FINGER_HALF_MM = 40.0       # how long the drawn finger line is either side of the grasp point
AXIS_HALF_MM = 50.0
CLASS_COLOURS = {0: (255, 140, 0), 1: (160, 160, 160), 2: (30, 120, 255), 3: (255, 220, 0)}


def import_rerun():
    try:
        import rerun as rr
    except ImportError:
        raise SystemExit("--viz needs Rerun, which is not installed in this environment.\n"
                         "Install it with:  pip install rerun-sdk") from None
    return rr


def _set_frame(rr, index: int) -> None:
    # rerun >= 0.23 has set_time(timeline, sequence=); older releases had set_time_sequence.
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=index)
    else:
        rr.set_time_sequence("frame", index)


def _direction(degrees: float) -> np.ndarray:
    return np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees)), 0.0])


def grasp_geometry(target) -> dict:
    """What to draw for one target, in base millimetres: points and segments, no Rerun involved."""
    x, y, z = target.xyz_base_mm
    grasp_z = grasp_z_for(target)
    rz = gripper_rz_deg(target.base_axis_deg)
    top = np.array([x, y, z])
    grasp = np.array([x, y, grasp_z])
    geometry = {"top": top, "grasp": grasp, "rz": rz,
                "approach": np.array([[x, y, APPROACH_Z_MM], [x, y, grasp_z]])}
    if rz is not None:
        closing = _direction(FINGER_CLOSE_AXIS_AT_RZ0_DEG + rz) * FINGER_HALF_MM
        geometry["fingers"] = np.array([top - closing, top + closing])
    if target.base_axis_deg is not None:
        along = _direction(target.base_axis_deg) * AXIS_HALF_MM
        geometry["axis"] = np.array([top - along, top + along])
    return geometry


class LiveView:
    """Logs a run to a Rerun viewer. Every method is cheap to call when there is nothing to log."""

    def __init__(self, T_base_cam: np.ndarray, *, every: int = DEFAULT_EVERY,
                 stride: int = DEFAULT_STRIDE, spawn: bool = True, rr=None):
        self.rr = rr if rr is not None else import_rerun()
        self.T = np.asarray(T_base_cam, dtype=np.float64)
        self.every = max(1, int(every))
        self.stride = stride
        self.rr.init(APP_ID, spawn=spawn)
        self.rr.log("world", self.rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.rr.log("world/camera", self.rr.Transform3D(
            translation=self.T[:3, 3] * 1000.0, mat3x3=self.T[:3, :3]), static=True)
        self._pinhole_logged = False

    def _pinhole(self, intrinsics, shape) -> None:
        if self._pinhole_logged:
            return
        K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                      [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]])
        self.rr.log("world/camera/image", self.rr.Pinhole(
            image_from_camera=K, resolution=[shape[1], shape[0]],
            camera_xyz=self.rr.ViewCoordinates.RDF, image_plane_distance=150.0), static=True)
        self._pinhole_logged = True

    def frame(self, index: int, image_bgr: np.ndarray, depth_frame, intrinsics, targets,
              annotated: np.ndarray) -> None:
        """One preview frame. Skipped unless index is a multiple of `every`."""
        if index % self.every:
            return
        rr = self.rr
        _set_frame(rr, index)
        self._pinhole(intrinsics, image_bgr.shape)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rr.log("world/camera/image", rr.Image(rgb))
        rr.log("image/annotated", rr.Image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
        metres = depth_map_metres(depth_frame)
        if metres is not None and metres.shape[:2] == image_bgr.shape[:2]:
            try:
                points, colours = cloud_in_base(image_bgr, metres, intrinsics, self.T, self.stride)
            except ValueError:
                points = None
            if points is not None:
                rr.log("world/cloud", rr.Points3D(points, colors=colours, radii=1.0))
        self._targets(targets)

    def _targets(self, targets) -> None:
        rr = self.rr
        if not targets:
            rr.log("world/targets", rr.Clear(recursive=False))
            return
        rr.log("world/targets", rr.Points3D(
            [target.xyz_base_mm for target in targets], radii=6.0,
            colors=[CLASS_COLOURS.get(target.cls, (255, 0, 255)) for target in targets],
            labels=[f"{rank} {target.name} {target.conf:.2f}"
                    for rank, target in enumerate(targets, 1)]))

    def snapshot(self, index: int, targets) -> None:
        """The frame the robot works from: draw every planned grasp."""
        rr = self.rr
        _set_frame(rr, index)
        self._targets(targets)
        for rank, target in enumerate(targets, 1):
            geometry = grasp_geometry(target)
            base = f"world/grasp/{rank}"
            rr.log(f"{base}/approach", rr.LineStrips3D([geometry["approach"]],
                                                       colors=[(0, 200, 255)], radii=1.5))
            rr.log(f"{base}/grasp", rr.Points3D([geometry["grasp"]], radii=5.0, colors=[(0, 200, 255)],
                                                labels=[f"{rank}: z {geometry['grasp'][2]:.0f}"]))
            if "fingers" in geometry:
                rr.log(f"{base}/fingers", rr.LineStrips3D([geometry["fingers"]],
                                                          colors=[(255, 60, 60)], radii=3.0))
            if "axis" in geometry:
                rr.log(f"{base}/axis", rr.LineStrips3D([geometry["axis"]],
                                                       colors=[(60, 255, 60)], radii=1.5))

    def pick_result(self, rank: int, target, outcome: str) -> None:
        """Mark a target with what happened to it: 'success', 'fail', 'skipped' or 'not run'."""
        colour = {"success": (0, 220, 0), "fail": (230, 0, 0)}.get(outcome, (150, 150, 150))
        self.rr.log(f"world/grasp/{rank}/result", self.rr.Points3D(
            [target.xyz_base_mm], radii=9.0, colors=[colour], labels=[f"{rank}: {outcome}"]))
