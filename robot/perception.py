"""One detected mask -> one robot target: centroid, orientation, z, class, position in both frames.

camera/segmentation_realsense.py has most of this already, and it is reused where it fits:
mask_principal_axis for the orientation, depth_map_metres for the depth, annotate for the picture.
measure() itself is not used, because it answers a different question. It takes the median depth
over the whole mask, which is the right summary of how far away an object is, while a robot wants
the z *at the point it is going to*. On a tilted bottle or a stack, the two differ.

Coordinates
-----------
Pixels are (u, v) = (column, row) in the frame handed to the detector, which is what
rs2_deproject_pixel_to_point takes. Arrays index as [row, column], so [v, u]. Keep the two apart.
A rosbag frame is scaled to 640x360 for the model, and its intrinsics and depth map are scaled with
it (see camera/rosbag_frames.py), so every length here is consistent within that frame.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.segmentation_realsense import (Detection, Observation, annotate,  # noqa: E402
                                           depth_map_metres, mask_principal_axis)
from robot.extrinsics import camera_to_base_mm, rays_at_height_mm  # noqa: E402

# The side of the window read around the centroid, from yolo_robot_test.py. One pixel is not enough:
# a depth map is full of holes on exactly the glass and plastic this project sorts.
CENTROID_WINDOW_PX = 9

# How many of a mask's pixels to follow down to the plane: enough for a direction, and few enough to
# do for every detection on every frame of the preview.
AXIS_SAMPLE_PIXELS = 400

CENTROID_WINDOW = "centroid-window"     # valid depths of the mask inside the window round the centroid
MASK_MEDIAN = "mask-median"             # nothing valid there, so the median over the whole mask
NO_DEPTH = "none"


@dataclass(frozen=True)
class Target:
    """One detection, measured, with everything the pick needs."""

    detection: Detection
    name: str
    centroid_px: tuple[float, float]        # (u, v), sub-pixel: the mean of the mask's pixels
    # The first principal component of the mask's pixels. Degrees in [0, 180), 0 along the bottom
    # of the *image*, growing anticlockwise on screen: see mask_principal_axis. It is an image-plane
    # angle and not yet a gripper yaw. None for a one-pixel mask, which has no direction. Arbitrary
    # for a round one, such as a can standing up, which has no long axis to find.
    orientation_deg: float | None
    axis_half_px: float | None              # for drawing only
    z_cam_m: float                          # along the optical axis. What the list is sorted by.
    z_source: str
    centroid_on_mask: bool                  # False for a C-shaped mask: it then lies off the object
    xyz_cam_m: tuple[float, float, float]
    xyz_base_mm: tuple[float, float, float]
    # The mask's first principal component on the horizontal plane through xyz_base_mm: see
    # base_axis_deg. Degrees in [0, 180), 0 along the base's +x, growing anticlockwise seen from above.
    # None for a one-pixel mask, as orientation_deg is.
    base_axis_deg: float | None

    @property
    def cls(self) -> int:
        return self.detection.cls

    @property
    def conf(self) -> float:
        return self.detection.conf


def depth_at_centroid(metres: np.ndarray, mask: np.ndarray, u: float, v: float,
                      window: int = CENTROID_WINDOW_PX) -> tuple[float, str]:
    """(depth at the centroid in metres, where it came from), or (0.0, NO_DEPTH).

    The median of the valid depths in a small window round the centroid, counting only pixels of
    the mask, so a thin object does not pick up the table beside it. A depth of 0 is "no reading"
    and never a distance. If the window holds nothing usable -- a hole on glass, or a centroid that
    lies off a C-shaped mask -- the median over the whole mask is the next best thing.
    """
    height, width = mask.shape[:2]
    half = window // 2
    column, row = int(round(u)), int(round(v))
    top, bottom = max(0, row - half), min(height, row + half + 1)
    left, right = max(0, column - half), min(width, column + half + 1)

    patch = metres[top:bottom, left:right][mask[top:bottom, left:right]]
    patch = patch[patch > 0]
    if patch.size:
        return float(np.median(patch)), CENTROID_WINDOW

    inside = metres[mask]
    inside = inside[inside > 0]
    if inside.size:
        return float(np.median(inside)), MASK_MEDIAN
    return 0.0, NO_DEPTH


def base_axis_deg(T_base_cam: np.ndarray, intrinsics, rs, rows: np.ndarray, columns: np.ndarray,
                  z_base_mm: float) -> float | None:
    """A mask's first principal component, measured on the horizontal plane z_base_mm of the base.

    The camera's tilt shears what it sees, so the mask's axis in the image is not the object's, and
    the less elongated the object the further out it is. So this follows the mask's pixels down to the
    plane and takes the principal component there. The plane's height only scales the result and does
    not change the angle, which leaves it deciding one thing: None if the plane cannot be reached, as
    with a z above the camera, which only a calibration that does not match the scene can produce.

    This assumes the object lies level. A long axis 20 degrees out of the horizontal reads about 17
    degrees off. None as well for a one-pixel mask, which has no direction.
    """
    step = max(1, rows.size // AXIS_SAMPLE_PIXELS)
    rays = [rs.rs2_deproject_pixel_to_point(intrinsics, [float(column), float(row)], 1.0)
            for row, column in zip(rows[::step], columns[::step])]
    points = rays_at_height_mm(T_base_cam, rays, z_base_mm)
    if points is None:
        return None
    # mask_principal_axis negates its rows, because an image's grow downward. y goes in negated so
    # that it comes out as y, and the angle runs from +x towards +y, anticlockwise seen from above.
    axis = mask_principal_axis(-points[:, 1], points[:, 0])
    return None if axis is None else axis[0]


def measure_targets(detections, depth_frame, intrinsics, rs, names,
                    T_base_cam) -> tuple[list[Target], list[str]]:
    """(targets, why each other detection was dropped), in the detector's order.

    A detection with no mask or no depth cannot become a position, so it is dropped with a reason
    rather than given a made-up one. A depth map that does not match the mask is a geometry bug and
    raises: carrying on would mean sending the robot to the wrong place.
    """
    metres = depth_map_metres(depth_frame) if detections else None

    targets, dropped = [], []
    for detection in detections:
        name = names.get(detection.cls, str(detection.cls))
        if detection.mask is None:
            dropped.append(f"{name}: the model gave no mask")
            continue
        mask = detection.mask.astype(bool)
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            dropped.append(f"{name}: empty mask")
            continue
        if metres is None or metres.shape[:2] != mask.shape[:2]:
            raise ValueError(f"the depth map {None if metres is None else metres.shape[:2]} does "
                             f"not match the mask {mask.shape[:2]}")

        u, v = float(columns.mean()), float(rows.mean())
        axis = mask_principal_axis(rows, columns)
        orientation, axis_half = axis if axis is not None else (None, None)

        z, source = depth_at_centroid(metres, mask, u, v)
        if z <= 0:
            dropped.append(f"{name}: no depth reading anywhere on its mask")
            continue

        X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z)
        xyz_cam = (float(X), float(Y), float(Z))
        xyz_base = camera_to_base_mm(T_base_cam, xyz_cam)
        targets.append(Target(
            detection=detection, name=name, centroid_px=(u, v), orientation_deg=orientation,
            axis_half_px=axis_half, z_cam_m=xyz_cam[2], z_source=source,
            centroid_on_mask=bool(mask[int(round(v)), int(round(u))]),
            xyz_cam_m=xyz_cam, xyz_base_mm=xyz_base,
            base_axis_deg=base_axis_deg(T_base_cam, intrinsics, rs, rows, columns, xyz_base[2])))
    return targets, dropped


def sort_nearest_first(targets) -> list[Target]:
    """Nearest the camera first, by distance along its optical axis."""
    return sorted(targets, key=lambda target: target.z_cam_m)


def draw_targets(frame: np.ndarray, targets, names) -> np.ndarray:
    """The frame with each target's mask, box, centroid, axis and its rank in the list."""
    observations = [
        Observation(
            object_id=rank, cls=target.cls, conf=target.conf, xyxy=target.detection.xyxy,
            box_centre=((target.detection.xyxy[0] + target.detection.xyxy[2]) // 2,
                        (target.detection.xyxy[1] + target.detection.xyxy[3]) // 2),
            centroid=(int(round(target.centroid_px[0])), int(round(target.centroid_px[1]))),
            xyz=target.xyz_cam_m, depth_source=target.z_source,
            orientation_deg=target.orientation_deg, axis_half_px=target.axis_half_px)
        for rank, target in enumerate(targets, 1)]
    return annotate(frame, [target.detection for target in targets], observations, names,
                    verbose=False, show_ids=True, show_axis=True)


def format_targets(targets) -> str:
    """The list as a table, in the order the robot will take it."""
    lines = [f"{'#':>2}  {'class':<8} {'conf':>4}  {'centroid px':>13}  {'orient':>6}  "
             f"{'z cam m':>7}  {'z from':<15}  {'base x':>7} {'base y':>7} {'base z':>7} (mm)  "
             f"{'axis':>5} (deg)"]
    for rank, target in enumerate(targets, 1):
        u, v = target.centroid_px
        x, y, z = target.xyz_base_mm
        orientation = ("-" if target.orientation_deg is None
                       else f"{target.orientation_deg:.1f}")
        axis = "-" if target.base_axis_deg is None else f"{target.base_axis_deg:.1f}"
        note = "" if target.centroid_on_mask else "  (centroid is off the mask)"
        lines.append(f"{rank:>2}  {target.name:<8} {target.conf:>4.2f}  {u:>6.1f},{v:>6.1f}  "
                     f"{orientation:>6}  {target.z_cam_m:>7.3f}  {target.z_source:<15}  "
                     f"{x:>7.1f} {y:>7.1f} {z:>7.1f}      {axis:>5}{note}")
    return "\n".join(lines)
