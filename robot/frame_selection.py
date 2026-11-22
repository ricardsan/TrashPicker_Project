"""Where the detections come from: a chosen frame of an unpacked rosbag, or the camera on 'n'.

preview_until is deliberately not camera/segmentation_realsense.run_frames. That loop holds the
project's timing contract and runs until ESC; it has no way to stop on a frame or hand back what it
found. This one times and logs nothing, so nothing here needs keeping in step with it. Both share
the same frame-source interface, so RealSenseSource and ExportedFrameSource are reused unchanged.
"""
from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.rosbag_frames import (ExportedFrameSource, default_export_dir,  # noqa: E402
                                  export_bag, is_exported)
from camera.segmentation_realsense import depth_map_metres  # noqa: E402
from robot.art_camera_client import Z_TABLE_MM  # noqa: E402
from robot.operator_keys import QUIT_KEYS  # noqa: E402
from robot.pick_sequence import REACH_MAX_MM, REACH_MIN_MM  # noqa: E402
from robot.perception import (Target, draw_targets, measure_targets,  # noqa: E402
                              sort_nearest_first)

ROSBAG_DIR = ROOT / "rosbags"
EXPORT_DIR = ROSBAG_DIR / "export"

WINDOW = "robot trash picker"

# How long to distrust the depth after a stream starts. Measured on all three clips at 15 Hz: up to
# frame 9 or 10, 33-77% of the depth pixels are holes and the rest are up to 158 mm (p95) from where
# the settled frames put them, and from frame 10 or 11 on it is under 4% and 9 mm. A z that wrong can
# still pass the guard in pick_sequence.py, so it would send the gripper too deep. That the live
# camera settles the same way is an assumption: there was none to measure.
SETTLE_S = 1.0


@dataclass(frozen=True)
class Snapshot:
    """The frame the robot will work from."""

    frame_index: int
    annotated: np.ndarray
    targets: list[Target]       # nearest the camera first
    dropped: list[str]          # why the other detections are not in the list
    # The raw frame, for the experiment log's point cloud. None where a source cannot give them.
    image: np.ndarray | None = None
    metres: np.ndarray | None = None
    intrinsics: object = None


def resolve_bag(name) -> Path:
    """The .db3 meant by --rosbag: a bare name is looked up in rosbags/, .db3 is optional."""
    path = Path(name)
    if path.suffix != ".db3":
        path = path.with_name(path.name + ".db3")
    return ROSBAG_DIR / path if path.parent == Path(".") else path


def open_recording(name) -> ExportedFrameSource:
    """An unpacked bag as a frame source. Unpacks it first only if that has not been done.

    real_time=False because the clock has no business here: real-time pacing skips the frames a
    slow model misses, which could be the very frame that was asked for. preload=False because only
    one pass is made and no latency is being measured.
    """
    bag = resolve_bag(name)
    export_dir = default_export_dir(bag, EXPORT_DIR)
    if not is_exported(export_dir) and not bag.exists():
        available = ", ".join(sorted(p.name for p in ROSBAG_DIR.glob("*.db3"))) or "none"
        raise SystemExit(f"no such rosbag: {bag}\nAvailable in {ROSBAG_DIR}: {available}")
    export_bag(bag, export_dir, clouds=False)
    return ExportedFrameSource(export_dir, real_time=False, preload=False)


def resolve_frame_index(requested: int, n_frames: int) -> int:
    """--frame as a position in the clip. Negative counts from the end, so -1 is the last frame."""
    index = requested if requested >= 0 else n_frames + requested
    if not 0 <= index < n_frames:
        raise SystemExit(f"--frame {requested} is outside this clip, which has {n_frames} frames "
                         f"(0..{n_frames - 1}, or -1..-{n_frames} from the end)")
    return index


def settle_frames(source) -> int:
    """How many frames of this source come before its depth can be trusted. See SETTLE_S."""
    return math.ceil(SETTLE_S * source.stream_fps)


@contextmanager
def started(source):
    """`source` running for the length of the block, and stopped however the block ends.

    The caller opens this before it moves the robot, so a camera that is not plugged in, or a
    missing pyrealsense2, costs a message and no motion.
    """
    try:
        source.start()
    except (RuntimeError, ImportError) as error:
        raise SystemExit(f"could not start {source.describe()}: {error}") from error
    try:
        yield source
    finally:
        source.stop()


def unreachable_mask(intrinsics, T_base_cam: np.ndarray, floor_z_mm: float = Z_TABLE_MM) -> np.ndarray:
    """True where the floor seen at that pixel is outside the arm's reach, or not seen at all.

    Each pixel's ray is followed down to the floor, z = floor_z_mm, in the base frame.
    """
    w, h = int(intrinsics.width), int(intrinsics.height)
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(u - intrinsics.ppx) / intrinsics.fx, (v - intrinsics.ppy) / intrinsics.fy,
                     np.ones_like(u)], axis=-1) @ np.asarray(T_base_cam)[:3, :3].T
    centre_mm = np.asarray(T_base_cam)[:3, 3] * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (floor_z_mm - centre_mm[2]) / rays[..., 2]
    x = centre_mm[0] + scale * rays[..., 0]
    y = centre_mm[1] + scale * rays[..., 1]
    radius = np.hypot(x, y)
    seen = np.isfinite(scale) & (scale > 0)
    return ~(seen & (radius >= REACH_MIN_MM) & (radius <= REACH_MAX_MM))


def shade_unreachable(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The image with the unreachable part tinted red and outlined."""
    if mask.shape != image.shape[:2] or not mask.any():
        return image
    out = image.copy()
    red = np.zeros_like(out)
    red[..., 2] = 255
    out[mask] = cv2.addWeighted(out, 0.6, red, 0.4, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


def preview_until(source, detector, T_base_cam, keys, *, stop_at: int | None = None,
                  window: str = WINDOW, view=None) -> Snapshot | None:
    """Show inference frame by frame, and return the Snapshot of the frame to work from.

    With `stop_at` that is frame number `stop_at`; without it, the frame on screen when 'n' is
    pressed, once the depth has settled. None means the operator quit. `source` must be running:
    see started(). `view`, a live_view.LiveView, also gets every frame.
    """
    settling = settle_frames(source)
    # A key pressed while the robot was going home was not pressed at this picture.
    keys.drain()
    index = -1
    reach = None        # (intrinsics key, mask): the mask only changes with the intrinsics
    while True:
        try:
            grabbed = source.grab()
        except StopIteration:
            raise SystemExit(f"{source.describe()} ran out before frame {stop_at}") from None
        except RuntimeError as error:
            explanation = source.explain_stall(error)
            if explanation is None:
                raise
            raise SystemExit(explanation) from error
        if grabbed is None:
            continue

        index += 1
        image, depth_frame, intrinsics = grabbed
        targets, dropped = measure_targets(detector.predict(image), depth_frame, intrinsics,
                                           source.rs, detector.names, T_base_cam)
        targets = sort_nearest_first(targets)
        size = (int(intrinsics.width), int(intrinsics.height), intrinsics.fx, intrinsics.ppx)
        if reach is None or reach[0] != size:
            reach = (size, unreachable_mask(intrinsics, T_base_cam))
        annotated = draw_targets(shade_unreachable(image, reach[1]), targets, detector.names)
        cv2.imshow(window, annotated)
        if view is not None:
            view.frame(index, image, depth_frame, intrinsics, targets, annotated)

        key = keys.poll()
        if key in QUIT_KEYS:
            return None
        if stop_at is None:
            if key != "n":
                continue
            if index < settling:
                print(f"The depth is still settling (frame {index + 1} of {settling}). "
                      f"Press 'n' again in a moment.")
                continue
            if not targets:
                print("Nothing pickable in this frame. Press 'n' again when there is.")
                continue
        elif index != stop_at:
            continue
        return Snapshot(index, annotated, targets, dropped, image=image,
                        metres=depth_map_metres(depth_frame), intrinsics=intrinsics)
