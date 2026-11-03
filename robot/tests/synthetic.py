"""Synthetic scenes and stand-ins, shared by the robot tests."""
from __future__ import annotations

import types
from dataclasses import dataclass

import numpy as np

from camera.segmentation_realsense import Detection
from robot.perception import CENTROID_WINDOW, Target

NAMES = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}


@dataclass
class Intrinsics:
    """Distortion-free, as the colour stream in meta.json is: its coefficients are all zero."""

    width: int = 640
    height: int = 480
    fx: float = 605.0
    fy: float = 607.0           # different from fx on purpose, so the axes cannot be swapped
    ppx: float = 318.0
    ppy: float = 243.0


def pinhole_rs() -> types.ModuleType:
    """Enough of pyrealsense2 for deprojection and for ExportedFrameSource, with no driver.

    The real function reduces to this formula when the coefficients are zero, which
    test_perception.test_the_real_deproject_is_this_pinhole checks where pyrealsense2 is installed.
    """
    module = types.ModuleType("pyrealsense2")
    module.__version__ = "stub"
    module.intrinsics = types.SimpleNamespace
    module.distortion = types.SimpleNamespace(
        none=0, brown_conrady=1, inverse_brown_conrady=2, __members__={})
    module.rs2_deproject_pixel_to_point = lambda intr, pixel, depth: [
        (pixel[0] - intr.ppx) / intr.fx * depth, (pixel[1] - intr.ppy) / intr.fy * depth, depth]
    return module


class FakeDepth:
    """A depth frame that is already a map in metres, like rosbag_frames.ExportedDepth."""

    def __init__(self, metres: np.ndarray):
        self._metres = metres

    def metres(self) -> np.ndarray:
        return self._metres


def pixel_grid(intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of every pixel of the frame."""
    return np.meshgrid(np.arange(intr.width, dtype=np.float64),
                       np.arange(intr.height, dtype=np.float64))


def disc_mask(u0: float, v0: float, radius: float, intr: Intrinsics) -> np.ndarray:
    u, v = pixel_grid(intr)
    return (u - u0) ** 2 + (v - v0) ** 2 <= radius ** 2


def ellipse_mask(u0: float, v0: float, angle_deg: float, long_px: float, short_px: float,
                 intr: Intrinsics) -> np.ndarray:
    """An ellipse whose long axis is at angle_deg anticlockwise on screen from the bottom edge."""
    u, v = pixel_grid(intr)
    dx, dy = u - u0, v0 - v        # dy points up the screen
    theta = np.radians(angle_deg)
    along = dx * np.cos(theta) + dy * np.sin(theta)
    across = -dx * np.sin(theta) + dy * np.cos(theta)
    return (along / long_px) ** 2 + (across / short_px) ** 2 <= 1.0


def detection(mask: np.ndarray, cls: int = 0, conf: float = 0.9) -> Detection:
    return Detection((0, 0, 10, 10), cls, conf, mask)


def make_target(cls: int = 0, base=(400.0, 100.0, 120.0), z_cam: float = 1.0,
                name: str | None = None, base_axis_deg: float | None = None) -> Target:
    """A Target as the pick sequence sees it, with only the fields it reads being meaningful."""
    return Target(detection=Detection((0, 0, 10, 10), cls, 0.9),
                  name=name or NAMES.get(cls, str(cls)), centroid_px=(5.0, 5.0),
                  orientation_deg=0.0, axis_half_px=1.0, z_cam_m=z_cam,
                  z_source=CENTROID_WINDOW, centroid_on_mask=True, xyz_cam_m=(0.0, 0.0, z_cam),
                  xyz_base_mm=tuple(float(value) for value in base),
                  base_axis_deg=base_axis_deg)


class ScriptedKeys:
    """OperatorKeys with the keys written down in advance. None means nothing pressed."""

    def __init__(self, polled=(), waited=()):
        self.polled = list(polled)
        self.waited = list(waited)
        self.asked = []

    def poll(self, delay_ms: int = 1):
        return self.polled.pop(0) if self.polled else None

    def drain(self) -> None:
        """Does nothing: the script says what happens after the drain. The real one is tested in
        test_operator_keys.py and test_frame_selection.py."""

    def wait_for(self, allowed: str) -> str:
        self.asked.append(allowed)
        return self.waited.pop(0)


class FakeSource:
    """A frame source over a list of (image, depth, intrinsics), or None for a dropped frame."""

    rs = None
    stream_fps = 0.0        # so nothing settles unless a test asks for it: settle_frames is then 0

    def __init__(self, frames, rs=None):
        self.frames = list(frames)
        self.rs = rs or pinhole_rs()
        self.position = 0
        self.started = self.stopped = False

    def describe(self) -> str:
        return "a fake source"

    def start(self) -> None:
        self.started = True

    def grab(self):
        if self.position >= len(self.frames):
            raise StopIteration("out of frames")
        frame = self.frames[self.position]
        self.position += 1
        if isinstance(frame, Exception):
            raise frame
        return frame

    def explain_stall(self, error):
        return "the camera stalled" if "didn't arrive" in str(error) else None

    def stop(self) -> None:
        self.stopped = True


class FakeDetector:
    """Returns the next scripted list of detections on each call."""

    names = NAMES

    def __init__(self, per_frame):
        self.per_frame = list(per_frame)
        self.calls = 0

    def predict(self, image):
        detections = self.per_frame[min(self.calls, len(self.per_frame) - 1)]
        self.calls += 1
        return detections
