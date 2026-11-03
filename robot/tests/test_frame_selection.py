"""Which frame the robot works from, and from where."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from camera.rosbag_frames import ExportedFrameSource
from robot import frame_selection
from robot.extrinsics import load_T_base_cam
from robot.frame_selection import (ROSBAG_DIR, open_recording, preview_until, resolve_bag,
                                   resolve_frame_index, settle_frames, started)
from robot.operator_keys import OperatorKeys
from synthetic import (FakeDepth, FakeDetector, FakeSource, Intrinsics, ScriptedKeys, detection,
                       disc_mask)

INTR = Intrinsics()
T = load_T_base_cam()


def frame(depth: float = 1.0):
    return (np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8),
            FakeDepth(np.full((INTR.height, INTR.width), depth)), INTR)


def by_frame(n):
    """One detection per frame, whose class says which frame it came from."""
    return [[detection(disc_mask(320, 240, 10, INTR), cls=i)] for i in range(n)]


def preview(source, detector, keys, **kwargs):
    """What the command line does: the source running for the length of the preview."""
    with started(source):
        return preview_until(source, detector, T, keys, **kwargs)


# =============================================================================
# resolving --rosbag and --frame
# =============================================================================
def test_a_bare_name_is_looked_up_in_rosbags():
    assert resolve_bag("clip.db3") == ROSBAG_DIR / "clip.db3"


def test_the_extension_is_optional():
    assert resolve_bag("clip") == ROSBAG_DIR / "clip.db3"


def test_a_path_is_taken_as_given():
    assert resolve_bag("elsewhere/clip.db3") == Path("elsewhere/clip.db3")


@pytest.mark.parametrize("requested, expected", [(0, 0), (3, 3), (9, 9), (-1, 9), (-10, 0)])
def test_frame_numbers(requested, expected):
    assert resolve_frame_index(requested, 10) == expected


@pytest.mark.parametrize("requested", [10, 500, -11])
def test_a_frame_outside_the_clip_is_refused(requested):
    with pytest.raises(SystemExit, match="outside this clip"):
        resolve_frame_index(requested, 10)


def test_an_unknown_bag_lists_what_there_is(tmp_path, monkeypatch):
    (tmp_path / "real.db3").write_bytes(b"")
    monkeypatch.setattr(frame_selection, "ROSBAG_DIR", tmp_path)
    monkeypatch.setattr(frame_selection, "EXPORT_DIR", tmp_path / "export")
    with pytest.raises(SystemExit, match="real.db3"):
        open_recording("missing.db3")


# =============================================================================
# starting and stopping the source
# =============================================================================
def test_the_source_is_stopped_when_the_block_ends():
    source = FakeSource([])
    with started(source):
        assert source.started and not source.stopped
    assert source.stopped


def test_the_source_is_stopped_when_the_block_fails():
    source = FakeSource([])
    with pytest.raises(ValueError):
        with started(source):
            raise ValueError("the robot went away")
    assert source.stopped


@pytest.mark.parametrize("failure", [RuntimeError("No device connected"),
                                     ModuleNotFoundError("No module named 'pyrealsense2'")])
def test_a_source_that_cannot_start_is_a_message_and_is_not_stopped(failure):
    """Stopping a RealSense pipeline that never started raises, which would bury the real reason."""
    source = FakeSource([])

    def refuse():
        raise failure

    source.start = refuse
    with pytest.raises(SystemExit, match="could not start"):
        with started(source):
            pytest.fail("the block must not run")
    assert not source.stopped


# =============================================================================
# a clip: stop on the chosen frame
# =============================================================================
@pytest.mark.parametrize("stop_at", [0, 2, 4])
def test_the_detections_are_those_of_the_chosen_frame(no_window, stop_at):
    source = FakeSource([frame() for _ in range(5)])
    snapshot = preview(source, FakeDetector(by_frame(5)), ScriptedKeys(), stop_at=stop_at)
    assert snapshot.frame_index == stop_at
    assert [target.cls for target in snapshot.targets] == [stop_at]


def test_it_runs_inference_on_every_frame_up_to_the_chosen_one_and_no_further(no_window):
    source, detector = FakeSource([frame() for _ in range(6)]), FakeDetector(by_frame(6))
    preview(source, detector, ScriptedKeys(), stop_at=3)
    assert detector.calls == 4
    assert source.position == 4


def test_the_list_comes_back_nearest_first(no_window):
    metres = np.full((INTR.height, INTR.width), 1.2)
    metres[60:140, 60:140] = 0.8
    metres[300:380, 300:380] = 1.0
    detections = [detection(disc_mask(500, 400, 20, INTR), cls=0),      # 1.2
                  detection(disc_mask(100, 100, 20, INTR), cls=1),      # 0.8
                  detection(disc_mask(340, 340, 20, INTR), cls=2)]      # 1.0
    source = FakeSource([(frame()[0], FakeDepth(metres), INTR)])
    snapshot = preview(source, FakeDetector([detections]), ScriptedKeys(), stop_at=0)
    assert [target.cls for target in snapshot.targets] == [1, 2, 0]


def test_a_dropped_frame_does_not_count_as_a_frame(no_window):
    source = FakeSource([None, frame(), None, frame()])
    snapshot = preview(source, FakeDetector(by_frame(2)), ScriptedKeys(), stop_at=1)
    assert [target.cls for target in snapshot.targets] == [1]


def test_quitting_during_a_clip_returns_none_and_stops_the_source(no_window):
    source = FakeSource([frame() for _ in range(5)])
    assert preview(source, FakeDetector(by_frame(5)), ScriptedKeys(polled=[None, "q"]),
                   stop_at=4) is None
    assert source.stopped


def test_quit_wins_on_the_chosen_frame_itself(no_window):
    source = FakeSource([frame()])
    assert preview(source, FakeDetector(by_frame(1)), ScriptedKeys(polled=["\x1b"]),
                   stop_at=0) is None


def test_the_chosen_frame_with_nothing_on_it_is_an_empty_snapshot(no_window):
    source = FakeSource([frame()])
    snapshot = preview(source, FakeDetector([[]]), ScriptedKeys(), stop_at=0)
    assert snapshot.targets == []


def test_detections_it_could_not_measure_are_reported(no_window):
    source = FakeSource([frame(depth=0.0)])
    detector = FakeDetector([[detection(disc_mask(320, 240, 10, INTR), cls=3)]])
    snapshot = preview(source, detector, ScriptedKeys(), stop_at=0)
    assert snapshot.targets == []
    assert snapshot.dropped == ["plastic: no depth reading anywhere on its mask"]


def test_a_clip_can_still_be_asked_for_an_unsettled_frame(no_window):
    """--frame is the operator's choice. The command line warns about it, but it is honoured."""
    source = FakeSource([frame() for _ in range(4)])
    source.stream_fps = 15.0
    assert preview(source, FakeDetector(by_frame(4)), ScriptedKeys(), stop_at=2).frame_index == 2


# =============================================================================
# the camera: stop on 'n'
# =============================================================================
def test_n_takes_the_frame_on_screen(no_window):
    source = FakeSource([frame() for _ in range(5)])
    keys = ScriptedKeys(polled=[None, None, "n"])
    snapshot = preview(source, FakeDetector(by_frame(5)), keys)
    assert snapshot.frame_index == 2
    assert [target.cls for target in snapshot.targets] == [2]


def test_nothing_but_n_takes_a_frame(no_window):
    source = FakeSource([frame() for _ in range(4)])
    keys = ScriptedKeys(polled=["c", "s", "x", "n"])
    assert preview(source, FakeDetector(by_frame(4)), keys).frame_index == 3


@pytest.mark.parametrize("key", ["q", "\x1b"])
def test_quitting_the_live_preview(no_window, key):
    source = FakeSource([frame() for _ in range(3)])
    assert preview(source, FakeDetector(by_frame(3)), ScriptedKeys(polled=[key])) is None
    assert source.stopped


def test_n_on_an_empty_frame_keeps_waiting_and_says_why(no_window, capsys):
    source = FakeSource([frame(), frame()])
    detector = FakeDetector([[], [detection(disc_mask(320, 240, 10, INTR), cls=1)]])
    snapshot = preview(source, detector, ScriptedKeys(polled=["n", "n"]))
    assert snapshot.frame_index == 1
    assert "Nothing pickable" in capsys.readouterr().out


def test_the_camera_is_stopped_once_the_frame_is_taken(no_window):
    source = FakeSource([frame()])
    preview(source, FakeDetector(by_frame(1)), ScriptedKeys(polled=["n"]))
    assert source.stopped


def test_a_stalled_camera_ends_with_its_explanation(no_window):
    source = FakeSource([RuntimeError("Frame didn't arrive within 5000")])
    with pytest.raises(SystemExit, match="stalled"):
        preview(source, FakeDetector([[]]), ScriptedKeys())
    assert source.stopped


def test_any_other_runtime_error_is_not_swallowed(no_window):
    source = FakeSource([RuntimeError("something else")])
    with pytest.raises(RuntimeError, match="something else"):
        preview(source, FakeDetector([[]]), ScriptedKeys())


# =============================================================================
# the depth settling after a stream starts
# =============================================================================
def test_n_is_ignored_until_the_depth_has_settled(no_window, capsys):
    """Taking frame 0 on an early 'n' would start a pick from the worst depth in the stream."""
    source = FakeSource([frame() for _ in range(6)])
    source.stream_fps = 3.0     # settles after 3 frames
    snapshot = preview(source, FakeDetector(by_frame(6)),
                       ScriptedKeys(polled=["n", "n", "n", "n"]))
    assert snapshot.frame_index == 3
    assert "still settling" in capsys.readouterr().out


def test_quitting_works_while_the_depth_is_settling(no_window):
    source = FakeSource([frame() for _ in range(6)])
    source.stream_fps = 3.0
    assert preview(source, FakeDetector(by_frame(6)), ScriptedKeys(polled=["q"])) is None


def test_a_key_pressed_while_the_robot_went_home_is_forgotten(no_window, monkeypatch):
    """With the real OperatorKeys. The first 'n' was queued before the preview started."""
    script = [ord("n"), -1,     # the drain: the old key, then nothing left
              -1,               # frame 0, nothing pressed
              ord("n")]         # frame 1, the real one
    monkeypatch.setattr("cv2.waitKey", lambda delay=0: script.pop(0) if script else -1)
    source = FakeSource([frame() for _ in range(3)])
    snapshot = preview(source, FakeDetector(by_frame(3)), OperatorKeys())
    assert snapshot.frame_index == 1


@pytest.mark.parametrize("fps, expected", [(15.0, 15), (30.0, 30), (0.0, 0)])
def test_the_settling_period_is_a_time_and_not_a_frame_count(fps, expected):
    """The clips run at 15 Hz and the camera at 30, and depth settles by the clock."""
    source = FakeSource([])
    source.stream_fps = fps
    assert settle_frames(source) == expected


# =============================================================================
# the real ExportedFrameSource, over a small export on disk
# =============================================================================
@pytest.fixture
def small_export(tmp_path, monkeypatch):
    """Five frames as export_bag writes them. Each colour frame is a distinguishable grey."""
    rosbags = tmp_path / "rosbags"
    directory = rosbags / "export" / "clip"
    (directory / "color").mkdir(parents=True)
    (directory / "depth").mkdir()
    records = []
    for i in range(5):
        cv2.imwrite(str(directory / f"color/{i:06d}.png"),
                    np.full((48, 64, 3), i * 40, dtype=np.uint8))
        cv2.imwrite(str(directory / f"depth/{i:06d}.png"),
                    np.full((48, 64), 1000 + i, dtype=np.uint16))
        records.append({"index": i, "timestamp_ms": 1000.0 + i * 66.7, "frame_number": i + 1,
                        "color": f"color/{i:06d}.png", "depth": f"depth/{i:06d}.png",
                        "cloud": None})
    (directory / "meta.json").write_text(json.dumps({
        "bag_name": "clip", "stream_fps": 15.0, "duration_s": 0.33, "depth_scale": 0.001,
        "color_intrinsics": {"width": 64, "height": 48, "ppx": 31.5, "ppy": 23.5, "fx": 40.0,
                             "fy": 41.0, "model": "inverse_brown_conrady", "coeffs": [0.0] * 5},
        "depth_aligned_to_color": True, "clouds": False, "cloud_stride": 1, "complete": True,
        "frames": records}))
    monkeypatch.setattr(frame_selection, "ROSBAG_DIR", rosbags)
    monkeypatch.setattr(frame_selection, "EXPORT_DIR", rosbags / "export")
    return directory


class RecordingDetector:
    """Finds nothing, and keeps what it was shown."""

    names = {0: "glass"}

    def __init__(self):
        self.images = []

    def predict(self, image):
        self.images.append(image)
        return []


def test_an_existing_export_is_used_without_the_bag(small_export):
    """There is no .db3 in this tree at all. "Use the unpacked rosbags" means exactly this."""
    source = open_recording("clip.db3")
    assert isinstance(source, ExportedFrameSource)
    assert source.export_dir == small_export
    assert len(source.frames) == 5


def test_a_clip_is_not_paced_by_the_clock(small_export):
    """Real-time pacing skips the frames a slow model misses, which could be the one asked for."""
    source = open_recording("clip.db3")
    assert source.real_time is False


@pytest.mark.parametrize("stop_at", [0, 3, 4])
def test_the_chosen_frame_is_the_right_picture(small_export, no_window, rs_or_stub, stop_at):
    source = open_recording("clip.db3")
    detector = RecordingDetector()
    snapshot = preview(source, detector, ScriptedKeys(), stop_at=stop_at)
    assert snapshot.frame_index == stop_at
    assert source.skipped == 0
    assert detector.images[-1][0, 0, 0] == stop_at * 40


def test_the_export_scales_the_frame_the_depth_and_the_intrinsics_together(small_export, no_window,
                                                                           rs_or_stub):
    """The frame is halved for the model. The position has to come out the same regardless."""
    source = open_recording("clip.db3")
    source.start()
    image, depth, intrinsics = source.grab()
    assert image.shape[:2] == (24, 32)
    assert depth.metres().shape == (24, 32)
    assert (intrinsics.fx, intrinsics.ppx) == (20.0, 15.75)
    # The same physical point, seen at half the resolution, deprojects to the same place.
    half = rs_or_stub.rs2_deproject_pixel_to_point(intrinsics, [10.0, 5.0], 1.0)
    assert half[0] == pytest.approx((20.0 - 31.5) / 40.0, abs=1e-5)
    assert half[1] == pytest.approx((10.0 - 23.5) / 41.0, abs=1e-5)
