"""The live loop, driven by a fake RealSense.

run_live holds every timing decision in the project -- what is inside the measured region and
what is not -- and none of it was covered. A fake `pyrealsense2` in sys.modules plus a detector
of scripted latency makes the whole loop testable with no camera and no model.
"""
from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

from camera.benchmark import RunLogger
from camera.segmentation_realsense import Detection, run_live

FRAME_H, FRAME_W = 480, 640


# --- the fake camera ---------------------------------------------------------
class FakeDepthFrame:
    def get_distance(self, x, y):
        if not (0 <= x < FRAME_W and 0 <= y < FRAME_H):
            # What the real pyrealsense2 does, and the reason boxes must be clipped.
            raise RuntimeError(f"({x}, {y}) is out of the depth frame")
        return 0.5


class FakeColorFrame:
    def __init__(self, image):
        self.image = image

    def get_data(self):
        return self.image


class FakeFrames:
    def __init__(self, color, depth):
        self.color, self.depth = color, depth

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class FakePipeline:
    """Serves a scripted list of frames, then raises to end the loop if ESC never came."""

    def __init__(self, script):
        self.script = list(script)
        self.started = False
        self.stopped = False
        self.waits = 0

    def start(self, config):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_active_profile(self):
        return types.SimpleNamespace(get_stream=lambda stream: None)

    def wait_for_frames(self):
        if not self.script:
            raise StopIteration("the fake camera ran out of frames")
        self.waits += 1
        item = self.script.pop(0)
        if callable(item):
            item = item()
        return item


def fake_rs(pipeline):
    """A module object with the handful of names run_live touches."""
    module = types.ModuleType("pyrealsense2")
    module.pipeline = lambda: pipeline
    module.config = lambda: types.SimpleNamespace(enable_stream=lambda *a, **k: None)
    module.stream = types.SimpleNamespace(color="color", depth="depth")
    module.format = types.SimpleNamespace(bgr8="bgr8", z16="z16")
    module.align = lambda stream: types.SimpleNamespace(process=lambda frames: frames)
    module.video_stream_profile = lambda stream: types.SimpleNamespace(
        get_intrinsics=lambda: "intrinsics")
    module.rs2_deproject_pixel_to_point = lambda intrinsics, pixel, distance: (0.1, 0.2, distance)
    return module


class FakeDetector:
    """Burns a scripted amount of time per call and returns scripted detections."""

    names = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}

    def __init__(self, latencies_s, detections_per_frame=None):
        self.latencies = list(latencies_s)
        self.detections_per_frame = detections_per_frame
        self.calls = 0

    def predict(self, frame):
        latency = self.latencies[min(self.calls, len(self.latencies) - 1)]
        self.calls += 1
        deadline = time.perf_counter() + latency
        while time.perf_counter() < deadline:
            pass
        if self.detections_per_frame is None:
            return [Detection((10, 10, 110, 110), 0, 0.9)]
        n = self.detections_per_frame[min(self.calls - 1,
                                          len(self.detections_per_frame) - 1)]
        return [Detection((10, 10, 110, 110), i % 4, 0.9) for i in range(n)]


def image():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def good_frame():
    return FakeFrames(FakeColorFrame(image()), FakeDepthFrame())


def dropped_frame():
    """What the RealSense hands back when a frame is incomplete."""
    return FakeFrames(None, None)


@pytest.fixture
def live(monkeypatch):
    """Runs run_live against a fake camera and a fake display, returning the pipeline + logger."""

    def run(script, detector, logger=None, keys=None, **kwargs):
        pipeline = FakePipeline(script)
        monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs(pipeline))

        # cv2 would open a window and block; the key script decides when ESC arrives.
        pressed = list(keys) if keys else []
        monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)
        monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)
        monkeypatch.setattr("cv2.waitKey",
                            lambda delay: pressed.pop(0) if pressed else -1)

        # Exposed before running, so a test can inspect the pipeline even when run_live raises.
        run.pipeline = pipeline
        try:
            run_live(detector, logger, window="test", **kwargs)
        except StopIteration:
            pass            # the fake camera ran dry, which ends the loop
        return pipeline

    run.pipeline = None
    return run


# --- what the loop does ------------------------------------------------------
def test_one_row_per_delivered_frame(live, tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live([good_frame() for _ in range(5)], FakeDetector([0.001]), logger)
    assert len(logger.rows) == 5
    assert [r["frame"] for r in logger.rows] == [0, 1, 2, 3, 4]


def test_a_dropped_frame_logs_no_row(live, tmp_path):
    """An incomplete frame hits the `continue`, so it must not become a row."""
    script = [good_frame(), dropped_frame(), dropped_frame(), good_frame()]
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live(script, FakeDetector([0.001]), logger)
    assert len(logger.rows) == 2


def test_a_dropped_frame_still_costs_time_in_the_next_period(live, tmp_path):
    """Its time has to land somewhere, or the achieved rate would flatter itself."""
    slow_drop = [good_frame, lambda: (time.sleep(0.05), dropped_frame())[1], good_frame]
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live(slow_drop, FakeDetector([0.001]), logger)

    assert len(logger.rows) == 2
    # The second delivered frame's period spans the dropped one too.
    assert logger.rows[1]["loop_ms"] >= 50


def test_the_detector_is_called_once_per_delivered_frame(live):
    detector = FakeDetector([0.001])
    live([good_frame(), dropped_frame(), good_frame()], detector)
    assert detector.calls == 2


def test_esc_ends_the_run(live, tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        pipeline = live([good_frame() for _ in range(10)], FakeDetector([0.001]), logger,
                        keys=[-1, -1, 27])
    assert len(logger.rows) == 3
    assert pipeline.script, "the loop should have stopped before exhausting the frames"


def test_the_pipeline_is_stopped_on_the_way_out(live):
    pipeline = live([good_frame()], FakeDetector([0.001]), keys=[27])
    assert pipeline.started and pipeline.stopped


def test_the_pipeline_is_stopped_when_the_loop_raises(live):
    """The finally has to survive a detector that blows up mid-run, or the camera stays open."""
    class Exploding(FakeDetector):
        def predict(self, frame):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA"):
        live([good_frame()], Exploding([0.0]))

    assert live.pipeline.started
    assert live.pipeline.stopped, "run_live must stop the pipeline even when the loop raises"


def test_a_run_with_no_logger_still_works(live):
    """--no-log must not break the loop."""
    detector = FakeDetector([0.001])
    live([good_frame(), good_frame()], detector, logger=None)
    assert detector.calls == 2


def test_detection_count_is_logged(live, tmp_path):
    detector = FakeDetector([0.001], detections_per_frame=[0, 3, 7])
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live([good_frame() for _ in range(3)], detector, logger)
    assert [r["n_detections"] for r in logger.rows] == [0, 3, 7]


# --- what is inside the measured region --------------------------------------
def test_inference_ms_tracks_the_model_not_the_loop(live, tmp_path):
    """inference_ms must follow the detector's own cost and nothing else."""
    detector = FakeDetector([0.002, 0.030, 0.002])
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live([good_frame() for _ in range(3)], detector, logger)

    latencies = [r["inference_ms"] for r in logger.rows]
    assert latencies[1] > 25, latencies          # the slow frame shows up
    assert latencies[0] < 15 and latencies[2] < 15


def test_loop_ms_includes_the_camera_wait(live, tmp_path):
    """The blocking grab is part of the achieved period, which is why it is not model speed."""
    slow_camera = [good_frame, lambda: (time.sleep(0.04), good_frame())[1], good_frame]
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live(slow_camera, FakeDetector([0.001]), logger)

    rows = logger.rows
    # The frame whose own grab blocked is the one that pays for it.
    assert rows[1]["loop_ms"] >= 40, rows
    # ...while inference stayed fast throughout: the two measure different things.
    assert max(r["inference_ms"] for r in rows) < 15


def test_loop_ms_describes_the_same_frame_as_inference_ms(live, tmp_path):
    """The row must be internally consistent.

    A top-to-top period looks equivalent but attributes the *previous* iteration's duration to
    this row. A real run caught it: the first grab blocked for 7 s, and row 1 ended up pairing a
    13.9 ms inference with a 7747 ms period, i.e. model_fps 72 next to fps 0.13 for one frame.
    """
    # Only the first grab is slow, exactly as in that run.
    script = [lambda: (time.sleep(0.25), good_frame())[1], good_frame, good_frame]
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live(script, FakeDetector([0.002]), logger)

    rows = logger.rows
    assert rows[0]["loop_ms"] >= 250, "the slow grab belongs to the frame that waited"
    for row in rows[1:]:
        assert row["loop_ms"] < 100, f"a later frame inherited the first grab's wait: {rows}"
        # The two rates in a row must be about one frame, so the period cannot be wildly
        # larger than the inference it is paired with.
        assert row["loop_ms"] < row["inference_ms"] + 100


def test_loop_ms_includes_waitkey(live, tmp_path, monkeypatch):
    """waitKey used to fall outside the span, so loop_ms was neither one thing nor the other."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        pipeline = FakePipeline([good_frame() for _ in range(3)])
        monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs(pipeline))
        monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)
        monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)
        # A waitKey that costs 30 ms, the way a real one does when the window is busy.
        monkeypatch.setattr("cv2.waitKey", lambda delay: (time.sleep(0.03), -1)[1])
        try:
            run_live(FakeDetector([0.001]), logger, window="test")
        except StopIteration:
            pass

    assert all(r["loop_ms"] >= 30 for r in logger.rows), logger.rows


def test_esc_still_logs_the_frame_it_was_pressed_on(live, tmp_path):
    """waitKey now runs before the log, so the final frame must not be dropped."""
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live([good_frame() for _ in range(5)], FakeDetector([0.001]), logger, keys=[-1, 27])
    assert len(logger.rows) == 2


def test_the_first_frame_has_a_period(live, tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live([good_frame()], FakeDetector([0.005]), logger, keys=[27])
    assert logger.rows[0]["loop_ms"] > 0


# --- the camera giving up -----------------------------------------------------
def test_a_stalled_stream_ends_the_run_without_a_traceback(live, tmp_path, capsys):
    """librealsense raises a bare RuntimeError when frames stop; a USB 2 link does this."""
    def stall():
        raise RuntimeError("Frame didn't arrive within 5000")

    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        pipeline = live([good_frame, good_frame, stall, good_frame],
                        FakeDetector([0.001]), logger)

    assert len(logger.rows) == 2, "the frames already captured must be kept"
    assert pipeline.stopped
    message = capsys.readouterr().out
    assert "USB" in message and "stopped delivering" in message


def test_other_runtime_errors_still_propagate(live):
    """Only the frame-timeout is swallowed; a real fault must not be hidden behind it."""
    def boom():
        raise RuntimeError("hardware is on fire")

    with pytest.raises(RuntimeError, match="on fire"):
        live([boom], FakeDetector([0.001]))


# --- drawing -----------------------------------------------------------------
def test_a_box_on_the_frame_edge_does_not_kill_the_run(live, tmp_path):
    """get_distance raises outside the frame; a centre at width-1 must stay inside."""
    class EdgeDetector(FakeDetector):
        def predict(self, frame):
            super().predict(frame)
            return [Detection((FRAME_W - 1, FRAME_H - 1, FRAME_W - 1, FRAME_H - 1), 0, 0.9)]

    with RunLogger(tmp_path / "run.csv", "yolo26", "taco") as logger:
        live([good_frame(), good_frame()], EdgeDetector([0.001]), logger)
    assert len(logger.rows) == 2


def test_masks_can_be_turned_off(live):
    """--no-masks reaches the drawing, which matters because the overlay is not free."""
    drawn = []

    class MaskDetector(FakeDetector):
        def predict(self, frame):
            super().predict(frame)
            mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
            mask[20:40, 20:40] = True
            return [Detection((20, 20, 40, 40), 1, 0.9, mask)]

    import camera.segmentation_realsense as module
    original = module.draw_masks
    try:
        module.draw_masks = lambda annotated, detections: drawn.append(len(detections))
        live([good_frame()], MaskDetector([0.001]), keys=[27], masks=False)
        assert drawn == []
        live([good_frame()], MaskDetector([0.001]), keys=[27], masks=True)
        assert drawn == [1]
    finally:
        module.draw_masks = original


# =============================================================================
# MEASURING: ONE DEPTH LOOKUP PER DETECTION, AND FAST MODE STAYS FAST
# =============================================================================
# Measuring used to be a side effect of drawing. It is its own step now because the per-object CSV
# wants the same numbers, and the thing to prove is that wanting them twice did not start paying
# for them twice.
from camera.benchmark import DetectionLogger, read_log          # noqa: E402
from camera.segmentation_realsense import (Observation, depth_map_metres,  # noqa: E402
                                           mask_principal_axis, measure, run_frames)
from camera.tracking import Tracker                             # noqa: E402

CLASS_NAMES = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}


class CountingDepthFrame(FakeDepthFrame):
    """A depth frame that remembers how often it was asked, and offers no map."""

    def __init__(self, distance=0.5):
        self.calls = []
        self.distance = distance

    def get_distance(self, x, y):
        super().get_distance(x, y)       # keeps the out-of-frame check
        self.calls.append((x, y))
        return self.distance


class MapDepthFrame(CountingDepthFrame):
    """A depth frame that can hand over the whole map, the way ExportedDepth does."""

    def __init__(self, distance=0.5, fill=None):
        super().__init__(distance)
        self.map_calls = 0
        self.fill = fill if fill is not None else np.full((FRAME_H, FRAME_W), 0.8)

    def metres(self):
        self.map_calls += 1
        return self.fill


def mask_at(x1, y1, x2, y2):
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def diagonal_mask(x0, y0, length, rise):
    """A one-pixel diagonal band from (x0, y0), stepping `rise` rows per column.

    rise=-1 draws "/" as seen on screen (x up, y down the array means up the picture); rise=+1
    draws "\\". Written with numpy rather than cv2 so the mask helpers stay dependency-free, like
    mask_at above.
    """
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    for i in range(length):
        mask[y0 + rise * i, x0 + i] = True
    return mask


def fake_rs_module():
    """Just the deprojection, which is all measure() uses."""
    return types.SimpleNamespace(
        rs2_deproject_pixel_to_point=lambda intrinsics, pixel, distance: (
            pixel[0] / 1000.0, pixel[1] / 1000.0, distance))


# --- fast mode ---------------------------------------------------------------
def test_fast_mode_makes_exactly_one_depth_lookup_per_detection():
    """The lookup is cheap, but two of them would be two, and the loop is measured."""
    depth = CountingDepthFrame()
    detections = [Detection((10, 10, 110, 110), 0, 0.9, mask_at(10, 10, 110, 110)),
                  Detection((200, 200, 300, 300), 1, 0.8, mask_at(200, 200, 300, 300))]

    measure(detections, depth, "intrinsics", fake_rs_module())
    assert len(depth.calls) == 2


def test_fast_mode_never_asks_for_the_depth_map():
    """Building it is the cost --objects gates; fast mode must not pay it at all."""
    depth = MapDepthFrame()
    measure([Detection((10, 10, 110, 110), 0, 0.9, mask_at(10, 10, 110, 110))],
            depth, "intrinsics", fake_rs_module())
    assert depth.map_calls == 0


def test_fast_mode_measures_at_the_box_centre_as_it_always_did():
    depth = CountingDepthFrame()
    # A mask off to one side, so a mask centroid would be somewhere else entirely.
    detection = Detection((0, 0, 200, 200), 0, 0.9, mask_at(0, 0, 20, 20))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module())

    assert depth.calls == [(100, 100)]
    assert observation.centroid == (100, 100) == observation.box_centre
    assert observation.depth_source == "centre-pixel"
    assert observation.mask_px == 0, "fast mode should not even look at the mask"


# --- --objects ---------------------------------------------------------------
def test_mask_depth_takes_the_median_over_the_mask():
    fill = np.zeros((FRAME_H, FRAME_W))
    fill[10:110, 10:110] = 1.0
    # A handful of wild values inside the mask: the median must shrug them off.
    fill[10:13, 10:13] = 9.0
    depth = MapDepthFrame(fill=fill)

    detection = Detection((10, 10, 110, 110), 0, 0.9, mask_at(10, 10, 110, 110))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module(), mask_depth=True)

    assert observation.depth_source == "mask-median"
    assert observation.xyz[2] == pytest.approx(1.0)
    assert observation.mask_px == 100 * 100
    assert observation.depth_px == 100 * 100
    assert depth.calls == [], "the per-pixel path should not have been used at all"


def test_mask_depth_deprojects_at_the_mask_centroid_not_the_box_centre():
    """The box centre of a diagonal object is often the floor behind it."""
    depth = MapDepthFrame()
    # The box spans 0..200 but the object only occupies its top-left corner.
    detection = Detection((0, 0, 200, 200), 0, 0.9, mask_at(0, 0, 40, 40))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module(), mask_depth=True)

    assert observation.box_centre == (100, 100)
    assert observation.centroid == pytest.approx((19, 19), abs=1)
    assert observation.centroid != observation.box_centre


def test_mask_depth_counts_only_the_pixels_that_had_a_reading():
    """depth_px against mask_px is how a reading from three pixels is told from a real one."""
    fill = np.zeros((FRAME_H, FRAME_W))
    fill[10:20, 10:110] = 0.6       # only the top tenth of the mask came back
    depth = MapDepthFrame(fill=fill)

    detection = Detection((10, 10, 110, 110), 0, 0.9, mask_at(10, 10, 110, 110))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module(), mask_depth=True)

    assert observation.mask_px == 100 * 100
    assert observation.depth_px == 10 * 100
    assert observation.xyz[2] == pytest.approx(0.6)


def test_the_depth_map_is_built_once_per_frame_not_once_per_detection():
    depth = MapDepthFrame()
    detections = [Detection((x, 10, x + 50, 60), 0, 0.9, mask_at(x, 10, x + 50, 60))
                  for x in (10, 100, 200, 300)]
    measure(detections, depth, "intrinsics", fake_rs_module(), mask_depth=True)
    assert depth.map_calls == 1


def test_a_detection_with_no_mask_falls_back_to_the_centre_pixel():
    """A detect-only checkpoint, or a frame ultralytics returned no masks for."""
    depth = MapDepthFrame()
    observation, = measure([Detection((10, 10, 110, 110), 0, 0.9, None)],
                           depth, "intrinsics", fake_rs_module(), mask_depth=True)
    assert observation.depth_source == "centre-pixel"
    assert observation.mask_px == 0
    assert depth.calls == [(60, 60)]


def test_a_depth_map_of_the_wrong_shape_falls_back_rather_than_crashing():
    depth = MapDepthFrame(fill=np.full((100, 100), 0.4))        # not the frame's shape
    detection = Detection((10, 10, 110, 110), 0, 0.9, mask_at(10, 10, 110, 110))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module(), mask_depth=True)

    assert observation.depth_source == "centre-pixel"
    assert observation.xyz[2] == pytest.approx(0.5)             # the per-pixel value


def test_an_empty_mask_falls_back_to_the_box_centre():
    """A mask thresholded away to nothing has no centroid to speak of."""
    depth = MapDepthFrame()
    empty = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    observation, = measure([Detection((10, 10, 110, 110), 0, 0.9, empty)],
                           depth, "intrinsics", fake_rs_module(), mask_depth=True)
    assert observation.mask_px == 0
    assert observation.centroid == (60, 60)
    assert observation.depth_source == "centre-pixel"


# --- the mask's principal axis -----------------------------------------------
def axis_of(mask):
    return mask_principal_axis(*np.nonzero(mask))


def test_a_wide_object_is_horizontal_and_a_tall_one_is_vertical():
    assert axis_of(mask_at(100, 200, 300, 210))[0] == pytest.approx(0.0)
    assert axis_of(mask_at(200, 100, 210, 300))[0] == pytest.approx(90.0)


def test_the_two_diagonals_are_not_the_same_angle():
    """The y-flip test, and the reason both diagonals are here rather than one.

    Image rows grow downward. Measure the angle in those coordinates without negating the row
    deltas and every angle in the CSV comes out mirrored -- "/" reporting 135 -- which is
    perfectly plausible-looking and wrong. "/" is 45 because the angle grows anticlockwise as
    seen on screen, which is the convention the column is documented under.
    """
    assert axis_of(diagonal_mask(100, 300, 100, rise=-1))[0] == pytest.approx(45.0)
    assert axis_of(diagonal_mask(100, 100, 100, rise=+1))[0] == pytest.approx(135.0)


def test_the_angle_is_always_in_zero_to_one_eighty():
    """An axis is a line, not an arrow: there is no 200 degrees, and no -45 either."""
    masks = [mask_at(100, 200, 300, 210), mask_at(200, 100, 210, 300),
             diagonal_mask(100, 300, 100, rise=-1), diagonal_mask(100, 100, 100, rise=+1)]
    for mask in masks:
        angle, _ = axis_of(mask)
        assert 0.0 <= angle < 180.0, angle


def test_a_hair_clockwise_of_horizontal_wraps_to_just_under_one_eighty():
    """The boundary of the wrap: it must not come out as a small negative number."""
    shallow = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    for i in range(200):
        shallow[200 + i // 40, 100 + i] = True       # drops 5 rows over 200 columns
    angle, _ = axis_of(shallow)
    assert 175.0 < angle < 180.0, angle


def test_a_near_circular_mask_never_reports_exactly_one_eighty():
    """The wrap's float edge, and the reason the range needs a clamp and not just a modulo.

    -1e-15 % 180.0 is 180.0 in Python, not 0.0. A round object's mu11 is pure cancellation
    residue, so it reaches that input routinely -- this ellipse is 31.36 x 31.02 px and used to
    come out as exactly 180.0, outside the range every docstring and the CSV documentation
    promise. 180 and 0 are the same orientation; only one of them is in range.
    """
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    dx, dy = xx - 416.0869, yy - 192.5046
    angle = np.radians(0.5617)
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    round_mask = (u / 31.3555) ** 2 + (v / 31.0214) ** 2 <= 1.0

    assert axis_of(round_mask)[0] < 180.0


def test_a_single_pixel_has_no_orientation():
    """atan2(0, 0) is 0, so without the guard one pixel would read as "lying horizontally"."""
    one = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    one[5, 5] = True
    assert axis_of(one) is None
    assert axis_of(np.zeros((FRAME_H, FRAME_W), dtype=bool)) is None


def test_a_one_pixel_line_has_an_orientation_despite_zero_variance_on_one_axis():
    """Zero variance in x is not a degenerate mask -- it is a vertical object."""
    line = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    line[100:200, 50] = True
    assert axis_of(line)[0] == pytest.approx(90.0)


def test_an_exact_square_reports_zero_which_is_a_documented_limitation():
    """A perfectly isotropic mask has no principal axis; atan2(0, 0) picks 0.

    Pinned rather than fixed: real masks are never exactly isotropic, and blanking "round enough"
    objects would need an arbitrary threshold. The column's documentation says to judge a
    near-round object's angle by its mask_px and box aspect instead.
    """
    assert axis_of(mask_at(10, 10, 110, 110))[0] == pytest.approx(0.0)


def test_the_half_length_scales_with_the_object():
    """It is what --show-axis draws, so it has to reach roughly the ends of the object."""
    _, half = axis_of(mask_at(100, 200, 300, 210))      # a 200 px bar
    assert 100.0 <= half <= 140.0, half
    _, double = axis_of(mask_at(100, 200, 500, 210))    # twice as long
    assert double == pytest.approx(2 * half, rel=0.05)


def test_the_orientation_reaches_the_observation_under_objects():
    depth = MapDepthFrame()
    detection = Detection((100, 200, 300, 210), 0, 0.9, mask_at(100, 200, 300, 210))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module(), mask_depth=True)
    assert observation.orientation_deg == pytest.approx(0.0)
    assert observation.axis_half_px > 0


def test_fast_mode_computes_no_orientation_even_though_the_mask_is_there():
    """Same bargain as mask_px: --centre-depth does not look at masks, so there is no axis."""
    depth = CountingDepthFrame()
    detection = Detection((100, 200, 300, 210), 0, 0.9, mask_at(100, 200, 300, 210))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module())
    assert observation.orientation_deg is None
    assert observation.axis_half_px is None


def test_a_detection_with_no_mask_has_no_orientation():
    depth = MapDepthFrame()
    observation, = measure([Detection((10, 10, 110, 110), 0, 0.9, None)],
                           depth, "intrinsics", fake_rs_module(), mask_depth=True)
    assert observation.orientation_deg is None


def test_an_empty_mask_has_no_orientation():
    empty = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    observation, = measure([Detection((10, 10, 110, 110), 0, 0.9, empty)],
                           MapDepthFrame(), "intrinsics", fake_rs_module(), mask_depth=True)
    assert observation.orientation_deg is None


# --- a depth of zero is not a distance of zero -------------------------------
def test_no_depth_reading_anywhere_gives_no_position():
    """Z16 uses 0 for "nothing came back", which is the normal case on glass."""
    depth = MapDepthFrame(distance=0.0, fill=np.zeros((FRAME_H, FRAME_W)))
    detection = Detection((10, 10, 110, 110), 0, 0.9, mask_at(10, 10, 110, 110))
    observation, = measure([detection], depth, "intrinsics", fake_rs_module(), mask_depth=True)

    assert observation.xyz is None
    assert observation.depth_source == "none"
    assert observation.depth_px == 0


def test_no_depth_reading_in_fast_mode_too():
    """The fix is not part of the --objects trade: it applies on the cheap path as well."""
    depth = CountingDepthFrame(distance=0.0)
    observation, = measure([Detection((10, 10, 110, 110), 0, 0.9)],
                           depth, "intrinsics", fake_rs_module())
    assert observation.xyz is None
    assert observation.depth_source == "none"


def test_a_frame_with_no_depth_is_labelled_rather_than_claiming_zero_metres(live):
    """It used to draw "Z=0.00m" under a bottle the camera could not see."""
    import camera.segmentation_realsense as module

    labels = []

    class BlindDepth(FakeDepthFrame):
        def get_distance(self, x, y):
            return 0.0

    def blind_frame():
        return FakeFrames(FakeColorFrame(image()), BlindDepth())

    original = module.cv2.putText
    try:
        module.cv2.putText = lambda img, text, *a, **k: labels.append(text)
        live([blind_frame()], FakeDetector([0.001]), keys=[27])
    finally:
        module.cv2.putText = original

    assert labels and all("Z=?" in text for text in labels), labels
    assert not any("Z=0.00" in text for text in labels)


# --- depth_map_metres' fallback chain ----------------------------------------
def test_depth_map_metres_prefers_an_explicit_metres_method():
    frame = MapDepthFrame(fill=np.full((4, 4), 1.25))
    assert depth_map_metres(frame)[0, 0] == pytest.approx(1.25)


def test_depth_map_metres_uses_get_data_and_get_units_for_a_real_frame():
    """librealsense's own depth_frame has both; get_units saves plumbing the device scale."""
    raw = np.full((4, 4), 500, dtype=np.uint16)
    frame = types.SimpleNamespace(get_data=lambda: raw, get_units=lambda: 0.001)
    assert depth_map_metres(frame)[0, 0] == pytest.approx(0.5)


def test_depth_map_metres_returns_none_for_a_frame_that_offers_neither():
    """None is a supported answer: measure() then uses the per-pixel path."""
    assert depth_map_metres(FakeDepthFrame()) is None


# --- the loop's wiring -------------------------------------------------------
def test_the_per_object_log_gets_a_row_per_detection(live, tmp_path):
    detector = FakeDetector([0.001], detections_per_frame=[2, 0, 3])
    with DetectionLogger(tmp_path / "objects.csv") as objects:
        live([good_frame() for _ in range(3)], detector, detection_logger=objects)

    rows = read_log(tmp_path / "objects.csv")
    assert [row["frame"] for row in rows] == ["0", "0", "2", "2", "2"]


def test_the_two_logs_agree_on_the_frame_number(live, tmp_path):
    """They are joined on it, and a dropped frame must not shift one against the other."""
    detector = FakeDetector([0.001], detections_per_frame=[1, 1, 1])
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as frames:
        with DetectionLogger(tmp_path / "objects.csv", start=frames.start) as objects:
            live([good_frame(), dropped_frame(), good_frame(), good_frame()],
                 detector, frames, detection_logger=objects)

    frame_numbers = [row["frame"] for row in read_log(tmp_path / "run.csv")]
    object_frames = [row["frame"] for row in read_log(tmp_path / "objects.csv")]
    assert frame_numbers == ["0", "1", "2"]
    assert object_frames == ["0", "1", "2"], "the dropped frame shifted the per-object log"


def test_object_ids_persist_across_frames_in_the_log(live, tmp_path):
    """The point of the id column: the same object is recognisable from row to row."""
    class Drifting(FakeDetector):
        def predict(self, frame):
            super().predict(frame)
            x = 100 + self.calls * 6
            return [Detection((x, 100, x + 80, 180), 3, 0.9)]

    with DetectionLogger(tmp_path / "objects.csv") as objects:
        live([good_frame() for _ in range(5)], Drifting([0.001]),
             detection_logger=objects, tracker=Tracker())

    ids = [row["object_id"] for row in read_log(tmp_path / "objects.csv")]
    assert len(ids) == 5
    assert len(set(ids)) == 1, f"one object got {len(set(ids))} ids: {ids}"


def test_without_a_tracker_the_id_is_the_index_within_the_frame(live, tmp_path):
    detector = FakeDetector([0.001], detections_per_frame=[3, 3])
    with DetectionLogger(tmp_path / "objects.csv") as objects:
        live([good_frame(), good_frame()], detector, detection_logger=objects, tracker=None)

    ids = [row["object_id"] for row in read_log(tmp_path / "objects.csv")]
    assert ids == ["0", "1", "2", "0", "1", "2"]


def test_the_per_class_counts_reach_the_per_frame_log(live, tmp_path):
    detector = FakeDetector([0.001], detections_per_frame=[4])
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom",
                   class_names=CLASS_NAMES) as logger:
        live([good_frame()], detector, logger)

    row = read_log(tmp_path / "run.csv")[0]
    # FakeDetector cycles the classes, so four detections is one of each.
    assert row["n_detections"] == "4"
    assert row["n_glass"] == row["n_metal"] == row["n_paper"] == row["n_plastic"] == "1"


def test_the_ids_are_only_drawn_when_asked(live):
    import camera.segmentation_realsense as module

    labels = []
    original = module.cv2.putText
    try:
        module.cv2.putText = lambda img, text, *a, **k: labels.append(text)
        live([good_frame()], FakeDetector([0.001]), keys=[27], tracker=Tracker())
        assert not any("#" in text for text in labels), labels

        labels.clear()
        live([good_frame()], FakeDetector([0.001]), keys=[27], tracker=Tracker(),
             show_ids=True)
        assert all(text.startswith("#") for text in labels), labels
    finally:
        module.cv2.putText = original


# --- --show-axis -------------------------------------------------------------
class MaskedDetector(FakeDetector):
    """One detection per frame, carrying a real mask, so there is an axis to draw."""

    def __init__(self, mask, box=(100, 200, 300, 400)):
        super().__init__([0.001])
        self.mask = mask
        self.box = box

    def predict(self, frame):
        super().predict(frame)
        return [Detection(self.box, 0, 0.9, self.mask)]


def line_spy(monkeypatch):
    """Collects (start, end) of every cv2.line the annotation draws.

    cv2.line is used by neither annotate() nor draw_masks(), so nothing else lands in here.
    """
    drawn = []
    import camera.segmentation_realsense as module
    monkeypatch.setattr(module.cv2, "line",
                        lambda img, start, end, *a, **k: drawn.append((start, end)))
    return drawn


def test_the_axis_is_only_drawn_when_asked(live, monkeypatch):
    drawn = line_spy(monkeypatch)
    detector = MaskedDetector(diagonal_mask(100, 300, 100, rise=-1))

    live([good_frame()], detector, keys=[27], mask_depth=True)
    assert drawn == [], "nothing should be drawn without --show-axis"

    live([good_frame()], detector, keys=[27], mask_depth=True, show_axis=True)
    assert len(drawn) == 1


def test_the_drawn_axis_leans_the_way_the_angle_says(live, monkeypatch):
    """The second half of the sign problem, and it is a different line of code from the first.

    The angle is stored in a y-up frame and the canvas is y-down, so draw_axis negates the row
    component on the way back out. Forget that and the picture is mirrored while the CSV stays
    right -- which defeats the whole point of having an overlay to check the CSV against. On a
    "/" mask the endpoint further right must therefore sit *higher* up the frame, i.e. at the
    smaller row.
    """
    drawn = line_spy(monkeypatch)
    live([good_frame()], MaskedDetector(diagonal_mask(100, 300, 100, rise=-1)),
         keys=[27], mask_depth=True, show_axis=True)

    (start, end), = drawn
    right, left = sorted((start, end))[::-1]
    assert right[1] < left[1], f"a '/' object was drawn leaning like '\\': {start} -> {end}"

    drawn.clear()
    live([good_frame()], MaskedDetector(diagonal_mask(100, 100, 100, rise=+1)),
         keys=[27], mask_depth=True, show_axis=True)
    (start, end), = drawn
    right, left = sorted((start, end))[::-1]
    assert right[1] > left[1], f"a '\\' object was drawn leaning like '/': {start} -> {end}"


def test_the_drawn_axis_is_centred_on_the_centroid(live, monkeypatch):
    drawn = line_spy(monkeypatch)
    mask = mask_at(100, 200, 300, 210)
    live([good_frame()], MaskedDetector(mask), keys=[27], mask_depth=True, show_axis=True)

    (start, end), = drawn
    rows, columns = np.nonzero(mask)
    assert (start[0] + end[0]) / 2 == pytest.approx(columns.mean(), abs=1)
    assert (start[1] + end[1]) / 2 == pytest.approx(rows.mean(), abs=1)


def test_nothing_is_drawn_for_a_detection_with_no_orientation(live, monkeypatch):
    """A box-only checkpoint under --show-axis draws no axis rather than a fabricated one."""
    drawn = line_spy(monkeypatch)
    live([good_frame()], FakeDetector([0.001]), keys=[27], mask_depth=True, show_axis=True)
    assert drawn == []


def test_the_axis_reaches_the_video(live):
    """--show-axis needs no video wiring of its own: the recorder is handed the drawn frame.

    Asserted on the pixels rather than on identity alone, because identity is already pinned by
    test_the_recorder_is_handed_the_frame_that_was_displayed and would pass even if the axis were
    drawn onto some other copy of the frame.
    """
    import camera.segmentation_realsense as module

    recorder = RecordingSpy()
    live([good_frame()], MaskedDetector(diagonal_mask(100, 300, 100, rise=-1)),
         keys=[27], recorder=recorder, mask_depth=True, show_axis=True)

    frame, = recorder.frames
    assert np.any(np.all(frame == module.AXIS_COLOUR, axis=-1)), \
        "the axis colour never made it into the frame handed to the recorder"


# --- the recorder stays out of the way ---------------------------------------
class RecordingSpy:
    """Stands in for a VideoRecorder: records what it was handed, instantly."""

    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)


def test_the_recorder_is_handed_the_frame_that_was_displayed(live):
    shown = []
    recorder = RecordingSpy()

    import camera.segmentation_realsense as module
    original = module.cv2.imshow
    try:
        module.cv2.imshow = lambda window, frame: shown.append(frame)
        live([good_frame() for _ in range(3)], FakeDetector([0.001]), recorder=recorder)
    finally:
        module.cv2.imshow = original

    assert len(recorder.frames) == 3
    for displayed, recorded in zip(shown, recorder.frames):
        assert recorded is displayed, "the video must be what the window showed"


def test_a_dropped_frame_is_not_recorded(live):
    recorder = RecordingSpy()
    live([good_frame(), dropped_frame(), good_frame()], FakeDetector([0.001]),
         recorder=recorder)
    assert len(recorder.frames) == 2


def test_the_recorders_write_is_outside_the_measured_span(live, tmp_path):
    """The claim --save-video rests on. A slow write() must not inflate loop_ms."""
    class SlowWrite:
        def write(self, frame):
            time.sleep(0.04)

    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        live([good_frame() for _ in range(4)], FakeDetector([0.001]), logger,
             recorder=SlowWrite())

    # 40 ms a frame would be unmissable inside a loop whose work is ~1 ms.
    assert all(row["loop_ms"] < 30 for row in logger.rows), logger.rows


def test_the_per_object_log_is_written_outside_the_measured_span(live, tmp_path):
    class SlowLog:
        def log(self, frame, observations, names=None):
            time.sleep(0.04)
            return len(observations)

    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        live([good_frame() for _ in range(4)], FakeDetector([0.001]), logger,
             detection_logger=SlowLog())

    assert all(row["loop_ms"] < 30 for row in logger.rows), logger.rows


def test_the_tracker_is_inside_the_measured_span(live, tmp_path):
    """It is per-frame pipeline work, so it belongs in the measurement -- and it is cheap."""
    class SlowTracker:
        def update(self, detections, frame_shape=None):
            time.sleep(0.04)
            return list(range(len(detections)))

    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        live([good_frame() for _ in range(3)], FakeDetector([0.001]), logger,
             tracker=SlowTracker())

    assert all(row["loop_ms"] >= 40 for row in logger.rows), logger.rows


def test_inference_ms_is_untouched_by_any_of_the_extras(live, tmp_path):
    """Whatever else changes, the model's own cost must mean the same thing in every run."""
    class SlowEverything:
        def write(self, frame):
            time.sleep(0.02)

        def log(self, frame, observations, names=None):
            time.sleep(0.02)
            return len(observations)

        def update(self, detections, frame_shape=None):
            time.sleep(0.02)
            return list(range(len(detections)))

    slow = SlowEverything()
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        live([good_frame() for _ in range(3)], FakeDetector([0.002]), logger,
             recorder=slow, detection_logger=slow, tracker=slow, mask_depth=True)

    assert all(row["inference_ms"] < 15 for row in logger.rows), logger.rows
