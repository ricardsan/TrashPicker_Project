"""Replaying a rosbag through the segmentation pipeline.

Everything here is synthetic -- a handful of tiny PNGs and a meta.json written into tmp_path --
so none of it needs a bag, a camera or a checkpoint. The one test that does is marked `slow`.

The three things worth pinning are the ones a reader cannot check by eye:

  * the geometry. A frame is halved before the model sees it, so the intrinsics are halved too
    and the depth lookup has to map back. If that is wrong every X/Y/Z is quietly wrong, and
    nothing crashes to say so.
  * the pacing. Which frames a too-slow model misses decides which frames its detection counts
    describe, and the warm-up resync is the difference between losing a fifth of a 10 s clip
    and losing none of it.
  * that the replay really does run the *same* loop as the camera. The assertions at the bottom
    deliberately mirror camera/tests/test_run_live.py's: if the two sources ever stop sharing
    run_frames, these are what notice.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from camera import rosbag_frames
from camera.benchmark import (CAMERA_FPS, RunLogger, read_log, stream_fps_of,
                              summarise)
from camera.rosbag_frames import (ExportedDepth, ExportedFrameSource, default_export_dir,
                                  intrinsics_from_dict, is_exported, read_meta,
                                  scaled_intrinsics)
from camera.segmentation_realsense import (Detection, build_parser, build_source, main,
                                           run_frames, yolo_padded_shape)

# Deliberately not 2:1, so a test cannot pass by confusing the two axes.
NATIVE_W, NATIVE_H = 64, 48
STREAM_FPS = 15.0
FRAME_PERIOD = 1.0 / STREAM_FPS

INTRINSICS = {
    "width": NATIVE_W,
    "height": NATIVE_H,
    "ppx": 31.5,
    "ppy": 23.5,
    "fx": 40.0,
    "fy": 41.0,          # different from fx on purpose: a single scale factor must do both
    "model": "inverse_brown_conrady",
    "coeffs": [0.0] * 5,
}


# =============================================================================
# a synthetic export
# =============================================================================
@pytest.fixture
def export(tmp_path):
    """Factory for an unpacked-bag directory: PNGs plus meta.json, as export_bag writes them."""

    def make(n_frames=4, depth_scale=0.001, fps=STREAM_FPS, complete=True, name="bag"):
        directory = tmp_path / name
        (directory / "color").mkdir(parents=True)
        (directory / "depth").mkdir(parents=True)

        records = []
        for i in range(n_frames):
            # Each frame a uniform, distinguishable value, so order is checkable from the pixels.
            color = np.full((NATIVE_H, NATIVE_W, 3), (i * 10) % 256, dtype=np.uint8)
            # A gradient, so a lookup landing on the wrong pixel gives a different number rather
            # than coincidentally the right one.
            depth = np.arange(NATIVE_H * NATIVE_W, dtype=np.uint16).reshape(NATIVE_H, NATIVE_W)
            depth = depth + i
            cv2.imwrite(str(directory / f"color/{i:06d}.png"), color)
            cv2.imwrite(str(directory / f"depth/{i:06d}.png"), depth)
            records.append({
                "index": i,
                "timestamp_ms": 1000.0 + i * (1000.0 / fps),
                "frame_number": i + 1,
                "color": f"color/{i:06d}.png",
                "depth": f"depth/{i:06d}.png",
                "cloud": None,
            })

        meta = {
            "bag": "/nowhere/bag.db3",
            "bag_name": name,
            "stream_fps": fps,
            "duration_s": n_frames / fps,
            "depth_scale": depth_scale,
            "color_intrinsics": INTRINSICS,
            "depth_aligned_to_color": True,
            "clouds": False,
            "cloud_stride": 1,
            "complete": complete,
            "frames": records,
        }
        (directory / "meta.json").write_text(json.dumps(meta))
        return directory

    return make


class FakeClock:
    """A clock the test drives, so pacing can be checked without sleeping through a bag."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += max(seconds, 0.0)

    def advance(self, seconds):
        """Simulate time passing outside the source, i.e. the model working."""
        self.now += seconds


def paced(export_dir, clock, **kwargs):
    kwargs.setdefault("preload", False)
    return ExportedFrameSource(export_dir, sleep=clock.sleep, clock=clock, verbose=False,
                               **kwargs)


# =============================================================================
# geometry
# =============================================================================
def test_intrinsics_round_trip_unscaled(export):
    meta = read_meta(export())
    intrinsics = intrinsics_from_dict(meta["color_intrinsics"])
    assert (intrinsics.width, intrinsics.height) == (NATIVE_W, NATIVE_H)
    assert intrinsics.fx == pytest.approx(INTRINSICS["fx"])
    assert intrinsics.fy == pytest.approx(INTRINSICS["fy"])
    assert intrinsics.ppx == pytest.approx(INTRINSICS["ppx"])
    assert intrinsics.ppy == pytest.approx(INTRINSICS["ppy"])
    assert list(intrinsics.coeffs) == INTRINSICS["coeffs"]


@pytest.mark.parametrize("model", ["none", "modified_brown_conrady", "inverse_brown_conrady",
                                   "ftheta", "brown_conrady", "kannala_brandt4"])
def test_every_distortion_model_survives_the_json(model):
    """intrinsics_to_dict writes the enum as `str(model).split('.')[-1]`, which is the fragile
    half of the round trip: a spelling that does not come back is a silently wrong projection."""
    import pyrealsense2 as rs

    from camera.rosbag_frames import intrinsics_to_dict

    original = rs.intrinsics()
    original.width, original.height = NATIVE_W, NATIVE_H
    original.ppx, original.ppy = INTRINSICS["ppx"], INTRINSICS["ppy"]
    original.fx, original.fy = INTRINSICS["fx"], INTRINSICS["fy"]
    original.model = getattr(rs.distortion, model)
    original.coeffs = [0.1, -0.2, 0.3, -0.4, 0.5]

    restored = intrinsics_from_dict(intrinsics_to_dict(original))
    assert restored.model == original.model
    assert list(restored.coeffs) == pytest.approx(list(original.coeffs))


def test_scaling_halves_every_length(export):
    half = scaled_intrinsics(read_meta(export()), 0.5)
    assert (half.width, half.height) == (NATIVE_W // 2, NATIVE_H // 2)
    assert half.fx == pytest.approx(INTRINSICS["fx"] / 2)
    assert half.fy == pytest.approx(INTRINSICS["fy"] / 2)
    assert half.ppx == pytest.approx(INTRINSICS["ppx"] / 2)
    assert half.ppy == pytest.approx(INTRINSICS["ppy"] / 2)
    # Distortion is dimensionless and must NOT be scaled.
    assert list(half.coeffs) == INTRINSICS["coeffs"]


@pytest.mark.parametrize("coeffs", [[0.0] * 5, [0.09, -0.21, 0.001, -0.002, 0.13]])
def test_deprojection_survives_the_downscale(export, coeffs):
    """The whole point of halving the intrinsics: the same point, in metres, either way.

    If fx/fy were scaled but ppx/ppy were not (or vice versa) every off-centre X and Y would be
    wrong while Z stayed right -- which no amount of looking at the window would reveal.

    Run with nonzero distortion as well as without, because that is the case that proves the
    other half of the decision: coeffs are dimensionless and must NOT be scaled. With all-zero
    coeffs -- which is what the D435i colour stream reports -- scaling them wrongly would make
    no difference and the test would pass regardless.
    """
    import pyrealsense2 as rs

    meta = read_meta(export())
    meta["color_intrinsics"]["coeffs"] = coeffs
    native = intrinsics_from_dict(meta["color_intrinsics"])
    half = scaled_intrinsics(meta, 0.5)

    for x, y in [(0, 0), (5, 3), (16, 12), (31, 23)]:
        z = 0.65
        in_half = rs.rs2_deproject_pixel_to_point(half, [x, y], z)
        in_native = rs.rs2_deproject_pixel_to_point(native, [x * 2, y * 2], z)
        assert in_half == pytest.approx(in_native, abs=1e-9), f"at ({x}, {y})"


def test_an_unknown_distortion_model_is_refused(export):
    """Better a clear error than deprojecting with whatever enum happened to be first."""
    meta = read_meta(export())
    meta["color_intrinsics"]["model"] = "not_a_real_model"
    with pytest.raises(ValueError, match="not_a_real_model"):
        intrinsics_from_dict(meta["color_intrinsics"])


def test_yolo_padded_shape_matches_the_two_cases_that_matter():
    """What makes a bag latency comparable with a live one, or not."""
    assert yolo_padded_shape((480, 640)) == (480, 640)      # live: 307k px
    assert yolo_padded_shape((360, 640)) == (384, 640)      # bag:  246k px


# =============================================================================
# depth
# =============================================================================
def depth_of(export_dir, index=0):
    meta = read_meta(export_dir)
    array = cv2.imread(str(export_dir / meta["frames"][index]["depth"]), cv2.IMREAD_UNCHANGED)
    return array, meta["depth_scale"]


def test_depth_png_round_trips_exactly(export):
    """16-bit has to survive the write/read, or every distance is out by a factor of 256."""
    array, _ = depth_of(export())
    assert array.dtype == np.uint16
    expected = (np.arange(NATIVE_H * NATIVE_W, dtype=np.uint16).reshape(NATIVE_H, NATIVE_W))
    assert np.array_equal(array, expected)


def test_get_distance_is_the_raw_value_times_the_scale(export):
    array, scale = depth_of(export())
    depth = ExportedDepth(array, scale, scale=1.0)
    assert depth.get_distance(7, 3) == pytest.approx(array[3, 7] * scale)


def test_get_distance_maps_scaled_coordinates_back(export):
    """A box centre is in the halved frame; the depth map never was."""
    array, scale = depth_of(export())
    depth = ExportedDepth(array, scale, scale=0.5)
    # (7, 3) in the halved frame is (14, 6) natively.
    assert depth.get_distance(7, 3) == pytest.approx(array[6, 14] * scale)


def test_get_distance_clamps_instead_of_raising(export):
    """pyrealsense2 raises here, and a centre one pixel past the edge used to kill whole runs.

    camera/tests/test_run_live.py guards the live path against exactly this; a replay must not
    reintroduce it by another route.
    """
    array, scale = depth_of(export())
    depth = ExportedDepth(array, scale, scale=0.5)
    assert depth.get_distance(10_000, 10_000) == pytest.approx(array[-1, -1] * scale)
    assert depth.get_distance(-5, -5) == pytest.approx(array[0, 0] * scale)


def test_zero_depth_reads_as_zero_metres(export):
    """0 in a Z16 frame means "no reading", which librealsense reports as 0.0."""
    array, scale = depth_of(export())
    array = array.copy()
    array[3, 7] = 0
    assert ExportedDepth(array, scale, scale=1.0).get_distance(7, 3) == 0.0


# --- the whole map at once, for measuring over a mask ------------------------
# measure() asks for metres() once a frame when it is taking an object's distance over its mask
# rather than at one point. The two paths must not be able to disagree: a difference between them
# would be invisible in the picture and present in every row of the per-object CSV.


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.75, 0.3])
def test_metres_agrees_with_get_distance_everywhere(export, scale):
    """Pixel for pixel, at every scale -- which is why it is built with get_distance's own
    index arithmetic rather than with cv2.resize. A nearest-neighbour resize maps output column x
    to floor((x + 0.5) / scale) while get_distance uses round(x / scale), and at COLOR_SCALE those
    differ almost everywhere.
    """
    array, depth_scale = depth_of(export())
    depth = ExportedDepth(array, depth_scale, scale=scale)

    metres = depth.metres()
    height, width = metres.shape
    for y in range(height):
        for x in range(width):
            assert metres[y, x] == depth.get_distance(x, y), f"disagreed at ({x}, {y})"


def test_metres_is_the_colour_frames_shape_so_a_mask_can_index_it(export):
    """The mask is the colour frame's size; the depth array is not."""
    directory = export()
    array, depth_scale = depth_of(directory)
    source = ExportedFrameSource(directory, verbose=False)
    depth = ExportedDepth(array, depth_scale, scale=rosbag_frames.COLOR_SCALE)
    assert depth.metres().shape == source.frame_shape


def test_metres_keeps_a_no_reading_as_zero(export):
    """0 has to survive as 0, because that is the signal measure() excludes it by."""
    array, depth_scale = depth_of(export())
    array = array.copy()
    array[:, :] = 0
    array[10, 10] = 1000
    metres = ExportedDepth(array, depth_scale, scale=1.0).metres()
    assert metres[0, 0] == 0.0
    assert metres[10, 10] == pytest.approx(1000 * depth_scale)


def test_metres_does_not_alias_the_stored_depth(export):
    """It is handed to measure(), which must not be able to corrupt the frame cache."""
    array, depth_scale = depth_of(export())
    depth = ExportedDepth(array, depth_scale, scale=1.0)
    metres = depth.metres()
    metres[0, 0] = 99.0
    assert depth.get_distance(0, 0) != 99.0


# =============================================================================
# the export directory
# =============================================================================
def test_read_meta_says_what_is_wrong(tmp_path):
    with pytest.raises(SystemExit, match="not an unpacked bag"):
        read_meta(tmp_path)


def test_is_exported_requires_a_complete_export(export, tmp_path):
    assert is_exported(export())
    assert not is_exported(tmp_path / "nothing")


def test_a_half_written_export_is_not_reused(export):
    """export_bag writes `complete` last, so an interrupted unpack is redone rather than read."""
    directory = export(complete=False)
    assert not is_exported(directory)


def test_a_corrupt_meta_is_not_reused(export):
    directory = export()
    (directory / "meta.json").write_text("{ this is not json")
    assert not is_exported(directory)


def test_default_export_dir_is_named_after_the_bag():
    assert default_export_dir("rosbags/20261016_233733_stacked.db3", "rosbags/export").name \
        == "20261016_233733_stacked"


def test_the_estimate_warns_about_the_clouds():
    """The point clouds are two orders of magnitude bigger than the images."""
    without = rosbag_frames.export_estimate(142, clouds=False, cloud_stride=1)
    with_clouds = rosbag_frames.export_estimate(142, clouds=True, cloud_stride=1)
    assert "MiB" in without
    assert "GiB" in with_clouds


@pytest.mark.parametrize("stride", [0, -1])
def test_a_useless_cloud_stride_is_refused(stride):
    """0 divides by zero in the estimate and in `index % stride`; a negative one would export
    every frame while reading like it thinned them."""
    with pytest.raises(SystemExit, match="cloud-stride"):
        rosbag_frames.check_cloud_stride(stride)


def test_a_write_that_fails_is_not_ignored(tmp_path, monkeypatch):
    """cv2.imwrite returns False on a full disk rather than raising.

    Ignored, that stamps an export `complete: True` with frames missing from it -- which
    is_exported then happily reuses forever, and the run dies much later, after a checkpoint has
    been loaded.
    """
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(SystemExit, match="full disk"):
        rosbag_frames.write_image(tmp_path / "x.png", np.zeros((4, 4), dtype=np.uint8))


# --- reusing an export ---------------------------------------------------------
def test_an_export_with_what_was_asked_for_is_reused(export):
    assert rosbag_frames.reuse_shortfall(read_meta(export()), clouds=False, cloud_stride=1) \
        is None


def test_an_export_without_clouds_is_not_reused_when_clouds_are_wanted(export):
    """The live export in rosbags/ was made with --cloud-stride 40, and before this check a
    plain --rosbag reused it and produced four clouds while reporting success."""
    meta = read_meta(export())           # the fixture writes clouds: False
    assert "no-clouds" in rosbag_frames.reuse_shortfall(meta, clouds=True, cloud_stride=1)


def test_a_coarser_cloud_stride_is_not_reused(export):
    meta = read_meta(export())
    meta["clouds"], meta["cloud_stride"] = True, 40
    assert "coarser" in rosbag_frames.reuse_shortfall(meta, clouds=True, cloud_stride=1)
    # ...but a finer existing export covers a coarser request.
    assert rosbag_frames.reuse_shortfall(meta, clouds=True, cloud_stride=40) is None
    meta["cloud_stride"] = 1
    assert rosbag_frames.reuse_shortfall(meta, clouds=True, cloud_stride=10) is None


def test_a_truncated_meta_explains_itself(export):
    """is_exported handles this; --from-export comes straight through read_meta."""
    directory = export()
    (directory / "meta.json").write_text('{"frames": [')
    with pytest.raises(SystemExit, match="interrupted"):
        read_meta(directory)


# =============================================================================
# serving frames
# =============================================================================
def test_frames_come_out_in_order_and_then_stop(export):
    clock = FakeClock()
    source = paced(export(n_frames=3), clock, real_time=False)
    source.start()

    seen = [source.grab() for _ in range(3)]
    assert all(frame is not None for frame in seen)
    # Each synthetic colour frame has its own constant value, so order is checkable.
    assert [int(frame[0][0, 0, 0]) for frame in seen] == [0, 10, 20]

    with pytest.raises(StopIteration):
        source.grab()


def test_the_colour_frame_is_halved_and_the_depth_is_not(export):
    clock = FakeClock()
    source = paced(export(), clock, real_time=False)
    source.start()
    color, depth, intrinsics = source.grab()

    assert color.shape == (NATIVE_H // 2, NATIVE_W // 2, 3)
    assert depth.depth.shape == (NATIVE_H, NATIVE_W)
    assert (intrinsics.width, intrinsics.height) == (NATIVE_W // 2, NATIVE_H // 2)
    assert source.frame_shape == (NATIVE_H // 2, NATIVE_W // 2)


def test_preloading_serves_the_same_pixels(export):
    """The cache must not change what the model is given, only when it was decoded."""
    directory = export(n_frames=3)
    clock = FakeClock()

    lazy = paced(directory, clock, real_time=False)
    lazy.start()
    eager = ExportedFrameSource(directory, real_time=False, preload=True, verbose=False)
    eager.start()

    for _ in range(3):
        a, depth_a, _ = lazy.grab()
        b, depth_b, _ = eager.grab()
        assert np.array_equal(a, b)
        assert np.array_equal(depth_a.depth, depth_b.depth)


def test_preloading_a_trimmed_export_serves_the_right_frames(export):
    """The cache is keyed on each record's own `index`, not on its position in the list.

    export_bag numbers contiguously from 0 so the two agree for anything it writes, but a
    meta.json holding a subrange -- a hand-trimmed export -- would make a positional cache serve
    the wrong frame for every record while the lazy path stayed correct. Wrong pixels, no error.
    """
    directory = export(n_frames=6)
    meta = json.loads((directory / "meta.json").read_text())
    meta["frames"] = meta["frames"][3:]          # indices 3, 4, 5 -- positions 0, 1, 2
    (directory / "meta.json").write_text(json.dumps(meta))

    eager = ExportedFrameSource(directory, real_time=False, preload=True, verbose=False)
    lazy = ExportedFrameSource(directory, real_time=False, preload=False, verbose=False)
    eager.start()
    lazy.start()

    for _ in range(3):
        assert np.array_equal(eager.grab()[0], lazy.grab()[0])


def test_an_oversized_clip_is_not_preloaded(export, monkeypatch, capsys):
    """RAM is finite; the fallback has to say what it costs the numbers."""
    monkeypatch.setattr(rosbag_frames, "PRELOAD_LIMIT_BYTES", 1)
    source = ExportedFrameSource(export(), preload=True, verbose=True)
    assert source._decoded is None
    assert "not preloading" in capsys.readouterr().out


def test_a_replay_never_blames_a_stall(export):
    """explain_stall is the camera's USB-2 story. A RuntimeError in a replay is a real fault."""
    clock = FakeClock()
    source = paced(export(), clock)
    assert source.explain_stall(RuntimeError("Frame didn't arrive within 5000")) is None


# =============================================================================
# pacing
# =============================================================================
def test_a_fast_model_misses_nothing(export):
    clock = FakeClock()
    source = paced(export(n_frames=5), clock)
    source.start()

    for _ in range(5):
        assert source.grab() is not None
        clock.advance(0.001)            # a model far quicker than the frame period
    assert source.skipped == 0


def replay(source, clock, model_s, limit=200):
    """Drive a source the way run_frames does, with a model of a given cost.

    The distinction that matters: only a frame that is actually handed over costs model time. A
    skipped one is a cheap trip round the loop, which is how a too-slow model catches back up to
    the wall clock instead of falling behind forever.
    """
    shown = 0
    for _ in range(limit):
        try:
            frame = source.grab()
        except StopIteration:
            break
        if frame is not None:
            shown += 1
            clock.advance(model_s)
    return shown


def test_a_slow_model_misses_frames_but_keeps_up_with_the_clock(export):
    """A model at 2.5x the frame period shows roughly every third frame -- not the first and then
    nothing.

    Falling permanently behind would be the easy bug here, and it would be invisible in the
    summary: the run would still end, just with almost no rows. The real Mask R-CNN run is the
    reference point -- 206 ms a frame on a 66.6 ms stream, 45 of 142 frames shown, which is the
    142 / 3.1 this predicts.
    """
    n_frames = 30
    clock = FakeClock()
    source = paced(export(n_frames=n_frames), clock)
    source.start()

    shown = replay(source, clock, model_s=FRAME_PERIOD * 2.5)

    assert source.skipped > 0, "a model slower than the stream has to miss frames"
    assert shown + source.skipped == n_frames, "every frame is either shown or skipped"
    # 1 in 2.5, give or take the warm-up resync and rounding at the ends.
    assert n_frames / 3.5 <= shown <= n_frames / 2.0, shown


def test_the_warm_up_frame_does_not_cost_the_rest_of_the_clip(export):
    """The one-off CUDA warm-up is ~1.7 s, which is 26 frame periods of a 15 Hz bag.

    Without the resync a fifth of a 10 s clip is skipped before the model has run at its real
    speed even once -- and a model with a slower warm-up is then scored on fewer frames than a
    quicker one, so their detection counts are not comparable. Measured on the real bag: 27 of
    142 frames skipped before this, 0 after.
    """
    clock = FakeClock()
    source = paced(export(n_frames=10), clock)
    source.start()

    assert source.grab() is not None    # the warm-up frame itself
    clock.advance(1.7)                  # what loading CUDA actually costs

    assert source.grab() is not None, "the frame after the warm-up must not be skipped"
    assert source.rebased

    for _ in range(8):
        clock.advance(0.002)
        assert source.grab() is not None
    assert source.skipped == 0


def test_the_resync_happens_only_once(export):
    """Or it would hide real slowness as repeated forgiveness."""
    clock = FakeClock()
    source = paced(export(n_frames=12), clock)
    source.start()

    source.grab()
    clock.advance(1.7)
    source.grab()                       # the resync
    assert source.rebased and source.skipped == 0

    replay(source, clock, model_s=FRAME_PERIOD * 4)
    assert source.skipped > 0, "after the resync, being late must cost frames again"


def test_nothing_is_skipped_when_pacing_is_off(export):
    clock = FakeClock()
    source = paced(export(n_frames=6), clock, real_time=False)
    source.start()
    for _ in range(6):
        clock.advance(10.0)
        assert source.grab() is not None
    assert source.skipped == 0
    assert not clock.slept


# =============================================================================
# the shared loop -- these mirror camera/tests/test_run_live.py on purpose
# =============================================================================
class FakeDetector:
    names = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}

    def __init__(self, latency_s=0.0, detections=1):
        self.latency_s = latency_s
        self.detections = detections
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        if self.latency_s:
            deadline = time.perf_counter() + self.latency_s
            while time.perf_counter() < deadline:
                pass
        return [Detection((1, 1, 5, 5), i % 4, 0.9) for i in range(self.detections)]


@pytest.fixture
def headless(monkeypatch):
    """cv2 without a window, and ESC never pressed."""
    monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)
    monkeypatch.setattr("cv2.waitKey", lambda delay: -1)


def test_one_row_per_shown_frame(export, headless, tmp_path):
    source = ExportedFrameSource(export(n_frames=5), real_time=False, preload=False,
                                 verbose=False)
    detector = FakeDetector()
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom",
                   stream_fps=source.stream_fps) as logger:
        run_frames(source, detector, logger, window="test", verbose=False, masks=False)

    assert detector.calls == 5
    assert [r["frame"] for r in logger.rows] == [0, 1, 2, 3, 4]


def test_a_skipped_frame_logs_no_row_but_still_costs_time(export, headless, tmp_path):
    """Exactly what test_run_live asserts of a dropped camera frame.

    Both sources return None for a frame that did not make it, and run_frames carries its time
    into the next logged row rather than losing it -- otherwise the achieved rate flatters itself.
    """
    class Skipping(ExportedFrameSource):
        """Serves frame, None (having burnt 50 ms), frame."""

        def grab(self):
            if self.index == 1:
                self.index += 1
                time.sleep(0.05)
                return None
            return super().grab()

    source = Skipping(export(n_frames=3), real_time=False, preload=False, verbose=False)
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        run_frames(source, FakeDetector(), logger, window="test", verbose=False, masks=False)

    assert len(logger.rows) == 2, "the skipped frame must not become a row"
    assert logger.rows[1]["loop_ms"] >= 50, "its time has to land in the next period"


def test_the_real_pacer_drives_the_real_loop(export, headless, tmp_path):
    """The two halves together, which nothing else here covers.

    Everything above tests the pacer with grab() called by hand, or the loop with the pacing
    switched off. This is the arrangement an actual replay uses: a model too slow for the stream,
    frames genuinely skipped by _wait_for, and run_frames deciding what gets a row.
    """
    clock = FakeClock()
    n_frames = 20
    source = paced(export(n_frames=n_frames), clock)

    class Slow(FakeDetector):
        """Costs two frame periods of the fake clock, not of real time."""

        def predict(self, frame):
            clock.advance(FRAME_PERIOD * 2)
            return super().predict(frame)

    with RunLogger(tmp_path / "run.csv", "yolo26", "custom",
                   stream_fps=source.stream_fps) as logger:
        run_frames(source, Slow(), logger, window="test", verbose=False, masks=False)

    assert source.skipped > 0, "a model at twice the frame period has to miss frames"
    assert len(logger.rows) + source.skipped == n_frames
    assert len(logger.rows) == source.shown, "a row per frame shown, and only those"
    assert [r["frame"] for r in logger.rows] == list(range(len(logger.rows)))


def test_the_frame_after_the_warm_up_still_waits_for_the_stream(export, headless, tmp_path):
    """The resync must not hand frame 1 a period with no stream wait in it.

    Rebasing to "due now" leaves frame 1's loop_ms as model-and-draw only, so its fps reads far
    above the stream rate -- up to 38 against a 15 Hz cap on the real bag -- and plotted_rows
    drops only frame 0, so that spike opens every figure this produces.
    """
    # Real time, not the FakeClock: run_frames measures loop_ms with perf_counter, so a fake
    # clock would leave every period at ~0 and the assertion would mean nothing. A 100 Hz
    # synthetic stream keeps the whole test to about a fifth of a second.
    fast_fps = 100.0
    source = ExportedFrameSource(export(n_frames=6, fps=fast_fps), preload=True, verbose=False)

    class WarmingUp(FakeDetector):
        """One slow first call -- five frame periods of it -- then quick, like a CUDA warm-up."""

        def predict(self, frame):
            time.sleep(0.05 if self.calls == 0 else 0.0)
            return super().predict(frame)

    with RunLogger(tmp_path / "run.csv", "yolo26", "custom",
                   stream_fps=source.stream_fps) as logger:
        run_frames(source, WarmingUp(), logger, window="test", verbose=False, masks=False)

    assert source.skipped == 0, "the warm-up must not cost the rest of the clip"
    # Frame 0 carries the warm-up and is excluded from the statistics. Every frame after it
    # waited for the stream, so none can report a rate far above it -- a "due now" rebase gives
    # frame 1 no wait at all and sends its fps into the hundreds.
    assert len(logger.rows) == 6
    for row in logger.rows[1:]:
        assert row["fps"] <= fast_fps * 1.5, logger.rows


def test_the_end_of_the_bag_ends_the_run_quietly(export, headless, tmp_path):
    """StopIteration is a clean finish, not a traceback."""
    source = ExportedFrameSource(export(n_frames=2), real_time=False, preload=False,
                                 verbose=False)
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        run_frames(source, FakeDetector(), logger, window="test", verbose=False, masks=False)
    assert len(logger.rows) == 2


def test_inference_ms_still_tracks_the_model_alone(export, headless, tmp_path):
    """The reason the loop is shared: this has to mean the same thing on a bag as on a camera."""
    source = ExportedFrameSource(export(n_frames=3), real_time=False, preload=False,
                                 verbose=False)
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom") as logger:
        run_frames(source, FakeDetector(latency_s=0.03), logger, window="test", verbose=False,
                   masks=False)

    assert all(r["inference_ms"] >= 25 for r in logger.rows), logger.rows
    assert all(r["loop_ms"] >= r["inference_ms"] for r in logger.rows)


def test_a_detector_that_blows_up_still_closes_the_source(export, headless, tmp_path):
    class Exploding(FakeDetector):
        def predict(self, frame):
            raise RuntimeError("CUDA out of memory")

    source = ExportedFrameSource(export(), real_time=False, preload=False, verbose=False)
    stopped = []
    source.stop = lambda: stopped.append(True)

    with pytest.raises(RuntimeError, match="CUDA"):
        run_frames(source, Exploding(), None, window="test", verbose=False, masks=False)
    assert stopped


# =============================================================================
# the stream rate in the log
# =============================================================================
def test_the_log_records_the_rate_the_frames_arrived_at(tmp_path):
    with RunLogger(tmp_path / "run.csv", "yolo26", "custom", stream_fps=15.0) as logger:
        logger.log(1, 0.02, 0.066)
    assert float(logger.rows[0]["stream_fps"]) == 15.0
    assert summarise(logger.rows).stream_fps == 15.0
    assert "15 Hz stream" in "\n".join(summarise(logger.rows).as_lines())


def test_a_log_from_before_the_column_existed_still_reads(tmp_path):
    """Everything already in runs/camera/ lacks it, and was recorded at CAMERA_FPS."""
    old = [{"inference_ms": "20", "loop_ms": "33", "n_detections": "2", "elapsed_s": "0.1"}]
    assert stream_fps_of(old) == CAMERA_FPS
    assert summarise(old).stream_fps == CAMERA_FPS


@pytest.mark.parametrize("value", ["", None, "nonsense", "0"])
def test_an_unusable_stream_fps_falls_back(value):
    assert stream_fps_of([{"stream_fps": value}]) == CAMERA_FPS


# =============================================================================
# the command line
# =============================================================================
def test_the_camera_is_still_the_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args([])
    source, out_dir = build_source(args)
    assert type(source).__name__ == "RealSenseSource"
    assert out_dir == type(out_dir)(".")


def test_from_export_selects_the_bag(export):
    directory = export(name="stacked")
    args = build_parser().parse_args(["--from-export", str(directory)])
    source, out_dir = build_source(args)
    assert isinstance(source, ExportedFrameSource)
    assert source.stream_fps == STREAM_FPS
    # Each bag's runs go in their own folder, so three clips do not pile up together.
    assert out_dir.name == "stacked"


def test_a_bag_and_an_export_cannot_both_be_asked_for():
    """argparse's exclusion, enforced before anything loads a checkpoint."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--rosbag", "a.db3", "--from-export", "b"])


def test_export_only_needs_a_bag():
    with pytest.raises(SystemExit, match="needs --rosbag"):
        main(["--export-only"])


def test_no_preload_reaches_the_source(export):
    directory = export()
    args = build_parser().parse_args(["--from-export", str(directory), "--no-preload"])
    source, _ = build_source(args)
    assert source._decoded is None


# =============================================================================
# the real thing
# =============================================================================
REAL_BAG = rosbag_frames.Path(__file__).resolve().parents[2] / "rosbags"


@pytest.mark.slow
def test_a_real_bag_unpacks_and_replays(tmp_path):
    """End to end on an actual .db3, with no clouds so it stays quick."""
    bags = sorted(REAL_BAG.glob("*.db3"))
    if not bags:
        pytest.skip("no bags in rosbags/")

    destination = tmp_path / "export"
    rosbag_frames.export_bag(bags[0], destination, clouds=False, verbose=False)

    meta = read_meta(destination)
    assert meta["complete"] and meta["frames"]
    assert meta["stream_fps"] > 0
    assert meta["color_intrinsics"]["width"] > 0

    source = ExportedFrameSource(destination, real_time=False, preload=False, verbose=False)
    source.start()
    color, depth, intrinsics = source.grab()

    native = meta["color_intrinsics"]
    assert color.shape[:2] == (round(native["height"] * 0.5), round(native["width"] * 0.5))
    assert depth.depth.dtype == np.uint16
    assert 0.0 <= depth.get_distance(color.shape[1] // 2, color.shape[0] // 2) < 10.0
    assert intrinsics.fx == pytest.approx(native["fx"] * 0.5)


# =============================================================================
# A WHOLE RUN'S WORTH OF ARTEFACTS, FROM A REPLAY
# =============================================================================
# The replay is the only path that can be driven end to end with no camera and no checkpoint, so
# it is where main()'s wiring gets exercised: the run folder, both CSVs, the video, and the
# promise that fast mode is still the fast mode.


@pytest.fixture
def replay_run(export, headless, monkeypatch, tmp_path):
    """Run main() over a synthetic export with a fake detector, and hand back the output dir."""

    def run(*flags, n_frames=6, detections=2):
        directory = export(name=f"bag-{abs(hash(flags)) % 10_000}", n_frames=n_frames)
        detector = FakeDetector(detections=detections)

        monkeypatch.setattr("camera.segmentation_realsense.resolve_weights",
                            lambda args: (tmp_path / "best.pt", "yolo26", "custom"))
        monkeypatch.setattr("camera.segmentation_realsense.build_detector",
                            lambda weights, architecture, conf: detector)

        out = tmp_path / f"out-{abs(hash(flags)) % 10_000}"
        assert main(["--from-export", str(directory), "--out-dir", str(out),
                     "--quiet", "--no-preload", *flags]) == 0
        return out, detector

    return run


def artefacts(out_dir):
    """{suffix: path} for the files a run left, keyed by what distinguishes them."""
    found = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            key = ("objects" if path.name.endswith("-objects.csv")
                   else path.suffix.lstrip("."))
            found[key] = path
    return found


def test_fast_mode_writes_only_the_per_frame_csv(replay_run):
    out, _ = replay_run()
    found = artefacts(out)
    assert set(found) == {"csv"}
    assert "objects" not in found, "--objects was not asked for"


def test_a_run_gets_a_folder_of_its_own(replay_run):
    out, _ = replay_run("--objects", "--plot")
    # One subdirectory, named after the run, holding everything.
    subdirectories = [p for p in out.iterdir() if p.is_dir()]
    assert len(subdirectories) == 1
    run_dir = subdirectories[0]
    assert run_dir.name.endswith("-yolo26-custom")
    for key in ("csv", "objects", "png"):
        assert artefacts(out)[key].parent == run_dir


def test_flat_puts_them_side_by_side_as_before(replay_run):
    out, _ = replay_run("--objects", "--plot", "--flat")
    assert not [p for p in out.iterdir() if p.is_dir()]
    for key in ("csv", "objects", "png"):
        assert artefacts(out)[key].parent == out


def test_the_four_artefacts_share_the_run_stem(replay_run):
    out, _ = replay_run("--objects", "--plot", "--save-video")
    found = artefacts(out)
    assert set(found) == {"csv", "objects", "png", "mp4"}
    stem = found["csv"].stem
    assert found["png"].stem == stem
    assert found["mp4"].stem == stem
    assert found["objects"].name == f"{stem}-objects.csv"


def test_the_objects_csv_has_a_row_per_detection_per_frame(replay_run):
    out, detector = replay_run("--objects", n_frames=5, detections=3)
    rows = read_log(artefacts(out)["objects"])
    assert len(rows) == detector.calls * 3
    assert {row["class_name"] for row in rows} <= set(FakeDetector.names.values())


def test_the_objects_csv_records_a_real_distance_from_the_bags_depth(replay_run):
    """End to end: the synthetic export's depth gradient must come out as metres."""
    out, _ = replay_run("--objects")
    rows = read_log(artefacts(out)["objects"])
    distances = [float(row["Z_m"]) for row in rows if row["Z_m"]]
    assert distances, "every row came back with no depth"
    assert all(d > 0 for d in distances)
    assert {row["depth_source"] for row in rows} <= {"mask-median", "centre-pixel", "none"}


def test_object_ids_are_stable_across_the_clip(replay_run):
    """FakeDetector returns the same boxes every frame, so the ids must not churn."""
    out, _ = replay_run("--objects", n_frames=6, detections=2)
    rows = read_log(artefacts(out)["objects"])
    assert {row["object_id"] for row in rows} == {"0", "1"}


def test_no_track_falls_back_to_the_index_within_the_frame(replay_run):
    out, _ = replay_run("--objects", "--no-track", detections=3)
    rows = read_log(artefacts(out)["objects"])
    assert {row["object_id"] for row in rows} == {"0", "1", "2"}


def test_the_per_frame_csv_breaks_the_count_out_per_class(replay_run):
    out, _ = replay_run(detections=4)
    rows = read_log(artefacts(out)["csv"])
    assert "n_glass" in rows[0]
    for row in rows:
        per_class = sum(int(row[column]) for column in
                        ("n_glass", "n_metal", "n_paper", "n_plastic"))
        assert per_class == int(row["n_detections"])


def test_the_extras_column_records_which_mode_the_run_was_in(replay_run):
    fast, _ = replay_run()
    assert {row["extras"] for row in read_log(artefacts(fast)["csv"])} == {""}

    full, _ = replay_run("--objects", "--save-video")
    assert {row["extras"] for row in read_log(artefacts(full)["csv"])} == {"objects+video"}


def test_the_video_holds_the_frames_that_were_shown(replay_run):
    out, detector = replay_run("--save-video", n_frames=6)
    path = artefacts(out)["mp4"]
    assert path.stat().st_size > 0

    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        decoded = 0
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    assert decoded == detector.calls


def test_the_video_is_the_size_of_the_frames_the_model_saw(replay_run, export):
    """Not the bag's native size: the replay halves the colour frame for the model."""
    out, _ = replay_run("--save-video")
    capture = cv2.VideoCapture(str(artefacts(out)["mp4"]))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    assert (height, width) == ExportedFrameSource(export(name="shape"), verbose=False).frame_shape


def test_an_explicit_video_path_is_taken_literally(replay_run, tmp_path):
    """--save-video PATH names the destination, so it gets no run folder prefixed on."""
    destination = tmp_path / "slides" / "clip.mp4"
    out, _ = replay_run("--save-video", str(destination))
    assert destination.exists()
    assert "mp4" not in artefacts(out)


def test_no_log_still_writes_the_video_and_creates_its_folder(replay_run):
    """There is no RunLogger to make the directory in this combination."""
    out, _ = replay_run("--no-log", "--save-video")
    found = artefacts(out)
    assert set(found) == {"mp4"}
    assert found["mp4"].stat().st_size > 0


def test_no_log_suppresses_both_csvs(replay_run):
    out, _ = replay_run("--no-log", "--objects")
    assert artefacts(out) == {}


def test_plot_from_refuses_the_per_object_log(replay_run, capsys):
    """The two differ by a suffix, so pointing the figure at the wrong one is easy."""
    out, _ = replay_run("--objects")
    objects_csv = artefacts(out)["objects"]
    with pytest.raises(SystemExit, match="per-object log"):
        main(["--plot-from", str(objects_csv)])


def test_plot_from_still_works_on_the_per_frame_log(replay_run):
    out, _ = replay_run("--objects")
    assert main(["--plot-from", str(artefacts(out)["csv"])]) == 0
    assert artefacts(out)["png"].exists()


def test_the_run_names_its_folder_and_files_on_the_way_out(replay_run, capsys):
    out, _ = replay_run("--objects", "--save-video")
    printed = capsys.readouterr().out
    assert "writing this run to" in printed
    assert "objects: " in printed
    assert "video: " in printed
    assert "extras: objects+video" in printed


def test_fast_mode_says_so(replay_run, capsys):
    replay_run()
    assert "fast path: no extra per-frame work" in capsys.readouterr().out


# --- the artefacts survive a failure in the figure ---------------------------
def test_a_failing_figure_does_not_corrupt_the_video(replay_run, monkeypatch):
    """plot_run used to run before the recorder was closed.

    When it raised, ffmpeg's stdin was never closed and the daemon worker died at interpreter
    exit, leaving an MP4 with no moov atom: a file of plausible size that no player will open.
    The artefacts are the point of a run; the figure can be rebuilt with --plot-from.
    """
    import camera.segmentation_realsense as module

    def explode(*a, **k):
        raise RuntimeError("no display, no disk, pick one")

    monkeypatch.setattr(module, "plot_run", explode)
    with pytest.raises(RuntimeError, match="pick one"):
        replay_run("--objects", "--save-video", "--plot")


def test_the_video_from_that_failed_run_still_plays(replay_run, monkeypatch, tmp_path):
    """The same scenario, checked on the file rather than on the exception."""
    import camera.segmentation_realsense as module

    monkeypatch.setattr(module, "plot_run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        replay_run("--save-video", "--plot", n_frames=6)

    videos = list(tmp_path.rglob("*.mp4"))
    assert len(videos) == 1, videos
    capture = cv2.VideoCapture(str(videos[0]))
    try:
        assert capture.isOpened(), "the MP4 was left unfinalised"
        decoded = 0
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    assert decoded > 0


# --- extras has to mean what benchmark.py says it means ----------------------
def test_centre_depth_is_labelled_as_the_fast_path_it_is(replay_run):
    """--objects --centre-depth writes the per-object CSV on the identical fast code path.

    benchmark.py tells a reader that two runs are comparable on loop_ms when their `extras`
    agree, so a run costing 0.01 ms a frame must not be labelled the same as one costing 3.4.
    """
    out, _ = replay_run("--objects", "--centre-depth")
    assert {row["extras"] for row in read_log(artefacts(out)["csv"])} == {""}
    # ...and the per-object CSV is written all the same.
    rows = read_log(artefacts(out)["objects"])
    assert rows
    assert {row["depth_source"] for row in rows} <= {"centre-pixel", "none"}


def test_objects_without_centre_depth_is_labelled(replay_run):
    out, _ = replay_run("--objects")
    assert {row["extras"] for row in read_log(artefacts(out)["csv"])} == {"objects"}


def test_no_track_alone_does_not_change_the_label(replay_run):
    """The tracker is 0.05 ms a frame, below the noise in loop_ms."""
    out, _ = replay_run("--objects", "--no-track")
    assert {row["extras"] for row in read_log(artefacts(out)["csv"])} == {"objects"}


def unwritable_video_path(tmp_path):
    """A destination whose parent is a regular file -- refused on every OS, unlike /proc/..."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    return str(blocker / "clip.mp4")


def test_an_unwritable_video_path_is_refused_before_the_model_runs(replay_run, tmp_path):
    """A typo in --save-video must not cost a whole session."""
    with pytest.raises(SystemExit, match="cannot record to"):
        replay_run("--save-video", unwritable_video_path(tmp_path))


def test_a_refused_video_path_leaves_no_half_written_run(replay_run, tmp_path):
    """The destination is checked before the checkpoint loads and before any file is made."""
    with pytest.raises(SystemExit, match="cannot record to"):
        replay_run("--save-video", unwritable_video_path(tmp_path))
    assert not list(tmp_path.rglob("*.csv")), "an empty CSV was left behind"
