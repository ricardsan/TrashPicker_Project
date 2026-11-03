"""The MP4 writer, and the two claims it makes about itself.

The claims are that the file plays and that writing it does not reach back into the measured loop.
Both are checked against the real thing rather than by inspection: the videos written here are
opened again and decoded, and the ffmpeg backend's output is inspected with ffprobe where it is
available, because "the file exists and is non-empty" is exactly the assertion a 0-byte
VideoWriter failure would pass.
"""
from __future__ import annotations

import shutil
import threading
import subprocess
import time

import cv2
import numpy as np
import pytest

from camera.recording import (BACKENDS, VideoRecorder, choose_backend,
                              ffmpeg_available)

HEIGHT, WIDTH = 120, 160        # small, so a hundred frames encode in no time

BACKENDS_TO_TEST = [
    pytest.param("opencv", id="opencv"),
    pytest.param("ffmpeg", id="ffmpeg",
                 marks=pytest.mark.skipif(not ffmpeg_available(),
                                          reason="no ffmpeg on PATH")),
]


def frame(index: int = 0, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """A frame whose content varies with the index, so a decoder sees real motion."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (index * 7 % 255, 64, 192)
    image[index % height, :] = 255
    return image


def decoded_frame_count(path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened(), f"{path} would not open for reading"
        count = 0
        while capture.read()[0]:
            count += 1
        return count
    finally:
        capture.release()


# Enough slack that a tight producer loop never fills it. The real loop is paced by the camera or
# the bag -- 15 to 30 frames a second, with an inference between each -- so it cannot outrun the
# encoder the way a test that writes a hundred frames back to back can. The drop-rather-than-block
# policy is tested on its own, with SlowSink, where it is the behaviour under examination.
ROOMY = 256


def ffprobe(path, entries: str) -> dict[str, str]:
    """ffprobe's answer as {field: value}.

    Keyed rather than positional: ffprobe emits the fields in the stream's own order, not in the
    order they were asked for, so unpacking a csv row silently pairs the wrong name with the wrong
    value.
    """
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={entries}", "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in output.stdout.strip().splitlines() if "=" in line)


# --- choosing a backend -------------------------------------------------------
def test_auto_prefers_ffmpeg_when_it_is_installed():
    assert choose_backend("auto") == ("ffmpeg" if ffmpeg_available() else "opencv")


def test_an_explicit_backend_is_honoured():
    for backend in ("ffmpeg", "opencv"):
        assert choose_backend(backend) == backend


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown video backend"):
        choose_backend("x264")


def test_the_cli_choices_and_the_module_agree():
    """--video-backend's choices come from here, so a new backend cannot be half-added."""
    from camera.segmentation_realsense import BACKENDS as cli_backends

    assert tuple(cli_backends) == tuple(BACKENDS)


# --- the file it writes -------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
def test_the_video_can_be_decoded_again(backend, tmp_path):
    path = tmp_path / "run.mp4"
    with VideoRecorder(path, 15.0, backend=backend, queue_size=ROOMY,
                       verbose=False) as recorder:
        for index in range(40):
            recorder.write(frame(index))
    stats = recorder.stats()

    assert stats["error"] is None, stats
    assert stats["written"] == 40
    assert stats["dropped"] == 0
    assert path.stat().st_size > 0
    # The real test: a decoder gets the frames back out.
    assert decoded_frame_count(path) == 40


@pytest.mark.skipif(not ffmpeg_available() or not shutil.which("ffprobe"),
                    reason="needs ffmpeg and ffprobe")
def test_the_ffmpeg_backend_writes_h264_that_plays_everywhere():
    """The whole reason for the ffmpeg backend: OpenCV's mp4v does not play in a browser.

    yuv420p matters as much as h264 does -- libx264 would otherwise keep the 4:4:4 it was handed,
    which QuickTime and most browsers refuse.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.mp4"
        with VideoRecorder(path, 15.0, backend="ffmpeg", queue_size=ROOMY,
                           verbose=False) as recorder:
            for index in range(20):
                recorder.write(frame(index))
        assert recorder.stats()["error"] is None

        stream = ffprobe(path, "codec_name,pix_fmt,width,height,nb_frames")
        assert stream["codec_name"] == "h264"
        assert stream["pix_fmt"] == "yuv420p"
        assert (int(stream["width"]), int(stream["height"])) == (WIDTH, HEIGHT)
        assert int(stream["nb_frames"]) == 20


@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
def test_the_size_is_taken_from_the_first_frame(backend, tmp_path):
    """Opening lazily is what makes a mismatched size impossible rather than silent."""
    path = tmp_path / "run.mp4"
    with VideoRecorder(path, 10.0, backend=backend, queue_size=ROOMY, verbose=False) as recorder:
        assert recorder.size is None, "nothing should be opened before the first frame"
        for index in range(10):
            recorder.write(frame(index, height=64, width=96))
        assert recorder.frame_size() == (96, 64)

    assert recorder.stats()["error"] is None
    assert decoded_frame_count(path) == 10


@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
def test_a_frame_of_the_wrong_shape_is_counted_not_written(backend, tmp_path):
    """Mixing sizes makes a file no player accepts; resizing would misreport what was shown."""
    path = tmp_path / "run.mp4"
    with VideoRecorder(path, 10.0, backend=backend, queue_size=ROOMY, verbose=False) as recorder:
        for index in range(5):
            recorder.write(frame(index))
        recorder.write(frame(5, height=HEIGHT * 2, width=WIDTH * 2))
        recorder.write(frame(6))

    stats = recorder.stats()
    assert stats["written"] == 6
    assert stats["rejected"] == 1
    assert stats["error"] is None
    assert "1 frames skipped" in "\n".join(recorder.summary_lines())


def test_a_recorder_that_never_saw_a_frame_closes_cleanly(tmp_path):
    """ESC on the very first frame, or a camera that died before delivering one."""
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, verbose=False)
    stats = recorder.close()
    assert stats["error"] is None
    assert stats["written"] == 0
    assert not (tmp_path / "run.mp4").exists(), "nothing should be created for no frames"
    assert recorder.summary_lines() == ["video: no frames were recorded"]


def test_close_is_idempotent(tmp_path):
    with VideoRecorder(tmp_path / "run.mp4", 15.0, verbose=False) as recorder:
        recorder.write(frame(0))
    first = recorder.close()
    assert first == recorder.close()


def test_write_after_close_is_ignored(tmp_path):
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, verbose=False)
    recorder.write(frame(0))
    recorder.close()
    recorder.write(frame(1))        # must not raise, must not reopen anything
    assert recorder.stats()["written"] == 1


def test_the_parent_folder_is_created(tmp_path):
    """--no-log --save-video has no RunLogger to make the run's folder."""
    path = tmp_path / "runs" / "camera" / "deeper" / "run.mp4"
    with VideoRecorder(path, 15.0, verbose=False) as recorder:
        recorder.write(frame(0))
    assert recorder.stats()["error"] is None
    assert path.exists()


# --- staying out of the measured loop ----------------------------------------
class SlowSink:
    """A sink that takes 50 ms a frame, standing in for an encoder that cannot keep up."""

    name = "slow"
    codec = "slow"

    def __init__(self, path, fps, size, crf):
        self.path = path
        self.written = 0

    def write(self, frame):
        time.sleep(0.05)
        self.written += 1

    def close(self):
        self.path.write_bytes(b"x")     # so the non-empty check passes
        return None


@pytest.fixture
def slow_backend(monkeypatch):
    monkeypatch.setitem(
        __import__("camera.recording", fromlist=["SINKS"]).SINKS, "slow", SlowSink)
    monkeypatch.setattr("camera.recording.BACKENDS", BACKENDS + ("slow",))
    monkeypatch.setattr("camera.recording.choose_backend", lambda backend="auto": "slow")


def test_write_does_not_block_on_a_slow_encoder(slow_backend, tmp_path):
    """The point of the thread. 50 ms a frame must not become 50 ms of loop time a frame."""
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend="slow", queue_size=4,
                             verbose=False)
    try:
        start = time.perf_counter()
        for index in range(40):
            recorder.write(frame(index))
        elapsed = time.perf_counter() - start
    finally:
        recorder.close()

    # 40 frames at 50 ms would be 2 s if write() blocked. The first few go into the queue
    # instantly and the rest are dropped, so the loop's side should cost milliseconds.
    assert elapsed < 0.5, f"write() blocked the caller for {elapsed:.2f} s"


def test_frames_are_dropped_and_counted_when_the_queue_fills(slow_backend, tmp_path):
    """Dropping is the correct failure: a stalled encoder must not change the measurement."""
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend="slow", queue_size=2,
                             verbose=False)
    try:
        for index in range(30):
            recorder.write(frame(index))
    finally:
        stats = recorder.close()

    assert stats["dropped"] > 0, "a 50 ms sink and a 2-frame queue must drop something"
    assert stats["written"] + stats["dropped"] == 30, stats
    assert "frames dropped" in "\n".join(recorder.summary_lines())


# --- failing legibly ----------------------------------------------------------
class ExplodingSink:
    """A sink whose pipe breaks partway, the way a killed ffmpeg's does."""

    name = "exploding"
    codec = "exploding"

    def __init__(self, path, fps, size, crf):
        self.path = path
        self.calls = 0

    def write(self, frame):
        self.calls += 1
        if self.calls > 3:
            raise BrokenPipeError("the encoder went away")

    def close(self):
        return None


def test_an_encoder_that_dies_does_not_take_the_run_down(monkeypatch, tmp_path):
    """Losing the video is survivable. Losing the run and its CSV with it is not."""
    monkeypatch.setitem(
        __import__("camera.recording", fromlist=["SINKS"]).SINKS, "exploding", ExplodingSink)
    monkeypatch.setattr("camera.recording.choose_backend", lambda backend="auto": "exploding")

    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend="exploding", verbose=False)
    for index in range(20):
        recorder.write(frame(index))        # none of these may raise
    stats = recorder.close()

    assert stats["error"] is not None
    assert "stopped accepting frames" in stats["error"]
    assert recorder.summary_lines()[0].startswith("video: FAILED")


def test_an_empty_file_is_reported_rather_than_passed_off_as_a_video(monkeypatch, tmp_path):
    """A cv2.VideoWriter given a bad configuration writes 0 bytes without complaining."""
    class SilentlyEmptySink:
        name = codec = "empty"

        def __init__(self, path, fps, size, crf):
            self.path = path

        def write(self, frame):
            pass

        def close(self):
            return None

    monkeypatch.setitem(
        __import__("camera.recording", fromlist=["SINKS"]).SINKS, "empty", SilentlyEmptySink)
    monkeypatch.setattr("camera.recording.choose_backend", lambda backend="auto": "empty")

    with VideoRecorder(tmp_path / "run.mp4", 15.0, backend="empty", verbose=False) as recorder:
        recorder.write(frame(0))
    assert "empty or missing" in (recorder.stats()["error"] or "")


def test_an_unwritable_destination_is_refused_before_the_run_starts(tmp_path):
    """prepare() exists so a typo in --save-video does not cost a camera session.

    It used to surface as a traceback out of write(), i.e. out of the middle of run_frames, after
    the model was loaded and the camera was streaming.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    recorder = VideoRecorder(blocker / "run.mp4", 15.0, verbose=False)
    with pytest.raises(OSError):
        recorder.prepare()


def test_prepare_refuses_a_forced_ffmpeg_backend_without_ffmpeg(monkeypatch, tmp_path):
    """--video-backend ffmpeg is explicit, so choose_backend does not probe PATH for it."""
    monkeypatch.setattr("camera.recording.ffmpeg_available", lambda: False)
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend="ffmpeg", verbose=False)
    with pytest.raises(RuntimeError, match="needs ffmpeg on PATH"):
        recorder.prepare()


def test_prepare_creates_the_folder_without_creating_the_file(tmp_path):
    recorder = VideoRecorder(tmp_path / "deep" / "run.mp4", 15.0, verbose=False)
    recorder.prepare()
    assert (tmp_path / "deep").is_dir()
    assert not (tmp_path / "deep" / "run.mp4").exists()
    recorder.close()


def test_a_failure_to_open_is_reported_rather_than_raised(monkeypatch, tmp_path):
    """write() is called from inside the loop, so it must never raise -- even on a bad open.

    And a half-built recorder must not then count every live frame as a drop while reporting no
    error at all, which read as "no frames were recorded".
    """
    class WontOpen:
        name = codec = "wont-open"

        def __init__(self, path, fps, size, crf):
            raise RuntimeError("the encoder refused the configuration")

    monkeypatch.setitem(
        __import__("camera.recording", fromlist=["SINKS"]).SINKS, "wont-open", WontOpen)
    monkeypatch.setattr("camera.recording.choose_backend", lambda backend="auto": "wont-open")

    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, verbose=False)
    for index in range(20):
        recorder.write(frame(index))        # none of these may raise
    stats = recorder.close()

    assert "could not start the" in (stats["error"] or "")
    assert "refused the configuration" in stats["error"]
    assert stats["written"] == 0
    assert stats["dropped"] == 0, "a failed open must not be reported as dropped frames"
    assert recorder.size is None, "nothing should have been opened"


# --- what it says at the end --------------------------------------------------
def test_the_summary_warns_when_playback_is_not_real_speed(tmp_path):
    """A model too slow for the stream yields a clip that plays faster than the run went."""
    with VideoRecorder(tmp_path / "run.mp4", 15.0, backend="opencv", verbose=False) as recorder:
        for index in range(30):
            recorder.write(frame(index))

    # 30 frames at 15 fps is a 2 s clip; say it took 10 s of real time.
    text = "\n".join(recorder.summary_lines(wall_s=10.0))
    assert "not real speed" in text
    # ...and stays quiet when the two agree.
    assert "not real speed" not in "\n".join(recorder.summary_lines(wall_s=2.0))


def test_the_summary_names_the_codec_and_the_file(tmp_path):
    path = tmp_path / "run.mp4"
    with VideoRecorder(path, 15.0, backend="opencv", verbose=False) as recorder:
        recorder.write(frame(0))
    line = recorder.summary_lines()[0]
    assert "run.mp4" in line and "mp4v" in line


# =============================================================================
# SHUTTING DOWN WHEN THE ENCODER HAS STOPPED CONSUMING
# =============================================================================
# close() runs in main()'s finally, with the window already gone. A hang there has no way out
# but Ctrl-C, and it leaves the MP4 unfinalised. Both of these hung before being fixed.


class StuckSink:
    """A sink that never returns from write(), like an ffmpeg stopped with SIGSTOP."""

    name = codec = "stuck"

    def __init__(self, path, fps, size, crf):
        self.path = path
        self.release = threading.Event()

    def write(self, frame):
        self.release.wait(timeout=60)

    def close(self):
        self.release.set()
        self.path.write_bytes(b"x")
        return None


class DyingSink:
    """A sink that raises something _drain used to not catch, killing the worker thread."""

    name = codec = "dying"

    def __init__(self, path, fps, size, crf):
        self.path = path

    def write(self, frame):
        raise ValueError("not an OSError, and not a BrokenPipeError either")

    def close(self):
        self.path.write_bytes(b"x")
        return None


@pytest.fixture
def sink(monkeypatch):
    """Install a sink class under its own backend name and select it."""

    def install(name, cls):
        monkeypatch.setitem(
            __import__("camera.recording", fromlist=["SINKS"]).SINKS, name, cls)
        monkeypatch.setattr("camera.recording.choose_backend", lambda backend="auto": name)
        return name

    return install


def test_close_does_not_hang_on_a_stalled_encoder(sink, tmp_path, monkeypatch):
    """The sentinel used to go on with an untimed put(), which blocks on a full queue."""
    monkeypatch.setattr("camera.recording.SENTINEL_TIMEOUT_S", 0.3)
    name = sink("stuck", StuckSink)

    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend=name, queue_size=2,
                             verbose=False)
    for index in range(50):
        recorder.write(frame(index))        # fills the queue, then drops

    start = time.perf_counter()
    stats = recorder.close()
    elapsed = time.perf_counter() - start

    assert elapsed < 10, f"close() took {elapsed:.1f} s -- it hung"
    assert stats["error"] is not None
    assert "stopped consuming" in stats["error"] or "did not finish" in stats["error"]
    assert stats["dropped"] > 0


def test_close_does_not_hang_when_the_worker_has_died(sink, tmp_path, monkeypatch):
    """A worker killed by an uncaught exception never takes the sentinel."""
    monkeypatch.setattr("camera.recording.SENTINEL_TIMEOUT_S", 0.3)
    name = sink("dying", DyingSink)

    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend=name, queue_size=2,
                             verbose=False)
    for index in range(50):
        recorder.write(frame(index))

    start = time.perf_counter()
    stats = recorder.close()
    assert time.perf_counter() - start < 10, "close() hung on a dead worker"

    # And the death itself is reported, rather than coming out as "no frames were recorded".
    assert stats["error"] is not None, stats
    assert "stopped accepting frames" in stats["error"]
    assert "not an OSError" in stats["error"]


def test_a_sink_raising_cv2_error_is_reported(sink, tmp_path):
    """cv2.VideoWriter.write raises cv2.error, which is not an OSError.

    The except used to be (BrokenPipeError, OSError), so this killed the worker silently and
    close() reported nothing wrong at all.
    """
    class Cv2Failing:
        name = codec = "cv2-failing"

        def __init__(self, path, fps, size, crf):
            self.path = path

        def write(self, frame):
            raise cv2.error("the writer refused the frame")

        def close(self):
            self.path.write_bytes(b"x")
            return None

    name = sink("cv2-failing", Cv2Failing)
    with VideoRecorder(tmp_path / "run.mp4", 15.0, backend=name, queue_size=ROOMY,
                       verbose=False) as recorder:
        for index in range(5):
            recorder.write(frame(index))

    assert "stopped accepting frames" in (recorder.stats()["error"] or "")
    assert recorder.summary_lines()[0].startswith("video: FAILED")


def test_a_frame_with_the_wrong_channel_count_is_refused(sink, tmp_path):
    """Same height and width, four channels: a bgr24 encoder would take it and mangle it."""
    with VideoRecorder(tmp_path / "run.mp4", 15.0, backend="opencv", queue_size=ROOMY,
                       verbose=False) as recorder:
        recorder.write(frame(0))
        recorder.write(np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8))

    stats = recorder.stats()
    assert stats["written"] == 1
    assert stats["rejected"] == 1


@pytest.mark.skipif(not ffmpeg_available(), reason="no ffmpeg on PATH")
def test_close_does_not_hang_on_a_real_stalled_ffmpeg(tmp_path, monkeypatch):
    """The scenario a fake sink does not model, and the reason `abort` exists.

    SIGSTOP a real ffmpeg and two separate things block: the worker inside `stdin.write`, and
    then `stdin.close()`'s own flush into a pipe nobody is draining. The first made a fixed
    `join` timeout useless (it cannot tell slow from stuck); the second meant close() blocked
    even after giving up on the worker. A stalled encoder must still let the run exit.
    """
    import os
    import signal

    monkeypatch.setattr("camera.recording.STALL_TIMEOUT_S", 1.0)
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend="ffmpeg", queue_size=4,
                             verbose=False)
    pid = None
    try:
        recorder.write(frame(0))
        deadline = time.perf_counter() + 5
        while recorder._sink is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        pid = recorder._sink.process.pid
        os.kill(pid, signal.SIGSTOP)

        # The loop's side must stay free even while the encoder is frozen.
        start = time.perf_counter()
        for index in range(200):
            recorder.write(frame(index))
        assert time.perf_counter() - start < 0.5

        start = time.perf_counter()
        stats = recorder.close()
        elapsed = time.perf_counter() - start
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGCONT)
            except (ProcessLookupError, PermissionError):
                pass

    assert elapsed < 15, f"close() took {elapsed:.1f} s on a stalled ffmpeg"
    assert stats["error"] is not None
    assert stats["dropped"] > 0
    assert recorder.summary_lines()[0].startswith("video: FAILED")


@pytest.mark.skipif(not ffmpeg_available(), reason="no ffmpeg on PATH")
def test_a_healthy_close_is_not_slowed_by_the_stall_watchdog(tmp_path):
    """The watchdog must cost a working run nothing: it waits on progress, not on a clock."""
    recorder = VideoRecorder(tmp_path / "run.mp4", 15.0, backend="ffmpeg", queue_size=ROOMY,
                             verbose=False)
    for index in range(60):
        recorder.write(frame(index))

    start = time.perf_counter()
    stats = recorder.close()
    elapsed = time.perf_counter() - start

    assert stats["error"] is None, stats
    assert stats["written"] == 60
    assert elapsed < 5, f"a healthy close took {elapsed:.1f} s"
