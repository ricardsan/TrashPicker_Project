"""Writing the annotated frames of a run to an MP4, without putting the encode in the timed loop.

The point of the file is a presentation: something to play that shows the model segmenting waste,
rather than a figure of how fast it did it.

Two things make this more than a `cv2.VideoWriter` call.

**The encode must not land inside run_frames' measured region.** Every latency figure in the report
comes out of that loop, and `loop_ms` is quoted as the rate the pipeline achieved. Encoding a
640x480 frame costs single-digit milliseconds, which against a 20 ms inference is not noise -- a
run with --save-video would report a visibly lower achieved FPS than the same model without it, for
reasons that have nothing to do with the model. So `write()` only hands the frame to a bounded
queue, and a worker thread does the encoding; the main thread pays a pointer copy. When the queue
is full the frame is dropped and counted rather than blocking the loop, because a stalled encoder
must not be able to reach back and change the measurement. The count is reported at close(), so a
drop is never silent.

No `.copy()` is needed on the way in: `segmentation_realsense.annotate` already returns a fresh
array per frame and nothing mutates it afterwards.

**OpenCV cannot necessarily encode H.264.** On this machine it cannot -- `cv2.VideoWriter` opens
for `mp4v` and fails for `avc1`/`H264`/`X264` with "Encoder not found", because the bundled FFmpeg
was built without libx264. `mp4v` is MPEG-4 Part 2, which PowerPoint and VLC play but Chrome,
Safari and Google Slides do not, so it is a poor choice for a file whose whole purpose is to be
shown. The system `ffmpeg` *does* have libx264, so the default is to pipe raw frames to it and let
it write H.264 + yuv420p + faststart, which plays everywhere. That also puts the encode in another
process entirely, which is the cheapest place for it. `cv2.VideoWriter` with `mp4v` remains as the
fallback for a machine with no ffmpeg on PATH.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

# Frames in flight between the loop and the encoder. 32 is ~28 MB at 640x480 and ~85 MB at 720p,
# and is far more slack than a steady encoder needs -- it exists to absorb a hiccup, not to buffer
# the run. Past it, dropping is the correct behaviour: see the module docstring.
QUEUE_SIZE = 32

DEFAULT_CRF = 20                # visually near-lossless for a screen recording; lower is bigger
CLOSE_TIMEOUT_S = 30.0          # generous: the worker only has to drain QUEUE_SIZE frames

# How long close() will wait to hand the worker its sentinel. Short, and separate from the join
# above, because this one is waiting on a *full queue* rather than on work in progress: if the
# encoder has stopped consuming -- stalled, killed, or dead of an exception -- the queue never
# drains and an untimed put() would hang the run at exit with the window already gone. Losing the
# last few frames of a video whose encoder has died is the right trade.
SENTINEL_TIMEOUT_S = 2.0

# How long the drain may go without encoding a single frame before close() calls it stuck.
#
# CLOSE_TIMEOUT_S alone cannot tell "slow" from "stuck": a worker blocked forever inside
# stdin.write() -- an ffmpeg that stopped reading -- looks exactly like one working through a
# queue, so close() sat out the full 30 s before giving up. Watching `written` advance
# distinguishes the two, and makes the wait proportional to the fault rather than to the
# timeout. A healthy drain of QUEUE_SIZE frames moves the counter every few milliseconds.
STALL_TIMEOUT_S = 5.0

BACKENDS = ("auto", "ffmpeg", "opencv")

_SENTINEL = object()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def choose_backend(backend: str = "auto") -> str:
    """Which backend a given --video-backend resolves to, given what is installed."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown video backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    if backend != "auto":
        return backend
    return "ffmpeg" if ffmpeg_available() else "opencv"


class _FFmpegSink:
    """Raw BGR frames down a pipe to the system ffmpeg, which writes H.264."""

    name = "ffmpeg"
    codec = "h264"

    def __init__(self, path: Path, fps: float, size, crf: int):
        width, height = size
        self.path = path
        self.command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            # yuv420p is the subsampling every player assumes; libx264 would otherwise keep the
            # 4:4:4 it was handed, which QuickTime and most browsers refuse. +faststart moves the
            # index to the front so the file starts playing before it has all downloaded.
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        # ascontiguousarray because tobytes() on a non-contiguous view silently reorders; a sliced
        # frame would come out scrambled rather than failing.
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def abort(self) -> None:
        """Kill ffmpeg, for when it has stopped reading its own stdin.

        Closing stdin flushes it, and a flush into a pipe nobody is draining **blocks** -- which
        is the one thing close() must not do, since it runs at the end of a run with the window
        already gone. Killing the reader first turns that flush into an EPIPE, which close()
        catches. The video is lost either way at this point; the choice is between losing it and
        hanging the process.
        """
        self.process.kill()

    def close(self) -> str | None:
        """Shut ffmpeg down. Returns its complaint if it had one, else None."""
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            # Already dead, or the worker thread got there first. Its stderr below is the
            # interesting part. ValueError is "flush of closed file", which is a race with the
            # worker rather than a fault.
            pass
        try:
            self.process.wait(timeout=CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            return "ffmpeg did not exit; it was killed"

        stderr = (self.process.stderr.read() or b"").decode("utf-8", "replace").strip()
        self.process.stderr.close()
        if self.process.returncode != 0:
            tail = "\n".join(stderr.splitlines()[-5:]) or "(it said nothing)"
            return f"ffmpeg exited {self.process.returncode}:\n{tail}"
        return None


class _OpenCVSink:
    """cv2.VideoWriter with mp4v, for a machine with no ffmpeg on PATH."""

    name = "opencv"
    codec = "mp4v"

    def __init__(self, path: Path, fps: float, size, crf: int):
        import cv2

        self.path = path
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
        if not self.writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter could not open {path} for mp4v at {size[0]}x{size[1]}. "
                f"Install ffmpeg and use --video-backend ffmpeg, which also gives H.264.")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> str | None:
        self.writer.release()
        return None


SINKS = {"ffmpeg": _FFmpegSink, "opencv": _OpenCVSink}


class VideoRecorder:
    """The annotated frames of a run, as an MP4. See the module docstring.

    Opened lazily on the first frame, because the frame itself is the only trustworthy source of
    the size: a writer given a size the frames do not match writes an empty file without
    complaining, and a replayed bag's frames are scaled by a factor the caller would have to
    reproduce. A later frame of a different shape is refused and counted, not silently mangled.
    """

    def __init__(self, path, fps: float, *, backend: str = "auto",
                 queue_size: int = QUEUE_SIZE, crf: int = DEFAULT_CRF, verbose: bool = True):
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.backend = choose_backend(backend)
        self.crf = crf
        self.verbose = verbose

        self.size: tuple[int, int, int] | None = None   # the first frame's shape, (h, w, c)
        self.written = 0
        self.dropped = 0        # frames the queue had no room for
        self.rejected = 0       # frames whose shape did not match the first one's
        self.error: str | None = None

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._sink = None
        self._worker: threading.Thread | None = None
        self._closed = False
        self._open_failed = False

    # --- before the run ------------------------------------------------------
    def prepare(self) -> None:
        """Fail now, on the command line, rather than on the first frame.

        The frame *size* genuinely cannot be known until a frame arrives -- which is why the sink
        is opened lazily -- but the destination folder and the encoder's existence can be. A typo
        in --save-video used to surface as a traceback out of the middle of the loop, throwing
        away a camera session that had already started, and `--video-backend ffmpeg` on a machine
        without ffmpeg did the same. Raising here means argparse-time behaviour for an
        argparse-time mistake.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "ffmpeg" and not ffmpeg_available():
            raise RuntimeError(
                "--video-backend ffmpeg needs ffmpeg on PATH, and there is none.\n"
                "Use --video-backend opencv, which writes mp4v (PowerPoint and VLC play it, "
                "browsers do not), or install ffmpeg.")

    # --- the loop's side ------------------------------------------------------
    def write(self, frame: np.ndarray) -> None:
        """Hand a frame over. Never blocks, never raises, never encodes on this thread.

        "Never raises" is the load-bearing half: this is called from inside run_frames, and a
        recorder that threw would take down a run whose CSV is the artefact that actually
        matters. Everything that can fail is turned into `self.error` and reported at close().
        """
        if self._closed or self._open_failed:
            return

        if self.size is None:
            if not self._open(frame):
                return
        elif frame.shape != self.size:
            # Counted rather than written: mixing sizes produces a file no player will accept,
            # resizing would quietly misrepresent what the model was shown, and a 4-channel frame
            # of the same width and height would be piped straight into a bgr24 encoder.
            self.rejected += 1
            return

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _open(self, frame: np.ndarray) -> bool:
        """Start the sink and its worker on the first frame. False if that failed.

        `size` is set only once the sink is actually up, so a failed open cannot leave the
        recorder half-built -- counting every later frame as a drop and reporting no error, which
        is how a destination typo came out as "no frames were recorded".
        """
        height, width = frame.shape[:2]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            sink = SINKS[self.backend](self.path, self.fps, (width, height), self.crf)
        except Exception as failure:        # noqa: BLE001 -- see the docstring: never raise
            self._open_failed = True
            self.error = f"could not start the {self.backend} encoder: {failure}"
            return False

        self._sink = sink
        self.size = frame.shape
        self._worker = threading.Thread(target=self._drain, name="video-recorder", daemon=True)
        self._worker.start()
        if self.verbose:
            print(f"recording {width}x{height} at {self.fps:.0f} fps to {self.path} "
                  f"({sink.name}/{sink.codec})")
        return True

    # --- the worker's side ----------------------------------------------------
    def _drain(self) -> None:
        """Encode until the sentinel. Any failure is recorded and reported at close().

        The except is deliberately broad. It used to catch (BrokenPipeError, OSError), which
        covers ffmpeg's pipe dying but not `cv2.error` -- what cv2.VideoWriter.write raises, and
        not an OSError. That combination killed this thread silently: close() found no error and
        nothing written, so a hard failure was reported as "no frames were recorded". A thread
        that cannot report its own death is worse than a broad except.

        Draining must also continue after a failure, or the queue fills and write() starts
        counting live frames as drops.
        """
        while True:
            frame = self._queue.get()
            if frame is _SENTINEL:
                return
            if self.error is not None:
                continue        # already broken; keep draining so write() never blocks
            try:
                self._sink.write(frame)
                self.written += 1
            except Exception as failure:    # noqa: BLE001 -- see the docstring
                # ffmpeg died, or the writer refused the frame. Losing the video is survivable;
                # taking the run and its CSV down with it is not.
                self.error = f"the encoder stopped accepting frames: {failure}"

    # --- shutting down --------------------------------------------------------
    def close(self) -> dict:
        """Drain, close the sink, and report what happened. Safe to call twice."""
        if self._closed:
            return self.stats()
        self._closed = True

        if self._worker is not None:
            # is_alive first: a worker that has already died of something _drain could not
            # report would never take the sentinel, and on a full queue an untimed put() would
            # then hang here forever -- at exit, with the window already closed, so the only way
            # out was Ctrl-C and the MP4 was left unfinalised.
            delivered = False
            if self._worker.is_alive():
                try:
                    self._queue.put(_SENTINEL, timeout=SENTINEL_TIMEOUT_S)
                    delivered = True
                except queue.Full:
                    self.error = self.error or (
                        f"the encoder stopped consuming frames: {self._queue.qsize()} are still "
                        f"queued after {SENTINEL_TIMEOUT_S:.0f} s")

            # Only wait for a worker that has been told to stop. If the sentinel never landed the
            # worker is provably not consuming, so joining would just spend CLOSE_TIMEOUT_S
            # proving it again -- 30 s of apparent hang at the end of a run. Fall through to
            # _sink.close() instead, which closes ffmpeg's stdin and kills it, and so unblocks
            # the worker from the other end. That write() may then fail, which _drain records.
            if delivered:
                self._drain_or_give_up()

            if self._worker.is_alive():
                # Given up on it. Tell the sink to abandon ship before close() tries to flush
                # into it -- see _FFmpegSink.abort: that flush is itself a blocking call when the
                # encoder has stopped reading, which is exactly the case we are in.
                abort = getattr(self._sink, "abort", None)
                if callable(abort):
                    abort()

        if self._sink is not None:
            complaint = self._sink.close()
            if complaint and self.error is None:
                self.error = complaint
            if self.error is None and not self._written_file_looks_real():
                self.error = (f"{self.path} is empty or missing after writing {self.written} "
                              f"frames")
        return self.stats()

    def _drain_or_give_up(self) -> None:
        """Wait for the worker to finish the queue, but not on one that has stopped moving.

        Returns as soon as the worker exits (the normal case: a handful of queued frames), after
        STALL_TIMEOUT_S without a single frame encoded, or after CLOSE_TIMEOUT_S overall. See
        STALL_TIMEOUT_S for why the progress check is what makes this bounded in practice.
        """
        deadline = time.monotonic() + CLOSE_TIMEOUT_S
        written, last_progress = self.written, time.monotonic()

        while self._worker.is_alive():
            now = time.monotonic()
            if now >= deadline:
                self.error = self.error or "the encoder thread did not finish in time"
                return
            if self.written != written:
                written, last_progress = self.written, now
            elif now - last_progress > STALL_TIMEOUT_S:
                self.error = self.error or (
                    f"the encoder stalled after {self.written} frames, with "
                    f"{self._queue.qsize()} still queued")
                return
            self._worker.join(timeout=0.05)

    def _written_file_looks_real(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def frame_size(self) -> tuple[int, int] | None:
        """(width, height) of what is being written, or None before the first frame.

        `self.size` is the numpy shape -- (height, width, channels) -- because that is what
        write() compares against. This is the same thing in the order a video is spoken about in.
        """
        return None if self.size is None else (self.size[1], self.size[0])

    def stats(self) -> dict:
        return {
            "path": self.path,
            "backend": self.backend,
            "codec": self._sink.codec if self._sink else None,
            "fps": self.fps,
            "size": self.frame_size(),
            "written": self.written,
            "dropped": self.dropped,
            "rejected": self.rejected,
            "error": self.error,
        }

    def summary_lines(self, wall_s: float | None = None) -> list[str]:
        """What to print at the end of a run.

        The duration is worth stating next to the wall clock: the file is written at a fixed rate,
        so a model too slow to keep up with the stream yields a clip that plays back faster than
        the run really went. Saying both numbers puts that on the record.
        """
        stats = self.stats()
        if stats["error"]:
            return [f"video: FAILED -- {stats['error']}"]
        if self.written == 0:
            return ["video: no frames were recorded"]

        clip_s = self.written / self.fps
        line = (f"video: {self.written} frames to {self.path} "
                f"({stats['codec']}, {clip_s:.1f} s at {self.fps:.0f} fps)")
        if wall_s and abs(clip_s - wall_s) > max(0.5, 0.1 * wall_s):
            line += f", against {wall_s:.1f} s of real time -- playback is not real speed"
        lines = [line]
        if self.dropped:
            lines.append(f"  {self.dropped} frames dropped: the encoder could not keep up")
        if self.rejected:
            width, height = self.frame_size()
            lines.append(f"  {self.rejected} frames skipped for not being {width}x{height}")
        return lines

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
