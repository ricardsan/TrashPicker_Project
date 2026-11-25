"""The snapshot frame shown in the running RobotStudio station: BoardOverlay's texture and a cloud.

On 'n' (camera) or at --frame (rosbag), frame_selection.preview_until hands the frame here. Two
files are built from it, and the RobotStudio add-in robot/rapid/TrashPickerStation is told to load
them:

    station_floor.png   the annotated frame laid on the floor (floor_overlay.floor_picture), turned
                        for RobotStudio, which the add-in puts on the part BoardOverlay
    station_cloud.bin   the scene in the base frame (scene_cloud.cloud_in_base), which the add-in
                        shows as the point cloud LiveCloud

Nothing polls and nothing is sent on any other frame. The add-in listens on STATION_HOST:STATION_PORT;
the request is one JSON line naming both files, and the answer one JSON line.

Why the two line up exactly
---------------------------
Both are built from one grab, with one T_base_cam and one set of intrinsics, under one projection:
the floor texel at (x, y, 0) takes the colour of the pixel K R^T (p - t) lands on, and the cloud
deprojects each pixel with the same K and moves it with the same T. So a cloud point on the floor
lands on the texel that took its pixel's colour. scene_cloud.require_pinhole refuses a frame with
distortion for both, rather than letting one path model it and the other not. The add-in sets
BoardOverlay's texture coordinates from its vertices' positions and the rectangle sent here,
u = (x - Xmin) / W and v = (Ymax - y) / H, which is how for_robotstudio turns the picture.

The background thread
---------------------
Building the files and the exchange run on a daemon thread, so a slow or hung RobotStudio never
freezes the image window or holds up the robot. One frame is on its way at a time; a frame chosen
meanwhile goes next, so the station ends on the latest. 'h' (stop()) closes the socket, which ends
the exchange; a frame waiting then goes, and the next 'n' tries again. A failure is a warning and
never the run, as for the floor picture.

    # the exactness check in RobotStudio: coloured squares, a cloud dot over each centre
    python -m robot.station_feed --test-pattern

    # what RobotStudio's 3D view shows now, as a PNG (optionally turned straight down first)
    python -m robot.station_feed --screenshot view.png [--look-down]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.segmentation_realsense import depth_map_metres  # noqa: E402
from robot import floor_overlay  # noqa: E402
from robot.floor_overlay import (FLOOR_X_MM, FLOOR_Y_MM, MM_PER_PX,  # noqa: E402
                                 floor_picture, floor_px, floor_size_px, for_robotstudio)
from robot.scene_cloud import DEFAULT_STRIDE, cloud_in_base, require_pinhole  # noqa: E402

# Mirrored in robot/rapid/TrashPickerStation/StationAddin.cs; test_station_feed diffs the two.
STATION_HOST = "127.0.0.1"
STATION_PORT = 5003
PART_NAME = "BoardOverlay"

TIMEOUT_S = 10.0
CLOUD_MAGIC = b"TPC1"
FLOOR_FILE = "station_floor.png"
CLOUD_FILE = "station_cloud.bin"
STOP_KEY = "h"
STOP_KEY_HELP = (f"'{STOP_KEY}' stops the RobotStudio display thread if it hangs; the run and the "
                 f"robot carry on, and the next 'n' tries again")


def station_dir() -> Path:
    # Read when called, not at import, so the tests' redirect of LIVE_DIR applies here too.
    return floor_overlay.LIVE_DIR / "station"


def write_cloud(path, points_mm: np.ndarray, colours: np.ndarray) -> Path:
    """b"TPC1", uint32 N, float32 N x 3 xyz in METRES in the base frame, uint8 N x 3 RGB."""
    path = Path(path)
    metres = (np.asarray(points_mm, dtype=np.float64) / 1000.0).astype("<f4")
    rgb = np.ascontiguousarray(colours, dtype=np.uint8).reshape(-1, 3)
    if len(metres) != len(rgb):
        raise ValueError(f"{len(metres)} points but {len(rgb)} colours")
    with open(path, "wb") as handle:
        handle.write(CLOUD_MAGIC)
        handle.write(np.uint32(len(metres)).astype("<u4").tobytes())
        handle.write(metres.tobytes())
        handle.write(rgb.tobytes())
    return path


def read_cloud(path) -> tuple[np.ndarray, np.ndarray]:
    """(N x 3 metres, N x 3 RGB) out of write_cloud's file."""
    data = Path(path).read_bytes()
    if data[:4] != CLOUD_MAGIC:
        raise ValueError(f"{path} is not a station cloud")
    count = int(np.frombuffer(data, "<u4", 1, 4)[0])
    points = np.frombuffer(data, "<f4", count * 3, 8).reshape(-1, 3)
    colours = np.frombuffer(data, np.uint8, count * 3, 8 + count * 12).reshape(-1, 3)
    return points, colours


def floor_uv(x_mm, y_mm) -> tuple[np.ndarray, np.ndarray]:
    """The texture coordinates the add-in gives floor point (x, y): u along X, v down from +Y max.

    The same formula is in StationAddin.cs. In the picture for_robotstudio writes, column c' holds
    X = Xmin + (c' + 0.5) mm and row r' holds Y = Ymax - (r' + 0.5) mm, so this lands every texel
    centre on its own floor point.
    """
    width, height = FLOOR_X_MM[1] - FLOOR_X_MM[0], FLOOR_Y_MM[1] - FLOOR_Y_MM[0]
    return ((np.asarray(x_mm, dtype=np.float64) - FLOOR_X_MM[0]) / width,
            (FLOOR_Y_MM[1] - np.asarray(y_mm, dtype=np.float64)) / height)


def build_request(frame: int, floor_path: Path, cloud_path: Path, points: int,
                  look_down: bool = False, cloud_visible: bool | None = None,
                  floor_visible: bool | None = None) -> dict:
    """The one JSON line the add-in takes. `look_down` turns the operator's view straight down on
    BoardOverlay, for the test pattern; a run leaves the view alone. `cloud_visible` and
    `floor_visible` show or hide LiveCloud and BoardOverlay; None leaves them as the add-in has
    them. A hidden one is still updated, so showing it again shows this frame."""
    request = {"cmd": "show", "frame": int(frame), "floor": str(Path(floor_path).resolve()),
               "cloud": str(Path(cloud_path).resolve()), "points": int(points),
               "floor_x_mm": list(FLOOR_X_MM), "floor_y_mm": list(FLOOR_Y_MM),
               "mm_per_px": MM_PER_PX, "part": PART_NAME}
    if look_down:
        request["look_down"] = True
    if cloud_visible is not None:
        request["cloud_visible"] = bool(cloud_visible)
    if floor_visible is not None:
        request["floor_visible"] = bool(floor_visible)
    return request


def visibility_request(cloud_visible: bool, floor_visible: bool) -> dict:
    """Show or hide LiveCloud and BoardOverlay without sending a frame: what a run sends at start."""
    return {"cmd": "visibility", "part": PART_NAME, "cloud_visible": bool(cloud_visible),
            "floor_visible": bool(floor_visible)}


def _shown(visible: bool) -> str:
    return "shown" if visible else "hidden"


class StationFeed:
    """Pushes a frame to the RobotStudio add-in on a background thread. See the module docstring.

    show() only copies the frame: building the files (a few tenths of a second) and the exchange
    with RobotStudio both happen on the thread, so 'n' never freezes the image window. One frame is
    sent at a time; a frame chosen while another is on its way goes next, replacing any frame that
    was already waiting, so the station always ends on the latest one.
    """

    def __init__(self, T_base_cam: np.ndarray, *, out_dir: Path | None = None,
                 host: str = STATION_HOST, port: int = STATION_PORT, timeout_s: float = TIMEOUT_S,
                 stride: int = DEFAULT_STRIDE, show_cloud: bool = True, show_floor: bool = True):
        self.T = np.asarray(T_base_cam, dtype=np.float64)
        # --hide-cloud / --hide-rgb: sent at start (set_visibility) and with every frame.
        self.show_cloud, self.show_floor = show_cloud, show_floor
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.host, self.port, self.timeout_s, self.stride = host, port, timeout_s, stride
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._pending = None                    # ("frame", ...) or ("request", dict), sent next
        self._current = None                    # the frame number the thread is on
        self._sock: socket.socket | None = None
        self._abort = False                     # 'h' was pressed during the current job
        self._idle = threading.Event()
        self._idle.set()
        self._job_done = threading.Event()
        self._job_done.set()
        self.last_reply: dict | None = None     # for the tests and the end of a run

    # --- building the files --------------------------------------------------------------------
    def write_frame(self, image, depth_frame, intrinsics, targets, annotated) -> tuple[Path, Path, int]:
        """(floor file, cloud file, number of points) for one frame. Raises if it cannot be built."""
        metres = depth_map_metres(depth_frame)
        if metres is None:
            raise ValueError("this depth frame gives no depth map")
        return self._write(image, annotated, metres, intrinsics, targets)

    def _write(self, image, annotated, metres, intrinsics, targets) -> tuple[Path, Path, int]:
        require_pinhole(intrinsics)
        picture = for_robotstudio(floor_picture(annotated, self.T, intrinsics, targets))
        points, colours = cloud_in_base(image, metres, intrinsics, self.T, self.stride)
        out_dir = self.out_dir or station_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        floor_path, cloud_path = out_dir / FLOOR_FILE, out_dir / CLOUD_FILE
        # Level 1: a fifth of level 3's time for a 1420 x 770 picture, and it is read back at once.
        if not cv2.imwrite(str(floor_path), picture, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"could not write {floor_path}")
        write_cloud(cloud_path, points, colours)
        return floor_path, cloud_path, len(points)

    # --- the caller's side ---------------------------------------------------------------------
    def show(self, index: int, image, depth_frame, intrinsics, targets, annotated) -> bool:
        """Hand this frame to the background thread. False if it cannot be shown at all.

        Returns at once: whether RobotStudio took it is printed by the thread.
        """
        try:
            require_pinhole(intrinsics)
            metres = depth_map_metres(depth_frame)
            if metres is None:
                raise ValueError("this depth frame gives no depth map")
            # Copies: a camera reuses its buffers, and the preview carries on drawing on its own.
            frame = (index, np.array(image, copy=True), np.array(annotated, copy=True),
                     np.array(metres, copy=True), intrinsics, list(targets))
        except Exception as failure:                # noqa: BLE001 - only a picture
            print(f"WARNING: frame {index} not shown in RobotStudio: {failure}")
            return False
        self._queue(("frame", frame), index)
        return True

    def set_visibility(self) -> bool:
        """Tell RobotStudio which of LiveCloud and BoardOverlay to show. A run sends this at start,
        with both stated, so a run without --hide-* shows again what an earlier one hid."""
        return self.send(visibility_request(self.show_cloud, self.show_floor))

    def send(self, request: dict) -> bool:
        """Send a request that needs no building, e.g. the test pattern or a screenshot."""
        self._queue(("request", request), request.get("frame", request.get("cmd")))
        return True

    def _queue(self, job, label) -> None:
        with self._lock:
            replaced = self._pending
            self._pending = job
            running = self._worker is not None
            current = self._current
            if not running:
                self._idle.clear()
                self._worker = threading.Thread(target=self._work, name="robotstudio-station",
                                                daemon=True)
                self._worker.start()
        if running:
            instead = "" if replaced is None else f", instead of frame {_label(replaced)}"
            print(f"RobotStudio is still busy with frame {current}; frame {label} goes next"
                  f"{instead}. Press '{STOP_KEY}' to stop the one it is on.")

    def busy(self) -> bool:
        return self._worker is not None

    def stop(self) -> None:
        """The 'h' key: end what the thread is doing by closing its socket. A frame waiting goes
        next, as the latest one chosen."""
        with self._lock:
            if self._worker is None:
                print("RobotStudio thread: nothing was running.")
                return
            frame, waiting = self._current, self._pending
            self._abort = True
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._sock.close()
        ended = self._job_done.wait(2.0)
        then = "" if waiting is None else f"; frame {_label(waiting)} goes next"
        if ended:
            print(f"RobotStudio thread stopped frame {frame}{then}.")
        else:
            print(f"RobotStudio thread asked to stop frame {frame}; it gives up within "
                  f"{self.timeout_s:.0f} s at the latest{then}.")

    def wait(self, keys=None, timeout_s: float | None = None, quiet: bool = False) -> None:
        """Let what the thread holds reach RobotStudio before the program exits, which would cut
        a daemon thread off. `keys` keeps the window painted and hears 'h' meanwhile."""
        if self._idle.is_set():
            return
        if not quiet:
            print(f"Waiting for RobotStudio to take frame {self._current}. "
                  f"Press '{STOP_KEY}' if it hangs.")
        end = time.monotonic() + (self.timeout_s * 2 if timeout_s is None else timeout_s)
        while not self._idle.wait(0.05) and time.monotonic() < end:
            if keys is not None:
                keys.poll(30)

    # --- the background thread -----------------------------------------------------------------
    def _work(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
                if job is None:
                    self._worker = self._current = None
                    self._idle.set()
                    return
                kind, payload = job
                self._current = _label(job)
                self._abort = False
                self._job_done.clear()
            try:
                if kind == "frame":
                    index, image, annotated, metres, intrinsics, targets = payload
                    try:
                        floor_path, cloud_path, count = self._write(image, annotated, metres,
                                                                    intrinsics, targets)
                    except Exception as failure:    # noqa: BLE001 - only a picture
                        print(f"WARNING: frame {index} not shown in RobotStudio: {failure}")
                        continue
                    payload = build_request(index, floor_path, cloud_path, count,
                                            cloud_visible=self.show_cloud,
                                            floor_visible=self.show_floor)
                if not self._abort:
                    self._exchange(payload)
            finally:
                self._job_done.set()

    def _exchange(self, request: dict) -> None:
        frame = request.get("frame")
        self.last_reply = None
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            with self._lock:
                if self._abort:
                    sock.close()
                    return
                self._sock = sock
            with sock:
                sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
                reply = b""
                while not reply.endswith(b"\n"):
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    reply += chunk
            answer = json.loads(reply.decode("utf-8")) if reply.strip() else {
                "ok": False, "error": "the add-in closed the connection without answering"}
        except (OSError, ValueError) as failure:
            answer = {"ok": False, "error": self._explain(failure)}
        finally:
            with self._lock:
                self._sock = None
        if self._abort:
            return                                  # 'h': stop() says what happened
        self.last_reply = answer
        if answer.get("ok") and request.get("cmd") == "show":
            hidden = [name for name, key in ((request.get("part"), "floor_visible"),
                                             ("LiveCloud", "cloud_visible"))
                      if request.get(key) is False]
            print(f"shown in RobotStudio: frame {frame} on {request.get('part')}, "
                  f"{answer.get('points', request.get('points'))} points in LiveCloud"
                  + (f" ({' and '.join(hidden)} hidden, but updated)" if hidden else ""))
        elif not answer.get("ok") and request.get("cmd") == "show":
            print(f"WARNING: frame {frame} not shown in RobotStudio: {answer.get('error')}\n"
                  f"  the files are {request.get('floor')} and {request.get('cloud')}")
        elif answer.get("ok") and request.get("cmd") == "visibility":
            print(f"RobotStudio: {request.get('part')} {_shown(request['floor_visible'])}, "
                  f"LiveCloud {_shown(request['cloud_visible'])}")
        elif not answer.get("ok"):
            print(f"WARNING: RobotStudio did not do {request.get('cmd')!r}: {answer.get('error')}")

    def _explain(self, failure: Exception) -> str:
        if isinstance(failure, ConnectionRefusedError):
            return (f"nothing listens on {self.host}:{self.port}. Is RobotStudio open with the "
                    f"TrashPickerStation add-in loaded? (robot/rapid/TrashPickerStation)")
        if isinstance(failure, socket.timeout):
            return f"no answer within {self.timeout_s:.0f} s"
        return str(failure)


def _label(job):
    """The frame number a queued job is for, or its command if it has none."""
    kind, payload = job
    return payload[0] if kind == "frame" else payload.get("frame", payload.get("cmd"))


# =============================================================================
# the test pattern: is the texture where the cloud is?
# =============================================================================
PATTERN_POINTS_MM = ((0, 0), (600, 0), (600, 300), (1200, -300), (300, -300), (1200, 300))
PATTERN_COLOURS_BGR = ((255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                       (255, 0, 255))
PATTERN_HALF_MM = 40
PATTERN_Z_MM = 2.0


def pattern_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(turned floor picture, cloud points in mm, RGB colours): a square at each of
    PATTERN_POINTS_MM, and a 5 x 5 grid of cloud points over each, 2 mm above it."""
    rows, columns = floor_size_px()
    picture = np.full((rows, columns, 3), 90, np.uint8)
    for x in range(int(np.ceil(FLOOR_X_MM[0] / 100) * 100), int(FLOOR_X_MM[1]) + 1, 100):
        cv2.line(picture, floor_px(x, FLOOR_Y_MM[1]), floor_px(x, FLOOR_Y_MM[0]), (150, 150, 150), 1)
    for y in range(int(np.ceil(FLOOR_Y_MM[0] / 100) * 100), int(FLOOR_Y_MM[1]) + 1, 100):
        cv2.line(picture, floor_px(FLOOR_X_MM[1], y), floor_px(FLOOR_X_MM[0], y), (150, 150, 150), 1)
    points, colours = [], []
    offsets = np.linspace(-PATTERN_HALF_MM * 0.8, PATTERN_HALF_MM * 0.8, 5)
    for (x, y), bgr in zip(PATTERN_POINTS_MM, PATTERN_COLOURS_BGR):
        cv2.rectangle(picture, floor_px(x + PATTERN_HALF_MM, y + PATTERN_HALF_MM),
                      floor_px(x - PATTERN_HALF_MM, y - PATTERN_HALF_MM), bgr, -1)
        for dx in offsets:
            for dy in offsets:
                points.append((x + dx, y + dy, PATTERN_Z_MM))
                colours.append(bgr[::-1])
    turned = for_robotstudio(picture)
    # Labels go on the turned picture, so they read upright in the station: there column c holds
    # X = Xmin + c and row r holds Y = Ymax - r (floor_uv).
    for (x, y), bgr in zip(PATTERN_POINTS_MM, PATTERN_COLOURS_BGR):
        cv2.putText(turned, f"X{x} Y{y}",
                    (int(x - FLOOR_X_MM[0] - PATTERN_HALF_MM), int(FLOOR_Y_MM[1] - y - PATTERN_HALF_MM - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2, cv2.LINE_AA)
    return turned, np.array(points), np.array(colours, dtype=np.uint8)


def send_test_pattern(feed: StationFeed | None = None) -> dict | None:
    feed = feed or StationFeed(np.eye(4))
    picture, points, colours = pattern_frame()
    out_dir = feed.out_dir or station_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    floor_path, cloud_path = out_dir / FLOOR_FILE, out_dir / CLOUD_FILE
    if not cv2.imwrite(str(floor_path), picture):
        raise OSError(f"could not write {floor_path}")
    write_cloud(cloud_path, points, colours)
    feed.send(build_request(-1, floor_path, cloud_path, len(points), look_down=True,
                            cloud_visible=True, floor_visible=True))
    feed.wait(quiet=True)
    return feed.last_reply


def request_screenshot(path, *, look_down: bool = False, port: int = STATION_PORT) -> dict | None:
    """Ask the add-in to save its 3D view as a PNG at `path`; changes nothing in the station."""
    feed = StationFeed(np.eye(4), port=port)
    request = {"cmd": "screenshot", "path": str(Path(path).resolve()),
               "floor_x_mm": list(FLOOR_X_MM), "floor_y_mm": list(FLOOR_Y_MM)}
    if look_down:
        request["look_down"] = True
    feed.send(request)
    feed.wait(quiet=True)
    return feed.last_reply


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Push a test pattern to the RobotStudio add-in.")
    parser.add_argument("--test-pattern", action="store_true",
                        help="squares at known floor X/Y on BoardOverlay, and a cloud dot grid "
                             f"{PATTERN_Z_MM:.0f} mm over each: the dots must sit on the squares")
    parser.add_argument("--screenshot", default=None, metavar="PNG",
                        help="save RobotStudio's 3D view to this file (after the test pattern, if "
                             "both are given)")
    parser.add_argument("--look-down", action="store_true",
                        help="with --screenshot: turn the view straight down on BoardOverlay first")
    parser.add_argument("--port", type=int, default=STATION_PORT)
    args = parser.parse_args(argv)
    if not (args.test_pattern or args.screenshot):
        parser.error("nothing to do: pass --test-pattern and/or --screenshot PNG")
    ok = True
    if args.test_pattern:
        reply = send_test_pattern(StationFeed(np.eye(4), port=args.port))
        print(f"add-in reply: {reply}")
        for (x, y), bgr in zip(PATTERN_POINTS_MM, PATTERN_COLOURS_BGR):
            print(f"  square at X {x:5d} Y {y:5d} mm, colour BGR {bgr}")
        ok = bool(reply and reply.get("ok"))
    if args.screenshot:
        reply = request_screenshot(args.screenshot, look_down=args.look_down, port=args.port)
        print(f"add-in reply: {reply}")
        ok = ok and bool(reply and reply.get("ok"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
