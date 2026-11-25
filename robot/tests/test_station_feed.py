"""The snapshot pushed into RobotStudio: alignment, files, the socket, the 'h' key, the preview hook."""
from __future__ import annotations

import json
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from robot import station_feed as sf
from robot.floor_overlay import FLOOR_X_MM, FLOOR_Y_MM, floor_size_px
from robot.frame_selection import preview_until, started
from robot.operator_keys import OperatorKeys
from robot.scene_cloud import MAX_RANGE_M
from synthetic import (FakeDepth, FakeDetector, FakeSource, Intrinsics, ScriptedKeys, detection,
                       disc_mask)

REAL_EXCHANGE = sf.StationFeed._exchange   # conftest swaps it for a recorder; some tests want it
ADDIN_SOURCE = (Path(__file__).resolve().parents[1] / "rapid" / "TrashPickerStation"
                / "StationAddin.cs")
INTR = Intrinsics()
T_DOWN = np.array([[0.0, -1.0, 0.0, 0.6],
                   [-1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0]])


def looking_at(eye, target) -> np.ndarray:
    """T_base_cam of a camera at `eye` (m) whose optical axis passes through `target`, image
    right roughly along base -y: tilted and offset, as the real camera is."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


T_TILTED = looking_at((-0.15, 0.08, 1.1), (0.65, -0.02, 0.0))


def floor_depth(T: np.ndarray) -> np.ndarray:
    """The depth map, in metres along the optical axis, of a camera at T seeing only the floor."""
    u, v = np.meshgrid(np.arange(INTR.width, dtype=float), np.arange(INTR.height, dtype=float))
    rays = np.stack([(u - INTR.ppx) / INTR.fx, (v - INTR.ppy) / INTR.fy, np.ones_like(u)], -1)
    down = rays @ T[:3, :3].T
    scale = -T[2, 3] / down[..., 2]
    return np.where(scale > 0, scale, 0.0)


def blocky_image(seed: int = 0, block: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 256, (INTR.height // block + 1, INTR.width // block + 1, 3), np.uint8)
    return np.ascontiguousarray(
        np.repeat(np.repeat(cells, block, 0), block, 1)[:INTR.height, :INTR.width])


def sample_texture(picture: np.ndarray, x_mm, y_mm) -> np.ndarray:
    """What a renderer draws at floor points (x, y): the texture, bilinear, at the add-in's UVs."""
    u, v = sf.floor_uv(x_mm, y_mm)
    height, width = picture.shape[:2]
    count = len(u)
    # remap wants a 2-D map under 32767 a side, so the samples go in rows of 256.
    padded = -(-count // 256) * 256
    columns = np.zeros(padded, np.float32)
    rows = np.zeros(padded, np.float32)
    columns[:count], rows[:count] = u * width - 0.5, v * height - 0.5
    drawn = cv2.remap(picture, columns.reshape(-1, 256), rows.reshape(-1, 256), cv2.INTER_LINEAR)
    return drawn.reshape(-1, 3)[:count]


def uniform_around(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Pixels whose whole (2r+1)^2 neighbourhood has their colour: no interpolation can blur them."""
    grey = image.astype(np.int32) @ np.array([1, 256, 65536])
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    low = cv2.erode(grey.astype(np.float32), kernel)
    high = cv2.dilate(grey.astype(np.float32), kernel)
    calm = low == high
    # Floor just past the image's edge is UNSEEN grey, and texels there blend into it.
    calm[:radius + 1], calm[-radius - 1:], calm[:, :radius + 1], calm[:, -radius - 1:] = (False,) * 4
    return calm


# =============================================================================
# the texture and the cloud line up
# =============================================================================
@pytest.mark.parametrize("T", [T_DOWN, T_TILTED], ids=["straight-down", "tilted"])
def test_every_floor_point_of_the_cloud_sits_on_its_own_colour_in_the_texture(tmp_path, T):
    image = blocky_image()
    metres = floor_depth(T)
    feed = sf.StationFeed(T, out_dir=tmp_path, stride=1)
    floor_path, cloud_path, count = feed.write_frame(image, FakeDepth(metres), INTR, [], image)
    picture = cv2.imread(str(floor_path))
    points_m, rgb = sf.read_cloud(cloud_path)
    assert len(points_m) == count > 0.5 * INTR.width * INTR.height

    points_mm = points_m.astype(np.float64) * 1000.0
    assert np.abs(points_mm[:, 2]).max() < 0.5                  # on the floor, as built
    inside = ((points_mm[:, 0] > FLOOR_X_MM[0] + 2) & (points_mm[:, 0] < FLOOR_X_MM[1] - 2)
              & (points_mm[:, 1] > FLOOR_Y_MM[0] + 2) & (points_mm[:, 1] < FLOOR_Y_MM[1] - 2))
    # Which pixel each point came from (cloud_in_camera's order and test), to keep only those
    # well inside a block, where no interpolation can change the colour.
    rows, columns = np.nonzero((metres > 0) & (metres < MAX_RANGE_M))
    calm = uniform_around(image)[rows, columns]
    keep = inside & calm
    assert keep.sum() > 50_000
    drawn = sample_texture(picture, points_mm[keep, 0], points_mm[keep, 1])
    expected = rgb[keep][:, ::-1]                               # the picture is BGR
    assert np.abs(drawn.astype(int) - expected.astype(int)).max() <= 2


@pytest.mark.parametrize("T", [T_DOWN, T_TILTED], ids=["straight-down", "tilted"])
def test_a_spot_lands_at_the_same_x_y_in_the_texture_and_in_the_cloud(tmp_path, T):
    """The position check: a white spot's centre in the cloud and in the texture agree."""
    image = np.zeros((INTR.height, INTR.width, 3), np.uint8)
    cv2.circle(image, (400, 300), 12, (255, 255, 255), -1)
    feed = sf.StationFeed(T, out_dir=tmp_path, stride=1)
    floor_path, cloud_path, _ = feed.write_frame(image, FakeDepth(floor_depth(T)), INTR, [], image)
    points_m, rgb = sf.read_cloud(cloud_path)
    white = rgb[:, 0] == 255
    cloud_centre = points_m[white, :2].astype(float).mean(0) * 1000.0

    picture = cv2.imread(str(floor_path))
    rows, columns = np.nonzero(picture[..., 0] > 128)
    height, width = picture.shape[:2]
    # Back from texel centres to floor millimetres, inverting floor_uv.
    texture_x = FLOOR_X_MM[0] + (columns + 0.5) / width * (FLOOR_X_MM[1] - FLOOR_X_MM[0])
    texture_y = FLOOR_Y_MM[1] - (rows + 0.5) / height * (FLOOR_Y_MM[1] - FLOOR_Y_MM[0])
    texture_centre = np.array([texture_x.mean(), texture_y.mean()])
    assert np.linalg.norm(cloud_centre - texture_centre) < 0.5


@pytest.mark.parametrize("T", [T_DOWN, T_TILTED], ids=["straight-down", "tilted"])
def test_each_texel_holds_the_pixel_its_floor_point_projects_to(tmp_path, T):
    """Exact, not statistical: an image whose pixels hold their own (u, v), laid on the floor, gives
    each texel the (u, v) its centre -- found through floor_uv's inverse, as the add-in maps it --
    projects to. Bilinear sampling of a linear image is exact to remap's 1/32 px, so a half-texel
    (0.5 mm) shift, about 0.3 px, shows."""
    u, v = np.meshgrid(np.arange(INTR.width, dtype=np.float32),
                       np.arange(INTR.height, dtype=np.float32))
    image = np.dstack([u, v, np.zeros_like(u)])
    turned = sf.for_robotstudio(sf.floor_overlay.lay_on_floor(image, T, INTR))
    rows, columns = np.mgrid[0:turned.shape[0]:7, 0:turned.shape[1]:7]
    x_mm = FLOOR_X_MM[0] + (columns + 0.5) * (FLOOR_X_MM[1] - FLOOR_X_MM[0]) / turned.shape[1]
    y_mm = FLOOR_Y_MM[1] - (rows + 0.5) * (FLOOR_Y_MM[1] - FLOOR_Y_MM[0]) / turned.shape[0]
    base = np.stack([x_mm, y_mm, np.zeros_like(x_mm)], -1).reshape(-1, 3) / 1000.0
    cam = (base - T[:3, 3]) @ T[:3, :3]
    expected_u = INTR.fx * cam[:, 0] / cam[:, 2] + INTR.ppx
    expected_v = INTR.fy * cam[:, 1] / cam[:, 2] + INTR.ppy
    got = turned[rows, columns].reshape(-1, 3)
    seen = ((expected_u > 1) & (expected_u < INTR.width - 2)
            & (expected_v > 1) & (expected_v < INTR.height - 2))
    assert seen.sum() > 1000
    assert np.abs(got[seen, 0] - expected_u[seen]).max() < 0.05     # remap works in 1/32 px
    assert np.abs(got[seen, 1] - expected_v[seen]).max() < 0.05


def test_floor_uv_puts_each_texel_centre_on_its_floor_point():
    rows, columns = floor_size_px()
    turned_width, turned_height = rows, columns                 # for_robotstudio swaps them
    u, v = sf.floor_uv(FLOOR_X_MM[0] + 0.5, FLOOR_Y_MM[1] - 0.5)
    assert u * turned_width == pytest.approx(0.5) and v * turned_height == pytest.approx(0.5)
    u, v = sf.floor_uv(FLOOR_X_MM[1], FLOOR_Y_MM[0])
    assert (u, v) == pytest.approx((1.0, 1.0))


def test_distortion_is_refused_for_both_and_nothing_is_sent(tmp_path, capsys, station_requests):
    @dataclass
    class Distorted(Intrinsics):
        coeffs: list = field(default_factory=lambda: [0.1, 0.0, 0.0, 0.0, 0.0])

    image = blocky_image()
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path)
    assert feed.show(3, image, FakeDepth(floor_depth(T_DOWN)), Distorted(), [], image) is False
    assert "distortion" in capsys.readouterr().out and station_requests == []


# =============================================================================
# the files and the request
# =============================================================================
def test_the_cloud_file_round_trips(tmp_path):
    points = np.array([[600.0, 0.0, 2.0], [-40.0, 380.0, 0.0]])
    colours = np.array([[255, 0, 0], [1, 2, 3]], np.uint8)
    read_points, read_colours = sf.read_cloud(sf.write_cloud(tmp_path / "c.bin", points, colours))
    assert read_points == pytest.approx(points / 1000.0)
    assert (read_colours == colours).all()


def test_show_writes_both_files_and_names_them_in_one_request(tmp_path, station_requests):
    image = blocky_image()
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path)
    assert feed.show(7, image, FakeDepth(floor_depth(T_DOWN)), INTR, [], image) is True
    feed.wait()
    (request,) = station_requests
    assert request["cmd"] == "show" and request["frame"] == 7 and request["part"] == "BoardOverlay"
    assert Path(request["floor"]).is_absolute() and Path(request["floor"]).exists()
    assert len(sf.read_cloud(request["cloud"])[0]) == request["points"]
    assert request["floor_x_mm"] == list(FLOOR_X_MM) and request["floor_y_mm"] == list(FLOOR_Y_MM)
    assert cv2.imread(request["floor"]).shape[:2] == floor_size_px()[::-1]


def test_a_frame_carries_the_feed_s_visibility(tmp_path, station_requests):
    image = blocky_image()
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path, show_cloud=False, show_floor=True)
    feed.show(2, image, FakeDepth(floor_depth(T_DOWN)), INTR, [], image)
    feed.wait()
    (request,) = station_requests
    assert request["cloud_visible"] is False and request["floor_visible"] is True


def test_set_visibility_states_both(station_requests):
    feed = sf.StationFeed(T_DOWN, show_cloud=True, show_floor=False)
    feed.set_visibility()
    feed.wait()
    assert station_requests == [{"cmd": "visibility", "part": "BoardOverlay",
                                 "cloud_visible": True, "floor_visible": False}]


def test_a_visibility_reply_is_reported(addin, capsys):
    server = addin("ok")
    feed = sf.StationFeed(T_DOWN, port=server.port, show_cloud=False)
    feed.set_visibility()
    feed.wait()
    assert "RobotStudio: BoardOverlay shown, LiveCloud hidden" in capsys.readouterr().out


def test_a_frame_with_no_depth_map_is_a_warning(tmp_path, capsys, station_requests):
    image = blocky_image()
    assert sf.StationFeed(T_DOWN, out_dir=tmp_path).show(0, image, object(), INTR, [], image) is False
    assert "WARNING" in capsys.readouterr().out and station_requests == []


# =============================================================================
# the socket, against a stand-in for the add-in
# =============================================================================
class FakeAddin:
    """Listens like StationAddin.cs. `mode`: 'ok', 'error', 'silent' (never answers), 'close'."""

    def __init__(self, mode="ok"):
        self.mode, self.requests = mode, []
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                request = json.loads(data)
                self.requests.append(request)
                if self.mode == "silent":
                    self.release.wait(10)
                elif self.mode == "ok":
                    conn.sendall(json.dumps({"ok": True, "points": request.get("points")}).encode()
                                 + b"\n")
                elif self.mode == "error":
                    conn.sendall(b'{"ok": false, "error": "no part called BoardOverlay"}\n')

    def close(self):
        self.release.set()
        self.server.close()


@pytest.fixture
def addin(monkeypatch):
    monkeypatch.setattr(sf.StationFeed, "_exchange", REAL_EXCHANGE)
    servers = []

    def start(mode="ok"):
        servers.append(FakeAddin(mode))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def show_once(feed, index=5):
    image = blocky_image()
    return feed.show(index, image, FakeDepth(floor_depth(T_DOWN)), INTR, [], image)


def test_a_frame_the_addin_takes_is_reported(tmp_path, addin, capsys):
    server = addin("ok")
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path, port=server.port)
    assert show_once(feed)
    feed.wait()
    assert feed.last_reply["ok"] and server.requests[0]["frame"] == 5
    assert "shown in RobotStudio: frame 5" in capsys.readouterr().out


def test_an_error_from_the_addin_is_a_warning_with_the_files(tmp_path, addin, capsys):
    server = addin("error")
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path, port=server.port)
    show_once(feed)
    feed.wait()
    out = capsys.readouterr().out
    assert "WARNING" in out and "no part called BoardOverlay" in out and "station_floor.png" in out


def test_no_robotstudio_is_a_warning_and_not_an_exception(tmp_path, addin, capsys):
    closed = socket.create_server(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path, port=port)
    show_once(feed)
    feed.wait()
    assert "nothing listens" in capsys.readouterr().out and not feed.last_reply["ok"]


def until(condition, timeout_s=10.0):
    """Wait for a background thread's effect, failing the test rather than hanging the suite."""
    end = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_a_hung_robotstudio_does_not_hold_up_the_caller_and_h_ends_the_thread(tmp_path, addin,
                                                                             capsys):
    hung = addin("silent")
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path, port=hung.port, timeout_s=30.0)
    start = time.monotonic()
    assert show_once(feed)
    assert time.monotonic() - start < 1.0 and feed.busy()     # the building is on the thread too
    until(lambda: hung.requests)                                # the request did go out

    # A frame chosen meanwhile is not dropped: it waits, and the latest one wins.
    assert show_once(feed, index=6) and show_once(feed, index=7)
    out = capsys.readouterr().out
    assert "busy with frame 5; frame 6 goes next" in out and "instead of frame 6" in out

    healthy = addin("ok")
    feed.port = healthy.port
    start = time.monotonic()
    feed.stop()
    assert time.monotonic() - start < 1.0
    assert "stopped frame 5; frame 7 goes next" in capsys.readouterr().out
    feed.wait()
    assert not feed.busy() and [r["frame"] for r in healthy.requests] == [7]
    assert feed.last_reply["ok"] and len(hung.requests) == 1

    assert show_once(feed, index=8)                             # and the next 'n' works as usual
    feed.wait()
    assert [r["frame"] for r in healthy.requests] == [7, 8]


def test_h_says_nothing_more_once_it_has_stopped_a_send(tmp_path, addin, capsys):
    """The closed socket must not come back as a 'closed without answering' warning."""
    hung = addin("silent")
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path, port=hung.port, timeout_s=30.0)
    show_once(feed)
    until(lambda: hung.requests)
    feed.stop()
    feed.wait()
    out = capsys.readouterr().out
    assert "stopped frame 5." in out and "WARNING" not in out and feed.last_reply is None


def test_the_frame_is_built_on_the_thread_from_copies(tmp_path, station_requests):
    """The camera reuses its buffers: drawing over the image after show() must not reach the
    picture that is sent."""
    image = blocky_image()
    feed = sf.StationFeed(T_DOWN, out_dir=tmp_path)
    depth = FakeDepth(floor_depth(T_DOWN))
    feed.show(1, image, depth, INTR, [], image)
    image[:] = 0
    feed.wait()
    assert cv2.imread(station_requests[0]["floor"]).max() > 0


def test_stop_with_nothing_running_says_so(capsys):
    sf.StationFeed(T_DOWN).stop()
    assert "nothing was running" in capsys.readouterr().out


# =============================================================================
# the 'h' key
# =============================================================================
@pytest.fixture
def pressed(monkeypatch):
    def install(window=()):
        window = list(window)
        monkeypatch.setattr("cv2.waitKey", lambda delay=0: window.pop(0) if window else -1)
        monkeypatch.setattr("robot.operator_keys.console_key", lambda: None)
        monkeypatch.setattr("robot.operator_keys.time.sleep", lambda seconds: None)
    return install


def test_h_acts_and_is_not_handed_on(pressed):
    stopped = []
    pressed(window=[ord("h"), ord("n")])
    keys = OperatorKeys()
    keys.on("h", lambda: stopped.append(True))
    assert keys.poll() is None and stopped == [True]
    assert keys.poll() == "n"


@pytest.mark.parametrize("allowed, answer", [("cs", "s"), ("ynx", "x"), ("ynx", "y")])
def test_h_works_at_a_prompt_without_being_taken_as_its_answer(pressed, allowed, answer):
    stopped = []
    pressed(window=[-1, -1, ord("h"), -1, ord(answer)])    # after the prompt's drain
    keys = OperatorKeys()
    keys.on("h", lambda: stopped.append(True))
    assert keys.wait_for(allowed) == answer and stopped == [True]


def test_h_pressed_during_a_move_is_not_lost_to_the_drain(pressed):
    stopped = []
    pressed(window=[ord("h"), -1, ord("c")])
    keys = OperatorKeys()
    keys.on("h", lambda: stopped.append(True))
    assert keys.wait_for("cs") == "c" and stopped == [True]


def test_h_does_not_cut_the_drain_short(pressed):
    """A 'c' queued behind an 'h' during a move is still stale, and must not start the next pick."""
    stopped = []
    pressed(window=[ord("h"), ord("c"), -1, ord("s")])
    keys = OperatorKeys()
    keys.on("h", lambda: stopped.append(True))
    assert keys.wait_for("cs") == "s" and stopped == [True]


def test_h_is_heard_while_a_move_is_awaited_and_other_keys_are_dropped(pressed):
    stopped = []
    pressed(window=[ord("c"), ord("h"), -1, ord("s")])
    keys = OperatorKeys()
    keys.on("h", lambda: stopped.append(True))
    keys.pump()                                             # the 'c' mid-move: dropped
    keys.pump()
    assert stopped == [True] and keys.wait_for("cs") == "s"


def test_the_key_help_names_h():
    assert sf.STOP_KEY == "h" and "'h'" in sf.STOP_KEY_HELP and "RobotStudio" in sf.STOP_KEY_HELP


# =============================================================================
# the preview sends only the chosen frame
# =============================================================================
class RecordingStation:
    def __init__(self):
        self.shown = []

    def show(self, index, image, depth_frame, intrinsics, targets, annotated):
        self.shown.append((index, len(targets)))


def frame():
    return (np.zeros((INTR.height, INTR.width, 3), np.uint8),
            FakeDepth(np.full((INTR.height, INTR.width), 1.0)), INTR)


def preview(source, detector, keys, **kwargs):
    with started(source):
        return preview_until(source, detector, T_DOWN, keys, **kwargs)


def test_n_on_a_frame_with_nothing_pickable_still_shows_it(no_window):
    station = RecordingStation()
    source = FakeSource([frame() for _ in range(4)])
    detector = FakeDetector([[], [], [], [detection(disc_mask(320, 240, 10, INTR))]])
    snapshot = preview(source, detector, ScriptedKeys(polled=[None, "n", None, "n"]),
                       station=station)
    assert snapshot.frame_index == 3
    assert station.shown == [(1, 0), (3, 1)]


def test_nothing_is_shown_while_the_depth_settles_or_without_a_key(no_window):
    station = RecordingStation()
    source = FakeSource([frame() for _ in range(4)])
    source.stream_fps = 2.0                                     # settles after 2 frames
    detector = FakeDetector([[detection(disc_mask(320, 240, 10, INTR))]])
    preview(source, detector, ScriptedKeys(polled=["n", None, None, "n"]), station=station)
    assert station.shown == [(3, 1)]


def test_quitting_shows_nothing(no_window):
    station = RecordingStation()
    preview(FakeSource([frame(), frame()]), FakeDetector([[]]), ScriptedKeys(polled=[None, "q"]),
            station=station)
    assert station.shown == []


def test_a_clip_shows_only_the_frame_asked_for(no_window):
    station = RecordingStation()
    source = FakeSource([frame() for _ in range(5)])
    preview(source, FakeDetector([[]]), ScriptedKeys(), stop_at=3, station=station)
    assert station.shown == [(3, 0)]


# =============================================================================
# the add-in's copies
# =============================================================================
@pytest.fixture(scope="module")
def addin_source():
    return ADDIN_SOURCE.read_text(encoding="utf-8")


def test_the_addin_listens_where_python_sends(addin_source):
    assert re.search(r'StationHost\s*=\s*"([^"]+)"', addin_source).group(1) == sf.STATION_HOST
    assert int(re.search(r"StationPort\s*=\s*(\d+)", addin_source).group(1)) == sf.STATION_PORT


def test_the_addin_reads_the_cloud_file_python_writes(addin_source):
    assert f'"{sf.CLOUD_MAGIC.decode()}"' in addin_source


def test_the_addin_takes_the_visibility_python_sends(addin_source):
    assert '"visibility" => Visibility(request)' in addin_source
    for key in ("cloud_visible", "floor_visible"):
        assert f'TryGetProperty("{key}"' in addin_source


def test_the_addin_maps_texture_coordinates_as_floor_uv_does(addin_source):
    """The formula in floor_uv, written the same way in C#, so a change to one shows up here."""
    assert re.search(r"\(\s*x\s*-\s*xMin\s*\)\s*/\s*width", addin_source)
    assert re.search(r"\(\s*yMax\s*-\s*y\s*\)\s*/\s*height", addin_source)
