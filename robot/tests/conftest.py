"""Shared fixtures for the robot tests.

Run these with `pytest robot/tests`, never `pytest robot`: robot/test_robot.py is an old script that
opens a socket as soon as it is imported.

Everything is synthetic except the tests marked `slow`, which need a real checkpoint and an unpacked
clip. Nothing here needs a robot, a camera or a display: the controller is mock/fake_rapid.py.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot.art_camera_client import ArtCameraClient  # noqa: E402
from robot.mock.fake_rapid import FakeRapid  # noqa: E402
from synthetic import pinhole_rs  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: loads a real checkpoint; deselect with -m 'not slow'")


@pytest.fixture(autouse=True)
def floor_pictures_in_tmp(monkeypatch, tmp_path):
    """A run writes the floor picture for RobotStudio; in a test it goes to tmp, not the repo."""
    monkeypatch.setattr("robot.floor_overlay.LIVE_DIR", tmp_path / "robotstudio_live")
    return tmp_path / "robotstudio_live"


@pytest.fixture(autouse=True)
def station_requests(monkeypatch):
    """A run pushes its snapshot to the RobotStudio add-in; in a test it is recorded, never sent,
    so a RobotStudio open on this machine is not touched. test_station_feed puts the real one back
    where it is the thing under test."""
    sent = []

    def record(self, request):
        sent.append(request)
        self.last_reply = {"ok": True, "points": request.get("points")}

    monkeypatch.setattr("robot.station_feed.StationFeed._exchange", record)
    yield sent
    # A run that ends on an error does not wait for its send thread. Finish it here, or it records
    # into the next test's list.
    for thread in threading.enumerate():
        if thread.name == "robotstudio-station":
            thread.join(timeout=5.0)


@pytest.fixture
def rapid():
    """A running fake controller. Scripted variants are built in the test with FakeRapid(script=)."""
    with FakeRapid() as server:
        yield server


@pytest.fixture
def client(rapid):
    with ArtCameraClient("127.0.0.1", rapid.port, reply_timeout_s=5.0) as connected:
        yield connected


@pytest.fixture
def no_window(monkeypatch):
    """cv2 would open a window and block. Nothing is pressed unless a test says so."""
    monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)
    monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)
    monkeypatch.setattr("cv2.waitKey", lambda delay=0: -1)
    monkeypatch.setattr("robot.operator_keys.console_key", lambda: None)


@pytest.fixture
def rs_or_stub(monkeypatch):
    """The real pyrealsense2 where it is installed, otherwise the pinhole stand-in.

    The colour stream's distortion coefficients are all zero in every clip, so the two give the same
    answer on this material. test_the_real_deproject_is_this_pinhole skips rather than passes when
    the real one is missing, so a machine without it says so in the summary.
    """
    try:
        import pyrealsense2
    except ImportError:
        module = pinhole_rs()
        monkeypatch.setitem(sys.modules, "pyrealsense2", module)
        return module
    return pyrealsense2
