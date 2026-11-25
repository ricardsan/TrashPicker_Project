"""The command line, through main().

The first half needs no checkpoint. The tests marked `slow` run the real model on real clips against
the fake controller, and skip if either is not on this machine.
"""
from __future__ import annotations

import functools
import json
import time

import cv2
import numpy as np
import pytest

import robot.robot_trash_picker as cli
from camera.segmentation_realsense import is_lfs_pointer
from robot import pick_sequence
from robot.extrinsics import DEFAULT_CALIBRATION, HARDCODED_T_BASE_CAM, load_T_base_cam
from robot.art_camera_client import MOCK_STEPS
from robot.frame_selection import EXPORT_DIR, SETTLE_S
from robot.mock.fake_rapid import DROP, FakeRapid, parse_message
from robot.operator_keys import OperatorKeys
from robot.station_feed import StationFeed
from synthetic import FakeDepth, FakeDetector, FakeSource, Intrinsics, detection, disc_mask
# Imported at collection, before the autouse fixture swaps _exchange for a recorder.
from test_station_feed import REAL_EXCHANGE, FakeAddin

HOME, GRIP_OPEN, GRIP_CLOSE = "-1,0,0,0", "-2,0,0,0", "-3,0,0,0"
DEMO_CALIBRATION = DEFAULT_CALIBRATION.parent / "calibration_rosbag_demo.txt"
CLIPS = ("20261016_231841_clutter_med", "20261016_232523_clutter_high2", "20261016_233733_stacked")
SETTLED_FRAME = "20"    # past the depth's first second, without running inference on a whole clip
INTR = Intrinsics()


# =============================================================================
# flags
# =============================================================================
@pytest.mark.parametrize("argv", [
    ["--mock", "--rosbag", "clip.db3"],
    ["--mock", "--model", "best.pt"],
    ["--mock", "--frame", "3"],
    ["--mock", "--calibration", "x.txt"],
    ["--mock", "--conf", "0.5"],
    ["--mock", "--no-station"],
    ["--mock", "--hide-cloud"],
    ["--mock", "--hide-rgb"],
    ["--rosbag", "clip", "--no-station", "--hide-rgb"],
    ["--rosbag", "clip", "--no-station", "--hide-cloud"],
    ["--frame", "3"],
])
def test_flags_that_mean_nothing_together_are_refused(argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2


def test_the_default_checkpoint_is_the_one_asked_for():
    assert cli.DEFAULT_WEIGHTS.relative_to(cli.ROOT).as_posix() == (
        "runs/segment/runs/segment/original-seg/original-finetune-augment_v/weights/best.pt")


def test_the_default_controller_is_the_virtual_one():
    args = cli.build_parser().parse_args([])
    assert (args.host, args.port) == ("127.0.0.1", 5002)


# =============================================================================
# --mock
# =============================================================================
def test_mock_passes_and_exits_zero(rapid, no_window, capsys):
    assert cli.main(["--mock", "--port", str(rapid.port)]) == 0
    assert f"{len(MOCK_STEPS)} of {len(MOCK_STEPS)}" in capsys.readouterr().out


def test_mock_exits_non_zero_when_a_reply_differs(no_window):
    with FakeRapid(script={2: "REJECTED"}) as server:
        assert cli.main(["--mock", "--port", str(server.port)]) == 1


def test_no_controller_is_a_message_and_not_a_traceback(no_window, capsys):
    with FakeRapid() as server:
        port = server.port
    assert cli.main(["--mock", "--port", str(port)]) == 1
    assert "RAPID program" in capsys.readouterr().err


# =============================================================================
# a whole session with a fake model and a fake clip
# =============================================================================
@pytest.fixture
def fake_inputs(monkeypatch, tmp_path, no_window):
    """main() with the model and the frames swapped out, and everything else real.

    The camera looks straight down from 1.5 m above the base, with the image centre over x = 450,
    so depths read directly as heights: a depth of 1.30 m is z = 200. The three objects sit along
    the image's vertical, which is the base's x, all inside the workspace.
    """
    calibration = tmp_path / "T.txt"
    calibration.write_text("[[0 -1 0 0.45]\n [-1 0 0 0.0]\n [0 0 -1 1.5]\n [0 0 0 1]]")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"\x80\x05" + b"\x00" * 4096)

    metres = np.full((INTR.height, INTR.width), 1.45)
    metres[225:256, 305:336] = 1.30     # z = 200, class 1
    metres[165:196, 305:336] = 1.25     # z = 250, class 2
    metres[285:316, 305:336] = 1.35     # z = 150, class 3
    detections = [detection(disc_mask(320, 240, 15, INTR), cls=1),
                  detection(disc_mask(320, 180, 15, INTR), cls=2),
                  detection(disc_mask(320, 300, 15, INTR), cls=3)]
    image = np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)

    source = FakeSource([(image, FakeDepth(metres), INTR) for _ in range(3)])
    monkeypatch.setattr(cli, "open_recording", lambda name: source)
    monkeypatch.setattr(cli, "YoloDetector", lambda path, conf: FakeDetector([detections]))

    # The real wait_for loops forever. A regression that reaches an unplanned prompt has to fail
    # the test and not hang the suite, so any test expecting a prompt says what it answers.
    def unplanned_prompt(self, allowed):
        pytest.fail("reached a prompt this test did not expect")

    monkeypatch.setattr(OperatorKeys, "wait_for", unplanned_prompt)
    return ["--calibration", str(calibration), "--model", str(weights)], source


def picks_in(received):
    """The class of each object that was gripped, in order."""
    return [received[i - 1].split(",")[0] for i, message in enumerate(received)
            if message == GRIP_CLOSE]


def test_a_session_homes_opens_then_picks_nearest_first(rapid, fake_inputs, monkeypatch):
    args, _ = fake_inputs
    keys = iter("cc")
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: next(keys))
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", *args]) == 0
    assert rapid.received[:2] == [HOME, GRIP_OPEN]
    assert picks_in(rapid.received) == ["2", "1", "3"]      # depths 1.25, 1.30, 1.35
    assert rapid.received[-1] == HOME


def test_s_after_the_first_pick_ends_the_session(rapid, fake_inputs, monkeypatch):
    args, _ = fake_inputs
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", *args]) == 0
    assert picks_in(rapid.received) == ["2"]


def test_the_model_loads_then_the_robot_goes_home_then_inference_runs(rapid, fake_inputs,
                                                                      monkeypatch):
    """The order the run was asked for. Each stage records what the robot had been told by then."""
    args, _ = fake_inputs
    at_load, at_inference = [], []

    class Spy(FakeDetector):
        def predict(self, image):
            at_inference.append(list(rapid.received))
            return super().predict(image)

    def build(path, conf):
        at_load.append(list(rapid.received))
        return Spy([[]])

    monkeypatch.setattr(cli, "YoloDetector", build)
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", *args]) == 0
    assert at_load == [[]]
    assert at_inference[0] == [HOME, GRIP_OPEN]


def test_a_camera_session_takes_the_frame_on_n_and_picks_from_it(rapid, fake_inputs, monkeypatch):
    """The same session with no --rosbag: the real command line's camera path, on a fake camera."""
    args, source = fake_inputs
    monkeypatch.setattr(cli, "RealSenseSource", lambda: source)
    presses = iter([None,       # the drain finds nothing queued
                    None,       # frame 0
                    "n"])       # frame 1
    monkeypatch.setattr(OperatorKeys, "poll", lambda self, delay_ms=1: next(presses, None))
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), *args]) == 0
    assert source.position == 2
    assert source.stopped
    assert picks_in(rapid.received) == ["2"]


def test_quitting_the_preview_leaves_the_robot_at_home(rapid, fake_inputs, monkeypatch, capsys):
    args, source = fake_inputs
    monkeypatch.setattr(cli, "RealSenseSource", lambda: source)
    presses = iter([None, "q"])
    monkeypatch.setattr(OperatorKeys, "poll", lambda self, delay_ms=1: next(presses, None))
    assert cli.main(["--port", str(rapid.port), *args]) == 0
    assert rapid.received == [HOME, GRIP_OPEN]
    assert "Quit before picking" in capsys.readouterr().out


def test_a_camera_that_is_not_there_is_a_message_and_the_robot_does_not_move(rapid, fake_inputs,
                                                                             monkeypatch):
    args, source = fake_inputs

    def refuse():
        raise RuntimeError("No device connected")

    source.start = refuse
    monkeypatch.setattr(cli, "RealSenseSource", lambda: source)
    with pytest.raises(SystemExit, match="could not start.*No device connected"):
        cli.main(["--port", str(rapid.port), *args])
    assert rapid.received == []


def test_the_chosen_frame_and_only_it_is_sent_to_robotstudio(rapid, fake_inputs, monkeypatch,
                                                            capsys, station_requests):
    args, _ = fake_inputs
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--frame", "1", *args]) == 0
    assert [request["cmd"] for request in station_requests] == ["visibility", "show"]
    request = station_requests[1]
    assert request["frame"] == 1
    assert request["part"] == "BoardOverlay" and request["points"] > 0
    out = capsys.readouterr().out
    assert "'h' stops the RobotStudio display thread" in out


def test_the_real_send_finishes_before_the_program_exits(rapid, fake_inputs, monkeypatch, capsys):
    """The thread is a daemon: without the wait on the way out, a quick exit would cut it off."""
    monkeypatch.setattr(StationFeed, "_exchange", REAL_EXCHANGE)
    addin = FakeAddin("ok")
    try:
        monkeypatch.setattr(cli, "StationFeed", functools.partial(StationFeed, port=addin.port))
        monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
        assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--frame", "2", *args_of(
            fake_inputs)]) == 0
    finally:
        addin.close()
    assert [request.get("frame") for request in addin.requests] == [None, 2]   # visibility first
    assert "shown in RobotStudio: frame 2" in capsys.readouterr().out


def test_a_robot_error_is_not_held_up_by_a_hung_robotstudio(fake_inputs, monkeypatch, capsys):
    """The wait on the way out is for a normal end only: a lost controller is reported at once."""
    monkeypatch.setattr(StationFeed, "_exchange", REAL_EXCHANGE)
    hung = FakeAddin("silent")
    script = {2: DROP}                          # home, grip open, then the controller hangs up
    try:
        monkeypatch.setattr(cli, "StationFeed",
                            functools.partial(StationFeed, port=hung.port, timeout_s=30.0))
        with FakeRapid(script=script) as rapid:
            start = time.monotonic()
            assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", *args_of(
                fake_inputs)]) == 1
            assert time.monotonic() - start < 10.0
    finally:
        hung.close()
    assert "STOPPED" in capsys.readouterr().err


def args_of(fake_inputs):
    return fake_inputs[0]


@pytest.mark.parametrize("flags, cloud, floor", [
    ([], True, True),                    # both stated, so an earlier run's --hide-* is undone
    (["--hide-cloud"], False, True),
    (["--hide-rgb"], True, False),
    (["--hide-cloud", "--hide-rgb"], False, False),
])
def test_the_hide_options_are_sent_at_start_and_with_the_frame(rapid, fake_inputs, monkeypatch,
                                                               capsys, station_requests, flags,
                                                               cloud, floor):
    args, _ = fake_inputs
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--frame", "1", *flags,
                     *args]) == 0
    start, frame = station_requests
    assert start == {"cmd": "visibility", "part": "BoardOverlay", "cloud_visible": cloud,
                     "floor_visible": floor}
    assert (frame["cmd"], frame["cloud_visible"], frame["floor_visible"]) == ("show", cloud, floor)
    out = capsys.readouterr().out
    assert ("as LiveCloud" in out) is cloud and ("on BoardOverlay" in out) is floor


def test_the_visibility_goes_out_before_the_robot_is_contacted(fake_inputs, monkeypatch,
                                                               station_requests):
    """'Sent when starting': before connecting, so it is there even if the controller is not."""
    args, _ = fake_inputs
    with FakeRapid() as rapid:
        port = rapid.port
    assert cli.main(["--port", str(port), "--rosbag", "clip", "--hide-cloud", *args]) == 1
    assert [request["cmd"] for request in station_requests] == ["visibility"]


def test_no_station_sends_nothing(rapid, fake_inputs, monkeypatch, station_requests):
    args, _ = fake_inputs
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--no-station", *args]) == 0
    assert station_requests == []


@pytest.mark.parametrize("frame, warned", [("0", True), ("1", False)])
def test_a_clip_frame_from_the_unsettled_start_is_warned_about(rapid, fake_inputs, monkeypatch,
                                                               capsys, frame, warned):
    args, source = fake_inputs
    source.stream_fps = 1.0     # settles after 1 frame
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--frame", frame, *args]) == 0
    assert ("WARNING" in capsys.readouterr().out) is warned


@pytest.mark.parametrize("host", ["192.168.125.1", "192.168.1.131", "robot.local"])
def test_the_synthetic_calibration_is_refused_for_any_other_machine(no_window, host):
    """Its positions correspond to nothing in the room, and a real arm would still go to them."""
    with pytest.raises(SystemExit, match="synthetic"):
        cli.main(["--host", host, "--rosbag", "clip", "--calibration", str(DEMO_CALIBRATION)])


def test_the_real_calibration_is_not_refused_for_another_machine(fake_inputs, monkeypatch):
    """Stopped by the bag lookup, which comes after the check under test."""
    args, _ = fake_inputs

    def stop_here(name):
        raise SystemExit("reached the bag lookup")

    monkeypatch.setattr(cli, "open_recording", stop_here)
    with pytest.raises(SystemExit, match="reached the bag lookup"):
        cli.main(["--host", "192.168.125.1", "--rosbag", "clip", "--model", args[3],
                  "--calibration", str(DEFAULT_CALIBRATION)])


def extrinsics_loaded(monkeypatch) -> list:
    """Stops a session where the preview would start, and records the T it was given."""
    seen = []

    def stop(source, detector, T_base_cam, keys, **kwargs):
        seen.append(T_base_cam)
        return None

    monkeypatch.setattr(cli, "preview_until", stop)
    return seen


def test_the_program_loads_the_hardcoded_extrinsics_by_default(rapid, fake_inputs, monkeypatch,
                                                               capsys):
    args, _ = fake_inputs
    seen = extrinsics_loaded(monkeypatch)
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--model", args[3]]) == 0
    assert np.array_equal(seen[0], HARDCODED_T_BASE_CAM)
    assert "calibration: the hardcoded extrinsics" in capsys.readouterr().out


def test_a_camera_session_loads_them_as_well(rapid, fake_inputs, monkeypatch):
    args, source = fake_inputs
    monkeypatch.setattr(cli, "RealSenseSource", lambda: source)
    seen = extrinsics_loaded(monkeypatch)
    assert cli.main(["--port", str(rapid.port), "--model", args[3]]) == 0
    assert np.array_equal(seen[0], HARDCODED_T_BASE_CAM)


def test_calibration_still_overrides_them(rapid, fake_inputs, monkeypatch):
    args, _ = fake_inputs
    seen = extrinsics_loaded(monkeypatch)
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--model", args[3],
                     "--calibration", str(DEFAULT_CALIBRATION)]) == 0
    assert np.array_equal(seen[0], load_T_base_cam(DEFAULT_CALIBRATION))


@pytest.mark.parametrize("host", ["192.168.125.1", "192.168.1.131", "robot.local"])
def test_the_hardcoded_extrinsics_are_refused_for_any_other_machine(rapid, no_window, host):
    """Their position is chosen, and a real arm would still go where they say."""
    with pytest.raises(SystemExit, match="hardcoded.*synthetic.*--calibration"):
        cli.main(["--host", host, "--port", str(rapid.port), "--rosbag", "clip"])
    assert rapid.connections == 0


def test_the_synthetic_calibration_works_on_this_machine(rapid, fake_inputs, monkeypatch):
    args, _ = fake_inputs
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", "clip",
                     "--calibration", str(DEMO_CALIBRATION), "--model", args[3]]) == 0
    assert rapid.received[:2] == [HOME, GRIP_OPEN]


def test_a_bad_drop_location_fails_before_the_robot_is_contacted(rapid, fake_inputs, monkeypatch):
    args, _ = fake_inputs
    monkeypatch.setitem(pick_sequence.DROP_LOCATIONS_MM, 0, (100.0, 0.0, 250.0))
    with pytest.raises(SystemExit, match="DROP_LOCATIONS_MM"):
        cli.main(["--port", str(rapid.port), "--rosbag", "clip", *args])
    assert rapid.connections == 0


def test_the_summary_does_not_claim_an_object_was_held(rapid, fake_inputs, monkeypatch, capsys):
    """RAPID's GripClose answers DONE even when it times out, so only the sequence is known."""
    args, _ = fake_inputs
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    cli.main(["--port", str(rapid.port), "--rosbag", "clip", *args])
    out = capsys.readouterr().out
    assert "Finished 1 pick sequence" in out and "does not say" in out


def test_a_bad_frame_fails_before_the_robot_is_even_contacted(rapid, fake_inputs):
    args, _ = fake_inputs
    with pytest.raises(SystemExit, match="outside this clip"):
        cli.main(["--port", str(rapid.port), "--rosbag", "clip", "--frame", "99", *args])
    assert rapid.connections == 0


def test_a_missing_checkpoint_fails_before_the_robot_is_contacted(rapid, no_window, tmp_path):
    with pytest.raises(SystemExit, match="no checkpoint"):
        cli.main(["--port", str(rapid.port), "--rosbag", "clip",
                  "--model", str(tmp_path / "nothing.pt")])
    assert rapid.connections == 0


def test_a_lost_controller_mid_session_is_a_message_and_a_stop(fake_inputs, monkeypatch, capsys):
    args, _ = fake_inputs
    with FakeRapid(script={3: None}) as server:     # the first descent
        assert cli.main(["--port", str(server.port), "--rosbag", "clip", *args]) == 1
        assert len(server.received) == 4
    assert "STOPPED" in capsys.readouterr().err


def test_a_rejection_while_holding_is_reported_as_a_stop(fake_inputs, capsys):
    args, _ = fake_inputs
    with FakeRapid(script={5: "REJECTED"}) as server:       # home, open, over, down, close, up
        assert cli.main(["--port", str(server.port), "--rosbag", "clip", *args]) == 1
        assert GRIP_OPEN not in server.received[2:]
    assert "still holding" in capsys.readouterr().err


# =============================================================================
# the real thing, on the fake controller
# =============================================================================
real_inputs = pytest.mark.skipif(
    not cli.DEFAULT_WEIGHTS.exists() or is_lfs_pointer(cli.DEFAULT_WEIGHTS)
    or not all((EXPORT_DIR / clip / "meta.json").exists() for clip in CLIPS),
    reason="needs the default checkpoint (git lfs pull) and the unpacked clips")


@pytest.mark.slow
@real_inputs
@pytest.mark.parametrize("clip", CLIPS)
def test_a_real_clip_with_the_demo_calibration_completes_real_picks(
        rapid, no_window, rs_or_stub, monkeypatch, clip):
    """The real checkpoint and real pixels, on a frame just past where the depth has settled."""
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "c")
    assert cli.main(["--port", str(rapid.port), "--rosbag", f"{clip}.db3", "--frame", SETTLED_FRAME,
                     "--calibration", str(DEMO_CALIBRATION)]) == 0
    assert rapid.received[:2] == [HOME, GRIP_OPEN]
    assert picks_in(rapid.received), "no object on this clip was picked with the demo calibration"
    assert rapid.received[-1] == HOME
    assert rapid.gripper == "open"

    # What RobotStudio showed going wrong: every descent has to be one, onto a level scene.
    for index, message in enumerate(rapid.received):
        if message != GRIP_CLOSE:
            continue
        over, down, up, drop = (parse_message(rapid.received[index + offset])
                                for offset in (-2, -1, 1, 2))
        assert over[3] == up[3] == pick_sequence.APPROACH_Z_MM
        assert pick_sequence.GRASP_Z_MIN_MM <= down[3] < over[3]
        assert over[1:3] == down[1:3] == up[1:3]
        assert over[4] == down[4] == up[4]
        assert -90.0 <= down[4] <= 90.0     # 89.96 goes out as 90.0
        assert rapid.received[index + 2].count(",") == 3, "the drop move carries no rz"
        assert drop[4] == 0.0


@pytest.mark.slow
@real_inputs
@pytest.mark.parametrize("clip", CLIPS)
def test_a_real_clip_with_the_real_calibration_picks_nothing(
        rapid, no_window, rs_or_stub, monkeypatch, clip):
    """Pins a known mismatch: the clips were not filmed on the calibrated rig.

    If this starts failing, the clips and the calibration have come to agree, which is good news:
    delete this test and calibration_rosbag_demo.txt.
    """
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "c")
    assert cli.main(["--port", str(rapid.port), "--rosbag", f"{clip}.db3",
                     "--frame", SETTLED_FRAME, "--calibration", str(DEFAULT_CALIBRATION)]) == 0
    assert picks_in(rapid.received) == []


@pytest.mark.slow
@real_inputs
@pytest.mark.parametrize("clip", CLIPS)
def test_the_settling_period_covers_the_bad_depth_of_every_clip(clip):
    """Pins SETTLE_S against the recordings: holes well above 30% before it, under 5% after it."""
    meta = json.loads((EXPORT_DIR / clip / "meta.json").read_text())
    settle = int(SETTLE_S * meta["stream_fps"])

    def holes(index):
        depth = cv2.imread(str(EXPORT_DIR / clip / meta["frames"][index]["depth"]),
                           cv2.IMREAD_UNCHANGED)
        return float((depth == 0).mean())

    assert all(holes(index) > 0.30 for index in range(0, 8))
    assert all(holes(index) < 0.05 for index in range(settle, settle + 10))


@pytest.mark.slow
@real_inputs
def test_the_default_frame_is_the_last_one(rapid, no_window, rs_or_stub, monkeypatch, capsys):
    monkeypatch.setattr(OperatorKeys, "wait_for", lambda self, allowed: "s")
    assert cli.main(["--port", str(rapid.port), "--rosbag", f"{CLIPS[2]}.db3",
                     "--calibration", str(DEMO_CALIBRATION)]) == 0
    assert "Detections on frame 141" in capsys.readouterr().out
