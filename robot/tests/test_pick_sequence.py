"""The order of things a robot is told, and what happens when it says no."""
from __future__ import annotations

import pytest

from robot.art_camera_client import Z_TABLE_MM, ArtCameraClient, RobotError
from robot.mock.fake_rapid import DROP, FakeRapid
from robot import pick_sequence
from robot.pick_sequence import (APPROACH_Z_MM, DROP_LOCATIONS_MM, FINGER_CLOSE_AXIS_AT_RZ0_DEG,
                                 GRASP_Z_MAX_MM, GRASP_Z_MIN_MM, PickAborted, gripper_rz_deg,
                                 pick_and_place, run_picks)
from synthetic import ScriptedKeys, make_target

HOME, GRIP_OPEN, GRIP_CLOSE = "-1,0,0,0", "-2,0,0,0", "-3,0,0,0"


def session(script):
    """A scripted controller and a client on it, for a failure at a chosen message."""
    server = FakeRapid(script=script)
    server.__enter__()
    client = ArtCameraClient("127.0.0.1", server.port, reply_timeout_s=2.0)
    client.connect()
    return server, client


def test_the_whole_sequence_in_order(client, rapid):
    """Over it at home's height, down, grip, back up, drop, open, home. Nothing else."""
    assert pick_and_place(client, make_target(cls=2, base=(400.0, 100.0, 120.0))) is True
    assert rapid.received == [
        "2,400.0,100.0,400.0",      # over the object, z unchanged from home
        "2,400.0,100.0,120.0",      # straight down to the z it perceived
        GRIP_CLOSE,
        "2,400.0,100.0,400.0",      # back up to where it started
        "2,500.0,-300.0,250.0",     # the drop location for class 2
        GRIP_OPEN,
        HOME,
    ]


def test_it_ends_at_home_with_the_gripper_open(client, rapid):
    pick_and_place(client, make_target())
    assert rapid.pose == rapid.limits["pHome"]
    assert rapid.gripper == "open"


@pytest.mark.parametrize("cls", sorted(DROP_LOCATIONS_MM))
def test_each_class_goes_to_its_own_drop_location(client, rapid, cls):
    pick_and_place(client, make_target(cls=cls))
    x, y, z = DROP_LOCATIONS_MM[cls]
    assert rapid.received[4] == f"{cls},{x:.1f},{y:.1f},{z:.1f}"


def test_the_descent_is_vertical_and_the_climb_retraces_it(client, rapid):
    pick_and_place(client, make_target(base=(321.5, -77.2, 150.0)))
    over, down, _, up = rapid.received[:4]
    assert over.split(",")[1:3] == down.split(",")[1:3] == up.split(",")[1:3]
    assert over == up


@pytest.mark.parametrize("z", [GRASP_Z_MIN_MM, 136.3, 245.0, GRASP_Z_MAX_MM - 0.1])
def test_going_down_is_a_lower_positive_z(client, rapid, z):
    """The base's +z is up, as checked in RobotStudio, so a positive z below the approach plane is a
    descent. This is the sign a scene has to arrive with: it is sent as it is."""
    pick_and_place(client, make_target(base=(400.0, 100.0, z)))
    over, down = (float(message.split(",")[3]) for message in rapid.received[:2])
    assert over == APPROACH_Z_MM
    assert down == pytest.approx(z, abs=0.05)
    assert 0.0 < down < over


# =============================================================================
# the z guard
# =============================================================================
@pytest.mark.parametrize("z", [GRASP_Z_MIN_MM - 0.1, -50.0, -136.3, GRASP_Z_MAX_MM + 0.1,
                               416.3,   # the tall bottle in the stacked clip: would climb
                               628.0,   # a clip's sheet through the real calibration: would climb
                               780.0, Z_TABLE_MM, float("nan"), float("inf"), float("-inf")])
def test_a_z_outside_the_guard_does_not_move_the_robot_at_all(client, rapid, capsys, z):
    """The controller would accept these: InWorkspace does not look at z."""
    assert pick_and_place(client, make_target(base=(400.0, 0.0, z))) is False
    assert rapid.received == []
    assert "SKIPPED" in capsys.readouterr().out


def test_only_a_z_above_the_approach_plane_points_at_the_extrinsics(client, capsys):
    """A z under the floor is a flat object or a bad reading, with any extrinsics."""
    pick_and_place(client, make_target(base=(400.0, 0.0, 628.0)))
    assert "extrinsics in use" in capsys.readouterr().out
    pick_and_place(client, make_target(base=(400.0, 0.0, 80.0)))
    assert "extrinsics in use" not in capsys.readouterr().out


def test_a_rejected_position_points_at_the_extrinsics_and_at_no_particular_file(client, capsys):
    pick_and_place(client, make_target(base=(100.0, 0.0, 120.0)))
    out = capsys.readouterr().out
    assert "extrinsics in use" in out
    assert "calibration_rosbag_demo" not in out and "calibration.txt" not in out


@pytest.mark.parametrize("z", [GRASP_Z_MIN_MM, GRASP_Z_MAX_MM])
def test_the_ends_of_the_guard_are_allowed(client, z):
    assert pick_and_place(client, make_target(base=(400.0, 0.0, z))) is True


def test_a_class_with_no_drop_location_does_not_move_the_robot(client, rapid, capsys):
    assert pick_and_place(client, make_target(cls=9)) is False
    assert rapid.received == []
    assert "no drop location" in capsys.readouterr().out


# =============================================================================
# the gripper's rz
# =============================================================================
def line_distance_deg(a: float, b: float) -> float:
    """Between two lines, which wrap at 180."""
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def test_the_fingers_close_along_the_bases_x_at_rz_0():
    """Read off the RobotStudio station."""
    assert FINGER_CLOSE_AXIS_AT_RZ0_DEG == 0.0


@pytest.mark.parametrize("axis, rz", [
    (0.0, -90.0),       # lying along x: the fingers have to turn to close along y
    (90.0, 0.0),        # lying along y: the fingers already close across it
    (45.0, -45.0),
    (135.0, 45.0),
    (1.0, -89.0),
    (179.0, 89.0),
])
def test_the_rz_for_an_object_axis(axis, rz):
    assert gripper_rz_deg(axis) == pytest.approx(rz)


def test_the_fingers_end_up_across_the_axis_at_every_angle_with_the_smaller_turn():
    for axis in [value / 2.0 for value in range(360)]:
        rz = gripper_rz_deg(axis)
        assert -90.0 <= rz < 90.0
        assert line_distance_deg(FINGER_CLOSE_AXIS_AT_RZ0_DEG + rz, axis) == pytest.approx(90.0)


def test_the_rz_follows_how_the_fingers_are_mounted(monkeypatch):
    """30 and not 90, which is its own negative modulo 180 and so cannot catch a wrong sign."""
    monkeypatch.setattr(pick_sequence, "FINGER_CLOSE_AXIS_AT_RZ0_DEG", 30.0)
    assert gripper_rz_deg(0.0) == pytest.approx(60.0)
    assert gripper_rz_deg(90.0) == pytest.approx(-30.0)


def test_no_axis_means_no_rz():
    assert gripper_rz_deg(None) is None


def test_over_down_and_up_carry_the_rz_and_the_drop_does_not(client, rapid):
    target = make_target(cls=2, base=(400.0, 100.0, 120.0), base_axis_deg=45.0)
    assert pick_and_place(client, target) is True
    assert rapid.received == [
        "2,400.0,100.0,400.0,-45.0",
        "2,400.0,100.0,120.0,-45.0",
        GRIP_CLOSE,
        "2,400.0,100.0,400.0,-45.0",
        "2,500.0,-300.0,250.0",
        GRIP_OPEN,
        HOME,
    ]


def test_the_tool_turns_before_it_goes_down(client, rapid):
    """So the fingers reach the object already lined up, and do not sweep round it on the way."""
    pick_and_place(client, make_target(base_axis_deg=30.0))
    assert rapid.received[0].endswith(",-60.0")


def test_a_target_with_no_axis_sends_four_fields_on_every_move(client, rapid):
    pick_and_place(client, make_target(base_axis_deg=None))
    assert all(message.count(",") == 3 for message in rapid.received)


def test_the_operator_is_shown_the_rz(client, capsys):
    pick_and_place(client, make_target(base_axis_deg=45.0))
    assert "rz -45.0" in capsys.readouterr().out


def test_every_rz_a_pick_can_send_is_one_the_controller_accepts(client, rapid):
    """Otherwise a turn could be refused after the gripper has closed, with an object in the air."""
    for axis in [value * 7.5 for value in range(24)]:
        assert pick_and_place(client, make_target(base_axis_deg=axis)) is True


# =============================================================================
# when the controller says no
# =============================================================================
def test_rejected_over_the_object_skips_it_and_goes_home(capsys):
    server, client = session({0: "REJECTED"})
    try:
        assert pick_and_place(client, make_target(base=(400.0, 0.0, 120.0))) is False
        assert server.received == ["0,400.0,0.0,400.0", HOME]
        assert GRIP_CLOSE not in server.received
        assert "SKIPPED" in capsys.readouterr().out
    finally:
        client.close()
        server.__exit__(None, None, None)


@pytest.mark.parametrize("x, y", [(0.0, 0.0), (250.0, 100.0), (820.0, 0.0), (600.0, -600.0)])
def test_out_of_reach_does_not_move_the_robot_at_all(client, rapid, capsys, x, y):
    """Too close to the base or too far out: the arm would fold or stretch, so nothing is sent."""
    assert pick_and_place(client, make_target(base=(x, y, 120.0))) is False
    assert rapid.received == []
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "reach" in out and "extrinsics in use" in out


@pytest.mark.parametrize("x, y", [(300.0, 0.0), (0.0, -800.0), (500.0, 400.0)])
def test_the_edges_of_the_reach_are_allowed(client, x, y):
    assert pick_and_place(client, make_target(base=(x, y, 120.0))) is True


def test_the_grasp_z_is_the_top_of_the_object_and_stays_above_the_floor():
    from robot.pick_sequence import grasp_z_for
    assert grasp_z_for(make_target(base=(400.0, 0.0, 88.3))) == 88.3
    assert GRASP_Z_MIN_MM > Z_TABLE_MM == 0.0


def test_rejected_on_the_way_down_goes_home_without_gripping():
    server, client = session({1: "REJECTED"})
    try:
        assert pick_and_place(client, make_target()) is False
        assert server.received[2:] == [HOME]
        assert GRIP_CLOSE not in server.received
    finally:
        client.close()
        server.__exit__()


@pytest.mark.parametrize("failing", [3, 4], ids=["climbing", "to the drop location"])
def test_rejected_while_holding_stops_without_letting_go(failing):
    """Opening or going home from here would drop whatever it holds, somewhere unplanned."""
    server, client = session({failing: "REJECTED"})
    try:
        with pytest.raises(PickAborted, match="still holding"):
            pick_and_place(client, make_target())
        assert len(server.received) == failing + 1
        assert GRIP_OPEN not in server.received
        assert HOME not in server.received
        assert server.gripper == "closed"
    finally:
        client.close()
        server.__exit__()


@pytest.mark.parametrize("failing", range(7))
def test_a_parse_error_anywhere_sends_nothing_more(failing):
    server, client = session({failing: "PARSE_ERROR"})
    try:
        with pytest.raises(RobotError):
            pick_and_place(client, make_target())
        assert len(server.received) == failing + 1
    finally:
        client.close()
        server.__exit__()


@pytest.mark.parametrize("failing", range(7))
def test_a_lost_connection_anywhere_is_a_stop(failing):
    server, client = session({failing: DROP})
    try:
        with pytest.raises(RobotError, match="closed the connection"):
            pick_and_place(client, make_target())
    finally:
        client.close()
        server.__exit__()


# =============================================================================
# between picks
# =============================================================================
def test_c_moves_on_to_the_second_target_in_the_list(client, rapid):
    targets = [make_target(cls=0, base=(300.0, 0.0, 100.0)),
               make_target(cls=1, base=(350.0, 0.0, 100.0))]
    assert run_picks(client, targets, ScriptedKeys(waited=["c"])) == 2
    moves = [message for message in rapid.received if not message.startswith("-")]
    assert moves[0].startswith("0,300.0") and moves[4].startswith("1,350.0")


def test_s_stops_with_the_robot_at_home_and_the_rest_untouched(client, rapid, capsys):
    targets = [make_target(base=(300.0, 0.0, 100.0)), make_target(base=(350.0, 0.0, 100.0))]
    assert run_picks(client, targets, ScriptedKeys(waited=["s"])) == 1
    assert not any("350.0" in message for message in rapid.received)
    assert rapid.received[-1] == HOME
    assert "Stopping" in capsys.readouterr().out


def test_it_prints_what_to_press(client, capsys):
    run_picks(client, [make_target(), make_target()], ScriptedKeys(waited=["s"]))
    out = capsys.readouterr().out
    assert "Press 'c'" in out and "'s' to stop" in out


def test_only_c_and_s_are_asked_for(client):
    keys = ScriptedKeys(waited=["s"])
    run_picks(client, [make_target(), make_target()], keys)
    assert keys.asked == ["cs"]


def test_there_is_no_prompt_after_the_last_target(client, capsys):
    keys = ScriptedKeys()
    assert run_picks(client, [make_target()], keys) == 1
    assert keys.asked == []
    assert "last detection" in capsys.readouterr().out


def test_a_skipped_target_still_asks_before_the_next(client, rapid):
    """Skipping is not a go-ahead: the operator decides whether the next one is wanted."""
    targets = [make_target(base=(100.0, 0.0, 120.0)), make_target(base=(350.0, 0.0, 100.0))]
    keys = ScriptedKeys(waited=["c"])
    assert run_picks(client, targets, keys) == 1
    assert keys.asked == ["cs"]


def test_every_target_gets_a_turn_and_the_count_is_of_placed_objects(client):
    targets = [make_target(base=(300.0, 0.0, 100.0)),
               make_target(base=(300.0, 0.0, 900.0)),      # refused by the guard
               make_target(base=(350.0, 0.0, 100.0))]
    assert run_picks(client, targets, ScriptedKeys(waited=["c", "c"])) == 2


def test_nothing_in_the_list_sends_nothing(client, rapid):
    assert run_picks(client, [], ScriptedKeys()) == 0
    assert rapid.received == []
