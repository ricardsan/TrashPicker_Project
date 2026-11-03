"""The socket client, against the fake controller."""
from __future__ import annotations

import socket

import pytest

from robot.art_camera_client import (MOCK_STEPS, ArtCameraClient, MoveRejected, RobotError,
                                     format_move, run_mock)
from robot.mock.fake_rapid import DROP, HANG, FakeRapid, parse_message


def connect(server: FakeRapid, **kwargs) -> ArtCameraClient:
    client = ArtCameraClient("127.0.0.1", server.port, **{"reply_timeout_s": 2.0, **kwargs})
    client.connect()
    return client


def test_a_move_is_four_numbers_to_a_tenth_of_a_millimetre():
    assert format_move(2, 400.04, -120.26, 70.0) == "2,400.0,-120.3,70.0"


def test_rz_is_a_fifth_field_only_when_there_is_one():
    assert format_move(2, 400.0, 0.0, 170.0, -45.04) == "2,400.0,0.0,170.0,-45.0"
    assert format_move(2, 400.0, 0.0, 170.0, 0.0) == "2,400.0,0.0,170.0,0.0"
    assert format_move(2, 400.0, 0.0, 170.0, None) == "2,400.0,0.0,170.0"


def test_a_move_with_rz_reaches_the_controller(client, rapid):
    client.move_to(1, 400.0, 10.0, 200.0, 35.0)
    assert rapid.received == ["1,400.0,10.0,200.0,35.0"]
    assert rapid.pose == (400.0, 10.0, 200.0)
    assert rapid.rz == 35.0


def test_a_move_with_no_rz_has_homes_orientation(client, rapid):
    client.move_to(1, 400.0, 10.0, 200.0, 35.0)
    client.move_to(1, 400.0, 10.0, 200.0)
    assert rapid.rz == 0.0


@pytest.mark.parametrize("rz", [180.0, -180.0])
def test_the_ends_of_the_rz_range_are_allowed(client, rapid, rz):
    client.move_to(0, 400.0, 0.0, 200.0, rz)
    assert rapid.rz == rz


@pytest.mark.parametrize("bad", [180.1, -180.1, float("nan"), float("inf"), float("-inf")])
def test_an_rz_that_could_not_be_sent_is_not_sent(client, rapid, bad):
    with pytest.raises(ValueError, match="rotation"):
        client.move_to(0, 400.0, 0.0, 200.0, bad)
    assert rapid.received == []


@pytest.mark.parametrize("message", ["0,400,0,170,181", "0,400,0,170,-181"])
def test_the_controller_rejects_an_rz_beyond_its_limit_and_nothing_moves(client, rapid, message):
    assert client.exchange(message) == "REJECTED"
    assert rapid.pose is None


def test_the_rz_limit_does_not_apply_to_a_command(client, rapid):
    """As in main: the check is on nClass >= 0, and commands carry whatever they are sent."""
    assert client.exchange("-1,0,0,0,999") == "DONE"
    assert rapid.pose == rapid.limits["pHome"]


def test_going_home_undoes_the_turn(client, rapid):
    client.move_to(1, 400.0, 10.0, 200.0, 35.0)
    client.home()
    assert rapid.rz == 0.0


@pytest.mark.parametrize("message, parsed", [
    ("1,2,3,4", (1.0, 2.0, 3.0, 4.0, 0.0)),
    ("1,2,3,4,5", (1.0, 2.0, 3.0, 4.0, 5.0)),
    ("-1, 1000,2000,3000", (-1.0, 1000.0, 2000.0, 3000.0, 0.0)),
    ("1,2,3,4,", None),
    ("1,2,3,4,5,6", None),
    ("1,2,3,4,abc", None),
    ("1,2,3", None),
])
def test_the_fake_parses_as_parsemessage_does(message, parsed):
    assert parse_message(message) == parsed


def test_commands_carry_the_negative_class_and_an_ignored_position(client, rapid):
    client.home()
    client.grip_close()
    client.grip_open()
    assert rapid.received == ["-1,0,0,0", "-3,0,0,0", "-2,0,0,0"]
    assert rapid.gripper == "open"
    assert rapid.pose == rapid.limits["pHome"]


def test_a_valid_move_reaches_the_controller(client, rapid):
    client.move_to(1, 400.0, 10.0, 200.0)
    assert rapid.received == ["1,400.0,10.0,200.0"]
    assert rapid.pose == (400.0, 10.0, 200.0)


@pytest.mark.parametrize("x, y", [(100.0, 0.0), (700.0, 0.0), (400.0, 400.0), (400.0, -400.0)])
def test_outside_the_workspace_is_rejected_and_nothing_moves(client, rapid, x, y):
    with pytest.raises(MoveRejected, match="rejected"):
        client.move_to(0, x, y, 200.0)
    assert rapid.pose is None


def test_a_negative_class_can_never_be_sent_as_a_move(client, rapid):
    """-1 would turn a move into a GoHome, with the rest of the sequence carrying on regardless."""
    with pytest.raises(ValueError, match="command"):
        client.move_to(-1, 400.0, 0.0, 200.0)
    assert rapid.received == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_position_that_is_not_a_number_is_not_sent(client, rapid, bad):
    with pytest.raises(ValueError, match="not a position"):
        client.move_to(0, 400.0, 0.0, bad)
    assert rapid.received == []


@pytest.mark.parametrize("call", ["home", "grip_open", "grip_close"])
def test_a_command_that_does_not_come_back_done_is_an_error(call):
    with FakeRapid(script={0: "PARSE_ERROR"}) as server:
        client = connect(server)
        with pytest.raises(RobotError, match="PARSE_ERROR"):
            getattr(client, call)()
        client.close()


def test_a_parse_error_on_a_move_is_an_error_not_a_rejection():
    with FakeRapid(script={0: "PARSE_ERROR"}) as server:
        client = connect(server)
        with pytest.raises(RobotError, match="PARSE_ERROR"):
            client.move_to(0, 400.0, 0.0, 200.0)
        client.close()


def test_a_reply_it_does_not_know_is_an_error():
    with FakeRapid(script={0: "WHAT"}) as server:
        client = connect(server)
        with pytest.raises(RobotError, match="WHAT"):
            client.move_to(0, 400.0, 0.0, 200.0)
        client.close()


def test_the_connection_closing_is_an_error_that_says_so():
    with FakeRapid(script={0: DROP}) as server:
        client = connect(server)
        with pytest.raises(RobotError, match="closed the connection"):
            client.home()
        client.close()


def test_no_reply_is_a_timeout_error_and_not_a_hang():
    with FakeRapid(script={0: HANG}) as server:
        client = connect(server, reply_timeout_s=0.2)
        with pytest.raises(RobotError, match="no reply"):
            client.home()
        client.close()


def test_nothing_listening_is_an_error_that_says_what_to_check():
    # Bound and never listened on, so this port is known to refuse.
    with socket.socket() as placeholder:
        placeholder.bind(("127.0.0.1", 0))
        port = placeholder.getsockname()[1]
        client = ArtCameraClient("127.0.0.1", port, connect_timeout_s=1.0)
        with pytest.raises(RobotError, match="RAPID program"):
            client.connect()


def test_sending_before_connecting_is_an_error():
    with pytest.raises(RobotError, match="not connected"):
        ArtCameraClient().home()


def test_closing_twice_is_harmless(rapid):
    client = connect(rapid)
    client.close()
    client.close()


# =============================================================================
# the mock run
# =============================================================================
def test_the_mock_run_passes_against_the_controller_logic(client, capsys):
    assert run_mock(client) is True
    assert f"{len(MOCK_STEPS)} of {len(MOCK_STEPS)}" in capsys.readouterr().out


def test_the_mock_run_leaves_the_robot_at_home_with_the_gripper_open(client, rapid):
    run_mock(client)
    assert rapid.pose == rapid.limits["pHome"]
    assert rapid.gripper == "open"


def test_the_mock_run_starts_from_home(client, rapid):
    run_mock(client)
    assert parse_message(rapid.received[0])[0] == -1


def test_the_mock_run_stops_at_a_reply_that_differs(capsys):
    """These steps move a robot, and this one has just said something unexpected."""
    with FakeRapid(script={2: "REJECTED"}) as server:
        client = connect(server)
        assert run_mock(client) is False
        assert len(server.received) == 3
        client.close()
    assert f"stopped at step 3 of {len(MOCK_STEPS)}" in capsys.readouterr().out


# =============================================================================
# staying responsive during a long move
# =============================================================================
def test_on_wait_is_called_while_a_reply_is_awaited():
    """What keeps the image window from going Not Responding during a move."""
    calls = []
    with FakeRapid(script={0: HANG}) as server:
        client = connect(server, reply_timeout_s=0.35, on_wait=lambda: calls.append(1))
        with pytest.raises(RobotError, match="no reply"):
            client.home()
        client.close()
    assert len(calls) >= 2


def test_a_reply_after_several_waits_still_comes_through():
    calls = []
    with FakeRapid(script={0: HANG}) as server:
        client = connect(server, on_wait=lambda: calls.append(1))
        # The reply to a hung message never comes, so answer it by hand after a few waits.
        original = client.on_wait

        def answer_late():
            original()
            if len(calls) == 3:
                server.send_late("DONE")

        client.on_wait = answer_late
        client.home()
        client.close()
    assert len(calls) >= 3


def test_every_message_shape_from_yolo_robot_test_is_kept():
    """The point of the mock run is that it is that test's coverage: each reply path, and each
    sentinel command with a wild position."""
    expected = {step.expected for step in MOCK_STEPS}
    assert expected == {"DONE", "REJECTED", "PARSE_ERROR"}
    assert {"-1, 1000,2000,3000", "-2, 1000,2000,3000", "-3, 1000,2000,3000",
            "abc,def", "1,300"} <= {step.message for step in MOCK_STEPS}


def test_the_mock_run_turns_the_tool_and_covers_every_way_rz_can_fail(client, rapid):
    run_mock(client)
    assert any(parse_message(step.message) is not None and parse_message(step.message)[4] != 0.0
               and step.expected == "DONE" for step in MOCK_STEPS)
    assert {"0,400,0,170,181", "0,400,0,170,abc", "0,400,0,170,90,1",
            "0,400,0,170,"} <= set(rapid.received)
