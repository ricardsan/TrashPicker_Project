"""The client's copies of the RAPID constants, diffed against the .modx itself.

Python cannot read the robot's pose back and cannot ask for its limits, so these constants are
copies by necessity. This is what stops a copy going stale without anyone noticing.
"""
from __future__ import annotations

import re

import pytest

from robot import art_camera_client as acc
from robot.mock.fake_rapid import MODX, parse_message, read_modx
from robot.pick_sequence import (APPROACH_Z_MM, DROP_LOCATIONS_MM, GRASP_Z_MARGIN_MM,
                                 GRASP_Z_MAX_MM, GRASP_Z_MIN_MM, check_drop_locations)


@pytest.fixture(scope="module")
def modx():
    return read_modx()


@pytest.mark.parametrize("name, value", [
    ("PORT", acc.DEFAULT_PORT),
    ("X_MIN", acc.X_MIN_MM), ("X_MAX", acc.X_MAX_MM),
    ("Y_MIN", acc.Y_MIN_MM), ("Y_MAX", acc.Y_MAX_MM),
    ("Z_TABLE", acc.Z_TABLE_MM), ("Z_SAFE", acc.Z_SAFE_MM),
])
def test_constant_matches_the_modx(modx, name, value):
    assert modx[name] == value


def test_home_z_matches_phome(modx):
    assert modx["pHome"][2] == acc.HOME_Z_MM


def test_the_rz_limit_matches_the_modx(modx):
    assert modx["RZ_MAX"] == acc.RZ_MAX_DEG


def test_default_host_matches_the_modx(modx):
    assert modx["HOST"] == acc.DEFAULT_HOST


def test_the_approach_plane_is_homes_z():
    """"Without changing the z" has only one meaning: the robot has just come from home."""
    assert APPROACH_Z_MM == acc.HOME_Z_MM


def test_the_grasp_floor_is_above_the_table_by_a_margin():
    """Z_TABLE is the table's surface, so a TCP sent exactly there is touching it."""
    assert GRASP_Z_MARGIN_MM > 0
    assert GRASP_Z_MIN_MM == acc.Z_TABLE_MM + GRASP_Z_MARGIN_MM
    assert GRASP_Z_MAX_MM == APPROACH_Z_MM


# =============================================================================
# what the messages mean, on the RAPID side
# =============================================================================
@pytest.fixture(scope="module")
def source():
    """The .modx with comments removed, so a commented-out line cannot satisfy these."""
    text = MODX.read_text(encoding="utf-8", errors="replace")
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


@pytest.mark.parametrize("command, proc", [(acc.CMD_HOME, "GoHome"),
                                           (acc.CMD_GRIP_OPEN, "GripOpen"),
                                           (acc.CMD_GRIP_CLOSE, "GripClose")])
def test_each_command_number_calls_the_right_procedure(source, command, proc):
    """A swap here would open the gripper where this client means to close it."""
    assert re.search(rf"nClass\s*=\s*{command}\s+THEN\s+{proc}\s*;", source)


@pytest.mark.parametrize("reply", [acc.DONE, acc.REJECTED, acc.PARSE_ERROR])
def test_each_reply_is_one_the_controller_sends(source, reply):
    assert re.search(rf'SocketSend\s+clientSocket\s+\\Str:="{reply}"', source)


def test_a_move_only_moves(source):
    """The sequence is built from single straight moves. A GoHome inside MoveToPoint, which is
    commented out today, would send the arm home in the middle of every one of them."""
    body = re.search(r"PROC\s+MoveToPoint.*?ENDPROC", source, re.S).group(0)
    assert re.findall(r"\bMove[LJ]\b", body) == ["MoveL"]
    assert "GoHome" not in body


def test_the_workspace_check_only_applies_to_moves(source):
    """Why the commands can carry any position, and why this client's z guard is needed."""
    assert re.search(r"nClass\s*>=\s*0\s+AND\s+\(NOT\s+InWorkspace", source)
    body = re.search(r"FUNC\s+bool\s+InWorkspace.*?ENDFUNC", source, re.S).group(0)
    assert re.search(r"\bx\s*<", body) and re.search(r"\by\s*<", body)
    assert not re.search(r"\bz\s*[<>]", body), "RAPID checks z now: the client's guard can relax"


def test_rz_turns_the_tool_about_the_base_on_top_of_homes_orientation(source):
    """What gripper_rz_deg assumes. The rotation on the left applies in the base's frame, so a
    positive rz is anticlockwise seen from above, whatever pHome's orientation is."""
    body = re.search(r"PROC\s+MoveToPoint.*?ENDPROC", source, re.S).group(0)
    assert re.search(r"pTarget\.rot\s*:=\s*OrientZYX\(\s*nRz\s*,\s*0\s*,\s*0\s*\)\s*\*\s*pHome\.rot",
                     body)


def test_home_points_the_tool_straight_down(source):
    assert re.search(r"pHome\s*:=\s*\[\[[^\]]+\],\s*\[0,\s*0,\s*1,\s*0\]", source)


def test_no_fifth_field_means_no_turn(source):
    """Without this default, a move would inherit the rz of the one before it."""
    body = re.search(r"FUNC\s+bool\s+ParseMessage.*?ENDFUNC", source, re.S).group(0)
    assert re.search(r"nRz\s*:=\s*0\s*;\s*IF\s+i4\s*<\s*StrLen", body)


def test_the_rz_check_only_applies_to_moves(source):
    assert re.search(r"nClass\s*>=\s*0\s+AND\s+\(NOT\s+InWorkspace\([^)]*\)\s+OR\s+Abs\(nRz\)\s*>",
                     source)


def test_the_mock_z_stays_above_the_table():
    """yolo_robot_test.py moves to z = 70, which is the table itself on a real robot."""
    for step in acc.MOCK_STEPS:
        parsed = parse_message(step.message)
        if parsed is not None and parsed[0] >= 0 and step.expected == acc.DONE:
            assert parsed[3] >= acc.Z_TABLE_MM + acc.Z_SAFE_MM


def test_the_drop_locations_are_acceptable_as_shipped():
    """Inside the workspace, not equal to pDrop: these are meant to be edited by hand."""
    check_drop_locations()


def test_there_is_a_drop_location_for_every_class():
    assert set(DROP_LOCATIONS_MM) == {0, 1, 2, 3}


@pytest.mark.parametrize("location", [
    (100.0, 0.0, 250.0),        # x below X_MIN
    (800.0, 0.0, 250.0),        # x above X_MAX
    (400.0, 500.0, 250.0),      # y above Y_MAX
    (400.0, 0.0, 10.0),         # below the table
    (400.0, 0.0, 5000.0),       # the controller would accept this: it does not check z
    (400.0, 0.0, float("nan")),
    (float("nan"), 0.0, 250.0),
    (400.0, float("inf"), 250.0),
])
def test_a_bad_drop_location_stops_the_run_before_it_starts(location):
    with pytest.raises(SystemExit, match="DROP_LOCATIONS_MM"):
        check_drop_locations({0: location})
