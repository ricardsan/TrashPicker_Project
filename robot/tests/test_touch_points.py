"""robot/touch_points.py: the TCP over, down to and back up from each point, then home."""
from __future__ import annotations

import pytest

from robot import touch_points
from robot.pick_sequence import APPROACH_Z_MM

HOME = "-1,0,0,0"


def test_each_point_is_visited_down_to_z_and_the_robot_ends_at_home(client, rapid):
    waited = []
    results = touch_points.touch(client, [(500.0, 40.0), (500.0, -121.0)], 0.0, wait=waited.append)
    over = f"{APPROACH_Z_MM:.1f}"
    assert rapid.received == [HOME,
                              f"0,500.0,40.0,{over}", "0,500.0,40.0,0.0", f"0,500.0,40.0,{over}",
                              f"0,500.0,-121.0,{over}", "0,500.0,-121.0,0.0", f"0,500.0,-121.0,{over}",
                              HOME]
    assert len(waited) == 2
    assert all("reached" in line for line in results)


def test_a_point_outside_the_pick_reach_is_still_tried_but_flagged(client, rapid):
    results = touch_points.touch(client, [(900.0, 0.0)], 0.0, wait=lambda prompt: None)
    assert "outside the 300..800 mm" in results[0]


@pytest.mark.parametrize("argv, message", [
    (["--points", "500,0", "--host", "192.168.125.1"], "only for the simulator"),
    (["--points", "500,0", "--z", "-5"], "from 0"),
])
def test_unsafe_heights_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit):
        touch_points.main(argv)
    assert message in capsys.readouterr().err
