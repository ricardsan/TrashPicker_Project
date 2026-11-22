"""robot/frame_selection.py: the part of the image the arm cannot reach, shaded in the preview."""
from __future__ import annotations

import numpy as np

from robot.frame_selection import shade_unreachable, unreachable_mask
from robot.pick_sequence import REACH_MAX_MM, REACH_MIN_MM
from synthetic import Intrinsics

INTR = Intrinsics(width=80, height=60, fx=40.0, fy=40.0, ppx=39.5, ppy=29.5)


def looking_down(at_mm, height_mm=800.0) -> np.ndarray:
    """Straight down: camera x is base +x, camera y base -y."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = np.array([at_mm[0], at_mm[1], height_mm]) / 1000.0
    return T


def test_the_middle_of_the_reach_is_all_reachable():
    middle = (REACH_MIN_MM + REACH_MAX_MM) / 2
    mask = unreachable_mask(Intrinsics(width=8, height=6, fx=4000.0, fy=4000.0, ppx=3.5, ppy=2.5),
                            looking_down((middle, 0.0)))
    assert not mask.any()                                   # a narrow view, all inside the ring


def test_the_base_and_the_far_side_are_shaded():
    mask = unreachable_mask(INTR, looking_down((REACH_MAX_MM - 100.0, 0.0)))
    # the floor at x = 700 +- 800 mm: the image's left edge is near the base, its right edge far out
    assert mask[30, 0] and mask[30, 79]
    assert not mask[30, 40] and not mask[30, 30]


def test_a_camera_looking_up_sees_no_floor():
    T = looking_down((500.0, 0.0))
    T[:3, :3] = np.eye(3)                                   # z up: no ray meets the floor
    assert unreachable_mask(INTR, T).all()


def test_shading_tints_only_the_masked_part_red():
    image = np.full((60, 80, 3), 100, np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[:, :40] = True
    out = shade_unreachable(image, mask)
    assert out[30, 10, 2] > 150 and out[30, 10, 0] < 100   # red, BGR
    assert out[30, 70].tolist() == [100, 100, 100]
    assert shade_unreachable(image, np.zeros_like(mask)) is image
