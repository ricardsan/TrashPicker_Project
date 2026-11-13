"""robot/scene_cloud.py: depth -> base-frame cloud, and the PLY / PTS files."""
from __future__ import annotations

import numpy as np
import pytest

from robot.scene_cloud import cloud_in_base, cloud_in_camera, write_ply, write_pts
from synthetic import Intrinsics

INTR = Intrinsics(width=64, height=48, fx=60.0, fy=61.0, ppx=31.5, ppy=23.5)


def looking_down(height_m=0.8, at=(0.4, 0.1)) -> np.ndarray:
    """A camera straight above (at, height_m): its z is base -z, its x base +x, its y base -y."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [at[0], at[1], height_m]
    return T


def test_the_principal_point_deprojects_along_the_optical_axis():
    metres = np.zeros((48, 64))
    metres[24, 32] = 1.0
    intr = Intrinsics(width=64, height=48, fx=60.0, fy=61.0, ppx=32.0, ppy=24.0)
    points, pixels = cloud_in_camera(metres, intr, stride=1)
    assert points.tolist() == [[0.0, 0.0, 1.0]]
    assert pixels.tolist() == [[24, 32]]


def test_a_flat_floor_seen_from_above_comes_out_level_in_the_base_frame():
    metres = np.full((48, 64), 0.8)
    image = np.zeros((48, 64, 3), np.uint8)
    points, colours = cloud_in_base(image, metres, INTR, looking_down(0.8), stride=4)
    assert len(points) == 12 * 16
    assert np.allclose(points[:, 2], 0.0, atol=1e-9)                  # the floor is at base z = 0
    centre = points[np.argmin(np.hypot(points[:, 0] - 400, points[:, 1] - 100))]
    assert np.hypot(centre[0] - 400, centre[1] - 100) < 15            # optical axis meets it at (400, 100)


def test_colours_are_rgb_and_invalid_depth_is_left_out():
    metres = np.full((48, 64), 0.8)
    metres[:, :32] = 0.0                                               # no depth on the left half
    image = np.zeros((48, 64, 3), np.uint8)
    image[...] = (10, 20, 30)                                          # BGR
    points, colours = cloud_in_base(image, metres, INTR, looking_down(), stride=1)
    assert len(points) == 48 * 32
    assert colours[0].tolist() == [30, 20, 10]


def test_distortion_and_misaligned_frames_are_refused():
    class Distorted(Intrinsics):
        coeffs = [0.1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="distortion"):
        cloud_in_camera(np.ones((48, 64)), Distorted())
    with pytest.raises(ValueError, match="not aligned"):
        cloud_in_base(np.zeros((10, 10, 3), np.uint8), np.ones((48, 64)), INTR, np.eye(4))


def test_ply_and_pts_hold_the_points_in_metres(tmp_path):
    points = np.array([[1000.0, -500.0, 70.0], [0.0, 0.0, 0.0]])
    colours = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    ply = write_ply(tmp_path / "c.ply", points, colours).read_bytes()
    header, body = ply.split(b"end_header\n")
    assert b"element vertex 2" in header
    records = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("r", "u1"), ("g", "u1"), ("b", "u1")])
    assert records["x"].tolist() == pytest.approx([1.0, 0.0])
    assert records["z"][0] == pytest.approx(0.07)
    assert records["r"].tolist() == [255, 0]

    lines = write_pts(tmp_path / "c.pts", points, colours).read_text().splitlines()
    assert lines[0] == "2"
    assert lines[1].split() == ["1.00000", "-0.50000", "0.07000", "0", "255", "0", "0"]
