"""calibration.txt and how it is applied. A mistake here is a robot move, not a wrong number."""
from __future__ import annotations

import numpy as np
import pytest

from robot.extrinsics import (DEFAULT_CALIBRATION, HARDCODED_T_BASE_CAM, MAX_TRANSLATION_M,
                              camera_to_base_mm, check_transform, is_synthetic, load_T_base_cam,
                              parse_calibration_text, rays_at_height_mm)

EXPECTED = np.array([[-0.0166, -0.9399, 0.3412, -0.1627],
                     [-0.9995, 0.0252, 0.0208, 0.3009],
                     [-0.0281, -0.3407, -0.9398, 1.3917],
                     [0.0, 0.0, 0.0, 1.0]])


def as_text(T: np.ndarray) -> str:
    rows = "\n ".join("[" + " ".join(str(value) for value in row) + "]" for row in T)
    return f"[{rows}]"


def test_the_real_file_parses_to_the_matrix_it_prints():
    assert np.array_equal(load_T_base_cam(DEFAULT_CALIBRATION), EXPECTED)


def test_the_camera_origin_lands_where_the_file_says_the_camera_is():
    """calibration.txt's own prose: X = -163 mm, Y = 301 mm, Z = 1392 mm."""
    assert camera_to_base_mm(EXPECTED, (0.0, 0.0, 0.0)) == pytest.approx((-162.7, 300.9, 1391.7))


def test_the_optical_axis_points_down_and_towards_plus_x():
    """Also from the prose: mostly downward, tilted about 20 degrees from vertical, toward +X."""
    origin = np.array(camera_to_base_mm(EXPECTED, (0.0, 0.0, 0.0)))
    along_axis = np.array(camera_to_base_mm(EXPECTED, (0.0, 0.0, 1.0)))
    direction = (along_axis - origin) / 1000.0
    assert direction == pytest.approx([0.3412, 0.0208, -0.9398])
    assert np.degrees(np.arccos(-direction[2])) == pytest.approx(20.0, abs=0.1)


def test_the_translation_is_added_once_in_the_right_unit():
    """Metres in, millimetres out. Converting first would leave the translation 1000x too small."""
    point = np.array([0.1, -0.2, 0.9])
    by_hand = (EXPECTED[:3, :3] @ point + EXPECTED[:3, 3]) * 1000.0
    assert camera_to_base_mm(EXPECTED, point) == pytest.approx(tuple(by_hand))

    wrong_order = (EXPECTED @ np.append(point * 1000.0, 1.0))[:3]
    assert np.linalg.norm(np.array(camera_to_base_mm(EXPECTED, point)) - wrong_order) > 1000.0


def test_it_applies_T_and_not_its_inverse():
    point = (0.1, -0.2, 0.9)
    inverse_applied = (np.linalg.inv(EXPECTED) @ np.append(point, 1.0))[:3] * 1000.0
    assert np.linalg.norm(np.array(camera_to_base_mm(EXPECTED, point)) - inverse_applied) > 100.0


def test_the_inverse_round_trips_a_point():
    """The rounding of calibration.txt to four decimals costs well under a millimetre here."""
    point = np.array([0.1, -0.2, 0.9])
    base_m = np.array(camera_to_base_mm(EXPECTED, point)) / 1000.0
    back = (np.linalg.inv(EXPECTED) @ np.append(base_m, 1.0))[:3]
    assert back == pytest.approx(point)


@pytest.mark.parametrize("ray", [(0.0, 0.0, 1.0), (0.3, -0.2, 1.0), (-0.4, 0.3, 1.0)])
@pytest.mark.parametrize("z", [70.0, 250.0])
def test_a_ray_lands_on_the_plane_it_was_sent_to(ray, z):
    (x_mm, y_mm, z_mm), = rays_at_height_mm(EXPECTED, [ray], z)
    assert z_mm == pytest.approx(z)
    # And it is on that ray: back in the camera's frame, it is a multiple of it.
    back = (np.linalg.inv(EXPECTED) @ np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, 1.0]))[:3]
    assert np.cross(back, ray) == pytest.approx(np.zeros(3), abs=1e-9)
    assert back @ np.array(ray) > 0


def test_many_rays_land_where_camera_to_base_mm_puts_them():
    """The same answer as the one function every position goes through, in metres then millimetres."""
    rays = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 1.0], [-0.4, 0.3, 1.0]])
    landed = rays_at_height_mm(EXPECTED, rays, 70.0)
    for ray, point in zip(rays, landed):
        distance = (0.070 - EXPECTED[2, 3]) / (EXPECTED[2, :3] @ ray)
        assert tuple(point) == pytest.approx(camera_to_base_mm(EXPECTED, distance * ray))


def test_a_ray_does_not_depend_on_its_length():
    assert rays_at_height_mm(EXPECTED, [(0.5, 0.1, 2.0)], 70.0) == pytest.approx(
        rays_at_height_mm(EXPECTED, [(0.25, 0.05, 1.0)], 70.0))


def test_a_ray_pointing_away_from_the_plane_does_not_reach_it():
    """The optical axis looks down, so going backwards along it never gets to the table."""
    assert rays_at_height_mm(EXPECTED, [(0.0, 0.0, -1.0)], 70.0) is None


def test_a_ray_along_the_plane_does_not_reach_it():
    along = np.cross(EXPECTED[2, :3], (1.0, 0.0, 0.0))
    assert rays_at_height_mm(EXPECTED, [along], 70.0) is None


def test_a_plane_above_the_camera_is_not_reached_looking_down():
    assert rays_at_height_mm(EXPECTED, [(0.0, 0.0, 1.0)], 2000.0) is None


def test_one_ray_that_misses_is_enough_for_no_answer():
    """Half a mask on a plane says nothing reliable about its direction."""
    assert rays_at_height_mm(EXPECTED, [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)], 70.0) is None


def test_a_npy_from_the_calibration_script_loads(tmp_path):
    path = tmp_path / "T_base_cam.npy"
    np.save(path, EXPECTED)
    assert np.array_equal(load_T_base_cam(path), EXPECTED)


def test_commas_and_exponents_parse():
    text = ("[[1e0, 0, 0, 1.5e-1]\n [0, 1.0, 0, 0.0]\n [0, 0, 1., -.25]\n [0, 0, 0, 1]]")
    assert parse_calibration_text(text)[2, 3] == -0.25


def test_prose_around_the_matrix_is_ignored():
    text = "Camera position: X = −163 mm, 20°.\n" + as_text(EXPECTED) + "\nIn base coordinates."
    assert np.array_equal(parse_calibration_text(text), EXPECTED)


@pytest.mark.parametrize("text, found", [("no matrix here", 0), ("[1 0 0 0]\n[0 1 0 0]", 2)])
def test_a_file_without_four_rows_is_refused(text, found):
    with pytest.raises(SystemExit, match=f"found {found}"):
        parse_calibration_text(text)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no calibration file"):
        load_T_base_cam(tmp_path / "nothing.txt")


def test_a_matrix_in_millimetres_is_refused():
    in_mm = EXPECTED.copy()
    in_mm[:3, 3] *= 1000.0
    with pytest.raises(SystemExit, match="metres"):
        check_transform(in_mm)
    assert np.abs(EXPECTED[:3, 3]).max() < MAX_TRANSLATION_M


def test_a_rotation_that_is_not_orthonormal_is_refused():
    skewed = EXPECTED.copy()
    skewed[0, 0] += 0.2
    with pytest.raises(SystemExit, match="orthonormal"):
        check_transform(skewed)


def test_a_reflection_is_refused():
    mirrored = EXPECTED.copy()
    mirrored[:3, 0] *= -1.0
    with pytest.raises(SystemExit, match="reflection"):
        check_transform(mirrored)


def test_a_bad_last_row_is_refused():
    bad = EXPECTED.copy()
    bad[3, 0] = 1.0
    with pytest.raises(SystemExit, match="last row"):
        check_transform(bad)


def test_a_wrong_shape_is_refused():
    with pytest.raises(SystemExit, match="4x4"):
        check_transform(np.eye(3))


def test_a_value_that_is_not_a_number_is_refused(tmp_path):
    """Only reachable from a .npy, and it would otherwise slip past every comparison below it."""
    bad = EXPECTED.copy()
    bad[0, 3] = np.nan
    path = tmp_path / "T.npy"
    np.save(path, bad)
    with pytest.raises(SystemExit, match="not a number"):
        load_T_base_cam(path)


@pytest.mark.parametrize("edit, message", [
    (lambda T: T[:3, 3].__imul__(1000.0), "metres"),
    (lambda T: T[:3, 0].__imul__(-1.0), "reflection"),
], ids=["millimetres", "reflection"])
@pytest.mark.parametrize("suffix", [".txt", ".npy"])
def test_loading_a_file_checks_what_it_holds(tmp_path, edit, message, suffix):
    """Through load_T_base_cam, because that is the only path the robot's positions come from."""
    bad = EXPECTED.copy()
    edit(bad)
    path = tmp_path / f"T{suffix}"
    if suffix == ".npy":
        np.save(path, bad)
    else:
        path.write_text(as_text(bad))
    with pytest.raises(SystemExit, match=message):
        load_T_base_cam(path)


# =============================================================================
# the hardcoded extrinsics
# =============================================================================
def test_the_hardcoded_camera_is_at_x_y_minus_390_at_the_measured_height():
    assert camera_to_base_mm(HARDCODED_T_BASE_CAM, (0.0, 0.0, 0.0)) == pytest.approx(
        (-390.0, -390.0, 789.8))


def test_the_hardcoded_extrinsics_are_a_valid_transform():
    check_transform(HARDCODED_T_BASE_CAM)
    R = HARDCODED_T_BASE_CAM[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


def test_the_hardcoded_tilt_is_the_one_measured_in_clutter_med():
    R = HARDCODED_T_BASE_CAM[:3, :3]
    assert R[2] == pytest.approx([-0.0250, -0.2314, -0.9725], abs=2e-4)
    assert np.degrees(np.arccos(-R[2, 2])) == pytest.approx(13.46, abs=0.02)


def test_the_hardcoded_heading_is_the_calibrated_one():
    """The rotation from calibration.txt's has a horizontal axis, so no spin about the vertical."""
    relative = HARDCODED_T_BASE_CAM[:3, :3] @ EXPECTED[:3, :3].T
    assert abs(relative[1, 0] - relative[0, 1]) < 0.002


def test_the_surface_is_at_z_0_and_up_is_towards_the_camera():
    R, t = HARDCODED_T_BASE_CAM[:3, :3], HARDCODED_T_BASE_CAM[:3, 3]
    on_surface = 0.7898 / -R[2, 2]      # along the optical axis
    assert camera_to_base_mm(HARDCODED_T_BASE_CAM, (0.0, 0.0, on_surface))[2] == pytest.approx(0.0, abs=1e-6)
    assert camera_to_base_mm(HARDCODED_T_BASE_CAM, (0.0, 0.0, on_surface - 0.1))[2] > 0.0
    assert t[2] == pytest.approx(0.7898)


def test_the_real_calibration_is_not_synthetic():
    assert is_synthetic(DEFAULT_CALIBRATION) is False


def test_the_demo_calibration_is_synthetic():
    assert is_synthetic(DEFAULT_CALIBRATION.parent / "calibration_rosbag_demo.txt") is True


def test_a_npy_is_not_read_as_text(tmp_path):
    path = tmp_path / "T.npy"
    np.save(path, EXPECTED)
    assert is_synthetic(path) is False


def test_the_rosbag_demo_file_is_the_real_rotation_levelled_and_says_what_it_is():
    """6.5 degrees is the gap between the clips' tilt and the calibration's, and it is about a
    horizontal axis, because the heading is the one thing a clip cannot speak to."""
    demo_path = DEFAULT_CALIBRATION.parent / "calibration_rosbag_demo.txt"
    demo = load_T_base_cam(demo_path)
    relative = demo[:3, :3] @ EXPECTED[:3, :3].T
    assert np.degrees(np.arccos((np.trace(relative) - 1.0) / 2.0)) == pytest.approx(6.5, abs=0.2)
    assert abs(relative[1, 0] - relative[0, 1]) < 0.002     # the rotation's axis has no z to speak of
    assert "NEVER USE THIS ON THE REAL ROBOT" in demo_path.read_text(encoding="utf-8")
