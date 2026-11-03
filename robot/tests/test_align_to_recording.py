"""Levelling a calibration to a recorded sheet. The scenes here are built with a known answer."""
from __future__ import annotations

import numpy as np
import pytest

from camera.segmentation_realsense import depth_map_metres
from robot import align_to_recording as align
from robot.art_camera_client import X_MAX_MM, X_MIN_MM, Y_MAX_MM, Y_MIN_MM, Z_TABLE_MM
from robot.extrinsics import (HARDCODED_T_BASE_CAM, camera_to_base_mm, check_transform, is_synthetic,
                              load_T_base_cam)
from robot.frame_selection import EXPORT_DIR, open_recording, started
from synthetic import Intrinsics, pixel_grid

T = load_T_base_cam()
INTR = Intrinsics()
UP = np.array([0.03, -0.25, -0.97]) / np.linalg.norm([0.03, -0.25, -0.97])
HEIGHT_M = 0.8
Z_TABLE_M = Z_TABLE_MM / 1000.0
CENTRE_MM = (450.0, 100.0)
CLIPS = ("20261016_231841_clutter_med", "20261016_232523_clutter_high2", "20261016_233733_stacked")


def rays() -> np.ndarray:
    u, v = pixel_grid(INTR)
    return np.stack([(u - INTR.ppx) / INTR.fx, (v - INTR.ppy) / INTR.fy, np.ones_like(u)], axis=-1)


def plane_points(up=UP, height=HEIGHT_M) -> np.ndarray:
    """Every pixel of a camera looking at the plane up @ p = -height."""
    directions = rays()
    return (directions * (-height / (directions @ up))[..., None]).reshape(-1, 3)


def to_base_mm(T_base_cam: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (points @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]) * 1000.0


def levelled() -> np.ndarray:
    return align.level_transform(T, UP, HEIGHT_M, Z_TABLE_M, CENTRE_MM)


def rotation_angle_deg(relative: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


# =============================================================================
# fitting the sheet
# =============================================================================
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_plane_is_found_past_noise_and_the_objects_standing_on_it(seed):
    rng = np.random.default_rng(seed)
    points = plane_points()
    points = points + rng.normal(0.0, 0.003, points.shape)
    standing = rng.random(len(points)) < 0.15
    points[standing] = points[standing] + UP * rng.uniform(0.04, 0.3, (standing.sum(), 1))
    sample = points[rng.choice(len(points), 8000, replace=False)]

    plane = align.fit_plane(sample, rng)
    assert align.angle_between_deg(plane.up, UP) < 0.3
    assert plane.height_m == pytest.approx(HEIGHT_M, abs=0.002)
    assert plane.on_plane == pytest.approx(0.85, abs=0.03)
    assert plane.rms_m == pytest.approx(0.003, abs=0.001)


def test_up_points_at_the_camera_whichever_way_the_points_come():
    """The sign of a fitted normal is arbitrary, and the wrong one makes the sheet's height negative."""
    points = plane_points()[::40]
    for seed in range(6):
        plane = align.fit_plane(points, np.random.default_rng(seed))
        assert plane.up[2] < 0
        assert plane.height_m > 0


@pytest.mark.parametrize("points", [np.zeros((2, 3)), np.zeros((50, 3)),
                                    np.outer(np.arange(50.0), [1.0, 2.0, 3.0])],
                         ids=["too few", "one point", "one line"])
def test_points_with_no_plane_are_refused(points):
    with pytest.raises(SystemExit, match="plane"):
        align.fit_plane(points, np.random.default_rng(0))


# =============================================================================
# levelling
# =============================================================================
def test_the_sheet_reads_level_at_the_tables_z_at_every_pixel():
    """The test that matters: a tilted view of a flat sheet, as one height everywhere."""
    z = to_base_mm(levelled(), plane_points())[:, 2]
    assert z == pytest.approx(np.full_like(z, Z_TABLE_MM), abs=1e-6)


def test_through_the_real_calibration_it_does_not():
    """So the test above has something to catch: this scene is tilted and far too high."""
    z = to_base_mm(T, plane_points())[:, 2]
    assert z.max() - z.min() > 50.0
    assert np.median(z) > 400.0


def test_it_is_a_true_rotation():
    level = levelled()
    assert np.allclose(level[:3, :3].T @ level[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(level[:3, :3]) == pytest.approx(1.0)
    check_transform(level)


def test_a_rounded_rotation_comes_out_true_as_well():
    """calibration.txt's four decimals are not quite a rotation, and this has to fix that, not keep it."""
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) > 1e-5
    assert abs(np.linalg.det(levelled()[:3, :3]) - 1.0) < 1e-12


def test_it_turns_by_exactly_the_tilt_between_the_two():
    R = align._nearest_rotation(T[:3, :3])
    gap = align.angle_between_deg(R @ UP, np.array([0.0, 0.0, 1.0]))
    assert rotation_angle_deg(levelled()[:3, :3] @ R.T) == pytest.approx(gap, abs=1e-6)


def test_the_heading_is_kept():
    """The smallest rotation has a horizontal axis, so it adds no spin about the vertical. A spin is
    what would carry the calibrated heading off with it, and a clip has no way to justify one."""
    relative = levelled()[:3, :3] @ align._nearest_rotation(T[:3, :3]).T
    assert relative[1, 0] - relative[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_the_optical_axis_meets_the_sheet_where_it_was_asked_to():
    level = levelled()
    assert camera_to_base_mm(level, (0.0, 0.0, HEIGHT_M / -UP[2])) == pytest.approx(
        (*CENTRE_MM, Z_TABLE_MM))


def test_up_in_the_base_is_towards_the_camera():
    """+z is up, as checked in RobotStudio. A point lifted off the sheet towards the camera reads
    higher by what it was lifted, and the camera itself is the highest thing in the scene."""
    level = levelled()
    on_sheet = plane_points()[1000]
    assert camera_to_base_mm(level, on_sheet + 0.1 * UP)[2] == pytest.approx(Z_TABLE_MM + 100.0)
    assert camera_to_base_mm(level, (0.0, 0.0, 0.0))[2] == pytest.approx(Z_TABLE_MM + HEIGHT_M * 1000.0)


def test_a_view_that_is_already_level_only_gets_its_height():
    R = align._nearest_rotation(T[:3, :3])
    already = R[2]      # the base's +z in the camera's frame
    level = align.level_transform(T, already, HEIGHT_M, Z_TABLE_M, CENTRE_MM)
    assert level[:3, :3] == pytest.approx(R)


def test_a_view_that_disagrees_by_more_than_a_tilt_is_refused():
    with pytest.raises(SystemExit, match="not a tilt"):
        align.level_transform(T, -align._nearest_rotation(T[:3, :3])[2], HEIGHT_M, Z_TABLE_M,
                              CENTRE_MM)


def test_a_normal_that_is_not_unit_length_gives_the_same_answer():
    assert align.level_transform(T, UP * 3.0, HEIGHT_M, Z_TABLE_M, CENTRE_MM) == pytest.approx(levelled())


# =============================================================================
# from a depth map to points
# =============================================================================
def test_the_points_are_the_pinhole_formula_in_the_right_order():
    metres = np.zeros((INTR.height, INTR.width))
    metres[100, 500] = 1.5      # row 100, column 500: not the same as (100, 500) the other way round
    intrinsics = Intrinsics()
    intrinsics.coeffs = [0.0] * 5
    (x, y, z), = align.sheet_points(metres, intrinsics)
    assert (x, y, z) == pytest.approx(((500 - INTR.ppx) / INTR.fx * 1.5,
                                       (100 - INTR.ppy) / INTR.fy * 1.5, 1.5))


def test_holes_are_not_points():
    metres = np.full((INTR.height, INTR.width), 1.0)
    metres[:10] = 0.0
    intrinsics = Intrinsics()
    intrinsics.coeffs = [0.0] * 5
    assert len(align.sheet_points(metres, intrinsics)) == (INTR.height - 10) * INTR.width


def test_a_lens_with_distortion_is_refused_rather_than_measured_wrongly():
    intrinsics = Intrinsics()
    intrinsics.coeffs = [0.1, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(SystemExit, match="distortion"):
        align.sheet_points(np.ones((INTR.height, INTR.width)), intrinsics)


# =============================================================================
# one transform for every clip
# =============================================================================
def plane(up=UP, height=HEIGHT_M) -> align.Plane:
    return align.Plane(np.asarray(up, dtype=np.float64), height, 0.9, 0.004)


def test_clips_that_agree_are_combined():
    up, height = align.combine([plane(height=0.799), plane(height=0.801)])
    assert up == pytest.approx(UP)
    assert height == pytest.approx(0.8)


def test_clips_at_different_heights_are_refused():
    with pytest.raises(SystemExit, match="do not agree"):
        align.combine([plane(height=0.78), plane(height=0.82)])


def test_clips_at_different_tilts_are_refused():
    other = UP + np.array([0.0, 0.08, 0.0])
    with pytest.raises(SystemExit, match="do not agree"):
        align.combine([plane(), plane(up=other / np.linalg.norm(other))])


# =============================================================================
# the file
# =============================================================================
def written(tmp_path, monkeypatch, planes) -> tuple:
    monkeypatch.setattr(align, "available_clips", lambda: ["a.db3"])
    monkeypatch.setattr(align, "measure_clip", lambda name, rng: planes)
    path = tmp_path / "demo.txt"
    return align.main(["--output", str(path)]), path


def test_the_file_says_it_is_synthetic_and_loads_back_as_what_was_computed(tmp_path, monkeypatch):
    code, path = written(tmp_path, monkeypatch, [plane()])
    assert code == 0
    assert is_synthetic(path) is True
    assert "NEVER USE THIS ON THE REAL ROBOT" in path.read_text(encoding="utf-8")
    assert load_T_base_cam(path) == pytest.approx(levelled(), abs=1e-6)


def test_clips_that_disagree_write_nothing(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="do not agree"):
        written(tmp_path, monkeypatch, [plane(height=0.78), plane(height=0.82)])
    assert not (tmp_path / "demo.txt").exists()


# =============================================================================
# the real clips, and the file that is checked in
# =============================================================================
real_clips = pytest.mark.skipif(
    not all((EXPORT_DIR / clip / "meta.json").exists() for clip in CLIPS),
    reason="needs the unpacked clips")


def settled_points(clip: str, frame: int = 60) -> np.ndarray:
    source = open_recording(f"{clip}.db3")
    with started(source):
        index = 0
        while index <= frame:
            grabbed = source.grab()
            if grabbed is None:
                continue
            index += 1
    _, depth, intrinsics = grabbed
    return align.sheet_points(depth_map_metres(depth), intrinsics)


@pytest.mark.slow
@real_clips
@pytest.mark.parametrize("clip", CLIPS)
def test_the_recorded_sheet_reads_level_through_the_demo_file(rs_or_stub, clip):
    """No plane fit in this one: most of the frame is sheet, so most of its pixels have to read as the
    table. Through the real calibration, none of them do."""
    points = settled_points(clip)
    demo = load_T_base_cam(align.DEFAULT_OUTPUT)
    z = to_base_mm(demo, points)[:, 2]
    assert np.median(z) == pytest.approx(Z_TABLE_MM, abs=5.0)
    assert (np.abs(z - Z_TABLE_MM) < 15.0).mean() > 0.70
    assert (np.abs(to_base_mm(T, points)[:, 2] - Z_TABLE_MM) < 15.0).mean() == 0.0


@pytest.mark.slow
@real_clips
def test_the_top_left_corner_of_clutter_med_is_at_z_0_through_the_hardcoded_extrinsics(rs_or_stub):
    """Where HARDCODED_T_BASE_CAM's height was read from. It moves by a few mm from frame to frame."""
    source = open_recording(f"{CLIPS[0]}.db3")
    corner_z, sheet = [], None
    with started(source):
        index = 0
        while index <= max(align.SETTLED_FRAMES):
            grabbed = source.grab()
            if grabbed is None:
                continue
            if index in align.SETTLED_FRAMES:
                _, depth, intrinsics = grabbed
                metres = depth_map_metres(depth)
                corner = align.sheet_points(metres[:21, :21], intrinsics)
                corner_z.append(np.median(to_base_mm(HARDCODED_T_BASE_CAM, corner)[:, 2]))
                sheet = to_base_mm(HARDCODED_T_BASE_CAM, align.sheet_points(metres, intrinsics))
            index += 1
    assert np.median(corner_z) == pytest.approx(0.0, abs=5.0)
    # And the corner is not a fluke: most of the frame is sheet, and most of it reads as level.
    assert np.median(sheet[:, 2]) == pytest.approx(0.0, abs=5.0)
    assert (np.abs(sheet[:, 2]) < 15.0).mean() > 0.70


@pytest.mark.slow
@real_clips
def test_the_image_centre_of_the_demo_file_is_inside_the_workspace(rs_or_stub):
    x, y, z = camera_to_base_mm(load_T_base_cam(align.DEFAULT_OUTPUT),
                                (0.0, 0.0, align.fit_plane(settled_points(CLIPS[0]),
                                                           np.random.default_rng(0)).height_m))
    assert X_MIN_MM < x < X_MAX_MM
    assert Y_MIN_MM < y < Y_MAX_MM


@pytest.mark.slow
@real_clips
def test_the_file_that_is_checked_in_is_what_the_script_writes(rs_or_stub, tmp_path):
    """So a change to the script, or to calibration.txt, cannot leave the demo file behind it."""
    path = tmp_path / "demo.txt"
    assert align.main(["--output", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == align.DEFAULT_OUTPUT.read_text(encoding="utf-8")
