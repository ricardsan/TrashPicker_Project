"""From a mask to a position. The geometry tests build a scene whose answer is known in advance."""
from __future__ import annotations

import types

import numpy as np
import pytest

from camera.segmentation_realsense import Detection
from robot.extrinsics import DEFAULT_CALIBRATION, load_T_base_cam
from robot.perception import (AXIS_SAMPLE_PIXELS, CENTROID_WINDOW, CENTROID_WINDOW_PX, MASK_MEDIAN,
                              NO_DEPTH, base_axis_deg, depth_at_centroid, draw_targets,
                              format_targets, measure_targets, sort_nearest_first)
from synthetic import (NAMES, FakeDepth, Intrinsics, detection, disc_mask, ellipse_mask, make_target,
                       pinhole_rs, pixel_grid)

INTR = Intrinsics()
T = load_T_base_cam()
RS = pinhole_rs()


def measure(detections, metres, T_base_cam=T):
    return measure_targets(detections, FakeDepth(metres), INTR, RS, NAMES, T_base_cam)


def flat_depth(value: float = 1.0) -> np.ndarray:
    return np.full((INTR.height, INTR.width), value, dtype=np.float64)


def plane_depth(z_base_m: float, T_base_cam=T) -> np.ndarray:
    """What the camera reads looking at the horizontal plane z = z_base_m in the base frame.

    A pixel's ray is r = ((u-ppx)/fx, (v-ppy)/fy, 1), the point on it at depth d is d*r in the
    camera's frame, and that lands on the plane when R[2] @ (d*r) + t[2] = z. So d comes from the
    third row alone, with none of the code under test involved.
    """
    u, v = pixel_grid(INTR)
    rays = np.stack([(u - INTR.ppx) / INTR.fx, (v - INTR.ppy) / INTR.fy, np.ones_like(u)], axis=-1)
    return (z_base_m - T_base_cam[2, 3]) / (rays @ T_base_cam[2, :3])


def pixel_of(base_mm, T_base_cam=T) -> tuple[float, float, float]:
    """A base-frame point seen by the camera: (u, v, depth). Goes through T's inverse."""
    cam = (np.linalg.inv(T_base_cam) @ np.append(np.asarray(base_mm) / 1000.0, 1.0))[:3]
    return (INTR.fx * cam[0] / cam[2] + INTR.ppx, INTR.fy * cam[1] / cam[2] + INTR.ppy, cam[2])


def circular_distance(a: float, b: float) -> float:
    """Between two line orientations, which wrap at 180."""
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def ellipse_lying_on_a_plane(base_mm, axis_deg: float, long_mm: float, short_mm: float,
                             T_base_cam=T) -> tuple[np.ndarray, np.ndarray]:
    """(mask, depth map) of an ellipse lying flat in the base frame, as the camera sees it.

    Every pixel's ray is followed to the plane and tested against the ellipse there, with none of the
    code under test involved, so the mask carries whatever a tilted camera does to its shape.
    """
    x0, y0, z0 = base_mm
    u, v = pixel_grid(INTR)
    rays = np.stack([(u - INTR.ppx) / INTR.fx, (v - INTR.ppy) / INTR.fy, np.ones_like(u)], axis=-1)
    depth = (z0 / 1000.0 - T_base_cam[2, 3]) / (rays @ T_base_cam[2, :3])
    base_m = (depth[..., None] * rays) @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    dx, dy = base_m[..., 0] * 1000.0 - x0, base_m[..., 1] * 1000.0 - y0
    theta = np.radians(axis_deg)
    along, across = dx * np.cos(theta) + dy * np.sin(theta), -dx * np.sin(theta) + dy * np.cos(theta)
    mask = (along / long_mm) ** 2 + (across / short_mm) ** 2 <= 1.0
    # An object the frame cuts off has a different centroid and axis, whatever the geometry does.
    assert mask.any() and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return mask, depth


# As near the workspace's corners as an object can be and still be whole in this 640 x 480 frame.
WORKSPACE_POINTS = [(250.0, -260.0), (650.0, -260.0), (650.0, 350.0), (250.0, 350.0), (450.0, 0.0)]


# =============================================================================
# the whole chain, against a scene with a known answer
# =============================================================================
@pytest.mark.parametrize("u, v", [(320, 240), (100, 80), (540, 90), (120, 400), (500, 420)])
def test_the_table_reads_back_as_the_table_everywhere_in_the_image(u, v):
    """The test that catches a wrong transform: a tilted camera sees a flat table at a depth that
    varies across the image, and only the right T turns all of that back into one height."""
    metres = plane_depth(0.070)
    (target,), dropped = measure([detection(disc_mask(u, v, 15, INTR))], metres)
    assert dropped == []
    assert target.xyz_base_mm[2] == pytest.approx(70.0, abs=1.0)


@pytest.mark.parametrize("base_mm", [(400.0, 0.0, 70.0), (250.0, -300.0, 120.0),
                                     (650.0, 200.0, 200.0), (300.0, 300.0, 90.0)])
def test_a_known_point_in_the_workspace_comes_back_where_it_was(base_mm):
    """Base -> pixel through T's inverse, then pixel -> base through the code. Rounding the disc's
    centre to a pixel costs up to Z/fx, about 2.3 mm at this range, and nothing else may."""
    u, v, depth = pixel_of(base_mm)
    (target,), _ = measure([detection(disc_mask(round(u), round(v), 15, INTR))], flat_depth(depth))
    assert np.linalg.norm(np.array(target.xyz_base_mm) - np.array(base_mm)) < 4.0


def test_the_camera_z_is_kept_apart_from_the_base_z():
    (target,), _ = measure([detection(disc_mask(320, 240, 15, INTR))], flat_depth(1.3))
    assert target.z_cam_m == pytest.approx(1.3)
    assert target.xyz_cam_m[2] == pytest.approx(1.3)
    assert abs(target.xyz_base_mm[2] - 1300.0) > 500.0


# =============================================================================
# the centroid
# =============================================================================
def test_the_centroid_is_u_then_v():
    """In a frame that is not square, with the object far from the diagonal."""
    (target,), _ = measure([detection(disc_mask(500, 100, 20, INTR))], flat_depth())
    assert target.centroid_px == pytest.approx((500.0, 100.0))


def test_the_centroid_is_the_masks_and_not_the_boxs():
    """An L: its box centre and its centroid are in different places, and the box is what the
    earlier scripts measured from."""
    mask = np.zeros((INTR.height, INTR.width), dtype=bool)
    mask[100:300, 100:120] = True       # the long stroke
    mask[280:300, 100:300] = True       # the foot
    rows, columns = np.nonzero(mask)
    (target,), _ = measure([Detection((100, 100, 299, 299), 0, 0.9, mask)], flat_depth())
    assert target.centroid_px == pytest.approx((columns.mean(), rows.mean()))
    assert target.centroid_px != pytest.approx((199.5, 199.5), abs=5.0)


def test_the_depth_is_read_at_v_u_and_not_u_v():
    """Two blobs, each sitting at the other's swapped coordinates, both in the mask. So a swap reads
    a valid depth of the wrong value, where it would otherwise be rescued by the fallback."""
    here, swapped = disc_mask(300, 100, 6, INTR), disc_mask(100, 300, 6, INTR)
    metres = flat_depth(1.4)
    metres[here] = 0.7
    metres[swapped] = 1.1
    assert depth_at_centroid(metres, here | swapped, 300.0, 100.0) == (0.7, CENTROID_WINDOW)


def test_a_depth_gradient_reads_as_its_value_at_the_centroid():
    u, v = pixel_grid(INTR)
    metres = 0.5 + v / 1000.0 + u / 10000.0
    (target,), _ = measure([detection(disc_mask(600, 50, 3, INTR))], metres)
    assert target.z_cam_m == pytest.approx(0.5 + 50 / 1000.0 + 600 / 10000.0, abs=1e-3)


def test_a_centroid_on_the_mask_is_reported_as_on_it():
    """(u, v) = (300, 100) is on this disc, and (row 300, column 100) is not."""
    (target,), _ = measure([detection(disc_mask(300, 100, 20, INTR))], flat_depth())
    assert target.centroid_on_mask is True


def test_the_position_is_deprojected_at_the_sub_pixel_centroid():
    mask = np.zeros((INTR.height, INTR.width), dtype=bool)
    mask[100:102, 200:202] = True       # centroid at (200.5, 100.5)
    (target,), _ = measure([detection(mask)], flat_depth())
    assert target.centroid_px == pytest.approx((200.5, 100.5))
    assert target.xyz_cam_m[0] == pytest.approx((200.5 - INTR.ppx) / INTR.fx)
    assert target.xyz_cam_m[1] == pytest.approx((100.5 - INTR.ppy) / INTR.fy)


# =============================================================================
# the first principal component
# =============================================================================
@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 170.0])
def test_the_orientation_of_a_rotated_ellipse(angle):
    mask = ellipse_mask(320, 240, angle, 80, 20, INTR)
    (target,), _ = measure([detection(mask)], flat_depth())
    assert circular_distance(target.orientation_deg, angle) < 1.0


@pytest.mark.parametrize("angle", [10.0, 37.0, 80.0, 101.0, 155.0])
def test_the_orientation_agrees_with_an_independent_eigendecomposition(angle):
    mask = ellipse_mask(300, 260, angle, 90, 25, INTR)
    rows, columns = np.nonzero(mask)
    points = np.stack([columns - columns.mean(), rows.mean() - rows])      # y up
    values, vectors = np.linalg.eigh(np.cov(points))
    major = vectors[:, np.argmax(values)]
    independent = np.degrees(np.arctan2(major[1], major[0])) % 180.0

    (target,), _ = measure([detection(mask)], flat_depth())
    assert circular_distance(target.orientation_deg, independent) < 0.01


def test_the_two_diagonals_are_not_confused():
    """Getting image rows backwards mirrors every angle while looking entirely plausible."""
    u, v = pixel_grid(INTR)
    rising = np.abs((u - 320) - (240 - v)) < 8      # "/" on screen
    rising &= np.abs(u - 320) < 100
    falling = np.abs((u - 320) - (v - 240)) < 8     # "\" on screen
    falling &= np.abs(u - 320) < 100
    (slash,), _ = measure([detection(rising)], flat_depth())
    (backslash,), _ = measure([detection(falling)], flat_depth())
    assert slash.orientation_deg == pytest.approx(45.0, abs=1.0)
    assert backslash.orientation_deg == pytest.approx(135.0, abs=1.0)


def test_it_is_the_long_axis_and_not_the_short_one():
    mask = np.zeros((INTR.height, INTR.width), dtype=bool)
    mask[230:250, 200:440] = True       # 240 wide, 20 tall: lies along the bottom edge
    (target,), _ = measure([detection(mask)], flat_depth())
    assert circular_distance(target.orientation_deg, 0.0) < 0.5


def test_the_orientation_is_always_in_range():
    for angle in np.arange(0.0, 180.0, 7.0):
        (target,), _ = measure([detection(ellipse_mask(320, 240, angle, 70, 30, INTR))],
                               flat_depth())
        assert 0.0 <= target.orientation_deg < 180.0


def test_one_pixel_has_no_orientation_but_is_still_a_position():
    mask = np.zeros((INTR.height, INTR.width), dtype=bool)
    mask[100, 200] = True
    (target,), _ = measure([detection(mask)], flat_depth())
    assert target.orientation_deg is None
    assert target.base_axis_deg is None
    assert target.centroid_px == (200.0, 100.0)


# =============================================================================
# the axis in the base frame
# =============================================================================
@pytest.mark.parametrize("x, y", WORKSPACE_POINTS)
@pytest.mark.parametrize("axis", [0.0, 30.0, 45.0, 90.0, 135.0, 150.0])
def test_an_object_lying_at_a_known_angle_reads_back_as_that_angle(x, y, axis):
    """Across the whole workspace, as far from the image's centre as the tilt can make it matter."""
    mask, depth = ellipse_lying_on_a_plane((x, y, 150.0), axis, 80.0, 25.0)
    (target,), _ = measure([detection(mask)], depth)
    assert circular_distance(target.base_axis_deg, axis) < 1.5


@pytest.mark.parametrize("axis", [20.0, 70.0, 110.0, 160.0])
def test_the_angle_is_the_same_through_the_levelled_demo_calibration(axis):
    demo = load_T_base_cam(DEFAULT_CALIBRATION.parent / "calibration_rosbag_demo.txt")
    mask, depth = ellipse_lying_on_a_plane((450.0, 100.0, 150.0), axis, 110.0, 30.0, demo)
    (target,), _ = measure([detection(mask)], depth, demo)
    assert circular_distance(target.base_axis_deg, axis) < 1.5


def test_turning_the_image_angle_by_the_rotation_alone_would_be_wrong():
    """Why base_axis_deg follows the rays: the tilt is what this gets wrong, by this much."""
    worst = 0.0
    R = T[:3, :3]
    for x, y in WORKSPACE_POINTS:
        for axis in (30.0, 45.0, 135.0, 150.0):
            mask, depth = ellipse_lying_on_a_plane((x, y, 150.0), axis, 80.0, 25.0)
            (target,), _ = measure([detection(mask)], depth)
            theta = np.radians(target.orientation_deg)
            turned = R @ np.array([np.cos(theta), -np.sin(theta), 0.0])
            naive = np.degrees(np.arctan2(turned[1], turned[0])) % 180.0
            worst = max(worst, circular_distance(naive, axis))
    assert worst > 5.0


def test_the_two_diagonals_are_not_confused_in_the_base_either():
    """Not mirrored by the image's rows growing downward, or by looking at the base from above."""
    for axis in (45.0, 135.0):
        mask, depth = ellipse_lying_on_a_plane((450.0, 0.0, 150.0), axis, 110.0, 30.0)
        (target,), _ = measure([detection(mask)], depth)
        assert circular_distance(target.base_axis_deg, axis) < 1.5


@pytest.mark.parametrize("x, y", WORKSPACE_POINTS)
@pytest.mark.parametrize("axis", [30.0, 60.0, 135.0])
@pytest.mark.parametrize("short", [40.0, 53.0, 61.0], ids=["2 to 1", "1.5 to 1", "1.3 to 1"])
def test_a_stubby_object_reads_back_as_well(x, y, axis, short):
    """The tilt shears the mask, and the axis in the image moves most when there is little elongation
    to hold it: at 1.5 to 1 near a corner, the image's own axis is about 10 degrees out."""
    mask, depth = ellipse_lying_on_a_plane((x, y, 150.0), axis, 80.0, short)
    (target,), _ = measure([detection(mask)], depth)
    assert circular_distance(target.base_axis_deg, axis) < 2.0


def test_the_axis_is_in_range_and_does_not_depend_on_the_height_of_the_plane():
    """Changing the height only scales what lands on the plane, about the point under the camera."""
    mask, _ = ellipse_lying_on_a_plane((450.0, 0.0, 150.0), 80.0, 110.0, 30.0)
    rows, columns = np.nonzero(mask)
    angles = [base_axis_deg(T, INTR, RS, rows, columns, z) for z in (-500.0, 70.0, 150.0, 1000.0)]
    assert angles == pytest.approx([angles[0]] * 4, abs=1e-6)
    assert 0.0 <= angles[0] < 180.0


def test_a_big_mask_is_sampled_and_keeps_its_axis():
    calls = []

    def counted(intrinsics, pixel, depth):
        calls.append(pixel)
        return RS.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)

    spy = types.SimpleNamespace(rs2_deproject_pixel_to_point=counted)
    mask, _ = ellipse_lying_on_a_plane((450.0, 0.0, 150.0), 120.0, 200.0, 60.0)
    rows, columns = np.nonzero(mask)
    assert rows.size > 5 * AXIS_SAMPLE_PIXELS
    angle = base_axis_deg(T, INTR, spy, rows, columns, 150.0)
    assert len(calls) < 2 * AXIS_SAMPLE_PIXELS
    assert circular_distance(angle, 120.0) < 1.5


def test_a_plane_the_rays_cannot_reach_means_no_axis():
    """A z above the camera, which only a calibration that does not match the scene can produce."""
    rows, columns = np.nonzero(disc_mask(320, 240, 10, INTR))
    assert base_axis_deg(T, INTR, RS, rows, columns, 5000.0) is None


def test_one_pixel_has_no_axis_in_the_base_either():
    assert base_axis_deg(T, INTR, RS, np.array([100]), np.array([200]), 150.0) is None


def test_the_table_shows_the_axis_or_a_dash():
    table = format_targets([make_target(base_axis_deg=37.25), make_target()]).splitlines()
    assert "37.2" in table[1] or "37.3" in table[1]
    assert table[2].rstrip().endswith("-")


# =============================================================================
# z at the centroid
# =============================================================================
def test_z_is_at_the_centroid_and_not_the_median_of_the_mask():
    """Most of the object is farther away, so the mask's median is not the centroid's depth."""
    mask = disc_mask(320, 240, 60, INTR)
    metres = flat_depth(1.2)
    metres[235:246, 315:326] = 0.8
    (target,), _ = measure([detection(mask)], metres)
    assert target.z_cam_m == pytest.approx(0.8)
    assert target.z_source == CENTROID_WINDOW
    assert float(np.median(metres[mask])) == pytest.approx(1.2)


def test_the_window_only_counts_pixels_of_the_mask():
    """A thin bar: most of a window round its centroid is the table beside it."""
    mask = np.zeros((INTR.height, INTR.width), dtype=bool)
    mask[239:242, 200:440] = True
    metres = flat_depth(1.4)
    metres[mask] = 0.9
    (target,), _ = measure([detection(mask)], metres)
    assert target.z_cam_m == pytest.approx(0.9)


def test_the_window_takes_the_median_so_a_few_bad_pixels_do_not_move_z():
    mask = disc_mask(320, 240, 30, INTR)
    metres = flat_depth(1.0)
    metres[236:238, 316:325] = 0.2      # 18 of the window's 81 pixels
    (target,), _ = measure([detection(mask)], metres)
    assert target.z_cam_m == pytest.approx(1.0)


def test_holes_in_the_window_are_not_distances():
    mask = disc_mask(320, 240, 30, INTR)
    metres = flat_depth(1.0)
    metres[238:243, 318:323] = 0.0      # the centre of the window, not all of it
    (target,), _ = measure([detection(mask)], metres)
    assert target.z_cam_m == pytest.approx(1.0)
    assert target.z_source == CENTROID_WINDOW


def test_a_window_with_no_reading_falls_back_to_the_masks_median():
    """A specular patch at the centroid, which is the usual case on glass."""
    mask = disc_mask(320, 240, 30, INTR)
    metres = flat_depth(1.0)
    half = CENTROID_WINDOW_PX // 2
    metres[240 - half:240 + half + 1, 320 - half:320 + half + 1] = 0.0
    (target,), _ = measure([detection(mask)], metres)
    assert target.z_cam_m == pytest.approx(1.0)
    assert target.z_source == MASK_MEDIAN


def test_a_centroid_off_the_mask_falls_back_and_says_so():
    """A ring: the centroid is in the hole, whose depth belongs to the table, not to the object."""
    u, v = pixel_grid(INTR)
    radius = np.hypot(u - 320, v - 240)
    mask = (radius > 40) & (radius < 60)
    metres = flat_depth(1.4)
    metres[mask] = 0.9
    (target,), _ = measure([detection(mask)], metres)
    assert target.centroid_on_mask is False
    assert target.z_cam_m == pytest.approx(0.9)
    assert target.z_source == MASK_MEDIAN
    assert "off the mask" in format_targets([target])


def test_no_depth_anywhere_drops_the_detection_with_a_reason():
    (targets, dropped) = measure([detection(disc_mask(320, 240, 30, INTR), cls=2)], flat_depth(0.0))
    assert targets == []
    assert dropped == ["paper: no depth reading anywhere on its mask"]


def test_a_window_at_the_frame_edge_does_not_wrap():
    mask = np.zeros((INTR.height, INTR.width), dtype=bool)
    mask[0:3, 0:3] = True
    z, source = depth_at_centroid(flat_depth(), mask, 1.0, 1.0)
    assert (z, source) == (1.0, CENTROID_WINDOW)


def test_depth_at_centroid_with_nothing_to_read():
    mask = disc_mask(320, 240, 10, INTR)
    assert depth_at_centroid(flat_depth(0.0), mask, 320.0, 240.0) == (0.0, NO_DEPTH)


# =============================================================================
# what cannot become a target
# =============================================================================
def test_no_mask_is_dropped_with_a_reason():
    targets, dropped = measure([Detection((0, 0, 9, 9), 1, 0.9, None)], flat_depth())
    assert targets == []
    assert dropped == ["metal: the model gave no mask"]


def test_an_empty_mask_is_dropped_with_a_reason():
    targets, dropped = measure([detection(np.zeros((INTR.height, INTR.width), dtype=bool))],
                               flat_depth())
    assert targets == []
    assert dropped == ["glass: empty mask"]


def test_a_depth_map_the_wrong_size_is_refused_rather_than_guessed_at():
    """This is what a scaled frame with an unscaled depth map would look like. Silently resampling
    it, or reading the wrong pixels, would send the robot to the wrong place."""
    with pytest.raises(ValueError, match="does not match"):
        measure([detection(disc_mask(320, 240, 10, INTR))], np.ones((720, 1280)))


def test_no_detections_is_not_an_error():
    assert measure([], flat_depth()) == ([], [])


def test_an_unnamed_class_is_kept_and_labelled_by_its_number():
    (target,), _ = measure([detection(disc_mask(320, 240, 10, INTR), cls=7)], flat_depth())
    assert (target.cls, target.name) == (7, "7")


# =============================================================================
# sorting and showing
# =============================================================================
def test_the_nearest_to_the_camera_comes_first():
    far, near, middle = (make_target(z_cam=z) for z in (1.4, 0.7, 1.0))
    assert sort_nearest_first([far, near, middle]) == [near, middle, far]


def test_sorting_is_by_camera_z_and_not_base_z():
    """A tilted camera orders these two the opposite way round from their heights in the base."""
    nearer_to_camera = make_target(base=(400.0, 0.0, 100.0), z_cam=0.8)
    higher_in_base = make_target(base=(400.0, 0.0, 300.0), z_cam=1.2)
    assert sort_nearest_first([higher_in_base, nearer_to_camera])[0] is nearer_to_camera


def test_drawing_does_not_touch_the_frame_and_marks_the_centroid():
    frame = np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)
    (target,), _ = measure([detection(ellipse_mask(320, 240, 30.0, 80, 20, INTR))], flat_depth())
    drawn = draw_targets(frame, [target], NAMES)
    assert frame.sum() == 0
    assert drawn[240, 320].sum() > 0


def test_the_table_has_one_row_per_target_in_order():
    table = format_targets([make_target(cls=0, base=(1.0, 2.0, 3.0)),
                            make_target(cls=3, base=(4.0, 5.0, 6.0))])
    lines = table.splitlines()
    assert len(lines) == 3
    assert "glass" in lines[1] and "plastic" in lines[2]


# =============================================================================
# the real pyrealsense2
# =============================================================================
def test_the_real_deproject_is_this_pinhole():
    """What justifies the stand-in in synthetic.py. Skips where pyrealsense2 is not installed."""
    rs = pytest.importorskip("pyrealsense2")
    intrinsics = rs.intrinsics()
    intrinsics.width, intrinsics.height = INTR.width, INTR.height
    intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy = (INTR.fx, INTR.fy, INTR.ppx,
                                                                    INTR.ppy)
    intrinsics.model = rs.distortion.inverse_brown_conrady
    intrinsics.coeffs = [0.0] * 5
    for pixel in ([320.0, 240.0], [10.5, 470.25], [600.75, 30.0]):
        real = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, 1.23)
        assert real == pytest.approx(RS.rs2_deproject_pixel_to_point(INTR, pixel, 1.23), abs=1e-5)
