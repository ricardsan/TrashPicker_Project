"""The tracker that gives a detection the same id on the next frame.

What matters here is not that the matching is clever -- it is greedy IoU, which is not -- but that
an id means something: the same physical object keeps it, a different object does not get it, and
the same recording twice gives the same answer. Those are the properties the per-object CSV rests
on, so they are what is pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from camera.segmentation_realsense import Detection
from camera.tracking import Tracker, centre_distances, centroid_of, iou_matrix

FRAME = (480, 640)


def detection(x1, y1, x2, y2, cls=0, conf=0.9):
    return Detection((x1, y1, x2, y2), cls, conf)


def box(x, y, size=100, cls=0):
    """A square of `size` with its top-left at (x, y)."""
    return detection(x, y, x + size, y + size, cls)


# --- the geometry it is built on ---------------------------------------------
def test_iou_of_a_box_with_itself_is_one():
    boxes = np.array([[10.0, 10.0, 110.0, 110.0]])
    assert iou_matrix(boxes, boxes)[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_of_half_overlapping_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    # 50 of overlap against a union of 150.
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)


def test_iou_of_a_degenerate_box_is_zero_not_a_division_by_zero():
    """YoloDetector clips inclusively, so an edge detection can have zero area."""
    edge = np.array([[639.0, 479.0, 639.0, 479.0]])
    other = np.array([[600.0, 440.0, 640.0, 480.0]])
    result = iou_matrix(edge, other)
    assert result[0, 0] == 0.0
    assert np.isfinite(result).all()


def test_iou_matrix_shape_with_no_boxes_on_one_side():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_centroid_of_a_box_is_its_middle():
    assert centroid_of((10, 20, 30, 60)) == (20.0, 40.0)


def test_centre_distances_is_euclidean():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    assert centre_distances(a, b)[0, 0] == pytest.approx(5.0)


# --- what an id means --------------------------------------------------------
def test_a_single_object_keeps_its_id_while_it_drifts():
    tracker = Tracker()
    ids = [tracker.update([box(100 + step * 8, 100)], FRAME)[0] for step in range(10)]
    assert ids == [ids[0]] * 10
    assert tracker.next_id == 1, "a drifting object must not spawn new tracks"


def test_a_second_object_gets_a_different_id():
    tracker = Tracker()
    tracker.update([box(100, 100)], FRAME)
    first, second = tracker.update([box(104, 100), box(400, 300)], FRAME)
    assert first != second


def test_ids_come_back_in_the_detections_own_order():
    """The CSV pairs ids with detections positionally, so the order is load-bearing."""
    tracker = Tracker()
    tracker.update([box(100, 100), box(400, 300)], FRAME)
    # Same two objects, handed over the other way round.
    swapped = tracker.update([box(402, 300), box(102, 100)], FRAME)
    assert swapped == [1, 0]


def test_the_class_flipping_does_not_change_the_id():
    """These checkpoints flip glass and plastic on the same bottle; the object is still one object.

    Minting a new id on every flicker would make the distinct-object count meaningless exactly
    where the model is least certain.
    """
    tracker = Tracker()
    first = tracker.update([box(100, 100, cls=0)], FRAME)[0]
    second = tracker.update([box(104, 100, cls=3)], FRAME)[0]
    third = tracker.update([box(108, 100, cls=0)], FRAME)[0]
    assert first == second == third
    assert tracker.next_id == 1


def test_a_short_occlusion_keeps_the_id():
    tracker = Tracker()
    first = tracker.update([box(100, 100)], FRAME)[0]
    for _ in range(3):              # inside max_misses
        tracker.update([], FRAME)
    assert tracker.update([box(112, 100)], FRAME)[0] == first


def test_a_long_absence_gives_a_new_id():
    """A track cannot be kept forever, or two unrelated objects minutes apart would share an id."""
    tracker = Tracker(max_misses=5)
    first = tracker.update([box(100, 100)], FRAME)[0]
    for _ in range(6):              # past max_misses
        tracker.update([], FRAME)
    assert tracker.update([box(100, 100)], FRAME)[0] != first


def test_an_object_that_jumps_too_far_is_a_new_object():
    tracker = Tracker()
    first = tracker.update([box(50, 50, size=40)], FRAME)[0]
    # Right across the frame: no IoU, and far beyond the centroid gate.
    assert tracker.update([box(560, 400, size=40)], FRAME)[0] != first


def test_a_fast_small_object_is_matched_by_its_centroid():
    """Zero IoU between consecutive frames does not mean a different object at 15 Hz."""
    tracker = Tracker()
    first = tracker.update([box(100, 100, size=20)], FRAME)[0]
    # 25 px of travel: clear of its own 20 px box, well inside the 5%-of-diagonal gate (40 px).
    assert tracker.update([box(125, 100, size=20)], FRAME)[0] == first


def test_two_objects_passing_each_other_do_not_swap_ids():
    tracker = Tracker()
    left, right = tracker.update([box(100, 200, size=60), box(400, 200, size=60)], FRAME)
    for step in range(1, 5):
        ids = tracker.update([box(100 + step * 20, 200, size=60),
                              box(400 - step * 20, 200, size=60)], FRAME)
        assert ids == [left, right], f"ids swapped at step {step}: {ids}"


def test_the_same_frames_twice_give_the_same_ids():
    """Ties are broken deterministically, so a recording is reproducible."""
    script = [
        [box(100, 100), box(300, 100)],
        [box(100, 100), box(300, 100)],              # identical boxes: every pair ties
        [box(300, 100), box(100, 100)],
        [box(102, 100)],
        [box(102, 100), box(500, 400), box(300, 100)],
    ]

    def run():
        tracker = Tracker()
        return [tracker.update(frame, FRAME) for frame in script]

    assert run() == run()


def test_an_empty_frame_returns_no_ids_and_does_not_crash():
    tracker = Tracker()
    assert tracker.update([], FRAME) == []


def test_tracks_are_forgotten_so_the_table_does_not_grow_without_bound():
    """A run of thousands of frames must not accumulate a track per object ever seen."""
    tracker = Tracker(max_misses=2)
    for step in range(40):
        # A new object every frame, each one somewhere else entirely.
        tracker.update([box((step * 137) % 500, (step * 61) % 300, size=30)], FRAME)
    assert len(tracker.tracks) <= 4, f"{len(tracker.tracks)} tracks still live"


def test_every_detection_gets_an_id_even_when_they_all_overlap():
    """Greedy matching must not leave a detection without an id, whatever the geometry."""
    tracker = Tracker()
    tracker.update([box(100, 100)], FRAME)
    ids = tracker.update([box(100, 100), box(102, 100), box(104, 100)], FRAME)
    assert len(ids) == 3
    assert len(set(ids)) == 3, f"two detections on one frame shared an id: {ids}"


def test_ids_are_unique_within_a_frame_over_a_long_run():
    rng = np.random.default_rng(7)
    tracker = Tracker()
    for _ in range(200):
        count = int(rng.integers(0, 6))
        frame = [box(int(rng.integers(0, 540)), int(rng.integers(0, 380)), size=60)
                 for _ in range(count)]
        ids = tracker.update(frame, FRAME)
        assert len(ids) == count
        assert len(set(ids)) == count


# --- the greedy matcher, against the obvious spelling of it ------------------
def reference_greedy(scores, track_ids, free_tracks, free_detections, *,
                     better_is_larger, gate):
    """The plain double loop over pairs, sorted as tuples.

    _greedy does the gating and the ordering in numpy instead, because the loop was O(T x D) in
    Python inside run_frames' measured region -- 11 ms a frame at RF-DETR's max_det of 300. This
    is kept as the definition of what the fast version has to agree with.
    """
    candidates = []
    for row in free_tracks:
        for column in free_detections:
            score = float(scores[row, column])
            if (score >= gate) if better_is_larger else (score <= gate):
                candidates.append((-score if better_is_larger else score,
                                   track_ids[row], column, row))
    pairs, used_rows, used_columns = [], set(), set()
    for _, _, column, row in sorted(candidates):
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        pairs.append((row, column))
    return pairs


@pytest.mark.parametrize("better_is_larger,gate", [(True, 0.3), (False, 0.5)])
def test_the_vectorised_matcher_pairs_exactly_as_the_plain_loop_does(better_is_larger, gate):
    """Including the tie-breaking, which is what makes a recording reproducible.

    Scores are quantised so exact ties are common: a change in the sort order would show up
    here and nowhere else.
    """
    from camera.tracking import _greedy

    rng = np.random.default_rng(0)
    for _ in range(600):
        tracks, detections = int(rng.integers(0, 8)), int(rng.integers(0, 8))
        scores = (np.round(rng.random((tracks, detections)) * 3) / 3
                  if tracks and detections else np.zeros((tracks, detections)))
        ids = list(rng.permutation(tracks * 3)[:tracks]) if tracks else []
        rows, columns = list(range(tracks)), list(range(detections))

        assert _greedy(scores, ids, rows, columns,
                       better_is_larger=better_is_larger, gate=gate) == \
            reference_greedy(scores, ids, rows, columns,
                             better_is_larger=better_is_larger, gate=gate)


def test_matching_a_full_max_det_frame_is_not_ruinous():
    """RF-DETR's max_det is 300, reachable with --objects --conf 0.01, and this runs inside the
    measured region. It is not free, but it must not dominate a frame."""
    import time

    tracker = Tracker()
    detections = [box(i % 60 * 10, i // 60 * 10, size=40) for i in range(300)]
    tracker.update(detections, FRAME)

    samples = []
    for _ in range(20):
        start = time.perf_counter()
        tracker.update(detections, FRAME)
        samples.append(time.perf_counter() - start)
    median_ms = float(np.median(samples)) * 1000

    # The double-loop version was 11.4 ms here. Generous bound: this is a timing test on shared
    # hardware, and the point is the order of magnitude, not the digit.
    assert median_ms < 8.0, f"{median_ms:.1f} ms to match 300 detections"
