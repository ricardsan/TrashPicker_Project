"""Giving a detection the same id on the next frame, so an object can be followed through a run.

Deliberately free of cv2, torch and ultralytics: it works on `Detection.xyxy` tuples and nothing
else, which is the point. The whole repository exists to compare three architectures under the same
conditions, so a tracker that only worked for one of them -- ultralytics' own `model.track()`, say,
which Mask R-CNN and RF-DETR have no equivalent of -- would put an id column in the YOLO logs and
leave the other two blank. Greedy matching on boxes is weaker than a Kalman tracker, but it is the
same weakness for all three.

The algorithm, in order:

  1. IoU between every live track and every detection on this frame; take pairs greedily, highest
     IoU first, down to `iou_threshold`.
  2. Whatever is left over gets a second pass on centroid distance, gated at `max_centre_fraction`
     of the frame diagonal. A small object moving quickly can have *zero* IoU between consecutive
     frames at 15 Hz while obviously still being the same object, and pass 1 cannot see that.
  3. A track nothing matched ages by a miss, and is forgotten after `max_misses` of them -- so an
     object that goes behind another one keeps its id when it comes back out.
  4. Anything still unmatched is a new object and takes the next id.

Matching ignores the predicted class on purpose. These checkpoints flip glass and clear plastic back
and forth on the same bottle from frame to frame, and requiring the class to agree would mint a new
id on every flicker -- which would make the object count useless precisely where the model is least
sure. The track keeps its id and the per-row `cls` records what was predicted *that* frame, so the
flicker stays visible in the data instead of being hidden by it.

Ties are broken on (track id, detection index) so that the same frames always produce the same ids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_IOU = 0.3
DEFAULT_CENTRE_FRACTION = 0.05      # of the frame diagonal: 40 px on a 640x480 frame
DEFAULT_MAX_MISSES = 5              # a third of a second at 15 Hz


@dataclass
class Track:
    """One object being followed, as last seen."""

    id: int
    xyxy: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cls: int
    misses: int = 0                 # consecutive frames it has not been matched on
    hits: int = 1                   # frames it has been matched on, this one included


def centroid_of(xyxy) -> tuple[float, float]:
    """The centre of a box, as floats -- not the integer pixel the dot is drawn at."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Pairwise IoU between an (T, 4) and a (D, 4) array of xyxy boxes.

    Boxes here can be degenerate: YoloDetector clips to width - 1 inclusive, so a detection on the
    frame edge can come back with x1 == x2 and zero area. That must give an IoU of 0, not a
    division by zero, which is why the union is clamped rather than assumed positive.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)

    t = tracks[:, None, :]          # (T, 1, 4)
    d = detections[None, :, :]      # (1, D, 4)

    left = np.maximum(t[..., 0], d[..., 0])
    top = np.maximum(t[..., 1], d[..., 1])
    right = np.minimum(t[..., 2], d[..., 2])
    bottom = np.minimum(t[..., 3], d[..., 3])

    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_t = np.clip(t[..., 2] - t[..., 0], 0, None) * np.clip(t[..., 3] - t[..., 1], 0, None)
    area_d = np.clip(d[..., 2] - d[..., 0], 0, None) * np.clip(d[..., 3] - d[..., 1], 0, None)

    union = area_t + area_d - overlap
    return np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0)


def centre_distances(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Pairwise euclidean distance between two (N, 2) arrays of centroids."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=float)
    return np.linalg.norm(tracks[:, None, :] - detections[None, :, :], axis=2)


def _greedy(scores: np.ndarray, track_ids, free_tracks, free_detections, *,
            better_is_larger: bool, gate: float) -> list[tuple[int, int]]:
    """Greedily pair rows with columns by score, best first, each used at most once.

    `track_ids` makes the tie-break stable across frames: two candidate pairs with identical
    scores would otherwise be resolved by whatever order the rows happen to be in, and the same
    recording could then produce different ids on a second run.

    The gating and the ordering are done in numpy rather than with a double loop over the pairs.
    The loop was the obvious spelling and is fine at five detections, but it is O(T x D) in
    *Python* and this runs inside run_frames' measured region: it cost 11.4 ms a frame at 300
    detections, which is RF-DETR's max_det and reachable with --objects --conf 0.01. This costs
    4.0 ms there, and 0.06 ms at the five-ish detections a sane --conf gives. Only the pairs that
    pass the gate are walked, and the walk is the part that genuinely has to be sequential.

    It pairs identically to the loop, ties included -- test_tracking.py keeps the loop as the
    reference and diffs the two over several thousand tie-heavy cases.
    """
    if not free_tracks or not free_detections:
        return []

    rows = np.asarray(free_tracks, dtype=int)
    columns = np.asarray(free_detections, dtype=int)
    window = scores[np.ix_(rows, columns)]
    passed = window >= gate if better_is_larger else window <= gate
    if not passed.any():
        return []

    local_rows, local_columns = np.nonzero(passed)
    values = window[local_rows, local_columns]
    ranked = -values if better_is_larger else values
    # lexsort orders by the last key first: score, then the track's own id, then the detection
    # index -- the same total order the sorted() of tuples had.
    order = np.lexsort((columns[local_columns],
                        np.asarray(track_ids, dtype=int)[rows[local_rows]],
                        ranked))

    pairs = []
    used_rows, used_columns = set(), set()
    for position in order:
        row = int(rows[local_rows[position]])
        column = int(columns[local_columns[position]])
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        pairs.append((row, column))
    return pairs


class Tracker:
    """Frame-to-frame identity for detections, by greedy box matching.

    `update` is the whole interface: hand it this frame's detections and it hands back one id per
    detection, in the same order.
    """

    def __init__(self, iou_threshold: float = DEFAULT_IOU,
                 max_centre_fraction: float = DEFAULT_CENTRE_FRACTION,
                 max_misses: int = DEFAULT_MAX_MISSES):
        self.iou_threshold = iou_threshold
        self.max_centre_fraction = max_centre_fraction
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self.next_id = 0

    def update(self, detections, frame_shape=(480, 640)) -> list[int]:
        """One id per detection, in the detections' own order.

        `frame_shape` is (height, width) and only sets the scale of the centroid gate, so a
        caller that does not have it can leave it at the live camera's geometry.
        """
        if not detections:
            self._age_out(matched=set())
            return []

        boxes = np.asarray([d.xyxy for d in detections], dtype=float).reshape(-1, 4)
        centroids = np.asarray([centroid_of(d.xyxy) for d in detections],
                               dtype=float).reshape(-1, 2)

        if not self.tracks:
            return [self._start(d, centroids[i]) for i, d in enumerate(detections)]

        track_boxes = np.asarray([t.xyxy for t in self.tracks], dtype=float).reshape(-1, 4)
        track_centroids = np.asarray([t.centroid for t in self.tracks],
                                     dtype=float).reshape(-1, 2)
        track_ids = [t.id for t in self.tracks]

        free_tracks = list(range(len(self.tracks)))
        free_detections = list(range(len(detections)))

        pairs = _greedy(iou_matrix(track_boxes, boxes), track_ids,
                        free_tracks, free_detections,
                        better_is_larger=True, gate=self.iou_threshold)

        # Pass 2, on whatever pass 1 could not pair: an object can move clear of its own box
        # between two frames and still plainly be the same object.
        matched_rows = {row for row, _ in pairs}
        matched_columns = {column for _, column in pairs}
        leftover_tracks = [r for r in free_tracks if r not in matched_rows]
        leftover_detections = [c for c in free_detections if c not in matched_columns]
        if leftover_tracks and leftover_detections:
            height, width = frame_shape[:2]
            gate = self.max_centre_fraction * float(np.hypot(height, width))
            pairs += _greedy(centre_distances(track_centroids, centroids), track_ids,
                             leftover_tracks, leftover_detections,
                             better_is_larger=False, gate=gate)

        ids: list[int | None] = [None] * len(detections)
        for row, column in pairs:
            track = self.tracks[row]
            track.xyxy = detections[column].xyxy
            track.centroid = tuple(centroids[column])
            track.cls = detections[column].cls
            track.misses = 0
            track.hits += 1
            ids[column] = track.id

        self._age_out(matched={row for row, _ in pairs})

        for column, detection in enumerate(detections):
            if ids[column] is None:
                ids[column] = self._start(detection, centroids[column])
        return [int(i) for i in ids]

    # --- bookkeeping ----------------------------------------------------------
    def _start(self, detection, centroid) -> int:
        """Remember a new object and return its id."""
        self.tracks.append(Track(id=self.next_id, xyxy=detection.xyxy,
                                 centroid=tuple(centroid), cls=detection.cls))
        self.next_id += 1
        return self.tracks[-1].id

    def _age_out(self, matched: set) -> None:
        """Charge a miss to every track nothing matched, and forget the ones that ran out."""
        kept = []
        for row, track in enumerate(self.tracks):
            if row in matched:
                kept.append(track)
                continue
            track.misses += 1
            if track.misses <= self.max_misses:
                kept.append(track)
        self.tracks = kept
