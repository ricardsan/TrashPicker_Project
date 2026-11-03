"""mAP, matching, and the fitness rule that picks best.pt."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from other_models.metrics import IOUV, Evaluator, mask_iou, match_predictions, to_mask_resolution

NAMES = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}

# A flawless detector scores 0.995, not 1.0. ultralytics' compute_ap appends a sentinel at
# recall 1.0 with precision 0, and one of the 101 interpolation points lands on it. Matching that
# number exactly is the whole point of reusing their code, so it is asserted rather than
# worked around.
PERFECT = 0.995


def boxes_and_masks(offsets=(0, 0), size=200):
    boxes = np.array([[10, 10, 50, 50], [60, 60, 120, 130]], np.float32) + np.array(
        [*offsets, *offsets], np.float32)
    masks = np.zeros((2, size, size), np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        masks[i, y1:y2, x1:x2] = 1
    return boxes, masks


def ground_truth():
    boxes, masks = boxes_and_masks()
    return {"boxes": boxes, "labels": np.array([0, 2]), "masks": masks}


def perfect_prediction():
    truth = ground_truth()
    return {**truth, "scores": np.array([0.9, 0.8], np.float32)}


# =============================================================================
# match_predictions
# =============================================================================
def test_match_predictions_is_one_to_one():
    """Two predictions on one ground truth: only the better one may be a true positive."""
    iou = np.array([[0.9, 0.8]])                 # 1 gt, 2 predictions
    correct = match_predictions(iou, np.array([0, 0]), np.array([0]))
    assert correct.shape == (2, len(IOUV))
    assert correct[0, 0] and not correct[1, 0]


def test_match_predictions_respects_class():
    iou = np.array([[0.95]])
    assert not match_predictions(iou, np.array([1]), np.array([0])).any()
    assert match_predictions(iou, np.array([0]), np.array([0]))[0, 0]


def test_match_predictions_applies_every_threshold():
    iou = np.array([[0.72]])
    correct = match_predictions(iou, np.array([0]), np.array([0]))
    # true at 0.50..0.70, false from 0.75 up
    assert list(correct[0]) == [t <= 0.72 for t in IOUV]


def test_match_predictions_handles_nothing_to_match():
    assert match_predictions(np.zeros((0, 0)), np.array([]), np.array([])).shape == (0, len(IOUV))


# =============================================================================
# mask_iou
# =============================================================================
def test_mask_iou_identical_and_disjoint():
    a = np.zeros((1, 10, 10), np.uint8); a[0, 0:5, 0:5] = 1
    b = np.zeros((2, 10, 10), np.uint8); b[0, 0:5, 0:5] = 1; b[1, 5:10, 5:10] = 1
    iou = mask_iou(a, b)
    assert iou.shape == (1, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)


def test_to_mask_resolution_shrinks_and_stays_binary():
    masks = np.zeros((2, 64, 64), np.uint8)
    masks[0, 8:56, 8:56] = 1
    out = to_mask_resolution(masks, 16)
    assert out.shape == (2, 16, 16)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 1}
    assert out[0].sum() > 0 and out[1].sum() == 0


def test_to_mask_resolution_tracks_true_area_better_than_subsampling():
    """Why bilinear+threshold rather than masks[:, ::4, ::4] (besides matching ultralytics).

    On a curved boundary, strided subsampling systematically under-counts. A radius-20 disc has
    1245 pixels, so 4x down it should be about 78.
    """
    yy, xx = np.mgrid[0:64, 0:64]
    disc = (((yy - 32) ** 2 + (xx - 32) ** 2) < 20 ** 2)[None].astype(np.uint8)
    expected = disc.sum() / 16

    resized = to_mask_resolution(disc, 16).sum()
    strided = disc[:, ::4, ::4].sum()
    assert abs(resized - expected) < abs(strided - expected)


def test_to_mask_resolution_loses_structures_thinner_than_the_stride():
    """Documents a real limitation rather than papering over it.

    Scoring masks at imgsz/4 cannot represent a 3-pixel bar, whatever the operator. This applies
    to the YOLO runs too -- they score at the same resolution -- so it does not bias the
    comparison, but it does mean very thin masks score badly.
    """
    bar = np.zeros((1, 64, 64), np.uint8)
    bar[0, 30:33, 4:60] = 1
    assert to_mask_resolution(bar, 16).sum() == 0
    # an 8-pixel bar, wider than the stride, survives intact
    thick = np.zeros((1, 64, 64), np.uint8)
    thick[0, 28:36, 4:60] = 1
    assert to_mask_resolution(thick, 16).sum() > 0


def test_to_mask_resolution_accepts_torch_and_empty_input():
    assert to_mask_resolution(torch.zeros(0, 32, 32, dtype=torch.uint8), 8).shape == (0, 8, 8)
    out = to_mask_resolution(torch.ones(1, 32, 32, dtype=torch.uint8), 8)
    assert out.shape == (1, 8, 8) and out.all()


def test_mask_iou_half_overlap():
    a = np.zeros((1, 10, 10), np.uint8); a[0, 0:10, 0:4] = 1
    b = np.zeros((1, 10, 10), np.uint8); b[0, 0:10, 2:6] = 1
    assert mask_iou(a, b)[0, 0] == pytest.approx(2 / 6)


# =============================================================================
# the Evaluator
# =============================================================================
def test_perfect_predictions_score_the_ultralytics_maximum():
    evaluator = Evaluator(NAMES, with_masks=True)
    for _ in range(5):
        evaluator.add(perfect_prediction(), ground_truth())
    results = evaluator.results()

    assert results["metrics/precision(B)"] == pytest.approx(1.0)
    assert results["metrics/recall(B)"] == pytest.approx(1.0)
    assert results["metrics/mAP50(B)"] == pytest.approx(PERFECT)
    assert results["metrics/mAP50-95(B)"] == pytest.approx(PERFECT)
    assert results["metrics/mAP50(M)"] == pytest.approx(PERFECT)
    assert results["fitness"] == pytest.approx(2 * PERFECT)


def test_wrong_class_scores_nothing():
    evaluator = Evaluator(NAMES, with_masks=True)
    prediction = perfect_prediction()
    prediction["labels"] = np.array([1, 3])      # right boxes, wrong classes
    for _ in range(3):
        evaluator.add(prediction, ground_truth())
    assert evaluator.results()["metrics/mAP50(B)"] == pytest.approx(0.0)


def test_a_shifted_box_fails_the_strict_thresholds_first():
    """Shifting a 40 px box by 10 px gives IoU ~0.6, so mAP50 survives but mAP50-95 collapses."""
    truth = ground_truth()
    boxes, masks = boxes_and_masks(offsets=(10, 0))
    shifted = {"boxes": boxes, "masks": masks, "labels": truth["labels"],
               "scores": np.array([0.9, 0.8], np.float32)}
    evaluator = Evaluator(NAMES, with_masks=True)
    for _ in range(5):
        evaluator.add(shifted, truth)
    results = evaluator.results()
    assert results["metrics/mAP50(B)"] > 0.9
    assert results["metrics/mAP50-95(B)"] < results["metrics/mAP50(B)"]


def test_extra_detections_cost_precision_not_recall():
    truth = ground_truth()
    prediction = perfect_prediction()
    prediction["boxes"] = np.vstack([prediction["boxes"], [[150, 150, 190, 190]]]).astype(np.float32)
    prediction["labels"] = np.append(prediction["labels"], 0)
    prediction["scores"] = np.append(prediction["scores"], 0.3).astype(np.float32)
    prediction["masks"] = np.concatenate([prediction["masks"], np.zeros((1, 200, 200), np.uint8)])

    evaluator = Evaluator(NAMES, with_masks=True)
    for _ in range(5):
        evaluator.add(prediction, truth)
    results = evaluator.results()
    assert results["metrics/recall(B)"] == pytest.approx(1.0)
    assert results["metrics/precision(B)"] < 1.0


def test_box_only_model_reports_nan_masks_not_zero():
    """NaN says 'not measured'. Zero would read as 'measured and scored nothing'."""
    evaluator = Evaluator(NAMES, with_masks=False)
    for _ in range(3):
        evaluator.add(perfect_prediction(), ground_truth())
    results = evaluator.results()

    assert results["metrics/mAP50(B)"] == pytest.approx(PERFECT)
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        assert np.isnan(results[f"metrics/{key}(M)"])
    # and fitness falls back to the box-only rule
    assert results["fitness"] == pytest.approx(results["metrics/mAP50-95(B)"])


def test_fitness_is_the_ultralytics_formula():
    """SegmentMetrics.fitness = mAP50-95(M) + mAP50-95(B); DetMetrics.fitness = the box term.

    Checked against the YOLO run we compare with, whose best.pt records
    fitness 1.76641 = 0.89402 + 0.87239.
    """
    metrics = {"metrics/mAP50-95(B)": 0.89402, "metrics/mAP50-95(M)": 0.87239}
    assert Evaluator(NAMES, True).fitness(metrics) == pytest.approx(1.76641)
    assert Evaluator(NAMES, False).fitness(metrics) == pytest.approx(0.89402)


def test_empty_validation_does_not_explode():
    evaluator = Evaluator(NAMES, with_masks=True)
    results = evaluator.results()
    assert results["metrics/mAP50(B)"] == 0.0
    assert results["fitness"] == 0.0


def test_missed_detections_cost_recall():
    truth = ground_truth()
    prediction = perfect_prediction()
    for key in ("boxes", "labels", "scores", "masks"):
        prediction[key] = prediction[key][:1]     # only find the first of two objects
    evaluator = Evaluator(NAMES, with_masks=True)
    for _ in range(5):
        evaluator.add(prediction, truth)
    results = evaluator.results()
    assert results["metrics/mAP50(B)"] < PERFECT


# =============================================================================
# the plots this module is responsible for
# =============================================================================
def test_results_writes_the_eight_curves_and_two_matrices(tmp_path):
    evaluator = Evaluator(NAMES, with_masks=True)
    for _ in range(5):
        evaluator.add(perfect_prediction(), ground_truth())
    evaluator.results(save_dir=tmp_path, plot=True)
    evaluator.plot_confusion(tmp_path)

    expected = {f"{prefix}{curve}_curve.png"
                for prefix in ("Box", "Mask") for curve in ("P", "R", "F1", "PR")}
    expected |= {"confusion_matrix.png", "confusion_matrix_normalized.png"}
    assert expected <= {p.name for p in tmp_path.iterdir()}


def test_confusion_matrix_has_a_background_row_and_column():
    evaluator = Evaluator(NAMES, with_masks=True)
    evaluator.add(perfect_prediction(), ground_truth())
    assert evaluator.confusion.matrix.shape == (len(NAMES) + 1, len(NAMES) + 1)
