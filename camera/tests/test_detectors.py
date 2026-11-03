"""The adapters that turn three different model outputs into the same Detection list."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from camera.segmentation_realsense import (
    ROOT,
    Detection,
    MaskRCNNDetector,
    OtherModelsDetector,
    RFDETRDetector,
    YoloDetector,
    build_detector,
    draw_masks,
    is_lfs_pointer,
    letterbox_frame,
    weights_of,
)

FRAME_H, FRAME_W = 480, 640


def frame(height=FRAME_H, width=FRAME_W):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


# --- letterboxing ------------------------------------------------------------
@pytest.mark.parametrize("imgsz", [640, 832])
@pytest.mark.parametrize("shape", [(480, 640), (640, 480), (512, 512), (300, 900)])
def test_letterbox_frame_matches_the_training_letterbox(imgsz, shape):
    """Inference must see the geometry training produced, so the two must not drift apart."""
    from other_models.augment import letterbox

    img = frame(*shape)
    mine, r, top, left = letterbox_frame(img, imgsz)
    theirs, _, _, _ = letterbox(img.copy(), np.zeros((0, 4), np.float32), np.zeros(0),
                               np.zeros((0, *shape), np.uint8), imgsz)

    assert mine.shape == (imgsz, imgsz, 3)
    np.testing.assert_array_equal(mine, theirs)
    # And the transform it reports is the one it applied.
    h, w = shape
    assert r == pytest.approx(min(imgsz / h, imgsz / w))
    assert top == (imgsz - round(h * r)) // 2
    assert left == (imgsz - round(w * r)) // 2


def test_letterbox_frame_pads_with_the_training_pad_value():
    img = frame(480, 640)
    out, _, top, _ = letterbox_frame(img, 640)
    assert (out[:top] == 114).all()          # other_models.augment.PAD


def test_letterbox_of_a_landscape_frame_at_640():
    """The numbers the box round-trip test below relies on."""
    _, r, top, left = letterbox_frame(frame(480, 640), 640)
    assert (r, top, left) == (1.0, 80, 0)


# --- Mask R-CNN output -------------------------------------------------------
def maskrcnn_output(boxes, labels, scores, masks=None):
    """What torchvision's Mask R-CNN returns for one image, in letterboxed coordinates."""
    output = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }
    if masks is not None:
        output["masks"] = torch.tensor(masks, dtype=torch.float32)
    return output


def test_maskrcnn_labels_drop_the_background_offset():
    """torchvision reserves 0 for background, so our four classes arrive as 1..4."""
    output = maskrcnn_output([[0, 80, 100, 180]] * 4, [1, 2, 3, 4], [0.9] * 4)
    detections = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    assert [d.cls for d in detections] == [0, 1, 2, 3]


def test_maskrcnn_boxes_come_back_in_frame_coordinates():
    # 640x480 letterboxed into 640x640 is r=1, top=80, left=0, so y shifts down by 80.
    output = maskrcnn_output([[100, 180, 200, 280]], [1], [0.9])
    detection, = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    assert detection.xyxy == (100, 100, 200, 200)


def test_maskrcnn_boxes_undo_a_scaling_letterbox():
    """The 832 pretrain scales as well as pads; both have to be undone."""
    _, r, top, left = letterbox_frame(frame(480, 640), 832)
    # A box that in the letterboxed image corresponds to frame (100, 100)-(200, 200).
    scaled = [[100 * r + left, 100 * r + top, 200 * r + left, 200 * r + top]]
    detection, = MaskRCNNDetector._to_detections(
        maskrcnn_output(scaled, [2], [0.9]), (FRAME_H, FRAME_W), r, top, left, 0.25)
    assert detection.xyxy == (100, 100, 200, 200)


def test_maskrcnn_applies_the_display_threshold():
    """build() leaves box_score_thresh at the training conf of 0.001; --conf acts here."""
    output = maskrcnn_output([[0, 80, 10, 90]] * 3, [1, 2, 3], [0.9, 0.3, 0.05])
    detections = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    assert [pytest.approx(d.conf, abs=1e-6) for d in detections] == [0.9, 0.3]


def test_maskrcnn_confidence_matches_its_own_detection():
    """Filtering boxes and scores separately is an easy way to pair them up wrongly."""
    output = maskrcnn_output([[0, 80, 10, 90], [100, 180, 110, 190], [200, 280, 210, 290]],
                             [1, 2, 3], [0.1, 0.9, 0.5])
    detections = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    assert [(d.cls, round(d.conf, 2)) for d in detections] == [(1, 0.9), (2, 0.5)]
    assert [d.xyxy[0] for d in detections] == [100, 200]


def test_maskrcnn_empty_output():
    output = maskrcnn_output(np.zeros((0, 4)), [], [])
    assert MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25) == []


def test_maskrcnn_everything_below_threshold():
    output = maskrcnn_output([[0, 80, 10, 90]], [1], [0.01])
    assert MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25) == []


def test_maskrcnn_boxes_are_clipped_to_the_frame():
    """A box predicted inside the grey padding would otherwise land off the frame."""
    output = maskrcnn_output([[-30, 0, 900, 639]], [1], [0.9])
    detection, = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    x1, y1, x2, y2 = detection.xyxy
    assert 0 <= x1 <= x2 <= FRAME_W - 1
    assert 0 <= y1 <= y2 <= FRAME_H - 1


def test_maskrcnn_masks_land_on_the_frame():
    masks = np.zeros((1, 1, 640, 640), dtype=np.float32)
    masks[0, 0, 180:280, 100:200] = 1.0          # letterboxed; frame y is 80 less
    detection, = MaskRCNNDetector._to_detections(
        maskrcnn_output([[100, 180, 200, 280]], [1], [0.9], masks),
        (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)

    assert detection.mask is not None
    assert detection.mask.shape == (FRAME_H, FRAME_W)
    assert detection.mask.dtype == bool
    assert detection.mask[150, 150]              # inside
    assert not detection.mask[50, 50]            # outside


def test_maskrcnn_without_a_mask_head():
    """fasterrcnn_v2 is in the models registry and returns no masks at all."""
    output = maskrcnn_output([[100, 180, 200, 280]], [1], [0.9])
    detection, = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    assert detection.mask is None


def test_masks_are_filtered_before_the_host_copy(monkeypatch):
    """RF-DETR returns max_det masks every frame; copying the discarded ones is pure cost.

    ~30 MB of device-to-host traffic per frame, inside run_frames' timed region, for planes
    that are thrown away one line later. The filter has to happen on the device.
    """
    shapes = []
    real_cpu = torch.Tensor.cpu

    def recording_cpu(self, *args, **kwargs):
        shapes.append(tuple(self.shape))
        return real_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", recording_cpu)

    masks = np.zeros((5, 1, 640, 640), dtype=np.float32)
    output = maskrcnn_output([[0, 80, 10, 90]] * 5, [1] * 5, [0.9, 0.01, 0.8, 0.02, 0.03], masks)
    detections = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)

    assert len(detections) == 2
    assert (2, 1, 640, 640) in shapes, "the kept masks should be the ones copied"
    assert (5, 1, 640, 640) not in shapes, "all five masks were copied, then filtered"


# --- RF-DETR output ----------------------------------------------------------
# Same dict as Mask R-CNN's (rfdetr_adapter.RFDETRAdapter exists to make that true), with one
# difference: PostProcess is built with upsample_masks_to_image_size=False, so the mask planes
# arrive at imgsz // mask_downsample_ratio rather than at the letterbox size.
MASK_RATIO = 4          # other_models.metrics.MASK_RATIO, and rfdetr's mask_downsample_ratio


def rfdetr_output(boxes, labels, scores, masks=None):
    return maskrcnn_output(boxes, labels, scores, masks)


@pytest.mark.parametrize("imgsz", [648, 840])
def test_rfdetr_quarter_resolution_masks_land_on_their_own_box(imgsz):
    """The whole point of the mask scale: a head-resolution plane must not be read as a
    letterbox-resolution one, which would crop the wrong rows and slide the mask off its box."""
    _, r, top, left = letterbox_frame(frame(), imgsz)
    side = imgsz // MASK_RATIO

    # The frame box (100, 100)-(200, 200), expressed in letterboxed and then in mask-head
    # coordinates.
    box = [100 * r + left, 100 * r + top, 200 * r + left, 200 * r + top]
    plane = np.zeros((1, 1, side, side), dtype=np.float32)
    x0, y0, x1, y1 = (int(v / MASK_RATIO) for v in box)
    plane[0, 0, y0:y1, x0:x1] = 1.0

    detection, = OtherModelsDetector._to_detections(
        rfdetr_output([box], [1], [0.9], plane), (FRAME_H, FRAME_W), r, top, left, 0.25,
        imgsz=imgsz)

    assert detection.xyxy == (100, 100, 200, 200)
    assert detection.mask.shape == (FRAME_H, FRAME_W)
    assert detection.mask.dtype == bool
    assert detection.mask[150, 150]              # inside the box
    assert not detection.mask[50, 50]            # outside it
    assert not detection.mask[300, 400]

    # And it sits on the box, give or take the resolution it was predicted at. One mask-head
    # pixel is MASK_RATIO letterbox pixels, which is MASK_RATIO / r frame pixels.
    slop = 2 * MASK_RATIO / r
    ys, xs = np.nonzero(detection.mask)
    assert xs.min() == pytest.approx(100, abs=slop)
    assert ys.min() == pytest.approx(100, abs=slop)
    assert xs.max() == pytest.approx(200, abs=slop)
    assert ys.max() == pytest.approx(200, abs=slop)


def test_passing_imgsz_does_not_change_a_full_resolution_mask():
    """Mask R-CNN pastes to the letterbox size, so its scale is 1 and nothing may move."""
    masks = np.zeros((1, 1, 640, 640), dtype=np.float32)
    masks[0, 0, 180:280, 100:200] = 1.0
    output = maskrcnn_output([[100, 180, 200, 280]], [1], [0.9], masks)

    without = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)
    with_imgsz = MaskRCNNDetector._to_detections(output, (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25,
                                                 imgsz=640)
    np.testing.assert_array_equal(without[0].mask, with_imgsz[0].mask)


# --- ultralytics output ------------------------------------------------------
class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.asarray(xyxy, dtype=np.float32)]
        self.cls = [cls]
        self.conf = [conf]


class FakeTensor:
    """Just enough of a torch tensor for .cpu().numpy()."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeMasks:
    def __init__(self, array):
        self.data = FakeTensor(array)


class FakeResults:
    def __init__(self, boxes, masks=None, orig_shape=(FRAME_H, FRAME_W)):
        self.boxes = boxes
        self.masks = FakeMasks(masks) if masks is not None else None
        self.orig_shape = orig_shape


def test_yolo_detections():
    result = FakeResults([FakeBox([10, 20, 110, 220], 2, 0.87),
                          FakeBox([0, 0, 5, 5], 0, 0.31)])
    detections = YoloDetector._to_detections(result)
    assert [d.xyxy for d in detections] == [(10, 20, 110, 220), (0, 0, 5, 5)]
    assert [d.cls for d in detections] == [2, 0]
    assert detections[0].conf == pytest.approx(0.87)


def test_yolo_no_detections():
    assert YoloDetector._to_detections(FakeResults([])) == []


def test_yolo_handles_a_missing_boxes_attribute():
    """A classification-only Results has boxes=None."""
    assert YoloDetector._to_detections(FakeResults(None)) == []


def test_yolo_masks_land_on_their_own_box():
    """retina_masks hands back frame-sized masks, so a mask must sit where its box is.

    The previous version of this test asserted only .shape and .dtype, and passed with its own
    fixture mask 50 px above its own box -- which is exactly the padding bug it was meant to
    catch.
    """
    masks = np.zeros((1, FRAME_H, FRAME_W), dtype=np.float32)
    masks[0, 100:200, 100:200] = 1.0
    detection, = YoloDetector._to_detections(
        FakeResults([FakeBox([100, 100, 200, 200], 1, 0.9)], masks))

    assert detection.mask.shape == (FRAME_H, FRAME_W)
    assert detection.mask.dtype == bool
    assert detection.mask[150, 150]              # inside the box
    assert not detection.mask[50, 50]            # outside it
    assert not detection.mask[250, 250]
    # And the mask's bounding box is the detection's box, give or take the 0.5 threshold.
    ys, xs = np.nonzero(detection.mask)
    assert (xs.min(), ys.min()) == (100, 100)
    assert (xs.max(), ys.max()) == (199, 199)


def test_yolo_mask_count_matches_the_box_count():
    masks = np.zeros((2, FRAME_H, FRAME_W), dtype=np.float32)
    masks[0, 10:20, 10:20] = 1.0
    masks[1, 300:320, 300:320] = 1.0
    detections = YoloDetector._to_detections(
        FakeResults([FakeBox([10, 10, 20, 20], 0, 0.9),
                     FakeBox([300, 300, 320, 320], 1, 0.8)], masks))
    assert len(detections) == 2
    assert detections[0].mask[15, 15] and not detections[0].mask[310, 310]
    assert detections[1].mask[310, 310] and not detections[1].mask[15, 15]


def test_yolo_requests_retina_masks(monkeypatch):
    """The padding crop is ultralytics' job via process_mask_native; the flag is what asks for it."""
    captured = {}

    class FakeYolo:
        names = {0: "glass"}

        def __call__(self, frame, **kwargs):
            captured.update(kwargs)
            return [FakeResults([])]

    detector = YoloDetector.__new__(YoloDetector)
    detector.model = FakeYolo()
    detector.conf = 0.4
    detector.names = dict(FakeYolo.names)

    detector.predict(frame())
    assert captured["retina_masks"] is True
    assert captured["conf"] == 0.4


def test_yolo_boxes_are_clipped_to_the_frame():
    """clip_boxes clamps inclusively, so an edge box yields a centre one pixel off the frame."""
    detection, = YoloDetector._to_detections(
        FakeResults([FakeBox([600, 400, FRAME_W, FRAME_H], 1, 0.9)]))
    x1, y1, x2, y2 = detection.xyxy
    assert x2 <= FRAME_W - 1 and y2 <= FRAME_H - 1
    # The centre is what annotate() hands to depth_frame.get_distance().
    assert 0 <= (x1 + x2) // 2 < FRAME_W
    assert 0 <= (y1 + y2) // 2 < FRAME_H


def test_yolo_without_masks_is_a_detect_checkpoint():
    detection, = YoloDetector._to_detections(FakeResults([FakeBox([1, 2, 3, 4], 0, 0.5)]))
    assert detection.mask is None


# --- both adapters agree on the shape of their output ------------------------
def test_both_adapters_produce_the_same_type():
    """The live loop never branches on the model, so the two must be interchangeable."""
    yolo = YoloDetector._to_detections(FakeResults([FakeBox([100, 100, 200, 200], 1, 0.9)]))
    maskrcnn = MaskRCNNDetector._to_detections(
        maskrcnn_output([[100, 180, 200, 280]], [2], [0.9]),
        (FRAME_H, FRAME_W), 1.0, 80, 0, 0.25)

    for detections in (yolo, maskrcnn):
        detection, = detections
        assert isinstance(detection, Detection)
        assert detection.xyxy == (100, 100, 200, 200)
        assert detection.cls == 1
        assert isinstance(detection.conf, float)


# --- the mask overlay --------------------------------------------------------
def box_mask(y0, y1, x0, x1):
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_draw_masks_tints_inside_and_leaves_outside_alone():
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    draw_masks(canvas, [Detection((100, 100, 200, 200), 1, 0.9, box_mask(100, 200, 100, 200))])

    assert canvas[150, 150].any(), "a masked pixel should be tinted"
    assert not canvas[50, 50].any(), "an unmasked pixel must be untouched"


def test_draw_masks_is_translucent_not_opaque():
    """A solid fill would hide the object; the point is to see the frame through the tint."""
    canvas = np.full((FRAME_H, FRAME_W, 3), 200, dtype=np.uint8)
    draw_masks(canvas, [Detection((0, 0, 50, 50), 0, 0.9, box_mask(0, 50, 0, 50))])

    from ultralytics.utils.plotting import colors
    tint = np.array(colors(0, bgr=True), dtype=float)
    blended = canvas[25, 25].astype(float)
    assert not np.allclose(blended, tint), "fully opaque"
    assert not np.allclose(blended, 200), "not blended at all"


def test_draw_masks_uses_the_training_palette():
    """Same colours as the run's val_batch*_pred.jpg, so a class looks the same in both."""
    from ultralytics.utils.plotting import colors

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    draw_masks(canvas, [Detection((0, 0, 50, 50), 2, 0.9, box_mask(0, 50, 0, 50))])

    expected = np.array(colors(2, bgr=True), dtype=float) * 0.45
    np.testing.assert_allclose(canvas[25, 25].astype(float), expected, atol=1.5)


def test_draw_masks_separates_classes_by_colour():
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    draw_masks(canvas, [
        Detection((0, 0, 50, 50), 0, 0.9, box_mask(0, 50, 0, 50)),
        Detection((100, 100, 150, 150), 3, 0.9, box_mask(100, 150, 100, 150)),
    ])
    assert not np.array_equal(canvas[25, 25], canvas[125, 125])


def test_draw_masks_ignores_detections_without_one():
    """A detect-only checkpoint has no masks; the overlay must simply not happen."""
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    draw_masks(canvas, [Detection((0, 0, 50, 50), 0, 0.9, None)])
    assert not canvas.any()


def test_draw_masks_on_an_empty_list():
    canvas = np.full((FRAME_H, FRAME_W, 3), 7, dtype=np.uint8)
    draw_masks(canvas, [])
    assert (canvas == 7).all()


def test_draw_masks_blends_once_for_many_detections(monkeypatch):
    """Per-detection blending costs the whole frame N times over for the same picture."""
    calls = []
    real = cv2.addWeighted
    monkeypatch.setattr(cv2, "addWeighted",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    draw_masks(canvas, [Detection((0, 0, 10, 10), i % 4, 0.9, box_mask(i * 10, i * 10 + 10, 0, 10))
                        for i in range(6)])
    assert len(calls) == 1


# --- the real checkpoints ----------------------------------------------------
# These are the only tests that need the trained runs. Both of these runs are real files in a
# normal working tree; an unfetched LFS stub skips rather than fails.
MASKRCNN_RUN = "runs/segment/maskrcnn/taco/taco-finetune-augment"
YOLO_RUN = "runs/segment/runs/segment/taco-yolo26-seg/taco-finetune-nms_augment_v"
RFDETR_RUN = "runs/segment/rfdetr/finetune_augment_v-3"


def usable(run_dir):
    weights = weights_of(run_dir)
    if not weights.exists():
        pytest.skip(f"{run_dir} is not in this working tree")
    if is_lfs_pointer(weights):
        pytest.skip(f"{run_dir}/weights/best.pt is an unfetched git LFS stub")
    return weights


@pytest.mark.slow
def test_maskrcnn_detector_loads_and_runs():
    weights = usable(MASKRCNN_RUN)
    detector = MaskRCNNDetector(weights, conf=0.25, device="cpu")

    assert set(detector.names.values()) == {"glass", "metal", "paper", "plastic"}
    # The run trained at imgsz 640 and the detector must letterbox to the same size.
    assert detector.imgsz == 640
    # --conf has to reach box_score_thresh, not just the post-filter.
    assert detector.hyp.conf == pytest.approx(0.25)

    detections = detector.predict(frame())
    assert isinstance(detections, list)
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        assert 0 <= x1 <= x2 < FRAME_W and 0 <= y1 <= y2 < FRAME_H
        assert detection.cls in range(len(detector.names))
        assert 0.25 <= detection.conf <= 1.0
        if detection.mask is not None:
            assert detection.mask.shape == (FRAME_H, FRAME_W)


@pytest.mark.slow
def test_yolo_detector_loads_and_runs():
    detector = YoloDetector(usable(YOLO_RUN), conf=0.25)
    assert set(detector.names.values()) == {"glass", "metal", "paper", "plastic"}

    detections = detector.predict(frame())
    assert isinstance(detections, list)
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        assert 0 <= x1 <= x2 <= FRAME_W and 0 <= y1 <= y2 <= FRAME_H
        assert detection.cls in range(len(detector.names))
        assert detection.conf >= 0.25


@pytest.mark.slow
def test_rfdetr_detector_loads_and_runs():
    weights = usable(RFDETR_RUN)
    detector = RFDETRDetector(weights, conf=0.25, device="cpu")

    assert set(detector.names.values()) == {"glass", "metal", "paper", "plastic"}
    # 648, not 640: RF-DETR's resolution must be a multiple of patch_size * num_windows.
    assert detector.imgsz == 648

    detections = detector.predict(frame())
    assert isinstance(detections, list)
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        assert 0 <= x1 <= x2 < FRAME_W and 0 <= y1 <= y2 < FRAME_H
        assert detection.cls in range(len(detector.names))
        assert 0.25 <= detection.conf <= 1.0
        if detection.mask is not None:
            assert detection.mask.shape == (FRAME_H, FRAME_W)


@pytest.mark.slow
@pytest.mark.parametrize("detector_class,run", [
    (MaskRCNNDetector, RFDETR_RUN),
    (RFDETRDetector, MASKRCNN_RUN),
])
def test_the_wrong_family_is_refused_rather_than_built(detector_class, run):
    """--model and --weights disagreeing must stop the run, not produce a mislabelled one.

    Both halves of this are failure modes that do not announce themselves:
    models.load_checkpoint prints `loaded 0/307 tensors` and returns, so a Mask R-CNN built
    for an RF-DETR run would serve random weights and log max_det confident-looking
    detections a frame under the name "maskrcnn"; and the reverse gets as far as RF-DETR's
    resolution check, which complains that imgsz 640 is not divisible by 24 -- true, and no
    help at all in working out what actually went wrong.
    """
    weights = usable(run)
    with pytest.raises(SystemExit) as error:
        detector_class(weights, conf=0.25, device="cpu")
    message = str(error.value)
    assert "--model" in message
    assert detector_class.family in message


@pytest.mark.slow
def test_rfdetr_downloads_nothing(monkeypatch):
    """models.build is given the run's own best.pt, so load_pretrain_weights is never reached.

    If it were, rfdetr would fetch its 134 MB published COCO checkpoint -- the failure mode
    rfdetr_adapter.model_config has a long comment about.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the published COCO checkpoint must not be fetched")

    monkeypatch.setattr("other_models.rfdetr_adapter.load_pretrain_weights", explode)
    RFDETRDetector(usable(RFDETR_RUN), conf=0.25, device="cpu")


@pytest.mark.slow
def test_build_detector_dispatches_on_architecture():
    assert isinstance(build_detector(usable(YOLO_RUN), "yolo26", 0.25), YoloDetector)
    assert isinstance(build_detector(usable(YOLO_RUN), "yolo8", 0.25), YoloDetector)
    assert isinstance(build_detector(usable(MASKRCNN_RUN), "maskrcnn", 0.25), MaskRCNNDetector)
    assert isinstance(build_detector(usable(RFDETR_RUN), "rfdetr", 0.25), RFDETRDetector)


@pytest.mark.slow
def test_a_custom_weights_path_loads_the_same_model():
    """--weights <run dir> must end up at the same checkpoint the registry would pick."""
    from camera.segmentation_realsense import build_parser, resolve_weights

    usable(MASKRCNN_RUN)
    args = build_parser().parse_args(["--weights", str(ROOT / MASKRCNN_RUN)])
    weights, architecture, dataset = resolve_weights(args)

    assert weights == weights_of(MASKRCNN_RUN)
    assert architecture == "maskrcnn"
    assert dataset == "custom"
    assert isinstance(build_detector(weights, architecture, 0.25), MaskRCNNDetector)
