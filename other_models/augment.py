"""Data augmentation, matching what Ultralytics does to a YOLO segment dataset.

EDIT POINT 3 of 4.

Every function here takes and returns the same four things, so geometry never falls out of sync:

    img    uint8  [H, W, 3]   BGR, the way cv2 reads it
    boxes  float32[N, 4]      x1, y1, x2, y2 in pixels
    cls    int64  [N]         class id, 0-based (glass=0 ... plastic=3)
    masks  uint8  [N, H, W]   one binary mask per instance

Boxes are always *recomputed from the masks* after a geometric change (masks_to_boxes). That is
simpler than transforming box corners and gives a tighter box, which is what YOLO's segment
pipeline also ends up with.

The pipeline order matches ultralytics' v8_transforms:

    mosaic -> copy_paste -> random_perspective -> mixup / cutmix -> hsv -> flipud -> fliplr

copy_paste sits between the mosaic and the warp because that is where ultralytics inserts it
(`pre_transform.insert(1, CopyPaste(...))`), and letterbox stands in for the warp whenever there is
no warp to do.

To change the augmentation: edit a function, or reorder/disable steps in Augmenter.__call__.
To change *how strong* it is: edit the numbers in config.py instead, they are the YOLO hyps.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from ultralytics.utils.metrics import bbox_ioa

# Grey used to pad: the same value ultralytics letterboxes and warps with.
PAD = 114


# =============================================================================
# 1. HELPERS
# =============================================================================
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Tight x1,y1,x2,y2 box around each mask. Empty masks get an all-zero row."""
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return boxes


def drop_empty(boxes, cls, masks, before: np.ndarray | None = None,
               min_side: int = 2, max_aspect: float = 100.0, min_area_ratio: float = 0.01):
    """Remove instances that augmentation ruined.

    This is ultralytics' box_candidates (data/augment.py): an instance survives only if it is at
    least min_side pixels on both sides, not absurdly elongated, and -- when `before` is given,
    the same boxes prior to the warp -- still has at least min_area_ratio of the area it started
    with. Without the area test a mosaic crop leaves slivers of objects labelled as whole
    objects, which is a label-quality difference, not a cosmetic one.
    """
    if len(cls) == 0:
        return boxes, cls, masks
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_side) & (h >= min_side) & (masks.reshape(len(masks), -1).any(1))
    keep &= np.maximum(w / (h + 1e-16), h / (w + 1e-16)) < max_aspect
    if before is not None and len(before) == len(boxes):
        area_before = (before[:, 2] - before[:, 0]) * (before[:, 3] - before[:, 1])
        keep &= (w * h) / (area_before + 1e-16) > min_area_ratio
    return boxes[keep], cls[keep], masks[keep]


def _warp_masks(masks, matrix, out_w, out_h, perspective: bool) -> np.ndarray:
    """Apply one affine/perspective matrix to every instance mask, nearest-neighbour."""
    if len(masks) == 0:
        return np.zeros((0, out_h, out_w), dtype=np.uint8)
    warp = cv2.warpPerspective if perspective else cv2.warpAffine
    m = matrix if perspective else matrix[:2]
    return np.stack([
        warp(mask, m, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)
        for mask in masks
    ])


# =============================================================================
# 2. NOT AUGMENTATION -- always applied, to train and val alike
# =============================================================================
def letterbox(img, boxes, cls, masks, imgsz: int):
    """Resize so the long side is imgsz, then pad to a square imgsz x imgsz.

    This is what makes a batch stackable. Mask R-CNN can take ragged sizes, but YOLO trains on
    square letterboxed images, so we do the same and keep the geometry comparable.
    """
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2

    out = np.full((imgsz, imgsz, 3), PAD, dtype=np.uint8)
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=interp)

    out_masks = np.zeros((len(masks), imgsz, imgsz), dtype=np.uint8)
    for i, m in enumerate(masks):
        out_masks[i, top:top + nh, left:left + nw] = cv2.resize(
            m, (nw, nh), interpolation=cv2.INTER_NEAREST
        )
    return out, masks_to_boxes(out_masks), cls, out_masks


# =============================================================================
# 3. THE AUGMENTATIONS
# =============================================================================
def hsv(img, hsv_h: float, hsv_s: float, hsv_v: float):
    """Random hue/saturation/brightness shift, via a lookup table (ultralytics' RandomHSV)."""
    if not (hsv_h or hsv_s or hsv_v):
        return img
    gains = np.random.uniform(-1, 1, 3) * [hsv_h, hsv_s, hsv_v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=gains.dtype)
    lut = (
        ((x * gains[0]) % 180).astype(np.uint8),
        np.clip(x * gains[1], 0, 255).astype(np.uint8),
        np.clip(x * gains[2], 0, 255).astype(np.uint8),
    )
    merged = cv2.merge((cv2.LUT(hue, lut[0]), cv2.LUT(sat, lut[1]), cv2.LUT(val, lut[2])))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def random_perspective(img, boxes, cls, masks, out_size: int,
                       degrees: float, translate: float, scale: float,
                       shear: float, perspective: float):
    """One combined rotate / scale / shear / perspective / translate warp.

    Built as a single matrix the way ultralytics' RandomPerspective does it, so the image is
    resampled once:  M = Translate @ Shear @ Rotate+Scale @ Perspective @ Center
    The output is out_size x out_size, which is also how mosaic's 2*imgsz canvas gets cropped
    back down to imgsz.
    """
    h, w = img.shape[:2]

    center = np.eye(3)
    center[0, 2] = -w / 2
    center[1, 2] = -h / 2

    persp = np.eye(3)
    persp[2, 0] = random.uniform(-perspective, perspective)
    persp[2, 1] = random.uniform(-perspective, perspective)

    rot = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    gain = random.uniform(1 - scale, 1 + scale)
    rot[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=gain)

    shr = np.eye(3)
    shr[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    shr[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    trans = np.eye(3)
    trans[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    trans[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size

    matrix = trans @ shr @ rot @ persp @ center
    use_persp = perspective != 0

    warp = cv2.warpPerspective if use_persp else cv2.warpAffine
    img = warp(img, matrix if use_persp else matrix[:2], (out_size, out_size),
               borderValue=(PAD, PAD, PAD))
    masks = _warp_masks(masks, matrix, out_size, out_size, use_persp)
    return img, masks_to_boxes(masks), cls, masks


def flipud(img, boxes, cls, masks):
    """Mirror top to bottom."""
    img = np.ascontiguousarray(img[::-1])
    masks = np.ascontiguousarray(masks[:, ::-1])
    return img, masks_to_boxes(masks), cls, masks


def fliplr(img, boxes, cls, masks):
    """Mirror left to right."""
    img = np.ascontiguousarray(img[:, ::-1])
    masks = np.ascontiguousarray(masks[:, :, ::-1])
    return img, masks_to_boxes(masks), cls, masks


def mosaic(samples, imgsz: int):
    """Stitch 4 samples into one 2*imgsz canvas.

    `samples` is a list of 4 (img, boxes, cls, masks). The mosaic centre wanders inside the
    middle half of the canvas, so each image contributes a different amount. random_perspective
    is expected to run next and crop the canvas back down to imgsz.
    """
    s = imgsz
    xc = int(random.uniform(s * 0.5, s * 1.5))
    yc = int(random.uniform(s * 0.5, s * 1.5))

    canvas = np.full((s * 2, s * 2, 3), PAD, dtype=np.uint8)
    out_masks, out_cls = [], []

    for i, (img, _, cls, masks) in enumerate(samples):
        # scale this image so its long side is imgsz
        h0, w0 = img.shape[:2]
        r = s / max(h0, w0)
        img = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]

        # destination rectangle on the canvas, and the matching source rectangle
        if i == 0:    # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b = 0, h - (y2a - y1a)
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b = w - (x2a - x1a), 0
        else:         # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b = 0, 0
        x2b, y2b = x1b + (x2a - x1a), y1b + (y2a - y1a)

        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        for m in masks:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            placed = np.zeros((s * 2, s * 2), dtype=np.uint8)
            placed[y1a:y2a, x1a:x2a] = m[y1b:y2b, x1b:x2b]
            out_masks.append(placed)
        out_cls.append(cls)

    masks = (np.stack(out_masks) if out_masks
             else np.zeros((0, s * 2, s * 2), dtype=np.uint8))
    cls = (np.concatenate(out_cls) if out_cls else np.zeros(0, dtype=np.int64))
    return canvas, masks_to_boxes(masks), cls, masks


def copy_paste(img, boxes, cls, masks, p: float, mode: str = "flip"):
    """Duplicate some of the image's own objects, mirrored, into the space they leave free.

    This is ultralytics' CopyPaste in "flip" mode (data/augment.py), and the paper it comes from is
    "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation". Every
    instance is mirrored left-to-right; a mirrored copy is *eligible* only if it would cover less
    than 30% of every object already in the image, and the eligible ones are taken least-intrusive
    first until round(p * eligible) of them have been pasted. So `p` is a fraction of objects, not
    a probability -- at copy_paste=0.25 a quarter of the eligible mirrors land.

    Only the pixels under the pasted masks are copied, not whole rectangles, which is what makes
    this different from cutmix. The instances already present are left alone: the 30% rule is what
    keeps a paste from burying one of them.

    One deviation from ultralytics, which is working from polygons: it re-rasterises the selected
    segments with drawContours, while we already hold rasterised masks and so mirror those
    directly. Same result, without the polygon round trip.
    """
    if mode != "flip":
        raise NotImplementedError(
            f"copy_paste_mode={mode!r} is not implemented, only 'flip'. ultralytics' 'mixup' mode "
            f"takes the pasted objects from a second image instead of from this one's mirror; to "
            f"add it, extend augment.copy_paste to accept a donor sample like mixup() does. All "
            f"the configs in yolo_train_configs/ use 'flip'."
        )
    if p <= 0 or len(cls) == 0:
        return img, boxes, cls, masks

    mirrored_masks = np.ascontiguousarray(masks[:, :, ::-1])
    mirrored_boxes = masks_to_boxes(mirrored_masks)

    # ioa[i, j] is the fraction of existing object j that mirrored copy i would cover
    ioa = bbox_ioa(mirrored_boxes, boxes)
    eligible = np.nonzero((ioa < 0.30).all(1))[0]
    eligible = eligible[np.argsort(ioa.max(1)[eligible])]       # least intrusive first
    selected = eligible[:round(p * len(eligible))]
    if len(selected) == 0:
        return img, boxes, cls, masks

    paste = np.ascontiguousarray(mirrored_masks[selected].any(0).astype(np.uint8))
    img = cv2.copyTo(cv2.flip(img, 1), paste, img.copy())

    masks = np.concatenate([masks, mirrored_masks[selected]])
    cls = np.concatenate([cls, cls[selected]])
    return img, masks_to_boxes(masks), cls, masks


def mixup(a, b):
    """Blend two equally sized samples and keep every instance from both."""
    img_a, _, cls_a, masks_a = a
    img_b, _, cls_b, masks_b = b
    ratio = np.random.beta(32.0, 32.0)  # ~0.5, the ultralytics choice
    img = (img_a * ratio + img_b * (1 - ratio)).astype(np.uint8)
    masks = np.concatenate([masks_a, masks_b])
    cls = np.concatenate([cls_a, cls_b])
    return img, masks_to_boxes(masks), cls, masks


def _intersection_over_area(boxes: np.ndarray, patch) -> np.ndarray:
    """For each box, the fraction of its area covered by the patch (x1, y1, x2, y2)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    px1, py1, px2, py2 = patch
    iw = np.clip(np.minimum(boxes[:, 2], px2) - np.maximum(boxes[:, 0], px1), 0, None)
    ih = np.clip(np.minimum(boxes[:, 3], py2) - np.maximum(boxes[:, 1], py1), 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (iw * ih) / (area + 1e-16)


def cutmix(a, b, beta: float = 1.0, num_areas: int = 3):
    """Paste a rectangle of sample b over sample a, without damaging a's objects.

    Follows ultralytics' CutMix: several candidate patches are sampled and the first that covers
    none of image a's boxes is used; if every candidate would clip one, cutmix is skipped
    entirely. That matters -- pasting over an object leaves a partially erased thing still
    labelled as the whole object. Instances of b survive only if the patch keeps at least 1% of
    them, and their masks are cropped to it.
    """
    img_a, boxes_a, cls_a, masks_a = a
    img_b, boxes_b, cls_b, masks_b = b
    h, w = img_a.shape[:2]

    patch = None
    for _ in range(num_areas):
        ratio = np.random.beta(beta, beta)
        cut_w, cut_h = int(w * math.sqrt(1 - ratio)), int(h * math.sqrt(1 - ratio))
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        candidate = (max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0),
                     min(cx + cut_w // 2, w), min(cy + cut_h // 2, h))
        if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
            continue
        if _intersection_over_area(boxes_a, candidate).sum() <= 0:
            patch = candidate
            break
    if patch is None:
        return a

    x1, y1, x2, y2 = patch
    img = img_a.copy()
    img[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]

    keep_b = _intersection_over_area(boxes_b, patch) >= 0.01
    cropped_b = np.zeros_like(masks_b[keep_b])
    cropped_b[:, y1:y2, x1:x2] = masks_b[keep_b][:, y1:y2, x1:x2]

    masks = np.concatenate([masks_a, cropped_b])
    cls = np.concatenate([cls_a, cls_b[keep_b]])
    return img, masks_to_boxes(masks), cls, masks


# =============================================================================
# 4. THE PIPELINE
# =============================================================================
class Augmenter:
    """Applies the steps above with the probabilities from config.py.

    `dataset.load_raw(i)` is used to fetch the extra images mosaic and mixup/cutmix need, so the
    augmenter stays independent of how the data is stored.
    """

    def __init__(self, hyp, imgsz: int | None = None):
        self.hyp = hyp
        self.imgsz = imgsz or hyp.imgsz
        self.mosaic_open = True  # close_mosaic() flips this for the final epochs

    def close_mosaic(self) -> None:
        """Turn mosaic, copy_paste, mixup and cutmix off, as ultralytics does for the last N epochs.

        All four, not just mosaic: YOLODataset.close_mosaic sets `hyp.mosaic = hyp.copy_paste =
        hyp.mixup = hyp.cutmix = 0.0`. The final epochs are meant to see unaltered images.
        """
        self.mosaic_open = False

    def _others(self, dataset, n: int):
        return [dataset.load_raw(random.randrange(len(dataset))) for _ in range(n)]

    def __call__(self, sample, dataset):
        h = self.hyp
        img, boxes, cls, masks = sample

        # --- mosaic: 4 images on a 2*imgsz canvas, cropped back by the warp below ----
        mosaicked = False
        if self.mosaic_open and h.mosaic and random.random() < h.mosaic:
            img, boxes, cls, masks = mosaic([sample, *self._others(dataset, 3)], self.imgsz)
            mosaicked = True

        # --- copy-paste: mirrored duplicates of this image's own objects -------------
        # Before the warp, like ultralytics, so the pasted objects are warped along with the
        # originals rather than looking pasted onto a finished image.
        if self.mosaic_open and h.copy_paste:
            img, boxes, cls, masks = copy_paste(
                img, boxes, cls, masks, h.copy_paste, h.copy_paste_mode
            )

        # --- one combined geometric warp; also what crops the mosaic canvas ---------
        if mosaicked or any((h.degrees, h.translate, h.scale, h.shear, h.perspective)):
            before = boxes
            img, boxes, cls, masks = random_perspective(
                img, boxes, cls, masks, self.imgsz,
                h.degrees, h.translate, h.scale, h.shear, h.perspective,
            )
            # the warp both moves and shrinks things, so this is where the area test matters
            boxes, cls, masks = drop_empty(boxes, cls, masks, before=before)
        else:
            img, boxes, cls, masks = letterbox(img, boxes, cls, masks, self.imgsz)
            boxes, cls, masks = drop_empty(boxes, cls, masks)

        # --- mixup / cutmix: need a second sample at the same size ------------------
        if self.mosaic_open and h.mixup and random.random() < h.mixup:
            other = self._prepared(dataset)
            img, boxes, cls, masks = mixup((img, boxes, cls, masks), other)
            boxes, cls, masks = drop_empty(boxes, cls, masks)
        if self.mosaic_open and h.cutmix and random.random() < h.cutmix:
            other = self._prepared(dataset)
            img, boxes, cls, masks = cutmix((img, boxes, cls, masks), other)
            boxes, cls, masks = drop_empty(boxes, cls, masks)

        # --- colour and flips ------------------------------------------------------
        img = hsv(img, h.hsv_h, h.hsv_s, h.hsv_v)
        if h.flipud and random.random() < h.flipud:
            img, boxes, cls, masks = flipud(img, boxes, cls, masks)
        if h.fliplr and random.random() < h.fliplr:
            img, boxes, cls, masks = fliplr(img, boxes, cls, masks)

        return img, boxes, cls, masks

    def _prepared(self, dataset):
        """A second sample, letterboxed to imgsz so mixup/cutmix can combine it with the first."""
        img, boxes, cls, masks = dataset.load_raw(random.randrange(len(dataset)))
        return letterbox(img, boxes, cls, masks, self.imgsz)
