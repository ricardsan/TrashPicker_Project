"""The augmentation pipeline.

The invariant worth protecting is that image, boxes and masks never drift apart. Every geometric
op recomputes boxes from the masks it just warped, so "box == extent of its mask" should hold
after any of them -- that is what most of these tests assert.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from other_models.augment import (Augmenter, copy_paste, cutmix, drop_empty, fliplr, flipud, hsv,
                            letterbox, masks_to_boxes, mixup, mosaic, random_perspective)
from other_models.datasets import CocoInstanceDataset

IMAGE_SIZE = 96


def assert_boxes_match_masks(boxes, masks):
    assert len(boxes) == len(masks)
    for box, mask in zip(boxes, masks):
        ys, xs = np.nonzero(mask)
        assert len(xs), "an empty mask should have been dropped"
        assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == tuple(box)


# =============================================================================
# the individual operations
# =============================================================================
def test_masks_to_boxes(sample):
    _, boxes, _, masks = sample
    assert_boxes_match_masks(boxes, masks)
    assert tuple(boxes[0]) == (8, 8, 32, 32)


def test_letterbox_squares_the_image_and_keeps_geometry():
    img = np.full((40, 80, 3), 10, dtype=np.uint8)
    masks = np.zeros((1, 40, 80), dtype=np.uint8)
    masks[0, 10:30, 20:60] = 1
    cls = np.array([0])
    out, boxes, cls, out_masks = letterbox(img, masks_to_boxes(masks), cls, masks, 64)

    assert out.shape == (64, 64, 3)
    assert out_masks.shape == (1, 64, 64)
    assert_boxes_match_masks(boxes, out_masks)
    # 80 -> 64 is a 0.8 scale, and the content is centred vertically
    assert boxes[0][0] == pytest.approx(16, abs=2)


def test_flips_are_involutive(sample):
    img, boxes, cls, masks = sample
    for flip in (fliplr, flipud):
        once = flip(img, boxes, cls, masks)
        twice = flip(*once)
        assert np.array_equal(twice[0], img)
        assert np.array_equal(twice[3], masks)
        assert_boxes_match_masks(once[1], once[3])


def test_fliplr_mirrors_the_box(sample):
    img, boxes, cls, masks = sample
    _, flipped, _, _ = fliplr(img, boxes, cls, masks)
    width = img.shape[1]
    assert flipped[0][0] == width - boxes[0][2]
    assert flipped[0][2] == width - boxes[0][0]


def test_hsv_with_zero_gains_is_a_no_op(sample):
    img = sample[0]
    assert np.array_equal(hsv(img, 0.0, 0.0, 0.0), img)


def test_hsv_changes_colour_but_not_shape(sample):
    random.seed(0)
    np.random.seed(0)
    img = sample[0]
    out = hsv(img, 0.5, 0.9, 0.9)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_random_perspective_keeps_boxes_on_masks(sample):
    random.seed(0)
    img, boxes, cls, masks = sample
    for _ in range(10):
        out, out_boxes, out_cls, out_masks = random_perspective(
            img, boxes, cls, masks, IMAGE_SIZE,
            degrees=90.0, translate=0.2, scale=0.5, shear=2.5, perspective=0.0001,
        )
        out_boxes, out_cls, out_masks = drop_empty(out_boxes, out_cls, out_masks)
        assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert_boxes_match_masks(out_boxes, out_masks)
        assert len(out_cls) == len(out_boxes)


def test_random_perspective_with_no_rotation_preserves_class_order(sample):
    random.seed(1)
    img, boxes, cls, masks = sample
    _, out_boxes, out_cls, out_masks = random_perspective(
        img, boxes, cls, masks, IMAGE_SIZE,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    )
    assert list(out_cls) == list(cls)
    assert_boxes_match_masks(out_boxes, out_masks)


def test_mosaic_draws_from_four_images(sample):
    random.seed(0)
    samples = [sample] * 4
    canvas, boxes, cls, masks = mosaic(samples, IMAGE_SIZE)
    assert canvas.shape == (IMAGE_SIZE * 2, IMAGE_SIZE * 2, 3)
    assert len(cls) == 8                       # 2 instances from each of 4 images
    boxes, cls, masks = drop_empty(boxes, cls, masks)
    assert_boxes_match_masks(boxes, masks)


def test_mixup_keeps_every_instance(sample):
    a = letterbox(*sample, IMAGE_SIZE)
    b = letterbox(*sample, IMAGE_SIZE)
    img, boxes, cls, masks = mixup(a, b)
    assert len(cls) == len(a[2]) + len(b[2])
    assert img.shape == a[0].shape
    boxes, cls, masks = drop_empty(boxes, cls, masks)
    assert_boxes_match_masks(boxes, masks)


def test_cutmix_crops_the_pasted_instances(sample):
    random.seed(3)
    a = letterbox(*sample, IMAGE_SIZE)
    b = letterbox(*sample, IMAGE_SIZE)
    img, boxes, cls, masks = cutmix(a, b)
    boxes, cls, masks = drop_empty(boxes, cls, masks)
    assert img.shape == a[0].shape
    assert_boxes_match_masks(boxes, masks)
    # nothing may be invented: at most the two originals plus the two pasted
    assert len(cls) <= len(a[2]) + len(b[2])


def test_copy_paste_appends_the_mirror_of_each_object(sample):
    """The fixture's two objects are both clear of their own mirror image, so both get pasted.

    copy_paste draws no random numbers at all -- unlike every other augmentation here, what it
    does is decided entirely by `p` and the geometry, which is also true of ultralytics' CopyPaste
    in flip mode.
    """
    img, boxes, cls, masks = copy_paste(*sample, p=1.0)
    assert len(cls) == 4
    assert list(cls) == [0, 1, 0, 1]                      # the mirrors keep their classes
    assert_boxes_match_masks(boxes, masks)
    assert np.array_equal(masks[:2], sample[3]), "the originals must be left alone"

    # instance 0 sits at rows 8:32, cols 8:32, so its mirror lands at cols 64:88 in a 96-wide image
    assert tuple(boxes[2]) == (64, 8, 88, 32)
    # and the pixels there are the real pixels of instance 0, not a grey rectangle
    assert np.array_equal(img[8:32, 64:88], sample[0][8:32, 8:32])
    # nothing outside the pasted masks is touched
    pasted = masks[2:].any(0) > 0
    assert np.array_equal(img[~pasted], sample[0][~pasted])


def test_copy_paste_pastes_a_fraction_of_the_eligible_objects():
    """`p` is a fraction of objects, not a probability: round(p * eligible) of them land."""
    masks = np.zeros((4, 96, 96), dtype=np.uint8)
    for i in range(4):
        # four small objects stacked down the left-hand side, so their mirrors land down the
        # right-hand side and overlap nothing. (Objects on the centre line mirror onto
        # themselves and are never eligible -- which is correct, and not what is being tested.)
        masks[i, 4 + i * 24:20 + i * 24, 10:26] = 1
    sample = (np.zeros((96, 96, 3), np.uint8), masks_to_boxes(masks),
              np.arange(4, dtype=np.int64), masks)

    assert len(copy_paste(*sample, p=1.0)[2]) == 4 + 4
    assert len(copy_paste(*sample, p=0.5)[2]) == 4 + 2
    assert len(copy_paste(*sample, p=0.1)[2]) == 4         # round(0.4) -> nothing to paste


def test_copy_paste_refuses_to_bury_an_existing_object():
    """The 30% rule: a mirror that would cover an object already in the image is not eligible.

    Both objects here are each other's mirror, so every candidate would land squarely on the
    other one and nothing may be pasted.
    """
    masks = np.zeros((2, 96, 96), dtype=np.uint8)
    masks[0, 10:40, 10:40] = 1
    masks[1, 10:40, 56:86] = 1                             # exactly the mirror of masks[0]
    img = np.zeros((96, 96, 3), np.uint8)
    cls = np.array([0, 1], dtype=np.int64)

    out = copy_paste(img, masks_to_boxes(masks), cls, masks, p=1.0)
    assert len(out[2]) == 2
    assert np.array_equal(out[0], img) and np.array_equal(out[3], masks)


def test_copy_paste_with_p_zero_or_no_objects_is_a_no_op(sample):
    img, boxes, cls, masks = sample
    assert copy_paste(img, boxes, cls, masks, p=0.0)[2] is cls
    empty = (img, np.zeros((0, 4), np.float32), np.zeros(0, np.int64),
             np.zeros((0, 96, 96), np.uint8))
    assert len(copy_paste(*empty, p=1.0)[2]) == 0


def test_copy_paste_rejects_the_mode_it_does_not_implement(sample):
    with pytest.raises(NotImplementedError, match="mixup"):
        copy_paste(*sample, p=1.0, mode="mixup")


def test_drop_empty_removes_vanished_instances(sample):
    _, boxes, cls, masks = sample
    masks = masks.copy()
    masks[1] = 0                                    # instance 1 disappears
    boxes = masks_to_boxes(masks)
    boxes, cls, masks = drop_empty(boxes, cls, masks)
    assert len(cls) == 1 and cls[0] == 0


# =============================================================================
# the pipeline
# =============================================================================
def test_augmenter_is_deterministic_for_a_fixed_seed(source, hyp):
    augmented = hyp.replace(mosaic=1.0, mixup=0.5, cutmix=0.3, degrees=90.0, shear=2.5)
    outputs = []
    for _ in range(2):
        random.seed(11)
        np.random.seed(11)
        dataset = CocoInstanceDataset(source, "train", IMAGE_SIZE,
                                      Augmenter(augmented, IMAGE_SIZE), cache=False)
        image, target = dataset[0]
        outputs.append((image.numpy(), target["boxes"].numpy(), target["labels"].numpy()))

    for a, b in zip(*outputs):
        assert np.array_equal(a, b)


def test_augmenter_keeps_the_contract_across_many_draws(source, hyp):
    """The real test: run the full aggressive pipeline repeatedly and check nothing drifts."""
    random.seed(5)
    np.random.seed(5)
    aggressive = hyp.replace(mosaic=1.0, copy_paste=0.5, mixup=0.5, cutmix=0.3, degrees=90.0,
                             translate=0.2, scale=0.5, shear=2.5, perspective=0.0001,
                             flipud=0.5, fliplr=0.5)
    dataset = CocoInstanceDataset(source, "train", IMAGE_SIZE,
                                  Augmenter(aggressive, IMAGE_SIZE), cache=False)
    for _ in range(25):
        image, target = dataset[random.randrange(len(dataset))]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert_boxes_match_masks(target["boxes"].numpy(), target["masks"].numpy())
        assert set(target["labels"].tolist()) <= {1, 2, 3, 4}


def test_close_mosaic_disables_exactly_mosaic_copy_paste_mixup_and_cutmix(source, hyp, monkeypatch):
    """ultralytics zeroes mosaic, copy_paste, mixup and cutmix for the last N epochs.

    All four: YOLODataset.close_mosaic sets every one of them to 0.0, so leaving copy_paste on
    would keep altering the images through the epochs that are meant to see them untouched.
    """
    augmenter = Augmenter(hyp.replace(mosaic=1.0, copy_paste=1.0, mixup=1.0, cutmix=1.0),
                          IMAGE_SIZE)
    dataset = CocoInstanceDataset(source, "train", IMAGE_SIZE, augmenter, cache=False)

    called = set()
    import other_models.augment as module
    for name in ("mosaic", "copy_paste", "mixup", "cutmix"):
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            called.add(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    random.seed(0)
    dataset[0]
    assert called == {"mosaic", "copy_paste", "mixup", "cutmix"}

    called.clear()
    augmenter.close_mosaic()
    for _ in range(5):
        dataset[0]
    assert called == set()


def count_instances_through_loader(dataset, workers: int) -> int:
    """Total instances over one pass of a DataLoader, which is how training really reads data."""
    from torch.utils.data import DataLoader
    from other_models.datasets import collate_fn
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=workers,
                        persistent_workers=workers > 0)
    return sum(len(t["labels"]) for _, targets in loader for t in targets)


def test_close_mosaic_reaches_the_dataloader_workers(source, hyp):
    """Flipping the flag is not enough: persistent workers hold their own copy of the dataset.

    Without rebuilding the loader this passes in-process and fails in production, so the whole
    point of the test is that it goes through a DataLoader with workers.
    """
    from other_models.engine import Trainer
    from other_models.models import MODELS, ModelSpec

    augmenter = Augmenter(hyp.replace(mosaic=1.0, copy_paste=1.0), IMAGE_SIZE)
    dataset = CocoInstanceDataset(source, "train", IMAGE_SIZE, augmenter, cache=False)
    plain = len(dataset) * 2            # the fixture puts exactly 2 instances in every image

    trainer = Trainer.__new__(Trainer)
    trainer.hyp = hyp.replace(workers=2)
    trainer.device = torch.device("cpu")
    trainer.augmenter = augmenter
    trainer.train_set = dataset
    trainer.train_loader = trainer._build_train_loader()

    def instances() -> int:
        return sum(len(t["labels"]) for _, targets in trainer.train_loader for t in targets)

    # Iterate FIRST, so the persistent workers are actually spawned while mosaic is still on and
    # each one pickles an Augmenter with mosaic_open=True. Closing mosaic before the first pass
    # would hide the bug entirely, which is what an earlier version of this test did.
    random.seed(0)
    before = instances()
    assert before > plain, "mosaic should be pulling in instances from other images"

    trainer.close_mosaic()
    random.seed(0)
    after = instances()
    assert after == plain, (
        f"close_mosaic did not reach the workers: {after} instances, expected {plain}"
    )


def test_augmenter_still_produces_images_after_close_mosaic(source, hyp):
    augmenter = Augmenter(hyp.replace(mosaic=1.0, copy_paste=1.0, mixup=1.0, cutmix=1.0),
                          IMAGE_SIZE)
    augmenter.close_mosaic()
    dataset = CocoInstanceDataset(source, "train", IMAGE_SIZE, augmenter, cache=False)
    image, target = dataset[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert_boxes_match_masks(target["boxes"].numpy(), target["masks"].numpy())
