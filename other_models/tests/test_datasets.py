"""The COCO -> Mask R-CNN target contract."""
from __future__ import annotations

import numpy as np
import torch

from other_models.datasets import CocoInstanceDataset, collate_fn

IMAGE_SIZE = 96


def build(source, split="train", **kwargs):
    return CocoInstanceDataset(source, split, IMAGE_SIZE, cache=False, **kwargs)


def test_target_shapes_and_dtypes(source):
    dataset = build(source)
    image, target = dataset[0]

    assert image.dtype == torch.float32 and image.shape[0] == 3
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert target["boxes"].dtype == torch.float32 and target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    assert target["masks"].dtype == torch.uint8
    assert len(target["boxes"]) == len(target["labels"]) == len(target["masks"])
    assert target["masks"].shape[1:] == image.shape[1:]


def test_labels_are_one_based_because_zero_is_background(source):
    """torchvision reserves label 0 for background, so our 0..3 classes must become 1..4."""
    dataset = build(source)
    seen = set()
    for i in range(len(dataset)):
        labels = dataset[i][1]["labels"].tolist()
        seen.update(labels)
    assert seen and 0 not in seen
    assert seen <= {1, 2, 3, 4}
    assert seen == {1, 2}  # the fixture only uses glass and metal


def test_boxes_are_the_extent_of_their_mask(source):
    dataset = build(source)
    for i in range(len(dataset)):
        _, target = dataset[i]
        for box, mask in zip(target["boxes"].numpy(), target["masks"].numpy()):
            ys, xs = np.nonzero(mask)
            assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == tuple(box)


def test_polygons_are_detected_as_real_masks(source):
    dataset = build(source)
    assert dataset.has_real_masks is True
    mask = dataset[0][1]["masks"][0].numpy()
    # the fixture's first instance is a 24 px square at (8, 8) on a 96 px image
    assert mask.sum() > 400


def test_boxes_only_export_falls_back_to_rectangles(boxes_only_source):
    """An export with segmentation: [] must still train, and must say so.

    has_real_masks is what makes the engine write NaN into the Mask columns instead of scoring
    rectangles as if they were segmentations.
    """
    dataset = build(boxes_only_source)
    assert dataset.has_real_masks is False

    _, target = dataset[0]
    assert len(target["masks"]) == 2
    for box, mask in zip(target["boxes"].numpy(), target["masks"].numpy()):
        x1, y1, x2, y2 = box.astype(int)
        assert mask[y1:y2, x1:x2].all()          # the rectangle is filled
        assert mask.sum() == (x2 - x1) * (y2 - y1)  # and nothing outside it is


def test_letterboxing_makes_a_square_batch(source):
    dataset = build(source)
    images, targets = collate_fn([dataset[0], dataset[1]])
    assert len(images) == len(targets) == 2
    for image in images:
        assert image.shape[1] == image.shape[2] == IMAGE_SIZE


def test_cache_does_not_change_what_comes_out(source):
    """The RAM cache holds downscaled images; it must not alter a single pixel of the result."""
    uncached = CocoInstanceDataset(source, "train", IMAGE_SIZE, cache=False)
    cached = CocoInstanceDataset(source, "train", IMAGE_SIZE, cache=True)
    for i in range(len(uncached)):
        a_image, a_target = uncached[i]
        b_image, b_target = cached[i]
        assert torch.equal(a_image, b_image)
        assert torch.equal(a_target["boxes"], b_target["boxes"])
        assert torch.equal(a_target["masks"], b_target["masks"])


def test_load_raw_never_hands_out_the_cached_array(source):
    """Augmentation writes into what load_raw returns, so it must not alias the cache."""
    dataset = CocoInstanceDataset(source, "train", IMAGE_SIZE, cache=True)
    image, _, _, masks = dataset.load_raw(0)
    image[:] = 0
    masks[:] = 0
    again = dataset.load_raw(0)[0]
    assert again.any(), "load_raw returned a view of the cache and it got clobbered"
