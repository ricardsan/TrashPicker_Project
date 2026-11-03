"""Shared fixtures. Everything here is synthetic, CPU-only and fast.

The tests that touch the real datasets live in test_prepare_data.py and skip themselves when the
files are not present, so the suite still runs on a fresh clone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from other_models.datasets import Source  # noqa: E402

# Two of the four real classes, so class ids stay meaningful (glass=0, metal=1).
N_IMAGES = 6
IMAGE_SIZE = 96


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs a real training run; deselect with -m 'not slow'")


def square(x: int, y: int, size: int) -> np.ndarray:
    """A polygon ring, as COCO stores it: [x1, y1, x2, y2, ...]."""
    return [float(v) for v in
            (x, y, x + size, y, x + size, y + size, x, y + size)]


def _write_split(directory: Path, split: str, n_images: int, with_polygons: bool) -> None:
    images_dir = directory / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []

    for i in range(n_images):
        name = f"{split}_{i:03d}.png"
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
        # two instances per image, placed so they never touch
        boxes = [(8, 8, 24), (52, 56, 28)]
        for n, (x, y, size) in enumerate(boxes):
            cv2.rectangle(image, (x, y), (x + size, y + size),
                          (200, 120, 60) if n == 0 else (60, 200, 120), -1)
        cv2.imwrite(str(images_dir / name), image)

        images.append({"id": i, "file_name": name,
                       "width": IMAGE_SIZE, "height": IMAGE_SIZE})
        for n, (x, y, size) in enumerate(boxes):
            annotations.append({
                "id": len(annotations),
                "image_id": i,
                "category_id": n,                       # 0 = glass, 1 = metal
                "segmentation": [square(x, y, size)] if with_polygons else [],
                "bbox": [float(x), float(y), float(size), float(size)],
                "area": float(size * size),
                "iscrowd": 0,
            })

    annotations_dir = directory / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"{split}.json").write_text(json.dumps({
        "info": {"description": f"synthetic {split} split for the other_models/ tests"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "glass"}, {"id": 1, "name": "metal"},
                       {"id": 2, "name": "paper"}, {"id": 3, "name": "plastic"}],
    }))


def _make_source(tmp_path: Path, with_polygons: bool) -> Source:
    for split, n in (("train", N_IMAGES), ("val", 2)):
        _write_split(tmp_path, split, n, with_polygons)
    return Source(
        annotations=tmp_path / "annotations" / "{split}.json",
        images=tmp_path / "images" / "{split}",
        splits=("train", "val"),
    )


@pytest.fixture
def source(tmp_path) -> Source:
    """A tiny dataset with real polygons."""
    return _make_source(tmp_path, with_polygons=True)


@pytest.fixture
def boxes_only_source(tmp_path) -> Source:
    """The same dataset with segmentation: [], like the 2026-10-05 Label Studio export."""
    return _make_source(tmp_path, with_polygons=False)


@pytest.fixture
def hyp():
    """A preset shrunk to something that trains in seconds on a CPU."""
    from other_models.config import FINETUNE_NMS
    return FINETUNE_NMS.replace(
        epochs=2, batch=2, imgsz=IMAGE_SIZE, workers=0, nbs=2, amp=False, cache=False,
        warmup_epochs=0.0, close_mosaic=0, mosaic=0.0, patience=100,
    )


@pytest.fixture
def sample():
    """One (img, boxes, cls, masks) tuple, with the box exactly matching the mask."""
    from other_models.augment import masks_to_boxes
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
    masks = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    masks[0, 8:32, 8:32] = 1
    masks[1, 56:84, 52:80] = 1
    img[masks[0] > 0] = (200, 120, 60)
    img[masks[1] > 0] = (60, 200, 120)
    cls = np.array([0, 1], dtype=np.int64)
    return img, masks_to_boxes(masks), cls, masks
