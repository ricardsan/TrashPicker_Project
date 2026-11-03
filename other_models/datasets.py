"""Which data to train on, and how a COCO annotation becomes a Mask R-CNN target.

EDIT POINT 2 of 4.

All three datasets are read from COCO JSON, which prepare_data.py writes. Everything downstream
sees one format, so swapping datasets is one word.

One rule holds for every source, and it is worth stating because it is counter-intuitive: OpenCV
APPLIES EXIF rotation. cv2.imread hands back the upright image, not the stored one, and every
annotation prepare_data.py writes is in that same upright frame. Ultralytics reads through
cv2.imdecode, which behaves the same way, so this is also the frame the YOLO runs trained in.

The photos in dataset_v2/ and dataset_seg/ all carry EXIF orientation 6, so their stored pixels
are 4032x3024 while everything here sees 3024x4032. Nothing needs rotating; it just means the
width/height recorded in result.json (which report the stored size) must not be trusted, and
prepare_data.py does not use them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from other_models.augment import drop_empty, letterbox, masks_to_boxes

ROOT = Path(__file__).resolve().parent.parent

# Class order is shared by data_v2.yaml, dataset_seg/data.yaml, classes.txt and every COCO file
# we produce. COCO category_id is 0-based here; Mask R-CNN labels are these + 1, because
# torchvision reserves label 0 for background.
CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
NAMES = dict(enumerate(CLASS_NAMES))


@dataclass(frozen=True)
class Source:
    """One dataset: where its COCO files are, where its images are, which splits exist."""

    annotations: Path          # formatted with {split}
    images: Path               # formatted with {split}
    splits: tuple[str, ...]
    note: str = ""

    def ann_file(self, split: str) -> Path:
        return Path(str(self.annotations).format(split=split))

    def image_dir(self, split: str) -> Path:
        return Path(str(self.images).format(split=split))


# =============================================================================
#  DATA SOURCE -- change ACTIVE_SOURCE to train on a different dataset
# =============================================================================
SOURCES = {
    "dataset_seg": Source(
        annotations=ROOT / "datasets" / "dataset_seg" / "annotations" / "{split}.json",
        images=ROOT / "datasets" / "dataset_seg" / "images" / "{split}",
        splits=("train", "val"),
        note="Real polygon masks, 611 train + 152 val. Gives both Box and Mask metrics.",
    ),
    "dataset_v2": Source(
        annotations=ROOT / "dataset_v2" / "annotations" / "{split}.json",
        images=ROOT / "dataset_v2" / "images" / "{split}",
        splits=("train", "val"),
        note="From the Label Studio export. The 2026-10-05 export has boxes only, so masks fall "
             "back to filled rectangles and Mask metrics are reported as NaN. Drop in an export "
             "with polygons and it starts producing real Mask metrics with no code change.",
    ),
    "taco": Source(
        annotations=ROOT / "datasets" / "taco_yolo" / "annotations" / "{split}.json",
        images=ROOT / "datasets" / "taco_yolo" / "images" / "{split}",
        splits=("train", "val", "test"),
        note="TACO, reduced class mapping. Used for pretraining.",
    ),
}

ACTIVE_SOURCE = "dataset_seg"   # <- the default. train.py --source NAME overrides it.
# =============================================================================


def get_source(name: str | Source | None = None) -> Source:
    """Look a source up by name. A Source is passed through, which is how tests supply their own."""
    if isinstance(name, Source):
        return name
    name = name or ACTIVE_SOURCE
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {sorted(SOURCES)}")
    return SOURCES[name]


# =============================================================================
#  TURNING COCO INTO MASK R-CNN TARGETS
# =============================================================================
@dataclass
class Record:
    """One image's annotations, decoded from COCO but not yet rasterised."""

    path: Path
    width: int
    height: int
    cls: np.ndarray                    # int64 [N], 0-based
    boxes: np.ndarray                  # float32 [N, 4] xyxy
    polygons: list[list[np.ndarray]]   # per instance, a list of (P, 2) float32 rings


class CocoInstanceDataset(Dataset):
    """Reads one split of one Source and yields what torchvision's detection models want.

        image   FloatTensor [3, H, W]  RGB in 0..1 (the model normalises internally)
        target  boxes   FloatTensor [N, 4]  x1, y1, x2, y2 in pixels
                labels  Int64Tensor  [N]    1..4, because 0 is background
                masks   UInt8Tensor  [N, H, W]
                image_id, orig_size   kept for evaluation and plotting
    """

    def __init__(self, source: Source, split: str, imgsz: int,
                 augmenter=None, cache: bool = True):
        self.source = source
        self.split = split
        self.imgsz = imgsz
        self.augmenter = augmenter
        self.names = NAMES

        ann_file = source.ann_file(split)
        if not ann_file.exists():
            raise FileNotFoundError(
                f"{ann_file} not found. Build it first:\n"
                f"    python other_models/prepare_data.py --source <name>"
            )
        coco = json.loads(ann_file.read_text())
        self.records, self.has_real_masks = self._decode(coco, source.image_dir(split))
        self._cache: dict[int, tuple] | None = {} if cache else None
        if cache:
            self._fill_cache()

    # --- loading ------------------------------------------------------------
    @staticmethod
    def _decode(coco: dict, image_dir: Path) -> tuple[list[Record], bool]:
        by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd"):
                continue  # a crowd region marks many objects at once; not trainable
            by_image.setdefault(ann["image_id"], []).append(ann)

        records, with_polygon, total = [], 0, 0
        for info in coco["images"]:
            anns = by_image.get(info["id"], [])
            if not anns:
                continue
            cls, boxes, polys = [], [], []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w < 1 or h < 1:
                    continue
                seg = ann.get("segmentation") or []
                rings = [
                    np.asarray(p, dtype=np.float32).reshape(-1, 2)
                    for p in seg
                    if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0
                ]
                total += 1
                with_polygon += bool(rings)
                cls.append(int(ann["category_id"]))
                boxes.append([x, y, x + w, y + h])
                polys.append(rings)
            if not cls:
                continue
            records.append(Record(
                path=image_dir / info["file_name"],
                width=int(info["width"]),
                height=int(info["height"]),
                cls=np.asarray(cls, dtype=np.int64),
                boxes=np.asarray(boxes, dtype=np.float32),
                polygons=polys,
            ))
        # Every annotation must carry a polygon before Mask metrics mean anything. A partly
        # polygonised export would otherwise score filled rectangles as if they were
        # segmentations, which is worse than reporting nothing.
        return records, total > 0 and with_polygon == total

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, i: int) -> tuple[np.ndarray, float]:
        """The image with its long side scaled down to imgsz, plus the scale that was applied.

        Downscaling here rather than later is what makes caching viable: these photos are
        4032x3024, so a cache of originals would need ~22 GB for 611 of them. It costs nothing,
        because every path out of this class letterboxes or warps to imgsz anyway. It is also
        what ultralytics' load_image does.
        """
        if self._cache is not None and i in self._cache:
            img, scale = self._cache[i]
            return img, scale

        record = self.records[i]
        # BGR, and upright: IMREAD_COLOR applies EXIF rotation (suppressing it would need
        # IMREAD_IGNORE_ORIENTATION). That is the frame every annotation is in -- see the
        # module docstring.
        img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image {record.path}")

        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        if scale != 1:
            interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)
        if self._cache is not None:
            self._cache[i] = (img, scale)
        return img, scale

    def _fill_cache(self) -> None:
        """Decode every image once, here in the parent process, so no worker ever decodes a JPEG.

        Be aware of what this costs. Python 3.14 defaults to the `forkserver` start method on
        Linux, so DataLoader workers do NOT share these pages copy-on-write the way they would
        under `fork` -- the whole dataset, cache included, is pickled into each worker. The
        printed total is per process; multiply by workers for the real resident cost, and pass
        cache=False (as TACO_PRETRAIN does) if that is too much.
        """
        for i in range(len(self.records)):
            self._load_image(i)
        if self._cache:
            megabytes = sum(img.nbytes for img, _ in self._cache.values()) / 1e6
            print(f"cached {len(self._cache)} {self.split} images at imgsz={self.imgsz} "
                  f"({megabytes:.0f} MB per process)")

    def load_raw(self, i: int):
        """(img, boxes, cls, masks) for one image, before any augmentation.

        augment.py calls this to fetch the extra images mosaic and mixup need. Masks are
        rasterised on every call rather than cached: fillPoly is cheap, and augmentation writes
        into them, so a cached copy would have to be duplicated anyway.
        """
        record = self.records[i]
        img, scale = self._load_image(i)
        h, w = img.shape[:2]

        masks = np.zeros((len(record.cls), h, w), dtype=np.uint8)
        for n, rings in enumerate(record.polygons):
            if rings:
                cv2.fillPoly(masks[n], [np.round(r * scale).astype(np.int32) for r in rings], 1)
            else:
                # No polygon in the export: fall back to the filled box so the mask head still
                # has something to learn from. has_real_masks stays False and the engine
                # reports Mask metrics as NaN rather than pretending these are segmentations.
                box = np.clip(record.boxes[n] * scale, [0, 0, 0, 0], [w, h, w, h]).astype(int)
                masks[n, box[1]:box[3], box[0]:box[2]] = 1

        boxes = masks_to_boxes(masks)
        return (img.copy(), *drop_empty(boxes, record.cls.copy(), masks))

    def __getitem__(self, i: int):
        img, boxes, cls, masks = self.load_raw(i)
        if self.augmenter is not None:
            img, boxes, cls, masks = self.augmenter((img, boxes, cls, masks), self)
        else:
            img, boxes, cls, masks = letterbox(img, boxes, cls, masks, self.imgsz)
        boxes, cls, masks = drop_empty(boxes, cls, masks)

        image = torch.from_numpy(
            np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR HWC -> RGB CHW
        ).float().div_(255)
        target = {
            # reshape keeps the dtype and rank right when an image ends up with no instances
            "boxes": torch.from_numpy(boxes).float().reshape(-1, 4),
            "labels": torch.from_numpy(cls).long().reshape(-1) + 1,  # 0 is background
            "masks": torch.from_numpy(masks).reshape(-1, img.shape[0], img.shape[1]),
            "image_id": i,
            "path": str(self.records[i].path),
        }
        return image, target


def collate_fn(batch):
    """Detection models take a list of images and a list of targets, not stacked tensors."""
    return tuple(zip(*batch))
