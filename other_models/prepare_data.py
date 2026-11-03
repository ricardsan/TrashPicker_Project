#!/usr/bin/env python3
"""Build the COCO annotation files other_models/datasets.py reads, from what is on this machine.

Three sources, one command each. Every run also writes a .provenance.json sidecar next to the
output (input sha256s, counts, per-class histogram, anything skipped), the same convention as
datasets/annotations-richard.provenance.json.

    dataset_seg   YOLO polygon labels  -> datasets/dataset_seg/annotations/{train,val}.json
                  The default training source: these are the only real masks in the repo.

    dataset_v2    the Label Studio export -> dataset_v2/annotations/{train,val}.json
                  Its file_name fields point at another machine
                  ("../../home/rica/.local/share/label-studio/media/upload/2/..."), so they are
                  rewritten to the basename and resolved against dataset_v2/images/{train,val}.
                  672 of the 763 images resolve; the other 91 are IMG_17xx.jpeg photos that are
                  not in dataset_v2 at all, and are listed in the sidecar rather than dropped
                  silently. NOTE the 2026-10-05 export has segmentation: [] on all 1848
                  annotations -- it is boxes only. A newer export with polygons needs no code
                  change here.

    taco          TACO/data/annotations.json + taco_class_mapping_taco_yolo_reduced.json
                  -> datasets/taco_yolo/annotations/{train,val,test}.json
                  Built from the REDUCED mapping: its 25 "skip" categories are not written at
                  all. See the long comment above build_taco() for why it is built this way and
                  not from annotations-richard.json or from the YOLO .txt labels.

Usage:
    python other_models/prepare_data.py --source dataset_seg
    python other_models/prepare_data.py --source dataset_seg --source dataset_v2 --source taco
    python other_models/prepare_data.py --all

A note on EXIF, because it is the easiest thing to get wrong here. OpenCV APPLIES EXIF rotation:
cv2.imread and cv2.imdecode both hand back the upright image, and ultralytics loads through
cv2.imdecode, so the YOLO runs trained on upright pixels. All three sets of annotations turn out
to be in that same upright frame:

  * dataset_seg's YOLO labels are normalised, so they follow whatever frame they are denormalised
    into -- upright, to match the pixels.
  * result.json's coordinates are upright, even though its width/height fields report the raw
    size. Measured: all 1512 of its resolved annotations fit the upright frame, only 951 fit the
    raw one. The 561 that do not fit raw are the ones that spill past y = 3024.
  * TACO's JSON records upright sizes already (verified on all 1014 images).

So there is nothing to rotate; image_size() below just has to report the size cv2 will return,
not the size PIL reports before applying EXIF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from other_models.datasets import CLASS_NAMES, ROOT  # noqa: E402

CATEGORIES = [{"id": i, "name": n, "supercategory": n} for i, n in enumerate(CLASS_NAMES)]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# =============================================================================
# 1. SMALL HELPERS
# =============================================================================
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    """Repo-relative POSIX path for provenance, so it does not pin one machine's layout."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def stem_key(name: str) -> str:
    """Case-insensitive stem, because on-disk extensions vary (batch_7_000125.JPG)."""
    return Path(name).stem.lower()


#  Orientation values 5-8 mean the stored pixels are rotated a quarter turn. PIL reports the
#  stored size; OpenCV rotates on load and so reports the transpose. Everything downstream reads
#  images through OpenCV, so this returns OpenCV's answer -- and does it from the header alone,
#  without decoding 12 megapixels.
def image_size(path: Path) -> tuple[int, int]:
    """(width, height) as cv2.imread will return it, i.e. with EXIF rotation applied."""
    with Image.open(path) as im:
        width, height = im.size
        return (height, width) if im.getexif().get(274) in (5, 6, 7, 8) else (width, height)


def polygon_area(ring: np.ndarray) -> float:
    """Shoelace area, for the COCO `area` field."""
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def rings_to_annotation(ann_id: int, image_id: int, category_id: int,
                        rings: list[np.ndarray], w: int, h: int) -> dict | None:
    """One COCO annotation from one or more polygon rings, clipped to the image."""
    # clip to [0, w] x [0, h]: COCO's convention is that a box may touch the far edge, and
    # w-1/h-1 would shave a pixel off every annotation that does
    rings = [np.clip(r, [0, 0], [w, h]) for r in rings if len(r) >= 3]
    if not rings:
        return None
    allpts = np.concatenate(rings)
    x1, y1 = allpts.min(0)
    x2, y2 = allpts.max(0)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [[round(float(v), 2) for v in r.reshape(-1)] for r in rings],
        "area": round(sum(polygon_area(r) for r in rings), 2),
        "bbox": [round(float(x1), 2), round(float(y1), 2),
                 round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
        "iscrowd": 0,
    }


def write_coco(out: Path, images: list[dict], annotations: list[dict], description: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "info": {
            "description": description,
            "contributor": "other_models/prepare_data.py",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "coordinate_frame": "EXIF-upright pixels, i.e. exactly what cv2.imread returns",
        },
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }, indent=1))


def write_provenance(out: Path, payload: dict) -> None:
    out.write_text(json.dumps(payload, indent=1))


def class_histogram(annotations: list[dict]) -> dict[str, int]:
    counts = Counter(a["category_id"] for a in annotations)
    return {name: counts.get(i, 0) for i, name in enumerate(CLASS_NAMES)}


def report(split: str, images: list[dict], annotations: list[dict]) -> None:
    hist = ", ".join(f"{k} {v}" for k, v in class_histogram(annotations).items())
    print(f"  {split:5s}  {len(images):5d} images  {len(annotations):5d} annotations   {hist}")


def index_images(directory: Path) -> dict[str, Path]:
    """stem (lowercased) -> path, for every image in a directory."""
    if not directory.is_dir():
        return {}
    return {stem_key(p.name): p for p in sorted(directory.iterdir())
            if p.suffix in IMAGE_SUFFIXES}


# =============================================================================
# 2. YOLO POLYGON LABELS -> COCO   (dataset_seg, and anything else in that layout)
# =============================================================================
def build_from_yolo(name: str, images_root: Path, labels_root: Path, out_dir: Path,
                    splits: tuple[str, ...]) -> None:
    """Each label line is `cls x1 y1 x2 y2 ...` normalised to the image; 4 numbers means a box."""
    print(f"{name}: {labels_root} -> {out_dir}")
    per_split = {}

    for split in splits:
        label_dir, image_dir = labels_root / split, images_root / split
        by_stem = index_images(image_dir)
        images, annotations, missing, box_only = [], [], [], 0
        ann_id = 0

        for label_file in sorted(label_dir.glob("*.txt")):
            key = stem_key(label_file.name)
            if key not in by_stem:
                missing.append(label_file.name)
                continue
            image_path = by_stem[key]
            w, h = image_size(image_path)
            image_id = len(images)
            kept = 0

            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, coords = int(parts[0]), np.asarray(parts[1:], dtype=np.float64)
                if len(coords) == 4:      # a detection box: cx cy bw bh
                    cx, cy, bw, bh = coords * [w, h, w, h]
                    ring = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                                     [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
                    box_only += 1
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    ring = coords.reshape(-1, 2) * [w, h]
                else:
                    continue
                ann = rings_to_annotation(ann_id, image_id, cls, [ring], w, h)
                if ann:
                    annotations.append(ann)
                    ann_id += 1
                    kept += 1

            if kept:
                images.append({"id": image_id, "file_name": image_path.name,
                               "width": w, "height": h})

        write_coco(out_dir / f"{split}.json", images, annotations,
                   f"{name} {split}, converted from YOLO polygon labels in {rel(labels_root)}")
        report(split, images, annotations)
        per_split[split] = {"images": len(images), "annotations": len(annotations),
                            "per_class": class_histogram(annotations),
                            "labels_without_image": missing,
                            "box_only_lines": box_only}
        if missing:
            print(f"         {len(missing)} label files had no matching image: {missing[:5]}")
        if box_only:
            print(f"         {box_only} lines were boxes, not polygons -> stored as rectangles")

    write_provenance(out_dir / "provenance.json", {
        "what_this_is": f"provenance for the COCO files in {rel(out_dir)}",
        "script": "other_models/prepare_data.py",
        "created": datetime.now(timezone.utc).isoformat(),
        "source": {"images": rel(images_root), "labels": rel(labels_root)},
        "coordinate_frame": "EXIF-upright pixels, i.e. exactly what cv2.imread returns",
        "splits": per_split,
    })


def build_dataset_seg() -> None:
    build_from_yolo(
        "dataset_seg",
        images_root=ROOT / "datasets" / "dataset_seg" / "images",
        labels_root=ROOT / "datasets" / "dataset_seg" / "labels",
        out_dir=ROOT / "datasets" / "dataset_seg" / "annotations",
        splits=("train", "val"),
    )


# =============================================================================
# 3. THE LABEL STUDIO EXPORT -> COCO   (dataset_v2)
# =============================================================================
def build_dataset_v2() -> None:
    export = next(ROOT.glob("coco-project*/result.json"), None)
    if export is None:
        raise FileNotFoundError(f"no coco-project*/result.json under {ROOT}")
    out_dir = ROOT / "dataset_v2" / "annotations"
    print(f"dataset_v2: {export.relative_to(ROOT)} -> {out_dir}")

    coco = json.loads(export.read_text())
    # Label Studio exports category ids that already match data_v2.yaml (glass=0 ... plastic=3).
    # Do NOT apply ultralytics convert_coco's `cls = cat_id - 1` shift here.
    source_names = {c["id"]: c["name"] for c in coco["categories"]}
    if [source_names[i] for i in sorted(source_names)] != CLASS_NAMES:
        raise ValueError(f"unexpected categories in {export}: {coco['categories']}")

    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    # the split is whichever folder the file actually sits in, so it matches the YOLO split
    disk = {split: index_images(ROOT / "dataset_v2" / "images" / split)
            for split in ("train", "val")}

    out = {split: {"images": [], "annotations": []} for split in disk}
    unresolved, no_polygon, out_of_bounds = [], 0, []

    for info in coco["images"]:
        # "../../home/rica/.local/share/label-studio/media/upload/2/5c9060cd-img_0001.jpg"
        # -> "5c9060cd-img_0001.jpg". Keep the 8-hex upload prefix: it is part of the real name.
        basename = Path(info["file_name"]).name
        split = next((s for s, idx in disk.items() if stem_key(basename) in idx), None)
        if split is None:
            unresolved.append(basename)
            continue

        image_path = disk[split][stem_key(basename)]
        w, h = image_size(image_path)
        bucket = out[split]
        image_id = len(bucket["images"])
        kept = 0

        for ann in by_image.get(info["id"], []):
            if ann.get("iscrowd"):
                continue
            seg = ann.get("segmentation") or []
            rings = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                     for p in seg if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
            if not rings:
                # boxes-only export: store the rectangle. datasets.py keeps has_real_masks False
                # for these, and the engine then reports Mask metrics as NaN.
                x, y, bw, bh = ann["bbox"]
                rings = [np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])]
                no_polygon += 1
            new = rings_to_annotation(len(bucket["annotations"]), image_id,
                                      int(ann["category_id"]), rings, w, h)
            if new:
                bucket["annotations"].append(new)
                kept += 1
            else:
                # Collapsed to nothing once clipped, i.e. it really does lie outside the
                # image. Expected to be empty: the 20 that looked out of bounds when this was
                # measured in the raw frame all fit once the upright frame is used.
                out_of_bounds.append({"image": image_path.name, "bbox": ann["bbox"],
                                      "class": CLASS_NAMES[int(ann["category_id"])]})

        if kept:
            bucket["images"].append({"id": image_id, "file_name": image_path.name,
                                     "width": w, "height": h})

    per_split = {}
    for split, bucket in out.items():
        write_coco(out_dir / f"{split}.json", bucket["images"], bucket["annotations"],
                   f"dataset_v2 {split}, from {export.name} with file_name rewritten for this machine")
        report(split, bucket["images"], bucket["annotations"])
        per_split[split] = {"images": len(bucket["images"]),
                            "annotations": len(bucket["annotations"]),
                            "per_class": class_histogram(bucket["annotations"])}

    resolved = sum(v["images"] for v in per_split.values())
    print(f"         resolved {resolved}/{len(coco['images'])} images; "
          f"{len(unresolved)} unresolved (listed in provenance.json)")
    if out_of_bounds:
        print(f"         {len(out_of_bounds)} annotations lie outside their image and were "
              f"dropped (listed in provenance.json)")
    if no_polygon:
        print(f"         WARNING: {no_polygon} annotations had no polygon and became rectangles.")
        print("         This export is boxes only -- Mask metrics will be reported as NaN.")

    write_provenance(out_dir / "provenance.json", {
        "what_this_is": f"provenance for the COCO files in {rel(out_dir)}",
        "script": "other_models/prepare_data.py",
        "created": datetime.now(timezone.utc).isoformat(),
        "source": {"path": rel(export), "sha256": sha256(export),
                   "bytes": export.stat().st_size,
                   "images": len(coco["images"]),
                   "annotations": len(coco["annotations"])},
        "file_name_rewrite": "basename of the Label Studio path, matched against "
                             "dataset_v2/images/{train,val}; the 8-hex upload prefix is kept",
        "coordinate_frame": "EXIF-upright pixels, i.e. exactly what cv2.imread returns. NOTE the export's own width/height fields report the pre-EXIF size and are not used",
        "annotations_without_polygon": no_polygon,
        "annotations_outside_image": out_of_bounds,
        "unresolved_images": sorted(unresolved),
        "splits": per_split,
    })


# =============================================================================
# 4. TACO -> COCO, THROUGH THE REDUCED MAPPING
# =============================================================================
# Why it is built this way:
#
#   * from TACO/data/annotations.json, because that is the only file carrying the original
#     60 category ids, and the reduced mapping keys on exactly those ids.
#   * not from annotations-richard.json, which was merged with the FULL mapping and so still
#     contains the categories the reduced mapping drops.
#   * not from taco_yolo/labels/*.txt, which only store the merged target id 0-3 -- they cannot
#     tell you which source category an instance came from.
#
# The split for each surviving image is read off taco_yolo/images/{train,val,test}, so the split
# boundary stays exactly where taco_to_yolo.py put it.
#
# Expected output, measured against the real files: 35 of 60 categories kept, 1985 of 4784
# annotations, 1014 of 1500 images, train 706 / val 208 / test 100,
# {plastic 1000, paper 497, metal 372, glass 116}. taco_yolo/source_index.json is the same
# content produced independently by taco_merge_classes.py, so tests compare against it.
#
# Heads-up: the reduced mapping's own "annotation_totals_per_target" field is stale -- it still
# reads skip: 1231, copied from the full mapping, where the real reduced drop is 2799. Always
# recompute from "mapping"; never read that field.
def build_taco() -> None:
    taco_dir = ROOT / "datasets"
    annotations_file = taco_dir / "TACO" / "data" / "annotations.json"
    mapping_file = taco_dir / "taco_class_mapping_taco_yolo_reduced.json"
    out_dir = taco_dir / "taco_yolo" / "annotations"
    print(f"taco: {annotations_file.relative_to(ROOT)} + {mapping_file.name} -> {out_dir}")

    mapping = json.loads(mapping_file.read_text())["mapping"]
    keep = {int(k): v for k, v in mapping.items() if not v["drop"]}
    dropped_categories = [
        {"id": int(k), "name": v["taco_name"], "n_annotations": v["n_annotations"]}
        for k, v in sorted(mapping.items(), key=lambda kv: int(kv[0])) if v["drop"]
    ]
    print(f"         reduced mapping keeps {len(keep)}/{len(mapping)} source categories")

    coco = json.loads(annotations_file.read_text())
    images_by_id = {i["id"]: i for i in coco["images"]}

    # which split each flattened filename lives in: batch_1/000006.jpg -> batch_1_000006.jpg
    split_of, path_of = {}, {}
    for split in ("train", "val", "test"):
        for key, path in index_images(taco_dir / "taco_yolo" / "images" / split).items():
            split_of[key] = split
            path_of[key] = path

    out = {s: {"images": [], "annotations": []} for s in ("train", "val", "test")}
    by_image, n_dropped_anns = {}, 0
    for ann in coco["annotations"]:
        entry = mapping[str(ann["category_id"])]
        if entry["drop"]:
            n_dropped_anns += 1
            continue
        by_image.setdefault(ann["image_id"], []).append((ann, entry))

    not_on_disk, empty_after_merge, size_mismatch = [], 0, []

    for image_id, anns in sorted(by_image.items()):
        info = images_by_id[image_id]
        key = stem_key(info["file_name"].replace("/", "_"))
        if key not in path_of:
            not_on_disk.append(info["file_name"])
            continue

        image_path = path_of[key]
        w, h = image_size(image_path)
        # TACO records upright sizes and OpenCV returns upright pixels, so these must agree. If
        # one ever does not, its polygons are in a different frame from its pixels and the only
        # honest thing is to say so rather than quietly scale them.
        if (w, h) != (info["width"], info["height"]):
            size_mismatch.append({"file": info["file_name"],
                                  "recorded": [info["width"], info["height"]],
                                  "on_disk_upright": [w, h]})

        bucket = out[split_of[key]]
        new_image_id = len(bucket["images"])
        kept = 0
        for ann, entry in anns:
            rings = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                     for p in (ann.get("segmentation") or [])
                     if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
            new = rings_to_annotation(len(bucket["annotations"]), new_image_id,
                                      entry["target_id"], rings, w, h)
            if new:
                bucket["annotations"].append(new)
                kept += 1
        if kept:
            bucket["images"].append({
                "id": new_image_id,
                "file_name": image_path.name,
                "width": w, "height": h,
                "taco_file_name": info["file_name"],
            })
        else:
            empty_after_merge += 1

    per_split, totals = {}, Counter()
    for split, bucket in out.items():
        write_coco(out_dir / f"{split}.json", bucket["images"], bucket["annotations"],
                   f"TACO {split}, reduced class mapping ({mapping_file.name}); "
                   f"skip categories are not written")
        report(split, bucket["images"], bucket["annotations"])
        per_split[split] = {"images": len(bucket["images"]),
                            "annotations": len(bucket["annotations"]),
                            "per_class": class_histogram(bucket["annotations"])}
        totals["images"] += len(bucket["images"])
        totals["annotations"] += len(bucket["annotations"])

    print(f"         total {totals['images']} images / {totals['annotations']} annotations "
          f"({n_dropped_anns} annotations dropped by the reduced mapping)")
    if size_mismatch:
        print(f"         WARNING: {len(size_mismatch)} images disagree with their recorded size "
              f"(listed in provenance.json)")
    if not_on_disk:
        print(f"         {len(not_on_disk)} images are not in taco_yolo/images/*: "
              f"{not_on_disk[:3]}")

    write_provenance(out_dir / "provenance.json", {
        "what_this_is": f"provenance for the COCO files in {rel(out_dir)}",
        "script": "other_models/prepare_data.py",
        "created": datetime.now(timezone.utc).isoformat(),
        "source": {"path": rel(annotations_file), "sha256": sha256(annotations_file),
                   "categories": len(coco["categories"]),
                   "images": len(coco["images"]),
                   "annotations": len(coco["annotations"])},
        "mapping": {"path": rel(mapping_file), "sha256": sha256(mapping_file),
                    "kept_categories": len(keep), "dropped_categories": len(dropped_categories)},
        "note": "built from the REDUCED mapping, not annotations-richard.json: the categories "
                "below are not written at all. The mapping file's own "
                "annotation_totals_per_target field is stale and is not used.",
        "coordinate_frame": "EXIF-upright pixels, which is what OpenCV returns and what TACO's "
                            "recorded sizes already use; no rotation was applied",
        "images_disagreeing_with_recorded_size": size_mismatch,
        "annotations_dropped_by_mapping": n_dropped_anns,
        "images_left_empty_by_mapping": empty_after_merge,
        "images_not_on_disk": not_on_disk,
        "dropped_category_table": dropped_categories,
        "splits": per_split,
    })


# =============================================================================
# 5. CLI
# =============================================================================
BUILDERS = {
    "dataset_seg": build_dataset_seg,
    "dataset_v2": build_dataset_v2,
    "taco": build_taco,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", choices=sorted(BUILDERS), default=[],
                   help="which dataset to prepare; repeat for several")
    p.add_argument("--all", action="store_true", help="prepare all three")
    args = p.parse_args()

    sources = sorted(BUILDERS) if args.all else args.source
    if not sources:
        p.error("pass --source NAME (repeatable) or --all")
    for name in sources:
        BUILDERS[name]()
        print()


if __name__ == "__main__":
    main()
