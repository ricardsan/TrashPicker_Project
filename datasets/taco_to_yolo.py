#!/usr/bin/env python3
"""Convert the TACO dataset (COCO-style annotations.json) into an Ultralytics YOLO dataset.

Output layout (what Ultralytics expects - labels/ mirrors images/):

    <out>/
      images/{train,val[,test]}/batch_1_000006.jpg   (symlink or copy of the original)
      labels/{train,val[,test]}/batch_1_000006.txt
      dataset.yaml                                     (pass this to `yolo train data=...`)
      source_index.json                                (YOLO stem -> original TACO image, used by the visualiser)

TACO specifics handled here:
  * category ids are 0-based (0..59), unlike COCO's 1-based ids, so they map 1:1 onto YOLO class ids.
    (Ultralytics' own `convert_coco` subtracts 1 from every id and would shift all TACO classes.)
  * the same file name appears in several batches (batch_1/000006.jpg, batch_2/000006.jpg, ...),
    so the batch name is folded into the output name to avoid one label file overwriting another.
  * a few boxes extend past the image border; they are clipped.
  * annotations are in the EXIF-rotated (upright) frame, and width/height in the JSON are the
    upright size. Ultralytics loads images upright too, so labels are normalised by the JSON size.
    `--verify-images` opens each image and checks that its upright size agrees with the JSON.
  * some objects are annotated as several polygons; in segment mode they are joined into one
    polygon with zero-width bridges so each object stays a single instance.

Usage:
    python taco_to_yolo.py --taco-dir TACO/data --out taco_yolo
    python taco_to_yolo.py --taco-dir TACO/data --out taco_yolo_seg --task segment
    python taco_to_yolo.py --taco-dir TACO/data --out taco_yolo_super --classes super
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def portable(path: Path) -> str:
    """Repo-relative POSIX path, so the written metadata does not pin one machine's layout."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taco-dir", type=Path, required=True,
                   help="TACO data folder holding annotations.json and the batch_* image folders")
    p.add_argument("--annotations", type=Path, default=None,
                   help="annotation file (default: <taco-dir>/annotations.json)")
    p.add_argument("--out", type=Path, default=Path("taco_yolo"), help="output dataset folder")
    p.add_argument("--task", choices=["detect", "segment"], default="detect",
                   help="detect: class cx cy w h | segment: class x1 y1 x2 y2 ... (polygon)")
    p.add_argument("--classes", choices=["all", "super"], default="all",
                   help="all: the 60 TACO categories | super: the 28 supercategories")
    p.add_argument("--val", type=float, default=0.2, help="fraction of images for validation")
    p.add_argument("--test", type=float, default=0.0, help="fraction of images for test")
    p.add_argument("--seed", type=int, default=0, help="seed for the image-level split")
    p.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="symlink",
                   help="how images are placed in the output (falls back to copy if linking fails)")
    p.add_argument("--min-size", type=float, default=1.0,
                   help="drop boxes narrower or shorter than this many pixels after clipping")
    p.add_argument("--verify-images", action="store_true",
                   help="open every image and check its EXIF-corrected size against the JSON (slower)")
    return p.parse_args()


def build_class_map(categories: list[dict], mode: str) -> tuple[dict[int, int], list[str]]:
    """Return (TACO category id -> YOLO class id, YOLO class names)."""
    cats = sorted(categories, key=lambda c: c["id"])
    if mode == "all":
        return {c["id"]: i for i, c in enumerate(cats)}, [c["name"] for c in cats]
    supers = list(dict.fromkeys(c["supercategory"] for c in cats))  # keep first-appearance order
    return {c["id"]: supers.index(c["supercategory"]) for c in cats}, supers


def clip_box(bbox: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] in pixels -> clipped (x1, y1, x2, y2)."""
    x, y, bw, bh = bbox
    return max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)


def merge_polygons(polys: list[np.ndarray]) -> np.ndarray:
    """Join several polygons into one ring by inserting each part at its nearest vertex.

    The ring walks the current outline up to vertex i, jumps to the nearest vertex j of the next
    part, goes all the way round that part, and comes back to i. The jump is a zero-width bridge,
    so a filled mask of the result is the union of the parts.
    """
    merged = polys[0]
    for part in polys[1:]:
        d = ((merged[:, None, :] - part[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(d.argmin(), d.shape)
        part = np.roll(part, -j, axis=0)
        merged = np.concatenate([merged[: i + 1], part, part[:1], merged[i:]])
    return merged


def polygon_row(segmentation: list[list[float]], w: int, h: int) -> np.ndarray | None:
    polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
             for p in segmentation if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
    if not polys:
        return None
    ring = merge_polygons(polys) if len(polys) > 1 else polys[0]
    ring = np.clip(ring, 0, [w, h]) / [w, h]
    return ring.reshape(-1)


def place_image(src: Path, dst: Path, how: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        if how == "symlink":
            dst.symlink_to(src.resolve())
            return
        if how == "hardlink":
            os.link(src, dst)
            return
    except OSError:
        pass  # e.g. Windows without symlink rights, or a different filesystem for hardlinks
    shutil.copy2(src, dst)


def upright_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(274) in (5, 6, 7, 8):  # 90/270 degree EXIF rotations swap the sides
            w, h = h, w
    return w, h


def main() -> None:
    args = parse_args()
    ann_path = args.annotations or args.taco_dir / "annotations.json"
    data = json.loads(ann_path.read_text())

    cls_map, names = build_class_map(data["categories"], args.classes)
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in data["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    # Image-level split, so no picture contributes to two splits.
    images = sorted(data["images"], key=lambda im: im["id"])
    random.Random(args.seed).shuffle(images)
    n_val, n_test = round(len(images) * args.val), round(len(images) * args.test)
    split_of = {}
    for k, im in enumerate(images):
        split_of[im["id"]] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    splits = [s for s in ("train", "val", "test") if s in split_of.values()]

    for s in splits:
        (args.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    per_class = Counter()
    index = {}
    for im in sorted(images, key=lambda im: im["id"]):
        src = args.taco_dir / im["file_name"]
        if not src.exists():
            stats["missing images"] += 1
            continue
        w, h = im["width"], im["height"]
        if args.verify_images and upright_size(src) != (w, h):
            stats["size mismatch (skipped)"] += 1
            print(f"  size mismatch: {im['file_name']} JSON {w}x{h} vs file {upright_size(src)}", file=sys.stderr)
            continue

        split = split_of[im["id"]]
        stem = im["file_name"].replace("/", "_").rsplit(".", 1)[0]  # batch_1/000006.jpg -> batch_1_000006
        rows = []
        for a in anns_by_image.get(im["id"], []):
            x1, y1, x2, y2 = clip_box(a["bbox"], w, h)
            bx, by, bw, bh = a["bbox"]
            if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
                stats["boxes clipped"] += 1
            if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                stats["boxes dropped (too small)"] += 1
                continue
            c = cls_map[a["category_id"]]
            if args.task == "detect":
                vals = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
            else:
                seg = a.get("segmentation") if isinstance(a.get("segmentation"), list) else []
                poly = polygon_row(seg, w, h)
                if poly is None:  # no usable polygon: fall back to the box outline
                    stats["polygons replaced by box"] += 1
                    poly = np.array([x1, y1, x2, y1, x2, y2, x1, y2]) / np.array([w, h] * 4)
                elif len(seg) > 1:
                    stats["multi-part polygons merged"] += 1
                vals = poly.tolist()
            rows.append(f"{c} " + " ".join(f"{v:.6f}" for v in vals))
            per_class[c] += 1

        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        place_image(src, args.out / "images" / split / f"{stem}{src.suffix}", args.link)
        index[stem] = {"image_id": im["id"], "file_name": im["file_name"], "split": split}
        stats[f"images {split}"] += 1
        stats[f"instances {split}"] += len(rows)

    # No `path:` -- Ultralytics then resolves images/... next to this yaml, wherever it lives.
    yaml_lines = [f"{s}: images/{s}" for s in splits]
    yaml_lines += [f"# converted from {portable(ann_path)} | task={args.task} classes={args.classes} seed={args.seed}",
                   f"nc: {len(names)}", "names:"]
    yaml_lines += [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    (args.out / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    (args.out / "source_index.json").write_text(json.dumps(
        {"annotations": portable(ann_path), "taco_dir": portable(args.taco_dir),
         "task": args.task, "classes": args.classes, "names": names,
         "class_map": {str(k): v for k, v in cls_map.items()}, "images": index}, indent=1))

    print(f"Wrote {args.out}/  ({args.task}, {len(names)} classes)")
    for k in sorted(stats):
        print(f"  {k:28s} {stats[k]}")
    rare = [(names[c], n) for c, n in sorted(per_class.items(), key=lambda kv: kv[1])][:5]
    print("  rarest classes:", ", ".join(f"{n} ({k})" for n, k in rare))
    if stats["missing images"] == len(images):
        sys.exit(f"No images found under {args.taco_dir} - point --taco-dir at the folder with batch_1/, batch_2/, ...")


if __name__ == "__main__":
    main()
