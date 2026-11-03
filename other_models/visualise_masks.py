#!/usr/bin/env python3
"""Show the masks Mask R-CNN will actually be trained on, next to the labels they came from.

Same idea and the same flags as datasets/visualise_taco_yolo.py, but for the COCO
files other_models/prepare_data.py writes.

Two modes, chosen with --labels:

  --labels source  (default)  Left panel is the original label, read straight from its source of
                              truth: the YOLO polygon .txt for dataset_seg and taco, the bbox in
                              result.json for dataset_v2. Right panel is the converted COCO. Use
                              this to check the conversion.
  --labels coco               Both panels come from the converted COCO file itself -- outlines on
                              the left, the rasterised masks on the right. Use this to look at the
                              labels you are actually training from, without needing the .txt
                              files to still be around.

The right panel is always rasterised exactly the way other_models/datasets.py rasterises it for the
model. For an export with no polygons that is a filled rectangle, and seeing that is the point.

With --labels source the figure title reports a round-trip check: for every source label, the mask
IoU against the best-matching prepared annotation of the same class. 1.000 means prepare_data.py
changed nothing, and the title turns red below 0.99. With --labels coco there is nothing to
compare against, so the title just counts the annotations.

Everything is drawn in the EXIF-upright frame, because that is what OpenCV returns and therefore
what the model is trained on. These photos are stored rotated (EXIF orientation 6), so the frame
here is 3024x4032 even though the file holds 4032x3024.

Three ways to pick images:
    a random image (default), or several with --n
    a random image containing a given class  --class glass
    named images                             --images 00958062-image297.jpg batch_11_000028.jpg

Usage:
    python other_models/visualise_masks.py --source dataset_seg --n 6
    python other_models/visualise_masks.py --source dataset_v2 --show
    python other_models/visualise_masks.py --source taco --class glass --n 4
    python other_models/visualise_masks.py --source taco --images batch_1_000008.jpg
    python other_models/visualise_masks.py --source dataset_seg --split val
    python other_models/visualise_masks.py --source dataset_seg --labels coco --n 6

Without --show the figures are written to other_models/viz/<source>/. Random picks change every run;
the seed is printed so a pick can be repeated with --seed.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from other_models.datasets import CLASS_NAMES, ROOT, get_source  # noqa: E402
from other_models.prepare_data import index_images, stem_key  # noqa: E402

# Where each source's original labels live, so the left panel never reads our own output.
SOURCE_LABELS = {
    "dataset_seg": ("yolo", ROOT / "datasets" / "dataset_seg" / "labels" / "{split}"),
    "taco": ("yolo", ROOT / "datasets" / "taco_yolo" / "labels" / "{split}"),
    "dataset_v2": ("label-studio", None),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="dataset_seg", choices=sorted(SOURCE_LABELS))
    p.add_argument("--split", default="train", help="split to draw from (default: train)")
    pick = p.add_mutually_exclusive_group()
    pick.add_argument("--class", dest="cls", metavar="CLASS",
                      help=f"only images containing this class: {', '.join(CLASS_NAMES)}")
    pick.add_argument("--images", nargs="+", metavar="IMAGE",
                      help="specific file names (extension optional)")
    p.add_argument("--labels", default="source", choices=("source", "coco"),
                   help="what the left panel reads: 'source' (default) compares the original "
                        "YOLO .txt / result.json against the converted COCO and reports a "
                        "round-trip IoU; 'coco' shows the converted COCO on its own, which is "
                        "what you want for inspecting the labels you are training from")
    p.add_argument("--n", type=int, default=1, help="how many random images (default: 1)")
    p.add_argument("--seed", type=int, default=None, help="repeat an earlier random pick")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: other_models/viz)")
    p.add_argument("--max-side", type=int, default=1400, help="downscale for drawing")
    p.add_argument("--show", action="store_true", help="open windows instead of writing PNGs")
    return p.parse_args()


# =============================================================================
# READING THE TWO SIDES
# =============================================================================
def load_prepared(source_name: str, split: str) -> tuple[dict, dict]:
    """The COCO file prepare_data.py wrote, indexed by image stem."""
    path = get_source(source_name).ann_file(split)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run:\n"
                         f"    python other_models/prepare_data.py --source {source_name}")
    coco = json.loads(path.read_text())
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    return {stem_key(i["file_name"]): (i, by_image.get(i["id"], [])) for i in coco["images"]}, coco


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    """YOLO polygon (or box) lines, denormalised onto a width x height frame."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, coords = int(parts[0]), np.asarray(parts[1:], dtype=np.float64)
        if len(coords) == 4:
            cx, cy, bw, bh = coords * [width, height, width, height]
            ring = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                             [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            ring = coords.reshape(-1, 2) * [width, height]
        else:
            continue
        out.append((cls, ring))
    return out


@functools.lru_cache(maxsize=1)
def _label_studio_export() -> dict | None:
    """result.json, read once. It is several MB and gets consulted per image."""
    export = next(ROOT.glob("coco-project*/result.json"), None)
    return json.loads(export.read_text()) if export else None


def read_label_studio(name: str) -> list[tuple[int, np.ndarray]]:
    """The bbox (or polygon, if a newer export has one) for one image, from result.json."""
    coco = _label_studio_export()
    if coco is None:
        return []
    image_id = next((i["id"] for i in coco["images"]
                     if stem_key(Path(i["file_name"]).name) == stem_key(name)), None)
    if image_id is None:
        return []
    out = []
    for ann in coco["annotations"]:
        if ann["image_id"] != image_id:
            continue
        rings = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                 for p in (ann.get("segmentation") or [])
                 if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
        if not rings:
            x, y, w, h = ann["bbox"]
            rings = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])]
        out.extend((int(ann["category_id"]), r) for r in rings)
    return out


def source_labels(source_name: str, split: str, image_path: Path,
                  size: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
    """The left panel's instances, in the upright frame so both panels are comparable."""
    kind, template = SOURCE_LABELS[source_name]
    width, height = size
    if kind == "label-studio":
        return read_label_studio(image_path.name)
    label_file = Path(str(template).format(split=split)) / f"{image_path.stem}.txt"
    return read_yolo_labels(label_file, width, height)


def coco_rings(annotations: list[dict]) -> list[tuple[int, list[np.ndarray]]]:
    """(class, rings) per annotation, straight out of a prepared COCO file."""
    return [(int(a["category_id"]),
             [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in a["segmentation"]])
            for a in annotations]


# =============================================================================
# THE ROUND-TRIP CHECK
# =============================================================================
def rasterise(rings: list[np.ndarray], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ring in rings:
        cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], 1)
    return mask


def round_trip(left: list[tuple[int, np.ndarray]], right: list[tuple[int, list[np.ndarray]]],
               width: int, height: int) -> list[float]:
    """For each source label, the best mask IoU against a prepared annotation of its class."""
    prepared = [(cls, rasterise(rings, width, height)) for cls, rings in right]
    scores = []
    for cls, ring in left:
        source = rasterise([ring], width, height)
        best = 0.0
        for other_cls, other in prepared:
            if other_cls != cls:
                continue
            union = (source | other).sum()
            if union:
                best = max(best, float((source & other).sum() / union))
        scores.append(best)
    return scores


# =============================================================================
# DRAWING
# =============================================================================
def colour(cls: int) -> tuple[float, float, float]:
    palette = [(0.20, 0.60, 0.95), (0.95, 0.45, 0.15), (0.25, 0.75, 0.35), (0.80, 0.30, 0.75)]
    return palette[cls % len(palette)]


def draw(ax, image: np.ndarray, instances, filled: bool, scale: float, title: str) -> None:
    from matplotlib.patches import Polygon
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for cls, rings in instances:
        for ring in (rings if isinstance(rings, list) else [rings]):
            ax.add_patch(Polygon(
                ring * scale, closed=True, facecolor=colour(cls) if filled else "none",
                edgecolor=colour(cls), alpha=0.45 if filled else 1.0, linewidth=1.6,
            ))
        first = (rings[0] if isinstance(rings, list) else rings) * scale
        ax.text(first[:, 0].min(), first[:, 1].min() - 4, CLASS_NAMES[cls],
                color="white", fontsize=7,
                bbox=dict(facecolor=colour(cls), edgecolor="none", pad=1))


def figure_for(source_name: str, split: str, image_path: Path, prepared, args):
    import matplotlib.pyplot as plt

    info, annotations = prepared
    # cv2 applies EXIF, so this is already the upright frame every annotation lives in
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read {image_path}")
    height, width = image.shape[:2]

    right = coco_rings(annotations)
    if args.labels == "coco":
        # Show the converted file on its own: outlines on the left, the masks the model is
        # actually handed on the right. No round-trip check -- there is nothing to compare to.
        left, ious = [(c, r) for c, rings in right for r in rings], []
    else:
        left = source_labels(source_name, split, image_path, (width, height))
        ious = round_trip(left, right, width, height)

    # A source label with no overlap at all has no counterpart, which for taco is expected --
    # see the note in main(). Judge the conversion on the labels that do have one.
    matched = [v for v in ious if v > 0]
    orphans = len(ious) - len(matched)
    worst = min(matched) if matched else 0.0

    image = image[:, :, ::-1]
    scale = min(1.0, args.max_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    kind = "converted COCO" if args.labels == "coco" else SOURCE_LABELS[source_name][0]
    draw(axes[0], image, left, False, scale,
         f"{kind} outlines -- {len(left)} polygons")
    draw(axes[1], image, right, True, scale,
         f"the same COCO, rasterised as the model sees it -- "
         f"{sum(len(r[1]) for r in right)} instances")

    if args.labels == "coco":
        summary = (f"{image_path.name}   {width}x{height} upright   "
                   f"{len(right)} annotations from {get_source(source_name).ann_file(split).name}")
        ok = True
    elif not ious:
        summary, ok = f"{image_path.name}   no source labels found", False
    else:
        summary = (f"{image_path.name}   {width}x{height} upright   "
                   f"{len(matched)}/{len(ious)} matched, round-trip IoU min {worst:.3f}")
        if orphans:
            summary += f"   ({orphans} source label(s) with no counterpart)"
        ok = worst >= 0.99
    fig.suptitle(summary, fontsize=10, color="black" if ok else "crimson")
    fig.tight_layout()
    return fig, (1.0 if args.labels == "coco" else worst), orphans


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    random.seed(seed)
    print(f"seed {seed} (pass --seed {seed} to repeat this pick)")

    prepared, _ = load_prepared(args.source, args.split)
    image_dir = get_source(args.source).image_dir(args.split)
    on_disk = index_images(image_dir)

    if args.images:
        keys = [stem_key(n) for n in args.images]
        unknown = [n for n, k in zip(args.images, keys) if k not in prepared]
        if unknown:
            raise SystemExit(f"not in {args.source}/{args.split}: {unknown}")
    else:
        keys = sorted(prepared)
        if args.cls:
            if args.cls not in CLASS_NAMES:
                raise SystemExit(f"--class must be one of {CLASS_NAMES}")
            wanted = CLASS_NAMES.index(args.cls)
            keys = [k for k in keys
                    if any(a["category_id"] == wanted for a in prepared[k][1])]
            if not keys:
                raise SystemExit(f"no {args.cls} in {args.source}/{args.split}")
        keys = random.sample(keys, min(args.n, len(keys)))

    out_dir = args.out or (ROOT / "other_models" / "viz" / args.source)
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures, orphan_total = 0, 0
    for key in keys:
        if key not in on_disk:
            print(f"  {key}: image missing from {image_dir}")
            continue
        fig, worst, orphans = figure_for(args.source, args.split, on_disk[key],
                                         prepared[key], args)
        if args.labels == "coco":
            flag, extra = "", f"   {len(prepared[key][1])} annotations"
            print(f"  {on_disk[key].name:42s}{extra}")
        else:
            flag = "" if worst >= 0.99 else "   <-- CHECK"
            extra = f"   {orphans} source label(s) with no counterpart" if orphans else ""
            print(f"  {on_disk[key].name:42s} round-trip IoU min {worst:.3f}{extra}{flag}")
        failures += bool(flag)
        orphan_total += orphans
        if args.show:
            plt.show()
        else:
            target = out_dir / f"{on_disk[key].stem}.png"
            fig.savefig(target, dpi=110)
            print(f"      -> {target}")
        plt.close(fig)

    if failures:
        print(f"\n{failures} of {len(keys)} images need a look (matched IoU below 0.99)")
    if orphan_total and args.source == "taco":
        print(f"\nThe {orphan_total} source label(s) with no counterpart are expected: "
              f"taco_yolo/labels/ was\nbuilt from the FULL class mapping, while these COCO files "
              f"come from the reduced one, so\nevery instance the reduced mapping drops still has "
              f"a YOLO line on disk.")
    elif orphan_total:
        print(f"\n{orphan_total} source label(s) had no counterpart -- worth a look.")


if __name__ == "__main__":
    main()
