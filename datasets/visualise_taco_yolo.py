#!/usr/bin/env python3
"""Show TACO's original annotations next to the YOLO labels made from them.

Works on a YOLO dataset built from TACO's own annotations.json, and on one built from the
4-class file written by taco_merge_classes.py:

    annotations.json --taco_merge_classes.py--> annotations-richard.json --taco_to_yolo.py--> taco_yolo/

Left panel : the original TACO annotations, with TACO's 60 class names. For a merged dataset each
             tag also shows where the class went ("Glass bottle -> glass"). Annotations the merge
             discarded are drawn grey and dashed, tagged "-> discard", and are not expected on the right.
Right panel: the YOLO .txt labels read back from disk and drawn on the same image.

Both panels use the EXIF-corrected (upright) image, which is the frame TACO annotated in and the
frame Ultralytics trains on. Each figure title reports a round-trip check: for every annotation
that should have been written, the IoU of its box (its polygon's extent, for segment labels) with
the best-matching YOLO label of the class it was mapped to. 1.000 means the conversion is exact.
The title turns red if that drops below 0.99 or if the number of labels is not the number expected.

Three ways to pick images:
  * a random image (default)          - from all splits, or one split with --split
  * a random image of a chosen class  - --class "Glass bottle"; the chosen class is highlighted
                                        and everything else in the picture is faded
  * specific images                   - --images batch_1/000006.jpg batch_9/000032.jpg

--dropped switches to the images the merge threw away whole: every one of their annotations was
in a discarded class, so they have no YOLO label file and are not part of the dataset at all.
They are drawn with the usual left panel and an empty right panel saying why. Use it to check
the merge discarded the right pictures. It cannot be combined with --split (they are in none),
but it does combine with --class. Named images are always found, --dropped or not.

--class accepts a TACO class name, its id, or a supercategory name (all its classes), case-insensitive.
For a merged dataset it also accepts the merged classes (glass, metal, paper, plastic, discard),
selecting every TACO class mapped there. A unique part of a name is enough ("cigar" -> Cigarette).
Exact class names win over supercategories ("Paper bag" is both); use "super:Paper bag" for the
supercategory, e.g. "super:Paper" for TACO's Paper supercategory on a merged dataset.

What the script reads, and where it looks for it:
  * the annotation file the dataset was converted from: --annotations; else source_index.json in the
    dataset (written by taco_to_yolo.py); else the "# converted from" line of dataset.yaml; else the
    annotations*.json in ./TACO/data, ./data, . or next to the dataset that matches its labels
  * for a merged file: the .provenance.json sidecar named in the file's info (--provenance overrides)
    and, through it, the original annotations.json, so the 60 classes and discarded objects can be shown
  * the images: <yolo-dir>/images/<split>/, else --taco-dir, else the annotation files' folder

Usage:
    python visualise_taco_yolo.py --yolo-dir taco_yolo --list-classes
    python visualise_taco_yolo.py --yolo-dir taco_yolo --show                        # one random image
    python visualise_taco_yolo.py --yolo-dir taco_yolo --class "Glass bottle" --show
    python visualise_taco_yolo.py --yolo-dir taco_yolo --class glass --n 5           # merged class, 5 PNGs
    python visualise_taco_yolo.py --yolo-dir taco_yolo --class discard --show        # kept images that lost a box
    python visualise_taco_yolo.py --yolo-dir taco_yolo --dropped --n 5               # images dropped whole
    python visualise_taco_yolo.py --yolo-dir taco_yolo --dropped --class Cigarette --show
    python visualise_taco_yolo.py --yolo-dir taco_yolo --images batch_1/000006.jpg --show
    python visualise_taco_yolo.py --yolo-dir taco_yolo --annotations TACO/data/annotations-richard.json

Without --show the figures are saved as PNGs in <yolo-dir>/viz/. Random picks change on every run;
the seed is printed so a pick can be repeated with --seed. Notes about which files were used go to
stderr, so --list-classes output can be piped into grep.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib
    import numpy as np
    from PIL import Image, ImageOps
except ImportError as e:  # the system python has none of these; the project's .venv has them all
    raise SystemExit(f"{e.name} is required by this script: pip install matplotlib numpy Pillow\n"
                     f"(the project's .venv already has them: activate it, then python {__file__} ...)")

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
DISCARD_COLOUR = (0.55, 0.55, 0.55)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yolo-dir", type=Path, required=True, help="YOLO dataset folder (with images/ and labels/)")
    p.add_argument("--annotations", type=Path, default=None,
                   help="annotation file the dataset was converted from (default: found automatically)")
    p.add_argument("--provenance", type=Path, default=None,
                   help="taco_merge_classes.py sidecar, if the one named in the merged file has moved")
    p.add_argument("--taco-dir", type=Path, default=None, help="TACO data folder with the batch_* image folders")
    pick = p.add_mutually_exclusive_group()
    pick.add_argument("--list-classes", action="store_true",
                      help="print every TACO class with its id, supercategory, YOLO class and counts, then exit")
    pick.add_argument("--class", dest="cls", metavar="CLASS",
                      help="show random images containing this class (name, id, supercategory or merged class)")
    pick.add_argument("--images", nargs="+", metavar="IMAGE",
                      help="specific images, as TACO paths (batch_1/000006.jpg) or YOLO stems (batch_1_000006)")
    p.add_argument("--dropped", action="store_true",
                   help="show only the images the merge discarded whole (no YOLO label file); "
                        "combines with --class, not with --split")
    p.add_argument("--split", default=None, help="split to pick random images from: all, train, val, ... (default: all)")
    p.add_argument("--n", type=int, default=1, help="number of random images to show (default: 1)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the random pick (default: a new one each run, printed so it can be repeated)")
    p.add_argument("--out", type=Path, default=None, help="folder for the PNGs (default: <yolo-dir>/viz)")
    p.add_argument("--max-side", type=int, default=1600, help="downscale images to this longest side for drawing")
    p.add_argument("--show", action="store_true", help="open interactive windows instead of saving PNGs")
    args = p.parse_args()
    if args.dropped and args.split is not None:
        p.error("--dropped and --split cannot be combined: the dropped images are in no split "
                "(they have no label file). Drop --split, or drop --dropped.")
    args.split = args.split or "all"
    return args


def note(msg: str) -> None:
    print(msg, file=sys.stderr)


# --- locating and loading everything ------------------------------------------------------------

DROPPED_SPLIT = "dropped"            # pseudo-split for images that never reached the dataset


@dataclass
class Data:
    names: list[str]                 # YOLO class names
    index: dict[str, dict]           # YOLO stem -> {image_id, file_name, split, label, image}
    cats: list[dict]                 # TACO classes, as shown on the left
    anns: dict[int, list[dict]]      # image id -> original TACO annotations
    to_yolo: dict[int, int | None]   # TACO category id -> YOLO class id; None = discarded by the merge
    target: dict[int, str] | None    # TACO category id -> merged class name (merged datasets only)
    source: str                      # file the left panel comes from
    image_dirs: list[Path]           # folders that hold TACO's batch_*/ image folders
    dropped: dict[str, dict]         # same shape as index, for images with no label file (label=None)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        sys.exit(f"File not found: {path}")
    except ValueError as e:
        sys.exit(f"{path} is not valid JSON: {e}")


def locate(recorded: str | Path | None, *dirs: Path) -> Path | None:
    """A recorded path if it still exists, else a file of the same name in one of `dirs`."""
    if not recorded:
        return None
    p = Path(recorded)
    return next((c for c in [p, *(d / p.name for d in dirs)] if c.is_file()), None)


def image_keys(coco: dict) -> dict[str, dict]:
    """Every name a converter may have given an image: batch_1/000006 and batch_1_000006."""
    keys = {}
    for im in coco["images"]:
        base = im["file_name"].rsplit(".", 1)[0]
        keys[base] = keys[base.replace("/", "_")] = im
    return keys


def is_coco(obj) -> bool:
    return isinstance(obj, dict) and all(
        isinstance(obj.get(k), list) for k in ("images", "annotations", "categories"))


def read_source_index(path: Path) -> tuple[dict, dict | None]:
    """(index, coco) from source_index.json: whichever of the two shapes the file actually has.

    taco_to_yolo.py writes {"annotations": "<path>", "taco_dir": ..., "names": [...],
    "images": {stem: {...}}}. A COCO annotation file copied or renamed over it has the same
    key names with entirely different types ("annotations" is a list of boxes), which used to
    blow up as `Path(list)`. Recognise that case and hand the file back as annotations instead.
    """
    if not path.is_file():
        return {}, None
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        note(f"warning: ignoring {path.name}, it is not readable JSON ({e})")
        return {}, None
    if isinstance(obj, dict) and isinstance(obj.get("annotations"), str) and isinstance(obj.get("images"), dict):
        return obj, None
    if is_coco(obj):
        note(f"note: {path} is a COCO annotation file, not a taco_to_yolo index - using it as the "
             f"annotation file. (Put the real source_index.json back, or pass --annotations.)")
        return {}, obj
    note(f"warning: ignoring {path.name}, it is not a taco_to_yolo source index")
    return {}, None


def read_yaml(yolo_dir: Path) -> tuple[Path | None, list[str] | None, str]:
    """(yaml path, class names, raw text) of dataset.yaml / data.yaml / the first *.yaml in the folder."""
    for y in [yolo_dir / "dataset.yaml", yolo_dir / "data.yaml", *sorted(yolo_dir.glob("*.yaml"))]:
        if not y.is_file():
            continue
        text = y.read_text()
        try:
            import yaml
            raw = (yaml.safe_load(text) or {}).get("names")
        except ImportError:  # minimal fallback for the "  0: name" layout
            raw = {int(k): v for k, v in re.findall(r'^\s+(\d+):\s*"?(.*?)"?\s*$', text, re.M)}
        if isinstance(raw, dict):
            raw = [raw[k] for k in sorted(raw, key=int)]
        return y, (list(map(str, raw)) if raw else None), text
    return None, None, ""


def scan_labels(yolo_dir: Path) -> tuple[list[tuple[str, str, Path]], int]:
    """[(split, key, label file)] for labels/<split>/**.txt, and the highest class id used."""
    root = yolo_dir / "labels"
    out, max_cls = [], -1
    for lf in sorted(root.rglob("*.txt")):
        rel = lf.relative_to(root)
        split, rest = (rel.parts[0], Path(*rel.parts[1:])) if len(rel.parts) > 1 else ("", rel)
        out.append((split, rest.with_suffix("").as_posix(), lf))
        for line in lf.read_text().splitlines():
            if line.split():
                max_cls = max(max_cls, int(float(line.split()[0])))
    return out, max_cls


def find_annotations(yolo_dir: Path, keys: list[str], nc: int | None, max_cls: int) -> tuple[Path | None, dict | None]:
    """The annotations*.json nearby whose images and class count match the labels best."""
    dirs = [Path.cwd() / "TACO" / "data", Path.cwd() / "data", Path.cwd(),
            yolo_dir.parent / "TACO" / "data", yolo_dir.parent / "data", yolo_dir.parent, yolo_dir]
    seen, scored = set(), []
    for d in dirs:
        for f in sorted(d.glob("annotations*.json")) if d.is_dir() else []:
            if f.resolve() in seen or f.name.endswith(".provenance.json"):
                continue
            seen.add(f.resolve())
            try:
                coco = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(coco, dict) or not all(k in coco for k in ("images", "annotations", "categories")):
                continue
            n_cls = {len(coco["categories"]), len({c.get("supercategory") for c in coco["categories"]})}
            if (nc is not None and nc not in n_cls) or max_cls >= max(n_cls):
                continue
            have = image_keys(coco)
            hit = sum(k in have for k in keys) / max(1, len(keys))
            # Prefer: covers the labels' images, then the fewest classes that still fit the label ids
            # (a 4-class merged file over a 60-class one for 0-3 labels), then the closest image count.
            fit = min(n for n in n_cls if n > max_cls)
            scored.append((round(hit, 3), -fit, -abs(len(coco["images"]) - len(keys)), f, coco))
    scored = [s for s in scored if s[0] >= 0.9]
    if not scored:
        return None, None
    scored.sort(key=lambda s: s[:3], reverse=True)
    return scored[0][3], scored[0][4]


def build_class_map(categories: list[dict], names: list[str], path: Path) -> dict[int, int]:
    """Category id -> YOLO class id, by matching class names (or supercategories) to the dataset's names."""
    pos = {n: i for i, n in enumerate(names)}
    for field in ("name", "supercategory"):
        if all(c.get(field) in pos for c in categories):
            return {c["id"]: pos[c[field]] for c in categories}
    missing = [c["name"] for c in categories if c["name"] not in pos][:5]
    sys.exit(f"The classes in {path} ({', '.join(missing)}, ...) are not the dataset's classes "
             f"({', '.join(names[:5])}, ...). Pass --annotations with the file this dataset was converted from.")


def load_data(args: argparse.Namespace) -> Data:
    yolo_dir = args.yolo_dir
    if not (yolo_dir / "labels").is_dir():
        sys.exit(f"No labels/ folder in {yolo_dir} - --yolo-dir must point at a YOLO dataset (images/ and labels/)")
    idx, idx_coco = read_source_index(yolo_dir / "source_index.json")
    idx_coco_path = yolo_dir / "source_index.json" if idx_coco else None
    yaml_path, names, yaml_text = read_yaml(yolo_dir)
    idx_names = idx.get("names")
    names = (list(map(str, idx_names)) if isinstance(idx_names, list) else None) or names
    labels, max_cls = scan_labels(yolo_dir)
    if not labels:
        sys.exit(f"No label files under {yolo_dir}/labels")

    # 1. The annotation file the YOLO labels were converted from.
    conv_path, conv, how = None, None, ""
    comment = re.search(r"^# converted from (.+?) \| task=", yaml_text, re.M)
    if args.annotations:
        conv_path, how = args.annotations, "--annotations"
    elif isinstance(idx.get("annotations"), str):
        conv_path, how = locate(idx["annotations"], yolo_dir.parent), "source_index.json"
    elif idx_coco is not None:
        conv_path, conv, how = idx_coco_path, idx_coco, "source_index.json (a COCO file)"
    elif comment:
        conv_path, how = locate(comment.group(1), yolo_dir.parent), yaml_path.name
    if conv_path is None:
        conv_path, conv = find_annotations(yolo_dir, [k for _, k, _ in labels], len(names) if names else None, max_cls)
        how = "found automatically, pass --annotations to use another"
        if conv_path is None:
            sys.exit(f"Could not tell which annotation file {yolo_dir} was converted from. "
                     "Pass it with --annotations, e.g. --annotations TACO/data/annotations-richard.json")
    conv = conv or load_json(conv_path)
    note(f"labels converted from: {conv_path} ({how})")
    names = names or [c["name"] for c in sorted(conv["categories"], key=lambda c: c["id"])]
    class_map = build_class_map(conv["categories"], names, conv_path)

    # 2. Which TACO image each label file belongs to.
    keys = image_keys(conv)
    index, unmatched = {}, 0
    for split, key, lf in labels:
        im = keys.get(key)
        if im is None:
            unmatched += 1
            continue
        stem = key.replace("/", "_")
        ext = Path(im["file_name"]).suffix
        index[stem] = {"image_id": im["id"], "file_name": im["file_name"], "width": im["width"],
                       "height": im["height"], "split": split or "-", "label": lf,
                       "image": yolo_dir / "images" / split / f"{key}{ext}"}
    if not index:
        sys.exit(f"None of the {len(labels)} label files match an image in {conv_path}. "
                 "Pass --annotations with the file this dataset was converted from.")
    if unmatched:
        note(f"warning: {unmatched} of {len(labels)} label files match no image in {conv_path.name}, skipped")

    # 3. The original 60-class annotations: the converted file itself, or what the merge started from.
    info = conv.get("info") or {}
    prov_path = args.provenance or (locate(info.get("provenance"), conv_path.parent) if "merged_by" in info else None)
    if "merged_by" in info and prov_path is None:
        note(f"warning: {conv_path.name} was made by taco_merge_classes.py, but its provenance file "
             f"{info.get('provenance')} was not found, so the left panel shows the merged classes. "
             "Pass --provenance to show the original TACO classes.")
    anns = defaultdict(list)
    pre_images: dict[int, dict] = {}   # every image known before conversion, by id
    merge_dropped: dict[int, dict] = {}  # those the merge itself discarded, from the sidecar
    taco_dir = idx.get("taco_dir")
    target, image_dirs = None, [args.taco_dir, Path(taco_dir) if isinstance(taco_dir, str) else None,
                                conv_path.parent]
    if prov_path is None:
        cats, source = conv["categories"], conv_path.name
        for a in conv["annotations"]:
            anns[a["image_id"]].append(a)
        pre_images = {im["id"]: im for im in conv["images"]}
        to_yolo = dict(class_map)
    else:
        prov = load_json(prov_path)
        if "forward_map" not in prov:
            sys.exit(f"{prov_path} is not a taco_merge_classes.py provenance file")
        fwd = {int(k): e for k, e in prov["forward_map"].items()}
        to_yolo = {k: None if e["dropped"] else class_map[e["target_id"]] for k, e in fwd.items()}
        target = {k: e["target"] for k, e in fwd.items()}
        merge_dropped = {d["id"]: d for d in prov.get("dropped_images", [])}
        orig_path = locate(prov.get("input", {}).get("path"), conv_path.parent, prov_path.parent)
        note(f"merge record: {prov_path}")
        if orig_path is not None:
            orig = load_json(orig_path)
            cats, source = orig["categories"], f"{orig_path.name}, before the merge"
            for a in orig["annotations"]:
                anns[a["image_id"]].append(a)
            pre_images = {im["id"]: im for im in orig["images"]}
            image_dirs.append(orig_path.parent)
            note(f"original annotations: {orig_path}")
        else:  # rebuild from the merged file and the sidecar's reverse map (+ dropped records if embedded)
            rev = prov["reverse_annotation_map"]
            for a in conv["annotations"]:
                anns[a["image_id"]].append({**a, "category_id": rev[f"{a['image_id']}:{a['id']}"]})
            for a in prov.get("dropped_annotation_records", []):
                anns[a["image_id"]].append(a)
            cats, source = prov["source_categories"], f"{conv_path.name} + {prov_path.name}"
            # width/height only come with the full records; the sidecar's dropped_images list
            # carries just id/file_name/category_ids, so those sizes are read off the image later.
            pre_images = {im["id"]: im for im in
                          [*conv["images"], *prov.get("dropped_image_records", []), *merge_dropped.values()]}
            if not prov.get("dropped_annotation_records"):
                note(f"warning: original file {prov.get('input', {}).get('path')} not found; "
                     "annotations discarded by the merge cannot be shown")
    image_dirs = list(dict.fromkeys(d for d in image_dirs if d is not None))

    # 4. Images that never reached the dataset: in the pre-merge annotations, but with no label file.
    in_dataset = {e["image_id"] for e in index.values()}
    dropped = {}
    for iid, im in sorted(pre_images.items()):
        if iid in in_dataset or "file_name" not in im:
            continue
        n = len(anns.get(iid, []))
        reason = (f"all {n} annotation{'s' if n != 1 else ''} discarded by the merge"
                  if iid in merge_dropped else "not converted (image missing, or skipped by taco_to_yolo.py)")
        dropped[im["file_name"].replace("/", "_").rsplit(".", 1)[0]] = {
            "image_id": iid, "file_name": im["file_name"], "width": im.get("width"),
            "height": im.get("height"), "split": DROPPED_SPLIT, "label": None,
            "image": None, "reason": reason}
    return Data(names, index, cats, dict(anns), to_yolo, target, source, image_dirs, dropped)


# --- class selection and listing ----------------------------------------------------------------

def resolve_class(query: str, d: Data) -> tuple[set[int], str]:
    """Turn --class into (set of TACO category ids, display name).

    Tried in order: super: prefix, id, exact class name, exact merged class, exact supercategory,
    then a unique partial match on any of those.
    """
    cats = d.cats
    merged = defaultdict(set)
    for cid, t in (d.target or {}).items():
        merged[t].add(cid)
    supers = {c["supercategory"]: {x["id"] for x in cats if x["supercategory"] == c["supercategory"]} for c in cats}
    q = query.strip().lower()
    if q.startswith("super:"):
        q = q[len("super:"):].strip()
        for s, ids in supers.items():
            if s.lower() == q:
                return ids, f"{s} (supercategory)"
        sys.exit(f"No TACO supercategory called '{query[6:].strip()}'. Run with --list-classes to see them.")
    if q.isdigit():
        hit = [c for c in cats if c["id"] == int(q)]
        if not hit:
            sys.exit(f"No TACO class with id {q}. Run with --list-classes to see them.")
        return {hit[0]["id"]}, hit[0]["name"]

    options = {}  # display name -> ids, in order of preference
    options.update({c["name"]: frozenset({c["id"]}) for c in cats})
    options.update({f"{t} (merged class)": frozenset(ids) for t, ids in sorted(merged.items())})
    options.update({f"{s} (supercategory)": frozenset(ids) for s, ids in supers.items()})
    bare = lambda label: label.rsplit(" (", 1)[0].lower() if label.endswith(")") else label.lower()
    for label, ids in options.items():
        if bare(label) == q:
            return set(ids), label
    partial = {label: ids for label, ids in options.items() if q in bare(label)}
    if len(set(partial.values())) == 1:  # one class, possibly found by name and by supercategory
        label, ids = next(iter(partial.items()))
        return set(ids), label
    if partial:
        sys.exit(f"'{query}' matches several classes - be more specific:\n  " + "\n  ".join(partial))
    lower = {bare(label): label for label in options}
    close = [lower[m] for m in difflib.get_close_matches(q, list(lower), n=5, cutoff=0.5)]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    sys.exit(f"No class matches '{query}'.{hint} Run with --list-classes to see them all.")


def read_yolo(label_file: Path, w: int, h: int) -> list[tuple[int, np.ndarray | None, np.ndarray]]:
    """Return [(class, polygon_px or None, box_xyxy_px)] from a YOLO detect or segment label file."""
    out = []
    for line in label_file.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        c, nums = int(float(v[0])), np.array(v[1:], dtype=np.float64)
        if len(nums) == 4:  # detect: cx cy w h
            cx, cy, bw, bh = nums * [w, h, w, h]
            out.append((c, None, np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])))
        else:  # segment: x1 y1 x2 y2 ...
            poly = nums.reshape(-1, 2) * [w, h]
            out.append((c, poly, np.r_[poly.min(0), poly.max(0)]))
    return out


def list_classes(d: Data) -> None:
    split_of = {e["image_id"]: e["split"] for e in d.index.values()}
    splits = sorted(set(split_of.values()), key=lambda s: (SPLIT_ORDER.get(s, 9), s))
    images, instances = defaultdict(lambda: defaultdict(set)), Counter()
    for iid, split in split_of.items():
        for a in d.anns.get(iid, []):
            images[a["category_id"]][split].add(iid)
            instances[a["category_id"]] += 1
    show_yolo_name = d.target is not None or any(
        d.to_yolo[c["id"]] is not None and d.names[d.to_yolo[c["id"]]] != c["name"] for c in d.cats)

    def yolo_txt(cid: int) -> str:
        y = d.to_yolo.get(cid)
        return "discard" if y is None else f"{y} {d.names[y]}" if show_yolo_name else str(y)

    width = max(4, *(len(yolo_txt(c["id"])) for c in d.cats))
    per_split = "/".join(splits)
    print(f"TACO classes, counted on the {len(split_of)} images in the dataset ({d.source})")
    print(f"{'id':>3}  {'class':<26}{'supercategory':<23}{'yolo':<{width}}  {'instances':>9}  {'images':>6}  {per_split}")
    for c in sorted(d.cats, key=lambda c: c["id"]):
        counts = [len(images[c["id"]][s]) for s in splits]
        print(f"{c['id']:>3}  {c['name']:<26}{c['supercategory']:<23}{yolo_txt(c['id']):<{width}}  "
              f"{instances[c['id']]:>9}  {sum(counts):>6}  {'/'.join(map(str, counts))}")
    if not show_yolo_name:
        return

    # The dataset's own classes differ from TACO's: summarise them, counted from the label files.
    y_inst, y_imgs = Counter(), defaultdict(lambda: defaultdict(set))
    for stem, e in d.index.items():
        if e["label"] is None:
            continue
        for line in e["label"].read_text().splitlines():
            if line.split():
                y = int(float(line.split()[0]))
                y_inst[y] += 1
                y_imgs[y][e["split"]].add(stem)
    print(f"\nYOLO classes in the label files ({len(d.names)})")
    print(f"{'yolo':>4}  {'class':<24}{'instances':>9}  {'images':>6}  {per_split:<12}  from TACO ids")
    for y, name in enumerate(d.names):
        counts = [len(y_imgs[y][s]) for s in splits]
        src = ",".join(str(c["id"]) for c in sorted(d.cats, key=lambda c: c["id"]) if d.to_yolo[c["id"]] == y)
        print(f"{y:>4}  {name:<24}{y_inst[y]:>9}  {sum(counts):>6}  {'/'.join(map(str, counts)):<12}  {src}")
    dropped = sorted(c for c, y in d.to_yolo.items() if y is None)
    if dropped:
        n = sum(instances[c] for c in dropped)
        print(f"{'-':>4}  {'discard (not written)':<24}{n:>9}  {'':>6}  {'':<12}  {','.join(map(str, dropped))}")
    if d.dropped:
        n = sum(len(d.anns.get(e["image_id"], [])) for e in d.dropped.values())
        print(f"\n{len(d.dropped)} further images are not in the dataset at all ({n} annotations): "
              f"nothing survived the merge on them.\nThe counts above exclude them - see --dropped.")


# --- drawing ------------------------------------------------------------------------------------

def colour(cls: int):
    """Saturated colour per YOLO class (golden-ratio hue steps), readable under white label text."""
    from matplotlib.colors import hsv_to_rgb
    return tuple(hsv_to_rgb(((cls * 0.618034) % 1.0, 0.85, 0.8 if cls % 2 else 0.95)))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def round_trip(anns: list[dict], rows: list, to_yolo: dict[int, int | None], w: int, h: int) -> list[float]:
    """Best same-class IoU for every annotation that should be in the labels (discarded ones skipped)."""
    lim = np.array([w, h, w, h], dtype=float)
    ious = []
    for a in anns:
        c = to_yolo.get(a["category_id"])
        if c is None:
            continue
        x, y, bw, bh = a["bbox"]
        box = np.clip([x, y, x + bw, y + bh], 0, lim)
        parts = [np.asarray(p, float).reshape(-1, 2) for p in a.get("segmentation") or [] if isinstance(p, list) and p]
        ext = np.clip(np.r_[np.concatenate(parts).min(0), np.concatenate(parts).max(0)], 0, lim) if parts else box
        ious.append(max((iou(ext if poly is not None else box, b) for rc, poly, b in rows if rc == c), default=0.0))
    return ious


def label(ax, x, y, text, col, alpha):
    inside = y < 25  # box touches the top edge: put the tag inside it so it is not cut off
    ax.text(max(x, 0), max(y, 0), text, fontsize=7, color="white", alpha=min(1.0, alpha + 0.1),
            va="top" if inside else "bottom", ha="left", clip_on=True,
            bbox=dict(facecolor=col, edgecolor="none", pad=1.2, alpha=alpha))


def draw(ax, img, items, s):
    """items: [(colour, [polygons_px], box_xyxy_px, text, highlighted, dashed)]; s: display scale."""
    from matplotlib.patches import Polygon, Rectangle
    ax.imshow(img)
    for col, polys, box, text, on, dashed in sorted(items, key=lambda it: it[4]):  # highlighted on top
        edge, face, lw = (1.0, 0.35, 1.8) if on else (0.35, 0.08, 0.8)
        if dashed:
            face = face / 2
        ls = (0, (4, 2)) if dashed else "-"
        for poly in polys:
            ax.add_patch(Polygon(poly * s, closed=True, facecolor=(*col, face), edgecolor=(*col, edge),
                                 lw=lw * 0.67, linestyle=ls))
        x1, y1, x2, y2 = box * s
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=(*col, edge), lw=lw, linestyle=ls))
        label(ax, x1, y1, text, col, alpha=0.9 if on else 0.35)
    ax.set_axis_off()


# --- main ---------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    d = load_data(args)

    if args.list_classes:
        list_classes(d)
        return

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cat_name = {c["id"]: c["name"] for c in d.cats}
    # Named images are looked up in the dataset and among the dropped ones alike.
    everything = {**d.index, **d.dropped}
    focus, focus_name = None, None  # TACO category ids to highlight when --class is used
    if args.images:
        by_file = {e["file_name"]: k for k, e in everything.items()}
        stems = [by_file.get(x, x) for x in args.images]
        unknown = [s for s in stems if s not in everything]
        if unknown:
            sys.exit(f"Not in the dataset: {unknown}")
    else:
        if args.dropped and not d.dropped:
            sys.exit(f"Nothing was dropped: every image in the annotations {args.yolo_dir} was built "
                     "from has a label file. --dropped only applies to a dataset converted from a "
                     "taco_merge_classes.py file, where whole images can lose all their annotations.")
        source = d.dropped if args.dropped else d.index
        splits = sorted({e["split"] for e in d.index.values()})
        if args.split != "all" and args.split not in splits:
            sys.exit(f"No split called '{args.split}' in {args.yolo_dir}; it has: all, {', '.join(splits)}")
        where = ("the images dropped by the merge" if args.dropped else
                 "all splits" if args.split == "all" else f"the {args.split} split")
        pool = sorted(k for k, e in source.items() if args.dropped or args.split in ("all", e["split"]))
        if args.cls:
            focus, focus_name = resolve_class(args.cls, d)
            pool = [k for k in pool if any(a["category_id"] in focus for a in d.anns.get(source[k]["image_id"], []))]
            if not pool:
                tip = (" Try without --dropped." if args.dropped else
                       " Try --split all." if args.split != "all" else "")
                sys.exit(f"No image contains {focus_name} in {where}.{tip}")
            print(f"{focus_name}: {len(pool)} images in {where}")
        if not pool:
            sys.exit(f"No images in {where}.")
        seed = args.seed if args.seed is not None else random.randrange(1_000_000)
        stems = random.Random(seed).sample(pool, min(args.n, len(pool)))
        if args.n > len(pool):
            print(f"Only {len(pool)} images to choose from, showing all of them")
        print(f"seed {seed} (add --seed {seed} to get the same pick again)")
    focus_yolo = {d.to_yolo[c] for c in focus if d.to_yolo.get(c) is not None} if focus else None

    out = args.out or args.yolo_dir / "viz"
    if not args.show:
        out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        e = everything[stem]
        gone = e["label"] is None              # image never made it into the dataset
        anns = d.anns.get(e["image_id"], [])
        src = e["image"] if e["image"] and e["image"].is_file() else next(
            (p / e["file_name"] for p in d.image_dirs if (p / e["file_name"]).is_file()), None)
        if src is None:
            sys.exit(f"Image {e['file_name']} not found in {args.yolo_dir}/images or {d.image_dirs}; pass --taco-dir")
        with Image.open(src) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")  # upright, as annotated and as YOLO sees it
        # the size the labels were normalised by; dropped images may have no recorded size
        w, h = e["width"] or img.width, e["height"] or img.height
        size_ok = img.size == (w, h)
        s = min(1.0, args.max_side / max(img.size))
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)

        # Left: original TACO annotations, unmodified (boxes not clipped, every polygon part drawn).
        left = []
        for a in anns:
            cid, y = a["category_id"], d.to_yolo.get(a["category_id"])
            x0, y0, bw, bh = a["bbox"]
            polys = [np.asarray(p, float).reshape(-1, 2) for p in a.get("segmentation") or [] if isinstance(p, list) and p]
            text = cat_name.get(cid, str(cid)) + (f" → {d.target[cid]}" if d.target else "")
            left.append((DISCARD_COLOUR if y is None else colour(y), polys, np.array([x0, y0, x0 + bw, y0 + bh]),
                         text, focus is None or cid in focus, y is None))

        # Right: YOLO labels read back from disk - unless there is no label file to read.
        rows = [] if gone else read_yolo(e["label"], w, h)
        right = [(colour(c), [poly] if poly is not None else [], box,
                  d.names[c] if c < len(d.names) else f"class {c}", focus_yolo is None or c in focus_yolo, False)
                 for c, poly, box in rows]

        ious = round_trip(anns, rows, d.to_yolo, w, h)
        n_disc = len(anns) - len(ious)
        check = e["reason"] if gone else f"min IoU vs original {min(ious):.3f}" if ious else "nothing to check"
        bad = not gone and (bool(ious) and min(ious) < 0.99 or len(rows) != len(ious))
        if bad and len(rows) != len(ious):
            check += f"; {len(rows)} labels for {len(ious)} annotations"

        n_left, n_right = sum(it[4] for it in left), sum(it[4] for it in right)
        hl_left = f", {n_left} highlighted" if focus else ""
        hl_right = f", {n_right} highlighted" if focus else ""
        disc = f", {n_disc} discarded by the merge" if n_disc else ""
        fig_w = min(20.0, max(10.0, 2 * 7 * w / h + 0.5))  # fits a screen for portrait and panoramic shots
        fig_h = min(8.2, (fig_w - 0.5) / 2 * h / w + 1.3)
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        draw(axes[0], img, left, s)
        draw(axes[1], img, right, s)
        axes[0].set_title(f"Original TACO annotations: {len(anns)}{disc}{hl_left}\n({d.source})", fontsize=10)
        if gone:
            axes[1].set_title(f"YOLO labels: none - this image is not in the dataset\n({check})",
                              fontsize=10, color=DISCARD_COLOUR)
            axes[1].text(0.5, 0.5, f"no YOLO label file\n{e['reason']}", transform=axes[1].transAxes,
                         ha="center", va="center", fontsize=12, color="white",
                         bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=8))
        else:
            axes[1].set_title(f"YOLO labels read back: {len(rows)}{hl_right}\n({check})", fontsize=10,
                              color="red" if bad else "black")
        cls_txt = f"   |   class: {focus_name}" if focus else ""
        warn = "" if size_ok else f"\nWARNING: upright image is {img.size}, JSON says {(w, h)}"
        dest = f"{DROPPED_SPLIT} (no label file)" if gone else f"{e['split']}/{stem}"
        fig.suptitle(f"{e['file_name']}  ->  {dest}{cls_txt}{warn}", fontsize=11,
                     color="black" if size_ok else "red")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        counts = f"{len(anns):3d} original{disc}{hl_left}  {len(rows):3d} yolo{hl_right}"
        print(f"{e['file_name']:28s} {counts}   {check}{warn}")
        if args.show:
            plt.show()
        else:
            fig.savefig(out / f"{stem}.png", dpi=110)
            plt.close(fig)

    if not args.show:
        print(f"Saved {len(stems)} figure(s) to {out}/")


if __name__ == "__main__":
    main()
