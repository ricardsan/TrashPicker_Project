#!/usr/bin/env python3
"""Collapse TACO's 60 litter categories onto the project's 4 material classes.

TACO's annotations.json is COCO-style with 60 fine-grained categories (0-based ids).
This project trains on `glass 0, metal 1, paper 2, plastic 3` (data_v2.yaml), so the
60 categories are merged onto those 4. Categories whose material cannot be determined
(Unlabeled litter, Cigarette, Shoe, Rope & strings, Battery, Food waste, ...) map to a
virtual `discard` class: those annotations are not written, and an image left with no
annotation at all is not written either.

The merge is lossy, so every run also writes a provenance sidecar next to the output
(<stem>.provenance.json) holding the full forward map, a per-annotation map back to the
original fine-grained category, and the ids of everything dropped. That file alone is
enough to explain or reverse the merge - see --undo.

Note on annotation ids: TACO's annotations.json reuses two ids (309 and 4040) on
different images - 4784 annotations but only 4782 distinct ids - so the reverse map is
keyed by "<image_id>:<annotation id>", which is unique across the file. Ids are passed
through to the output unchanged (duplicates included) so the two files stay comparable.

The input annotation file is never modified.

Usage:
    # report what would happen, write nothing
    python taco_merge_classes.py --annotations TACO/data/annotations.json \
        --mapping TACO/taco_class_mapping.json \
        --out TACO/data/annotations-richard.json --report-only

    # do it
    python taco_merge_classes.py --annotations TACO/data/annotations.json \
        --mapping TACO/taco_class_mapping.json \
        --out TACO/data/annotations-richard.json

    # reverse it, driven entirely by the sidecar
    python taco_merge_classes.py --undo TACO/data/annotations-richard.provenance.json \
        --out TACO/data/annotations-restored.json

The output drops straight into the existing converter:
    python taco_to_yolo.py --taco-dir TACO/data \
        --annotations TACO/data/annotations-richard.json --out taco_yolo_richard
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent

# Must match data_v2.yaml / dataset_seg/data.yaml so the ids need no further remapping.
TARGET_CLASSES = {"glass": 0, "metal": 1, "paper": 2, "plastic": 3}
DISCARD = "discard"

# taco_class_mapping.json calls the virtual class "skip"; "discard" is the name used here.
DISCARD_ALIASES = {"discard", "skip", "drop", "none"}

def ann_key(ann: dict) -> str:
    """Unique key for one annotation.

    Plain annotation ids are not unique in TACO (309 and 4040 each appear on two
    different images), so the image id is folded in; the pair is unique across the file.
    """
    return f"{ann['image_id']}:{ann['id']}"


HOW_TO_UNDO = """\
Two ways back, in order of preference.

1. Full restore (nothing to run). The merge never touched its input, so
   `input.path` below IS the original 60-class file. Check it still hashes to
   `input.sha256` and use it directly. Anything derived from the merged file
   (e.g. a YOLO dataset built by taco_to_yolo.py) can simply be rebuilt from it.

2. Self-contained restore, when the original file is gone or you only have this
   sidecar plus the merged json:

       python taco_merge_classes.py --undo <this file> --out annotations-restored.json

   This rebuilds the 60 `categories` from `forward_map` and puts each surviving
   annotation's original `category_id` back via `reverse_annotation_map`, keyed
   "<image_id>:<annotation id>". Both were preserved through the merge and the pair is
   unique in the source (plain annotation ids are not - TACO reuses 309 and 4040), so
   the lookup is exact: no geometry matching, no guessing.

   Limit: unless this sidecar was written with --embed-dropped (see
   `dropped_records_embedded`), the annotations and images that the merge discarded
   cannot be recreated - their bboxes and polygons were never written to the merged
   file. `dropped_annotations` and `dropped_images` list their image ids, annotation
   ids and original categories - enough to identify them in the original file, not to
   reconstruct them. With --embed-dropped the full records are in
   `dropped_annotation_records` / `dropped_image_records` /
   `dropped_scene_annotation_records` and the restore reproduces the original exactly.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path,
                   help="source COCO file (TACO data/annotations.json)")
    p.add_argument("--mapping", type=Path,
                   help="json with the 60 -> 4 mapping (TACO/taco_class_mapping.json)")
    p.add_argument("--out", type=Path, required=True, help="file to write")
    p.add_argument("--undo", type=Path, default=None, metavar="PROVENANCE",
                   help="reverse a previous merge using its .provenance.json sidecar")
    p.add_argument("--report-only", action="store_true",
                   help="print the report and exit without writing anything")
    p.add_argument("--embed-dropped", action="store_true",
                   help="store the full records of the dropped annotations, images and "
                        "scene annotations in the sidecar, making --undo lossless "
                        "(costs ~560 KB)")
    p.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = p.parse_args()
    if args.undo is None and (args.annotations is None or args.mapping is None):
        p.error("--annotations and --mapping are required unless --undo is given")
    return args


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def portable(path: Path) -> str:
    """Repo-relative POSIX path, so the written metadata does not pin one machine's layout."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def file_ref(path: Path) -> dict:
    return {"path": portable(path), "sha256": sha256_of(path),
            "bytes": path.stat().st_size}


def write_json(path: Path, obj: dict, force: bool) -> None:
    if path.exists() and not force:
        sys.exit(f"{path} already exists - pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def provenance_path(out: Path) -> Path:
    return out.with_suffix("").with_suffix(".provenance.json")


# --------------------------------------------------------------------------- forward

def build_forward_map(categories: list[dict], mapping: dict, counts: Counter) -> dict:
    """{source cat id (str) -> {name, target, target_id, dropped, n_annotations}}.

    Validated against the category list: a source id missing from the mapping file, or
    carrying a target that is neither one of the 4 classes nor a discard alias, is a
    hard error - silently defaulting either way would corrupt the labels.
    """
    raw = mapping.get("mapping")
    if not isinstance(raw, dict):
        sys.exit("mapping file has no 'mapping' object")

    fwd, problems = {}, []
    for c in sorted(categories, key=lambda c: c["id"]):
        cid = c["id"]
        entry = raw.get(str(cid))
        if entry is None:
            problems.append(f"  category {cid} ({c['name']!r}) is not in the mapping file")
            continue
        target = str(entry.get("target", "")).lower()
        if target in DISCARD_ALIASES:
            fwd[str(cid)] = {"name": c["name"], "supercategory": c.get("supercategory"),
                             "target": DISCARD, "target_id": None, "dropped": True,
                             "n_annotations": counts[cid]}
        elif target in TARGET_CLASSES:
            fwd[str(cid)] = {"name": c["name"], "supercategory": c.get("supercategory"),
                             "target": target, "target_id": TARGET_CLASSES[target],
                             "dropped": False, "n_annotations": counts[cid]}
        else:
            problems.append(f"  category {cid} ({c['name']!r}) has unknown target "
                            f"{entry.get('target')!r}")
    if problems:
        sys.exit("mapping is incomplete or invalid:\n" + "\n".join(problems) +
                 f"\nexpected targets: {sorted(TARGET_CLASSES)} or a discard alias "
                 f"{sorted(DISCARD_ALIASES)}")
    return fwd


def print_report(data: dict, fwd: dict, counts: Counter, kept_ann: list,
                 dropped_ann: list, kept_imgs: list, dropped_imgs: list,
                 per_target: Counter, imgs_with_target: Counter) -> None:
    print(f"\n{'id':>3}  {'supercategory':22s} {'name':26s} {'n':>5}  -> new class")
    print("-" * 78)
    for cid in sorted(fwd, key=int):
        e = fwd[cid]
        flag = "" if not e["dropped"] else "   (not written)"
        tgt = e["target"] if e["dropped"] else f"{e['target']} ({e['target_id']})"
        print(f"{cid:>3}  {str(e['supercategory'] or '')[:22]:22s} {e['name'][:26]:26s} "
              f"{e['n_annotations']:5d}  -> {tgt}{flag}")

    print("\nnew classes")
    for name, idx in sorted(TARGET_CLASSES.items(), key=lambda kv: kv[1]):
        src = [cid for cid in sorted(fwd, key=int) if fwd[cid]["target"] == name]
        print(f"  {idx} {name:8s} {per_target[idx]:5d} annotations in "
              f"{imgs_with_target[idx]:5d} images   from {len(src)} TACO categories: "
              f"{','.join(src)}")
    disc = [cid for cid in sorted(fwd, key=int) if fwd[cid]["dropped"]]
    print(f"  - {DISCARD:8s} {len(dropped_ann):5d} annotations"
          f"{'':20s}from {len(disc)} TACO categories: {','.join(disc)}")
    for cid in disc:
        print(f"      {cid:>3} {fwd[cid]['name']:26s} {fwd[cid]['n_annotations']:5d}")

    n_partial = len({a["image_id"] for a in dropped_ann} & {i["id"] for i in kept_imgs})
    print(f"\nannotations  {len(data['annotations']):5d} in -> {len(kept_ann):5d} kept, "
          f"{len(dropped_ann):5d} discarded")
    print(f"images       {len(data['images']):5d} in -> {len(kept_imgs):5d} kept, "
          f"{len(dropped_imgs):5d} dropped (nothing left to label)")
    print(f"             {n_partial:5d} kept images lost at least one annotation")
    print(f"categories   {len(data['categories']):5d} in -> "
          f"{len(TARGET_CLASSES):5d} out\n")


def merge(args: argparse.Namespace) -> None:
    prov_out = provenance_path(args.out)
    if not args.report_only and not args.force:        # fail before doing the work
        for p in (args.out, prov_out):
            if p.exists():
                sys.exit(f"{p} already exists - pass --force to overwrite it, or "
                         f"--report-only to just see what the merge would do")

    data = json.loads(args.annotations.read_text())
    mapping = json.loads(args.mapping.read_text())

    counts = Counter(a["category_id"] for a in data["annotations"])
    fwd = build_forward_map(data["categories"], mapping, counts)

    unknown = sorted(set(counts) - {int(k) for k in fwd})
    if unknown:
        sys.exit(f"annotations reference category ids absent from 'categories': {unknown}")

    # TACO reuses a couple of annotation ids across images; ann_key() handles that, but
    # if even (image_id, id) collided the reverse map would be ambiguous - refuse instead.
    id_counts = Counter(a["id"] for a in data["annotations"])
    duplicate_ids = sorted(i for i, n in id_counts.items() if n > 1)
    key_counts = Counter(ann_key(a) for a in data["annotations"])
    collisions = sorted(k for k, n in key_counts.items() if n > 1)
    if collisions:
        sys.exit(f"{len(collisions)} annotations share an (image_id, id) pair, so the "
                 f"merge could not be reversed unambiguously: {collisions[:10]}")

    kept_ann, dropped_ann, reverse = [], [], {}
    for a in data["annotations"]:
        e = fwd[str(a["category_id"])]
        if e["dropped"]:
            dropped_ann.append(a)
            continue
        new = dict(a)                            # bbox, area, iscrowd, segmentation verbatim
        new["category_id"] = e["target_id"]      # only the class changes
        kept_ann.append(new)
        reverse[ann_key(a)] = a["category_id"]

    live = {a["image_id"] for a in kept_ann}
    kept_imgs = [im for im in data["images"] if im["id"] in live]
    dropped_imgs = [im for im in data["images"] if im["id"] not in live]

    per_target = Counter(a["category_id"] for a in kept_ann)
    by_img = defaultdict(set)
    for a in kept_ann:
        by_img[a["image_id"]].add(a["category_id"])
    imgs_with_target = Counter(c for cls in by_img.values() for c in cls)

    print_report(data, fwd, counts, kept_ann, dropped_ann, kept_imgs, dropped_imgs,
                 per_target, imgs_with_target)
    if duplicate_ids:
        print(f"note: the source reuses {len(duplicate_ids)} annotation id(s) on "
              f"different images: {duplicate_ids}.\n      Ids are passed through "
              f"unchanged; the reverse map is keyed '<image_id>:<id>', which is unique.\n")

    if args.report_only:
        print("--report-only: nothing written.")
        return

    discarded_cats = [{"id": int(cid), "name": fwd[cid]["name"],
                       "n_annotations": fwd[cid]["n_annotations"]}
                      for cid in sorted(fwd, key=int) if fwd[cid]["dropped"]]
    out = {
        "info": {
            **(data.get("info") or {}),
            "description": f"TACO, {len(data['categories'])} categories merged onto "
                           f"{len(TARGET_CLASSES)} material classes "
                           f"({', '.join(f'{n}={i}' for n, i in sorted(TARGET_CLASSES.items(), key=lambda kv: kv[1]))}). "
                           f"Annotations of non-sortable categories, and images left with "
                           f"no annotation, are not included.",
            "source": portable(args.annotations),
            "merged_by": f"taco_merge_classes.py v{SCRIPT_VERSION}",
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "provenance": portable(prov_out),
            "target_classes": TARGET_CLASSES,
            "discarded_categories": discarded_cats,
            "counts": {"annotations_in": len(data["annotations"]),
                       "annotations_kept": len(kept_ann),
                       "images_in": len(data["images"]),
                       "images_kept": len(kept_imgs)},
        },
        "images": kept_imgs,
        "annotations": kept_ann,
        "scene_annotations": [s for s in data.get("scene_annotations", [])
                              if s["image_id"] in live],
        "licenses": data.get("licenses", []),
        "categories": [{"supercategory": n, "id": i, "name": n}
                       for n, i in sorted(TARGET_CLASSES.items(), key=lambda kv: kv[1])],
        "scene_categories": data.get("scene_categories", []),
    }
    write_json(args.out, out, force=True)        # existence already checked above

    dropped_by_img = defaultdict(list)
    for a in dropped_ann:
        dropped_by_img[a["image_id"]].append(a["category_id"])
    prov = {
        "what_this_is": f"Record of the lossy {len(data['categories'])}->"
                        f"{len(TARGET_CLASSES)} class merge that produced "
                        f"{args.out.name} from {args.annotations.name}.",
        "how_to_undo": HOW_TO_UNDO,
        "created": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "command": " ".join(sys.argv),
        "input": file_ref(args.annotations),
        "output": file_ref(args.out),
        "mapping_file": file_ref(args.mapping),
        "target_classes": TARGET_CLASSES,
        "discard_class": {"name": DISCARD,
                          "meaning": "material not determinable or not one of the four "
                                     "streams; annotation is not written"},
        "forward_map": fwd,
        # why the debatable categories went where they did, carried over from the
        # mapping file so the reasoning travels with the record
        "mapping_notes": mapping.get("notes", {}),
        "reverse_annotation_map_key": "<image_id>:<annotation id>",
        "reverse_annotation_map": reverse,
        "duplicate_source_annotation_ids": duplicate_ids,
        "dropped_records_embedded": bool(args.embed_dropped),
        "dropped_annotations": [{"id": a["id"], "image_id": a["image_id"],
                                 "category_id": a["category_id"]} for a in dropped_ann],
        "dropped_images": [{"id": im["id"], "file_name": im["file_name"],
                            "category_ids": sorted(set(dropped_by_img[im["id"]]))}
                           for im in dropped_imgs],
        "counts": {
            "images": {"in": len(data["images"]), "kept": len(kept_imgs),
                       "dropped": len(dropped_imgs)},
            "annotations": {"in": len(data["annotations"]), "kept": len(kept_ann),
                            "dropped": len(dropped_ann)},
            "kept_per_target": {n: per_target[i]
                                for n, i in sorted(TARGET_CLASSES.items(),
                                                   key=lambda kv: kv[1])},
            "images_per_target": {n: imgs_with_target[i]
                                  for n, i in sorted(TARGET_CLASSES.items(),
                                                     key=lambda kv: kv[1])},
        },
        "source_categories": data["categories"],
        "source_info": data.get("info"),
    }
    if args.embed_dropped:
        prov["dropped_annotation_records"] = dropped_ann
        prov["dropped_image_records"] = dropped_imgs
        prov["dropped_scene_annotation_records"] = [
            s for s in data.get("scene_annotations", []) if s["image_id"] not in live]
    write_json(prov_out, prov, force=True)

    print(f"wrote {args.out}  ({len(kept_imgs)} images, {len(kept_ann)} annotations, "
          f"{len(TARGET_CLASSES)} classes)")
    print(f"wrote {prov_out}  (how the merge was done / how to undo it"
          f"{', dropped records embedded' if args.embed_dropped else ''})")


# ----------------------------------------------------------------------------- undo

def undo(args: argparse.Namespace) -> None:
    prov = json.loads(args.undo.read_text())
    if "reverse_annotation_map" not in prov or "forward_map" not in prov:
        sys.exit(f"{args.undo} is not a taco_merge_classes provenance file")

    merged_path = Path(prov["output"]["path"])
    if not merged_path.exists():
        sys.exit(f"merged file recorded in the sidecar is missing: {merged_path}")
    got = sha256_of(merged_path)
    if got != prov["output"]["sha256"]:
        print(f"warning: {merged_path} has changed since the merge\n"
              f"  recorded sha256 {prov['output']['sha256']}\n"
              f"  actual   sha256 {got}\n"
              f"  annotation ids not in the sidecar will be left untouched.",
              file=sys.stderr)

    src_path = Path(prov["input"]["path"])
    if src_path.exists() and sha256_of(src_path) == prov["input"]["sha256"]:
        print(f"note: the original file is still intact at\n  {src_path}\n"
              f"  (sha256 matches the sidecar) - that file is the complete undo and "
              f"nothing needs rebuilding. Continuing anyway.")

    data = json.loads(merged_path.read_text())
    reverse = prov["reverse_annotation_map"]
    fwd = prov["forward_map"]

    restored, unmapped = [], []
    for a in data["annotations"]:
        key = ann_key(a)
        if key not in reverse and str(a["id"]) in reverse:
            key = str(a["id"])          # sidecar written before the key included image_id
        if key not in reverse:
            unmapped.append(a["id"])
            restored.append(dict(a))
            continue
        new = dict(a)
        new["category_id"] = reverse[key]
        restored.append(new)

    embedded = prov.get("dropped_records_embedded") and "dropped_annotation_records" in prov
    images = list(data["images"])
    scenes = list(data.get("scene_annotations", []))
    if embedded:
        restored += [dict(a) for a in prov["dropped_annotation_records"]]
        images += [dict(im) for im in prov["dropped_image_records"]]
        scenes += [dict(s) for s in prov.get("dropped_scene_annotation_records", [])]
        restored.sort(key=lambda a: (a["image_id"], a["id"]))
        images.sort(key=lambda im: im["id"])
        scenes.sort(key=lambda s: s["image_id"])

    out = {
        "info": {
            **(prov.get("source_info") or {}),
            "description": f"{(prov.get('source_info') or {}).get('description', 'TACO')} "
                           f"- restored from {merged_path.name} via {args.undo.name}",
            "restored_by": f"taco_merge_classes.py v{SCRIPT_VERSION} --undo",
            "restored_at": datetime.now(timezone.utc).isoformat(),
            "complete": bool(embedded),
        },
        "images": images,
        "annotations": restored,
        "scene_annotations": scenes,
        "licenses": data.get("licenses", []),
        "categories": prov.get("source_categories") or
                      [{"supercategory": e.get("supercategory") or e["name"],
                        "id": int(cid), "name": e["name"]}
                       for cid, e in sorted(fwd.items(), key=lambda kv: int(kv[0]))],
        "scene_categories": data.get("scene_categories", []),
    }
    write_json(args.out, out, args.force)

    n_in = prov["counts"]["annotations"]["in"]
    print(f"wrote {args.out}  ({len(images)} images, {len(restored)} annotations, "
          f"{len(out['categories'])} categories)")
    print(f"  restored the original category of {len(restored) - len(unmapped)} annotations")
    if unmapped:
        print(f"  {len(unmapped)} annotations were not in the sidecar and kept their "
              f"merged class: {unmapped[:10]}{' ...' if len(unmapped) > 10 else ''}")
    if embedded:
        print(f"  dropped records were embedded: complete restore "
              f"({len(restored)}/{n_in} annotations, "
              f"{len(images)}/{prov['counts']['images']['in']} images)")
    else:
        print(f"  INCOMPLETE: {prov['counts']['annotations']['dropped']} discarded "
              f"annotations and {prov['counts']['images']['dropped']} dropped images "
              f"cannot be recreated (no geometry was kept).\n"
              f"  Their ids are listed in 'dropped_annotations' / 'dropped_images' of "
              f"{args.undo.name}.\n  For the full original use {src_path}")


def main() -> None:
    args = parse_args()
    undo(args) if args.undo else merge(args)


if __name__ == "__main__":
    main()
