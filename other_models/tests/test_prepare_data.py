"""The data preparation, checked against the real files in this repo.

These are the tests that would have caught the two bugs that actually happened: annotations
silently landing in the wrong coordinate frame, and the TACO build quietly using the wrong class
mapping. Each test skips itself if its input is missing, so the suite still runs on a clone
without the datasets.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from other_models.datasets import CLASS_NAMES, ROOT, get_source
from other_models.prepare_data import image_size, index_images, rings_to_annotation, stem_key

EXPORT = next(ROOT.glob("coco-project*/result.json"), None)
TACO_MAPPING = ROOT / "datasets" / "taco_class_mapping_taco_yolo_reduced.json"
TACO_ANNOTATIONS = ROOT / "datasets" / "TACO" / "data" / "annotations.json"
TACO_INDEX = ROOT / "datasets" / "taco_yolo" / "source_index.json"


def prepared(source: str, split: str):
    path = get_source(source).ann_file(split)
    if not path.exists():
        pytest.skip(f"{path} not built; run other_models/prepare_data.py --source {source}")
    return json.loads(path.read_text())


# =============================================================================
# image_size: the EXIF trap
# =============================================================================
@pytest.mark.parametrize("source,split", [("dataset_seg", "train"), ("dataset_v2", "train"),
                                          ("taco", "train"), ("taco", "val")])
def test_image_size_agrees_with_opencv(source, split):
    """image_size must report what cv2 returns, EXIF applied, not what PIL reports.

    OpenCV rotates on load. Reading the size from PIL instead puts every annotation in a
    transposed frame, which is exactly the bug this guards against.
    """
    directory = get_source(source).image_dir(split)
    if not directory.is_dir():
        pytest.skip(f"{directory} missing")
    files = sorted(directory.iterdir())[:8]
    for path in files:
        height, width = cv2.imread(str(path)).shape[:2]
        assert image_size(path) == (width, height), path


def test_prepared_sizes_match_the_images():
    """Every width/height we write must be the size the dataset loader will actually see."""
    coco = prepared("dataset_seg", "train")
    directory = get_source("dataset_seg").image_dir("train")
    for info in coco["images"][:8]:
        height, width = cv2.imread(str(directory / info["file_name"])).shape[:2]
        assert (info["width"], info["height"]) == (width, height), info["file_name"]


@pytest.mark.parametrize("source,split", [("dataset_seg", "train"), ("dataset_seg", "val"),
                                          ("dataset_v2", "train"), ("dataset_v2", "val"),
                                          ("taco", "train"), ("taco", "val"), ("taco", "test")])
def test_annotations_lie_inside_their_image(source, split):
    coco = prepared(source, split)
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    for ann in coco["annotations"]:
        width, height = sizes[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        assert 0 <= x and 0 <= y, ann
        assert x + w <= width and y + h <= height, ann
        for polygon in ann["segmentation"]:
            points = np.asarray(polygon).reshape(-1, 2)
            assert points[:, 0].max() <= width and points[:, 1].max() <= height, ann["id"]


@pytest.mark.parametrize("source,split", [("dataset_seg", "train"), ("dataset_v2", "train"),
                                          ("taco", "train")])
def test_category_ids_stay_zero_based(source, split):
    """COCO ids here are already 0..3, matching data_v2.yaml.

    ultralytics' convert_coco subtracts one from category ids; doing that to these files would
    silently relabel everything. Also catches a 'skip' class (id 4) leaking through from TACO.
    """
    coco = prepared(source, split)
    assert [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])] == CLASS_NAMES
    ids = {a["category_id"] for a in coco["annotations"]}
    assert ids <= set(range(len(CLASS_NAMES))), ids


# =============================================================================
# dataset_seg
# =============================================================================
def test_dataset_seg_counts():
    assert len(prepared("dataset_seg", "train")["images"]) == 611
    assert len(prepared("dataset_seg", "val")["images"]) == 152


def test_dataset_seg_matches_the_yolo_labels_it_came_from():
    """The polygons must round-trip against the .txt files, which is what YOLO trained on."""
    coco = prepared("dataset_seg", "train")
    images = get_source("dataset_seg").image_dir("train")
    labels = ROOT / "datasets" / "dataset_seg" / "labels" / "train"
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    checked = 0
    for info in coco["images"][:15]:
        label_file = labels / f"{stem_key(info['file_name'])}.txt"
        if not label_file.exists():
            continue
        width, height = info["width"], info["height"]
        lines = [line for line in label_file.read_text().splitlines() if line.strip()]
        annotations = by_image[info["id"]]
        assert len(lines) == len(annotations), info["file_name"]

        for line, ann in zip(lines, annotations):
            parts = line.split()
            assert int(parts[0]) == ann["category_id"]
            expected = np.asarray(parts[1:], float).reshape(-1, 2) * [width, height]
            actual = np.asarray(ann["segmentation"][0]).reshape(-1, 2)
            assert np.allclose(expected, actual, atol=0.01), info["file_name"]
            checked += 1
    assert checked > 10, "expected to have compared some polygons"


# =============================================================================
# dataset_v2 / the Label Studio export
# =============================================================================
@pytest.mark.skipif(EXPORT is None, reason="no coco-project*/result.json")
def test_dataset_v2_resolves_672_of_763_images():
    export = json.loads(EXPORT.read_text())
    train, val = prepared("dataset_v2", "train"), prepared("dataset_v2", "val")
    assert len(train["images"]) == 537
    assert len(val["images"]) == 135
    assert len(train["images"]) + len(val["images"]) == 672
    assert len(export["images"]) == 763


@pytest.mark.skipif(EXPORT is None, reason="no coco-project*/result.json")
def test_dataset_v2_unresolved_images_are_listed_not_dropped():
    sidecar = get_source("dataset_v2").ann_file("train").parent / "provenance.json"
    if not sidecar.exists():
        pytest.skip("provenance.json not built")
    unresolved = json.loads(sidecar.read_text())["unresolved_images"]
    assert len(unresolved) == 91
    assert all("IMG_17" in name for name in unresolved), unresolved[:3]


@pytest.mark.skipif(EXPORT is None, reason="no coco-project*/result.json")
def test_dataset_v2_keeps_every_annotation_of_a_resolved_image():
    """1512 annotations, matching the count data_v2_all.yaml records for these 672 images.

    This is the upright-frame check with teeth: read in the raw frame instead and 20 of these
    collapse to nothing, because their coordinates run past the stored height.
    """
    total = sum(len(prepared("dataset_v2", s)["annotations"]) for s in ("train", "val"))
    assert total == 1512

    counts = {name: 0 for name in CLASS_NAMES}
    for split in ("train", "val"):
        for ann in prepared("dataset_v2", split)["annotations"]:
            counts[CLASS_NAMES[ann["category_id"]]] += 1
    assert counts == {"glass": 343, "metal": 341, "paper": 415, "plastic": 413}


@pytest.mark.skipif(EXPORT is None, reason="no coco-project*/result.json")
def test_the_current_export_has_no_polygons():
    """Documents why Mask metrics are NaN for dataset_v2 today.

    When a newer export with real polygons is dropped in, this test is expected to fail -- that
    is the signal that the mask path is now live, and the assertion can simply be inverted.
    """
    export = json.loads(EXPORT.read_text())
    with_polygons = [a for a in export["annotations"] if a.get("segmentation")]
    assert len(with_polygons) == 0, (
        f"{len(with_polygons)} annotations now have polygons -- the export has been updated, so "
        f"flip this assertion and re-run prepare_data.py"
    )


# =============================================================================
# TACO, through the reduced mapping
# =============================================================================
@pytest.mark.skipif(not TACO_MAPPING.exists(), reason="reduced mapping missing")
def test_reduced_mapping_keeps_35_of_60_categories():
    mapping = json.loads(TACO_MAPPING.read_text())["mapping"]
    kept = [v for v in mapping.values() if not v["drop"]]
    assert len(mapping) == 60
    assert len(kept) == 35
    assert all(v["target_id"] in range(4) for v in kept)


@pytest.mark.skipif(not TACO_MAPPING.exists(), reason="reduced mapping missing")
def test_reduced_mapping_totals_field_is_stale_and_must_not_be_used():
    """Guards a trap rather than our code.

    The mapping file's annotation_totals_per_target was copied from the full mapping and says
    skip: 1231, where the reduced mapping really drops 2799. prepare_data.py recomputes from
    "mapping"; if a future edit reaches for this field instead, the numbers silently shift.
    """
    data = json.loads(TACO_MAPPING.read_text())
    recomputed = sum(v["n_annotations"] for v in data["mapping"].values() if v["drop"])
    assert recomputed == 2799
    assert data["annotation_totals_per_target"]["skip"] == 1231 != recomputed


def test_taco_counts_and_splits():
    counts = {s: prepared("taco", s) for s in ("train", "val", "test")}
    assert [len(counts[s]["images"]) for s in ("train", "val", "test")] == [706, 208, 100]
    assert sum(len(c["images"]) for c in counts.values()) == 1014
    assert sum(len(c["annotations"]) for c in counts.values()) == 1985

    per_class = {name: 0 for name in CLASS_NAMES}
    for coco in counts.values():
        for ann in coco["annotations"]:
            per_class[CLASS_NAMES[ann["category_id"]]] += 1
    assert per_class == {"glass": 116, "metal": 372, "paper": 497, "plastic": 1000}


@pytest.mark.skipif(not TACO_ANNOTATIONS.exists(), reason="TACO annotations missing")
def test_taco_keeps_exactly_what_the_reduced_mapping_says_per_image():
    """Replay the mapping over the source and compare image by image.

    Stronger than a total: a dropped category leaking in, or a kept one going missing, shows up
    on the one image it affects even when the overall counts happen to balance out.
    """
    mapping = json.loads(TACO_MAPPING.read_text())["mapping"]
    source = json.loads(TACO_ANNOTATIONS.read_text())
    names = {i["id"]: stem_key(i["file_name"].replace("/", "_")) for i in source["images"]}

    expected: dict[str, list[int]] = {}
    for ann in source["annotations"]:
        entry = mapping[str(ann["category_id"])]
        if entry["drop"]:
            continue
        expected.setdefault(names[ann["image_id"]], []).append(entry["target_id"])

    actual: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        coco = prepared("taco", split)
        stems = {i["id"]: stem_key(i["file_name"]) for i in coco["images"]}
        for ann in coco["annotations"]:
            actual.setdefault(stems[ann["image_id"]], []).append(ann["category_id"])

    assert set(actual) <= set(expected), set(actual) - set(expected)
    for stem, classes in actual.items():
        assert sorted(classes) == sorted(expected[stem]), stem


@pytest.mark.skipif(not TACO_INDEX.exists(), reason="source_index.json missing")
def test_taco_build_reproduces_source_index():
    """taco_merge_classes.py produced the same reduced COCO independently, so compare.

    Our build goes annotations.json + reduced mapping -> COCO; source_index.json was written by
    a different script from the same inputs. Agreeing on both counts and the per-class histogram
    means neither is drifting.
    """
    index = json.loads(TACO_INDEX.read_text())
    assert len(index["images"]) == 1014
    assert len(index["annotations"]) == 1985

    ours = {name: 0 for name in CLASS_NAMES}
    for split in ("train", "val", "test"):
        for ann in prepared("taco", split)["annotations"]:
            ours[CLASS_NAMES[ann["category_id"]]] += 1
    theirs = {name: 0 for name in CLASS_NAMES}
    for ann in index["annotations"]:
        theirs[CLASS_NAMES[ann["category_id"]]] += 1
    assert ours == theirs

    # and the same images, by flattened name
    ours_names = {stem_key(i["file_name"]) for split in ("train", "val", "test")
                  for i in prepared("taco", split)["images"]}
    theirs_names = {stem_key(i["file_name"].replace("/", "_")) for i in index["images"]}
    assert ours_names == theirs_names


def test_taco_splits_match_the_folders_on_disk():
    """The split boundary must stay where taco_to_yolo.py put it."""
    for split in ("train", "val", "test"):
        on_disk = index_images(get_source("taco").image_dir(split))
        if not on_disk:
            pytest.skip(f"taco {split} images missing")
        for info in prepared("taco", split)["images"]:
            assert stem_key(info["file_name"]) in on_disk, info["file_name"]


# =============================================================================
# rings_to_annotation
# =============================================================================
def test_rings_to_annotation_builds_a_tight_box():
    ring = np.array([[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]])
    ann = rings_to_annotation(0, 0, 2, [ring], 100, 100)
    assert ann["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert ann["area"] == pytest.approx(1200.0)
    assert ann["category_id"] == 2
    assert len(ann["segmentation"]) == 1


def test_rings_to_annotation_rejects_what_cannot_be_trained_on():
    outside = np.array([[200.0, 200.0], [260.0, 200.0], [260.0, 240.0], [200.0, 240.0]])
    assert rings_to_annotation(0, 0, 0, [outside], 100, 100) is None   # clips to nothing
    assert rings_to_annotation(0, 0, 0, [], 100, 100) is None          # no rings
    sliver = np.array([[10.0, 10.0], [10.4, 10.0], [10.4, 10.4]])
    assert rings_to_annotation(0, 0, 0, [sliver], 100, 100) is None    # sub-pixel
