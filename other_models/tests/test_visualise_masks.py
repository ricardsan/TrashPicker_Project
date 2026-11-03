"""The label visualiser.

Light coverage: the two --labels modes have different code paths through figure_for, and the
`coco` one is what you look at before committing to a long training run, so it should not be the
thing that breaks silently.
"""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from other_models.datasets import ROOT, get_source
from other_models.visualise_masks import coco_rings, figure_for, load_prepared, round_trip
from other_models.prepare_data import index_images, stem_key


def args(labels="source", max_side=400):
    return SimpleNamespace(labels=labels, max_side=max_side)


def one_real_image(source="dataset_seg", split="train"):
    if not get_source(source).ann_file(split).exists():
        pytest.skip(f"{source}/{split} not prepared")
    prepared, _ = load_prepared(source, split)
    on_disk = index_images(get_source(source).image_dir(split))
    key = next((k for k in sorted(prepared) if k in on_disk and prepared[k][1]), None)
    if key is None:
        pytest.skip("no usable image")
    return key, prepared[key], on_disk[key]


def test_coco_rings_reads_every_polygon():
    annotations = [
        {"category_id": 2, "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]},
        {"category_id": 0, "segmentation": [[1.0, 1.0, 5.0, 1.0, 5.0, 5.0],
                                            [6.0, 6.0, 9.0, 6.0, 9.0, 9.0]]},
    ]
    rings = coco_rings(annotations)
    assert [c for c, _ in rings] == [2, 0]
    assert [len(r) for _, r in rings] == [1, 2]       # second annotation has two rings
    assert rings[0][1][0].shape == (3, 2)             # flat list reshaped to (points, 2)


def test_round_trip_is_one_for_identical_geometry():
    ring = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
    assert round_trip([(1, ring)], [(1, [ring])], 64, 64) == pytest.approx([1.0])
    # a different class cannot match, however well the shapes line up
    assert round_trip([(0, ring)], [(1, [ring])], 64, 64) == pytest.approx([0.0])


@pytest.mark.parametrize("labels", ["source", "coco"])
def test_figure_for_builds_both_modes(labels):
    import matplotlib.pyplot as plt
    _, prepared, path = one_real_image()
    fig, worst, orphans = figure_for("dataset_seg", "train", path, prepared, args(labels))
    assert len(fig.axes) == 2
    assert orphans == 0
    plt.close(fig)


def test_coco_mode_needs_no_yolo_labels(monkeypatch, tmp_path):
    """The point of --labels coco: it reads only the COCO file.

    Pointed at a directory with no .txt in it, `source` mode finds nothing while `coco` mode is
    unaffected -- which is what makes it usable on a dataset shipped as COCO alone.
    """
    import matplotlib.pyplot as plt
    import other_models.visualise_masks as module
    monkeypatch.setitem(module.SOURCE_LABELS, "dataset_seg", ("yolo", tmp_path / "{split}"))
    _, prepared, path = one_real_image()

    fig, _, _ = figure_for("dataset_seg", "train", path, prepared, args("source"))
    assert "0 polygons" in fig.axes[0].get_title()
    plt.close(fig)

    fig, _, _ = figure_for("dataset_seg", "train", path, prepared, args("coco"))
    assert "0 polygons" not in fig.axes[0].get_title()
    plt.close(fig)


@pytest.mark.slow
def test_cli_writes_a_png(tmp_path):
    if not get_source("dataset_seg").ann_file("train").exists():
        pytest.skip("dataset_seg not prepared")
    result = subprocess.run(
        [sys.executable, str(ROOT / "other_models" / "visualise_masks.py"),
         "--source", "dataset_seg", "--labels", "coco", "--n", "1", "--seed", "0",
         "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert list(tmp_path.glob("*.png")), result.stdout
