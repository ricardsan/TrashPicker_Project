"""The run directory: results.csv, the plots, the checkpoints, args.yaml.

The goal is a folder that looks like an Ultralytics run, because the figures are meant to go
next to the YOLO ones in a paper. Compare against:

    runs/segment/runs/segment/original-yolo26-seg/original-finetune-nms_v/

which contains exactly these 24 files, all reproduced here:

    args.yaml  results.csv  results.png  labels.jpg
    BoxP_curve.png  BoxR_curve.png  BoxF1_curve.png  BoxPR_curve.png
    MaskP_curve.png MaskR_curve.png MaskF1_curve.png MaskPR_curve.png
    confusion_matrix.png  confusion_matrix_normalized.png
    train_batch0.jpg train_batch1.jpg train_batch2.jpg  (+ the three final batches)
    val_batch{0,1,2}_labels.jpg  val_batch{0,1,2}_pred.jpg
    weights/best.pt  weights/last.pt

results.csv is the load-bearing part. ultralytics' plot_results does not hardcode a layout: it
splits the columns into those whose name contains "loss" and those containing "metric", halves
each, and lays them out as (2, n/2). Ten loss columns and eight metric columns therefore give
the same 2x9 grid as the YOLO segment run, panel for panel. The metrics/* names are also the
ones collect_results.py reads.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results

# Mask R-CNN's five losses, in the slots YOLO's five occupy (box, seg, cls, l1, sem).
# This is the default; a ModelSpec can carry its own (see RFDETR_LOSS_COLUMNS), because the
# five names are a property of the architecture's loss, not of the CSV.
LOSS_COLUMNS = {
    "loss_box_reg": "box_loss",
    "loss_mask": "mask_loss",
    "loss_classifier": "cls_loss",
    "loss_rpn_box_reg": "rpn_box_loss",
    "loss_objectness": "rpn_obj_loss",
}

# RF-DETR's five, in the same slots. Lives here rather than in rfdetr_adapter.py so that
# models.py can name it without importing rfdetr, which costs ~10 s.
# `aux_loss` is RF-DETR's auxiliary per-decoder-layer and encoder losses summed: they are a
# large share of the gradient, and without a column of their own the engine's `if k in
# totals` filter would drop them from results.csv while still backpropagating them. See
# rfdetr_adapter.aggregate_losses.
RFDETR_LOSS_COLUMNS = {
    "loss_bbox": "box_loss",
    "loss_mask": "mask_loss",
    "loss_ce": "cls_loss",
    "loss_giou": "giou_loss",
    "loss_aux": "aux_loss",
}

METRIC_COLUMNS = [
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
]


def csv_columns(loss_columns: dict | None = None) -> list[str]:
    """The 23 results.csv column names for a model with these five losses.

    Always 2 + 5 + 8 + 5 + 3 = 23, in this order, whatever the losses are called -- which is
    what keeps every run in this package diffable against the YOLO one column for column.
    """
    names = (loss_columns or LOSS_COLUMNS).values()
    return (
        ["epoch", "time"]
        + [f"train/{c}" for c in names]
        + METRIC_COLUMNS
        + [f"val/{c}" for c in names]
        + ["lr/pg0", "lr/pg1", "lr/pg2"]
    )


CSV_COLUMNS = csv_columns(LOSS_COLUMNS)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """plot_images and plot_labels both want centre-based xywh."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


class RunDirectory:
    """Owns one run's folder and everything written into it."""

    def __init__(self, hyp, names: dict[int, str], extra_args: dict | None = None,
                 loss_columns: dict | None = None):
        self.dir = Path(increment_path(Path(hyp.project) / hyp.name, exist_ok=hyp.exist_ok))
        self.weights = self.dir / "weights"
        self.weights.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "results.csv"
        self.hyp = hyp
        self.names = names
        # Which five losses this model has. The engine passes the spec's; the default keeps
        # every existing caller and test working unchanged.
        self.loss_columns = loss_columns or LOSS_COLUMNS
        self.csv_columns = csv_columns(self.loss_columns)
        self.start = time.time()
        self.best_fitness: float | None = None
        self._write_args(extra_args or {})

    # --- args.yaml ----------------------------------------------------------
    def _write_args(self, extra: dict) -> None:
        args = {**asdict(self.hyp), **extra, "save_dir": str(self.dir)}
        (self.dir / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))

    # --- results.csv --------------------------------------------------------
    def log_epoch(self, epoch: int, train_losses: dict, metrics: dict,
                  val_losses: dict, lrs: list[float]) -> None:
        """Append one row, in the same `%.6g` style ultralytics' save_metrics uses."""
        row = [epoch + 1, time.time() - self.start]
        row += [train_losses.get(k, float("nan")) for k in self.loss_columns]
        row += [metrics.get(k, float("nan")) for k in METRIC_COLUMNS]
        row += [val_losses.get(k, float("nan")) for k in self.loss_columns]
        if len(lrs) > 3:
            raise ValueError(f"results.csv has 3 lr columns, got {len(lrs)} param groups")
        row += list(lrs) + [float("nan")] * (3 - len(lrs))

        header = "" if self.csv.exists() else ",".join(self.csv_columns) + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(header + ",".join(f"{v:.6g}" for v in row) + "\n")

    # --- checkpoints --------------------------------------------------------
    def save_checkpoint(self, model, epoch: int, fitness: float,
                        metrics: dict, train_args: dict) -> bool:
        """Write last.pt, and best.pt when fitness improves. Returns True if it was the best.

        The optimiser state is deliberately not stored. It would triple the file (Adam keeps two
        moments per parameter: 180 MB of ResNet-50 weights becomes 550 MB) and is only useful for
        resuming, which this trainer does not do. Ultralytics ends up in the same place -- its
        final_eval strips the optimiser out of best.pt and last.pt.
        """
        improved = self.best_fitness is None or fitness > self.best_fitness
        if improved:
            self.best_fitness = fitness

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "train_args": train_args,
            "train_metrics": {**metrics, "fitness": fitness},
            "train_results": self._results_dict(),
            "date": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(ckpt, self.weights / "last.pt")
        if improved:
            torch.save(ckpt, self.weights / "best.pt")
        if self.hyp.save_period > 0 and (epoch + 1) % self.hyp.save_period == 0:
            torch.save(ckpt, self.weights / f"epoch{epoch + 1}.pt")
        return improved

    def _results_dict(self) -> dict[str, list]:
        """results.csv as a dict of lists, the way ultralytics embeds it in a checkpoint."""
        if not self.csv.exists():
            return {}
        lines = self.csv.read_text().strip().splitlines()
        keys = lines[0].split(",")
        rows = [[float(v) for v in line.split(",")] for line in lines[1:]]
        return {k: [r[i] for r in rows] for i, k in enumerate(keys)}

    # --- plots --------------------------------------------------------------
    def plot_dataset_labels(self, dataset) -> None:
        """labels.jpg: class histogram, box shapes, box centres, box width/height.

        Straight off the decoded COCO records, so no image is read and nothing is augmented --
        the same thing ultralytics does with its cached label list.
        """
        boxes, classes = [], []
        for record in dataset.records:
            if len(record.cls) == 0:
                continue
            scale = np.array([record.width, record.height] * 2, dtype=np.float32)
            boxes.append(xyxy_to_xywh(record.boxes) / scale)  # plot_labels wants normalised xywh
            classes.append(record.cls)
        if not boxes:
            return
        plot_labels(np.concatenate(boxes), np.concatenate(classes),
                    names=dataset.names, save_dir=self.dir)

    def plot_batch(self, images, targets, index: int, prefix: str = "train") -> None:
        """train_batch<N>.jpg / val_batch<N>_labels.jpg -- ground truth on the real input."""
        plot_images(
            self._as_plot_dict(images, targets),
            fname=self.dir / f"{prefix}_batch{index}{'_labels' if prefix == 'val' else ''}.jpg",
            names=self.names,
        )

    def plot_predictions(self, images, predictions, index: int,
                         with_masks: bool = True) -> None:
        """val_batch<N>_pred.jpg -- the same images with what the model predicted."""
        cls, boxes, batch_idx, masks, conf = [], [], [], [], []
        for i, pred in enumerate(predictions):
            # already filtered by hyp.conf upstream; plot_images applies its own 0.25 for drawing
            n = len(pred["scores"])
            if n == 0:
                continue
            cls.append(pred["labels"])
            boxes.append(pred["boxes"])
            conf.append(pred["scores"])
            masks.append(pred["masks"])
            batch_idx.append(np.full(n, i))

        labels = {
            "img": torch.stack([im.cpu() for im in images]),
            "cls": np.concatenate(cls) if cls else np.zeros(0, dtype=np.int64),
            "bboxes": xyxy_to_xywh(np.concatenate(boxes)) if boxes else np.zeros((0, 4), np.float32),
            "conf": np.concatenate(conf) if conf else np.zeros(0, np.float32),
            "batch_idx": np.concatenate(batch_idx) if batch_idx else np.zeros(0, dtype=np.int64),
        }
        if with_masks and masks:
            labels["masks"] = np.concatenate(masks)
        plot_images(labels, fname=self.dir / f"val_batch{index}_pred.jpg", names=self.names)

    def _as_plot_dict(self, images, targets) -> dict:
        cls, boxes, batch_idx, masks = [], [], [], []
        for i, t in enumerate(targets):
            n = len(t["labels"])
            if n == 0:
                continue
            cls.append(t["labels"].cpu().numpy() - 1)        # back to 0-based for the names dict
            boxes.append(t["boxes"].cpu().numpy())
            masks.append(t["masks"].cpu().numpy())
            batch_idx.append(np.full(n, i))
        out = {
            "img": torch.stack([im.cpu() for im in images]),
            "cls": np.concatenate(cls) if cls else np.zeros(0, dtype=np.int64),
            "bboxes": xyxy_to_xywh(np.concatenate(boxes)) if boxes else np.zeros((0, 4), np.float32),
            "batch_idx": np.concatenate(batch_idx) if batch_idx else np.zeros(0, dtype=np.int64),
        }
        if masks:
            out["masks"] = np.concatenate(masks)
        return out

    def plot_metrics(self) -> None:
        """results.png, from results.csv."""
        if self.csv.exists():
            plot_results(file=self.csv)
