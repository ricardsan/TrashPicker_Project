"""Box and mask mAP, computed by the same code Ultralytics uses.

The point of this file is comparability. Rather than reimplement average precision (or reach for
pycocotools, whose summary is laid out differently), it feeds plain numpy arrays into
ultralytics.utils.metrics. So "mAP50-95" here means exactly what it means in the YOLO runs, down
to the 101-point interpolation and the max-F1 operating point that precision and recall are read
off, and the eight curve PNGs are produced by the same plotting functions.

The flow for one validation pass:

    1. per image: match predictions to ground truth, once by box IoU and once by mask IoU
       -> a (n_pred, 10) boolean "is this prediction correct at IoU 0.50, 0.55, ... 0.95" matrix
    2. concatenate everything over the split
    3. ap_per_class(...) twice, prefix="Box" and prefix="Mask"
    4. fitness = mAP50-95(M) + mAP50-95(B), which is how YOLO picks best.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class

# The 10 IoU thresholds COCO averages over, and ultralytics' validator uses.
IOUV = np.linspace(0.5, 0.95, 10)

# What the validator uses when building the confusion matrix (args.conf is None -> 0.25).
CONF_MATRIX_CONF = 0.25
CONF_MATRIX_IOU = 0.45

# Masks are compared at imgsz/MASK_RATIO, not full resolution. That is what ultralytics'
# segment validator does -- it scores against the prototype-resolution masks -- and it also
# keeps the IoU matmul 16x cheaper, which matters when every image can carry max_det=300
# predicted masks.
MASK_RATIO = 4


def match_predictions(iou: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray) -> np.ndarray:
    """(n_pred, 10) bool: is prediction j correct at each IoU threshold?

    A numpy transcription of ultralytics' BaseValidator.match_predictions. Predictions must
    already be sorted by descending confidence. Matching is one-to-one: at each threshold the
    highest-IoU pairs win, and neither a ground truth nor a prediction is used twice.

    iou is (n_gt, n_pred).
    """
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    if iou.size == 0:
        return correct

    iou = iou * (gt_cls[:, None] == pred_cls[None, :])  # a wrong class can never be a match
    for i, threshold in enumerate(IOUV):
        matches = np.argwhere(iou >= threshold)
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            order = iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
            matches = matches[order]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # one gt per pred
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # one pred per gt
        correct[matches[:, 1], i] = True
    return correct


def to_mask_resolution(masks, size: int) -> np.ndarray:
    """Binary masks resized to size x size, as uint8 numpy.

    Bilinear then threshold at 0.5, because that is the operator ultralytics' segment validator
    uses on its ground-truth masks, and matching it is the point. It is also the more faithful
    choice on curved boundaries: for a radius-20 disc downscaled 4x, bilinear keeps 79 pixels
    against a true area of ~78, where plain strided subsampling keeps 69.

    Neither operator preserves structures thinner than the stride -- a 3-pixel bar at 4x is lost
    by bilinear and only accidentally sampled by striding -- so a very thin mask will score badly
    either way. That is a property of scoring masks at imgsz/4 at all, and it applies equally to
    the YOLO runs being compared against.

    Accepts a torch tensor (so predictions can be shrunk on the GPU before they are copied back)
    or a numpy array.
    """
    if not torch.is_tensor(masks):
        masks = torch.from_numpy(np.ascontiguousarray(masks))
    if masks.numel() == 0:
        return np.zeros((len(masks), size, size), dtype=np.uint8)
    resized = torch.nn.functional.interpolate(
        masks[:, None].float(), (size, size), mode="bilinear", align_corners=False
    )
    return (resized[:, 0] > 0.5).to(torch.uint8).cpu().numpy()


def mask_iou(gt_masks: np.ndarray, pred_masks: np.ndarray) -> np.ndarray:
    """(n_gt, n_pred) IoU between two sets of binary masks of the same H x W."""
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    a = torch.from_numpy(np.ascontiguousarray(gt_masks)).flatten(1).float()
    b = torch.from_numpy(np.ascontiguousarray(pred_masks)).flatten(1).float()
    intersection = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    return (intersection / union.clamp(min=1e-9)).numpy()


class Evaluator:
    """Accumulates one validation pass, then turns it into metrics and plots."""

    def __init__(self, names: dict[int, str], with_masks: bool):
        self.names = names
        self.with_masks = with_masks
        self.stats = {"conf": [], "pred_cls": [], "target_cls": [], "tp": [], "tp_m": []}
        # "detect" either way: the matrix is built from box IoU in both cases, and ultralytics
        # only branches on "classify"/"semantic" here, so "segment" would just be an unknown
        # string taking the same path.
        self.confusion = ConfusionMatrix(names=names, task="detect")

    # --- accumulate ---------------------------------------------------------
    def add(self, prediction: dict, target: dict) -> None:
        """One image. Both dicts hold numpy arrays.

        prediction: boxes (n, 4) xyxy, scores (n,), labels (n,) 0-based, masks (n, H, W) uint8
        target:     boxes (m, 4) xyxy, labels (m,) 0-based, masks (m, H, W) uint8

        Boxes are in image pixels. The two sets of masks only have to agree with *each other* on
        H and W -- the caller decides the resolution, and the engine shrinks both to
        imgsz // MASK_RATIO first (see to_mask_resolution).
        """
        gt_cls = target["labels"]
        self.stats["target_cls"].append(gt_cls)

        # Sorting by confidence is not required by ap_per_class (it sorts internally) nor by
        # match_predictions (which is purely IoU-greedy). It is done so that the rows of the
        # accumulated tp/conf arrays line up with each other in a readable order.
        order = np.argsort(-prediction["scores"])
        pred_cls = prediction["labels"][order]
        self.stats["conf"].append(prediction["scores"][order])
        self.stats["pred_cls"].append(pred_cls)

        boxes = prediction["boxes"][order]
        if len(gt_cls) == 0 or len(pred_cls) == 0:
            self.stats["tp"].append(np.zeros((len(pred_cls), len(IOUV)), dtype=bool))
            self.stats["tp_m"].append(np.zeros((len(pred_cls), len(IOUV)), dtype=bool))
        else:
            iou = box_iou(torch.from_numpy(target["boxes"]).float(),
                          torch.from_numpy(boxes).float()).numpy()
            self.stats["tp"].append(match_predictions(iou, pred_cls, gt_cls))
            if self.with_masks:
                iou_m = mask_iou(target["masks"], prediction["masks"][order])
                self.stats["tp_m"].append(match_predictions(iou_m, pred_cls, gt_cls))
            else:
                self.stats["tp_m"].append(np.zeros((len(pred_cls), len(IOUV)), dtype=bool))

        # the confusion matrix wants torch tensors and xyxy boxes
        self.confusion.process_batch(
            detections={"cls": torch.from_numpy(pred_cls),
                        "conf": torch.from_numpy(self.stats["conf"][-1]),
                        "bboxes": torch.from_numpy(boxes).float()},
            batch={"cls": torch.from_numpy(gt_cls),
                   "bboxes": torch.from_numpy(target["boxes"]).float()},
            conf=CONF_MATRIX_CONF, iou_thres=CONF_MATRIX_IOU,
        )

    # --- summarise ----------------------------------------------------------
    def results(self, save_dir: Path | None = None, plot: bool = False) -> dict[str, float]:
        """The eight metrics/* numbers, plus fitness. Writes the curve PNGs when plot=True."""
        cat = {k: (np.concatenate(v) if v else np.array([])) for k, v in self.stats.items()}
        empty = len(cat["conf"]) == 0 or len(cat["target_cls"]) == 0

        box = self._ap(cat, "tp", "Box", save_dir, plot and not empty, empty)
        if self.with_masks:
            mask = self._ap(cat, "tp_m", "Mask", save_dir, plot and not empty, empty)
        else:
            # A box-only model has no mask metrics. NaN, not 0: zero would read as "it tried and
            # scored nothing", which is a different and misleading claim.
            mask = dict.fromkeys(("precision", "recall", "mAP50", "mAP50-95"), float("nan"))

        out = {
            "metrics/precision(B)": box["precision"],
            "metrics/recall(B)": box["recall"],
            "metrics/mAP50(B)": box["mAP50"],
            "metrics/mAP50-95(B)": box["mAP50-95"],
            "metrics/precision(M)": mask["precision"],
            "metrics/recall(M)": mask["recall"],
            "metrics/mAP50(M)": mask["mAP50"],
            "metrics/mAP50-95(M)": mask["mAP50-95"],
        }
        out["fitness"] = self.fitness(out)
        return out

    def _ap(self, cat, tp_key, prefix, save_dir, plot, empty) -> dict[str, float]:
        if empty:
            return dict.fromkeys(("precision", "recall", "mAP50", "mAP50-95"), 0.0)
        p, r, _f1, ap = ap_per_class(
            cat[tp_key], cat["conf"], cat["pred_cls"], cat["target_cls"],
            plot=plot, save_dir=Path(save_dir or "."), names=self.names, prefix=prefix,
        )[2:6]
        return {
            "precision": float(p.mean()) if len(p) else 0.0,
            "recall": float(r.mean()) if len(r) else 0.0,
            "mAP50": float(ap[:, 0].mean()) if len(ap) else 0.0,
            "mAP50-95": float(ap.mean()) if len(ap) else 0.0,
        }

    def fitness(self, metrics: dict[str, float]) -> float:
        """What best.pt is chosen by, copied from ultralytics.

        SegmentMetrics.fitness is mAP50-95(M) + mAP50-95(B); DetMetrics.fitness is just the box
        term. (Checked against the YOLO run we compare with: its best.pt records fitness
        1.76641 = 0.89402 + 0.87239.)
        """
        box = metrics["metrics/mAP50-95(B)"]
        if not self.with_masks:
            return float(box)
        return float(metrics["metrics/mAP50-95(M)"] + box)

    def plot_confusion(self, save_dir: Path) -> None:
        """The two confusion matrix PNGs, normalized and not."""
        for normalize in (True, False):
            self.confusion.plot(normalize=normalize, save_dir=save_dir)
