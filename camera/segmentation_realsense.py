"""RealSense segmentation with any of the project's trained models, logging its own speed.

Frames come from the camera, or from a recorded rosbag with --rosbag.

What this adds over camera/yolo_realsense.py (which is left alone):

  * every trained segmentation checkpoint is reachable, not just the two "original" fine-tunes --
    including the TACO pretrain/fine-tune runs and all three Mask R-CNN and three RF-DETR runs,
    which are plain state dicts that ultralytics' YOLO() cannot load at all;
  * an arbitrary checkpoint can be pointed at with --weights;
  * every frame's detection count and timing go to a CSV, and --plot turns that into the
    report figure. --plot-from rebuilds the figure from an old CSV with no camera attached;
  * --rosbag replays one of the clips in rosbags/ through the identical loop, so the models can
    be compared on byte-identical input instead of on whatever was in front of the camera.
    camera/rosbag_frames.py does the unpacking and serves the frames; everything downstream of
    source.grab() is the same code either way;
  * --save-video writes the annotated frames to an H.264 MP4, for a presentation;
  * --objects follows each object across frames and writes a second CSV describing them one by
    one -- id, class, where it was, how far away, and which way it was lying: the orientation of
    the first principal component of its mask, in degrees from the bottom edge of the frame.
    --show-axis draws that axis on the frame, and so into the video.

Fast by default
---------------
--objects is the only thing here that adds work to the measured loop, and it is off unless asked
for. Without it the loop does exactly what it has always done, so `loop_ms` stays comparable with
every CSV already in runs/camera/. With it, each detection's distance is the median of the valid
depths inside its mask rather than one pixel at the box centre: measured on
rosbags/20261016_233733_stacked at 640x360 with 5.3 detections a frame, that is 3.4 ms a frame
against 0.01 ms, or about 5% of a 15 Hz frame period -- enough to be worth a flag, not enough to
drop a frame on these clips. It scales with the detection count and the frame size, not with the
architecture, so a --objects run is still internally comparable across the three families.
`inference_ms` is untouched either way, and the CSV's `extras` column records which mode a run was
in so the question is answerable from the file.

--save-video deliberately is *not* in that trade: the encode runs on a worker thread feeding a
separate ffmpeg process, so the loop pays a pointer handoff. See camera/recording.py.

Examples
--------
    python camera/segmentation_realsense.py --list-models
    python camera/segmentation_realsense.py --model yolo26 --variant taco-finetune-augment
    python camera/segmentation_realsense.py --model maskrcnn --variant taco-finetune-augment \
        --conf 0.4 --plot --out-dir runs/camera
    python camera/segmentation_realsense.py --model rfdetr --variant finetune-augment --plot
    python camera/segmentation_realsense.py --weights runs/segment/maskrcnn/finetune-augment_v
    python camera/segmentation_realsense.py --plot-from 2026-10-20-143512-yolo26-taco.csv

    # for the report and the slides: a video, and a table of the individual objects
    python camera/segmentation_realsense.py --model yolo26 \
        --variant taco-finetune-augment --objects --save-video --show-ids --show-axis --plot

    # a recorded clip instead of the camera: unpacked once, then replayed at its own 15 Hz
    python camera/segmentation_realsense.py --rosbag rosbags/20261016_233733_stacked.db3 \
        --model yolo26 --variant taco-finetune-augment --plot
    python camera/segmentation_realsense.py --rosbag rosbags/..._stacked.db3 \
        --export-only --no-clouds
    python camera/segmentation_realsense.py --from-export rosbags/export/20261016_233733_stacked \
        --model maskrcnn --variant taco-finetune-augment --plot

ESC closes the window and, with --plot, writes the figure. A run's artefacts go in their own folder
under --out-dir, named after the run; --flat puts them side by side instead.

One caveat on comparing inference_ms across architectures
---------------------------------------------------------
The three families are not fed the same number of pixels for the same frame, and none of them is
wrong -- each sees the geometry it trained on. For a live 640x480 frame:

    ultralytics   pads to the frame's own aspect ratio, 480x640          307k px
    Mask R-CNN    letterboxes to a square 640x640                        409k px  (1.33x)
    RF-DETR       letterboxes to a square 648x648, or 840x840 for the
                  TACO pretrain                                          420k px  (1.37x)
                                                                         706k px  (2.30x)

MaskRCNNDetector's square comes from other_models.models.build pinning min_size = max_size =
imgsz. RF-DETR's 648 and 840 are not a preference: its resolution must be divisible by
patch_size * num_windows = 24, and 640 and 832 are not, so the presets use the nearest valid
values (see other_models/config.py section 3). So none of these latencies is a like-for-like
comparison against a YOLO one at the same nominal imgsz. The figure does not correct for it; the
input shape is printed at startup so it is on the record.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from camera.benchmark import (CAMERA_FPS, DetectionLogger,  # noqa: E402
                              RunLogger, is_detection_log, plot_run, read_log,
                              run_log_name, summarise)
# Both are free of cv2, torch and ultralytics at import time -- recording.py only reaches for cv2
# inside its OpenCV sink -- so importing them here costs nothing a --list-models run would notice.
from camera.recording import BACKENDS, DEFAULT_CRF, VideoRecorder  # noqa: E402
from camera.tracking import Tracker  # noqa: E402

DEFAULT_MODEL = "yolo26"
DEFAULT_VARIANT = "original-finetune-augment"

# Where --rosbag unpacks to, and where a bag run's CSV and figure go. Bag runs get their own
# subdirectory per bag so the three clips in rosbags/ stay apart, while the run log keeps the
# four-field date-time-model-dataset stem that benchmark.labels_from falls back on.
BAG_EXPORT_DIR = "rosbags/export"
BAG_RUN_DIR = "runs/camera"

# =============================================================================
# 1. WHICH CHECKPOINTS ARE ON OFFER
# =============================================================================
# An explicit table rather than a folder name built out of the --variant pieces (what
# yolo_realsense.resolve_model_path does). The run folders are not consistent enough for a
# formula: the yolo26 TACO pretrain that matters is `taco-pretrain-nms-augment_v-11` out of
# twelve numbered siblings, the yolo8 and yolo26 fine-tunes differ by `_` vs `-` before
# `augment`, and the Mask R-CNN runs have no `nms` in their names at all. Spelling them out also
# means the v-2..v-12 scratch runs cannot be selected by accident.


@dataclass(frozen=True)
class RunEntry:
    run_dir: str          # relative to ROOT
    dataset: str          # the label that goes in the log filename


# "custom" = trained or fine-tuned on our dataset_seg annotations, "taco" = the TACO pretrain,
# "richard" = Richard's original run. dataset_label() derives the same thing for --weights runs.
MODEL_RUNS: dict[tuple[str, str], RunEntry] = {
    ("yolo8", "original-finetune"): RunEntry(
        "runs/segment/runs/segment/original-seg/original-finetune_v", "custom"),
    ("yolo8", "original-finetune-augment"): RunEntry(
        "runs/segment/runs/segment/original-seg/original-finetune-augment_v", "custom"),
    ("yolo8", "taco-pretrain-nms-augment"): RunEntry(
        "runs/segment/runs/segment/taco-seg/taco-pretrain-nms-augment_v", "taco"),
    ("yolo8", "taco-finetune-augment"): RunEntry(
        "runs/segment/runs/segment/taco-seg/taco-finetune-nms_augment_v", "custom"),
    ("yolo8", "richard-train-seg"): RunEntry(
        "runs/segment/train_seg", "richard"),

    ("yolo26", "original-finetune"): RunEntry(
        "runs/segment/runs/segment/original-yolo26-seg/original-finetune-nms_v", "custom"),
    ("yolo26", "original-finetune-augment"): RunEntry(
        "runs/segment/runs/segment/original-yolo26-seg/original-finetune-nms-augment_v", "custom"),
    ("yolo26", "taco-pretrain-nms-augment"): RunEntry(
        "runs/segment/runs/segment/taco-yolo26-seg/taco-pretrain-nms-augment_v-11", "taco"),
    ("yolo26", "taco-finetune-augment"): RunEntry(
        "runs/segment/runs/segment/taco-yolo26-seg/taco-finetune-nms_augment_v", "custom"),

    ("maskrcnn", "finetune-augment"): RunEntry(
        "runs/segment/maskrcnn/finetune-augment_v", "custom"),
    ("maskrcnn", "taco-pretrain-augment"): RunEntry(
        "runs/segment/maskrcnn/taco/taco-pretrain-augment", "taco"),
    ("maskrcnn", "taco-finetune-augment"): RunEntry(
        "runs/segment/maskrcnn/taco/taco-finetune-augment", "custom"),

    # The RF-DETR runs mirror the three Mask R-CNN ones. Their folders spell the variant with
    # underscores and carry a numbered suffix; the CLI spelling stays dashed and unnumbered,
    # as it already is for the yolo runs, so the `_v` / `_v-2` siblings and `smoke_v` are not
    # selectable.
    ("rfdetr", "finetune-augment"): RunEntry(
        "runs/segment/rfdetr/finetune_augment_v-3", "custom"),
    ("rfdetr", "taco-pretrain-augment"): RunEntry(
        "runs/segment/rfdetr/taco/taco_pretrain_augment_v-2", "taco"),
    ("rfdetr", "taco-finetune-augment"): RunEntry(
        "runs/segment/rfdetr/taco/taco_finetune_augment_v-2", "custom"),
}

ARCHITECTURES = ("yolo8", "yolo26", "maskrcnn", "rfdetr")

# The filename wants the name the report uses, which is not quite the CLI spelling. No label
# may contain a dash: benchmark.labels_from falls back to reading a log's stem as four
# dash-separated fields, and "rf-detr" would make that split land on the wrong two.
MODEL_LABELS = {"yolo8": "yolov8", "yolo26": "yolo26", "maskrcnn": "maskrcnn",
                "rfdetr": "rfdetr"}

DATASET_LABELS = ("custom", "taco", "richard")

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def weights_of(run_dir) -> Path:
    """The checkpoint inside a run directory. An absolute run_dir wins over ROOT, per pathlib."""
    return ROOT / run_dir / "weights" / "best.pt"


def list_models() -> str:
    """The registry as a table, with whether each checkpoint is usable right now."""
    width = max(len(m) + len(v) for m, v in MODEL_RUNS) + 2
    lines = [f"{'--model --variant'.ljust(width)}  {'dataset'.ljust(8)}  run"]
    for (model, variant), entry in MODEL_RUNS.items():
        weights = weights_of(entry.run_dir)
        if not weights.exists():
            note = "   [best.pt missing]"
        elif is_lfs_pointer(weights):
            note = "   [git LFS stub -- see --help]"
        else:
            note = ""
        lines.append(f"{(model + ' ' + variant).ljust(width)}  {entry.dataset.ljust(8)}  "
                     f"{entry.run_dir}{note}")
    lines += ["", "Any other checkpoint: --weights <run dir or best.pt>"]
    return "\n".join(lines)


# =============================================================================
# 2. RESOLVING A CHECKPOINT AND ITS LABELS
# =============================================================================
def is_lfs_pointer(path: Path) -> bool:
    """True for an unfetched git LFS stub: a ~130-byte text file where weights should be."""
    # Real checkpoints here are 23-180 MB, so the size test alone is decisive; reading the first
    # line as well means some other small file is never misreported as a stub.
    if path.stat().st_size >= 2048:
        return False
    with open(path, "rb") as f:
        return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def check_weights(path: Path) -> Path:
    """Fail early and legibly on a checkpoint that is not really there.

    .gitattributes tracks *.pt through git LFS, and a clone without `git lfs pull` leaves pointer
    stubs behind -- several of the runs in MODEL_RUNS are stubs in a typical working tree.
    torch.load on one of those raises an UnpicklingError several frames deep that says nothing
    about LFS. Nothing is fetched here; this only reports.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}\n--list-models shows which runs are available.")
    if is_lfs_pointer(path):
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = path
        raise RuntimeError(
            f"{path} is an unfetched git LFS pointer ({path.stat().st_size} bytes), not weights.\n"
            f"Fetch it with:\n    git lfs pull --include={relative}")
    return path


def resolve_registry(model: str, variant: str) -> tuple[Path, str, str]:
    """(weights path, architecture, dataset label) for a --model/--variant pair."""
    if model not in ARCHITECTURES:
        raise SystemExit(f"unknown --model {model!r}; choose one of {', '.join(ARCHITECTURES)}")
    if (model, variant) not in MODEL_RUNS:
        available = sorted(v for m, v in MODEL_RUNS if m == model)
        raise SystemExit(
            f"unknown --variant {variant!r} for --model {model}; available: "
            f"{', '.join(available)}\n"
            f"(--list-models prints the whole table; --weights takes any other checkpoint)")
    entry = MODEL_RUNS[(model, variant)]
    return weights_of(entry.run_dir), model, entry.dataset


def run_dir_of(weights) -> Path:
    """The run folder a best.pt belongs to, i.e. the one holding args.yaml."""
    weights = Path(weights)
    return weights.parent.parent if weights.parent.name == "weights" else weights.parent


def read_run_args(run_dir) -> dict:
    """A run's args.yaml, or {} when there is none."""
    path = Path(run_dir) / "args.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def architecture_from_name(name) -> str | None:
    """yolov8s-seg.pt -> yolo8, maskrcnn_v2 -> maskrcnn, rfdetr_seg_small -> rfdetr, else None."""
    stem = Path(str(name)).name.lower()
    if stem.startswith("yolo26"):
        return "yolo26"
    if stem.startswith(("yolov8", "yolo8")):
        return "yolo8"
    if stem.startswith(("maskrcnn", "mask_rcnn")):
        return "maskrcnn"
    if stem.startswith(("rfdetr", "rf_detr", "rf-detr")):
        return "rfdetr"
    return None


def infer_architecture(run_dir, _depth: int = 0) -> str:
    """Which architecture a run trained, read off its args.yaml `model:` field.

    The fine-tune runs record their *pretrain checkpoint path* there rather than an architecture
    name -- that is how the pretrain -> fine-tune chain documents itself -- so one hop along such
    a path is followed. Anything else raises rather than guesses: loading the wrong class of
    model would either blow up deep inside torch or silently produce nonsense.
    """
    args = read_run_args(run_dir)
    recorded = args.get("model")
    if recorded:
        architecture = architecture_from_name(recorded)
        if architecture:
            return architecture

        candidate = Path(str(recorded))
        if _depth == 0 and candidate.suffix == ".pt":
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            # The checkpoint itself has to be there. Following a merely plausible-looking path
            # would walk off to whatever directory it happens to resolve to and then complain
            # about *that* one, which is a confusing way to say "pass --model".
            if candidate.exists():
                return infer_architecture(run_dir_of(candidate), _depth + 1)

    raise SystemExit(
        f"cannot tell which architecture {run_dir} trained"
        + (f" (its args.yaml says model: {recorded!r})" if recorded
           else " -- no args.yaml with a `model:` field")
        + f"\nPass it explicitly: --model {'|'.join(ARCHITECTURES)}")


def dataset_label(run_dir) -> str:
    """Which dataset a run trained on, as the one-word label used in the log filename.

    YOLO runs record `data: <path to a dataset yaml>`, Mask R-CNN runs record
    `source: <a datasets.SOURCES key>`. Anything unrecognised falls back to "richard", which is
    where the pre-existing runs that carry no usable metadata belong.
    """
    args = read_run_args(run_dir)
    recorded = str(args.get("data") or args.get("source") or "").replace("\\", "/").lower()
    if not recorded:
        return "richard"
    if "taco" in recorded:
        return "taco"
    if "dataset_seg" in recorded or "dataset_v2" in recorded:
        return "custom"
    return "richard"


def resolve_custom_weights(given) -> Path:
    """The best.pt meant by a --weights argument: a checkpoint, a run dir, or a weights dir."""
    given = Path(given).expanduser()
    if not given.is_absolute():
        given = (Path.cwd() / given).resolve()
    if given.suffix == ".pt":
        return given
    for candidate in (given / "weights" / "best.pt", given / "best.pt"):
        if candidate.exists():
            return candidate
    # Nothing there; name the conventional location so check_weights' message is the useful one.
    return given / "weights" / "best.pt"


def resolve_weights(args) -> tuple[Path, str, str]:
    """(weights path, architecture, dataset label) for whatever the command line asked for."""
    if args.weights:
        weights = check_weights(resolve_custom_weights(args.weights))
        run_dir = run_dir_of(weights)
        return weights, args.model or infer_architecture(run_dir), dataset_label(run_dir)

    weights, architecture, dataset = resolve_registry(args.model or DEFAULT_MODEL,
                                                      args.variant or DEFAULT_VARIANT)
    return check_weights(weights), architecture, dataset


# =============================================================================
# 3. ONE INTERFACE OVER THREE KINDS OF MODEL
# =============================================================================
@dataclass
class Detection:
    """One predicted object, in the coordinates of the frame handed to the detector."""

    xyxy: tuple[int, int, int, int]
    cls: int
    conf: float
    mask: np.ndarray | None = None      # bool, frame-sized; None when the model gave none


PAD = 114                               # other_models.augment.PAD


def letterbox_frame(img: np.ndarray, imgsz: int):
    """Resize-and-pad to a square imgsz, returning the inverse transform as well.

    The arithmetic is deliberately identical to other_models.augment.letterbox, which is what
    produced the training inputs; the difference is only that this returns r/top/left so the
    predictions can be mapped back onto the frame. test_detectors.py pins the two together.
    """
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2

    out = np.full((imgsz, imgsz, 3), PAD, dtype=np.uint8)
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out, r, top, left


class YoloDetector:
    """ultralytics YOLO, v8 or 26, segmentation checkpoint."""

    def __init__(self, weights, conf: float):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.names = dict(self.model.names)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        # retina_masks routes ultralytics through ops.process_mask_native -> ops.scale_masks,
        # which crops the letterbox padding off before interpolating and hands back masks already
        # at the frame's own resolution. Doing it here by hand instead means reimplementing that
        # crop, and getting it wrong stretches every mask over the padding.
        result = self.model(frame, verbose=False, conf=self.conf, retina_masks=True)[0]
        return self._to_detections(result)

    @staticmethod
    def _to_detections(result) -> list[Detection]:
        """Pull a Results object apart. Separate from predict() so the tests can feed a fake."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks = None
        if getattr(result, "masks", None) is not None:
            # Already frame-sized, courtesy of retina_masks above.
            masks = result.masks.data.cpu().numpy()

        height, width = result.orig_shape
        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            # ultralytics' clip_boxes clamps to (0, w) and (0, h) *inclusive*, so a box touching
            # the right or bottom edge yields a centre one pixel outside the frame and
            # depth_frame.get_distance() raises on it, taking the whole run down. Clip the way
            # MaskRCNNDetector already does.
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))

            mask = masks[i] > 0.5 if masks is not None and i < len(masks) else None
            detections.append(
                Detection((x1, y1, x2, y2), int(box.cls[0]), float(box.conf[0]), mask))
        return detections


class OtherModelsDetector:
    """Anything other_models.models.build can build, rebuilt from its args.yaml and best.pt.

    Mask R-CNN and RF-DETR share this class because they share an interface, not because they
    are alike: other_models/rfdetr_adapter.py exists precisely so that RF-DETR answers
    torchvision's detection contract -- eval mode, `model([tensor])` -> one dict per image of
    boxes (xyxy pixels), scores, 1-based labels and masks. So everything from the letterbox to
    the Detection list is one code path, and the two subclasses below differ only in which
    `other_models.models.MODELS` family they expect.

    The checkpoints are plain state dicts, so the model has to be constructed exactly as
    training constructed it before the weights will fit. other_models.models.build does that
    already -- the 4-class head swap, the subtleties about why torchvision's COCO weights must
    still be passed, and RF-DETR's position-embedding interpolation -- so it is reused rather
    than reimplemented here.
    """

    # Which ModelSpec.family a run's `model:` field must name for this detector, and what to
    # build when it names something unrecognised.
    family = "torchvision"
    default_spec = "maskrcnn"
    build_note = ""                 # appended to the "building ..." line

    def __init__(self, weights, conf: float, device: str | None = None):
        import torch

        from other_models.config import Hyp
        from other_models.datasets import CLASS_NAMES
        from other_models.models import MODELS, build

        self.torch = torch
        self.conf = conf

        run_dir = run_dir_of(weights)
        args = read_run_args(run_dir)
        if not args:
            print(f"WARNING: no args.yaml in {run_dir}; falling back to Hyp() defaults, so "
                  f"imgsz/iou/max_det may not match how this model was trained.")

        known = {f.name for f in fields(Hyp)}
        # conf is overridden because build() feeds it to torchvision's box_score_thresh, and
        # every one of our runs trained with conf=0.001 -- right for scoring a PR curve, but it
        # would make the model paste up to max_det masks back to full resolution on every
        # single live frame. Thresholding inside roi_heads instead is both faster and what
        # --conf means. RF-DETR has no equivalent knob (its PostProcess takes a count, not a
        # threshold, and rfdetr_adapter.build ignores hyp.conf), so there the filtering happens
        # in _to_detections -- which is why that filter runs before the host copy.
        self.hyp = Hyp(**{**{k: v for k, v in args.items() if k in known}, "conf": conf})
        self.imgsz = self.hyp.imgsz
        self.names = dict(enumerate(args.get("classes") or CLASS_NAMES))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        spec_name = str(args.get("model") or self.default_spec)
        if spec_name not in MODELS:
            print(f"WARNING: args.yaml names model {spec_name!r}, which is not in "
                  f"other_models.models.MODELS; building {self.default_spec!r} instead.")
            spec_name = self.default_spec
        elif MODELS[spec_name].family != self.family:
            # A known model of the wrong family: --model and --weights disagree. Refuse,
            # rather than build either of them.
            #
            # Falling back to default_spec here would not even fail loudly -- load_checkpoint
            # prints `loaded 0/307 tensors` and carries on, so the run would serve random
            # weights, report max_det plausible-looking detections a frame, and log them under
            # the wrong model name. Honouring args.yaml instead would put the right
            # architecture behind the wrong label, which the CSV and the figure would then
            # repeat on every row. Neither is a thing to do quietly.
            raise SystemExit(
                f"{run_dir} trained {spec_name!r}, a {MODELS[spec_name].family!r} model, but "
                f"it was asked for as {self.family!r}.\n"
                f"Drop --model and let it be read from args.yaml, or pass the one that "
                f"matches: --model {architecture_from_name(spec_name) or '<architecture>'}")
        self.spec_name = spec_name

        print(f"building {spec_name} on {self.device}{self.build_note} ...")
        self.model, _ = build(spec_name, len(self.names), self.hyp, str(weights))
        self.model.to(self.device).eval()

    def mask_resolution(self) -> int | None:
        """The side of the mask planes predict() will get back, or None if it cannot be said.

        Only for the startup line. torchvision pastes its masks to the letterbox size; the
        subclass that does not override this answers for that case.
        """
        return self.imgsz

    def predict(self, frame: np.ndarray) -> list[Detection]:
        letterboxed, r, top, left = letterbox_frame(frame, self.imgsz)
        # BGR uint8 -> RGB float in 0..1. That is what torchvision's transform expects (it
        # applies the ImageNet normalisation itself) and also what RFDETRAdapter.forward
        # expects (it does the same normalisation explicitly).
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)

        with self.torch.inference_mode():
            output = self.model([tensor])[0]

        return self._to_detections(output, frame.shape[:2], r, top, left, self.conf,
                                   imgsz=self.imgsz)

    @staticmethod
    def _to_detections(output, frame_shape, r, top, left, conf, *,
                       imgsz: int | None = None) -> list[Detection]:
        """One image's torchvision-shaped output as frame-space Detections.

        `imgsz` is the letterbox side the model was fed, and is needed only to place masks
        that come back smaller than it -- see the scale below. Omitting it asserts the masks
        are already at letterbox resolution.
        """
        # Filter on the device, before anything is copied to the host. Mask R-CNN has already
        # dropped everything below --conf inside roi_heads so this is nearly a no-op for it,
        # but RF-DETR's PostProcess returns max_det (300) detections unconditionally, and
        # copying 300 masks across to discard most of them would put a ~30 MB transfer inside
        # run_frames' timed region on every frame.
        scores_t = output["scores"].detach()
        keep = scores_t >= conf
        if not bool(keep.any()):
            return []

        scores = scores_t[keep].cpu().numpy()
        boxes = output["boxes"].detach()[keep].cpu().numpy()
        # torchvision reserves label 0 for background, so our classes sit at 1..N -- the same
        # correction other_models/engine.py makes when it scores a validation batch, and the
        # one rfdetr_adapter._to_torchvision applies to make RF-DETR agree.
        labels = output["labels"].detach()[keep].cpu().numpy() - 1
        masks = output["masks"].detach()[keep].cpu().numpy() if "masks" in output else None

        height, width = frame_shape
        nh, nw = round(height * r), round(width * r)

        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = ((box[0] - left) / r, (box[1] - top) / r,
                              (box[2] - left) / r, (box[3] - top) / r)
            x1, x2 = int(round(min(max(x1, 0), width - 1))), int(round(min(max(x2, 0), width - 1)))
            y1, y2 = (int(round(min(max(y1, 0), height - 1))),
                      int(round(min(max(y2, 0), height - 1))))

            mask = None
            if masks is not None:
                mask = _mask_to_frame(masks[i][0], (height, width), (nh, nw), top, left, imgsz)

            detections.append(Detection((x1, y1, x2, y2), int(labels[i]), float(scores[i]), mask))
        return detections


def _mask_to_frame(plane: np.ndarray, frame_shape, scaled_shape, top: int, left: int,
                   imgsz: int | None) -> np.ndarray:
    """One mask plane, in letterbox space, placed back on the frame: crop the padding, resize.

    The plane is not always at letterbox resolution. torchvision's mask head pastes to it, but
    rfdetr_adapter.build asks PostProcess for masks at the mask head's own resolution
    (`upsample_masks_to_image_size=False`, imgsz // mask_downsample_ratio = imgsz // 4), which
    is deliberate: it is the resolution metrics.MASK_RATIO already scores mAP(M) at, and
    letting PostProcess interpolate all max_det planes up to imgsz^2 every frame would cost far
    more than the resize below. The price is up to `imgsz // plane height` pixels of slop at
    the crop boundary -- 4 px at the usual ratio, invisible under a 0.45-alpha overlay.

    The scale is read off the plane rather than hardcoded, so a variant with a different
    downsample ratio needs no change here.
    """
    height, width = frame_shape
    nh, nw = scaled_shape
    scale = plane.shape[0] / imgsz if imgsz else 1.0
    if scale != 1.0:
        top, left, nh, nw = (round(top * scale), round(left * scale),
                             round(nh * scale), round(nw * scale))
    cropped = plane[top:top + nh, left:left + nw]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5


class MaskRCNNDetector(OtherModelsDetector):
    """torchvision Mask R-CNN. See OtherModelsDetector."""

    family = "torchvision"
    default_spec = "maskrcnn"
    build_note = " (this loads the COCO backbone, which torchvision downloads on first use)"


class RFDETRDetector(OtherModelsDetector):
    """RF-DETR, through other_models/rfdetr_adapter.py. See OtherModelsDetector.

    Nothing is downloaded to build one: models.build passes the run's own best.pt as
    `weights_path`, which makes rfdetr_adapter.build skip load_pretrain_weights entirely, so
    the published COCO checkpoint is never fetched. The first import of rfdetr still costs
    ~10 s (it pulls in transformers), which is why the "building ..." line is printed first.
    """

    family = "rfdetr"
    default_spec = "rfdetr_seg_small"
    build_note = " (importing rfdetr pulls in transformers, which takes ~10 s)"

    def mask_resolution(self) -> int | None:
        """imgsz // mask_downsample_ratio -- see _mask_to_frame.

        The ratio is read off the variant's own config class rather than written down as 4,
        for the same reason _mask_to_frame reads its scale off the plane: this is the one
        number that would quietly become a lie for a variant with a different mask head.
        """
        from other_models.models import MODELS, rfdetr_config_class

        field = rfdetr_config_class(MODELS[self.spec_name]).model_fields.get(
            "mask_downsample_ratio")
        return self.imgsz // field.default if field else None


def build_detector(weights, architecture: str, conf: float):
    if architecture == "maskrcnn":
        return MaskRCNNDetector(weights, conf)
    if architecture == "rfdetr":
        return RFDETRDetector(weights, conf)
    return YoloDetector(weights, conf)


YOLO_STRIDE = 32                        # ultralytics pads each side up to a multiple of this
YOLO_IMGSZ = 640                        # and scales the long side to this first


def yolo_padded_shape(frame_shape, imgsz: int = YOLO_IMGSZ,
                      stride: int = YOLO_STRIDE) -> tuple[int, int]:
    """(height, width) ultralytics will actually feed the network for a frame of this shape.

    It scales the long side to imgsz, keeping the aspect ratio, then pads each side up to a
    multiple of the stride. Spelled out because the pixel count is what makes two inference_ms
    figures comparable or not: a live 640x480 frame becomes 640x480 (307k px) while a 640x360
    rosbag frame becomes 384x640 (246k px), so the same checkpoint is a little quicker on the bag
    for reasons that have nothing to do with the model.
    """
    height, width = frame_shape[:2]
    ratio = imgsz / max(height, width)
    scaled = [round(height * ratio), round(width * ratio)]
    return tuple(int(-(-side // stride) * stride) for side in scaled)


def describe_input(detector, frame_shape=(480, 640)) -> str:
    """What the detector will actually feed the network, printed so it is on the record.

    See the note in the module docstring: the two families do not see the same pixel count for
    the same frame, which is correct but matters when reading inference_ms across them.
    """
    if isinstance(detector, OtherModelsDetector):
        side = detector.imgsz
        masks = detector.mask_resolution()
        note = ""
        if masks is not None and masks != side:
            # Worth saying: it is why the masks drawn here are coarser than the boxes, and it
            # is the resolution the run's own mAP(M) was scored at. See _mask_to_frame.
            note = f"; masks come back at the mask head's own {masks}x{masks}"
        return (f"{side}x{side} letterboxed square ({side * side / 1e3:.0f}k px), "
                f"matching how this run was trained{note}")
    height, width = frame_shape[:2]
    padded_h, padded_w = yolo_padded_shape(frame_shape)
    # Both sides of the arrow are width x height. They used to disagree -- the left read w x h and
    # the right h x w -- which printed "640x480 -> 480x640" for a live frame and looked like a
    # transposition bug even though both numbers were right.
    return (f"the frame at its own aspect ratio, padded to a multiple of {YOLO_STRIDE} by "
            f"ultralytics ({width}x{height} -> {padded_w}x{padded_h}, "
            f"{padded_h * padded_w / 1e3:.0f}k px)")


# =============================================================================
# 4. THE LIVE LOOP
# =============================================================================
MASK_ALPHA = 0.45
# --show-axis's line. Cyan because green is already the box and red is the centroid dot, and the
# axis has to be told apart from both at a glance on a tinted mask.
AXIS_COLOUR = (255, 255, 0)
AXIS_THICKNESS = 2


def draw_masks(annotated: np.ndarray, detections) -> None:
    """Tint each detection's pixels with its class colour, in place.

    Composited through one scratch image and a single addWeighted rather than one blend per
    detection: blending is over the whole frame whatever it covers, so per-detection blending
    would cost the full frame N times over for the same picture.
    """
    from ultralytics.utils.plotting import colors

    masked = [d for d in detections if d.mask is not None]
    if not masked:
        return

    overlay = annotated.copy()
    for detection in masked:
        # The same palette as the training figures, so a class is the same colour here as in
        # the run's val_batch*_pred.jpg.
        overlay[detection.mask] = colors(detection.cls, bgr=True)
    cv2.addWeighted(overlay, MASK_ALPHA, annotated, 1 - MASK_ALPHA, 0, dst=annotated)


# =============================================================================
# 4a. MEASURING A DETECTION
# =============================================================================
# Measuring used to happen inside annotate(), which was fine while the only consumer was the label
# drawn on the frame. It is separate now because the per-object CSV wants the same numbers, and
# doing it twice would mean two depth lookups per detection inside the timed region.


@dataclass
class Observation:
    """One detection, measured: where it is on the frame and where it is in the camera's frame.

    `centroid` is the pixel the position was deprojected from -- the mask's centroid when the
    distance came from the mask, the box centre otherwise. `box_centre` is always the box centre,
    which is where the dot used to be drawn and what every earlier run measured at.

    `xyz` is None when no depth reading was available, which is a real and common outcome rather
    than an error: see measure().

    `orientation_deg` is None for the same kind of reason: there was no mask to take an axis over,
    which is not the same as an object lying horizontally. It is None when mask_px is 0, and also
    when it is 1 -- a single pixel has no direction.
    """

    object_id: int
    cls: int
    conf: float
    xyxy: tuple[int, int, int, int]
    box_centre: tuple[int, int]
    centroid: tuple[int, int]
    xyz: tuple[float, float, float] | None
    depth_source: str
    mask_px: int = 0
    depth_px: int = 0
    orientation_deg: float | None = None
    # Drawing only, for --show-axis, and deliberately not a CSV column: how long to make the line.
    # It is here rather than computed in annotate() because annotate() computes nothing -- every
    # number it draws comes from measure(), and the moments this falls out of are measure()'s.
    axis_half_px: float | None = None


# What produced an Observation's distance, and the only values the CSV's depth_source column ever
# holds. measure() asserts against this, so the column and this list cannot drift apart.
MASK_MEDIAN = "mask-median"     # the median of the valid depths inside the mask (--objects)
CENTRE_PIXEL = "centre-pixel"   # one depth pixel at `centroid`, the box centre in fast mode
NO_DEPTH = "none"               # nothing came back; xyz is None
DEPTH_SOURCES = (MASK_MEDIAN, CENTRE_PIXEL, NO_DEPTH)


def depth_map_metres(depth_frame) -> np.ndarray | None:
    """The whole depth map in metres, or None if this frame cannot produce one.

    Two shapes of depth frame reach here and neither is a numpy array:
    rosbag_frames.ExportedDepth, which offers metres() directly, and librealsense's own
    rs.depth_frame, which offers get_data() and get_units().

    **None is a supported answer, not a failure.** measure() falls back to the per-pixel
    get_distance path, which is the only thing a depth frame is otherwise required to provide --
    so a stand-in that implements nothing else keeps working, and the frame-source contract in
    run_frames' docstring does not grow a third method.
    """
    metres = getattr(depth_frame, "metres", None)
    if callable(metres):
        return metres()

    get_data = getattr(depth_frame, "get_data", None)
    get_units = getattr(depth_frame, "get_units", None)
    if callable(get_data) and callable(get_units):
        return np.asanyarray(get_data()).astype(np.float64) * float(get_units())
    return None


def mask_principal_axis(rows, columns) -> tuple[float, float] | None:
    r"""The mask's first principal component: (orientation in degrees, half-length in pixels).

    `rows` and `columns` are np.nonzero(mask), which measure() has already computed -- which is
    why this takes the pixel coordinates rather than the mask. The frame-sized boolean pass is the
    cost and it is already paid; everything here is a few passes over two 1-D arrays, so adding it
    to a --objects run is free in a way that adding it to a fast run would not be.

    **0 degrees is parallel to the bottom of the frame, and the angle grows anticlockwise as seen
    on screen, wrapped into [0, 180).** A principal axis is a line, not an arrow -- the sign of an
    eigenvector is arbitrary -- so 135 and -45 are one orientation and this reports 135. Reporting
    the full circle would be inventing a distinction the maths does not make.

        horizontal  --     0.0        diagonal  /     45.0
        vertical    |     90.0        diagonal  \    135.0

    Image rows grow *downward*, so the row deltas are negated up front and everything after that
    is in an ordinary y-up frame. Getting this one sign wrong mirrors every angle in the file --
    "/" reads as "\" -- while looking entirely plausible, which is why both diagonals are tested.

    The angle is the major axis by construction: theta maximises the variance of the projected
    pixels. camera/livegrasp/live_grasp.py's pca_from_mask has an extra swap to the axis of
    greatest pixel *extent*, which is a different quantity and not what the first principal
    component means; the formula below is otherwise the same as that one's.

    The half-length is 2*sqrt(lambda1), two standard deviations along the axis, and exists only so
    --show-axis has something to draw. It is not a CSV column. For a uniform bar of length L it
    works out to 0.58*L against a true half-length of 0.50*L, so the drawn line reaches just past
    the ends of the object, which is what makes it readable as *that object's* axis. It falls out
    of the same three moments as the angle; taking the true extent instead would mean projecting
    every mask pixel, a second pass per detection.

    Returns None when there is no axis to measure: an empty mask, or a single pixel, whose moments
    are all 0. atan2(0, 0) is 0, so without this guard one pixel would be logged as a confident
    "lying horizontally" -- the same class of untruth as a depth of 0.0 meaning "no reading".

    One honest limit: for an isotropic mask -- a perfect square or disc, where mu20 == mu02 and
    mu11 == 0 -- the principal axis is genuinely undefined, and this returns an arbitrary angle
    rather than None. Exactly isotropic gives 0.0, by the atan2 convention. *Near* isotropic is
    the case that turns up in real data, and there the answer is decided by whichever way mu11's
    last bits fell, so it can be anywhere in the range and will jump between frames. There is no
    threshold at which "round" could honestly become None, so the column documents how to spot
    one instead: a near-square box, or an angle that swings while the object sits still.
    """
    if rows.size == 0:
        return None

    dx = columns - columns.mean()
    # The whole of the y-down correction, done once here rather than by negating the result later.
    dy = rows.mean() - rows

    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())
    if mu20 == 0.0 and mu02 == 0.0 and mu11 == 0.0:
        return None

    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    spread = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2)
    lam1 = 0.5 * (mu20 + mu02) + 0.5 * spread       # the larger eigenvalue
    angle = float(np.degrees(theta)) % 180.0
    # Python's float modulo returns exactly 180.0 for any tiny negative input -- -1e-15 % 180.0
    # is 180.0, not 0.0 -- so the half-open range it looks like it guarantees is not guaranteed.
    # A mask a hair anticlockwise of horizontal gets there through mu11's rounding residue alone,
    # and a near-round object hits it often: 101 of 4000 random masks within 0.1 degree of
    # horizontal, measured. 180 and 0 are the same orientation, so fold it onto the end that is
    # inside the range.
    if angle >= 180.0:
        angle = 0.0
    return angle, float(2.0 * np.sqrt(lam1))


def measure(detections, depth_frame, intrinsics, rs, ids=None, *,
            mask_depth: bool = False) -> list[Observation]:
    """Each detection's position in the camera's frame. One depth lookup per detection.

    Two ways of asking how far away an object is:

    `mask_depth=False` (the default, and what every run before this did) takes a single depth
    pixel at the box centre. It is one array lookup, so it costs nothing -- and it is wrong
    whenever the box centre is not on the object. The box around a bottle lying diagonally is
    mostly floor, and its centre routinely lands on the floor rather than the bottle.

    `mask_depth=True` takes the median of every valid depth inside the mask and deprojects at the
    mask's centroid. That is the number the segmentation actually supports, and the median makes
    it robust to the speckle around a mask's edge. It costs one depth map per frame plus a
    boolean pass per detection -- 3.4 ms a frame against 0.01 ms, measured on the stacked bag at
    640x360 with 5.3 detections a frame -- which is why it is behind --objects rather than always
    on. The cost grows with the detection count and the frame size, never with the architecture.

    **A depth of 0 means "no reading", not "zero metres away".** Z16 uses 0 for a pixel nothing
    came back from, and that is the normal case on glass, clear plastic and anything specular --
    which is to say, on most of what this project sorts. Deprojecting it yields a confident
    (0, 0, 0), which is how `Z=0.00m` used to appear under a bottle the camera could not see.
    Here it gives `xyz = None` and a depth_source of "none", so the label reads `Z=?` and the CSV
    leaves the field blank. This applies in both modes: it is a bug fix, not part of the trade.

    `mask_depth=True` also yields each object's **orientation** -- the first principal component of
    its mask, see mask_principal_axis() -- because that falls out of the pixel coordinates this
    mode has already unpacked. It is therefore part of the same trade and not a separate cost:
    `mask_depth=False` leaves orientation_deg None, exactly as it leaves mask_px 0.
    """
    metres = depth_map_metres(depth_frame) if mask_depth else None

    observations = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.xyxy
        box_centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        cx, cy = box_centre
        mask_px = depth_px = 0
        orientation = axis_half = None
        distance = 0.0
        source = CENTRE_PIXEL

        mask = detection.mask if mask_depth else None
        if mask is not None:
            rows, columns = np.nonzero(mask)
            mask_px = int(rows.size)
            if mask_px:
                # The mask is frame-sized, so its centroid is always a pixel of the frame and
                # get_distance cannot be handed a coordinate it would raise on.
                cx = int(round(float(columns.mean())))
                cy = int(round(float(rows.mean())))
                # Rides along inside this block on purpose: it needs exactly the pixel
                # coordinates the lines above already paid for, so it adds no pass over the mask.
                axis = mask_principal_axis(rows, columns)
                if axis is not None:
                    orientation, axis_half = axis
                if metres is not None and metres.shape[:2] == mask.shape[:2]:
                    inside = metres[rows, columns]
                    inside = inside[inside > 0]
                    depth_px = int(inside.size)
                    if depth_px:
                        distance = float(np.median(inside))
                        source = MASK_MEDIAN

        if source != MASK_MEDIAN:
            distance = float(depth_frame.get_distance(cx, cy))

        if distance > 0:
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance)
            xyz = (float(X), float(Y), float(Z))
        else:
            xyz, source = None, NO_DEPTH
        assert source in DEPTH_SOURCES, source

        observations.append(Observation(
            object_id=int(ids[index]) if ids is not None else index,
            cls=int(detection.cls), conf=float(detection.conf), xyxy=(x1, y1, x2, y2),
            box_centre=box_centre, centroid=(cx, cy), xyz=xyz, depth_source=source,
            mask_px=mask_px, depth_px=depth_px,
            orientation_deg=orientation, axis_half_px=axis_half))
    return observations


def draw_axis(annotated: np.ndarray, observation) -> None:
    """Draw one object's principal axis through its centroid, in place.

    **The sign of dy is the point of this function.** mask_principal_axis() reports the angle in
    an ordinary y-up frame, and this canvas is y-down, so the row component has to be negated
    again on the way back out. Forget it and the picture is mirrored while the CSV stays right --
    which is precisely the mistake the overlay exists to reveal, so it is the worst place in the
    file to make it.

    cv2.line clips its own endpoints, so a line running off the frame needs no arithmetic here.
    """
    radians = np.radians(observation.orientation_deg)
    half = observation.axis_half_px
    dx = half * np.cos(radians)
    dy = -half * np.sin(radians)        # back to y-down, for the canvas
    cx, cy = observation.centroid
    cv2.line(annotated, (int(round(cx - dx)), int(round(cy - dy))),
             (int(round(cx + dx)), int(round(cy + dy))), AXIS_COLOUR, AXIS_THICKNESS)


def annotate(frame: np.ndarray, detections, observations, names,
             verbose: bool = True, masks: bool = True, show_ids: bool = False,
             show_axis: bool = False) -> np.ndarray:
    """Mask tint, then box, centre dot and the measured distance on top.

    The box/dot/label part is what camera/yolo_realsense.py draws, and it is identical for all
    three architectures on purpose -- the log compares their inference cost, which only means
    something if the per-frame drawing cost is the same. `show_ids` and `show_axis` add to it, but
    both default off and both add the same work for every architecture, so they cost the
    comparison nothing.

    Drawing only: every number here was computed by measure(). The dot sits on the pixel the
    position was deprojected from, so in fast mode it is the box centre exactly as before, and
    under --objects it marks the mask centroid the distance was actually taken over.
    """
    annotated = frame.copy()
    if masks:
        draw_masks(annotated, detections)

    for detection, observation in zip(detections, observations):
        x1, y1, x2, y2 = detection.xyxy
        name = names.get(detection.cls, str(detection.cls))

        if observation.xyz is None:
            # Not "Z=0.00m": there was no reading. See measure().
            depth_text = "Z=?"
            if verbose:
                print(f"{name}: no depth reading")
        else:
            X, Y, Z = observation.xyz
            depth_text = f"Z={Z:.2f}m"
            if verbose:
                print(f"{name}: X={X:.3f} Y={Y:.3f} Z={Z:.3f}")

        label = f"#{observation.object_id} {name}" if show_ids else name
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Both fields, because draw_axis needs both. measure() always sets them together, but
        # Observation defaults them to None independently, so a hand-built one can carry an
        # angle and no length -- and that would be a TypeError inside the drawing rather than a
        # skipped overlay.
        if show_axis and None not in (observation.orientation_deg, observation.axis_half_px):
            draw_axis(annotated, observation)
        # After the axis, so the dot marking where the distance was taken stays on top of it.
        cv2.circle(annotated, observation.centroid, 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"{label} {depth_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


class RealSenseSource:
    """The live camera, as a frame source for run_frames.

    This is the code that used to sit inline at the top of run_live, moved behind the same
    interface camera/rosbag_frames.ExportedFrameSource implements, so that the loop below can
    serve either without knowing which it has. The interface is: `start()`, `grab()`,
    `explain_stall(error)`, `stop()`, and the `rs`, `stream_fps` attributes.
    """

    stream_fps = CAMERA_FPS     # what enable_stream asks for, and what the figure's cap line means
    frame_shape = (480, 640)    # (height, width), matching the enable_stream calls below

    def __init__(self):
        self.rs = None
        self.pipeline = None
        self.intrinsics = None
        self.align = None

    def describe(self) -> str:
        return "the RealSense at 640x480, depth aligned to colour"

    def start(self) -> None:
        # Imported here rather than at module scope so --plot-from, --list-models and the tests
        # work on a machine with no RealSense runtime installed.
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.intrinsics = color_profile.get_intrinsics()
        self.align = rs.align(rs.stream.color)

    def grab(self):
        """(colour image, depth frame, intrinsics), or None for a frame that did not arrive whole.

        Returning None rather than raising keeps the dropped-frame accounting in one place: the
        loop carries the lost time into the next logged row.
        """
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        return np.asanyarray(color_frame.get_data()), depth_frame, self.intrinsics

    def explain_stall(self, error: RuntimeError) -> str | None:
        """A human explanation for a stalled stream, or None if this error is not that.

        librealsense raises a bare RuntimeError when the stream stalls. Ending the run with
        something actionable beats a traceback: on this hardware it almost always means the camera
        negotiated USB 2, where the D415 cannot sustain colour + depth. `rs-enumerate-devices`
        reports the link as usb_type_descriptor.
        """
        if "didn't arrive" not in str(error):
            return None
        return (f"\nthe camera stopped delivering frames ({error}).\n"
                f"Most often this is a USB 2 link: check `usb_type_descriptor` -- a D415 "
                f"on USB 3 reports 3.2, and on 2.1 the colour stream starves.")

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


def run_frames(source, detector, logger: RunLogger | None, window: str, verbose: bool = True,
               masks: bool = True, *, detection_logger: DetectionLogger | None = None,
               recorder=None, tracker=None, show_ids: bool = False, show_axis: bool = False,
               mask_depth: bool = False) -> None:
    """Grab, infer, draw, log, until ESC or the source runs out.

    Every timing decision in the project lives here, and there is deliberately only one copy of
    it: a live camera and a replayed rosbag differ in `source` alone, so their CSVs measure the
    same thing and can be put side by side.

    A source must offer start(), grab(), explain_stall(error), stop(), and -- for main(), not for
    the loop -- describe(), stream_fps and frame_shape. The loop itself uses only the first four
    and the `rs` attribute that start() sets.

    grab() returns (colour image, depth frame, intrinsics), or None for a frame that did not
    arrive whole, and raises StopIteration when it has nothing left -- the end of a bag, which is
    not an error. Only the grab sits inside that guard, so a StopIteration from anywhere else
    still propagates.

    The keyword-only arguments are the optional extras, and every one of them defaults off so
    that the measured region is byte-for-byte the work it has always been:

      tracker, mask_depth   inside the timed span, because they are per-frame pipeline work.
                            The tracker is ~0.05 ms at a handful of detections, rising to ~1 ms
                            at a hundred; mask_depth is the one addition with a real cost at any
                            detection count, which is why --objects gates it.
      detection_logger      written after the span closes, beside logger.log.
      recorder              likewise, and its write() only queues the frame -- the encode is on
                            another thread feeding another process. See camera/recording.py.
    """
    source.start()

    # Time the iteration from its own top to past its own cv2.waitKey, so that the period and the
    # inference_ms logged beside it describe the *same* frame. Measuring top-to-top instead looks
    # equivalent but is not: it attributes the previous iteration's duration to this row, which on
    # a run whose first grab blocked for 7 s paired a 13.9 ms inference with a 7747 ms period in
    # one row. Only the logger's own write falls outside, which cannot be helped and is noise.
    #
    # An incomplete frame logs no row, so its cost is carried into the next delivered one rather
    # than disappearing from the accounting.
    carried_s = 0.0
    # Advances once per delivered frame, exactly as RunLogger's own `frame` does, so the two CSVs
    # agree on which frame a row belongs to and can be joined on it.
    frame_index = 0

    try:
        while True:
            loop_top = time.perf_counter()

            try:
                grabbed = source.grab()
            except StopIteration:
                # The source has nothing left -- the end of a bag. Not an error.
                break
            except RuntimeError as error:
                explanation = source.explain_stall(error)
                if explanation is None:
                    raise
                print(f"{explanation}\n"
                      f"Keeping the {len(logger.rows) if logger else 0} frames logged so far.")
                break

            if grabbed is None:
                carried_s += time.perf_counter() - loop_top
                continue

            color_image, depth_frame, intrinsics = grabbed

            inference_start = time.perf_counter()
            detections = detector.predict(color_image)
            inference_s = time.perf_counter() - inference_start

            ids = tracker.update(detections, color_image.shape) if tracker else None
            observations = measure(detections, depth_frame, intrinsics, source.rs, ids,
                                   mask_depth=mask_depth)
            annotated = annotate(color_image, detections, observations, detector.names,
                                 verbose, masks, show_ids, show_axis)
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1)

            # The measured span ends here. Everything below writes the run's artefacts, which is
            # the same noise logger.log has always been outside it.
            loop_s = time.perf_counter() - loop_top + carried_s
            if logger is not None:
                logger.log(len(detections), inference_s, loop_s,
                           per_class=Counter(d.cls for d in detections))
            if detection_logger is not None:
                detection_logger.log(frame_index, observations, detector.names)
            if recorder is not None:
                recorder.write(annotated)
            carried_s = 0.0
            frame_index += 1

            if key == 27:
                break
    finally:
        source.stop()
        cv2.destroyAllWindows()


def run_live(detector, logger: RunLogger | None, window: str, verbose: bool = True,
             masks: bool = True, **extras) -> None:
    """Grab, infer, draw, log, until ESC. The live-camera spelling of run_frames."""
    run_frames(RealSenseSource(), detector, logger, window, verbose, masks, **extras)


# =============================================================================
# 5. COMMAND LINE
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live RealSense segmentation with speed logging.",
        epilog="An unfetched checkpoint is a git LFS stub; fetch one with "
               "`git lfs pull --include=<path>`.",
    )
    p.add_argument("--model", choices=ARCHITECTURES, default=None,
                   help=f"which architecture to load (default: {DEFAULT_MODEL}). With --weights "
                        f"this overrides what would be read from the run's args.yaml.")
    # Both exclusions are argparse's rather than hand-rolled, so they are enforced during
    # parsing -- before anything loads a checkpoint. Checking them later meant `--plot --no-log`
    # paid a torch.load, and possibly a COCO-backbone download, only to exit.
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument("--variant", default=None,
                            help=f"which run of that architecture (default: {DEFAULT_VARIANT}); "
                                 f"see --list-models")
    checkpoint.add_argument("--weights", default=None,
                            help="load an arbitrary checkpoint: a run directory or a best.pt")

    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence threshold for display and for the detection count "
                        "(default: 0.25)")
    p.add_argument("--out-dir", default=None,
                   help="parent folder for the run's own folder (default: the current directory, "
                        f"or {BAG_RUN_DIR}/<bag> for a rosbag, or next to the CSV for "
                        f"--plot-from)")
    p.add_argument("--flat", action="store_true",
                   help="put this run's files straight in --out-dir rather than in a folder of "
                        "their own. The layout runs used to have.")

    # --- the optional extras: a video, and a table of the individual objects --------------
    extras = p.add_argument_group(
        "recording and per-object logging",
        "--objects is the only thing here that adds work inside the measured loop. Without it "
        "the loop is exactly as fast as it has always been, and its loop_ms stays comparable "
        "with the runs already logged. The CSV's `extras` column records which was used.")
    extras.add_argument("--objects", action="store_true",
                        help="follow each object across frames and write <stem>-objects.csv: one "
                             "row per detection per frame, with its id, class, box, centroid, "
                             "distance and the orientation of its mask's principal axis. Also "
                             "measures distance as the median over the object's "
                             "mask instead of one pixel at the box centre, which is more "
                             "truthful and costs ~3.4 ms a frame at five detections.")
    extras.add_argument("--centre-depth", action="store_true",
                        help="with --objects: keep the single-pixel distance, so the per-object "
                             "CSV is written without the mask-median cost. It never looks at a "
                             "mask, so orientation_deg comes out blank and --show-axis has "
                             "nothing to draw -- the two cannot be combined.")
    extras.add_argument("--no-track", action="store_true",
                        help="with --objects: object_id is the detection's index within its "
                             "frame rather than an id that follows it across frames")
    extras.add_argument("--show-ids", action="store_true",
                        help="draw the object id in each label, and so in the video. Implies "
                             "--objects, since there is no id without the tracker.")
    extras.add_argument("--show-axis", action="store_true",
                        help="draw each object's principal axis -- the first PCA component of "
                             "its mask, the orientation_deg column -- through its centroid, and "
                             "so in the video. Implies --objects, since the axis comes out of "
                             "the same mask pass that writes the per-object log.")
    extras.add_argument("--save-video", nargs="?", const=True, default=None, metavar="PATH",
                        help="write the annotated frames to an MP4 for a presentation. Defaults "
                             "to <stem>.mp4 in the run's folder; give a path to override. The "
                             "encode is on a worker thread feeding ffmpeg, so it stays out of "
                             "the measured loop.")
    extras.add_argument("--video-fps", type=float, default=None, metavar="N",
                        help="frames per second to write (default: the stream's own rate). The "
                             "file holds only the frames that were shown, so a model too slow "
                             "to keep up yields a clip that plays faster than real time.")
    extras.add_argument("--video-backend", choices=BACKENDS, default="auto",
                        help="ffmpeg writes H.264, which plays everywhere including browsers and "
                             "Google Slides; opencv writes mp4v, which PowerPoint and VLC play "
                             "but browsers do not. Default: ffmpeg if it is on PATH.")
    extras.add_argument("--video-crf", type=int, default=DEFAULT_CRF, metavar="N",
                        help=f"H.264 quality for the ffmpeg backend, lower is better and bigger "
                             f"(default: {DEFAULT_CRF})")

    # --- the frame source: the camera unless one of these says otherwise -----------------
    source = p.add_mutually_exclusive_group()
    source.add_argument("--rosbag", default=None, metavar="BAG.db3",
                        help="replay a RealSense rosbag instead of the camera. It is unpacked to "
                             f"{BAG_EXPORT_DIR}/<bag>/ first (reused if already there), then run "
                             f"through the identical loop, metrics and figure.")
    source.add_argument("--from-export", default=None, metavar="DIR",
                        help="replay an already-unpacked bag, without opening the .db3 at all")

    p.add_argument("--export-only", action="store_true",
                   help="with --rosbag: unpack and stop, loading no checkpoint")
    p.add_argument("--force", action="store_true",
                   help="with --rosbag: re-unpack over an existing export")
    p.add_argument("--no-clouds", action="store_true",
                   help="with --rosbag: skip the .ply point clouds, which are ~13 MB a frame and "
                        "are not read back by the run")
    p.add_argument("--cloud-stride", type=int, default=1, metavar="N",
                   help="with --rosbag: export a point cloud every N frames (default: 1)")
    p.add_argument("--no-preload", action="store_true",
                   help="replay without decoding the clip into RAM first. Saves ~350 MiB on a "
                        "10 s bag but puts the ~20 ms PNG decode inside the timed loop, where it "
                        "inflates loop_ms and skips frames the model was quick enough for.")

    logging = p.add_mutually_exclusive_group()
    logging.add_argument("--no-log", action="store_true",
                         help="do not write any CSV -- neither the per-frame one nor, with "
                              "--objects, the per-object one")
    logging.add_argument("--plot", action="store_true",
                         help="write the speed figure when the run ends")

    p.add_argument("--plot-from", default=None, metavar="CSV",
                   help="build the figure from an existing run log and exit; needs no camera")
    p.add_argument("--list-models", action="store_true",
                   help="print the available --model/--variant pairs and exit")
    p.add_argument("--quiet", action="store_true",
                   help="do not print a line per detection. Worth using when benchmarking: at a "
                        "low --conf the printing itself can cost more than the inference.")
    p.add_argument("--no-masks", action="store_true",
                   help="draw boxes only. Worth using when comparing architectures: retina "
                        "masks cost more than the default mask path, inside the timed region.")
    return p


def run_out_dir(parent, stem: str, flat: bool = False) -> Path:
    """Where a run's files go: a folder of its own under `parent`, unless --flat.

    A run writes up to four files now -- the per-frame CSV, the figure, the per-object CSV and the
    video -- and the repository root is a demonstration of what a flat layout turns into after a
    dozen runs. The stem carries the seconds, so two runs a minute apart cannot land in the same
    folder.
    """
    return Path(parent) if flat else Path(parent) / stem


def extras_label(mask_depth: bool, video: bool) -> str:
    """The CSV's `extras` column: which optional per-frame work this run actually did.

    Keyed on `mask_depth` rather than on --objects, because --objects is not the thing that costs
    anything -- the mask-median depth is. `--objects --centre-depth` writes the per-object CSV
    while taking the identical code path a fast run takes, so labelling it `objects` would put a
    3.4 ms-a-frame run and a 0.01 ms-a-frame run under the same name. benchmark.py tells a reader
    that two runs are comparable on loop_ms when their `extras` agree, and that has to be true.

    The tracker is not in the label: it is 0.05 ms a frame at a sane --conf, which is below the
    noise in loop_ms.
    """
    return "+".join(name for name, on in (("objects", mask_depth), ("video", video)) if on)


def build_recorder(args, stem: str, directory: Path, stream_fps: float) -> VideoRecorder | None:
    """The VideoRecorder --save-video asked for, or None.

    An explicit --save-video PATH is taken literally: the user named the destination, so it does
    not get the run folder prefixed onto it.

    prepare() is called here, before the loop starts, so a bad destination or a missing ffmpeg is
    a message now rather than a traceback out of the middle of a camera session.
    """
    if not args.save_video:
        return None
    path = directory / f"{stem}.mp4" if args.save_video is True else Path(args.save_video)
    recorder = VideoRecorder(path, args.video_fps or stream_fps, backend=args.video_backend,
                             crf=args.video_crf)
    try:
        recorder.prepare()
    except (OSError, RuntimeError) as failure:
        raise SystemExit(f"cannot record to {path}: {failure}")
    return recorder


def build_source(args):
    """The frame source the command line asked for, and the default --out-dir that goes with it.

    The camera unless --rosbag or --from-export says otherwise. With --rosbag the bag is unpacked
    first; an export that is already complete is reused, so running three architectures over one
    bag pays the unpack once.
    """
    if not (args.rosbag or args.from_export):
        return RealSenseSource(), Path(".")

    # Imported here so that a camera run, --list-models and --plot-from never touch it.
    from camera import rosbag_frames

    if args.rosbag:
        export_dir = rosbag_frames.default_export_dir(args.rosbag, BAG_EXPORT_DIR)
        rosbag_frames.export_bag(args.rosbag, export_dir, clouds=not args.no_clouds,
                                 cloud_stride=args.cloud_stride, force=args.force)
    else:
        export_dir = Path(args.from_export)

    source = rosbag_frames.ExportedFrameSource(export_dir, preload=not args.no_preload)
    return source, Path(BAG_RUN_DIR) / source.bag_name


def check_extras(args) -> None:
    """Reject the extras flags that mean nothing on their own, rather than ignoring them.

    --show-ids and --show-axis are the exceptions: they imply --objects, because asking for
    something to be drawn is unambiguous about wanting it computed.

    --show-axis --centre-depth is rejected outright rather than implied away. --centre-depth is
    defined as the mode that never touches a mask, so there is no axis to draw and the run would
    come up with an empty overlay and a blank column -- an obedient answer to a request that
    cannot be met. --show-ids has no such conflict and still works with --centre-depth.
    """
    if args.show_ids or args.show_axis:
        args.objects = True
    if args.show_axis and args.centre_depth:
        raise SystemExit(
            "--show-axis needs the object masks, and --centre-depth is the mode that does not "
            "look at them: there would be nothing to draw and orientation_deg would be blank.\n"
            "Drop one of the two.")
    for flag in ("centre_depth", "no_track"):
        if getattr(args, flag) and not args.objects:
            spelling = "--" + flag.replace("_", "-")
            raise SystemExit(
                f"{spelling} only means something with --objects, which is what writes the "
                f"per-object log. Add --objects, or drop {spelling}.")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        print(list_models())
        return 0

    if args.plot_from:
        csv_path = Path(args.plot_from)
        if not csv_path.exists():
            raise SystemExit(f"no such run log: {csv_path}")
        if is_detection_log(read_log(csv_path)):
            # The two CSVs of a run differ by an "-objects" suffix, so this is an easy mistake.
            # The figure is of latency over time, which the per-object log has no column for.
            raise SystemExit(
                f"{csv_path} is a per-object log, not a per-frame one -- it has no timings to "
                f"plot.\nThe figure comes from the run's other CSV: "
                f"{csv_path.name.replace('-objects.csv', '.csv')}")
        # Taken literally: --out-dir here means "put the PNG in this folder", so it gets no run
        # subfolder. The default already writes beside the CSV, which is inside the run's folder.
        destination = (Path(args.out_dir) / f"{csv_path.stem}.png" if args.out_dir
                       else csv_path.with_suffix(".png"))
        print(f"wrote {plot_run(csv_path, destination)}")
        return 0

    if args.export_only:
        if not args.rosbag:
            raise SystemExit("--export-only needs --rosbag: there is nothing to unpack otherwise")
        # Before resolve_weights, so unpacking a bag needs no checkpoint present at all.
        from camera import rosbag_frames

        rosbag_frames.export_bag(args.rosbag,
                                 rosbag_frames.default_export_dir(args.rosbag, BAG_EXPORT_DIR),
                                 clouds=not args.no_clouds, cloud_stride=args.cloud_stride,
                                 force=args.force)
        return 0

    check_extras(args)

    # Resolved before the source, so a missing or LFS-stubbed checkpoint is reported in
    # milliseconds rather than after a two-minute unpack.
    weights, architecture, dataset = resolve_weights(args)
    model_label = MODEL_LABELS[architecture]

    source, default_out_dir = build_source(args)
    parent = Path(args.out_dir) if args.out_dir else default_out_dir

    stem = run_log_name(model_label, dataset)
    out_dir = run_out_dir(parent, stem, args.flat)
    mask_depth = args.objects and not args.centre_depth
    extras = extras_label(mask_depth, bool(args.save_video))

    # Before the checkpoint is loaded and before any file is created, so an unwritable
    # --save-video destination costs a second and leaves nothing behind, rather than being
    # discovered after a model load with an empty CSV already on disk.
    recorder = build_recorder(args, stem, out_dir, source.stream_fps)

    print(f"loading {model_label} / {dataset} from {weights}")
    detector = build_detector(weights, architecture, args.conf)
    print(f"frames from: {source.describe()}")
    print(f"input to the model: {describe_input(detector, source.frame_shape)}")
    print(f"writing this run to {out_dir}"
          + (f" (extras: {extras})" if extras else " (fast path: no extra per-frame work)"))

    logger = detection_logger = None
    if not args.no_log:
        logger = RunLogger(out_dir / f"{stem}.csv", model_label, dataset,
                           stream_fps=source.stream_fps, class_names=detector.names,
                           extras=extras)
        print(f"  {logger.path.name} -- one row per frame")
        if args.objects:
            # Sharing the per-frame log's clock so elapsed_s means the same thing in both files.
            detection_logger = DetectionLogger(out_dir / f"{stem}-objects.csv",
                                               start=logger.start)
            print(f"  {detection_logger.path.name} -- one row per detection per frame")

    tracker = Tracker() if args.objects and not args.no_track else None

    try:
        run_frames(source, detector, logger, window=f"segmentation: {model_label} / {dataset}",
                   verbose=not args.quiet, masks=not args.no_masks,
                   detection_logger=detection_logger, recorder=recorder, tracker=tracker,
                   show_ids=args.show_ids, show_axis=args.show_axis, mask_depth=mask_depth)
    finally:
        # In a finally block so that a camera that dies mid-run still leaves closed files and, if
        # asked, a figure of however many frames it managed.
        #
        # Every file is closed *first*, before anything that could fail. The figure used to be
        # drawn in the middle of this, and a plot_run that raised -- a matplotlib backend
        # problem, a full disk -- left ffmpeg's stdin open and the worker to die at interpreter
        # exit, which produces an MP4 with no moov atom: a file that exists, has a plausible
        # size, and no player will open. The artefacts are the point of the run; the figure can
        # always be rebuilt afterwards with --plot-from.
        summary = None
        if logger is not None:
            logger.close()
            summary = summarise(logger.rows)
        if detection_logger is not None:
            detection_logger.close()
        if recorder is not None:
            recorder.close()

        if logger is not None:
            if summary is None:
                print("no frames were logged")
            else:
                for line in summary.as_lines():
                    print(line)
                skipped = getattr(source, "skipped", 0)
                if skipped:
                    # Worth saying out loud on a replay: unlike a live run, the frame count is
                    # known in advance, so a shortfall means this model could not hold the
                    # stream's rate -- which is a result, not a glitch.
                    print(f"skipped: {skipped} of {skipped + summary.frames} frames, which this "
                          f"model was too slow to show at {source.stream_fps:.0f} Hz")

        if detection_logger is not None:
            print(f"objects: {detection_logger.rows_written} rows in "
                  f"{detection_logger.path}"
                  + (f", {tracker.next_id} distinct objects" if tracker else ""))

        if recorder is not None:
            for line in recorder.summary_lines(summary.wall_s if summary else None):
                print(line)

        # Last, because it is the only step here that is recoverable: everything above is
        # already safely on disk by now.
        if logger is not None and summary is not None and args.plot:
            print(f"wrote {plot_run(logger.path, out_dir / f'{stem}.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
