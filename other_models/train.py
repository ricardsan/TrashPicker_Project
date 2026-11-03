#!/usr/bin/env python3
"""Train Mask R-CNN (or any model in other_models/models.py) and write Ultralytics-shaped artifacts.

Examples:

    # the two fine-tunes the paper compares against, both Mask R-CNN generations
    python other_models/train.py --model maskrcnn maskrcnn_v2 --preset FINETUNE_NMS
    python other_models/train.py --model maskrcnn maskrcnn_v2 --preset FINETUNE_AUGMENT

    # pretrain on TACO first, then fine-tune from that checkpoint. This is the pair of commands
    # 26-taco-pretrain.yaml and 26-taco-finetune.yaml describe; the fine-tune stage's recipe is
    # FINETUNE_AUGMENT, which those two YAMLs agree on hyperparameter for hyperparameter.
    # Both datasets use the same four classes, so the pretrained heads transfer as they are and
    # nothing like YOLO's cls_remap is needed.
    python other_models/train.py --model maskrcnn_v2 --source taco --preset TACO_PRETRAIN
    python other_models/train.py --model maskrcnn_v2 --preset FINETUNE_AUGMENT \
        --weights runs/segment/maskrcnn-taco/taco-pretrain_v/weights/best.pt

    # a quick check that the artifacts come out right before committing to 70 epochs
    python other_models/train.py --model maskrcnn_v2 --epochs 3 --name smoke_v

    # RF-DETR, the transformer. It takes the RFDETR_ presets, which differ from the ones above
    # only in imgsz, batch and patience -- see config.py section 4 for why each has to. Crossing
    # a preset with the wrong model family is refused rather than silently retrained.
    python other_models/train.py --model rfdetr_seg_small --preset RFDETR_FINETUNE_NMS
    python other_models/train.py --model rfdetr_seg_small --source taco \
        --preset RFDETR_TACO_PRETRAIN
    python other_models/train.py --model rfdetr_seg_small \
        --preset RFDETR_FINETUNE_AUGMENT \
        --weights runs/segment/rfdetr-taco/taco-pretrain_v/weights/best.pt

    # NOTE: the RFDETR_ fine-tune presets set warmup_bias_lr to 0.001 rather than the YAMLs'
    # 0.1, because 0.1 trains this architecture to mAP 0.002 (measured; see config.py section
    # 4 and the README). To reproduce the YAML value anyway:
    python other_models/train.py --model rfdetr_seg_small \
        --preset RFDETR_FINETUNE_AUGMENT --warmup_bias_lr 0.1

Any hyperparameter in config.py can be overridden on the command line, e.g. --epochs 3
--batch 4 --lr0 0.0005 --imgsz 512 --workers 0.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from other_models.config import Hyp, PRESETS  # noqa: E402
from other_models.datasets import ACTIVE_SOURCE, SOURCES  # noqa: E402
from other_models.engine import Trainer  # noqa: E402
from other_models.models import MODELS  # noqa: E402

def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


# Every Hyp field is exposed as a flag. `from __future__ import annotations` makes f.type a
# string, hence the lookup table rather than the annotation object itself.
TYPES = {"int": int, "float": float, "str": str, "bool": _as_bool}
OVERRIDABLE = {f.name: TYPES.get(f.type, str) for f in fields(Hyp)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["maskrcnn_v2"], choices=sorted(MODELS),
                   help="model(s) to train, in sequence (default: maskrcnn_v2)")
    p.add_argument("--source", default=ACTIVE_SOURCE, choices=sorted(SOURCES),
                   help=f"dataset to train on (default: {ACTIVE_SOURCE})")
    p.add_argument("--preset", default="FINETUNE_NMS", choices=sorted(PRESETS),
                   help="hyperparameter preset from config.py (default: FINETUNE_NMS)")
    p.add_argument("--weights", default=None,
                   help="start from this checkpoint instead of the COCO weights "
                        "(this is how a TACO pretrain is carried into the fine-tune)")
    p.add_argument("--device", default=None, help="cuda, cuda:0 or cpu (default: auto)")

    override = p.add_argument_group("hyperparameter overrides (see other_models/config.py)")
    for name, kind in OVERRIDABLE.items():
        override.add_argument(f"--{name}", type=kind, default=None,
                              metavar="BOOL" if kind is _as_bool else None)
    return p.parse_args()


def check_preset_suits_the_model(preset_name: str, model_name: str) -> None:
    """Refuse an RFDETR_ preset for a torchvision model, and vice versa.

    The RFDETR_ presets carry deviations that exist only because RF-DETR forces them -- a
    different imgsz, a smaller batch, an early stop. Handing them to Mask R-CNN would
    quietly train it at 648/4 instead of 640/16, write the result into
    runs/segment/rfdetr/, and produce a run that is no longer the like-for-like comparison
    against the YOLO one that this package exists to provide. The opposite pairing already
    fails loudly (rfdetr_adapter rejects imgsz 640 with the divisibility arithmetic); this
    makes the quiet direction loud too.
    """
    wants_rfdetr = preset_name.startswith("RFDETR_")
    is_rfdetr = MODELS[model_name].family == "rfdetr"
    if wants_rfdetr == is_rfdetr:
        return
    raise SystemExit(
        f"--preset {preset_name} is for the "
        f"{'rfdetr' if wants_rfdetr else 'torchvision'} family, but {model_name} is "
        f"{MODELS[model_name].family}. The RFDETR_ presets differ from the others in imgsz, "
        f"batch and patience for reasons specific to RF-DETR (see other_models/config.py "
        f"section 4), so crossing them over would silently train a different recipe than "
        f"the YAMLs describe.\n"
        f"Use --preset {preset_name.removeprefix('RFDETR_') if wants_rfdetr else 'RFDETR_' + preset_name}"
        f" instead, or pick a model from the other family."
    )


def main() -> None:
    args = parse_args()
    overrides = {name: getattr(args, name) for name in OVERRIDABLE
                 if getattr(args, name) is not None}
    hyp = PRESETS[args.preset].replace(**overrides) if overrides else PRESETS[args.preset]

    for model_name in args.model:
        check_preset_suits_the_model(args.preset, model_name)

    results = []
    for model_name in args.model:
        # Training several models at once gives each its own directory, prefixed with the model
        # name so the generations sit side by side. A single model keeps the preset's name.
        per_model = hyp.replace(name=f"{model_name}-{hyp.name}") if len(args.model) > 1 else hyp
        trainer = Trainer(model_name, args.source, per_model,
                          weights_path=args.weights, device=args.device)
        results.append((model_name, trainer.train()))

    print("\nsummary")
    for model_name, result in results:
        print(f"  {model_name:14s} fitness {result['best_fitness']:.4f}  {result['run_dir']}")


if __name__ == "__main__":  # DataLoader workers need this
    main()
