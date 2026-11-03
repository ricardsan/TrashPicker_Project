"""The training loop.

These details are copied from ultralytics' BaseTrainer rather than reinvented, because they are
the difference between "the same learning rate" and "the same recipe":

  * the optimiser is split into three parameter groups -- weights with decay, norm weights
    without, biases without -- which is why results.csv has lr/pg0, lr/pg1, lr/pg2.
  * the schedule is linear, lf(e) = max(1 - e/epochs, 0) * (1 - lrf) + lrf, stepped once per
    epoch before the epoch's batches (one_cycle when cos_lr), with last_epoch reset so that
    epoch e really gets lf(e).
  * warmup runs for round(warmup_epochs * batches_per_epoch) iterations, during which the bias
    group's LR falls from warmup_bias_lr to lr0*lf, every other group rises from 0, momentum
    rises from warmup_momentum, and gradient accumulation ramps from 1 to nbs/batch.
  * gradients are clipped to max_norm=10.0 after unscaling, every optimiser step.
  * close_mosaic rebuilds the dataloader, because persistent workers would otherwise never see
    that mosaic was switched off.

Validation runs the model twice per epoch, which is a torchvision quirk rather than a choice:
in eval() it returns predictions and no losses, in train() it returns losses and no predictions,
and results.csv wants both.
"""
from __future__ import annotations

import math
import os
import random
import warnings
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from other_models.artifacts import LOSS_COLUMNS, RunDirectory
from other_models.augment import Augmenter
from other_models.datasets import CLASS_NAMES, CocoInstanceDataset, collate_fn, get_source
from other_models.metrics import MASK_RATIO, Evaluator, to_mask_resolution
from other_models.models import build, predicts_masks

# plot_images caps at 16 subplots, and the validator only ever plots three batches.
PLOT_BATCHES = 3

# How much GPU memory the predicted masks of one validation batch may take. See Trainer.val_batch.
VAL_MASK_BUDGET = 2 * 1024 ** 3


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed every RNG this training touches.

    DataLoader workers do not need seeding here: torch reseeds `random`, `torch` and `np.random`
    in each worker from a base derived from the global torch RNG, so the augmentation is
    reproducible across workers once torch.manual_seed has been called.

    What `deterministic` buys, precisely: identical data order and augmentation, and cuDNN
    picking deterministic convolution algorithms. It does NOT force bit-exact determinism for
    every CUDA op -- ultralytics additionally calls torch.use_deterministic_algorithms, which on
    Mask R-CNN measured 1.39 s/step against 0.23 s/step here, a 6x tax for a guarantee this code
    does not need. Enable it yourself if you ever need to reproduce a run bit for bit.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, hyp):
    """Three parameter groups, matching ultralytics' build_optimizer."""
    norm_types = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    decay, no_decay, biases = [], [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if "bias" in name:
                biases.append(param)
            elif isinstance(module, norm_types):
                no_decay.append(param)
            else:
                decay.append(param)

    groups = [
        {"params": decay, "weight_decay": hyp.weight_decay, "param_group": "weight"},
        {"params": no_decay, "weight_decay": 0.0, "param_group": "bn"},
        {"params": biases, "weight_decay": 0.0, "param_group": "bias"},
    ]
    # match ultralytics, which canonicalises the name rather than rejecting "adam"
    known = {"Adam", "AdamW", "NAdam", "RAdam", "Adamax", "SGD", "RMSProp"}
    name = {x.lower(): x for x in known}.get(str(hyp.optimizer).lower(), str(hyp.optimizer))
    if name in {"Adam", "AdamW", "NAdam", "RAdam", "Adamax"}:
        optimizer = getattr(torch.optim, name)(groups, lr=hyp.lr0, betas=(hyp.momentum, 0.999))
    elif name == "SGD":
        optimizer = torch.optim.SGD(groups, lr=hyp.lr0, momentum=hyp.momentum, nesterov=True)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(groups, lr=hyp.lr0, momentum=hyp.momentum)
    else:
        raise NotImplementedError(f"optimizer {name!r} is not wired up")
    for group in optimizer.param_groups:
        group["initial_lr"] = hyp.lr0
    return optimizer


def lr_schedule(hyp):
    """The per-epoch learning rate multiplier."""
    if hyp.cos_lr:
        return lambda e: max((1 - math.cos(e * math.pi / hyp.epochs)) / 2, 0) * (hyp.lrf - 1) + 1
    return lambda e: max(1 - e / hyp.epochs, 0) * (1.0 - hyp.lrf) + hyp.lrf


class Trainer:
    def __init__(self, model_name: str, source_name, hyp,
                 weights_path: str | None = None, device: str | None = None):
        """source_name is a key of datasets.SOURCES, or a Source object (used by the tests)."""
        seed_everything(hyp.seed, hyp.deterministic)
        self.hyp = hyp
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        source = get_source(source_name)
        self.augmenter = Augmenter(hyp)
        self.train_set = CocoInstanceDataset(source, "train", hyp.imgsz, self.augmenter, hyp.cache)
        self.val_set = CocoInstanceDataset(source, "val", hyp.imgsz, None, hyp.cache)

        self.model, self.spec = build(model_name, len(CLASS_NAMES), hyp, weights_path)
        self.model.to(self.device)
        # Which five losses this architecture has. Mask R-CNN's and RF-DETR's are different
        # terms in the same five CSV slots -- see artifacts.LOSS_COLUMNS.
        self.loss_columns = self.spec.loss_columns or LOSS_COLUMNS
        self.with_masks = predicts_masks(self.spec) and self.train_set.has_real_masks
        if predicts_masks(self.spec) and not self.train_set.has_real_masks:
            print("WARNING: this dataset has no polygons, so the masks are filled boxes.\n"
                  "         The mask head still trains, but Mask metrics would be meaningless\n"
                  "         and are written to results.csv as NaN.")

        self.train_loader = self._build_train_loader()
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.val_batch(), shuffle=False, collate_fn=collate_fn,
            num_workers=hyp.workers, pin_memory=self.device.type == "cuda",
            # iterated twice per epoch (predictions, then losses), so keep the workers alive
            persistent_workers=hyp.workers > 0,
        )

        self.optimizer = build_optimizer(self.model, hyp)
        self.lf = lr_schedule(hyp)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # LambdaLR's constructor already performs one step. Undoing it means the step at the top
        # of epoch e lands on lf(e), not lf(e + 1) -- without this the whole schedule runs one
        # epoch ahead and disagrees with the warmup's own target of initial_lr * lf(epoch).
        # ultralytics does exactly this, with the comment "do not move".
        self.scheduler.last_epoch = -1
        # fp16 is a CUDA feature; on CPU amp is simply off whatever the config says
        self.amp = bool(hyp.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)

        self.batches = max(len(self.train_loader), 1)
        warmup_epochs = min(hyp.warmup_epochs, max(hyp.epochs - 1, 0))
        self.warmup_iters = round(warmup_epochs * self.batches) if warmup_epochs > 0 else 0
        self.accumulate = max(round(hyp.nbs / hyp.batch), 1)
        # Masks are scored at imgsz // MASK_RATIO, which is both what ultralytics' segment
        # validator does and what keeps the predicted masks a sane size in memory.
        self.mask_size = max(hyp.imgsz // MASK_RATIO, 1)

        # Which global batch indices get a train_batch<N>.jpg: the first three, and the first
        # three after mosaic closes. That is where the YOLO run's train_batch2340-2342.jpg
        # come from, and seeing both is the point -- one set is mosaicked, the other is not.
        self.plot_idx = {0, 1, 2}
        if hyp.close_mosaic:
            base = (hyp.epochs - hyp.close_mosaic) * self.batches
            self.plot_idx |= {base, base + 1, base + 2}

        self.run = RunDirectory(hyp, dict(enumerate(CLASS_NAMES)), extra_args={
            "model": model_name,
            "family": self.spec.family,
            "source": source_name if isinstance(source_name, str) else "custom",
            "pretrained_from": weights_path or "COCO",
            "device": str(self.device),
            "task": "segment" if self.with_masks else "detect",
            "classes": CLASS_NAMES,
            # What max_det actually resolved to. A DETR cannot return more detections than
            # it has queries, so hyp.max_det is clamped (rfdetr_adapter.build); recording
            # the effective number keeps args.yaml from claiming 300 when it means 100.
            "max_det_effective": self.effective_max_det(),
        }, loss_columns=self.loss_columns)

    def effective_max_det(self) -> int:
        """The most detections this model can actually return for one image.

        torchvision takes hyp.max_det literally via box_detections_per_img. A DETR is
        bounded by its own top-k: PostProcess scores every (query, class column) pair and
        keeps num_select of them, so its ceiling is num_queries * (num_classes + 1) -- 500
        for RFDETRSegSmall's 100 queries and our 5 columns, comfortably above max_det.
        Recorded in args.yaml so the number is visible rather than inferred.
        """
        postprocess = getattr(self.model, "postprocess", None)
        if postprocess is None:
            return self.hyp.max_det
        ceiling = postprocess.num_select
        net = getattr(self.model, "net", None)
        if net is not None and hasattr(net, "num_queries"):
            ceiling = min(ceiling, net.num_queries * self.model.class_columns)
        return min(self.hyp.max_det, ceiling)

    def val_batch(self) -> int:
        """Batch size for validation, which is memory-bound in a way training is not.

        A torchvision mask head pastes every one of its max_det detections back to full
        resolution before returning, so one validated image costs roughly
        max_det * imgsz^2 * 4 bytes of masks -- 0.49 GB at max_det=300, imgsz=640, measured at
        0.92 GB of peak allocation once intermediates are counted. Validating at the training
        batch size of 16 would therefore want ~15 GB for masks alone and OOM on most cards, even
        though training the same batch peaks at 5.3 GB.

        RF-DETR does not pay this at all: its mask head predicts at imgsz/4 and the adapter
        asks PostProcess not to upsample (upsample_masks_to_image_size=False), which is both
        16x less memory per mask and already the resolution metrics.MASK_RATIO scores at. So
        the cap is a torchvision concern and gating it on the family is the honest version --
        gating on `kind` would needlessly halve RF-DETR's validation batch.

        Metrics are unaffected: the evaluator accumulates over the whole split either way. The
        only visible difference is that val_batch*.jpg shows this many images per file.
        """
        if not predicts_masks(self.spec) or self.spec.family != "torchvision":
            return self.hyp.batch
        per_image = self.hyp.max_det * self.hyp.imgsz ** 2 * 4
        return max(1, min(self.hyp.batch, VAL_MASK_BUDGET // per_image))

    def _build_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set, batch_size=self.hyp.batch, shuffle=True, collate_fn=collate_fn,
            num_workers=self.hyp.workers, pin_memory=self.device.type == "cuda",
            drop_last=False, persistent_workers=self.hyp.workers > 0,
        )

    def close_mosaic(self) -> None:
        """Turn mosaic/mixup/cutmix off, and rebuild the loader so the workers notice.

        Persistent workers hold their own pickled copy of the dataset, so flipping a flag on the
        parent's Augmenter would change nothing at all -- the run would keep mosaicking to the
        end. Ultralytics has the same problem and solves it the same way, by resetting the
        dataloader after closing mosaic.
        """
        self.augmenter.close_mosaic()
        self.train_loader = self._build_train_loader()

    # --- one training epoch -------------------------------------------------
    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals, n_batches = {k: 0.0 for k in self.loss_columns}, 0
        self.optimizer.zero_grad()
        last_step = -1

        for i, (images, targets) in enumerate(self.train_loader):
            iteration = i + self.batches * epoch
            if iteration < self.warmup_iters:
                self._warmup(iteration, epoch)

            images = [im.to(self.device, non_blocking=True) for im in images]
            targets = [{k: (v.to(self.device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]

            with torch.amp.autocast(self.device.type, enabled=self.amp):
                losses = self.model(images, targets)
                loss = sum(losses.values())

            # Divided by accumulate because torchvision's losses are already mean-reduced over
            # the batch; ultralytics does not divide only because its YOLO loss is multiplied by
            # the batch size first. Same effective step, different starting point.
            #
            # RF-DETR arrives at the same place by a different route: SetCriterion normalises
            # by *this* micro-batch's box count, so averaging the micro-batches weights them
            # equally rather than by how many objects each held. That is not identical to a
            # true batch-64 step, but it is exactly what rfdetr's own automatic-optimisation
            # path does (its box-count-aware rescaling is reserved for the manual keypoint
            # path), so the comparison stays like for like.
            self.scaler.scale(loss / self.accumulate).backward()
            # `i - last_step` rather than a modulo: _warmup ramps self.accumulate mid-epoch, and
            # a modulo would reset its phase every time the ramp ticks up. This is ultralytics'
            # `ni - last_opt_step >= self.accumulate`.
            if i - last_step >= self.accumulate or i + 1 == self.batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                last_step = i

            for k, v in losses.items():
                if k in totals:
                    totals[k] += v.item()
            n_batches += 1

            if iteration in self.plot_idx and self.hyp.plots:
                self.run.plot_batch(images, targets, iteration, prefix="train")

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _warmup(self, iteration: int, epoch: int) -> None:
        xi = [0, self.warmup_iters]
        self.accumulate = max(1, int(round(float(
            np.interp(iteration, xi, [1, self.hyp.nbs / self.hyp.batch])
        ))))
        for group in self.optimizer.param_groups:
            start = self.hyp.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
            group["lr"] = float(np.interp(
                iteration, xi, [start, group["initial_lr"] * self.lf(epoch)]
            ))
            if "momentum" in group:
                group["momentum"] = float(np.interp(
                    iteration, xi, [self.hyp.warmup_momentum, self.hyp.momentum]
                ))

    # --- validation ---------------------------------------------------------
    @torch.no_grad()
    def validate(self, plot: bool = False, with_losses: bool = True) -> tuple[dict, dict]:
        """Metrics, and optionally the val/* loss columns (a second pass -- see val_losses)."""
        evaluator = Evaluator(dict(enumerate(CLASS_NAMES)), self.with_masks)

        self.model.eval()
        for index, (images, targets) in enumerate(self.val_loader):
            gpu_images = [im.to(self.device, non_blocking=True) for im in images]
            outputs = self.model(gpu_images)
            predictions = [self._to_numpy(o) for o in outputs]

            for prediction, target in zip(predictions, targets):
                evaluator.add(prediction, {
                    "boxes": target["boxes"].numpy(),
                    "labels": target["labels"].numpy() - 1,     # back to 0-based
                    # same resolution as the predicted masks, so the IoU is well defined
                    "masks": to_mask_resolution(target["masks"], self.mask_size),
                })
            if plot and index < PLOT_BATCHES and self.hyp.plots:
                self.run.plot_batch(images, targets, index, prefix="val")
                self.run.plot_predictions(images, predictions, index,
                                          with_masks=self.with_masks)

        metrics = evaluator.results(save_dir=self.run.dir, plot=plot)
        if plot and self.hyp.plots:
            evaluator.plot_confusion(self.run.dir)
        return metrics, (self.val_losses() if with_losses else {})

    def _to_numpy(self, output: dict) -> dict:
        """One image's predictions as numpy, filtered by conf, masks binarised and shrunk.

        Masks are reduced to mask_size on the GPU before they are copied back. At full
        resolution a single batch of 16 images x max_det=300 masks would be 2 GB of uint8.
        """
        keep = output["scores"] >= self.hyp.conf
        masks = output.get("masks")
        return {
            "boxes": output["boxes"][keep].float().cpu().numpy(),
            "scores": output["scores"][keep].float().cpu().numpy(),
            "labels": output["labels"][keep].cpu().numpy() - 1,   # back to 0-based
            "masks": (to_mask_resolution(masks[keep][:, 0] > 0.5, self.mask_size)
                      if masks is not None
                      else np.zeros((int(keep.sum()), self.mask_size, self.mask_size), np.uint8)),
        }

    @torch.no_grad()
    def val_losses(self) -> dict[str, float]:
        """The val/* loss columns.

        torchvision only returns losses from a model in train() mode, so this is a second pass.
        no_grad stops the gradients, but it does NOT stop BatchNorm from updating its running
        mean and variance -- and Mask R-CNN v2 has 69 live BatchNorm2d modules (v1 has none, its
        backbone is FrozenBatchNorm2d once pretrained weights are loaded). Left alone, every
        epoch would quietly fit the model's normalisation statistics to the validation set.
        So the BatchNorm layers are held in eval() for this pass.
        """
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        totals, n = {k: 0.0 for k in self.loss_columns}, 0
        for images, targets in self.val_loader:
            images = [im.to(self.device, non_blocking=True) for im in images]
            targets = [{k: (v.to(self.device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]
            with torch.amp.autocast(self.device.type, enabled=self.amp):
                losses = self.model(images, targets)
            for k, v in losses.items():
                if k in totals:
                    totals[k] += v.item()
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    # --- the whole run ------------------------------------------------------
    def train(self) -> dict:
        hyp = self.hyp
        print(f"\n{self.model_name} -> {self.run.dir}")
        print(f"{len(self.train_set)} train / {len(self.val_set)} val images at "
              f"imgsz={hyp.imgsz}, batch={hyp.batch}, accumulate={self.accumulate}, "
              f"device={self.device}")
        if self.val_batch() != hyp.batch:
            print(f"validating in batches of {self.val_batch()} to bound predicted-mask memory "
                  f"(metrics are unaffected)")
        if hyp.plots:
            self.run.plot_dataset_labels(self.train_set)

        header = f"{'epoch':>8}{'loss':>10}{'mAP50(B)':>11}{'mAP50-95(B)':>13}"
        header += f"{'mAP50(M)':>11}{'mAP50-95(M)':>13}" if self.with_masks else ""
        print(header + f"{'fitness':>10}")

        no_improvement = 0
        metrics: dict = {}
        for epoch in range(hyp.epochs):
            if hyp.close_mosaic and epoch == hyp.epochs - hyp.close_mosaic:
                # ultralytics zeroes mosaic, mixup and cutmix for the final epochs
                self.close_mosaic()
                print(f"closing mosaic/mixup/cutmix for the final {hyp.close_mosaic} epochs")

            # Stepped at the top of every epoch, including the first, because that is what
            # ultralytics' training loop does; warmup overrides the LR for the early iterations
            # anyway, so the trajectory matches theirs rather than textbook LambdaLR usage.
            # (Same reason they suppress the "step() before optimizer.step()" warning.)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            train_losses = self.train_epoch(epoch)

            metrics, val_losses = self.validate(plot=False)
            fitness = metrics.pop("fitness")
            lrs = [g["lr"] for g in self.optimizer.param_groups]
            self.run.log_epoch(epoch, train_losses, metrics, val_losses, lrs)

            improved = self.run.save_checkpoint(
                self.model, epoch, fitness, metrics,
                {**asdict(hyp), "model": self.model_name},
            )
            row = (f"{epoch + 1:>4}/{hyp.epochs:<3}{sum(train_losses.values()):>10.4f}"
                   f"{metrics['metrics/mAP50(B)']:>11.4f}{metrics['metrics/mAP50-95(B)']:>13.4f}")
            if self.with_masks:
                row += (f"{metrics['metrics/mAP50(M)']:>11.4f}"
                        f"{metrics['metrics/mAP50-95(M)']:>13.4f}")
            print(row + f"{fitness:>10.4f}{'  *' if improved else '':>5}")

            no_improvement = 0 if improved else no_improvement + 1
            if no_improvement >= hyp.patience:
                print(f"stopping early: no improvement for {hyp.patience} epochs")
                break

        metrics = self.final_eval()
        print(f"best fitness {self.run.best_fitness:.4f}  ->  {self.run.weights / 'best.pt'}")
        return {"run_dir": str(self.run.dir), "best_fitness": self.run.best_fitness, **metrics}

    def final_eval(self) -> dict:
        """Re-validate the best checkpoint and write every plot, as ultralytics' final_eval does.

        Doing it here rather than on the last training epoch means the curves, the confusion
        matrices and the val_batch*_pred images all describe best.pt, and that they are written
        even when training stopped early.
        """
        best = self.run.weights / "best.pt"
        if best.exists():
            ckpt = torch.load(best, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
        metrics, _ = self.validate(plot=self.hyp.plots, with_losses=False)
        metrics.pop("fitness", None)
        if self.hyp.plots:
            self.run.plot_metrics()
        return metrics
