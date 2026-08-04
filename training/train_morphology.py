#!/usr/bin/env python3
"""Train the four-head morphology network on MHSMA, or on simulator crops.

    python training/train_morphology.py --source synthetic --epochs 2 \
        -s morphology.backbone=simplecnn -o runs/morph_smoke

What this script is careful about
---------------------------------

**Polarity.** The network emits a logit for ``P(abnormal)``. The training
target is therefore the MHSMA integer label *verbatim* -- ``0 = normal``,
``1 = abnormal`` -- and there is no ``1 - y`` anywhere in this file. The single
permitted flip lives in :func:`sperm_sorting.morphology.polarity.flip_polarity`
and is applied only by the inference adapter. The convention string is written
into every checkpoint and every calibration bundle by the library functions
this script calls, and both refuse to load under a different one.

**Imbalance.** Verified MHSMA train prevalences are acrosome 30.1%, head
27.3%, vacuole 17.0% and tail **4.6%**. Three consequences, all implemented
here:

* ``pos_weight`` is computed **per aspect** from the training split's own
  prevalence via
  :func:`~sperm_sorting.morphology.model.pos_weight_from_prevalence`. A single
  shared weight would either ignore the tail or drown the acrosome.
* **Raw accuracy is never computed, logged or selected on.** A model that calls
  every tail normal scores 95.4% and is worthless. Model selection uses
  macro-F1 (default), balanced accuracy or MCC on the macro row.
* Any aspect whose validation split holds fewer than
  :data:`LOW_POSITIVE_WARNING` positives is flagged in the console output, in
  the JSON and in ``experiment.json``. On the real MHSMA validation split the
  tail aspect has **7** abnormal examples out of 240, and a sensitivity
  computed from seven positives moves by 0.14 when one of them changes side.

**Threshold used for the validation metrics.** Not 0.5 -- at 4.6% prevalence
that threshold predicts "normal" for every tail and makes macro-F1 a constant.
Per-aspect thresholds are fitted each epoch on the validation split by Youden's
J, which is prevalence-invariant. That is mildly optimistic, because the
threshold sees the split it is scored on; it is stated here, it is stated in
the output, and the threshold-free ROC-AUC and PR-AUC are reported alongside so
a reader can check the selection against a metric that has no such bias. The
final shipped operating point is fitted the same way on the same split, so the
selection metric and the deployed threshold are consistent rather than merely
close.

**Calibration is fitted on validation, never on test.** After training, the
best checkpoint is reloaded, validation logits are recomputed, and
:func:`~sperm_sorting.morphology.calibration.fit_calibration_bundle` fits one
temperature and one threshold per aspect. The result is written as
``calibration.json`` next to the weights, where
:meth:`MorphologyEngine.find_calibration_sidecar` looks for it.

**Provenance.** ``weights_provenance`` is stamped from
:mod:`sperm_sorting.constants`: ``public-research-baseline`` for MHSMA,
``synthetic-bootstrap`` for simulator data. Neither is ever
``device-finetuned``; only a run on real device data may claim that.

Outputs, in ``--out``: ``best.pt``, ``last.pt``, ``calibration.json``,
``metrics.json``, ``metrics.jsonl``, ``experiment.json``, ``plots/`` and, when
TensorBoard is installed, ``tensorboard/``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.bootstrap import ensure_importable  # noqa: E402

ensure_importable()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from sperm_sorting.constants import MORPHOLOGY_ASPECTS  # noqa: E402
from sperm_sorting.errors import SpermSortingError  # noqa: E402
from sperm_sorting.morphology.calibration import (  # noqa: E402
    fit_calibration_bundle,
    fit_thresholds,
    sigmoid,
)
from sperm_sorting.morphology.metrics import (  # noqa: E402
    evaluate_aspects,
    format_metrics_table,
    metrics_to_json_dict,
)
from sperm_sorting.morphology.model import (  # noqa: E402
    MorphologyLoss,
    MultiTaskMorphologyNet,
    pos_weight_from_prevalence,
    save_checkpoint,
)
from sperm_sorting.morphology.polarity import POLARITY_CONVENTION  # noqa: E402
from training.common.amp import AmpContext  # noqa: E402
from training.common.args import (  # noqa: E402
    build_parser,
    describe_device,
    dump_json,
    resolve_config,
    resolve_device,
)
from training.common.augment import MorphologyAugmentation  # noqa: E402
from training.common.checkpoints import CheckpointManager, TrainingState  # noqa: E402
from training.common.earlystop import EarlyStopping  # noqa: E402
from training.common.experiment import ExperimentRecord  # noqa: E402
from training.common.logging_utils import (  # noqa: E402
    JsonlLogger,
    TensorBoardWriter,
    console_table,
    format_duration,
    print_block,
)
from training.common.morphology_report import (  # noqa: E402
    all_four_normal_agreement,
    low_positive_warnings,
    per_aspect_confusion,
    write_morphology_plots,
)
from training.common.morphology_data import (  # noqa: E402
    MHSMA_TRAIN_PREVALENCE,
    SOURCE_KINDS,
    MorphologyArrayDataset,
    MorphologySource,
    load_morphology_source,
)
from training.common.schedules import SCHEDULE_KINDS, build_scheduler  # noqa: E402
from training.common.seeding import make_generator, seed_everything, seed_worker  # noqa: E402

#: Metrics that may drive model selection. Raw accuracy is deliberately absent.
SELECTION_METRICS: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
    "val_loss",
)


# ==========================================================================
# Arguments
# ==========================================================================


def build_argument_parser() -> Any:
    parser = build_parser(
        description=__doc__.split("\n\n")[0] if __doc__ else "Train the morphology network.",
        epilog=(
            "Examples:\n"
            "  python training/train_morphology.py --source synthetic --epochs 2 \\\n"
            "      -s morphology.backbone=simplecnn -o runs/morph_smoke\n"
            "  python training/train_morphology.py --source mhsma --data-root data/mhsma \\\n"
            "      --epochs 60 --batch-size 32 -o runs/morph_mhsma\n"
            "  python training/train_morphology.py --resume runs/morph_mhsma/last.pt \\\n"
            "      --source mhsma --data-root data/mhsma -o runs/morph_mhsma\n"
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--source",
        choices=SOURCE_KINDS,
        default="synthetic",
        help=(
            "mhsma preserves the official train/valid/test split exactly. "
            "synthetic renders labelled crops with the in-repo simulator; it is "
            "the bootstrap path before real data exists."
        ),
    )
    data.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root, passed to the MHSMA adapter. Ignored by --source synthetic.",
    )
    data.add_argument("--n-train", type=int, default=2000, help="Synthetic train size.")
    data.add_argument("--n-valid", type=int, default=500, help="Synthetic valid size.")
    data.add_argument("--n-test", type=int, default=500, help="Synthetic test size.")
    data.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Synthetic crop edge, 64 or 128. Defaults to the configured "
            "morphology.input_size."
        ),
    )
    data.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=40)
    optim.add_argument("--batch-size", type=int, default=32)
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument(
        "--schedule", choices=SCHEDULE_KINDS, default="cosine", help="LR schedule shape."
    )
    optim.add_argument(
        "--warmup-epochs",
        type=float,
        default=1.0,
        help="Linear LR ramp length, in epochs. May be fractional.",
    )
    optim.add_argument("--lr-min-factor", type=float, default=0.01)
    optim.add_argument("--step-size-epochs", type=float, default=10.0)
    optim.add_argument("--step-gamma", type=float, default=0.1)
    optim.add_argument(
        "--clip-grad-norm",
        type=float,
        default=5.0,
        help="Global gradient-norm clip. 0 disables it.",
    )
    optim.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Request mixed precision. Honoured on CUDA only; on CPU it is "
            "reported as ignored rather than silently pretended."
        ),
    )
    optim.add_argument(
        "--uncertainty-weighting",
        action="store_true",
        help=(
            "Learn the per-aspect loss balance (Kendall et al. homoscedastic "
            "uncertainty) instead of fixing it. Adds four parameters whose final "
            "values are reported in experiment.json."
        ),
    )

    select = parser.add_argument_group("selection")
    select.add_argument(
        "--select-metric",
        choices=SELECTION_METRICS,
        default="macro_f1",
        help="Macro-row metric that decides best.pt and early stopping.",
    )
    select.add_argument("--patience", type=int, default=10, help="0 disables early stopping.")
    select.add_argument("--min-delta", type=float, default=0.0)
    select.add_argument(
        "--threshold-criterion",
        choices=("youden", "f1", "balanced_accuracy", "mcc"),
        default="youden",
        help="Criterion for the per-aspect thresholds fitted on validation.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--no-tensorboard", dest="tensorboard", action="store_false", default=True
    )
    output.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    output.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        default=True,
        help="Disable training augmentation. Validation is never augmented.",
    )
    return parser


# ==========================================================================
# Model assembly
# ==========================================================================


def build_model(cfg: Any, image_size: tuple[int, int]) -> MultiTaskMorphologyNet:
    """Construct the network from the ``morphology`` config section.

    ``pretrained=False`` unconditionally. ImageNet weights would trigger a
    network fetch on first use, which a training script must not do implicitly,
    and phase-contrast microscopy crops share very little low-level statistics
    with ImageNet photographs. Set it deliberately in a fine-tuning script if
    it is ever wanted.
    """
    return MultiTaskMorphologyNet(
        backbone=cfg.morphology.backbone,
        in_channels=1,
        pretrained=False,
        input_size=image_size,
        aspects=MORPHOLOGY_ASPECTS,
    )


def build_criterion(
    train_prevalence: dict[str, float], *, uncertainty_weighting: bool
) -> tuple[MorphologyLoss, dict[str, float]]:
    """Build the loss with per-aspect ``pos_weight``.

    A split that happens to contain no positives for an aspect cannot produce a
    weight -- ``(1 - 0) / 0`` is undefined and the aspect is untrainable -- so
    the published MHSMA train prevalence is substituted and the substitution is
    returned for the record. That is a documented fallback, not a silent fix:
    it keeps the run alive while making the degenerate split visible in
    ``experiment.json``.
    """
    usable: dict[str, float] = {}
    for aspect in MORPHOLOGY_ASPECTS:
        value = float(train_prevalence.get(aspect, 0.0))
        usable[aspect] = value if 0.0 < value < 1.0 else MHSMA_TRAIN_PREVALENCE[aspect]
    weights = pos_weight_from_prevalence(usable)
    criterion = MorphologyLoss(
        MORPHOLOGY_ASPECTS,
        pos_weight=weights,
        uncertainty_weighting=uncertainty_weighting,
    )
    return criterion, weights


# ==========================================================================
# Train / validate
# ==========================================================================


def train_one_epoch(
    *,
    model: Any,
    criterion: MorphologyLoss,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    amp: AmpContext,
    device: Any,
    clip_grad_norm: float,
    state: TrainingState,
    writer: TensorBoardWriter,
    progress: bool,
) -> dict[str, float]:
    """One pass over the training split. Returns the mean losses.

    Per-aspect losses are returned as well as the total. With prevalences
    spanning 30% to 4.6% a falling total says almost nothing about whether the
    tail head is learning, and the tail head is the one that decides whether
    this model is usable.
    """
    import torch

    model.train()
    criterion.train()

    n_batches = 0
    total_loss = 0.0
    per_aspect = dict.fromkeys(MORPHOLOGY_ASPECTS, 0.0)
    grad_norms: list[float] = []

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(loader, desc=f"epoch {state.epoch + 1} train", leave=False)
        except ImportError:
            iterator = loader

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = model.logits_tensor(images)
            loss = criterion(logits, targets)
        amp.backward(loss)
        norm = amp.step(
            optimizer,
            clip_grad_norm=clip_grad_norm if clip_grad_norm > 0 else None,
            parameters=model.parameters(),
        )
        scheduler.step()
        state.global_step += 1

        total_loss += float(loss.detach())
        for name, value in criterion.last_per_aspect_losses.items():
            per_aspect[name] += value
        if norm is not None:
            grad_norms.append(norm)
            writer.add_scalar("train/grad_norm", norm, state.global_step)
        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), state.global_step)
        writer.add_scalar("train/loss_step", float(loss.detach()), state.global_step)
        n_batches += 1

    del torch  # imported only so the caller need not; nothing else uses it here
    divisor = max(n_batches, 1)
    out = {"train_loss": total_loss / divisor}
    for name in MORPHOLOGY_ASPECTS:
        out[f"train_loss_{name}"] = per_aspect[name] / divisor
    if grad_norms:
        out["train_grad_norm_mean"] = float(np.mean(grad_norms))
    return out


def collect_logits(
    *, model: Any, criterion: MorphologyLoss | None, loader: Any, device: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Run the model over a split and return ``(logits, labels, mean_loss)``.

    Logits, not probabilities: temperature scaling operates on logits, and
    round-tripping through a sigmoid and back loses precision exactly where the
    ``pos_weight``-trained heads put their confident negatives.
    """
    import torch

    model.eval()
    if criterion is not None:
        criterion.eval()

    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            logits = model.logits_tensor(images)
            if criterion is not None:
                total_loss += float(criterion(logits, targets_device))
            logit_chunks.append(logits.detach().float().cpu().numpy())
            label_chunks.append(targets.detach().cpu().numpy())
            n_batches += 1

    if not logit_chunks:
        empty = {name: np.zeros(0, dtype=np.float64) for name in MORPHOLOGY_ASPECTS}
        return empty, dict(empty), float("nan")

    stacked_logits = np.concatenate(logit_chunks, axis=0)
    stacked_labels = np.concatenate(label_chunks, axis=0)
    logits_by_aspect = {
        name: stacked_logits[:, i].astype(np.float64)
        for i, name in enumerate(MORPHOLOGY_ASPECTS)
    }
    labels_by_aspect = {
        name: stacked_labels[:, i].astype(np.int64)
        for i, name in enumerate(MORPHOLOGY_ASPECTS)
    }
    mean_loss = total_loss / max(n_batches, 1) if criterion is not None else float("nan")
    return logits_by_aspect, labels_by_aspect, mean_loss


def evaluate_split(
    logits: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    *,
    criterion: str = "youden",
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, np.ndarray]]:
    """Metrics at per-aspect thresholds fitted on this split.

    Returns ``(results, thresholds, probabilities)``. All three are in
    ``P(abnormal)`` space; there is no flip here.

    The thresholds are fitted on the same data they are scored on, which is
    mildly optimistic. It is done anyway because the alternative -- scoring at
    0.5 -- is not mildly anything: on a 4.6%-prevalence aspect the 0.5
    threshold predicts "normal" for everything, so macro-F1 becomes a constant
    and model selection stops working entirely. The threshold-free ROC-AUC and
    PR-AUC in the same table are what a reader should check the selection
    against.
    """
    probabilities = {name: sigmoid(values) for name, values in logits.items()}
    thresholds = fit_thresholds(probabilities, labels, criterion)  # type: ignore[arg-type]
    results = evaluate_aspects(labels, probabilities, thresholds)
    return results, thresholds, probabilities


# ==========================================================================
# Main
# ==========================================================================


def main(argv: list[str] | None = None) -> int:
    """Run one training job. Returns a process exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        common = resolve_config(args)
    except SpermSortingError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    cfg = common.cfg
    out_dir = common.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    record = ExperimentRecord(script="train_morphology", out_dir=out_dir)
    record.args = {**common.to_json_dict(), **_script_args(args)}

    with record:
        try:
            _run(args, common, record)
        except SpermSortingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            record.finish("failed", str(exc))
            record.save()
            return 1
    return 0


def _script_args(args: Any) -> dict[str, Any]:
    """The script-specific flags, for the experiment record."""
    keys = (
        "source", "data_root", "n_train", "n_valid", "n_test", "image_size",
        "num_workers", "epochs", "batch_size", "lr", "weight_decay", "schedule",
        "warmup_epochs", "lr_min_factor", "step_size_epochs", "step_gamma",
        "clip_grad_norm", "amp", "uncertainty_weighting", "select_metric",
        "patience", "min_delta", "threshold_criterion", "tensorboard", "plots",
        "augment",
    )
    return {key: getattr(args, key, None) for key in keys}


def _run(args: Any, common: Any, record: ExperimentRecord) -> None:
    import torch
    from torch.utils.data import DataLoader

    cfg = common.cfg
    out_dir = common.out_dir
    plot_dir = out_dir / "plots"

    # --- determinism -----------------------------------------------------
    determinism = seed_everything(cfg.run.seed, cfg.run.deterministic)
    record.determinism = determinism
    record.set_config(cfg)

    device = resolve_device(common.device)
    record.hardware = describe_device(device)

    # --- data ------------------------------------------------------------
    image_edge = int(args.image_size or cfg.morphology.input_size[0])
    source: MorphologySource = load_morphology_source(
        args.source,
        root=args.data_root,
        seed=cfg.run.seed,
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        image_size=image_edge,
    )
    train_split = source.splits["train"]
    valid_split = source.splits["valid"]

    reserved = {"name", "licence", "source", "splits"}
    record.set_dataset(
        name=str(source.info.get("name", args.source)),
        licence=str(source.info.get("licence", "unrecorded")),
        splits={name: len(split) for name, split in source.splits.items()},
        source=str(source.info.get("source", "")),
        split_detail={
            name: split.to_json_dict() for name, split in source.splits.items()
        },
        **{k: v for k, v in source.to_json_dict().items() if k not in reserved},
    )

    augmentation = MorphologyAugmentation(enabled=bool(args.augment))
    train_dataset = MorphologyArrayDataset(
        train_split, augmentation=augmentation, base_seed=cfg.run.seed
    )
    # Validation is never augmented: augmenting an evaluation split makes the
    # reported metric a measurement of the augmentation, not of the model.
    valid_dataset = MorphologyArrayDataset(valid_split, augmentation=None, base_seed=cfg.run.seed)

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=make_generator(cfg.run.seed),
        **loader_kwargs,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    # --- model, loss, optimiser -----------------------------------------
    model = build_model(cfg, (image_edge, image_edge)).to(device)
    train_prevalence = train_split.prevalence()
    criterion, pos_weights = build_criterion(
        train_prevalence, uncertainty_weighting=bool(args.uncertainty_weighting)
    )
    criterion = criterion.to(device)

    parameters = list(model.parameters()) + [
        p for p in criterion.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * int(args.epochs))
    warmup_steps = min(
        int(round(float(args.warmup_epochs) * steps_per_epoch)), max(total_steps - 1, 0)
    )
    scheduler = build_scheduler(
        optimizer,
        kind=str(args.schedule),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_factor=float(args.lr_min_factor),
        step_size=max(1, int(round(float(args.step_size_epochs) * steps_per_epoch))),
        step_gamma=float(args.step_gamma),
    )
    amp = AmpContext(device, enabled=bool(args.amp))

    record.model = {
        **model.describe(),
        "pos_weight": pos_weights,
        "train_prevalence": train_prevalence,
        "loss": "sum of four BCEWithLogits terms, per-aspect pos_weight",
        "uncertainty_weighting": bool(args.uncertainty_weighting),
        "label_polarity": POLARITY_CONVENTION,
    }
    record.training = {
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "schedule": str(args.schedule),
        "clip_grad_norm": float(args.clip_grad_norm),
        "amp": amp.to_json_dict(),
        "augmentation": augmentation.to_json_dict(),
        "select_metric": str(args.select_metric),
        "threshold_criterion": str(args.threshold_criterion),
        "validation_threshold_policy": (
            "per-aspect thresholds re-fitted on the validation split every epoch by "
            f"'{args.threshold_criterion}'; mildly optimistic, see the module "
            "docstring. Threshold-free ROC-AUC and PR-AUC are reported alongside."
        ),
    }
    record.save()

    # --- checkpoints, early stopping ------------------------------------
    provenance = source.weights_provenance
    model_id = f"{cfg.morphology.backbone}-{args.source}-seed{cfg.run.seed}"

    def deploy_writer(path: Path) -> None:
        """Write the deployable checkpoint, contract and all."""
        save_checkpoint(
            model,
            path,
            metadata={
                "trained_by": "training/train_morphology.py",
                "source": args.source,
                "dataset": record.dataset,
                "pos_weight": pos_weights,
                "train_prevalence": train_prevalence,
                "select_metric": str(args.select_metric),
                "git_commit": record.git.get("commit"),
            },
            model_id=model_id,
            weights_provenance=provenance,
        )

    select_mode = "min" if args.select_metric == "val_loss" else "max"
    manager = CheckpointManager(
        out_dir,
        deploy_writer=deploy_writer,
        metric_name=f"val_{args.select_metric}",
        mode=select_mode,
        min_delta=float(args.min_delta),
    )
    stopper = EarlyStopping(
        patience=int(args.patience), mode=select_mode, min_delta=float(args.min_delta)
    )
    extra_modules = {"criterion": criterion} if args.uncertainty_weighting else None

    state = TrainingState(metric_name=manager.metric_name, mode=select_mode)
    if common.resume is not None:
        state = manager.resume(
            common.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=amp.scaler,
            extra_modules=extra_modules,
            map_location=str(device),
        )
        stopper.load_state(state.best_metric, state.best_epoch, state.epochs_without_improvement)
        print_block(
            [
                f"resumed from {common.resume}",
                f"  epochs completed : {state.epoch}",
                f"  optimiser steps  : {state.global_step}",
                f"  best {state.metric_name} : {state.best_metric:.6f} (epoch {state.best_epoch})",
                f"  patience counter : {state.epochs_without_improvement}",
            ]
        )
        record.training["resumed_from"] = str(common.resume)
        record.training["resumed_at_epoch"] = state.epoch
        record.training["resumed_at_step"] = state.global_step

    writer = TensorBoardWriter(out_dir / "tensorboard", enabled=bool(args.tensorboard))
    jsonl = JsonlLogger(out_dir / "metrics.jsonl")
    writer.add_text("config", f"```\n{cfg.to_yaml()}\n```")
    writer.add_text("polarity", POLARITY_CONVENTION)

    # --- training loop ---------------------------------------------------
    import time

    started = time.monotonic()
    epoch_columns = (
        "train_loss", "val_loss", "val_macro_f1", "val_balanced_accuracy",
        "val_mcc", "val_roc_auc", "val_pr_auc", "lr",
    )

    try:
        while state.epoch < int(args.epochs):
            train_dataset.set_epoch(state.epoch)
            epoch_started = time.monotonic()

            train_metrics = train_one_epoch(
                model=model,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=amp,
                device=device,
                clip_grad_norm=float(args.clip_grad_norm),
                state=state,
                writer=writer,
                progress=True,
            )

            val_logits, val_labels, val_loss = collect_logits(
                model=model, criterion=criterion, loader=valid_loader, device=device
            )
            results, thresholds, _ = evaluate_split(
                val_logits, val_labels, criterion=str(args.threshold_criterion)
            )
            macro = results.get("macro", {})

            state.epoch += 1
            row: dict[str, Any] = {
                "epoch": state.epoch,
                "global_step": state.global_step,
                **train_metrics,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": round(time.monotonic() - epoch_started, 3),
            }
            for key in ("macro_f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc",
                        "sensitivity", "specificity", "precision", "ece"):
                row[f"val_{key}"] = float(macro.get(key, float("nan")))
            for aspect in MORPHOLOGY_ASPECTS:
                aspect_row = results.get(aspect, {})
                row[f"val_{aspect}_macro_f1"] = float(aspect_row.get("macro_f1", float("nan")))
                row[f"val_{aspect}_sensitivity"] = float(
                    aspect_row.get("sensitivity", float("nan"))
                )
                row[f"val_{aspect}_threshold"] = float(thresholds.get(aspect, float("nan")))

            state.history.append(row)
            jsonl.log(row)
            writer.add_scalars("train", train_metrics, state.epoch)
            writer.add_scalar("val/loss", val_loss, state.epoch)
            for aspect, aspect_row in results.items():
                writer.add_scalars(
                    f"val_{aspect}",
                    {k: v for k, v in aspect_row.items() if isinstance(v, float)},
                    state.epoch,
                )

            metric_value = (
                float(val_loss)
                if args.select_metric == "val_loss"
                else float(macro.get(args.select_metric, float("nan")))
            )
            last_path, best_path = manager.save(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=amp.scaler,
                state=state,
                metric_value=metric_value,
                extra_modules=extra_modules,
            )
            improved = stopper.step(metric_value, state.epoch)

            print_block(
                [
                    console_table(
                        [row],
                        epoch_columns,
                        title=(
                            f"epoch {state.epoch}/{args.epochs}  "
                            f"({format_duration(row['epoch_seconds'])})"
                            + ("   [best]" if best_path else "")
                        ),
                        index_column=None,
                    )
                ]
            )
            del improved, last_path

            if stopper.should_stop:
                print_block([f"early stop: {stopper.reason}"])
                record.training["early_stop"] = stopper.to_json_dict()
                break
    finally:
        writer.flush()

    duration = time.monotonic() - started
    record.training["epochs_completed"] = state.epoch
    record.training["global_steps"] = state.global_step
    record.training["wall_clock_s"] = round(duration, 3)
    record.training["early_stop"] = stopper.to_json_dict()
    if args.uncertainty_weighting:
        record.training["learned_task_weights"] = criterion.task_weights
    record.artifact("last_checkpoint", manager.last_path)
    record.artifact("best_checkpoint", manager.best_path)
    record.save()

    # --- final evaluation and calibration, on the BEST checkpoint --------
    final = _finalise(
        args=args,
        cfg=cfg,
        model=model,
        criterion=criterion,
        manager=manager,
        valid_loader=valid_loader,
        device=device,
        source=source,
        out_dir=out_dir,
        plot_dir=plot_dir,
        model_id=model_id,
        provenance=provenance,
        record=record,
        state=state,
    )

    writer.add_text("final_metrics", f"```\n{final['table']}\n```")
    writer.close()
    jsonl.close()

    record.metrics = final["metrics"]
    record.training["tensorboard"] = writer.to_json_dict()
    record.save()

    print_block(
        [
            "",
            final["table"],
            "",
            *final["warnings"],
            "",
            f"trained {state.epoch} epoch(s) in {format_duration(duration)} on {device}",
            f"best {manager.metric_name} = {state.best_metric:.6f} at epoch {state.best_epoch}",
            f"weights_provenance = {provenance}",
            f"outputs in {out_dir}",
        ]
    )


def _finalise(
    *,
    args: Any,
    cfg: Any,
    model: Any,
    criterion: MorphologyLoss,
    manager: CheckpointManager,
    valid_loader: Any,
    device: Any,
    source: MorphologySource,
    out_dir: Path,
    plot_dir: Path,
    model_id: str,
    provenance: str,
    record: ExperimentRecord,
    state: TrainingState,
) -> dict[str, Any]:
    """Reload the best checkpoint, fit calibration on validation, write outputs.

    Calibration is fitted on the **validation** split and never on test. Test
    exists to estimate what the shipped operating point does on data it has not
    influenced; fitting the operating point on test destroys that, and it does
    so invisibly -- the numbers get better.
    """
    from sperm_sorting.morphology.model import load_checkpoint

    # Reload the best weights rather than keeping whatever the last epoch left
    # in memory. Going through load_checkpoint also exercises the polarity
    # guard on the artefact that is about to be shipped.
    if manager.best_path.exists():
        best_model, best_info = load_checkpoint(manager.best_path, map_location=str(device))
        model = best_model.to(device)
        checkpoint_used = str(manager.best_path)
        recorded_polarity = str(best_info.get("label_polarity"))
    else:  # pragma: no cover - only when zero epochs completed
        checkpoint_used = "in-memory (no best.pt was written)"
        recorded_polarity = POLARITY_CONVENTION

    val_logits, val_labels, val_loss = collect_logits(
        model=model, criterion=criterion, loader=valid_loader, device=device
    )

    bundle = fit_calibration_bundle(
        val_logits,
        val_labels,
        aspects=MORPHOLOGY_ASPECTS,
        criterion=str(args.threshold_criterion),
        fitted_on=f"{source.info.get('name', args.source)}:valid",
        model_id=model_id,
    )
    calibration_path = bundle.save_json(out_dir / "calibration.json")

    calibrated = bundle.apply(val_logits)
    results = evaluate_aspects(val_labels, calibrated, bundle.thresholds)
    table = format_metrics_table(
        results,
        title=(
            "morphology metrics on the VALIDATION split, at the calibrated "
            "operating point (positive class = ABNORMAL, MHSMA label 1)"
        ),
    )
    warnings = low_positive_warnings(val_labels, "validation")

    joint = all_four_normal_agreement(val_labels, calibrated, bundle.thresholds)

    plot_paths: dict[str, str] = {}
    if args.plots:
        plot_paths = write_morphology_plots(
            labels=val_labels,
            probabilities=calibrated,
            thresholds=bundle.thresholds,
            results=results,
            plot_dir=plot_dir,
            history=state.history,
        )

    metrics = {
        "checkpoint": checkpoint_used,
        "label_polarity": recorded_polarity,
        "split": "valid",
        "val_loss_at_best": val_loss,
        "calibrated": True,
        "temperatures": dict(bundle.temperatures),
        "thresholds_p_abnormal": dict(bundle.thresholds),
        "threshold_space": bundle.threshold_space,
        "calibration_notes": dict(bundle.notes),
        "calibration_fit_metrics": {k: dict(v) for k, v in bundle.metrics.items()},
        "per_aspect": metrics_to_json_dict(results),
        "confusion": per_aspect_confusion(val_labels, calibrated, bundle.thresholds),
        "all_four_normal": joint,
        "low_positive_warnings": warnings,
        "weights_provenance": provenance,
    }
    metrics_path = dump_json(out_dir / "metrics.json", metrics)

    record.artifact("calibration", calibration_path)
    record.artifact("metrics", metrics_path)
    for key, path in plot_paths.items():
        record.artifact(f"plot_{key}", path)
    for warning in warnings:
        record.note(warning)
    record.note(
        "Calibration (temperature and thresholds) was fitted on the VALIDATION "
        "split. The test split was not used for any fitting decision."
    )
    if cfg.morphology.weights is None:
        record.note(
            "configs still point morphology.weights at null; set it to "
            f"{manager.best_path} (and keep {calibration_path} beside it) to use "
            "these weights at runtime."
        )

    return {"metrics": metrics, "table": table, "warnings": warnings}


if __name__ == "__main__":
    raise SystemExit(main())
