#!/usr/bin/env python3
"""Train P2Net or TOD-CNN with the shared CenterNet head.

    python training/train_detector.py --source synthetic --epochs 2 \\
        -s detection.architecture=todcnn -o runs/det_smoke

Splitting is by video, and that is enforced
-------------------------------------------
At 30-160 FPS, consecutive frames of one recording are near-duplicates. A
random *frame*-level split therefore trains on near-copies of the validation
set, and the resulting AP measures memorisation. This script calls
:func:`~training.common.detection_data.assert_no_video_leakage` before the
first optimiser step and **raises** on any overlap -- it does not warn. The
failure it guards against makes the metric look *better*, so nothing
downstream would ever flag it, and a warning in a log nobody reads is
indistinguishable from no check at all.

Warmup is not optional here
---------------------------
CenterNet's penalty-reduced focal loss divides the total by the number of true
centres. In the first few hundred steps the heatmap head still sits at its
``prior_prob`` bias and the softplus size head predicts near-zero widths, so the
gradient is large and badly conditioned; at a normal learning rate the run
diverges inside the first epoch. A linear ramp over ``--warmup-steps`` costs
nothing and removes the failure. Gradient clipping is on for the same reason.

Augmentation
------------
Flips, a small rotation and mild brightness/contrast. **No colour jitter** --
the sensor is ``Mono8``, there is nothing to jitter. **No scale augmentation**
-- the premise of both architectures is that the object scale distribution is
effectively a point (see ``detection/heads.py``), so inventing scale variance
spends capacity on a distribution the optics cannot produce. Rotation inflates
axis-aligned boxes by ``|cos t| + |sin t|``, which is why the default angle is
small; see :func:`~training.common.augment.rotate_boxes`.

Frames are padded, never resized
--------------------------------
A sperm head is a handful of pixels across. Resizing either destroys it or
fabricates scale variation, so frames are padded to the network's
``size_divisor`` with the image's own median and the box coordinates are left
untouched.

Outputs, in ``--out``: ``best.pt``, ``last.pt``, ``metrics.json``,
``metrics.jsonl``, ``experiment.json``, and ``tensorboard/`` when available.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.bootstrap import ensure_importable

ensure_importable()

import datetime as _dt  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from sperm_sorting.constants import (  # noqa: E402
    WEIGHTS_PROVENANCE_PUBLIC,
    WEIGHTS_PROVENANCE_SYNTHETIC,
)
from sperm_sorting.detection.heads import centernet_loss  # noqa: E402
from sperm_sorting.errors import SpermSortingError  # noqa: E402
from training.common.amp import AmpContext  # noqa: E402
from training.common.args import (  # noqa: E402
    build_parser,
    describe_device,
    dump_json,
    resolve_config,
    resolve_device,
)
from training.common.augment import DetectionAugmentation  # noqa: E402
from training.common.checkpoints import CheckpointManager, TrainingState  # noqa: E402
from training.common.detection_data import (  # noqa: E402
    DETECTION_SOURCE_KINDS,
    DetectionClipDataset,
    assert_no_video_leakage,
    collate_detection_batch,
    load_detection_source,
)
from training.common.earlystop import EarlyStopping  # noqa: E402
from training.common.experiment import ExperimentRecord  # noqa: E402
from training.common.logging_utils import (  # noqa: E402
    JsonlLogger,
    TensorBoardWriter,
    console_table,
    format_duration,
    print_block,
)
from training.common.plots import plot_training_curves  # noqa: E402
from training.common.schedules import SCHEDULE_KINDS, build_scheduler  # noqa: E402
from training.common.seeding import (  # noqa: E402
    make_generator,
    seed_everything,
    seed_worker,
)
from training.eval_detector import FrameAnnotations, evaluate_detections  # noqa: E402

#: Selection metrics. AP50 by default: validation loss on a CenterNet head is
#: dominated by the heatmap term and can fall while detection gets worse.
SELECTION_METRICS: tuple[str, ...] = ("ap50", "map50_95", "recall", "val_loss")


# ==========================================================================
# Arguments
# ==========================================================================


def build_argument_parser() -> Any:
    parser = build_parser(
        description="Train P2Net or TOD-CNN with CenterNet targets, split by video.",
        epilog=(
            "Examples:\n"
            "  python training/train_detector.py --source synthetic --epochs 2 \\\n"
            "      -s detection.architecture=todcnn -o runs/det_smoke\n"
            "  python training/train_detector.py --source visem --data-root data/visem \\\n"
            "      --epochs 80 -s detection.architecture=p2net -o runs/det_visem\n"
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--source",
        choices=DETECTION_SOURCE_KINDS,
        default="synthetic",
        help="visem uses datasets.adapters.visem; synthetic generates clips in-repo.",
    )
    data.add_argument("--data-root", type=Path, default=None)
    data.add_argument("--n-clips", type=int, default=8, help="Synthetic clips (== videos).")
    data.add_argument("--frames-per-clip", type=int, default=12)
    data.add_argument("--frame-width", type=int, default=320)
    data.add_argument("--frame-height", type=int, default=256)
    data.add_argument(
        "--split-fractions",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VALID", "TEST"),
        help="Fractions of VIDEOS (never frames) assigned to each split.",
    )
    data.add_argument("--num-workers", type=int, default=0)

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--width", type=int, default=16, help="Base channel count.")
    arch.add_argument(
        "--blocks-per-stage",
        type=int,
        nargs="+",
        default=(1, 1, 1, 1),
        help="P2Net only: depth of each encoder stage.",
    )
    arch.add_argument("--fpn-channels", type=int, default=32, help="P2Net only.")
    arch.add_argument("--head-channels", type=int, default=32)
    arch.add_argument(
        "--block",
        choices=("basic", "mobile"),
        default="basic",
        help="P2Net only: residual block type.",
    )

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=60)
    optim.add_argument("--batch-size", type=int, default=4)
    optim.add_argument("--lr", type=float, default=1.25e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--schedule", choices=SCHEDULE_KINDS, default="cosine")
    optim.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help=(
            "Linear LR ramp, in optimiser steps. Not optional in practice: "
            "CenterNet's focal loss is unstable in the first few hundred steps."
        ),
    )
    optim.add_argument("--lr-min-factor", type=float, default=0.01)
    optim.add_argument("--step-size-epochs", type=float, default=20.0)
    optim.add_argument("--step-gamma", type=float, default=0.1)
    optim.add_argument(
        "--clip-grad-norm",
        type=float,
        default=5.0,
        help="Global gradient-norm clip. 0 disables it; leaving it on is advised.",
    )
    optim.add_argument("--size-weight", type=float, default=0.1, help="CenterNet size-loss weight.")
    optim.add_argument("--offset-weight", type=float, default=1.0)
    optim.add_argument("--amp", action="store_true", help="Honoured on CUDA only.")

    select = parser.add_argument_group("selection")
    select.add_argument("--select-metric", choices=SELECTION_METRICS, default="ap50")
    select.add_argument("--patience", type=int, default=15)
    select.add_argument("--min-delta", type=float, default=0.0)

    output = parser.add_argument_group("output")
    output.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", default=True)
    output.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    output.add_argument("--no-augment", dest="augment", action="store_false", default=True)
    return parser


# ==========================================================================
# Model
# ==========================================================================


def build_network(cfg: Any, args: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Build the network and its :class:`Detector` wrapper around the same object.

    The wrapper is not a second copy. It holds a reference to the very module
    being trained, so validation runs through the *deployment* path -- the same
    preprocessing, decode, NMS and box-size filtering the runtime uses. A
    validation AP computed from raw heatmaps instead would be measuring
    something the product never does, and would silently hide a decode/target
    disagreement, which is the single most damaging bug available in a
    CenterNet pipeline.
    """
    architecture = cfg.detection.architecture
    if architecture not in ("p2net", "todcnn"):
        raise SpermSortingError(
            f"detection.architecture is '{architecture}', which this trainer cannot "
            "build. Use 'p2net' or 'todcnn'; 'onnx' and 'oracle' are inference-only."
        )

    num_classes = len(cfg.detection.class_names)
    if architecture == "p2net":
        from sperm_sorting.detection.p2net import P2Net, P2NetDetector

        blocks = tuple(int(b) for b in args.blocks_per_stage)
        kwargs: dict[str, Any] = {
            "in_channels": 1,
            "width": int(args.width),
            "num_classes": num_classes,
            "num_stages": len(blocks),
            "blocks_per_stage": blocks,
            "fpn_channels": int(args.fpn_channels),
            "head_channels": int(args.head_channels),
            "block": str(args.block),
        }
        net = P2Net(**kwargs)
        detector = P2NetDetector(
            cfg.detection,
            width=int(args.width),
            num_stages=len(blocks),
            blocks_per_stage=blocks,
            fpn_channels=int(args.fpn_channels),
            head_channels=int(args.head_channels),
            block=str(args.block),
            net=net,
        )
    else:
        from sperm_sorting.detection.todcnn import TodCnnDetector, TodCnnNet

        kwargs = {
            "in_channels": 1,
            "width": int(args.width),
            "num_classes": num_classes,
            "stride": 4,
            "head_channels": int(args.head_channels),
        }
        net = TodCnnNet(**kwargs)
        detector = TodCnnDetector(
            cfg.detection,
            width=int(args.width),
            stride=4,
            head_channels=int(args.head_channels),
            net=net,
        )

    description = {
        "architecture": architecture,
        "arch_kwargs": kwargs,
        "stride": int(detector.stride),
        "size_divisor": int(detector.size_divisor),
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "in_channels": 1,
        "num_classes": num_classes,
    }
    return net, detector, description


# ==========================================================================
# Train / validate
# ==========================================================================


def train_one_epoch(
    *,
    net: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    amp: AmpContext,
    device: Any,
    clip_grad_norm: float,
    size_weight: float,
    offset_weight: float,
    state: TrainingState,
    writer: TensorBoardWriter,
    progress: bool,
) -> dict[str, float]:
    """One pass over the training split.

    The three loss terms are logged separately. A total that falls only because
    the size term is falling is a run that is not learning to *detect*, and
    that is invisible from the total -- which is exactly why
    :func:`~sperm_sorting.detection.heads.centernet_loss` returns them apart.
    """
    net.train()
    totals = {"loss": 0.0, "loss_heatmap": 0.0, "loss_size": 0.0, "loss_offset": 0.0}
    grad_norms: list[float] = []
    n_batches = 0

    iterator = loader
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(loader, desc=f"epoch {state.epoch + 1} train", leave=False)
        except ImportError:
            iterator = loader

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in targets.items()
            if key in ("heatmap", "size", "offset", "mask")
        }

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            outputs = net(images)
            losses = centernet_loss(outputs, targets, size_weight, offset_weight)
        amp.backward(losses["loss"])
        norm = amp.step(
            optimizer,
            clip_grad_norm=clip_grad_norm if clip_grad_norm > 0 else None,
            parameters=net.parameters(),
        )
        scheduler.step()
        state.global_step += 1

        for key in totals:
            totals[key] += float(losses[key].detach())
        if norm is not None:
            grad_norms.append(norm)
            writer.add_scalar("train/grad_norm", norm, state.global_step)
        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), state.global_step)
        writer.add_scalar("train/loss_step", float(losses["loss"].detach()), state.global_step)
        n_batches += 1

    divisor = max(n_batches, 1)
    out = {f"train_{key}": value / divisor for key, value in totals.items()}
    out["train_loss"] = totals["loss"] / divisor
    if grad_norms:
        out["train_grad_norm_mean"] = float(np.mean(grad_norms))
        out["train_grad_norm_max"] = float(np.max(grad_norms))
    return out


def validate(
    *,
    net: Any,
    detector: Any,
    loader: Any,
    frames: Any,
    device: Any,
    size_weight: float,
    offset_weight: float,
    score_threshold: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Validation loss plus full detection metrics through the deployment path."""
    import torch

    from sperm_sorting.schemas.enums import SourceKind, TimestampSource
    from sperm_sorting.schemas.frame import FramePacket

    net.eval()
    totals = {"loss": 0.0, "loss_heatmap": 0.0, "loss_size": 0.0, "loss_offset": 0.0}
    n_batches = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = {
                key: value.to(device, non_blocking=True)
                for key, value in targets.items()
                if key in ("heatmap", "size", "offset", "mask")
            }
            outputs = net(images)
            losses = centernet_loss(outputs, targets, size_weight, offset_weight)
            for key in totals:
                totals[key] += float(losses[key])
            n_batches += 1

    truth: list[FrameAnnotations] = []
    predictions: list[FrameAnnotations] = []
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            packet = FramePacket(
                frame_id=index,
                image=np.ascontiguousarray(frame.image),
                capture_time_s=index / 160.0,
                timestamp_source=TimestampSource.SYNTHETIC,
                source_kind=SourceKind.SYNTHETIC,
            )
            detections = detector.detect(packet)
            truth.append(FrameAnnotations(index, frame.boxes, frame.class_ids))
            predictions.append(FrameAnnotations.from_detections(index, detections))

    metrics = evaluate_detections(truth, predictions, score_threshold=score_threshold)
    divisor = max(n_batches, 1)
    losses_out = {f"val_{key}": value / divisor for key, value in totals.items()}
    losses_out["val_loss"] = totals["loss"] / divisor
    return losses_out, metrics


# ==========================================================================
# Main
# ==========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        common = resolve_config(args)
    except SpermSortingError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    out_dir = common.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentRecord(script="train_detector", out_dir=out_dir)
    record.args = {
        **common.to_json_dict(),
        **{
            key: getattr(args, key, None)
            for key in (
                "source", "data_root", "n_clips", "frames_per_clip", "frame_width",
                "frame_height", "split_fractions", "num_workers", "width",
                "blocks_per_stage", "fpn_channels", "head_channels", "block",
                "epochs", "batch_size", "lr", "weight_decay", "schedule",
                "warmup_steps", "clip_grad_norm", "size_weight", "offset_weight",
                "amp", "select_metric", "patience", "augment",
            )
        },
    }

    with record:
        try:
            _run(args, common, record)
        except SpermSortingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            record.finish("failed", str(exc))
            record.save()
            return 1
    return 0


def _run(args: Any, common: Any, record: ExperimentRecord) -> None:
    import torch
    from torch.utils.data import DataLoader

    cfg = common.cfg
    out_dir = common.out_dir

    record.determinism = seed_everything(cfg.run.seed, cfg.run.deterministic)
    record.set_config(cfg)
    device = resolve_device(common.device)
    record.hardware = describe_device(device)

    # --- data, split by video -------------------------------------------
    source = load_detection_source(
        args.source,
        root=args.data_root,
        seed=cfg.run.seed,
        fractions=tuple(float(f) for f in args.split_fractions),
        n_clips=int(args.n_clips),
        frames_per_clip=int(args.frames_per_clip),
        width=int(args.frame_width),
        height=int(args.frame_height),
    )
    # Called a second time, explicitly, on the video lists the loader produced.
    # Belt and braces on purpose: this is the one invariant whose violation
    # improves every number downstream, so it is checked where it is used and
    # not only where the split was made.
    leakage = assert_no_video_leakage(source.video_splits)
    record.training = {"video_leakage_check": leakage}

    reserved = {"name", "licence", "source", "splits"}
    record.set_dataset(
        name=str(source.info.get("name", args.source)),
        licence=str(source.info.get("licence", "unrecorded")),
        splits={name: len(frames) for name, frames in source.splits.items()},
        source=str(source.info.get("source", "")),
        **{k: v for k, v in source.to_json_dict().items() if k not in reserved},
    )

    net, detector, description = build_network(cfg, args)
    net = net.to(device)
    record.model = description

    augmentation = DetectionAugmentation(enabled=bool(args.augment))
    train_dataset = DetectionClipDataset(
        source.splits["train"],
        stride=int(detector.stride),
        size_divisor=int(detector.size_divisor),
        num_classes=len(cfg.detection.class_names),
        augmentation=augmentation,
        base_seed=cfg.run.seed,
    )
    valid_dataset = DetectionClipDataset(
        source.splits["valid"],
        stride=int(detector.stride),
        size_divisor=int(detector.size_divisor),
        num_classes=len(cfg.detection.class_names),
        augmentation=None,  # never augment an evaluation split
        base_seed=cfg.run.seed,
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "collate_fn": collate_detection_batch,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=make_generator(cfg.run.seed), **loader_kwargs
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * int(args.epochs))
    warmup_steps = min(int(args.warmup_steps), max(total_steps - 1, 0))
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

    provenance = (
        WEIGHTS_PROVENANCE_SYNTHETIC if args.source == "synthetic" else WEIGHTS_PROVENANCE_PUBLIC
    )
    record.training.update(
        {
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "epochs_requested": int(args.epochs),
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "schedule": str(args.schedule),
            "clip_grad_norm": float(args.clip_grad_norm),
            "size_weight": float(args.size_weight),
            "offset_weight": float(args.offset_weight),
            "amp": amp.to_json_dict(),
            "augmentation": augmentation.to_json_dict(),
            "select_metric": str(args.select_metric),
            "weights_provenance": provenance,
            "frames_resized": False,
            "frame_geometry_note": (
                "frames are padded to size_divisor with their own median, never "
                "resized: a sperm head is a handful of pixels and resizing either "
                "destroys it or fabricates scale variation"
            ),
        }
    )
    record.save()

    # --- checkpoints -----------------------------------------------------
    def deploy_writer(path: Path) -> None:
        """Write a checkpoint ``load_state_dict_from_checkpoint`` can read."""
        from training.common.checkpoints import write_checkpoint

        write_checkpoint(
            path,
            {
                "state_dict": net.state_dict(),
                "architecture": description["architecture"],
                "arch_kwargs": description["arch_kwargs"],
                "stride": description["stride"],
                "size_divisor": description["size_divisor"],
                "class_names": list(cfg.detection.class_names),
                "weights_provenance": provenance,
                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "torch_version": str(torch.__version__),
                "trained_by": "training/train_detector.py",
                "git_commit": record.git.get("commit") or "",
            },
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
    state = TrainingState(metric_name=manager.metric_name, mode=select_mode)

    if common.resume is not None:
        state = manager.resume(
            common.resume,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=amp.scaler,
            map_location=str(device),
        )
        stopper.load_state(state.best_metric, state.best_epoch, state.epochs_without_improvement)
        print_block(
            [
                f"resumed from {common.resume}",
                f"  epochs completed : {state.epoch}",
                f"  optimiser steps  : {state.global_step}",
                f"  best {state.metric_name} : {state.best_metric:.6f} (epoch {state.best_epoch})",
            ]
        )
        record.training["resumed_from"] = str(common.resume)
        record.training["resumed_at_epoch"] = state.epoch

    writer = TensorBoardWriter(out_dir / "tensorboard", enabled=bool(args.tensorboard))
    jsonl = JsonlLogger(out_dir / "metrics.jsonl")
    writer.add_text("config", f"```\n{cfg.to_yaml()}\n```")

    started = time.monotonic()
    columns = ("train_loss", "train_loss_heatmap", "val_loss", "val_ap50",
               "val_map50_95", "val_recall", "lr")
    final_metrics: dict[str, Any] = {}

    try:
        while state.epoch < int(args.epochs):
            train_dataset.set_epoch(state.epoch)
            epoch_started = time.monotonic()

            train_metrics = train_one_epoch(
                net=net,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                amp=amp,
                device=device,
                clip_grad_norm=float(args.clip_grad_norm),
                size_weight=float(args.size_weight),
                offset_weight=float(args.offset_weight),
                state=state,
                writer=writer,
                progress=True,
            )
            val_losses, val_metrics = validate(
                net=net,
                detector=detector,
                loader=valid_loader,
                frames=source.splits["valid"],
                device=device,
                size_weight=float(args.size_weight),
                offset_weight=float(args.offset_weight),
                score_threshold=float(cfg.detection.score_threshold),
            )
            final_metrics = val_metrics

            state.epoch += 1
            row: dict[str, Any] = {
                "epoch": state.epoch,
                "global_step": state.global_step,
                **train_metrics,
                **val_losses,
                "val_ap50": val_metrics["ap50"],
                "val_map50_95": val_metrics["map50_95"],
                "val_precision": val_metrics["operating_point"]["precision"],
                "val_recall": val_metrics["operating_point"]["recall"],
                "val_f1": val_metrics["operating_point"]["f1"],
                "val_recall_small": val_metrics["small_objects"]["recall_small"],
                "val_count_bias": val_metrics["counting"].get("mean_signed_error"),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": round(time.monotonic() - epoch_started, 3),
            }
            state.history.append(row)
            jsonl.log(row)
            writer.add_scalars("train", train_metrics, state.epoch)
            writer.add_scalars(
                "val", {k: v for k, v in row.items() if k.startswith("val_") and isinstance(v, float)},
                state.epoch,
            )

            metric_value = {
                "ap50": val_metrics["ap50"],
                "map50_95": val_metrics["map50_95"],
                "recall": val_metrics["operating_point"]["recall"],
                "val_loss": val_losses["val_loss"],
            }[args.select_metric]

            _, best_path = manager.save(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=amp.scaler,
                state=state,
                metric_value=float(metric_value),
            )
            stopper.step(float(metric_value), state.epoch)

            print_block(
                [
                    console_table(
                        [row],
                        columns,
                        title=(
                            f"epoch {state.epoch}/{args.epochs}  "
                            f"({format_duration(row['epoch_seconds'])})"
                            + ("   [best]" if best_path else "")
                        ),
                    )
                ]
            )
            if stopper.should_stop:
                print_block([f"early stop: {stopper.reason}"])
                break
    finally:
        writer.flush()

    duration = time.monotonic() - started
    record.training["epochs_completed"] = state.epoch
    record.training["global_steps"] = state.global_step
    record.training["wall_clock_s"] = round(duration, 3)
    record.training["early_stop"] = stopper.to_json_dict()
    record.training["tensorboard"] = writer.to_json_dict()
    record.artifact("last_checkpoint", manager.last_path)
    record.artifact("best_checkpoint", manager.best_path)

    if args.plots:
        path = plot_training_curves(
            state.history,
            out_dir / "plots" / "training_curves.png",
            loss_keys=("train_loss", "train_loss_heatmap", "val_loss"),
            metric_keys=("val_ap50", "val_map50_95", "val_recall", "val_recall_small"),
            title="detector",
        )
        if path:
            record.artifact("plot_training_curves", path)

    metrics_path = dump_json(
        out_dir / "metrics.json",
        {
            "final_validation": final_metrics,
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
            "select_metric": manager.metric_name,
            "weights_provenance": provenance,
            "video_splits": source.video_splits,
            "leakage_check": leakage,
        },
    )
    record.artifact("metrics", metrics_path)
    record.metrics = {
        "final_validation": final_metrics,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
    }
    writer.close()
    jsonl.close()
    record.save()

    print_block(
        [
            "",
            f"trained {state.epoch} epoch(s) in {format_duration(duration)} on {device}",
            f"best {manager.metric_name} = {state.best_metric:.6f} at epoch {state.best_epoch}",
            "video splits: " + ", ".join(
                f"{name}={len(ids)}" for name, ids in source.video_splits.items()
            ),
            f"weights_provenance = {provenance}",
            f"outputs in {out_dir}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
