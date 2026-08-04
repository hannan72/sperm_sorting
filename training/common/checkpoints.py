"""Checkpoints that actually resume.

A checkpoint holding only ``model.state_dict()`` does not resume a run, it
restarts one with warm weights. Adam's first and second moment estimates, the
LR scheduler's step counter, the AMP scaler's loss scale, the epoch counter and
the best-metric state all have to travel with the weights, or ``--resume``
silently produces a different trajectory from the uninterrupted run -- which is
the worst kind of difference, because it looks like it worked.

Layout, in the output directory:

``last.pt``
    Written every epoch. This is what ``--resume`` reads.
``best.pt``
    Written only when the selection metric improves. This is what evaluation
    and deployment read.

Both files are *supersets* of the deployable checkpoint format, not a separate
format: they carry every key the architecture's own loader expects, plus a
``training_state`` block. So
:func:`sperm_sorting.morphology.model.load_checkpoint` reads ``best.pt``
directly, polarity check and all, and there is no export step to forget.

The deployable half is produced by calling the architecture's own writer into a
scratch file and reading it back, rather than by re-listing its keys here.
That costs one extra small write per epoch and buys the guarantee that this
module can never drift out of sync with the checkpoint contract it is wrapping.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CheckpointManager",
    "TrainingState",
    "read_checkpoint",
    "write_checkpoint",
]

#: Bumped when ``training_state`` gains or re-interprets a key.
TRAINING_STATE_VERSION: str = "1"


def _is_better(candidate: float, incumbent: float, mode: str, min_delta: float) -> bool:
    """Comparison used for both best-checkpoint selection and early stopping.

    Shared with :mod:`training.common.earlystop` through this one function so
    that the checkpoint written as "best" and the epoch that reset the patience
    counter can never be different epochs.
    """
    if mode == "max":
        return candidate > incumbent + min_delta
    if mode == "min":
        return candidate < incumbent - min_delta
    raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")


@dataclass
class TrainingState:
    """Everything needed to continue a run, minus the tensors.

    Attributes
    ----------
    epoch
        Number of epochs **completed**. A fresh run starts at 0 and the first
        epoch to execute is ``epoch``; resuming therefore continues at
        ``epoch`` without re-running it. Storing "completed" rather than "next"
        removes the off-by-one that otherwise re-runs or skips one epoch.
    global_step
        Optimiser steps taken. Drives the warmup/cosine schedule, so it must
        survive a resume or the LR jumps back to the warmup ramp.
    best_metric
        Best value of ``metric_name`` seen so far, in ``mode`` direction.
    best_epoch
        Which epoch produced it.
    epochs_without_improvement
        Early-stopping patience counter, carried so that a resume does not hand
        the run a fresh budget of patience it has already spent.
    history
        Per-epoch metric dicts, appended in order. Kept in the checkpoint (and
        not only in the JSONL log) so a resumed run can plot complete training
        curves without having to find the previous run's log.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("nan")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    metric_name: str = "val_macro_f1"
    mode: str = "max"
    history: list[dict[str, Any]] = field(default_factory=list)
    version: str = TRAINING_STATE_VERSION

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")
        if self.best_epoch < 0 and self.best_metric != self.best_metric:  # NaN check
            self.best_metric = float("-inf") if self.mode == "max" else float("inf")

    def update_best(self, value: float, epoch: int, min_delta: float = 0.0) -> bool:
        """Record ``value`` and return whether it is a new best.

        A NaN candidate is never an improvement. That case is not theoretical:
        macro-F1 on an aspect with no positives in the validation split is NaN,
        and treating NaN as "better than -inf" would pin ``best.pt`` to the
        first epoch forever.
        """
        if value != value:  # NaN
            self.epochs_without_improvement += 1
            return False
        if _is_better(value, self.best_metric, self.mode, min_delta):
            self.best_metric = float(value)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingState:
        """Rebuild from a checkpoint, tolerating keys a future version adds."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def write_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` with ``torch.save``, atomically.

    Temp-file-plus-rename, for the same reason
    :func:`sperm_sorting.morphology.model.save_checkpoint` does it: a run
    killed mid-write must not leave a truncated ``last.pt`` that makes the next
    ``--resume`` fail with an unpickling error instead of resuming.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return path


def read_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Read a checkpoint with ``weights_only=True``.

    ``weights_only=True`` is not negotiable: checkpoints get copied between
    machines and emailed around, and a pickle that can execute arbitrary code
    on load is not an acceptable thing for a training script to accept from a
    path an operator typed. Every payload this module writes is restricted to
    tensors and plain data so that the flag never has to be relaxed.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} is not a checkpoint written by this harness "
            f"(top level is {type(payload).__name__}, expected dict)"
        )
    return payload


class CheckpointManager:
    """Owns ``last.pt`` and ``best.pt`` for one training run.

    Parameters
    ----------
    out_dir
        Directory the two files live in.
    deploy_writer
        Callable that writes the *architecture's own* checkpoint format to a
        given path. For morphology this is a closure over
        :func:`sperm_sorting.morphology.model.save_checkpoint`, so the file
        keeps its polarity string and model config; for detection it writes the
        state dict plus the architecture arguments that
        :func:`sperm_sorting.detection.torch_base.load_state_dict_from_checkpoint`
        needs. Passing the writer in, rather than re-implementing either
        format here, is what keeps this module architecture-agnostic *and*
        guarantees the deployable keys are always exactly right.
    metric_name, mode
        Which metric decides "best", and in which direction.
    min_delta
        Improvement below this does not count. Matches the early stopper's, so
        the two never disagree about whether an epoch improved.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        deploy_writer: Callable[[Path], None],
        metric_name: str = "val_macro_f1",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.deploy_writer = deploy_writer
        self.metric_name = str(metric_name)
        self.mode = str(mode)
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.min_delta = float(min_delta)

    # ------------------------------------------------------------------ paths

    @property
    def last_path(self) -> Path:
        return self.out_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.out_dir / "best.pt"

    # ------------------------------------------------------------------ write

    def _deploy_payload(self) -> dict[str, Any]:
        """Ask the architecture's writer for its payload, via a scratch file.

        Round-tripping through disk looks wasteful and is: it costs one extra
        write of a few megabytes per epoch. It buys something worth more --
        this module never has to know, or restate, which keys a deployable
        checkpoint carries, so a change to the checkpoint contract in
        ``src/`` propagates here for free instead of producing a checkpoint
        that loads everywhere except in production.
        """
        scratch = self.out_dir / ".deploy.scratch.pt"
        try:
            self.deploy_writer(scratch)
            payload = read_checkpoint(scratch)
        finally:
            scratch.unlink(missing_ok=True)
            scratch.with_name(scratch.name + ".tmp").unlink(missing_ok=True)
        return payload

    def save(
        self,
        *,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        state: TrainingState,
        metric_value: float,
        extra_modules: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path | None]:
        """Write ``last.pt`` and, when the metric improved, ``best.pt``.

        ``extra_modules`` names any other stateful object that must survive a
        resume -- the loss module when it carries learned task weights, for
        instance. Each value must expose ``state_dict()``. They are stored
        under ``extra_state`` and restored by :meth:`resume`.

        Returns ``(last_path, best_path_or_None)``. The best-path element is
        ``None`` when this epoch did not improve, which is what the caller logs
        -- printing "saved best" every epoch trains people to ignore the line.
        """
        state.metric_name = self.metric_name
        state.mode = self.mode
        improved = state.update_best(metric_value, state.epoch, self.min_delta)

        payload = self._deploy_payload()
        payload["training_state"] = state.to_dict()
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = (
            scheduler.state_dict() if scheduler is not None else None
        )
        payload["scaler_state_dict"] = scaler.state_dict() if scaler is not None else None
        if extra_modules:
            payload["extra_state"] = {
                name: module.state_dict() for name, module in extra_modules.items()
            }

        last = write_checkpoint(self.last_path, payload)
        best = write_checkpoint(self.best_path, payload) if improved else None
        return last, best

    # ------------------------------------------------------------------- read

    def resume(
        self,
        path: str | Path,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        extra_modules: Mapping[str, Any] | None = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> TrainingState:
        """Restore model, optimizer, scheduler and scaler from ``path``.

        Every restorable component that is *present in the checkpoint but not
        passed in*, or vice versa, is an error rather than a warning. Resuming
        a cosine-schedule run without its scheduler state restarts the LR ramp
        from zero at epoch 40, which produces a plausible-looking loss curve
        and a worse model; that must not be a thing you can do by accident.
        """
        payload = read_checkpoint(path, map_location=map_location)

        if "state_dict" not in payload:
            raise ValueError(
                f"{path} has no 'state_dict'; it was not written by this harness"
            )
        model.load_state_dict(payload["state_dict"], strict=strict)

        opt_state = payload.get("optimizer_state_dict")
        if opt_state is None:
            raise ValueError(
                f"{path} carries no optimizer state, so it cannot resume a run -- "
                "it can only initialise one. Start a fresh run from these weights "
                "instead of passing --resume."
            )
        optimizer.load_state_dict(opt_state)

        sched_state = payload.get("scheduler_state_dict")
        if scheduler is not None and sched_state is not None:
            scheduler.load_state_dict(sched_state)
        elif scheduler is not None and sched_state is None:
            raise ValueError(
                f"{path} has no scheduler state but this run uses an LR schedule; "
                "resuming would restart the warmup ramp mid-training."
            )

        scaler_state = payload.get("scaler_state_dict")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        stored_extra = payload.get("extra_state") or {}
        for name, module in (extra_modules or {}).items():
            if name not in stored_extra:
                raise ValueError(
                    f"{path} carries no saved state for '{name}', but this run needs "
                    "it restored. Resuming without it would silently reset learned "
                    "task weights mid-training."
                )
            module.load_state_dict(stored_extra[name])

        raw_state = payload.get("training_state")
        if not isinstance(raw_state, dict):
            raise ValueError(f"{path} has no 'training_state' block; cannot resume.")
        state = TrainingState.from_dict(raw_state)
        state.metric_name = self.metric_name
        state.mode = self.mode
        return state
