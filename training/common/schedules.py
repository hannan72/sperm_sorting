"""Learning-rate schedules with warmup.

Both trainers step the schedule **per optimiser step**, not per epoch. On the
small splits this project trains on -- MHSMA is ~1000 crops per aspect -- an
epoch is a few dozen steps, so an epoch-granular cosine has a dozen distinct
learning rates over the whole run and warmup is impossible to express at all.

Warmup is not decoration for the detector. CenterNet's penalty-reduced focal
loss divides by the number of true centres, and in the first few hundred steps
the heatmap head is still at its ``prior_prob`` bias while the size head is
predicting softplus(~0) pixels; the resulting gradient is large and badly
conditioned, and at the target LR it routinely diverges within the first
epoch. A linear ramp over the first few hundred steps costs nothing and removes
the failure entirely. The morphology net does not strictly need it, but it uses
the same code path so that both runs are described by the same three numbers.

The implementation is a plain ``LambdaLR`` over a pure function of the step
index. That keeps the whole schedule reproducible from ``global_step`` alone,
which is what makes ``--resume`` restore the right learning rate instead of
restarting the ramp.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

__all__ = ["SCHEDULE_KINDS", "build_scheduler", "schedule_factor"]

#: Schedules this module can build.
SCHEDULE_KINDS: tuple[str, ...] = ("cosine", "step", "constant")


def schedule_factor(
    step: int,
    *,
    kind: str,
    warmup_steps: int,
    total_steps: int,
    min_factor: float,
    step_size: int,
    step_gamma: float,
) -> float:
    """Multiplier applied to the base learning rate at ``step``.

    Pure, so the schedule can be plotted, unit-tested and reasoned about
    without constructing an optimiser.

    Parameters
    ----------
    step
        Zero-based optimiser step index.
    kind
        ``'cosine'``, ``'step'`` or ``'constant'``.
    warmup_steps
        Length of the linear ramp. The ramp starts at ``1 / warmup_steps``
        rather than at exactly 0: a first step with a learning rate of exactly
        zero is a wasted forward and backward pass, and with Adam it also
        leaves the moment estimates initialised from a zero update.
    total_steps
        Total steps in the run, used by the cosine tail.
    min_factor
        Floor on the multiplier, so the LR decays towards a small positive
        value rather than to zero. Reaching exactly zero freezes the model for
        the final steps, which wastes them.
    step_size, step_gamma
        Step-decay parameters: multiply by ``step_gamma`` every ``step_size``
        steps *after* warmup.
    """
    if kind not in SCHEDULE_KINDS:
        raise ValueError(f"unknown schedule '{kind}'; available: {', '.join(SCHEDULE_KINDS)}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    after_warmup = step - max(warmup_steps, 0)

    if kind == "constant":
        return 1.0

    if kind == "step":
        if step_size <= 0:
            raise ValueError(f"step_size must be positive for a step schedule, got {step_size}")
        decays = after_warmup // step_size
        return max(min_factor, float(step_gamma) ** decays)

    # cosine
    span = max(total_steps - max(warmup_steps, 0), 1)
    progress = min(max(after_warmup / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_factor + (1.0 - min_factor) * cosine)


def build_scheduler(
    optimizer: Any,
    *,
    kind: str = "cosine",
    warmup_steps: int = 0,
    total_steps: int = 1,
    min_factor: float = 0.01,
    step_size: int = 1000,
    step_gamma: float = 0.1,
) -> Any:
    """Build a per-step ``LambdaLR`` implementing :func:`schedule_factor`.

    ``LambdaLR`` rather than ``CosineAnnealingLR`` plus ``LinearLR`` in a
    ``SequentialLR``: the composed form stores three interacting internal
    counters whose ``state_dict`` round-trip has been subtly wrong in more than
    one torch release, and a schedule that does not survive ``--resume``
    correctly is worse than no schedule.
    """
    from torch.optim.lr_scheduler import LambdaLR

    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if warmup_steps >= total_steps and kind == "cosine":
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be smaller than total_steps "
            f"({total_steps}); otherwise the run is entirely warmup and the "
            "cosine tail never executes"
        )
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError(f"min_factor must lie in [0, 1], got {min_factor}")

    def _lambda(step: int) -> float:
        return schedule_factor(
            step,
            kind=kind,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_factor=min_factor,
            step_size=step_size,
            step_gamma=step_gamma,
        )

    lambda_fn: Callable[[int], float] = _lambda
    return LambdaLR(optimizer, lr_lambda=lambda_fn)
