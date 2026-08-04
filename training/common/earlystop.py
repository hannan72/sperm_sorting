"""Early stopping.

Small enough to inline, kept separate anyway for one reason: the stopping rule
and the best-checkpoint rule must be the *same* rule, and giving it a name is
how that stays true. :class:`EarlyStopping` and
:class:`~training.common.checkpoints.CheckpointManager` both compare through
:func:`is_improvement`, so "the epoch that reset patience" and "the epoch
written to best.pt" are the same epoch by construction rather than by
coincidence.

NaN handling is deliberate and load-bearing on this project. Macro-F1 for an
aspect whose validation split holds no positives is NaN, and the MHSMA
validation split holds seven abnormal tails out of 240 -- a fold with zero is
not hypothetical. A NaN is treated as "no improvement", never as a new best,
so a degenerate epoch can neither win the checkpoint nor extend the patience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["EarlyStopping", "is_improvement"]


def is_improvement(
    candidate: float, incumbent: float, mode: str, min_delta: float = 0.0
) -> bool:
    """Whether ``candidate`` beats ``incumbent`` by more than ``min_delta``.

    ``min_delta`` is applied on the *improving* side in both modes, so
    ``mode='min'`` requires ``candidate < incumbent - min_delta``. NaN loses
    against everything, including another NaN.
    """
    if candidate != candidate:  # NaN
        return False
    if incumbent != incumbent:  # NaN incumbent: anything real is better
        return True
    if mode == "max":
        return candidate > incumbent + min_delta
    if mode == "min":
        return candidate < incumbent - min_delta
    raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")


@dataclass
class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs.

    Parameters
    ----------
    patience
        Epochs of no improvement tolerated before :attr:`should_stop` is set.
        ``0`` disables stopping entirely (the counter is still maintained, so
        the diagnostics remain meaningful).
    mode
        ``'max'`` for metrics where higher is better (macro-F1, balanced
        accuracy, MCC), ``'min'`` for losses.
    min_delta
        Improvements smaller than this do not reset the counter. Worth setting
        above zero on small validation splits: with 240 validation crops, one
        sample changing side moves macro-F1 by ~0.004, so a zero threshold
        makes patience effectively infinite on noise alone.

    Notes
    -----
    This class does not decide *what* to monitor. That belongs to the training
    script, which must not monitor raw accuracy -- with 4.6% abnormal tails a
    constant "normal" predictor scores 95.4% and would win every epoch.
    """

    patience: int = 10
    mode: str = "max"
    min_delta: float = 0.0

    best: float = field(default=float("nan"), init=False)
    best_epoch: int = field(default=-1, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)
    #: Human-readable explanation, for the console line and experiment.json.
    reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")
        self.best = float("-inf") if self.mode == "max" else float("inf")

    def step(self, value: float, epoch: int) -> bool:
        """Feed one epoch's metric. Returns whether it was an improvement."""
        if is_improvement(value, self.best, self.mode, self.min_delta):
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.reason = ""
            return True

        self.counter += 1
        if self.patience and self.counter >= self.patience:
            self.should_stop = True
            self.reason = (
                f"no improvement in {self.counter} epoch(s) "
                f"(best {self.best:.6f} at epoch {self.best_epoch}, "
                f"min_delta {self.min_delta})"
            )
        return False

    def load_state(self, best: float, best_epoch: int, counter: int) -> None:
        """Restore the counter after ``--resume``.

        Without this a resumed run gets a fresh patience budget, so a job that
        is restarted every few hours by a scheduler can never early-stop -- it
        just runs to the epoch limit while the metric flatlines.
        """
        self.best = float(best)
        self.best_epoch = int(best_epoch)
        self.counter = int(counter)
        self.should_stop = bool(self.patience and self.counter >= self.patience)

    def to_json_dict(self) -> dict[str, Any]:
        """State for the experiment record."""
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best": None if self.best in (float("inf"), float("-inf")) else self.best,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.counter,
            "stopped_early": self.should_stop,
            "reason": self.reason,
        }
