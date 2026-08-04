"""Field-command schemas.

A :class:`FieldCommand` is a *future-dated* instruction: it is created when a
shot is decided, but it names the monotonic instant at which the field must
actually change, which is later by the transport delay between the imaging
region and the magnetic region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..constants import SCHEMA_VERSION
from .enums import CommandOrigin, CommandOutcome, FieldCommandKind


@dataclass(slots=True)
class FieldCommand:
    """One scheduled change of the magnetic field state.

    Attributes
    ----------
    activate_at_s
        Monotonic instant at which the field must be in the commanded state.
        The scheduler dispatches *earlier* than this by the configured field
        rise/fall time so that the physical transition completes on time.
    dispatch_at_s
        Monotonic instant at which the command should leave the scheduler.
        Derived, not supplied: ``activate_at_s - lead_time_s``.
    deadline_s
        Latest dispatch instant that still counts as on-time. Dispatching
        after this marks the command :attr:`CommandOutcome.LATE`.
    """

    command_id: int
    kind: FieldCommandKind
    origin: CommandOrigin

    #: When the field must be in the commanded state.
    activate_at_s: float
    #: When the scheduler should hand it to the actuator.
    dispatch_at_s: float
    #: Latest acceptable dispatch instant.
    deadline_s: float
    #: How long the field should be held; ``None`` means "until superseded".
    duration_s: float | None = None

    #: Shot this command answers, when it came from a decision.
    shot_id: int | None = None
    #: Creation instant, monotonic.
    created_at_s: float = 0.0

    outcome: CommandOutcome = CommandOutcome.PENDING
    #: Actual dispatch instant, filled in by the scheduler.
    dispatched_at_s: float | None = None
    #: Actual acknowledgement instant, filled in by the actuator layer.
    acknowledged_at_s: float | None = None
    #: ``dispatched_at_s - dispatch_at_s``; positive means late.
    timing_error_s: float | None = None
    failure_reason: str = ""

    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def is_terminal(self) -> bool:
        return self.outcome in (
            CommandOutcome.ACKNOWLEDGED,
            CommandOutcome.SUPERSEDED,
            CommandOutcome.FAILED,
        )

    def __lt__(self, other: FieldCommand) -> bool:
        """Ordering for the scheduler's priority queue.

        Earlier dispatch first; ties broken by command id so that the heap is
        deterministic and replay reproduces the same dispatch order.
        """
        if self.dispatch_at_s != other.dispatch_at_s:
            return self.dispatch_at_s < other.dispatch_at_s
        return self.command_id < other.command_id

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "kind": str(self.kind),
            "origin": str(self.origin),
            "activate_at_s": float(self.activate_at_s),
            "dispatch_at_s": float(self.dispatch_at_s),
            "deadline_s": float(self.deadline_s),
            "duration_s": self.duration_s,
            "shot_id": self.shot_id,
            "created_at_s": float(self.created_at_s),
            "outcome": str(self.outcome),
            "dispatched_at_s": self.dispatched_at_s,
            "acknowledged_at_s": self.acknowledged_at_s,
            "timing_error_s": self.timing_error_s,
            "failure_reason": self.failure_reason,
            "schema_version": self.schema_version,
        }
