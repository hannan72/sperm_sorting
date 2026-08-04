"""Actuation scheduler.

The imaging region and the magnetic region are physically separated, so a
decision made about the fluid under the microscope must be *applied later*, to
the moment that same fluid reaches the magnet. The scheduler owns that delay.

Timeline for one decision::

    t_gate      the shot's fluid segment passes the counting gate
    t_activate  = t_gate + transport_delay          field must be in state
    t_dispatch  = t_activate - rise_time - margin   command must leave here
    t_deadline  = t_dispatch + late_tolerance       after this it is LATE

Every one of ``transport_delay``, ``rise_time`` and the margins is a property
of the built kit, not of the software. They default to zero and
:attr:`SchedulingConfig.calibrated` is false until they are measured, which is
why the scheduler refuses to arm at all unless calibration is present or has
been explicitly waived for a bench test.

Failure policy, in order of severity:

* a command dispatched later than its deadline is marked ``LATE`` and counted;
* a command already later than ``drop_if_late_by_ms`` is **dropped**, because
  firing it would gate the wrong fluid segment -- acting on the wrong segment
  is worse than not acting;
* a command superseded before dispatch is dropped as ``SUPERSEDED``;
* if the watchdog is not fed, the field goes to FIELD_OFF regardless of what
  is queued.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable
from typing import Any

from ..config import SchedulingConfig
from ..errors import LateCommandError
from ..schemas.command import FieldCommand
from ..schemas.enums import CommandOrigin, CommandOutcome, FieldCommandKind
from .clock import Clock, MonotonicClock

logger = logging.getLogger(__name__)


class ActuationScheduler:
    """A priority queue of future field commands, drained against a clock.

    Not thread-safe; drive it from one loop. The pipeline calls :meth:`submit`
    when a shot is decided and :meth:`poll` every tick.
    """

    def __init__(
        self,
        cfg: SchedulingConfig,
        clock: Clock | None = None,
        *,
        dispatch: Callable[[FieldCommand], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.clock = clock or MonotonicClock()
        #: Called to actually drive the hardware. Returns True on success.
        #: Injected rather than imported so the scheduler can be tested with
        #: no actuator at all.
        self._dispatch = dispatch
        self._heap: list[FieldCommand] = []
        self._next_command_id = 0
        self._history: list[FieldCommand] = []
        #: The last state the field was commanded into.
        self._current_state = FieldCommandKind.FIELD_OFF
        self._armed = False

        self.n_dispatched = 0
        self.n_late = 0
        self.n_dropped_late = 0
        self.n_superseded = 0
        self.n_failed = 0
        #: Signed dispatch timing errors in seconds, for P50/P95 reporting.
        self.timing_errors_s: list[float] = []

    # ------------------------------------------------------------------ arm

    def arm(self) -> None:
        """Permit dispatch. Refuses if the kit timing is uncalibrated.

        Unknown physical timing is not a detail to be filled in with a
        plausible number: a wrong transport delay applies the field to the
        wrong fluid, which is a silent, invisible failure that would corrupt
        every sorted sample without ever raising an error.
        """
        self.cfg.require_calibrated()
        self._armed = True
        logger.info(
            "scheduler armed (transport=%.1fms rise=%.1fms fall=%.1fms "
            "calibration=%s)",
            self.cfg.transport_delay_ms,
            self.cfg.field_rise_time_ms,
            self.cfg.field_fall_time_ms,
            self.cfg.calibration_id or "none",
        )

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def current_state(self) -> FieldCommandKind:
        return self._current_state

    # --------------------------------------------------------------- submit

    def submit(
        self,
        kind: FieldCommandKind,
        *,
        gate_time_s: float,
        shot_id: int | None = None,
        origin: CommandOrigin = CommandOrigin.DECISION,
        duration_s: float | None = None,
    ) -> FieldCommand:
        """Queue a command for the fluid segment that passed the gate.

        Parameters
        ----------
        gate_time_s
            Monotonic instant at which the segment crossed the counting gate.
            The transport delay is added to it to find the activation instant.
        """
        transport_s = self.cfg.transport_delay_ms / 1000.0
        activate_at = gate_time_s + transport_s

        # A rising field needs its rise time; a collapsing one needs its fall
        # time. Using the wrong one biases every command in one direction.
        if kind is FieldCommandKind.FIELD_ON:
            settle_s = self.cfg.field_rise_time_ms / 1000.0
        else:
            settle_s = self.cfg.field_fall_time_ms / 1000.0
        lead_s = settle_s + self.cfg.pre_activation_margin_ms / 1000.0

        dispatch_at = activate_at - lead_s
        deadline = dispatch_at + self.cfg.late_tolerance_ms / 1000.0

        if duration_s is None and self.cfg.pulse_duration_ms is not None:
            duration_s = self.cfg.pulse_duration_ms / 1000.0

        command = FieldCommand(
            command_id=self._next_command_id,
            kind=kind,
            origin=origin,
            activate_at_s=activate_at,
            dispatch_at_s=dispatch_at,
            deadline_s=deadline,
            duration_s=duration_s,
            shot_id=shot_id,
            created_at_s=self.clock.now(),
        )
        self._next_command_id += 1
        heapq.heappush(self._heap, command)
        self._history.append(command)
        logger.debug(
            "queued command %d %s for t=%.4f (dispatch %.4f)",
            command.command_id,
            kind,
            activate_at,
            dispatch_at,
        )
        return command

    def submit_immediate(
        self,
        kind: FieldCommandKind,
        *,
        origin: CommandOrigin,
        reason: str = "",
    ) -> FieldCommand:
        """Queue a command for right now. Used for safe defaults and faults.

        Bypasses the transport delay: a safety command is about the actuator's
        state, not about a particular fluid segment.
        """
        now = self.clock.now()
        command = FieldCommand(
            command_id=self._next_command_id,
            kind=kind,
            origin=origin,
            activate_at_s=now,
            dispatch_at_s=now,
            deadline_s=now + self.cfg.late_tolerance_ms / 1000.0,
            duration_s=None,
            shot_id=None,
            created_at_s=now,
            failure_reason=reason,
        )
        self._next_command_id += 1
        heapq.heappush(self._heap, command)
        self._history.append(command)
        return command

    # ----------------------------------------------------------------- poll

    def poll(self, now_s: float | None = None) -> list[FieldCommand]:
        """Dispatch every command whose time has come. Returns those handled.

        Call this every tick. It is cheap when the queue is empty.
        """
        now = self.clock.now() if now_s is None else now_s
        handled: list[FieldCommand] = []

        while self._heap and self._heap[0].dispatch_at_s <= now:
            command = heapq.heappop(self._heap)

            if command.outcome is CommandOutcome.SUPERSEDED:
                handled.append(command)
                continue

            lateness = now - command.dispatch_at_s
            if lateness > self.cfg.drop_if_late_by_ms / 1000.0:
                # The fluid this command was meant for is long gone. Applying
                # the field now would divert an unrelated segment.
                command.outcome = CommandOutcome.FAILED
                command.timing_error_s = lateness
                command.failure_reason = (
                    f"dropped: {lateness * 1000:.1f} ms late, beyond the "
                    f"{self.cfg.drop_if_late_by_ms:.1f} ms limit; firing it "
                    "would act on the wrong fluid segment"
                )
                self.n_dropped_late += 1
                logger.error(
                    "command %d dropped: %.1f ms late", command.command_id, lateness * 1000
                )
                handled.append(command)
                continue

            ok = True
            if self._dispatch is not None:
                if not self._armed:
                    command.outcome = CommandOutcome.FAILED
                    command.failure_reason = "scheduler is not armed"
                    self.n_failed += 1
                    handled.append(command)
                    continue
                try:
                    ok = self._dispatch(command)
                except Exception as exc:  # actuator faults must not stop the loop
                    ok = False
                    command.failure_reason = f"actuator raised: {exc}"
                    logger.exception("actuator failed on command %d", command.command_id)

            command.dispatched_at_s = now
            command.timing_error_s = lateness
            self.timing_errors_s.append(lateness)

            if not ok:
                command.outcome = CommandOutcome.FAILED
                self.n_failed += 1
            elif now > command.deadline_s:
                command.outcome = CommandOutcome.LATE
                self.n_late += 1
                self.n_dispatched += 1
                self._current_state = command.kind
                logger.warning(
                    "command %d dispatched %.2f ms late",
                    command.command_id,
                    lateness * 1000,
                )
            else:
                command.outcome = CommandOutcome.DISPATCHED
                self.n_dispatched += 1
                self._current_state = command.kind

            handled.append(command)

        return handled

    def next_dispatch_time(self) -> float | None:
        """When :meth:`poll` will next have work, or ``None`` if idle."""
        return self._heap[0].dispatch_at_s if self._heap else None

    def supersede(self, before_time_s: float) -> int:
        """Mark queued commands activating before ``before_time_s`` as stale.

        Used when a newer decision covers the same fluid: the older command
        would fight it. Returns how many were superseded.
        """
        count = 0
        for command in self._heap:
            if (
                command.outcome is CommandOutcome.PENDING
                and command.activate_at_s < before_time_s
            ):
                command.outcome = CommandOutcome.SUPERSEDED
                count += 1
        self.n_superseded += count
        return count

    def check_late(self, now_s: float | None = None) -> None:
        """Raise if anything at the head of the queue is unacceptably late.

        Only for tests and diagnostics -- the normal path handles lateness by
        logging and dropping, because raising inside the control loop would
        leave the field in whatever state it happened to be in.
        """
        now = self.clock.now() if now_s is None else now_s
        if self._heap:
            head = self._heap[0]
            lateness = now - head.dispatch_at_s
            if lateness > self.cfg.drop_if_late_by_ms / 1000.0:
                raise LateCommandError(
                    f"command {head.command_id} is {lateness * 1000:.1f} ms late"
                )

    def pending_count(self) -> int:
        return sum(1 for c in self._heap if c.outcome is CommandOutcome.PENDING)

    def flush(self) -> list[FieldCommand]:
        """Discard everything queued. Returns the discarded commands."""
        discarded = list(self._heap)
        for command in discarded:
            if command.outcome is CommandOutcome.PENDING:
                command.outcome = CommandOutcome.SUPERSEDED
                command.failure_reason = "flushed at shutdown"
                self.n_superseded += 1
        self._heap.clear()
        return discarded

    # ------------------------------------------------------------ reporting

    @property
    def history(self) -> list[FieldCommand]:
        return list(self._history)

    def timing_percentiles(self) -> dict[str, float | None]:
        """P50/P95/P99 of dispatch timing error, in milliseconds."""
        if not self.timing_errors_s:
            return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
        ordered = sorted(self.timing_errors_s)
        n = len(ordered)

        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return ordered[idx] * 1000.0

        return {
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": ordered[-1] * 1000.0,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "armed": self._armed,
            "current_state": str(self._current_state),
            "queued": len(self._heap),
            "pending": self.pending_count(),
            "n_dispatched": self.n_dispatched,
            "n_late": self.n_late,
            "n_dropped_late": self.n_dropped_late,
            "n_superseded": self.n_superseded,
            "n_failed": self.n_failed,
            "timing": self.timing_percentiles(),
        }
