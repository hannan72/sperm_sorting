"""Mock actuator.

Records every state change instead of driving hardware. This is what the
integration tests assert against: they check the *sequence* of field states,
which is the only externally-visible behaviour the product has.

It can also be told to fail on demand, which is how the fault-handling paths
(rejected write, acknowledgement mismatch) get exercised without a rig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import ActuationConfig
from ..schemas.enums import FieldCommandKind
from ..scheduling.clock import Clock, MonotonicClock
from .base import MagneticActuator

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class StateChange:
    """One recorded transition."""

    time_s: float
    kind: FieldCommandKind
    index: int


class MockActuator(MagneticActuator):
    """In-memory actuator with a full transition history."""

    name = "mock"

    def __init__(
        self,
        cfg: ActuationConfig,
        clock: Clock | None = None,
        *,
        fail_writes: bool = False,
        report_wrong_state: bool = False,
        can_acknowledge: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.clock = clock or MonotonicClock()
        self.history: list[StateChange] = []
        #: Every write fails while this is set. Simulates a dead driver.
        self.fail_writes = fail_writes
        #: Acknowledgement reports the opposite state. Simulates a stuck relay
        #: that accepts commands but does not actually switch.
        self.report_wrong_state = report_wrong_state
        #: When false, the actuator cannot report its state at all, which is
        #: the realistic case for a bare GPIO line with no sense feedback.
        self.can_acknowledge = can_acknowledge

    def open(self) -> None:
        self._open = True
        self._write(FieldCommandKind.FIELD_OFF)
        self._state = FieldCommandKind.FIELD_OFF
        logger.info("mock actuator opened in the safe state")

    def close(self) -> None:
        if self._open:
            self.safe_state()
            self._open = False
            logger.info("mock actuator closed in the safe state")

    def _write(self, kind: FieldCommandKind) -> bool:
        if self.fail_writes:
            return False
        self.history.append(
            StateChange(time_s=self.clock.now(), kind=kind, index=len(self.history))
        )
        return True

    def _read_acknowledgement(self) -> FieldCommandKind | None:
        if not self.can_acknowledge:
            return None
        if self.report_wrong_state:
            return (
                FieldCommandKind.FIELD_OFF
                if self._state is FieldCommandKind.FIELD_ON
                else FieldCommandKind.FIELD_ON
            )
        return self._state

    # ------------------------------------------------------------ test aids

    def state_sequence(self, *, collapse_repeats: bool = False) -> list[str]:
        """The recorded states as plain strings, for assertions.

        ``collapse_repeats`` removes consecutive duplicates, which is usually
        what a test means by "the sequence of commands": re-asserting FIELD_OFF
        when the field is already off is a no-op physically.
        """
        seq = [str(change.kind) for change in self.history]
        if not collapse_repeats:
            return seq
        collapsed: list[str] = []
        for item in seq:
            if not collapsed or collapsed[-1] != item:
                collapsed.append(item)
        return collapsed

    def time_in_state(self, kind: FieldCommandKind, until_s: float | None = None) -> float:
        """Total seconds spent in ``kind``, for duty-cycle checks."""
        end = self.clock.now() if until_s is None else until_s
        total = 0.0
        for i, change in enumerate(self.history):
            stop = self.history[i + 1].time_s if i + 1 < len(self.history) else end
            if change.kind is kind:
                total += max(0.0, stop - change.time_s)
        return total

    def reset(self) -> None:
        self.history.clear()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["n_transitions"] = len(self.history)
        return info
