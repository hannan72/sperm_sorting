"""Magnetic actuator interface and safety watchdog.

The prototype's entire physical output is one bit: energise the magnet, or
don't. Everything above this layer exists to decide that bit correctly;
everything in this layer exists to make sure the bit is never left in the
wrong state when something goes wrong.

**FIELD_OFF is the safe state.** With the field off, the sample flows to the
collection reservoir unsorted. With it on, the labelled population is diverted
to waste. So a fault that leaves the field off degrades the device to "no
sorting", whereas a fault that leaves it on would silently divert material.
Every failure path here therefore ends in FIELD_OFF.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config import ActuationConfig
from ..errors import ActuatorError, WatchdogTimeout
from ..schemas.command import FieldCommand
from ..schemas.enums import CommandOutcome, FieldCommandKind
from ..scheduling.clock import Clock, MonotonicClock

logger = logging.getLogger(__name__)


class MagneticActuator(ABC):
    """Drives the magnetic field into one of two states."""

    name: str = "actuator"

    def __init__(self, cfg: ActuationConfig) -> None:
        self.cfg = cfg
        self._state = FieldCommandKind.FIELD_OFF
        self._open = False
        self.n_commands = 0
        self.n_ack_failures = 0

    # ---------------------------------------------------------- lifecycle

    @abstractmethod
    def open(self) -> None:
        """Acquire the hardware and drive it to the safe state."""

    @abstractmethod
    def close(self) -> None:
        """Drive to the safe state and release the hardware.

        Must be idempotent and must not raise: it runs on the shutdown path,
        including the path taken after another exception.
        """

    @abstractmethod
    def _write(self, kind: FieldCommandKind) -> bool:
        """Actually change the hardware state. Returns success."""

    def _read_acknowledgement(self) -> FieldCommandKind | None:
        """Read the state the hardware reports.

        ``None`` means the actuator cannot report its state, in which case
        acknowledgement checking is skipped and that fact is logged once at
        open time. An actuator that cannot be verified is a real limitation
        and should be visible in the audit log, not silently assumed good.
        """
        return None

    # ------------------------------------------------------------- command

    @property
    def state(self) -> FieldCommandKind:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._open

    def apply(self, command: FieldCommand) -> bool:
        """Execute a command, verifying the acknowledgement if available."""
        if not self._open:
            raise ActuatorError(f"{self.name} is not open")

        ok = self._write(command.kind)
        self.n_commands += 1
        if not ok:
            command.outcome = CommandOutcome.FAILED
            command.failure_reason = f"{self.name} rejected the write"
            self.safe_state()
            return False

        self._state = command.kind

        if self.cfg.require_acknowledgement:
            reported = self._read_acknowledgement()
            if reported is not None and reported is not command.kind:
                self.n_ack_failures += 1
                command.outcome = CommandOutcome.FAILED
                command.failure_reason = (
                    f"acknowledgement mismatch: commanded {command.kind}, "
                    f"hardware reports {reported}"
                )
                logger.error(
                    "actuator acknowledgement mismatch on command %d; forcing "
                    "safe state",
                    command.command_id,
                )
                self.safe_state()
                return False
            if reported is not None:
                command.outcome = CommandOutcome.ACKNOWLEDGED

        return True

    def safe_state(self) -> None:
        """Force FIELD_OFF. Never raises; this runs on fault paths."""
        try:
            self._write(FieldCommandKind.FIELD_OFF)
            self._state = FieldCommandKind.FIELD_OFF
        except Exception:  # a failing safe path must still not propagate
            logger.exception("%s failed while forcing the safe state", self.name)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": str(self._state),
            "open": self._open,
            "n_commands": self.n_commands,
            "n_ack_failures": self.n_ack_failures,
            "can_acknowledge": self._read_acknowledgement() is not None,
        }

    def __enter__(self) -> MagneticActuator:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Watchdog:
    """Forces FIELD_OFF when the pipeline stops feeding it.

    The failure this guards against is not a crash -- a crash unwinds through
    ``close()`` and the field goes off. It guards against the pipeline
    *hanging*: a stuck inference call, a deadlocked queue, a camera that stops
    delivering. In that state the process is alive, no exception is raised, and
    without a watchdog the field would stay in whatever state the last decision
    left it, applied to fluid nobody is looking at any more.
    """

    def __init__(
        self,
        actuator: MagneticActuator,
        timeout_ms: float,
        clock: Clock | None = None,
    ) -> None:
        self.actuator = actuator
        self.timeout_s = timeout_ms / 1000.0
        self.clock = clock or MonotonicClock()
        self._last_fed = self.clock.now()
        self._tripped = False
        self.n_trips = 0

    def feed(self) -> None:
        """Report that the pipeline is alive. Call every tick."""
        self._last_fed = self.clock.now()
        if self._tripped:
            logger.info("watchdog recovered after %d trip(s)", self.n_trips)
            self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def time_since_fed(self, now_s: float | None = None) -> float:
        now = self.clock.now() if now_s is None else now_s
        return now - self._last_fed

    def check(self, now_s: float | None = None) -> bool:
        """Trip and force FIELD_OFF if the timeout has expired.

        Returns ``True`` when it trips on this call. Deliberately does not
        raise: the whole point is to keep working while the rest of the
        pipeline is not.
        """
        if self.time_since_fed(now_s) <= self.timeout_s:
            return False
        if not self._tripped:
            self._tripped = True
            self.n_trips += 1
            logger.error(
                "watchdog expired after %.1f ms without a feed; forcing FIELD_OFF",
                self.time_since_fed(now_s) * 1000,
            )
            self.actuator.safe_state()
            return True
        return False

    def check_or_raise(self, now_s: float | None = None) -> None:
        """As :meth:`check`, but raises after forcing the safe state.

        For callers that want the run to stop rather than continue degraded.
        """
        if self.check(now_s):
            raise WatchdogTimeout(
                f"watchdog expired after {self.time_since_fed(now_s) * 1000:.1f} ms; "
                "field forced to FIELD_OFF"
            )

    def stats(self) -> dict[str, Any]:
        return {
            "timeout_s": self.timeout_s,
            "tripped": self._tripped,
            "n_trips": self.n_trips,
            "time_since_fed_s": self.time_since_fed(),
        }
