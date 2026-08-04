"""Clocks.

Every timing decision in this system is made on a *monotonic* timeline.
Wall-clock time is unsuitable: NTP steps, daylight-saving transitions and
manual clock changes can move it backwards, and a scheduler that sees time go
backwards will either fire everything at once or stall.

The abstraction exists mainly so that replay and tests can drive time
deterministically. :class:`ManualClock` makes "wait 300 ms" instant and exact,
which is what lets the scheduler tests assert timing to the microsecond
without sleeping.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class Clock(ABC):
    """A monotonic time source."""

    @abstractmethod
    def now(self) -> float:
        """Seconds since an arbitrary epoch. Never decreases."""

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Block for approximately ``seconds``. Negative values return at once."""


class MonotonicClock(Clock):
    """Wraps :func:`time.monotonic`. The production clock."""

    __slots__ = ()

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock(Clock):
    """A clock advanced explicitly. For tests and deterministic replay.

    ``sleep`` advances the clock rather than blocking, so a test can exercise
    a one-second shot timeout in microseconds of real time and get exactly the
    same ordering of events every run.
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += seconds

    def advance(self, seconds: float) -> float:
        """Move the clock forward. Refuses to go backwards."""
        if seconds < 0:
            raise ValueError("a monotonic clock cannot be advanced backwards")
        self._now += seconds
        return self._now

    def set(self, value: float) -> None:
        if value < self._now:
            raise ValueError(
                f"a monotonic clock cannot be set backwards "
                f"({value} < {self._now})"
            )
        self._now = float(value)


class ScaledClock(Clock):
    """Real time, rescaled. Used to replay a recording faster or slower.

    A scale of 2.0 makes time appear to pass twice as fast, so a replay drives
    the pipeline at double rate while every relative interval stays correct.
    """

    __slots__ = ("_base", "_origin", "_scale")

    def __init__(self, scale: float = 1.0, base: Clock | None = None) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._scale = float(scale)
        self._base = base or MonotonicClock()
        self._origin = self._base.now()

    def now(self) -> float:
        return self._origin + (self._base.now() - self._origin) * self._scale

    def sleep(self, seconds: float) -> None:
        self._base.sleep(seconds / self._scale)
