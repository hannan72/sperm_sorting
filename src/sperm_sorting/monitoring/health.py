"""Health monitoring.

Watches for the conditions that mean the pipeline is no longer doing its job,
and classifies each as degraded (keep running, tell someone) or failed (stop,
safe state). The distinction matters because this device runs unattended:
stopping on every transient would make it useless, and continuing through a
real fault would make it dangerous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import MonitoringConfig
from ..scheduling.clock import Clock, MonotonicClock

logger = logging.getLogger(__name__)


class HealthState(str, Enum):
    HEALTHY = "healthy"
    #: Working, but outside its intended operating envelope.
    DEGRADED = "degraded"
    #: Not working; the caller must go to the safe state.
    FAILED = "failed"


@dataclass(slots=True)
class HealthIssue:
    """One named problem, with enough context to act on it."""

    source: str
    state: HealthState
    message: str
    first_seen_s: float
    last_seen_s: float
    count: int = 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "state": self.state.value,
            "message": self.message,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
            "count": self.count,
        }


class HealthMonitor:
    """Tracks liveness and operating-envelope conditions.

    Issues are keyed by source so that a condition recurring every frame
    produces one issue with a rising count, not a million log lines.
    """

    def __init__(self, cfg: MonitoringConfig, clock: Clock | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or MonotonicClock()
        self._issues: dict[str, HealthIssue] = {}
        self._last_frame_s: float | None = None
        self._started_s = self.clock.now()

    # ------------------------------------------------------------- liveness

    def on_frame(self) -> None:
        self._last_frame_s = self.clock.now()

    def check_frame_liveness(self) -> HealthIssue | None:
        """Fail when the source has gone quiet.

        Before the first frame arrives the grace period runs from start-up, so
        a slow camera open does not immediately look like a dead camera.
        """
        now = self.clock.now()
        reference = self._last_frame_s if self._last_frame_s is not None else self._started_s
        silence = now - reference
        if silence > self.cfg.frame_timeout_s:
            what = "since start-up" if self._last_frame_s is None else "since the last frame"
            return self.report(
                "acquisition",
                HealthState.FAILED,
                f"no frame for {silence:.2f} s {what} "
                f"(limit {self.cfg.frame_timeout_s:.2f} s)",
            )
        return None

    def check_queue(self, name: str, depth: int, capacity: int) -> HealthIssue | None:
        """Warn when a queue stays near capacity.

        A persistently full queue means a downstream stage cannot keep up, so
        frames are about to be dropped or the producer is about to block.
        """
        if capacity <= 0:
            return None
        occupancy = depth / capacity
        if occupancy >= self.cfg.queue_high_water:
            return self.report(
                f"queue:{name}",
                HealthState.DEGRADED,
                f"queue '{name}' is {occupancy:.0%} full ({depth}/{capacity}); "
                "a downstream stage is not keeping up",
            )
        self.clear(f"queue:{name}")
        return None

    def check_drop_rate(self, drop_rate: float, threshold: float = 0.05) -> HealthIssue | None:
        if drop_rate > threshold:
            return self.report(
                "frame_drops",
                HealthState.DEGRADED,
                f"dropping {drop_rate:.1%} of frames (limit {threshold:.1%}); "
                "tracks will fragment and velocities will be unreliable",
            )
        self.clear("frame_drops")
        return None

    # --------------------------------------------------------------- issues

    def report(self, source: str, state: HealthState, message: str) -> HealthIssue:
        now = self.clock.now()
        existing = self._issues.get(source)
        if existing is not None and existing.state is state:
            existing.last_seen_s = now
            existing.count += 1
            existing.message = message
            return existing
        issue = HealthIssue(
            source=source,
            state=state,
            message=message,
            first_seen_s=now,
            last_seen_s=now,
        )
        self._issues[source] = issue
        log = logger.error if state is HealthState.FAILED else logger.warning
        log("health [%s] %s: %s", state.value, source, message)
        return issue

    def clear(self, source: str) -> None:
        if self._issues.pop(source, None) is not None:
            logger.info("health issue cleared: %s", source)

    @property
    def state(self) -> HealthState:
        """Worst current state across all issues."""
        if any(i.state is HealthState.FAILED for i in self._issues.values()):
            return HealthState.FAILED
        if any(i.state is HealthState.DEGRADED for i in self._issues.values()):
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    @property
    def issues(self) -> list[HealthIssue]:
        return list(self._issues.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "issues": [i.to_json_dict() for i in self._issues.values()],
        }
