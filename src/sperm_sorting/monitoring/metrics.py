"""Runtime metrics.

Measures what the device is actually doing, as opposed to what it was asked to
do: achieved frame rate, how many frames were dropped and where, per-stage
latency distributions, and queue occupancy.

Latency is reported as percentiles, never as a mean. A mean latency hides
exactly the tail that breaks a real-time system -- a pipeline that meets its
deadline 99% of the time and misses catastrophically 1% of the time has a fine
mean and is unusable.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LatencyTracker:
    """Bounded-history latency statistics for one pipeline stage.

    The window is bounded so that a run of arbitrary length uses constant
    memory. That does mean the percentiles are over the recent past rather
    than the whole run, which is what you want for a live health display; the
    all-time count, min and max are tracked separately and exactly.
    """

    name: str
    window: int = 2048
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    count: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0

    def __post_init__(self) -> None:
        if self.samples.maxlen != self.window:
            self.samples = deque(maxlen=self.window)

    def record(self, seconds: float) -> None:
        self.samples.append(seconds)
        self.count += 1
        self.total_s += seconds
        self.min_s = min(self.min_s, seconds)
        self.max_s = max(self.max_s, seconds)

    def percentile(self, p: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return ordered[idx]

    def snapshot(self) -> dict[str, Any]:
        def ms(value: float | None) -> float | None:
            return None if value is None else value * 1000.0

        return {
            "count": self.count,
            "mean_ms": ms(self.total_s / self.count) if self.count else None,
            "p50_ms": ms(self.percentile(0.50)),
            "p95_ms": ms(self.percentile(0.95)),
            "p99_ms": ms(self.percentile(0.99)),
            "min_ms": ms(self.min_s) if self.count else None,
            "max_ms": ms(self.max_s) if self.count else None,
        }


class StageTimer:
    """Context manager that records into a :class:`LatencyTracker`."""

    __slots__ = ("_start", "_tracker")

    def __init__(self, tracker: LatencyTracker) -> None:
        self._tracker = tracker
        self._start = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._tracker.record(time.perf_counter() - self._start)


class RuntimeMetrics:
    """Aggregate metrics for one run."""

    def __init__(self, window: int = 2048) -> None:
        self._start = time.monotonic()
        self.window = window
        self.stages: dict[str, LatencyTracker] = {}

        self.frames_acquired = 0
        self.frames_processed = 0
        self.frames_dropped_source = 0
        self.frames_dropped_quality = 0
        self.frames_dropped_backpressure = 0

        self.detections_total = 0
        self.tracks_created = 0
        self.tracks_gated = 0
        self.crops_extracted = 0
        self.morphology_completed = 0
        self.morphology_failed = 0
        self.morphology_deadline_missed = 0

        #: Recent queue depths, keyed by queue name.
        self.queue_depths: dict[str, deque[int]] = {}
        self.queue_high_water: dict[str, int] = {}

        #: Timestamps of recent acquisitions, for an instantaneous FPS.
        self._recent_frames: deque[float] = deque(maxlen=256)

    # ---------------------------------------------------------------- stages

    def stage(self, name: str) -> LatencyTracker:
        tracker = self.stages.get(name)
        if tracker is None:
            tracker = LatencyTracker(name=name, window=self.window)
            self.stages[name] = tracker
        return tracker

    def time_stage(self, name: str) -> StageTimer:
        return StageTimer(self.stage(name))

    # ---------------------------------------------------------------- frames

    def on_frame_acquired(self, dropped_before: int = 0) -> None:
        self.frames_acquired += 1
        self.frames_dropped_source += dropped_before
        self._recent_frames.append(time.monotonic())

    def on_frame_processed(self) -> None:
        self.frames_processed += 1

    def on_queue_depth(self, name: str, depth: int, capacity: int) -> None:
        history = self.queue_depths.get(name)
        if history is None:
            history = deque(maxlen=256)
            self.queue_depths[name] = history
        history.append(depth)
        self.queue_high_water[name] = max(self.queue_high_water.get(name, 0), depth)

    # ------------------------------------------------------------ derived

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    @property
    def acquisition_fps(self) -> float:
        """Instantaneous rate over the recent window, not the run average."""
        if len(self._recent_frames) < 2:
            return 0.0
        span = self._recent_frames[-1] - self._recent_frames[0]
        if span <= 0:
            return 0.0
        return (len(self._recent_frames) - 1) / span

    @property
    def processed_fps(self) -> float:
        elapsed = self.elapsed_s
        return self.frames_processed / elapsed if elapsed > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        """Fraction of frames offered that never completed the pipeline."""
        offered = self.frames_acquired + self.frames_dropped_source
        return 0.0 if offered == 0 else 1.0 - (self.frames_processed / offered)

    def snapshot(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(self.elapsed_s, 3),
            "acquisition_fps": round(self.acquisition_fps, 2),
            "processed_fps": round(self.processed_fps, 2),
            "frames_acquired": self.frames_acquired,
            "frames_processed": self.frames_processed,
            "frames_dropped_source": self.frames_dropped_source,
            "frames_dropped_quality": self.frames_dropped_quality,
            "frames_dropped_backpressure": self.frames_dropped_backpressure,
            "drop_rate": round(self.drop_rate, 5),
            "detections_total": self.detections_total,
            "tracks_created": self.tracks_created,
            "tracks_gated": self.tracks_gated,
            "crops_extracted": self.crops_extracted,
            "morphology_completed": self.morphology_completed,
            "morphology_failed": self.morphology_failed,
            "morphology_deadline_missed": self.morphology_deadline_missed,
            "queue_high_water": dict(self.queue_high_water),
            "latency": {name: t.snapshot() for name, t in self.stages.items()},
        }

    def format_summary(self) -> str:
        s = self.snapshot()
        lines = [
            f"  elapsed            : {s['elapsed_s']:.2f} s",
            f"  acquisition FPS    : {s['acquisition_fps']:.1f}",
            f"  processed FPS      : {s['processed_fps']:.1f}",
            f"  frames acquired    : {s['frames_acquired']}",
            f"  frames processed   : {s['frames_processed']}",
            f"  drops (source)     : {s['frames_dropped_source']}",
            f"  drops (quality)    : {s['frames_dropped_quality']}",
            f"  drops (backpressure): {s['frames_dropped_backpressure']}",
            f"  drop rate          : {s['drop_rate'] * 100:.2f}%",
            f"  tracks created     : {s['tracks_created']}",
            f"  tracks gated       : {s['tracks_gated']}",
            f"  morphology ok/fail/late: {s['morphology_completed']}/"
            f"{s['morphology_failed']}/{s['morphology_deadline_missed']}",
        ]
        if s["latency"]:
            lines.append("  stage latency (p50 / p95 / p99 ms):")
            for name, stat in sorted(s["latency"].items()):
                if stat["count"]:
                    lines.append(
                        f"    {name:<22s} {stat['p50_ms']:7.2f} / "
                        f"{stat['p95_ms']:7.2f} / {stat['p99_ms']:7.2f}"
                        f"   (n={stat['count']})"
                    )
        return "\n".join(lines)
