"""Bounded inter-stage queues.

Every queue in the pipeline is bounded. An unbounded queue does not remove
back-pressure, it converts it into unbounded memory growth followed by a
process death several hours into a run -- which is the worst possible time to
discover that a stage was too slow.

When a queue fills, the policy decides who suffers:

* ``block`` -- the producer waits. Correct for replay, where nothing may be
  lost and determinism matters more than latency.
* ``drop_oldest`` -- the oldest item is discarded. Correct for live capture,
  where a stale frame has no value: by the time it is processed, the fluid it
  shows has already passed the magnet.

The drop is always *counted* and surfaced, never silent.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class QueueStats:
    """Counters for one queue, reported to the metrics layer."""

    name: str
    capacity: int
    put_count: int = 0
    get_count: int = 0
    dropped_count: int = 0
    blocked_count: int = 0
    high_water: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "put_count": self.put_count,
            "get_count": self.get_count,
            "dropped_count": self.dropped_count,
            "blocked_count": self.blocked_count,
            "high_water": self.high_water,
        }


class BoundedQueue(Generic[T]):
    """A bounded queue with an explicit overflow policy.

    Wraps :class:`queue.Queue` rather than reimplementing it: the standard
    library's condition-variable handling is correct and well tested, and the
    only thing missing is a drop-oldest policy with accounting.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        policy: str = "drop_oldest",
    ) -> None:
        if capacity < 1:
            raise ValueError(f"queue capacity must be at least 1, got {capacity}")
        if policy not in ("block", "drop_oldest"):
            raise ValueError(f"unknown queue policy: {policy!r}")
        self.name = name
        self.capacity = capacity
        self.policy = policy
        self._q: queue.Queue[T] = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self.stats = QueueStats(name=name, capacity=capacity)
        self._closed = threading.Event()

    # ------------------------------------------------------------------ put

    def put(self, item: T, timeout: float | None = None) -> bool:
        """Enqueue. Returns ``False`` if the item could not be accepted.

        Under ``drop_oldest`` this returns ``True`` even when something else
        was discarded to make room -- the *new* item was accepted, which is
        what the caller asked about. Check :attr:`stats` for drops.
        """
        if self._closed.is_set():
            return False

        if self.policy == "block":
            try:
                self._q.put(item, block=True, timeout=timeout)
            except queue.Full:
                with self._lock:
                    self.stats.blocked_count += 1
                return False
            with self._lock:
                self.stats.put_count += 1
                self.stats.high_water = max(self.stats.high_water, self._q.qsize())
            return True

        # drop_oldest
        while True:
            try:
                self._q.put_nowait(item)
                with self._lock:
                    self.stats.put_count += 1
                    self.stats.high_water = max(self.stats.high_water, self._q.qsize())
                return True
            except queue.Full:
                try:
                    self._q.get_nowait()
                    with self._lock:
                        self.stats.dropped_count += 1
                except queue.Empty:
                    # A consumer drained it between the two calls; retry.
                    continue

    # ------------------------------------------------------------------ get

    def get(self, timeout: float | None = None) -> T | None:
        """Dequeue, or return ``None`` on timeout / when closed and drained."""
        try:
            item = self._q.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            self.stats.get_count += 1
        return item

    def get_nowait(self) -> T | None:
        try:
            item = self._q.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self.stats.get_count += 1
        return item

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        self._closed.set()

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def occupancy(self) -> float:
        return self._q.qsize() / self.capacity

    def drain(self) -> list[T]:
        """Remove and return everything currently queued."""
        items: list[T] = []
        while True:
            item = self.get_nowait()
            if item is None:
                break
            items.append(item)
        return items

    def __len__(self) -> int:
        return self._q.qsize()
