"""Pipeline orchestration.

The pipeline is synchronous and deterministic; the worker topology wraps it
for live capture, where acquisition must never be blocked by inference.
"""

from __future__ import annotations

from .pipeline import FrameResult, Pipeline
from .queues import BoundedQueue, QueueStats
from .workers import PipelineRunner, Worker, WorkerStatus

__all__ = [
    "BoundedQueue",
    "FrameResult",
    "Pipeline",
    "PipelineRunner",
    "QueueStats",
    "Worker",
    "WorkerStatus",
]
