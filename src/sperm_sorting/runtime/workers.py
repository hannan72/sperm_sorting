"""Threaded worker topology for live capture.

For replay and tests, :class:`~sperm_sorting.runtime.pipeline.Pipeline` runs
synchronously and deterministically. Live capture cannot: the camera delivers
at a fixed rate whether or not inference has finished, and a driver whose
buffers are not drained promptly reports overruns and drops frames inside the
SDK, where the pipeline cannot see or count them.

So acquisition gets its own thread whose only job is to drain the camera into a
bounded queue, and the analysis stages run in another. The queue makes the
back-pressure explicit and countable: under overload the configured policy
either blocks the producer or drops the oldest frame, and either way the drop
appears in the metrics rather than vanishing into the driver.

Threads rather than asyncio: every stage here is CPU-bound or blocks in a C
extension (pypylon, OpenCV, torch), none of which cooperate with an event
loop. Threads also let the GIL be released inside those extensions, which is
where the real parallelism comes from.

Why not one thread per named stage: sixteen queues between sixteen threads
would add sixteen hand-offs of a 2.3 MB frame and a great deal of lock
traffic, to parallelise stages that are individually sub-millisecond. The
split that pays for itself is the one between "must never block" (acquisition)
and "everything else". Finer decomposition is a measurement away, and the
stage timings needed to justify it are already collected.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..acquisition.base import FrameSource
from ..config import AppConfig
from ..errors import CameraError
from ..schemas.frame import FramePacket
from .pipeline import FrameResult, Pipeline
from .queues import BoundedQueue

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerStatus:
    """Liveness of one worker thread."""

    name: str
    started: bool = False
    finished: bool = False
    error: BaseException | None = None
    iterations: int = 0
    last_activity_s: float = field(default_factory=time.monotonic)


class Worker(threading.Thread):
    """A cancellable loop with a status record.

    Cancellation is cooperative via an :class:`threading.Event`. There is no
    forcible kill: a thread killed inside a C extension leaves the SDK's
    internal state undefined, and this process owns a magnet.
    """

    def __init__(
        self,
        name: str,
        target: Callable[[], bool],
        stop_event: threading.Event,
        *,
        poll_interval_s: float = 0.001,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._target = target
        self._stop = stop_event
        self._poll = poll_interval_s
        self.status = WorkerStatus(name=name)

    def run(self) -> None:
        self.status.started = True
        logger.debug("worker %s started", self.name)
        try:
            while not self._stop.is_set():
                keep_going = self._target()
                self.status.iterations += 1
                self.status.last_activity_s = time.monotonic()
                if not keep_going:
                    break
        except BaseException as exc:  # recorded, then the run is stopped
            self.status.error = exc
            logger.exception("worker %s failed", self.name)
            # One worker dying must stop the rest: a pipeline running with a
            # missing stage would keep actuating on stale analysis.
            self._stop.set()
        finally:
            self.status.finished = True
            logger.debug(
                "worker %s finished after %d iterations",
                self.name,
                self.status.iterations,
            )


class PipelineRunner:
    """Drives a :class:`Pipeline` from a :class:`FrameSource`.

    Two modes:

    * :meth:`run_synchronous` -- one thread, source and pipeline in lockstep.
      Deterministic; used for replay, synthetic runs and every test.
    * :meth:`run_threaded` -- acquisition decoupled from analysis by a bounded
      queue. Used for live capture.

    Both call the identical :meth:`Pipeline.process_frame`, so the two modes
    cannot drift apart in behaviour, only in timing.
    """

    def __init__(
        self,
        cfg: AppConfig,
        source: FrameSource,
        pipeline: Pipeline,
        *,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.source = source
        self.pipeline = pipeline
        self.on_frame = on_frame
        self._stop = threading.Event()
        self._workers: list[Worker] = []
        self.frame_queue: BoundedQueue[FramePacket] = BoundedQueue(
            "frames", cfg.runtime.queue_size, cfg.runtime.backpressure
        )
        self.n_frames = 0
        self.error: BaseException | None = None

    # ------------------------------------------------------------- control

    def stop(self) -> None:
        """Request a graceful stop. Safe from a signal handler."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def _should_continue(self) -> bool:
        if self._stop.is_set():
            return False
        limit = self.cfg.run.max_frames
        if limit is not None and self.n_frames >= limit:
            return False
        duration = self.cfg.run.max_duration_s
        return not (duration is not None and self.pipeline.metrics.elapsed_s >= duration)

    # --------------------------------------------------------- synchronous

    def run_synchronous(self) -> int:
        """Read and process in lockstep. Returns the frame count."""
        logger.info("running synchronously (%s)", self.cfg.run.mode)
        try:
            while self._should_continue():
                packet = self.source.read()
                if packet is None:
                    logger.info("source exhausted after %d frames", self.n_frames)
                    break
                result = self.pipeline.process_frame(packet)
                self.n_frames += 1
                if self.on_frame is not None:
                    self.on_frame(result)
        except BaseException as exc:
            self.error = exc
            raise
        return self.n_frames

    # ------------------------------------------------------------ threaded

    def _acquire_once(self) -> bool:
        """One acquisition iteration. Returns False to end the loop."""
        if not self._should_continue():
            return False
        try:
            packet = self.source.read()
        except CameraError as exc:
            # A camera fault must stop the run: continuing would leave the
            # field acting on analysis of frames that no longer arrive.
            logger.error("acquisition failed: %s", exc)
            self.pipeline.health.report(
                "acquisition",
                self.pipeline.health.state.__class__.FAILED,
                f"camera error: {exc}",
            )
            return False
        if packet is None:
            return False
        if not self.frame_queue.put(packet, timeout=1.0):
            self.pipeline.metrics.frames_dropped_backpressure += 1
        return True

    def _process_once(self) -> bool:
        """One analysis iteration. Returns False to end the loop."""
        packet = self.frame_queue.get(timeout=self.cfg.runtime.poll_interval_s)
        if packet is None:
            # Idle. Keep the time-driven machinery alive so a shot can still
            # time out and the watchdog still sees progress.
            if self._stop.is_set() and self.frame_queue.qsize() == 0:
                return False
            self.pipeline.health.check_frame_liveness()
            return True
        result = self.pipeline.process_frame(packet)
        self.n_frames += 1
        self.pipeline.metrics.on_queue_depth(
            "frames", self.frame_queue.qsize(), self.frame_queue.capacity
        )
        self.pipeline.health.check_queue(
            "frames", self.frame_queue.qsize(), self.frame_queue.capacity
        )
        if self.on_frame is not None:
            self.on_frame(result)
        return True

    def run_threaded(self) -> int:
        """Acquisition in its own thread, analysis in another."""
        logger.info(
            "running threaded (queue=%d, backpressure=%s)",
            self.cfg.runtime.queue_size,
            self.cfg.runtime.backpressure,
        )
        acquirer = Worker("acquisition", self._acquire_once, self._stop)
        processor = Worker("analysis", self._process_once, self._stop)
        self._workers = [acquirer, processor]

        acquirer.start()
        processor.start()

        try:
            # The acquisition thread finishing means the source is done; then
            # the analysis thread drains what is left and stops.
            acquirer.join()
            self.frame_queue.close()
            self._stop.set()
            processor.join(timeout=self.cfg.runtime.shutdown_grace_s)
            if processor.is_alive():
                logger.warning(
                    "analysis worker did not stop within %.1f s; %d frames "
                    "remain queued and will be discarded",
                    self.cfg.runtime.shutdown_grace_s,
                    self.frame_queue.qsize(),
                )
        finally:
            self._stop.set()

        for worker in self._workers:
            if worker.status.error is not None:
                self.error = worker.status.error
                raise worker.status.error

        return self.n_frames

    def run(self) -> int:
        """Pick the topology from the configuration."""
        if self.cfg.run.mode == "live":
            return self.run_threaded()
        return self.run_synchronous()

    # ------------------------------------------------------------ reporting

    def worker_status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": w.status.name,
                "started": w.status.started,
                "finished": w.status.finished,
                "alive": w.is_alive(),
                "iterations": w.status.iterations,
                "error": repr(w.status.error) if w.status.error else None,
            }
            for w in self._workers
        ]

    def queue_stats(self) -> dict[str, Any]:
        return {"frames": self.frame_queue.stats.to_json_dict()}
