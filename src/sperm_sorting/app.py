"""Application assembly.

Builds every component from one :class:`AppConfig`, wires them together, runs
the pipeline, and tears everything down safely -- including on the paths where
something has gone wrong, which is where teardown matters most.

The startup order is deliberate:

1. seed and logging, so everything after is reproducible and observable;
2. the audit log, so failures during the rest of startup are recorded;
3. the actuator, driven immediately to FIELD_OFF -- before anything can
   possibly decide otherwise;
4. models and the frame source, the parts most likely to fail;
5. the scheduler is *armed last*, and only if the kit timing is calibrated.

Shutdown reverses it, and the field goes off first.
"""

from __future__ import annotations

import logging
import random
import signal
import sys
from types import FrameType
from typing import Any

import numpy as np

from .acquisition.factory import build_frame_source
from .actuation.base import MagneticActuator, Watchdog
from .actuation.factory import build_actuator
from .backends.runtime_backend import describe_environment
from .config import AppConfig
from .decision.engine import Decision
from .detection.factory import build_detector
from .errors import CalibrationError
from .monitoring.audit import AuditLogger
from .monitoring.health import HealthMonitor
from .monitoring.logging import setup_logging
from .monitoring.metrics import RuntimeMetrics
from .morphology.factory import build_morphology_engine
from .runtime.pipeline import FrameResult, Pipeline
from .runtime.workers import PipelineRunner
from .schemas.command import FieldCommand
from .schemas.enums import CommandOrigin, FieldCommandKind
from .scheduling.clock import Clock, ManualClock, MonotonicClock
from .scheduling.scheduler import ActuationScheduler
from .shots.feasibility import assess_feasibility
from .tracking.factory import build_tracker

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline can reach.

    Determinism is not a nicety here: the replay test asserts that the same
    recording produces byte-identical decisions, and that guarantee is what
    makes an audit log worth keeping.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class Application:
    """One configured run, from construction to teardown."""

    def __init__(self, cfg: AppConfig, clock: Clock | None = None) -> None:
        self.cfg = cfg
        self.clock = clock or (
            ManualClock() if cfg.run.mode == "replay" and cfg.run.deterministic
            else MonotonicClock()
        )
        self.audit: AuditLogger | None = None
        self.actuator: MagneticActuator | None = None
        self.watchdog: Watchdog | None = None
        self.pipeline: Pipeline | None = None
        self.runner: PipelineRunner | None = None
        self.source: Any = None
        self.decisions: list[Decision] = []
        self.commands: list[FieldCommand] = []
        self._closed = False
        self._interrupted = False

    # ------------------------------------------------------------- startup

    def setup(self) -> None:
        cfg = self.cfg

        setup_logging(cfg.monitoring.log_level, json_logs=cfg.monitoring.json_logs)
        seed_everything(cfg.run.seed, cfg.run.deterministic)
        logger.info(
            "starting run %r in %s mode (seed %d)",
            cfg.run.name,
            cfg.run.mode,
            cfg.run.seed,
        )

        self.audit = AuditLogger(
            cfg.monitoring.audit_dir,
            run_name=cfg.run.name,
            audit_tracks=cfg.monitoring.audit_tracks,
            audit_track_points=cfg.monitoring.audit_track_points,
        )

        feasibility = assess_feasibility(cfg)
        for warning in feasibility.warnings:
            logger.warning("feasibility: %s", warning)

        self.audit.write_manifest(
            cfg.summary(),
            environment=describe_environment(),
            feasibility=feasibility.to_json_dict(),
            full_config=cfg.model_dump(mode="json"),
        )

        # The actuator comes up first and is immediately in the safe state, so
        # there is no window in which the magnet is energised by whatever the
        # driver happened to be doing before this process started.
        self.actuator = build_actuator(cfg.actuation, clock=self.clock)
        self.actuator.open()
        self.watchdog = Watchdog(
            self.actuator, cfg.scheduling.watchdog_timeout_ms, clock=self.clock
        )

        scheduler = ActuationScheduler(
            cfg.scheduling, clock=self.clock, dispatch=self._dispatch
        )

        detector = build_detector(cfg.detection)
        tracker = build_tracker(cfg.tracking)
        morphology = build_morphology_engine(cfg.morphology)

        self.source = build_frame_source(cfg.acquisition)
        self.source.open()

        metrics = RuntimeMetrics()
        health = HealthMonitor(cfg.monitoring, clock=self.clock)

        self.pipeline = Pipeline(
            cfg,
            detector=detector,
            tracker=tracker,
            morphology=morphology,
            scheduler=scheduler,
            clock=self.clock,
            audit=self.audit,
            metrics=metrics,
            health=health,
        )

        # Arming last, and only when the kit timing has been measured. An
        # uncalibrated transport delay would apply the field to the wrong
        # fluid -- a failure that raises nothing and looks like success.
        try:
            scheduler.arm()
        except CalibrationError as exc:
            if cfg.actuation.kind == "mock":
                logger.warning(
                    "scheduler not armed (%s). The mock actuator will record "
                    "commands but nothing will be driven.",
                    exc,
                )
            else:
                self.close()
                raise

        self.runner = PipelineRunner(
            cfg, self.source, self.pipeline, on_frame=self._on_frame
        )

    def _dispatch(self, command: FieldCommand) -> bool:
        if self.actuator is None:
            return False
        return self.actuator.apply(command)

    def _on_frame(self, result: FrameResult) -> None:
        if self.watchdog is not None:
            self.watchdog.feed()
        self.decisions.extend(result.decisions)
        self.commands.extend(result.commands)

    # ----------------------------------------------------------------- run

    def install_signal_handlers(self) -> None:
        """Turn Ctrl-C into a graceful stop rather than a stack trace.

        A hard interrupt would skip teardown and could leave the field
        energised, so the handler asks the runner to stop and lets the normal
        shutdown path drive the magnet off.
        """

        def handler(signum: int, frame: FrameType | None) -> None:
            if self._interrupted:
                logger.error("second interrupt; exiting immediately")
                if self.actuator is not None:
                    self.actuator.safe_state()
                sys.exit(130)
            self._interrupted = True
            logger.warning(
                "interrupted; stopping gracefully (interrupt again to force)"
            )
            if self.runner is not None:
                self.runner.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or the platform lacks the signal.
                logger.debug("could not install a handler for %s", sig)

    def run(self) -> int:
        if self.runner is None or self.pipeline is None:
            raise RuntimeError("setup() must be called before run()")
        try:
            frames = self.runner.run()
        finally:
            # Flush regardless: a partial shot that has already been counted
            # still deserves a decision and an audit record.
            if self.pipeline is not None:
                self.decisions.extend(self.pipeline.flush())
        return frames

    # -------------------------------------------------------------- report

    def summary(self) -> dict[str, Any]:
        if self.pipeline is None:
            return {}
        summary = self.pipeline.summary()
        summary["watchdog"] = self.watchdog.stats() if self.watchdog else None
        summary["actuator"] = self.actuator.describe() if self.actuator else None
        if self.runner is not None:
            summary["queues"] = self.runner.queue_stats()
            summary["workers"] = self.runner.worker_status()
        return summary

    def format_report(self) -> str:
        if self.pipeline is None:
            return "pipeline was never constructed"
        s = self.summary()
        lines = [
            "",
            "=" * 66,
            f"Run: {self.cfg.run.name}   mode={self.cfg.run.mode}   "
            f"seed={self.cfg.run.seed}",
            "=" * 66,
            self.pipeline.metrics.format_summary(),
            "",
            "  shots:",
            f"    total          : {s['shots']}",
            f"    ACCEPT         : {s['accept']}   (field off, to collection)",
            f"    REJECT         : {s['reject']}   (field on, diverted)",
            f"    INDETERMINATE  : {s['indeterminate']}   (too few sperm)",
            f"    mean trackable : {s['mean_trackable_per_shot']:.1f} per shot",
            "",
            "  actuation:",
            f"    dispatched     : {s['commands_dispatched']}",
            f"    late           : {s['commands_late']}",
            f"    dropped (late) : {s['commands_dropped_late']}",
        ]
        if self.audit is not None and self.audit.run_dir is not None:
            lines += ["", f"  audit log: {self.audit.run_dir}"]
        lines.append("=" * 66)
        return "\n".join(lines)

    # ------------------------------------------------------------ shutdown

    def close(self) -> None:
        """Tear down safely. Idempotent, and never raises."""
        if self._closed:
            return
        self._closed = True

        # The field goes off first, before anything else can fail.
        if self.actuator is not None:
            try:
                self.actuator.close()
            except Exception:
                logger.exception("actuator shutdown failed")

        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                logger.exception("frame source shutdown failed")

        if self.pipeline is not None:
            try:
                self.pipeline.detector.close()
            except Exception:
                logger.exception("detector shutdown failed")

        if self.audit is not None:
            try:
                self.audit.close(summary=self.summary())
            except Exception:
                logger.exception("audit shutdown failed")

    def __enter__(self) -> Application:
        self.setup()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def run_app(cfg: AppConfig, clock: Clock | None = None) -> Application:
    """Build, run and tear down one application. Returns it for inspection."""
    app = Application(cfg, clock=clock)
    try:
        app.setup()
        app.install_signal_handlers()
        app.run()
    finally:
        app.close()
    return app


__all__ = ["Application", "run_app", "seed_everything"]
