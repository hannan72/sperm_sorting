"""The pipeline.

Implements the mandated processing order exactly, once, so that live capture,
replay and synthetic runs all go through the same code::

    frame
      -> preprocess -> quality gate
      -> detect (bounding boxes)
      -> track (persistent unique IDs)
      -> counting gate -> shot assignment          [denominator committed here]
      -> on track completion:
           motion features (flow-corrected)
           progressive classification
           if progressive:  best frame -> crop -> morphology (4 aspects)
           eligibility  (progressive AND all four normal AND in time)
      -> shot finalisation -> ratio -> decision
      -> scheduled FIELD_ON / FIELD_OFF

Two ordering commitments are worth stating because they are easy to violate
while "optimising":

**Morphology never runs before tracking.** A crop is only meaningful if it can
be bound to the motion measurement of the *same* cell, and that binding does
not exist until the cell has an ID and a trajectory. Running morphology on raw
detections would be cheaper and would measure something else entirely.

**Track quality is assessed once, at the gate.** The denominator is fixed the
moment a track is committed to a shot. Re-assessing later and removing a track
would let the denominator shrink after the numerator was known, which is the
precise manipulation that would make a bad sample look good.

This class is synchronous and deterministic: one call to :meth:`process_frame`
advances everything by one frame. That is what makes replay reproducible and
what lets the tests drive a whole run in milliseconds. The threaded
worker topology in :mod:`.workers` wraps these same stage methods for live
capture, where acquisition must not be blocked by inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..cropping.extractor import CropExtractor
from ..decision.engine import Decision, DecisionEngine
from ..detection.base import Detector
from ..monitoring.audit import AuditLogger
from ..monitoring.health import HealthMonitor
from ..monitoring.metrics import RuntimeMetrics
from ..morphology.inference import BaseMorphologyEngine
from ..motion.classifier import ProgressiveMotilityClassifier, assess_track_quality
from ..motion.flow import FlowEstimator, build_flow_estimator
from ..preprocessing.preprocessor import FramePreprocessor
from ..preprocessing.quality_gate import ImageQualityGate
from ..quality.selector import BestFrameSelector, FrameBuffer
from ..schemas.command import FieldCommand
from ..schemas.detection import Detection
from ..schemas.enums import (
    CommandOrigin,
    FieldCommandKind,
    MorphologyStatus,
    QualityVerdict,
    ShotStatus,
)
from ..schemas.frame import FramePacket
from ..schemas.morphology import MorphologyResult
from ..schemas.shot import ShotRecord
from ..schemas.track import TrackRecord
from ..scheduling.clock import Clock, MonotonicClock
from ..scheduling.scheduler import ActuationScheduler
from ..shots.gate import CountingGate
from ..shots.manager import ShotManager
from ..tracking.base import Tracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameResult:
    """What one frame produced. Returned so callers can observe progress."""

    frame_id: int
    accepted: bool
    n_detections: int = 0
    n_active_tracks: int = 0
    gated_track_ids: list[int] = field(default_factory=list)
    finished_track_ids: list[int] = field(default_factory=list)
    closed_shot_ids: list[int] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    commands: list[FieldCommand] = field(default_factory=list)
    reject_reason: str = ""


class Pipeline:
    """Owns every stage and the state that flows between them."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        detector: Detector,
        tracker: Tracker,
        morphology: BaseMorphologyEngine,
        scheduler: ActuationScheduler,
        clock: Clock | None = None,
        audit: AuditLogger | None = None,
        metrics: RuntimeMetrics | None = None,
        health: HealthMonitor | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.clock = clock or MonotonicClock()
        self.detector = detector
        self.tracker = tracker
        self.morphology = morphology
        self.scheduler = scheduler
        self.audit = audit
        self.metrics = metrics or RuntimeMetrics()
        self.health = health or HealthMonitor(cfg.monitoring, clock=self.clock)

        self.preprocessor = FramePreprocessor(cfg.preprocess)
        self.quality_gate = ImageQualityGate(cfg.quality_gate)
        self.motility = ProgressiveMotilityClassifier(cfg.motion, cfg.calibration.optical)
        self.flow_estimator: FlowEstimator = build_flow_estimator(
            cfg.motion.flow_correction
        )
        self.selector = BestFrameSelector(cfg.best_frame)
        self.cropper = CropExtractor(cfg.crop)
        self.decider = DecisionEngine(cfg.decision)

        roi = cfg.preprocess.roi
        width = roi[2] if roi else (frame_width or cfg.acquisition.synthetic.width)
        height = roi[3] if roi else (frame_height or cfg.acquisition.synthetic.height)
        self.gate = CountingGate(cfg.shots.gate, width, height)
        self.shots = ShotManager(cfg.shots, cfg.morphology.deadline_ms / 1000.0)

        # Frames must outlive the moment they are processed, because best-frame
        # selection looks back over a whole track. Sized to cover the longest
        # plausible track: max_age plus a generous transit.
        buffer_capacity = max(64, cfg.tracking.max_age * 4, 256)
        self.frames = FrameBuffer(capacity=buffer_capacity)
        self.detections_by_frame: dict[int, list[Detection]] = {}

        #: Every track ever seen, so a shot can look its members up at
        #: finalisation even after the tracker has forgotten them.
        self.tracks_by_id: dict[int, TrackRecord] = {}

        self.frame_count = 0
        self.last_frame: FramePacket | None = None

    # ================================================================= frame

    def process_frame(self, packet: FramePacket) -> FrameResult:
        """Advance the whole pipeline by one frame."""
        self.frame_count += 1
        self.metrics.on_frame_acquired(packet.dropped_before)
        self.health.on_frame()

        # --- preprocess -------------------------------------------------
        with self.metrics.time_stage("preprocess"):
            packet = self.preprocessor.process(packet)

        # --- quality gate -----------------------------------------------
        with self.metrics.time_stage("quality_gate"):
            gated = self.quality_gate.apply(packet)
        if gated is None:
            self.metrics.frames_dropped_quality += 1
            reason = packet.quality.reason if packet.quality else "rejected"
            # Time still advances on a rejected frame: a shot must be able to
            # time out during a run of unusable frames, and the watchdog must
            # still see the pipeline as alive.
            result = FrameResult(
                frame_id=packet.frame_id, accepted=False, reject_reason=reason
            )
            self._advance_time_only(packet, result)
            return result
        packet = gated
        self.last_frame = packet

        self.frames.put(packet)

        # --- detect ------------------------------------------------------
        with self.metrics.time_stage("detect"):
            detections = self.detector.detect(packet)
        for det in detections:
            det.capture_time_s = packet.capture_time_s
        self.detections_by_frame[packet.frame_id] = detections
        self.metrics.detections_total += len(detections)

        # --- track -------------------------------------------------------
        with self.metrics.time_stage("track"):
            active = self.tracker.update(detections, packet)
        for track in active:
            if track.track_id not in self.tracks_by_id:
                self.tracks_by_id[track.track_id] = track
                self.metrics.tracks_created += 1

        result = FrameResult(
            frame_id=packet.frame_id,
            accepted=True,
            n_detections=len(detections),
            n_active_tracks=len(active),
        )

        # --- flow estimate ----------------------------------------------
        # Estimated from the current population, before any track is graded,
        # so the same estimate applies to every track on this frame.
        with self.metrics.time_stage("flow"):
            flow = self.flow_estimator.estimate(active, packet)

        # --- counting gate -> shot assignment ----------------------------
        with self.metrics.time_stage("gate"):
            for track in active:
                crossing = self.gate.update(track)
                if crossing is None:
                    continue
                # Track quality is judged here, once, and the verdict is final.
                assess_track_quality(track, self.cfg.track_quality)
                closed = self.shots.add_track(
                    track, crossing.time_s, crossing.frame_id
                )
                result.gated_track_ids.append(track.track_id)
                self.metrics.tracks_gated += 1
                if closed is not None:
                    result.closed_shot_ids.append(closed.shot_id)

        # A shot must also be able to close on time alone.
        closed = self.shots.poll(packet.capture_time_s, packet.frame_id)
        if closed is not None:
            result.closed_shot_ids.append(closed.shot_id)

        # --- finished tracks: motion, morphology, eligibility -------------
        for track in self.tracker.finished_tracks():
            self._finalise_track(track, flow)
            result.finished_track_ids.append(track.track_id)
            self.gate.forget(track.track_id)

        # --- shot finalisation and decision ------------------------------
        for shot in self.shots.ready_shots(packet.capture_time_s, self.tracks_by_id):
            decision, commands = self._decide_shot(shot)
            result.decisions.append(decision)
            result.commands.extend(commands)

        # --- dispatch -----------------------------------------------------
        self.scheduler.poll(packet.capture_time_s)

        self._evict_old_frames(packet.frame_id)
        self.metrics.on_frame_processed()
        return result

    def _advance_time_only(self, packet: FramePacket, result: FrameResult) -> None:
        """Keep time-driven machinery running through an unusable frame."""
        closed = self.shots.poll(packet.capture_time_s, packet.frame_id)
        if closed is not None:
            result.closed_shot_ids.append(closed.shot_id)
        for shot in self.shots.ready_shots(packet.capture_time_s, self.tracks_by_id):
            decision, commands = self._decide_shot(shot)
            result.decisions.append(decision)
            result.commands.extend(commands)
        self.scheduler.poll(packet.capture_time_s)

    # ================================================================ track

    def _finalise_track(
        self, track: TrackRecord, flow: tuple[float, float] | None
    ) -> None:
        """Run the per-sperm analysis on a track that can no longer grow.

        A track that never crossed the gate belongs to no shot and is not
        analysed: it is not in anyone's denominator, so its morphology could
        not change any decision, and the morphology budget is finite.
        """
        self.tracks_by_id.setdefault(track.track_id, track)

        if track.shot_id is None:
            return

        # --- motion ------------------------------------------------------
        with self.metrics.time_stage("motion"):
            self.motility.classify(track, flow_vector=flow)

        # --- morphology, only for progressive sperm -----------------------
        if not track.is_progressive:
            track.morphology = MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.NOT_REQUIRED,
                f"track was graded {track.motion.motility_class if track.motion else 'ungraded'}; "
                "morphology is only evaluated for progressive sperm",
            )
            track.evaluation_complete = True
            self._resolve(track)
            return

        deadline = self.clock.now() + self.cfg.morphology.deadline_ms / 1000.0
        track.evaluation_deadline_s = deadline

        # --- best frame ---------------------------------------------------
        with self.metrics.time_stage("best_frame"):
            candidates = self.selector.select(
                track, self.frames, self.detections_by_frame
            )
        if not candidates:
            track.morphology = MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.NO_VALID_CROP,
                "no frame in this track met the crop-quality bar",
            )
            track.evaluation_complete = True
            self._resolve(track)
            return

        candidate = candidates[0]
        frame = self.frames.get(candidate.frame_id)
        if frame is None:
            # The frame aged out of the buffer before we got here, which means
            # the buffer is too small for the track lengths being produced.
            track.morphology = MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.NO_VALID_CROP,
                f"frame {candidate.frame_id} was evicted from the frame buffer "
                f"(capacity {self.frames.capacity}) before the crop was cut",
            )
            track.evaluation_complete = True
            self.health.report(
                "frame_buffer",
                self.health.state.__class__.DEGRADED,
                "frames are ageing out of the buffer before morphology runs; "
                "increase the buffer capacity or shorten track lifetimes",
            )
            self._resolve(track)
            return

        # --- crop ---------------------------------------------------------
        with self.metrics.time_stage("crop"):
            crop_image, crop_record = self.cropper.extract(
                track,
                candidate,
                frame,
                neighbours=self.detections_by_frame.get(candidate.frame_id),
            )
        track.crop = crop_record
        self.metrics.crops_extracted += 1

        # --- morphology ----------------------------------------------------
        with self.metrics.time_stage("morphology"):
            track.morphology = self.morphology.evaluate_track(
                track, crop_image, deadline
            )

        status = track.morphology.status
        if status is MorphologyStatus.COMPLETE:
            self.metrics.morphology_completed += 1
            track.evaluation_complete = True
        elif status is MorphologyStatus.DEADLINE_MISSED:
            self.metrics.morphology_deadline_missed += 1
            track.evaluation_complete = False
        else:
            self.metrics.morphology_failed += 1
            track.evaluation_complete = False

        self._resolve(track)

    def _resolve(self, track: TrackRecord) -> None:
        """Compute eligibility and tell the shot manager this member is done."""
        track.compute_eligibility()
        self.shots.notify_track_resolved(track.track_id)
        if self.audit is not None:
            self.audit.track(track)

    # ================================================================= shot

    def _decide_shot(self, shot: ShotRecord) -> tuple[Decision, list[FieldCommand]]:
        """Apply the rule and schedule the resulting field commands."""
        with self.metrics.time_stage("decision"):
            decision = self.decider.evaluate(shot)

        commands = self._schedule_for(shot, decision)

        if self.audit is not None:
            self.audit.shot_decided(shot, decision)
            for command in commands:
                self.audit.command_dispatched(command)
        return decision, commands

    def _schedule_for(
        self, shot: ShotRecord, decision: Decision
    ) -> list[FieldCommand]:
        """Queue the field commands that answer one shot.

        A shot describes a *segment* of fluid, not an instant, so a rejected
        shot is energised from the moment its first member reaches the magnet
        until its last member has passed, plus the configured margins. An
        accepted shot needs no command at all when the field is already off --
        re-asserting FIELD_OFF would add pointless actuator traffic -- but one
        is issued whenever the current state is not already safe.
        """
        commands: list[FieldCommand] = []
        first_gate = shot.first_gate_time_s
        last_gate = shot.last_gate_time_s
        if first_gate is None or last_gate is None:
            # A shot with no crossings (possible only on a timeout with zero
            # members) has no fluid segment to act on.
            return commands

        if decision.field_command is FieldCommandKind.FIELD_ON:
            commands.append(
                self.scheduler.submit(
                    FieldCommandKind.FIELD_ON,
                    gate_time_s=first_gate,
                    shot_id=shot.shot_id,
                    origin=CommandOrigin.DECISION,
                )
            )
            # And release it once the segment has passed.
            release_at = last_gate + self.cfg.scheduling.post_activation_margin_ms / 1000.0
            commands.append(
                self.scheduler.submit(
                    FieldCommandKind.FIELD_OFF,
                    gate_time_s=release_at,
                    shot_id=shot.shot_id,
                    origin=CommandOrigin.DECISION,
                )
            )
        elif self.scheduler.current_state is not FieldCommandKind.FIELD_OFF:
            commands.append(
                self.scheduler.submit(
                    FieldCommandKind.FIELD_OFF,
                    gate_time_s=first_gate,
                    shot_id=shot.shot_id,
                    origin=CommandOrigin.DECISION,
                )
            )
        return commands

    # ============================================================ lifecycle

    def _evict_old_frames(self, current_frame_id: int) -> None:
        """Drop per-frame detection lists for frames the buffer has released.

        The frame buffer evicts on its own; this keeps the parallel detection
        dictionary from becoming the unbounded structure instead.
        """
        cutoff = current_frame_id - self.frames.capacity
        if cutoff <= 0:
            return
        stale = [fid for fid in self.detections_by_frame if fid < cutoff]
        for fid in stale:
            self.detections_by_frame.pop(fid, None)

    def flush(self) -> list[Decision]:
        """Close and decide everything outstanding. Called at shutdown."""
        decisions: list[Decision] = []
        packet = self.last_frame
        now = packet.capture_time_s if packet else self.clock.now()
        frame_id = packet.frame_id if packet else self.frame_count

        for track in self.tracker.finished_tracks():
            self._finalise_track(track, None)

        # Tracks still alive at shutdown have been gated but never finalised;
        # resolve them so their shots are not left waiting on a deadline that
        # will never be fed.
        for track in self.tracker.all_tracks():
            if track.shot_id is not None and track.morphology is None:
                self._finalise_track(track, None)

        self.shots.flush(now, frame_id)
        for shot in self.shots.drain_pending(now, self.tracks_by_id):
            decision, _ = self._decide_shot(shot)
            decisions.append(decision)

        # Anything still queued describes fluid that has already passed.
        self.scheduler.flush()
        return decisions

    def stats(self) -> dict[str, Any]:
        return {
            "frames": self.frame_count,
            "metrics": self.metrics.snapshot(),
            "gate": self.gate.stats(),
            "shots": self.shots.stats(),
            "decision": self.decider.stats(),
            "scheduler": self.scheduler.stats(),
            "health": self.health.snapshot(),
            "quality_gate": {
                "n_pass": getattr(self.quality_gate, "n_pass", None),
                "n_degraded": getattr(self.quality_gate, "n_degraded", None),
                "n_reject": getattr(self.quality_gate, "n_reject", None),
            },
        }

    def summary(self) -> dict[str, Any]:
        """Compact end-of-run summary for the console and the audit log."""
        shots = self.shots.history
        decided = [s for s in shots if s.status is not None]
        return {
            "frames_processed": self.frame_count,
            "tracks_created": len(self.tracks_by_id),
            "tracks_gated": self.gate.n_crossings,
            "shots": len(shots),
            "accept": sum(1 for s in decided if s.status is ShotStatus.ACCEPT),
            "reject": sum(1 for s in decided if s.status is ShotStatus.REJECT),
            "indeterminate": sum(
                1 for s in decided if s.status is ShotStatus.INDETERMINATE
            ),
            "mean_trackable_per_shot": (
                sum(s.trackable_count for s in shots) / len(shots) if shots else 0.0
            ),
            "commands_dispatched": self.scheduler.n_dispatched,
            "commands_late": self.scheduler.n_late,
            "commands_dropped_late": self.scheduler.n_dropped_late,
        }
