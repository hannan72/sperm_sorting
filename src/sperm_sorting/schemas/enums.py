"""Enumerations shared across the pipeline.

All enums derive from :class:`str` so that they serialise to readable JSON in
the audit log without a custom encoder, and compare equal to their wire value.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """``str``-valued enum with a readable ``repr``."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


class SourceKind(StrEnum):
    """Where frames come from. All three feed the identical downstream graph."""

    BASLER = "basler"
    VIDEO = "video"
    SYNTHETIC = "synthetic"


class TimestampSource(StrEnum):
    """Provenance of a frame's timestamp.

    Motion analysis must know whether it is working from real hardware
    timestamps or from a software approximation, because the latter carries
    jitter that propagates directly into velocity estimates.
    """

    #: Camera-provided hardware tick, converted to seconds.
    HARDWARE = "hardware"
    #: Host monotonic clock read at grab time.
    HOST_MONOTONIC = "host_monotonic"
    #: Derived from container/media PTS during replay.
    CONTAINER_PTS = "container_pts"
    #: Synthesised at a known exact interval by the simulator.
    SYNTHETIC = "synthetic"


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------


class QualityVerdict(StrEnum):
    """Outcome of the image-quality gate for a whole frame."""

    PASS = "pass"
    #: Usable for tracking continuity but not for morphology crops.
    DEGRADED = "degraded"
    #: Unusable; the frame is dropped and the drop is counted.
    REJECT = "reject"


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


class TrackState(StrEnum):
    """Lifecycle of a track inside the multi-object tracker."""

    #: Created this frame, not yet confirmed by the min-hits rule.
    TENTATIVE = "tentative"
    #: Confirmed and currently matched.
    CONFIRMED = "confirmed"
    #: Confirmed but unmatched for >= 1 frame; predicted forward.
    LOST = "lost"
    #: Terminated; will never be revived and its ID is never reused.
    REMOVED = "removed"


# --------------------------------------------------------------------------
# Motility
# --------------------------------------------------------------------------


class MotilityClass(StrEnum):
    """WHO-inspired motility grade assigned from *flow-corrected* kinematics.

    ``RAPID_PROGRESSIVE`` and ``SLOW_PROGRESSIVE`` both satisfy the progressive
    motility filter; see :meth:`is_progressive`.
    """

    RAPID_PROGRESSIVE = "rapid_progressive"
    SLOW_PROGRESSIVE = "slow_progressive"
    NON_PROGRESSIVE = "non_progressive"
    IMMOTILE = "immotile"
    #: Kinematics could not be computed (too few points, no calibration, ...).
    UNDETERMINED = "undetermined"

    @property
    def is_progressive(self) -> bool:
        """True for the two grades that pass the progressive-motility filter."""
        return self in (
            MotilityClass.RAPID_PROGRESSIVE,
            MotilityClass.SLOW_PROGRESSIVE,
        )


class FlowCorrectionMode(StrEnum):
    """How bulk fluid motion is removed from observed trajectories."""

    #: No correction. For controlled, still-fluid test recordings only.
    DISABLED = "disabled"
    #: Single calibrated (vx, vy) subtracted everywhere.
    FIXED_VECTOR = "fixed_vector"
    #: Position-dependent velocity field sampled from a calibrated map.
    FLOW_MAP = "flow_map"
    #: Per-frame robust estimate from debris / non-motile objects.
    ROBUST_ESTIMATE = "robust_estimate"


# --------------------------------------------------------------------------
# Morphology
# --------------------------------------------------------------------------


class MorphologyStatus(StrEnum):
    """Whether a morphology evaluation produced a usable result."""

    COMPLETE = "complete"
    #: Not finished before the track's evaluation deadline.
    DEADLINE_MISSED = "deadline_missed"
    #: No frame in the track met the minimum crop-quality bar.
    NO_VALID_CROP = "no_valid_crop"
    #: The model raised or the backend was unavailable.
    INFERENCE_FAILED = "inference_failed"
    #: Track was never classified progressive, so morphology was never run.
    NOT_REQUIRED = "not_required"


# --------------------------------------------------------------------------
# Shots and decision
# --------------------------------------------------------------------------


class ShotCloseReason(StrEnum):
    """Why a shot stopped accepting new tracks."""

    TARGET_REACHED = "target_reached"
    HARD_MAXIMUM = "hard_maximum"
    TIMEOUT = "timeout"
    #: Pipeline is shutting down; partial shot flushed.
    SHUTDOWN = "shutdown"


class ShotStatus(StrEnum):
    """Terminal classification of a shot."""

    ACCEPT = "accept"
    REJECT = "reject"
    #: Fewer than the minimum trackable sperm; no reliable ratio exists.
    INDETERMINATE = "indeterminate"


class FieldCommandKind(StrEnum):
    """The only two logical outputs of the prototype."""

    FIELD_ON = "FIELD_ON"
    FIELD_OFF = "FIELD_OFF"


class CommandOrigin(StrEnum):
    """Why a field command exists. Drives audit and priority."""

    #: Normal per-shot decision.
    DECISION = "decision"
    #: Startup / shutdown default.
    SAFE_DEFAULT = "safe_default"
    #: Watchdog expiry.
    WATCHDOG = "watchdog"
    #: Operator or test harness override.
    MANUAL = "manual"


class CommandOutcome(StrEnum):
    """What actually happened to a scheduled command."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    #: Dispatched after its deadline by more than the tolerance.
    LATE = "late"
    #: Dropped because it was superseded before dispatch.
    SUPERSEDED = "superseded"
    FAILED = "failed"


# --------------------------------------------------------------------------
# Eligibility bookkeeping
# --------------------------------------------------------------------------


class IneligibilityReason(StrEnum):
    """Why a valid, counted track was not ``ai_eligible``.

    Every track in a shot's denominator that is not in its numerator carries
    exactly one of these, so any decision can be explained after the fact.
    """

    NONE = "none"
    TRACK_QUALITY_FAIL = "track_quality_fail"
    NOT_PROGRESSIVE = "not_progressive"
    MOTILITY_UNDETERMINED = "motility_undetermined"
    ABNORMAL_HEAD = "abnormal_head"
    ABNORMAL_ACROSOME = "abnormal_acrosome"
    ABNORMAL_VACUOLE = "abnormal_vacuole"
    ABNORMAL_TAIL = "abnormal_tail"
    MORPHOLOGY_INCOMPLETE = "morphology_incomplete"
    DEADLINE_MISSED = "deadline_missed"
