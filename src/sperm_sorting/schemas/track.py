"""Track-level schemas.

A :class:`TrackRecord` is the single accounting unit of the whole product: one
physical sperm, one persistent ID, counted exactly once. Every quantity that
feeds the shot ratio hangs off this record, and every reason a track failed to
qualify is recorded on it, so that any FIELD_ON/FIELD_OFF decision can be
reconstructed from the audit log alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..constants import MOTILITY_PROFILE_VERSION, SCHEMA_VERSION
from .detection import BoundingBox
from .enums import (
    FlowCorrectionMode,
    IneligibilityReason,
    MotilityClass,
    TimestampSource,
    TrackState,
)
from .morphology import MorphologyResult


@dataclass(slots=True)
class TrackPoint:
    """One observation of a track: where it was, when, and how sure we were."""

    frame_id: int
    capture_time_s: float
    box: BoundingBox
    score: float
    #: False when the position was predicted by the motion model rather than
    #: measured, i.e. the track was unmatched on this frame. Interpolated
    #: points are excluded from best-frame selection and are flagged in the
    #: kinematics so that velocity is never computed from pure prediction.
    observed: bool = True

    @property
    def x(self) -> float:
        return self.box.cx

    @property
    def y(self) -> float:
        return self.box.cy

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "capture_time_s": float(self.capture_time_s),
            "box_xyxy": list(self.box.as_xyxy()),
            "score": float(self.score),
            "observed": self.observed,
        }


@dataclass(slots=True)
class MotionFeatures:
    """CASA-style kinematics for one track.

    Velocities exist in two unit systems and both are retained:

    * ``*_px_s``    -- always available, pixels per second.
    * ``*_um_s``    -- ``None`` unless an optical calibration was loaded.

    and in two frames of reference:

    * *raw*         -- as observed, including bulk fluid transport.
    * *corrected*   -- with the estimated flow field removed.

    Progressive classification must use the **corrected** values. The raw
    values are kept so that a reviewer can see how large the correction was.
    """

    # -- provenance -------------------------------------------------------
    n_points: int
    n_observed_points: int
    duration_s: float
    mean_frame_interval_s: float
    timestamp_source: TimestampSource
    flow_correction_mode: FlowCorrectionMode
    profile_version: str = MOTILITY_PROFILE_VERSION
    #: ``True`` once micrometre values are trustworthy.
    optically_calibrated: bool = False
    um_per_px: float | None = None

    # -- raw (uncorrected) kinematics, pixels/second -----------------------
    vcl_px_s: float = 0.0
    vsl_px_s: float = 0.0
    vap_px_s: float = 0.0

    # -- flow-corrected kinematics, pixels/second --------------------------
    vcl_corrected_px_s: float = 0.0
    vsl_corrected_px_s: float = 0.0
    vap_corrected_px_s: float = 0.0

    # -- flow-corrected kinematics, micrometres/second ---------------------
    vcl_um_s: float | None = None
    vsl_um_s: float | None = None
    vap_um_s: float | None = None

    # -- dimensionless ratios (computed from corrected values) -------------
    #: LIN = VSL / VCL
    lin: float | None = None
    #: STR = VSL / VAP
    str_: float | None = None
    #: WOB = VAP / VCL
    wob: float | None = None

    # -- optional, frame-rate dependent ------------------------------------
    #: Amplitude of lateral head displacement. ``None`` when the sampling rate
    #: or track length cannot support a trustworthy estimate.
    alh_um: float | None = None
    alh_unavailable_reason: str = ""
    #: Beat-cross frequency, Hz. Same availability caveat as ALH.
    bcf_hz: float | None = None
    bcf_unavailable_reason: str = ""

    # -- geometry ----------------------------------------------------------
    #: Net displacement start->end, pixels.
    net_displacement_px: float = 0.0
    #: Total path length, pixels.
    path_length_px: float = 0.0
    #: Net direction of travel in radians, atan2(dy, dx), image coordinates.
    direction_rad: float | None = None
    #: Circular standard deviation of per-step headings. Low = straight swim.
    direction_stability: float | None = None

    # -- estimated flow actually subtracted --------------------------------
    flow_vx_px_s: float = 0.0
    flow_vy_px_s: float = 0.0

    # -- classification ----------------------------------------------------
    motility_class: MotilityClass = MotilityClass.UNDETERMINED
    motility_reason: str = ""

    @property
    def is_progressive(self) -> bool:
        return self.motility_class.is_progressive

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "n_observed_points": self.n_observed_points,
            "duration_s": float(self.duration_s),
            "mean_frame_interval_s": float(self.mean_frame_interval_s),
            "timestamp_source": str(self.timestamp_source),
            "flow_correction_mode": str(self.flow_correction_mode),
            "profile_version": self.profile_version,
            "optically_calibrated": self.optically_calibrated,
            "um_per_px": self.um_per_px,
            "vcl_px_s": float(self.vcl_px_s),
            "vsl_px_s": float(self.vsl_px_s),
            "vap_px_s": float(self.vap_px_s),
            "vcl_corrected_px_s": float(self.vcl_corrected_px_s),
            "vsl_corrected_px_s": float(self.vsl_corrected_px_s),
            "vap_corrected_px_s": float(self.vap_corrected_px_s),
            "vcl_um_s": self.vcl_um_s,
            "vsl_um_s": self.vsl_um_s,
            "vap_um_s": self.vap_um_s,
            "lin": self.lin,
            "str": self.str_,
            "wob": self.wob,
            "alh_um": self.alh_um,
            "alh_unavailable_reason": self.alh_unavailable_reason,
            "bcf_hz": self.bcf_hz,
            "bcf_unavailable_reason": self.bcf_unavailable_reason,
            "net_displacement_px": float(self.net_displacement_px),
            "path_length_px": float(self.path_length_px),
            "direction_rad": self.direction_rad,
            "direction_stability": self.direction_stability,
            "flow_vx_px_s": float(self.flow_vx_px_s),
            "flow_vy_px_s": float(self.flow_vy_px_s),
            "motility_class": str(self.motility_class),
            "motility_reason": self.motility_reason,
        }


@dataclass(slots=True)
class CropRecord:
    """The crop that was actually sent to the morphology model.

    :attr:`track_id` is duplicated here on purpose. The specification requires
    that the crop belong to the *same* tracked sperm whose motion was
    evaluated, and ``tests/test_crop_track_identity.py`` asserts that this
    field matches the owning :class:`TrackRecord`. Carrying the ID on the crop
    itself makes the invariant checkable rather than assumed.
    """

    track_id: int
    frame_id: int
    capture_time_s: float
    #: Box in source-frame pixels *after* padding, before resize.
    source_box: BoundingBox
    #: Output crop size ``(h, w)`` fed to the model.
    output_size: tuple[int, int]
    #: Composite best-frame quality score that won the selection.
    quality_score: float
    #: Per-term breakdown of ``quality_score`` for auditability.
    quality_terms: dict[str, float] = field(default_factory=dict)
    #: True when the padded box had to be clipped at the frame border.
    truncated: bool = False
    #: Fraction of the padded box that lay inside the frame, 0-1.
    visible_fraction: float = 1.0
    #: Best-effort judgement of whether the full tail is inside the crop.
    tail_complete: bool | None = None
    #: Maximum IoU with any other detection in the same frame. High values
    #: mean the crop is contaminated by a neighbouring sperm.
    max_overlap_iou: float = 0.0
    detector_score: float = 0.0
    track_confidence: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "capture_time_s": float(self.capture_time_s),
            "source_box_xyxy": list(self.source_box.as_xyxy()),
            "output_size": list(self.output_size),
            "quality_score": float(self.quality_score),
            "quality_terms": {k: float(v) for k, v in self.quality_terms.items()},
            "truncated": self.truncated,
            "visible_fraction": float(self.visible_fraction),
            "tail_complete": self.tail_complete,
            "max_overlap_iou": float(self.max_overlap_iou),
            "detector_score": float(self.detector_score),
            "track_confidence": float(self.track_confidence),
        }


@dataclass(slots=True)
class TrackRecord:
    """One uniquely-identified sperm across its whole observed lifetime.

    The eligibility rule is deliberately implemented once, here, in
    :meth:`compute_eligibility`. No other module is permitted to decide that a
    sperm is ``ai_eligible``.
    """

    track_id: int
    state: TrackState = TrackState.TENTATIVE
    points: list[TrackPoint] = field(default_factory=list)

    first_frame_id: int = -1
    last_frame_id: int = -1
    first_time_s: float = 0.0
    last_time_s: float = 0.0

    #: Consecutive frames matched to a measurement since creation.
    hit_count: int = 0
    #: Frames since the last successful match.
    time_since_update: int = 0
    #: Mean detector score over observed points; a crude track confidence.
    mean_score: float = 0.0

    motion: MotionFeatures | None = None
    crop: CropRecord | None = None
    morphology: MorphologyResult | None = None

    #: Shot this track was assigned to when it crossed the counting gate.
    #: ``None`` means it has not been gated (and so is not counted anywhere).
    shot_id: int | None = None
    #: Frame at which the gate crossing happened.
    gate_crossing_frame_id: int | None = None
    gate_crossing_time_s: float | None = None

    #: Whether the track passed the minimum-length / minimum-quality bar to be
    #: counted at all. A failing track is not in the numerator *or* the
    #: denominator: it is not a trustworthy observation of a sperm.
    track_quality_pass: bool = False
    track_quality_reason: str = ""

    #: Set once morphology has resolved one way or the other.
    evaluation_complete: bool = False
    #: Monotonic deadline by which morphology had to finish.
    evaluation_deadline_s: float | None = None

    ai_eligible: bool = False
    ineligibility_reason: IneligibilityReason = IneligibilityReason.NONE

    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # ---------------------------------------------------------------- points

    def add_point(self, point: TrackPoint) -> None:
        """Append an observation and refresh the cached span/score."""
        self.points.append(point)
        if self.first_frame_id < 0:
            self.first_frame_id = point.frame_id
            self.first_time_s = point.capture_time_s
        self.last_frame_id = point.frame_id
        self.last_time_s = point.capture_time_s
        observed = [p.score for p in self.points if p.observed]
        self.mean_score = float(sum(observed) / len(observed)) if observed else 0.0

    @property
    def observed_points(self) -> list[TrackPoint]:
        return [p for p in self.points if p.observed]

    @property
    def n_observed(self) -> int:
        return sum(1 for p in self.points if p.observed)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_time_s - self.first_time_s)

    @property
    def last_box(self) -> BoundingBox | None:
        return self.points[-1].box if self.points else None

    # ----------------------------------------------------------- eligibility

    @property
    def is_progressive(self) -> bool:
        return self.motion is not None and self.motion.is_progressive

    @property
    def all_four_normal(self) -> bool:
        """The all-four morphology rule. Never an average, never a mean score."""
        return self.morphology is not None and self.morphology.all_four_normal

    def compute_eligibility(self) -> bool:
        """Evaluate the per-sperm eligibility rule and record *why*.

        ``ai_eligible`` is true only when every one of the following holds:

        1. the track is a valid unique track (it has an ID and was gated),
        2. it passed the track-quality bar,
        3. its flow-corrected motility is progressive (rapid **or** slow),
        4. head, acrosome, vacuole and tail are **all** normal,
        5. the evaluation completed before its deadline.

        A track that fails only (3), (4) or (5) still counts in the shot
        denominator -- it is a real observed sperm that simply did not qualify.
        Only a track failing (1) or (2) is excluded from the shot entirely, and
        that exclusion is handled by the shot manager, not here.
        """
        self.ineligibility_reason = IneligibilityReason.NONE

        if not self.track_quality_pass:
            self.ai_eligible = False
            self.ineligibility_reason = IneligibilityReason.TRACK_QUALITY_FAIL
            return False

        if self.motion is None or self.motion.motility_class.name == "UNDETERMINED":
            self.ai_eligible = False
            self.ineligibility_reason = IneligibilityReason.MOTILITY_UNDETERMINED
            return False

        if not self.motion.is_progressive:
            self.ai_eligible = False
            self.ineligibility_reason = IneligibilityReason.NOT_PROGRESSIVE
            return False

        if self.morphology is None or not self.morphology.is_complete:
            self.ai_eligible = False
            self.ineligibility_reason = (
                IneligibilityReason.DEADLINE_MISSED
                if self.morphology is not None
                and self.morphology.status.name == "DEADLINE_MISSED"
                else IneligibilityReason.MORPHOLOGY_INCOMPLETE
            )
            return False

        if not self.evaluation_complete:
            self.ai_eligible = False
            self.ineligibility_reason = IneligibilityReason.DEADLINE_MISSED
            return False

        # All four aspects must be normal. Report the first failing one so the
        # audit log explains the rejection concretely.
        first_abnormal = self.morphology.first_abnormal_aspect()
        if first_abnormal is not None:
            self.ai_eligible = False
            self.ineligibility_reason = {
                "head": IneligibilityReason.ABNORMAL_HEAD,
                "acrosome": IneligibilityReason.ABNORMAL_ACROSOME,
                "vacuole": IneligibilityReason.ABNORMAL_VACUOLE,
                "tail": IneligibilityReason.ABNORMAL_TAIL,
            }[first_abnormal]
            return False

        self.ai_eligible = True
        return True

    # ----------------------------------------------------------------- audit

    def to_json_dict(self, *, include_points: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "track_id": self.track_id,
            "state": str(self.state),
            "first_frame_id": self.first_frame_id,
            "last_frame_id": self.last_frame_id,
            "first_time_s": float(self.first_time_s),
            "last_time_s": float(self.last_time_s),
            "duration_s": float(self.duration_s),
            "n_points": len(self.points),
            "n_observed": self.n_observed,
            "mean_score": float(self.mean_score),
            "shot_id": self.shot_id,
            "gate_crossing_frame_id": self.gate_crossing_frame_id,
            "gate_crossing_time_s": self.gate_crossing_time_s,
            "track_quality_pass": self.track_quality_pass,
            "track_quality_reason": self.track_quality_reason,
            "evaluation_complete": self.evaluation_complete,
            "evaluation_deadline_s": self.evaluation_deadline_s,
            "ai_eligible": self.ai_eligible,
            "ineligibility_reason": str(self.ineligibility_reason),
            "motion": self.motion.to_json_dict() if self.motion else None,
            "crop": self.crop.to_json_dict() if self.crop else None,
            "morphology": self.morphology.to_json_dict() if self.morphology else None,
            "schema_version": self.schema_version,
        }
        if include_points:
            out["points"] = [p.to_json_dict() for p in self.points]
        return out
