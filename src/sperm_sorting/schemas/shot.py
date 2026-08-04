"""Shot schemas.

A *shot* is a software-defined segment of the physically continuous flow: the
portion passing the imaging region that contains, on average, 25 +/- 5 uniquely
trackable sperm, treated as one independent AI decision unit.

The ratio arithmetic lives here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..constants import (
    ACCEPT_RATIO_THRESHOLD,
    EPS,
    MINIMUM_TRACKABLE_SPERM,
    SCHEMA_VERSION,
)
from .enums import IneligibilityReason, ShotCloseReason, ShotStatus


def exceeds_threshold(numerator: int, denominator: int, threshold: float) -> bool:
    """Exact ``numerator / denominator > threshold``, without float error.

    The 60% rule is a hard product boundary: exactly 60% must REJECT. Binary
    floating point cannot represent 0.60, so a naive ``n / d > 0.60`` risks
    flipping a boundary case if the compiler, platform or a later refactor
    changes the rounding of either side.

    Both sides are therefore converted to exact rationals.
    ``Fraction(str(threshold))`` is used rather than ``Fraction(threshold)``
    because the latter would capture the binary approximation
    (0.59999999999999997...) and make exactly-60% compare as *above* threshold.

    >>> exceeds_threshold(15, 25, 0.60)   # 60.0% -- not above
    False
    >>> exceeds_threshold(16, 25, 0.60)   # 64.0%
    True
    >>> exceeds_threshold(12, 20, 0.60)   # 60.0% -- not above
    False
    >>> exceeds_threshold(13, 20, 0.60)   # 65.0%
    True
    """
    if denominator <= 0:
        return False
    return Fraction(numerator, denominator) > Fraction(str(threshold))


@dataclass(slots=True)
class ShotRecord:
    """One decision unit: its members, its ratio, and its verdict.

    The invariant this type exists to protect is that the denominator is
    *every* valid trackable sperm assigned to the shot -- not just the
    progressive ones, and not just the morphologically normal ones.
    """

    shot_id: int
    opened_at_s: float
    #: Frame on which the shot opened.
    opened_frame_id: int

    #: Track IDs assigned to this shot, in gate-crossing order. This list *is*
    #: the denominator; its length is ``trackable_count``. Membership is
    #: enforced unique by :meth:`add_track`.
    track_ids: list[int] = field(default_factory=list)
    #: Subset of :attr:`track_ids` that satisfied the full eligibility rule.
    eligible_track_ids: list[int] = field(default_factory=list)

    closed_at_s: float | None = None
    closed_frame_id: int | None = None
    close_reason: ShotCloseReason | None = None

    #: Gate-crossing instants of the first and last members. Together these
    #: delimit the *fluid segment* this shot describes, which is what the
    #: field has to cover: a rejected shot needs FIELD_ON from
    #: ``first_gate_time_s + transport_delay`` until
    #: ``last_gate_time_s + transport_delay``, not at a single instant.
    #: They differ from :attr:`opened_at_s` / :attr:`closed_at_s` when a shot
    #: closes on timeout rather than on a crossing.
    first_gate_time_s: float | None = None
    last_gate_time_s: float | None = None

    status: ShotStatus | None = None
    #: Cached ratio at decision time; ``None`` until decided.
    ai_eligible_ratio: float | None = None
    #: Threshold in force when this shot was decided, recorded so that a
    #: later threshold change does not retroactively reinterpret old logs.
    threshold_applied: float = ACCEPT_RATIO_THRESHOLD
    minimum_trackable_applied: int = MINIMUM_TRACKABLE_SPERM

    #: Histogram of why non-eligible members failed, for operator diagnostics.
    ineligibility_histogram: dict[str, int] = field(default_factory=dict)

    #: Number of tracks that crossed the gate but were rejected as invalid
    #: (too short, poor quality) and therefore never entered the denominator.
    rejected_track_count: int = 0

    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------- membership

    def add_track(self, track_id: int) -> bool:
        """Assign a track to this shot. Returns ``False`` if already present.

        Duplicate assignment is the single most dangerous failure mode in this
        product -- it would let one physical sperm be counted many times -- so
        it is rejected here rather than deduplicated silently downstream.
        """
        if track_id in self.track_ids:
            return False
        self.track_ids.append(track_id)
        return True

    @property
    def trackable_count(self) -> int:
        """The denominator: unique valid trackable sperm in this shot."""
        return len(self.track_ids)

    @property
    def ai_eligible_count(self) -> int:
        """The numerator: unique ``ai_eligible`` sperm in this shot."""
        return len(self.eligible_track_ids)

    @property
    def is_closed(self) -> bool:
        return self.close_reason is not None

    @property
    def duration_s(self) -> float:
        if self.closed_at_s is None:
            return 0.0
        return max(0.0, self.closed_at_s - self.opened_at_s)

    @property
    def gate_span_s(self) -> float:
        """How long this shot's fluid segment took to pass the gate."""
        if self.first_gate_time_s is None or self.last_gate_time_s is None:
            return 0.0
        return max(0.0, self.last_gate_time_s - self.first_gate_time_s)

    def note_gate_crossing(self, time_s: float) -> None:
        """Extend the fluid segment to include a crossing at ``time_s``."""
        if self.first_gate_time_s is None or time_s < self.first_gate_time_s:
            self.first_gate_time_s = time_s
        if self.last_gate_time_s is None or time_s > self.last_gate_time_s:
            self.last_gate_time_s = time_s

    # ------------------------------------------------------------------ ratio

    def compute_ratio(self) -> float:
        """``ai_eligible_count / trackable_count``, or 0.0 for an empty shot.

        Deliberately *not* divided by the progressive count, and abnormal
        sperm are deliberately *not* removed from the denominator.
        """
        if self.trackable_count <= 0:
            return 0.0
        return self.ai_eligible_count / max(self.trackable_count, EPS)

    def record_ineligibility(self, reason: IneligibilityReason) -> None:
        key = str(reason)
        self.ineligibility_histogram[key] = self.ineligibility_histogram.get(key, 0) + 1

    # ------------------------------------------------------------------ audit

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "opened_at_s": float(self.opened_at_s),
            "opened_frame_id": self.opened_frame_id,
            "closed_at_s": self.closed_at_s,
            "closed_frame_id": self.closed_frame_id,
            "close_reason": str(self.close_reason) if self.close_reason else None,
            "duration_s": float(self.duration_s),
            "first_gate_time_s": self.first_gate_time_s,
            "last_gate_time_s": self.last_gate_time_s,
            "gate_span_s": float(self.gate_span_s),
            "trackable_count": self.trackable_count,
            "ai_eligible_count": self.ai_eligible_count,
            "ai_eligible_ratio": self.ai_eligible_ratio,
            "status": str(self.status) if self.status else None,
            "threshold_applied": float(self.threshold_applied),
            "minimum_trackable_applied": self.minimum_trackable_applied,
            "track_ids": list(self.track_ids),
            "eligible_track_ids": list(self.eligible_track_ids),
            "ineligibility_histogram": dict(self.ineligibility_histogram),
            "rejected_track_count": self.rejected_track_count,
            "schema_version": self.schema_version,
        }
