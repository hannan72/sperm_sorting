"""Decision engine.

This module contains the whole of the product's decision rule. It is
deliberately small, deliberately free of dependencies on the rest of the
pipeline, and deliberately pure: given a count of eligible sperm and a count of
trackable sperm, it returns a status and a field command, and nothing else.

The rule, verbatim::

    if trackable_count < 20:
        shot_status  = "INDETERMINATE"
        field_command = "FIELD_OFF"
    else:
        ratio = ai_eligible_count / trackable_count
        if ratio > 0.60:
            shot_status  = "ACCEPT"
            field_command = "FIELD_OFF"
        else:
            shot_status  = "REJECT"
            field_command = "FIELD_ON"

Two things about it are counter-intuitive often enough to be worth stating
plainly, because both have been "fixed" into bugs before:

* **Exactly 60% is a REJECT.** The comparison is a strict ``>``. 15/25 is not
  acceptable; 16/25 is.
* **FIELD_ON is the rejection.** Energising the magnet diverts the
  Annexin-V-labelled, bead-bound population toward the waste channel, so the
  field is switched *on* for a segment the AI has judged poor. An accepted
  segment is passed through with the field off. Reading FIELD_ON as "good"
  inverts the product.

What the AI is judging is visible phenotype only -- motility and morphology.
The magnetic separation acts on Annexin V binding, which this software does not
and cannot observe. The two mechanisms are complementary, not equivalent; see
``docs/safety_and_claims.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import DecisionConfig
from ..schemas.enums import FieldCommandKind, ShotStatus
from ..schemas.shot import ShotRecord, exceeds_threshold

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Decision:
    """The outcome of applying the rule to one shot."""

    shot_id: int
    status: ShotStatus
    field_command: FieldCommandKind
    ai_eligible_count: int
    trackable_count: int
    ratio: float
    threshold: float
    minimum_trackable: int
    #: Human-readable justification, written into the audit log.
    rationale: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "status": str(self.status),
            "field_command": str(self.field_command),
            "ai_eligible_count": self.ai_eligible_count,
            "trackable_count": self.trackable_count,
            "ratio": float(self.ratio),
            "threshold": float(self.threshold),
            "minimum_trackable": self.minimum_trackable,
            "rationale": self.rationale,
        }


def decide(
    ai_eligible_count: int,
    trackable_count: int,
    *,
    threshold: float,
    minimum_trackable: int,
    shot_id: int = -1,
) -> Decision:
    """Apply the decision rule. Pure function; no I/O, no state.

    Parameters
    ----------
    ai_eligible_count
        Number of unique sperm in the shot that satisfied *both* progressive
        motility *and* all four normal morphology aspects.
    trackable_count
        Number of unique valid trackable sperm assigned to the shot. This is
        every counted sperm, not only the progressive ones and not only the
        morphologically normal ones.

    Notes
    -----
    The ratio comparison goes through :func:`exceeds_threshold`, which uses
    exact rational arithmetic. 0.60 has no exact binary representation, so a
    plain ``ratio > 0.60`` would rest on the rounding of two separate
    floating-point conversions happening to agree. They do agree today; making
    the boundary depend on that is not worth the risk when the alternative
    costs nothing.
    """
    if ai_eligible_count < 0 or trackable_count < 0:
        raise ValueError(
            f"counts must be non-negative, got eligible={ai_eligible_count} "
            f"trackable={trackable_count}"
        )
    if ai_eligible_count > trackable_count:
        raise ValueError(
            f"eligible count ({ai_eligible_count}) exceeds trackable count "
            f"({trackable_count}); the numerator must be a subset of the "
            "denominator"
        )

    ratio = (ai_eligible_count / trackable_count) if trackable_count > 0 else 0.0

    if trackable_count < minimum_trackable:
        return Decision(
            shot_id=shot_id,
            status=ShotStatus.INDETERMINATE,
            field_command=FieldCommandKind.FIELD_OFF,
            ai_eligible_count=ai_eligible_count,
            trackable_count=trackable_count,
            ratio=ratio,
            threshold=threshold,
            minimum_trackable=minimum_trackable,
            rationale=(
                f"only {trackable_count} trackable sperm, below the minimum of "
                f"{minimum_trackable}; the ratio would not be a reliable "
                "estimate, so no sorting decision is made and the field stays "
                "off"
            ),
        )

    if exceeds_threshold(ai_eligible_count, trackable_count, threshold):
        return Decision(
            shot_id=shot_id,
            status=ShotStatus.ACCEPT,
            field_command=FieldCommandKind.FIELD_OFF,
            ai_eligible_count=ai_eligible_count,
            trackable_count=trackable_count,
            ratio=ratio,
            threshold=threshold,
            minimum_trackable=minimum_trackable,
            rationale=(
                f"{ai_eligible_count}/{trackable_count} = {ratio:.4f} exceeds "
                f"{threshold:.2f}; segment accepted and passed to collection "
                "with the field off"
            ),
        )

    return Decision(
        shot_id=shot_id,
        status=ShotStatus.REJECT,
        field_command=FieldCommandKind.FIELD_ON,
        ai_eligible_count=ai_eligible_count,
        trackable_count=trackable_count,
        ratio=ratio,
        threshold=threshold,
        minimum_trackable=minimum_trackable,
        rationale=(
            f"{ai_eligible_count}/{trackable_count} = {ratio:.4f} does not "
            f"exceed {threshold:.2f}; segment rejected and the field is "
            "energised to divert it"
        ),
    )


class DecisionEngine:
    """Applies :func:`decide` to closed shots and records the outcome.

    Thin by design. Everything interesting is in the pure function above, so
    that the rule can be exhaustively tested without constructing a pipeline.
    """

    def __init__(self, cfg: DecisionConfig) -> None:
        self.cfg = cfg
        self.n_accept = 0
        self.n_reject = 0
        self.n_indeterminate = 0

    def evaluate(self, shot: ShotRecord) -> Decision:
        """Decide one shot and write the verdict back onto its record."""
        decision = decide(
            shot.ai_eligible_count,
            shot.trackable_count,
            threshold=self.cfg.threshold,
            minimum_trackable=self.cfg.minimum_trackable_sperm,
            shot_id=shot.shot_id,
        )

        shot.status = decision.status
        shot.ai_eligible_ratio = decision.ratio
        # Stamp the rule that was actually in force, so a later config change
        # cannot retroactively reinterpret this log line.
        shot.threshold_applied = self.cfg.threshold
        shot.minimum_trackable_applied = self.cfg.minimum_trackable_sperm

        if decision.status is ShotStatus.ACCEPT:
            self.n_accept += 1
        elif decision.status is ShotStatus.REJECT:
            self.n_reject += 1
        else:
            self.n_indeterminate += 1

        logger.info(
            "shot %d: %s (%d/%d = %.4f) -> %s",
            shot.shot_id,
            decision.status,
            decision.ai_eligible_count,
            decision.trackable_count,
            decision.ratio,
            decision.field_command,
        )
        return decision

    def stats(self) -> dict[str, Any]:
        total = self.n_accept + self.n_reject + self.n_indeterminate
        return {
            "n_accept": self.n_accept,
            "n_reject": self.n_reject,
            "n_indeterminate": self.n_indeterminate,
            "n_total": total,
            "indeterminate_rate": (self.n_indeterminate / total) if total else None,
            "threshold": self.cfg.threshold,
            "minimum_trackable": self.cfg.minimum_trackable_sperm,
        }
