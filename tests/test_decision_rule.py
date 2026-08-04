"""The decision rule.

These are the product's acceptance criteria expressed as code. If any test in
this file fails, the device sorts differently than specified, and no amount of
model accuracy compensates for that.

The five cases marked "mandated" are stated explicitly in the specification.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sperm_sorting.config import DecisionConfig
from sperm_sorting.decision.engine import DecisionEngine, decide
from sperm_sorting.schemas.enums import FieldCommandKind, ShotStatus
from sperm_sorting.schemas.shot import ShotRecord, exceeds_threshold

THRESHOLD = 0.60
MINIMUM = 20


# --------------------------------------------------------------------------
# The five mandated cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("eligible", "trackable", "status", "command"),
    [
        # 15/25 = exactly 60% -> REJECT. Exactly at the threshold is NOT above it.
        (15, 25, ShotStatus.REJECT, FieldCommandKind.FIELD_ON),
        # 16/25 = 64% -> ACCEPT.
        (16, 25, ShotStatus.ACCEPT, FieldCommandKind.FIELD_OFF),
        # 12/20 = exactly 60% -> REJECT.
        (12, 20, ShotStatus.REJECT, FieldCommandKind.FIELD_ON),
        # 13/20 = 65% -> ACCEPT.
        (13, 20, ShotStatus.ACCEPT, FieldCommandKind.FIELD_OFF),
    ],
)
def test_mandated_ratio_cases(
    eligible: int, trackable: int, status: ShotStatus, command: FieldCommandKind
) -> None:
    result = decide(
        eligible, trackable, threshold=THRESHOLD, minimum_trackable=MINIMUM
    )
    assert result.status is status
    assert result.field_command is command


def test_mandated_timeout_case_is_indeterminate() -> None:
    """19 trackable at timeout -> INDETERMINATE + FIELD_OFF.

    Even at 100% eligibility: below the minimum there is no reliable estimate,
    so no sorting decision is made and the field stays off.
    """
    for eligible in (0, 10, 19):
        result = decide(
            eligible, 19, threshold=THRESHOLD, minimum_trackable=MINIMUM
        )
        assert result.status is ShotStatus.INDETERMINATE
        assert result.field_command is FieldCommandKind.FIELD_OFF


# --------------------------------------------------------------------------
# The boundary itself
# --------------------------------------------------------------------------


def test_exactly_threshold_rejects_for_every_representable_ratio() -> None:
    """Every (n, d) with n/d == 3/5 exactly must REJECT.

    This is the test that would catch a floating-point regression at the
    boundary: 0.60 has no exact binary representation, so a naive comparison
    can flip depending on how each side happens to round.
    """
    exact_sixty = [
        (n, d)
        for d in range(MINIMUM, 201)
        for n in range(d + 1)
        if Fraction(n, d) == Fraction(3, 5)
    ]
    assert len(exact_sixty) >= 20, "expected many exactly-60% pairs to test"
    for n, d in exact_sixty:
        result = decide(n, d, threshold=THRESHOLD, minimum_trackable=MINIMUM)
        assert result.status is ShotStatus.REJECT, f"{n}/{d} should REJECT"
        assert result.field_command is FieldCommandKind.FIELD_ON


def test_just_above_threshold_accepts() -> None:
    """The smallest possible step above 60% must flip the decision."""
    for d in range(MINIMUM, 101):
        n = 0
        while Fraction(n, d) <= Fraction(3, 5):
            n += 1
        if n > d:
            continue
        result = decide(n, d, threshold=THRESHOLD, minimum_trackable=MINIMUM)
        assert result.status is ShotStatus.ACCEPT, f"{n}/{d} should ACCEPT"


def test_exceeds_threshold_is_exact() -> None:
    assert exceeds_threshold(15, 25, 0.60) is False
    assert exceeds_threshold(16, 25, 0.60) is True
    assert exceeds_threshold(12, 20, 0.60) is False
    assert exceeds_threshold(13, 20, 0.60) is True
    assert exceeds_threshold(3, 5, 0.60) is False
    assert exceeds_threshold(0, 0, 0.60) is False


def test_threshold_uses_decimal_not_binary_interpretation() -> None:
    """0.60 must mean three fifths, not its binary approximation.

    ``Fraction(0.60)`` is 5404319552844595/9007199254740992, which is very
    slightly *less* than 3/5. Comparing against that would make exactly-60%
    read as above threshold and would silently invert the boundary case.
    """
    assert Fraction(str(0.60)) == Fraction(3, 5)
    assert Fraction(0.60) < Fraction(3, 5)
    assert exceeds_threshold(3, 5, 0.60) is False


# --------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------


def test_field_on_is_the_rejection() -> None:
    """Energising the magnet diverts to waste, so FIELD_ON means "rejected".

    Reading FIELD_ON as "good" inverts the product, which is why this is
    asserted directly rather than left implicit in the parametrised cases.
    """
    poor = decide(1, 25, threshold=THRESHOLD, minimum_trackable=MINIMUM)
    assert poor.status is ShotStatus.REJECT
    assert poor.field_command is FieldCommandKind.FIELD_ON

    good = decide(25, 25, threshold=THRESHOLD, minimum_trackable=MINIMUM)
    assert good.status is ShotStatus.ACCEPT
    assert good.field_command is FieldCommandKind.FIELD_OFF


def test_indeterminate_is_fail_safe() -> None:
    """Every sub-minimum count is FIELD_OFF regardless of the ratio."""
    for trackable in range(0, MINIMUM):
        for eligible in range(trackable + 1):
            result = decide(
                eligible, trackable, threshold=THRESHOLD, minimum_trackable=MINIMUM
            )
            assert result.status is ShotStatus.INDETERMINATE
            assert result.field_command is FieldCommandKind.FIELD_OFF


def test_minimum_count_boundary() -> None:
    """Exactly the minimum is decidable; one below is not."""
    at_min = decide(20, MINIMUM, threshold=THRESHOLD, minimum_trackable=MINIMUM)
    assert at_min.status is ShotStatus.ACCEPT

    below = decide(19, MINIMUM - 1, threshold=THRESHOLD, minimum_trackable=MINIMUM)
    assert below.status is ShotStatus.INDETERMINATE


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_numerator_may_not_exceed_denominator() -> None:
    """The eligible set is a subset of the trackable set, by construction.

    A violation means the shot accounting is broken upstream, and continuing
    would produce a ratio above 1.0 that would always accept.
    """
    with pytest.raises(ValueError, match="exceeds trackable"):
        decide(26, 25, threshold=THRESHOLD, minimum_trackable=MINIMUM)


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        decide(-1, 25, threshold=THRESHOLD, minimum_trackable=MINIMUM)


# --------------------------------------------------------------------------
# Engine integration
# --------------------------------------------------------------------------


def test_engine_writes_verdict_onto_the_shot() -> None:
    engine = DecisionEngine(DecisionConfig())
    shot = ShotRecord(shot_id=7, opened_at_s=0.0, opened_frame_id=0)
    for i in range(25):
        shot.add_track(i)
    shot.eligible_track_ids = list(range(15))  # exactly 60%

    decision = engine.evaluate(shot)

    assert shot.status is ShotStatus.REJECT
    assert shot.ai_eligible_ratio == pytest.approx(0.60)
    assert decision.field_command is FieldCommandKind.FIELD_ON
    # The rule in force is stamped, so a later config change cannot
    # retroactively reinterpret this record.
    assert shot.threshold_applied == 0.60
    assert shot.minimum_trackable_applied == 20


def test_engine_counts_outcomes() -> None:
    engine = DecisionEngine(DecisionConfig())
    for eligible, trackable in [(25, 25), (15, 25), (1, 5)]:
        shot = ShotRecord(shot_id=0, opened_at_s=0.0, opened_frame_id=0)
        for i in range(trackable):
            shot.add_track(i)
        shot.eligible_track_ids = list(range(eligible))
        engine.evaluate(shot)

    stats = engine.stats()
    assert stats["n_accept"] == 1
    assert stats["n_reject"] == 1
    assert stats["n_indeterminate"] == 1


def test_configurable_threshold_is_honoured_and_stamped() -> None:
    """A different threshold changes the decision and is recorded as applied."""
    engine = DecisionEngine(DecisionConfig(threshold=0.75))
    shot = ShotRecord(shot_id=0, opened_at_s=0.0, opened_frame_id=0)
    for i in range(20):
        shot.add_track(i)
    shot.eligible_track_ids = list(range(15))  # 75%, exactly at the threshold

    engine.evaluate(shot)
    assert shot.status is ShotStatus.REJECT  # exactly at threshold still rejects
    assert shot.threshold_applied == 0.75


def test_rationale_explains_the_decision() -> None:
    """Every decision carries a human-readable reason for the audit log."""
    reject = decide(15, 25, threshold=THRESHOLD, minimum_trackable=MINIMUM)
    assert "15/25" in reject.rationale
    assert "0.60" in reject.rationale

    indet = decide(5, 10, threshold=THRESHOLD, minimum_trackable=MINIMUM)
    assert "10" in indet.rationale and "20" in indet.rationale
