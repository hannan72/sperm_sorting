"""Per-sperm eligibility, and the all-four morphology rule.

The rule is conjunctive on five conditions. The tests below assert both that
all five are required and, individually, that *removing* each one changes the
answer -- because a rule that passes for the wrong reason is indistinguishable
from a correct one until it matters.
"""

from __future__ import annotations

import pytest

from sperm_sorting.constants import LABEL_ABNORMAL, LABEL_NORMAL, MORPHOLOGY_ASPECTS
from sperm_sorting.schemas.enums import (
    FlowCorrectionMode,
    IneligibilityReason,
    MorphologyStatus,
    MotilityClass,
    TimestampSource,
)
from sperm_sorting.schemas.morphology import AspectResult, MorphologyResult
from sperm_sorting.schemas.track import MotionFeatures, TrackRecord

from builders import make_track


def make_morphology(
    track_id: int = 1,
    *,
    head: bool = True,
    acrosome: bool = True,
    vacuole: bool = True,
    tail: bool = True,
    status: MorphologyStatus = MorphologyStatus.COMPLETE,
) -> MorphologyResult:
    """Build a result with the given per-aspect normality."""

    def aspect(name: str, normal: bool) -> AspectResult:
        return AspectResult(name=name, p_normal=0.9 if normal else 0.1, threshold=0.5)

    return MorphologyResult(
        track_id=track_id,
        status=status,
        head=aspect("head", head),
        acrosome=aspect("acrosome", acrosome),
        vacuole=aspect("vacuole", vacuole),
        tail=aspect("tail", tail),
    )


def make_motion(motility: MotilityClass) -> MotionFeatures:
    return MotionFeatures(
        n_points=20,
        n_observed_points=20,
        duration_s=0.125,
        mean_frame_interval_s=1 / 160,
        timestamp_source=TimestampSource.SYNTHETIC,
        flow_correction_mode=FlowCorrectionMode.FIXED_VECTOR,
        optically_calibrated=True,
        um_per_px=0.0345,
        motility_class=motility,
    )


def eligible_track(track_id: int = 1) -> TrackRecord:
    """A track that satisfies every condition. The baseline for the tests."""
    track = make_track(track_id)
    track.track_quality_pass = True
    track.motion = make_motion(MotilityClass.RAPID_PROGRESSIVE)
    track.morphology = make_morphology(track_id)
    track.evaluation_complete = True
    return track


# --------------------------------------------------------------------------
# The all-four rule
# --------------------------------------------------------------------------


def test_all_four_normal_requires_all_four() -> None:
    assert make_morphology().all_four_normal is True
    for aspect in MORPHOLOGY_ASPECTS:
        result = make_morphology(**{aspect: False})
        assert result.all_four_normal is False, f"{aspect} abnormal must fail"
        assert result.first_abnormal_aspect() == aspect


def test_all_four_is_not_an_average() -> None:
    """Three excellent aspects cannot outvote one abnormal one.

    An averaged score of 0.99/0.99/0.99/0.02 is 0.75 and would pass any
    sensible mean threshold. The rule is conjunctive precisely so that it
    cannot.
    """
    result = MorphologyResult(
        track_id=1,
        status=MorphologyStatus.COMPLETE,
        head=AspectResult(name="head", p_normal=0.99, threshold=0.5),
        acrosome=AspectResult(name="acrosome", p_normal=0.99, threshold=0.5),
        vacuole=AspectResult(name="vacuole", p_normal=0.99, threshold=0.5),
        tail=AspectResult(name="tail", p_normal=0.02, threshold=0.5),
    )
    mean_p = sum(a.p_normal for a in result.aspects) / 4  # type: ignore[union-attr]
    assert mean_p > 0.7, "the mean would pass a mean-based rule"
    assert result.all_four_normal is False


def test_missing_aspect_is_not_normal() -> None:
    """An absent aspect must never read as normal."""
    result = make_morphology()
    result.tail = None
    assert result.is_complete is False
    assert result.all_four_normal is False
    assert result.first_abnormal_aspect() == "tail"


def test_incomplete_status_is_not_normal() -> None:
    for status in (
        MorphologyStatus.DEADLINE_MISSED,
        MorphologyStatus.NO_VALID_CROP,
        MorphologyStatus.INFERENCE_FAILED,
        MorphologyStatus.NOT_REQUIRED,
    ):
        result = make_morphology(status=status)
        assert result.all_four_normal is False, f"{status} must not be normal"


def test_label_polarity_is_mhsma_convention() -> None:
    """0 is normal and 1 is abnormal, on the schema as well as the dataset."""
    assert LABEL_NORMAL == 0
    assert LABEL_ABNORMAL == 1
    normal = AspectResult(name="head", p_normal=0.9, threshold=0.5)
    abnormal = AspectResult(name="head", p_normal=0.1, threshold=0.5)
    assert normal.normal is True and normal.label == LABEL_NORMAL == 0
    assert abnormal.normal is False and abnormal.label == LABEL_ABNORMAL == 1


def test_per_aspect_thresholds_are_independent() -> None:
    """Each aspect decides against its own threshold.

    MHSMA abnormality prevalence ranges from ~30% (acrosome) to ~4.6% (tail),
    so a shared threshold is dominated by the majority aspects.
    """
    result = MorphologyResult(
        track_id=1,
        status=MorphologyStatus.COMPLETE,
        head=AspectResult(name="head", p_normal=0.55, threshold=0.50),
        acrosome=AspectResult(name="acrosome", p_normal=0.55, threshold=0.50),
        vacuole=AspectResult(name="vacuole", p_normal=0.55, threshold=0.50),
        # A stricter threshold for the rare class flips this one.
        tail=AspectResult(name="tail", p_normal=0.55, threshold=0.80),
    )
    assert result.head.normal is True  # type: ignore[union-attr]
    assert result.tail.normal is False  # type: ignore[union-attr]
    assert result.all_four_normal is False


# --------------------------------------------------------------------------
# The five-condition eligibility rule
# --------------------------------------------------------------------------


def test_fully_qualifying_track_is_eligible() -> None:
    track = eligible_track()
    assert track.compute_eligibility() is True
    assert track.ai_eligible is True
    assert track.ineligibility_reason is IneligibilityReason.NONE


def test_track_quality_failure_blocks_eligibility() -> None:
    track = eligible_track()
    track.track_quality_pass = False
    assert track.compute_eligibility() is False
    assert track.ineligibility_reason is IneligibilityReason.TRACK_QUALITY_FAIL


@pytest.mark.parametrize(
    ("motility", "reason"),
    [
        (MotilityClass.NON_PROGRESSIVE, IneligibilityReason.NOT_PROGRESSIVE),
        (MotilityClass.IMMOTILE, IneligibilityReason.NOT_PROGRESSIVE),
        (MotilityClass.UNDETERMINED, IneligibilityReason.MOTILITY_UNDETERMINED),
    ],
)
def test_non_progressive_blocks_eligibility(
    motility: MotilityClass, reason: IneligibilityReason
) -> None:
    track = eligible_track()
    track.motion = make_motion(motility)
    assert track.compute_eligibility() is False
    assert track.ineligibility_reason is reason


@pytest.mark.parametrize(
    "motility",
    [MotilityClass.RAPID_PROGRESSIVE, MotilityClass.SLOW_PROGRESSIVE],
)
def test_both_progressive_grades_pass(motility: MotilityClass) -> None:
    """Rapid and slow progressive both satisfy the motility filter.

    WHO 6th ed. reports progressive motility as rapid + slow (a + b), so
    excluding the slow grade would be a departure from the standard, not a
    conservative choice.
    """
    track = eligible_track()
    track.motion = make_motion(motility)
    assert track.compute_eligibility() is True


@pytest.mark.parametrize(
    ("aspect", "reason"),
    [
        ("head", IneligibilityReason.ABNORMAL_HEAD),
        ("acrosome", IneligibilityReason.ABNORMAL_ACROSOME),
        ("vacuole", IneligibilityReason.ABNORMAL_VACUOLE),
        ("tail", IneligibilityReason.ABNORMAL_TAIL),
    ],
)
def test_each_abnormal_aspect_blocks_eligibility(
    aspect: str, reason: IneligibilityReason
) -> None:
    track = eligible_track()
    track.morphology = make_morphology(**{aspect: False})
    assert track.compute_eligibility() is False
    assert track.ineligibility_reason is reason


def test_missing_morphology_blocks_eligibility() -> None:
    track = eligible_track()
    track.morphology = None
    assert track.compute_eligibility() is False
    assert track.ineligibility_reason is IneligibilityReason.MORPHOLOGY_INCOMPLETE


def test_deadline_miss_blocks_eligibility_but_keeps_the_track() -> None:
    """A missed deadline excludes from the numerator, not the denominator.

    The sperm was really there and was really counted; we simply could not
    show that it qualified. Dropping it from the denominator would inflate
    the ratio every time the pipeline fell behind -- the ratio would improve
    precisely when the system was least trustworthy.
    """
    track = eligible_track()
    track.morphology = MorphologyResult.failed(
        track.track_id,
        MorphologyStatus.DEADLINE_MISSED,
        "did not complete in time",
    )
    track.evaluation_complete = False

    assert track.compute_eligibility() is False
    assert track.ineligibility_reason is IneligibilityReason.DEADLINE_MISSED
    # Still a valid, counted observation.
    assert track.track_quality_pass is True


def test_evaluation_incomplete_blocks_eligibility() -> None:
    """Complete morphology with the completion flag unset still fails."""
    track = eligible_track()
    track.evaluation_complete = False
    assert track.compute_eligibility() is False
    assert track.ineligibility_reason is IneligibilityReason.DEADLINE_MISSED


def test_morphology_alone_is_not_enough() -> None:
    """Perfect morphology with no progression must not qualify."""
    track = eligible_track()
    track.motion = make_motion(MotilityClass.IMMOTILE)
    assert track.morphology is not None and track.morphology.all_four_normal
    assert track.compute_eligibility() is False


def test_motility_alone_is_not_enough() -> None:
    """Fast progression with abnormal morphology must not qualify."""
    track = eligible_track()
    track.morphology = make_morphology(head=False)
    assert track.motion is not None and track.motion.is_progressive
    assert track.compute_eligibility() is False


def test_eligibility_is_idempotent() -> None:
    """Recomputing must not change the answer or accumulate state."""
    track = eligible_track()
    first = track.compute_eligibility()
    second = track.compute_eligibility()
    assert first is second is True
    assert track.ineligibility_reason is IneligibilityReason.NONE
