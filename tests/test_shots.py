"""Shot assembly: the counting gate, closure conditions and the denominator.

The invariant under test throughout is that one physical sperm is counted
exactly once. Everything else in this file is a consequence of that.
"""

from __future__ import annotations

import pytest

from sperm_sorting.config import CountingGateConfig, ShotConfig
from sperm_sorting.schemas.enums import (
    IneligibilityReason,
    MorphologyStatus,
    ShotCloseReason,
)
from sperm_sorting.schemas.morphology import MorphologyResult
from sperm_sorting.schemas.track import TrackPoint, TrackRecord
from sperm_sorting.shots.gate import CountingGate
from sperm_sorting.shots.manager import ShotManager

from builders import make_track

ROI_W, ROI_H = 1000, 500


def gate(**kwargs: object) -> CountingGate:
    cfg = CountingGateConfig(**kwargs)  # type: ignore[arg-type]
    return CountingGate(cfg, ROI_W, ROI_H)


def walk(g: CountingGate, track: TrackRecord, xs: list[float]) -> list[int]:
    """Step a track through a series of x positions; return crossing frames."""
    crossings: list[int] = []
    for i, x in enumerate(xs):
        from sperm_sorting.schemas.detection import BoundingBox

        track.add_point(
            TrackPoint(
                frame_id=i,
                capture_time_s=i / 160,
                box=BoundingBox.from_cxcywh(x, 250.0, 20.0, 14.0),
                score=0.9,
            )
        )
        crossing = g.update(track)
        if crossing is not None:
            crossings.append(crossing.frame_id)
    return crossings


# --------------------------------------------------------------------------
# Counting gate
# --------------------------------------------------------------------------


def test_gate_counts_a_crossing_once() -> None:
    g = gate(position_fraction=0.5, direction=1, min_axis_displacement_px=5.0)
    track = TrackRecord(track_id=1)
    crossings = walk(g, track, [100.0, 300.0, 480.0, 520.0, 700.0, 900.0])
    assert len(crossings) == 1
    assert g.n_crossings == 1


def test_gate_never_counts_the_same_track_twice() -> None:
    """A track loitering on the line must not be counted repeatedly.

    This is the failure mode that would silently multiply the denominator and
    make every shot look worse than it is.
    """
    g = gate(position_fraction=0.5, direction=1, min_axis_displacement_px=5.0)
    track = TrackRecord(track_id=1)
    # Cross, come back, cross again, several times.
    xs = [400.0, 600.0, 400.0, 600.0, 400.0, 600.0, 900.0]
    walk(g, track, xs)
    assert g.n_crossings == 1
    assert g.has_crossed(1)


def test_gate_ignores_wrong_direction() -> None:
    """A sperm swimming upstream against the flow is not counted."""
    g = gate(position_fraction=0.5, direction=1, min_axis_displacement_px=5.0)
    track = TrackRecord(track_id=1)
    walk(g, track, [900.0, 700.0, 520.0, 480.0, 300.0, 100.0])
    assert g.n_crossings == 0
    assert g.n_rejected_wrong_direction > 0


def test_gate_requires_lifetime_displacement() -> None:
    """Jitter across the line does not qualify as transit."""
    g = gate(position_fraction=0.5, direction=1, min_axis_displacement_px=50.0)
    track = TrackRecord(track_id=1)
    walk(g, track, [498.0, 499.0, 501.0, 502.0])
    assert g.n_crossings == 0
    assert g.n_rejected_insufficient_displacement > 0


def test_gate_forget_does_not_forget_the_crossing() -> None:
    """Dropping a finished track's cache must not re-arm it for counting."""
    g = gate(position_fraction=0.5, direction=1, min_axis_displacement_px=5.0)
    track = TrackRecord(track_id=1)
    walk(g, track, [100.0, 400.0, 600.0, 900.0])
    assert g.n_crossings == 1
    g.forget(1)
    assert g.has_crossed(1) is True


def test_gate_on_y_axis() -> None:
    g = gate(axis="y", position_fraction=0.5, direction=1, min_axis_displacement_px=5.0)
    from sperm_sorting.schemas.detection import BoundingBox

    track = TrackRecord(track_id=1)
    crossings = 0
    for i, y in enumerate([50.0, 150.0, 240.0, 260.0, 400.0]):
        track.add_point(
            TrackPoint(
                frame_id=i,
                capture_time_s=i / 160,
                box=BoundingBox.from_cxcywh(500.0, y, 20.0, 14.0),
                score=0.9,
            )
        )
        if g.update(track) is not None:
            crossings += 1
    assert crossings == 1


# --------------------------------------------------------------------------
# Shot closure
# --------------------------------------------------------------------------


def gated(manager: ShotManager, n: int, *, t0: float = 0.0, dt: float = 0.01) -> None:
    for i in range(n):
        track = make_track(i, quality_pass=True)
        manager.add_track(track, t0 + i * dt, i)


def test_shot_closes_at_target_count() -> None:
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    gated(manager, 25)
    assert manager.open_shot is None
    history = manager.history
    assert len(history) == 1
    assert history[0].close_reason is ShotCloseReason.TARGET_REACHED
    assert history[0].trackable_count == 25


def test_shot_closes_at_hard_maximum_when_target_raised() -> None:
    cfg = ShotConfig(target_trackable_sperm=30, maximum_trackable_sperm=30)
    manager = ShotManager(cfg, morphology_deadline_s=0.25)
    gated(manager, 30)
    assert manager.history[0].close_reason in (
        ShotCloseReason.HARD_MAXIMUM,
        ShotCloseReason.TARGET_REACHED,
    )
    assert manager.history[0].trackable_count == 30


def test_shot_never_exceeds_the_hard_maximum() -> None:
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    gated(manager, 100)
    for shot in manager.history:
        assert shot.trackable_count <= 30


def test_shot_closes_on_timeout() -> None:
    """A shot must close after one second even if it is under-filled."""
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    for i in range(5):
        manager.add_track(make_track(i), i * 0.01, i)
    assert manager.open_shot is not None

    closed = manager.poll(now_s=1.5, frame_id=999)
    assert closed is not None
    assert closed.close_reason is ShotCloseReason.TIMEOUT
    assert closed.trackable_count == 5


def test_timeout_is_measured_from_the_shot_opening() -> None:
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    manager.add_track(make_track(0), 10.0, 0)
    assert manager.poll(10.5, 1) is None      # 0.5 s in: still open
    assert manager.poll(10.99, 2) is None     # just under the limit
    assert manager.poll(11.0, 3) is not None  # exactly at the limit closes


def test_empty_channel_does_not_open_a_shot() -> None:
    """Polling with nothing gated must not manufacture empty shots."""
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    for i in range(100):
        assert manager.poll(i * 0.1, i) is None
    assert manager.history == []


# --------------------------------------------------------------------------
# The denominator
# --------------------------------------------------------------------------


def test_low_quality_track_is_excluded_from_both_counts() -> None:
    """A track that fails the quality bar is not a trustworthy observation.

    It goes in neither the numerator nor the denominator, but it is counted
    separately so an operator can see how much of the field is discarded.
    """
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    for i in range(10):
        manager.add_track(make_track(i, quality_pass=True), i * 0.01, i)
    for i in range(10, 15):
        manager.add_track(make_track(i, quality_pass=False), i * 0.01, i)

    shot = manager.open_shot
    assert shot is not None
    assert shot.trackable_count == 10
    assert shot.rejected_track_count == 5


def test_duplicate_assignment_is_refused() -> None:
    """One track cannot enter two shots, or the same shot twice."""
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    track = make_track(42)
    manager.add_track(track, 0.0, 0)
    manager.add_track(track, 0.1, 10)

    assert manager.open_shot is not None
    assert manager.open_shot.trackable_count == 1
    assert manager.n_duplicate_assignments_rejected == 1


def test_shot_record_refuses_duplicate_track_ids() -> None:
    from sperm_sorting.schemas.shot import ShotRecord

    shot = ShotRecord(shot_id=0, opened_at_s=0.0, opened_frame_id=0)
    assert shot.add_track(5) is True
    assert shot.add_track(5) is False
    assert shot.trackable_count == 1


def test_gate_span_covers_the_fluid_segment() -> None:
    """A shot's first and last crossings delimit the fluid it describes."""
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    for i in range(5):
        manager.add_track(make_track(i), 100.0 + i * 0.05, i)
    shot = manager.open_shot
    assert shot is not None
    assert shot.first_gate_time_s == pytest.approx(100.0)
    assert shot.last_gate_time_s == pytest.approx(100.20)
    assert shot.gate_span_s == pytest.approx(0.20)


# --------------------------------------------------------------------------
# Finalisation
# --------------------------------------------------------------------------


def resolved_track(track_id: int, *, eligible: bool) -> TrackRecord:
    from test_eligibility_rule import eligible_track, make_morphology

    track = eligible_track(track_id)
    if not eligible:
        track.morphology = make_morphology(track_id, head=False)
    return track


def test_shot_becomes_ready_when_all_members_resolve() -> None:
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    tracks = {}
    for i in range(25):
        track = resolved_track(i, eligible=i < 16)
        tracks[i] = track
        manager.add_track(track, i * 0.01, i)

    assert manager.pending_count() == 1
    assert manager.ready_shots(now_s=0.30, tracks_by_id=tracks) == []

    for i in range(25):
        manager.notify_track_resolved(i)

    ready = manager.ready_shots(now_s=0.30, tracks_by_id=tracks)
    assert len(ready) == 1
    assert ready[0].trackable_count == 25
    assert ready[0].ai_eligible_count == 16


def test_unresolved_members_at_deadline_stay_in_the_denominator() -> None:
    """The deadline excludes from the numerator, never from the denominator."""
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    tracks = {}
    for i in range(25):
        track = resolved_track(i, eligible=True)
        if i >= 20:
            # Genuinely still in flight: a track whose morphology has not
            # finished has no result yet. This mirrors the pipeline, where
            # the result and the resolution notification are set together.
            track.morphology = None
            track.evaluation_complete = False
        tracks[i] = track
        manager.add_track(track, i * 0.01, i)

    # Only 20 of the 25 resolve in time.
    for i in range(20):
        manager.notify_track_resolved(i)

    closed_at = manager.history[0].closed_at_s or 0.0
    ready = manager.ready_shots(now_s=closed_at + 1.0, tracks_by_id=tracks)

    assert len(ready) == 1
    shot = ready[0]
    assert shot.trackable_count == 25, "denominator must not shrink"
    assert shot.ai_eligible_count == 20
    assert (
        shot.ineligibility_histogram.get(str(IneligibilityReason.DEADLINE_MISSED)) == 5
    )
    for i in range(20, 25):
        assert tracks[i].morphology is not None
        assert tracks[i].morphology.status is MorphologyStatus.DEADLINE_MISSED


def test_missing_member_record_stays_in_the_denominator() -> None:
    """A member we can no longer inspect cannot be shown eligible.

    Dropping it would inflate the ratio by shrinking the divisor, so it is
    kept and recorded as incomplete instead.
    """
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    tracks = {}
    for i in range(25):
        track = resolved_track(i, eligible=True)
        if i < 24:
            tracks[i] = track
        manager.add_track(track, i * 0.01, i)
        manager.notify_track_resolved(i)

    ready = manager.ready_shots(now_s=1.0, tracks_by_id=tracks)
    shot = ready[0]
    assert shot.trackable_count == 25
    assert shot.ai_eligible_count == 24
    assert (
        shot.ineligibility_histogram.get(
            str(IneligibilityReason.MORPHOLOGY_INCOMPLETE)
        )
        == 1
    )


def test_flush_on_shutdown_decides_a_partial_shot() -> None:
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.25)
    tracks = {}
    for i in range(7):
        track = resolved_track(i, eligible=True)
        tracks[i] = track
        manager.add_track(track, i * 0.01, i)

    manager.flush(now_s=0.5, frame_id=99)
    drained = manager.drain_pending(now_s=0.5, tracks_by_id=tracks)

    assert len(drained) == 1
    assert drained[0].close_reason is ShotCloseReason.SHUTDOWN
    assert drained[0].trackable_count == 7


def test_pending_shots_do_not_leak() -> None:
    """Every finalised shot must leave the pending list."""
    manager = ShotManager(ShotConfig(), morphology_deadline_s=0.01)
    tracks = {}
    for shot_index in range(4):
        for i in range(25):
            tid = shot_index * 100 + i
            track = resolved_track(tid, eligible=True)
            tracks[tid] = track
            manager.add_track(track, shot_index + i * 0.001, tid)
            manager.notify_track_resolved(tid)
        manager.ready_shots(now_s=shot_index + 10.0, tracks_by_id=tracks)

    assert manager.pending_count() == 0
    assert len(manager.history) == 4
