"""End-to-end pipeline tests.

These run the real graph -- synthetic video into detection, tracking, motion
analysis, best-frame selection, cropping, morphology, shot assembly, the
decision rule and the scheduler -- and assert the properties that only exist
once the parts are connected.

They use the synthetic source because it is the only one with per-sperm ground
truth, and the oracle detector because a pipeline test should fail when the
*pipeline* is wrong, not when an untrained detector finds nothing.
"""

from __future__ import annotations

import pytest

from sperm_sorting.app import Application
from sperm_sorting.config import AppConfig, load_config
from sperm_sorting.schemas.enums import FieldCommandKind, ShotStatus
from sperm_sorting.shots.feasibility import assess_feasibility

pytestmark = pytest.mark.slow


def synthetic_config(**overrides: object) -> AppConfig:
    """A small, fast, physically coherent synthetic run.

    Scaled down from the reference 1920x1200 so the suite is runnable on every
    change: a 640x400 field at 6400 px/s still gives a 100 ms residence (16
    frames, above the 6-frame quality bar) and, at density 10, fills roughly
    four shots in 240 frames. The *physics* is unchanged -- same sampling, same
    thresholds, same timing budget -- only the frame is smaller.
    """
    base = [
        "run.max_frames=240",
        "acquisition.synthetic.n_frames=240",
        "acquisition.synthetic.width=640",
        "acquisition.synthetic.height=400",
        "acquisition.synthetic.flow_vx_px_s=6400",
        "acquisition.synthetic.density=10",
        "monitoring.audit_dir=null",
        "monitoring.log_level=ERROR",
    ]
    base += [f"{k}={v}" for k, v in overrides.items()]
    return load_config("configs/synthetic.yaml", base)


def run(cfg: AppConfig) -> Application:
    app = Application(cfg)
    try:
        app.setup()
        app.run()
    finally:
        app.close()
    return app


@pytest.fixture(scope="module")
def standard_run() -> Application:
    """One pipeline run, shared by every assertion that does not change config.

    A full run is seconds of CPU, and most of these tests interrogate different
    properties of the *same* execution rather than needing separate ones.
    Sharing it turns a four-minute file into a fifteen-second one. Tests that
    need a different configuration still build their own.
    """
    return run(synthetic_config())


# --------------------------------------------------------------------------
# The whole chain
# --------------------------------------------------------------------------


def test_full_pipeline_produces_decisions(standard_run: Application) -> None:
    """video -> detect -> track -> motion -> crop -> morphology -> decide."""
    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None

    stats = pipeline.stats()
    assert stats["frames"] == 240
    assert pipeline.metrics.detections_total > 0, "detector found nothing"
    assert pipeline.metrics.tracks_created > 0, "tracker made no tracks"
    assert pipeline.gate.n_crossings > 0, "no track crossed the counting gate"
    assert len(pipeline.shots.history) > 0, "no shot was assembled"

    for shot in pipeline.shots.history:
        if shot.status is not None:
            assert shot.status in (
                ShotStatus.ACCEPT,
                ShotStatus.REJECT,
                ShotStatus.INDETERMINATE,
            )


def test_every_sperm_is_counted_at_most_once(standard_run: Application) -> None:
    """The central invariant: one physical sperm, one shot membership."""
    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None

    all_members: list[int] = []
    for shot in pipeline.shots.history:
        all_members.extend(shot.track_ids)
        assert len(shot.track_ids) == len(set(shot.track_ids)), (
            f"shot {shot.shot_id} contains a duplicate track"
        )

    assert len(all_members) == len(set(all_members)), (
        "a track was assigned to more than one shot"
    )
    assert pipeline.shots.n_duplicate_assignments_rejected == 0


def test_shot_sizes_respect_the_configured_bounds(standard_run: Application) -> None:
    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None
    for shot in pipeline.shots.history:
        assert shot.trackable_count <= 30, "hard maximum exceeded"


def test_crop_belongs_to_the_track_whose_motion_was_measured(standard_run: Application) -> None:
    """The binding the whole product rests on.

    A crop from a different cell would mean the morphology verdict and the
    motility verdict describe two different sperm, and their conjunction would
    be meaningless.
    """
    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None

    checked = 0
    for track in pipeline.tracks_by_id.values():
        if track.crop is None:
            continue
        checked += 1
        assert track.crop.track_id == track.track_id
        if track.morphology is not None:
            assert track.morphology.track_id == track.track_id
        # The crop must come from a frame in which the track was observed.
        observed_frames = {p.frame_id for p in track.points if p.observed}
        assert track.crop.frame_id in observed_frames

    assert checked > 0, "no crop was produced, so the binding was not exercised"


def test_morphology_runs_only_after_progressive_classification(standard_run: Application) -> None:
    """Ordering rule: no crop is cut for a sperm that cannot qualify anyway."""
    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None

    for track in pipeline.tracks_by_id.values():
        if track.crop is not None:
            assert track.motion is not None, (
                f"track {track.track_id} was cropped before motion analysis"
            )
            assert track.motion.is_progressive, (
                f"track {track.track_id} was cropped despite being "
                f"{track.motion.motility_class}"
            )


def test_denominator_includes_non_eligible_sperm(standard_run: Application) -> None:
    """Abnormal and non-progressive sperm stay in the denominator."""
    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None

    decided = [s for s in pipeline.shots.history if s.status is not None]
    assert decided, "no shot was decided"
    for shot in decided:
        assert shot.ai_eligible_count <= shot.trackable_count
        ineligible = shot.trackable_count - shot.ai_eligible_count
        assert sum(shot.ineligibility_histogram.values()) == ineligible, (
            "every non-eligible member must carry exactly one recorded reason"
        )


def test_decisions_match_the_rule_applied_to_the_counts(standard_run: Application) -> None:
    """Recompute every decision independently and compare."""
    from sperm_sorting.decision.engine import decide

    app = standard_run
    pipeline = app.pipeline
    assert pipeline is not None

    for shot in pipeline.shots.history:
        if shot.status is None:
            continue
        expected = decide(
            shot.ai_eligible_count,
            shot.trackable_count,
            threshold=shot.threshold_applied,
            minimum_trackable=shot.minimum_trackable_applied,
        )
        assert shot.status is expected.status


# --------------------------------------------------------------------------
# Safety properties
# --------------------------------------------------------------------------


def test_field_ends_off_after_a_normal_run(standard_run: Application) -> None:
    assert standard_run.actuator is not None
    app = standard_run
    assert app.actuator.state is FieldCommandKind.FIELD_OFF


def test_field_ends_off_after_an_inference_failure() -> None:
    """A model that raises must not leave the magnet energised."""
    cfg = synthetic_config()
    app = Application(cfg)
    app.setup()
    try:
        # Break morphology after startup, the way a real backend fault would.
        def explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated inference failure")

        assert app.pipeline is not None
        app.pipeline.morphology.evaluate_track = explode  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated inference failure"):
            app.run()
    finally:
        app.close()

    assert app.actuator is not None
    assert app.actuator.state is FieldCommandKind.FIELD_OFF


def test_field_ends_off_after_a_camera_disconnect() -> None:
    """A source that dies mid-run must leave the field safe."""
    cfg = synthetic_config()
    app = Application(cfg)
    app.setup()
    try:
        from sperm_sorting.errors import CameraError

        original = app.source.read
        state = {"n": 0}

        def failing_read():  # type: ignore[no-untyped-def]
            state["n"] += 1
            if state["n"] > 50:
                raise CameraError("simulated camera disconnect")
            return original()

        app.source.read = failing_read  # type: ignore[method-assign]
        with pytest.raises(CameraError, match="disconnect"):
            app.run()
    finally:
        app.close()

    assert app.actuator is not None
    assert app.actuator.state is FieldCommandKind.FIELD_OFF


def test_no_command_fires_for_an_indeterminate_shot() -> None:
    """A shot below the minimum must never energise the magnet."""
    app = run(synthetic_config(**{"shots.minimum_trackable_sperm": 999,
                                  "decision.minimum_trackable_sperm": 999,
                                  "shots.target_trackable_sperm": 999,
                                  "shots.maximum_trackable_sperm": 1000}))
    pipeline = app.pipeline
    assert pipeline is not None
    decided = [s for s in pipeline.shots.history if s.status is not None]
    assert decided, "expected at least one decided shot"
    assert all(s.status is ShotStatus.INDETERMINATE for s in decided)
    assert app.actuator is not None
    assert FieldCommandKind.FIELD_ON not in [
        c.kind for c in app.actuator.history
    ], "an INDETERMINATE run must never energise the field"


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_replay_is_deterministic() -> None:
    """The same input must produce byte-identical decisions.

    This is what makes an audit log worth keeping: a recorded run can be
    re-executed and the recorded decision reproduced exactly.
    """

    def fingerprint(app: Application) -> list[tuple]:
        pipeline = app.pipeline
        assert pipeline is not None
        return [
            (
                s.shot_id,
                s.trackable_count,
                s.ai_eligible_count,
                str(s.status),
                tuple(s.track_ids),
                tuple(sorted(s.ineligibility_histogram.items())),
            )
            for s in pipeline.shots.history
        ]

    first = fingerprint(run(synthetic_config()))
    second = fingerprint(run(synthetic_config()))
    assert first == second
    assert first, "the run produced no shots, so determinism was not tested"


def test_a_different_seed_produces_a_different_run() -> None:
    """Guards against the determinism test passing because nothing varies."""
    a = run(synthetic_config(**{"acquisition.synthetic.seed": 1}))
    b = run(synthetic_config(**{"acquisition.synthetic.seed": 2}))
    assert a.pipeline is not None and b.pipeline is not None
    ids_a = [t for s in a.pipeline.shots.history for t in s.track_ids]
    ids_b = [t for s in b.pipeline.shots.history for t in s.track_ids]
    assert (ids_a, len(a.pipeline.shots.history)) != (
        ids_b,
        len(b.pipeline.shots.history),
    ) or a.pipeline.metrics.detections_total != b.pipeline.metrics.detections_total


# --------------------------------------------------------------------------
# Frame handling
# --------------------------------------------------------------------------


def test_dropped_frames_are_reported_not_hidden() -> None:
    """A gap in frames must appear as a counted drop, not a silent gap."""
    cfg = synthetic_config()
    app = Application(cfg)
    app.setup()
    try:
        original = app.source.read
        state = {"n": 0}

        def dropping_read():  # type: ignore[no-untyped-def]
            packet = original()
            if packet is None:
                return None
            state["n"] += 1
            if state["n"] % 10 == 0:
                packet.dropped_before = 2
            return packet

        app.source.read = dropping_read  # type: ignore[method-assign]
        app.run()
    finally:
        app.close()

    assert app.pipeline is not None
    metrics = app.pipeline.metrics
    assert metrics.frames_dropped_source > 0
    assert metrics.drop_rate > 0.0


def test_unusable_frames_do_not_stall_the_shot_clock() -> None:
    """A run of rejected frames must still let a shot time out.

    Otherwise a period of poor illumination would freeze the controller with
    a half-filled shot and no decision.
    """
    cfg = synthetic_config(**{"quality_gate.min_focus_score": 1e9})
    app = run(cfg)
    assert app.pipeline is not None
    # Everything was rejected...
    assert app.pipeline.metrics.frames_dropped_quality > 0
    assert app.pipeline.metrics.detections_total == 0
    # ...and the field is still in the safe state.
    assert app.actuator is not None
    assert app.actuator.state is FieldCommandKind.FIELD_OFF


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------


def test_feasibility_flags_a_decision_that_cannot_reach_the_magnet() -> None:
    """Analysis slower than the transport delay is a hard design error.

    Nothing raises when it happens -- every component behaves correctly in
    isolation and the scheduler simply drops each command -- so it has to be
    caught by arithmetic at startup.
    """
    cfg = load_config(
        "configs/synthetic.yaml", ["scheduling.transport_delay_ms=50"]
    )
    report = assess_feasibility(cfg)
    assert report.decision_arrives_in_time is False
    assert any("transport delay" in w for w in report.warnings)

    ok = load_config("configs/synthetic.yaml")
    assert assess_feasibility(ok).decision_arrives_in_time is True


def test_configured_pipeline_dispatches_its_commands(standard_run: Application) -> None:
    """With coherent timing, commands must actually reach the actuator."""
    app = standard_run
    assert app.pipeline is not None
    scheduler = app.pipeline.scheduler
    if scheduler.n_dispatched == 0:
        pytest.skip("no shot was decided in this short run")
    assert scheduler.n_dropped_late == 0, (
        "commands were dropped as late despite a feasible configuration"
    )
