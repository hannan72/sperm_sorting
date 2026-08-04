"""Real-time scheduling, actuation and the fail-safe paths.

The property that matters here is not throughput but *what state the magnet is
left in*. FIELD_OFF lets the sample flow to collection unsorted; FIELD_ON
diverts it to waste. So every fault must end in FIELD_OFF, and a command that
can no longer reach the fluid it was meant for must be dropped rather than
fired at whatever fluid happens to be passing.
"""

from __future__ import annotations

import pytest

from sperm_sorting.actuation.base import Watchdog
from sperm_sorting.actuation.mock import MockActuator
from sperm_sorting.config import ActuationConfig, SchedulingConfig
from sperm_sorting.errors import ActuatorError, CalibrationError, WatchdogTimeout
from sperm_sorting.schemas.enums import (
    CommandOrigin,
    CommandOutcome,
    FieldCommandKind,
)
from sperm_sorting.scheduling.clock import ManualClock, MonotonicClock, ScaledClock
from sperm_sorting.scheduling.scheduler import ActuationScheduler

ON = FieldCommandKind.FIELD_ON
OFF = FieldCommandKind.FIELD_OFF


def advance_and_poll(
    scheduler: ActuationScheduler,
    clock: ManualClock,
    duration_s: float,
    step_s: float = 1 / 160,
) -> list:
    """Step time forward the way the pipeline does, polling as it goes.

    The scheduler drops a command that is more than ``drop_if_late_by_ms``
    past its dispatch instant, because by then the fluid it was meant for has
    moved on. Jumping the clock forward in one leap therefore makes every
    queued command look catastrophically late -- an artefact of the test, not
    of the scheduler. Stepping at the frame period reproduces the real duty
    cycle, where ``poll`` is called once per frame.
    """
    handled: list = []
    elapsed = 0.0
    while elapsed < duration_s:
        clock.advance(step_s)
        elapsed += step_s
        handled.extend(scheduler.poll())
    return handled


# --------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------


def test_manual_clock_cannot_go_backwards() -> None:
    clock = ManualClock(start=100.0)
    clock.advance(5.0)
    assert clock.now() == 105.0
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1.0)
    with pytest.raises(ValueError, match="backwards"):
        clock.set(50.0)


def test_manual_clock_sleep_advances_instead_of_blocking() -> None:
    clock = ManualClock()
    clock.sleep(3.0)
    assert clock.now() == 3.0


def test_monotonic_clock_never_decreases() -> None:
    clock = MonotonicClock()
    samples = [clock.now() for _ in range(200)]
    assert all(b >= a for a, b in zip(samples, samples[1:], strict=False))


def test_scaled_clock_rescales_intervals() -> None:
    base = ManualClock(start=0.0)
    clock = ScaledClock(scale=2.0, base=base)
    base.advance(10.0)
    assert clock.now() == pytest.approx(20.0)


# --------------------------------------------------------------------------
# Arming
# --------------------------------------------------------------------------


def test_scheduler_refuses_to_arm_uncalibrated() -> None:
    """An unmeasured transport delay would gate the wrong fluid, silently."""
    sched = ActuationScheduler(SchedulingConfig(), clock=ManualClock())
    with pytest.raises(CalibrationError, match="not calibrated"):
        sched.arm()
    assert sched.is_armed is False


def test_bench_override_allows_arming_uncalibrated() -> None:
    cfg = SchedulingConfig(require_calibration_to_actuate=False)
    sched = ActuationScheduler(cfg, clock=ManualClock())
    sched.arm()
    assert sched.is_armed is True


def test_unarmed_scheduler_refuses_to_dispatch(
    calibrated_scheduling: SchedulingConfig, clock: ManualClock, actuator: MockActuator
) -> None:
    sched = ActuationScheduler(
        calibrated_scheduling, clock=clock, dispatch=actuator.apply
    )
    sched.submit(ON, gate_time_s=clock.now())
    clock.advance(1.0)
    handled = sched.poll()
    assert handled[0].outcome is CommandOutcome.FAILED
    assert actuator.state is OFF


# --------------------------------------------------------------------------
# Transport delay and lead time
# --------------------------------------------------------------------------


def test_activation_is_delayed_by_the_transport_time(
    scheduler: ActuationScheduler, clock: ManualClock
) -> None:
    """A decision about imaged fluid applies when that fluid reaches the magnet."""
    gate_time = clock.now()
    command = scheduler.submit(ON, gate_time_s=gate_time, shot_id=1)
    # transport 100 ms
    assert command.activate_at_s == pytest.approx(gate_time + 0.100)
    # dispatched early by rise time (8 ms) + pre-activation margin (9 ms)
    assert command.dispatch_at_s == pytest.approx(gate_time + 0.100 - 0.017)


def test_fall_time_is_used_for_field_off(
    scheduler: ActuationScheduler, clock: ManualClock
) -> None:
    """A collapsing field has its own settling time.

    Using the rise time for both would bias every release in one direction.
    """
    gate_time = clock.now()
    off = scheduler.submit(OFF, gate_time_s=gate_time)
    # fall 6 ms + margin 9 ms
    assert off.dispatch_at_s == pytest.approx(gate_time + 0.100 - 0.015)


def test_command_is_not_dispatched_early(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    scheduler.submit(ON, gate_time_s=clock.now())
    assert scheduler.poll() == []
    assert actuator.state is OFF

    clock.advance(0.082)  # just before dispatch at +83 ms
    assert scheduler.poll() == []

    clock.advance(0.002)
    handled = scheduler.poll()
    assert len(handled) == 1
    assert actuator.state is ON


# --------------------------------------------------------------------------
# Ordering, lateness and dropping
# --------------------------------------------------------------------------


def test_commands_dispatch_in_time_order(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    """Submission order must not determine dispatch order."""
    t0 = clock.now()
    scheduler.submit(OFF, gate_time_s=t0 + 0.30)
    scheduler.submit(ON, gate_time_s=t0 + 0.10)
    scheduler.submit(OFF, gate_time_s=t0 + 0.20)

    handled = advance_and_poll(scheduler, clock, 0.6)
    assert [c.kind for c in handled] == [ON, OFF, OFF]


def test_late_command_is_marked_but_still_fires(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    """Slightly late is still useful; it is recorded, not discarded."""
    scheduler.submit(ON, gate_time_s=clock.now())
    clock.advance(0.083 + 0.010)  # 10 ms past dispatch, tolerance is 5 ms
    handled = scheduler.poll()
    assert handled[0].outcome is CommandOutcome.LATE
    assert scheduler.n_late == 1
    assert actuator.state is ON


def test_very_late_command_is_dropped(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    """Past the drop limit the target fluid has gone; firing would divert
    an unrelated segment, which is worse than not acting at all."""
    scheduler.submit(ON, gate_time_s=clock.now())
    clock.advance(0.083 + 0.200)  # far past the 50 ms drop limit
    handled = scheduler.poll()
    assert handled[0].outcome is CommandOutcome.FAILED
    assert "wrong fluid segment" in handled[0].failure_reason
    assert scheduler.n_dropped_late == 1
    assert actuator.state is OFF, "the magnet must not fire for stale fluid"


def test_superseded_command_does_not_fire(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    t0 = clock.now()
    scheduler.submit(ON, gate_time_s=t0)
    n = scheduler.supersede(before_time_s=t0 + 1.0)
    assert n == 1
    advance_and_poll(scheduler, clock, 0.2)
    assert actuator.state is OFF


def test_timing_percentiles_are_reported(
    scheduler: ActuationScheduler, clock: ManualClock
) -> None:
    for i in range(20):
        scheduler.submit(ON if i % 2 else OFF, gate_time_s=clock.now() + i * 0.001)
    advance_and_poll(scheduler, clock, 0.5)
    timing = scheduler.timing_percentiles()
    assert timing["p50_ms"] is not None
    assert timing["p99_ms"] >= timing["p50_ms"]


# --------------------------------------------------------------------------
# The mandated command sequence
# --------------------------------------------------------------------------


def test_consecutive_command_sequence(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    """OFF, ON, OFF, OFF, ON must reach the actuator in that order.

    The repeated OFF is deliberate: re-asserting a state the field is already
    in must be harmless and must not reorder or coalesce the surrounding
    commands.
    """
    wanted = [OFF, ON, OFF, OFF, ON]
    t0 = clock.now()
    for i, kind in enumerate(wanted):
        scheduler.submit(kind, gate_time_s=t0 + i * 0.5, shot_id=i)

    advance_and_poll(scheduler, clock, 3.0)

    # The mock records an initial OFF at open(); drop it before comparing.
    recorded = [str(k) for k in actuator.state_sequence()][1:]
    assert recorded == [str(k) for k in wanted]
    assert scheduler.n_dispatched == 5
    assert scheduler.n_dropped_late == 0


# --------------------------------------------------------------------------
# Actuator faults
# --------------------------------------------------------------------------


def test_actuator_opens_in_the_safe_state(clock: ManualClock) -> None:
    act = MockActuator(ActuationConfig(), clock=clock)
    act.open()
    assert act.state is OFF
    act.close()


def test_actuator_closes_in_the_safe_state(clock: ManualClock) -> None:
    act = MockActuator(ActuationConfig(), clock=clock)
    act.open()
    act._write(ON)
    act._state = ON
    act.close()
    assert act.state is OFF


def test_failed_write_forces_the_safe_state(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    actuator.fail_writes = True
    scheduler.submit(ON, gate_time_s=clock.now())
    handled = advance_and_poll(scheduler, clock, 0.12)
    assert handled[0].outcome is CommandOutcome.FAILED
    assert actuator.state is OFF


def test_acknowledgement_mismatch_forces_the_safe_state(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    """A relay that accepts commands but does not switch must be caught."""
    actuator.report_wrong_state = True
    scheduler.submit(ON, gate_time_s=clock.now())
    handled = advance_and_poll(scheduler, clock, 0.12)
    assert handled[0].outcome is CommandOutcome.FAILED
    assert "acknowledgement mismatch" in handled[0].failure_reason
    assert actuator.state is OFF
    assert actuator.n_ack_failures == 1


def test_actuator_without_feedback_is_reported_as_such(clock: ManualClock) -> None:
    """An actuator that cannot report its state must not be assumed good."""
    act = MockActuator(ActuationConfig(), clock=clock, can_acknowledge=False)
    act.open()
    assert act.describe()["can_acknowledge"] is False
    act.close()


def test_applying_to_a_closed_actuator_raises(clock: ManualClock) -> None:
    from sperm_sorting.schemas.command import FieldCommand

    act = MockActuator(ActuationConfig(), clock=clock)
    command = FieldCommand(
        command_id=0,
        kind=ON,
        origin=CommandOrigin.MANUAL,
        activate_at_s=0.0,
        dispatch_at_s=0.0,
        deadline_s=0.0,
    )
    with pytest.raises(ActuatorError, match="not open"):
        act.apply(command)


def test_safe_state_never_raises(clock: ManualClock) -> None:
    """The fault path must survive a broken actuator."""
    act = MockActuator(ActuationConfig(), clock=clock)
    act.open()
    act.fail_writes = True
    act.safe_state()  # must not raise
    act.close()


# --------------------------------------------------------------------------
# Watchdog
# --------------------------------------------------------------------------


def test_watchdog_does_not_trip_while_fed(
    actuator: MockActuator, clock: ManualClock
) -> None:
    dog = Watchdog(actuator, timeout_ms=500.0, clock=clock)
    for _ in range(20):
        clock.advance(0.1)
        dog.feed()
        assert dog.check() is False
    assert dog.is_tripped is False


def test_watchdog_trips_and_forces_field_off(
    actuator: MockActuator, clock: ManualClock
) -> None:
    """A hung pipeline must not leave the magnet energised.

    This is the case a crash does not cover: the process is alive, nothing
    raises, and without the watchdog the field would stay wherever the last
    decision left it, applied to fluid nobody is analysing.
    """
    actuator._write(ON)
    actuator._state = ON
    dog = Watchdog(actuator, timeout_ms=500.0, clock=clock)

    clock.advance(0.4)
    assert dog.check() is False
    assert actuator.state is ON

    clock.advance(0.2)  # 600 ms total, past the 500 ms timeout
    assert dog.check() is True
    assert actuator.state is OFF
    assert dog.n_trips == 1


def test_watchdog_trips_only_once_per_episode(
    actuator: MockActuator, clock: ManualClock
) -> None:
    dog = Watchdog(actuator, timeout_ms=100.0, clock=clock)
    clock.advance(0.5)
    assert dog.check() is True
    for _ in range(5):
        clock.advance(0.5)
        assert dog.check() is False
    assert dog.n_trips == 1


def test_watchdog_recovers_when_fed_again(
    actuator: MockActuator, clock: ManualClock
) -> None:
    dog = Watchdog(actuator, timeout_ms=100.0, clock=clock)
    clock.advance(0.5)
    dog.check()
    assert dog.is_tripped is True

    dog.feed()
    assert dog.is_tripped is False
    clock.advance(0.05)
    assert dog.check() is False


def test_watchdog_check_or_raise(actuator: MockActuator, clock: ManualClock) -> None:
    dog = Watchdog(actuator, timeout_ms=100.0, clock=clock)
    clock.advance(0.5)
    with pytest.raises(WatchdogTimeout):
        dog.check_or_raise()
    # The field is off before the exception propagates.
    assert actuator.state is OFF


def test_watchdog_grace_runs_from_construction(
    actuator: MockActuator, clock: ManualClock
) -> None:
    """A watchdog must not trip on the instant it is created."""
    dog = Watchdog(actuator, timeout_ms=500.0, clock=clock)
    assert dog.check() is False


# --------------------------------------------------------------------------
# Flush
# --------------------------------------------------------------------------


def test_flush_discards_queued_commands(
    scheduler: ActuationScheduler, clock: ManualClock, actuator: MockActuator
) -> None:
    """At shutdown, queued commands describe fluid that has already passed."""
    for i in range(5):
        scheduler.submit(ON, gate_time_s=clock.now() + i)
    discarded = scheduler.flush()
    assert len(discarded) == 5
    assert all(c.outcome is CommandOutcome.SUPERSEDED for c in discarded)

    clock.advance(100.0)
    assert scheduler.poll() == []
    assert actuator.state is OFF
