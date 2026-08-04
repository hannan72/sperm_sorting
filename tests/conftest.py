"""Shared fixtures.

Deliberately small. The pipeline's own factories are used to build components
wherever possible, so that a test exercises the real construction path rather
than a parallel one assembled for testing -- a fixture that quietly diverges
from production is a test that passes while the product is broken.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sperm_sorting.actuation.mock import MockActuator
from sperm_sorting.config import (
    ActuationConfig,
    AppConfig,
    DecisionConfig,
    SchedulingConfig,
    ShotConfig,
    load_config,
)
from sperm_sorting.scheduling.clock import ManualClock
from sperm_sorting.scheduling.scheduler import ActuationScheduler


@pytest.fixture
def clock() -> ManualClock:
    """A clock the test advances by hand, so timing assertions are exact."""
    return ManualClock(start=1000.0)


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def calibrated_scheduling() -> SchedulingConfig:
    """Scheduling config with plausible measured values.

    The numbers stand in for a calibration; they are internally consistent so
    that timing assertions mean something, and they are not presented anywhere
    as measurements of a real instrument.
    """
    return SchedulingConfig(
        calibrated=True,
        calibration_id="test-fixture-not-a-real-instrument",
        transport_delay_ms=100.0,
        transport_delay_std_ms=3.0,
        field_rise_time_ms=8.0,
        field_fall_time_ms=6.0,
        pre_activation_margin_ms=9.0,
        post_activation_margin_ms=9.0,
        watchdog_timeout_ms=500.0,
        late_tolerance_ms=5.0,
        drop_if_late_by_ms=50.0,
    )


@pytest.fixture
def actuator(clock: ManualClock) -> Iterator[MockActuator]:
    act = MockActuator(ActuationConfig(kind="mock"), clock=clock)
    act.open()
    yield act
    act.close()


@pytest.fixture
def scheduler(
    calibrated_scheduling: SchedulingConfig,
    clock: ManualClock,
    actuator: MockActuator,
) -> ActuationScheduler:
    sched = ActuationScheduler(
        calibrated_scheduling, clock=clock, dispatch=actuator.apply
    )
    sched.arm()
    return sched


@pytest.fixture
def shot_cfg() -> ShotConfig:
    return ShotConfig()


@pytest.fixture
def decision_cfg() -> DecisionConfig:
    return DecisionConfig()


# Builders live in ``builders.py`` and are re-exported here so that a test may
# reach them either way without two copies drifting apart.
from builders import make_detection, make_frame, make_track  # noqa: E402

__all__ = ["make_detection", "make_frame", "make_track"]
