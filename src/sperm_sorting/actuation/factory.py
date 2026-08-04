"""Actuator construction."""

from __future__ import annotations

from ..config import ActuationConfig
from ..errors import ConfigurationError
from ..scheduling.clock import Clock
from .base import MagneticActuator, Watchdog
from .mock import MockActuator


def build_actuator(
    cfg: ActuationConfig, clock: Clock | None = None
) -> MagneticActuator:
    """Construct the configured actuator.

    Hardware backends are imported lazily so that a machine with no GPIO and
    no serial port can still import this module and run every test that uses
    the mock.
    """
    if cfg.kind == "mock":
        return MockActuator(cfg, clock=clock)
    if cfg.kind == "gpio":
        from .gpio import GpioActuator

        return GpioActuator(cfg)
    if cfg.kind == "serial":
        from .serial_actuator import SerialActuator

        return SerialActuator(cfg)
    raise ConfigurationError(f"unknown actuation.kind: {cfg.kind!r}")


def available_actuators() -> list[str]:
    return ["mock", "gpio", "serial"]


__all__ = ["MagneticActuator", "Watchdog", "available_actuators", "build_actuator"]
