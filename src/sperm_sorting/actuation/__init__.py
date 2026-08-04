"""Magnetic field actuation.

FIELD_OFF is the safe state throughout: it lets the sample flow to collection
unsorted, whereas a stuck FIELD_ON would silently divert everything to waste.
Every fault path in this package ends in FIELD_OFF.
"""

from __future__ import annotations

from .base import MagneticActuator, Watchdog
from .factory import available_actuators, build_actuator
from .mock import MockActuator, StateChange

__all__ = [
    "MagneticActuator",
    "MockActuator",
    "StateChange",
    "Watchdog",
    "available_actuators",
    "build_actuator",
]
