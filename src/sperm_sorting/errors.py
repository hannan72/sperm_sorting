"""Exception hierarchy.

The runtime distinguishes three broad failure classes, because they drive
different responses in :mod:`sperm_sorting.runtime.pipeline`:

* :class:`ConfigurationError` -- refuse to start.
* :class:`CalibrationError` -- start, but refuse to report physical units or
  (depending on policy) refuse to actuate.
* :class:`HardwareError` / :class:`InferenceError` -- degrade safely at
  runtime; the watchdog drives the field to FIELD_OFF.
"""

from __future__ import annotations


class SpermSortingError(Exception):
    """Base class for every error raised by this package."""


# --------------------------------------------------------------------------
# Startup-time
# --------------------------------------------------------------------------


class ConfigurationError(SpermSortingError):
    """Configuration is missing, malformed or internally inconsistent."""


class CalibrationError(SpermSortingError):
    """A physical calibration is missing, stale or invalid.

    Raised, for example, when velocities are requested in micrometres per
    second but no optical calibration has been loaded. The system must never
    silently substitute pixel units for physical units.
    """


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


class HardwareError(SpermSortingError):
    """Base class for camera / actuator / bus faults."""


class CameraError(HardwareError):
    """The camera could not be opened, configured, or has disconnected."""


class ActuatorError(HardwareError):
    """The magnetic-field actuator failed to accept or acknowledge a command."""


class WatchdogTimeout(HardwareError):
    """The actuator watchdog was not fed within its timeout.

    Handling this exception must always drive the field to FIELD_OFF.
    """


class InferenceError(SpermSortingError):
    """A model failed to load or to produce an output."""


class BackendUnavailableError(InferenceError):
    """The requested inference backend is not installed or not usable."""


# --------------------------------------------------------------------------
# Data / dataset
# --------------------------------------------------------------------------


class DatasetError(SpermSortingError):
    """Base class for dataset adapter and validator failures."""


class DatasetNotFoundError(DatasetError):
    """The dataset root does not exist or lacks its expected files."""


class DatasetValidationError(DatasetError):
    """The dataset is present but violates a documented invariant.

    Raised, notably, when MHSMA label polarity cannot be confirmed as
    ``0 = normal`` / ``1 = abnormal``.
    """


class LeakageError(DatasetError):
    """A requested split would place frames from one video/patient on both
    sides of a train/validation boundary."""


# --------------------------------------------------------------------------
# Real-time scheduling
# --------------------------------------------------------------------------


class SchedulingError(SpermSortingError):
    """Base class for scheduler faults."""


class LateCommandError(SchedulingError):
    """A command's scheduled activation time is already in the past by more
    than the configured tolerance."""


class DeadlineMissed(SchedulingError):
    """A per-sperm or per-shot evaluation did not complete before its
    deadline. This is an expected, logged condition -- not a crash."""
