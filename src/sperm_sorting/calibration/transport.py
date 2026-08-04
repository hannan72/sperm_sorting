"""Transport-delay calibration.

The imaging region and the magnetic region are separated along the channel, so
a decision about the fluid under the microscope must be applied to the moment
that fluid arrives at the magnet. That interval -- the transport delay -- is a
property of the built kit and is unknown until measured.

It cannot be guessed. A wrong transport delay is the worst class of failure
available in this device: nothing raises, nothing looks wrong, and every shot
gates the wrong segment of fluid. The scheduler therefore refuses to arm until
this measurement exists.

Two measurement methods:

* :func:`estimate_from_tracer` -- inject a visible bolus (dye, bead
  suspension, an air gap) and time its passage between the imaging region and
  a downstream observation point. Direct, and the method of record.
* :func:`estimate_from_geometry` -- compute the delay from channel geometry
  and volumetric flow rate. Useful as a sanity check on the measurement, and
  as a starting estimate for the search window; explicitly *not* a substitute,
  because it assumes plug flow and ignores the parabolic velocity profile,
  dispersion, and any dead volume.

Also provided is the field rise/fall measurement, which is separate: the
transport delay says *when* the field must be on, and the rise time says *how
much earlier* to command it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SchedulingConfig
from ..errors import CalibrationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TransportCalibrationResult:
    """Measured timing between the imaging region and the magnetic region."""

    transport_delay_ms: float
    transport_delay_std_ms: float
    n_trials: int
    method: str
    #: Populated when field switching was characterised in the same session.
    field_rise_time_ms: float | None = None
    field_fall_time_ms: float | None = None
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "transport_delay_ms": self.transport_delay_ms,
            "transport_delay_std_ms": self.transport_delay_std_ms,
            "n_trials": self.n_trials,
            "method": self.method,
            "field_rise_time_ms": self.field_rise_time_ms,
            "field_fall_time_ms": self.field_fall_time_ms,
            "notes": self.notes,
        }

    def to_config(
        self, calibration_id: str, base: SchedulingConfig | None = None
    ) -> SchedulingConfig:
        """Build a scheduling config marked calibrated.

        The pre-activation margin defaults to three standard deviations of the
        measured delay, so the field is on before the segment arrives in
        essentially every case rather than only on average.
        """
        base = base or SchedulingConfig()
        margin = max(
            base.pre_activation_margin_ms, 3.0 * self.transport_delay_std_ms
        )
        return base.model_copy(
            update={
                "calibrated": True,
                "calibration_id": calibration_id,
                "transport_delay_ms": self.transport_delay_ms,
                "transport_delay_std_ms": self.transport_delay_std_ms,
                "field_rise_time_ms": (
                    self.field_rise_time_ms
                    if self.field_rise_time_ms is not None
                    else base.field_rise_time_ms
                ),
                "field_fall_time_ms": (
                    self.field_fall_time_ms
                    if self.field_fall_time_ms is not None
                    else base.field_fall_time_ms
                ),
                "pre_activation_margin_ms": margin,
                "post_activation_margin_ms": margin,
            }
        )


def estimate_from_tracer(
    imaging_times_s: list[float],
    magnet_times_s: list[float],
    *,
    method: str = "tracer_bolus",
) -> TransportCalibrationResult:
    """Delay from paired arrival times of a tracer bolus.

    Each pair is one trial: the instant the bolus was seen in the imaging
    region and the instant it reached the magnetic region. The spread across
    trials is what widens the activation window, so several trials are
    required -- a single measurement gives no way to size the margin.
    """
    if len(imaging_times_s) != len(magnet_times_s):
        raise CalibrationError(
            f"paired arrival times must be the same length, got "
            f"{len(imaging_times_s)} and {len(magnet_times_s)}"
        )
    if len(imaging_times_s) < 3:
        raise CalibrationError(
            f"at least 3 trials are needed to estimate the spread of the "
            f"transport delay; got {len(imaging_times_s)}. The spread sets the "
            "activation margin, so it cannot be skipped."
        )

    deltas = np.asarray(magnet_times_s, dtype=np.float64) - np.asarray(
        imaging_times_s, dtype=np.float64
    )
    if np.any(deltas <= 0):
        raise CalibrationError(
            "every magnet arrival must be later than its imaging arrival; "
            "check that the two time series are paired and in the same order"
        )

    mean_ms = float(np.mean(deltas) * 1000.0)
    std_ms = float(np.std(deltas, ddof=1) * 1000.0)

    if std_ms > 0.5 * mean_ms:
        logger.warning(
            "transport delay is highly variable (%.1f +/- %.1f ms). The "
            "activation window will have to be wide, which reduces sorting "
            "precision. Check for pump pulsation, bubbles or a leak.",
            mean_ms,
            std_ms,
        )

    return TransportCalibrationResult(
        transport_delay_ms=mean_ms,
        transport_delay_std_ms=std_ms,
        n_trials=len(deltas),
        method=method,
        notes=(
            f"{len(deltas)} tracer trials, range "
            f"{deltas.min() * 1000:.1f}-{deltas.max() * 1000:.1f} ms"
        ),
    )


def estimate_from_geometry(
    channel_length_mm: float,
    channel_width_um: float,
    channel_height_um: float,
    volumetric_flow_ul_min: float,
) -> float:
    """Plug-flow transport delay from geometry, in milliseconds.

    A cross-check and a starting estimate only. It assumes uniform plug flow,
    so it ignores the parabolic velocity profile of pressure-driven flow (the
    centreline moves about 1.5-2x the mean), Taylor dispersion, and any dead
    volume in connectors. Expect the measured delay to differ, and trust the
    measurement.
    """
    for name, value in (
        ("channel_length_mm", channel_length_mm),
        ("channel_width_um", channel_width_um),
        ("channel_height_um", channel_height_um),
        ("volumetric_flow_ul_min", volumetric_flow_ul_min),
    ):
        if value <= 0:
            raise CalibrationError(f"{name} must be positive, got {value}")

    # Cross-sectional area in m^2, length in m, flow in m^3/s.
    area_m2 = (channel_width_um * 1e-6) * (channel_height_um * 1e-6)
    length_m = channel_length_mm * 1e-3
    flow_m3_s = volumetric_flow_ul_min * 1e-9 / 60.0
    mean_velocity_m_s = flow_m3_s / area_m2
    return float(length_m / mean_velocity_m_s * 1000.0)


def estimate_field_switching(
    command_times_s: list[float],
    field_reached_times_s: list[float],
    *,
    rising: bool,
) -> tuple[float, float]:
    """Field rise or fall time from paired command and settle instants.

    Returns ``(mean_ms, std_ms)``. Measured with a Hall probe or a pickup coil
    at the magnetic region; ``field_reached_times_s`` is when the field first
    crosses its settling threshold, not when it finally asymptotes.
    """
    if len(command_times_s) != len(field_reached_times_s):
        raise CalibrationError("paired switching times must be the same length")
    if not command_times_s:
        raise CalibrationError("no switching trials supplied")

    deltas = np.asarray(field_reached_times_s, dtype=np.float64) - np.asarray(
        command_times_s, dtype=np.float64
    )
    if np.any(deltas < 0):
        raise CalibrationError(
            "the field cannot settle before it was commanded; check pairing"
        )
    edge = "rise" if rising else "fall"
    logger.info(
        "field %s time: %.2f +/- %.2f ms over %d trials",
        edge,
        deltas.mean() * 1000,
        deltas.std() * 1000,
        len(deltas),
    )
    return float(np.mean(deltas) * 1000.0), float(np.std(deltas) * 1000.0)


def save_transport_calibration(
    result: TransportCalibrationResult, path: Path | str
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    logger.info("wrote transport calibration to %s", path)


def load_transport_calibration(path: Path | str) -> TransportCalibrationResult:
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"transport calibration not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return TransportCalibrationResult(**json.load(fh))
