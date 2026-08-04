"""Optical calibration: micrometres per pixel.

Every velocity this system reports in physical units passes through one
number. Getting it wrong scales every VSL, and therefore moves every sperm
across the 25 and 5 um/s WHO boundaries, and therefore changes the shot ratio
and the sort. It is the single most leveraged constant in the product, which
is why it is measured rather than assumed.

The measurement is made against a stage micrometer (a graticule with rulings
of certified pitch, usually 10 um). Two routes are provided:

* :func:`calibrate_from_known_distance` -- the operator marks two rulings a
  known distance apart. Simple, and the fallback when automatic detection
  fails on a low-contrast image.
* :func:`calibrate_from_graticule` -- detects the periodic ruling pattern
  automatically via the image's power spectrum, which uses every ruling in the
  field rather than two, and so averages down the marking error.

Both cross-check against the nominal scale implied by the optical train and
refuse a result that disagrees by more than the configured factor. The failure
this catches is a reducing C-mount coupler: a 0.5x adapter is easy not to
notice and puts every velocity out by exactly a factor of two.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import OpticalCalibration, OpticsConfig
from ..errors import CalibrationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpticalCalibrationResult:
    """Outcome of one optical calibration."""

    um_per_px: float
    nominal_um_per_px: float
    relative_uncertainty: float
    method: str
    n_samples: int
    #: Ratio of measured to nominal; ~1.0 means the optical train is as
    #: described, ~2.0 means an unnoticed 0.5x reducing coupler.
    nominal_ratio: float
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "um_per_px": self.um_per_px,
            "nominal_um_per_px": self.nominal_um_per_px,
            "relative_uncertainty": self.relative_uncertainty,
            "method": self.method,
            "n_samples": self.n_samples,
            "nominal_ratio": self.nominal_ratio,
            "notes": self.notes,
        }

    def to_config(self, calibration_id: str, optics: OpticsConfig) -> OpticalCalibration:
        return OpticalCalibration(
            calibrated=True,
            calibration_id=calibration_id,
            um_per_px=self.um_per_px,
            optics=optics,
            relative_uncertainty=self.relative_uncertainty,
        )


def _check_against_nominal(
    um_per_px: float, optics: OpticsConfig, max_discrepancy: float
) -> float:
    nominal = optics.nominal_um_per_px
    ratio = um_per_px / nominal
    if not (1.0 / max_discrepancy <= ratio <= max_discrepancy):
        raise CalibrationError(
            f"measured scale {um_per_px:.5f} um/px disagrees with the nominal "
            f"{nominal:.5f} um/px by {ratio:.2f}x. The most common cause is a "
            f"reducing C-mount coupler (a 0.5x adapter gives exactly 2.0x). "
            f"Check the adapter, the objective, and the graticule pitch before "
            f"accepting this calibration."
        )
    return ratio


def calibrate_from_known_distance(
    pixel_distance: float,
    physical_distance_um: float,
    optics: OpticsConfig | None = None,
    *,
    max_discrepancy: float = 1.5,
    marking_uncertainty_px: float = 2.0,
) -> OpticalCalibrationResult:
    """Scale from a single measured span.

    ``marking_uncertainty_px`` is how precisely the operator can place each of
    the two marks; the resulting relative uncertainty is what propagates into
    every reported velocity, so it is recorded rather than discarded.
    """
    if pixel_distance <= 0:
        raise CalibrationError("pixel_distance must be positive")
    if physical_distance_um <= 0:
        raise CalibrationError("physical_distance_um must be positive")

    optics = optics or OpticsConfig()
    um_per_px = physical_distance_um / pixel_distance
    # Two independent marks, so the errors add in quadrature.
    uncertainty = float(np.sqrt(2.0) * marking_uncertainty_px / pixel_distance)
    ratio = _check_against_nominal(um_per_px, optics, max_discrepancy)

    return OpticalCalibrationResult(
        um_per_px=um_per_px,
        nominal_um_per_px=optics.nominal_um_per_px,
        relative_uncertainty=uncertainty,
        method="known_distance",
        n_samples=1,
        nominal_ratio=ratio,
        notes=(
            f"{physical_distance_um:.2f} um spans {pixel_distance:.1f} px; "
            f"assumed marking uncertainty {marking_uncertainty_px:.1f} px"
        ),
    )


def calibrate_from_graticule(
    image: np.ndarray,
    ruling_pitch_um: float = 10.0,
    optics: OpticsConfig | None = None,
    *,
    axis: int = 1,
    max_discrepancy: float = 1.5,
    min_period_px: float = 4.0,
) -> OpticalCalibrationResult:
    """Scale from the periodic rulings of a stage micrometer.

    Collapses the image along the ruling direction to get a 1-D profile, then
    finds the dominant spatial frequency of that profile. Using the whole
    field averages over every ruling present, so the result is far less
    sensitive to any single edge than a two-point measurement.

    Parameters
    ----------
    axis
        ``1`` when the rulings run vertically (period measured across x),
        ``0`` when they run horizontally.
    """
    optics = optics or OpticsConfig()
    if image.ndim != 2:
        raise CalibrationError(f"expected a 2-D image, got shape {image.shape}")

    profile = image.astype(np.float64).mean(axis=0 if axis == 1 else 1)
    n = profile.size
    if n < 32:
        raise CalibrationError(f"profile is too short to find a period ({n} samples)")

    # Remove the DC level and any illumination ramp: a gradient across the
    # field puts a huge spike at low frequency that would swamp the rulings.
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, profile, 1)
    detrended = profile - (slope * x + intercept)
    detrended *= np.hanning(n)

    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(n, d=1.0)

    # Ignore frequencies whose period is longer than a third of the field
    # (residual illumination structure) or shorter than min_period_px (noise).
    valid = (freqs > 3.0 / n) & (freqs < 1.0 / min_period_px)
    if not valid.any():
        raise CalibrationError("no plausible ruling frequency in the spectrum")

    masked = np.where(valid, spectrum, 0.0)
    peak = int(np.argmax(masked))

    # Parabolic interpolation around the peak bin: the true period almost
    # never lands exactly on a bin, and this recovers most of the sub-bin
    # accuracy for free.
    if 0 < peak < len(masked) - 1:
        y0, y1, y2 = masked[peak - 1], masked[peak], masked[peak + 1]
        denom = y0 - 2 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        offset = 0.0

    peak_freq = freqs[peak] + offset * (freqs[1] - freqs[0])
    if peak_freq <= 0:
        raise CalibrationError("recovered a non-positive ruling frequency")

    period_px = 1.0 / peak_freq
    um_per_px = ruling_pitch_um / period_px

    # Signal-to-noise of the peak against the rest of the valid band gives a
    # usable uncertainty estimate without needing repeated measurements.
    band = masked[valid]
    noise = float(np.median(band)) if band.size else 0.0
    snr = float(masked[peak] / noise) if noise > 0 else float("inf")
    uncertainty = float(min(0.5, 1.0 / max(snr, 1.0)))

    ratio = _check_against_nominal(um_per_px, optics, max_discrepancy)
    n_rulings = int(n / period_px)

    return OpticalCalibrationResult(
        um_per_px=um_per_px,
        nominal_um_per_px=optics.nominal_um_per_px,
        relative_uncertainty=uncertainty,
        method="graticule_fft",
        n_samples=n_rulings,
        nominal_ratio=ratio,
        notes=(
            f"ruling period {period_px:.2f} px for a {ruling_pitch_um:.1f} um "
            f"pitch, {n_rulings} rulings in the field, peak SNR {snr:.1f}"
        ),
    )


def save_calibration(result: OpticalCalibrationResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    logger.info("wrote optical calibration to %s", path)


def load_calibration(path: Path | str) -> OpticalCalibrationResult:
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"calibration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return OpticalCalibrationResult(**data)


def px_s_to_um_s(value_px_s: float, calibration: OpticalCalibration) -> float:
    """Convert a pixel velocity to micrometres per second, or refuse."""
    return value_px_s * calibration.require_calibrated()
