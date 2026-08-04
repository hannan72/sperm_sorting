"""Calibration, and what the system refuses to do without it.

The theme: an uncalibrated system must *say so and stop*, never substitute a
plausible number. A guessed micrometres-per-pixel silently rescales every
velocity across the WHO motility boundaries; a guessed transport delay applies
the field to the wrong fluid. Neither failure raises anything on its own, which
is exactly why the refusals below are tested as hard as the calculations.
"""

from __future__ import annotations

import numpy as np
import pytest

from builders import make_track
from sperm_sorting.calibration.flow import calibrate_fixed_vector, calibrate_flow_map
from sperm_sorting.calibration.optics import (
    calibrate_from_graticule,
    calibrate_from_known_distance,
    load_calibration,
    px_s_to_um_s,
    save_calibration,
)
from sperm_sorting.calibration.transport import (
    estimate_field_switching,
    estimate_from_geometry,
    estimate_from_tracer,
)
from sperm_sorting.config import (
    OpticalCalibration,
    OpticsConfig,
    SchedulingConfig,
)
from sperm_sorting.errors import CalibrationError

TRUE_UM_PER_PX = 0.0345


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_uncalibrated_optics_refuses_to_convert() -> None:
    optical = OpticalCalibration()
    assert optical.calibrated is False
    with pytest.raises(CalibrationError, match="optical calibration is missing"):
        optical.require_calibrated()
    with pytest.raises(CalibrationError):
        px_s_to_um_s(100.0, optical)


def test_uncalibrated_scheduling_refuses_to_arm() -> None:
    with pytest.raises(CalibrationError, match="not calibrated"):
        SchedulingConfig().require_calibrated()


def test_calibrated_flag_without_a_value_is_rejected() -> None:
    """Marking a calibration done without supplying it must not be possible."""
    with pytest.raises(Exception, match="um_per_px"):
        OpticalCalibration(calibrated=True, um_per_px=None)


# --------------------------------------------------------------------------
# Nominal optics
# --------------------------------------------------------------------------


def test_nominal_scale_matches_the_reference_build() -> None:
    """3.45 um pixels behind a 100x objective and a 1x coupler."""
    optics = OpticsConfig()
    assert optics.nominal_um_per_px == pytest.approx(0.0345)
    assert optics.abbe_limit_um == pytest.approx(0.220, abs=1e-4)
    assert optics.rayleigh_limit_um == pytest.approx(0.2684, abs=1e-4)
    # Comfortably above Nyquist, which is what leaves room for 2x2 binning.
    assert optics.nyquist_oversampling == pytest.approx(3.89, abs=0.01)


def test_field_of_view_is_smaller_than_a_whole_sperm() -> None:
    """The finding that forces head detection rather than whole-cell detection."""
    optics = OpticsConfig()
    width_um, height_um = optics.field_of_view_um(1920, 1200)
    assert width_um == pytest.approx(66.24, abs=0.01)
    assert height_um == pytest.approx(41.40, abs=0.01)

    whole_sperm_um = 53.1  # WHO 6th ed.: head 4.1 + midpiece 4.0 + tail ~45
    assert whole_sperm_um < width_um, "fits along the frame"
    assert whole_sperm_um > height_um, "does not fit across it"

    head_px = 4.1 / optics.nominal_um_per_px
    assert head_px == pytest.approx(118.8, abs=0.5)


def test_reducing_coupler_halves_the_scale() -> None:
    assert OpticsConfig(coupler_magnification=0.5).nominal_um_per_px == pytest.approx(
        0.069
    )


# --------------------------------------------------------------------------
# Optical measurement
# --------------------------------------------------------------------------


def test_known_distance_calibration() -> None:
    result = calibrate_from_known_distance(
        pixel_distance=1000.0, physical_distance_um=34.5
    )
    assert result.um_per_px == pytest.approx(TRUE_UM_PER_PX)
    assert result.nominal_ratio == pytest.approx(1.0)
    assert 0.0 < result.relative_uncertainty < 0.01


def test_graticule_calibration_recovers_the_scale() -> None:
    """A synthetic 10 um graticule must recover 0.0345 um/px."""
    period_px = 10.0 / TRUE_UM_PER_PX
    x = np.arange(1920)
    profile = 200 + 40 * np.sign(np.sin(2 * np.pi * x / period_px))
    image = np.tile(profile, (100, 1))
    image = image + np.random.default_rng(0).normal(0, 3, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    result = calibrate_from_graticule(image, ruling_pitch_um=10.0)
    relative_error = abs(result.um_per_px - TRUE_UM_PER_PX) / TRUE_UM_PER_PX
    assert relative_error < 0.02, f"recovered {result.um_per_px}"
    assert result.n_samples >= 5


def test_reducing_coupler_is_caught() -> None:
    """A 0.5x adapter gives exactly a 2x discrepancy and must be refused.

    This is the single most likely optical mistake, and it is invisible: the
    images look fine and every velocity is out by a factor of two.
    """
    with pytest.raises(CalibrationError, match="coupler"):
        calibrate_from_known_distance(
            pixel_distance=500.0, physical_distance_um=34.5  # 0.069 um/px
        )


def test_calibration_round_trips_through_disk(tmp_path) -> None:
    result = calibrate_from_known_distance(
        pixel_distance=1000.0, physical_distance_um=34.5
    )
    path = tmp_path / "optics.json"
    save_calibration(result, path)
    loaded = load_calibration(path)
    assert loaded.um_per_px == pytest.approx(result.um_per_px)
    assert loaded.method == result.method


def test_calibration_config_rejects_an_implausible_measurement() -> None:
    with pytest.raises(Exception, match="nominal"):
        OpticalCalibration(calibrated=True, um_per_px=0.5)


def test_conversion_uses_the_measured_scale() -> None:
    optical = OpticalCalibration(
        calibrated=True, calibration_id="test", um_per_px=TRUE_UM_PER_PX
    )
    # A sperm crossing 1000 px/s is 34.5 um/s: slow progressive by WHO 6th ed.
    assert px_s_to_um_s(1000.0, optical) == pytest.approx(34.5)


# --------------------------------------------------------------------------
# Flow calibration
# --------------------------------------------------------------------------


def test_fixed_vector_flow_from_drifting_objects() -> None:
    """Slow tracks are assumed passively transported and set the estimate."""
    tracks = [
        make_track(i, n_points=20, dx=2.0, dy=0.0, dt=0.01) for i in range(12)
    ]
    result = calibrate_fixed_vector(tracks, quantile=0.5, min_tracks=8)
    assert result.vx_px_s == pytest.approx(200.0, rel=0.01)
    assert result.vy_px_s == pytest.approx(0.0, abs=1e-6)


def test_flow_estimate_resists_fast_swimmers() -> None:
    """A median, not a mean, so a few fast cells cannot drag the estimate."""
    drifting = [make_track(i, n_points=20, dx=2.0, dt=0.01) for i in range(12)]
    swimmers = [
        make_track(100 + i, n_points=20, dx=40.0, dt=0.01) for i in range(4)
    ]
    result = calibrate_fixed_vector(drifting + swimmers, quantile=0.5, min_tracks=8)
    assert result.vx_px_s == pytest.approx(200.0, rel=0.05)


def test_flow_calibration_refuses_too_few_tracks() -> None:
    tracks = [make_track(i, n_points=20) for i in range(3)]
    with pytest.raises(CalibrationError, match="at least"):
        calibrate_fixed_vector(tracks, min_tracks=8)


def test_flow_map_has_the_requested_shape() -> None:
    tracks = [
        make_track(i, n_points=20, x0=10.0 + i * 30, y0=20.0 + i * 12, dx=2.0, dt=0.01)
        for i in range(40)
    ]
    field, summary = calibrate_flow_map(tracks, height=120, width=200, grid=4)
    assert field.shape == (120, 200, 2)
    assert summary.map_shape == (120, 200)
    assert np.isfinite(field).all()


# --------------------------------------------------------------------------
# Transport delay
# --------------------------------------------------------------------------


def test_tracer_calibration() -> None:
    imaging = [0.0, 1.0, 2.0, 3.0, 4.0]
    magnet = [0.452, 1.449, 2.455, 3.447, 4.451]
    result = estimate_from_tracer(imaging, magnet)
    assert result.transport_delay_ms == pytest.approx(450.8, abs=0.5)
    assert result.transport_delay_std_ms < 10.0
    assert result.n_trials == 5


def test_tracer_calibration_requires_repeats() -> None:
    """One trial gives no spread, and the spread sets the activation margin."""
    with pytest.raises(CalibrationError, match="at least 3 trials"):
        estimate_from_tracer([0.0, 1.0], [0.5, 1.5])


def test_tracer_calibration_rejects_reversed_pairs() -> None:
    with pytest.raises(CalibrationError, match="later than"):
        estimate_from_tracer([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])


def test_transport_result_marks_the_config_calibrated() -> None:
    result = estimate_from_tracer([0.0, 1.0, 2.0], [0.45, 1.46, 2.44])
    cfg = result.to_config("cal-test")
    assert cfg.calibrated is True
    assert cfg.calibration_id == "cal-test"
    cfg.require_calibrated()  # must not raise
    # The margin absorbs the measured jitter rather than assuming none.
    assert cfg.pre_activation_margin_ms >= 3.0 * result.transport_delay_std_ms


def test_geometry_estimate_is_only_a_cross_check() -> None:
    delay_ms = estimate_from_geometry(
        channel_length_mm=15.0,
        channel_width_um=500.0,
        channel_height_um=50.0,
        volumetric_flow_ul_min=10.0,
    )
    # 15 mm at a mean velocity of (10 uL/min) / (500 x 50 um) = 6.67 mm/s.
    assert delay_ms == pytest.approx(2250.0, rel=0.01)


def test_geometry_estimate_rejects_nonsense() -> None:
    with pytest.raises(CalibrationError, match="must be positive"):
        estimate_from_geometry(0.0, 500.0, 50.0, 10.0)


def test_field_switching_measurement() -> None:
    mean_ms, std_ms = estimate_field_switching(
        [0.0, 1.0, 2.0], [0.008, 1.009, 2.007], rising=True
    )
    assert mean_ms == pytest.approx(8.0, abs=0.5)
    assert std_ms < 2.0


def test_field_switching_rejects_settling_before_command() -> None:
    with pytest.raises(CalibrationError, match="cannot settle before"):
        estimate_field_switching([1.0], [0.5], rising=True)
