"""CASA kinematics for one track.

Definitions implemented here
----------------------------
Let :math:`p_i` be the measured head positions and :math:`t_i` their capture
times, with :math:`T = t_{N-1} - t_0`, and let :math:`a_i` be the average path
(a moving average of :math:`p_i`; see :mod:`.smoothing`).

======  ==================================================  ====================
Symbol  Definition                                          Units
======  ==================================================  ====================
VCL     :math:`\\sum_i \\lVert p_{i+1}-p_i \\rVert / T`      px/s and um/s
VSL     :math:`\\lVert p_{N-1}-p_0 \\rVert / T`              px/s and um/s
VAP     :math:`\\sum_i \\lVert a_{i+1}-a_i \\rVert / T`      px/s and um/s
LIN     VSL / VCL                                           --
STR     VSL / VAP                                           --
WOB     VAP / VCL                                           --
ALH     mean :math:`\\lvert` lateral offset of :math:`p`     um
        from the average path :math:`\\rvert`
BCF     crossings of :math:`p` through the average path / T  Hz
======  ==================================================  ====================

Five properties of this implementation are load-bearing:

1. **Only measured points are used.** A track that went unmatched for three
   frames still has points for them, flagged ``observed=False``, because the
   tracker predicted the positions with its motion model. Those positions are
   the model's opinion. Computing a velocity from them measures the Kalman
   filter, not the sperm, and it does so in the flattering direction: a
   constant-velocity predictor produces a perfectly straight, perfectly smooth
   segment, which inflates LIN and suppresses ALH.

2. **Real timestamps, never a nominal frame interval.** ``frame_index / fps``
   is wrong whenever the camera drops a frame, and a dropped frame is exactly
   the situation in which the error is largest. :attr:`mean_frame_interval_s`
   is measured from the timestamps of the observed points, and the effective
   sampling rate used to gate ALH/BCF is derived from it -- so a track that was
   only matched on every third frame is correctly judged to be sampled at a
   third of the camera's rate.

3. **Micrometres only when calibrated.** Without an optical calibration the
   ``*_um_s`` fields stay ``None`` and :attr:`optically_calibrated` is false.
   Reporting pixel numbers under a micrometre label would be a fabricated
   physical measurement.

4. **Ratios come from the corrected velocities**, with guarded denominators:
   an immotile cell has VCL of zero, and ``0/0`` must be reported as unknown,
   not as ``nan`` propagating into a comparison that then silently evaluates
   false.

5. **VAP and its family are algorithm-dependent.** VAP, STR, WOB, ALH and BCF
   all depend on the average-path smoother, and are therefore not comparable
   with another CASA system's values.

   The window is configured in *milliseconds* (``MotionConfig.vap_window_ms``)
   and resolved to a frame count here, against the frame rate actually
   measured from this track's timestamps. A fixed frame count would not be
   equivalent: five frames is 100 ms at 50 FPS but 31 ms at 160 FPS, and
   Mortimer, van der Horst & Mortimer (Asian J Androl 2015;17:545-53) show
   that a fixed point count gives inadequate smoothing and widely aberrant ALH
   once the acquisition rate changes. Fixing the window in time keeps the
   average path comparable across frame rates.

   Because the resolved window is part of the definition of the numbers, it is
   stamped into :attr:`MotionFeatures.profile_version` per track, e.g.
   ``who6-inspired-v1+vap=moving_average:17f@100ms``. The configured threshold
   profile remains the prefix, so a log written before this suffix existed
   still matches on ``startswith``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..config import MotionConfig, OpticalCalibration
from ..constants import EPS
from ..errors import CalibrationError
from ..schemas.enums import FlowCorrectionMode, MotilityClass, TimestampSource
from ..schemas.track import MotionFeatures, TrackPoint, TrackRecord
from .flow import apply_flow_correction
from .smoothing import net_displacement, path_length, smooth_path

__all__ = [
    "FlowSampler",
    "compute_motion_features",
    "lateral_deviations",
]

#: A callable mapping ``(N, 2)`` positions to the ``(N, 2)`` flow velocity at
#: those positions, in px/s. :meth:`~.flow.FlowEstimator.sample_points`
#: satisfies it; taking a callable rather than the estimator keeps this module
#: independent of the estimator hierarchy.
FlowSampler = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_NOT_CLASSIFIED = ""


def compute_motion_features(
    track: TrackRecord,
    cfg: MotionConfig,
    optical: OpticalCalibration | None = None,
    flow_vector: tuple[float, float] | None = None,
    timestamp_source: TimestampSource = TimestampSource.HOST_MONOTONIC,
    *,
    flow_sampler: FlowSampler | None = None,
) -> MotionFeatures:
    """Compute the full CASA kinematic record for one track.

    Parameters
    ----------
    track
        The track to analyse. Only its ``observed=True`` points are used.
    cfg
        Motion configuration: the average-path algorithm, the minimum point
        counts, and the ALH/BCF sampling-rate gate.
    optical
        Pixel-to-micrometre calibration. ``None`` or uncalibrated leaves every
        ``*_um_s`` field and ALH as ``None``.
    flow_vector
        Bulk flow ``(vx, vy)`` in px/s to subtract, or ``None`` when the
        estimator could not produce one. ``None`` means *no correction is
        applied*: the corrected values equal the raw ones, and the record says
        so by reporting :attr:`MotionFeatures.flow_correction_mode` as
        ``DISABLED`` (see below).
    timestamp_source
        Provenance of the capture times, copied into the record so a reader can
        tell a hardware-timed velocity from a software-timed one.
    flow_sampler
        Optional position-dependent flow, used in preference to
        ``flow_vector``. Called once with the observed positions and must
        return the flow at each of them. This is how a flow *map* is applied.

    Returns
    -------
    MotionFeatures
        Always a complete record, never an exception, for any track. A track
        too short or too brief to analyse comes back with
        ``motility_class=UNDETERMINED`` and an explanatory ``motility_reason``,
        because "this sperm could not be measured" is a normal outcome that the
        shot accounting has to be able to record.

        This function does **not** grade motility. A successfully measured
        track therefore also comes back ``UNDETERMINED``, but with an *empty*
        reason -- the two are distinguishable, and the safe default for a
        record nobody has graded yet is "unknown", not "immotile". Grading is
        :mod:`.classifier`'s job.

    Notes
    -----
    ``flow_correction_mode`` records **what was actually applied**, not what
    was configured. A configured ``ROBUST_ESTIMATE`` that yielded no estimate
    for this track is recorded as ``DISABLED``, so the audit log never claims a
    correction that did not happen; the *configured* mode is already in the
    audit header via :meth:`AppConfig.summary`.
    ``ProgressiveMotilityClassifier`` compares the two and refuses to grade
    uncorrected kinematics when a correction was expected.
    """
    points = list(track.points)
    observed = _sorted_observed(points)
    n_points = len(points)
    n_observed = len(observed)

    um_per_px = _resolve_um_per_px(optical)
    configured_mode = cfg.flow_correction.mode
    correction_available = flow_sampler is not None or flow_vector is not None
    applied_mode = configured_mode if correction_available else FlowCorrectionMode.DISABLED

    def unmeasurable(reason: str) -> MotionFeatures:
        """A complete, honest, all-zero record plus the reason it is empty."""
        return MotionFeatures(
            n_points=n_points,
            n_observed_points=n_observed,
            duration_s=_observed_duration(observed),
            mean_frame_interval_s=_mean_interval(observed),
            timestamp_source=timestamp_source,
            flow_correction_mode=applied_mode,
            # No average path was computed, so no smoother is named: the bare
            # threshold-profile version is the whole truth about this record.
            profile_version=cfg.thresholds.profile_version,
            optically_calibrated=um_per_px is not None,
            um_per_px=um_per_px,
            alh_unavailable_reason=reason,
            bcf_unavailable_reason=reason,
            motility_class=MotilityClass.UNDETERMINED,
            motility_reason=reason,
        )

    min_points = max(2, int(cfg.min_points_for_kinematics))
    if n_observed < min_points:
        return unmeasurable(
            f"kinematics not computed: {n_observed} observed point(s), "
            f"{min_points} required (min_points_for_kinematics="
            f"{cfg.min_points_for_kinematics}); predicted points are excluded "
            "because velocity from a motion model is not a measurement"
        )

    times = np.array([p.capture_time_s for p in observed], dtype=np.float64)
    duration = float(times[-1] - times[0])
    if duration <= EPS:
        return unmeasurable(
            f"kinematics not computed: {n_observed} observed points span "
            f"{duration:.6g} s of capture time; velocity is undefined over a "
            "zero interval (check the timestamp source)"
        )

    raw = np.array([[p.x, p.y] for p in observed], dtype=np.float64)
    intervals = np.diff(times)
    mean_interval = float(intervals.mean())
    effective_fps = 1.0 / mean_interval if mean_interval > EPS else 0.0

    # ---------------------------------------------------------------- flow
    if flow_sampler is not None:
        sampled = np.asarray(flow_sampler(raw), dtype=np.float64)
        if sampled.shape != raw.shape:
            raise ValueError(
                f"flow_sampler returned shape {sampled.shape}, expected {raw.shape}"
            )
        flow_x: NDArray[np.float64] | float = sampled[:, 0]
        flow_y: NDArray[np.float64] | float = sampled[:, 1]
        # Recorded flow is the mean over the track: one number for the audit
        # log, while the correction itself stayed position-dependent.
        recorded_flow = (float(sampled[:, 0].mean()), float(sampled[:, 1].mean()))
    elif flow_vector is not None:
        flow_x, flow_y = float(flow_vector[0]), float(flow_vector[1])
        recorded_flow = (flow_x, flow_y)
    else:
        flow_x, flow_y = 0.0, 0.0
        recorded_flow = (0.0, 0.0)

    corrected = apply_flow_correction(raw, times, flow_x, flow_y)

    # ---------------------------------------------------------- velocities
    # The average-path window is specified in milliseconds and resolved here
    # against the frame rate this track was actually sampled at, so that a
    # track matched on every frame and one matched on every third frame are
    # smoothed over the same duration of trajectory rather than the same
    # number of samples.
    vap_window = cfg.vap_window_frames(effective_fps)
    raw_path = smooth_path(raw, cfg.smoothing, vap_window, cfg.savgol_polyorder)
    corrected_path = smooth_path(
        corrected, cfg.smoothing, vap_window, cfg.savgol_polyorder
    )

    vcl_px_s = path_length(raw) / duration
    vsl_px_s = net_displacement(raw) / duration
    vap_px_s = path_length(raw_path) / duration

    corrected_path_length = path_length(corrected)
    corrected_net = net_displacement(corrected)
    vcl_c = corrected_path_length / duration
    vsl_c = corrected_net / duration
    vap_c = path_length(corrected_path) / duration

    # ------------------------------------------------------------ um/s
    if um_per_px is not None:
        vcl_um_s: float | None = vcl_c * um_per_px
        vsl_um_s: float | None = vsl_c * um_per_px
        vap_um_s: float | None = vap_c * um_per_px
    else:
        vcl_um_s = vsl_um_s = vap_um_s = None

    # ---------------------------------------------------------- ratios
    lin = _ratio(vsl_c, vcl_c)
    str_ = _ratio(vsl_c, vap_c)
    wob = _ratio(vap_c, vcl_c)

    # -------------------------------------------------------- direction
    delta = corrected[-1] - corrected[0]
    direction_rad = (
        float(math.atan2(delta[1], delta[0])) if corrected_net > EPS else None
    )
    direction_stability = _circular_std(corrected)

    # ---------------------------------------------------------- ALH/BCF
    alh_um, alh_reason, bcf_hz, bcf_reason = _alh_and_bcf(
        corrected,
        corrected_path,
        duration=duration,
        n_observed=n_observed,
        effective_fps=effective_fps,
        cfg=cfg,
        um_per_px=um_per_px,
    )

    return MotionFeatures(
        n_points=n_points,
        n_observed_points=n_observed,
        duration_s=duration,
        mean_frame_interval_s=mean_interval,
        timestamp_source=timestamp_source,
        flow_correction_mode=applied_mode,
        profile_version=_profile_tag(cfg, vap_window),
        optically_calibrated=um_per_px is not None,
        um_per_px=um_per_px,
        vcl_px_s=vcl_px_s,
        vsl_px_s=vsl_px_s,
        vap_px_s=vap_px_s,
        vcl_corrected_px_s=vcl_c,
        vsl_corrected_px_s=vsl_c,
        vap_corrected_px_s=vap_c,
        vcl_um_s=vcl_um_s,
        vsl_um_s=vsl_um_s,
        vap_um_s=vap_um_s,
        lin=lin,
        str_=str_,
        wob=wob,
        alh_um=alh_um,
        alh_unavailable_reason=alh_reason,
        bcf_hz=bcf_hz,
        bcf_unavailable_reason=bcf_reason,
        # Geometry describes the *corrected* trajectory, consistently with
        # direction_rad: dividing these by duration_s reproduces
        # vcl_corrected_px_s and vsl_corrected_px_s exactly.
        net_displacement_px=corrected_net,
        path_length_px=corrected_path_length,
        direction_rad=direction_rad,
        direction_stability=direction_stability,
        flow_vx_px_s=recorded_flow[0],
        flow_vy_px_s=recorded_flow[1],
        motility_reason=_NOT_CLASSIFIED,
    )


# ==========================================================================
# Lateral deviation, ALH and BCF
# ==========================================================================


def lateral_deviations(
    points_xy: NDArray[np.float64], path_xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Signed perpendicular offset of each point from the average path, px.

    The sign is the z-component of ``tangent x offset``, i.e. positive on one
    side of the average path and negative on the other, which is what makes
    counting *crossings* (BCF) possible at all -- an unsigned distance never
    changes sign and would give BCF = 0 for every track.

    The local tangent is a central difference of the average path. Where the
    average path is stationary (a cell going nowhere) there is no local
    direction and therefore no defined lateral axis; the net direction of the
    path is used as a fallback, and if that is degenerate too the deviation is
    reported as zero, which is the honest answer for a cell that does not move.
    """
    if points_xy.shape != path_xy.shape:
        raise ValueError(
            f"track {points_xy.shape} and average path {path_xy.shape} must match"
        )
    n = points_xy.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    tangents = np.gradient(path_xy, axis=0)
    norms = np.hypot(tangents[:, 0], tangents[:, 1])

    net = path_xy[-1] - path_xy[0]
    net_norm = float(np.hypot(net[0], net[1]))
    degenerate = norms <= EPS
    if np.any(degenerate):
        fallback = net / net_norm if net_norm > EPS else np.zeros(2)
        tangents = tangents.copy()
        tangents[degenerate] = fallback
        norms = np.where(degenerate, 1.0 if net_norm > EPS else np.inf, norms)

    offset = points_xy - path_xy
    cross = tangents[:, 0] * offset[:, 1] - tangents[:, 1] * offset[:, 0]
    return np.asarray(cross / norms, dtype=np.float64)


def _alh_and_bcf(
    corrected: NDArray[np.float64],
    corrected_path: NDArray[np.float64],
    *,
    duration: float,
    n_observed: int,
    effective_fps: float,
    cfg: MotionConfig,
    um_per_px: float | None,
) -> tuple[float | None, str, float | None, str]:
    """ALH and BCF, or ``None`` plus the reason they are untrustworthy.

    ALH is reported here as the **mean absolute lateral deviation of the
    measured track from its average path** -- a half-amplitude measured *about*
    the path, following the WHO wording ("magnitude of lateral displacement of
    a sperm head about its average path ... expressed as a maximum or an
    average"). Systems that report the peak-to-peak width instead publish
    numbers roughly twice these. The choice is fixed and documented rather than
    configurable, so that a number in an audit log always means one thing.

    **BCF is not a beat frequency.** It is the rate at which the curvilinear
    path crosses the average path, and nothing more. WHO 6th ed. section 4.5.1.4
    states explicitly that BCF has been shown *not* to correlate with the
    flagellar beat frequency (Gallagher et al., Hum Reprod 2019;34:1173-85):
    the head trace is a sparse, aliased projection of a three-dimensional
    flagellar wave. Presenting BCF as "beat frequency" in a UI or a report
    would be a physiological claim this measurement does not support. Note also
    that a crossing is counted in each direction, so a purely sinusoidal
    excursion at *f* Hz yields BCF = 2*f* crossings per second.

    Both quantities are computed on the flow-corrected trajectory: lateral
    wobble about the average path is a property of the cell, and bulk transport
    is not part of it.

    The sampling gate is not a nicety. Mortimer et al. (2015) call 60 images/s
    "really the minimum imaging frequency required for reliable human sperm
    track analysis"; below it VCL is under-estimated while LIN and ALH are
    over-estimated, all three in the direction that manufactures progressive
    motility. An aliased BCF in particular is an arbitrary number that still
    looks like a frequency, so both are refused outright rather than reported
    with a caveat nobody reads.
    """
    min_points = int(cfg.min_points_for_alh_bcf)
    if n_observed < min_points:
        reason = (
            f"insufficient sampling: {n_observed} observed points, "
            f"{min_points} required (min_points_for_alh_bcf)"
        )
        return None, reason, None, reason
    if effective_fps < float(cfg.min_fps_for_alh_bcf):
        reason = (
            f"under-sampled: effective rate {effective_fps:.1f} Hz from the "
            f"observed timestamps is below the {cfg.min_fps_for_alh_bcf:.1f} Hz "
            "needed to resolve the flagellar beat; an aliased value would be "
            "meaningless"
        )
        return None, reason, None, reason

    deviations = lateral_deviations(corrected, corrected_path)

    # BCF: crossings of the average path per second. Exact zeros carry no side
    # and are skipped rather than counted as a crossing in each direction --
    # with a symmetric shrinking window the endpoints sit exactly on the path.
    signs = np.sign(deviations)
    nonzero = signs[signs != 0.0]
    crossings = int(np.count_nonzero(np.diff(nonzero) != 0.0)) if nonzero.size > 1 else 0
    bcf_hz = float(crossings) / duration if duration > EPS else None
    bcf_reason = "" if bcf_hz is not None else "zero duration"

    if um_per_px is None:
        return (
            None,
            "ALH is a physical length: no optical calibration, so it cannot be "
            "reported in micrometres (the lateral deviation in pixels is "
            "available from the trajectory)",
            bcf_hz,
            bcf_reason,
        )

    alh_um = float(np.mean(np.abs(deviations))) * um_per_px
    return alh_um, "", bcf_hz, bcf_reason


# ==========================================================================
# Helpers
# ==========================================================================


def _profile_tag(cfg: MotionConfig, vap_window: int) -> str:
    """Threshold-profile version plus the average-path algorithm actually used.

    VAP, STR, WOB, ALH and BCF are only meaningful alongside the smoother that
    produced them, and the frame count is resolved per track from that track's
    measured frame rate -- so it belongs in the per-track record, not only in
    the run header. The configured profile version stays as the prefix so that
    equality-on-prefix still identifies the threshold set.
    """
    return (
        f"{cfg.thresholds.profile_version}"
        f"+vap={cfg.smoothing}:{vap_window}f@{cfg.vap_window_ms:g}ms"
    )


def _sorted_observed(points: list[TrackPoint]) -> list[TrackPoint]:
    """Measured points only, in time order.

    The tracker appends in frame order, so this sort is normally a no-op; it is
    done anyway (stably, on ``(capture_time_s, frame_id)``) because every
    downstream formula assumes monotone time, and an out-of-order point would
    otherwise show up as a negative interval inside a mean.
    """
    return sorted(
        (p for p in points if p.observed),
        key=lambda p: (p.capture_time_s, p.frame_id),
    )


def _observed_duration(observed: list[TrackPoint]) -> float:
    if len(observed) < 2:
        return 0.0
    return max(0.0, observed[-1].capture_time_s - observed[0].capture_time_s)


def _mean_interval(observed: list[TrackPoint]) -> float:
    if len(observed) < 2:
        return 0.0
    return _observed_duration(observed) / (len(observed) - 1)


def _resolve_um_per_px(optical: OpticalCalibration | None) -> float | None:
    """The calibrated scale, or ``None``. Never a default, never a guess."""
    if optical is None:
        return None
    try:
        return float(optical.require_calibrated())
    except CalibrationError:
        # Uncalibrated is an expected state, not a failure: the pipeline runs
        # in pixel units and simply refuses to report micrometres.
        return None


def _ratio(numerator: float, denominator: float) -> float | None:
    """Guarded dimensionless ratio, clamped to ``[0, 1]``.

    ``None`` rather than ``inf``/``nan`` when the denominator vanishes: an
    immotile cell has VCL of zero and its LIN is genuinely unknown, and a
    ``nan`` would compare false against every threshold, silently grading the
    cell as non-progressive for the wrong reason.

    The clamp is a numerical guard, not a fudge. VSL <= VAP <= VCL holds
    exactly in the continuum (the straight line is the shortest path, and the
    average path shares its endpoints with the raw one), so the only way to
    exceed 1 is float round-off on a perfectly straight track.
    """
    if denominator <= EPS:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        return None
    return float(min(1.0, max(0.0, value)))


def _circular_std(points_xy: NDArray[np.float64]) -> float | None:
    """Circular standard deviation of the per-step headings, radians.

    Headings are directions, so they must be averaged on the circle: the
    arithmetic mean of 359 degrees and 1 degree is 180 degrees, which is the
    opposite of the truth. With :math:`\\theta_i` the heading of step *i*,

    .. math::
        R = \\left\\lvert \\frac{1}{n}\\sum_i e^{j\\theta_i} \\right\\rvert,
        \\qquad s = \\sqrt{-2 \\ln R}

    is the standard circular SD (Mardia). ``R = 1`` (every step in the same
    direction) gives ``s = 0``; the more the headings scatter, the larger
    ``s`` grows, without bound as ``R -> 0``.

    Zero-length steps have no heading and are dropped rather than counted as
    heading zero, which would fake straightness for a stationary cell. Fewer
    than two usable headings gives ``None``: one step is always perfectly
    "stable", which is not information.
    """
    if points_xy.shape[0] < 3:
        return None
    deltas = np.diff(points_xy, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    usable = lengths > EPS
    if int(np.count_nonzero(usable)) < 2:
        return None
    headings = np.arctan2(deltas[usable, 1], deltas[usable, 0])
    resultant = float(
        np.hypot(np.mean(np.cos(headings)), np.mean(np.sin(headings)))
    )
    # R can round to just above 1 for identical headings, and to exactly 0 for
    # perfectly opposed ones; both are clamped into the domain of the log.
    resultant = min(1.0, max(EPS, resultant))
    # max(0.0, ...) turns the -0.0 that sqrt(-2*log(1.0)) produces for a
    # perfectly straight track into a plain 0.0.
    return float(math.sqrt(max(0.0, -2.0 * math.log(resultant))))
