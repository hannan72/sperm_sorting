"""Motility grading and the track-quality bar.

The grade
---------
The four grades are the WHO laboratory manual's own four-category motility
system, 6th edition (2021) section 2.4.6.1, which reinstated the split that the
5th edition had collapsed into PR/NP/IM: rapidly progressive, slowly
progressive, non-progressive, immotile. WHO's PR is rapid + slow, NP is
non-progressive, IM is immotile. The default cut-points, 25 um/s and 5 um/s,
are WHO's own approximate velocity limits, not values invented here -- see
:class:`~sperm_sorting.config.MotilityThresholds` for the manual's caveats
about how approximate they are.

The rule, on **flow-corrected** kinematics and in micrometres per second:

===================  =========================================================
Grade                Condition
===================  =========================================================
RAPID_PROGRESSIVE    ``VSL >= rapid_progressive_vsl_um_s`` and LIN sufficient
SLOW_PROGRESSIVE     ``slow <= VSL < rapid`` and LIN sufficient
NON_PROGRESSIVE      ``VSL < slow`` but ``VCL > immotile_vcl_um_s``; also any
                     fast-but-not-linear cell demoted by the LIN floor
IMMOTILE             no motion above the calibrated noise floor
UNDETERMINED         the grade cannot be established (see below)
===================  =========================================================

Temperature
-----------
WHO section 2.4.6 requires the sample to be at 37 C, because sperm velocity is
temperature-dependent: the same cell graded at room temperature is slower and
falls into a lower category. When ``thresholds.temperature_in_spec`` is false
the grade is still produced -- a bench test at room temperature is a legitimate
thing to run -- but every reason string carries a note that the grading is not
WHO-comparable, so the caveat travels with the record into the audit log
instead of living only in someone's memory of how the run was set up.

Three refusals are deliberate and each of them prevents a specific wrong
answer:

**No optical calibration, no grade.** The thresholds are in micrometres per
second. A pixel velocity compared against them is a category error whose
outcome depends on the objective in the turret: at 0.5 um/px a cell moving 20
px/s is 10 um/s -- *slow progressive* -- while the uncorrected comparison of
``20 >= 25`` says non-progressive, and at 2 um/px the same comparison is wrong
in the other direction. It would not raise, it would not warn, and the shot
ratio would just be wrong. So an uncalibrated system returns UNDETERMINED with
a reason naming the missing calibration.

**No flow estimate, no grade** (:class:`ProgressiveMotilityClassifier` only,
which is the layer that knows what was configured). A dead cell drifting at
120 px/s in the bulk flow is straight and fast; uncorrected it grades rapid
progressive. If a correction was configured but none could be produced for
this track, the honest answer is "unknown", not "progressive".

**Fast but circling is not progressing.** A cell swimming tight circles has a
large VCL and a small VSL, and hyperactivated cells in particular can post a
high VSL over a short window while going nowhere. Both progressive grades
therefore also require ``LIN >= min_lin_for_progressive``. When LIN is
unavailable (VCL of zero, i.e. no measurable path at all) the cell is
*demoted*, never promoted, and the reason says so.

This LIN floor is **stricter than WHO**, and is this implementation's choice
rather than a WHO criterion: the manual's own wording admits progression
"either linearly or in a large circle", which a linearity floor will reject.
It is applied because this pipeline's downstream action is a physical sort, and
a large-circle swimmer does not reliably leave the imaging region. Setting
``min_lin_for_progressive = 0.0`` disables the floor completely -- including
the demotion for an unavailable LIN -- for anyone who wants WHO's wording
followed literally.

The track-quality bar
---------------------
:func:`assess_track_quality` decides whether a track is a trustworthy
observation of one sperm at all. A track that fails is excluded from the shot
entirely -- numerator *and* denominator -- so this is a stricter thing than a
bad grade, and is kept separate from grading for that reason.
"""

from __future__ import annotations

from typing import Any

from ..config import (
    MotilityThresholds,
    MotionConfig,
    OpticalCalibration,
    TrackQualityConfig,
)
from ..constants import EPS
from ..schemas.enums import FlowCorrectionMode, MotilityClass, TimestampSource
from ..schemas.track import MotionFeatures, TrackRecord
from .features import FlowSampler, compute_motion_features

__all__ = [
    "ProgressiveMotilityClassifier",
    "assess_track_quality",
    "classify_motility",
]


# ==========================================================================
# Grading
# ==========================================================================


def classify_motility(
    features: MotionFeatures, thresholds: MotilityThresholds
) -> tuple[MotilityClass, str]:
    """Grade one track's motility from its kinematics.

    Parameters
    ----------
    features
        Kinematics from :func:`~.features.compute_motion_features`. The
        **corrected** micrometre velocities are what is compared; the raw ones
        are ignored here entirely.
    thresholds
        Versioned cut-points. ``thresholds.profile_version`` is what makes a
        historical log re-interpretable, so it is never bypassed with literals.

    Returns
    -------
    (MotilityClass, str)
        The grade and a human-readable reason, suitable for the audit log.
        Pure: nothing is mutated, including ``features``.

    Notes
    -----
    A record that already carries an UNDETERMINED verdict *with* a reason was
    settled upstream (too few observed points, zero elapsed time) and is passed
    through unchanged rather than being re-explained less accurately here.

    When the sample temperature is out of WHO specification the grade is still
    returned, with the non-comparability note appended to the reason. The note
    is attached only to the branches that actually applied a velocity
    threshold; a track refused for want of calibration has not been graded at
    any temperature.
    """
    if features.motility_class is MotilityClass.UNDETERMINED and features.motility_reason:
        return MotilityClass.UNDETERMINED, features.motility_reason

    if features.n_observed_points < 2 or features.duration_s <= EPS:
        return (
            MotilityClass.UNDETERMINED,
            f"undetermined: {features.n_observed_points} observed point(s) over "
            f"{features.duration_s:.6g} s cannot define a velocity",
        )

    if not features.optically_calibrated or features.vsl_um_s is None:
        return (
            MotilityClass.UNDETERMINED,
            "undetermined: no optical calibration, so the micrometre-per-second "
            "motility thresholds cannot be applied to pixel velocities "
            f"(corrected VSL {features.vsl_corrected_px_s:.2f} px/s, corrected "
            f"VCL {features.vcl_corrected_px_s:.2f} px/s). Run "
            "scripts/calibrate_optics.py.",
        )

    vsl = float(features.vsl_um_s)
    vcl = float(features.vcl_um_s) if features.vcl_um_s is not None else 0.0
    lin = features.lin
    rapid = thresholds.rapid_progressive_vsl_um_s
    slow = thresholds.slow_progressive_vsl_um_s
    min_lin = thresholds.min_lin_for_progressive
    # A floor of zero is not "LIN >= 0", which would still demote a track whose
    # LIN could not be computed. It means the linearity criterion is switched
    # off entirely, which is how WHO's own wording is followed literally.
    lin_floor_active = min_lin > 0.0
    note = _temperature_note(thresholds)

    def graded(grade: MotilityClass, reason: str) -> tuple[MotilityClass, str]:
        return grade, reason + note

    # ---------------------------------------------------- progressive grades
    if vsl >= slow:
        grade = (
            MotilityClass.RAPID_PROGRESSIVE
            if vsl >= rapid
            else MotilityClass.SLOW_PROGRESSIVE
        )
        band = "rapid" if grade is MotilityClass.RAPID_PROGRESSIVE else "slow"
        if lin_floor_active and lin is None:
            return graded(
                MotilityClass.NON_PROGRESSIVE,
                f"non-progressive: corrected VSL {vsl:.1f} um/s meets the {band} "
                f"progressive cut, but LIN is unavailable (corrected VCL "
                f"{vcl:.1f} um/s), so progression cannot be confirmed and the "
                "grade is demoted rather than assumed",
            )
        if lin_floor_active and lin is not None and lin < min_lin:
            return graded(
                MotilityClass.NON_PROGRESSIVE,
                f"non-progressive: corrected VSL {vsl:.1f} um/s meets the {band} "
                f"progressive cut but LIN {lin:.2f} < {min_lin:.2f}; the cell is "
                "moving fast without going anywhere (circling)",
            )
        cut = rapid if grade is MotilityClass.RAPID_PROGRESSIVE else slow
        linearity = (
            f"with LIN {lin:.2f} >= {min_lin:.2f}"
            if lin_floor_active and lin is not None
            else "with no linearity floor applied (min_lin_for_progressive=0)"
        )
        return graded(
            grade,
            f"{band} progressive: corrected VSL {vsl:.1f} um/s >= {cut:.1f} um/s "
            f"{linearity}",
        )

    # ------------------------------------------- non-progressive vs immotile
    if vcl > thresholds.immotile_vcl_um_s:
        return graded(
            MotilityClass.NON_PROGRESSIVE,
            f"non-progressive: corrected VSL {vsl:.1f} um/s < {slow:.1f} um/s so "
            f"there is no net progression, but corrected VCL {vcl:.1f} um/s > "
            f"{thresholds.immotile_vcl_um_s:.1f} um/s shows local movement",
        )
    return graded(
        MotilityClass.IMMOTILE,
        f"immotile: corrected VCL {vcl:.1f} um/s <= "
        f"{thresholds.immotile_vcl_um_s:.1f} um/s and corrected VSL "
        f"{vsl:.1f} um/s; no motion distinguishable from measurement noise",
    )


def _temperature_note(thresholds: MotilityThresholds) -> str:
    """Non-comparability note for a sample that was not held at 37 C.

    Velocity is temperature-dependent, so the WHO cut-points only mean what
    they say at the temperature WHO specifies. The run is not refused -- bench
    tests at room temperature are legitimate -- but the caveat is attached to
    the individual grade so that it survives into the audit log rather than
    depending on someone remembering how the run was configured.
    """
    if thresholds.temperature_in_spec:
        return ""
    return (
        f" [not WHO-comparable: sample held at "
        f"{thresholds.sample_temperature_c:.1f} C, outside 37.0 +/- "
        f"{thresholds.temperature_tolerance_c:.1f} C; sperm velocity is "
        "temperature-dependent, so these micrometre-per-second cut-points do "
        "not carry their WHO meaning at this temperature]"
    )


class ProgressiveMotilityClassifier:
    """Config-bound wrapper around :func:`classify_motility`.

    Holds the pieces the free function deliberately does not see -- the
    configured flow-correction mode, the tracker noise floor and the optical
    calibration -- and applies the two safety rules that need them:

    1. **Refuse to grade uncorrected kinematics** when a flow correction was
       configured but none was applied to this track. Without it, every
       passively drifting cell in a live channel grades progressive.
    2. **Collapse sub-noise-floor movement to IMMOTILE.** ``noise_floor_px_s``
       is the tracker's own positional jitter expressed as a velocity. Below
       it, "local movement" is the box centre trembling, not the cell.
    """

    name = "progressive_motility_classifier"

    def __init__(
        self,
        cfg: MotionConfig,
        optical: OpticalCalibration | None = None,
        timestamp_source: TimestampSource = TimestampSource.HOST_MONOTONIC,
    ) -> None:
        self.cfg = cfg
        self.optical = optical
        self.timestamp_source = timestamp_source

    # ------------------------------------------------------------- grading

    def classify(
        self,
        track: TrackRecord,
        flow_vector: tuple[float, float] | None = None,
        *,
        flow_sampler: FlowSampler | None = None,
        recompute: bool = False,
    ) -> MotilityClass:
        """Grade ``track``, computing and attaching its kinematics if needed.

        ``track.motion`` is created when absent (or when ``recompute`` is set)
        and is updated in place with the grade and its reason, so the track
        record remains the single place a decision can be reconstructed from.
        Returns the grade for convenience.
        """
        if track.motion is None or recompute:
            track.motion = compute_motion_features(
                track,
                self.cfg,
                self.optical,
                flow_vector,
                self.timestamp_source,
                flow_sampler=flow_sampler,
            )
        features = track.motion

        grade, reason = self._decide(features)
        features.motility_class = grade
        features.motility_reason = reason
        return grade

    def classify_features(self, features: MotionFeatures) -> tuple[MotilityClass, str]:
        """Grade an existing record without mutating it."""
        return self._decide(features)

    def _decide(self, features: MotionFeatures) -> tuple[MotilityClass, str]:
        configured = self.cfg.flow_correction.mode
        applied = features.flow_correction_mode
        if (
            configured is not FlowCorrectionMode.DISABLED
            and applied is FlowCorrectionMode.DISABLED
        ):
            return (
                MotilityClass.UNDETERMINED,
                f"undetermined: flow correction '{configured}' was configured "
                "but no flow estimate was available for this track, and grading "
                "uncorrected kinematics would report bulk transport as "
                "progressive swimming",
            )

        grade, reason = classify_motility(features, self.cfg.thresholds)

        if grade is MotilityClass.NON_PROGRESSIVE:
            floor = float(self.cfg.noise_floor_px_s)
            if features.vcl_corrected_px_s <= floor:
                return (
                    MotilityClass.IMMOTILE,
                    f"immotile: corrected VCL {features.vcl_corrected_px_s:.2f} "
                    f"px/s is at or below the tracker noise floor {floor:.2f} "
                    "px/s, so the apparent local movement is positional jitter"
                    + _temperature_note(self.cfg.thresholds),
                )
        return grade, reason

    def describe(self) -> dict[str, Any]:
        """Metadata stamped into the audit log header."""
        thresholds = self.cfg.thresholds
        return {
            "name": self.name,
            "profile_version": thresholds.profile_version,
            "rapid_progressive_vsl_um_s": thresholds.rapid_progressive_vsl_um_s,
            "slow_progressive_vsl_um_s": thresholds.slow_progressive_vsl_um_s,
            "immotile_vcl_um_s": thresholds.immotile_vcl_um_s,
            "min_lin_for_progressive": thresholds.min_lin_for_progressive,
            "lin_floor_is_stricter_than_who": thresholds.min_lin_for_progressive > 0.0,
            "sample_temperature_c": thresholds.sample_temperature_c,
            "temperature_in_spec": thresholds.temperature_in_spec,
            "noise_floor_px_s": self.cfg.noise_floor_px_s,
            "vap_window_ms": self.cfg.vap_window_ms,
            "smoothing": self.cfg.smoothing,
            "flow_correction_mode": str(self.cfg.flow_correction.mode),
            "optically_calibrated": bool(
                self.optical is not None and self.optical.calibrated
            ),
        }


# ==========================================================================
# Track quality
# ==========================================================================


def assess_track_quality(
    track: TrackRecord, cfg: TrackQualityConfig
) -> tuple[bool, str]:
    """Decide whether a track is a trustworthy observation of one sperm.

    Four independent ways a track can fail to be evidence:

    * **too few measured points** -- a two-frame track has one velocity sample
      and no way to tell a real swim from an association error;
    * **too short a lifetime** -- velocity over a millisecond is dominated by
      centroid noise;
    * **too much interpolation** -- a track more than half predicted is mostly
      the motion model's story about a sperm rather than an observation of it;
    * **too low a mean detector score** -- weak evidence that the thing being
      tracked is a sperm at all.

    Every failing criterion is reported, not just the first, because when a
    batch of tracks is being rejected the pattern across criteria is what
    identifies the cause.

    Side effect: writes ``track.track_quality_pass`` and
    ``track.track_quality_reason``. This is the one mutation the function
    performs, and it is the documented contract -- the shot manager reads those
    fields rather than re-deriving the rule.
    """
    n_points = len(track.points)
    observed = track.observed_points
    n_observed = len(observed)

    failures: list[str] = []

    if n_points == 0:
        failures.append("track has no points")
    if n_observed < cfg.min_observed_points:
        failures.append(
            f"{n_observed} observed points < {cfg.min_observed_points} required"
        )

    duration = (
        observed[-1].capture_time_s - observed[0].capture_time_s
        if n_observed >= 2
        else 0.0
    )
    if duration < cfg.min_duration_s:
        failures.append(
            f"observed lifetime {duration:.4f} s < {cfg.min_duration_s:.4f} s required"
        )

    interpolated_fraction = (
        1.0 - (n_observed / n_points) if n_points > 0 else 1.0
    )
    if interpolated_fraction > cfg.max_interpolated_fraction:
        failures.append(
            f"interpolated fraction {interpolated_fraction:.2f} > "
            f"{cfg.max_interpolated_fraction:.2f} allowed "
            f"({n_points - n_observed} of {n_points} points predicted)"
        )

    if track.mean_score < cfg.min_mean_score:
        failures.append(
            f"mean detector score {track.mean_score:.3f} < "
            f"{cfg.min_mean_score:.3f} required"
        )

    if failures:
        passed, reason = False, "track quality fail: " + "; ".join(failures)
    else:
        passed, reason = True, (
            f"track quality pass: {n_observed} observed points over "
            f"{duration:.4f} s, {interpolated_fraction:.2f} interpolated, "
            f"mean score {track.mean_score:.3f}"
        )

    track.track_quality_pass = passed
    track.track_quality_reason = reason
    return passed, reason
