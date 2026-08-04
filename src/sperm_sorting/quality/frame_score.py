"""Per-candidate crop quality scoring.

This module scores **one sperm in one frame**, not a whole frame. The
whole-frame gate in :mod:`sperm_sorting.preprocessing.quality_gate` answers
"is this frame usable at all"; this answers "of the frames in which this
particular track was actually seen, which one gives the morphology model its
best look at *this* cell". A frame can be globally excellent and still be a
poor look at one sperm that happens to be overlapping a neighbour, clipped by
the frame border, or smeared along its direction of travel.

The composite score is a weighted sum of eight terms, each mapped into
``[0, 1]`` before weighting, with weights taken from
:class:`~sperm_sorting.config.BestFrameConfig` (validated there to sum to 1.0).
Because every term is in ``[0, 1]`` and the weights sum to 1, the composite is
in ``[0, 1]`` by construction -- there is no post-hoc clamp hiding a bug.

Detector confidence
-------------------
``BestFrameConfig`` refuses ``w_detector_score >= 0.5``. That rule is easy to
defeat by accident, because ``track_confidence`` is the track's *mean detector
score* and is therefore the same quantity averaged over time. This module
closes that loophole explicitly: :func:`validate_weights` refuses a config
whose detector-derived weights (``w_detector_score + w_track_confidence``)
reach 0.5 in total, and no other term is computed from any detector output.
The remaining six terms are measured from pixels and geometry alone.

Normalisation constants
-----------------------
Every raw measurement is a physical quantity in sensor units and has to be
mapped into ``[0, 1]`` by some scale constant. Those constants live in
:class:`ScoreNormalisation` with defaults chosen for 8-bit brightfield
microscopy at the design magnification. **They are illumination-, optics- and
sensor-dependent and must be re-tuned on device data**; the raw measurements
are therefore also reported in the breakdown dict (``raw_*`` keys) so a
recording can be replayed and the constants fitted without re-running
detection. The mappings are:

============= =================================== ==============================
term          raw measurement                     mapping into [0, 1]
============= =================================== ==============================
focus         variance of Laplacian inside the    ``v / (v + focus_half_sat)``
              box, 8-bit-equivalent units         saturating, 0.5 at the constant
motion_blur   structure-tensor coherence          ``1 - coherence``
              ``(l1 - l2) / (l1 + l2)`` in [0, 1]
local_contrast intensity std inside the box, 0-1  ``min(std / contrast_ref, 1)``
exposure      saturated + underexposed fraction   ``1 - min(f / exposure_tol, 1)``
              inside the box
overlap       max IoU with neighbouring           ``1 - iou``
              detections
truncation    fraction of the *padded* box that   used directly
              lies inside the frame
detector_score detector confidence                clipped to [0, 1]
track_confidence track mean detector score        clipped to [0, 1]
============= =================================== ==============================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import cv2
import numpy as np

from ..config import BestFrameConfig
from ..errors import ConfigurationError
from ..preprocessing.preprocessor import ensure_mono2d, to_unit_float
from ..schemas.detection import BoundingBox, Detection
from ..schemas.enums import QualityVerdict
from ..schemas.frame import FrameQuality

__all__ = [
    "DEFAULT_NORMALISATION",
    "ScoreNormalisation",
    "describe_normalisation",
    "padded_box",
    "score_candidate",
    "validate_weights",
    "visible_fraction_of",
]

#: Combined ceiling on detector-derived weights. Mirrors the ``< 0.5`` rule
#: ``BestFrameConfig`` applies to ``w_detector_score`` alone.
_MAX_DETECTOR_DERIVED_WEIGHT: Final[float] = 0.5

#: Below this many pixels on a side, a box is too small for a meaningful
#: Laplacian or structure tensor (a 3x3 kernel needs at least 3 rows/columns
#: and a 1-pixel margin to avoid being dominated by border replication).
_MIN_ANALYSIS_SIDE_PX: Final[int] = 5


@dataclass(frozen=True, slots=True)
class ScoreNormalisation:
    """Scale constants mapping raw measurements into ``[0, 1]``.

    Frozen so a tuned instance can be shared between the selector and any
    offline analysis without the risk of one of them mutating it.
    """

    #: Variance of Laplacian (8-bit-equivalent) at which the focus term
    #: reaches 0.5. Raising it makes the term stricter about sharpness. Note
    #: the term saturates for high-contrast imagery (synthetic scenes with
    #: hard edges reach 1e5 and score ~1.0 whether or not they are smeared);
    #: on such data the ranking is carried by the motion-blur and contrast
    #: terms, which is the intended division of labour but is another reason
    #: this constant has to be re-fitted on real device data.
    focus_half_saturation: float = 250.0
    #: Normalised intensity std at which the contrast term saturates at 1.0.
    #: 0.12 is roughly 30 grey levels in 8-bit.
    contrast_reference: float = 0.12
    #: Fraction of clipped pixels inside the box at which the exposure term
    #: reaches 0. A morphology crop with a tenth of its pixels at the rails
    #: has lost the acrosome/vacuole detail the model needs.
    exposure_tolerance: float = 0.10
    #: Normalised intensity at/above which a pixel counts as saturated.
    saturation_level: float = 0.99
    #: Normalised intensity at/below which a pixel counts as underexposed.
    black_level: float = 0.01
    #: Gradient-magnitude floor, in 8-bit-equivalent units, below which the
    #: structure tensor is pure noise and its coherence is meaningless. Such a
    #: box scores 0 on motion blur rather than getting a random number.
    min_structure_energy: float = 1.0
    #: IoU at or above which a "neighbour" is taken to be the candidate's own
    #: detection passed in by mistake, and is skipped.
    self_iou_threshold: float = 0.999


DEFAULT_NORMALISATION: Final[ScoreNormalisation] = ScoreNormalisation()


def validate_weights(cfg: BestFrameConfig) -> None:
    """Refuse a weighting in which detector confidence dominates.

    ``BestFrameConfig`` already rejects ``w_detector_score >= 0.5``. This adds
    the check that rule implies but cannot express, because it only sees one
    field at a time: ``track_confidence`` is the same detector score averaged
    over the track, so the two together are the detector's total vote. A
    weighting that gave them 0.5 between them would let a confidently-detected
    but motion-blurred sperm win, which is exactly the failure the rule exists
    to prevent.
    """
    detector_derived = float(cfg.w_detector_score) + float(cfg.w_track_confidence)
    if detector_derived >= _MAX_DETECTOR_DERIVED_WEIGHT:
        raise ConfigurationError(
            "detector-derived weights may not dominate best-frame selection: "
            f"w_detector_score ({cfg.w_detector_score}) + w_track_confidence "
            f"({cfg.w_track_confidence}) = {detector_derived:.3f} >= "
            f"{_MAX_DETECTOR_DERIVED_WEIGHT}. track_confidence is the track's "
            "mean detector score, so it counts toward the detector's vote."
        )


# ==========================================================================
# Geometry helpers
# ==========================================================================


def padded_box(
    box: BoundingBox, padding_fraction: float, min_padding_px: float
) -> BoundingBox:
    """Grow ``box`` the way :class:`CropExtractor` will.

    The truncation term has to be measured on the box that will actually be
    cut, not on the detection box: a detection sitting comfortably inside the
    frame can still have its padded crop hang over the edge, and it is the
    padding that carries the tail.

    Kept here (rather than only in the extractor) so the selector can evaluate
    truncation *before* committing to a frame; the extractor imports this same
    function so the two can never drift apart.
    """
    side = max(box.width, box.height)
    pad = max(float(min_padding_px), float(padding_fraction) * side)
    return box.expanded(pad, pad)


def visible_fraction_of(box: BoundingBox, width: int, height: int) -> float:
    """Fraction of ``box`` lying inside a ``width`` x ``height`` frame."""
    area = box.area
    if area <= 0.0:
        return 0.0
    clipped = box.clipped(float(width), float(height))
    return float(min(1.0, max(0.0, clipped.area / area)))


def _integer_region(
    box: BoundingBox, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Pixel index range covered by ``box``, clipped; ``None`` if empty."""
    x1 = int(np.floor(box.x1))
    y1 = int(np.floor(box.y1))
    x2 = int(np.ceil(box.x2))
    y2 = int(np.ceil(box.y2))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _as_box(item: Detection | BoundingBox) -> BoundingBox:
    return item.box if isinstance(item, Detection) else item


# ==========================================================================
# Individual terms
# ==========================================================================


def _focus_term(
    patch_8bit: np.ndarray, norm: ScoreNormalisation
) -> tuple[float, float]:
    """Local sharpness: variance of the Laplacian inside the box.

    Returns ``(term, raw_variance)``. The saturating map ``v / (v + k)`` is
    used rather than a linear ramp because sharpness has no natural maximum:
    beyond "clearly in focus", more Laplacian energy mostly means more noise,
    and a linear map would keep rewarding it.
    """
    if min(patch_8bit.shape) < _MIN_ANALYSIS_SIDE_PX:
        return 0.0, 0.0
    laplacian = cv2.Laplacian(patch_8bit, cv2.CV_32F, ksize=3)
    # Drop a 1-pixel frame: OpenCV replicates the border, which fabricates
    # gradient energy that has nothing to do with the object.
    interior = laplacian[1:-1, 1:-1]
    variance = float(interior.var()) if interior.size else 0.0
    k = max(1e-9, norm.focus_half_saturation)
    return variance / (variance + k), variance


def _motion_blur_term(
    patch_8bit: np.ndarray, norm: ScoreNormalisation
) -> tuple[float, float]:
    """Directional-gradient anisotropy from the gradient structure tensor.

    Estimator
    ---------
    Sobel gradients ``gx``, ``gy`` are accumulated over the box into the
    gradient covariance (structure) matrix::

        J = [[sum gx*gx, sum gx*gy],
             [sum gx*gy, sum gy*gy]]

    and reduced to the *coherence*::

        C = (l1 - l2) / (l1 + l2)
          = sqrt((Jxx - Jyy)^2 + 4 Jxy^2) / (Jxx + Jyy)

    with ``l1 >= l2`` the eigenvalues. ``C`` is 0 for gradients spread evenly
    over all directions and 1 when every gradient shares one orientation. It
    needs no eigen-decomposition (the closed form above), no threshold, and no
    estimate of the blur direction, which is what makes it cheap enough for a
    6 ms frame budget.

    Why it detects motion blur: a linear smear is a convolution with a line
    kernel. It destroys the intensity variation *along* the direction of
    travel while leaving the variation *across* it intact, so the surviving
    gradients collapse onto one axis and ``C`` rises toward 1. Defocus, by
    contrast, is isotropic and lowers gradient magnitude everywhere without
    raising ``C`` -- which is why blur and focus are two separate terms rather
    than one.

    The term is ``1 - C``, so isotropic (unsmeared) detail scores high.

    Known limitation, stated plainly: a sperm is an intrinsically elongated
    object, so even a perfectly frozen one has somewhat anisotropic gradients
    and never reaches ``C = 0``. The term therefore is not an absolute measure
    of blur. It is used only to *rank frames of the same cell against each
    other*, where the intrinsic elongation is common to all candidates and
    cancels out of the comparison. Do not interpret an absolute value.

    Returns ``(term, raw_coherence)``.
    """
    if min(patch_8bit.shape) < _MIN_ANALYSIS_SIDE_PX:
        return 0.0, 0.0
    gx = cv2.Sobel(patch_8bit, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_8bit, cv2.CV_32F, 0, 1, ksize=3)
    gx = gx[1:-1, 1:-1]
    gy = gy[1:-1, 1:-1]
    if gx.size == 0:
        return 0.0, 0.0
    jxx = float(np.mean(gx * gx, dtype=np.float64))
    jyy = float(np.mean(gy * gy, dtype=np.float64))
    jxy = float(np.mean(gx * gy, dtype=np.float64))
    trace = jxx + jyy
    if trace <= norm.min_structure_energy:
        # No structure at all: featureless patch. Coherence would be the ratio
        # of two noise terms, so report the worst case rather than a number
        # that is really a coin flip.
        return 0.0, 1.0
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / trace)
    coherence = min(1.0, max(0.0, coherence))
    return 1.0 - coherence, coherence


def _contrast_term(
    patch_unit: np.ndarray, norm: ScoreNormalisation
) -> tuple[float, float]:
    """Local contrast: intensity std inside the box, on the 0-1 view.

    A low-contrast crop of a sperm is one where head, midpiece and tail are
    not separable from the medium; the morphology model cannot recover what
    the optics did not deliver. Linear up to ``contrast_reference``, then
    saturated -- beyond that, more contrast does not make the crop better.
    """
    std = float(patch_unit.std()) if patch_unit.size else 0.0
    ref = max(1e-9, norm.contrast_reference)
    return min(1.0, std / ref), std


def _exposure_term(
    patch_unit: np.ndarray, norm: ScoreNormalisation
) -> tuple[float, float, float]:
    """Penalise clipping *inside the box*, at both rails.

    Both directions matter and for different reasons: saturated pixels erase
    the acrosome boundary, crushed pixels erase the tail against the
    background. A frame-level exposure check cannot see either, because a few
    hundred clipped pixels on one sperm are invisible in a 2.3 megapixel mean.

    Returns ``(term, saturated_fraction, underexposed_fraction)``.
    """
    if patch_unit.size == 0:
        return 0.0, 0.0, 0.0
    n = float(patch_unit.size)
    saturated = float(np.count_nonzero(patch_unit >= norm.saturation_level)) / n
    underexposed = float(np.count_nonzero(patch_unit <= norm.black_level)) / n
    tolerance = max(1e-9, norm.exposure_tolerance)
    clipped = (saturated + underexposed) / tolerance
    return max(0.0, 1.0 - min(1.0, clipped)), saturated, underexposed


def _overlap_term(
    box: BoundingBox,
    neighbours: Sequence[Detection | BoundingBox] | None,
    norm: ScoreNormalisation,
) -> tuple[float, float]:
    """``1 - max IoU`` against the other detections in the same frame.

    An overlapping neighbour puts a second cell's pixels inside the crop, and
    the morphology model has no way to know which cell it is being asked
    about. The resulting judgement would be bound to the wrong track -- the
    exact failure the crop/track identity invariant exists to prevent.

    Neighbours are expected to exclude the candidate's own detection; one that
    is not excluded (IoU ~ 1) is skipped defensively so a caller mistake
    degrades into a warning-free correct answer rather than a permanent
    zero-overlap score for every candidate.

    Returns ``(term, raw_max_iou)``.
    """
    if not neighbours:
        return 1.0, 0.0
    max_iou = 0.0
    for item in neighbours:
        other = _as_box(item)
        if other is box:
            continue
        iou = box.iou(other)
        if iou >= norm.self_iou_threshold:
            continue
        if iou > max_iou:
            max_iou = iou
    return 1.0 - max_iou, max_iou


# ==========================================================================
# Composite
# ==========================================================================


def score_candidate(
    image: np.ndarray,
    box: BoundingBox,
    neighbours: Sequence[Detection | BoundingBox] | None,
    detector_score: float,
    track_confidence: float,
    frame_quality: FrameQuality | None,
    cfg: BestFrameConfig,
    *,
    padding_fraction: float = 0.35,
    min_padding_px: float = 4.0,
    normalisation: ScoreNormalisation = DEFAULT_NORMALISATION,
) -> tuple[float, dict[str, float]]:
    """Score one sperm in one frame as a morphology crop candidate.

    Parameters
    ----------
    image
        The frame the candidate was observed in, ``uint8``/``uint16`` or
        ``float32`` in ``[0, 1]``. Both dtypes are accepted and give the same
        score for the same scene, because every measurement is taken on the
        normalised view.
    box
        The candidate's detection box, in the coordinate frame of ``image``.
    neighbours
        The other detections in the same frame (``Detection`` or bare
        ``BoundingBox``), used for the overlap term. ``None`` or empty means
        an isolated cell.
    detector_score, track_confidence
        The detector's confidence for this observation and the track's mean
        detector score. Both are clipped to ``[0, 1]``; together they carry at
        most ``w_detector_score + w_track_confidence`` of the composite, which
        :func:`validate_weights` holds below 0.5.
    frame_quality
        The whole-frame verdict, if measured. It is **recorded** in the
        breakdown (``diag_*`` keys) but deliberately **not weighted**: the
        frame verdict is already a hard admission filter in
        :class:`~sperm_sorting.quality.selector.BestFrameSelector`
        (``require_frame_quality_pass``), and folding it in here as well would
        count the same evidence twice with an undeclared weight.
    cfg
        Weights and thresholds.
    padding_fraction, min_padding_px
        Must match :class:`~sperm_sorting.config.CropConfig`, because the
        truncation term is measured on the padded box that will actually be
        cut. :class:`BestFrameSelector` passes the crop config's values.
    normalisation
        Scale constants; see :class:`ScoreNormalisation`.

    Returns
    -------
    (score, terms)
        ``score`` is the weighted composite in ``[0, 1]``. ``terms`` holds
        every normalised term under its own name, the raw measurements under
        ``raw_*``, and frame-level diagnostics under ``diag_*``. The whole
        dict is stored on
        :attr:`~sperm_sorting.schemas.track.CropRecord.quality_terms`, so a
        crop's selection can be re-derived from the audit log alone.
    """
    validate_weights(cfg)
    mono = ensure_mono2d(image)
    height, width = mono.shape

    region = _integer_region(box, width, height)
    if region is None:
        # The box lies entirely outside the image. Every pixel-based term is
        # undefined; report a zero score rather than inventing measurements.
        empty_terms: dict[str, float] = {
            "focus": 0.0,
            "motion_blur": 0.0,
            "local_contrast": 0.0,
            "exposure": 0.0,
            "overlap": 0.0,
            "truncation": 0.0,
            "detector_score": float(min(1.0, max(0.0, detector_score))),
            "track_confidence": float(min(1.0, max(0.0, track_confidence))),
            "raw_focus_variance": 0.0,
            "raw_coherence": 1.0,
            "raw_contrast_std": 0.0,
            "raw_saturated_fraction": 0.0,
            "raw_underexposed_fraction": 0.0,
            "raw_max_overlap_iou": 1.0,
            "raw_visible_fraction": 0.0,
        }
        _add_frame_diagnostics(empty_terms, frame_quality)
        return 0.0, empty_terms

    x1, y1, x2, y2 = region
    patch_unit = to_unit_float(mono[y1:y2, x1:x2])
    # 8-bit-equivalent scale keeps the Laplacian and Sobel magnitudes in the
    # same units as the frame-level focus score, so the two are comparable and
    # the normalisation constants transfer between them.
    patch_8bit = patch_unit.astype(np.float32) * np.float32(255.0)

    focus, raw_focus = _focus_term(patch_8bit, normalisation)
    blur, raw_coherence = _motion_blur_term(patch_8bit, normalisation)
    contrast, raw_std = _contrast_term(patch_unit, normalisation)
    exposure, raw_sat, raw_under = _exposure_term(patch_unit, normalisation)
    overlap, raw_iou = _overlap_term(box, neighbours, normalisation)

    padded = padded_box(box, padding_fraction, min_padding_px)
    truncation = visible_fraction_of(padded, width, height)

    det = float(min(1.0, max(0.0, detector_score)))
    trk = float(min(1.0, max(0.0, track_confidence)))

    terms: dict[str, float] = {
        "focus": float(focus),
        "motion_blur": float(blur),
        "local_contrast": float(contrast),
        "exposure": float(exposure),
        "overlap": float(overlap),
        "truncation": float(truncation),
        "detector_score": det,
        "track_confidence": trk,
        # Raw measurements travel with the record so the normalisation
        # constants above can be re-fitted from a recording without having to
        # re-run detection and tracking.
        "raw_focus_variance": float(raw_focus),
        "raw_coherence": float(raw_coherence),
        "raw_contrast_std": float(raw_std),
        "raw_saturated_fraction": float(raw_sat),
        "raw_underexposed_fraction": float(raw_under),
        "raw_max_overlap_iou": float(raw_iou),
        "raw_visible_fraction": float(truncation),
    }
    _add_frame_diagnostics(terms, frame_quality)

    score = (
        cfg.w_focus * terms["focus"]
        + cfg.w_motion_blur * terms["motion_blur"]
        + cfg.w_local_contrast * terms["local_contrast"]
        + cfg.w_exposure * terms["exposure"]
        + cfg.w_overlap * terms["overlap"]
        + cfg.w_truncation * terms["truncation"]
        + cfg.w_detector_score * terms["detector_score"]
        + cfg.w_track_confidence * terms["track_confidence"]
    )
    # Weights sum to 1 and every term is in [0, 1], so this is already in
    # [0, 1]; the clamp only absorbs floating-point dust at the ends.
    return float(min(1.0, max(0.0, score))), terms


def _add_frame_diagnostics(
    terms: dict[str, float], frame_quality: FrameQuality | None
) -> None:
    """Record whole-frame context without weighting it. See ``score_candidate``."""
    if frame_quality is None:
        return
    terms["diag_frame_focus"] = float(frame_quality.focus_score)
    terms["diag_frame_contrast"] = float(frame_quality.contrast)
    terms["diag_frame_mean_intensity"] = float(frame_quality.mean_intensity)
    terms["diag_frame_quality_pass"] = float(
        frame_quality.verdict is QualityVerdict.PASS
    )


def describe_normalisation(norm: ScoreNormalisation = DEFAULT_NORMALISATION) -> dict[str, Any]:
    """Constants stamped into the audit-log header.

    Logged because the composite score is meaningless without them: the same
    crop scores differently under a different ``focus_half_saturation``, and a
    log read six months later has to be able to tell which was in force.
    """
    return {
        "focus_half_saturation": norm.focus_half_saturation,
        "contrast_reference": norm.contrast_reference,
        "exposure_tolerance": norm.exposure_tolerance,
        "saturation_level": norm.saturation_level,
        "black_level": norm.black_level,
        "min_structure_energy": norm.min_structure_energy,
        "tuning_status": "defaults for 8-bit brightfield; re-tune on device data",
    }
