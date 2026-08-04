"""Whole-frame image-quality gate.

The gate exists to stop unusable frames from contaminating measurements that
are reported as facts. A defocused frame still yields detections and still
yields a velocity; it just yields a *wrong* one, and nothing downstream can
tell the difference after the fact. Rejecting the frame and counting the drop
keeps the failure visible instead of folding it into the result.

Three verdicts, three different downstream consequences:

``PASS``
    Usable for everything, including morphology crops.
``DEGRADED``
    Usable for tracking continuity -- dropping frames mid-track breaks
    identity, which costs more than a slightly soft frame -- but never
    eligible for a morphology crop. The best-frame selector enforces that via
    ``BestFrameConfig.require_frame_quality_pass``.
``REJECT``
    Dropped. :meth:`ImageQualityGate.apply` returns ``None`` and the caller
    counts the drop.

All measurements are taken on a normalised ``0-1`` view of the frame (see
:func:`~sperm_sorting.preprocessing.preprocessor.to_unit_float`) so that a
``uint8`` frame and the ``float32`` frame the preprocessor derives from it
measure identically and one set of thresholds covers both.

The one deliberate exception is the focus score, which is reported in
**8-bit-equivalent grey levels**: the normalised view is multiplied by 255
before the Laplacian. Variance of Laplacian scales with the square of the
intensity scale, so a 0-1 image would put ``min_focus_score`` around 1e-4 and
make the configured defaults (8.0 / 20.0, chosen for 8-bit imagery)
meaningless. Multiplying by a constant keeps the number dtype-independent
*and* keeps the configured thresholds interpretable.
"""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np

from ..config import QualityGateConfig
from ..schemas.enums import QualityVerdict
from ..schemas.frame import FramePacket, FrameQuality
from .preprocessor import ensure_mono2d, to_unit_float

__all__ = ["ImageQualityGate"]

#: Normalised intensity at or above which a pixel counts as saturated. 0.99
#: corresponds to grey level 253 in 8-bit, i.e. the top three codes.
_DEFAULT_SATURATION_LEVEL: Final[float] = 0.99

#: Normalised intensity at or below which a pixel counts as underexposed.
_DEFAULT_BLACK_LEVEL: Final[float] = 0.01

#: Multiplicative margin defining the soft (DEGRADED) bar from the hard
#: (REJECT) one. An upper limit's soft bar is ``margin * limit``; a lower
#: limit's soft bar is ``limit / margin``. 0.5 therefore means "you are within
#: a factor of two of being rejected".
_DEFAULT_SOFT_MARGIN: Final[float] = 0.5

#: Absolute distance from the mean-intensity bars at which a frame is called
#: soft. Mean intensity is already a bounded 0-1 quantity, so an absolute
#: margin is more meaningful here than a multiplicative one.
_DEFAULT_MEAN_SOFT_MARGIN: Final[float] = 0.05


class ImageQualityGate:
    """Measures and classifies whole-frame image quality.

    Parameters
    ----------
    cfg
        Validated :class:`~sperm_sorting.config.QualityGateConfig`.
    saturation_level, black_level
        Normalised intensities defining "saturated" and "underexposed". These
        are sensor and illumination dependent and should be re-tuned on device
        data; the defaults correspond to the top and bottom three 8-bit codes.
    soft_margin, mean_soft_margin
        How close to a hard bar a frame may come before it is called
        ``DEGRADED`` rather than ``PASS``. The point of the soft band is that
        a frame drifting toward rejection is visible in the metrics *before*
        frames start disappearing.
    """

    __slots__ = (
        "_black_level",
        "_mean_soft_margin",
        "_saturation_level",
        "_soft_margin",
        "cfg",
        "n_degraded",
        "n_pass",
        "n_reject",
    )

    def __init__(
        self,
        cfg: QualityGateConfig,
        *,
        saturation_level: float = _DEFAULT_SATURATION_LEVEL,
        black_level: float = _DEFAULT_BLACK_LEVEL,
        soft_margin: float = _DEFAULT_SOFT_MARGIN,
        mean_soft_margin: float = _DEFAULT_MEAN_SOFT_MARGIN,
    ) -> None:
        self.cfg = cfg
        self._saturation_level = float(saturation_level)
        self._black_level = float(black_level)
        self._soft_margin = float(soft_margin)
        self._mean_soft_margin = float(mean_soft_margin)
        self.n_pass = 0
        self.n_degraded = 0
        self.n_reject = 0

    # --------------------------------------------------------------- counters

    @property
    def n_total(self) -> int:
        """Frames evaluated through :meth:`apply` since the last reset."""
        return self.n_pass + self.n_degraded + self.n_reject

    def counters(self) -> dict[str, int]:
        """Snapshot for the metrics layer.

        Exposed as a plain dict because the monitoring layer serialises it
        straight into a metrics record; a live view would be racy.
        """
        return {
            "n_pass": self.n_pass,
            "n_degraded": self.n_degraded,
            "n_reject": self.n_reject,
            "n_total": self.n_total,
        }

    def reset_counters(self) -> None:
        self.n_pass = 0
        self.n_degraded = 0
        self.n_reject = 0

    # ------------------------------------------------------------ evaluation

    def evaluate(self, frame: FramePacket) -> FrameQuality:
        """Measure one frame and classify it. Does not touch the counters.

        Pure with respect to the gate's state, so it can be used for offline
        threshold tuning over a recording without corrupting the live
        statistics.
        """
        unit = to_unit_float(ensure_mono2d(frame.image))

        # Variance of the Laplacian on the 8-bit-equivalent scale; see the
        # module docstring for why the factor of 255 is there.
        scaled = unit.astype(np.float32) * np.float32(255.0)
        laplacian = cv2.Laplacian(scaled, cv2.CV_32F, ksize=3)
        focus = float(laplacian.var())

        mean_intensity = float(unit.mean())
        contrast = float(unit.std())
        n_pixels = float(unit.size) if unit.size else 1.0
        saturated = float(np.count_nonzero(unit >= self._saturation_level)) / n_pixels
        underexposed = float(np.count_nonzero(unit <= self._black_level)) / n_pixels

        verdict, reason = self._classify(
            focus, mean_intensity, contrast, saturated, underexposed
        )
        return FrameQuality(
            verdict=verdict,
            focus_score=focus,
            mean_intensity=mean_intensity,
            contrast=contrast,
            saturated_fraction=saturated,
            underexposed_fraction=underexposed,
            reason=reason,
        )

    def apply(self, frame: FramePacket) -> FramePacket | None:
        """Evaluate, count, attach and return -- or ``None`` if rejected.

        Returning ``None`` rather than raising is deliberate: a rejected frame
        is an expected operating condition (a bubble drifting through the
        field, a momentary defocus), not an error. The caller increments its
        drop counter and carries on.

        The quality record is attached to the packet in place. That is what
        :class:`~sperm_sorting.schemas.frame.FramePacket` documents ("populated
        by the quality gate"), and it costs no allocation on the hot path. The
        pixel buffer is never touched, so replay determinism is unaffected.

        ``DEGRADED`` frames are returned, not dropped, regardless of
        ``degraded_frames_feed_tracking``: whether they reach the tracker is
        the caller's routing decision, exposed by :meth:`should_feed_tracking`.
        Dropping them here would deny the caller the chance to count them.
        """
        quality = self.evaluate(frame)
        if quality.verdict is QualityVerdict.PASS:
            self.n_pass += 1
        elif quality.verdict is QualityVerdict.DEGRADED:
            self.n_degraded += 1
        else:
            self.n_reject += 1
            frame.quality = quality
            return None
        frame.quality = quality
        return frame

    def should_feed_tracking(self, quality: FrameQuality | None) -> bool:
        """Whether a frame with this verdict may be handed to the tracker.

        A ``DEGRADED`` frame is usually still worth tracking: losing a frame
        mid-track can break a track's identity, and identity is the accounting
        unit of the whole product, so continuity normally outweighs a slightly
        soft image. ``degraded_frames_feed_tracking=False`` overrides that for
        setups where soft frames produce bad boxes rather than merely blurry
        ones.
        """
        if quality is None:
            return True
        if quality.verdict is QualityVerdict.REJECT:
            return False
        if quality.verdict is QualityVerdict.DEGRADED:
            return self.cfg.degraded_frames_feed_tracking
        return True

    @staticmethod
    def is_morphology_eligible(quality: FrameQuality | None) -> bool:
        """Only a ``PASS`` frame may supply a morphology crop.

        An absent quality record is *not* a pass: morphology decides whether a
        sperm is counted as eligible, and that must never rest on an
        unmeasured frame.
        """
        return quality is not None and quality.verdict is QualityVerdict.PASS

    def describe(self) -> dict[str, Any]:
        """Metadata for the audit-log header."""
        return {
            "enabled": self.cfg.enabled,
            "min_focus_score": self.cfg.min_focus_score,
            "degraded_focus_score": self.cfg.degraded_focus_score,
            "saturation_level": self._saturation_level,
            "black_level": self._black_level,
            "soft_margin": self._soft_margin,
            "focus_units": "8-bit-equivalent variance of Laplacian",
        }

    # -------------------------------------------------------------- internal

    def _classify(
        self,
        focus: float,
        mean_intensity: float,
        contrast: float,
        saturated: float,
        underexposed: float,
    ) -> tuple[QualityVerdict, str]:
        """Turn measurements into a verdict plus a human-readable reason.

        Hard bars reject; soft bars degrade. Every violated bar is reported,
        not just the first, because a frame that fails on three counts at once
        is diagnostically different from one that fails on a single marginal
        threshold.
        """
        if not self.cfg.enabled:
            # Measurements are still produced (they feed the metrics layer and
            # offline tuning), but the gate never drops a frame when disabled.
            return QualityVerdict.PASS, ""

        cfg = self.cfg
        hard: list[str] = []
        soft: list[str] = []

        # -- focus ----------------------------------------------------------
        if focus < cfg.min_focus_score:
            hard.append(
                f"defocused: focus {focus:.2f} < min_focus_score {cfg.min_focus_score:.2f}"
            )
        elif focus < cfg.degraded_focus_score:
            soft.append(
                f"soft focus: {focus:.2f} < degraded_focus_score "
                f"{cfg.degraded_focus_score:.2f}"
            )

        # -- exposure: mean level -------------------------------------------
        if mean_intensity < cfg.min_mean_intensity:
            hard.append(
                f"exposure too low: mean {mean_intensity:.3f} < "
                f"{cfg.min_mean_intensity:.3f}"
            )
        elif mean_intensity > cfg.max_mean_intensity:
            hard.append(
                f"exposure too high: mean {mean_intensity:.3f} > "
                f"{cfg.max_mean_intensity:.3f}"
            )
        elif mean_intensity < cfg.min_mean_intensity + self._mean_soft_margin:
            soft.append(f"exposure near lower bar: mean {mean_intensity:.3f}")
        elif mean_intensity > cfg.max_mean_intensity - self._mean_soft_margin:
            soft.append(f"exposure near upper bar: mean {mean_intensity:.3f}")

        # -- contrast --------------------------------------------------------
        if contrast < cfg.min_contrast:
            hard.append(
                f"low contrast: std {contrast:.4f} < {cfg.min_contrast:.4f}"
            )
        elif self._soft_margin > 0.0 and contrast < cfg.min_contrast / self._soft_margin:
            soft.append(f"contrast near lower bar: std {contrast:.4f}")

        # -- exposure: clipping ----------------------------------------------
        if saturated > cfg.max_saturated_fraction:
            hard.append(
                f"exposure clipped: saturated fraction {saturated:.3f} > "
                f"{cfg.max_saturated_fraction:.3f}"
            )
        elif saturated > cfg.max_saturated_fraction * self._soft_margin:
            soft.append(f"exposure: saturated fraction {saturated:.3f} approaching bar")

        if underexposed > cfg.max_underexposed_fraction:
            hard.append(
                f"exposure crushed: underexposed fraction {underexposed:.3f} > "
                f"{cfg.max_underexposed_fraction:.3f}"
            )
        elif underexposed > cfg.max_underexposed_fraction * self._soft_margin:
            soft.append(
                f"exposure: underexposed fraction {underexposed:.3f} approaching bar"
            )

        if hard:
            return QualityVerdict.REJECT, "; ".join(hard)
        if soft:
            return QualityVerdict.DEGRADED, "; ".join(soft)
        return QualityVerdict.PASS, ""
