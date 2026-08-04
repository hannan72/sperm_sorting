"""Crop extraction for the morphology model.

Turns a selected :class:`~sperm_sorting.quality.selector.CandidateFrame` into
the fixed-size image the morphology model consumes, plus the
:class:`~sperm_sorting.schemas.track.CropRecord` that makes the crop auditable.

Two decisions in here are load-bearing:

**Aspect ratio is preserved by letterboxing, never by squashing.** The head
morphology classifier keys on head shape -- length against width is most of
what distinguishes a normal head from a tapered or amorphous one. Resizing a
non-square crop to a square input by stretching one axis changes that ratio by
exactly the amount the box was non-square, which is a systematic,
shape-dependent distortion of the single feature the model is being asked
about. Letterboxing spends a few border pixels instead.

**The crop is bound to the track it came from.**
``CropRecord.track_id == TrackRecord.track_id`` is checked here, not assumed;
:meth:`CropExtractor.extract` raises :class:`CropIdentityError` if it ever
fails. A crop that silently ends up on the wrong track pairs one cell's shape
with another cell's velocity, and the resulting eligibility decision would be
wrong in a way no downstream check could detect.

Output dtype policy
-------------------
``normalize="none"``
    Same dtype as the source frame (``uint8`` in, ``uint8`` out).
``normalize="minmax"``
    ``float32`` in ``[0, 1]``.
``normalize="zscore"``
    ``float32``, zero mean and unit standard deviation over the crop's
    *content* region, clipped to +/-4 sigma. Note this is deliberately **not**
    the same convention as ``PreprocessConfig.normalize="zscore"``, which
    squashes into ``[0, 1]`` because it produces a viewable frame. Here the
    consumer is a CNN, and standardised input is what a CNN expects; forcing
    it into ``[0, 1]`` would throw away the property that makes z-scoring
    worth doing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import cv2
import numpy as np

from ..config import CropConfig
from ..errors import SpermSortingError
from ..preprocessing.preprocessor import ensure_mono2d, to_uint8, to_unit_float
from ..quality.frame_score import padded_box, visible_fraction_of
from ..quality.selector import CandidateFrame
from ..schemas.detection import BoundingBox, Detection
from ..schemas.frame import FramePacket
from ..schemas.track import CropRecord, TrackRecord

__all__ = ["CropExtractor", "CropIdentityError"]

#: Standardised values are clipped here. Letterbox fill is a constant far from
#: the object's intensity distribution, so without a clip a large border could
#: produce a handful of extreme values that dominate the model's first layer.
_ZSCORE_CLIP: Final[float] = 4.0

#: Smallest crop side, in pixels, on which tail completeness is attempted. The
#: margin band would otherwise be a pixel or two wide and the answer would be
#: noise.
_MIN_TAIL_ANALYSIS_SIDE_PX: Final[int] = 16

#: Minimum separation, in 8-bit grey levels, between the Otsu foreground and
#: background means before the segmentation is trusted for tail completeness.
_MIN_TAIL_SEGMENT_SEPARATION: Final[float] = 8.0

#: Above this foreground area fraction, the "object" found by Otsu is really
#: the background (or a wall of debris) and nothing can be concluded.
_MAX_TAIL_FOREGROUND_FRACTION: Final[float] = 0.60


class CropIdentityError(SpermSortingError):
    """A crop was about to be recorded against the wrong track.

    This is a hard failure rather than a logged warning: the crop/track
    binding is the invariant that lets a morphology verdict be attributed to
    the sperm whose motion was measured. Continuing past a violation would
    produce an audit log that looks correct and is not.
    """


class CropExtractor:
    """Cuts, pads, letterboxes and normalises one morphology crop.

    Parameters
    ----------
    cfg
        Validated :class:`~sperm_sorting.config.CropConfig`. Its
        ``output_size`` is cross-checked against ``morphology.input_size`` by
        :class:`~sperm_sorting.config.AppConfig`, so no size negotiation is
        needed here.
    """

    __slots__ = ("cfg",)

    def __init__(self, cfg: CropConfig) -> None:
        out_h, out_w = cfg.output_size
        if out_h < 1 or out_w < 1:
            raise ValueError(f"crop.output_size must be positive, got {cfg.output_size}")
        if not 0.0 <= cfg.tail_margin_fraction < 0.5:
            raise ValueError(
                "crop.tail_margin_fraction must lie in [0, 0.5), got "
                f"{cfg.tail_margin_fraction}"
            )
        self.cfg = cfg

    # ------------------------------------------------------------------ api

    def extract(
        self,
        track: TrackRecord,
        candidate: CandidateFrame,
        frame: FramePacket,
        *,
        neighbours: Sequence[Detection | BoundingBox] | None = None,
    ) -> tuple[np.ndarray, CropRecord]:
        """Produce the model input and its audit record.

        Parameters
        ----------
        track
            The track the crop belongs to. Its ``track_id`` is stamped onto
            the record and verified before returning.
        candidate
            The winning candidate from
            :class:`~sperm_sorting.quality.selector.BestFrameSelector`.
        frame
            The frame named by ``candidate.frame_id``. Its ``image`` must be
            in the same coordinate frame as the candidate's box -- i.e. the
            *preprocessed* frame, after any ROI crop.
        neighbours
            Other detections in this frame, for ``max_overlap_iou``. When
            omitted, the value already measured during selection is reused
            rather than recomputed; pass them to recompute (for instance after
            a second detection pass).

        Returns
        -------
        (crop, record)
            ``crop`` has shape ``cfg.output_size`` and the dtype described in
            the module docstring. ``record`` is fully populated -- no field is
            left at its default because it was inconvenient to compute.
        """
        if candidate.frame_id != frame.frame_id:
            raise ValueError(
                f"candidate names frame {candidate.frame_id} but was handed frame "
                f"{frame.frame_id}; the crop must come from the frame that was scored"
            )

        image = ensure_mono2d(frame.image)
        height, width = image.shape

        padded = padded_box(
            candidate.box, self.cfg.padding_fraction, self.cfg.min_padding_px
        )
        visible_fraction = visible_fraction_of(padded, width, height)
        canvas, truncated = self._cut(image, padded, width, height)

        tail_complete = self._estimate_tail_complete(canvas, truncated)
        crop, content = self._resize_letterbox(canvas)
        crop = self._normalize(crop, content)

        max_overlap_iou = (
            self._max_iou(candidate.box, neighbours)
            if neighbours is not None
            else float(candidate.max_overlap_iou)
        )

        record = CropRecord(
            track_id=track.track_id,
            frame_id=candidate.frame_id,
            capture_time_s=float(frame.capture_time_s),
            source_box=padded,
            output_size=tuple(self.cfg.output_size),  # type: ignore[arg-type]
            quality_score=float(candidate.score),
            quality_terms=dict(candidate.terms),
            truncated=truncated,
            visible_fraction=float(visible_fraction),
            tail_complete=tail_complete,
            max_overlap_iou=float(max_overlap_iou),
            detector_score=float(candidate.detector_score),
            track_confidence=float(candidate.track_confidence),
        )

        # The product invariant, checked rather than assumed. Not an ``assert``
        # statement: those vanish under ``python -O`` and this must hold in
        # production, where the audit log is the only record of the decision.
        if record.track_id != track.track_id:  # pragma: no cover - guard
            raise CropIdentityError(
                f"crop bound to track {record.track_id} but extracted for track "
                f"{track.track_id}: the crop must belong to the same tracked "
                "sperm whose motion was measured"
            )
        return crop, record

    def describe(self) -> dict[str, Any]:
        """Metadata for the audit-log header."""
        return {
            "padding_fraction": self.cfg.padding_fraction,
            "min_padding_px": self.cfg.min_padding_px,
            "output_size": list(self.cfg.output_size),
            "preserve_aspect_ratio": self.cfg.preserve_aspect_ratio,
            "letterbox_value": self.cfg.letterbox_value,
            "normalize": self.cfg.normalize,
            "tail_margin_fraction": self.cfg.tail_margin_fraction,
        }

    # -------------------------------------------------------------- internal

    def _cut(
        self, image: np.ndarray, padded: BoundingBox, width: int, height: int
    ) -> tuple[np.ndarray, bool]:
        """Extract the padded box, filling anything outside the frame.

        The missing part is filled with ``letterbox_value`` rather than the
        box being shrunk to what is available. Shrinking would change the
        object's scale inside the crop -- a sperm clipped by the border would
        arrive at the model larger than an identical one in mid-frame, and
        scale is the other half of the head length:width story.
        """
        px1 = int(np.floor(padded.x1))
        py1 = int(np.floor(padded.y1))
        px2 = int(np.ceil(padded.x2))
        py2 = int(np.ceil(padded.y2))
        canvas_w = max(1, px2 - px1)
        canvas_h = max(1, py2 - py1)

        sx1, sy1 = max(0, px1), max(0, py1)
        sx2, sy2 = min(width, px2), min(height, py2)
        if sx2 <= sx1 or sy2 <= sy1:
            raise ValueError(
                f"padded box {padded.as_xyxy()} lies entirely outside the "
                f"{width}x{height} frame; there is nothing to crop"
            )

        fully_inside = (sx1, sy1, sx2, sy2) == (px1, py1, px2, py2)
        if fully_inside:
            # A view of the caller's frame. Never written to: every later step
            # allocates its own output.
            return image[sy1:sy2, sx1:sx2], False

        fill = self._fill_value(image.dtype)
        canvas = np.full((canvas_h, canvas_w), fill, dtype=image.dtype)
        canvas[sy1 - py1 : sy2 - py1, sx1 - px1 : sx2 - px1] = image[sy1:sy2, sx1:sx2]
        return canvas, True

    def _fill_value(self, dtype: np.dtype) -> Any:
        """``letterbox_value`` expressed in the source dtype's units.

        The config states the fill as an 8-bit grey level because that is how
        anyone looking at a crop thinks about it; float frames are in ``[0, 1]``
        so the same visual value has to be rescaled.
        """
        value = float(self.cfg.letterbox_value)
        if dtype.kind == "f":
            return np.array(value / 255.0, dtype=dtype)
        if dtype == np.uint16:
            return np.array(min(65535.0, value * 257.0), dtype=dtype)
        return np.array(value, dtype=dtype)

    def _resize_letterbox(
        self, canvas: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Resize to ``output_size``, letterboxing to preserve aspect ratio.

        Returns the output and the content rectangle ``(top, left, h, w)``
        inside it, so that normalisation can use the real pixels' statistics
        and ignore the synthetic border.
        """
        out_h, out_w = (int(v) for v in self.cfg.output_size)
        src_h, src_w = canvas.shape

        if not self.cfg.preserve_aspect_ratio:
            resized = cv2.resize(
                canvas, (out_w, out_h), interpolation=self._interpolation(
                    out_w / src_w, out_h / src_h
                )
            )
            return resized, (0, 0, out_h, out_w)

        scale = min(out_h / src_h, out_w / src_w)
        new_w = max(1, min(out_w, round(src_w * scale)))
        new_h = max(1, min(out_h, round(src_h * scale)))
        resized = cv2.resize(
            canvas, (new_w, new_h), interpolation=self._interpolation(scale, scale)
        )
        if (new_h, new_w) == (out_h, out_w):
            return resized, (0, 0, out_h, out_w)

        top = (out_h - new_h) // 2
        left = (out_w - new_w) // 2
        out = np.full(
            (out_h, out_w), self._fill_value(canvas.dtype), dtype=canvas.dtype
        )
        out[top : top + new_h, left : left + new_w] = resized
        return out, (top, left, new_h, new_w)

    @staticmethod
    def _interpolation(scale_x: float, scale_y: float) -> int:
        """``INTER_AREA`` when shrinking, ``INTER_LINEAR`` when growing.

        Area averaging is the correct anti-aliased downsample; using bilinear
        to shrink a 200-pixel sperm to 128 pixels aliases the tail into a
        dotted line, which is a morphology defect the model would then report.
        """
        return cv2.INTER_AREA if min(scale_x, scale_y) < 1.0 else cv2.INTER_LINEAR

    def _normalize(
        self, crop: np.ndarray, content: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Apply the per-crop normalisation. See the module dtype policy.

        Statistics come from the content region only. The letterbox border is
        a constant this module invented; letting it into the statistics would
        make the normalisation depend on how non-square the box happened to
        be, which is precisely the coupling letterboxing exists to remove.
        The affine map is then applied to the whole output so the border stays
        consistent with the content around it.
        """
        mode = self.cfg.normalize
        if mode == "none":
            return crop

        top, left, c_h, c_w = content
        unit = to_unit_float(crop, clip=True).astype(np.float32, copy=True)
        region = unit[top : top + c_h, left : left + c_w]

        if mode == "minmax":
            lo = float(region.min())
            hi = float(region.max())
            span = hi - lo
            if span <= 1e-12:
                # Flat crop: nothing to stretch. Zeros are deterministic and
                # will be visibly wrong to anyone inspecting the crop, which
                # is preferable to a division that quietly returns NaN.
                return np.zeros_like(unit)
            out = (unit - np.float32(lo)) * np.float32(1.0 / span)
            return np.clip(out, 0.0, 1.0, out=out)

        if mode == "zscore":
            mean = float(region.mean())
            std = float(region.std())
            if std <= 1e-12:
                return np.zeros_like(unit)
            out = (unit - np.float32(mean)) * np.float32(1.0 / std)
            return np.clip(out, -_ZSCORE_CLIP, _ZSCORE_CLIP, out=out)

        raise ValueError(f"unknown crop.normalize mode: {mode!r}")

    def _estimate_tail_complete(
        self, canvas: np.ndarray, truncated: bool
    ) -> bool | None:
        """Whether the whole tail appears to lie inside the crop.

        A tail leaving the crop cannot be judged: the classifier would be
        asked whether a tail is normal while looking at a fragment of it, and
        would answer confidently either way. The record therefore carries
        three states, and ``None`` -- "could not be determined" -- is a real
        answer that the morphology stage must handle, not a placeholder.

        Method: Otsu-threshold the crop, take the minority class as object
        pixels, and test whether any of them fall in a border band of width
        ``tail_margin_fraction * min(h, w)``.

        ``None`` is returned when:

        * the crop is truncated at the frame border -- part of the padded box
          was never imaged, so a tail extending into it is unobservable;
        * the crop is smaller than a few tens of pixels, where the band is too
          thin to mean anything;
        * Otsu finds no real separation (a flat or fogged crop), or the
          "object" covers most of the crop, in which case the segmentation is
          describing the background rather than the cell.

        Conservative by construction: a neighbouring cell's pixels in the band
        also count as touching, so a crowded crop is reported incomplete
        rather than complete. Under-claiming completeness costs a morphology
        evaluation; over-claiming it produces a tail verdict from a fragment.
        """
        if truncated:
            return None
        if min(canvas.shape) < _MIN_TAIL_ANALYSIS_SIDE_PX:
            return None

        grey = to_uint8(canvas)
        # OpenCV 5.0's Otsu path is 8-bit only, hence the conversion above.
        _threshold, mask = cv2.threshold(
            grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = mask > 0
        n_true = int(np.count_nonzero(binary))
        n_total = binary.size
        if n_true == 0 or n_true == n_total:
            return None

        # The cell is the minority class, whichever polarity the frame uses:
        # dark objects on a bright field (brightfield) or the inverse.
        object_mask = binary if n_true * 2 <= n_total else ~binary
        object_fraction = float(np.count_nonzero(object_mask)) / float(n_total)
        if object_fraction > _MAX_TAIL_FOREGROUND_FRACTION:
            return None

        fg_mean = float(grey[object_mask].mean())
        bg_mean = float(grey[~object_mask].mean())
        if abs(fg_mean - bg_mean) < _MIN_TAIL_SEGMENT_SEPARATION:
            # Otsu always returns a threshold, even for a flat image. Without
            # this check a fogged crop would report a crisp answer.
            return None

        height, width = object_mask.shape
        band = max(1, round(self.cfg.tail_margin_fraction * min(height, width)))
        if band * 2 >= min(height, width):
            return None

        touches = (
            bool(object_mask[:band, :].any())
            or bool(object_mask[-band:, :].any())
            or bool(object_mask[:, :band].any())
            or bool(object_mask[:, -band:].any())
        )
        return not touches

    @staticmethod
    def _max_iou(
        box: BoundingBox, neighbours: Sequence[Detection | BoundingBox] | None
    ) -> float:
        """Largest IoU between the crop's detection box and any neighbour.

        Measured against the unpadded detection box, matching the convention
        used during scoring, so ``CropRecord.max_overlap_iou`` is comparable
        with ``BestFrameConfig.max_overlap_iou``.
        """
        if not neighbours:
            return 0.0
        max_iou = 0.0
        for item in neighbours:
            other = item.box if isinstance(item, Detection) else item
            if other is box:
                continue
            iou = box.iou(other)
            # An IoU of 1 means the caller left the candidate's own detection
            # in the list; that is not a neighbour.
            if iou >= 0.999:
                continue
            max_iou = max(max_iou, iou)
        return float(max_iou)
