"""Frame preprocessing: ROI, inversion, background subtraction, normalisation.

Everything in this module runs once per frame at up to ~164 Hz for hours, so
three properties are non-negotiable and are the reason for most of the design
choices below:

* **Bounded memory.** The rolling-median background estimator owns exactly one
  pre-allocated ``(window, H, W)`` buffer and overwrites slots in place. No
  container grows with the number of frames processed.
* **No mutation of the caller's array.** Replay determinism means the same
  input bytes must produce the same output bytes on every run; if this stage
  wrote into the acquisition buffer, a second pass over the same recording
  would see different pixels. Every operation that changes values allocates a
  new array. Operations that only change *extent* (the ROI crop) return a
  numpy view, which is free and safe because it is never written to.
* **Determinism.** No randomness, no time-dependent behaviour, no
  data-dependent early exits other than the documented ones.

Output dtype policy
-------------------
``normalize="none"``
    The output keeps the input's integer dtype (``uint8`` in, ``uint8`` out;
    ``uint16`` in, ``uint16`` out). Full sensor range preserved.
``normalize="minmax" | "zscore" | "clahe"``
    The output is ``float32`` with values in ``[0, 1]``.

Both forms are legal inputs to everything downstream. Detectors and the
quality gate must therefore accept either; :func:`to_unit_float` and
:func:`to_uint8` are the two canonical converters and are used by the quality
gate, the crop scorer and the crop extractor so that a single convention is
applied everywhere rather than one per call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final

import cv2
import numpy as np

from ..config import PreprocessConfig
from ..errors import ConfigurationError
from ..schemas.detection import BoundingBox, Detection
from ..schemas.frame import FramePacket

__all__ = [
    "FramePreprocessor",
    "ensure_mono2d",
    "to_uint8",
    "to_unit_float",
    "translate_boxes_to_roi",
]

#: Rows processed per stripe when the rolling median is recomputed. The median
#: needs a temporary of ``window * stripe_rows * W`` elements; striping keeps
#: that temporary small and constant instead of proportional to the full frame
#: (a 64-frame window over 1920x1200 would otherwise need ~147 MB at once).
_MEDIAN_STRIPE_ROWS: Final[int] = 128

#: ``zscore`` maps ``z`` in ``[-ZSCORE_DISPLAY_SIGMA, +ZSCORE_DISPLAY_SIGMA]``
#: onto ``[0, 1]``. Three sigma covers essentially the whole distribution of a
#: well-exposed microscopy frame; values beyond it are clipped rather than
#: allowed to leave the documented output range.
_ZSCORE_DISPLAY_SIGMA: Final[float] = 3.0

#: Keys accepted when a ground-truth box is supplied as a mapping.
_GT_BOX_KEYS: Final[tuple[str, ...]] = ("box", "bbox", "box_xyxy", "xyxy")


# ==========================================================================
# Intensity conversion helpers (the one place these conventions live)
# ==========================================================================


def to_unit_float(image: np.ndarray, *, clip: bool = False) -> np.ndarray:
    """Return a ``float32`` view of ``image`` scaled to ``[0, 1]``.

    Integer inputs are divided by the maximum representable value of their
    dtype (255 for ``uint8``, 65535 for ``uint16``), which allocates. Float
    inputs are *contractually already* in ``[0, 1]`` -- that is what this
    package's normalising modes produce -- and are returned unchanged, which
    costs nothing. Pass ``clip=True`` to force the range when the array came
    from somewhere that does not honour the contract.

    Why this exists: thresholds such as ``QualityGateConfig.min_contrast`` are
    expressed in normalised units. Without a single converter, a ``uint8``
    frame and a ``float32`` frame of the same scene would be measured on
    different scales and the same threshold would mean two different things.
    """
    arr = np.asarray(image)
    if arr.dtype.kind == "f":
        out = arr.astype(np.float32, copy=False)
        return np.clip(out, 0.0, 1.0) if clip else out
    if arr.dtype.kind in "ui":
        max_value = float(np.iinfo(arr.dtype).max)
        out = arr.astype(np.float32) * np.float32(1.0 / max_value)
        # Signed dtypes can go negative; the pipeline never produces them, but
        # clamping is cheaper than a surprising negative contrast measurement.
        return np.clip(out, 0.0, 1.0) if arr.dtype.kind == "i" or clip else out
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    raise TypeError(f"unsupported image dtype for intensity conversion: {arr.dtype}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit view of ``image``.

    ``uint8`` passes through untouched (zero copy). ``uint16`` is shifted down
    by 8 bits, which is exact and branch-free. Floats are assumed to be in
    ``[0, 1]`` and are scaled with round-half-up.

    Needed because several OpenCV entry points this package relies on are
    8-bit-only in OpenCV 5.0 -- ``CLAHE.apply`` asserts ``CV_8UC1`` or
    ``CV_16UC1`` and Otsu thresholding refuses float input -- so the
    conversion has to happen somewhere explicit rather than by accident.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    if arr.dtype.kind == "f":
        return np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    if arr.dtype.kind in "ui":
        unit = to_unit_float(arr)
        return np.clip(unit * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    raise TypeError(f"unsupported image dtype for 8-bit conversion: {arr.dtype}")


def ensure_mono2d(image: np.ndarray) -> np.ndarray:
    """Validate the monochrome contract, tolerating a trailing singleton axis.

    :class:`FramePacket` documents a 2-D monochrome buffer because the target
    camera has no colour filter array. A 3-channel array here means an unclear
    provenance, and silently averaging it would hide the mistake.
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(
            f"expected a 2-D monochrome frame, got shape {arr.shape}; the "
            "pipeline never assumes three channels"
        )
    return arr


# ==========================================================================
# Ground-truth box translation
# ==========================================================================


def _shifted_xyxy(
    values: Sequence[float], dx: float, dy: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in values[:4])
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _clip_xyxy(
    xyxy: tuple[float, float, float, float], width: float, height: float
) -> tuple[float, float, float, float] | None:
    """Clip into the ROI, or return ``None`` when nothing is left inside it."""
    x1 = min(max(xyxy[0], 0.0), width)
    y1 = min(max(xyxy[1], 0.0), height)
    x2 = min(max(xyxy[2], 0.0), width)
    y2 = min(max(xyxy[3], 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def translate_boxes_to_roi(
    items: Any,
    offset_x: float,
    offset_y: float,
    roi_width: float,
    roi_height: float,
    *,
    clip: bool = True,
) -> Any:
    """Translate ground-truth boxes from sensor coordinates into ROI ones.

    The synthetic source publishes per-object ground truth in
    ``frame.meta["gt_detections"]``. The oracle detector reads it back and
    emits detections directly from it, and every synthetic accuracy test
    compares detector output against it. Those boxes are in *sensor* pixels;
    once an ROI has been cropped, the image the detector sees no longer starts
    at sensor ``(0, 0)``. Leaving the boxes untranslated does not raise -- it
    just silently shifts every ground-truth box by the ROI offset, which turns
    into a mysterious drop in measured recall. Hence this is done eagerly and
    unconditionally by :meth:`FramePreprocessor.process`.

    Accepted item forms (each is returned in the same form):

    * :class:`~sperm_sorting.schemas.detection.Detection`
    * :class:`~sperm_sorting.schemas.detection.BoundingBox`
    * a mapping with one of ``box`` / ``bbox`` / ``box_xyxy`` / ``xyxy``, or
      with explicit ``x1``/``y1``/``x2``/``y2`` keys
    * a 4-or-more element sequence ``(x1, y1, x2, y2, ...)``
    * an ``(N, >=4)`` numpy array, whose first four columns are translated

    With ``clip=True`` (the default) boxes are additionally clipped to the ROI
    and those with no remaining overlap are dropped, because an object outside
    the ROI is not present in the image the detector is handed and counting it
    as a miss would understate recall.
    """
    dx, dy = -float(offset_x), -float(offset_y)

    if isinstance(items, np.ndarray):
        if items.size == 0:
            return items.copy()
        if items.ndim == 1:
            # A single box stored bare rather than as a one-row table.
            single = _translate_one(items, dx, dy, roi_width, roi_height, clip)
            return single if single is not None else items[:0].copy()
        out = items.astype(np.float64, copy=True)
        out[:, 0] += dx
        out[:, 2] += dx
        out[:, 1] += dy
        out[:, 3] += dy
        if clip:
            np.clip(out[:, 0], 0.0, roi_width, out=out[:, 0])
            np.clip(out[:, 2], 0.0, roi_width, out=out[:, 2])
            np.clip(out[:, 1], 0.0, roi_height, out=out[:, 1])
            np.clip(out[:, 3], 0.0, roi_height, out=out[:, 3])
            keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
            out = out[keep]
        return out.astype(items.dtype, copy=False)

    if not isinstance(items, (list, tuple)):
        # Unknown container: leave it alone rather than corrupt it. The caller
        # owns meta and may legitimately store something exotic there.
        return items

    translated: list[Any] = []
    for item in items:
        moved = _translate_one(item, dx, dy, roi_width, roi_height, clip)
        if moved is not None:
            translated.append(moved)
    return type(items)(translated) if isinstance(items, tuple) else translated


def _translate_one(
    item: Any, dx: float, dy: float, roi_w: float, roi_h: float, clip: bool
) -> Any:
    """Translate one ground-truth item; ``None`` means "drop it"."""
    if isinstance(item, Detection):
        xyxy = _shifted_xyxy(item.box.as_xyxy(), dx, dy)
        if clip:
            clipped = _clip_xyxy(xyxy, roi_w, roi_h)
            if clipped is None:
                return None
            xyxy = clipped
        return replace(item, box=BoundingBox.from_xyxy(*xyxy))

    if isinstance(item, BoundingBox):
        xyxy = _shifted_xyxy(item.as_xyxy(), dx, dy)
        if clip:
            clipped = _clip_xyxy(xyxy, roi_w, roi_h)
            if clipped is None:
                return None
            xyxy = clipped
        return BoundingBox.from_xyxy(*xyxy)

    if isinstance(item, dict):
        out = dict(item)
        # Every recognised box key is translated, not just the first found: a
        # record carrying two spellings of the same box (``box`` and
        # ``box_xyxy``) must not end up with one of them in ROI coordinates and
        # the other in sensor coordinates.
        translated_any = False
        for key in _GT_BOX_KEYS:
            value = out.get(key)
            if value is None:
                continue
            xyxy = _shifted_xyxy(list(np.asarray(value, dtype=np.float64)), dx, dy)
            if clip:
                clipped = _clip_xyxy(xyxy, roi_w, roi_h)
                if clipped is None:
                    return None
                xyxy = clipped
            if isinstance(value, np.ndarray):
                out[key] = np.asarray(xyxy, dtype=value.dtype)
            elif isinstance(value, tuple):
                out[key] = tuple(xyxy)
            else:
                out[key] = list(xyxy)
            translated_any = True
        if translated_any:
            return out
        if {"x1", "y1", "x2", "y2"} <= out.keys():
            xyxy = _shifted_xyxy(
                (out["x1"], out["y1"], out["x2"], out["y2"]), dx, dy
            )
            if clip:
                clipped = _clip_xyxy(xyxy, roi_w, roi_h)
                if clipped is None:
                    return None
                xyxy = clipped
            out["x1"], out["y1"], out["x2"], out["y2"] = xyxy
            return out
        return out

    if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 4:
        xyxy = _shifted_xyxy(list(np.asarray(item, dtype=np.float64)), dx, dy)
        if clip:
            clipped = _clip_xyxy(xyxy, roi_w, roi_h)
            if clipped is None:
                return None
            xyxy = clipped
        rest = list(item[4:])
        if isinstance(item, np.ndarray):
            return np.asarray(list(xyxy) + rest, dtype=item.dtype)
        return type(item)(list(xyxy) + rest)

    # Anything else is not a box; pass it through untouched.
    return item


# ==========================================================================
# Rolling-median background
# ==========================================================================


class _RollingMedianBackground:
    """Fixed-capacity ring buffer holding the last ``window`` frames.

    Memory is ``window * H * W * itemsize`` and is allocated exactly once, on
    the first frame, in the frame's own dtype. Eviction is O(1): the oldest
    slot is simply overwritten. There is no list that can grow, and no
    per-frame allocation on the steady-state path other than the median
    temporaries, which are bounded by the stripe size.
    """

    __slots__ = (
        "_buffer",
        "_cached",
        "_count",
        "_next",
        "_refresh_interval",
        "_since_refresh",
        "_window",
    )

    def __init__(self, window: int, refresh_interval: int) -> None:
        if window < 1:
            raise ConfigurationError(
                f"preprocess.background_window must be >= 1, got {window}"
            )
        self._window = int(window)
        self._refresh_interval = max(1, int(refresh_interval))
        self._buffer: np.ndarray | None = None
        self._cached: np.ndarray | None = None
        self._count = 0
        self._next = 0
        self._since_refresh = 0

    # ------------------------------------------------------------------ api

    @property
    def window(self) -> int:
        return self._window

    @property
    def n_filled(self) -> int:
        return self._count

    def reset(self) -> None:
        """Forget the current estimate.

        Called when the frame geometry changes or the acquisition session is
        restarted: a background estimated from a different ROI, or from before
        a camera reconnect, describes an illumination field that no longer
        exists.
        """
        self._buffer = None
        self._cached = None
        self._count = 0
        self._next = 0
        self._since_refresh = 0

    def update(self, image: np.ndarray) -> np.ndarray | None:
        """Insert ``image`` and return the current background, or ``None``.

        ``None`` is returned while the buffer holds fewer than three frames:
        with one frame the "median" is the frame itself and subtraction would
        erase the image entirely, which is worse than not subtracting at all.
        """
        # A change of geometry or dtype means the buffered frames describe a
        # different image than the one arriving now, so the estimate restarts.
        if (
            self._buffer is None
            or self._buffer.shape[1:] != image.shape
            or self._buffer.dtype != image.dtype
        ):
            self.reset()
            self._buffer = np.empty((self._window, *image.shape), dtype=image.dtype)

        self._buffer[self._next] = image
        self._next = (self._next + 1) % self._window
        self._count = min(self._count + 1, self._window)

        if self._count < 3:
            return None
        if self._cached is None or self._since_refresh >= self._refresh_interval:
            self._cached = self._median()
            self._since_refresh = 0
        else:
            self._since_refresh += 1
        return self._cached

    # -------------------------------------------------------------- internal

    def _median(self) -> np.ndarray:
        """Per-pixel order statistic across the filled slots.

        Uses ``np.partition`` rather than ``np.median`` so the result keeps the
        buffer's integer dtype: ``np.median`` promotes to ``float64``, which
        would triple the working set for no gain. For an even number of filled
        slots this is the upper of the two central values rather than their
        mean -- a deterministic, documented choice, and a sub-grey-level
        difference on a background estimate.
        """
        buffer = self._buffer
        if buffer is None:  # pragma: no cover - update() allocates before calling
            raise RuntimeError("background buffer accessed before allocation")
        n = self._count
        kth = n // 2
        out = np.empty(buffer.shape[1:], dtype=buffer.dtype)
        rows = buffer.shape[1]
        for start in range(0, rows, _MEDIAN_STRIPE_ROWS):
            stop = min(start + _MEDIAN_STRIPE_ROWS, rows)
            stripe = buffer[:n, start:stop]
            out[start:stop] = np.partition(stripe, kth, axis=0)[kth]
        return out


# ==========================================================================
# Preprocessor
# ==========================================================================


class FramePreprocessor:
    """Applies ROI, inversion, background subtraction and normalisation.

    The order is fixed and meaningful:

    1. **ROI crop** -- everything after it is cheaper, and the background
       buffer only ever has to hold the region actually analysed.
    2. **Inversion** -- before background estimation, so the stored background
       matches the polarity the rest of the stage works in.
    3. **Background subtraction** -- removes static illumination
       non-uniformity and fixed-pattern sensor artefacts, which would
       otherwise be a permanent, position-dependent bias on every focus and
       contrast measurement downstream.
    4. **Normalisation** -- last, because it is the only step that changes the
       dtype, and doing it earlier would force the background buffer to hold
       floats (four times the memory) for no benefit.

    Parameters
    ----------
    cfg
        The validated :class:`~sperm_sorting.config.PreprocessConfig`.
    background_refresh_interval
        Frames between recomputations of the rolling median. ``None`` selects
        ``max(1, background_window // 8)``. A full-frame 64-deep median costs
        tens of milliseconds, which does not fit in a 6 ms frame budget; the
        background of a rigidly-mounted microscope changes on a timescale of
        seconds, so refreshing it every few frames loses nothing measurable
        and is fully deterministic. Set to ``1`` for an exact per-frame median
        when throughput does not matter (offline analysis, tests).
    clip_gt_to_roi
        Whether ground-truth boxes translated into ROI coordinates are also
        clipped to the ROI, dropping those that fall entirely outside it.
    """

    __slots__ = ("_background", "_clahe", "_clip_gt", "_last_session_id", "cfg")

    def __init__(
        self,
        cfg: PreprocessConfig,
        *,
        background_refresh_interval: int | None = None,
        clip_gt_to_roi: bool = True,
    ) -> None:
        self.cfg = cfg
        self._clip_gt = bool(clip_gt_to_roi)
        self._last_session_id: int | None = None

        if cfg.roi is not None:
            x, y, w, h = cfg.roi
            if w <= 0 or h <= 0:
                raise ConfigurationError(
                    f"preprocess.roi must have positive width and height, got {cfg.roi}"
                )
            if x < 0 or y < 0:
                raise ConfigurationError(
                    f"preprocess.roi origin must be non-negative, got {cfg.roi}"
                )

        interval = (
            max(1, cfg.background_window // 8)
            if background_refresh_interval is None
            else int(background_refresh_interval)
        )
        self._background = _RollingMedianBackground(cfg.background_window, interval)

        # CLAHE objects are stateless between calls but expensive to build, so
        # one is created per preprocessor rather than per frame.
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=float(cfg.clahe_clip_limit),
                tileGridSize=(int(cfg.clahe_tile_grid), int(cfg.clahe_tile_grid)),
            )
            if cfg.normalize == "clahe"
            else None
        )

    # ------------------------------------------------------------------ api

    def reset(self) -> None:
        """Drop the background estimate. Call between independent recordings."""
        self._background.reset()
        self._last_session_id = None

    @property
    def background_frames_buffered(self) -> int:
        """How many frames currently contribute to the background estimate."""
        return self._background.n_filled

    def process(self, frame: FramePacket) -> FramePacket:
        """Return a new :class:`FramePacket` holding the preprocessed image.

        The input packet and its pixel buffer are never modified. When no
        operation is configured the returned packet shares the caller's buffer
        (or a numpy view of it, for an ROI) rather than copying ~2.3 MB per
        frame; when any value-changing step runs, that step allocates.
        """
        image = ensure_mono2d(frame.image)

        # A session restart invalidates the background: the camera may have
        # been re-configured, and a stale background would be subtracted from
        # an illumination field it never described.
        if self._last_session_id is not None and frame.session_id != self._last_session_id:
            self._background.reset()
        self._last_session_id = frame.session_id

        image, roi, meta = self._apply_roi(image, frame)

        if self.cfg.invert:
            image = self._invert(image)

        if self.cfg.background_subtraction:
            image = self._subtract_background(image)

        image = self._normalize(image)

        return FramePacket(
            frame_id=frame.frame_id,
            image=image,
            capture_time_s=frame.capture_time_s,
            timestamp_source=frame.timestamp_source,
            source_kind=frame.source_kind,
            received_time_s=frame.received_time_s,
            dropped_before=frame.dropped_before,
            session_id=frame.session_id,
            quality=frame.quality,
            roi=roi,
            meta=meta,
            schema_version=frame.schema_version,
        )

    def describe(self) -> dict[str, Any]:
        """Metadata stamped into the audit-log header."""
        return {
            "roi": list(self.cfg.roi) if self.cfg.roi is not None else None,
            "normalize": self.cfg.normalize,
            "invert": self.cfg.invert,
            "background_subtraction": self.cfg.background_subtraction,
            "background_window": self.cfg.background_window,
            "output_dtype": (
                "uint8/uint16 (unchanged)"
                if self.cfg.normalize == "none"
                else "float32 in [0, 1]"
            ),
        }

    # -------------------------------------------------------------- internal

    def _apply_roi(
        self, image: np.ndarray, frame: FramePacket
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None, dict[str, Any]]:
        """Crop to the configured ROI and move ground truth with it."""
        if self.cfg.roi is None:
            # ``meta`` is shared rather than copied: nothing was changed, and
            # copying a dict per frame at 160 Hz for no reason is waste.
            return image, frame.roi, frame.meta

        x, y, w, h = (int(v) for v in self.cfg.roi)
        height, width = image.shape
        if x + w > width or y + h > height:
            raise ConfigurationError(
                f"preprocess.roi {self.cfg.roi} does not fit inside a "
                f"{width}x{height} frame"
            )

        # A basic slice: a view, not a copy. Nothing downstream writes to it.
        cropped = image[y : y + h, x : x + w]

        # ROIs compose. If acquisition already cropped the sensor, the packet
        # carries that offset and the recorded ROI must stay expressed in
        # original sensor pixels, as :class:`FramePacket` documents.
        if frame.roi is not None:
            roi = (frame.roi[0] + x, frame.roi[1] + y, w, h)
        else:
            roi = (x, y, w, h)

        meta = frame.meta
        if "gt_detections" in meta:
            meta = dict(meta)  # shallow copy: never mutate the caller's meta
            meta["gt_detections"] = translate_boxes_to_roi(
                meta["gt_detections"],
                offset_x=x,
                offset_y=y,
                roi_width=float(w),
                roi_height=float(h),
                clip=self._clip_gt,
            )
            meta["roi_offset"] = (x, y)
        return cropped, roi, meta

    @staticmethod
    def _invert(image: np.ndarray) -> np.ndarray:
        """Flip intensity polarity, allocating a new array.

        Brightfield microscopy gives dark objects on a bright field; some
        detectors and most segmentation heuristics are trained on the opposite
        convention.
        """
        if image.dtype.kind == "f":
            return (1.0 - image).astype(np.float32, copy=False)
        return (np.iinfo(image.dtype).max - image).astype(image.dtype, copy=False)

    def _subtract_background(self, image: np.ndarray) -> np.ndarray:
        """Subtract the rolling median, preserving the overall grey level.

        The residual is re-centred on the background's own mean instead of
        being left around zero. That keeps absolute intensities comparable
        with an unsubtracted frame, so the quality gate's mean-intensity and
        contrast thresholds keep their meaning whether or not background
        subtraction is enabled, and the output can stay in the input's integer
        dtype as the dtype policy promises.
        """
        background = self._background.update(image)
        if background is None:
            # Too few frames for a trustworthy estimate. Passing the frame
            # through unchanged is the honest answer; inventing a background
            # from one or two frames would remove the objects themselves.
            return image
        offset = float(np.mean(background, dtype=np.float64))
        residual = image.astype(np.float32) - background.astype(np.float32) + offset
        if image.dtype.kind == "f":
            return np.clip(residual, 0.0, 1.0)
        info = np.iinfo(image.dtype)
        np.clip(residual, float(info.min), float(info.max), out=residual)
        return residual.astype(image.dtype)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured intensity normalisation.

        See the module docstring for the dtype contract: ``none`` keeps the
        integer dtype, everything else returns ``float32`` in ``[0, 1]``.
        """
        mode = self.cfg.normalize
        if mode == "none":
            return image

        if mode == "minmax":
            unit = to_unit_float(image)
            lo = float(unit.min())
            hi = float(unit.max())
            span = hi - lo
            if span <= 1e-12:
                # A flat frame carries no information to stretch. Returning a
                # constant zero image is deterministic and lets the quality
                # gate reject it on contrast rather than on a NaN.
                return np.zeros_like(unit, dtype=np.float32)
            return ((unit - lo) * np.float32(1.0 / span)).astype(np.float32, copy=False)

        if mode == "zscore":
            unit = to_unit_float(image)
            mean = float(unit.mean())
            std = float(unit.std())
            if std <= 1e-12:
                return np.full(unit.shape, 0.5, dtype=np.float32)
            z = (unit - np.float32(mean)) * np.float32(1.0 / std)
            # Map z onto [0, 1] so the documented output range holds for every
            # normalising mode. +/-3 sigma is the display window; beyond it the
            # values are clipped rather than allowed to escape the contract.
            scaled = np.float32(0.5) + z * np.float32(
                1.0 / (2.0 * _ZSCORE_DISPLAY_SIGMA)
            )
            return np.clip(scaled, 0.0, 1.0, out=scaled)

        if mode == "clahe":
            # OpenCV 5.0's CLAHE asserts CV_8UC1 or CV_16UC1, so float input
            # has to be taken to 8-bit and back. Verified against cv2 5.0.0.
            source = image if image.dtype in (np.uint8, np.uint16) else to_uint8(image)
            clahe = self._clahe
            if clahe is None:  # pragma: no cover - built in __init__ for this mode
                raise ConfigurationError("CLAHE requested but not initialised")
            equalised = clahe.apply(source)
            return to_unit_float(equalised)

        raise ConfigurationError(f"unknown preprocess.normalize mode: {mode!r}")
