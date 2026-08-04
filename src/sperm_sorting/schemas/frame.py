"""Frame-level schemas.

:class:`FramePacket` is the unit that flows from acquisition into the rest of
the pipeline. It is a ``slots`` dataclass rather than a Pydantic model because
one is constructed per frame at up to ~164 Hz and it carries a large numpy
buffer; per-field validation on that path would be wasted work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import SCHEMA_VERSION
from .enums import QualityVerdict, SourceKind, TimestampSource


@dataclass(slots=True)
class FrameQuality:
    """Per-frame image-quality measurements produced by the quality gate."""

    verdict: QualityVerdict
    #: Variance of the Laplacian; higher is sharper. Scale is sensor-dependent.
    focus_score: float
    #: Mean intensity, 0-1 normalised.
    mean_intensity: float
    #: Standard deviation of intensity, 0-1 normalised. Proxy for contrast.
    contrast: float
    #: Fraction of pixels at the top of the dynamic range.
    saturated_fraction: float
    #: Fraction of pixels at the bottom of the dynamic range.
    underexposed_fraction: float
    #: Free-text explanation when ``verdict`` is not ``PASS``.
    reason: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "focus_score": float(self.focus_score),
            "mean_intensity": float(self.mean_intensity),
            "contrast": float(self.contrast),
            "saturated_fraction": float(self.saturated_fraction),
            "underexposed_fraction": float(self.underexposed_fraction),
            "reason": self.reason,
        }


@dataclass(slots=True)
class FramePacket:
    """One acquired frame plus everything known about *when* it was captured.

    Attributes
    ----------
    frame_id
        Strictly increasing, gap-free counter assigned by the frame source.
        Gaps are never used to signal drops -- drops are reported explicitly
        via :attr:`dropped_before` so that a missing frame cannot be confused
        with a counter reset.
    image
        2-D ``uint8`` or ``uint16`` array, shape ``(H, W)``. Monochrome: the
        target camera has no colour filter array and the pipeline never
        assumes three channels.
    capture_time_s
        Capture instant in seconds on a *monotonic* timeline. This is the only
        timestamp motion analysis may use. Never a wall-clock value.
    timestamp_source
        Provenance of :attr:`capture_time_s`; see :class:`TimestampSource`.
        Motion features record this so that a downstream reader can tell
        hardware-timed measurements from software-timed approximations.
    """

    frame_id: int
    image: np.ndarray
    capture_time_s: float
    timestamp_source: TimestampSource
    source_kind: SourceKind

    #: Host monotonic time at which the packet entered the pipeline. Used for
    #: latency accounting only, never for velocity.
    received_time_s: float = 0.0
    #: Number of frames the source knows it lost immediately before this one.
    dropped_before: int = 0
    #: Sequence number of the acquisition session; resets on reconnect.
    session_id: int = 0
    #: Populated by the quality gate; ``None`` until it runs.
    quality: FrameQuality | None = None
    #: Region of interest applied during preprocessing, ``(x, y, w, h)`` in
    #: pixels of the *original* sensor frame. ``None`` means the full frame.
    roi: tuple[int, int, int, int] | None = None
    #: Arbitrary source-specific extras (exposure, gain, sensor temperature,
    #: simulator ground truth, ...). Never load-bearing for the decision path.
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def to_json_dict(self, *, include_image: bool = False) -> dict[str, Any]:
        """Audit-log representation.

        The image is excluded by default: the audit log is a decision record,
        not a video store.
        """
        out: dict[str, Any] = {
            "frame_id": self.frame_id,
            "capture_time_s": float(self.capture_time_s),
            "timestamp_source": str(self.timestamp_source),
            "source_kind": str(self.source_kind),
            "received_time_s": float(self.received_time_s),
            "dropped_before": self.dropped_before,
            "session_id": self.session_id,
            "shape": list(self.shape),
            "roi": list(self.roi) if self.roi is not None else None,
            "quality": self.quality.to_json_dict() if self.quality else None,
            "schema_version": self.schema_version,
        }
        if include_image:
            out["image"] = self.image.tolist()
        return out
