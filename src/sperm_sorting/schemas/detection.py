"""Detection schemas.

Boxes are stored as ``(x1, y1, x2, y2)`` in pixel coordinates of the frame the
detector was run on, with ``x2``/``y2`` exclusive-style (i.e. width = x2 - x1).
Every converter in :mod:`datasets.converters` targets this one format so that
no module ever has to guess whether it is holding xywh or xyxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import SCHEMA_VERSION


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """Axis-aligned box in pixels, ``x2``/``y2`` exclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"degenerate box: ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def as_cxcywh(self) -> tuple[float, float, float, float]:
        return (self.cx, self.cy, self.width, self.height)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
        return cls(float(x1), float(y1), float(x2), float(y2))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        return cls(float(x), float(y), float(x) + float(w), float(y) + float(h))

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        return cls(
            float(cx) - 0.5 * float(w),
            float(cy) - 0.5 * float(h),
            float(cx) + 0.5 * float(w),
            float(cy) + 0.5 * float(h),
        )

    def clipped(self, width: float, height: float) -> BoundingBox:
        """Clip into ``[0, width] x [0, height]``."""
        return BoundingBox(
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    def expanded(self, pad_x: float, pad_y: float) -> BoundingBox:
        """Grow by an absolute pixel margin on each side."""
        return BoundingBox(
            self.x1 - pad_x, self.y1 - pad_y, self.x2 + pad_x, self.y2 + pad_y
        )

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with ``other``."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union


@dataclass(slots=True)
class Detection:
    """One detected object in one frame, before any tracking has occurred."""

    frame_id: int
    box: BoundingBox
    score: float
    #: Class index within the detector's label set. The MVP detector is
    #: single-class (sperm), but the interface keeps room for an explicit
    #: debris class, which is how false positives get measured rather than
    #: silently thresholded away.
    class_id: int = 0
    class_name: str = "sperm"
    #: Capture time of the owning frame, copied for convenience so that motion
    #: code never has to hold a frame reference.
    capture_time_s: float = 0.0
    #: Filled in by the tracker; ``None`` means "not yet associated".
    track_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "box_xyxy": list(self.box.as_xyxy()),
            "score": float(self.score),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "capture_time_s": float(self.capture_time_s),
            "track_id": self.track_id,
            "schema_version": self.schema_version,
        }


def detections_to_array(detections: list[Detection]) -> np.ndarray:
    """Stack detections into an ``(N, 5)`` ``[x1, y1, x2, y2, score]`` array.

    Returns a correctly-shaped empty array for an empty input so that callers
    never have to special-case the no-detection frame.
    """
    if not detections:
        return np.zeros((0, 5), dtype=np.float32)
    out = np.empty((len(detections), 5), dtype=np.float32)
    for i, det in enumerate(detections):
        out[i, 0] = det.box.x1
        out[i, 1] = det.box.y1
        out[i, 2] = det.box.x2
        out[i, 3] = det.box.y2
        out[i, 4] = det.score
    return out
