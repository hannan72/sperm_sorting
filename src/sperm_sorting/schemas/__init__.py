"""Typed schemas shared by every stage of the pipeline.

Runtime pipeline objects are ``slots`` dataclasses (constructed at frame rate,
carrying numpy buffers); configuration objects are Pydantic models
(constructed once, from untrusted YAML, and worth validating field by field).
That split is deliberate -- see ``docs/architecture.md``.
"""

from __future__ import annotations

from .command import FieldCommand
from .detection import BoundingBox, Detection, detections_to_array
from .enums import (
    CommandOrigin,
    CommandOutcome,
    FieldCommandKind,
    FlowCorrectionMode,
    IneligibilityReason,
    MorphologyStatus,
    MotilityClass,
    QualityVerdict,
    ShotCloseReason,
    ShotStatus,
    SourceKind,
    TimestampSource,
    TrackState,
)
from .frame import FramePacket, FrameQuality
from .morphology import AspectResult, MorphologyResult
from .shot import ShotRecord, exceeds_threshold
from .track import CropRecord, MotionFeatures, TrackPoint, TrackRecord

__all__ = [
    "AspectResult",
    "BoundingBox",
    "CommandOrigin",
    "CommandOutcome",
    "CropRecord",
    "Detection",
    "FieldCommand",
    "FieldCommandKind",
    "FlowCorrectionMode",
    "FramePacket",
    "FrameQuality",
    "IneligibilityReason",
    "MorphologyResult",
    "MorphologyStatus",
    "MotilityClass",
    "MotionFeatures",
    "QualityVerdict",
    "ShotCloseReason",
    "ShotRecord",
    "ShotStatus",
    "SourceKind",
    "TimestampSource",
    "TrackPoint",
    "TrackRecord",
    "TrackState",
    "detections_to_array",
    "exceeds_threshold",
]
