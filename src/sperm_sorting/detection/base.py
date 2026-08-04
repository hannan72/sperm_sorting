"""Detector interface.

Every detector -- learned, ONNX-exported, or the synthetic oracle -- presents
the same two methods, so the runtime never knows which one it is driving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..schemas.detection import Detection
from ..schemas.frame import FramePacket


class Detector(ABC):
    """Detects sperm in a single monochrome frame.

    Implementations must:

    * accept a 2-D ``uint8`` / ``uint16`` array or a :class:`FramePacket`;
    * return boxes in **source-frame pixel coordinates**, undoing any internal
      resize or tiling themselves, because nothing downstream knows the
      detector's internal geometry;
    * be safe to call repeatedly from one thread;
    * never raise on an empty frame -- return an empty list.
    """

    #: Human-readable identifier written into the audit log.
    name: str = "detector"
    #: Names indexed by ``class_id``.
    class_names: tuple[str, ...] = ("sperm",)

    @abstractmethod
    def detect(self, frame: FramePacket) -> list[Detection]:
        """Detect objects in one frame."""

    def detect_array(
        self, image: np.ndarray, frame_id: int = 0, capture_time_s: float = 0.0
    ) -> list[Detection]:
        """Convenience wrapper for callers that hold a bare array."""
        from ..schemas.enums import SourceKind, TimestampSource

        packet = FramePacket(
            frame_id=frame_id,
            image=image,
            capture_time_s=capture_time_s,
            timestamp_source=TimestampSource.SYNTHETIC,
            source_kind=SourceKind.SYNTHETIC,
        )
        return self.detect(packet)

    def warmup(self, height: int, width: int, iterations: int = 3) -> None:
        """Run a few dummy inferences so the first real frame is not slow.

        Lazy CUDA context creation and cuDNN autotuning otherwise land on the
        first live frame, which is exactly when latency matters most.
        """
        dummy = np.zeros((height, width), dtype=np.uint8)
        for _ in range(iterations):
            self.detect_array(dummy)

    def close(self) -> None:
        """Release backend resources. Safe to call more than once."""

    def describe(self) -> dict[str, Any]:
        """Metadata stamped into the audit log header."""
        return {"name": self.name, "class_names": list(self.class_names)}

    def __enter__(self) -> Detector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
