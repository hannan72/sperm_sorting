"""Frame preprocessing and the whole-frame image-quality gate.

Two stages, run in this order, immediately after acquisition:

1. :class:`FramePreprocessor` -- ROI crop, optional inversion, optional
   rolling-median background subtraction, intensity normalisation.
2. :class:`ImageQualityGate` -- measures focus, exposure and contrast and
   decides whether the frame is usable, degraded, or must be dropped.

The order matters: the gate's thresholds describe the image the detector will
actually see, so it must run on the preprocessed frame, not the raw one.

The intensity converters are exported because the dtype policy (``uint8``
through the ``none`` path, ``float32`` in ``[0, 1]`` through the normalising
paths) is a package-wide contract, and every consumer that has to handle both
should use the same converter rather than reinventing the scaling.
"""

from __future__ import annotations

from .preprocessor import (
    FramePreprocessor,
    ensure_mono2d,
    to_uint8,
    to_unit_float,
    translate_boxes_to_roi,
)
from .quality_gate import ImageQualityGate

__all__ = [
    "FramePreprocessor",
    "ImageQualityGate",
    "ensure_mono2d",
    "to_uint8",
    "to_unit_float",
    "translate_boxes_to_roi",
]
