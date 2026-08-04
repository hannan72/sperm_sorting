"""Format converters. Everything routes through the internal ``Detection``.

:mod:`~datasets.converters.to_detection_format`
    YOLO / Pascal VOC / COCO <-> :class:`sperm_sorting.schemas.detection.Detection`.
    Box round-trips are lossless to double precision.
:mod:`~datasets.converters.to_mot_format`
    Tracks <-> MOTChallenge text, so HOTA/IDF1 can be computed by tooling this
    project did not write and cannot accidentally bias.
:mod:`~datasets.converters.to_crops`
    Detection dataset -> per-track morphology crops, cut by the *same*
    :class:`sperm_sorting.cropping.extractor.CropExtractor` the live pipeline
    uses. Never fabricates a morphology label.

None of these imports anything from :mod:`datasets.adapters`; the dependency
runs the other way (``detection_sperm`` uses ``to_detection_format``), which
keeps the import graph acyclic.
"""

from __future__ import annotations

from .to_crops import (
    CropDatasetBuilder,
    CropDatasetSummary,
    ExtractedCrop,
    load_grayscale,
)
from .to_detection_format import (
    coco_to_detections,
    detections_to_coco,
    detections_to_voc,
    detections_to_yolo,
    read_yolo_file,
    voc_to_detections,
    write_yolo_file,
    yolo_to_detections,
)
from .to_mot_format import (
    MOT_COLUMNS,
    MOT_FRAME_OFFSET,
    detections_to_mot,
    mot_to_detections,
    mot_to_tracks,
    read_mot_file,
    tracks_to_mot,
    write_mot_file,
)

__all__ = [
    "MOT_COLUMNS",
    "MOT_FRAME_OFFSET",
    "CropDatasetBuilder",
    "CropDatasetSummary",
    "ExtractedCrop",
    "coco_to_detections",
    "detections_to_coco",
    "detections_to_mot",
    "detections_to_voc",
    "detections_to_yolo",
    "load_grayscale",
    "mot_to_detections",
    "mot_to_tracks",
    "read_mot_file",
    "read_yolo_file",
    "tracks_to_mot",
    "voc_to_detections",
    "write_mot_file",
    "write_yolo_file",
    "yolo_to_detections",
]
