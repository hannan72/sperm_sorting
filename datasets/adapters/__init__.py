"""Dataset adapters: one per corpus, all speaking the same internal format.

+--------------------+---------------------------------+---------------------+
| adapter            | supervision                     | commercial use      |
+====================+=================================+=====================+
| ``mhsma``          | 4 binary morphology aspects     | no (CC BY-NC-SA)    |
| ``visem_tracking`` | boxes + track IDs, 3 classes    | yes (CC BY 4.0)     |
| ``visem``          | sample-level only               | no (CC BY-NC)       |
| ``visem_graphs``   | graphs derived from the above   | yes (CC BY 4.0)     |
| ``detection_sperm``| boxes, 2 classes (sperm/debris) | **unclear**         |
| ``device``         | boxes + tracks + morphology     | yes (ours)          |
+--------------------+---------------------------------+---------------------+

Imports are lazy (PEP 562). ``MhsmaAdapter`` pulls in torch and
``VisemGraphsAdapter`` pulls in networkx, and a licence audit or a
documentation build should not have to pay for either. ``from datasets.adapters
import MhsmaAdapter`` works exactly as if these were ordinary imports; the
module is only executed at that moment.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import CaptureConditions, DatasetAdapter, DatasetInfo

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .detection_sperm import AnnotationFormat, DetectionSpermAdapter
    from .device import (
        CaptureMetadata,
        DeviceAnnotationWriter,
        DeviceCapture,
        DeviceDatasetAdapter,
        FrameRecord,
        ObjectRecord,
    )
    from .mhsma import MhsmaAdapter, MhsmaAugmentation, MhsmaDataset
    from .visem import VisemAdapter
    from .visem_graphs import VisemGraphsAdapter
    from .visem_tracking import FrameAnnotation, VisemTrackingAdapter

#: Public name -> defining module, for the lazy loader below.
_LAZY: dict[str, str] = {
    "AnnotationFormat": "detection_sperm",
    "CaptureMetadata": "device",
    "DetectionSpermAdapter": "detection_sperm",
    "DeviceAnnotationWriter": "device",
    "DeviceCapture": "device",
    "DeviceDatasetAdapter": "device",
    "FrameAnnotation": "visem_tracking",
    "FrameRecord": "device",
    "MhsmaAdapter": "mhsma",
    "MhsmaAugmentation": "mhsma",
    "MhsmaDataset": "mhsma",
    "ObjectRecord": "device",
    "VisemAdapter": "visem",
    "VisemGraphsAdapter": "visem_graphs",
    "VisemTrackingAdapter": "visem_tracking",
}

__all__ = [
    "AnnotationFormat",
    "CaptureConditions",
    "CaptureMetadata",
    "DatasetAdapter",
    "DatasetInfo",
    "DetectionSpermAdapter",
    "DeviceAnnotationWriter",
    "DeviceCapture",
    "DeviceDatasetAdapter",
    "FrameAnnotation",
    "FrameRecord",
    "MhsmaAdapter",
    "MhsmaAugmentation",
    "MhsmaDataset",
    "ObjectRecord",
    "VisemAdapter",
    "VisemGraphsAdapter",
    "VisemTrackingAdapter",
]


def __getattr__(name: str) -> Any:
    """Import an adapter on first use. See the module docstring."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache, so the second access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
