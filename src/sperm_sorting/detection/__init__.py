"""Detection subsystem.

Public surface: the :class:`Detector` interface, four concrete detectors, the
:func:`build_detector` factory, and the pre/post-processing helpers that keep
every backend's geometry identical.

Import policy
-------------
``torch`` and ``onnxruntime`` are optional extras, so the symbols that need
them are resolved lazily through :pep:`562` module ``__getattr__``. The
practical consequence: ``from sperm_sorting.detection import OracleDetector``
works on a bare numpy+opencv install, and only ``P2NetDetector`` (say) pulls
torch in. Without this, a deployment image carrying just ``onnxruntime`` could
not import this package at all.

Everything remains statically visible to type checkers via the ``TYPE_CHECKING``
block, so the laziness costs no editor completion or mypy coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Detector
from .factory import DETECTOR_ARCHITECTURES, available_detectors, build_detector
from .oracle import GT_META_KEY, OracleDetector
from .postprocess import (
    arrays_to_detections,
    batched_nms,
    clip_boxes,
    compute_tile_grid,
    decode_centernet_heatmap,
    filter_boxes_by_size,
    finalise_boxes,
    merge_tiled_detections,
    nms,
    scale_boxes,
    top_k_detections,
)
from .preprocess import (
    pad_to_divisor,
    prepare_input,
    resize_long_side,
    to_float_gray,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .heads import (
        CenterNetHead,
        build_centernet_targets,
        centernet_focal_loss,
        centernet_loss,
        masked_l1_loss,
    )
    from .onnx_detector import OnnxDetector
    from .p2net import P2Net, P2NetDetector
    from .todcnn import TodCnnDetector, TodCnnNet
    from .torch_base import TorchDetectorBase

#: Attribute name -> submodule that defines it, for the lazy loader.
_LAZY_EXPORTS: dict[str, str] = {
    "CenterNetHead": "heads",
    "build_centernet_targets": "heads",
    "centernet_focal_loss": "heads",
    "centernet_loss": "heads",
    "masked_l1_loss": "heads",
    "OnnxDetector": "onnx_detector",
    "P2Net": "p2net",
    "P2NetDetector": "p2net",
    "TodCnnDetector": "todcnn",
    "TodCnnNet": "todcnn",
    "TorchDetectorBase": "torch_base",
}

__all__ = [
    "DETECTOR_ARCHITECTURES",
    "GT_META_KEY",
    "CenterNetHead",
    "Detector",
    "OnnxDetector",
    "OracleDetector",
    "P2Net",
    "P2NetDetector",
    "TodCnnDetector",
    "TodCnnNet",
    "TorchDetectorBase",
    "arrays_to_detections",
    "available_detectors",
    "batched_nms",
    "build_centernet_targets",
    "build_detector",
    "centernet_focal_loss",
    "centernet_loss",
    "clip_boxes",
    "compute_tile_grid",
    "decode_centernet_heatmap",
    "filter_boxes_by_size",
    "finalise_boxes",
    "masked_l1_loss",
    "merge_tiled_detections",
    "nms",
    "pad_to_divisor",
    "prepare_input",
    "resize_long_side",
    "scale_boxes",
    "to_float_gray",
    "top_k_detections",
]


def __getattr__(name: str) -> Any:
    """Resolve the backend-dependent exports on first use."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    # Cache on the module so the import cost and the dict lookup are paid once.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
