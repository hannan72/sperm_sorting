"""Dataset adapters, converters and validators for the sperm-analysis pipeline.

**No dataset is redistributed in this repository.** Several of the corpora used
here forbid it, all of them are large, and all of them are downloadable from
their own sources. Every adapter therefore imports and constructs without any
data present, and fails with a message naming both the path it searched and the
URL to download from. See ``datasets/README.md``.

Layout
------
``datasets.adapters``
    One reader per corpus, all producing
    :class:`sperm_sorting.schemas.detection.Detection` objects in absolute
    pixels, plus :class:`~datasets.adapters.base.DatasetInfo` metadata carrying
    the licence, the capture conditions and the expected domain shift.
``datasets.converters``
    YOLO / VOC / COCO / MOTChallenge <-> the internal format, and crop
    extraction that reuses the pipeline's own
    :class:`~sperm_sorting.cropping.extractor.CropExtractor`.
``datasets.validators``
    Integrity checks, split-leakage detection and a fail-closed licence
    registry.

Start here::

    from datasets import get_adapter, list_adapters

    adapter = get_adapter("mhsma")("~/data/mhsma-dataset")
    report = adapter.validate()          # raises if label polarity is inverted
    print(report.format_text())

A note on the package name
--------------------------
This package is called ``datasets`` and lives at the repository root, so from
the repository root it **shadows** Hugging Face's ``datasets`` library. Nothing
here needs that library; if you do, import it inside a virtual environment whose
working directory is not this repository, or rename this package. The name is
kept because the layout (``datasets/adapters``, ``datasets/converters``,
``datasets/validators``) is the one the project's documentation refers to.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .adapters.base import CaptureConditions, DatasetAdapter, DatasetInfo
    from .validators.integrity import CheckStatus, ValidationReport
    from .validators.licenses import CommercialUse, check_commercial_use

#: Registry name -> ``module:ClassName``. The single source of truth for
#: ``get_adapter``; the licence registry in
#: :mod:`datasets.validators.licenses` uses the same keys, so a dataset that
#: gains an adapter without gaining a licence record fails loudly the first time
#: anything asks about its terms.
ADAPTER_REGISTRY: dict[str, str] = {
    "mhsma": "datasets.adapters.mhsma:MhsmaAdapter",
    "visem_tracking": "datasets.adapters.visem_tracking:VisemTrackingAdapter",
    "visem": "datasets.adapters.visem:VisemAdapter",
    "visem_graphs": "datasets.adapters.visem_graphs:VisemGraphsAdapter",
    "detection_sperm": "datasets.adapters.detection_sperm:DetectionSpermAdapter",
    "device": "datasets.adapters.device:DeviceDatasetAdapter",
}

#: Convenience aliases, so a plausible spelling does not become a lookup failure.
ADAPTER_ALIASES: dict[str, str] = {
    "visem-tracking": "visem_tracking",
    "visem_tracking_graphs": "visem_graphs",
    "visem-tracking-graphs": "visem_graphs",
    "graphs": "visem_graphs",
    "miamia": "detection_sperm",
    "miamia_svds": "detection_sperm",
    "tod_cnn": "detection_sperm",
    "todcnn": "detection_sperm",
}

_LAZY: dict[str, str] = {
    "CaptureConditions": "datasets.adapters.base",
    "DatasetAdapter": "datasets.adapters.base",
    "DatasetInfo": "datasets.adapters.base",
    "CheckStatus": "datasets.validators.integrity",
    "ValidationReport": "datasets.validators.integrity",
    "CommercialUse": "datasets.validators.licenses",
    "check_commercial_use": "datasets.validators.licenses",
}

__all__ = [
    "ADAPTER_ALIASES",
    "ADAPTER_REGISTRY",
    "CaptureConditions",
    "CheckStatus",
    "CommercialUse",
    "DatasetAdapter",
    "DatasetInfo",
    "ValidationReport",
    "check_commercial_use",
    "describe_adapters",
    "get_adapter",
    "list_adapters",
]


def _canonical(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    return ADAPTER_ALIASES.get(key, key)


def get_adapter(name: str) -> type[DatasetAdapter]:
    """Look up an adapter class by registry name.

    Returns the **class**, not an instance, so the caller supplies the root and
    any adapter-specific options::

        get_adapter("visem_tracking")("~/data/visem-tracking")

    The import happens here rather than at package import time: ``MhsmaAdapter``
    needs torch and ``VisemGraphsAdapter`` needs networkx, and neither should be
    required to ask what licence a dataset carries.

    Raises
    ------
    KeyError
        On an unknown name, listing the registry.
    """
    key = _canonical(name)
    try:
        target = ADAPTER_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown dataset adapter {name!r}. Registered: "
            f"{sorted(ADAPTER_REGISTRY)} (aliases: {sorted(ADAPTER_ALIASES)})"
        ) from None
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def list_adapters() -> list[str]:
    """Registered adapter names, sorted."""
    return sorted(ADAPTER_REGISTRY)


def describe_adapters() -> list[dict[str, Any]]:
    """Metadata for every adapter, without touching the filesystem.

    Used to generate documentation and to run a licence audit in CI. Importing
    every adapter module is unavoidable here (the metadata lives on the classes),
    so this is the one entry point that does pay the torch import.
    """
    return [get_adapter(name).describe() for name in list_adapters()]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
