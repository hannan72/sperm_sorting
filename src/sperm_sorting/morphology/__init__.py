"""Four-aspect sperm morphology: model, calibration, metrics and inference.

Read :mod:`sperm_sorting.morphology.polarity` first. It states the one thing
that must never be got wrong here:

    **Every logit the network emits is a logit for ``P(abnormal)``.**

MHSMA labels are ``0 = normal`` / ``1 = abnormal``, so the training target is
the dataset label verbatim, calibration and metrics live entirely in
abnormal-positive space, and the single flip to the schema's ``p_normal``
happens in :class:`~.inference.MorphologyEngine` by calling
:func:`~.polarity.flip_polarity`. Checkpoints and calibration bundles both
record :data:`~.polarity.POLARITY_CONVENTION` and refuse to load when it
differs.

Module map
----------
``polarity``    the convention, the one flip, and its self-check.
``backbones``   grayscale-native feature extractors, ``(module, feature_dim)``.
``model``       :class:`~.model.MultiTaskMorphologyNet`, its loss, checkpoints.
``calibration`` temperature scaling, threshold fitting, ECE/MCE. No torch.
``metrics``     per-aspect classification metrics. No torch.
``inference``   :class:`~.inference.MorphologyEngine` and the test double.
``factory``     configuration to engine.

Imports of the torch-dependent modules are **lazy** (PEP 562 ``__getattr__``),
because torch is an optional extra of this project: ``calibration`` and
``metrics`` are useful on their own -- for a report, a notebook, or an
ONNX-only deployment -- and must not drag a 200 MB dependency in with them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .calibration import (
    CalibrationBundle,
    ReliabilityCurve,
    TemperatureScaler,
    ThresholdFit,
    expected_calibration_error,
    fit_calibration_bundle,
    fit_threshold,
    fit_thresholds,
    maximum_calibration_error,
    reliability_curve,
)
from .metrics import (
    aspect_metrics,
    balanced_accuracy,
    confusion_counts,
    confusion_matrix,
    evaluate_aspects,
    format_metrics_table,
    macro_f1,
    matthews_corrcoef,
    pr_auc,
    precision,
    roc_auc,
    sensitivity,
    specificity,
)
from .polarity import (
    POLARITY_CONVENTION,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    describe_polarity,
    flip_polarity,
    p_normal_from_p_abnormal,
    p_normal_threshold_from_p_abnormal_threshold,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from .backbones import (
        available_backbones,
        build_backbone,
        build_efficientnet_b0,
        build_mobilenetv3_small,
        build_simplecnn,
    )
    from .factory import build_morphology_engine, build_morphology_model
    from .inference import (
        BaseMorphologyEngine,
        MorphologyEngine,
        RandomMorphologyEngine,
        preprocess_crops,
    )
    from .model import (
        AspectHead,
        MorphologyLoss,
        MultiTaskMorphologyNet,
        export_onnx,
        load_checkpoint,
        pos_weight_from_prevalence,
        save_checkpoint,
    )

#: Attribute name -> module it lives in, for the lazy loader below.
_LAZY: dict[str, str] = {
    "AspectHead": "model",
    "BaseMorphologyEngine": "inference",
    "MorphologyEngine": "inference",
    "MorphologyLoss": "model",
    "MultiTaskMorphologyNet": "model",
    "RandomMorphologyEngine": "inference",
    "available_backbones": "backbones",
    "build_backbone": "backbones",
    "build_efficientnet_b0": "backbones",
    "build_mobilenetv3_small": "backbones",
    "build_morphology_engine": "factory",
    "build_morphology_model": "factory",
    "build_simplecnn": "backbones",
    "export_onnx": "model",
    "load_checkpoint": "model",
    "pos_weight_from_prevalence": "model",
    "preprocess_crops": "inference",
    "save_checkpoint": "model",
}


def __getattr__(name: str) -> Any:
    """Import torch-dependent symbols on first use (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache, so the indirection costs one lookup
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "POLARITY_CONVENTION",
    "POSITIVE_CLASS",
    "POSITIVE_CLASS_NAME",
    "AspectHead",
    "BaseMorphologyEngine",
    "CalibrationBundle",
    "MorphologyEngine",
    "MorphologyLoss",
    "MultiTaskMorphologyNet",
    "RandomMorphologyEngine",
    "ReliabilityCurve",
    "TemperatureScaler",
    "ThresholdFit",
    "aspect_metrics",
    "available_backbones",
    "balanced_accuracy",
    "build_backbone",
    "build_efficientnet_b0",
    "build_mobilenetv3_small",
    "build_morphology_engine",
    "build_morphology_model",
    "build_simplecnn",
    "confusion_counts",
    "confusion_matrix",
    "describe_polarity",
    "evaluate_aspects",
    "expected_calibration_error",
    "export_onnx",
    "fit_calibration_bundle",
    "fit_threshold",
    "fit_thresholds",
    "flip_polarity",
    "format_metrics_table",
    "load_checkpoint",
    "macro_f1",
    "matthews_corrcoef",
    "maximum_calibration_error",
    "p_normal_from_p_abnormal",
    "p_normal_threshold_from_p_abnormal_threshold",
    "pos_weight_from_prevalence",
    "pr_auc",
    "precision",
    "preprocess_crops",
    "reliability_curve",
    "roc_auc",
    "save_checkpoint",
    "sensitivity",
    "specificity",
]
