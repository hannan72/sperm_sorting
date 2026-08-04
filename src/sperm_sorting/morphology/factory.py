"""Construction of morphology engines and models from configuration.

One function, one decision: which backend does :attr:`MorphologyConfig.backend`
ask for, and is it usable in this build? Everything else -- weight loading,
calibration discovery, warm-up -- belongs to
:class:`~.inference.MorphologyEngine` and is not duplicated here.

There is deliberately **no path from configuration to
:class:`~.inference.RandomMorphologyEngine`**. Its outputs are seeded noise, and
the only safe way to keep a test double out of a production run is to make it
unreachable from the config file rather than to guard it with a flag someone can
set. :attr:`sperm_sorting.config.BackendConfig.kind` is a ``Literal`` over
``torch``/``onnxruntime``/``tensorrt``, so no YAML can name the test engine, and
this factory never mentions it.

Torch is imported lazily throughout, because it is an *optional* extra of this
project: a deployment that runs morphology through ONNX Runtime must be able to
validate its configuration without PyTorch installed.
"""

from __future__ import annotations

from typing import Any

from ..config import MorphologyConfig
from ..errors import BackendUnavailableError, ConfigurationError
from .inference import BaseMorphologyEngine, MorphologyEngine

__all__ = [
    "available_backbones",
    "build_morphology_engine",
    "build_morphology_model",
]

#: Backends this module can construct.
_SUPPORTED_BACKENDS: tuple[str, ...] = ("torch", "onnxruntime")


def available_backbones() -> tuple[str, ...]:
    """Backbone names this build can actually construct.

    Empty when PyTorch/torchvision are absent -- an honest answer rather than a
    hard-coded list that would let configuration validation pass and then fail
    at model construction. An ONNX-only deployment legitimately returns ``()``:
    it consumes an exported graph and never builds a backbone.
    """
    try:
        from .backbones import available_backbones as _impl
    except ImportError:  # pragma: no cover - depends on the environment
        return ()
    return _impl()


def build_morphology_engine(
    cfg: MorphologyConfig, **kwargs: Any
) -> BaseMorphologyEngine:
    """Build the morphology engine described by ``cfg``.

    Parameters
    ----------
    cfg
        The ``morphology`` section of :class:`~sperm_sorting.config.AppConfig`.
    kwargs
        Forwarded to :class:`~.inference.MorphologyEngine` (``model``,
        ``calibration``, ``strict_deadline``, ``warmup_iterations``). Used by
        tests and by the training loop's validation pass; a production run
        passes none of them.

    Raises
    ------
    ConfigurationError
        If ``cfg.backbone`` cannot be constructed by this build. Checked at
        startup, which is the whole point of a factory: discovering it on the
        first track means discovering it mid-run.
    BackendUnavailableError
        If ``cfg.backend.kind`` names a backend that is not implemented, or one
        whose runtime is not installed.
    """
    kind = cfg.backend.kind
    if kind not in _SUPPORTED_BACKENDS:
        raise BackendUnavailableError(
            f"morphology.backend.kind='{kind}' is not implemented. Supported: "
            f"{', '.join(_SUPPORTED_BACKENDS)}. Morphology runs once per track "
            "rather than once per frame, so it has never justified a TensorRT "
            "engine; export the detector instead if throughput is the problem."
        )

    # The backbone only matters to the torch path: the ONNX path consumes a
    # graph whose architecture was fixed at export time.
    if kind == "torch":
        names = available_backbones()
        if not names:
            raise BackendUnavailableError(
                "morphology.backend.kind='torch' but PyTorch/torchvision are not "
                "installed; install the 'torch' extra or switch to "
                "morphology.backend.kind='onnxruntime'"
            )
        if cfg.backbone not in names:
            raise ConfigurationError(
                f"morphology.backbone='{cfg.backbone}' is not available in this "
                f"build; choose one of {', '.join(names)}"
            )

    return MorphologyEngine(cfg, **kwargs)


def build_morphology_model(cfg: MorphologyConfig, *, pretrained: bool = False) -> Any:
    """Build an untrained :class:`~.model.MultiTaskMorphologyNet` from ``cfg``.

    The entry point for the training script, so that the architecture a run
    trains is byte-for-byte the architecture the runtime will later build from
    the same config file.
    """
    from .model import MultiTaskMorphologyNet

    return MultiTaskMorphologyNet(
        backbone=cfg.backbone,
        in_channels=1,
        input_size=(int(cfg.input_size[0]), int(cfg.input_size[1])),
        pretrained=pretrained,
    )
