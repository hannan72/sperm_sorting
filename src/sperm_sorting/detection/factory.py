"""Detector construction from configuration.

One entry point, :func:`build_detector`, so that every caller -- the runtime,
the training evaluation loop, the benchmark scripts, the tests -- builds a
detector the same way. A second construction path is how a benchmark ends up
measuring a differently-configured model from the one that ships.

Backends are imported lazily inside the branch that needs them. ``torch`` and
``onnxruntime`` are optional extras; importing them at module scope would mean
that a deployment carrying only ``onnxruntime`` could not even import this
module, and that a config selecting the oracle detector would still require a
600 MB torch install.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import DetectionConfig
from ..errors import BackendUnavailableError, ConfigurationError
from .base import Detector

__all__ = ["DETECTOR_ARCHITECTURES", "available_detectors", "build_detector"]

logger = logging.getLogger(__name__)

#: Every value ``DetectionConfig.architecture`` may take. Kept in sync with the
#: ``Literal`` on that field by :func:`test_factory_covers_config_literal`.
DETECTOR_ARCHITECTURES: tuple[str, ...] = ("todcnn", "p2net", "onnx", "oracle")

#: Architectures that need torch, and therefore share the same import guard.
_TORCH_ARCHITECTURES = frozenset({"todcnn", "p2net"})

#: Default seed for the weight initialisation of an *untrained* network, so
#: that two builds of the same config produce byte-identical outputs. Matches
#: ``RunConfig.seed``; the runtime passes its own.
_DEFAULT_SEED = 1234


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "the 'todcnn' and 'p2net' detectors require PyTorch, which is not "
            "installed. Install it with `pip install 'sperm-sorting-ai[torch]'`. "
            "To run without torch, export the model to ONNX and set "
            "detection.architecture='onnx', or use 'oracle' for a synthetic run."
        ) from exc
    return torch


def available_detectors() -> dict[str, bool]:
    """Which architectures can actually be built in *this* environment.

    Returned as a mapping rather than a filtered list so a caller can tell
    "this name is unknown" from "this name is known but its backend is
    missing" -- the two need very different error messages.
    """
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False
    try:
        import onnxruntime  # noqa: F401

        has_onnx = True
    except ImportError:
        has_onnx = False

    return {
        "todcnn": has_torch,
        "p2net": has_torch,
        "onnx": has_onnx,
        # The oracle needs nothing beyond numpy, which is why it is always the
        # fallback for a smoke test on a bare environment.
        "oracle": True,
    }


def _warn_on_backend_mismatch(cfg: DetectionConfig) -> None:
    """Flag a backend/architecture combination that cannot mean what it says.

    A warning rather than an error: ``BackendConfig`` is shared with the
    morphology model, so a config can legitimately name a backend that this
    detector does not use.
    """
    kind = cfg.backend.kind
    if cfg.architecture == "onnx" and kind != "onnxruntime":
        logger.warning(
            "detection.architecture='onnx' but detection.backend.kind='%s'; the "
            "ONNX detector always runs through onnxruntime and ignores that "
            "setting.",
            kind,
        )
    elif cfg.architecture in _TORCH_ARCHITECTURES and kind != "torch":
        logger.warning(
            "detection.architecture='%s' runs through torch, but "
            "detection.backend.kind='%s'. Export the model and set "
            "detection.architecture='onnx' to actually use that backend.",
            cfg.architecture,
            kind,
        )


def build_detector(cfg: DetectionConfig, seed: int = _DEFAULT_SEED) -> Detector:
    """Construct the detector described by ``cfg``.

    Weight loading and backend selection happen here rather than in each
    detector's constructor so that "which checkpoint is loaded" and "which
    device it runs on" are answered in one readable place.

    Parameters
    ----------
    cfg
        The ``detection`` section of the application config.
    seed
        Seeds the *initialisation* of an untrained torch network, so that a
        config without ``weights`` still produces a reproducible detector. It
        is drawn from a forked RNG state, so building a detector never
        perturbs the process-wide torch seed that training relies on.

    Raises
    ------
    ConfigurationError
        The architecture name is not recognised.
    BackendUnavailableError
        The architecture is recognised but its backend is not installed.
    InferenceError
        Weights were configured but could not be loaded.
    """
    architecture = str(cfg.architecture)
    if architecture not in DETECTOR_ARCHITECTURES:
        raise ConfigurationError(
            f"unknown detection.architecture '{architecture}'; expected one of "
            f"{list(DETECTOR_ARCHITECTURES)}"
        )
    _warn_on_backend_mismatch(cfg)

    if architecture == "oracle":
        from .oracle import OracleDetector

        if cfg.weights is not None:
            logger.warning(
                "detection.weights=%s is ignored by the oracle detector, which "
                "reads ground truth from the frame rather than a model.",
                cfg.weights,
            )
        return OracleDetector(cfg, seed=seed)

    if architecture == "onnx":
        from .onnx_detector import OnnxDetector

        return OnnxDetector(cfg)

    torch = _require_torch()
    # fork_rng keeps weight initialisation reproducible without stomping on the
    # global RNG: a training script that builds a detector mid-run must not
    # have its own data shuffling silently reseeded.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if architecture == "todcnn":
            from .todcnn import TodCnnDetector

            detector: Detector = TodCnnDetector(cfg)
        else:
            from .p2net import P2NetDetector

            detector = P2NetDetector(cfg)

    if cfg.weights is not None:
        detector.load_weights(cfg.weights)  # type: ignore[attr-defined]
        logger.info("%s: loaded weights from %s", architecture, cfg.weights)
    else:
        logger.warning(
            "%s: no detection.weights configured -- the network is randomly "
            "initialised and its detections are meaningless. This is expected "
            "for a shape/latency test and for nothing else.",
            architecture,
        )
    return detector
