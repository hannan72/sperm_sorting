"""Inference-backend abstraction.

The target board is not fixed, so no module may assume CUDA, or ONNX Runtime,
or TensorRT. This resolves a :class:`BackendConfig` into a concrete runtime and
fails with an actionable message when the requested one is unavailable --
rather than silently falling back to CPU, which would turn a deployment
mistake into a mysterious latency regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import BackendConfig
from ..errors import BackendUnavailableError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedBackend:
    """What the process will actually run on."""

    kind: str
    device: str
    fp16: bool
    #: Populated for ONNX Runtime.
    providers: list[str] | None = None
    #: Human-readable description for the audit manifest.
    detail: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "device": self.device,
            "fp16": self.fp16,
            "providers": self.providers,
            "detail": self.detail,
        }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(cfg: BackendConfig) -> ResolvedBackend:
    """Validate the requested backend and report what it resolved to."""
    if cfg.kind == "torch":
        if not torch_available():
            raise BackendUnavailableError(
                "PyTorch is not installed. Install it with "
                "'pip install sperm-sorting-ai[torch]', or set "
                "backend.kind=onnxruntime."
            )
        import torch

        device = cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendUnavailableError(
                f"backend.device={device!r} was requested but PyTorch reports "
                "no CUDA device. Set backend.device='cpu' explicitly if that "
                "is what you intend -- silently falling back would hide a "
                "deployment error behind a latency regression."
            )
        if cfg.fp16 and device == "cpu":
            logger.warning(
                "fp16 was requested on CPU, where it is usually slower than "
                "fp32; ignoring it"
            )
        if cfg.num_threads:
            torch.set_num_threads(int(cfg.num_threads))
        detail = f"torch {torch.__version__}"
        if device.startswith("cuda"):
            detail += f", {torch.cuda.get_device_name(0)}"
        return ResolvedBackend(
            kind="torch",
            device=device,
            fp16=cfg.fp16 and device != "cpu",
            detail=detail,
        )

    if cfg.kind == "onnxruntime":
        if not onnxruntime_available():
            raise BackendUnavailableError(
                "onnxruntime is not installed. Install it with "
                "'pip install sperm-sorting-ai[onnx]'."
            )
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        requested = list(cfg.onnx_providers)
        usable = [p for p in requested if p in available]
        if not usable:
            raise BackendUnavailableError(
                f"none of the requested ONNX providers {requested} are "
                f"available; this build offers {sorted(available)}"
            )
        missing = [p for p in requested if p not in available]
        if missing:
            logger.warning(
                "ONNX providers %s are unavailable; running on %s", missing, usable
            )
        return ResolvedBackend(
            kind="onnxruntime",
            device=cfg.device,
            fp16=cfg.fp16,
            providers=usable,
            detail=f"onnxruntime {ort.__version__}",
        )

    if cfg.kind == "tensorrt":
        if not tensorrt_available():
            raise BackendUnavailableError(
                "TensorRT is not installed. It ships with the NVIDIA "
                "container images rather than from PyPI on most platforms."
            )
        import tensorrt as trt

        return ResolvedBackend(
            kind="tensorrt",
            device=cfg.device,
            fp16=cfg.fp16,
            detail=f"tensorrt {trt.__version__}",
        )

    raise BackendUnavailableError(f"unknown backend kind: {cfg.kind!r}")


def describe_environment() -> dict[str, Any]:
    """What is actually installed. Written into the audit manifest."""
    info: dict[str, Any] = {
        "torch": None,
        "cuda_available": False,
        "cuda_devices": [],
        "onnxruntime": None,
        "onnx_providers": [],
        "tensorrt": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnx_providers"] = list(ort.get_available_providers())
    except ImportError:
        pass
    try:
        import tensorrt as trt

        info["tensorrt"] = trt.__version__
    except ImportError:
        pass
    return info
