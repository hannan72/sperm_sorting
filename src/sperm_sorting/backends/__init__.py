"""Inference-backend abstraction for CPU, CUDA, ONNX Runtime and TensorRT."""

from __future__ import annotations

from .runtime_backend import (
    ResolvedBackend,
    describe_environment,
    onnxruntime_available,
    resolve_backend,
    tensorrt_available,
    torch_available,
)

__all__ = [
    "ResolvedBackend",
    "describe_environment",
    "onnxruntime_available",
    "resolve_backend",
    "tensorrt_available",
    "torch_available",
]
