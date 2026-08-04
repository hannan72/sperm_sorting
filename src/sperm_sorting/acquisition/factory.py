"""Frame-source construction."""

from __future__ import annotations

from ..config import AcquisitionConfig
from ..errors import ConfigurationError
from ..schemas.enums import SourceKind
from .base import FrameSource


def build_frame_source(cfg: AcquisitionConfig) -> FrameSource:
    """Construct the configured frame source.

    Backends are imported lazily so that a machine without pypylon can still
    import this module and run every replay and synthetic test.
    """
    if cfg.kind is SourceKind.SYNTHETIC:
        from .synthetic import SyntheticFrameSource

        return SyntheticFrameSource(cfg.synthetic)
    if cfg.kind is SourceKind.VIDEO:
        from .video import VideoFrameSource

        return VideoFrameSource(cfg.video)
    if cfg.kind is SourceKind.BASLER:
        from .basler import BaslerFrameSource

        return BaslerFrameSource(cfg.basler)
    raise ConfigurationError(f"unknown acquisition.kind: {cfg.kind!r}")


def available_sources() -> list[str]:
    return [str(k) for k in SourceKind]


__all__ = ["FrameSource", "available_sources", "build_frame_source"]
