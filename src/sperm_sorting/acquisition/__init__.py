"""Frame acquisition: live camera, recorded video, and the simulator.

All three feed the identical downstream graph, which is what makes replay a
test of the production path rather than a parallel implementation of it.
"""

from __future__ import annotations

from .base import FrameSource
from .factory import available_sources, build_frame_source

__all__ = ["FrameSource", "available_sources", "build_frame_source"]
