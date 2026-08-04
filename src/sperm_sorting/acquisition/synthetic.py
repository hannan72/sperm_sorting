"""Synthetic frame source.

Wraps the procedural scene generator as a :class:`FrameSource`, so the whole
production pipeline can be exercised end to end with **known per-sperm ground
truth**. No public dataset offers that: VISEM-Tracking has boxes and track IDs
but no morphology, MHSMA has morphology but no video, and VISEM has only
sample-level percentages. Only the simulator produces both for the same
virtual cell, which is what makes it possible to measure the accuracy of the
combined motility-and-morphology rule rather than of its two halves
separately.

Ground truth travels in ``FramePacket.meta`` under ``gt_detections`` and
``gt_states``, where the oracle detector and the evaluation harness pick it up.
Nothing on the decision path reads it -- that separation is what keeps the
measurement honest.

Timestamps are exact by construction (``TimestampSource.SYNTHETIC``): frame
``n`` is at ``n / fps``, with no jitter. Real velocity estimates are therefore
*better* here than on hardware, and a model that only works on synthetic data
should be assumed not to work on a camera.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ..config import SyntheticSourceConfig
from ..schemas.enums import SourceKind, TimestampSource
from ..schemas.frame import FramePacket
from .base import FrameSource

logger = logging.getLogger(__name__)


class SyntheticFrameSource(FrameSource):
    """Procedurally generated microscopy frames with ground truth attached."""

    kind = SourceKind.SYNTHETIC

    def __init__(self, cfg: SyntheticSourceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._generator: Any = None
        self._iterator: Any = None

    def open(self) -> None:
        # Imported here rather than at module scope so that the acquisition
        # package stays importable if the simulator is unavailable.
        from ..simulator.scene import SceneConfig, SceneGenerator

        # The simulator owns a richer configuration than the runtime needs to
        # know about (render parameters, agent lifetimes, per-agent defocus).
        # Bridging through ``from_source_config`` keeps those simulator-only
        # knobs out of ``AppConfig``, which describes the *device*, while still
        # letting the scene be driven from the same YAML.
        #
        # ``defocus_rate`` is the runtime's name for what the scene calls
        # ``out_of_focus_rate``; it is mapped rather than duplicated so the
        # device configuration does not grow simulator vocabulary.
        scene_cfg = SceneConfig.from_source_config(
            self.cfg,
            um_per_px=self.cfg.um_per_px,
            out_of_focus_rate=self.cfg.defocus_rate,
        )
        self._generator = SceneGenerator(scene_cfg)
        self._iterator = iter(self._generator.frames())
        self._open = True
        self._session_id += 1
        self._frame_id = 0
        logger.info(
            "synthetic source: %dx%d at %.1f FPS, density %.1f, flow %.0f px/s "
            "(%.0f um/s), seed %d",
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
            self.cfg.density,
            self.cfg.flow_vx_px_s,
            self.cfg.flow_vx_px_s * self.cfg.um_per_px,
            self.cfg.seed,
        )

    def close(self) -> None:
        self._iterator = None
        self._generator = None
        self._open = False

    def read(self) -> FramePacket | None:
        if not self._open or self._iterator is None:
            return None
        if self.cfg.n_frames and self._frame_id >= self.cfg.n_frames:
            return None

        try:
            image, ground_truth = next(self._iterator)
        except StopIteration:
            return None

        packet = FramePacket(
            frame_id=self._frame_id,
            image=np.ascontiguousarray(image),
            # Exact by construction: the simulator defines the timeline.
            capture_time_s=self._frame_id / self.cfg.fps,
            timestamp_source=TimestampSource.SYNTHETIC,
            source_kind=self.kind,
            received_time_s=time.monotonic(),
            session_id=self._session_id,
            meta=dict(ground_truth),
        )
        self._frame_id += 1
        return packet

    @property
    def nominal_fps(self) -> float | None:
        return self.cfg.fps

    @property
    def um_per_px(self) -> float:
        """Sampling the scene was rendered at.

        Exposed so that an evaluation run can compare micrometre-per-second
        estimates against the truth without having to guess the scale.
        """
        return self.cfg.um_per_px

    def describe(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "width": self.cfg.width,
            "height": self.cfg.height,
            "fps": self.cfg.fps,
            "n_frames": self.cfg.n_frames,
            "um_per_px": self.cfg.um_per_px,
            "density": self.cfg.density,
            "debris_density": self.cfg.debris_density,
            "flow_px_s": [self.cfg.flow_vx_px_s, self.cfg.flow_vy_px_s],
            "flow_um_s": [
                self.cfg.flow_vx_px_s * self.cfg.um_per_px,
                self.cfg.flow_vy_px_s * self.cfg.um_per_px,
            ],
            "progressive_rate": self.cfg.progressive_rate,
            "normal_morphology_rate": self.cfg.normal_morphology_rate,
            "seed": self.cfg.seed,
        }
