"""Video-file frame source.

Replays a recording through the production pipeline. Two details make this a
real test rather than a demo:

* Timestamps come from the container's presentation timestamps when they are
  usable, and are marked :attr:`TimestampSource.CONTAINER_PTS` so downstream
  code knows they are nominal. When the container has no usable rate the
  configured fallback is used and that fact is logged once, loudly -- a
  silently assumed frame interval is the most direct route to a wrong velocity.
* ``realtime=False`` replays as fast as the pipeline can consume, which is what
  makes the determinism test cheap; ``realtime=True`` paces to the recorded
  rate, which is what makes a latency measurement meaningful.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import VideoSourceConfig
from ..errors import CameraError
from ..schemas.enums import SourceKind, TimestampSource
from ..schemas.frame import FramePacket
from .base import FrameSource

logger = logging.getLogger(__name__)


class VideoFrameSource(FrameSource):
    """Reads frames from a file via OpenCV."""

    kind = SourceKind.VIDEO

    def __init__(self, cfg: VideoSourceConfig) -> None:
        super().__init__()
        if cfg.path is None:
            raise CameraError("acquisition.video.path is required for a video source")
        self.cfg = cfg
        self.path = Path(cfg.path)
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = cfg.fallback_fps
        self._fps_is_nominal = True
        self._n_frames: int | None = None
        self._start_wall: float = 0.0
        self._first_pts_s: float | None = None
        self._loops = 0

    def open(self) -> None:
        if not self.path.exists():
            raise CameraError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise CameraError(f"OpenCV could not open {self.path}")
        self._cap = cap

        reported = cap.get(cv2.CAP_PROP_FPS)
        if reported and reported > 0 and np.isfinite(reported):
            self._fps = float(reported)
            self._fps_is_nominal = False
        else:
            self._fps = self.cfg.fallback_fps
            self._fps_is_nominal = True
            logger.warning(
                "%s reports no usable frame rate; falling back to %.2f FPS from "
                "configuration. Every velocity derived from this replay depends "
                "on that number being correct.",
                self.path.name,
                self._fps,
            )

        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._n_frames = int(count) if count and count > 0 else None

        if self.cfg.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.cfg.start_frame))

        self._open = True
        self._start_wall = time.monotonic()
        logger.info(
            "video source: %s (%.2f FPS%s, %s frames)",
            self.path.name,
            self._fps,
            " nominal" if self._fps_is_nominal else "",
            self._n_frames if self._n_frames is not None else "unknown",
        )

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.exception("failed to release the video capture")
            self._cap = None
        self._open = False

    def _pts_seconds(self) -> float | None:
        """Presentation timestamp of the frame just read, in seconds."""
        if self._cap is None:
            return None
        msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec is None or msec <= 0 or not np.isfinite(msec):
            return None
        return float(msec) / 1000.0

    def read(self) -> FramePacket | None:
        if self._cap is None or not self._open:
            raise CameraError("video source is not open")

        ok, frame = self._cap.read()
        if not ok:
            if self.cfg.loop:
                self._loops += 1
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.cfg.start_frame))
                self._first_pts_s = None
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                return None

        # The sensor is monochrome, so anything with three channels came from
        # a codec that expanded it. Collapse rather than picking one channel:
        # a lossy codec distributes luma across all three.
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = np.ascontiguousarray(frame)

        pts = self._pts_seconds()
        if pts is not None:
            if self._first_pts_s is None:
                self._first_pts_s = pts
            capture_time = pts
            ts_source = TimestampSource.CONTAINER_PTS
        else:
            capture_time = self._frame_id / self._fps
            ts_source = TimestampSource.CONTAINER_PTS

        if self.cfg.realtime:
            target = self._start_wall + capture_time
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        packet = FramePacket(
            frame_id=self._frame_id,
            image=image,
            capture_time_s=capture_time,
            timestamp_source=ts_source,
            source_kind=self.kind,
            received_time_s=time.monotonic(),
            session_id=self._session_id,
            meta={
                "path": str(self.path),
                "fps": self._fps,
                "fps_is_nominal": self._fps_is_nominal,
                "loops": self._loops,
            },
        )
        self._frame_id += 1
        return packet

    @property
    def nominal_fps(self) -> float | None:
        return self._fps

    def describe(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "path": str(self.path),
            "fps": self._fps,
            "fps_is_nominal": self._fps_is_nominal,
            "n_frames": self._n_frames,
            "realtime": self.cfg.realtime,
            "loop": self.cfg.loop,
            "start_frame": self.cfg.start_frame,
        }
