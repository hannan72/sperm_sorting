"""Object builders shared by the tests.

Kept out of ``conftest.py`` because these are plain functions, not fixtures,
and test modules import them directly. ``tests/`` deliberately has no
``__init__.py``, so pytest puts this directory on ``sys.path`` and a plain
``from builders import make_track`` works from any test module.
"""

from __future__ import annotations

import numpy as np

from sperm_sorting.schemas.detection import BoundingBox, Detection
from sperm_sorting.schemas.enums import SourceKind, TimestampSource, TrackState
from sperm_sorting.schemas.frame import FramePacket
from sperm_sorting.schemas.track import TrackPoint, TrackRecord

def make_frame(
    frame_id: int = 0,
    capture_time_s: float = 0.0,
    width: int = 320,
    height: int = 240,
    value: int = 200,
) -> FramePacket:
    """A uniform frame with a little texture, so focus metrics are non-zero."""
    image = np.full((height, width), value, dtype=np.uint8)
    image[::8, :] = max(0, value - 60)
    image[:, ::8] = max(0, value - 60)
    return FramePacket(
        frame_id=frame_id,
        image=image,
        capture_time_s=capture_time_s,
        timestamp_source=TimestampSource.SYNTHETIC,
        source_kind=SourceKind.SYNTHETIC,
    )


def make_detection(
    frame_id: int,
    x: float,
    y: float,
    w: float = 20.0,
    h: float = 14.0,
    score: float = 0.9,
    capture_time_s: float = 0.0,
) -> Detection:
    return Detection(
        frame_id=frame_id,
        box=BoundingBox.from_cxcywh(x, y, w, h),
        score=score,
        capture_time_s=capture_time_s,
    )


def make_track(
    track_id: int,
    *,
    n_points: int = 20,
    x0: float = 10.0,
    y0: float = 50.0,
    dx: float = 5.0,
    dy: float = 0.0,
    dt: float = 1 / 160,
    score: float = 0.9,
    observed: bool = True,
    quality_pass: bool = True,
) -> TrackRecord:
    """A straight-line track with exact, regular timestamps."""
    track = TrackRecord(track_id=track_id, state=TrackState.CONFIRMED)
    for i in range(n_points):
        track.add_point(
            TrackPoint(
                frame_id=i,
                capture_time_s=i * dt,
                box=BoundingBox.from_cxcywh(x0 + i * dx, y0 + i * dy, 20.0, 14.0),
                score=score,
                observed=observed,
            )
        )
    track.track_quality_pass = quality_pass
    return track
