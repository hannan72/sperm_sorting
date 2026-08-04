"""Multi-object tracker interface.

The contract that matters for this product is *identity*: one physical sperm
gets exactly one ID for its whole visible lifetime, and that ID is never
reused. Everything downstream -- the shot denominator, the once-only counting
rule, the crop-to-motion binding -- rests on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas.detection import Detection
from ..schemas.frame import FramePacket
from ..schemas.track import TrackRecord


class Tracker(ABC):
    """Associates per-frame detections into persistent tracks.

    Implementations must guarantee:

    1. **Unique IDs.** Every ID returned by :meth:`update` identifies one
       track for the whole session. IDs are never recycled, even after a
       track is removed. ``tests/test_track_identity.py`` asserts this.
    2. **Idempotent records.** :meth:`update` returns the *same*
       :class:`TrackRecord` object for a given ID on every call, so callers
       may hold a reference and see it grow.
    3. **Explicit interpolation.** A point produced by the motion model rather
       than a measurement is appended with ``observed=False`` so that motion
       analysis can exclude it.
    """

    name: str = "tracker"

    @abstractmethod
    def update(
        self, detections: list[Detection], frame: FramePacket
    ) -> list[TrackRecord]:
        """Advance the tracker by one frame.

        Returns the tracks that are *currently active* (confirmed and matched
        or recently lost), not every track ever seen. Use :meth:`all_tracks`
        for the full history and :meth:`finished_tracks` to drain tracks that
        have just been removed and are therefore ready for final analysis.
        """

    @abstractmethod
    def all_tracks(self) -> list[TrackRecord]:
        """Every track created this session, in creation order."""

    @abstractmethod
    def finished_tracks(self) -> list[TrackRecord]:
        """Drain tracks removed since the last call.

        A track is only ready for final motion analysis and morphology once it
        can no longer grow, so this is the natural hand-off point into the
        rest of the pipeline.
        """

    def get(self, track_id: int) -> TrackRecord | None:
        for track in self.all_tracks():
            if track.track_id == track_id:
                return track
        return None

    def reset(self) -> None:
        """Clear all state. IDs restart; only valid between sessions."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
