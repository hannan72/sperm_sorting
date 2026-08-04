"""Virtual counting gate.

The flow through the kit is physically continuous, so "a shot" has to be
manufactured in software. The gate is the mechanism: a line across the channel,
near the downstream end of the imaging ROI, that each track crosses exactly
once on its way out.

Placing the gate downstream rather than upstream is deliberate. A track that
has reached the downstream edge has been observed for most of its transit, so
its kinematics are as complete as they are ever going to be at the moment it is
committed to a shot. Gating on entry would commit sperm to a decision unit
before there was any evidence about them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import CountingGateConfig
from ..schemas.track import TrackRecord


@dataclass(slots=True)
class GateCrossing:
    """Record of one track crossing the gate."""

    track_id: int
    frame_id: int
    time_s: float
    #: Coordinate along the gate axis before and after the crossing.
    from_coord: float
    to_coord: float
    #: Net displacement along the gate axis over the track's whole lifetime.
    axis_displacement: float


class CountingGate:
    """Detects one-way crossings of a virtual line.

    A crossing is counted only when all three of the following hold:

    1. the track's centre moved from one side of the line to the other,
    2. it moved in the configured flow direction, and
    3. its *lifetime* displacement along the flow axis exceeds
       ``min_axis_displacement_px``.

    Condition (3) is what stops a sperm loitering on the line from being
    counted repeatedly: a cell jittering back and forth about the line has a
    near-zero lifetime displacement and never qualifies. Condition (2) stops a
    cell swimming upstream against the flow from being counted on its way back.

    Independently of all that, a hard ``set`` of already-crossed track IDs
    makes double-counting structurally impossible rather than merely unlikely.
    """

    def __init__(
        self,
        cfg: CountingGateConfig,
        roi_width: int,
        roi_height: int,
    ) -> None:
        self.cfg = cfg
        self.roi_width = roi_width
        self.roi_height = roi_height
        self._extent = roi_width if cfg.axis == "x" else roi_height
        self.position = float(cfg.position_fraction) * float(self._extent)
        #: Last observed axis coordinate per track, so a crossing can be seen.
        self._last_coord: dict[int, float] = {}
        #: Tracks that have already been counted. Never removed: an ID is
        #: never reused, so this set is monotonically correct.
        self._crossed: set[int] = set()
        self.n_crossings = 0
        self.n_rejected_wrong_direction = 0
        self.n_rejected_insufficient_displacement = 0

    # ------------------------------------------------------------------ core

    def _axis_coord(self, track: TrackRecord) -> float | None:
        box = track.last_box
        if box is None:
            return None
        return box.cx if self.cfg.axis == "x" else box.cy

    def _lifetime_axis_displacement(self, track: TrackRecord) -> float:
        observed = track.observed_points
        if len(observed) < 2:
            return 0.0
        if self.cfg.axis == "x":
            return observed[-1].x - observed[0].x
        return observed[-1].y - observed[0].y

    def update(self, track: TrackRecord) -> GateCrossing | None:
        """Advance the gate with one track's current position.

        Returns a :class:`GateCrossing` on the frame the track crosses, and
        ``None`` on every other frame, including every subsequent frame after
        it has crossed.
        """
        track_id = track.track_id
        coord = self._axis_coord(track)
        if coord is None:
            return None

        previous = self._last_coord.get(track_id)
        self._last_coord[track_id] = coord

        if track_id in self._crossed or previous is None:
            return None

        direction = self.cfg.direction
        if direction > 0:
            crossed = previous < self.position <= coord
        else:
            crossed = previous > self.position >= coord

        if not crossed:
            # Distinguish "did not reach the line" from "crossed the wrong
            # way", because a high wrong-direction count means the configured
            # flow direction is wrong and every shot will be mis-assembled.
            wrong_way = (
                (previous > self.position >= coord)
                if direction > 0
                else (previous < self.position <= coord)
            )
            if wrong_way:
                self.n_rejected_wrong_direction += 1
            return None

        displacement = self._lifetime_axis_displacement(track)
        if abs(displacement) < self.cfg.min_axis_displacement_px:
            self.n_rejected_insufficient_displacement += 1
            return None
        if direction > 0 and displacement <= 0:
            self.n_rejected_wrong_direction += 1
            return None
        if direction < 0 and displacement >= 0:
            self.n_rejected_wrong_direction += 1
            return None

        self._crossed.add(track_id)
        self.n_crossings += 1
        return GateCrossing(
            track_id=track_id,
            frame_id=track.last_frame_id,
            time_s=track.last_time_s,
            from_coord=previous,
            to_coord=coord,
            axis_displacement=displacement,
        )

    def has_crossed(self, track_id: int) -> bool:
        return track_id in self._crossed

    def forget(self, track_id: int) -> None:
        """Drop a finished track's position cache.

        The ``_crossed`` set is deliberately *not* cleared: forgetting that a
        track was counted is exactly how a sperm would get counted twice.
        """
        self._last_coord.pop(track_id, None)

    def stats(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "axis": self.cfg.axis,
            "direction": self.cfg.direction,
            "n_crossings": self.n_crossings,
            "n_rejected_wrong_direction": self.n_rejected_wrong_direction,
            "n_rejected_insufficient_displacement": (
                self.n_rejected_insufficient_displacement
            ),
            "n_tracked_positions": len(self._last_coord),
        }
