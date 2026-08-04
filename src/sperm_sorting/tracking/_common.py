"""Track identity and lifecycle bookkeeping shared by every tracker.

The three trackers in this package differ only in *how they associate*.
Everything the product actually depends on -- that an ID is never reused, that
one physical sperm yields one growing :class:`TrackRecord`, that a predicted
position is flagged as predicted -- is implemented exactly once, here, so that
a fix or a bug applies uniformly to all three rather than to whichever one
happened to be audited.

Two decisions in this module are worth reading before changing anything:

**Observed points carry the detector's box, not the filter's.**
A point with ``observed=True`` is the raw measurement. Smoothing is a
downstream choice (``MotionConfig.smoothing``); if the tracker smoothed first,
downstream code would smooth a smoothed signal and no one could tell how much
of the reported velocity came from the Kalman gain rather than from the sperm.

**Trailing predicted points are dropped when a track is retired.**
While a track is alive, every frame it survives gets a point -- measured or
predicted -- so a caller can always ask "where was this sperm on frame N".
But a track dies precisely because its predictions were never confirmed, so
the ``max_age`` frames of extrapolation at the end are supported by no
measurement at all. Keeping them would push short tracks over
``TrackQualityConfig.max_interpolated_fraction`` and drop real sperm out of
the shot denominator. The trim happens at the single instant a record becomes
final -- the transition to ``REMOVED``, before ``finished_tracks()`` hands it
downstream -- so no caller ever sees a live record shrink. Interior gaps,
which *are* bracketed by real observations, are always kept.
"""

from __future__ import annotations

from typing import Any

from ..config import TrackingConfig
from ..schemas.detection import BoundingBox, Detection
from ..schemas.enums import TrackState
from ..schemas.frame import FramePacket
from ..schemas.track import TrackPoint, TrackRecord
from .base import Tracker
from .kalman import KalmanBoxTracker

#: Score written onto a predicted point. Zero rather than a carried-forward
#: value: there was no detection, so there is no detector confidence, and
#: inventing one would make an extrapolation indistinguishable from a
#: measurement in any plot or log.
PREDICTED_POINT_SCORE: float = 0.0


class TrackStore:
    """Owns track IDs and records for one tracker instance.

    The monotonic counter is the whole mechanism behind the never-reuse rule
    (``TrackingConfig.reuse_track_ids`` is typed ``Literal[False]`` to make
    that non-negotiable at the config level too). Records are kept forever, so
    :meth:`get` resolves an ID long after the track was removed -- which is
    what lets a downstream consumer prove that a "new" ID really is new.
    """

    def __init__(self) -> None:
        self._next_id: int = 1
        self._records: dict[int, TrackRecord] = {}
        self._order: list[int] = []
        self._pending_finished: list[int] = []
        self._ever_finished: set[int] = set()

    def new_record(self) -> TrackRecord:
        """Allocate the next never-before-used ID and its record."""
        track_id = self._next_id
        self._next_id += 1
        record = TrackRecord(track_id=track_id, state=TrackState.TENTATIVE)
        self._records[track_id] = record
        self._order.append(track_id)
        return record

    def get(self, track_id: int) -> TrackRecord | None:
        return self._records.get(track_id)

    def all_records(self) -> list[TrackRecord]:
        """Every record ever created, in creation order."""
        return [self._records[tid] for tid in self._order]

    def retire(self, record: TrackRecord) -> None:
        """Mark a record final: trim, set ``REMOVED``, queue it for draining.

        Idempotent, and a record is queued at most once *ever*, so
        :meth:`drain_finished` can never hand the same track to the analysis
        stage twice.
        """
        if record.track_id in self._ever_finished:
            return
        _trim_trailing_predictions(record)
        record.state = TrackState.REMOVED
        record.time_since_update = 0
        self._ever_finished.add(record.track_id)
        self._pending_finished.append(record.track_id)

    def drain_finished(self) -> list[TrackRecord]:
        """Return and clear the queue of newly-retired records."""
        drained = [self._records[tid] for tid in self._pending_finished]
        self._pending_finished.clear()
        return drained

    def reset(self) -> None:
        """Forget everything, including the ID counter. Between sessions only."""
        self._next_id = 1
        self._records.clear()
        self._order.clear()
        self._pending_finished.clear()
        self._ever_finished.clear()


def _trim_trailing_predictions(record: TrackRecord) -> None:
    """Drop the unconfirmed extrapolation at the end of a finished track."""
    while record.points and not record.points[-1].observed:
        record.points.pop()
    if record.points:
        last = record.points[-1]
        record.last_frame_id = last.frame_id
        record.last_time_s = last.capture_time_s


class ManagedTrack:
    """One live track: a Kalman filter, a lifecycle, and its growing record.

    Subclasses (OC-SORT, BoT-SORT) add their own bookkeeping by overriding
    :meth:`_apply_measurement` and :meth:`_on_matched`; the record-growing and
    state-machine logic stays here.
    """

    __slots__ = (
        "age",
        "hits",
        "kf",
        "last_box",
        "last_score",
        "record",
        "state",
        "time_since_update",
    )

    def __init__(
        self,
        record: TrackRecord,
        detection: Detection,
        frame: FramePacket,
        *,
        std_position: float,
        std_velocity: float,
    ) -> None:
        self.record = record
        self.kf = self._make_filter(
            detection.box, std_position=std_position, std_velocity=std_velocity
        )
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.last_box = detection.box
        self.last_score = float(detection.score)

        detection.track_id = record.track_id
        record.state = TrackState.TENTATIVE
        record.hit_count = 1
        record.time_since_update = 0
        record.add_point(
            TrackPoint(
                frame_id=frame.frame_id,
                capture_time_s=frame.capture_time_s,
                box=detection.box,
                score=float(detection.score),
                observed=True,
            )
        )

    # ---------------------------------------------------------- construction

    @staticmethod
    def _make_filter(
        box: BoundingBox, *, std_position: float, std_velocity: float
    ) -> KalmanBoxTracker:
        return KalmanBoxTracker(box, std_position=std_position, std_velocity=std_velocity)

    # -------------------------------------------------------------- identity

    @property
    def track_id(self) -> int:
        return self.record.track_id

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.LOST)

    # --------------------------------------------------------------- motion

    def predict(self) -> BoundingBox:
        """Advance the filter one frame; the result is this frame's prior."""
        self.age += 1
        self.last_box = self.kf.predict()
        return self.last_box

    @property
    def predicted_box(self) -> BoundingBox:
        """The box this track is currently betting on."""
        return self.last_box

    # -------------------------------------------------------------- outcome

    def mark_matched(
        self, detection: Detection, frame: FramePacket, *, min_hits: int
    ) -> None:
        """Fold in a measurement and append an ``observed=True`` point."""
        self._apply_measurement(detection.box)
        self._on_matched(detection, frame)

        self.hits += 1
        self.time_since_update = 0
        self.last_box = detection.box
        self.last_score = float(detection.score)
        if self.hits >= min_hits or self.state is TrackState.LOST:
            self.state = TrackState.CONFIRMED

        detection.track_id = self.record.track_id
        self.record.state = self.state
        self.record.hit_count = self.hits
        self.record.time_since_update = 0
        self.record.add_point(
            TrackPoint(
                frame_id=frame.frame_id,
                capture_time_s=frame.capture_time_s,
                box=detection.box,
                score=float(detection.score),
                observed=True,
            )
        )

    def mark_missed(self, frame: FramePacket) -> None:
        """Append the motion model's guess, flagged ``observed=False``."""
        self.time_since_update += 1
        if self.state is TrackState.CONFIRMED:
            self.state = TrackState.LOST

        self.record.state = self.state
        self.record.time_since_update = self.time_since_update
        self.record.add_point(
            TrackPoint(
                frame_id=frame.frame_id,
                capture_time_s=frame.capture_time_s,
                box=self.kf.to_box(),
                score=PREDICTED_POINT_SCORE,
                observed=False,
            )
        )

    # ------------------------------------------------------------ overridable

    def _apply_measurement(self, box: BoundingBox) -> None:
        """Feed a measurement to the filter. OC-SORT overrides this."""
        self.kf.update(box)

    def _on_matched(self, detection: Detection, frame: FramePacket) -> None:
        """Hook for per-tracker bookkeeping on a successful match."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(id={self.track_id}, state={self.state}, "
            f"hits={self.hits}, age={self.age}, tsu={self.time_since_update})"
        )


class TrackerBase(Tracker):
    """Store ownership, births and ageing, shared by all three trackers.

    Subclasses implement :meth:`update` -- i.e. association -- and nothing
    else. Anything that touches a track ID or a record lives here.
    """

    #: Per-track state class; subclasses swap in their own bookkeeping.
    track_class: type[ManagedTrack] = ManagedTrack

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config if config is not None else TrackingConfig()
        self._store = TrackStore()
        self._tracks: list[ManagedTrack] = []
        self._frame_count = 0

    # ------------------------------------------------------------- interface

    def all_tracks(self) -> list[TrackRecord]:
        return self._store.all_records()

    def finished_tracks(self) -> list[TrackRecord]:
        return self._store.drain_finished()

    def get(self, track_id: int) -> TrackRecord | None:
        return self._store.get(track_id)

    def reset(self) -> None:
        self._store.reset()
        self._tracks.clear()
        self._frame_count = 0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "high_score_threshold": self.config.high_score_threshold,
            "low_score_threshold": self.config.low_score_threshold,
            "match_iou_threshold": self.config.match_iou_threshold,
            "second_match_iou_threshold": self.config.second_match_iou_threshold,
            "max_age": self.config.max_age,
            "min_hits": self.config.min_hits,
        }

    # ------------------------------------------------------------- internals

    def _partition(
        self, detections: list[Detection]
    ) -> tuple[list[Detection], list[Detection]]:
        """Split into the high and low score bands, preserving input order.

        Detections below ``low_score_threshold`` are dropped: below that score
        a box is not evidence of anything, and admitting it would let noise
        keep dead tracks alive.
        """
        cfg = self.config
        high = [d for d in detections if d.score >= cfg.high_score_threshold]
        low = [
            d
            for d in detections
            if cfg.low_score_threshold <= d.score < cfg.high_score_threshold
        ]
        return high, low

    def _active_records(self) -> list[TrackRecord]:
        """Records to return from :meth:`update`: confirmed, matched or lost."""
        return [t.record for t in self._tracks if t.is_confirmed]

    def _spawn(self, detection: Detection, frame: FramePacket) -> ManagedTrack:
        """Create a track with a brand-new, never-before-issued ID."""
        record = self._store.new_record()
        track = self.track_class(
            record,
            detection,
            frame,
            std_position=self.config.kalman_std_position,
            std_velocity=self.config.kalman_std_velocity,
        )
        if track.hits >= self.config.min_hits:
            track.state = TrackState.CONFIRMED
            record.state = TrackState.CONFIRMED
        self._tracks.append(track)
        return track

    def _age_unmatched(
        self,
        existing: list[ManagedTrack],
        matched_ids: set[int],
        frame: FramePacket,
    ) -> None:
        """Extrapolate unmatched tracks, then retire the ones that are done.

        ``existing`` is the pre-birth snapshot of live tracks, so a track
        created on this frame is never counted as having missed it.
        """
        retired: list[ManagedTrack] = []
        for track in existing:
            if track.track_id in matched_ids:
                continue
            was_tentative = not track.is_confirmed
            track.mark_missed(frame)
            # An unconfirmed track that misses is almost always a detector
            # false positive; keeping it alive would let it steal a real
            # sperm's detection later.
            if was_tentative or track.time_since_update > self.config.max_age:
                retired.append(track)

        if not retired:
            return
        for track in retired:
            self._store.retire(track.record)
        dead = {t.track_id for t in retired}
        self._tracks = [t for t in self._tracks if t.track_id not in dead]
