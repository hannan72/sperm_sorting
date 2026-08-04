"""OC-SORT: observation-centric tracking.

SORT-style trackers trust their motion model. OC-SORT starts from the
observation that this trust is misplaced whenever an object is missed for a
few frames, and fixes it in three places. All three matter here, because a
sperm is a small, low-contrast, non-linearly moving object -- the exact regime
where a Kalman filter's own predictions are least worth believing.

**Observation-Centric Momentum (OCM).** Association uses direction as well as
overlap. The heading is measured between two *real* observations
``ocsort_delta_t`` frames apart, not read out of the filter's velocity
channel, because that channel is itself an estimate and drifts when the track
is coasting. Two sperm that swim past each other at similar speeds are
separable by heading long before they are separable by overlap.

**Observation-Centric Re-Update (ORU).** When a lost track is re-found, the
filter is rewound to its state at the last real observation and re-run along
the straight line between that observation and the new one. Every frame of
accumulated self-referential drift is discarded. Without this, a track that
survives a five-frame occlusion emerges with an inflated velocity and pollutes
the CASA numbers this product reports.

**Observation-Centric Recovery (OCR).** Leftover detections are matched
against tracks' *last observations* rather than their predictions. A sperm
that changed direction while occluded reappears nowhere near where a constant
velocity model expected it, but it is still close to where it was last seen.

``ocsort_use_byte`` additionally enables ByteTrack's low-score pass between
OCM and OCR; see :mod:`sperm_sorting.tracking.bytetrack` for why that pass
earns its keep on dim, partly-occluded sperm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas.detection import BoundingBox, Detection
from ..schemas.enums import TrackState
from ..schemas.frame import FramePacket
from ..schemas.track import TrackRecord
from ._common import ManagedTrack, TrackerBase
from .assignment import (
    DEFAULT_VDC_WEIGHT,
    iou_batch,
    iou_distance,
    linear_assignment,
    speed_direction,
    velocity_direction_cost,
)
from .kalman import OCRKalmanBoxTracker


class OCSortTrack(ManagedTrack):
    """A track that remembers its observations, not just its filter state."""

    __slots__ = ("delta_t", "last_observation", "last_observation_age", "observations")

    def __init__(
        self,
        record: TrackRecord,
        detection: Detection,
        frame: FramePacket,
        *,
        std_position: float,
        std_velocity: float,
        delta_t: int = 3,
    ) -> None:
        self.delta_t = max(int(delta_t), 1)
        #: Real observations keyed by track age, pruned to the momentum window.
        self.observations: dict[int, BoundingBox] = {}
        self.last_observation: BoundingBox | None = None
        self.last_observation_age: int = 0
        super().__init__(
            record,
            detection,
            frame,
            std_position=std_position,
            std_velocity=std_velocity,
        )
        self._record_observation(detection.box)

    @staticmethod
    def _make_filter(
        box: BoundingBox, *, std_position: float, std_velocity: float
    ) -> OCRKalmanBoxTracker:
        return OCRKalmanBoxTracker(
            box, std_position=std_position, std_velocity=std_velocity
        )

    # ----------------------------------------------------------- observations

    @property
    def observation_velocity(self) -> np.ndarray | None:
        """Unit heading between two real observations, or ``None``.

        Recomputed on demand from :attr:`observations` rather than cached, so
        it can never disagree with the observation history it claims to
        summarise.
        """
        if self.last_observation is None:
            return None
        previous: BoundingBox | None = None
        # Prefer the furthest-back observation inside the window: a longer
        # baseline gives a heading that is less sensitive to detector jitter.
        for dt in range(self.delta_t, 0, -1):
            candidate = self.observations.get(self.last_observation_age - dt)
            if candidate is not None:
                previous = candidate
                break
        if previous is None:
            return None
        direction = speed_direction(
            np.array(previous.as_xyxy()), np.array(self.last_observation.as_xyxy())
        )
        if not np.any(direction):
            return None
        return direction

    def _record_observation(self, box: BoundingBox) -> None:
        self.observations[self.age] = box
        self.last_observation = box
        self.last_observation_age = self.age
        cutoff = self.age - self.delta_t - 1
        for stale_age in [a for a in self.observations if a < cutoff]:
            del self.observations[stale_age]

    # -------------------------------------------------------------- overrides

    def _apply_measurement(self, box: BoundingBox) -> None:
        """Ordinary update when tracked; observation-centric re-update when not."""
        gap = self.age - self.last_observation_age
        if (
            self.time_since_update > 0
            and self.last_observation is not None
            and gap >= 1
            and isinstance(self.kf, OCRKalmanBoxTracker)
        ):
            self.kf.observation_centric_reupdate(self.last_observation, box, gap)
        else:
            self.kf.update(box)

    def _on_matched(self, detection: Detection, frame: FramePacket) -> None:
        del frame  # the observation history is indexed by track age, not frame
        self._record_observation(detection.box)


class OCSortTracker(TrackerBase):
    """Observation-Centric SORT.

    Track identity, births and ageing are the shared implementation in
    :class:`~sperm_sorting.tracking._common.TrackerBase`; only the association
    is OC-SORT's own.
    """

    name = "ocsort"
    track_class = OCSortTrack

    #: Weight on the direction-consistency term. The paper's value; not in
    #: ``TrackingConfig`` because the config models only expose knobs that are
    #: meant to be tuned per deployment, and this one is not.
    vdc_weight: float = DEFAULT_VDC_WEIGHT

    # ------------------------------------------------------------------- API

    def update(
        self, detections: list[Detection], frame: FramePacket
    ) -> list[TrackRecord]:
        cfg = self.config
        self._frame_count += 1

        high, low = self._partition(detections)

        for track in self._tracks:
            track.predict()

        existing = list(self._tracks)
        matched_ids: set[int] = set()
        pool: list[OCSortTrack] = [t for t in existing if isinstance(t, OCSortTrack)]

        # -- stage 1: OCM. Overlap plus direction consistency ----------------
        remaining_tracks, unmatched_high = self._associate_momentum(
            pool, high, frame, matched_ids
        )

        # -- stage 2: BYTE. Leftover tracks against the low-score band --------
        if cfg.ocsort_use_byte and low:
            remaining_tracks = self._associate_byte(
                remaining_tracks, low, frame, matched_ids
            )

        # -- stage 3: OCR. Leftover detections against last observations ------
        leftover_high = [high[i] for i in unmatched_high]
        leftover_high = self._associate_recovery(
            remaining_tracks, leftover_high, frame, matched_ids
        )

        # -- births ------------------------------------------------------------
        for detection in leftover_high:
            self._spawn(detection, frame)

        # -- misses and ageing --------------------------------------------------
        self._age_unmatched(existing, matched_ids, frame)

        return self._active_records()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["ocsort_delta_t"] = self.config.ocsort_delta_t
        info["ocsort_use_byte"] = self.config.ocsort_use_byte
        info["vdc_weight"] = self.vdc_weight
        return info

    # ------------------------------------------------------------- internals

    def _spawn(self, detection: Detection, frame: FramePacket) -> ManagedTrack:
        track = super()._spawn(detection, frame)
        if isinstance(track, OCSortTrack):
            track.delta_t = max(int(self.config.ocsort_delta_t), 1)
        return track

    def _associate_momentum(
        self,
        pool: list[OCSortTrack],
        detections: list[Detection],
        frame: FramePacket,
        matched_ids: set[int],
    ) -> tuple[list[OCSortTrack], list[int]]:
        """Stage 1: IoU on predictions, plus the momentum term.

        The gate is applied to *raw IoU*, not to the fused cost, exactly as in
        the reference implementation. The direction term is allowed to decide
        between candidates but never to admit a pair that overlap alone would
        have rejected -- otherwise a confident heading could drag a track onto
        a detection it does not actually overlap.
        """
        if not pool or not detections:
            return list(pool), list(range(len(detections)))

        track_boxes = [t.predicted_box for t in pool]
        det_boxes = [d.box for d in detections]
        overlap = iou_batch(track_boxes, det_boxes)

        last_obs = np.array(
            [(t.last_observation or t.predicted_box).as_xyxy() for t in pool],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                t.observation_velocity
                if t.observation_velocity is not None
                else (np.nan, np.nan)
                for t in pool
            ],
            dtype=np.float64,
        )
        scores = np.array([d.score for d in detections], dtype=np.float64)

        cost = (1.0 - overlap) + velocity_direction_cost(
            last_obs, det_boxes, velocities, scores, weight=self.vdc_weight
        )
        matches, _, _ = linear_assignment(cost, float("inf"))

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        for track_idx, det_idx in matches:
            ti, di = int(track_idx), int(det_idx)
            if overlap[ti, di] < self.config.match_iou_threshold:
                continue
            pool[ti].mark_matched(detections[di], frame, min_hits=self.config.min_hits)
            matched_ids.add(pool[ti].track_id)
            matched_tracks.add(ti)
            matched_dets.add(di)

        remaining = [t for i, t in enumerate(pool) if i not in matched_tracks]
        unmatched = [i for i in range(len(detections)) if i not in matched_dets]
        return remaining, unmatched

    def _associate_byte(
        self,
        tracks: list[OCSortTrack],
        low: list[Detection],
        frame: FramePacket,
        matched_ids: set[int],
    ) -> list[OCSortTrack]:
        """Stage 2: still-tracked leftovers against the low-score band."""
        pool = [t for t in tracks if t.state is TrackState.CONFIRMED]
        if not pool:
            return tracks

        cost = iou_distance([t.predicted_box for t in pool], [d.box for d in low])
        matches, _, _ = linear_assignment(
            cost, 1.0 - self.config.second_match_iou_threshold
        )
        matched_tracks: set[int] = set()
        for track_idx, det_idx in matches:
            ti, di = int(track_idx), int(det_idx)
            pool[ti].mark_matched(low[di], frame, min_hits=self.config.min_hits)
            matched_ids.add(pool[ti].track_id)
            matched_tracks.add(ti)

        claimed = {pool[i].track_id for i in matched_tracks}
        return [t for t in tracks if t.track_id not in claimed]

    def _associate_recovery(
        self,
        tracks: list[OCSortTrack],
        detections: list[Detection],
        frame: FramePacket,
        matched_ids: set[int],
    ) -> list[Detection]:
        """Stage 3: leftover detections against tracks' *last observations*.

        Returns the detections that survive unmatched and are therefore
        candidates for a new track.
        """
        pool = [t for t in tracks if t.last_observation is not None]
        if not pool or not detections:
            return list(detections)

        cost = iou_distance(
            [t.last_observation for t in pool if t.last_observation is not None],
            [d.box for d in detections],
        )
        matches, _, unmatched_dets = linear_assignment(
            cost, 1.0 - self.config.match_iou_threshold
        )
        for track_idx, det_idx in matches:
            ti, di = int(track_idx), int(det_idx)
            pool[ti].mark_matched(detections[di], frame, min_hits=self.config.min_hits)
            matched_ids.add(pool[ti].track_id)
        return [detections[int(i)] for i in unmatched_dets]
