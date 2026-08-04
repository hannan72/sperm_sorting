"""ByteTrack: association by IoU, in two score bands.

The idea ByteTrack contributes is small and specific. Every other tracker
throws away detections below a confidence threshold before associating;
ByteTrack keeps them and gives them a *second* chance, matched only against
tracks that nothing better claimed. A low-confidence box in isolation is
usually noise, but a low-confidence box sitting exactly where an established
track predicted it is almost always the object, dimmed or half-occluded.

That is the case this product lives in. A sperm that swims under a debris
particle, tumbles so its head presents edge-on, or drifts through a dim patch
of the field does not vanish -- its detector score collapses for a handful of
frames. Discarding those frames breaks the track, and a broken track is two
tracks: the same sperm counted twice in the shot denominator, with two
half-length velocity estimates instead of one good one. The second
association pass is what prevents that.

The passes, in order:

1. **High-score vs confirmed tracks.** The ordinary case, gated by
   ``match_iou_threshold``.
2. **Low-score vs still-tracked leftovers.** Only tracks that were matched on
   the previous frame take part, as in the reference implementation: a track
   that has already been lost for several frames has a stale prediction, and
   letting it grab weak detections is how ID switches happen.
3. **Leftover high-score vs tentative tracks.** New tracks get their
   confirmation hits before they are allowed to compete with established ones.

Then births from whatever high-score detections remain, then ageing.
"""

from __future__ import annotations

import numpy as np

from ..schemas.detection import Detection
from ..schemas.enums import TrackState
from ..schemas.frame import FramePacket
from ..schemas.track import TrackRecord
from ._common import ManagedTrack, TrackerBase
from .assignment import iou_distance, linear_assignment


class ByteTracker(TrackerBase):
    """ByteTrack over Kalman-predicted boxes.

    Identity guarantees (unique never-reused IDs, one growing record per
    track, ``observed=False`` on predicted points) come from
    :class:`~sperm_sorting.tracking._common.TrackerBase` and are shared with
    the other two trackers in this package.
    """

    name = "bytetrack"

    # ------------------------------------------------------------------ API

    def update(
        self, detections: list[Detection], frame: FramePacket
    ) -> list[TrackRecord]:
        cfg = self.config
        self._frame_count += 1

        high, low = self._partition(detections)

        for track in self._tracks:
            track.predict()
        self._compensate_camera_motion(frame)

        # Snapshot before any births, so a track created on this frame is not
        # immediately counted as having missed it.
        existing = list(self._tracks)
        confirmed = [t for t in existing if t.is_confirmed]
        tentative = [t for t in existing if not t.is_confirmed]
        matched_ids: set[int] = set()

        # -- pass 1: confirmed tracks against high-score detections ----------
        cost = self._association_cost(confirmed, high, frame)
        matches, unmatched_tracks, unmatched_high = linear_assignment(
            cost, 1.0 - cfg.match_iou_threshold
        )
        for track_idx, det_idx in matches:
            track = confirmed[int(track_idx)]
            track.mark_matched(high[int(det_idx)], frame, min_hits=cfg.min_hits)
            matched_ids.add(track.track_id)

        # -- pass 2: BYTE. Leftovers that were tracked last frame, against the
        #    low-score band. This is the pass the whole algorithm is named for.
        byte_pool = [
            confirmed[int(i)]
            for i in unmatched_tracks
            if confirmed[int(i)].state is TrackState.CONFIRMED
        ]
        cost = iou_distance([t.predicted_box for t in byte_pool], [d.box for d in low])
        matches, _, _ = linear_assignment(cost, 1.0 - cfg.second_match_iou_threshold)
        for track_idx, det_idx in matches:
            track = byte_pool[int(track_idx)]
            track.mark_matched(low[int(det_idx)], frame, min_hits=cfg.min_hits)
            matched_ids.add(track.track_id)

        # -- pass 3: leftover high-score detections against tentative tracks --
        leftover_high = [high[int(i)] for i in unmatched_high]
        cost = self._association_cost(tentative, leftover_high, frame)
        matches, _, unmatched_leftover = linear_assignment(
            cost, 1.0 - cfg.match_iou_threshold
        )
        for track_idx, det_idx in matches:
            track = tentative[int(track_idx)]
            track.mark_matched(leftover_high[int(det_idx)], frame, min_hits=cfg.min_hits)
            matched_ids.add(track.track_id)

        # -- births ----------------------------------------------------------
        for det_idx in unmatched_leftover:
            self._spawn(leftover_high[int(det_idx)], frame)

        # -- misses and ageing ------------------------------------------------
        self._age_unmatched(existing, matched_ids, frame)

        return self._active_records()

    # -------------------------------------------------------------- internals

    def _association_cost(
        self,
        tracks: list[ManagedTrack],
        detections: list[Detection],
        frame: FramePacket,
    ) -> np.ndarray:
        """Cost between predicted track boxes and detections.

        Plain IoU distance here; BoT-SORT overrides it to fuse appearance.
        """
        del frame  # unused in the IoU-only cost; BoT-SORT's override needs it
        return iou_distance(
            [t.predicted_box for t in tracks], [d.box for d in detections]
        )

    def _compensate_camera_motion(self, frame: FramePacket) -> None:
        """Hook, after prediction. No-op here; BoT-SORT overrides it."""
