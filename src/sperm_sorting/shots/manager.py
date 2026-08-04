"""Shot assembly.

Turns a continuous stream of gated tracks into discrete decision units, and
holds each closed unit open just long enough for its members' morphology to
resolve before handing it to the decision engine.

The two-phase lifecycle matters:

* **Assembly** -- tracks are added as they cross the gate. The shot closes on
  whichever of three conditions fires first: the target count, the hard
  maximum, or the duration limit.
* **Finalisation** -- a closed shot still has members whose morphology is in
  flight. It becomes decidable when every member has resolved, or when the
  morphology deadline passes, whichever comes first. Members that never
  resolved stay in the denominator and are recorded as deadline misses.

Collapsing those two phases into one would force a choice between deciding on
incomplete data and blocking the pipeline; keeping them apart lets the shot
wait a bounded time and then decide honestly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import ShotConfig
from ..schemas.enums import IneligibilityReason, MorphologyStatus, ShotCloseReason
from ..schemas.shot import ShotRecord
from ..schemas.track import TrackRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingShot:
    """A closed shot waiting for its members' morphology to resolve."""

    record: ShotRecord
    #: Track IDs still awaiting a morphology verdict.
    awaiting: set[int] = field(default_factory=set)
    #: Monotonic instant after which the shot is decided regardless.
    deadline_s: float = 0.0

    @property
    def is_ready(self) -> bool:
        return not self.awaiting


class ShotManager:
    """Assembles gated tracks into shots and finalises them.

    Not thread-safe by design: it is driven from one pipeline stage. Sharing it
    across workers would reintroduce exactly the ordering hazards the gate
    exists to remove.
    """

    def __init__(self, cfg: ShotConfig, morphology_deadline_s: float) -> None:
        self.cfg = cfg
        self.morphology_deadline_s = morphology_deadline_s
        self._next_shot_id = 0
        self._open: ShotRecord | None = None
        self._pending: list[PendingShot] = []
        #: Every shot ever produced, for the audit log and for tests.
        self._history: list[ShotRecord] = []
        #: Every track ever assigned to any shot. Belt and braces against
        #: double counting: the gate already guarantees one crossing per
        #: track, and this guarantees one shot membership per track.
        self._assigned: set[int] = set()
        #: Tracks whose morphology resolved *before* their shot closed.
        #:
        #: This is the common case, not an edge case. The counting gate sits
        #: near the downstream edge of the ROI, so a track crosses it and
        #: leaves the field shortly afterwards -- while the shot it joined is
        #: usually still filling toward 25 members. Its resolution therefore
        #: arrives before there is any pending shot to record it against.
        #: Without this set those notifications would be dropped and every
        #: such member would be counted as a deadline miss, which would empty
        #: the numerator of nearly every shot while leaving the denominator
        #: full -- a systematic bias toward REJECT.
        self._resolved_early: set[int] = set()
        self.n_duplicate_assignments_rejected = 0

    # ----------------------------------------------------------- assembly

    @property
    def open_shot(self) -> ShotRecord | None:
        return self._open

    def _open_new_shot(self, now_s: float, frame_id: int) -> ShotRecord:
        record = ShotRecord(
            shot_id=self._next_shot_id,
            opened_at_s=now_s,
            opened_frame_id=frame_id,
        )
        self._next_shot_id += 1
        self._open = record
        logger.debug("shot %d opened at t=%.4f", record.shot_id, now_s)
        return record

    def add_track(
        self, track: TrackRecord, now_s: float, frame_id: int
    ) -> ShotRecord | None:
        """Assign a gated track to the open shot, opening one if needed.

        Returns the shot that *closed* as a result of this addition, or
        ``None`` if the shot is still open. A track that fails the track-
        quality bar is counted as rejected and never enters the denominator.
        """
        if track.track_id in self._assigned:
            # Should be impossible: the gate counts each track once. Recorded
            # rather than raised, so a tracker bug degrades a single shot
            # instead of stopping the run, but it is a loud warning.
            self.n_duplicate_assignments_rejected += 1
            logger.warning(
                "track %d was already assigned to a shot; refusing to count it "
                "twice (this indicates a tracker or gate defect)",
                track.track_id,
            )
            return None

        shot = self._open or self._open_new_shot(now_s, frame_id)

        if not track.track_quality_pass:
            # Not a trustworthy observation of one sperm: excluded from the
            # numerator *and* the denominator, but counted so the operator can
            # see how much of the field is being discarded.
            shot.rejected_track_count += 1
            shot.note_gate_crossing(now_s)
            self._assigned.add(track.track_id)
            track.shot_id = None
            return self._maybe_close(now_s, frame_id)

        if not shot.add_track(track.track_id):
            self.n_duplicate_assignments_rejected += 1
            return None

        self._assigned.add(track.track_id)
        shot.note_gate_crossing(now_s)
        track.shot_id = shot.shot_id
        track.gate_crossing_frame_id = frame_id
        track.gate_crossing_time_s = now_s
        return self._maybe_close(now_s, frame_id)

    def _maybe_close(self, now_s: float, frame_id: int) -> ShotRecord | None:
        shot = self._open
        if shot is None:
            return None
        count = shot.trackable_count
        if count >= self.cfg.maximum_trackable_sperm:
            return self.close(ShotCloseReason.HARD_MAXIMUM, now_s, frame_id)
        if count >= self.cfg.target_trackable_sperm:
            return self.close(ShotCloseReason.TARGET_REACHED, now_s, frame_id)
        return None

    def poll(self, now_s: float, frame_id: int) -> ShotRecord | None:
        """Close the open shot if it has exceeded its duration limit.

        Must be called every frame, not only when a track is gated: a shot
        that stops receiving sperm entirely still has to time out, otherwise
        an empty channel would stall the controller indefinitely.
        """
        shot = self._open
        if shot is None:
            return None
        if now_s - shot.opened_at_s >= self.cfg.maximum_shot_duration_seconds:
            return self.close(ShotCloseReason.TIMEOUT, now_s, frame_id)
        return None

    def close(
        self, reason: ShotCloseReason, now_s: float, frame_id: int
    ) -> ShotRecord | None:
        """Close the open shot and move it to the pending-finalisation list."""
        shot = self._open
        if shot is None:
            return None
        shot.closed_at_s = now_s
        shot.closed_frame_id = frame_id
        shot.close_reason = reason
        self._open = None
        self._history.append(shot)
        # Members that already resolved while the shot was still filling are
        # not awaited, and their bookkeeping is dropped now that it has been
        # consumed.
        awaiting = set(shot.track_ids) - self._resolved_early
        self._resolved_early.difference_update(shot.track_ids)
        self._pending.append(
            PendingShot(
                record=shot,
                awaiting=awaiting,
                deadline_s=now_s + self.morphology_deadline_s,
            )
        )
        logger.debug(
            "shot %d closed: reason=%s trackable=%d",
            shot.shot_id,
            reason,
            shot.trackable_count,
        )
        return shot

    def flush(self, now_s: float, frame_id: int) -> ShotRecord | None:
        """Close a partially-filled shot at shutdown, if configured to."""
        if self._open is None or not self.cfg.flush_on_shutdown:
            return None
        return self.close(ShotCloseReason.SHUTDOWN, now_s, frame_id)

    # -------------------------------------------------------- finalisation

    def notify_track_resolved(self, track_id: int) -> None:
        """Mark a member's morphology as resolved (successfully or not).

        Safe to call before the member's shot has closed: the resolution is
        remembered and applied when the shot enters the pending list. See
        :attr:`_resolved_early` for why that ordering is the normal one.
        """
        found = False
        for pending in self._pending:
            if track_id in pending.awaiting:
                pending.awaiting.discard(track_id)
                found = True
        if not found:
            self._resolved_early.add(track_id)

    def ready_shots(
        self, now_s: float, tracks_by_id: dict[int, TrackRecord]
    ) -> list[ShotRecord]:
        """Return and remove shots that are ready to be decided.

        A shot is ready when every member has resolved, or when its deadline
        has passed. Members that have not resolved by the deadline are marked
        ``DEADLINE_MISSED`` -- they stay in the denominator, because they were
        genuinely observed sperm; they simply could not be shown to qualify.
        """
        ready: list[ShotRecord] = []
        still_pending: list[PendingShot] = []

        for pending in self._pending:
            expired = now_s >= pending.deadline_s
            if not (pending.is_ready or expired):
                still_pending.append(pending)
                continue

            if expired and pending.awaiting:
                for track_id in sorted(pending.awaiting):
                    track = tracks_by_id.get(track_id)
                    if track is None:
                        continue
                    track.evaluation_complete = False
                    if track.morphology is None:
                        from ..schemas.morphology import MorphologyResult

                        track.morphology = MorphologyResult.failed(
                            track_id,
                            MorphologyStatus.DEADLINE_MISSED,
                            "morphology did not complete before the shot's "
                            "finalisation deadline",
                        )
                        logger.info(
                            "track %d missed the morphology deadline for shot "
                            "%d; counted in the denominator only",
                            track_id,
                            pending.record.shot_id,
                        )
                    else:
                        # A member holding a finished result that never
                        # reached us means notify_track_resolved() was not
                        # called for it -- a pipeline wiring defect, not a
                        # slow model. The existing result is a real
                        # measurement and is left intact rather than
                        # overwritten, but the track is still excluded from
                        # the numerator, because the shot could not confirm
                        # it in time and we do not retroactively admit
                        # evidence that arrived after the decision window.
                        logger.warning(
                            "track %d held a %s morphology result at the "
                            "deadline for shot %d but was never reported as "
                            "resolved; excluding it from the numerator and "
                            "keeping the result for audit. This indicates a "
                            "missing notify_track_resolved() call.",
                            track_id,
                            track.morphology.status,
                            pending.record.shot_id,
                        )
                pending.awaiting.clear()

            self._finalise(pending.record, tracks_by_id)
            ready.append(pending.record)

        self._pending = still_pending
        return ready

    def _finalise(
        self, shot: ShotRecord, tracks_by_id: dict[int, TrackRecord]
    ) -> None:
        """Evaluate eligibility for every member and populate the numerator."""
        shot.eligible_track_ids.clear()
        shot.ineligibility_histogram.clear()

        for track_id in shot.track_ids:
            track = tracks_by_id.get(track_id)
            if track is None:
                # A member we can no longer inspect cannot be shown eligible.
                # It stays in the denominator; dropping it would inflate the
                # ratio by quietly shrinking the divisor.
                shot.record_ineligibility(IneligibilityReason.MORPHOLOGY_INCOMPLETE)
                continue
            if track.compute_eligibility():
                shot.eligible_track_ids.append(track_id)
            else:
                shot.record_ineligibility(track.ineligibility_reason)

    # ------------------------------------------------------------- reporting

    @property
    def history(self) -> list[ShotRecord]:
        return list(self._history)

    def pending_count(self) -> int:
        return len(self._pending)

    def stats(self) -> dict[str, Any]:
        return {
            "shots_opened": self._next_shot_id,
            "shots_pending": len(self._pending),
            "open_shot_id": self._open.shot_id if self._open else None,
            "open_shot_count": self._open.trackable_count if self._open else 0,
            "duplicate_assignments_rejected": self.n_duplicate_assignments_rejected,
            "tracks_assigned": len(self._assigned),
        }

    def drain_pending(
        self, now_s: float, tracks_by_id: dict[int, TrackRecord]
    ) -> list[ShotRecord]:
        """Force-finalise every pending shot. Used only at shutdown."""
        ready: list[ShotRecord] = []
        for pending in self._pending:
            if pending.awaiting:
                for track_id in sorted(pending.awaiting):
                    track = tracks_by_id.get(track_id)
                    if track is not None and track.morphology is None:
                        from ..schemas.morphology import MorphologyResult

                        track.morphology = MorphologyResult.failed(
                            track_id,
                            MorphologyStatus.DEADLINE_MISSED,
                            "pipeline shut down before morphology completed",
                        )
                pending.awaiting.clear()
            self._finalise(pending.record, tracks_by_id)
            ready.append(pending.record)
        self._pending.clear()
        return ready


def summarise_shots(shots: Iterable[ShotRecord]) -> dict[str, Any]:
    """Aggregate statistics over a run's shots, for the engineering report."""
    shots = list(shots)
    if not shots:
        return {"n_shots": 0}
    decided = [s for s in shots if s.status is not None]
    counts: dict[str, int] = {}
    for shot in decided:
        key = str(shot.status)
        counts[key] = counts.get(key, 0) + 1
    ratios = [s.ai_eligible_ratio for s in decided if s.ai_eligible_ratio is not None]
    return {
        "n_shots": len(shots),
        "n_decided": len(decided),
        "status_counts": counts,
        "mean_trackable_count": sum(s.trackable_count for s in shots) / len(shots),
        "mean_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "mean_duration_s": sum(s.duration_s for s in shots) / len(shots),
        "indeterminate_rate": (
            counts.get("indeterminate", 0) / len(decided) if decided else None
        ),
    }
