"""Best-frame selection for morphology.

The ordering rule
=================
Best-frame selection runs **only after** a sperm has been confirmed
progressive. :meth:`BestFrameSelector.select` takes an already-classified
:class:`~sperm_sorting.schemas.track.TrackRecord` and refuses -- raises
:class:`BestFrameOrderingError` -- if the track carries no motion features or
is not progressive. That is not defensive pedantry; it is the API making the
wrong order awkward on purpose, for two reasons:

1. **Budget.** Morphology is the most expensive stage in the pipeline and it
   has a per-track deadline. Only progressive sperm can ever be
   ``ai_eligible``, so selecting and cropping for every track would spend the
   entire morphology budget on cells that are disqualified before the model is
   even asked. At ~25 tracks per shot with a minority progressive, that is
   most of the budget burned for nothing.

2. **Binding.** The product invariant is that a crop belongs to the *same*
   tracked sperm whose motion was measured
   (``CropRecord.track_id == TrackRecord.track_id``,
   ``tests/test_crop_track_identity.py``). Evaluating morphology before
   tracking and motion analysis means there is no track to bind the crop to;
   the two measurements would be joined afterwards by position or by time,
   which is exactly the kind of implicit join that silently pairs one cell's
   shape with another cell's velocity in a crowded field.

Only frames with an ``observed=True`` track point are considered. A predicted
position is the motion model's opinion about where the cell probably is, not
evidence that it appeared there; cropping at a predicted box would hand the
morphology model a picture of the background, or of whatever else drifted
into that spot.

When nothing qualifies, an empty list comes back and the caller records
``MorphologyStatus.NO_VALID_CROP``. There is deliberately no "best of a bad
lot" fallback: a crop that fails the quality bar produces a morphology verdict
that looks exactly like a good one in the audit log, and that verdict decides
whether a sperm counts toward the 60% rule.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config import BestFrameConfig, CropConfig
from ..errors import SpermSortingError
from ..schemas.detection import BoundingBox, Detection
from ..schemas.enums import QualityVerdict
from ..schemas.frame import FramePacket
from ..schemas.track import TrackPoint, TrackRecord
from .frame_score import (
    DEFAULT_NORMALISATION,
    ScoreNormalisation,
    padded_box,
    score_candidate,
    validate_weights,
    visible_fraction_of,
)

__all__ = [
    "BestFrameOrderingError",
    "BestFrameSelector",
    "CandidateFrame",
    "FrameBuffer",
]

#: IoU above which a detection in the frame is taken to *be* the track's own
#: observation, when the tracker did not stamp ``track_id`` onto detections.
_SELF_MATCH_IOU: float = 0.5


class BestFrameOrderingError(SpermSortingError):
    """Best-frame selection was attempted out of order.

    Raised when :meth:`BestFrameSelector.select` is handed a track that has
    not been through motion analysis, or one that motion analysis classified
    as not progressive. See the module docstring for why this is an error
    rather than a silently empty result: an empty result would be recorded as
    ``NO_VALID_CROP`` (a statement about image quality), which would be a
    false explanation for what is actually a pipeline-ordering bug.
    """


# ==========================================================================
# Candidate
# ==========================================================================


@dataclass(slots=True)
class CandidateFrame:
    """One frame in which a track was observed, scored as a crop source.

    ``top_k_frames`` is 1 in the pipeline today, but this record carries
    everything a future multi-frame aggregation would need -- the box, the
    score, the full term breakdown and the geometry -- so that aggregating k
    crops means combining k of these rather than re-deriving them from frames
    that may have been evicted from the buffer by then.
    """

    frame_id: int
    capture_time_s: float
    #: Detection box for this track in this frame, in the frame's coordinates.
    box: BoundingBox
    #: The padded box that the extractor will cut. Stored so the extractor and
    #: the selector cannot disagree about what was evaluated.
    padded_box: BoundingBox
    #: Composite quality score in [0, 1].
    score: float
    #: Per-term breakdown; copied into ``CropRecord.quality_terms``.
    terms: dict[str, float] = field(default_factory=dict)
    detector_score: float = 0.0
    track_confidence: float = 0.0
    max_overlap_iou: float = 0.0
    visible_fraction: float = 1.0
    #: Position in the returned ranking, 0 = best.
    rank: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "capture_time_s": float(self.capture_time_s),
            "box_xyxy": list(self.box.as_xyxy()),
            "padded_box_xyxy": list(self.padded_box.as_xyxy()),
            "score": float(self.score),
            "terms": {k: float(v) for k, v in self.terms.items()},
            "detector_score": float(self.detector_score),
            "track_confidence": float(self.track_confidence),
            "max_overlap_iou": float(self.max_overlap_iou),
            "visible_fraction": float(self.visible_fraction),
            "rank": self.rank,
        }


# ==========================================================================
# Frame buffer
# ==========================================================================


class FrameBuffer(Mapping[int, FramePacket]):
    """Bounded ring buffer of recent frames, keyed by ``frame_id``.

    The selector has to look back over a track's whole lifetime, which means
    frames must outlive the moment they were processed. At ~164 FPS a 1920x1200
    ``uint8`` frame is 2.3 MB, so an unbounded cache reaches a gigabyte in
    about seven seconds and the process dies during a run that is supposed to
    last hours. This holds exactly ``capacity`` frames and evicts the oldest in
    O(1) by overwriting its slot and dropping one dict entry -- no scan, no
    re-hash, no reallocation.

    Sizing: capacity must exceed the longest track lifetime you intend to
    select over. ``TrackingConfig.max_age`` (30 frames) plus the track's
    observed span is the relevant figure; 256 covers ~1.6 s at 160 FPS.

    Implements :class:`collections.abc.Mapping`, so it can be passed directly
    as the ``frames`` argument of :meth:`BestFrameSelector.select` and can
    equally be replaced by a plain dict in tests.
    """

    __slots__ = ("_capacity", "_index", "_next", "_order", "_slots")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"FrameBuffer capacity must be >= 1, got {capacity}")
        self._capacity = int(capacity)
        self._slots: list[FramePacket | None] = [None] * self._capacity
        #: frame_id -> slot index
        self._index: dict[int, int] = {}
        #: slot index -> frame_id, so eviction knows which key to drop
        self._order: list[int | None] = [None] * self._capacity
        self._next = 0

    # ------------------------------------------------------------- container

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, frame_id: object) -> bool:
        return frame_id in self._index

    def __iter__(self) -> Iterator[int]:
        """Iterate frame ids oldest-first.

        Insertion order rather than numeric order: it is O(capacity) without
        sorting, and it is what "oldest first" means for a ring buffer even if
        the source ever restarts its counter.
        """
        for offset in range(self._capacity):
            slot = (self._next + offset) % self._capacity
            frame_id = self._order[slot]
            if frame_id is not None:
                yield frame_id

    def __getitem__(self, frame_id: int) -> FramePacket:
        slot = self._index[frame_id]
        packet = self._slots[slot]
        if packet is None:  # pragma: no cover - index and slots stay in step
            raise KeyError(frame_id)
        return packet

    def get(  # type: ignore[override]
        self, frame_id: int, default: FramePacket | None = None
    ) -> FramePacket | None:
        """Return the buffered frame, or ``default`` if it has been evicted."""
        slot = self._index.get(frame_id)
        if slot is None:
            return default
        return self._slots[slot] or default

    # ------------------------------------------------------------- mutation

    def put(self, frame: FramePacket) -> None:
        """Insert ``frame``, evicting the oldest entry when full. O(1).

        Re-inserting an existing ``frame_id`` overwrites it in place, so a
        frame re-processed after a quality re-evaluation does not consume a
        second slot.
        """
        existing = self._index.get(frame.frame_id)
        if existing is not None:
            self._slots[existing] = frame
            return

        slot = self._next
        evicted = self._order[slot]
        if evicted is not None:
            del self._index[evicted]
        self._slots[slot] = frame
        self._order[slot] = frame.frame_id
        self._index[frame.frame_id] = slot
        self._next = (slot + 1) % self._capacity

    #: Alias: reads naturally at a call site that is feeding a stream.
    append = put

    def clear(self) -> None:
        """Drop every buffered frame and release the references."""
        self._slots = [None] * self._capacity
        self._order = [None] * self._capacity
        self._index.clear()
        self._next = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"FrameBuffer(capacity={self._capacity}, held={len(self._index)})"


# ==========================================================================
# Selector
# ==========================================================================


class BestFrameSelector:
    """Ranks the frames of a progressive track as morphology crop sources.

    Parameters
    ----------
    cfg
        Weights and admission thresholds.
    crop_cfg
        The crop configuration that will actually cut the crop. Only its
        padding is used, and only so that the truncation term and the
        ``min_visible_fraction`` filter are measured on the box the extractor
        will really take. Defaults to :class:`CropConfig`'s own defaults.
    normalisation
        Scale constants for the raw-to-``[0, 1]`` mappings; see
        :class:`~sperm_sorting.quality.frame_score.ScoreNormalisation`.
    """

    __slots__ = (
        "_counters",
        "_crop_cfg",
        "_normalisation",
        "cfg",
    )

    def __init__(
        self,
        cfg: BestFrameConfig,
        crop_cfg: CropConfig | None = None,
        *,
        normalisation: ScoreNormalisation = DEFAULT_NORMALISATION,
    ) -> None:
        validate_weights(cfg)
        if cfg.top_k_frames < 1:
            raise ValueError(
                f"best_frame.top_k_frames must be >= 1, got {cfg.top_k_frames}"
            )
        self.cfg = cfg
        self._crop_cfg = crop_cfg if crop_cfg is not None else CropConfig()
        self._normalisation = normalisation
        self._counters: dict[str, int] = {
            "tracks_considered": 0,
            "tracks_with_candidates": 0,
            "points_considered": 0,
            "rejected_not_observed": 0,
            "rejected_frame_missing": 0,
            "rejected_frame_quality": 0,
            "rejected_low_score": 0,
            "rejected_overlap": 0,
            "rejected_truncated": 0,
            "accepted_candidates": 0,
        }

    # ------------------------------------------------------------------ api

    def counters(self) -> dict[str, int]:
        """Snapshot of why candidates were dropped, for the metrics layer.

        A run where every track ends ``NO_VALID_CROP`` looks identical from
        the outside whether the cause is defocus, crowding or an undersized
        frame buffer. These counters are what distinguishes them.
        """
        return dict(self._counters)

    def reset_counters(self) -> None:
        for key in self._counters:
            self._counters[key] = 0

    def select(
        self,
        track: TrackRecord,
        frames: Mapping[int, FramePacket],
        detections_by_frame: Mapping[int, list[Detection]],
    ) -> list[CandidateFrame]:
        """Return the best ``top_k_frames`` candidates, best first.

        Parameters
        ----------
        track
            A track that has already been through motion analysis **and** was
            classified progressive. Anything else raises
            :class:`BestFrameOrderingError`.
        frames
            Frames by id -- typically a :class:`FrameBuffer`. Frames already
            evicted are skipped and counted, not treated as an error: on a
            long-lived track the earliest frames are legitimately gone.
        detections_by_frame
            All detections per frame, used to measure crowding around the
            candidate. A missing entry is read as "no neighbours known", which
            makes the overlap term optimistic; supply it whenever it exists.

        Returns
        -------
        list[CandidateFrame]
            Between 0 and ``top_k_frames`` entries, sorted by descending
            score. Empty means nothing met the bar; the caller must record
            ``MorphologyStatus.NO_VALID_CROP``.
        """
        self._require_classified_progressive(track)
        self._counters["tracks_considered"] += 1

        candidates: list[CandidateFrame] = []
        for point in track.points:
            if not point.observed:
                # A predicted position is not evidence that the cell appeared.
                self._counters["rejected_not_observed"] += 1
                continue
            self._counters["points_considered"] += 1

            frame = frames.get(point.frame_id)
            if frame is None:
                self._counters["rejected_frame_missing"] += 1
                continue

            if self.cfg.require_frame_quality_pass and not _frame_quality_passes(frame):
                self._counters["rejected_frame_quality"] += 1
                continue

            candidate = self._score_point(track, point, frame, detections_by_frame)
            if candidate is None:
                continue
            candidates.append(candidate)

        if not candidates:
            return []

        # Descending score; ties broken by frame_id so the choice is
        # reproducible across runs and across Python versions.
        candidates.sort(key=lambda c: (-c.score, c.frame_id))
        top = candidates[: self.cfg.top_k_frames]
        for rank, candidate in enumerate(top):
            candidate.rank = rank
        self._counters["accepted_candidates"] += len(top)
        self._counters["tracks_with_candidates"] += 1
        return top

    # -------------------------------------------------------------- internal

    @staticmethod
    def _require_classified_progressive(track: TrackRecord) -> None:
        """Enforce the ordering rule. See the module docstring."""
        if track.motion is None:
            raise BestFrameOrderingError(
                f"track {track.track_id} has no motion features: best-frame "
                "selection runs only after tracking and motion analysis have "
                "classified the track. Selecting first would spend the "
                "morphology budget on cells that can never be eligible and "
                "would leave the crop with no motion measurement to bind to."
            )
        if not track.motion.is_progressive:
            raise BestFrameOrderingError(
                f"track {track.track_id} is classified "
                f"{track.motion.motility_class}, not progressive: morphology "
                "is only evaluated for progressive sperm, so no crop should "
                "be selected for it. Record MorphologyStatus.NOT_REQUIRED "
                "instead of calling select()."
            )

    def _score_point(
        self,
        track: TrackRecord,
        point: TrackPoint,
        frame: FramePacket,
        detections_by_frame: Mapping[int, list[Detection]],
    ) -> CandidateFrame | None:
        """Score one observed point, applying the admission filters."""
        neighbours = self._neighbours_for(track, point, detections_by_frame)
        score, terms = score_candidate(
            frame.image,
            point.box,
            neighbours,
            point.score,
            track.mean_score,
            frame.quality,
            self.cfg,
            padding_fraction=self._crop_cfg.padding_fraction,
            min_padding_px=self._crop_cfg.min_padding_px,
            normalisation=self._normalisation,
        )

        max_iou = float(terms.get("raw_max_overlap_iou", 0.0))
        if max_iou > self.cfg.max_overlap_iou:
            self._counters["rejected_overlap"] += 1
            return None

        padded = padded_box(
            point.box, self._crop_cfg.padding_fraction, self._crop_cfg.min_padding_px
        )
        visible = visible_fraction_of(padded, frame.width, frame.height)
        if visible < self.cfg.min_visible_fraction:
            self._counters["rejected_truncated"] += 1
            return None

        if score < self.cfg.min_quality_score:
            self._counters["rejected_low_score"] += 1
            return None

        return CandidateFrame(
            frame_id=point.frame_id,
            capture_time_s=point.capture_time_s,
            box=point.box,
            padded_box=padded,
            score=score,
            terms=terms,
            detector_score=float(point.score),
            track_confidence=float(track.mean_score),
            max_overlap_iou=max_iou,
            visible_fraction=visible,
        )

    @staticmethod
    def _neighbours_for(
        track: TrackRecord,
        point: TrackPoint,
        detections_by_frame: Mapping[int, list[Detection]],
    ) -> list[Detection]:
        """The other detections in the frame, excluding this track's own.

        Two ways to identify "own": the tracker normally stamps ``track_id``
        onto the detection it associated, which is exact. When it has not
        (raw detections straight from the detector, or a replayed log), fall
        back to dropping the single best-overlapping detection. Both are
        needed -- without the fallback every candidate would score an overlap
        of ~1.0 against itself and no crop would ever be selected.
        """
        detections = detections_by_frame.get(point.frame_id)
        if not detections:
            return []
        tagged = [d for d in detections if d.track_id == track.track_id]
        if tagged:
            return [d for d in detections if d.track_id != track.track_id]

        best_index = -1
        best_iou = _SELF_MATCH_IOU
        for i, det in enumerate(detections):
            iou = point.box.iou(det.box)
            if iou > best_iou:
                best_iou = iou
                best_index = i
        if best_index < 0:
            return list(detections)
        return [d for i, d in enumerate(detections) if i != best_index]


def _frame_quality_passes(frame: FramePacket) -> bool:
    """Whether a frame is admissible as a morphology crop source.

    An unmeasured frame (``quality is None``) is *not* admissible when
    ``require_frame_quality_pass`` is set. The absence of a measurement is not
    evidence of quality, and morphology feeds the eligibility decision.
    """
    quality = frame.quality
    return quality is not None and quality.verdict is QualityVerdict.PASS
