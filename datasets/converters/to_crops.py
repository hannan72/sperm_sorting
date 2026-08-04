"""Turn a detection/tracking dataset into per-track morphology crops.

The rule this module refuses to reimplement
-------------------------------------------
Crops are cut by :class:`sperm_sorting.cropping.extractor.CropExtractor` and
scored by :func:`sperm_sorting.quality.frame_score.score_candidate` -- the same
objects the live pipeline uses, imported, not copied. That matters more than it
looks:

* the padding rule (``max(min_padding_px, padding_fraction * longest_side)``)
  decides how much tail comes into the crop;
* the letterbox rule preserves aspect ratio, because head morphology is largely
  a length-to-width judgement and stretching a non-square box distorts exactly
  the feature the model is asked about;
* the normalisation takes its statistics from the content region only, ignoring
  the synthetic border.

A training set built with a *reimplementation* of those rules differs from what
the device feeds the model at inference in a way that is invisible in the code
and shows up only as a model that works in validation and not on the bench. So
there is one implementation, and this module is a caller of it.

What this module will not do
----------------------------
It does **not** invent morphology labels. VISEM-Tracking has boxes and track IDs
and no morphology annotation whatsoever; crops extracted from it are unlabelled,
and the index records ``"morphology": null``. They are useful for
self-supervised pre-training, for annotation by an embryologist, and for
measuring domain shift -- not for training the four-aspect head. Labels arrive
only from a source that actually has them: MHSMA (already cropped, so it does
not pass through here) or a device capture whose ``ObjectRecord.morphology`` an
operator filled in.

Best-frame selection here versus in the pipeline
------------------------------------------------
:class:`sperm_sorting.quality.selector.BestFrameSelector` deliberately refuses to
run on a track that has not been through motion analysis and classified
progressive -- see its module docstring: it is guarding the morphology budget and
the crop/track binding. At dataset-build time there is no motility
classification and no budget, so this module cannot use the selector. It uses
the selector's *scoring function* (:func:`score_candidate`) with the same
:class:`~sperm_sorting.config.BestFrameConfig`, and applies the admission
thresholds itself. The scores are therefore directly comparable with the
pipeline's; only the ordering constraint differs, and that difference is
recorded in every index record as ``"selection": "offline"``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sperm_sorting.config import BestFrameConfig, CropConfig
from sperm_sorting.constants import MORPHOLOGY_ASPECTS
from sperm_sorting.cropping.extractor import CropExtractor
from sperm_sorting.quality.frame_score import (
    padded_box,
    score_candidate,
    visible_fraction_of,
)
from sperm_sorting.quality.selector import CandidateFrame
from sperm_sorting.schemas.detection import Detection
from sperm_sorting.schemas.enums import SourceKind, TimestampSource
from sperm_sorting.schemas.frame import FramePacket
from sperm_sorting.schemas.track import TrackRecord

__all__ = [
    "CropDatasetBuilder",
    "CropDatasetSummary",
    "ExtractedCrop",
    "load_grayscale",
]


def load_grayscale(path: str | Path) -> np.ndarray:
    """Read an image as a 2-D array, preserving bit depth.

    ``IMREAD_UNCHANGED`` then an explicit channel reduction, rather than
    ``IMREAD_GRAYSCALE``: the latter converts 16-bit input to 8-bit silently,
    and a device camera's extra four bits of dynamic range are not something to
    throw away inside a helper.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@dataclass(slots=True)
class ExtractedCrop:
    """One crop on disk plus the record that makes it auditable."""

    track_key: str
    track_id: int
    frame_id: int
    path: Path
    #: ``CropRecord.to_json_dict()`` from the shared extractor.
    record: dict[str, Any]
    #: Per-aspect labels, or ``None`` when the source has no morphology
    #: annotation. Never fabricated.
    morphology: dict[str, int | None] | None = None
    group_id: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "track_key": self.track_key,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "group_id": self.group_id,
            "path": str(self.path),
            "morphology": self.morphology,
            "aspects": list(MORPHOLOGY_ASPECTS),
            "selection": "offline",
            "crop_record": self.record,
        }


@dataclass(slots=True)
class CropDatasetSummary:
    """What a build produced, and what it rejected and why."""

    output_dir: Path
    crops: list[ExtractedCrop] = field(default_factory=list)
    #: ``reason -> count`` for tracks/frames that produced no crop.
    rejected: dict[str, int] = field(default_factory=dict)
    n_tracks_seen: int = 0
    n_frames_seen: int = 0
    n_labelled: int = 0

    @property
    def n_crops(self) -> int:
        return len(self.crops)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "n_crops": self.n_crops,
            "n_tracks_seen": self.n_tracks_seen,
            "n_frames_seen": self.n_frames_seen,
            "n_crops_with_morphology_labels": self.n_labelled,
            "rejected": dict(sorted(self.rejected.items())),
        }


class CropDatasetBuilder:
    """Build a morphology crop set from a boxed, tracked dataset.

    Parameters
    ----------
    crop_cfg
        Padding, output size, letterboxing and normalisation. Defaults to
        :class:`~sperm_sorting.config.CropConfig`'s defaults, which are the
        pipeline's.
    best_frame_cfg
        Scoring weights and admission thresholds, shared with the runtime
        selector.
    top_k
        Crops to keep per track, best first. 1 matches the pipeline.
    image_loader
        ``path -> 2-D array``. Defaults to :func:`load_grayscale`. Override to
        read from a video container or an archive.
    """

    def __init__(
        self,
        crop_cfg: CropConfig | None = None,
        best_frame_cfg: BestFrameConfig | None = None,
        *,
        top_k: int = 1,
        image_loader: Callable[[str | Path], np.ndarray] | None = None,
    ) -> None:
        self.crop_cfg = crop_cfg or CropConfig()
        self.best_frame_cfg = best_frame_cfg or BestFrameConfig()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = int(top_k)
        self.extractor = CropExtractor(self.crop_cfg)
        self.load_image = image_loader or load_grayscale

    # ------------------------------------------------------------------ api

    def build(
        self,
        frames: Iterable[Any],
        output_dir: str | Path,
        *,
        group_id: str = "",
        morphology_labels: Mapping[int, Mapping[str, int | None]] | None = None,
        write_index: bool = True,
        min_track_length: int = 1,
    ) -> CropDatasetSummary:
        """Extract crops for every track appearing in ``frames``.

        Parameters
        ----------
        frames
            Any iterable of frame-like objects carrying ``frame_id``, an image
            (``image`` array or ``image_path``) and detections (a ``detections``
            sequence *or* a ``detections()`` method -- both shapes exist among
            this package's adapters and both are accepted). Optional
            ``capture_time_s`` is used when present.
        output_dir
            Crops and ``index.jsonl`` are written here.
        group_id
            Video / sample / patient identifier, copied onto every record. This
            is the key a leakage-safe split is later built on
            (:func:`datasets.validators.leakage.patient_level_split`), so a
            build that omits it produces a crop set that cannot be split safely.
        morphology_labels
            ``track_id -> {aspect: 0|1|None}``, from a source that genuinely has
            morphology annotation. Omit it and every crop is recorded with
            ``"morphology": null``. Nothing is ever inferred from box size,
            motion or anything else.
        write_index
            Write ``index.jsonl`` alongside the crops.
        min_track_length
            Skip tracks observed in fewer than this many frames. A one-frame
            track is as likely to be a detector blink as a sperm.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = CropDatasetSummary(output_dir=out_dir)

        # Buffer the frames: a track's best frame can be any frame it appears
        # in, so this needs random access. Images are loaded lazily and cached
        # only for the frames that win, which is what keeps a 1500-frame video
        # from becoming 1500 decoded images in RAM.
        buffered: list[tuple[int, float, Any, list[Detection]]] = []
        by_track: dict[int, list[tuple[int, Detection]]] = {}

        for index, frame in enumerate(frames):
            frame_id = int(getattr(frame, "frame_id", index))
            capture_time = float(getattr(frame, "capture_time_s", 0.0) or 0.0)
            detections = _detections_of(frame)
            buffered.append((frame_id, capture_time, frame, detections))
            summary.n_frames_seen += 1
            for det in detections:
                if det.track_id is None:
                    summary.reject("detection_without_track_id")
                    continue
                by_track.setdefault(int(det.track_id), []).append((len(buffered) - 1, det))

        summary.n_tracks_seen = len(by_track)
        index_records: list[dict[str, Any]] = []
        image_cache: dict[int, np.ndarray] = {}

        for track_id, observations in sorted(by_track.items()):
            if len(observations) < min_track_length:
                summary.reject("track_too_short")
                continue

            mean_score = float(np.mean([d.score for _, d in observations]))
            scored: list[tuple[float, dict[str, float], int, Detection]] = []
            for buffer_index, det in observations:
                frame_id, _capture_time, frame_obj, siblings = buffered[buffer_index]
                image = self._image_for(buffer_index, frame_obj, image_cache)
                if image is None:
                    summary.reject("frame_image_unavailable")
                    continue
                neighbours = [d for d in siblings if d is not det]
                score, terms = score_candidate(
                    image=image,
                    box=det.box,
                    neighbours=neighbours,
                    detector_score=float(det.score),
                    track_confidence=mean_score,
                    frame_quality=None,
                    cfg=self.best_frame_cfg,
                    padding_fraction=self.crop_cfg.padding_fraction,
                    min_padding_px=self.crop_cfg.min_padding_px,
                )
                scored.append((score, terms, buffer_index, det))

            if not scored:
                summary.reject("no_scorable_frame")
                continue

            scored.sort(key=lambda item: item[0], reverse=True)
            kept = 0
            for rank, (score, terms, buffer_index, det) in enumerate(scored):
                if kept >= self.top_k:
                    break
                if score < self.best_frame_cfg.min_quality_score:
                    summary.reject("below_min_quality_score")
                    continue
                frame_id, capture_time, frame_obj, siblings = buffered[buffer_index]
                image = image_cache[buffer_index]
                height, width = image.shape[:2]
                padded = padded_box(
                    det.box, self.crop_cfg.padding_fraction, self.crop_cfg.min_padding_px
                )
                visible = visible_fraction_of(padded, width, height)
                if visible < self.best_frame_cfg.min_visible_fraction:
                    summary.reject("crop_not_visible_enough")
                    continue
                overlap = max(
                    (det.box.iou(other.box) for other in siblings if other is not det),
                    default=0.0,
                )
                if overlap > self.best_frame_cfg.max_overlap_iou:
                    summary.reject("crop_overlaps_neighbour")
                    continue

                crop, record = self._extract(
                    track_id=track_id,
                    detection=det,
                    frame_id=frame_id,
                    capture_time_s=capture_time,
                    image=image,
                    padded=padded,
                    score=score,
                    terms=terms,
                    rank=rank,
                    visible_fraction=visible,
                    max_overlap_iou=overlap,
                    mean_score=mean_score,
                    siblings=siblings,
                )

                track_key = f"{group_id}:{track_id}" if group_id else str(track_id)
                path = self._write_crop(out_dir, track_key, frame_id, crop)
                labels = (
                    dict(morphology_labels[track_id])
                    if morphology_labels and track_id in morphology_labels
                    else None
                )
                extracted = ExtractedCrop(
                    track_key=track_key,
                    track_id=int(track_id),
                    frame_id=int(frame_id),
                    path=path,
                    record=record.to_json_dict(),
                    morphology=labels,
                    group_id=group_id,
                )
                summary.crops.append(extracted)
                if labels:
                    summary.n_labelled += 1
                index_records.append(extracted.to_json_dict())
                kept += 1

            if kept == 0:
                summary.reject("track_produced_no_crop")

        if write_index:
            index_path = out_dir / "index.jsonl"
            with index_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "header",
                            "aspects": list(MORPHOLOGY_ASPECTS),
                            "crop_config": self.extractor.describe(),
                            "best_frame_config": self.best_frame_cfg.model_dump(),
                            "group_id": group_id,
                            "labels_present": summary.n_labelled > 0,
                            "note": (
                                "morphology is null unless the source dataset carried "
                                "per-sperm morphology annotation; labels are never "
                                "inferred from boxes, motion or sample-level statistics"
                            ),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                for record in index_records:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return summary

    # -------------------------------------------------------------- internal

    def _image_for(
        self, buffer_index: int, frame_obj: Any, cache: dict[int, np.ndarray]
    ) -> np.ndarray | None:
        """Load (and cache) the image for one buffered frame."""
        if buffer_index in cache:
            return cache[buffer_index]
        image = getattr(frame_obj, "image", None)
        if image is None:
            path = getattr(frame_obj, "image_path", None)
            if path is None:
                return None
            try:
                image = self.load_image(path)
            except (FileNotFoundError, OSError):
                return None
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        cache[buffer_index] = array
        return array

    def _extract(
        self,
        *,
        track_id: int,
        detection: Detection,
        frame_id: int,
        capture_time_s: float,
        image: np.ndarray,
        padded: Any,
        score: float,
        terms: Mapping[str, float],
        rank: int,
        visible_fraction: float,
        max_overlap_iou: float,
        mean_score: float,
        siblings: Sequence[Detection],
    ) -> tuple[np.ndarray, Any]:
        """Assemble the pipeline objects the shared extractor expects and call it."""
        track = TrackRecord(track_id=int(track_id))
        track.mean_score = mean_score

        candidate = CandidateFrame(
            frame_id=int(frame_id),
            capture_time_s=float(capture_time_s),
            box=detection.box,
            padded_box=padded,
            score=float(score),
            terms=dict(terms),
            detector_score=float(detection.score),
            track_confidence=float(mean_score),
            max_overlap_iou=float(max_overlap_iou),
            visible_fraction=float(visible_fraction),
            rank=int(rank),
        )
        packet = FramePacket(
            frame_id=int(frame_id),
            image=image,
            capture_time_s=float(capture_time_s),
            # Offline extraction from stored frames: the honest timestamp
            # provenance is the container, not a hardware tick.
            timestamp_source=TimestampSource.CONTAINER_PTS,
            source_kind=SourceKind.VIDEO,
        )
        neighbours = [d for d in siblings if d is not detection]
        return self.extractor.extract(track, candidate, packet, neighbours=neighbours)

    @staticmethod
    def _write_crop(out_dir: Path, track_key: str, frame_id: int, crop: np.ndarray) -> Path:
        """Write one crop.

        ``uint8``/``uint16`` crops go to PNG (lossless, viewable, and a reviewer
        can open one). Normalised float crops go to ``.npy``, because writing a
        float32 z-scored crop to PNG would quantise it back to 8 bits and undo
        the normalisation the config asked for.
        """
        safe = track_key.replace(":", "_").replace("/", "_")
        directory = out_dir / "crops"
        directory.mkdir(parents=True, exist_ok=True)
        if crop.dtype in (np.uint8, np.uint16):
            path = directory / f"{safe}_f{frame_id:06d}.png"
            if not cv2.imwrite(str(path), crop):
                raise OSError(f"failed to write crop: {path}")
            return path
        path = directory / f"{safe}_f{frame_id:06d}.npy"
        np.save(path, crop)
        return path


def _detections_of(frame: Any) -> list[Detection]:
    """Pull detections off a frame-like object.

    Two shapes exist among this package's adapters and both are legitimate:
    :class:`~datasets.adapters.visem_tracking.FrameAnnotation` exposes
    ``detections`` as a list attribute, while
    :class:`~datasets.adapters.device.FrameRecord` exposes ``detections()`` as a
    method that converts its stored ``ObjectRecord``s. Duck-typing here keeps
    this converter free of any import from the adapter package, which is what
    keeps the import graph acyclic.
    """
    value = getattr(frame, "detections", None)
    if value is None:
        raise TypeError(
            f"{type(frame).__name__} has no 'detections'; a frame-like object must "
            "expose either a detections sequence or a detections() method returning "
            "Detection objects"
        )
    resolved = value() if callable(value) else value
    return list(resolved)
