"""VISEM-Tracking: the only public dataset here with per-sperm boxes and track IDs.

Source: Zenodo record 7293726 -- Thambawita et al., "VISEM-Tracking: a human
spermatozoa tracking dataset", Scientific Data 10:260 (2023), arXiv:2212.02842.
Licence CC BY 4.0, the only permissively-licensed set in this repository.

What it is
----------
20 annotated 30-second clips from 20 patients at 640x480, **45-50 FPS and not
uniform across videos**; 29,196 annotated frames and 656,334 bounding boxes,
over 1,121 unique sperm track IDs, 20 cluster IDs and 35 pinhead IDs. Three
classes: ``0 = sperm``, ``1 = cluster``, ``2 = small or pinhead``. Official
split is 16 train / 4 validation **by video**, with validation videos
82, 60, 54, 52. There is no official test split -- reporting a "test" number on
this dataset means you invented the split, and you must say so.

Two file-format traps, both silent
----------------------------------
1. ``labels/`` lines are plain YOLO: ``class x_center y_center width height``.
   ``labels_ftid/`` lines put the **tracking ID first**:
   ``sperm_id class x_center y_center width height``. Parse the second with the
   first one's field order and you get the track ID as the class and the class
   as the track ID -- 1,121 "classes" and three "tracks", which fails loudly, or
   worse, quietly trains a detector whose class head is fitting an arbitrary
   identifier. :func:`parse_labels_ftid_line` is the only place this order is
   encoded, and :meth:`VisemTrackingAdapter.validate` cross-checks that every
   parsed class lands in ``{0, 1, 2}`` rather than trusting the comment.

2. Coordinates are **normalised** to ``[0, 1]``. Everything downstream of this
   package works in absolute pixels
   (:class:`~sperm_sorting.schemas.detection.BoundingBox`), so conversion
   happens exactly once, here, using the frame's real size.

Known quirks, all real, all detected by :meth:`quirk_report`
------------------------------------------------------------
* ``video_23`` has 174 frames containing no sperm at all. A pipeline that
  assumes every annotated frame has at least one box will divide by zero on
  those; a detector trained without them never learns what an empty field looks
  like and hallucinates on it.
* Frame counts differ between videos (``video_35`` and ``video_52`` have 1440,
  ``video_82`` has 1500), so "frames per video" is not a constant and any code
  that indexes by a global frame number across videos is wrong.
* Boxes concentrate in the **upper-left** of the frame. That is a genuine
  spatial prior a detector will happily overfit; it will then fail on device
  data where the sperm are wherever the flow puts them. :meth:`quirk_report`
  measures the mean normalised centre so the bias is a number, not a rumour.

The published YOLOv5l baseline reaches mAP@0.5 = 0.2231. That is not a weak
baseline; it is what tiny, dense, low-contrast objects look like. Treat any
substantially higher number as evidence of a leaked split until proven
otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from sperm_sorting.errors import DatasetValidationError
from sperm_sorting.schemas.detection import BoundingBox, Detection

from ..validators.integrity import CheckStatus, ValidationReport, check_non_empty
from .base import CaptureConditions, DatasetAdapter, DatasetInfo

__all__ = [
    "CLASS_NAMES",
    "OFFICIAL_VAL_VIDEO_IDS",
    "FrameAnnotation",
    "VisemTrackingAdapter",
    "parse_labels_ftid_line",
    "parse_labels_line",
    "yolo_to_box",
]

#: Class index -> name, exactly as published.
CLASS_NAMES: Final[dict[int, str]] = {
    0: "sperm",
    1: "cluster",
    2: "small_or_pinhead",
}

#: The four validation video IDs of the official 16/4 split.
OFFICIAL_VAL_VIDEO_IDS: Final[tuple[int, ...]] = (52, 54, 60, 82)

#: Number of annotated videos in the release.
EXPECTED_N_VIDEOS: Final[int] = 20

#: Declared frame size. Used only when no image is present to measure.
DECLARED_FRAME_SIZE: Final[tuple[int, int]] = (640, 480)

_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp")

#: Trailing integer of a filename stem is the frame index (``11_frame_0.jpg``,
#: ``0000123.jpg``, ``frame_7.png`` all work). Anchored at the end so a video id
#: embedded in the prefix is never mistaken for the frame number.
_FRAME_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)$")

#: Leading integer of a directory name is the video id (``52``, ``video_52``).
_VIDEO_ID_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)")


# ==========================================================================
# Line parsing -- the field-order trap lives here and nowhere else
# ==========================================================================


def parse_labels_line(line: str) -> tuple[int, float, float, float, float]:
    """Parse one ``labels/`` line: ``class x_center y_center width height``.

    Returns the class index and the four YOLO-normalised floats.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(
            f"expected 5 fields (class x_center y_center width height) in a labels/ "
            f"line, got {len(parts)}: {line.strip()!r}"
        )
    return (int(float(parts[0])), *(float(p) for p in parts[1:5]))  # type: ignore[return-value]


def parse_labels_ftid_line(line: str) -> tuple[int, int, float, float, float, float]:
    """Parse one ``labels_ftid/`` line.

    The upstream field order is::

        sperm_id  class  x_center  y_center  width  height
        ^^^^^^^^  ^^^^^
        ID FIRST, THEN CLASS -- the opposite of every plain YOLO file.

    This is the single most expensive thing to get wrong in this dataset, and it
    fails silently: both fields are small non-negative integers, so a swapped
    parse produces a syntactically valid annotation set in which the "class" is
    a track identifier ranging over 1,121 values and the "track id" is one of
    three classes. Nothing downstream can detect that on its own.

    Returns
    -------
    ``(track_id, class_id, x_center, y_center, width, height)``

    Raises
    ------
    ValueError
        On the wrong field count. A 5-field line is *not* silently accepted as
        a plain-YOLO line: the caller asked for the ftid format, and quietly
        returning a fabricated track id would defeat the point of the check.
    """
    parts = line.split()
    if len(parts) != 6:
        raise ValueError(
            "expected 6 fields (sperm_id class x_center y_center width height) in a "
            f"labels_ftid/ line, got {len(parts)}: {line.strip()!r}. Note the tracking "
            "id comes FIRST, before the class."
        )
    return (
        int(float(parts[0])),  # sperm_id  -- first
        int(float(parts[1])),  # class     -- second
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
        float(parts[5]),
    )


def yolo_to_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    """YOLO-normalised centre/size -> absolute-pixel
    :class:`~sperm_sorting.schemas.detection.BoundingBox`.

    The box is **not** clipped to the frame. VISEM-Tracking boxes occasionally
    extend a fraction of a pixel past the border, and clipping here would change
    the box the annotator drew; whether to clip is a decision for the consumer
    (the crop extractor clips, the IoU metric should not).
    """
    w = float(width) * frame_width
    h = float(height) * frame_height
    cx = float(x_center) * frame_width
    cy = float(y_center) * frame_height
    return BoundingBox(cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


# ==========================================================================
# Frame records
# ==========================================================================


@dataclass(slots=True)
class FrameAnnotation:
    """Every annotation for one frame of one video, in absolute pixels."""

    video_id: int
    frame_id: int
    #: ``None`` when only the label files were downloaded.
    image_path: Path | None
    frame_width: int
    frame_height: int
    detections: list[Detection] = field(default_factory=list)
    #: True when track IDs came from ``labels_ftid/``; False when only
    #: ``labels/`` was available and every ``track_id`` is therefore ``None``.
    has_track_ids: bool = False

    def of_class(self, class_id: int) -> list[Detection]:
        return [d for d in self.detections if d.class_id == class_id]

    @property
    def n_sperm(self) -> int:
        """Boxes of class 0 only -- clusters and pinheads are not single sperm."""
        return sum(1 for d in self.detections if d.class_id == 0)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "image_path": str(self.image_path) if self.image_path else None,
            "frame_size": [self.frame_width, self.frame_height],
            "has_track_ids": self.has_track_ids,
            "detections": [d.to_json_dict() for d in self.detections],
        }


@dataclass(slots=True)
class _VideoLayout:
    """Where one video's pieces actually live on this disk."""

    video_id: int
    directory: Path
    labels_dir: Path | None
    labels_ftid_dir: Path | None
    images_dir: Path | None
    video_file: Path | None


# ==========================================================================
# Adapter
# ==========================================================================


class VisemTrackingAdapter(DatasetAdapter):
    """Reader for ``VISEM_Tracking_Train_v4/Train/<video_id>/``.

    Parameters
    ----------
    root
        Any of: the directory containing ``VISEM_Tracking_Train_v4``, that
        directory itself, or the ``Train`` folder inside it. All three are what
        somebody reasonably means by "the VISEM-Tracking folder".
    require_present
        See :class:`~datasets.adapters.base.DatasetAdapter`.
    frame_size
        Override the frame size instead of measuring it from an image. Only
        needed for a labels-only copy; when images are present their real size
        is measured once per video and cached.
    """

    info = DatasetInfo(
        name="visem_tracking",
        title="VISEM-Tracking",
        url="https://zenodo.org/records/7293726",
        license_key="visem_tracking",
        annotation_level="per-frame bounding boxes + per-sperm tracking IDs (3 classes)",
        approximate_size="~35 GB extracted (20 videos, 29,196 annotated frames)",
        capture=CaptureConditions(
            objective_magnification=None,
            total_magnification=None,
            contrast_mode="brightfield, unstained wet preparation",
            stained=False,
            camera="UEye UI-2210C on an Olympus CX31 (per the VISEM lineage)",
            fps_range=(45.0, 50.0),
            fps_uniform=False,
            resolution=(640, 480),
            um_per_px=None,
            notes=(
                "20 patients, one 30-second clip each. Frame rate varies between "
                "videos (45-50 FPS) and is NOT constant across the release, so "
                "per-frame velocities must use each video's own rate; frame counts "
                "also differ (1440 vs 1500)."
            ),
        ),
        domain_shift_notes=[
            "640x480 across the whole field: a sperm head is only a handful of "
            "pixels across. A device camera at higher resolution shows head "
            "structure this dataset physically cannot contain, so a detector "
            "trained here is tuned for a blob, not a shape.",
            "Frame rate 45-50 FPS and non-uniform. Displacement per frame -- the "
            "feature every tracker's motion model is built on -- differs by ~10% "
            "between videos here and by more against a device running faster.",
            "Boxes concentrate in the upper-left of the frame. A detector can and "
            "will learn that prior; measure it with quirk_report() before believing "
            "any localisation metric.",
            "No morphology labels at all. This dataset can train detection and "
            "tracking; it cannot train or validate the four-aspect morphology head.",
            "Free-swimming cells in a static chamber, not cells in a flow. Track "
            "kinematics carry no bulk-transport component, so flow correction "
            "cannot be validated on this data.",
            "Published YOLOv5l baseline is mAP@0.5 = 0.2231. A much higher score on "
            "a self-made split is evidence of leakage, not of a better model.",
        ],
        expected_layout=(
            "  <root>/VISEM_Tracking_Train_v4/Train/<video_id>/\n"
            "      <video_id>.mp4\n"
            "      images/                 (extracted frames; may sit directly in the\n"
            "                               video folder instead -- both are handled)\n"
            "      labels/<frame>.txt      class x_center y_center width height\n"
            "      labels_ftid/<frame>.txt sperm_id class x_center y_center width height\n"
            "  (<root> may also be VISEM_Tracking_Train_v4/ or its Train/ folder)"
        ),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        require_present: bool = True,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        self._frame_size_override = frame_size
        self._layouts: dict[int, _VideoLayout] | None = None
        self._frame_size_cache: dict[int, tuple[int, int]] = {}
        #: Directory names under the root that carried no numeric video id.
        self._unparsed_dirs: list[str] = []
        super().__init__(root, require_present=require_present)

    # ------------------------------------------------------------ discovery

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """Resolve to the ``Train`` directory, whichever level was handed in.

        A candidate only counts when it actually holds a numerically-named video
        folder. Accepting any existing directory would let ``VisemTrackingAdapter(".")``
        succeed and then report zero videos, which reads as an empty dataset
        rather than as a wrong path.
        """
        for candidate in (
            given / "VISEM_Tracking_Train_v4" / "Train",
            given / "Train",
            given,
        ):
            if candidate.is_dir() and any(
                entry.is_dir() and _VIDEO_ID_RE.search(entry.name)
                for entry in candidate.iterdir()
            ):
                return candidate
        return None

    def _discover(self) -> dict[int, _VideoLayout]:
        """Map video id -> on-disk layout, cached.

        Sniffs where the frames live because the release has been repackaged
        more than once: an ``images/`` subfolder and loose frames in the video
        folder are both in the wild, and guessing wrong produces "0 frames"
        rather than an error.
        """
        if self._layouts is not None:
            return self._layouts

        layouts: dict[int, _VideoLayout] = {}
        unparsed: list[str] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            match = _VIDEO_ID_RE.search(entry.name)
            if match is None:
                unparsed.append(entry.name)
                continue
            video_id = int(match.group(1))
            labels = entry / "labels"
            labels_ftid = entry / "labels_ftid"
            images = entry / "images"
            if not images.is_dir():
                images = entry if any(self._image_files(entry)) else None  # type: ignore[assignment]
            videos = sorted(entry.glob("*.mp4"))
            layouts[video_id] = _VideoLayout(
                video_id=video_id,
                directory=entry,
                labels_dir=labels if labels.is_dir() else None,
                labels_ftid_dir=labels_ftid if labels_ftid.is_dir() else None,
                images_dir=images if images is not None and images.is_dir() else None,
                video_file=videos[0] if videos else None,
            )
        if not layouts:
            raise self.not_found_error(self.root)
        # Not fatal, but validate() reports it: a skipped directory is a
        # silently smaller dataset.
        self._unparsed_dirs = unparsed
        self._layouts = layouts
        return layouts

    @staticmethod
    def _image_files(directory: Path) -> Iterator[Path]:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                yield path

    def videos(self) -> list[int]:
        """Video IDs present on disk, sorted."""
        return sorted(self._discover())

    def video_dir(self, video_id: int) -> Path:
        layouts = self._discover()
        if int(video_id) not in layouts:
            raise DatasetValidationError(
                f"VISEM-Tracking: video {video_id} is not present under {self.root}. "
                f"Present: {sorted(layouts)}"
            )
        return layouts[int(video_id)].directory

    # -------------------------------------------------------------- sizing

    def frame_size(self, video_id: int) -> tuple[int, int]:
        """``(width, height)`` for one video.

        Measured from the first image when images are present -- an explicit
        override wins, and the declared 640x480 is the last resort, recorded as
        such in :meth:`validate` so that a labels-only copy never silently
        pretends it measured something.
        """
        video_id = int(video_id)
        if self._frame_size_override is not None:
            return self._frame_size_override
        if video_id in self._frame_size_cache:
            return self._frame_size_cache[video_id]

        layout = self._discover()[video_id]
        size = DECLARED_FRAME_SIZE
        if layout.images_dir is not None:
            for path in self._image_files(layout.images_dir):
                measured = _read_image_size(path)
                if measured is not None:
                    size = measured
                break
        self._frame_size_cache[video_id] = size
        return size

    # ------------------------------------------------------------- indexing

    def frame_ids(self, video_id: int, *, with_track_ids: bool = True) -> list[int]:
        """Sorted frame indices that have an annotation file for this video."""
        layout = self._discover()[int(video_id)]
        directory = layout.labels_ftid_dir if with_track_ids else layout.labels_dir
        if directory is None:
            kind = "labels_ftid" if with_track_ids else "labels"
            raise DatasetValidationError(
                f"VISEM-Tracking: video {video_id} has no {kind}/ directory under "
                f"{layout.directory}. Re-download from {self.info.url}."
            )
        out: list[int] = []
        for path in directory.glob("*.txt"):
            frame = _frame_number(path.stem)
            if frame is not None:
                out.append(frame)
        return sorted(out)

    def label_file(self, video_id: int, frame_id: int, *, with_track_ids: bool) -> Path | None:
        """Locate one frame's annotation file, tolerating stem conventions.

        Stems in the wild include ``0``, ``00000``, ``11_frame_0`` and
        ``video_11_frame_0``. Rather than encode a convention that may not be
        the one on this disk, an index of stem -> path is built once per video
        and matched on the trailing integer.
        """
        layout = self._discover()[int(video_id)]
        directory = layout.labels_ftid_dir if with_track_ids else layout.labels_dir
        if directory is None:
            return None
        cache_attr = f"_index_{'ftid' if with_track_ids else 'plain'}_{video_id}"
        index: dict[int, Path] | None = getattr(self, cache_attr, None)
        if index is None:
            index = {}
            for path in directory.glob("*.txt"):
                frame = _frame_number(path.stem)
                if frame is not None:
                    index[frame] = path
            setattr(self, cache_attr, index)
        return index.get(int(frame_id))

    def image_file(self, video_id: int, frame_id: int) -> Path | None:
        """Locate one frame's image, or ``None`` for a labels-only copy."""
        layout = self._discover()[int(video_id)]
        if layout.images_dir is None:
            return None
        cache_attr = f"_index_images_{video_id}"
        index: dict[int, Path] | None = getattr(self, cache_attr, None)
        if index is None:
            index = {}
            for path in self._image_files(layout.images_dir):
                frame = _frame_number(path.stem)
                if frame is not None:
                    index[frame] = path
            setattr(self, cache_attr, index)
        return index.get(int(frame_id))

    # ---------------------------------------------------------------- reading

    def frame(
        self, video_id: int, frame_id: int, *, with_track_ids: bool = True
    ) -> FrameAnnotation:
        """Read one frame's annotations, converted to absolute pixels.

        ``with_track_ids=True`` reads ``labels_ftid/`` (id first, then class);
        ``False`` reads ``labels/``. Falling back from one to the other is
        deliberately *not* done: a caller who asked for track IDs and silently
        got ``None`` would build a tracking benchmark with no ground truth.
        """
        video_id, frame_id = int(video_id), int(frame_id)
        width, height = self.frame_size(video_id)
        path = self.label_file(video_id, frame_id, with_track_ids=with_track_ids)
        if path is None:
            kind = "labels_ftid" if with_track_ids else "labels"
            raise DatasetValidationError(
                f"VISEM-Tracking: no {kind} annotation for video {video_id} frame "
                f"{frame_id} under {self.video_dir(video_id)}"
            )

        detections: list[Detection] = []
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                if with_track_ids:
                    track_id, class_id, cx, cy, bw, bh = parse_labels_ftid_line(line)
                else:
                    class_id, cx, cy, bw, bh = parse_labels_line(line)
                    track_id = None  # type: ignore[assignment]
            except ValueError as exc:
                raise DatasetValidationError(f"{path}:{lineno}: {exc}") from exc
            detections.append(
                Detection(
                    frame_id=frame_id,
                    box=yolo_to_box(cx, cy, bw, bh, width, height),
                    # Ground truth, so the score is 1.0 by definition. It is
                    # carried anyway because every metric and converter in this
                    # package takes Detection, and a GT box with score 0 would
                    # be filtered out by any confidence threshold on the way.
                    score=1.0,
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, f"unknown_{class_id}"),
                    track_id=track_id if with_track_ids else None,
                    meta={"video_id": video_id, "source": path.name},
                )
            )
        return FrameAnnotation(
            video_id=video_id,
            frame_id=frame_id,
            image_path=self.image_file(video_id, frame_id),
            frame_width=width,
            frame_height=height,
            detections=detections,
            has_track_ids=with_track_ids,
        )

    def iter_frames(
        self, video_id: int, *, with_track_ids: bool = True
    ) -> Iterator[FrameAnnotation]:
        """Yield every annotated frame of one video, in frame order."""
        for frame_id in self.frame_ids(video_id, with_track_ids=with_track_ids):
            yield self.frame(video_id, frame_id, with_track_ids=with_track_ids)

    def iter_all_frames(
        self, video_ids: Iterable[int] | None = None, *, with_track_ids: bool = True
    ) -> Iterator[FrameAnnotation]:
        """Yield frames across several videos (default: all of them)."""
        for video_id in sorted(video_ids) if video_ids is not None else self.videos():
            yield from self.iter_frames(video_id, with_track_ids=with_track_ids)

    def tracks(self, video_id: int) -> dict[int, list[Detection]]:
        """Reconstruct per-track detection sequences from ``labels_ftid/``.

        Returns ``{track_id: [Detection, ...]}`` sorted by frame within each
        track, which is the shape MOT-style evaluation and
        :mod:`datasets.converters.to_mot_format` want.

        Returns :class:`~sperm_sorting.schemas.detection.Detection` objects
        rather than :class:`~sperm_sorting.schemas.track.TrackRecord` on
        purpose: a ``TrackRecord`` carries runtime state (eligibility, motility
        class, morphology verdict, the shot it was gated into) that ground truth
        simply does not have, and populating those fields with defaults would
        manufacture a record that looks like a pipeline output and is not one.

        Note that IDs are only unique **within a video**: track 3 in video 11 and
        track 3 in video 52 are different sperm. Callers merging videos must
        namespace the IDs themselves -- see
        :meth:`global_track_key`.
        """
        out: dict[int, list[Detection]] = {}
        for annotation in self.iter_frames(video_id, with_track_ids=True):
            for det in annotation.detections:
                if det.track_id is None:
                    continue
                out.setdefault(int(det.track_id), []).append(det)
        for detections in out.values():
            detections.sort(key=lambda d: d.frame_id)
        return dict(sorted(out.items()))

    @staticmethod
    def global_track_key(video_id: int, track_id: int) -> str:
        """Namespace a track ID by its video. IDs restart per video."""
        return f"{int(video_id)}:{int(track_id)}"

    # ---------------------------------------------------------------- splits

    def splits(self) -> list[str]:
        """``["train", "val"]``. There is **no official test split**."""
        return ["train", "val"]

    def official_split(self, *, restrict_to_present: bool = False) -> dict[str, list[int]]:
        """The published 16/4 split by video.

        Validation is videos ``52, 54, 60, 82``
        (:data:`OFFICIAL_VAL_VIDEO_IDS`); training is every other video.

        Parameters
        ----------
        restrict_to_present
            By default the validation list is the **published constant**,
            regardless of what is on this disk, so that a partial download
            cannot silently produce a different (and incomparable) split that
            still calls itself "official". Set True to intersect with the videos
            actually present -- useful for a smoke test, never for a reported
            number. :meth:`validate` flags a copy that is missing any of them.

        There is deliberately no ``test`` key. VISEM-Tracking publishes no test
        split, so any test number is on a split you invented and must describe.
        """
        present = set(self.videos()) if self.available else set()
        val = list(OFFICIAL_VAL_VIDEO_IDS)
        if restrict_to_present:
            val = [v for v in val if v in present]
        train = sorted(present - set(OFFICIAL_VAL_VIDEO_IDS))
        return {"train": train, "val": sorted(val)}

    def __len__(self) -> int:
        """Total annotated frames across every video present."""
        return sum(len(self.frame_ids(v, with_track_ids=True)) for v in self.videos())

    # ---------------------------------------------------------------- quirks

    def quirk_report(self, video_ids: Iterable[int] | None = None) -> dict[str, Any]:
        """Measure the documented quirks on *this* copy.

        Reports, per video: annotated frame count, number of frames with no
        class-0 box, per-class box counts, unique track IDs, and the mean
        normalised box centre (the upper-left concentration).

        This is a full pass over every label file, so it is slow on the complete
        release (29,196 files). It is worth running once per copy: the numbers it
        prints are the ones that explain a surprising training curve later.
        """
        videos = sorted(video_ids) if video_ids is not None else self.videos()
        per_video: dict[int, dict[str, Any]] = {}
        centres: list[tuple[float, float]] = []
        total_boxes = 0

        for video_id in videos:
            width, height = self.frame_size(video_id)
            n_empty = 0
            class_counts: dict[int, int] = {}
            track_ids: set[int] = set()
            frame_ids = self.frame_ids(video_id, with_track_ids=True)
            for frame_id in frame_ids:
                annotation = self.frame(video_id, frame_id, with_track_ids=True)
                if annotation.n_sperm == 0:
                    n_empty += 1
                for det in annotation.detections:
                    class_counts[det.class_id] = class_counts.get(det.class_id, 0) + 1
                    if det.track_id is not None:
                        track_ids.add(int(det.track_id))
                    cx, cy = det.box.center
                    centres.append((cx / width, cy / height))
                    total_boxes += 1
            per_video[video_id] = {
                "n_frames": len(frame_ids),
                "n_frames_without_sperm": n_empty,
                "class_counts": {CLASS_NAMES.get(k, str(k)): v for k, v in sorted(class_counts.items())},
                "n_unique_track_ids": len(track_ids),
                "frame_size": [width, height],
            }

        centre_array = np.asarray(centres, dtype=np.float64) if centres else np.zeros((0, 2))
        spatial = (
            {
                "mean_normalised_centre": [
                    float(centre_array[:, 0].mean()),
                    float(centre_array[:, 1].mean()),
                ],
                "fraction_in_upper_left_quadrant": float(
                    np.count_nonzero((centre_array[:, 0] < 0.5) & (centre_array[:, 1] < 0.5))
                    / len(centre_array)
                ),
            }
            if len(centre_array)
            else {"mean_normalised_centre": None, "fraction_in_upper_left_quadrant": None}
        )

        frame_counts = {v: per_video[v]["n_frames"] for v in videos}
        return {
            "n_videos": len(videos),
            "n_boxes": total_boxes,
            "per_video": per_video,
            "frame_counts_uniform": len(set(frame_counts.values())) <= 1,
            "frame_counts": frame_counts,
            "spatial_prior": spatial,
        }

    # ------------------------------------------------------------ validation

    def validate(self, *, deep: bool = False) -> ValidationReport:
        """Structural checks; ``deep=True`` also runs :meth:`quirk_report`.

        The shallow pass reads one annotation file per video, which is enough to
        catch the field-order trap, a labels-only copy, and a missing split
        video. The deep pass reads all 29,196 and is the one that reports the
        empty-sperm frames and the spatial prior.
        """
        report = self._new_report()
        layouts = self._discover()
        report.checks.append(check_non_empty(len(layouts), name="videos", what="video folders"))

        unparsed = self._unparsed_dirs
        if unparsed:
            report.add(
                "layout:unparsed_dirs",
                CheckStatus.WARN,
                f"{len(unparsed)} directory name(s) under {self.root} carry no numeric "
                f"video id and were skipped: {unparsed[:10]}",
            )

        if len(layouts) != EXPECTED_N_VIDEOS:
            report.add(
                "completeness:videos",
                CheckStatus.FAIL,
                f"found {len(layouts)} videos, the release has {EXPECTED_N_VIDEOS}. "
                "Metrics from an incomplete copy are not comparable with the published "
                f"baseline. Present: {sorted(layouts)}",
                found=sorted(layouts),
            )
        else:
            report.add(
                "completeness:videos",
                CheckStatus.PASS,
                f"all {EXPECTED_N_VIDEOS} videos present",
            )

        missing_val = [v for v in OFFICIAL_VAL_VIDEO_IDS if v not in layouts]
        if missing_val:
            report.add(
                "split:official_val_present",
                CheckStatus.FAIL,
                f"official validation video(s) {missing_val} are absent, so the official "
                "16/4 split cannot be reproduced on this copy",
                missing=missing_val,
            )
        else:
            report.add(
                "split:official_val_present",
                CheckStatus.PASS,
                f"official validation videos {list(OFFICIAL_VAL_VIDEO_IDS)} all present",
            )

        total_frames = 0
        for video_id, layout in sorted(layouts.items()):
            for label, directory in (
                ("labels", layout.labels_dir),
                ("labels_ftid", layout.labels_ftid_dir),
            ):
                if directory is None:
                    report.add(
                        f"layout:{label}:{video_id}",
                        CheckStatus.FAIL,
                        f"video {video_id}: no {label}/ directory under {layout.directory}",
                    )
            if layout.images_dir is None:
                report.add(
                    f"layout:images:{video_id}",
                    CheckStatus.WARN,
                    f"video {video_id}: no extracted frames found; boxes will be converted "
                    f"using the declared frame size {DECLARED_FRAME_SIZE}, which was not "
                    "measured on this copy",
                )
            if layout.video_file is None:
                report.add(
                    f"layout:mp4:{video_id}",
                    CheckStatus.WARN,
                    f"video {video_id}: no .mp4 found under {layout.directory}",
                )

            if layout.labels_ftid_dir is not None:
                frame_ids = self.frame_ids(video_id, with_track_ids=True)
                total_frames += len(frame_ids)
                if not frame_ids:
                    report.add(
                        f"annotations:{video_id}",
                        CheckStatus.FAIL,
                        f"video {video_id}: labels_ftid/ contains no parseable .txt files",
                    )
                else:
                    self._validate_field_order(report, video_id, frame_ids[0])

        report.context["n_annotated_frames"] = total_frames
        report.add(
            "annotations:total",
            CheckStatus.PASS if total_frames else CheckStatus.FAIL,
            f"{total_frames} annotated frames found "
            f"(the full release has 29,196 across 20 videos)",
            n_frames=total_frames,
        )

        report.add(
            "split:no_official_test",
            CheckStatus.UNVERIFIABLE,
            "VISEM-Tracking publishes no test split. Any 'test' number on this dataset "
            "comes from a split someone invented; it must be described alongside the "
            "number and cannot be compared with the published baseline.",
        )

        if deep:
            quirks = self.quirk_report()
            report.context["quirks"] = quirks
            empty = {
                v: d["n_frames_without_sperm"]
                for v, d in quirks["per_video"].items()
                if d["n_frames_without_sperm"]
            }
            report.add(
                "quirk:frames_without_sperm",
                CheckStatus.WARN if empty else CheckStatus.PASS,
                (
                    f"frames with no class-0 box, per video: {empty}. These are real "
                    "annotations (video_23 has 174 upstream); keep them in training so the "
                    "detector learns what an empty field looks like, and guard any "
                    "per-frame average against division by zero."
                )
                if empty
                else "every annotated frame contains at least one sperm box",
                per_video=empty,
            )
            if not quirks["frame_counts_uniform"]:
                report.add(
                    "quirk:frame_counts",
                    CheckStatus.WARN,
                    f"frame counts differ between videos: {quirks['frame_counts']}. Do not "
                    "index frames by a global counter across videos.",
                )
            centre = quirks["spatial_prior"]["mean_normalised_centre"]
            if centre is not None:
                skewed = centre[0] < 0.45 or centre[1] < 0.45
                report.add(
                    "quirk:spatial_prior",
                    CheckStatus.WARN if skewed else CheckStatus.PASS,
                    f"mean normalised box centre is ({centre[0]:.3f}, {centre[1]:.3f}); "
                    f"{quirks['spatial_prior']['fraction_in_upper_left_quadrant']:.1%} of "
                    "boxes fall in the upper-left quadrant. A detector will learn this "
                    "prior and it does not transfer.",
                    **quirks["spatial_prior"],
                )

        return report

    def _validate_field_order(
        self, report: ValidationReport, video_id: int, frame_id: int
    ) -> None:
        """Cross-check the ``labels_ftid`` field order on one real file.

        The documented order is ``sperm_id class ...``. If it were the other way
        round, the second field would be a track identifier and would range far
        outside ``{0, 1, 2}``. Checking that every parsed class is a valid class
        turns a silent misparse into a named failure -- which is the whole reason
        this check exists rather than a comment saying "id comes first".
        """
        path = self.label_file(video_id, frame_id, with_track_ids=True)
        if path is None:  # pragma: no cover - guarded by the caller
            return
        name = f"format:labels_ftid:{video_id}"
        classes: set[int] = set()
        ids: set[int] = set()
        try:
            for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                track_id, class_id, cx, cy, bw, bh = parse_labels_ftid_line(line)
                classes.add(class_id)
                ids.add(track_id)
                for value, label in ((cx, "x_center"), (cy, "y_center"), (bw, "width"), (bh, "height")):
                    if not -0.01 <= value <= 1.01:
                        report.add(
                            name,
                            CheckStatus.FAIL,
                            f"{path}:{lineno}: {label}={value} is outside [0, 1]; "
                            "labels_ftid coordinates are YOLO-normalised, so a value "
                            "outside that range means the fields are being read in the "
                            "wrong order (expected: sperm_id class x y w h)",
                        )
                        return
        except ValueError as exc:
            report.add(name, CheckStatus.FAIL, f"{path}: {exc}")
            return

        bad = sorted(c for c in classes if c not in CLASS_NAMES)
        if bad:
            report.add(
                name,
                CheckStatus.FAIL,
                f"{path}: parsed class value(s) {bad} outside the published set "
                f"{sorted(CLASS_NAMES)}. Almost certainly the labels_ftid field order: "
                "the tracking id comes FIRST, then the class.",
                classes=sorted(classes),
            )
            return
        report.add(
            name,
            CheckStatus.PASS,
            f"video {video_id}: labels_ftid parses as (sperm_id, class, x, y, w, h); "
            f"classes {sorted(classes)}, {len(ids)} track id(s) in frame {frame_id}",
            classes=sorted(classes),
            n_track_ids=len(ids),
        )


# ==========================================================================
# helpers
# ==========================================================================


def _frame_number(stem: str) -> int | None:
    """Trailing integer of a filename stem, or ``None``."""
    match = _FRAME_NUMBER_RE.search(stem)
    return int(match.group(1)) if match else None


def _read_image_size(path: Path) -> tuple[int, int] | None:
    """``(width, height)`` of an image file, or ``None`` if it cannot be read.

    OpenCV is already a hard dependency of this project, so no new one is
    introduced; a failure to decode returns ``None`` and the caller falls back
    to the declared size *and says so*.
    """
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if image is None:
        return None
    return int(image.shape[1]), int(image.shape[0])
