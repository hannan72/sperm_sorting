"""Device captures: the annotation schema for data recorded on the instrument.

This is the only corpus that is actually in-domain, and the only one with no
third-party licence constraint. Everything public in this package exists to
bootstrap a model that this data then adapts; the domain-adaptation path
consumes exactly this format.

The format: JSON Lines, header first
------------------------------------
One file, one capture session::

    {"record_type": "header", "schema_version": "1.0.0", "capture": {...}, ...}
    {"record_type": "frame", "frame_id": 0, "boxes": [...], ...}
    {"record_type": "frame", "frame_id": 1, "boxes": [...], ...}

JSON Lines rather than one big JSON because a capture is appended to while it
runs: a partially-written JSONL file is still readable up to its last complete
line, whereas a truncated JSON array is not readable at all. Losing a session's
annotations to a power cut during a clinic day is not an acceptable failure mode.

**Why the capture metadata lives in a header rather than on every frame.** It is
a property of the session, not of the frame -- ``um_per_px``, the operator and
the sample id do not change between frame 0 and frame 1. Repeating them 30,000
times invites them to drift, and a file where frame 400 claims a different
``um_per_px`` from frame 399 is one nobody can interpret. The fields that
genuinely *can* drift within a session -- exposure, gain, temperature -- may be
overridden per frame, and :meth:`DeviceCapture.effective_capture` resolves the
override against the header, so a reader never has to know which it was.

Required versus optional capture metadata
-----------------------------------------
:data:`REQUIRED_CAPTURE_FIELDS` is short and every entry earns its place:

``sample_id``, ``operator``
    Provenance. Without them a capture cannot be grouped for a patient-level
    split (:func:`datasets.validators.leakage.patient_level_split`) and cannot be
    excluded when consent is withdrawn.
``um_per_px``
    Without it, no velocity from this data can be expressed in physical units,
    and the WHO motility thresholds this product decides on are in um/s. This
    mirrors :class:`sperm_sorting.errors.CalibrationError`: the system must never
    silently substitute pixel units for physical ones.
``frame_rate_hz``
    Every kinematic quantity divides by the frame interval.
``exposure_us``, ``gain_db``
    The two knobs that change image appearance most, and therefore the two you
    need when a model trained on Tuesday's captures underperforms on Friday's.

:data:`OPTIONAL_CAPTURE_FIELDS` holds ``temperature_c`` and the descriptive
fields. Temperature is optional because not every rig measures it -- but it is
*warned* about rather than ignored, because sperm motility is strongly
temperature-dependent and a motility comparison across captures at unknown
temperatures is not a comparison.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from sperm_sorting.constants import (
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    MORPHOLOGY_ASPECTS,
    SCHEMA_VERSION,
)
from sperm_sorting.errors import DatasetValidationError
from sperm_sorting.schemas.detection import BoundingBox, Detection

from ..validators.integrity import CheckStatus, ValidationReport, check_non_empty
from .base import CaptureConditions, DatasetAdapter, DatasetInfo

__all__ = [
    "OPTIONAL_CAPTURE_FIELDS",
    "REQUIRED_CAPTURE_FIELDS",
    "CaptureMetadata",
    "DeviceAnnotationWriter",
    "DeviceCapture",
    "DeviceDatasetAdapter",
    "FrameRecord",
    "ObjectRecord",
]

#: Capture metadata without which a session cannot be used. See the module
#: docstring for why each one is here.
REQUIRED_CAPTURE_FIELDS: Final[tuple[str, ...]] = (
    "sample_id",
    "operator",
    "um_per_px",
    "frame_rate_hz",
    "exposure_us",
    "gain_db",
)

#: Recorded when available; their absence is a warning, not a failure.
OPTIONAL_CAPTURE_FIELDS: Final[tuple[str, ...]] = (
    "temperature_c",
    "camera_model",
    "objective",
    "illumination",
    "channel_id",
    "flow_rate_ul_min",
    "notes",
)

#: Fields that may legitimately be overridden on an individual frame.
PER_FRAME_OVERRIDABLE: Final[tuple[str, ...]] = (
    "exposure_us",
    "gain_db",
    "temperature_c",
)

_ANNOTATION_SUFFIX: Final[str] = ".jsonl"


# ==========================================================================
# Records
# ==========================================================================


@dataclass(slots=True)
class CaptureMetadata:
    """Session-level capture conditions.

    Every required field is a plain attribute with no default, so an incomplete
    record fails at construction rather than at analysis time three weeks later.
    """

    sample_id: str
    operator: str
    um_per_px: float
    frame_rate_hz: float
    exposure_us: float
    gain_db: float
    temperature_c: float | None = None
    camera_model: str | None = None
    objective: str | None = None
    illumination: str | None = None
    channel_id: str | None = None
    flow_rate_ul_min: float | None = None
    notes: str = ""
    #: Anything device-specific that has no field yet. Never load-bearing.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sample_id": self.sample_id,
            "operator": self.operator,
            "um_per_px": float(self.um_per_px),
            "frame_rate_hz": float(self.frame_rate_hz),
            "exposure_us": float(self.exposure_us),
            "gain_db": float(self.gain_db),
            "temperature_c": self.temperature_c,
            "camera_model": self.camera_model,
            "objective": self.objective,
            "illumination": self.illumination,
            "channel_id": self.channel_id,
            "flow_rate_ul_min": self.flow_rate_ul_min,
            "notes": self.notes,
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> CaptureMetadata:
        missing = [f for f in REQUIRED_CAPTURE_FIELDS if payload.get(f) is None]
        if missing:
            raise DatasetValidationError(
                f"device capture metadata is missing required field(s): {missing}. "
                f"Required: {list(REQUIRED_CAPTURE_FIELDS)}. Present: "
                f"{sorted(k for k, v in payload.items() if v is not None)}."
            )
        known = set(REQUIRED_CAPTURE_FIELDS) | set(OPTIONAL_CAPTURE_FIELDS)
        return cls(
            sample_id=str(payload["sample_id"]),
            operator=str(payload["operator"]),
            um_per_px=float(payload["um_per_px"]),
            frame_rate_hz=float(payload["frame_rate_hz"]),
            exposure_us=float(payload["exposure_us"]),
            gain_db=float(payload["gain_db"]),
            temperature_c=_opt_float(payload.get("temperature_c")),
            camera_model=_opt_str(payload.get("camera_model")),
            objective=_opt_str(payload.get("objective")),
            illumination=_opt_str(payload.get("illumination")),
            channel_id=_opt_str(payload.get("channel_id")),
            flow_rate_ul_min=_opt_float(payload.get("flow_rate_ul_min")),
            notes=str(payload.get("notes", "")),
            extra={k: v for k, v in payload.items() if k not in known and k != "extra"}
            | dict(payload.get("extra", {}) or {}),
        )

    def to_capture_conditions(self) -> CaptureConditions:
        """Express this session in the same terms as the public datasets.

        This is what makes a domain-shift table computable:
        ``public.capture.differences_from(session.to_capture_conditions())``
        enumerates the gap field by field.
        """
        return CaptureConditions(
            objective_magnification=_magnification(self.objective),
            total_magnification=None,
            contrast_mode=self.illumination,
            stained=False,
            camera=self.camera_model,
            fps_range=(self.frame_rate_hz, self.frame_rate_hz),
            fps_uniform=True,
            resolution=None,
            um_per_px=self.um_per_px,
            notes=self.notes,
        )


@dataclass(slots=True)
class ObjectRecord:
    """One annotated object in one frame.

    ``morphology`` maps an aspect name to ``0`` (normal) / ``1`` (abnormal), the
    same encoding as MHSMA and the same one the training path consumes verbatim
    (:mod:`sperm_sorting.morphology.polarity`). An aspect that was **not
    assessed** must be absent from the mapping or ``None`` -- never 0. "Nobody
    looked" and "looked, and it was normal" are different facts, and collapsing
    them silently biases every prevalence computed from this data toward normal.
    """

    box: BoundingBox
    class_id: int = 0
    class_name: str = "sperm"
    track_id: int | None = None
    score: float = 1.0
    morphology: dict[str, int | None] = field(default_factory=dict)
    #: Free-form annotator notes ("out of focus", "overlapping neighbour").
    notes: str = ""

    def __post_init__(self) -> None:
        for aspect, value in self.morphology.items():
            if aspect not in MORPHOLOGY_ASPECTS:
                raise ValueError(
                    f"unknown morphology aspect {aspect!r}; expected one of "
                    f"{list(MORPHOLOGY_ASPECTS)}"
                )
            if value is not None and value not in (LABEL_NORMAL, LABEL_ABNORMAL):
                raise ValueError(
                    f"morphology[{aspect!r}] must be {LABEL_NORMAL} (normal), "
                    f"{LABEL_ABNORMAL} (abnormal) or None (not assessed), got {value!r}"
                )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "box_xyxy": list(self.box.as_xyxy()),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "score": float(self.score),
        }
        if self.morphology:
            out["morphology"] = dict(self.morphology)
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> ObjectRecord:
        x1, y1, x2, y2 = payload["box_xyxy"]
        return cls(
            box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
            class_id=int(payload.get("class_id", 0)),
            class_name=str(payload.get("class_name", "sperm")),
            track_id=None if payload.get("track_id") is None else int(payload["track_id"]),
            score=float(payload.get("score", 1.0)),
            morphology={
                str(k): (None if v is None else int(v))
                for k, v in dict(payload.get("morphology", {})).items()
            },
            notes=str(payload.get("notes", "")),
        )

    def to_detection(self, frame_id: int, capture_time_s: float = 0.0) -> Detection:
        """Convert to the unified internal detection format."""
        return Detection(
            frame_id=frame_id,
            box=self.box,
            score=float(self.score),
            class_id=self.class_id,
            class_name=self.class_name,
            capture_time_s=float(capture_time_s),
            track_id=self.track_id,
            meta={"morphology": dict(self.morphology)} if self.morphology else {},
        )


@dataclass(slots=True)
class FrameRecord:
    """One frame's annotations plus any per-frame capture overrides."""

    frame_id: int
    capture_time_s: float
    width: int
    height: int
    boxes: list[ObjectRecord] = field(default_factory=list)
    image_path: str | None = None
    #: Only the fields in :data:`PER_FRAME_OVERRIDABLE` are honoured.
    capture_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = sorted(set(self.capture_overrides) - set(PER_FRAME_OVERRIDABLE))
        if bad:
            raise ValueError(
                f"per-frame capture override(s) {bad} are not permitted; only "
                f"{list(PER_FRAME_OVERRIDABLE)} can change within a session. Anything "
                "else is a property of the session and belongs in the header."
            )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "record_type": "frame",
            "frame_id": int(self.frame_id),
            "capture_time_s": float(self.capture_time_s),
            "width": int(self.width),
            "height": int(self.height),
            "boxes": [b.to_json_dict() for b in self.boxes],
        }
        if self.image_path:
            out["image_path"] = self.image_path
        if self.capture_overrides:
            out["capture_overrides"] = dict(self.capture_overrides)
        return out

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> FrameRecord:
        return cls(
            frame_id=int(payload["frame_id"]),
            capture_time_s=float(payload.get("capture_time_s", 0.0)),
            width=int(payload["width"]),
            height=int(payload["height"]),
            boxes=[ObjectRecord.from_json_dict(b) for b in payload.get("boxes", [])],
            image_path=payload.get("image_path"),
            capture_overrides=dict(payload.get("capture_overrides", {})),
        )

    def detections(self) -> list[Detection]:
        return [b.to_detection(self.frame_id, self.capture_time_s) for b in self.boxes]


@dataclass(slots=True)
class DeviceCapture:
    """One session: its header plus its frames."""

    path: Path
    capture: CaptureMetadata
    frames: list[FrameRecord] = field(default_factory=list)
    session_id: str = ""
    schema_version: str = SCHEMA_VERSION

    @property
    def sample_id(self) -> str:
        """Grouping key for a leakage-safe split. One sample = one group."""
        return self.capture.sample_id

    def effective_capture(self, frame: FrameRecord) -> dict[str, Any]:
        """Header metadata with this frame's overrides applied."""
        merged = self.capture.to_json_dict()
        merged.update(frame.capture_overrides)
        return merged

    def tracks(self) -> dict[int, list[Detection]]:
        """Per-track detection sequences, sorted by frame."""
        out: dict[int, list[Detection]] = {}
        for frame in self.frames:
            for det in frame.detections():
                if det.track_id is None:
                    continue
                out.setdefault(int(det.track_id), []).append(det)
        for detections in out.values():
            detections.sort(key=lambda d: d.frame_id)
        return dict(sorted(out.items()))

    def morphology_labels(self) -> dict[int, dict[str, int | None]]:
        """Per-track morphology, merged across the frames that carry it.

        A track annotated on one frame is annotated for the whole track -- the
        cell does not change shape between frames. Conflicting non-null values
        for one aspect raise: two annotators disagreeing about the same sperm is
        a fact worth surfacing, not something to resolve by taking the last one.
        """
        merged: dict[int, dict[str, int | None]] = {}
        for frame in self.frames:
            for box in frame.boxes:
                if box.track_id is None or not box.morphology:
                    continue
                target = merged.setdefault(int(box.track_id), {})
                for aspect, value in box.morphology.items():
                    if value is None:
                        continue
                    existing = target.get(aspect)
                    if existing is not None and existing != value:
                        raise DatasetValidationError(
                            f"{self.path}: track {box.track_id} has conflicting "
                            f"'{aspect}' labels ({existing} and {value}) on different "
                            "frames. One sperm has one morphology; resolve the "
                            "disagreement in the source annotations."
                        )
                    target[aspect] = value
        return merged


# ==========================================================================
# Writer
# ==========================================================================


class DeviceAnnotationWriter:
    """Append-only JSON Lines writer for one capture session.

    Usage::

        with DeviceAnnotationWriter(path, capture) as writer:
            writer.write_frame(FrameRecord(...))

    The header is written on open, so a file that exists always has one; frames
    are flushed per line, so an interrupted session leaves a file that reads
    cleanly up to its last complete frame.
    """

    def __init__(
        self,
        path: str | Path,
        capture: CaptureMetadata,
        *,
        session_id: str = "",
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"{self.path} already exists. Annotation files are append-only records "
                "of a capture session; pass overwrite=True only if you are certain the "
                "existing session is disposable."
            )
        self.capture = capture
        self.session_id = session_id or self.path.stem
        self._handle: Any = None
        self._n_frames = 0

    def __enter__(self) -> DeviceAnnotationWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        header = {
            "record_type": "header",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "capture": self.capture.to_json_dict(),
            "morphology_aspects": list(MORPHOLOGY_ASPECTS),
            "label_encoding": {"normal": LABEL_NORMAL, "abnormal": LABEL_ABNORMAL},
        }
        self._write(header)

    def write_frame(self, frame: FrameRecord) -> None:
        """Append one frame record."""
        if self._handle is None:
            raise RuntimeError("writer is not open; use it as a context manager")
        self._write(frame.to_json_dict())
        self._n_frames += 1

    def write_frames(self, frames: Iterable[FrameRecord]) -> None:
        for frame in frames:
            self.write_frame(frame)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        # Flush per record: an interrupted capture must leave every frame that
        # was written actually on disk, not in a buffer.
        self._handle.flush()


def read_capture(path: str | Path) -> DeviceCapture:
    """Read one ``.jsonl`` capture file.

    Raises
    ------
    DatasetValidationError
        On a missing or malformed header, or a malformed frame record, naming
        the line number.
    """
    target = Path(path)
    capture: CaptureMetadata | None = None
    session_id = ""
    schema_version = SCHEMA_VERSION
    frames: list[FrameRecord] = []

    with target.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if lineno == 1:
                    raise DatasetValidationError(
                        f"{target}:1 is not valid JSON: {exc}. A device annotation file "
                        "must begin with a header record."
                    ) from exc
                # A truncated final line is the expected shape of an interrupted
                # capture; everything before it is still valid data.
                raise DatasetValidationError(
                    f"{target}:{lineno} is not valid JSON: {exc}. If this is the last "
                    "line, the capture was interrupted mid-write; truncate the file to "
                    f"its first {lineno - 1} lines to recover the complete frames."
                ) from exc

            kind = payload.get("record_type")
            if kind == "header":
                capture = CaptureMetadata.from_json_dict(payload.get("capture", {}))
                session_id = str(payload.get("session_id", target.stem))
                schema_version = str(payload.get("schema_version", SCHEMA_VERSION))
            elif kind == "frame":
                if capture is None:
                    raise DatasetValidationError(
                        f"{target}:{lineno}: a frame record appears before the header. "
                        "The first line of a device annotation file must be the header "
                        "carrying the capture metadata."
                    )
                try:
                    frames.append(FrameRecord.from_json_dict(payload))
                except (KeyError, TypeError, ValueError) as exc:
                    raise DatasetValidationError(
                        f"{target}:{lineno}: malformed frame record: {exc}"
                    ) from exc
            else:
                raise DatasetValidationError(
                    f"{target}:{lineno}: unknown record_type {kind!r}; expected "
                    "'header' or 'frame'"
                )

    if capture is None:
        raise DatasetValidationError(
            f"{target}: no header record found. Every capture file must start with "
            f'{{"record_type": "header", "capture": {{...}}}} carrying at least '
            f"{list(REQUIRED_CAPTURE_FIELDS)}."
        )
    return DeviceCapture(
        path=target,
        capture=capture,
        frames=frames,
        session_id=session_id,
        schema_version=schema_version,
    )


# ==========================================================================
# Adapter
# ==========================================================================


class DeviceDatasetAdapter(DatasetAdapter):
    """Reader for a directory of device capture sessions.

    Parameters
    ----------
    root
        Directory containing ``*.jsonl`` capture files (optionally under an
        ``annotations/`` subfolder) and, usually, the frame images they name.
    require_present
        See :class:`~datasets.adapters.base.DatasetAdapter`.
    """

    info = DatasetInfo(
        name="device",
        title="Device capture (data recorded on this instrument)",
        url="(internal -- produced by this device, not downloaded)",
        license_key="device",
        annotation_level=(
            "per-frame boxes + track ids + optional per-track morphology aspects, "
            "with session capture metadata"
        ),
        approximate_size="depends on the capture campaign",
        capture=CaptureConditions(
            objective_magnification=None,
            total_magnification=None,
            contrast_mode=None,
            stained=False,
            camera=None,
            fps_range=None,
            fps_uniform=True,
            resolution=None,
            um_per_px=None,
            notes=(
                "Deliberately unspecified at the class level: the real values come "
                "from each session's header (CaptureMetadata.to_capture_conditions()). "
                "A class-level default here would be a guess that outlived the rig it "
                "described."
            ),
        ),
        domain_shift_notes=[
            "This is the in-domain corpus: by construction there is no domain shift "
            "between it and deployment, provided the optics and illumination in each "
            "session header match the deployed configuration -- which is why those "
            "fields are required rather than optional.",
            "Compare a public dataset against a session with "
            "CaptureConditions.differences_from() to enumerate the gap a fine-tune "
            "has to close.",
            "Weights fine-tuned on this data carry "
            "constants.WEIGHTS_PROVENANCE_DEVICE; weights from public data alone "
            "carry WEIGHTS_PROVENANCE_PUBLIC and are baseline research weights.",
        ],
        expected_layout=(
            "  <root>/*.jsonl                  (or <root>/annotations/*.jsonl)\n"
            "      line 1: {\"record_type\": \"header\", \"capture\": {...}}\n"
            "      line n: {\"record_type\": \"frame\", \"frame_id\": .., \"boxes\": [..]}\n"
            "  <root>/frames/...               (images referenced by image_path)"
        ),
    )

    def __init__(self, root: str | Path, *, require_present: bool = True) -> None:
        self._captures: dict[str, DeviceCapture] = {}
        super().__init__(root, require_present=require_present)

    # ------------------------------------------------------------ discovery

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """Accept the capture directory or an ``annotations/`` folder inside it.

        A directory with no ``.jsonl`` anywhere under it is not a capture
        corpus, so ``None`` is returned and the constructor raises -- rather
        than handing back an adapter that reports zero frames and looks like a
        dataset that happens to be empty.
        """
        for candidate in (given, given / "annotations"):
            if candidate.is_dir() and any(candidate.glob(f"*{_ANNOTATION_SUFFIX}")):
                return candidate
        if given.is_dir() and any(given.rglob(f"*{_ANNOTATION_SUFFIX}")):
            return given
        return None

    def annotation_files(self) -> list[Path]:
        """Capture files under the root, sorted."""
        direct = sorted(self.root.glob(f"*{_ANNOTATION_SUFFIX}"))
        if direct:
            return direct
        return sorted(self.root.rglob(f"*{_ANNOTATION_SUFFIX}"))

    def session(self, path: str | Path) -> DeviceCapture:
        """Read (and cache) one capture file.

        Named ``session``, not ``capture``: ``DatasetAdapter.capture`` is already
        the adapter-level :class:`~datasets.adapters.base.CaptureConditions`
        property, and shadowing it here would make ``adapter.capture`` mean two
        different things depending on which adapter you were holding.
        """
        key = str(Path(path))
        if key not in self._captures:
            self._captures[key] = read_capture(path)
        return self._captures[key]

    def sessions(self) -> list[DeviceCapture]:
        """Every capture session under the root."""
        return [self.session(p) for p in self.annotation_files()]

    def iter_frames(self) -> Iterator[tuple[DeviceCapture, FrameRecord]]:
        """Stream ``(session, frame)`` across every session."""
        for session in self.sessions():
            for frame in session.frames:
                yield session, frame

    def samples(self) -> list[str]:
        """Distinct ``sample_id`` values -- the grouping key for a safe split."""
        return sorted({c.sample_id for c in self.sessions()})

    def detections(self) -> list[Detection]:
        """Every annotated object across every session, as ``Detection``.

        Track IDs are namespaced by session (``meta["session_id"]``) because IDs
        restart at each capture; merging sessions without namespacing merges
        unrelated sperm into one track.
        """
        out: list[Detection] = []
        for session, frame in self.iter_frames():
            for det in frame.detections():
                det.meta["session_id"] = session.session_id
                det.meta["sample_id"] = session.sample_id
                out.append(det)
        return out

    # ------------------------------------------------------------- contract

    def splits(self) -> list[str]:
        """No fixed splits.

        Device captures accumulate over time; a split fixed at file level would
        go stale. Build one with
        :func:`datasets.validators.leakage.patient_level_split` keyed on
        :attr:`DeviceCapture.sample_id`.
        """
        return []

    def __len__(self) -> int:
        """Total annotated frames across every session."""
        return sum(len(c.frames) for c in self.sessions())

    def validate(self) -> ValidationReport:
        """Check every session parses and carries the required capture metadata."""
        report = self._new_report()
        files = self.annotation_files()
        report.checks.append(
            check_non_empty(len(files), name="captures", what="*.jsonl capture files")
        )
        if not files:
            return report

        n_frames = 0
        n_boxes = 0
        n_tracks = 0
        n_morphology = 0
        samples: set[str] = set()
        missing_temperature: list[str] = []

        for path in files:
            try:
                session = self.session(path)
            except DatasetValidationError as exc:
                report.add(f"capture:{path.name}", CheckStatus.FAIL, str(exc))
                continue

            samples.add(session.sample_id)
            n_frames += len(session.frames)
            n_boxes += sum(len(f.boxes) for f in session.frames)
            n_tracks += len(session.tracks())
            try:
                n_morphology += len(session.morphology_labels())
            except DatasetValidationError as exc:
                report.add(f"morphology:{path.name}", CheckStatus.FAIL, str(exc))

            # from_json_dict already enforced the required fields; this records
            # the pass explicitly so the report shows the check was performed.
            report.add(
                f"capture_metadata:{path.name}",
                CheckStatus.PASS,
                f"{path.name}: all required capture metadata present "
                f"({', '.join(REQUIRED_CAPTURE_FIELDS)}); sample={session.sample_id}, "
                f"um_per_px={session.capture.um_per_px}, "
                f"{session.capture.frame_rate_hz} Hz",
            )
            if session.capture.temperature_c is None:
                missing_temperature.append(path.name)

            if not session.frames:
                report.add(
                    f"frames:{path.name}",
                    CheckStatus.FAIL,
                    f"{path.name}: header present but no frame records",
                )
                continue

            frame_ids = [f.frame_id for f in session.frames]
            if len(set(frame_ids)) != len(frame_ids):
                duplicates = sorted({i for i in frame_ids if frame_ids.count(i) > 1})
                report.add(
                    f"frames:{path.name}",
                    CheckStatus.FAIL,
                    f"{path.name}: duplicate frame_id(s) {duplicates[:10]}",
                )
            sizes = {(f.width, f.height) for f in session.frames}
            if len(sizes) > 1:
                report.add(
                    f"frames:size:{path.name}",
                    CheckStatus.WARN,
                    f"{path.name}: frame size changes within the session: {sorted(sizes)}",
                )
            for frame in session.frames:
                for box in frame.boxes:
                    if box.box.x1 < -1 or box.box.y1 < -1 or box.box.x2 > frame.width + 1 or box.box.y2 > frame.height + 1:
                        report.add(
                            f"boxes:bounds:{path.name}",
                            CheckStatus.WARN,
                            f"{path.name} frame {frame.frame_id}: box "
                            f"{box.box.as_xyxy()} lies outside the declared "
                            f"{frame.width}x{frame.height} frame",
                        )
                        break

        if missing_temperature:
            report.add(
                "capture_metadata:temperature",
                CheckStatus.WARN,
                f"{len(missing_temperature)} session(s) record no temperature_c: "
                f"{missing_temperature[:10]}. Optional, but sperm motility is strongly "
                "temperature-dependent, so motility compared across captures at unknown "
                "temperatures is not a like-for-like comparison.",
            )

        report.context.update(
            {
                "n_sessions": len(files),
                "n_frames": n_frames,
                "n_boxes": n_boxes,
                "n_tracks": n_tracks,
                "n_tracks_with_morphology": n_morphology,
                "samples": sorted(samples),
            }
        )
        report.add(
            "totals",
            CheckStatus.PASS,
            f"{len(files)} session(s), {len(samples)} sample(s), {n_frames} frame(s), "
            f"{n_boxes} box(es), {n_tracks} track(s), {n_morphology} track(s) with "
            "morphology labels",
        )
        if n_morphology == 0:
            report.add(
                "morphology:coverage",
                CheckStatus.WARN,
                "no per-track morphology labels in this corpus. Detection and tracking "
                "can be fine-tuned on it; the four-aspect morphology head cannot.",
            )
        return report


# ==========================================================================
# helpers
# ==========================================================================


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _opt_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _magnification(objective: str | None) -> float | None:
    """Parse ``"20x"`` / ``"x40"`` into a number; ``None`` when unparseable.

    Returning ``None`` rather than a guess keeps
    :class:`~datasets.adapters.base.CaptureConditions` honest: an unparsed
    objective string is an unknown magnification.
    """
    if not objective:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]|[xX]\s*(\d+(?:\.\d+)?)", objective)
    if match is None:
        return None
    return float(match.group(1) or match.group(2))
