"""Tracks <-> MOTChallenge text format, so standard HOTA/IDF1 tooling can score us.

Why bother with somebody else's text format
-------------------------------------------
Tracking metrics are easy to compute wrongly and almost impossible to check by
eye. HOTA, IDF1, MOTA and the identity-switch count all depend on association
decisions that a bespoke implementation gets subtly different -- and a bespoke
implementation is always evaluated by the same person who wrote the tracker.
Emitting MOTChallenge text means the numbers come from ``TrackEval`` or
``py-motmetrics``, which nobody in this project can accidentally bias, and it
means the tracker can be compared against published results at all.

The format
----------
Nine comma-separated fields per line::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, cls, vis

* ``frame`` and ``id`` are **1-based**. This is the single most common bug in
  MOT export: a 0-based frame counter shifts every ground-truth association by
  one frame, which reads as a tracker that is slightly bad rather than an
  exporter that is exactly wrong. :data:`MOT_FRAME_OFFSET` is applied on write
  and removed on read, and both directions take the same ``frame_offset``
  argument so a round-trip cannot drift.
* ``bb_left, bb_top, bb_width, bb_height`` are pixels, top-left origin. Our
  boxes are ``x1 y1 x2 y2``, so the conversion is
  :meth:`~sperm_sorting.schemas.detection.BoundingBox.as_xywh`.
* ``conf`` is the detection confidence for results files; for ground truth it is
  a 0/1 *consider-this-target* flag. Both live in ``score``, and
  :func:`tracks_to_mot` takes ``as_ground_truth`` to make the choice explicit
  rather than accidental.
* ``cls`` is the object class. MOTChallenge reserves 1 for pedestrians; this
  project writes its own class ids and records the mapping in the sidecar
  written by :func:`write_mot_file`, because a silent renumbering is worse than
  a non-standard class id.
* ``vis`` is visibility in ``[0, 1]``. There is no visibility annotation in any
  sperm dataset used here, so it is written as 1.0 and preserved on round-trip
  when a file supplies it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sperm_sorting.schemas.detection import BoundingBox, Detection

__all__ = [
    "MOT_COLUMNS",
    "MOT_FRAME_OFFSET",
    "detections_to_mot",
    "mot_to_detections",
    "mot_to_tracks",
    "read_mot_file",
    "tracks_to_mot",
    "write_mot_file",
]

#: Column order of the MOTChallenge ground-truth / results text format.
MOT_COLUMNS: tuple[str, ...] = (
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "cls",
    "vis",
)

#: MOTChallenge frames and ids are 1-based; ours are 0-based.
MOT_FRAME_OFFSET: int = 1


# ==========================================================================
# Writing
# ==========================================================================


def detections_to_mot(
    detections: Iterable[Detection],
    *,
    frame_offset: int = MOT_FRAME_OFFSET,
    id_offset: int = 0,
    as_ground_truth: bool = False,
    precision: int | None = None,
    default_track_id: int = -1,
) -> list[str]:
    """Render detections as MOTChallenge lines, sorted by ``(frame, id)``.

    Parameters
    ----------
    detections
        Any iterable. Each must carry a ``track_id`` unless
        ``default_track_id`` is acceptable (``-1`` is the MOT convention for a
        detection with no identity).
    frame_offset
        Added to ``frame_id`` on write. Default 1, the MOT convention.
    id_offset
        Added to ``track_id``. Default 0; use 1 if your ids start at 0 and the
        consuming tool rejects id 0.
    as_ground_truth
        Write ``conf`` as the 0/1 consider-flag instead of the detector score.
        Ground-truth and results files have the same columns and different
        meanings for this one, so it is a parameter rather than a guess.
    precision
        ``None`` (default) writes full double precision, which is what makes the
        round-trip lossless. Pass an integer for a conventional-looking file.
    default_track_id
        Used when ``track_id`` is ``None``.
    """
    rows: list[tuple[int, int, str]] = []
    for det in detections:
        frame = int(det.frame_id) + int(frame_offset)
        track = (
            int(default_track_id)
            if det.track_id is None
            else int(det.track_id) + int(id_offset)
        )
        left, top, width, height = det.box.as_xywh()
        conf = 1.0 if as_ground_truth else float(det.score)
        visibility = float(det.meta.get("visibility", 1.0))
        fields = [
            str(frame),
            str(track),
            _fmt(left, precision),
            _fmt(top, precision),
            _fmt(width, precision),
            _fmt(height, precision),
            _fmt(conf, precision),
            str(int(det.class_id)),
            _fmt(visibility, precision),
        ]
        rows.append((frame, track, ",".join(fields)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [row[2] for row in rows]


def tracks_to_mot(
    tracks: Mapping[int, Sequence[Detection]] | Iterable[Detection],
    **kwargs: Any,
) -> list[str]:
    """Render tracks as MOTChallenge lines.

    Accepts either the ``{track_id: [Detection, ...]}`` shape returned by
    :meth:`datasets.adapters.visem_tracking.VisemTrackingAdapter.tracks` or a
    flat iterable of detections. When a mapping is given, the mapping's key wins
    over any ``track_id`` already on the detection -- the caller has stated the
    identity explicitly, and a disagreement there means the two disagree about
    which sperm this is, which the mapping is the authority on.
    """
    if isinstance(tracks, Mapping):
        flattened: list[Detection] = []
        for track_id, detections in tracks.items():
            for det in detections:
                if det.track_id == int(track_id):
                    flattened.append(det)
                else:
                    # Copy rather than mutate: the caller's ground truth must
                    # not change shape because it was exported.
                    flattened.append(
                        Detection(
                            frame_id=det.frame_id,
                            box=det.box,
                            score=det.score,
                            class_id=det.class_id,
                            class_name=det.class_name,
                            capture_time_s=det.capture_time_s,
                            track_id=int(track_id),
                            meta=dict(det.meta),
                        )
                    )
        return detections_to_mot(flattened, **kwargs)
    return detections_to_mot(tracks, **kwargs)


def write_mot_file(
    path: str | Path,
    tracks: Mapping[int, Sequence[Detection]] | Iterable[Detection],
    *,
    class_names: Mapping[int, str] | None = None,
    write_sidecar: bool = True,
    **kwargs: Any,
) -> Path:
    """Write a MOTChallenge file, plus a small JSON sidecar.

    The sidecar (``<name>.meta.json``) records the class-id -> name mapping and
    the offsets used. MOTChallenge's own class numbering is for pedestrian
    benchmarks and means nothing here; rather than renumber sperm to 1 and lose
    the distinction from clusters and pinheads, the ids are written as they are
    and their meaning is written next to them.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = tracks_to_mot(tracks, **kwargs)
    target.write_text("\n".join(lines) + ("\n" if lines else ""))

    if write_sidecar:
        sidecar = target.with_suffix(target.suffix + ".meta.json")
        sidecar.write_text(
            json.dumps(
                {
                    "columns": list(MOT_COLUMNS),
                    "frame_offset": int(kwargs.get("frame_offset", MOT_FRAME_OFFSET)),
                    "id_offset": int(kwargs.get("id_offset", 0)),
                    "as_ground_truth": bool(kwargs.get("as_ground_truth", False)),
                    "class_names": {str(k): v for k, v in (class_names or {}).items()},
                    "note": (
                        "class ids are this project's, not MOTChallenge's pedestrian "
                        "numbering; frames and ids in the .txt are offset as recorded here"
                    ),
                },
                indent=2,
            )
        )
    return target


# ==========================================================================
# Reading
# ==========================================================================


def mot_to_detections(
    text: str | Iterable[str],
    *,
    frame_offset: int = MOT_FRAME_OFFSET,
    id_offset: int = 0,
    class_names: Mapping[int, str] | None = None,
    keep_unidentified: bool = True,
) -> list[Detection]:
    """Parse MOTChallenge lines into detections.

    ``frame_offset`` and ``id_offset`` are **subtracted**, inverting
    :func:`detections_to_mot` exactly when the same values are used.

    ``keep_unidentified`` controls what happens to the ``id = -1`` rows that a
    raw detection file uses: kept with ``track_id=None`` by default, because
    dropping them silently would turn a detection dump into a much smaller and
    apparently better one.
    """
    lines = text.splitlines() if isinstance(text, str) else list(text)
    out: list[Detection] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            raise ValueError(
                f"line {lineno}: MOT lines need at least 6 fields "
                f"(frame,id,x,y,w,h), got {len(parts)}: {line!r}"
            )
        frame = int(float(parts[0])) - int(frame_offset)
        raw_id = int(float(parts[1]))
        left, top, width, height = (float(p) for p in parts[2:6])
        conf = float(parts[6]) if len(parts) > 6 and parts[6] != "" else 1.0
        class_id = int(float(parts[7])) if len(parts) > 7 and parts[7] != "" else 0
        visibility = float(parts[8]) if len(parts) > 8 and parts[8] != "" else 1.0

        track_id: int | None = None if raw_id < 0 else raw_id - int(id_offset)
        if track_id is None and not keep_unidentified:
            continue
        out.append(
            Detection(
                frame_id=frame,
                box=BoundingBox.from_xywh(left, top, width, height),
                score=conf,
                class_id=class_id,
                class_name=(class_names or {}).get(class_id, f"class_{class_id}"),
                track_id=track_id,
                meta={"visibility": visibility},
            )
        )
    return out


def mot_to_tracks(
    text: str | Iterable[str], **kwargs: Any
) -> dict[int, list[Detection]]:
    """Parse MOT lines and group them into ``{track_id: [Detection, ...]}``.

    Detections with no identity (``id = -1``) are excluded -- a track keyed by
    ``None`` is not a track. Use :func:`mot_to_detections` to keep them.
    """
    kwargs.setdefault("keep_unidentified", False)
    grouped: dict[int, list[Detection]] = {}
    for det in mot_to_detections(text, **kwargs):
        if det.track_id is None:
            continue
        grouped.setdefault(int(det.track_id), []).append(det)
    for detections in grouped.values():
        detections.sort(key=lambda d: d.frame_id)
    return dict(sorted(grouped.items()))


def read_mot_file(path: str | Path, **kwargs: Any) -> dict[int, list[Detection]]:
    """:func:`mot_to_tracks` on a file, honouring its sidecar when present.

    If ``<name>.meta.json`` exists, the offsets recorded there are used unless
    the caller overrode them -- which is what makes a file written by
    :func:`write_mot_file` reread correctly without the caller having to
    remember what it chose.
    """
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".meta.json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        kwargs.setdefault("frame_offset", int(meta.get("frame_offset", MOT_FRAME_OFFSET)))
        kwargs.setdefault("id_offset", int(meta.get("id_offset", 0)))
        names = meta.get("class_names") or {}
        if names and "class_names" not in kwargs:
            kwargs["class_names"] = {int(k): v for k, v in names.items()}
    return mot_to_tracks(target.read_text(), **kwargs)


def _fmt(value: float, precision: int | None) -> str:
    if precision is None:
        return repr(float(value))
    return f"{float(value):.{int(precision)}f}"
