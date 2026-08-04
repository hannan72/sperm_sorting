"""Annotation format conversion: YOLO / Pascal VOC / COCO <-> internal ``Detection``.

One hub, three spokes. Every public dataset arrives in one of these three
formats and every training framework wants one of them back, so the temptation
is to write YOLO->COCO directly and skip the middle. Doing that for three
formats needs six converters instead of six half-converters, and each one
re-derives the coordinate conventions independently -- which is where the
off-by-one lives. Everything therefore round-trips through
:class:`sperm_sorting.schemas.detection.Detection`, whose box is unambiguous:
``(x1, y1, x2, y2)`` in absolute pixels, ``x2``/``y2`` exclusive, so
``width == x2 - x1``.

The three conventions, spelled out because they genuinely differ
----------------------------------------------------------------
**YOLO** -- ``class x_center y_center width height``, every coordinate
normalised to ``[0, 1]`` by the image size. Needs the image size to convert in
either direction; there is no default here, because a wrong size silently
rescales every box and nothing downstream can detect it.

**Pascal VOC** -- ``<bndbox>`` with ``xmin ymin xmax ymax`` in pixels. The
original VOC specification is **1-based and inclusive**: a box from ``xmin=1``
to ``xmax=10`` is 10 pixels wide, not 9. Many modern tools (LabelImg among them)
write 0-based coordinates into the same tags, and no tag says which. There is no
way to detect it from the file, so :func:`voc_to_detections` takes an explicit
``origin`` parameter, defaults to the specification, and does the arithmetic in
one visible place instead of leaving a silent one-pixel bias in every box.

**COCO** -- ``bbox: [x, y, width, height]`` in pixels, 0-based, floats, with
``x2 = x + width``. Category ids are conventionally 1-based, which is why
:func:`detections_to_coco` offsets them and :func:`coco_to_detections` reads the
category table rather than assuming the id *is* the class index.

Losslessness
------------
Round-trips are exact to within double-precision arithmetic. That requires
writing full precision by default: the usual ``%.6f`` YOLO convention loses up
to half a thousandth of an image width -- about 0.3 px on a 640-wide frame,
which is a fifth of a sperm head in VISEM-Tracking. ``precision=None`` (the
default) writes ``repr``-grade floats and round-trips to ~1e-12 px; pass
``precision=6`` when a downstream tool insists on the conventional look, and
accept the rounding knowingly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from sperm_sorting.schemas.detection import BoundingBox, Detection

__all__ = [
    "coco_to_detections",
    "detections_to_coco",
    "detections_to_voc",
    "detections_to_yolo",
    "read_yolo_file",
    "voc_to_detections",
    "write_yolo_file",
    "yolo_to_detections",
]

#: VOC coordinate conventions. See the module docstring.
_VOC_ORIGINS = ("one_based_inclusive", "zero_based_exclusive")


# ==========================================================================
# YOLO
# ==========================================================================


def yolo_to_detections(
    text: str | Iterable[str],
    *,
    image_size: tuple[int, int],
    frame_id: int = 0,
    class_names: Mapping[int, str] | None = None,
    default_score: float = 1.0,
    track_id_first: bool = False,
) -> list[Detection]:
    """Parse YOLO annotations into absolute-pixel detections.

    Parameters
    ----------
    text
        The file contents, or any iterable of lines.
    image_size
        ``(width, height)`` in pixels. Required: YOLO coordinates are
        normalised and carry no scale of their own.
    frame_id
        Stamped onto every detection.
    class_names
        ``class_id -> name``. Unmapped ids get ``"class_<id>"`` rather than
        being dropped -- an unexpected class is data, not noise.
    default_score
        Ground-truth files carry no confidence, so 1.0. A 6-field line whose
        last field is a confidence (a *prediction* dump) is read as such.
    track_id_first
        Set True for VISEM-Tracking's ``labels_ftid`` layout, where the field
        order is ``sperm_id class x y w h`` -- **track id first, then class**.
        Off by default because plain YOLO is far more common, and reading one as
        the other silently swaps class for identity.

    Raises
    ------
    ValueError
        On a line whose field count is not recognised, naming the line.
    """
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    lines = text.splitlines() if isinstance(text, str) else list(text)
    out: list[Detection] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        track_id: int | None = None
        score = float(default_score)

        if track_id_first:
            if len(parts) != 6:
                raise ValueError(
                    f"line {lineno}: expected 6 fields "
                    f"(sperm_id class x y w h) with track_id_first=True, got "
                    f"{len(parts)}: {line!r}"
                )
            track_id = int(float(parts[0]))
            class_id = int(float(parts[1]))
            cx, cy, bw, bh = (float(p) for p in parts[2:6])
        elif len(parts) == 5:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
        elif len(parts) == 6:
            # Prediction dumps append a confidence. Track-id-first files have
            # the same field count, which is exactly why track_id_first is an
            # explicit parameter and not something guessed from the shape.
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            score = float(parts[5])
        else:
            raise ValueError(
                f"line {lineno}: expected 5 fields (class x y w h) or 6 "
                f"(class x y w h conf), got {len(parts)}: {line!r}. For "
                "VISEM-Tracking labels_ftid files pass track_id_first=True."
            )

        box_w = bw * width
        box_h = bh * height
        centre_x = cx * width
        centre_y = cy * height
        out.append(
            Detection(
                frame_id=int(frame_id),
                box=BoundingBox(
                    centre_x - 0.5 * box_w,
                    centre_y - 0.5 * box_h,
                    centre_x + 0.5 * box_w,
                    centre_y + 0.5 * box_h,
                ),
                score=score,
                class_id=class_id,
                class_name=(class_names or {}).get(class_id, f"class_{class_id}"),
                track_id=track_id,
            )
        )
    return out


def detections_to_yolo(
    detections: Sequence[Detection],
    *,
    image_size: tuple[int, int],
    precision: int | None = None,
    include_score: bool = False,
    track_id_first: bool = False,
) -> list[str]:
    """Render detections as YOLO lines.

    ``precision=None`` writes full double precision so that
    ``yolo -> Detection -> yolo`` is lossless; see the module docstring for why
    the conventional 6 decimals is not good enough for 640x480 sperm.
    """
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    lines: list[str] = []
    for det in detections:
        cx, cy, bw, bh = det.box.as_cxcywh()
        values = (cx / width, cy / height, bw / width, bh / height)
        rendered = [_fmt(v, precision) for v in values]
        fields: list[str] = []
        if track_id_first:
            if det.track_id is None:
                raise ValueError(
                    f"detection on frame {det.frame_id} has no track_id, so it cannot be "
                    "written in labels_ftid form (sperm_id class x y w h)"
                )
            fields.append(str(int(det.track_id)))
        fields.append(str(int(det.class_id)))
        fields.extend(rendered)
        if include_score:
            fields.append(_fmt(float(det.score), precision))
        lines.append(" ".join(fields))
    return lines


def read_yolo_file(path: str | Path, **kwargs: Any) -> list[Detection]:
    """:func:`yolo_to_detections` on a file. ``frame_id`` defaults to the stem."""
    target = Path(path)
    kwargs.setdefault("frame_id", _trailing_int(target.stem) or 0)
    return yolo_to_detections(target.read_text(), **kwargs)


def write_yolo_file(
    path: str | Path, detections: Sequence[Detection], **kwargs: Any
) -> Path:
    """Write YOLO lines to ``path``, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = detections_to_yolo(detections, **kwargs)
    target.write_text("\n".join(lines) + ("\n" if lines else ""))
    return target


# ==========================================================================
# Pascal VOC
# ==========================================================================


def voc_to_detections(
    xml_text: str,
    *,
    frame_id: int = 0,
    class_map: Mapping[str, tuple[int, str]] | None = None,
    origin: str = "one_based_inclusive",
    default_score: float = 1.0,
) -> list[Detection]:
    """Parse a Pascal VOC XML annotation into absolute-pixel detections.

    Parameters
    ----------
    xml_text
        Contents of the ``.xml`` file.
    frame_id
        Stamped onto every detection.
    class_map
        ``<name> -> (class_id, class_name)``. Unmapped names are assigned ids in
        first-seen order and keep their original name, so an unexpected class is
        surfaced rather than dropped.
    origin
        ``"one_based_inclusive"`` (the VOC specification: ``xmin=1, xmax=10`` is
        10 px wide) or ``"zero_based_exclusive"`` (what several modern tools
        write into the same tags). The file cannot tell you which it is, so this
        must be chosen deliberately -- the difference is one pixel on every edge
        of every box.
    default_score
        VOC has no confidence field.

    Raises
    ------
    ValueError
        On an unparseable document, a missing ``<bndbox>``, or an unknown
        ``origin``.
    """
    if origin not in _VOC_ORIGINS:
        raise ValueError(f"origin must be one of {list(_VOC_ORIGINS)}, got {origin!r}")

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"not a parseable Pascal VOC document: {exc}") from exc

    mapping = dict(class_map or {})
    assigned: dict[str, int] = {}
    out: list[Detection] = []

    for obj in root.iter("object"):
        name_element = obj.find("name")
        name = (name_element.text or "").strip() if name_element is not None else ""
        box_element = obj.find("bndbox")
        if box_element is None:
            raise ValueError(f"VOC <object> {name!r} has no <bndbox>")
        xmin = _voc_float(box_element, "xmin")
        ymin = _voc_float(box_element, "ymin")
        xmax = _voc_float(box_element, "xmax")
        ymax = _voc_float(box_element, "ymax")

        if origin == "one_based_inclusive":
            # 1-based inclusive -> 0-based exclusive: the left/top edge moves
            # back one pixel; the right/bottom edge is already exclusive once
            # the origin shifts. Width is preserved: xmax - (xmin - 1).
            x1, y1, x2, y2 = xmin - 1.0, ymin - 1.0, xmax, ymax
        else:
            x1, y1, x2, y2 = xmin, ymin, xmax, ymax

        if name in mapping:
            class_id, class_name = mapping[name]
        else:
            if name not in assigned:
                assigned[name] = len(mapping) + len(assigned)
            class_id, class_name = assigned[name], name or f"class_{assigned[name]}"

        difficult = obj.find("difficult")
        truncated = obj.find("truncated")
        out.append(
            Detection(
                frame_id=int(frame_id),
                box=BoundingBox(x1, y1, x2, y2),
                score=float(default_score),
                class_id=int(class_id),
                class_name=str(class_name),
                meta={
                    "voc_name": name,
                    "difficult": int(difficult.text or 0) if difficult is not None else 0,
                    "truncated": int(truncated.text or 0) if truncated is not None else 0,
                },
            )
        )
    return out


def detections_to_voc(
    detections: Sequence[Detection],
    *,
    image_size: tuple[int, int],
    filename: str = "image.jpg",
    folder: str = "images",
    origin: str = "one_based_inclusive",
    depth: int = 1,
) -> str:
    """Render detections as a Pascal VOC XML document.

    The exact inverse of :func:`voc_to_detections` under the same ``origin``, so
    VOC -> internal -> VOC preserves integer box coordinates exactly.
    """
    if origin not in _VOC_ORIGINS:
        raise ValueError(f"origin must be one of {list(_VOC_ORIGINS)}, got {origin!r}")

    width, height = int(image_size[0]), int(image_size[1])
    root = ElementTree.Element("annotation")
    ElementTree.SubElement(root, "folder").text = folder
    ElementTree.SubElement(root, "filename").text = filename
    size = ElementTree.SubElement(root, "size")
    ElementTree.SubElement(size, "width").text = str(width)
    ElementTree.SubElement(size, "height").text = str(height)
    ElementTree.SubElement(size, "depth").text = str(int(depth))
    ElementTree.SubElement(root, "segmented").text = "0"

    for det in detections:
        obj = ElementTree.SubElement(root, "object")
        ElementTree.SubElement(obj, "name").text = str(
            det.meta.get("voc_name") or det.class_name
        )
        ElementTree.SubElement(obj, "pose").text = "Unspecified"
        ElementTree.SubElement(obj, "truncated").text = str(int(det.meta.get("truncated", 0)))
        ElementTree.SubElement(obj, "difficult").text = str(int(det.meta.get("difficult", 0)))
        box = ElementTree.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = det.box.as_xyxy()
        if origin == "one_based_inclusive":
            x1, y1 = x1 + 1.0, y1 + 1.0
        for tag, value in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
            ElementTree.SubElement(box, tag).text = _fmt_coord(value)

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode")


# ==========================================================================
# COCO
# ==========================================================================


def coco_to_detections(
    payload: Mapping[str, Any] | str | Path,
    *,
    class_map: Mapping[str, tuple[int, str]] | None = None,
    image_ids: Iterable[int] | None = None,
) -> list[Detection]:
    """Parse a COCO detection JSON into absolute-pixel detections.

    ``frame_id`` is taken from the annotation's ``image_id``. Category ids are
    resolved through the ``categories`` table rather than being used directly as
    class indices: COCO category ids are arbitrary integers (commonly 1-based,
    frequently sparse), and treating them as a contiguous class index is how a
    two-class dataset acquires a class 91.

    ``class_map`` maps the *category name* to ``(class_id, class_name)``, so a
    dataset's own vocabulary can be normalised onto this repository's.
    """
    data = _load_json(payload)
    categories = {
        int(c["id"]): str(c.get("name", f"category_{c['id']}"))
        for c in data.get("categories", [])
    }
    # Contiguous fallback index, in category-table order, for categories the
    # caller did not map.
    contiguous = {cid: i for i, cid in enumerate(sorted(categories))}
    mapping = dict(class_map or {})
    wanted = {int(i) for i in image_ids} if image_ids is not None else None

    out: list[Detection] = []
    for annotation in data.get("annotations", []):
        image_id = int(annotation.get("image_id", 0))
        if wanted is not None and image_id not in wanted:
            continue
        x, y, w, h = (float(v) for v in annotation["bbox"])
        category_id = int(annotation.get("category_id", 0))
        name = categories.get(category_id, f"category_{category_id}")
        if name in mapping:
            class_id, class_name = mapping[name]
        else:
            class_id, class_name = contiguous.get(category_id, category_id), name
        out.append(
            Detection(
                frame_id=image_id,
                box=BoundingBox(x, y, x + w, y + h),
                score=float(annotation.get("score", 1.0)),
                class_id=int(class_id),
                class_name=str(class_name),
                track_id=(
                    int(annotation["track_id"]) if annotation.get("track_id") is not None else None
                ),
                meta={
                    "coco_category_id": category_id,
                    "coco_annotation_id": annotation.get("id"),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                },
            )
        )
    return out


def detections_to_coco(
    detections: Sequence[Detection],
    *,
    image_size: tuple[int, int],
    class_names: Mapping[int, str] | None = None,
    image_files: Mapping[int, str] | None = None,
    category_id_offset: int = 1,
    include_scores: bool = False,
) -> dict[str, Any]:
    """Render detections as a COCO detection dictionary.

    One ``images`` entry per distinct ``frame_id``. Category ids are
    ``class_id + category_id_offset`` (default 1, the COCO convention that ids
    start at 1); :func:`coco_to_detections` undoes it through the category table,
    so the round-trip is exact.

    ``include_scores`` writes a ``score`` field. Off by default because a COCO
    *ground-truth* file with confidences in it will be silently accepted by some
    evaluation tools as a results file.
    """
    width, height = int(image_size[0]), int(image_size[1])
    frame_ids = sorted({int(d.frame_id) for d in detections})
    names = dict(class_names or {})
    for det in detections:
        names.setdefault(int(det.class_id), det.class_name)

    images = [
        {
            "id": frame_id,
            "file_name": (image_files or {}).get(frame_id, f"{frame_id:06d}.jpg"),
            "width": width,
            "height": height,
        }
        for frame_id in frame_ids
    ]
    categories = [
        {"id": class_id + int(category_id_offset), "name": name, "supercategory": "cell"}
        for class_id, name in sorted(names.items())
    ]

    annotations: list[dict[str, Any]] = []
    for index, det in enumerate(detections, start=1):
        x, y, w, h = det.box.as_xywh()
        entry: dict[str, Any] = {
            "id": index,
            "image_id": int(det.frame_id),
            "category_id": int(det.class_id) + int(category_id_offset),
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": int(det.meta.get("iscrowd", 0)),
        }
        if det.track_id is not None:
            entry["track_id"] = int(det.track_id)
        if include_scores:
            entry["score"] = float(det.score)
        annotations.append(entry)

    return {
        "info": {
            "description": "converted by datasets.converters.to_detection_format",
            "schema": "COCO detection",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ==========================================================================
# helpers
# ==========================================================================


def _fmt(value: float, precision: int | None) -> str:
    """Format a float for a text annotation file.

    ``None`` means ``repr``, which is the shortest string that round-trips
    exactly through ``float()``. That is the whole reason it is the default.
    """
    if precision is None:
        return repr(float(value))
    return f"{float(value):.{int(precision)}f}"


def _fmt_coord(value: float) -> str:
    """VOC coordinates are conventionally integers; keep them that way when exact."""
    as_float = float(value)
    return str(int(as_float)) if as_float.is_integer() else repr(as_float)


def _voc_float(element: ElementTree.Element, tag: str) -> float:
    child = element.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"VOC <bndbox> is missing <{tag}>")
    return float(child.text)


def _load_json(payload: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    path = Path(payload)
    if path.exists():
        return json.loads(path.read_text())
    if isinstance(payload, str):
        return json.loads(payload)
    raise FileNotFoundError(f"no such COCO file: {payload}")


def _trailing_int(stem: str) -> int | None:
    digits = ""
    for char in reversed(stem):
        if char.isdigit():
            digits = char + digits
        else:
            break
    return int(digits) if digits else None
