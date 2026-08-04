"""MIaMIA-SVDS / Detection-Sperm (TOD-CNN). **On-disk format UNVERIFIED.**

Sources
-------
* Model code: https://github.com/Demozsj/Detection-Sperm -- a *model* repository,
  not a dataset repository. Its ``model_data/sperm_classes.txt`` is exactly two
  lines, ``S`` and ``Impurity``, and its anchors are all under 20 pixels
  (7,11  8,15  9,10  10,14  12,11  13,19), which is the clearest available
  statement of how small these objects are.
* Data: MIaMIA-SVDS on figshare, record 15074253, a 1.42 GB ``Data Set.rar``:
  Subset-A, over 125,000 objects with bounding boxes and categories from 101
  videos; Subset-B, over 26,000 segmented sperms in 10 videos; Subset-C, over
  125,000 cropped classification images.

What is verified and what is not
--------------------------------
**Verified**: the two class names and their order; the anchor sizes; the subset
descriptions and counts; that annotation was performed with LabelImg (which
writes Pascal VOC XML, or YOLO txt when switched); the capture rig -- a WLJY-9000
CASA system, 20x objective plus 20x electronic eyepiece, 30 FPS, clips of 1-3
seconds.

**Not verified**: the *on-disk* annotation format of the release. LabelImg
implies VOC XML at annotation time, but what the ``.rar`` actually contains has
not been confirmed, and the paper's 416x416 is a network input size, not the
native video resolution -- which is also unverified.

So this adapter **sniffs** rather than assumes. :meth:`DetectionSpermAdapter.sniff_format`
inspects the files present and decides between VOC XML, YOLO txt and COCO JSON
by content, reporting the evidence; an ambiguous or unrecognised layout raises
with a list of what was found. Hard-coding one format and failing with
``FileNotFoundError: annotations/*.xml`` on a release that ships something else
is exactly the mysterious failure this design avoids.

Licence conflict
----------------
There is no LICENSE file in the GitHub repository, the README welcomes
non-commercial research use, and the figshare metadata carries CC BY 4.0. The
registry records this as
:attr:`~datasets.validators.licenses.CommercialUse.UNCLEAR` and
:func:`~datasets.validators.licenses.check_commercial_use` treats it as a
blocker. This module does not pick a side.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree

from sperm_sorting.errors import DatasetValidationError
from sperm_sorting.schemas.detection import Detection

from ..converters.to_detection_format import (
    coco_to_detections,
    voc_to_detections,
    yolo_to_detections,
)
from ..validators.integrity import CheckStatus, ValidationReport, check_non_empty
from .base import CaptureConditions, DatasetAdapter, DatasetInfo

__all__ = [
    "CLASS_MAP",
    "SOURCE_CLASS_NAMES",
    "AnnotationFormat",
    "DetectionSpermAdapter",
    "FormatEvidence",
]

#: The two classes, in the order ``model_data/sperm_classes.txt`` lists them.
#: Index matters: a YOLO release encodes the class as this index.
SOURCE_CLASS_NAMES: Final[tuple[str, str]] = ("S", "Impurity")

#: Upstream name -> this repository's vocabulary. ``Impurity`` becomes
#: ``debris`` because that is what the rest of the pipeline calls a detected
#: non-sperm object -- and because keeping it as an explicit class is how false
#: positives get *measured* rather than thresholded away
#: (see :class:`sperm_sorting.schemas.detection.Detection`).
CLASS_MAP: Final[dict[str, tuple[int, str]]] = {
    "S": (0, "sperm"),
    "s": (0, "sperm"),
    "sperm": (0, "sperm"),
    "Impurity": (1, "debris"),
    "impurity": (1, "debris"),
    "debris": (1, "debris"),
}

_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_FRAME_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)$")

#: How many files the sniffer inspects before deciding. Enough to be confident,
#: small enough to run on a 125,000-object release in a second.
_SNIFF_SAMPLE: Final[int] = 25


class AnnotationFormat(str, Enum):
    """Annotation formats this adapter can read."""

    VOC_XML = "voc_xml"
    YOLO_TXT = "yolo_txt"
    COCO_JSON = "coco_json"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class FormatEvidence:
    """What the sniffer saw, and what it concluded.

    The evidence is kept because the conclusion is a *guess about somebody
    else's archive*: when it is wrong, the person debugging needs to know which
    files were counted and what was in them, not just that the answer was "YOLO".
    """

    format: AnnotationFormat
    #: Directory the annotations were found in.
    annotation_dir: Path | None
    n_files: int
    #: Human-readable observations ("47 .xml files, root tag <annotation>").
    evidence: tuple[str, ...] = ()
    #: Other formats that also matched, if any. Non-empty means ambiguity.
    also_matched: tuple[AnnotationFormat, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": str(self.format),
            "annotation_dir": str(self.annotation_dir) if self.annotation_dir else None,
            "n_files": self.n_files,
            "evidence": list(self.evidence),
            "also_matched": [str(f) for f in self.also_matched],
            "extra": dict(self.extra),
        }


class DetectionSpermAdapter(DatasetAdapter):
    """Reader for MIaMIA-SVDS, with a format sniffer instead of an assumption.

    Parameters
    ----------
    root
        Directory the ``Data Set.rar`` was extracted into.
    require_present
        See :class:`~datasets.adapters.base.DatasetAdapter`.
    frame_size
        ``(width, height)`` of the source frames. **Required for YOLO
        annotations** and ignored otherwise, because YOLO stores normalised
        coordinates and the native resolution of this release is unverified --
        the paper's 416x416 is a network input size, not the capture size.
        Passing a wrong value silently scales every box, so there is no default.
    class_map
        Override the upstream-name -> ``(class_id, class_name)`` mapping.
    """

    info = DatasetInfo(
        name="detection_sperm",
        title="MIaMIA-SVDS / Detection-Sperm (TOD-CNN)",
        url="https://github.com/Demozsj/Detection-Sperm",
        license_key="detection_sperm",
        annotation_level=(
            "Subset-A: boxes + 2 categories (>125,000 objects, 101 videos); "
            "Subset-B: segmentation (>26,000 sperms, 10 videos); "
            "Subset-C: classification crops (>125,000)"
        ),
        approximate_size="1.42 GB ('Data Set.rar' on figshare record 15074253)",
        capture=CaptureConditions(
            objective_magnification=20.0,
            total_magnification=400.0,
            contrast_mode="brightfield (CASA)",
            stained=False,
            camera="WLJY-9000 computer-aided sperm analysis system",
            fps_range=(30.0, 30.0),
            fps_uniform=True,
            resolution=None,
            um_per_px=None,
            notes=(
                "20x objective plus a 20x electronic eyepiece, so total magnification "
                "is 400x through a two-stage path. 30 FPS, clips of 1-3 seconds. "
                "NATIVE RESOLUTION UNVERIFIED: the 416x416 quoted in the paper is the "
                "network's input size after resizing, not the capture size."
            ),
        ),
        domain_shift_notes=[
            "Objects are tiny: the published anchors are all under 20 pixels (largest "
            "13x19). A detector trained here is tuned for objects an order of "
            "magnitude smaller in pixels than a higher-resolution device camera "
            "produces, and small-object detectors do not transfer up in scale for free.",
            "30 FPS against the device's much higher rate: per-frame displacement is "
            "several times larger here, so any motion prior learned from this data is "
            "wrong for the device.",
            "The second class is 'Impurity' (debris), not a second cell type. That is "
            "useful -- it is the only public set here that labels debris explicitly, "
            "which is what makes a false-positive rate measurable instead of assumed.",
            "A two-stage 20x + 20x electronic path has different aberration and "
            "contrast characteristics from a single high-NA objective.",
            "On-disk annotation format is unverified, so anything read here should be "
            "spot-checked visually before it is trained on.",
        ],
        expected_layout=(
            "  <root>/  (extracted 'Data Set.rar'; the exact tree is UNVERIFIED)\n"
            "  The sniffer looks for, in order:\n"
            "    *.xml   with a root <annotation> element      -> Pascal VOC\n"
            "    *.txt   with 5 numeric fields per line        -> YOLO (needs frame_size)\n"
            "    *.json  with images/annotations/categories    -> COCO\n"
            "  alongside images in Annotations/, labels/, annotations/ or the root itself"
        ),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        require_present: bool = True,
        frame_size: tuple[int, int] | None = None,
        class_map: dict[str, tuple[int, str]] | None = None,
    ) -> None:
        self._frame_size = frame_size
        self._class_map = dict(class_map) if class_map else dict(CLASS_MAP)
        self._evidence: FormatEvidence | None = None
        super().__init__(root, require_present=require_present)

    # ------------------------------------------------------------ discovery

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """First candidate directory that contains any file at all.

        The tree inside 'Data Set.rar' is unverified, so this cannot look for a
        named subdirectory the way the other adapters do -- but an *empty*
        directory is still unambiguously not a copy of the dataset, and saying
        so at construction beats a format sniffer failing on nothing.
        """
        for candidate in (given, given / "Data Set", given / "DataSet", given / "MIaMIA-SVDS"):
            if candidate.is_dir() and any(p.is_file() for p in candidate.rglob("*")):
                return candidate
        return None

    def subsets(self) -> dict[str, Path]:
        """Top-level directories that look like the published subsets.

        Matched by name containing ``a``/``b``/``c`` next to the word "subset",
        with everything else returned under its own name. The release's tree is
        unverified, so this reports what is there rather than asserting what
        should be.
        """
        found: dict[str, Path] = {}
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            key = entry.name.lower().replace(" ", "").replace("_", "").replace("-", "")
            if "subset" in key:
                letter = key.split("subset")[-1][:1]
                found[f"subset_{letter}" if letter else entry.name] = entry
            else:
                found[entry.name] = entry
        return found

    # ---------------------------------------------------------------- sniffer

    def sniff_format(self, directory: Path | None = None) -> FormatEvidence:
        """Detect the annotation format by inspecting the files, not the name.

        Order of investigation, and why:

        1. **COCO JSON** first. A single JSON with ``images``/``annotations``/
           ``categories`` keys is unambiguous, and a COCO release often also
           contains ``.txt`` files (licence, README) that would confuse a
           txt-based test.
        2. **VOC XML** next: a root ``<annotation>`` element containing
           ``<object><name>`` is equally unambiguous, and matches what LabelImg
           writes by default.
        3. **YOLO txt** last, and only when the lines really parse as
           ``int float float float float`` with the floats in ``[0, 1]``. A bare
           ``.txt`` file is the weakest signal in the tree, so it needs the
           strongest test.

        Raises
        ------
        DatasetValidationError
            When nothing matches, or when more than one format matches with real
            files behind each. Ambiguity is reported with the evidence rather
            than resolved by precedence -- picking silently is how you train on
            half a dataset.
        """
        if self._evidence is not None and directory is None:
            return self._evidence

        base = directory or self.root
        matches: list[FormatEvidence] = []

        coco = self._sniff_coco(base)
        if coco is not None:
            matches.append(coco)
        voc = self._sniff_voc(base)
        if voc is not None:
            matches.append(voc)
        yolo = self._sniff_yolo(base)
        if yolo is not None:
            matches.append(yolo)

        if not matches:
            listing = sorted({p.suffix.lower() for p in base.rglob("*") if p.is_file()})
            raise DatasetValidationError(
                f"MIaMIA-SVDS: no recognisable annotation format under {base}. File "
                f"extensions present: {listing or '(none)'}. Supported: Pascal VOC XML "
                "(root <annotation>), YOLO txt (class x y w h, normalised), COCO JSON "
                "(images/annotations/categories). The on-disk format of this release is "
                "unverified upstream, so if your copy uses something else, extend "
                "DetectionSpermAdapter.sniff_format rather than renaming files."
            )

        best = matches[0]
        others = tuple(m.format for m in matches[1:])
        if len(matches) > 1 and all(m.n_files >= 3 for m in matches):
            raise DatasetValidationError(
                f"MIaMIA-SVDS: ambiguous annotation format under {base} -- "
                + "; ".join(f"{m.format} ({m.n_files} files in {m.annotation_dir})" for m in matches)
                + ". Point the adapter at a single subset directory, or pass the "
                "annotation directory explicitly to sniff_format()."
            )
        resolved = FormatEvidence(
            format=best.format,
            annotation_dir=best.annotation_dir,
            n_files=best.n_files,
            evidence=best.evidence,
            also_matched=others,
            extra=best.extra,
        )
        if directory is None:
            self._evidence = resolved
        return resolved

    def _sniff_coco(self, base: Path) -> FormatEvidence | None:
        for path in sorted(base.rglob("*.json"))[:_SNIFF_SAMPLE]:
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            if {"images", "annotations"} <= set(payload):
                return FormatEvidence(
                    format=AnnotationFormat.COCO_JSON,
                    annotation_dir=path.parent,
                    n_files=1,
                    evidence=(
                        f"{path.name} is a JSON object with keys "
                        f"{sorted(set(payload) & {'images', 'annotations', 'categories'})}, "
                        f"{len(payload.get('images', []))} images and "
                        f"{len(payload.get('annotations', []))} annotations",
                    ),
                    extra={"json_path": str(path)},
                )
        return None

    def _sniff_voc(self, base: Path) -> FormatEvidence | None:
        paths = sorted(base.rglob("*.xml"))
        if not paths:
            return None
        n_ok = 0
        names: set[str] = set()
        for path in paths[:_SNIFF_SAMPLE]:
            try:
                root = ElementTree.parse(path).getroot()
            except (ElementTree.ParseError, OSError):
                continue
            if root.tag != "annotation":
                continue
            n_ok += 1
            names.update(e.text or "" for e in root.iter("name"))
        if n_ok == 0:
            return None
        return FormatEvidence(
            format=AnnotationFormat.VOC_XML,
            annotation_dir=paths[0].parent,
            n_files=len(paths),
            evidence=(
                f"{len(paths)} .xml file(s); {n_ok} of the first {min(len(paths), _SNIFF_SAMPLE)} "
                f"have a root <annotation> element; object names seen: {sorted(names)}",
            ),
            extra={"object_names": sorted(names)},
        )

    def _sniff_yolo(self, base: Path) -> FormatEvidence | None:
        paths = [p for p in sorted(base.rglob("*.txt")) if p.stem.lower() not in {"readme", "license", "classes"}]
        if not paths:
            return None
        n_ok = 0
        classes: set[int] = set()
        for path in paths[:_SNIFF_SAMPLE]:
            try:
                lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            except (OSError, UnicodeDecodeError):
                continue
            if not lines:
                continue
            if all(_looks_like_yolo(line, classes) for line in lines):
                n_ok += 1
        if n_ok == 0:
            return None
        return FormatEvidence(
            format=AnnotationFormat.YOLO_TXT,
            annotation_dir=paths[0].parent,
            n_files=len(paths),
            evidence=(
                f"{len(paths)} .txt file(s); {n_ok} of the first "
                f"{min(len(paths), _SNIFF_SAMPLE)} parse as 'class x y w h' with all "
                f"coordinates in [0, 1]; class indices seen: {sorted(classes)}",
            ),
            extra={"class_indices": sorted(classes)},
        )

    # ---------------------------------------------------------------- reading

    def annotation_files(self) -> list[Path]:
        """Annotation files, in the detected format, sorted."""
        evidence = self.sniff_format()
        if evidence.format is AnnotationFormat.COCO_JSON:
            return [Path(evidence.extra["json_path"])]
        suffix = ".xml" if evidence.format is AnnotationFormat.VOC_XML else ".txt"
        directory = evidence.annotation_dir or self.root
        return sorted(p for p in directory.rglob(f"*{suffix}") if p.is_file())

    def detections(self, path: Path | None = None) -> list[Detection]:
        """Read detections from one annotation file (or the whole COCO JSON).

        Boxes come back in absolute pixels, in this repository's
        :class:`~sperm_sorting.schemas.detection.Detection` form, with
        ``class_id`` 0 for sperm and 1 for debris.

        Raises
        ------
        DatasetValidationError
            For a YOLO release when ``frame_size`` was not supplied. YOLO
            coordinates are normalised and the native resolution of this release
            is unverified, so there is no safe default -- guessing 416x416 (the
            network input size) would scale every box wrongly and silently.
        """
        evidence = self.sniff_format()
        if evidence.format is AnnotationFormat.COCO_JSON:
            target = path or Path(evidence.extra["json_path"])
            return coco_to_detections(json.loads(target.read_text()), class_map=self._class_map)

        if path is None:
            raise ValueError(
                "detections() needs a file path for VOC/YOLO releases; use "
                "annotation_files() to enumerate them, or iter_detections() to stream."
            )
        if evidence.format is AnnotationFormat.VOC_XML:
            return voc_to_detections(
                path.read_text(),
                frame_id=_frame_number(path.stem) or 0,
                class_map=self._class_map,
            )
        if self._frame_size is None:
            raise DatasetValidationError(
                "MIaMIA-SVDS appears to ship YOLO-normalised annotations, but the native "
                "frame resolution of this release is UNVERIFIED upstream and no "
                "frame_size was supplied. Pass DetectionSpermAdapter(root, "
                "frame_size=(width, height)) with the size you measured from the images. "
                "The 416x416 in the TOD-CNN paper is the network input size after "
                "resizing, not the capture resolution -- using it would rescale every box."
            )
        width, height = self._frame_size
        return yolo_to_detections(
            path.read_text(),
            image_size=(width, height),
            frame_id=_frame_number(path.stem) or 0,
            class_names=dict(self._class_map.values()),
        )

    def iter_detections(self) -> Iterator[tuple[Path, list[Detection]]]:
        """Stream ``(annotation_path, detections)`` over the whole set."""
        evidence = self.sniff_format()
        if evidence.format is AnnotationFormat.COCO_JSON:
            path = Path(evidence.extra["json_path"])
            yield path, self.detections(path)
            return
        for path in self.annotation_files():
            yield path, self.detections(path)

    def images(self) -> list[Path]:
        """Image files found anywhere under the root."""
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)

    # ------------------------------------------------------------- contract

    def splits(self) -> list[str]:
        """No published split.

        MIaMIA-SVDS publishes no train/val/test division, and its 101 videos are
        the obvious grouping unit. Build a split with
        :func:`datasets.validators.leakage.patient_level_split` keyed on the
        video, never on the frame.
        """
        return []

    def __len__(self) -> int:
        """Number of annotation files (or annotations, for a COCO release)."""
        evidence = self.sniff_format()
        if evidence.format is AnnotationFormat.COCO_JSON:
            payload = json.loads(Path(evidence.extra["json_path"]).read_text())
            return len(payload.get("images", []))
        return len(self.annotation_files())

    def validate(self, *, sample: int = 20) -> ValidationReport:
        """Sniff the format, parse a sample, and report every unverified assumption."""
        report = self._new_report()
        report.context["subsets"] = {k: str(v) for k, v in self.subsets().items()}

        try:
            evidence = self.sniff_format()
        except DatasetValidationError as exc:
            report.add("format:sniff", CheckStatus.FAIL, str(exc))
            report.checks.append(self._unverified_note())
            return report

        report.context["format_evidence"] = evidence.to_json_dict()
        report.add(
            "format:sniff",
            CheckStatus.PASS,
            f"detected {evidence.format} from {evidence.n_files} file(s) in "
            f"{evidence.annotation_dir}: {'; '.join(evidence.evidence)}",
            **evidence.to_json_dict(),
        )
        if evidence.also_matched:
            report.add(
                "format:ambiguity",
                CheckStatus.WARN,
                f"other formats also matched, with few files each: "
                f"{[str(f) for f in evidence.also_matched]}. Verify the chosen format "
                "covers the whole release.",
            )

        files = self.annotation_files()
        report.checks.append(check_non_empty(len(files), name="annotations", what="annotation files"))

        n_boxes = 0
        class_ids: set[int] = set()
        parsed = 0
        for index, path in enumerate(files[: max(1, sample)]):
            try:
                detections = self.detections(path)
            except DatasetValidationError as exc:
                report.add("annotations:parse", CheckStatus.FAIL, f"{path}: {exc}")
                break
            except Exception as exc:
                report.add("annotations:parse", CheckStatus.FAIL, f"{path}: {exc!r}")
                break
            parsed = index + 1
            n_boxes += len(detections)
            class_ids.update(d.class_id for d in detections)
        else:
            report.add(
                "annotations:parse",
                CheckStatus.PASS,
                f"parsed {parsed} annotation file(s) yielding {n_boxes} box(es); class "
                f"ids seen: {sorted(class_ids)}",
                n_files_parsed=parsed,
                n_boxes=n_boxes,
                class_ids=sorted(class_ids),
            )

        unexpected = sorted(c for c in class_ids if c not in {0, 1})
        if unexpected:
            report.add(
                "classes",
                CheckStatus.FAIL,
                f"class id(s) {unexpected} outside the published two-class set "
                f"{list(SOURCE_CLASS_NAMES)} (0=S/sperm, 1=Impurity/debris)",
            )
        elif class_ids:
            report.add(
                "classes",
                CheckStatus.PASS,
                f"class ids {sorted(class_ids)} map onto {list(SOURCE_CLASS_NAMES)}",
            )

        n_images = len(self.images())
        report.add(
            "images",
            CheckStatus.PASS if n_images else CheckStatus.WARN,
            f"{n_images} image file(s) found under {self.root}",
            n_images=n_images,
        )

        report.checks.append(self._unverified_note())
        report.checks.append(self._licence_note())
        return report

    @staticmethod
    def _unverified_note() -> Any:
        from ..validators.integrity import CheckResult

        return CheckResult(
            name="provenance:unverified",
            status=CheckStatus.UNVERIFIABLE,
            message=(
                "Two properties of this release are unverified upstream and cannot be "
                "confirmed from the published material: (1) the on-disk annotation "
                "format -- LabelImg implies VOC XML at annotation time, but the "
                "contents of 'Data Set.rar' have not been confirmed, hence the sniffer; "
                "(2) the native video resolution -- the 416x416 in the TOD-CNN paper is "
                "a network input size after resizing. Spot-check boxes visually before "
                "training on them."
            ),
            details={"format_sniffed": True, "native_resolution_known": False},
        )

    @staticmethod
    def _licence_note() -> Any:
        from ..validators.integrity import CheckResult
        from ..validators.licenses import get_license

        record = get_license("detection_sperm")
        return CheckResult(
            name="licence:conflict",
            status=CheckStatus.UNVERIFIABLE,
            message=(
                "Licence terms conflict and this package does not resolve them: "
                + " | ".join(record.evidence)
                + ". check_commercial_use() reports this dataset as a blocker."
            ),
            details=record.to_json_dict(),
        )


# ==========================================================================
# helpers
# ==========================================================================


def _looks_like_yolo(line: str, classes: set[int]) -> bool:
    """Whether one line parses as ``class x_center y_center width height``.

    Requires an integral first field and four floats in ``[0, 1]``. A 6-field
    line is rejected: that is VISEM-Tracking's ``labels_ftid`` shape, and
    accepting it here would read a track id as a class.
    """
    parts = line.split()
    if len(parts) != 5:
        return False
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return False
    if values[0] != int(values[0]):
        return False
    if not all(0.0 <= v <= 1.0 for v in values[1:]):
        return False
    classes.add(int(values[0]))
    return True


def _frame_number(stem: str) -> int | None:
    match = _FRAME_NUMBER_RE.search(stem)
    return int(match.group(1)) if match else None
