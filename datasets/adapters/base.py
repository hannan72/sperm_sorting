"""The dataset adapter contract, and the metadata that makes public weights honest.

Why :class:`CaptureConditions` is a first-class part of dataset metadata
-----------------------------------------------------------------------
Every public sperm dataset in this repository was captured on somebody else's
microscope. MHSMA is stained, cropped, single-cell brightfield at x400/x600 from
an Iranian fertility clinic. VISEM-Tracking is 640x480 video of a live,
unstained wet preparation at 45-50 FPS from Oslo University Hospital.
MIaMIA-SVDS is 30 FPS CASA output through a 20x objective plus a 20x electronic
eyepiece. None of them is this device.

A model trained on them is a *baseline research model*: it demonstrates that the
task is learnable and gives a starting point for fine-tuning. It is not a device
model, and the difference is not a matter of degree -- illumination mode, optical
magnification, sensor pixel pitch, exposure and frame rate all change the
appearance of a 3-micrometre sperm head far more than most architectural
choices do.

Recording the capture conditions next to the loader is what makes that claim
checkable instead of rhetorical. When device data arrives, the gap between
:class:`CaptureConditions` for the public set and for the device set *is* the
domain shift, enumerated field by field, and :attr:`DatasetInfo.domain_shift_notes`
says what each gap is expected to do. ``constants.WEIGHTS_PROVENANCE_PUBLIC``
exists for the same reason at the checkpoint level.

The absence contract
--------------------
Public datasets are not redistributed in this repository (several forbid it, all
are large). So every adapter must be importable, and must fail *usefully*,
without any data on disk:

* constructing an adapter against a missing root raises
  :class:`~sperm_sorting.errors.DatasetNotFoundError` naming **the path it
  looked at and the URL to download from**;
* :meth:`DatasetAdapter.describe` is a classmethod and needs no data at all, so
  licence audits and documentation generation work on a bare checkout;
* ``require_present=False`` constructs a metadata-only adapter whose data
  accessors still raise the same error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from sperm_sorting.errors import DatasetNotFoundError

from ..validators.integrity import ValidationReport
from ..validators.licenses import CommercialUse, LicenseRecord, get_license

__all__ = [
    "CaptureConditions",
    "DatasetAdapter",
    "DatasetInfo",
]


@dataclass(frozen=True, slots=True)
class CaptureConditions:
    """How the images were physically produced.

    Every field is ``None``-able and ``None`` means **unknown**, never "not
    applicable" and never a default. An unknown magnification recorded as
    ``None`` shows up in a domain-shift table as a hole, which is a fact; an
    unknown magnification recorded as ``40.0`` is a fabrication that later
    justifies a scale assumption nobody checked.
    """

    #: Objective magnification, e.g. 20.0 for a 20x objective.
    objective_magnification: float | None = None
    #: Total optical magnification where the release states one (MIaMIA-SVDS
    #: pairs a 20x objective with a 20x electronic eyepiece, so the two numbers
    #: are genuinely different and both matter).
    total_magnification: float | None = None
    #: "brightfield", "phase-contrast", "DIC", ... Free text: the releases do
    #: not agree on a vocabulary and normalising it would lose information.
    contrast_mode: str | None = None
    #: Whether the sample was stained. Staining changes head contrast more than
    #: any other single variable, and live sorting cannot stain.
    stained: bool | None = None
    #: Camera / CASA system, as named upstream.
    camera: str | None = None
    #: Frame rate as ``(min, max)``. A range because VISEM-Tracking's is
    #: genuinely 45-50 FPS and *not uniform across videos*; collapsing that to a
    #: single number is how per-frame velocities acquire a silent 10% error.
    fps_range: tuple[float, float] | None = None
    #: ``True`` only when the release states a constant rate. ``None`` = unknown.
    fps_uniform: bool | None = None
    #: Image size ``(width, height)`` in pixels, when fixed by the release.
    resolution: tuple[int, int] | None = None
    #: Micrometres per pixel, when published. Almost never is -- which is
    #: exactly why physical velocities from public data are not trustworthy.
    um_per_px: float | None = None
    #: Anything else worth carrying into a domain-shift discussion.
    notes: str = ""

    @property
    def fps_nominal(self) -> float | None:
        """Midpoint of :attr:`fps_range`, for display only.

        Explicitly *not* for computing velocities: use per-video timestamps.
        """
        if self.fps_range is None:
            return None
        lo, hi = self.fps_range
        return 0.5 * (lo + hi)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "objective_magnification": self.objective_magnification,
            "total_magnification": self.total_magnification,
            "contrast_mode": self.contrast_mode,
            "stained": self.stained,
            "camera": self.camera,
            "fps_range": list(self.fps_range) if self.fps_range else None,
            "fps_uniform": self.fps_uniform,
            "resolution": list(self.resolution) if self.resolution else None,
            "um_per_px": self.um_per_px,
            "notes": self.notes,
        }

    def differences_from(self, other: CaptureConditions) -> list[str]:
        """Field-by-field description of how ``self`` differs from ``other``.

        This is the mechanical half of a domain-shift assessment: point it at
        (public dataset, device) and it enumerates every axis on which the
        training distribution is not the deployment distribution. Fields that
        are unknown on either side are reported as unknown rather than skipped,
        because an unmeasured difference is still a difference.
        """
        out: list[str] = []
        pairs: tuple[tuple[str, Any, Any], ...] = (
            ("objective magnification", self.objective_magnification, other.objective_magnification),
            ("total magnification", self.total_magnification, other.total_magnification),
            ("contrast mode", self.contrast_mode, other.contrast_mode),
            ("staining", self.stained, other.stained),
            ("camera", self.camera, other.camera),
            ("frame rate", self.fps_range, other.fps_range),
            ("resolution", self.resolution, other.resolution),
            ("um per pixel", self.um_per_px, other.um_per_px),
        )
        for label, mine, theirs in pairs:
            if mine is None or theirs is None:
                out.append(f"{label}: unknown on {'this' if mine is None else 'the other'} side "
                           f"({mine!r} vs {theirs!r}) -- difference unmeasurable")
            elif mine != theirs:
                out.append(f"{label}: {mine!r} vs {theirs!r}")
        return out


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Everything about a dataset that is true before any file is opened.

    The licence is *referenced*, not copied: :attr:`license_key` indexes
    :data:`datasets.validators.licenses.LICENSES` so that there is exactly one
    place in the repository where a licence is written down, and a CI licence
    check and an adapter can never disagree about it.
    """

    #: Registry key, e.g. ``"visem_tracking"``. Matches ``get_adapter(name)``.
    name: str
    #: Human-facing title.
    title: str
    #: Where to obtain it.
    url: str
    #: Key into the licence registry.
    license_key: str
    capture: CaptureConditions
    #: What kind of supervision it carries -- the field that decides whether a
    #: dataset can train a detector, a morphology head, or nothing at all.
    annotation_level: str = "unknown"
    #: One line per expected failure mode when a model trained on this data
    #: meets device data. Written from the capture conditions, not from vibes.
    domain_shift_notes: list[str] = field(default_factory=list)
    #: Expected on-disk layout, shown verbatim in DatasetNotFoundError messages.
    expected_layout: str = ""
    #: Size on disk, for the "why is this not in the repo" conversation.
    approximate_size: str = "unknown"

    @property
    def license_record(self) -> LicenseRecord:
        return get_license(self.license_key)

    @property
    def license(self) -> str:
        """Licence name as published upstream."""
        return self.license_record.license_name

    @property
    def commercial_use_permitted(self) -> bool:
        """Fail-closed: ``UNCLEAR`` licences read as *not* permitted."""
        return self.license_record.commercial_use_permitted

    @property
    def commercial_use(self) -> CommercialUse:
        """The three-state answer, for callers that must distinguish
        "forbidden" from "nobody has said"."""
        return self.license_record.commercial_use

    @property
    def share_alike(self) -> bool:
        return self.license_record.share_alike

    @property
    def citation(self) -> str:
        return self.license_record.citation

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "url": self.url,
            "annotation_level": self.annotation_level,
            "license": self.license,
            "license_key": self.license_key,
            "commercial_use": str(self.commercial_use),
            "commercial_use_permitted": self.commercial_use_permitted,
            "share_alike": self.share_alike,
            "citation": self.citation,
            "capture": self.capture.to_json_dict(),
            "domain_shift_notes": list(self.domain_shift_notes),
            "expected_layout": self.expected_layout,
            "approximate_size": self.approximate_size,
        }


class DatasetAdapter(ABC):
    """Base class for every dataset reader in this package.

    Subclasses set :attr:`info` and implement :meth:`splits`, :meth:`__len__`
    and :meth:`validate`. They get, for free, a root-resolution and
    absence-reporting protocol whose error messages always name both the path
    that was searched and the URL to download from -- because the two questions
    a person has when this fails are "where did it look" and "where do I get it".

    Parameters
    ----------
    root
        Directory holding the dataset. What exactly it should point at is
        subclass-specific and always documented in
        :attr:`DatasetInfo.expected_layout`; adapters generally accept either
        the archive's top directory or its parent.
    require_present
        When True (default) the constructor verifies the dataset is on disk and
        raises :class:`~sperm_sorting.errors.DatasetNotFoundError` if not. Set
        False to build a metadata-only adapter (documentation, licence audit);
        data accessors then raise the same error when they are actually called.
    """

    #: Class-level metadata. Set by every concrete subclass.
    info: ClassVar[DatasetInfo]

    #: Set True by adapters that carry no per-object annotation (VISEM). The
    #: training code checks it before wiring a dataset into a detector.
    sample_level_only: ClassVar[bool] = False

    def __init__(self, root: str | Path, *, require_present: bool = True) -> None:
        self._given_root = Path(root).expanduser()
        self._root: Path | None = None
        self._require_present = require_present
        resolved = self._resolve_root(self._given_root)
        if resolved is None:
            if require_present:
                raise self.not_found_error(self._given_root)
        else:
            self._root = resolved

    # ------------------------------------------------------------- metadata

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Full metadata without touching the filesystem."""
        return cls.info.to_json_dict()

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def license(self) -> str:
        return self.info.license

    @property
    def commercial_use_permitted(self) -> bool:
        return self.info.commercial_use_permitted

    @property
    def citation(self) -> str:
        return self.info.citation

    @property
    def capture(self) -> CaptureConditions:
        return self.info.capture

    @property
    def domain_shift_notes(self) -> list[str]:
        return list(self.info.domain_shift_notes)

    @property
    def root(self) -> Path:
        """The resolved dataset root.

        Raises
        ------
        DatasetNotFoundError
            When the adapter was built with ``require_present=False`` and the
            data is genuinely absent. Accessing data must fail the same way
            whenever the data is missing, whatever the constructor was told.
        """
        if self._root is None:
            raise self.not_found_error(self._given_root)
        return self._root

    @property
    def available(self) -> bool:
        """True when the dataset was found on disk. Never raises."""
        return self._root is not None

    # ------------------------------------------------------------- contract

    @abstractmethod
    def splits(self) -> list[str]:
        """Canonical split names this dataset offers, in a stable order."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of primary items -- images, annotated frames, or participants.

        Subclasses document which, because "length" is not the same unit across
        an image classification set and a video tracking set.
        """

    @abstractmethod
    def validate(self) -> ValidationReport:
        """Check this copy against everything known about the dataset."""

    # ------------------------------------------------------------ resolution

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """Find the real dataset root under ``given``, or ``None``.

        Default: ``given`` itself if it is a directory. Subclasses override to
        accept the several places an archive can plausibly be unpacked (the
        archive's own top directory, its parent, a nested ``Train`` folder),
        because forcing a user to guess which of three equally reasonable paths
        the code wanted is a bad first five minutes.

        Every override also checks for a *signature* of the dataset -- one of
        MHSMA's ``x_*.npy``, a numerically-named video folder, a CSV -- rather
        than accepting any directory that happens to exist. An adapter built on
        an empty directory would otherwise report zero items, which reads as "the
        dataset is empty" when the truth is "that is the wrong path".
        """
        return given if given.is_dir() else None

    @classmethod
    def not_found_error(cls, searched: Path) -> DatasetNotFoundError:
        """Build the canonical "you do not have this dataset" error.

        Always contains the searched path, the download URL, the expected
        layout and the licence -- the licence because several of these datasets
        cannot legally be redistributed and a user's next thought is usually
        "why is it not just in the repo".
        """
        info = cls.info
        layout = f"\nExpected layout:\n{info.expected_layout}" if info.expected_layout else ""
        return DatasetNotFoundError(
            f"{info.title} was not found at: {searched}\n"
            f"Download it from: {info.url}\n"
            f"Licence: {info.license} "
            f"(commercial use: {info.commercial_use}); approximate size: "
            f"{info.approximate_size}.{layout}\n"
            "No dataset is redistributed in this repository -- see datasets/README.md."
        )

    def require_path(self, path: Path, what: str) -> Path:
        """Return ``path`` or raise a :class:`DatasetNotFoundError` naming ``what``.

        Used for individual files inside a present root, so that "you have the
        dataset but the acrosome labels for the valid split are missing" is a
        distinguishable message from "you have no dataset".
        """
        if not path.exists():
            raise DatasetNotFoundError(
                f"{self.info.title}: missing {what} at {path}\n"
                f"Re-download from {self.info.url}; the copy under "
                f"{self._root or self._given_root} is incomplete."
            )
        return path

    def missing_paths(self, paths: Iterable[Path]) -> list[Path]:
        """Subset of ``paths`` that do not exist. For batch reporting."""
        return [p for p in paths if not p.exists()]

    # -------------------------------------------------------------- helpers

    def _new_report(self) -> ValidationReport:
        return ValidationReport(
            dataset=self.info.name,
            root=self._root or self._given_root,
            context={
                "license": self.info.license,
                "commercial_use": str(self.info.commercial_use),
                "url": self.info.url,
                "available": self.available,
            },
        )

    def __repr__(self) -> str:
        state = str(self._root) if self._root is not None else f"<absent: {self._given_root}>"
        return f"{type(self).__name__}(root={state!r})"
