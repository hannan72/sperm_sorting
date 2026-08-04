"""VISEM: 85 videos with **sample-level** annotations only.

Source: Zenodo record 2640506 -- Haugen et al., "VISEM: A Multimodal Video
Dataset of Human Spermatozoa", MMSys'19. Licence CC BY-NC 4.0 (non-commercial).
85 participants, one 2-7 minute video each, 640x480, 50 FPS, AVI, ~35.2 GB.

What this dataset does and does not contain
-------------------------------------------
It contains, in six CSV files: WHO semen analysis results, motility percentages
(progressive / non-progressive / immotile), sperm concentration, fatty acid
profiles of serum and spermatozoa, sex hormone levels, and participant
demographics -- all **per sample**, i.e. one row per participant.

It contains **no bounding boxes, no per-sperm labels, no track identifiers, and
no morphology labels for individual cells.**

Why that restriction is enforced in code
----------------------------------------
There is an obvious and completely wrong way to use this dataset: take a video
whose sample is "62% progressive", detect the sperm in it, and label 62% of the
detections progressive. Every individual label so produced is fabricated. The
per-sperm assignment is unknown -- the percentage constrains only the aggregate
-- and a model trained on such labels learns the *base rate* of the sample it
came from, then reports that base rate confidently for individual cells in a
different sample. In a device that physically sorts cells, that is a machine
that has learned to guess.

The same objection applies to weak/MIL-style formulations dressed up as
principled: they are defensible research, but they produce *sample-level*
predictions, and this repository's decision rule
(:meth:`sperm_sorting.schemas.track.TrackRecord.compute_eligibility`) is
per-sperm. So:

* :attr:`VisemAdapter.sample_level_only` is ``True``, and training code checks
  it before wiring a dataset into anything per-object;
* this adapter exposes **no** per-sperm interface, and asking for one by name
  raises an ``AttributeError`` that explains why rather than a bare
  ``AttributeError`` that reads like an oversight.

What VISEM *is* good for: sanity-checking that aggregate statistics produced by
the pipeline (progressive fraction over a whole video) land in the right
neighbourhood for a sample whose laboratory-measured value is known. That is a
real and useful validation, and it is aggregate-to-aggregate, which is the only
comparison the data supports.

Column names are matched, not assumed
-------------------------------------
The exact CSV filenames and column headers of the Zenodo release are not
reproduced here from memory. Files are located by fuzzy stem matching and
columns by keyword matching, both reported in :meth:`VisemAdapter.validate`, and
a column that cannot be found raises with the list of columns that *are*
present. Guessing a header name and silently returning the wrong column is
exactly the failure this module exists to prevent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from sperm_sorting.errors import DatasetValidationError

from ..validators.integrity import CheckStatus, ValidationReport, check_non_empty
from .base import CaptureConditions, DatasetAdapter, DatasetInfo

__all__ = ["MotilityPercentages", "VisemAdapter"]

#: Logical CSV name -> candidate filename stems (matched case-insensitively,
#: after stripping non-alphanumerics). The release ships six CSVs; these are the
#: stems seen in the Zenodo record, with plausible variants, because the exact
#: spelling has not been verified against a downloaded copy.
_CSV_SPECS: Final[dict[str, tuple[str, ...]]] = {
    "semen_analysis": ("semenanalysisdata", "semenanalysis", "semen"),
    "participant": ("participantrelateddata", "participantrelated", "participants", "participant"),
    "sex_hormones": ("sexhormones", "hormones"),
    "fatty_acids_spermatozoa": ("fattyacidsspermatozoa", "fattyacidsperm"),
    "fatty_acids_serum": ("fattyacidsserum", "fattyacidserum"),
    "videos": ("videos", "videodata", "video"),
}

#: Keyword sets for locating WHO columns. Matched against a normalised header
#: (lowercased, non-alphanumerics removed). Order matters: the more specific
#: pattern must come first, or "progressive motility" matches
#: "non-progressive motility" too.
_MOTILITY_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "non_progressive": ("nonprogressive", "nonprogressivemotility", "npmotility"),
    "immotile": ("immotile", "immotility"),
    "progressive": ("progressivemotility", "progressive", "prmotility"),
}

_WHO_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "concentration": ("spermconcentration", "concentration"),
    "volume": ("ejaculatevolume", "semenvolume", "volume"),
    "total_count": ("totalspermcount", "spermcount", "totalcount"),
    "normal_morphology_pct": ("normalmorphology", "morphologynormal", "morphology"),
    "vitality_pct": ("vitality", "viability"),
    "ph": ("ph",),
}

_PARTICIPANT_ID_PATTERNS: Final[tuple[str, ...]] = ("id", "participantid", "subjectid", "sampleid")

#: Attribute names this adapter refuses to provide, with the reason. Asking for
#: any of them is asking for per-sperm data that does not exist.
_FORBIDDEN_ATTRS: Final[dict[str, str]] = {
    "detections": "bounding boxes",
    "boxes": "bounding boxes",
    "annotations": "per-object annotations",
    "tracks": "per-sperm tracks",
    "track_ids": "per-sperm track identifiers",
    "crops": "per-sperm crops",
    "labels": "per-sperm labels",
    "per_sperm_labels": "per-sperm labels",
    "morphology": "per-sperm morphology labels",
    "frames": "per-frame annotations",
    "iter_frames": "per-frame annotations",
}


def _normalise(text: str) -> str:
    """Lowercase and strip everything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


@dataclass(frozen=True, slots=True)
class MotilityPercentages:
    """The three WHO motility fractions for one sample, as percentages.

    ``total`` is carried so a caller can see whether the row is internally
    consistent. The three are *supposed* to sum to 100; laboratory rounding
    means they often sum to 99 or 101, and a row that sums to 60 is a parse
    error rather than an unusual patient. Nothing is renormalised here --
    a silently rescaled percentage is a fabricated measurement.
    """

    participant_id: str
    progressive: float
    non_progressive: float
    immotile: float

    @property
    def total(self) -> float:
        return self.progressive + self.non_progressive + self.immotile

    @property
    def consistent(self) -> bool:
        """Whether the three fractions sum to 100 within 2 points."""
        return abs(self.total - 100.0) <= 2.0

    @property
    def motile(self) -> float:
        """Total motile percentage = progressive + non-progressive."""
        return self.progressive + self.non_progressive

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "progressive_pct": self.progressive,
            "non_progressive_pct": self.non_progressive,
            "immotile_pct": self.immotile,
            "total_pct": self.total,
            "consistent": self.consistent,
        }


class VisemAdapter(DatasetAdapter):
    """Reader for VISEM's sample-level CSVs. **No per-sperm interface exists.**

    Parameters
    ----------
    root
        Directory containing the CSV files (and, optionally, the videos).
    require_present
        See :class:`~datasets.adapters.base.DatasetAdapter`.

    Notes
    -----
    Fabricating per-sperm labels from these sample-level percentages is **not
    supported**, and is not an omission to be filled in later. A sample that is
    "62% progressive" says nothing about *which* sperm are progressive; assigning
    that label to individual cells invents 100% of the per-cell supervision and
    trains a model to reproduce a base rate. Check
    :attr:`sample_level_only` before using any dataset in a per-object training
    loop.
    """

    sample_level_only = True

    info = DatasetInfo(
        name="visem",
        title="VISEM: a multimodal video dataset of human spermatozoa",
        url="https://zenodo.org/records/2640506",
        license_key="visem",
        annotation_level="SAMPLE-LEVEL ONLY (one row per participant; no boxes, no per-sperm labels)",
        approximate_size="35.2 GB (85 AVI videos plus six CSV files)",
        capture=CaptureConditions(
            objective_magnification=400.0,
            total_magnification=400.0,
            contrast_mode="brightfield, unstained wet preparation",
            stained=False,
            camera="UEye UI-2210C camera on an Olympus CX31 microscope",
            fps_range=(50.0, 50.0),
            fps_uniform=True,
            resolution=(640, 480),
            um_per_px=None,
            notes=(
                "85 participants, one 2-7 minute video each. VISEM-Tracking's 20 "
                "annotated clips are drawn from this same acquisition setup, so the "
                "two share optics -- and share the same distance from this device."
            ),
        ),
        domain_shift_notes=[
            "No per-object annotation of any kind, so this dataset cannot train or "
            "validate detection, tracking or morphology. It is an aggregate "
            "cross-check, nothing more.",
            "640x480 at x400 on a clinical brightfield microscope: a different "
            "optical chain from the device, with the same low-resolution limitation "
            "as VISEM-Tracking.",
            "Laboratory motility percentages were measured by a technician under WHO "
            "protocol on a stationary chamber, not by tracking in a flow. Comparing "
            "them with pipeline output compares two different measurements of the "
            "same underlying quantity, and the difference between the methods is "
            "part of the disagreement.",
        ],
        expected_layout=(
            "  <root>/semen_analysis_data.csv        (WHO parameters, one row per participant)\n"
            "  <root>/participant_related_data.csv   (demographics)\n"
            "  <root>/sex_hormones.csv\n"
            "  <root>/fatty_acids_spermatozoa.csv\n"
            "  <root>/fatty_acids_serum.csv\n"
            "  <root>/videos.csv\n"
            "  <root>/videos/*.avi                   (optional; 35 GB)\n"
            "  (filenames are matched fuzzily -- exact spellings vary by download)"
        ),
    )

    def __init__(self, root: str | Path, *, require_present: bool = True) -> None:
        self._tables: dict[str, Any] = {}
        self._csv_paths: dict[str, Path] | None = None
        super().__init__(root, require_present=require_present)

    # ------------------------------------------------------------ discovery

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """First candidate directory that actually contains a CSV.

        VISEM *is* its CSVs as far as this adapter is concerned (the 35 GB of
        AVI carries no annotation), so a directory with none of them is not a
        VISEM copy and the constructor should say so rather than produce an
        adapter with zero participants.
        """
        for candidate in (given, given / "visem", given / "VISEM"):
            if candidate.is_dir() and any(candidate.rglob("*.csv")):
                return candidate
        return None

    def csv_paths(self) -> dict[str, Path]:
        """Logical name -> path, for every CSV that could be located.

        Missing entries are simply absent from the mapping; :meth:`validate`
        reports which of the six were not found.
        """
        if self._csv_paths is not None:
            return self._csv_paths
        by_stem = {
            _normalise(p.stem): p
            for p in sorted(self.root.rglob("*.csv"))
            if p.is_file()
        }
        found: dict[str, Path] = {}
        for logical, candidates in _CSV_SPECS.items():
            for candidate in candidates:
                exact = by_stem.get(candidate)
                if exact is not None:
                    found[logical] = exact
                    break
            else:
                # Fall back to substring containment, longest stem first so that
                # "fatty_acids_serum" is not swallowed by a bare "fatty" rule.
                for stem, path in sorted(by_stem.items(), key=lambda kv: -len(kv[0])):
                    if any(candidate in stem for candidate in candidates):
                        found[logical] = path
                        break
        self._csv_paths = found
        return found

    def table(self, name: str) -> Any:
        """Load one CSV as a :class:`pandas.DataFrame`, cached.

        Raises
        ------
        DatasetNotFoundError
            When that CSV is not present in this copy, naming what was searched.
        """
        if name not in _CSV_SPECS:
            raise ValueError(f"unknown VISEM table {name!r}; expected one of {sorted(_CSV_SPECS)}")
        if name in self._tables:
            return self._tables[name]
        paths = self.csv_paths()
        if name not in paths:
            found = sorted(paths)
            # require_path always raises here: the constructed path is a
            # placeholder that names what was wanted, not a real candidate.
            self.require_path(
                self.root / f"{name}.csv",
                f"the '{name}' CSV (searched every *.csv under the root; found: {found})",
            )
        import pandas as pd

        frame = pd.read_csv(paths[name])
        self._tables[name] = frame
        return frame

    # ----------------------------------------------------------- participants

    def _id_column(self, frame: Any) -> str:
        """Find the participant-identifier column, or raise listing the headers."""
        normalised = {_normalise(c): c for c in frame.columns}
        for pattern in _PARTICIPANT_ID_PATTERNS:
            if pattern in normalised:
                return normalised[pattern]
        for key, original in normalised.items():
            if key.endswith("id"):
                return original
        raise DatasetValidationError(
            "VISEM: no participant identifier column found. Columns present: "
            f"{list(frame.columns)}. Expected one matching {list(_PARTICIPANT_ID_PATTERNS)}."
        )

    def participants(self) -> list[str]:
        """Participant identifiers, as strings, sorted."""
        frame = self.table("semen_analysis")
        column = self._id_column(frame)
        return sorted({str(v) for v in frame[column].tolist()})

    def participant(self, participant_id: str | int) -> dict[str, Any]:
        """Every sample-level field for one participant, merged across the CSVs.

        Values are returned verbatim, with the source table recorded in the key
        prefix, so a caller can always tell which CSV a number came from.
        """
        wanted = str(participant_id)
        merged: dict[str, Any] = {"participant_id": wanted}
        for name in self.csv_paths():
            frame = self.table(name)
            try:
                column = self._id_column(frame)
            except DatasetValidationError:
                continue
            rows = frame[frame[column].astype(str) == wanted]
            if rows.empty:
                continue
            row = rows.iloc[0].to_dict()
            for key, value in row.items():
                merged[f"{name}.{key}"] = value
        if len(merged) == 1:
            raise DatasetValidationError(
                f"VISEM: participant {wanted!r} does not appear in any loaded CSV. "
                f"Known participants: {self.participants()[:20]}..."
            )
        return merged

    def motility(self, participant_id: str | int) -> MotilityPercentages:
        """The three WHO motility percentages for one participant.

        Raises
        ------
        DatasetValidationError
            When a required column cannot be located, listing the headers that
            *are* present. It does not fall back to a default or to zero: a
            fabricated motility percentage is a fabricated clinical measurement.
        """
        frame = self.table("semen_analysis")
        column = self._id_column(frame)
        rows = frame[frame[column].astype(str) == str(participant_id)]
        if rows.empty:
            raise DatasetValidationError(
                f"VISEM: participant {participant_id!r} not present in the semen analysis table"
            )
        row = rows.iloc[0]
        values: dict[str, float] = {}
        # _MOTILITY_PATTERNS is ordered most-specific-first and each match is
        # excluded from later ones. Without that, a table with headers
        # "progressive_motility" and "non_progressive_motility" returns the same
        # column twice -- the substring "progressive" is in both -- and the
        # immotile fraction silently absorbs the difference.
        claimed: list[str] = []
        for key, patterns in _MOTILITY_PATTERNS.items():
            column_name = _match_column(frame.columns, patterns, exclude=claimed)
            if column_name is None:
                raise DatasetValidationError(
                    f"VISEM: cannot locate the '{key}' motility column. Columns present: "
                    f"{list(frame.columns)}. Patterns tried: {list(patterns)}. Refusing to "
                    "guess -- a wrong column here silently mislabels every sample."
                )
            claimed.append(column_name)
            values[key] = float(row[column_name])
        return MotilityPercentages(
            participant_id=str(participant_id),
            progressive=values["progressive"],
            non_progressive=values["non_progressive"],
            immotile=values["immotile"],
        )

    def who_parameters(self, participant_id: str | int) -> dict[str, float | None]:
        """WHO semen-analysis parameters for one participant.

        A parameter whose column cannot be located comes back as ``None`` --
        explicitly absent, never zero and never imputed. Unlike
        :meth:`motility` this does not raise, because which optional WHO
        parameters a release carries varies and a missing pH should not stop a
        caller who wanted concentration.
        """
        frame = self.table("semen_analysis")
        column = self._id_column(frame)
        rows = frame[frame[column].astype(str) == str(participant_id)]
        if rows.empty:
            raise DatasetValidationError(
                f"VISEM: participant {participant_id!r} not present in the semen analysis table"
            )
        row = rows.iloc[0]
        out: dict[str, float | None] = {}
        for key, patterns in _WHO_PATTERNS.items():
            column_name = _match_column(frame.columns, patterns)
            if column_name is None:
                out[key] = None
                continue
            try:
                out[key] = float(row[column_name])
            except (TypeError, ValueError):
                out[key] = None
        return out

    def summary(self) -> dict[str, Any]:
        """Participant count and the tables that were located."""
        return {
            "info": self.info.to_json_dict(),
            "n_participants": len(self.participants()),
            "tables_found": {k: str(v) for k, v in self.csv_paths().items()},
            "tables_missing": sorted(set(_CSV_SPECS) - set(self.csv_paths())),
            "sample_level_only": self.sample_level_only,
        }

    # ------------------------------------------------------------- contract

    def splits(self) -> list[str]:
        """VISEM publishes no splits at all.

        Returns an empty list rather than inventing ``["train", "val"]``: any
        split of these 85 participants is one you made, and it must be built
        with :func:`datasets.validators.leakage.patient_level_split` keyed on
        the participant id, because one participant contributes one video and
        many frames.
        """
        return []

    def __len__(self) -> int:
        """Number of participants -- the only unit this dataset counts in."""
        return len(self.participants())

    def validate(self) -> ValidationReport:
        """Check the CSVs are present and readable and the percentages consistent."""
        report = self._new_report()
        paths = self.csv_paths()
        missing = sorted(set(_CSV_SPECS) - set(paths))
        report.context["tables_found"] = {k: str(v) for k, v in paths.items()}

        report.checks.append(check_non_empty(len(paths), name="csv:found", what="CSV files"))
        if missing:
            report.add(
                "csv:complete",
                CheckStatus.WARN,
                f"{len(missing)} of the six CSVs were not located: {missing}. Filenames are "
                "matched fuzzily; if your copy spells them differently, the data is fine "
                "but these tables are unavailable.",
                missing=missing,
            )
        else:
            report.add("csv:complete", CheckStatus.PASS, "all six CSVs located")

        if "semen_analysis" not in paths:
            report.add(
                "csv:semen_analysis",
                CheckStatus.FAIL,
                "the semen analysis CSV is the one table this adapter cannot work "
                f"without, and it was not found under {self.root}",
            )
            report.checks.append(self._sample_level_note())
            return report

        try:
            participants = self.participants()
        except Exception as exc:
            report.add("participants", CheckStatus.FAIL, f"cannot read participants: {exc!r}")
            report.checks.append(self._sample_level_note())
            return report

        report.checks.append(
            check_non_empty(len(participants), name="participants", what="participants")
        )
        if len(participants) != 85:
            report.add(
                "participants:count",
                CheckStatus.WARN,
                f"found {len(participants)} participants; the release has 85",
                n=len(participants),
            )
        else:
            report.add("participants:count", CheckStatus.PASS, "85 participants present")

        inconsistent: list[str] = []
        unreadable = 0
        for pid in participants:
            try:
                percentages = self.motility(pid)
            except DatasetValidationError:
                unreadable += 1
                continue
            if not percentages.consistent:
                inconsistent.append(f"{pid} (sums to {percentages.total:.1f})")
        if unreadable:
            report.add(
                "motility:columns",
                CheckStatus.FAIL,
                f"motility columns could not be located for {unreadable} participant(s); "
                "see the exception message from motility() for the available headers",
            )
        elif inconsistent:
            report.add(
                "motility:sums_to_100",
                CheckStatus.WARN,
                f"{len(inconsistent)} participant(s) have motility percentages that do not "
                f"sum to 100 +/- 2: {inconsistent[:10]}. Not renormalised here -- rescaling "
                "a clinical measurement to make it tidy is fabrication.",
                participants=inconsistent[:50],
            )
        else:
            report.add(
                "motility:sums_to_100",
                CheckStatus.PASS,
                "every participant's motility percentages sum to 100 +/- 2",
            )

        report.checks.append(self._sample_level_note())
        return report

    @staticmethod
    def _sample_level_note() -> Any:
        from ..validators.integrity import CheckResult

        return CheckResult(
            name="annotation_level",
            status=CheckStatus.UNVERIFIABLE,
            message=(
                "VISEM carries SAMPLE-LEVEL annotations only: WHO parameters and motility "
                "percentages per participant, with no bounding boxes and no per-sperm "
                "labels. Per-sperm supervision cannot be verified because it does not "
                "exist. Deriving per-sperm labels from these percentages is not supported "
                "-- see the module docstring."
            ),
            details={"sample_level_only": True},
        )

    # ----------------------------------------------- the refusal, made legible

    def __getattr__(self, name: str) -> Any:
        """Explain, rather than merely fail, when asked for per-sperm data.

        ``AttributeError`` is still the exception type -- ``hasattr`` and
        ``copy`` depend on that -- but for the specific names that mean "give me
        per-object annotations" the message says why they will never exist,
        which is more useful than the reader concluding the method was simply
        forgotten and writing it themselves.
        """
        reason = _FORBIDDEN_ATTRS.get(name)
        if reason is not None:
            raise AttributeError(
                f"VisemAdapter has no {name!r}: VISEM contains no {reason}. Its "
                "annotations are sample-level (one row per participant). Deriving "
                "per-sperm labels from sample-level percentages fabricates every "
                "individual label and is not supported. Use VISEM-Tracking "
                "(get_adapter('visem_tracking')) for per-sperm boxes and track IDs."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


# ==========================================================================
# helpers
# ==========================================================================


def _match_column(
    columns: Iterable[str], patterns: Sequence[str], exclude: Sequence[str] = ()
) -> str | None:
    """First column whose normalised name contains one of ``patterns``.

    Exact matches win over substring matches, so a header literally named
    ``progressive`` is chosen over ``non_progressive`` even though the latter
    also contains the substring. ``exclude`` holds columns already claimed by an
    earlier, more specific pattern.
    """
    normalised = [(c, _normalise(c)) for c in columns if c not in exclude]
    for pattern in patterns:
        for original, key in normalised:
            if key == pattern:
                return original
    for pattern in patterns:
        for original, key in normalised:
            if pattern in key:
                return original
    return None
