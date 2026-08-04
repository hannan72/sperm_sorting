"""Machine-readable licence registry for every dataset this repository can read.

Why this is code and not a paragraph in a README
------------------------------------------------
The licences here are not uniform and the differences are commercially
load-bearing:

* MHSMA is **CC BY-NC-SA 4.0** -- non-commercial *and* share-alike.
* VISEM (the 85-video sample-level set) is **CC BY-NC 4.0** -- non-commercial.
* VISEM-Tracking is **CC BY 4.0** -- commercial use permitted.
* MIaMIA-SVDS / Detection-Sperm is **contradictory**: no LICENSE file in the
  GitHub repository, a README that welcomes non-commercial research use, and
  figshare metadata tagged CC BY 4.0.

A model trained on a mixture inherits the strictest terms in the mixture, and
"which datasets went into this checkpoint" is a question that gets asked long
after the person who knows the answer has moved on. Encoding the terms next to
the loaders lets a training entry point call :func:`check_commercial_use` on the
list of datasets it is about to read and refuse, in CI, before the weights
exist.

Fail-closed on UNCLEAR
----------------------
:class:`CommercialUse` has three states and :func:`check_commercial_use` treats
``UNCLEAR`` as a blocker. Picking a side on Detection-Sperm's contradiction --
in either direction -- would be this file inventing a legal fact. Reporting the
contradiction and stopping is the only honest behaviour; a human with authority
to accept the risk can pass ``allow_unclear=True`` and that decision is then
visible in the call site rather than buried in a default.

**Nothing here is legal advice.** It is a transcription of what the upstream
sources state, with the source of each transcription recorded in
:attr:`LicenseRecord.evidence`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "LICENSES",
    "CommercialUse",
    "LicenseRecord",
    "check_commercial_use",
    "check_share_alike",
    "describe_licenses",
    "get_license",
    "strictest_terms",
]


class CommercialUse(str, Enum):
    """Whether a dataset's licence permits commercial use."""

    #: The licence explicitly permits commercial use (CC BY 4.0, MIT, ...).
    PERMITTED = "permitted"
    #: The licence explicitly forbids it (any CC *-NC-* variant).
    PROHIBITED = "prohibited"
    #: The published terms conflict or are absent. Treated as a blocker.
    UNCLEAR = "unclear"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    """The licence terms of one dataset, as published upstream."""

    #: Registry key; matches the adapter name in :mod:`datasets`.
    dataset: str
    #: Human-facing dataset title.
    title: str
    #: Licence name as stated upstream.
    license_name: str
    #: SPDX identifier where one exists, else ``None``. CC licences have SPDX
    #: ids (``CC-BY-NC-SA-4.0``); "unstated" does not.
    spdx_id: str | None
    commercial_use: CommercialUse
    #: True when redistribution of derivatives must carry the same licence.
    #: For CC BY-NC-SA this plausibly reaches trained weights; see
    #: :func:`check_share_alike`.
    share_alike: bool
    attribution_required: bool
    #: Where the licence statement was read from.
    url: str
    #: Canonical citation for the dataset.
    citation: str
    #: What was actually observed upstream, so a future reader can re-check
    #: rather than trusting this table.
    evidence: tuple[str, ...] = ()
    notes: str = ""
    #: Some releases license the data and the generator code differently.
    code_license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def commercial_use_permitted(self) -> bool:
        """Fail-closed boolean: only ``PERMITTED`` is true.

        ``UNCLEAR`` maps to *false* deliberately. A boolean that reads "we do
        not know" as "yes" is how an unlicensed dataset ends up in a product.
        """
        return self.commercial_use is CommercialUse.PERMITTED

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "title": self.title,
            "license_name": self.license_name,
            "spdx_id": self.spdx_id,
            "commercial_use": str(self.commercial_use),
            "commercial_use_permitted": self.commercial_use_permitted,
            "share_alike": self.share_alike,
            "attribution_required": self.attribution_required,
            "url": self.url,
            "citation": self.citation,
            "evidence": list(self.evidence),
            "notes": self.notes,
            "code_license": self.code_license,
            "extra": dict(self.extra),
        }


# ==========================================================================
# The registry
# ==========================================================================

LICENSES: dict[str, LicenseRecord] = {
    "mhsma": LicenseRecord(
        dataset="mhsma",
        title="MHSMA: Modified Human Sperm Morphology Analysis dataset",
        license_name="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
        spdx_id="CC-BY-NC-SA-4.0",
        commercial_use=CommercialUse.PROHIBITED,
        share_alike=True,
        attribution_required=True,
        url="https://github.com/soroushj/mhsma-dataset",
        citation=(
            "Javadi, S. and Mirroshandel, S.A. (2019). A novel deep learning method for "
            "automatic assessment of human sperm images. Computers in Biology and "
            "Medicine, 109, 182-194."
        ),
        evidence=(
            "LICENSE file in the soroushj/mhsma-dataset repository states "
            "CC BY-NC-SA 4.0.",
        ),
        notes=(
            "Non-commercial AND share-alike. The share-alike term is the one people "
            "forget: a checkpoint trained on MHSMA is arguably an adapted work, so "
            "redistributing those weights may require the same licence. Weights "
            "trained on this data are baseline research weights only."
        ),
    ),
    "visem_tracking": LicenseRecord(
        dataset="visem_tracking",
        title="VISEM-Tracking",
        license_name="Creative Commons Attribution 4.0 International",
        spdx_id="CC-BY-4.0",
        commercial_use=CommercialUse.PERMITTED,
        share_alike=False,
        attribution_required=True,
        url="https://zenodo.org/records/7293726",
        citation=(
            "Thambawita, V., Hicks, S.A., Storas, A.M. et al. (2023). VISEM-Tracking: "
            "a human spermatozoa tracking dataset. Scientific Data 10, 260. "
            "arXiv:2212.02842."
        ),
        evidence=("Zenodo record 7293726 states the licence as CC BY 4.0.",),
        notes=(
            "The only dataset here whose licence permits commercial use. It is also "
            "the only one with per-sperm bounding boxes and track IDs, which is why "
            "the detection/tracking baseline is built on it."
        ),
    ),
    "visem": LicenseRecord(
        dataset="visem",
        title="VISEM: a multimodal video dataset of human spermatozoa",
        license_name="Creative Commons Attribution-NonCommercial 4.0 International",
        spdx_id="CC-BY-NC-4.0",
        commercial_use=CommercialUse.PROHIBITED,
        share_alike=False,
        attribution_required=True,
        url="https://zenodo.org/records/2640506",
        citation=(
            "Haugen, T.B., Hicks, S.A., Andersen, J.M. et al. (2019). VISEM: A "
            "Multimodal Video Dataset of Human Spermatozoa. Proceedings of the 10th "
            "ACM Multimedia Systems Conference (MMSys'19), 261-266."
        ),
        evidence=("Zenodo record 2640506 states the licence as CC BY-NC 4.0.",),
        notes=(
            "Sample-level annotations only (WHO semen analysis, motility percentages, "
            "hormones, fatty acids). No bounding boxes, no per-sperm labels."
        ),
    ),
    "visem_graphs": LicenseRecord(
        dataset="visem_graphs",
        title="VISEM-Tracking-graphs (graph representations of VISEM-Tracking)",
        license_name="Creative Commons Attribution 4.0 International (data); MIT (code)",
        spdx_id="CC-BY-4.0",
        commercial_use=CommercialUse.PERMITTED,
        share_alike=False,
        attribution_required=True,
        url="https://huggingface.co/datasets/SimulaMet-HOST/visem-tracking-graphs",
        citation=(
            "SimulaMet-HOST. visem-tracking-graphs (Hugging Face dataset). Derived "
            "from VISEM-Tracking (Thambawita et al., Scientific Data 2023)."
        ),
        evidence=(
            "Hugging Face dataset card states CC BY 4.0 for the data and MIT for "
            "the generator code.",
        ),
        code_license="MIT",
        notes=(
            "Derived from VISEM-Tracking, so it inherits that dataset's attribution "
            "requirement in addition to its own. Optional extension; the MVP does "
            "not depend on it."
        ),
    ),
    "detection_sperm": LicenseRecord(
        dataset="detection_sperm",
        title="MIaMIA-SVDS / Detection-Sperm (TOD-CNN)",
        license_name="unstated (repository) vs CC BY 4.0 (figshare metadata)",
        spdx_id=None,
        commercial_use=CommercialUse.UNCLEAR,
        share_alike=False,
        attribution_required=True,
        url="https://github.com/Demozsj/Detection-Sperm",
        citation=(
            "Zhang, J. et al. TOD-CNN: An effective framework for tiny object "
            "detection in sperm videos with high object density. Dataset: "
            "MIaMIA-SVDS, figshare record 15074253."
        ),
        evidence=(
            "No LICENSE file is present in the Demozsj/Detection-Sperm repository.",
            "The repository README welcomes use for non-commercial research.",
            "The figshare record (15074253) carries CC BY 4.0 in its metadata.",
        ),
        notes=(
            "CONFLICT, deliberately left unresolved. A README sentence and a figshare "
            "metadata tag disagree about whether commercial use is permitted, and this "
            "registry is not the right place to decide which one governs. Resolve it "
            "with the authors in writing before any commercial use; until then "
            "check_commercial_use() reports it as a blocker."
        ),
        extra={"share_alike_known": False},
    ),
    "device": LicenseRecord(
        dataset="device",
        title="Device capture (data recorded on the instrument itself)",
        license_name="proprietary / internal",
        spdx_id=None,
        commercial_use=CommercialUse.PERMITTED,
        share_alike=False,
        attribution_required=False,
        url="(internal)",
        citation="(internal capture; see the capture metadata header of each file)",
        evidence=("Recorded by the operator of this instrument; no third-party terms.",),
        notes=(
            "The only corpus with no third-party licence constraint, and the only one "
            "captured under the device's own optics. Both facts are why the "
            "domain-adaptation path is built on it rather than on public data."
        ),
    ),
}


# ==========================================================================
# Queries
# ==========================================================================


def get_license(dataset: str) -> LicenseRecord:
    """Look up one dataset's licence record.

    Raises
    ------
    KeyError
        If the dataset is unknown. Deliberately not a ``None`` return: an
        unregistered dataset must not silently read as unconstrained.
    """
    try:
        return LICENSES[dataset]
    except KeyError:
        known = ", ".join(sorted(LICENSES))
        raise KeyError(
            f"no licence record for dataset {dataset!r}. Known datasets: {known}. "
            "Add a LicenseRecord in datasets/validators/licenses.py before using new "
            "data -- an unregistered dataset is not an unencumbered one."
        ) from None


def check_commercial_use(
    datasets: Iterable[str], *, allow_unclear: bool = False
) -> list[str]:
    """Return the blockers to commercial use of a model trained on ``datasets``.

    Parameters
    ----------
    datasets
        Registry keys (``"mhsma"``, ``"visem_tracking"``, ...).
    allow_unclear
        When True, ``UNCLEAR`` datasets are downgraded to a warning string
        prefixed ``UNCLEAR (accepted):`` instead of being reported as blockers.
        Pass it only where a human has accepted the risk *and* the call site
        records who; the default is to block.

    Returns
    -------
    A list of human-readable blocker strings. **Empty means no blockers**, so
    the natural ``if check_commercial_use(...):`` reads correctly.
    """
    blockers: list[str] = []
    for name in datasets:
        record = get_license(name)
        if record.commercial_use is CommercialUse.PROHIBITED:
            blockers.append(
                f"{record.dataset}: {record.license_name} forbids commercial use "
                f"({record.url})"
            )
        elif record.commercial_use is CommercialUse.UNCLEAR:
            detail = " | ".join(record.evidence)
            message = (
                f"{record.dataset}: licence terms are contradictory or absent, so "
                f"commercial use cannot be established. Evidence: {detail} ({record.url})"
            )
            if allow_unclear:
                blockers.append(f"UNCLEAR (accepted): {message}")
            else:
                blockers.append(message)
    return blockers


def check_share_alike(datasets: Iterable[str]) -> list[str]:
    """Return the share-alike obligations attaching to ``datasets``.

    Separate from :func:`check_commercial_use` because the two constraints bind
    differently: non-commercial stops you selling the model, share-alike stops
    you keeping the derivative closed. MHSMA is subject to both, and a plan that
    only ever checked the first would come apart at release.
    """
    return [
        f"{r.dataset}: {r.license_name} is share-alike; derivative works "
        f"(plausibly including trained weights) may have to carry the same licence "
        f"({r.url})"
        for r in (get_license(name) for name in datasets)
        if r.share_alike
    ]


def strictest_terms(datasets: Iterable[str]) -> LicenseRecord | None:
    """The record imposing the strictest terms in ``datasets``, or ``None``.

    Ordering, strictest first: ``UNCLEAR`` (unknown risk), then ``PROHIBITED``,
    then ``PERMITTED``; share-alike breaks ties. Useful for stamping "this
    checkpoint is governed by X" onto a weights file.
    """
    rank = {
        CommercialUse.UNCLEAR: 0,
        CommercialUse.PROHIBITED: 1,
        CommercialUse.PERMITTED: 2,
    }
    records = [get_license(name) for name in datasets]
    if not records:
        return None
    return min(records, key=lambda r: (rank[r.commercial_use], not r.share_alike))


def describe_licenses(datasets: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """JSON-able dump of the registry, for audit-log headers and CI artefacts."""
    names = list(datasets) if datasets is not None else sorted(LICENSES)
    return [get_license(name).to_json_dict() for name in names]
