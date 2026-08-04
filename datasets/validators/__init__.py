"""Dataset validators: integrity, split leakage and licence terms.

Three separable questions, three modules, deliberately no cross-imports between
them and the adapters:

:mod:`~datasets.validators.integrity`
    Is this copy of the data structurally what it claims to be? Files present,
    shapes and dtypes right, labels in range, nothing NaN. Defines
    :class:`~datasets.validators.integrity.ValidationReport`, which every
    adapter's ``validate()`` returns.
:mod:`~datasets.validators.leakage`
    Can this split produce a validation number that means anything? Group-level
    leakage, temporally adjacent frames across the boundary, and a grouped
    splitter that cannot leak by construction.
:mod:`~datasets.validators.licenses`
    May we use it, and under what terms? Fail-closed on unclear licences.

Keeping these free of adapter imports is what lets ``licenses.py`` be consulted
by CI without importing torch, and what keeps the import graph acyclic
(adapters import validators; validators import nothing from adapters).
"""

from __future__ import annotations

from .integrity import (
    CheckResult,
    CheckStatus,
    ValidationReport,
    check_array_finite,
    check_dir_present,
    check_file_present,
    check_label_range,
    check_non_empty,
    check_npy_header,
    npy_header,
)
from .leakage import (
    AdjacencyViolation,
    LeakageReport,
    assert_no_frame_leakage,
    check_adjacent_frames,
    default_frame_key,
    default_group_key,
    group_items,
    mhsma_split_leakage_note,
    patient_level_split,
    summarise_split,
)
from .licenses import (
    LICENSES,
    CommercialUse,
    LicenseRecord,
    check_commercial_use,
    check_share_alike,
    describe_licenses,
    get_license,
    strictest_terms,
)

__all__ = [
    "LICENSES",
    "AdjacencyViolation",
    "CheckResult",
    "CheckStatus",
    "CommercialUse",
    "LeakageReport",
    "LicenseRecord",
    "ValidationReport",
    "assert_no_frame_leakage",
    "check_adjacent_frames",
    "check_array_finite",
    "check_commercial_use",
    "check_dir_present",
    "check_file_present",
    "check_label_range",
    "check_non_empty",
    "check_npy_header",
    "check_share_alike",
    "default_frame_key",
    "default_group_key",
    "describe_licenses",
    "get_license",
    "group_items",
    "mhsma_split_leakage_note",
    "npy_header",
    "patient_level_split",
    "strictest_terms",
    "summarise_split",
]
