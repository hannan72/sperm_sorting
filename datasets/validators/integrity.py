"""Dataset integrity checking and the :class:`ValidationReport` every adapter returns.

Why a report object rather than bare assertions
-----------------------------------------------
An adapter that raises on the first problem tells you one thing per run, and
tells you nothing at all about the checks it never reached. A dataset that is
subtly wrong is usually wrong in several places at once -- a missing split, a
dtype that silently promoted to ``int64``, a label column that only ever takes
one value -- and the fastest way to diagnose that is to see every check at
once. So the default is: run everything, collect everything, report everything,
and let the caller decide whether to raise (:meth:`ValidationReport.raise_on_failure`).

The one exception is documented in :mod:`datasets.adapters.mhsma`: label
*polarity* inversion raises immediately, because a report that is merely
inspected and then ignored is exactly how an inverted classifier ships.

Five statuses, not two
----------------------
``PASS``/``FAIL`` alone force every unknown into one of the two, and the
tempting default is ``PASS``. That is how "we could not check this" becomes
"this is fine" in a slide deck. :class:`CheckStatus` therefore has:

``PASS``
    Checked, correct.
``FAIL``
    Checked, wrong. The only status that makes :attr:`ValidationReport.ok`
    false.
``WARN``
    Checked, legal, but likely to bite -- e.g. an incomplete copy of a dataset,
    or a class that is present but almost empty.
``UNVERIFIABLE``
    The property matters and **cannot be checked from the published files**.
    MHSMA's patient-level split is the canonical example: the risk is
    unverifiable, not absent. Reported loudly, never silently passed.
``SKIPPED``
    Not applicable to this copy (an optional subset is absent, an optional
    dependency is missing).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CheckResult",
    "CheckStatus",
    "ValidationReport",
    "check_array_finite",
    "check_dir_present",
    "check_file_present",
    "check_label_range",
    "check_non_empty",
    "check_npy_header",
    "npy_header",
]


class CheckStatus(str, Enum):
    """Outcome of a single integrity check. See the module docstring."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    UNVERIFIABLE = "unverifiable"
    SKIPPED = "skipped"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named check, its verdict, and a sentence a human can act on.

    ``message`` is written for someone who has just been handed a failing CI
    job and does not know this codebase: it names the file, the expectation and
    the observation, in that order.
    """

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAIL

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": str(self.status),
            "message": self.message,
            "details": _jsonable(self.details),
        }

    def format_line(self) -> str:
        return f"[{self.status.value.upper():<12}] {self.name}: {self.message}"


@dataclass(slots=True)
class ValidationReport:
    """The result of validating one dataset copy.

    ``ok`` is a *derived* property rather than a stored field on purpose. A
    stored boolean can disagree with the list of checks next to it -- someone
    appends a failing check and forgets to clear the flag -- and a validation
    report that lies is worse than no report at all.
    """

    dataset: str
    root: Path | None = None
    checks: list[CheckResult] = field(default_factory=list)
    #: Free-form context (counts, measured prevalences, discovered layout).
    context: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------- construction

    def add(
        self,
        name: str,
        status: CheckStatus,
        message: str,
        **details: Any,
    ) -> CheckResult:
        """Append a check and return it (handy for immediate inspection)."""
        result = CheckResult(name=name, status=status, message=message, details=details)
        self.checks.append(result)
        return result

    def extend(self, results: Iterable[CheckResult]) -> None:
        self.checks.extend(results)

    # -------------------------------------------------------------- verdicts

    @property
    def ok(self) -> bool:
        """True when no check FAILed.

        ``WARN`` and ``UNVERIFIABLE`` do not make a report not-ok: they are
        conditions a human must read, not conditions the data got wrong. Use
        :attr:`has_unverifiable` before claiming a dataset is "validated".
        """
        return not any(c.failed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.WARN]

    @property
    def unverifiable(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.UNVERIFIABLE]

    @property
    def has_unverifiable(self) -> bool:
        """True when at least one property that matters could not be checked."""
        return bool(self.unverifiable)

    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(1 for c in self.checks if c.status is status)
            for status in CheckStatus
        }

    # ------------------------------------------------------------- reporting

    def raise_on_failure(self, exc_type: type[Exception]) -> None:
        """Raise ``exc_type`` listing every failure, if there is one.

        The exception carries *all* failing messages rather than the first,
        because fixing datasets one error per CI round-trip is how an afternoon
        disappears.
        """
        if self.ok:
            return
        joined = "\n  - ".join(c.message for c in self.failures)
        raise exc_type(
            f"{self.dataset}: {len(self.failures)} integrity check(s) failed"
            f"{f' under {self.root}' if self.root is not None else ''}:\n  - {joined}"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "root": str(self.root) if self.root is not None else None,
            "ok": self.ok,
            "counts": self.counts(),
            "checks": [c.to_json_dict() for c in self.checks],
            "context": _jsonable(self.context),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=False)

    def format_text(self) -> str:
        """Human-readable report, failures and unverifiables last so they stick."""
        order = {
            CheckStatus.PASS: 0,
            CheckStatus.SKIPPED: 1,
            CheckStatus.WARN: 2,
            CheckStatus.UNVERIFIABLE: 3,
            CheckStatus.FAIL: 4,
        }
        head = f"{self.dataset} @ {self.root or '<no root>'} -> {'OK' if self.ok else 'NOT OK'}"
        lines = [head, "-" * len(head)]
        lines.extend(
            c.format_line() for c in sorted(self.checks, key=lambda c: order[c.status])
        )
        if self.has_unverifiable:
            lines.append(
                "NOTE: at least one property could not be verified from the published "
                "files. Unverifiable is not the same as absent -- read the entries above."
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.format_text()


# ==========================================================================
# Reusable checks
#
# Each returns a CheckResult rather than raising, so an adapter can compose a
# dozen of them into one report without a try/except per call.
# ==========================================================================


def check_file_present(path: Path, *, name: str | None = None) -> CheckResult:
    """File exists and is non-empty."""
    check = name or f"file:{path.name}"
    if not path.exists():
        return CheckResult(check, CheckStatus.FAIL, f"missing file: {path}")
    if not path.is_file():
        return CheckResult(check, CheckStatus.FAIL, f"expected a file, found a directory: {path}")
    size = path.stat().st_size
    if size == 0:
        return CheckResult(check, CheckStatus.FAIL, f"file is empty (0 bytes): {path}")
    return CheckResult(
        check, CheckStatus.PASS, f"present ({size} bytes): {path}", {"bytes": size}
    )


def check_dir_present(path: Path, *, name: str | None = None) -> CheckResult:
    """Directory exists and contains at least one entry."""
    check = name or f"dir:{path.name}"
    if not path.exists():
        return CheckResult(check, CheckStatus.FAIL, f"missing directory: {path}")
    if not path.is_dir():
        return CheckResult(check, CheckStatus.FAIL, f"expected a directory, found a file: {path}")
    n = sum(1 for _ in path.iterdir())
    if n == 0:
        return CheckResult(check, CheckStatus.FAIL, f"directory is empty: {path}")
    return CheckResult(check, CheckStatus.PASS, f"present ({n} entries): {path}", {"entries": n})


def npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype, bool]:
    """Read an ``.npy`` header without loading the array.

    MHSMA's largest array is 16 MB, so this is not about speed -- it is about
    being able to report a wrong dtype or shape on a file that is too large or
    too corrupt to load, which is precisely when you most want the answer.

    Only the documented public readers are used
    (``read_array_header_1_0`` / ``read_array_header_2_0``), dispatched on the
    version returned by ``read_magic``. The private ``_read_array_header``
    helper is tempting because it takes the version as an argument, but it was
    removed in NumPy 2.x and taking that shortcut breaks every check in this
    module on a version bump.
    """
    readers: dict[tuple[int, ...], Any] = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        reader = readers.get(tuple(version))
        if reader is None:
            raise ValueError(
                f"unsupported .npy format version {version} in {path}; this reader "
                f"handles {sorted(readers)}"
            )
        shape, fortran_order, dtype = reader(handle)
    return tuple(int(s) for s in shape), np.dtype(dtype), bool(fortran_order)


def check_npy_header(
    path: Path,
    *,
    expected_shape: Sequence[int | None] | None = None,
    expected_dtype: str | np.dtype | None = None,
    name: str | None = None,
) -> CheckResult:
    """Shape and dtype of an ``.npy`` file.

    ``expected_shape`` may contain ``None`` for "any length on this axis", so a
    caller can assert ``(None, 128, 128)`` when the split size is what it is
    checking elsewhere.
    """
    check = name or f"npy:{path.name}"
    presence = check_file_present(path, name=check)
    if presence.failed:
        return presence
    try:
        shape, dtype, fortran = npy_header(path)
    except Exception as exc:
        return CheckResult(
            check, CheckStatus.FAIL, f"unreadable .npy header for {path}: {exc!r}"
        )

    problems: list[str] = []
    if expected_shape is not None:
        if len(shape) != len(expected_shape):
            problems.append(
                f"rank {len(shape)} (shape {shape}) but expected rank {len(expected_shape)} "
                f"(shape {tuple(expected_shape)})"
            )
        else:
            for axis, (got, want) in enumerate(zip(shape, expected_shape, strict=True)):
                if want is not None and got != want:
                    problems.append(f"axis {axis} is {got}, expected {want}")
    if expected_dtype is not None and dtype != np.dtype(expected_dtype):
        problems.append(f"dtype is {dtype}, expected {np.dtype(expected_dtype)}")

    details = {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}
    if problems:
        return CheckResult(
            check, CheckStatus.FAIL, f"{path}: " + "; ".join(problems), details
        )
    return CheckResult(
        check, CheckStatus.PASS, f"{path.name}: shape {shape}, dtype {dtype}", details
    )


def check_label_range(
    labels: np.ndarray, *, allowed: Sequence[int], name: str
) -> CheckResult:
    """Every label value is in ``allowed``.

    A label set that has drifted outside ``{0, 1}`` -- because someone
    one-hot-encoded it, or because a ``-1`` "ignore" marker crept in -- must not
    reach a ``BCEWithLogitsLoss``, which will happily train on it.
    """
    unique = np.unique(labels)
    extra = sorted(int(v) for v in unique if int(v) not in set(allowed))
    details = {"unique": [int(v) for v in unique], "allowed": list(allowed)}
    if extra:
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"{name}: values {extra} are outside the allowed set {list(allowed)}",
            details,
        )
    missing = sorted(set(allowed) - {int(v) for v in unique})
    if missing:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"{name}: class(es) {missing} never occur; a split with an absent class "
            "cannot produce a meaningful metric for it",
            details,
        )
    return CheckResult(
        name, CheckStatus.PASS, f"{name}: values {details['unique']} within {list(allowed)}", details
    )


def check_array_finite(array: np.ndarray, *, name: str) -> CheckResult:
    """No NaN and no infinity.

    Integer arrays cannot hold either, so they pass trivially -- reported as
    PASS rather than SKIPPED because the property genuinely does hold.
    """
    if array.dtype.kind not in "fc":
        return CheckResult(
            name, CheckStatus.PASS, f"{name}: integer dtype {array.dtype}, cannot be NaN/inf"
        )
    n_nan = int(np.count_nonzero(np.isnan(array)))
    n_inf = int(np.count_nonzero(np.isinf(array)))
    details = {"n_nan": n_nan, "n_inf": n_inf, "size": int(array.size)}
    if n_nan or n_inf:
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"{name}: {n_nan} NaN and {n_inf} infinite value(s) out of {array.size}",
            details,
        )
    return CheckResult(name, CheckStatus.PASS, f"{name}: all {array.size} values finite", details)


def check_non_empty(count: int, *, name: str, what: str) -> CheckResult:
    """A count that must not be zero (frames, annotations, participants...)."""
    if count <= 0:
        return CheckResult(name, CheckStatus.FAIL, f"{name}: found 0 {what}")
    return CheckResult(name, CheckStatus.PASS, f"{name}: found {count} {what}", {"count": count})


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy/Path values so a report can be dumped."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return str(value)
    return value
