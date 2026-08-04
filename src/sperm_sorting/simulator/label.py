"""The health rule, in exactly one place.

Why a whole module for three functions
--------------------------------------
"Healthy" is a product definition, not an implementation detail. If the
dataset builder, the demo and the evaluation harness each re-derived it from
:class:`~.params.HealthState`, they would drift, and a model would end up
trained against one rule and scored against another. Everything that needs the
rule imports it from here.

The rule
--------
A synthetic sperm is **healthy (0)** when *all four* morphology aspects are
normal **and** its motility grade is progressive (rapid or slow both count).
Anything else is **unhealthy (1)**. That is the conjunction of the two filters
the real pipeline applies -- see
:class:`sperm_sorting.schemas.enums.IneligibilityReason`, whose members are
exactly the ways this conjunction can fail.

Naming note: downstream, the runtime calls this property ``ai_eligible``, not
"healthy", because the pipeline observes phenotype only (see the package
docstring). Inside the simulator the ground truth genuinely is the generative
health state, so ``overall_label`` is the honest name here.
"""

from __future__ import annotations

from itertools import product
from typing import Final

import numpy as np

from ..constants import LABEL_ABNORMAL, LABEL_NORMAL, MORPHOLOGY_ASPECTS
from ..schemas.enums import MotilityClass
from .params import SIMULATED_MOTILITY_CLASSES, HealthState

#: Integer encoding of the motility head's three classes. Progressive collapses
#: rapid and slow because the decision rule does not distinguish them; keeping
#: them separate here would create a class the loss must learn but nothing
#: consumes.
MOTILITY_LABEL_PROGRESSIVE: Final[int] = 0
MOTILITY_LABEL_NON_PROGRESSIVE: Final[int] = 1
MOTILITY_LABEL_IMMOTILE: Final[int] = 2

#: Names indexed by the integer above; used in ``meta.json`` and report tables.
MOTILITY_LABEL_NAMES: Final[tuple[str, str, str]] = (
    "progressive",
    "non_progressive",
    "immotile",
)

#: Names indexed by :func:`overall_label`.
OVERALL_LABEL_NAMES: Final[tuple[str, str]] = ("healthy", "unhealthy")


def aspect_labels(state: HealthState) -> np.ndarray:
    """The four binary morphology labels as ``uint8[4]``.

    Order is :data:`sperm_sorting.constants.MORPHOLOGY_ASPECTS`, which is also
    the model's head order and the threshold-vector order, so an index means
    the same thing everywhere.
    """
    return np.asarray(state.aspects, dtype=np.uint8)


def motility_label(state: HealthState) -> int:
    """0 progressive / 1 non-progressive / 2 immotile."""
    if state.motility.is_progressive:
        return MOTILITY_LABEL_PROGRESSIVE
    if state.motility is MotilityClass.NON_PROGRESSIVE:
        return MOTILITY_LABEL_NON_PROGRESSIVE
    if state.motility is MotilityClass.IMMOTILE:
        return MOTILITY_LABEL_IMMOTILE
    # Unreachable for a valid HealthState, which rejects UNDETERMINED at
    # construction; kept so a future enum member fails loudly rather than
    # silently folding into "immotile".
    raise ValueError(f"unlabelable motility grade: {state.motility!r}")


def morphology_label(state: HealthState) -> int:
    """0 when every aspect is normal, 1 when any aspect is abnormal."""
    return LABEL_NORMAL if state.is_morphology_normal else LABEL_ABNORMAL


def overall_label(state: HealthState) -> int:
    """0 healthy / 1 unhealthy.

    Healthy requires the conjunction: normal head **and** acrosome **and**
    vacuole **and** tail, **and** progressive motility. The asymmetry is
    intentional -- one defect is enough to disqualify, which is why the
    positive class is the common one and why the dataset builder balances
    explicitly rather than trusting the prevalences.
    """
    if not state.is_morphology_normal:
        return LABEL_ABNORMAL
    if not state.motility.is_progressive:
        return LABEL_ABNORMAL
    return LABEL_NORMAL


def label_dict(state: HealthState) -> dict[str, object]:
    """All labels for one state, in the shape the scene generator emits."""
    return {
        "aspects": [int(v) for v in state.aspects],
        "motility": str(state.motility),
        "motility_label": motility_label(state),
        "morphology": morphology_label(state),
        "overall": overall_label(state),
    }


def truth_table() -> list[tuple[tuple[int, int, int, int], MotilityClass, int]]:
    """Enumerate all ``2^4 x 4`` combinations with their expected label.

    Returned rather than asserted so that tests outside this module can use the
    same enumeration instead of re-deriving the rule (which would defeat the
    point of centralising it).
    """
    rows: list[tuple[tuple[int, int, int, int], MotilityClass, int]] = []
    for raw in product((LABEL_NORMAL, LABEL_ABNORMAL), repeat=len(MORPHOLOGY_ASPECTS)):
        combo: tuple[int, int, int, int] = (raw[0], raw[1], raw[2], raw[3])
        for grade in SIMULATED_MOTILITY_CLASSES:
            expected = (
                LABEL_NORMAL
                if all(v == LABEL_NORMAL for v in combo) and grade.is_progressive
                else LABEL_ABNORMAL
            )
            rows.append((combo, grade, expected))
    return rows


def self_check() -> int:
    """Verify :func:`overall_label` against the full truth table.

    Returns the number of rows checked so a caller can assert on it (a silent
    zero-row pass would be worse than a failure).
    """
    checked = 0
    for combo, grade, expected in truth_table():
        state = HealthState(
            head=combo[0],
            acrosome=combo[1],
            vacuole=combo[2],
            tail=combo[3],
            motility=grade,
        )
        got = overall_label(state)
        if got != expected:
            raise AssertionError(
                f"overall_label({combo}, {grade}) = {got}, expected {expected}"
            )
        if int(aspect_labels(state).sum()) != sum(combo):
            raise AssertionError(f"aspect_labels disagrees with the state for {combo}")
        checked += 1
    if checked != 2 ** len(MORPHOLOGY_ASPECTS) * len(SIMULATED_MOTILITY_CLASSES):
        raise AssertionError(f"truth table is incomplete: {checked} rows")
    return checked


if __name__ == "__main__":  # pragma: no cover - runnable self-check
    n_rows = self_check()
    assert n_rows == 64, f"expected 64 truth-table rows, got {n_rows}"

    # Exactly one combination per progressive grade is healthy: all-normal.
    healthy = [r for r in truth_table() if r[2] == LABEL_NORMAL]
    assert len(healthy) == 2, f"expected 2 healthy rows, got {len(healthy)}"
    assert all(combo == (0, 0, 0, 0) for combo, _, _ in healthy)
    assert {grade for _, grade, _ in healthy} == {
        MotilityClass.RAPID_PROGRESSIVE,
        MotilityClass.SLOW_PROGRESSIVE,
    }

    # A single defect is enough, for each aspect independently.
    for idx, name in enumerate(MORPHOLOGY_ASPECTS):
        combo = [0, 0, 0, 0]
        combo[idx] = 1
        st = HealthState(
            head=combo[0], acrosome=combo[1], vacuole=combo[2], tail=combo[3],
            motility=MotilityClass.RAPID_PROGRESSIVE,
        )
        assert overall_label(st) == LABEL_ABNORMAL, f"{name} defect must disqualify"
        assert int(aspect_labels(st)[idx]) == LABEL_ABNORMAL

    # Motility encoding.
    enc = {
        MotilityClass.RAPID_PROGRESSIVE: 0,
        MotilityClass.SLOW_PROGRESSIVE: 0,
        MotilityClass.NON_PROGRESSIVE: 1,
        MotilityClass.IMMOTILE: 2,
    }
    for grade, want in enc.items():
        assert motility_label(HealthState(motility=grade)) == want, grade

    # Perfect cell is the only healthy one among a random sample of defects.
    perfect = HealthState(motility=MotilityClass.RAPID_PROGRESSIVE)
    assert overall_label(perfect) == LABEL_NORMAL
    assert morphology_label(perfect) == LABEL_NORMAL
    assert aspect_labels(perfect).dtype == np.uint8
    assert aspect_labels(perfect).shape == (4,)
    assert label_dict(perfect)["overall"] == 0

    print(f"label.py self-check OK ({n_rows} truth-table rows, {len(healthy)} healthy)")
