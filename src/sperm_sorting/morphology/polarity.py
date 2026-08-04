"""THE label-polarity contract for morphology. One convention, one flip, one file.

Read this before touching anything else in :mod:`sperm_sorting.morphology`.

The convention
--------------
MHSMA labels every one of the four aspects as ``0 = normal`` / ``1 = abnormal``
(the dataset README confusingly calls the *normal* class "positive"; the integer
encoding is what matters and it is the one above, mirrored in
:data:`sperm_sorting.constants.LABEL_NORMAL` / ``LABEL_ABNORMAL``).

This package therefore fixes:

    **Every logit produced by the network is a logit for P(abnormal).**

Consequences, which are the whole point of writing this down:

* ``sigmoid(logit) == P(abnormal)``.
* The **training target is the MHSMA integer label verbatim**. There is no
  ``1 - y``, no ``.flip()``, no ``not`` anywhere in the training path. Training
  code that does the obvious thing is automatically correct, which is the only
  reliable defence against a silently inverted product.
* Calibration (:mod:`.calibration`) and metrics (:mod:`.metrics`) also live
  entirely in *abnormal-positive* space: a threshold there means "predict
  abnormal when ``P(abnormal) >= t``", and "sensitivity" there means recall of
  the abnormal class. Neither module contains a flip.
* The **inference adapter is the only place a flip happens**, and it happens by
  calling :func:`flip_polarity` below. :class:`~.inference.MorphologyEngine`
  calls it exactly twice per aspect -- once for the probability, once for the
  threshold -- but both calls route through this single implementation, so the
  arithmetic ``1 - x`` appears once in the codebase for polarity purposes.
* The product-facing schema is the other way round on purpose:
  :attr:`sperm_sorting.schemas.morphology.AspectResult.p_normal` is
  ``P(normal)`` and ``normal`` is ``p_normal >= threshold``. That is the
  boundary this module exists to police.

Why not train on ``P(normal)`` directly and skip the flip? Because then the
training target would be ``1 - label``, and every dataset adapter, augmentation
wrapper, sampler and class-weight computation would have to remember to invert.
One flip in one adapter is auditable; a flip smeared across the training stack
is how a sorter ends up keeping exactly the sperm it was built to discard.

Checkpoints and calibration bundles both carry :data:`POLARITY_CONVENTION` as a
string and refuse to load when it does not match, so an artefact produced under
a different convention can never be silently consumed under this one.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from ..constants import LABEL_ABNORMAL, LABEL_NORMAL

__all__ = [
    "POLARITY_CONVENTION",
    "POSITIVE_CLASS",
    "POSITIVE_CLASS_NAME",
    "describe_polarity",
    "flip_polarity",
    "p_normal_from_p_abnormal",
    "p_normal_threshold_from_p_abnormal_threshold",
    "self_check",
]

#: Machine-comparable statement of the convention. Stamped into every
#: checkpoint (:func:`~.model.save_checkpoint`) and every calibration bundle
#: (:meth:`~.calibration.CalibrationBundle.save_json`); loading an artefact
#: whose string differs raises rather than guessing. Bump the trailing version
#: only if the convention itself ever changes, which would invalidate every
#: existing checkpoint -- that is exactly the intent.
POLARITY_CONVENTION: str = (
    "logit=P(abnormal);target=mhsma_label_verbatim(0=normal,1=abnormal);"
    "flip=inference_adapter_only;v1"
)

#: The class the network's logit, the loss' ``pos_weight``, every calibration
#: threshold and every metric in this package treat as positive.
POSITIVE_CLASS: int = LABEL_ABNORMAL
POSITIVE_CLASS_NAME: str = "abnormal"

_T = TypeVar("_T", float, np.ndarray)


def flip_polarity(value: _T) -> _T:
    """Convert between abnormal-space and normal-space. **The only flip.**

    This is deliberately a one-line function with an oversized docstring: it is
    the single point at which the package changes which class is "positive". If
    you ever find yourself writing ``1 - p`` or ``1 - threshold`` elsewhere in
    :mod:`sperm_sorting.morphology` for polarity reasons, call this instead so
    that the flip stays countable by ``grep``.

    The map is an involution, so it serves both directions
    (``P(abnormal) -> P(normal)`` and back), and it applies unchanged to
    thresholds because thresholding is monotone: predicting abnormal when
    ``P(abnormal) >= t`` is the same rule as predicting normal when
    ``P(normal) <= 1 - t``. The schema's rule is ``normal`` iff
    ``p_normal >= threshold``, i.e. ``P(abnormal) <= t`` -- the two differ only
    on the measure-zero boundary ``P(abnormal) == t``, where this package
    resolves in favour of *normal*. That is a deliberate, documented tie-break
    and it is the only asymmetry introduced by the flip.

    Parameters
    ----------
    value
        A probability, an array of probabilities, or a threshold, in ``[0, 1]``.

    Returns
    -------
    The same type, in the opposite polarity.
    """
    return 1.0 - value  # type: ignore[return-value]


def p_normal_from_p_abnormal(p_abnormal: _T) -> _T:
    """``P(normal)`` from the network's ``P(abnormal)``. Named call site."""
    return flip_polarity(p_abnormal)


def p_normal_threshold_from_p_abnormal_threshold(threshold: _T) -> _T:
    """Schema-space threshold from a calibration-space threshold.

    Calibration fits thresholds on ``P(abnormal)`` because that is where the
    class imbalance lives and where "sensitivity" is meaningful. The schema
    compares against ``P(normal)``. This converts one into the other.
    """
    return flip_polarity(threshold)


def describe_polarity() -> dict[str, object]:
    """Machine-readable polarity block for audit-log headers."""
    return {
        "convention": POLARITY_CONVENTION,
        "logit_means": "P(abnormal)",
        "label_normal": LABEL_NORMAL,
        "label_abnormal": LABEL_ABNORMAL,
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": POSITIVE_CLASS_NAME,
        "flip_location": "sperm_sorting.morphology.polarity.flip_polarity",
        "flip_applied_by": "sperm_sorting.morphology.inference.MorphologyEngine",
    }


def self_check() -> None:
    """Assert the convention end to end. Cheap enough to run at import time.

    Verifies the chain a reviewer actually cares about: a large positive logit
    means *abnormal*, which must surface as a low ``p_normal``, which must make
    :attr:`AspectResult.normal` false and :attr:`AspectResult.label` equal to
    :data:`~sperm_sorting.constants.LABEL_ABNORMAL`.

    Raises
    ------
    AssertionError
        If any link in the chain is inverted.
    """
    from ..schemas.morphology import AspectResult

    # Local import so this module stays dependency-free for callers that only
    # want the constant; the schema import is cheap and torch-free.
    big = 8.0
    p_abnormal_hi = float(1.0 / (1.0 + np.exp(-big)))
    p_abnormal_lo = float(1.0 / (1.0 + np.exp(big)))

    assert p_abnormal_hi > 0.99, "sigmoid of a large positive logit must be ~1"
    assert LABEL_NORMAL == 0 and LABEL_ABNORMAL == 1, "MHSMA encoding changed"

    abnormal = AspectResult(
        name="head",
        p_normal=p_normal_from_p_abnormal(p_abnormal_hi),
        threshold=0.5,
    )
    assert abnormal.p_normal < 0.01, "a confident abnormal logit must give low p_normal"
    assert abnormal.normal is False, "a confident abnormal logit must not read as normal"
    assert abnormal.label == LABEL_ABNORMAL, "abnormal prediction must carry label 1"

    normal = AspectResult(
        name="head",
        p_normal=p_normal_from_p_abnormal(p_abnormal_lo),
        threshold=0.5,
    )
    assert normal.p_normal > 0.99, "a confident normal logit must give high p_normal"
    assert normal.normal is True, "a confident normal logit must read as normal"
    assert normal.label == LABEL_NORMAL, "normal prediction must carry label 0"

    # The flip is an involution, so applying it twice is a no-op. If this ever
    # fails, someone has replaced it with something that is not 1 - x.
    assert abs(flip_polarity(flip_polarity(0.37)) - 0.37) < 1e-12

    # A threshold fitted at, say, 0.12 in abnormal space must become 0.88 in
    # schema space, and must stay strictly inside (0, 1) as MorphologyConfig
    # requires.
    t_schema = p_normal_threshold_from_p_abnormal_threshold(0.12)
    assert abs(t_schema - 0.88) < 1e-12
    assert 0.0 < t_schema < 1.0


self_check()
