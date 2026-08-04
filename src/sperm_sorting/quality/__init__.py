"""Per-candidate crop quality scoring and best-frame selection.

This package answers "which frame gives the best look at *this* sperm", which
is a different question from the whole-frame gate in
:mod:`sperm_sorting.preprocessing.quality_gate`.

It runs **after** tracking and motion analysis, never before:
:meth:`BestFrameSelector.select` refuses a track that has not been classified,
and refuses one that is not progressive. See :mod:`.selector` for why that
ordering is enforced in the API rather than left to convention.
"""

from __future__ import annotations

from .frame_score import (
    DEFAULT_NORMALISATION,
    ScoreNormalisation,
    describe_normalisation,
    padded_box,
    score_candidate,
    validate_weights,
    visible_fraction_of,
)
from .selector import (
    BestFrameOrderingError,
    BestFrameSelector,
    CandidateFrame,
    FrameBuffer,
)

__all__ = [
    "DEFAULT_NORMALISATION",
    "BestFrameOrderingError",
    "BestFrameSelector",
    "CandidateFrame",
    "FrameBuffer",
    "ScoreNormalisation",
    "describe_normalisation",
    "padded_box",
    "score_candidate",
    "validate_weights",
    "visible_fraction_of",
]
