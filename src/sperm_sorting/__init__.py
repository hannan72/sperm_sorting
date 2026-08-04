"""Real-time AI-guided sperm analysis and conditional magnetic sorting.

A research prototype. This software analyses visible phenotype -- presence,
trajectory, velocity, direction, progression, linearity and morphology -- from
microscopy video, and emits a two-state FIELD_ON / FIELD_OFF command.

It does **not** measure DNA integrity, phosphatidylserine exposure, Annexin V
binding, apoptosis, magnetic labelling, fertility potential or pregnancy rate,
and it is not clinically validated. Internally, a sperm that satisfies the
combined motility-and-morphology rule is called ``ai_eligible``, never
"healthy". See ``docs/safety_and_claims.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .constants import (
    ACCEPT_RATIO_THRESHOLD,
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    MORPHOLOGY_ASPECTS,
    SCHEMA_VERSION,
)

__all__ = [
    "ACCEPT_RATIO_THRESHOLD",
    "LABEL_ABNORMAL",
    "LABEL_NORMAL",
    "MORPHOLOGY_ASPECTS",
    "SCHEMA_VERSION",
    "__version__",
]
