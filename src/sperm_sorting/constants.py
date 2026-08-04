"""Project-wide constants.

Everything in this module is either (a) a value fixed by the decision
specification and therefore *not* tunable at runtime, or (b) a sentinel /
enumeration-like literal shared across subsystems.

Physical values that depend on the built device (transport delay, field rise
time, micrometres-per-pixel, ...) deliberately do **not** live here. They are
configuration values that must be measured by the calibration utilities in
:mod:`sperm_sorting.calibration`; see ``configs/device_v1.yaml``.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Decision rule (specification-fixed; changing these changes the product)
# --------------------------------------------------------------------------

#: Acceptance threshold on the AI-eligible ratio. A shot is ACCEPTed only when
#: ``ratio > ACCEPT_RATIO_THRESHOLD``. Exactly 0.60 is a REJECT.
ACCEPT_RATIO_THRESHOLD: Final[float] = 0.60

#: Below this many unique trackable sperm, a shot cannot be decided.
MINIMUM_TRACKABLE_SPERM: Final[int] = 20

#: Nominal shot size; a shot closes normally once this many tracks are gated.
TARGET_TRACKABLE_SPERM: Final[int] = 25

#: Hard ceiling on shot size; a shot closes immediately upon reaching this.
MAXIMUM_TRACKABLE_SPERM: Final[int] = 30

#: A shot is force-closed after this wall-clock duration regardless of count.
MAXIMUM_SHOT_DURATION_S: Final[float] = 1.0

# --------------------------------------------------------------------------
# Morphology label convention (MHSMA)
# --------------------------------------------------------------------------

#: MHSMA encodes ``0 = normal`` and ``1 = abnormal`` for every one of the four
#: morphology aspects. Every adapter, loss, metric and inference path in this
#: repository uses this convention. Guarded by
#: ``tests/test_mhsma_label_direction.py``; do not "simplify" it away.
LABEL_NORMAL: Final[int] = 0
LABEL_ABNORMAL: Final[int] = 1

#: Canonical ordering of the four morphology aspects. Model head order, metric
#: report order and threshold-vector order all follow this tuple.
MORPHOLOGY_ASPECTS: Final[tuple[str, str, str, str]] = (
    "head",
    "acrosome",
    "vacuole",
    "tail",
)

# --------------------------------------------------------------------------
# Motility defaults (WHO/CASA-inspired; overridable via versioned config)
# --------------------------------------------------------------------------

#: Corrected VSL at or above which a track is "rapid progressive" (um/s).
#: This is WHO's own approximate limit, not an invented one: the WHO
#: laboratory manual, 6th edition (2021), section 2.4.6.1 defines rapidly
#: progressive as covering "at least 25 um (or 1/2 tail length) in one second".
DEFAULT_RAPID_PROGRESSIVE_VSL_UM_S: Final[float] = 25.0

#: Corrected VSL at or above which a track is at least "slow progressive".
#: Also WHO 6th ed. section 2.4.6.1: slowly progressive covers "5 to < 25 um
#: (or at least one head length to less than 1/2 tail length) in one second".
DEFAULT_SLOW_PROGRESSIVE_VSL_UM_S: Final[float] = 5.0

#: Version tag stamped into every :class:`MotionFeatures` record so that
#: historical audit logs remain interpretable after a threshold change.
#:
#: The four motility grades and both velocity limits are taken directly from
#: WHO 6th ed. section 2.4.6.1, which reinstated the four-category system the
#: 5th edition had collapsed to PR/NP/IM. The tag names that source rather
#: than claiming mere inspiration. The motion module appends the resolved
#: smoothing parameters to this prefix per track, because VAP -- and therefore
#: STR, WOB, ALH and BCF -- depends on the smoothing algorithm, so a bare
#: threshold version would not fully identify how a number was produced.
MOTILITY_PROFILE_VERSION: Final[str] = "who6-2021-s2.4.6.1-v1"

# --------------------------------------------------------------------------
# Numerical guards
# --------------------------------------------------------------------------

#: Denominators smaller than this are treated as zero (ratios become 0.0).
EPS: Final[float] = 1e-9

#: Sentinel used for "this quantity could not be computed"; paired with a
#: human-readable reason string on the owning record.
UNAVAILABLE: Final[None] = None

# --------------------------------------------------------------------------
# Schema / audit versioning
# --------------------------------------------------------------------------

#: Bumped whenever a runtime schema gains, loses or re-interprets a field.
#: Written into every audit record so replayed logs can be validated.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Weights trained on public research datasets are *baseline research weights*
#: and are never to be presented as device-validated. Stamped into checkpoints.
WEIGHTS_PROVENANCE_PUBLIC: Final[str] = "public-research-baseline"
WEIGHTS_PROVENANCE_DEVICE: Final[str] = "device-finetuned"
WEIGHTS_PROVENANCE_SYNTHETIC: Final[str] = "synthetic-bootstrap"
