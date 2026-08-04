"""Ground-truth health state: the single source of truth for "healthy".

Why this module exists
----------------------
Public data gives us either boxes and tracks (VISEM-Tracking) or morphology
crops (MHSMA), never both for the same cell. The simulator closes that gap by
*sampling the ground truth first* and deriving everything observable from it.
:class:`HealthState` is that ground truth. :mod:`~.render` turns it into an
image and :mod:`~.motility` turns it into a trajectory, so a single sample is
jointly labelled for morphology *and* motion at zero annotation cost.

The causality rule
------------------
A morphology flag is never an independent annotation bolted onto a random
picture. Each binary aspect flag *causes* its continuous knob to leave the
normal band (see :func:`sample_health_state`). If the flags were sampled
independently of the appearance, a classifier trained on this data would be
fitting noise: there would be nothing in the pixels to learn. Everything the
label asserts must be visible.

Normal ranges (WHO laboratory manual, 6th edition, strict criteria)
-------------------------------------------------------------------
Where a WHO number exists we use it; where it does not, the range is a
documented modelling choice, marked as such.

* Head length ~4.0-5.0 um, width ~2.5-3.5 um  (WHO strict criteria).
* Head length:width ratio ~1.5-1.75           (WHO strict criteria).
* Acrosome covering ~40-70% of the head area  (WHO strict criteria).
* Vacuoles: normal heads contain no large vacuole; >20% of the head area
  vacuolated is abnormal (WHO strict criteria).
* Midpiece ~1 um wide, ~7-8 um long; tail ~45 um, uncoiled, thinner than the
  midpiece (WHO strict criteria).
* Progressive velocity thresholds are the project's own, in
  :mod:`sperm_sorting.constants` (25 um/s rapid, 5 um/s slow), which is a
  CASA convention rather than a WHO one -- WHO grades progression
  qualitatively.

Units are micrometres and micrometres/second throughout; the renderer and the
trajectory generator convert to pixels using ``um_per_px``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from ..constants import LABEL_ABNORMAL, LABEL_NORMAL, MORPHOLOGY_ASPECTS
from ..schemas.enums import MotilityClass

# --------------------------------------------------------------------------
# Ground-truth normal bands
# --------------------------------------------------------------------------
# Each entry is (low, high) for a *normal* cell. The abnormal sampler pushes
# the value outside the band on the side(s) noted in the comment. Keeping the
# bands as module constants (rather than magic numbers inside the sampler)
# means the renderer, the docs and the tests all read the same numbers.

#: Head length in um. WHO strict: 4.0-5.0.
NORMAL_HEAD_LENGTH_UM: Final[tuple[float, float]] = (4.0, 5.0)
#: Head width in um. WHO strict: 2.5-3.5.
NORMAL_HEAD_WIDTH_UM: Final[tuple[float, float]] = (2.5, 3.5)
#: Head length:width. WHO strict: 1.5-1.75.
NORMAL_HEAD_AXIS_RATIO: Final[tuple[float, float]] = (1.50, 1.75)
#: Multiplicative size factor applied to the whole head. 1.0 is the WHO
#: mid-range head; a macrocephalic / microcephalic head sits outside this.
NORMAL_HEAD_SCALE: Final[tuple[float, float]] = (0.90, 1.10)
#: Fraction of head area covered by the acrosomal cap. WHO strict: 0.40-0.70.
NORMAL_ACROSOME_FRAC: Final[tuple[float, float]] = (0.40, 0.70)
#: Vacuole diameter as a fraction of head length. A normal head shows no
#: resolvable vacuole; we allow a sub-resolution trace so that "normal" is not
#: encoded as an exactly-zero pixel value the model could shortcut on.
NORMAL_VACUOLE_SIZE: Final[tuple[float, float]] = (0.00, 0.08)
#: Tail curvature, radians of total bend along the flagellum. A normal tail is
#: near-straight; coiling is the classic abnormality.
NORMAL_TAIL_CURVATURE: Final[tuple[float, float]] = (0.00, 0.55)
#: Tail length in um. WHO: ~45 um. Short/absent tails are abnormal.
NORMAL_TAIL_LENGTH_UM: Final[tuple[float, float]] = (40.0, 50.0)

#: Upper bound for a *short* abnormal tail, in um. Deliberately well under the
#: ~10 um of flagellum visible behind the midpiece in a morphology crop (see
#: :data:`sperm_sorting.simulator.render.CROP_FIELD_UM`): a "short tail" that
#: still ran off the edge of the crop would be labelled abnormal while looking
#: exactly like a normal one, which is label noise the model cannot resolve and
#: would simply learn to mis-fit.
ABNORMAL_SHORT_TAIL_MAX_UM: Final[float] = 8.0

#: Target **VCL** (curvilinear path speed) in um/s for each grade. Interpreting
#: ``speed_um_s`` as VCL rather than as a forward speed is what lets
#: :func:`sperm_sorting.simulator.motility.simulate_trajectory` hit both the
#: speed and the linearity a state claims: forward progression and lateral
#: wiggle are then a split of one budget rather than two independent numbers
#: that may contradict each other. Bands bracket clinical CASA VCL for human
#: sperm and sit well above the project's VSL cut-points (25 um/s rapid,
#: 5 um/s slow) once linearity is applied.
SPEED_BAND_UM_S: Final[dict[MotilityClass, tuple[float, float]]] = {
    MotilityClass.RAPID_PROGRESSIVE: (55.0, 110.0),
    MotilityClass.SLOW_PROGRESSIVE: (18.0, 38.0),
    MotilityClass.NON_PROGRESSIVE: (12.0, 30.0),
    MotilityClass.IMMOTILE: (0.0, 1.2),
}

#: Target **LIN** = VSL / VCL. Clinical progressive LIN is around 0.5-0.8; the
#: rapid band is kept at 0.65+ so a rapid-progressive cell is unambiguously
#: linear even after the beat and the finite-track-length scatter.
LINEARITY_BAND: Final[dict[MotilityClass, tuple[float, float]]] = {
    MotilityClass.RAPID_PROGRESSIVE: (0.65, 0.85),
    MotilityClass.SLOW_PROGRESSIVE: (0.55, 0.78),
    MotilityClass.NON_PROGRESSIVE: (0.02, 0.16),
    MotilityClass.IMMOTILE: (0.00, 0.20),
}

#: Free-running heading diffusion in radians per sqrt(second). Used directly by
#: the scene generator, whose agents are stepped one frame at a time and have
#: no known track length to solve a target linearity against. The batch
#: trajectory generator instead solves for the heading noise that realises
#: :attr:`HealthState.linearity`; see ``match_linearity`` there.
ANGLE_NOISE_BAND: Final[dict[MotilityClass, tuple[float, float]]] = {
    MotilityClass.RAPID_PROGRESSIVE: (0.15, 0.55),
    MotilityClass.SLOW_PROGRESSIVE: (0.45, 1.20),
    MotilityClass.NON_PROGRESSIVE: (11.0, 20.0),
    MotilityClass.IMMOTILE: (0.0, 0.5),
}

#: Requested lateral half-amplitude of the flagellar beat, in um; CASA ALH is
#: twice this. Clinical ALH is ~2-7 um, but those figures come from 50-60 Hz
#: systems that under-sample the beat and therefore under-report VCL. At this
#: project's 160 FPS the beat is resolved, so a clinically-typical ALH would
#: inflate VCL far past its clinical band. The generator therefore treats this
#: value as a *request*, capped at whatever the (VCL, LIN) budget allows;
#: realised ALH lands at the low end of the clinical range. Documented rather
#: than hidden, because it is a genuine limitation of the model.
BEAT_AMPLITUDE_BAND: Final[dict[MotilityClass, tuple[float, float]]] = {
    MotilityClass.RAPID_PROGRESSIVE: (0.6, 2.5),
    MotilityClass.SLOW_PROGRESSIVE: (0.4, 1.8),
    MotilityClass.NON_PROGRESSIVE: (0.3, 1.2),
    MotilityClass.IMMOTILE: (0.0, 0.10),
}

#: Flagellar beat-cross frequency in Hz, used directly as ground-truth BCF.
#: CASA BCF for human sperm is ~5-25 Hz; at 160 FPS this is comfortably
#: sampled (Nyquist 80 Hz), which is why the project runs the camera fast.
BEAT_FREQUENCY_BAND: Final[dict[MotilityClass, tuple[float, float]]] = {
    MotilityClass.RAPID_PROGRESSIVE: (8.0, 22.0),
    MotilityClass.SLOW_PROGRESSIVE: (5.0, 14.0),
    MotilityClass.NON_PROGRESSIVE: (3.0, 10.0),
    MotilityClass.IMMOTILE: (0.0, 1.0),
}

#: The four motility grades the simulator may emit. ``UNDETERMINED`` is a
#: runtime outcome ("we could not measure this"), never a ground truth, so it
#: is deliberately excluded.
SIMULATED_MOTILITY_CLASSES: Final[tuple[MotilityClass, ...]] = (
    MotilityClass.RAPID_PROGRESSIVE,
    MotilityClass.SLOW_PROGRESSIVE,
    MotilityClass.NON_PROGRESSIVE,
    MotilityClass.IMMOTILE,
)

#: Default per-aspect probability that the aspect is abnormal. Loosely
#: reflects MHSMA's class balance, where tail and acrosome defects dominate.
DEFAULT_PREVALENCES: Final[dict[str, float]] = {
    "head": 0.25,
    "acrosome": 0.35,
    "vacuole": 0.20,
    "tail": 0.30,
}

#: Given a non-progressive draw, how the remaining mass splits between
#: "moving but going nowhere" and "not moving at all".
NON_PROGRESSIVE_SHARE_OF_NON_PROGRESSIVE: Final[float] = 0.55

#: Given a progressive draw, the share that is rapid rather than slow.
RAPID_SHARE_OF_PROGRESSIVE: Final[float] = 0.65


def _uniform(rng: np.random.Generator, band: tuple[float, float]) -> float:
    """Draw uniformly inside an inclusive band."""
    return float(rng.uniform(band[0], band[1]))


def _outside(
    rng: np.random.Generator,
    band: tuple[float, float],
    *,
    margin: float,
    span: float,
    low_side_prob: float = 0.5,
    floor: float | None = None,
) -> float:
    """Draw a value strictly *outside* ``band``, on a randomly chosen side.

    ``margin`` is the gap left between the band edge and the nearest abnormal
    value, so that normal and abnormal never touch: an abnormal cell must be
    unambiguously abnormal or the label is noise. ``span`` is how far beyond
    the margin the draw may reach.

    ``floor`` clips the low side (e.g. a length may not go negative). When the
    low side collapses onto the floor the draw is forced high instead, which
    keeps "abnormal" meaningful rather than silently returning a normal value.
    """
    lo, hi = band
    go_low = bool(rng.random() < low_side_prob)
    if go_low:
        top = lo - margin
        bottom = top - span
        if floor is not None:
            bottom = max(bottom, floor)
            top = max(top, floor)
        if top - bottom < 1e-9 or (floor is not None and top <= floor + 1e-9):
            go_low = False  # low side is not reachable; use the high side
        else:
            return float(rng.uniform(bottom, top))
    return float(rng.uniform(hi + margin, hi + margin + span))


@dataclass(slots=True)
class HealthState:
    """One virtual sperm's complete ground truth.

    The four ``head`` / ``acrosome`` / ``vacuole`` / ``tail`` fields follow the
    MHSMA convention pinned in :mod:`sperm_sorting.constants`:
    ``0 = normal``, ``1 = abnormal``. They are the *cause* of the continuous
    knobs below, which in turn are the sole inputs to the renderer, so the
    label and the pixels can never disagree.

    Attributes
    ----------
    head, acrosome, vacuole, tail
        Binary morphology aspects, ordered as
        :data:`sperm_sorting.constants.MORPHOLOGY_ASPECTS`.
    motility
        One of :data:`SIMULATED_MOTILITY_CLASSES`.
    head_axis_ratio
        Head length divided by head width. Normal 1.50-1.75.
    head_scale
        Multiplier on overall head size; ``head=1`` may push this out of band
        (macro/microcephaly) instead of, or as well as, the axis ratio.
    acrosome_frac
        Fraction of the head *area* covered by the acrosomal cap. Normal
        0.40-0.70.
    vacuole_size
        Vacuole diameter as a fraction of head length. Normal <= 0.08.
    tail_curvature
        Total bend along the flagellum in radians. Normal <= 0.55; a coiled
        tail runs to several radians.
    tail_length_um
        Flagellum length. Normal 40-50 um; ~0 means an absent tail.
    speed_um_s
        Target curvilinear velocity (VCL) in um/s.
    linearity
        Target LIN = VSL / VCL.
    angle_noise
        Free-running heading diffusion, rad/sqrt(s); used by the scene's
        incremental stepper.
    beat_amplitude_um
        Requested lateral half-amplitude of the beat (ALH / 2), in um.
    beat_frequency_hz
        Ground-truth beat-cross frequency (BCF), in Hz.
    head_length_um
        Derived convenience value used by the renderer to size the head in
        micrometres before conversion to pixels.
    """

    # --- binary morphology ground truth (0 normal / 1 abnormal) ------------
    head: int = LABEL_NORMAL
    acrosome: int = LABEL_NORMAL
    vacuole: int = LABEL_NORMAL
    tail: int = LABEL_NORMAL

    # --- motility ground truth --------------------------------------------
    motility: MotilityClass = MotilityClass.RAPID_PROGRESSIVE

    # --- continuous appearance knobs --------------------------------------
    head_axis_ratio: float = 1.62
    head_scale: float = 1.0
    head_length_um: float = 4.5
    acrosome_frac: float = 0.55
    vacuole_size: float = 0.02
    tail_curvature: float = 0.20
    tail_length_um: float = 45.0

    # --- continuous motion knobs ------------------------------------------
    speed_um_s: float = 70.0
    linearity: float = 0.78
    angle_noise: float = 0.30
    beat_amplitude_um: float = 1.5
    beat_frequency_hz: float = 14.0

    # --- rendering nuisance parameters (not labelled, but not constant) ----
    #: Per-cell contrast multiplier; real cells are not equally dark.
    contrast: float = 1.0
    #: Per-cell defocus in supersampled pixels; the quality gate must be able
    #: to see some cells that are simply out of focus.
    defocus: float = 0.0

    def __post_init__(self) -> None:
        for name in MORPHOLOGY_ASPECTS:
            value = int(getattr(self, name))
            if value not in (LABEL_NORMAL, LABEL_ABNORMAL):
                raise ValueError(
                    f"morphology aspect '{name}' must be "
                    f"{LABEL_NORMAL} or {LABEL_ABNORMAL}, got {value}"
                )
            setattr(self, name, value)
        if self.motility not in SIMULATED_MOTILITY_CLASSES:
            raise ValueError(
                f"motility must be one of {[str(m) for m in SIMULATED_MOTILITY_CLASSES]}, "
                f"got {self.motility!r}"
            )

    # ------------------------------------------------------------------ views

    @property
    def aspects(self) -> tuple[int, int, int, int]:
        """The four binary flags in :data:`MORPHOLOGY_ASPECTS` order."""
        return (self.head, self.acrosome, self.vacuole, self.tail)

    @property
    def head_width_um(self) -> float:
        """Head width implied by length and axis ratio."""
        return self.head_length_um / max(self.head_axis_ratio, 1e-6)

    @property
    def is_morphology_normal(self) -> bool:
        """True when all four aspects are normal."""
        return all(v == LABEL_NORMAL for v in self.aspects)

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-JSON view for ``meta.json`` and audit records."""
        out = asdict(self)
        out["motility"] = str(self.motility)
        return out


@dataclass(slots=True)
class Prevalences:
    """Per-aspect probability of the abnormal label.

    A dataclass rather than a bare dict so that a typo in an aspect name is an
    error at construction rather than a silently-ignored key that quietly
    changes the class balance of a whole dataset.
    """

    head: float = DEFAULT_PREVALENCES["head"]
    acrosome: float = DEFAULT_PREVALENCES["acrosome"]
    vacuole: float = DEFAULT_PREVALENCES["vacuole"]
    tail: float = DEFAULT_PREVALENCES["tail"]

    def __post_init__(self) -> None:
        for name in MORPHOLOGY_ASPECTS:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"prevalence for '{name}' must lie in [0, 1], got {value}"
                )
            setattr(self, name, value)

    @classmethod
    def coerce(cls, value: Prevalences | Mapping[str, float] | None) -> Prevalences:
        """Accept a dict, an instance or ``None`` and always return an instance."""
        if value is None:
            return cls()
        if isinstance(value, Prevalences):
            return value
        unknown = set(value) - set(MORPHOLOGY_ASPECTS)
        if unknown:
            raise ValueError(
                f"unknown morphology aspect(s) in prevalences: {sorted(unknown)}; "
                f"expected a subset of {list(MORPHOLOGY_ASPECTS)}"
            )
        return cls(**{k: float(v) for k, v in value.items()})

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in MORPHOLOGY_ASPECTS}


def sample_motility(
    rng: np.random.Generator, progressive_rate: float
) -> MotilityClass:
    """Draw a motility grade with ``progressive_rate`` mass on progressive.

    The progressive mass is split between rapid and slow, and the remainder
    between non-progressive and immotile, using the module-level shares. Two
    nested Bernoullis rather than one categorical draw so that changing
    ``progressive_rate`` never disturbs the rapid:slow mix.
    """
    if not 0.0 <= progressive_rate <= 1.0:
        raise ValueError(f"progressive_rate must lie in [0, 1], got {progressive_rate}")
    if rng.random() < progressive_rate:
        if rng.random() < RAPID_SHARE_OF_PROGRESSIVE:
            return MotilityClass.RAPID_PROGRESSIVE
        return MotilityClass.SLOW_PROGRESSIVE
    if rng.random() < NON_PROGRESSIVE_SHARE_OF_NON_PROGRESSIVE:
        return MotilityClass.NON_PROGRESSIVE
    return MotilityClass.IMMOTILE


def sample_health_state(
    rng: np.random.Generator,
    prevalences: Prevalences | Mapping[str, float] | None = None,
    progressive_rate: float = 0.6,
    *,
    motility: MotilityClass | None = None,
    aspects: tuple[int, int, int, int] | None = None,
) -> HealthState:
    """Sample one ground-truth :class:`HealthState`.

    The four aspects are drawn independently (real defects do co-occur, but
    modelling that correlation would bake an unverified prior into the only
    labelled data we have; independence keeps the per-aspect heads honest).
    Each flag then *drives* its continuous knobs:

    ``head=1``
        The axis ratio leaves 1.50-1.75 (tapered / round head) and/or the head
        scale leaves 0.90-1.10 (macro / microcephaly). One of the two is always
        violated, sometimes both, so a flagged head is always visibly wrong.
    ``acrosome=1``
        The acrosomal cap covers materially less than 40% or more than 70% of
        the head area.
    ``vacuole=1``
        A resolvable vacuole appears, well above the 0.08 normal ceiling.
    ``tail=1``
        The flagellum is coiled (curvature far beyond 0.55 rad), and/or
        short-to-absent.

    Parameters
    ----------
    rng
        Explicit generator. The global numpy random state is never touched:
        determinism is a hard requirement of this package.
    prevalences
        Per-aspect abnormal probability; see :class:`Prevalences`.
    progressive_rate
        Probability the cell is progressively motile.
    motility, aspects
        Optional overrides used by the dataset builder to balance classes and
        by tests to pin a specific corner of the truth table. When given, the
        corresponding draw is skipped but the causal knob generation is not.
    """
    prev = Prevalences.coerce(prevalences)

    if aspects is None:
        flags = tuple(
            int(rng.random() < getattr(prev, name)) for name in MORPHOLOGY_ASPECTS
        )
    else:
        if len(aspects) != len(MORPHOLOGY_ASPECTS):
            raise ValueError(
                f"aspects must have {len(MORPHOLOGY_ASPECTS)} entries, got {len(aspects)}"
            )
        flags = tuple(int(v) for v in aspects)
    head_f, acro_f, vac_f, tail_f = flags

    grade = sample_motility(rng, progressive_rate) if motility is None else motility

    # -- head geometry -----------------------------------------------------
    head_length = _uniform(rng, NORMAL_HEAD_LENGTH_UM)
    if head_f == LABEL_ABNORMAL:
        # Violate the ratio, the scale, or both -- never neither.
        mode = int(rng.integers(0, 3))
        if mode in (0, 2):
            axis_ratio = _outside(
                rng, NORMAL_HEAD_AXIS_RATIO, margin=0.18, span=0.85, floor=0.55
            )
        else:
            axis_ratio = _uniform(rng, NORMAL_HEAD_AXIS_RATIO)
        if mode in (1, 2):
            head_scale = _outside(
                rng, NORMAL_HEAD_SCALE, margin=0.22, span=0.45, floor=0.35
            )
        else:
            head_scale = _uniform(rng, NORMAL_HEAD_SCALE)
    else:
        axis_ratio = _uniform(rng, NORMAL_HEAD_AXIS_RATIO)
        head_scale = _uniform(rng, NORMAL_HEAD_SCALE)

    # -- acrosome ----------------------------------------------------------
    if acro_f == LABEL_ABNORMAL:
        # Margin 0.12 keeps abnormal caps clear of the 0.40/0.70 boundary.
        acrosome_frac = _outside(
            rng, NORMAL_ACROSOME_FRAC, margin=0.12, span=0.24, floor=0.0
        )
        acrosome_frac = float(np.clip(acrosome_frac, 0.0, 0.98))
    else:
        acrosome_frac = _uniform(rng, NORMAL_ACROSOME_FRAC)

    # -- vacuole -----------------------------------------------------------
    if vac_f == LABEL_ABNORMAL:
        # Only the high side is meaningful: "less than no vacuole" does not
        # exist, so low_side_prob=0 rather than relying on the floor clamp.
        vacuole_size = _outside(
            rng, NORMAL_VACUOLE_SIZE, margin=0.10, span=0.26, low_side_prob=0.0
        )
    else:
        vacuole_size = _uniform(rng, NORMAL_VACUOLE_SIZE)

    # -- tail --------------------------------------------------------------
    if tail_f == LABEL_ABNORMAL:
        mode = int(rng.integers(0, 3))
        if mode in (0, 2):  # coiled / sharply bent
            tail_curvature = _outside(
                rng, NORMAL_TAIL_CURVATURE, margin=0.75, span=4.2, low_side_prob=0.0
            )
        else:
            tail_curvature = _uniform(rng, NORMAL_TAIL_CURVATURE)
        if mode in (1, 2):  # short or absent
            tail_length = float(rng.uniform(0.0, ABNORMAL_SHORT_TAIL_MAX_UM))
        else:
            tail_length = _uniform(rng, NORMAL_TAIL_LENGTH_UM)
    else:
        tail_curvature = _uniform(rng, NORMAL_TAIL_CURVATURE)
        tail_length = _uniform(rng, NORMAL_TAIL_LENGTH_UM)

    # -- motion ------------------------------------------------------------
    speed = _uniform(rng, SPEED_BAND_UM_S[grade])
    linearity = _uniform(rng, LINEARITY_BAND[grade])
    angle_noise = _uniform(rng, ANGLE_NOISE_BAND[grade])
    beat_amplitude = _uniform(rng, BEAT_AMPLITUDE_BAND[grade])
    beat_frequency = _uniform(rng, BEAT_FREQUENCY_BAND[grade])

    # A tail that is absent or coiled cannot propel the cell well. This is a
    # deliberate, documented coupling: it is real biology, and it means the
    # motion channel carries weak evidence about the tail label, which is what
    # a multi-modal model should be able to exploit.
    if tail_f == LABEL_ABNORMAL and grade.is_progressive:
        speed *= float(rng.uniform(0.60, 0.90))

    return HealthState(
        head=head_f,
        acrosome=acro_f,
        vacuole=vac_f,
        tail=tail_f,
        motility=grade,
        head_axis_ratio=axis_ratio,
        head_scale=head_scale,
        head_length_um=head_length,
        acrosome_frac=acrosome_frac,
        vacuole_size=vacuole_size,
        tail_curvature=tail_curvature,
        tail_length_um=tail_length,
        speed_um_s=speed,
        linearity=linearity,
        angle_noise=angle_noise,
        beat_amplitude_um=beat_amplitude,
        beat_frequency_hz=beat_frequency,
        contrast=float(rng.uniform(0.80, 1.20)),
        defocus=float(abs(rng.normal(0.0, 0.35))),
    )


def normal_state(rng: np.random.Generator | None = None) -> HealthState:
    """A fully normal, rapid-progressive reference cell.

    Used by tests and by the docs figure; ``rng=None`` gives the exact mid-band
    cell, which is handy as a fixed visual reference.
    """
    if rng is None:
        return HealthState(
            head_axis_ratio=float(np.mean(NORMAL_HEAD_AXIS_RATIO)),
            head_scale=1.0,
            head_length_um=float(np.mean(NORMAL_HEAD_LENGTH_UM)),
            acrosome_frac=float(np.mean(NORMAL_ACROSOME_FRAC)),
            vacuole_size=0.02,
            tail_curvature=0.15,
            tail_length_um=float(np.mean(NORMAL_TAIL_LENGTH_UM)),
            speed_um_s=float(np.mean(SPEED_BAND_UM_S[MotilityClass.RAPID_PROGRESSIVE])),
            linearity=0.78,
            angle_noise=0.25,
            beat_amplitude_um=1.5,
            beat_frequency_hz=14.0,
            contrast=1.0,
            defocus=0.0,
        )
    return sample_health_state(
        rng, aspects=(0, 0, 0, 0), motility=MotilityClass.RAPID_PROGRESSIVE
    )


def abnormal_state(rng: np.random.Generator | None = None) -> HealthState:
    """A fully abnormal, immotile reference cell (the visual opposite)."""
    if rng is None:
        return HealthState(
            head=1,
            acrosome=1,
            vacuole=1,
            tail=1,
            motility=MotilityClass.IMMOTILE,
            head_axis_ratio=1.05,
            head_scale=1.45,
            head_length_um=5.0,
            acrosome_frac=0.95,
            vacuole_size=0.30,
            tail_curvature=3.6,
            tail_length_um=14.0,
            speed_um_s=0.6,
            linearity=0.05,
            angle_noise=0.2,
            beat_amplitude_um=0.05,
            beat_frequency_hz=0.4,
            contrast=1.0,
            defocus=0.0,
        )
    return sample_health_state(
        rng, aspects=(1, 1, 1, 1), motility=MotilityClass.IMMOTILE
    )


if __name__ == "__main__":  # pragma: no cover - runnable self-check
    rng_ = np.random.default_rng(0)

    # -- determinism ------------------------------------------------------
    a = sample_health_state(np.random.default_rng(7))
    b = sample_health_state(np.random.default_rng(7))
    assert a == b, "same seed must give the same state"
    c = sample_health_state(np.random.default_rng(8))
    assert a != c, "different seeds must give different states"

    # -- the label causes the appearance ----------------------------------
    for _ in range(4000):
        s = sample_health_state(rng_, {"head": 0.5, "acrosome": 0.5, "vacuole": 0.5, "tail": 0.5})
        ratio_ok = NORMAL_HEAD_AXIS_RATIO[0] <= s.head_axis_ratio <= NORMAL_HEAD_AXIS_RATIO[1]
        scale_ok = NORMAL_HEAD_SCALE[0] <= s.head_scale <= NORMAL_HEAD_SCALE[1]
        if s.head == LABEL_ABNORMAL:
            assert not (ratio_ok and scale_ok), "head=1 must break ratio or scale"
        else:
            assert ratio_ok and scale_ok, "head=0 must stay in both normal bands"

        acro_ok = NORMAL_ACROSOME_FRAC[0] <= s.acrosome_frac <= NORMAL_ACROSOME_FRAC[1]
        assert acro_ok == (s.acrosome == LABEL_NORMAL), "acrosome flag must drive the cap"

        vac_ok = s.vacuole_size <= NORMAL_VACUOLE_SIZE[1]
        assert vac_ok == (s.vacuole == LABEL_NORMAL), "vacuole flag must drive the spot"

        curve_ok = s.tail_curvature <= NORMAL_TAIL_CURVATURE[1]
        len_ok = NORMAL_TAIL_LENGTH_UM[0] <= s.tail_length_um <= NORMAL_TAIL_LENGTH_UM[1]
        if s.tail == LABEL_ABNORMAL:
            assert not (curve_ok and len_ok), "tail=1 must break curvature or length"
        else:
            assert curve_ok and len_ok, "tail=0 must stay straight and full length"

        assert 0.0 <= s.acrosome_frac <= 1.0
        assert s.head_width_um > 0.0
        assert s.motility in SIMULATED_MOTILITY_CLASSES

    # -- prevalence is honoured -------------------------------------------
    rng2 = np.random.default_rng(3)
    n = 20000
    flags = np.array(
        [sample_health_state(rng2, {"head": 0.25, "acrosome": 0.35, "vacuole": 0.2, "tail": 0.3}).aspects
         for _ in range(n)]
    )
    empirical = flags.mean(axis=0)
    for name, want, got in zip(MORPHOLOGY_ASPECTS, (0.25, 0.35, 0.20, 0.30), empirical, strict=True):
        assert abs(got - want) < 0.02, f"{name}: prevalence {got:.3f} != {want}"

    # -- progressive_rate is honoured -------------------------------------
    rng3 = np.random.default_rng(11)
    grades = [sample_health_state(rng3, progressive_rate=0.6).motility for _ in range(20000)]
    prog = sum(g.is_progressive for g in grades) / len(grades)
    assert abs(prog - 0.6) < 0.02, f"progressive share {prog:.3f} != 0.6"

    # -- validation --------------------------------------------------------
    try:
        HealthState(head=2)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("HealthState must reject a non-binary aspect")
    try:
        HealthState(motility=MotilityClass.UNDETERMINED)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("HealthState must reject UNDETERMINED as ground truth")
    try:
        Prevalences.coerce({"heads": 0.1})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Prevalences must reject an unknown aspect name")

    assert normal_state().is_morphology_normal
    assert not abnormal_state().is_morphology_normal
    print(f"params.py self-check OK (empirical prevalence {np.round(empirical, 3)}, "
          f"progressive {prog:.3f})")
