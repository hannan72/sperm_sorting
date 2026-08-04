"""FastAPI service behind the sperm-analysis demo page.

What this service is for
------------------------
Three things about this prototype are routinely misunderstood, and each of them
is misunderstood in a way that would produce a wrong device. The demo exists to
make all three impossible to misread:

1. **The generator is the ground truth.** ``/generate`` samples one
   :class:`~sperm_sorting.simulator.params.HealthState`, then renders the image
   *and* simulates the trajectory from that same state. The labels are not
   annotations of the picture; the picture is a consequence of the labels. Once
   that is on screen next to a prediction, a wrong prediction is visible per
   aspect instead of being averaged into an accuracy number.
2. **The decision rule.** ``/decide`` calls
   :func:`sperm_sorting.decision.engine.decide` -- the real one, unmodified.
   Exactly 60% is a REJECT, and REJECT drives ``FIELD_ON``, because energising
   the magnet is what diverts a segment to waste. Both halves have been "fixed"
   into bugs before, so neither the API nor the JavaScript is allowed to hold a
   second copy of the rule.
3. **The optical budget.** ``/config`` returns
   :func:`sperm_sorting.shots.feasibility.assess_feasibility` verbatim,
   warnings included, so the fact that a whole spermatozoon does not fit across
   the field of view is stated by the same code that warns about it at startup.

Honesty constraints
-------------------
No morphology weights exist for this device yet. ``/classify`` therefore falls
back to :class:`~sperm_sorting.morphology.inference.RandomMorphologyEngine`,
which draws seeded noise and never looks at the pixels, and every classify
response carries a ``model`` block whose ``provenance`` and ``untrained_warning``
fields the page renders prominently. A demo that looks like a working classifier
when it is not is worse than no demo: it manufactures confidence that the
project has not earned.

The service is deliberately stateless apart from the three objects built once in
the lifespan handler (configuration, morphology engine, feasibility report).
Every request recomputes from its own inputs, so two browsers cannot interfere.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from sperm_sorting.config import AppConfig, load_config
from sperm_sorting.constants import (
    ACCEPT_RATIO_THRESHOLD,
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    MAXIMUM_SHOT_DURATION_S,
    MAXIMUM_TRACKABLE_SPERM,
    MINIMUM_TRACKABLE_SPERM,
    MORPHOLOGY_ASPECTS,
    TARGET_TRACKABLE_SPERM,
)
from sperm_sorting.decision.engine import decide
from sperm_sorting.morphology.inference import (
    BaseMorphologyEngine,
    RandomMorphologyEngine,
)
from sperm_sorting.morphology.polarity import describe_polarity, flip_polarity
from sperm_sorting.motion.classifier import classify_motility
from sperm_sorting.schemas.enums import (
    FieldCommandKind,
    FlowCorrectionMode,
    MorphologyStatus,
    MotilityClass,
    TimestampSource,
)
from sperm_sorting.schemas.morphology import AspectResult, MorphologyResult
from sperm_sorting.schemas.track import MotionFeatures, TrackRecord
from sperm_sorting.shots.feasibility import FeasibilityReport, assess_feasibility
from sperm_sorting.simulator.label import (
    MOTILITY_LABEL_NAMES,
    OVERALL_LABEL_NAMES,
    aspect_labels,
    morphology_label,
    motility_label,
    overall_label,
)
from sperm_sorting.simulator.motility import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    casa_features,
    normalize_features,
    simulate_trajectory,
)
from sperm_sorting.simulator.params import (
    ABNORMAL_SHORT_TAIL_MAX_UM,
    DEFAULT_PREVALENCES,
    NORMAL_ACROSOME_FRAC,
    NORMAL_HEAD_AXIS_RATIO,
    NORMAL_HEAD_LENGTH_UM,
    NORMAL_HEAD_SCALE,
    NORMAL_TAIL_CURVATURE,
    NORMAL_TAIL_LENGTH_UM,
    NORMAL_VACUOLE_SIZE,
    SIMULATED_MOTILITY_CLASSES,
    HealthState,
    Prevalences,
    sample_health_state,
)
from sperm_sorting.simulator.render import (
    CROP_FIELD_UM,
    SUPPORTED_SIZES,
    RenderConfig,
    render_sperm,
)

logger = logging.getLogger("web.app")

# ==========================================================================
# Constants that the page and the tests both depend on
# ==========================================================================

STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "static"
INDEX_HTML: Final[Path] = STATIC_DIR / "index.html"

#: Verbatim, single-line disclaimer. It appears in ``static/index.html`` as one
#: uninterrupted string so that ``test_api`` can assert the served page really
#: carries it; splitting it across lines in the HTML would break that check,
#: which is the point -- the check exists to stop the banner being quietly
#: softened or removed.
DISCLAIMER: Final[str] = (
    "Research prototype, not a medical device. It does not measure DNA "
    "integrity, apoptosis or fertility."
)

#: Rendered wherever a prediction is shown. The wording is fixed here so the
#: API and the page cannot drift apart on how strong the caveat is.
UNTRAINED_WARNING: Final[str] = "untrained — predictions are not meaningful"

#: Provenance strings that mean "this did not come from trained weights". The
#: random engine's own string is deliberately not one of the
#: ``WEIGHTS_PROVENANCE_*`` constants; ``unset`` is the config default and means
#: nobody has said where the weights came from, which is equally untrustworthy.
UNTRAINED_PROVENANCES: Final[frozenset[str]] = frozenset(
    {"random-test-engine", "unset", ""}
)

#: What each field command physically does. Written down because "FIELD_ON =
#: good" is the single most common misreading of this product, and a UI that
#: only shows the enum name invites exactly that reading.
FIELD_COMMAND_MEANING: Final[dict[str, str]] = {
    "FIELD_ON": (
        "magnet energised — the segment is diverted to the waste channel. "
        "FIELD_ON is the rejection."
    ),
    "FIELD_OFF": (
        "magnet de-energised — the segment passes through to collection. "
        "FIELD_OFF is the pass-through, and is also the safe state used when no "
        "decision could be made."
    ),
}

#: Head length used for the "how many pixels does a head span" figure, in um.
#: WHO 6th-edition median; passed explicitly to :func:`assess_feasibility` so
#: the number reported here is the number that function actually used.
HEAD_LENGTH_UM: Final[float] = 4.1

#: Whole-cell length used for the "does a spermatozoon fit" check, in um.
SPERM_LENGTH_UM: Final[float] = 53.1

#: Continuous ``HealthState`` fields a caller may override, with the slider
#: metadata the page needs and -- crucially -- whether overriding the field
#: breaks the guarantee that the pixels follow from the label.
#:
#: The appearance and motion knobs are *caused* by the binary aspect flags
#: (see :func:`sample_health_state`). Setting one by hand keeps the label but
#: changes the evidence, so the label and the picture can disagree. That is
#: occasionally what you want -- it is the only way to see what a 2.6 axis ratio
#: looks like -- but it must be announced, so every override is echoed back in
#: ``overridden_knobs`` and the page shows a warning while any are active.
KNOB_SPECS: Final[tuple[dict[str, Any], ...]] = (
    {
        "name": "head_axis_ratio",
        "label": "head axis ratio (length / width)",
        "group": "morphology",
        "min": 0.5,
        "max": 3.0,
        "step": 0.01,
        "unit": "",
        "normal_band": list(NORMAL_HEAD_AXIS_RATIO),
        "breaks_label_link": True,
        "driven_by": "head",
    },
    {
        "name": "head_scale",
        "label": "head size multiplier",
        "group": "morphology",
        "min": 0.3,
        "max": 2.0,
        "step": 0.01,
        "unit": "x",
        "normal_band": list(NORMAL_HEAD_SCALE),
        "breaks_label_link": True,
        "driven_by": "head",
    },
    {
        "name": "head_length_um",
        "label": "head length",
        "group": "morphology",
        "min": 2.0,
        "max": 8.0,
        "step": 0.05,
        "unit": "um",
        "normal_band": list(NORMAL_HEAD_LENGTH_UM),
        "breaks_label_link": True,
        "driven_by": "head",
    },
    {
        "name": "acrosome_frac",
        "label": "acrosomal cap area fraction",
        "group": "morphology",
        "min": 0.0,
        "max": 0.98,
        "step": 0.01,
        "unit": "",
        "normal_band": list(NORMAL_ACROSOME_FRAC),
        "breaks_label_link": True,
        "driven_by": "acrosome",
    },
    {
        "name": "vacuole_size",
        "label": "vacuole diameter / head length",
        "group": "morphology",
        "min": 0.0,
        "max": 0.45,
        "step": 0.005,
        "unit": "",
        "normal_band": list(NORMAL_VACUOLE_SIZE),
        "breaks_label_link": True,
        "driven_by": "vacuole",
    },
    {
        "name": "tail_curvature",
        "label": "total flagellar bend",
        "group": "morphology",
        "min": 0.0,
        "max": 6.0,
        "step": 0.05,
        "unit": "rad",
        "normal_band": list(NORMAL_TAIL_CURVATURE),
        "breaks_label_link": True,
        "driven_by": "tail",
    },
    {
        "name": "tail_length_um",
        "label": "flagellum length",
        "group": "morphology",
        "min": 0.0,
        "max": 60.0,
        "step": 0.5,
        "unit": "um",
        "normal_band": list(NORMAL_TAIL_LENGTH_UM),
        "breaks_label_link": True,
        "driven_by": "tail",
        "note": (
            f"abnormally short tails are drawn below "
            f"{ABNORMAL_SHORT_TAIL_MAX_UM:.0f} um"
        ),
    },
    {
        "name": "speed_um_s",
        "label": "target VCL",
        "group": "motion",
        "min": 0.0,
        "max": 200.0,
        "step": 1.0,
        "unit": "um/s",
        "normal_band": None,
        "breaks_label_link": True,
        "driven_by": "motility",
    },
    {
        "name": "linearity",
        "label": "target LIN (VSL / VCL)",
        "group": "motion",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "unit": "",
        "normal_band": None,
        "breaks_label_link": True,
        "driven_by": "motility",
    },
    {
        "name": "beat_amplitude_um",
        "label": "beat half-amplitude (ALH / 2)",
        "group": "motion",
        "min": 0.0,
        "max": 8.0,
        "step": 0.05,
        "unit": "um",
        "normal_band": None,
        "breaks_label_link": True,
        "driven_by": "motility",
    },
    {
        "name": "beat_frequency_hz",
        "label": "beat-cross frequency (BCF)",
        "group": "motion",
        "min": 0.0,
        "max": 40.0,
        "step": 0.5,
        "unit": "Hz",
        "normal_band": None,
        "breaks_label_link": True,
        "driven_by": "motility",
    },
    {
        "name": "contrast",
        "label": "per-cell contrast",
        "group": "imaging",
        "min": 0.3,
        "max": 2.0,
        "step": 0.01,
        "unit": "x",
        "normal_band": None,
        "breaks_label_link": False,
        "driven_by": None,
    },
    {
        "name": "defocus",
        "label": "per-cell defocus",
        "group": "imaging",
        "min": 0.0,
        "max": 4.0,
        "step": 0.05,
        "unit": "px",
        "normal_band": None,
        "breaks_label_link": False,
        "driven_by": None,
    },
)

_KNOB_NAMES: Final[frozenset[str]] = frozenset(spec["name"] for spec in KNOB_SPECS)

#: The five cases the specification mandates the demo demonstrate. Preloaded as
#: one-click buttons; the verdicts are still fetched from ``/decide`` so the
#: page never asserts an outcome it did not receive from the real engine.
MANDATED_DECISION_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "id": "15-25",
        "ai_eligible_count": 15,
        "trackable_count": 25,
        "caption": "15 / 25 = exactly 60%",
        "why": "the boundary case; a strict '>' makes exactly 60% a REJECT",
    },
    {
        "id": "16-25",
        "ai_eligible_count": 16,
        "trackable_count": 25,
        "caption": "16 / 25 = 64%",
        "why": "one sperm more than the boundary is the first accepting count",
    },
    {
        "id": "12-20",
        "ai_eligible_count": 12,
        "trackable_count": 20,
        "caption": "12 / 20 = exactly 60%",
        "why": "the same boundary at the minimum shot size",
    },
    {
        "id": "13-20",
        "ai_eligible_count": 13,
        "trackable_count": 20,
        "caption": "13 / 20 = 65%",
        "why": "accepted at the minimum shot size",
    },
    {
        "id": "19-timeout",
        "ai_eligible_count": 19,
        "trackable_count": 19,
        "caption": f"19 trackable at the {MAXIMUM_SHOT_DURATION_S:.0f} s timeout",
        "why": (
            f"below the minimum of {MINIMUM_TRACKABLE_SPERM}, so no ratio is "
            "trusted — INDETERMINATE even though every sperm was eligible"
        ),
    },
)


# ==========================================================================
# Request / response models
# ==========================================================================


class PrevalenceModel(BaseModel):
    """Per-aspect probability that the sampled aspect comes out abnormal.

    Mirrors :class:`sperm_sorting.simulator.params.Prevalences` rather than
    re-deriving its validation: the values are handed straight to
    ``Prevalences.coerce``, which is the module that owns the range check.
    """

    head: float = Field(default=DEFAULT_PREVALENCES["head"], ge=0.0, le=1.0)
    acrosome: float = Field(default=DEFAULT_PREVALENCES["acrosome"], ge=0.0, le=1.0)
    vacuole: float = Field(default=DEFAULT_PREVALENCES["vacuole"], ge=0.0, le=1.0)
    tail: float = Field(default=DEFAULT_PREVALENCES["tail"], ge=0.0, le=1.0)

    def to_prevalences(self) -> Prevalences:
        return Prevalences.coerce(self.model_dump())


class GenerateRequest(BaseModel):
    """Everything that determines one virtual sperm and how it is observed.

    A single ``seed`` fixes the whole response. Three independent child
    generators are spawned from it -- one for the health state, one for the
    render, one for the trajectory -- so that changing the frame rate cannot
    silently change which cell you are looking at.
    """

    seed: int = Field(default=1234, ge=0, le=2**32 - 1)
    prevalences: PrevalenceModel = Field(default_factory=PrevalenceModel)
    #: ``None`` samples a grade using ``progressive_rate``; a name forces it.
    motility: str | None = None
    #: Probability of a progressive grade when ``motility`` is ``None``.
    progressive_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    #: Force the four binary aspect flags instead of drawing them. Used by the
    #: tests to pin a corner of the truth table, and by the page's "flip one
    #: aspect" buttons.
    aspects: list[int] | None = None
    #: Explicit overrides for continuous knobs. See :data:`KNOB_SPECS`.
    knobs: dict[str, float] = Field(default_factory=dict)

    # -- imaging -----------------------------------------------------------
    image_size: int = 128
    #: ``None`` derives the crop scale from ``CROP_FIELD_UM`` and the size, so
    #: every supported size shows the same field of view.
    image_um_per_px: float | None = Field(default=None, gt=0.0, le=5.0)
    blur_px: float = Field(default=0.6, ge=0.0, le=6.0)
    noise_sigma: float = Field(default=4.0, ge=0.0, le=40.0)

    # -- tracking ----------------------------------------------------------
    n_points: int = Field(default=96, ge=8, le=1024)
    fps: float = Field(default=160.0, gt=1.0, le=2000.0)
    #: Sample-plane sampling assumed by the tracker, um/px.
    track_um_per_px: float = Field(default=0.5, gt=0.0, le=5.0)
    flow_vx_px_s: float = Field(default=0.0, ge=-2000.0, le=2000.0)
    flow_vy_px_s: float = Field(default=0.0, ge=-2000.0, le=2000.0)

    @model_validator(mode="after")
    def _check(self) -> GenerateRequest:
        if self.image_size not in SUPPORTED_SIZES:
            raise ValueError(
                f"image_size must be one of {list(SUPPORTED_SIZES)} (MHSMA parity), "
                f"got {self.image_size}"
            )
        if self.motility is not None and self.motility not in {
            str(m) for m in SIMULATED_MOTILITY_CLASSES
        }:
            raise ValueError(
                "motility must be null or one of "
                f"{[str(m) for m in SIMULATED_MOTILITY_CLASSES]}, got "
                f"{self.motility!r}"
            )
        if self.aspects is not None:
            if len(self.aspects) != len(MORPHOLOGY_ASPECTS):
                raise ValueError(
                    f"aspects must have {len(MORPHOLOGY_ASPECTS)} entries in "
                    f"{list(MORPHOLOGY_ASPECTS)} order, got {len(self.aspects)}"
                )
            for value in self.aspects:
                if value not in (LABEL_NORMAL, LABEL_ABNORMAL):
                    raise ValueError(
                        f"aspect flags must be {LABEL_NORMAL} or {LABEL_ABNORMAL}, "
                        f"got {value}"
                    )
        unknown = sorted(set(self.knobs) - _KNOB_NAMES)
        if unknown:
            raise ValueError(
                f"unknown knob(s): {unknown}; expected a subset of "
                f"{sorted(_KNOB_NAMES)}"
            )
        return self


class ClassifyRequest(BaseModel):
    """Either an image to judge, or a seed/params pair to regenerate one.

    The two forms are not equivalent and the response says which was used. An
    image alone carries no trajectory, so no motility grade can be produced from
    it; the seed form regenerates the same virtual sperm and therefore has the
    kinematics as well.
    """

    #: Base64 PNG, with or without a ``data:image/png;base64,`` prefix.
    image: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    params: GenerateRequest | None = None

    @model_validator(mode="after")
    def _one_of(self) -> ClassifyRequest:
        if self.image is None and self.seed is None and self.params is None:
            raise ValueError(
                "supply either {image: <base64 png>} or {seed, params}; with "
                "neither there is nothing to classify"
            )
        return self

    def resolved_params(self) -> GenerateRequest | None:
        """The generate request implied by this classify request, if any.

        A bare ``seed`` is enough: it means "the default scene at that seed".
        A ``seed`` given alongside ``params`` overrides the one inside them, so
        the page can re-roll without rebuilding the whole parameter block.
        """
        if self.params is None and self.seed is None:
            return None
        base = self.params or GenerateRequest()
        if self.seed is None:
            return base
        return base.model_copy(update={"seed": self.seed})


class DecideRequest(BaseModel):
    """Inputs to the decision rule.

    ``threshold`` and ``minimum_trackable`` default to the resolved
    configuration rather than to literals, so the demo cannot demonstrate a rule
    the device is not configured for.
    """

    ai_eligible_count: int = Field(ge=0, le=10_000)
    trackable_count: int = Field(ge=0, le=10_000)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    minimum_trackable: int | None = Field(default=None, ge=0, le=10_000)


# ==========================================================================
# Helpers
# ==========================================================================


def _encode_png(image: np.ndarray) -> str:
    """Base64-encode a ``uint8`` grayscale array as a PNG.

    No ``data:`` prefix: the payload is the PNG itself so a caller can
    ``base64.b64decode`` it straight into an image library. The page adds the
    prefix when it builds the ``src``. Pillow writes no timestamp chunk, so the
    same array always yields the same bytes -- which is what makes the
    "same seed, byte-identical response" guarantee testable.
    """
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_png(payload: str) -> np.ndarray:
    """Decode a base64 PNG into a ``uint8`` grayscale array.

    Accepts a bare base64 string or a full data URI, because the page has one
    and a script has the other, and rejecting either would be a papercut with no
    safety benefit.
    """
    text = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"image is not valid base64: {exc}"
        ) from exc
    try:
        with Image.open(io.BytesIO(raw)) as handle:
            return np.asarray(handle.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise HTTPException(
            status_code=422, detail=f"image is not a decodable PNG: {exc}"
        ) from exc


def _spawn(seed: int, count: int) -> list[np.random.Generator]:
    """Independent child generators from one seed.

    ``SeedSequence.spawn`` rather than ``default_rng(seed + k)``: seeds that
    differ by a small integer are not guaranteed to give independent streams,
    and the whole point of this demo is that the picture and the track describe
    the *same* cell without one biasing the other.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]


def _apply_knobs(state: HealthState, knobs: dict[str, float]) -> list[str]:
    """Apply continuous overrides in place; return the names actually changed.

    Values equal to the sampled value are still reported as overridden, because
    the caller *asked* for a fixed value and would otherwise see the override
    silently vanish when a re-roll happened to land on the same number.
    """
    applied: list[str] = []
    for spec in KNOB_SPECS:
        name = str(spec["name"])
        if name not in knobs:
            continue
        value = float(np.clip(float(knobs[name]), spec["min"], spec["max"]))
        setattr(state, name, value)
        applied.append(name)
    return applied


def _motion_features_from_casa(
    casa: dict[str, float],
    *,
    n_points: int,
    dt_s: float,
    um_per_px: float,
    flow_px_s: tuple[float, float],
) -> MotionFeatures:
    """Package simulated CASA kinematics as the record the classifier expects.

    The simulator's :func:`casa_features` and the runtime's
    :mod:`sperm_sorting.motion` are independent implementations on purpose. This
    adapter lets the demo feed the simulator's numbers into the *real* grading
    rule (:func:`classify_motility`), so the motility prediction on the page is
    the production rule applied to simulated measurements rather than a fourth
    copy of the WHO cut-points.

    ``optically_calibrated`` is set true because the simulator knows its own
    scale exactly. That is a property of the simulation, not a claim about the
    device -- ``/config`` reports the device's calibration state separately, and
    it is ``False``.
    """
    return MotionFeatures(
        n_points=n_points,
        n_observed_points=n_points,
        duration_s=(n_points - 1) * dt_s,
        mean_frame_interval_s=dt_s,
        timestamp_source=TimestampSource.SYNTHETIC,
        flow_correction_mode=(
            FlowCorrectionMode.FIXED_VECTOR
            if any(abs(v) > 0.0 for v in flow_px_s)
            else FlowCorrectionMode.DISABLED
        ),
        optically_calibrated=True,
        um_per_px=um_per_px,
        vcl_um_s=casa["vcl"],
        vsl_um_s=casa["vsl"],
        vap_um_s=casa["vap"],
        vcl_corrected_px_s=casa["vcl"] / um_per_px,
        vsl_corrected_px_s=casa["vsl"] / um_per_px,
        vap_corrected_px_s=casa["vap"] / um_per_px,
        lin=casa["lin"],
        str_=casa["str"],
        wob=casa["wob"],
        alh_um=casa["alh"],
        bcf_hz=casa["bcf"],
        flow_vx_px_s=flow_px_s[0],
        flow_vy_px_s=flow_px_s[1],
    )


def _simulate(req: GenerateRequest) -> dict[str, Any]:
    """Sample one virtual sperm and observe it. The core of ``/generate``.

    Returns a plain dict rather than a response model so that ``/classify`` can
    reuse it without paying for a second round of serialisation. Everything in
    it is a function of ``req`` alone.
    """
    state_rng, render_rng, track_rng = _spawn(req.seed, 3)

    motility = MotilityClass(req.motility) if req.motility is not None else None
    aspects = (
        (req.aspects[0], req.aspects[1], req.aspects[2], req.aspects[3])
        if req.aspects is not None
        else None
    )
    state = sample_health_state(
        state_rng,
        req.prevalences.to_prevalences(),
        req.progressive_rate,
        motility=motility,
        aspects=aspects,
    )
    overridden = _apply_knobs(state, req.knobs)

    render_cfg = RenderConfig(blur_px=req.blur_px, noise_sigma=req.noise_sigma)
    image = render_sperm(
        state,
        (req.image_size, req.image_size),
        render_rng,
        req.image_um_per_px,
        render_cfg,
    )

    dt_s = 1.0 / req.fps
    flow = (req.flow_vx_px_s, req.flow_vy_px_s)
    track = simulate_trajectory(
        state,
        req.n_points,
        dt_s,
        req.track_um_per_px,
        track_rng,
        flow_px_s=flow,
    )

    # Two CASA readings: what a camera would see, and what the pipeline grades.
    # They differ only when a bulk flow is present, and the difference is the
    # whole reason flow correction exists -- a 300 um/s stream makes a dead cell
    # look rapidly progressive.
    casa_observed = casa_features(track, dt_s, req.track_um_per_px)
    if any(abs(v) > 0.0 for v in flow):
        drift = np.arange(req.n_points, dtype=np.float64)[:, None] * dt_s * np.array(flow)
        casa_corrected = casa_features(track - drift, dt_s, req.track_um_per_px)
    else:
        casa_corrected = dict(casa_observed)

    image_scale = (
        CROP_FIELD_UM / float(req.image_size)
        if req.image_um_per_px is None
        else float(req.image_um_per_px)
    )
    return {
        "state": state,
        "image": image,
        "image_um_per_px": image_scale,
        "track": track,
        "dt_s": dt_s,
        "casa_observed": casa_observed,
        "casa_corrected": casa_corrected,
        "overridden": overridden,
        "flow": flow,
    }


def _truth_block(state: HealthState) -> dict[str, Any]:
    """Ground-truth labels for one state, from :mod:`.simulator.label` only."""
    labels = aspect_labels(state)
    overall = overall_label(state)
    motility_index = motility_label(state)
    return {
        "true_label": overall,
        "true_label_name": OVERALL_LABEL_NAMES[overall],
        "true_aspects": {
            name: int(labels[index])
            for index, name in enumerate(MORPHOLOGY_ASPECTS)
        },
        "true_morphology_label": morphology_label(state),
        "true_motility": str(state.motility),
        "true_motility_label": motility_index,
        "true_motility_label_name": MOTILITY_LABEL_NAMES[motility_index],
    }


def _build_engine(cfg: AppConfig) -> tuple[BaseMorphologyEngine, str]:
    """Construct the morphology engine, honestly reporting what was possible.

    The real engine is attempted first, every time, so that the day weights
    appear the demo starts using them without a code change. When it cannot be
    built the reason is carried into the response rather than swallowed: "no
    weights are configured" and "the weights failed to load" are very different
    situations and a viewer is entitled to know which one they are looking at.

    The fallback's ``p_normal_rate`` is 0.5 rather than the class's default of
    0.87. A 0.87 rate would produce a table that agrees with the ground truth
    most of the time and would read as a model that mostly works. A coin flip
    reads as what it is.
    """
    weights = cfg.morphology.weights
    if weights is not None and Path(weights).is_file():
        try:
            from sperm_sorting.morphology.factory import build_morphology_engine

            engine = build_morphology_engine(cfg.morphology)
            engine.warmup(2)
            return engine, f"loaded morphology weights from {weights}"
        # Broad on purpose: a torch import failure, a corrupt checkpoint and a
        # polarity mismatch all mean the same thing here -- there are no usable
        # weights -- and the demo must degrade to the honest fallback rather
        # than refuse to start. The exception text is carried into the response.
        except Exception as exc:
            reason = (
                f"morphology weights at {weights} could not be loaded ({exc}); "
                "falling back to the untrained random engine"
            )
            logger.warning(reason)
    else:
        reason = (
            "no morphology weights are configured (morphology.weights is "
            f"{weights!r}), so there is nothing trained to serve; falling back "
            "to the untrained random engine"
        )
        logger.warning(reason)
    return (
        RandomMorphologyEngine(seed=20240804, p_normal_rate=0.5, model_id="untrained"),
        reason,
    )


def _model_block(app: FastAPI) -> dict[str, Any]:
    """Provenance block attached to every ``/classify`` response.

    Every field here exists to close off a way the demo could be mistaken for a
    working classifier: ``trained`` for a machine reader, ``untrained_warning``
    and ``headline`` for the page, ``reads_the_image`` because a random engine
    ignoring its input is the most surprising property of all, and
    ``deterministic`` because classifying the same crop twice gives different
    answers and a viewer who did not expect that should find it explained
    rather than alarming.
    """
    engine: BaseMorphologyEngine = app.state.engine
    described = engine.describe()
    thresholds: dict[str, float] = dict(getattr(engine, "thresholds", {}))
    provenance = str(described.get("weights_provenance", ""))
    untrained = provenance in UNTRAINED_PROVENANCES
    return {
        "provenance": provenance,
        "trained": not untrained,
        "untrained": untrained,
        "headline": "UNTRAINED MODEL" if untrained else "trained weights loaded",
        "untrained_warning": UNTRAINED_WARNING if untrained else "",
        "detail": (
            app.state.engine_reason
            + ". The random engine draws a probability per aspect from a seeded "
            "generator and never looks at the pixels, so every prediction on "
            "this page is noise. Read the table as a demonstration of the "
            "comparison, not of the model."
            if untrained
            else app.state.engine_reason
        ),
        "engine_class": type(engine).__name__,
        "model_id": described.get("model_id", ""),
        "reads_the_image": not isinstance(engine, RandomMorphologyEngine),
        "deterministic": not isinstance(engine, RandomMorphologyEngine),
        "aspects": list(described.get("aspects", MORPHOLOGY_ASPECTS)),
        "thresholds": {name: float(value) for name, value in thresholds.items()},
        "label_polarity": described.get("label_polarity", ""),
    }


def _finite(value: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively.

    A zero-flow bench configuration gives an infinite residence time and an
    infinite implied concentration, both of which are meaningful answers. JSON
    has no way to spell them, and FastAPI serialises with ``allow_nan=False``,
    so an honest ``null`` beats a 500 on a legitimate configuration. The page
    renders ``null`` as "not applicable" next to the warning that explains why.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _feasibility_block(report: FeasibilityReport) -> dict[str, Any]:
    """The optical budget, plus the derived figures the page shows.

    ``head_span_px`` and the "does a whole cell fit" comparison are computed
    from the report's own ``um_per_px`` and from the same morphometry constants
    passed into :func:`assess_feasibility`, so the panel cannot quote a sampling
    figure that disagrees with the warning printed underneath it.
    """
    block = report.to_json_dict()
    block["sperm_length_um"] = report.sperm_length_um
    block["head_length_um"] = HEAD_LENGTH_UM
    block["head_span_px"] = HEAD_LENGTH_UM / report.um_per_px
    block["head_width_span_px"] = (HEAD_LENGTH_UM / 1.5) / report.um_per_px
    block["sperm_span_px"] = report.sperm_length_um / report.um_per_px
    block["field_width_px"] = report.field_width_um / report.um_per_px
    block["field_height_px"] = report.field_height_um / report.um_per_px
    block["fraction_of_sperm_across_field"] = (
        min(report.field_width_um, report.field_height_um) / report.sperm_length_um
    )
    block["formatted"] = report.format_report()
    return _finite(block)


# ==========================================================================
# Application
# ==========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the three long-lived objects once, tear the engine down cleanly.

    Lifespan rather than ``@app.on_event``: the event decorators are deprecated,
    and more importantly they give no place to run shutdown code that is
    guaranteed to pair with the startup code. The morphology engine may own an
    ONNX session once real weights exist, and leaking one per reload in
    development is the kind of thing that is discovered much later, on hardware.
    """
    cfg = load_config()
    engine, reason = _build_engine(cfg)
    app.state.config = cfg
    app.state.engine = engine
    app.state.engine_reason = reason
    # np.random.Generator is not thread-safe and FastAPI runs def endpoints in a
    # worker thread pool, so two concurrent /classify calls would race on the
    # random engine's stream. The lock costs nothing at demo request rates.
    app.state.engine_lock = threading.Lock()
    app.state.feasibility = assess_feasibility(
        cfg, sperm_length_um=SPERM_LENGTH_UM, head_length_um=HEAD_LENGTH_UM
    )
    logger.info("web demo ready: %s", reason)
    try:
        yield
    finally:
        engine.close()


app = FastAPI(
    title="Sperm-analysis research demo",
    description=DISCLAIMER,
    version="0.1.0",
    lifespan=lifespan,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe. Cheap, and says what is actually loaded."""
    engine: BaseMorphologyEngine = app.state.engine
    return {
        "status": "ok",
        "morphology_engine": type(engine).__name__,
        "weights_provenance": engine.weights_provenance,
        "trained": engine.weights_provenance not in UNTRAINED_PROVENANCES,
        "disclaimer": DISCLAIMER,
    }


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    """The demo page itself, served from disk with no templating.

    The device runs offline, so there is no CDN, no build step and no framework:
    three static files and this handler. ``no-store`` because the page is edited
    while the server is running and a stale cached copy during a demo is a
    needless way to look broken.
    """
    if not INDEX_HTML.is_file():
        raise HTTPException(
            status_code=500, detail=f"static page is missing at {INDEX_HTML}"
        )
    return FileResponse(
        INDEX_HTML, media_type="text/html", headers={"Cache-Control": "no-store"}
    )


@app.get("/aspects")
def aspects() -> dict[str, Any]:
    """The canonical vocabulary, so the page hard-codes none of it.

    Aspect order, the 0/1 label convention, the motility class names, the CASA
    feature order and the slider ranges are all properties of the Python
    package. A frontend copy of any of them is a second source of truth that
    will eventually disagree, and a UI that silently renders 'head' where the
    model meant 'tail' is exactly the failure this endpoint prevents.
    """
    return {
        "aspects": list(MORPHOLOGY_ASPECTS),
        "label_normal": LABEL_NORMAL,
        "label_abnormal": LABEL_ABNORMAL,
        "label_names": {
            str(LABEL_NORMAL): "normal",
            str(LABEL_ABNORMAL): "abnormal",
        },
        "overall_label_names": list(OVERALL_LABEL_NAMES),
        "motility_label_names": list(MOTILITY_LABEL_NAMES),
        "motility_classes": [str(m) for m in SIMULATED_MOTILITY_CLASSES],
        "progressive_classes": [
            str(m) for m in SIMULATED_MOTILITY_CLASSES if m.is_progressive
        ],
        "casa_feature_names": list(FEATURE_NAMES),
        "casa_feature_scales": dict(FEATURE_SCALES),
        "knobs": [dict(spec) for spec in KNOB_SPECS],
        "default_prevalences": dict(DEFAULT_PREVALENCES),
        "polarity": describe_polarity(),
        "health_rule": (
            "healthy (0) requires all four morphology aspects normal AND a "
            "progressive motility grade (rapid or slow). Any single defect, or "
            "any non-progressive grade, makes it unhealthy (1)."
        ),
        "field_command_meaning": dict(FIELD_COMMAND_MEANING),
        "mandated_decision_cases": [dict(case) for case in MANDATED_DECISION_CASES],
        "disclaimer": DISCLAIMER,
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    """Sample one virtual sperm; return its picture, its track and its truth.

    The image and the trajectory describe the *same* ``HealthState``. That is
    the claim the whole demo rests on, and it is why both are produced here in
    one call rather than by two endpoints a caller could accidentally seed
    differently.
    """
    try:
        sim = _simulate(req)
    except ValueError as exc:
        # Range violations that pydantic cannot express (for example a
        # trajectory too short for the CASA window) surface as 422, not 500:
        # they are bad input, not a broken server.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    state: HealthState = sim["state"]
    casa = sim["casa_corrected"]
    body: dict[str, Any] = {
        "seed": req.seed,
        "image": _encode_png(sim["image"]),
        "image_format": "png",
        "image_shape": [int(sim["image"].shape[0]), int(sim["image"].shape[1])],
        "image_um_per_px": sim["image_um_per_px"],
        "image_field_um": sim["image_um_per_px"] * float(sim["image"].shape[1]),
        "trajectory": [[float(x), float(y)] for x, y in sim["track"]],
        "trajectory_units": "pixels",
        "track_um_per_px": req.track_um_per_px,
        "dt_s": sim["dt_s"],
        "fps": req.fps,
        "casa": casa,
        "casa_observed": sim["casa_observed"],
        "casa_normalized": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES, normalize_features(casa), strict=True
            )
        },
        "flow_px_s": list(sim["flow"]),
        "flow_correction_applied": any(abs(v) > 0.0 for v in sim["flow"]),
        "state": state.to_json_dict(),
        "overridden_knobs": sim["overridden"],
        "label_pixel_link_intact": not any(
            spec["breaks_label_link"]
            for spec in KNOB_SPECS
            if spec["name"] in sim["overridden"]
        ),
        "generator": {
            "source": "sperm_sorting.simulator",
            "is_ground_truth": True,
            "note": (
                "the labels were drawn first and the picture and the track were "
                "generated from them; nothing here was annotated after the fact"
            ),
        },
    }
    body.update(_truth_block(state))
    return body


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict[str, Any]:
    """Predict the four aspects and the motility grade. **Untrained.**

    Morphology comes from the loaded engine, which today is the random one. The
    motility grade is a different matter: it comes from
    :func:`sperm_sorting.motion.classifier.classify_motility`, the production
    WHO-threshold rule, applied to the simulated kinematics. That half really is
    the shipping implementation, and the response says so per field
    (``motility_source``) so the two are not confused with one another.

    The overall verdict is not assembled here. A
    :class:`~sperm_sorting.schemas.track.TrackRecord` is populated and
    ``compute_eligibility`` is called, because that method is the one place the
    per-sperm rule is allowed to live -- and it also returns *why* a sperm was
    rejected, which a hand-rolled ``and`` chain would not.
    """
    params = req.resolved_params()
    crop: np.ndarray | None = None
    motion: MotionFeatures | None = None
    motility_source = "unavailable"
    casa: dict[str, float] | None = None

    if params is not None:
        try:
            sim = _simulate(params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        crop = sim["image"]
        casa = sim["casa_corrected"]
        motion = _motion_features_from_casa(
            casa,
            n_points=params.n_points,
            dt_s=sim["dt_s"],
            um_per_px=params.track_um_per_px,
            flow_px_s=sim["flow"],
        )
        grade, reason = classify_motility(motion, app.state.config.motion.thresholds)
        motion.motility_class = grade
        motion.motility_reason = reason
        motility_source = "casa_rule"
    elif req.image is not None:
        crop = _decode_png(req.image)

    if crop is None or crop.size == 0:
        raise HTTPException(
            status_code=422, detail="no image could be obtained from this request"
        )

    engine: BaseMorphologyEngine = app.state.engine
    with app.state.engine_lock:
        probabilities = engine.infer_batch([crop])[0]

    thresholds = getattr(engine, "thresholds", {})
    results = {
        name: AspectResult(
            name=name,
            p_normal=float(probabilities[name]),
            threshold=float(thresholds.get(name, 0.5)),
        )
        for name in MORPHOLOGY_ASPECTS
    }
    morphology = MorphologyResult(
        track_id=0,
        status=MorphologyStatus.COMPLETE,
        head=results["head"],
        acrosome=results["acrosome"],
        vacuole=results["vacuole"],
        tail=results["tail"],
        model_id=engine.model_id,
        weights_provenance=engine.weights_provenance,
    )

    if motion is None:
        # An image on its own cannot be graded for motility. Saying
        # "undetermined" with the reason attached is the honest answer; guessing
        # "progressive" so the table looks complete would be a lie the decision
        # rule would then act on.
        motion = MotionFeatures(
            n_points=0,
            n_observed_points=0,
            duration_s=0.0,
            mean_frame_interval_s=0.0,
            timestamp_source=TimestampSource.SYNTHETIC,
            flow_correction_mode=FlowCorrectionMode.DISABLED,
            motility_class=MotilityClass.UNDETERMINED,
            motility_reason=(
                "undetermined: a single image carries no trajectory, so no "
                "velocity and therefore no WHO motility grade can be computed. "
                "Send {seed, params} instead of {image} to get a graded track."
            ),
        )

    track = TrackRecord(
        track_id=0,
        track_quality_pass=True,
        motion=motion,
        morphology=morphology,
        evaluation_complete=True,
    )
    eligible = track.compute_eligibility()
    # ``labels()`` types every entry as optional because an incomplete result
    # has missing aspects. This result is COMPLETE by construction, so a missing
    # entry would be a bug in the schema rather than a case to paper over --
    # hence the explicit failure instead of a default.
    predicted_labels: dict[str, int] = {}
    for name in MORPHOLOGY_ASPECTS:
        label = morphology.labels()[name]
        if label is None:  # pragma: no cover - unreachable for a COMPLETE result
            raise HTTPException(
                status_code=500,
                detail=f"morphology result is COMPLETE but aspect '{name}' is missing",
            )
        predicted_labels[name] = int(label)

    return {
        "pred_label": LABEL_NORMAL if eligible else LABEL_ABNORMAL,
        "pred_label_name": OVERALL_LABEL_NAMES[
            LABEL_NORMAL if eligible else LABEL_ABNORMAL
        ],
        "pred_aspects": dict(predicted_labels),
        "pred_motility": str(motion.motility_class),
        "pred_motility_reason": motion.motility_reason,
        "pred_motility_progressive": motion.motility_class.is_progressive,
        "motility_source": motility_source,
        "motility_rule": (
            "sperm_sorting.motion.classifier.classify_motility, profile "
            f"{app.state.config.motion.thresholds.profile_version}"
            if motility_source == "casa_rule"
            else "not applicable"
        ),
        "probs": {
            name: float(results[name].p_normal) for name in MORPHOLOGY_ASPECTS
        },
        "probs_meaning": "P(normal) per aspect; P(abnormal) = 1 - P(normal)",
        "aspect_detail": {
            name: {
                **results[name].to_json_dict(),
                "p_abnormal": float(flip_polarity(results[name].p_normal)),
            }
            for name in MORPHOLOGY_ASPECTS
        },
        "all_four_normal": morphology.all_four_normal,
        "first_abnormal_aspect": morphology.first_abnormal_aspect(),
        "ai_eligible": eligible,
        "ineligibility_reason": str(track.ineligibility_reason),
        "casa": casa,
        "input_kind": "seed" if params is not None else "image",
        "model": _model_block(app),
    }


@app.post("/decide")
def decide_endpoint(req: DecideRequest) -> dict[str, Any]:
    """Apply the real decision rule to one (eligible, trackable) pair.

    This handler contains no arithmetic on the counts beyond passing them to
    :func:`sperm_sorting.decision.engine.decide`. The extra fields it adds are
    presentation only -- what the field command physically does, and whether the
    ratio landed exactly on the threshold -- and the second of those is answered
    with the same exact-rational comparison the engine itself uses, never with
    floating point.
    """
    cfg: AppConfig = app.state.config
    threshold = (
        cfg.decision.threshold if req.threshold is None else float(req.threshold)
    )
    minimum = (
        cfg.decision.minimum_trackable_sperm
        if req.minimum_trackable is None
        else int(req.minimum_trackable)
    )
    try:
        decision = decide(
            req.ai_eligible_count,
            req.trackable_count,
            threshold=threshold,
            minimum_trackable=minimum,
        )
    except ValueError as exc:
        # An eligible count above the trackable count is a caller error -- the
        # numerator must be a subset of the denominator -- so it must not
        # present as a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    body = decision.to_json_dict()
    at_threshold = req.trackable_count > 0 and Fraction(
        req.ai_eligible_count, req.trackable_count
    ) == Fraction(str(threshold))
    body.update(
        {
            "status_upper": str(decision.status).upper(),
            "field_command_meaning": FIELD_COMMAND_MEANING[
                str(decision.field_command)
            ],
            "is_rejection": decision.field_command is FieldCommandKind.FIELD_ON,
            "exactly_at_threshold": at_threshold,
            "percent": (
                100.0 * decision.ratio if req.trackable_count > 0 else None
            ),
            "boundary_rule": (
                "ACCEPT requires ratio > threshold, strictly. A ratio of exactly "
                f"{threshold:.0%} is a REJECT and energises the field."
            ),
            "minimum_rule": (
                f"fewer than {minimum} trackable sperm gives INDETERMINATE with "
                "FIELD_OFF: no ratio is trusted, and the safe state is to let the "
                "segment pass rather than to divert it on a guess."
            ),
            "comparison_is_exact": True,
            "engine": "sperm_sorting.decision.engine.decide",
        }
    )
    return body


@app.get("/config")
def config() -> dict[str, Any]:
    """Resolved configuration summary plus the shot-throughput feasibility budget.

    The feasibility report is passed through as the module produced it,
    ``warnings`` included. Those warnings are the honest part: on the reference
    build a whole spermatozoon does not fit across the field of view, and a demo
    that quietly dropped that line would be advertising an instrument that does
    not exist.
    """
    cfg: AppConfig = app.state.config
    report: FeasibilityReport = app.state.feasibility
    return {
        "summary": cfg.summary(),
        "feasibility": _feasibility_block(report),
        "decision": {
            "threshold": cfg.decision.threshold,
            "minimum_trackable": cfg.decision.minimum_trackable_sperm,
            "target_trackable": cfg.shots.target_trackable_sperm,
            "maximum_trackable": cfg.shots.maximum_trackable_sperm,
            "maximum_shot_duration_s": cfg.shots.maximum_shot_duration_seconds,
            "spec_defaults": {
                "threshold": ACCEPT_RATIO_THRESHOLD,
                "minimum_trackable": MINIMUM_TRACKABLE_SPERM,
                "target_trackable": TARGET_TRACKABLE_SPERM,
                "maximum_trackable": MAXIMUM_TRACKABLE_SPERM,
                "maximum_shot_duration_s": MAXIMUM_SHOT_DURATION_S,
            },
        },
        "motility_thresholds": cfg.motion.thresholds.model_dump(mode="json"),
        "morphology": _model_block(app),
        "optically_calibrated": cfg.calibration.optical.calibrated,
        "disclaimer": DISCLAIMER,
    }


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the demo with uvicorn.

    Provided so ``python -m web.app`` works for someone who does not want to
    remember the uvicorn incantation; ``uvicorn web.app:app --reload`` remains
    the development path.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()


__all__: Final[list[str]] = [
    "DISCLAIMER",
    "MANDATED_DECISION_CASES",
    "UNTRAINED_WARNING",
    "app",
    "main",
]
