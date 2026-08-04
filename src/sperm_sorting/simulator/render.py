"""Turn a :class:`~.params.HealthState` into pixels.

Why the renderer is parameter-driven rather than texture-driven
---------------------------------------------------------------
Every visible feature is computed from the state's continuous knobs, and those
knobs were themselves forced out of their normal band by the state's binary
labels (see :func:`~.params.sample_health_state`). The chain
``label -> knob -> geometry -> pixels`` is unbroken, so a model trained on this
data is learning the thing the label names. Pasting labels onto sampled
textures would break the chain and produce a dataset on which nothing is
learnable.

Imaging convention
------------------
Brightfield microscopy: the specimen absorbs light, so **objects are darker
than the background**. This is stated once, here, and implemented once, in
:func:`composite_ink`, with :attr:`RenderConfig.dark_objects` as the switch --
some detectors train better on the inverted convention, and
``PreprocessConfig.invert`` exists downstream for exactly that reason. Getting
this backwards silently would poison every model trained on the output, so it
is a named, tested parameter rather than an implicit sign.

Anti-aliasing
-------------
Shapes are drawn on a grid ``supersample`` times finer than the output and
reduced with an exact box filter. A sperm head is only ~20 px long in a crop
and ~13 px in a full scene, so aliasing on the head outline would be a
significant fraction of the very morphology the model must read.

Output sizes
------------
Both 64x64 and 128x128 are supported so results are directly comparable with
MHSMA, which ships those two crop variants.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .params import HealthState

# --------------------------------------------------------------------------
# Optical scale
# --------------------------------------------------------------------------

#: Field of view of a morphology crop, in micrometres, *independent of the
#: output size*. 64x64 and 128x128 therefore show the same content at two
#: resolutions, exactly as MHSMA's two variants do; scaling the field with the
#: pixel count instead would make the two variants show different amounts of
#: flagellum and stop being comparable.
#:
#: 25.6 um puts the head at ~22 px in a 128 px crop -- enough to read the
#: acrosomal boundary and a vacuole -- while leaving ~10 um of frame behind the
#: midpiece for the flagellum. A full 45 um tail therefore runs off the edge,
#: which is not a defect but the discriminating signal: a normal tail exits as a
#: straight line, a coiled tail curls up beside the head, and a short or absent
#: tail (capped at :data:`~.params.ABNORMAL_SHORT_TAIL_MAX_UM`, chosen against
#: this number) visibly stops inside the crop.
CROP_FIELD_UM: Final[float] = 25.6

#: Micrometres per pixel of a 128 px crop; the 64 px variant is twice this.
DEFAULT_CROP_UM_PER_PX: Final[float] = CROP_FIELD_UM / 128.0

#: Where the head centre sits, as a fraction of the crop width. The head points
#: towards +x and the flagellum trails towards -x, so the head goes on the
#: *right* and the tail gets the rest of the frame.
CROP_HEAD_X_FRACTION: Final[float] = 0.82

#: Micrometres per pixel for a full 1920x1200 scene: a 672 x 420 um field, a
#: plausible view for a 20x objective, in which the configured density of ~28
#: sperm is a realistic concentration rather than a crowd.
DEFAULT_SCENE_UM_PER_PX: Final[float] = 0.35

#: Midpiece geometry, WHO: ~7-8 um long, ~1 um wide, attached axially.
MIDPIECE_LENGTH_UM: Final[float] = 7.5
MIDPIECE_WIDTH_UM: Final[float] = 1.0
#: Principal piece is thinner than the midpiece.
TAIL_WIDTH_UM: Final[float] = 0.55

#: Supported output edges. Enforced so a typo cannot silently produce a size
#: no downstream consumer expects.
SUPPORTED_SIZES: Final[tuple[int, ...]] = (64, 128)


@dataclass(slots=True)
class RenderConfig:
    """Imaging parameters. Separate from :class:`HealthState` on purpose.

    A state describes the *cell*; this describes the *microscope*. Keeping them
    apart means the same cell can be re-imaged under different degradation to
    test the quality gate, and it keeps nuisance parameters out of the ground
    truth where they would look like labels.
    """

    #: Background grey level, 0-255. Matches ``SyntheticSourceConfig``'s default.
    background_level: int = 200
    #: Grey level of a fully opaque object. Brightfield: below the background.
    object_level: int = 35
    #: Peak absorbance of the head. 1.0 would render the head fully opaque.
    head_ink: float = 0.85
    #: Absorbance of the midpiece; denser than the tail, lighter than the head.
    midpiece_ink: float = 0.62
    #: Absorbance of the flagellum.
    tail_ink: float = 0.42
    #: Added to the head's absorbance inside the acrosomal cap. Negative means
    #: the cap is *lighter* than the post-acrosomal region, which is the usual
    #: unstained brightfield appearance; positive inverts it for stained-like
    #: contrast. Sign is configurable because both are seen in real data.
    acrosome_ink_delta: float = -0.30
    #: Absorbance *inside* a vacuole. Absolute rather than a delta on the head:
    #: a vacuole is a fluid-filled void, so how much light it removes does not
    #: depend on whether it happens to lie under the acrosomal cap. Expressing
    #: it as a delta made a vacuole under the cap visibly fainter than the same
    #: vacuole beside it -- a rendering artefact the label knows nothing about.
    vacuole_ink: float = 0.12
    #: Baseline optical blur in output pixels, before any per-cell defocus.
    blur_px: float = 0.6
    #: Sensor noise standard deviation in grey levels.
    noise_sigma: float = 4.0
    #: Peak-to-peak illumination gradient as a fraction of the background.
    illumination_amplitude: float = 0.10
    #: Linear supersampling factor used for anti-aliasing.
    supersample: int = 4
    #: False renders bright objects on a dark field (the inverted convention).
    dark_objects: bool = True

    def __post_init__(self) -> None:
        if self.supersample < 1:
            raise ValueError(f"supersample must be >= 1, got {self.supersample}")
        if not 0 <= self.background_level <= 255:
            raise ValueError(f"background_level must be 0-255, got {self.background_level}")
        if not 0 <= self.object_level <= 255:
            raise ValueError(f"object_level must be 0-255, got {self.object_level}")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CellGeometry:
    """Pixel-space dimensions of one cell at a given optical scale."""

    head_semi_major_px: float
    head_semi_minor_px: float
    midpiece_length_px: float
    midpiece_width_px: float
    tail_length_px: float
    tail_width_px: float
    vacuole_radius_px: float

    @property
    def total_length_px(self) -> float:
        """Head tip to tail tip along the cell axis, ignoring tail curvature."""
        return (
            2.0 * self.head_semi_major_px + self.midpiece_length_px + self.tail_length_px
        )


def cell_geometry(state: HealthState, um_per_px: float) -> CellGeometry:
    """Convert a state's micrometre knobs into pixel dimensions.

    ``head_scale`` multiplies both head axes, so a macrocephalic head is larger
    in both directions while the axis ratio independently controls how tapered
    or round it is. That separation is what lets ``head=1`` be caused by either
    defect (or both) and still always be visible.
    """
    if um_per_px <= 0.0:
        raise ValueError(f"um_per_px must be positive, got {um_per_px}")
    length_um = state.head_length_um * state.head_scale
    width_um = length_um / max(state.head_axis_ratio, 1e-6)
    head_a = 0.5 * length_um / um_per_px
    head_b = 0.5 * width_um / um_per_px
    return CellGeometry(
        head_semi_major_px=head_a,
        head_semi_minor_px=head_b,
        midpiece_length_px=MIDPIECE_LENGTH_UM * state.head_scale / um_per_px,
        midpiece_width_px=max(MIDPIECE_WIDTH_UM / um_per_px, 1.0),
        tail_length_px=state.tail_length_um / um_per_px,
        tail_width_px=max(TAIL_WIDTH_UM / um_per_px, 0.8),
        vacuole_radius_px=0.5 * state.vacuole_size * length_um / um_per_px,
    )


def _acrosome_cut(area_fraction: float) -> float:
    """Normalised long-axis coordinate whose anterior side holds ``area_fraction``.

    For a unit disc the area with ``x >= u`` is ``arccos(u) - u*sqrt(1-u^2)``.
    An affine scaling to an ellipse preserves area *ratios*, so the same ``u``
    works for any head shape -- which is why the acrosome fraction stays a true
    area fraction (the WHO criterion) rather than a length fraction, even as
    the axis ratio changes.
    """
    target = float(np.clip(area_fraction, 0.0, 1.0)) * math.pi
    lo, hi = -1.0, 1.0  # area(x>=u) decreases as u increases
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        area = math.acos(mid) - mid * math.sqrt(max(1.0 - mid * mid, 0.0))
        if area > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ellipse_points(
    cx: float, cy: float, a: float, b: float, angle: float, n: int = 96,
    t0: float = -math.pi, t1: float = math.pi,
) -> list[tuple[float, float]]:
    """Sampled boundary of a rotated ellipse arc, as polygon vertices."""
    t = np.linspace(t0, t1, n)
    x = a * np.cos(t)
    y = b * np.sin(t)
    ca, sa = math.cos(angle), math.sin(angle)
    return [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in zip(x, y, strict=True)]


def tail_polyline(
    state: HealthState,
    geom: CellGeometry,
    origin: tuple[float, float],
    angle: float,
    *,
    bend_sign: float = 1.0,
    wave_phase: float = 0.0,
    n_points: int = 160,
) -> list[tuple[float, float]]:
    """Sample the flagellum as a smooth curve leaving ``origin``.

    The heading turns at a constant rate so that the total bend over the whole
    flagellum equals ``state.tail_curvature`` radians: a normal tail
    (<= 0.55 rad) reads as gently curved, and a coiled one (several radians)
    wraps back on itself, which is precisely the WHO "coiled tail" defect. A
    small superimposed wave keeps a straight tail from looking synthetic
    without changing its total bend.
    """
    length = geom.tail_length_px
    if length <= 1.0 or n_points < 2:
        return []
    s = np.linspace(0.0, length, n_points)
    frac = s / length
    heading = angle + math.pi + bend_sign * float(state.tail_curvature) * frac
    heading = heading + 0.09 * np.sin(2.0 * math.pi * 1.5 * frac + wave_phase)
    # Integrate the heading to get the curve; cumulative trapezoid keeps the
    # arc length equal to `length` regardless of how sharply it turns.
    ds = float(s[1] - s[0])
    dx = np.cos(heading) * ds
    dy = np.sin(heading) * ds
    xs = origin[0] + np.cumsum(dx) - dx[0]
    ys = origin[1] + np.cumsum(dy) - dy[0]
    return [(float(px), float(py)) for px, py in zip(xs, ys, strict=True)]


# --------------------------------------------------------------------------
# Ink rendering
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CellPose:
    """Per-cell rendering choices that must stay fixed over a cell's lifetime.

    Which way the tail curls, where the vacuole sits and the phase of the tail's
    wave are properties of the individual cell, not of the frame. Re-drawing
    them each frame would make a tracked sperm visibly flicker and would defeat
    any appearance-based re-identification. A scene agent samples one of these
    at spawn and reuses it for every frame it appears in.
    """

    bend_sign: float = 1.0
    wave_phase: float = 0.0
    vacuole_u: float = 0.15
    vacuole_v: float = 0.10

    @classmethod
    def sample(cls, rng: np.random.Generator) -> CellPose:
        return cls(
            bend_sign=1.0 if rng.random() < 0.5 else -1.0,
            wave_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            vacuole_u=float(rng.uniform(-0.45, 0.45)),
            vacuole_v=float(rng.uniform(-0.35, 0.35)),
        )


def _draw_mask(
    size_px: tuple[int, int],
    supersample: int,
    draw_fn: Callable[[ImageDraw.ImageDraw, int], None],
) -> np.ndarray:
    """Rasterise at ``supersample`` resolution and box-reduce to output size.

    Returns ``float32`` in ``[0, 1]``. The box reduction is an exact area
    average, so a boundary pixel receives the fraction of its area the shape
    actually covers -- true anti-aliasing rather than a post-hoc blur.
    """
    h, w = size_px
    img = Image.new("L", (max(w, 1) * supersample, max(h, 1) * supersample), 0)
    draw_fn(ImageDraw.Draw(img), supersample)
    if supersample > 1:
        img = img.resize((max(w, 1), max(h, 1)), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


def cell_ink(
    state: HealthState,
    size_px: tuple[int, int],
    center_px: tuple[float, float],
    angle: float,
    um_per_px: float,
    cfg: RenderConfig,
    rng: np.random.Generator | None = None,
    pose: CellPose | None = None,
) -> np.ndarray:
    """Absorbance map of one cell, ``float32`` in ``[0, 1]``, shape ``size_px``.

    Absorbance, not intensity: keeping the cell as "how much light it removes"
    means several cells and a background can be composited with one rule
    (:func:`composite_ink`) instead of every renderer knowing about
    illumination.

    All five components are drawn into a *single* supersampled buffer whose
    grey value already is the absorbance, back-to-front (tail, midpiece, head,
    acrosomal cap, vacuole), then box-reduced once. Rasterising each component
    separately would be five times the work -- which at 28 cells per frame is
    the difference between a usable scene generator and an unusable one -- and
    would also blend components with the wrong weights at their shared edges.
    Because the buffer holds absorbance rather than a label, the box reduction
    averages physically meaningful quantities and stays correct at every
    boundary.

    ``pose`` fixes the cell's individual appearance; ``rng`` samples one when
    no pose is supplied. Both ``None`` gives the state's canonical, repeatable
    appearance, which is what training crops and documentation figures use.
    """
    h, w = size_px
    geom = cell_geometry(state, um_per_px)
    cx, cy = center_px
    ca, sa = math.cos(angle), math.sin(angle)

    if pose is None:
        pose = CellPose() if rng is None else CellPose.sample(rng)

    contrast = float(state.contrast)

    def _level(ink: float) -> int:
        """Absorbance -> 0-255 buffer value, clamped to a drawable range."""
        return round(float(np.clip(ink * contrast, 0.0, 1.0)) * 255.0)

    head_level = _level(cfg.head_ink)
    acrosome_level = _level(cfg.head_ink + cfg.acrosome_ink_delta)
    vacuole_level = _level(cfg.vacuole_ink)
    midpiece_level = _level(cfg.midpiece_ink)
    tail_level = _level(cfg.tail_ink)

    # Acrosomal cap: the anterior part of the head holding `acrosome_frac` of
    # its *area*, drawn as an ellipse arc closed by its chord.
    u = _acrosome_cut(state.acrosome_frac)
    t_half = math.acos(float(np.clip(u, -1.0, 1.0)))

    base_x = cx - ca * geom.head_semi_major_px
    base_y = cy - sa * geom.head_semi_major_px
    tail_x = base_x - ca * geom.midpiece_length_px
    tail_y = base_y - sa * geom.midpiece_length_px

    pts_tail = tail_polyline(
        state,
        geom,
        (tail_x, tail_y),
        angle,
        bend_sign=pose.bend_sign,
        wave_phase=pose.wave_phase,
    )

    vac_r = geom.vacuole_radius_px
    vac_x = (
        cx
        + (pose.vacuole_u * geom.head_semi_major_px) * ca
        - (pose.vacuole_v * geom.head_semi_minor_px) * sa
    )
    vac_y = (
        cy
        + (pose.vacuole_u * geom.head_semi_major_px) * sa
        + (pose.vacuole_v * geom.head_semi_minor_px) * ca
    )

    def _draw(draw: ImageDraw.ImageDraw, k: int) -> None:
        if len(pts_tail) >= 2 and tail_level > 0:
            draw.line(
                [(px * k, py * k) for px, py in pts_tail],
                fill=tail_level,
                width=max(round(geom.tail_width_px * k), 1),
                joint="curve",
            )
        if midpiece_level > 0:
            draw.line(
                [(base_x * k, base_y * k), (tail_x * k, tail_y * k)],
                fill=midpiece_level,
                width=max(round(geom.midpiece_width_px * k), 1),
            )
        if head_level > 0 or acrosome_level > 0:
            draw.polygon(
                _ellipse_points(
                    cx * k,
                    cy * k,
                    geom.head_semi_major_px * k,
                    geom.head_semi_minor_px * k,
                    angle,
                ),
                fill=head_level,
            )
            if t_half > 1e-4:
                cap = _ellipse_points(
                    cx * k,
                    cy * k,
                    geom.head_semi_major_px * k,
                    geom.head_semi_minor_px * k,
                    angle,
                    n=96,
                    t0=-t_half,
                    t1=t_half,
                )
                if len(cap) >= 3:
                    draw.polygon(cap, fill=acrosome_level)
            if vac_r > 0.15:
                draw.ellipse(
                    [
                        (vac_x - vac_r) * k,
                        (vac_y - vac_r) * k,
                        (vac_x + vac_r) * k,
                        (vac_y + vac_r) * k,
                    ],
                    fill=vacuole_level,
                )

    return _draw_mask((h, w), cfg.supersample, _draw).astype(np.float32)


def composite_ink(
    background: np.ndarray, ink: np.ndarray, cfg: RenderConfig
) -> np.ndarray:
    """Apply an absorbance map to a background, honouring the imaging convention.

    Brightfield (``dark_objects=True``): the object removes light, so intensity
    falls from the background towards :attr:`RenderConfig.object_level`.

    Inverted (``dark_objects=False``): the photometric negative of that -- a
    dark field with bright objects. Defined as ``255 - brightfield`` rather
    than "raise the object above the background", because that is exactly what
    ``PreprocessConfig.invert`` does downstream; defining it any other way here
    would mean a model trained on inverted synthetic data saw a different
    transform from the one applied to inverted device data.

    This one function is the only place the sign of the convention appears.
    """
    ink = np.clip(ink, 0.0, 1.0)
    dark = background - ink * (background - float(cfg.object_level))
    return dark if cfg.dark_objects else 255.0 - dark


def illumination_field(
    size_px: tuple[int, int],
    cfg: RenderConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Background plane with a linear illumination gradient.

    Real Koehler illumination is never perfectly flat, and an uncorrected
    gradient is one of the things the quality gate and any intensity
    normalisation must survive. Making it a first-class part of the simulator
    means those components are tested against it rather than against an
    unrealistically uniform field.
    """
    h, w = size_px
    base = float(cfg.background_level)
    if cfg.illumination_amplitude <= 0.0:
        return np.full((h, w), base, dtype=np.float32)
    if rng is None:
        theta, amp = 0.6, cfg.illumination_amplitude
    else:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        amp = float(cfg.illumination_amplitude * rng.uniform(0.4, 1.0))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx / max(w - 1, 1) - 0.5
    yy = yy / max(h - 1, 1) - 0.5
    ramp = math.cos(theta) * xx + math.sin(theta) * yy
    return (base * (1.0 + amp * ramp)).astype(np.float32)


def _blur(image: np.ndarray, radius_px: float) -> np.ndarray:
    """Gaussian blur via Pillow, kept in float by round-tripping through uint8."""
    if radius_px <= 1e-3:
        return image
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=float(radius_px)))
    return np.asarray(img, dtype=np.float32)


def finish_image(
    field: np.ndarray,
    cfg: RenderConfig,
    rng: np.random.Generator | None,
    *,
    extra_blur_px: float = 0.0,
) -> np.ndarray:
    """Blur, add sensor noise and quantise to ``uint8``.

    Blur precedes noise deliberately: optical defocus happens in front of the
    sensor, read noise behind it. Doing it the other way round would produce
    smooth, correlated "noise" that no denoiser or quality metric would ever
    see on a real camera.
    """
    out = _blur(field, cfg.blur_px + extra_blur_px)
    if cfg.noise_sigma > 0.0 and rng is not None:
        out = out + rng.normal(0.0, cfg.noise_sigma, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------
# Public renderers
# --------------------------------------------------------------------------


def render_sperm(
    state: HealthState,
    size: tuple[int, int] = (128, 128),
    rng: np.random.Generator | None = None,
    um_per_px: float | None = None,
    cfg: RenderConfig | None = None,
    *,
    angle: float | None = None,
    strict_size: bool = True,
) -> np.ndarray:
    """Render one cell as a ``uint8`` crop of exactly ``size``.

    The head sits at :data:`CROP_HEAD_X_FRACTION` of the width and points
    towards +x, so the flagellum trails across the rest of the frame. Centring
    the head would halve the visible tail and make a short tail
    indistinguishable from a normal one that merely left the crop.

    Parameters
    ----------
    size
        ``(height, width)``. With ``strict_size`` the edge must be one of
        :data:`SUPPORTED_SIZES` (64 or 128, matching MHSMA's two variants).
    rng
        ``None`` renders the state's canonical, noise-free appearance; a
        generator adds pose variation, illumination direction and sensor noise.
    um_per_px
        ``None`` derives the scale from :data:`CROP_FIELD_UM` and the output
        size, so every supported size shows the same field of view.
    angle
        Cell orientation in radians. ``None`` means "along +x", the canonical
        pose used for training crops.
    """
    h, w = int(size[0]), int(size[1])
    if strict_size and not (h == w and h in SUPPORTED_SIZES):
        raise ValueError(
            f"size must be square and one of {SUPPORTED_SIZES} (MHSMA parity), got {size}"
        )
    cfg = cfg or RenderConfig()
    theta = 0.0 if angle is None else float(angle)
    scale = CROP_FIELD_UM / float(max(w, 1)) if um_per_px is None else float(um_per_px)

    field = illumination_field((h, w), cfg, rng)
    ink = cell_ink(
        state, (h, w), (CROP_HEAD_X_FRACTION * w, 0.5 * h), theta, scale, cfg, rng
    )
    field = composite_ink(field, ink, cfg)
    return finish_image(field, cfg, rng, extra_blur_px=float(state.defocus))


def render_sperm_on_canvas(
    canvas: np.ndarray,
    state: HealthState,
    cx: float,
    cy: float,
    angle: float,
    um_per_px: float = DEFAULT_SCENE_UM_PER_PX,
    cfg: RenderConfig | None = None,
    rng: np.random.Generator | None = None,
    pose: CellPose | None = None,
    *,
    extra_blur_px: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """Composite one cell into a larger scene, in place.

    Returns the head+midpiece bounding box ``(x1, y1, x2, y2)`` clipped to the
    canvas, or ``None`` when the cell is entirely outside it. The box covers
    the head and midpiece rather than the whole flagellum because that is what
    a detector trained on VISEM-Tracking-style annotation predicts, and because
    a box around a 45 um tail would overlap half the neighbours and make IoU
    meaningless.

    ``canvas`` must be ``float32``; the caller quantises once at the end, so
    compositing many cells does not accumulate rounding error.

    Only a tight patch around the cell's *actual* extent is rasterised -- the
    flagellum is long and one-sided, so a square patch big enough to hold it in
    any direction would be an order of magnitude more pixels than the cell
    occupies, and at 28 cells a frame that is the whole frame budget.
    """
    if canvas.dtype != np.float32:
        raise TypeError(f"canvas must be float32, got {canvas.dtype}")
    cfg = cfg or RenderConfig()
    h, w = canvas.shape[:2]
    geom = cell_geometry(state, um_per_px)
    if pose is None:
        pose = CellPose() if rng is None else CellPose.sample(rng)

    # Tight extent: head ellipse bounds plus the sampled flagellum, padded by
    # the tail half-width and any defocus radius.
    ca, sa = math.cos(angle), math.sin(angle)
    base = (cx - ca * geom.head_semi_major_px, cy - sa * geom.head_semi_major_px)
    tail_root = (
        base[0] - ca * geom.midpiece_length_px,
        base[1] - sa * geom.midpiece_length_px,
    )
    pts = tail_polyline(
        state, geom, tail_root, angle,
        bend_sign=pose.bend_sign, wave_phase=pose.wave_phase, n_points=48,
    )
    xs = [cx - geom.head_semi_major_px, cx + geom.head_semi_major_px, tail_root[0]]
    ys = [cy - geom.head_semi_major_px, cy + geom.head_semi_major_px, tail_root[1]]
    xs.extend(p[0] for p in pts)
    ys.extend(p[1] for p in pts)
    pad = max(geom.tail_width_px, geom.midpiece_width_px) + 2.0 + 3.0 * extra_blur_px
    px0 = max(math.floor(min(xs) - pad), 0)
    py0 = max(math.floor(min(ys) - pad), 0)
    px1 = min(math.ceil(max(xs) + pad), w)
    py1 = min(math.ceil(max(ys) + pad), h)
    if px1 <= px0 or py1 <= py0:
        return None

    patch_h, patch_w = py1 - py0, px1 - px0
    ink = cell_ink(
        state,
        (patch_h, patch_w),
        (cx - px0, cy - py0),
        angle,
        um_per_px,
        cfg,
        rng,
        pose,
    )
    if extra_blur_px > 0.0:
        # Defocus is applied to the cell alone, so an out-of-focus sperm sits
        # in a sharp field -- which is what the quality gate has to detect.
        blurred = _blur(ink * 255.0, extra_blur_px) / 255.0
        ink = blurred.astype(np.float32)

    region = canvas[py0:py1, px0:px1]
    canvas[py0:py1, px0:px1] = composite_ink(region, ink, cfg)

    ca, sa = math.cos(angle), math.sin(angle)
    ax = geom.head_semi_major_px
    bx = geom.head_semi_minor_px
    mid_x = cx - ca * (ax + geom.midpiece_length_px)
    mid_y = cy - sa * (ax + geom.midpiece_length_px)
    tip_x, tip_y = cx + ca * ax, cy + sa * ax
    half = max(bx, geom.midpiece_width_px * 0.5) + 1.0
    bx1 = min(tip_x, mid_x) - half
    by1 = min(tip_y, mid_y) - half
    bx2 = max(tip_x, mid_x) + half
    by2 = max(tip_y, mid_y) + half
    cbx1, cby1 = max(bx1, 0.0), max(by1, 0.0)
    cbx2, cby2 = min(bx2, float(w)), min(by2, float(h))
    if cbx2 - cbx1 < 1.0 or cby2 - cby1 < 1.0:
        return None
    return (cbx1, cby1, cbx2, cby2)


def render_debris_on_canvas(
    canvas: np.ndarray,
    kind: str,
    cx: float,
    cy: float,
    angle: float,
    size_px: float,
    elongation: float,
    ink_level: float,
    cfg: RenderConfig | None = None,
) -> None:
    """Composite one non-sperm particle, in place.

    Debris exists so that debris-induced false positives can be *measured*
    rather than assumed away, so it must be visually plausible without ever
    being sperm-shaped: blobs and streaks, never a head-plus-tail silhouette.
    ``kind`` is ``"blob"`` (round, compact) or ``"streak"`` (elongated, but
    uniform-width and untapered, with no head).
    """
    if canvas.dtype != np.float32:
        raise TypeError(f"canvas must be float32, got {canvas.dtype}")
    cfg = cfg or RenderConfig()
    h, w = canvas.shape[:2]
    a = max(size_px, 0.8)
    b = max(size_px / max(elongation, 1.0), 0.6)
    reach = a + b + 4.0
    px0, py0 = max(int(cx - reach), 0), max(int(cy - reach), 0)
    px1, py1 = min(int(cx + reach) + 1, w), min(int(cy + reach) + 1, h)
    if px1 <= px0 or py1 <= py0:
        return
    lcx, lcy = cx - px0, cy - py0

    if kind == "streak":
        def _shape(draw: ImageDraw.ImageDraw, k: int) -> None:
            dx, dy = math.cos(angle) * a, math.sin(angle) * a
            draw.line(
                [((lcx - dx) * k, (lcy - dy) * k), ((lcx + dx) * k, (lcy + dy) * k)],
                fill=255,
                width=max(round(b * k), 1),
            )
    else:
        def _shape(draw: ImageDraw.ImageDraw, k: int) -> None:
            draw.polygon(_ellipse_points(lcx * k, lcy * k, a * k, b * k, angle, n=48), fill=255)

    mask = _draw_mask((py1 - py0, px1 - px0), cfg.supersample, _shape)
    region = canvas[py0:py1, px0:px1]
    canvas[py0:py1, px0:px1] = composite_ink(region, mask * float(ink_level), cfg)


if __name__ == "__main__":  # pragma: no cover - runnable self-check
    from .params import abnormal_state, normal_state, sample_health_state

    cfg_ = RenderConfig()

    # -- contract: exact size and dtype, both MHSMA variants ---------------
    for edge in SUPPORTED_SIZES:
        img = render_sperm(normal_state(), size=(edge, edge))
        assert img.shape == (edge, edge), img.shape
        assert img.dtype == np.uint8, img.dtype
    try:
        render_sperm(normal_state(), size=(100, 100))
    except ValueError:
        pass
    else:
        raise AssertionError("strict_size must reject an unsupported edge")

    # -- determinism -------------------------------------------------------
    st = sample_health_state(np.random.default_rng(3))
    a1 = render_sperm(st, rng=np.random.default_rng(4))
    a2 = render_sperm(st, rng=np.random.default_rng(4))
    assert np.array_equal(a1, a2), "same seed must give a byte-identical crop"
    a3 = render_sperm(st, rng=np.random.default_rng(5))
    assert not np.array_equal(a1, a3), "different seeds must differ"
    assert np.array_equal(render_sperm(st), render_sperm(st)), "rng=None must be canonical"

    # -- brightfield convention -------------------------------------------
    dark = render_sperm(normal_state())
    inv_cfg = RenderConfig(dark_objects=False)
    bright = render_sperm(normal_state(), cfg=inv_cfg)
    bg = cfg_.background_level
    assert float(dark.min()) < bg - 40, f"objects must be darker than {bg}, min {dark.min()}"
    assert float(bright.max()) > 255 - bg + 40, "inverted convention must brighten objects"
    assert float(bright.mean()) < float(dark.mean()), "inverted field must be darker overall"
    assert abs(float(dark.astype(np.int32).mean()) + float(bright.astype(np.int32).mean()) - 255.0) < 3.0

    # -- normal vs abnormal are visibly different --------------------------
    def _diff(a_state: HealthState, b_state: HealthState) -> tuple[float, float, float]:
        """Mean |dI| over the whole crop, over cell pixels, and its 99th pct.

        The whole-crop mean is reported because it is the obvious metric, but
        it is dominated by empty background: a cell covers ~4% of a crop, so
        even a total change of the head moves it by only a few grey levels.
        The cell-region mean says whether a feature moved a meaningful part of
        the cell, and the 99th percentile says how strong the change is where
        it happens -- a head-only defect necessarily dilutes across the tail
        pixels in the region mean, so both are needed to judge visibility.
        """
        ia = render_sperm(a_state).astype(np.int32)
        ib = render_sperm(b_state).astype(np.int32)
        diff = np.abs(ia - ib)
        bg = cfg_.background_level
        mask = (ia < bg - 15) | (ib < bg - 15)
        region = float(diff[mask].mean()) if mask.any() else 0.0
        # Percentile taken *within* the cell mask: over the whole crop the 99th
        # percentile still lands in background, because a head-only change
        # touches fewer than 1% of the pixels.
        p99 = float(np.percentile(diff[mask], 99.0)) if mask.any() else 0.0
        return float(diff.mean()), region, p99

    l1, l1_cell, l1_p99 = _diff(normal_state(), abnormal_state())
    assert l1_cell > 25.0, f"normal vs abnormal cell-region |dI| = {l1_cell:.2f} too small"
    assert l1 > 2.0, f"normal vs abnormal whole-crop |dI| = {l1:.2f} too small"
    assert l1_p99 > 60.0, f"normal vs abnormal p99 |dI| = {l1_p99:.2f} too small"

    # -- each aspect independently changes the image -----------------------
    base = normal_state()
    per_aspect: dict[str, tuple[float, float, float]] = {}
    for name, field_name, value in (
        ("head", "head_axis_ratio", 1.02),
        ("acrosome", "acrosome_frac", 0.95),
        ("vacuole", "vacuole_size", 0.34),
        ("tail", "tail_curvature", 4.0),
    ):
        variant = HealthState(**{**base.to_json_dict(), "motility": base.motility})
        setattr(variant, field_name, value)
        whole, region, p99 = _diff(base, variant)
        per_aspect[name] = (whole, region, p99)
        assert region > 4.0, (
            f"{name} defect barely changes the cell (region |dI| = {region:.2f})"
        )
        assert p99 > 25.0, (
            f"{name} defect is too faint where it acts (p99 |dI| = {p99:.2f})"
        )

    # -- acrosome fraction is a true *area* fraction -----------------------
    for want in (0.2, 0.4, 0.55, 0.7, 0.9):
        u = _acrosome_cut(want)
        got = (math.acos(u) - u * math.sqrt(max(1 - u * u, 0.0))) / math.pi
        assert abs(got - want) < 1e-6, (want, got)
    probe = normal_state()
    areas = []
    for frac in (0.30, 0.55, 0.80):
        probe.acrosome_frac = frac
        geom_ = cell_geometry(probe, DEFAULT_CROP_UM_PER_PX)
        ink_full = cell_ink(probe, (128, 128), (CROP_HEAD_X_FRACTION * 128.0, 64.0), 0.0, DEFAULT_CROP_UM_PER_PX,
                            RenderConfig(tail_ink=0.0, midpiece_ink=0.0), None)
        areas.append(float(ink_full.sum()))
        assert geom_.head_semi_major_px > 5.0
    assert areas[0] > areas[1] > areas[2], (
        "a larger lighter acrosomal cap must reduce total head absorbance"
    )

    # -- vacuole only appears when it is large enough to resolve -----------
    v0, v1 = normal_state(), normal_state()
    v1.vacuole_size = 0.30
    _, v_region, v_p99 = _diff(v0, v1)
    assert v_region > 4.0 and v_p99 > 40.0, (v_region, v_p99)
    v_sub = normal_state()
    v_sub.vacuole_size = 0.02  # within the normal band: must stay unresolvable
    _, sub_region, _ = _diff(v0, v_sub)
    assert sub_region < 1.0, (
        f"a sub-resolution vacuole must not be visible (region |dI| = {sub_region:.2f})"
    )

    # -- tail length drives how far the flagellum reaches ------------------
    short, long_ = normal_state(), normal_state()
    short.tail_length_um, long_.tail_length_um = 8.0, 48.0
    tail_only = RenderConfig(head_ink=0.0, midpiece_ink=0.0, acrosome_ink_delta=0.0)
    centre = (CROP_HEAD_X_FRACTION * 128.0, 64.0)
    ink_s = cell_ink(short, (128, 128), centre, 0.0, DEFAULT_CROP_UM_PER_PX, tail_only, None)
    ink_l = cell_ink(long_, (128, 128), centre, 0.0, DEFAULT_CROP_UM_PER_PX, tail_only, None)
    assert ink_l.sum() > ink_s.sum() * 1.15, "a long tail must deposit more ink than a stub"
    # The decisive cue is not total ink but *where the flagellum ends*: a short
    # tail must stop inside the crop while a normal one runs off the edge.
    # Only ~11 um of flagellum fits behind the midpiece, which is exactly why
    # ABNORMAL_SHORT_TAIL_MAX_UM is 8 um rather than something more clinical.
    reach_s = int(np.argmax(ink_s.sum(axis=0) > 0.05))
    reach_l = int(np.argmax(ink_l.sum(axis=0) > 0.05))
    assert reach_l == 0, "a 48 um flagellum must reach the edge of the crop"
    assert reach_s > 8, f"an 8 um flagellum must stop inside the crop, stopped at x={reach_s}"

    # -- canvas compositing ------------------------------------------------
    canvas = np.full((240, 320), 200.0, dtype=np.float32)
    box = render_sperm_on_canvas(canvas, normal_state(), 160.0, 120.0, 0.4)
    assert box is not None
    bx1, by1, bx2, by2 = box
    assert 0 <= bx1 < bx2 <= 320 and 0 <= by1 < by2 <= 240, box
    assert float(canvas.min()) < 199.0, "compositing must darken the canvas"
    assert render_sperm_on_canvas(canvas, normal_state(), -900.0, -900.0, 0.0) is None
    try:
        render_sperm_on_canvas(np.zeros((10, 10), dtype=np.uint8), normal_state(), 5, 5, 0)
    except TypeError:
        pass
    else:
        raise AssertionError("canvas dtype must be enforced")

    # -- the box actually contains the head --------------------------------
    canvas2 = np.full((240, 320), 200.0, dtype=np.float32)
    box2 = render_sperm_on_canvas(canvas2, normal_state(), 160.0, 120.0, 0.0)
    assert box2 is not None
    ix1, iy1, ix2, iy2 = (int(v) for v in box2)
    inside = canvas2[iy1:iy2, ix1:ix2]
    assert float(inside.min()) < 150.0, "the ground-truth box must contain dark head pixels"

    # -- debris is never sperm-shaped and still visible ---------------------
    canvas3 = np.full((120, 120), 200.0, dtype=np.float32)
    render_debris_on_canvas(canvas3, "blob", 60.0, 60.0, 0.0, 5.0, 1.2, 0.5)
    render_debris_on_canvas(canvas3, "streak", 30.0, 30.0, 0.8, 9.0, 5.0, 0.4)
    assert float(canvas3.min()) < 180.0

    print("render.py self-check OK")
    print(
        f"  normal vs abnormal mean |dI|: {l1:.2f} whole crop, "
        f"{l1_cell:.2f} over cell pixels, p99 {l1_p99:.2f} (grey levels)"
    )
    print("  per-aspect |dI|  (whole crop / cell pixels / p99):")
    for k, (whole, region, p99) in per_aspect.items():
        print(f"    {k:<9} {whole:6.2f} / {region:6.2f} / {p99:6.2f}")
