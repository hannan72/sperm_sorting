"""Trajectory synthesis and a self-contained CASA reference implementation.

Two jobs, deliberately in one module
------------------------------------
1. :func:`simulate_trajectory` turns a :class:`~.params.HealthState` into a
   pixel-space track. Because the state is the ground truth, the resulting
   track is labelled for free.
2. :func:`casa_features` measures VCL, VSL, VAP, LIN, STR, WOB, ALH and BCF
   from a track.

Why a *second* CASA implementation
----------------------------------
:mod:`sperm_sorting.motion` carries the production estimator, which works on
noisy tracker output, handles interpolated points, flow correction, missing
calibration and the ALH/BCF sampling-rate refusals. The one here is
intentionally independent and deliberately naive: it assumes a clean,
gap-free, evenly-sampled, flow-free track and computes the textbook
definitions directly. Having two implementations that were written from the
definitions rather than from each other is the only way to catch a shared
misunderstanding -- a bug reproduced identically in both the labeller and the
estimator would be invisible. Cross-checking the two on synthetic tracks with
known parameters is the intended test, so this file must **never** import from
:mod:`sperm_sorting.motion`.

Model
-----
A track is built from three superposed components:

``average path``
    A point advancing at ``v_fwd`` along a wandering heading -- mean-reverting
    (Ornstein-Uhlenbeck) for a progressive cell, which has a preferred
    direction, and a free random walk for a non-progressive one, which does
    not. Heading wander is what separates the two: it leaves path length (VCL)
    untouched while destroying net displacement (VSL), i.e. it lowers LIN
    through the numerator.
``flagellar beat``
    A sinusoid perpendicular to the current heading, half-amplitude ``A`` and
    frequency ``f``. Being periodic and bounded it leaves VSL untouched while
    adding path length, i.e. it lowers LIN through the denominator.
``bulk flow``
    A constant pixel-space velocity added to everything, sperm and debris
    alike. This is the term the production pipeline must estimate and remove;
    the simulator reports the true value so removal can be scored.

The two lowering mechanisms are complementary -- one works on the numerator of
LIN, the other on the denominator -- which is what makes it possible to hit a
target (VCL, LIN) pair *exactly* rather than on average. See
:func:`beat_and_forward_split` for the split and
:func:`_solve_linearity_scale` for the per-track correction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Final

import numpy as np

from ..constants import EPS
from ..schemas.enums import MotilityClass
from .params import HealthState

# --------------------------------------------------------------------------
# Algorithm constants. These are part of the definition of the numbers this
# module produces, so they are named, documented and logged -- CASA outputs are
# meaningless without the algorithm that produced them.
# --------------------------------------------------------------------------

#: Window, in frames, of the moving average that defines the *average path*
#: and therefore VAP, STR, WOB, ALH and BCF.
#:
#: The window must span roughly one flagellar beat cycle, or the "average
#: path" still follows the beat and ALH collapses towards zero. The classic
#: CASA default of 5 frames encodes that rule at the 50-60 Hz those systems
#: run at (5 / 60 Hz = 83 ms, about one 12 Hz beat). At this project's 160 FPS,
#: 5 frames is 31 ms -- less than half a beat -- and ALH comes out roughly 4x
#: too small. 11 frames is 69 ms, one cycle of a 14.5 Hz beat, so that is the
#: default here.
#:
#: Note for reviewers: ``MotionConfig.vap_window`` still defaults to 5. That is
#: correct for a 60 Hz recording and wrong for a 160 FPS one; it is exactly the
#: kind of shared-assumption bug an independent second implementation exists to
#: surface, and it should be revisited when the production estimator lands.
VAP_WINDOW: Final[int] = 11


def vap_window_for_fps(fps: float, beat_hz: float = 14.5) -> int:
    """Odd frame count spanning one beat cycle at ``fps``.

    Exposed so that a caller running at some other frame rate gets a defensible
    window instead of silently inheriting a constant tuned for 160 FPS.
    """
    if fps <= 0.0 or beat_hz <= 0.0:
        raise ValueError(f"fps and beat_hz must be positive, got {fps}, {beat_hz}")
    w = round(fps / beat_hz)
    return max(3, w + 1 - (w % 2))

#: Minimum points before the textbook definitions mean anything.
MIN_POINTS: Final[int] = 3

#: Canonical feature order. Model input order, ``feats.npy`` column order and
#: report order all follow this tuple.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "vcl",
    "vsl",
    "vap",
    "lin",
    "str",
    "wob",
    "alh",
    "bcf",
)

#: Fixed divisors used by :func:`normalize_features`. Chosen once, from the
#: physiological range, and frozen: a normalisation derived from a training
#: set's own statistics silently changes whenever the training set changes,
#: and then training and serving disagree. Velocities are um/s, ALH um, BCF Hz;
#: the three ratios are already in [0, 1] and are passed through with a divisor
#: of 1.0 so the vector stays interpretable.
FEATURE_SCALES: Final[dict[str, float]] = {
    "vcl": 150.0,
    "vsl": 150.0,
    "vap": 150.0,
    "lin": 1.0,
    "str": 1.0,
    "wob": 1.0,
    "alh": 10.0,
    "bcf": 30.0,
}

#: Normalised features are clipped here. A clip rather than a rescale so that
#: one freak track cannot shift the whole feature distribution, and so the
#: network never sees an unbounded input.
FEATURE_CLIP: Final[tuple[float, float]] = (0.0, 4.0)

#: Largest share of the lateral (linearity) budget the flagellar beat may take.
#: Giving the beat the whole budget makes the arithmetic work but the biology
#: wrong: the cell's *average path* comes out perfectly straight and the only
#: departure from a ruler is the beat itself. Real progressive sperm also yaw.
#: Reserving a quarter of the budget for heading wander keeps both mechanisms
#: active and, just as importantly, leaves the linearity solver something to
#: turn -- with the beat saturated there is no free parameter and the realised
#: LIN can only be whatever the beat happens to give.
BEAT_LATERAL_SHARE: Final[float] = 0.75

#: Nominal linearity used to size a *non-progressive* cell's beat. Such a cell
#: has a normal flagellar beat and a normal path speed; what it lacks is a
#: persistent heading. See :func:`simulate_trajectory`.
NON_PROGRESSIVE_BEAT_LIN: Final[float] = 0.75

#: Hard ceiling on the per-step heading standard deviation, in radians, for the
#: free-running random-walk branch. Beyond about this the heading is already
#: uniformly random each step and a larger value buys nothing.
MAX_HEADING_STEP_STD: Final[float] = 3.0

#: Ceiling on the stationary spread of a progressive cell's heading, radians.
#: At ~1.5 rad the mean resultant is already down to 0.32, well below any
#: progressive linearity, so nothing above it is reachable anyway.
MAX_HEADING_SPREAD: Final[float] = 1.5

#: Correlation time of a progressive cell's heading wander, in seconds. Short
#: relative to a typical track (40 ms against ~600 ms) so that a track contains
#: many independent excursions and its measured LIN concentrates on the target
#: instead of scattering. Physically it is the yaw wobble of the whole cell,
#: distinct from and slower than the flagellar beat.
HEADING_CORRELATION_S: Final[float] = 0.04


# --------------------------------------------------------------------------
# Trajectory synthesis
# --------------------------------------------------------------------------


def _ou_heading(
    target_resultant: float,
    n_points: int,
    dt_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Heading offsets for a *progressive* cell, realising a target linearity.

    A progressive sperm has a preferred direction that it wanders about and
    returns to. The natural model is therefore an Ornstein-Uhlenbeck process
    with stationary spread ``sigma`` and correlation time
    :data:`HEADING_CORRELATION_S`, not a random walk.

    Why not a random walk. A pure heading random walk has *no* preferred
    direction and unbounded spread, so the realised LIN of a finite track is
    the modulus of a small sum of weakly-correlated unit vectors: its
    expectation can be solved for exactly, but its scatter is enormous. In
    practice that put roughly 1 in 100 nominally rapid-progressive cells below
    LIN 0.6, i.e. mislabelled by the simulator's own definition. Mean-reverting
    headings fix that at the source rather than by widening a threshold: with a
    correlation time much shorter than the track, the time-average of
    ``cos(offset)`` concentrates, and the realised LIN lands tightly on target.

    For a stationary zero-mean Gaussian offset ``w`` with standard deviation
    ``sigma``, ``E[cos w] = exp(-sigma^2 / 2)``, so the spread that realises a
    target resultant ``P`` is ``sigma = sqrt(-2 ln P)`` -- closed form, no
    tuning.
    """
    p = float(np.clip(target_resultant, 1e-6, 1.0))
    sigma = math.sqrt(max(-2.0 * math.log(p), 0.0))
    sigma = min(sigma, MAX_HEADING_SPREAD)
    if sigma <= 0.0 or n_points < 2:
        return np.zeros(n_points, dtype=np.float64)
    rho = math.exp(-dt_s / max(HEADING_CORRELATION_S, 1e-6))
    innovation = sigma * math.sqrt(max(1.0 - rho * rho, 0.0))
    noise = rng.normal(0.0, 1.0, size=n_points)
    out = np.empty(n_points, dtype=np.float64)
    # Start from the stationary distribution so there is no burn-in transient.
    out[0] = sigma * noise[0]
    for k in range(1, n_points):
        out[k] = rho * out[k - 1] + innovation * noise[k]
    return out


def _track_linearity(points: np.ndarray) -> float:
    """LIN of a finished track: net displacement over path length."""
    seg = np.diff(points, axis=0)
    length = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    if length <= EPS:
        return 0.0
    net = points[-1] - points[0]
    return float(math.hypot(net[0], net[1]) / length)


def _solve_linearity_scale(
    build: Callable[[float], np.ndarray], target: float
) -> float:
    """Heading-wander scale whose *finished track* has LIN equal to ``target``.

    Matching the heading resultant is not enough. The flagellar beat is bounded
    and periodic, so it barely moves the path length, but its phase at the two
    *endpoints* does shift the net displacement -- by up to twice the beat
    amplitude. On a slow-progressive cell (VSL ~19 um/s, track 0.6 s) that is a
    LIN error of up to 0.2, which is the difference between a correctly and an
    incorrectly labelled sample.

    So the bisection runs on the completed trajectory instead: build it, measure
    its actual LIN, adjust. LIN falls monotonically as the wander widens, and
    each trial is a handful of vectorised operations on a ~100-point array, so
    this costs microseconds and makes the stored label exactly true rather than
    true on average.
    """
    if _track_linearity(build(0.0)) <= target:
        return 0.0  # even a perfectly straight heading cannot reach it
    lo, hi = 0.0, 1.0
    for _ in range(14):
        if _track_linearity(build(hi)) <= target:
            break
        lo, hi = hi, hi * 2.0
    else:
        return hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _track_linearity(build(mid)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _discrete_beat_rms(amplitude_um: float, freq_hz: float, dt_s: float) -> float:
    """RMS lateral speed of a *sampled* sinusoid, in um/s.

    The continuous answer ``2*pi*f*A/sqrt(2)`` overestimates what a camera
    actually measures, because the track is a polyline through the samples, not
    the smooth curve. Using the exact discrete difference keeps the realised
    VCL on target instead of consistently overshooting it.
    """
    if dt_s <= 0.0 or freq_hz <= 0.0 or amplitude_um <= 0.0:
        return 0.0
    return float(2.0 * amplitude_um * abs(math.sin(math.pi * freq_hz * dt_s)) / (dt_s * math.sqrt(2.0)))


def beat_and_forward_split(
    state: HealthState, dt_s: float, lin_for_budget: float
) -> tuple[float, float]:
    """Split the VCL budget into a forward speed and a beat half-amplitude.

    Returns ``(v_forward_um_s, beat_amplitude_um)``.

    The beat inflates VCL without touching VSL, so the largest beat compatible
    with a linearity of ``lin_for_budget`` is the one that uses up exactly the
    lateral share ``VCL * sqrt(1 - lin**2)``. The state's *requested* amplitude
    is honoured when it is smaller than that, and capped when it is not: VCL
    and LIN are what the decision rule depends on, so they win over ALH when
    the three cannot all be satisfied at once.
    """
    vcl = max(float(state.speed_um_s), 0.0)
    lin = float(np.clip(lin_for_budget, 0.0, 0.999))
    lateral_budget = vcl * math.sqrt(max(1.0 - lin * lin, 0.0)) * BEAT_LATERAL_SHARE
    beat_rms = _discrete_beat_rms(state.beat_amplitude_um, state.beat_frequency_hz, dt_s)
    if beat_rms > lateral_budget and beat_rms > 0.0:
        amplitude = float(state.beat_amplitude_um) * (lateral_budget / beat_rms)
        beat_rms = lateral_budget
    else:
        amplitude = float(state.beat_amplitude_um)
    v_forward = math.sqrt(max(vcl * vcl - beat_rms * beat_rms, 0.0))
    return v_forward, amplitude


def simulate_trajectory(
    state: HealthState,
    n_points: int,
    dt_s: float,
    um_per_px: float,
    rng: np.random.Generator,
    flow_px_s: tuple[float, float] = (0.0, 0.0),
    *,
    start_xy_px: tuple[float, float] = (0.0, 0.0),
    match_linearity: bool = True,
    normalize_vcl: bool = True,
    jitter_um: float = 0.0,
) -> np.ndarray:
    """Generate one trajectory, in **pixels**, shape ``(n_points, 2)``.

    Columns are ``(x, y)``. Pixels, not micrometres, because everything
    downstream of acquisition works in pixels and only converts at the point
    where a physical quantity is reported -- keeping the simulator in pixels
    means it exercises the same conversion path as the real system.

    Parameters
    ----------
    state
        Ground truth. ``speed_um_s`` is the target VCL, ``linearity`` the
        target LIN.
    n_points
        Number of samples, including the start point.
    dt_s
        Sampling interval; ``1 / fps``.
    um_per_px
        Optical scale. Must be positive: a zero or negative scale would make
        every velocity meaningless, so it is rejected rather than defaulted.
    rng
        Explicit generator; the global numpy state is never touched.
    flow_px_s
        Bulk flow added on top of the swimming component, in pixels/second.
    start_xy_px
        Where the track begins, in pixels.
    match_linearity
        When true (the default) the heading noise is *solved* so that the
        realised LIN matches ``state.linearity`` at this track length. When
        false, ``state.angle_noise`` is used as a free-running diffusion --
        the mode the scene generator needs, where an agent's lifetime is not
        known in advance.
    normalize_vcl
        Rescale the finished track so its measured VCL is exactly
        ``state.speed_um_s``. On by default so the ground truth is exact rather
        than approximate; turn it off to inspect the raw generative model.
    jitter_um
        Per-point isotropic localisation noise. Zero by default: at 160 FPS
        even sub-micrometre jitter adds tens of um/s to VCL, so injecting it
        into the *labelling* path would corrupt the very features the labels
        are built from. The scene generator adds detection noise separately,
        where it belongs.

    Notes
    -----
    Immotile cells take a separate branch: a pure Brownian walk whose step
    scale is set by ``speed_um_s``. Net displacement then grows as
    ``sqrt(n)`` rather than ``n``, so VSL falls towards zero as the track
    lengthens, which is exactly the behaviour that distinguishes a dead cell
    from a slow one.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if um_per_px <= 0.0:
        raise ValueError(f"um_per_px must be positive, got {um_per_px}")

    n_steps = n_points - 1
    px_per_um = 1.0 / um_per_px
    out = np.empty((n_points, 2), dtype=np.float64)

    if state.motility is MotilityClass.IMMOTILE:
        step_um = float(state.speed_um_s) * dt_s
        steps = rng.normal(0.0, step_um / math.sqrt(2.0), size=(n_points, 2))
        steps[0] = 0.0
        pos_um = np.cumsum(steps, axis=0)
    else:
        # A non-progressive cell beats like any other -- it simply gets
        # nowhere. Sizing its beat from its (near-zero) target LIN would spend
        # the entire path budget on the beat and leave no forward motion at
        # all, which looks nothing like a real thrashing sperm. So the beat is
        # sized from a nominal progressive linearity and the low LIN is
        # produced entirely by heading diffusion, which is the actual biology.
        progressive = state.motility.is_progressive
        lin_budget = (
            float(state.linearity) if progressive else NON_PROGRESSIVE_BEAT_LIN
        )
        v_forward, amplitude_um = beat_and_forward_split(state, dt_s, lin_budget)
        heading0 = float(rng.uniform(0.0, 2.0 * math.pi))
        # A preferred direction only exists for a progressive cell, so only a
        # progressive cell gets the mean-reverting model. Non-progressive cells
        # run free at their sampled heading diffusion, where LIN emerges (near
        # 1/sqrt(N)) instead of being dialled in -- which is what "going
        # nowhere" actually means.
        tune = match_linearity and progressive and v_forward > EPS
        if tune:
            target = min(
                float(state.speed_um_s) * float(state.linearity) / v_forward, 1.0
            )
            offsets = _ou_heading(target, n_points, dt_s, rng)
        else:
            step_std = float(
                min(state.angle_noise * math.sqrt(dt_s), MAX_HEADING_STEP_STD)
            )
            increments = rng.normal(0.0, step_std, size=n_points)
            increments[0] = 0.0
            offsets = np.cumsum(increments)

        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        t = np.arange(n_points, dtype=np.float64) * dt_s
        lateral = amplitude_um * np.sin(
            2.0 * math.pi * state.beat_frequency_hz * t + phase
        )

        def _positions(scale: float) -> np.ndarray:
            """Trajectory for a heading wander scaled by ``scale``."""
            heading = heading0 + scale * offsets
            step_vec = np.empty((n_points, 2), dtype=np.float64)
            step_vec[:, 0] = np.cos(heading) * v_forward * dt_s
            step_vec[:, 1] = np.sin(heading) * v_forward * dt_s
            step_vec[0] = 0.0
            path_um = np.cumsum(step_vec, axis=0)
            # Flagellar beat: perpendicular to the instantaneous heading.
            normal = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
            return path_um + normal * lateral[:, None]

        if tune:
            pos_um = _positions(_solve_linearity_scale(_positions, state.linearity))
        else:
            pos_um = _positions(1.0)

    if normalize_vcl:
        # Rescale so the realised path speed equals the state's target VCL
        # exactly. Two second-order effects otherwise leak in: the polyline
        # under-measures the sampled sinusoid, and a fast-rotating heading
        # swings the beat's own normal vector and adds path length (which
        # inflated non-progressive VCL by ~60% before this was added). A single
        # global scale about the start point fixes both, and because it scales
        # VSL and VCL together it leaves LIN, STR and WOB untouched.
        seg = np.diff(pos_um, axis=0)
        realised = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        duration = n_steps * dt_s
        target_len = float(state.speed_um_s) * duration
        if realised > EPS and target_len > 0.0:
            pos_um = (pos_um - pos_um[0]) * (target_len / realised)

    if jitter_um > 0.0:
        pos_um = pos_um + rng.normal(0.0, jitter_um, size=pos_um.shape)

    out[:] = pos_um * px_per_um
    out[:, 0] += start_xy_px[0]
    out[:, 1] += start_xy_px[1]

    # Bulk flow, in pixels, added on top of the swimming component.
    if flow_px_s[0] != 0.0 or flow_px_s[1] != 0.0:
        t_s = np.arange(n_points, dtype=np.float64) * dt_s
        out[:, 0] += flow_px_s[0] * t_s
        out[:, 1] += flow_px_s[1] * t_s
    return out


# --------------------------------------------------------------------------
# CASA features (independent reference implementation)
# --------------------------------------------------------------------------


def _pad_linear(points: np.ndarray, left: int, right: int) -> np.ndarray:
    """Pad by linear extrapolation of the end segments.

    Edge replication (``np.pad(mode="edge")``) would flatten the ends of the
    track, shortening the average path and making WOB come out below 1 even
    for a perfectly straight line -- a several-percent bias on short tracks,
    which at 160 FPS is every track. Linear extrapolation reproduces a
    straight line exactly, so the identities VAP = VCL and WOB = 1 hold when
    they should.
    """
    if left == 0 and right == 0:
        return points
    head = points[0] - np.arange(left, 0, -1)[:, None] * (points[1] - points[0])
    tail = points[-1] + np.arange(1, right + 1)[:, None] * (points[-1] - points[-2])
    return np.concatenate([head, points, tail], axis=0)


def _average_path(points: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, shape preserved.

    The array is padded rather than shortened so that ALH and BCF are defined
    over the whole track; truncating instead would make VAP and VCL cover
    different time spans, and their ratio (WOB) would be wrong.
    """
    n = points.shape[0]
    w = int(min(max(window, 1), n))
    if w <= 1 or n < 2:
        return points.astype(np.float64, copy=True)
    left = (w - 1) // 2
    right = w - 1 - left
    padded = _pad_linear(points.astype(np.float64), left, right)
    kernel = np.ones(w, dtype=np.float64) / float(w)
    smooth = np.empty_like(points, dtype=np.float64)
    for axis in range(points.shape[1]):
        smooth[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return smooth


def casa_features(
    points_xy: np.ndarray,
    dt_s: float,
    um_per_px: float,
    *,
    vap_window: int = VAP_WINDOW,
) -> dict[str, float]:
    """Textbook CASA kinematics for one clean track.

    This is the *labelling* implementation: an intentionally independent second
    opinion against which :mod:`sperm_sorting.motion` is cross-checked. It
    assumes what the production estimator may not -- clean, gap-free, evenly
    spaced, already flow-corrected points -- and is therefore short enough to
    read and verify against the definitions:

    ``VCL``
        Curvilinear velocity: total polyline length / duration.
    ``VSL``
        Straight-line velocity: |last - first| / duration.
    ``VAP``
        Average-path velocity: length of the ``vap_window`` moving average /
        duration.
    ``LIN``
        VSL / VCL. ``STR`` = VSL / VAP. ``WOB`` = VAP / VCL.
    ``ALH``
        Amplitude of lateral head displacement: mean *width* of the excursion
        about the average path, i.e. twice the mean absolute perpendicular
        deviation. Systems differ on mean-vs-maximum; the choice is stated
        here and is part of the number.
    ``BCF``
        Beat-cross frequency: sign changes of the perpendicular deviation per
        second, i.e. how often the head crosses its own average path.

    Parameters are in pixels and seconds; outputs are in um/s, um and Hz.

    Caveat on BCF and ALH for non-beating cells: both are measured against the
    average path, so for an immotile or thrashing cell the "deviation" is
    localisation noise and the crossing count saturates near ``fps / 4``
    regardless of biology. The numbers are still emitted -- refusing to compute
    them is the production estimator's job, and hard-coding a refusal here
    would hide the artefact from the cross-check -- but they carry no
    information for those grades and must not be read as a beat rate.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2), got {pts.shape}")
    n = pts.shape[0]
    if n < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} points, got {n}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if um_per_px <= 0.0:
        raise ValueError(f"um_per_px must be positive, got {um_per_px}")

    pts_um = pts * um_per_px
    duration_s = (n - 1) * dt_s

    seg = np.diff(pts_um, axis=0)
    path_len = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    vcl = path_len / duration_s

    net = pts_um[-1] - pts_um[0]
    vsl = float(math.hypot(net[0], net[1])) / duration_s

    avg = _average_path(pts_um, vap_window)
    avg_seg = np.diff(avg, axis=0)
    vap = float(np.sum(np.hypot(avg_seg[:, 0], avg_seg[:, 1]))) / duration_s

    lin = vsl / vcl if vcl > EPS else 0.0
    strr = vsl / vap if vap > EPS else 0.0
    wob = vap / vcl if vcl > EPS else 0.0

    # Perpendicular deviation of each raw point from the local average-path
    # direction. The direction is taken from the smoothed path so that the
    # beat itself does not rotate the frame it is being measured in.
    tangent = np.gradient(avg, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    safe = norm > EPS
    unit = np.zeros_like(tangent)
    unit[safe] = tangent[safe] / norm[safe, None]
    perp = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    delta = pts_um - avg
    dev = np.einsum("ij,ij->i", delta, perp)

    alh = float(2.0 * np.mean(np.abs(dev)))

    sign = np.sign(dev)
    nonzero = sign[sign != 0.0]
    crossings = int(np.count_nonzero(np.diff(nonzero) != 0.0)) if nonzero.size > 1 else 0
    bcf = crossings / duration_s / 2.0

    return {
        "vcl": float(vcl),
        "vsl": float(vsl),
        "vap": float(vap),
        "lin": float(np.clip(lin, 0.0, 1.0)),
        "str": float(np.clip(strr, 0.0, 1.0)),
        "wob": float(np.clip(wob, 0.0, 1.0)),
        "alh": float(alh),
        "bcf": float(bcf),
    }


def normalize_features(feats: dict[str, float] | np.ndarray) -> np.ndarray:
    """Fixed, documented normalisation to a ``float32[8]`` vector.

    The divisors live in :data:`FEATURE_SCALES` and never change with the data.
    A normalisation fitted to a training set is the classic source of
    train/serve skew: the serving process has no access to that set, so it
    either re-derives different statistics or silently ships stale ones. Fixed
    physical divisors have neither failure mode, and they keep every component
    interpretable (``0.5`` in the VCL slot always means 75 um/s).

    Accepts either the dict from :func:`casa_features` or a raw array already
    in :data:`FEATURE_NAMES` order.
    """
    if isinstance(feats, dict):
        missing = [k for k in FEATURE_NAMES if k not in feats]
        if missing:
            raise ValueError(f"missing CASA feature(s): {missing}")
        values = np.array([float(feats[k]) for k in FEATURE_NAMES], dtype=np.float64)
    else:
        values = np.asarray(feats, dtype=np.float64).ravel()
        if values.size != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features in {list(FEATURE_NAMES)} "
                f"order, got {values.size}"
            )
    scales = np.array([FEATURE_SCALES[k] for k in FEATURE_NAMES], dtype=np.float64)
    out = np.clip(values / scales, FEATURE_CLIP[0], FEATURE_CLIP[1])
    return out.astype(np.float32)


def features_for_state(
    state: HealthState,
    rng: np.random.Generator,
    *,
    n_points: int = 64,
    dt_s: float = 1.0 / 160.0,
    um_per_px: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Convenience: trajectory -> CASA dict + normalised vector.

    Used by the dataset builder so that the exact same path produces the stored
    features and the stored labels; generating them separately would risk the
    two describing different draws.
    """
    track = simulate_trajectory(state, n_points, dt_s, um_per_px, rng)
    feats = casa_features(track, dt_s, um_per_px)
    return feats, normalize_features(feats)


def describe() -> dict[str, Any]:
    """Algorithm metadata stamped into ``meta.json``."""
    return {
        "vap_window": VAP_WINDOW,
        "feature_names": list(FEATURE_NAMES),
        "feature_scales": dict(FEATURE_SCALES),
        "feature_clip": list(FEATURE_CLIP),
        "alh_definition": "2 * mean(|perpendicular deviation from average path|)",
        "bcf_definition": "average-path sign changes per second / 2",
    }


if __name__ == "__main__":  # pragma: no cover - runnable self-check
    from .params import (
        LINEARITY_BAND,
        SPEED_BAND_UM_S,
        sample_health_state,
    )

    DT = 1.0 / 160.0
    UM_PER_PX = 0.5
    N = 96

    # -- determinism -------------------------------------------------------
    st = sample_health_state(np.random.default_rng(1), motility=MotilityClass.RAPID_PROGRESSIVE)
    t1 = simulate_trajectory(st, N, DT, UM_PER_PX, np.random.default_rng(5))
    t2 = simulate_trajectory(st, N, DT, UM_PER_PX, np.random.default_rng(5))
    assert np.array_equal(t1, t2), "same seed must give a byte-identical track"
    t3 = simulate_trajectory(st, N, DT, UM_PER_PX, np.random.default_rng(6))
    assert not np.array_equal(t1, t3), "different seeds must differ"
    assert t1.shape == (N, 2) and t1.dtype == np.float64

    # -- a straight, beat-free track reproduces its own definitions --------
    v_um_s, angle = 60.0, 0.7
    t = np.arange(N) * DT
    straight = np.stack(
        [np.cos(angle) * v_um_s * t / UM_PER_PX, np.sin(angle) * v_um_s * t / UM_PER_PX],
        axis=1,
    )
    f = casa_features(straight, DT, UM_PER_PX)
    assert abs(f["vcl"] - v_um_s) < 1e-6, f["vcl"]
    assert abs(f["vsl"] - v_um_s) < 1e-6, f["vsl"]
    assert abs(f["vap"] - v_um_s) < 1e-6, f["vap"]
    assert abs(f["lin"] - 1.0) < 1e-6 and abs(f["str"] - 1.0) < 1e-6
    assert abs(f["wob"] - 1.0) < 1e-6 and f["alh"] < 1e-6

    # -- grade-wise behaviour ---------------------------------------------
    summary: dict[str, dict[str, float]] = {}
    for grade in (
        MotilityClass.RAPID_PROGRESSIVE,
        MotilityClass.SLOW_PROGRESSIVE,
        MotilityClass.NON_PROGRESSIVE,
        MotilityClass.IMMOTILE,
    ):
        rng = np.random.default_rng(20)
        rows: list[dict[str, float]] = []
        for _ in range(300):
            # Morphology-normal so the bands test the motility model alone;
            # the tail-defect speed coupling is asserted separately below.
            s = sample_health_state(rng, motility=grade, aspects=(0, 0, 0, 0))
            trk = simulate_trajectory(s, N, DT, UM_PER_PX, rng)
            rows.append(casa_features(trk, DT, UM_PER_PX))
        summary[str(grade)] = {
            k: float(np.mean([r[k] for r in rows])) for k in FEATURE_NAMES
        }
        summary[str(grade)]["lin_min"] = float(np.min([r["lin"] for r in rows]))
        summary[str(grade)]["lin_max"] = float(np.max([r["lin"] for r in rows]))
        summary[str(grade)]["vcl_min"] = float(np.min([r["vcl"] for r in rows]))
        summary[str(grade)]["vcl_max"] = float(np.max([r["vcl"] for r in rows]))
        summary[str(grade)]["vsl_max"] = float(np.max([r["vsl"] for r in rows]))

    rp = summary[str(MotilityClass.RAPID_PROGRESSIVE)]
    assert rp["lin_min"] > 0.6, f"rapid LIN floor {rp['lin_min']:.3f} must exceed 0.6"
    lo, hi = SPEED_BAND_UM_S[MotilityClass.RAPID_PROGRESSIVE]
    assert lo * 0.9 <= rp["vcl_min"] and rp["vcl_max"] <= hi * 1.1, (
        f"rapid VCL {rp['vcl_min']:.1f}-{rp['vcl_max']:.1f} outside band {lo}-{hi}"
    )
    assert rp["vsl"] > 25.0, f"rapid VSL {rp['vsl']:.1f} must clear the 25 um/s cut"

    sp = summary[str(MotilityClass.SLOW_PROGRESSIVE)]
    assert sp["vsl"] > 5.0, f"slow VSL {sp['vsl']:.1f} must clear the 5 um/s cut"
    assert sp["lin_min"] > 0.4, sp["lin_min"]

    npg = summary[str(MotilityClass.NON_PROGRESSIVE)]
    assert npg["lin_max"] < 0.45, f"non-progressive LIN max {npg['lin_max']:.3f} too high"
    assert npg["vcl"] > 8.0, f"non-progressive VCL {npg['vcl']:.1f} must stay non-trivial"
    assert npg["vsl"] < 5.0, f"non-progressive VSL {npg['vsl']:.2f} must stay small"

    imm = summary[str(MotilityClass.IMMOTILE)]
    assert imm["vsl_max"] < 1.0, f"immotile VSL max {imm['vsl_max']:.3f} must be ~0"
    assert imm["vcl"] < 2.0, f"immotile VCL {imm['vcl']:.3f} must be ~0"

    # -- realised VCL is the state's VCL, not merely close to it -----------
    rng_v = np.random.default_rng(41)
    for _ in range(200):
        s = sample_health_state(rng_v)
        trk = simulate_trajectory(s, N, DT, UM_PER_PX, rng_v)
        got = casa_features(trk, DT, UM_PER_PX)["vcl"]
        assert abs(got - s.speed_um_s) < 1e-6 * max(s.speed_um_s, 1.0) + 1e-9, (
            f"VCL {got:.6f} != target {s.speed_um_s:.6f}"
        )

    # -- the VAP window must span a beat, or ALH collapses -----------------
    probe_w = sample_health_state(
        np.random.default_rng(19), motility=MotilityClass.RAPID_PROGRESSIVE
    )
    trk_w = simulate_trajectory(probe_w, 320, DT, UM_PER_PX, np.random.default_rng(19))
    alh_short = casa_features(trk_w, DT, UM_PER_PX, vap_window=5)["alh"]
    alh_beat = casa_features(trk_w, DT, UM_PER_PX, vap_window=VAP_WINDOW)["alh"]
    assert alh_beat > 2.0 * alh_short, (
        f"a beat-spanning window must recover ALH: {alh_short:.3f} -> {alh_beat:.3f}"
    )
    assert vap_window_for_fps(160.0) == 11 and vap_window_for_fps(60.0) == 5

    # -- a defective tail costs speed (documented cross-modal coupling) ----
    rng_t = np.random.default_rng(77)
    good_tail, bad_tail = [], []
    for _ in range(400):
        ok = sample_health_state(
            rng_t, motility=MotilityClass.RAPID_PROGRESSIVE, aspects=(0, 0, 0, 0)
        )
        bad = sample_health_state(
            rng_t, motility=MotilityClass.RAPID_PROGRESSIVE, aspects=(0, 0, 0, 1)
        )
        good_tail.append(casa_features(simulate_trajectory(ok, N, DT, UM_PER_PX, rng_t), DT, UM_PER_PX)["vcl"])
        bad_tail.append(casa_features(simulate_trajectory(bad, N, DT, UM_PER_PX, rng_t), DT, UM_PER_PX)["vcl"])
    assert float(np.mean(bad_tail)) < 0.9 * float(np.mean(good_tail)), (
        "tail=1 must slow a progressive cell"
    )

    # -- LIN is achieved, not merely aimed at ------------------------------
    for grade in (MotilityClass.RAPID_PROGRESSIVE, MotilityClass.SLOW_PROGRESSIVE):
        rng = np.random.default_rng(33)
        errs = []
        for _ in range(200):
            s = sample_health_state(rng, motility=grade)
            trk = simulate_trajectory(s, 256, DT, UM_PER_PX, rng)
            errs.append(casa_features(trk, DT, UM_PER_PX)["lin"] - s.linearity)
        assert abs(float(np.mean(errs))) < 0.12, (grade, float(np.mean(errs)))
        band = LINEARITY_BAND[grade]
        assert band[0] <= 1.0 and band[1] <= 1.0

    # -- bulk flow is additive and recoverable -----------------------------
    s = sample_health_state(np.random.default_rng(2), motility=MotilityClass.IMMOTILE)
    with_flow = simulate_trajectory(
        s, N, DT, UM_PER_PX, np.random.default_rng(9), flow_px_s=(120.0, -30.0)
    )
    without = simulate_trajectory(s, N, DT, UM_PER_PX, np.random.default_rng(9))
    tt = np.arange(N) * DT
    recovered = with_flow - without
    assert np.allclose(recovered[:, 0], 120.0 * tt) and np.allclose(recovered[:, 1], -30.0 * tt)

    # -- BCF tracks the commanded beat frequency ---------------------------
    probe = sample_health_state(np.random.default_rng(4), motility=MotilityClass.RAPID_PROGRESSIVE)
    probe.beat_frequency_hz = 12.0
    probe.beat_amplitude_um = 2.0
    probe.linearity = 0.70
    bcfs = [
        casa_features(
            simulate_trajectory(probe, 320, DT, UM_PER_PX, np.random.default_rng(k)),
            DT,
            UM_PER_PX,
        )["bcf"]
        for k in range(20)
    ]
    assert 6.0 < float(np.mean(bcfs)) < 18.0, f"BCF {np.mean(bcfs):.1f} far from 12 Hz"

    # -- normalisation -----------------------------------------------------
    vec = normalize_features(rp | {k: rp[k] for k in FEATURE_NAMES})
    assert vec.shape == (8,) and vec.dtype == np.float32
    assert np.all(vec >= FEATURE_CLIP[0]) and np.all(vec <= FEATURE_CLIP[1])
    assert np.allclose(
        normalize_features(dict.fromkeys(FEATURE_NAMES, 0.0)), np.zeros(8, dtype=np.float32)
    )
    assert np.allclose(normalize_features(np.zeros(8)), np.zeros(8, dtype=np.float32))
    try:
        normalize_features({"vcl": 1.0})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("normalize_features must reject an incomplete dict")

    # -- guards ------------------------------------------------------------
    guards: tuple[Callable[[], object], ...] = (
        lambda: casa_features(np.zeros((2, 2)), DT, UM_PER_PX),
        lambda: casa_features(np.zeros((5, 3)), DT, UM_PER_PX),
        lambda: casa_features(np.zeros((5, 2)), 0.0, UM_PER_PX),
        lambda: simulate_trajectory(st, N, DT, 0.0, np.random.default_rng(0)),
    )
    for guard in guards:
        try:
            guard()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")  # pragma: no cover

    print("motility.py self-check OK")
    hdr = "grade                 " + "".join(f"{k:>9}" for k in FEATURE_NAMES)
    print(hdr)
    for name, row in summary.items():
        print(f"{name:<22}" + "".join(f"{row[k]:9.2f}" for k in FEATURE_NAMES))
    print(
        f"  rapid LIN range      {rp['lin_min']:.3f}-{rp['lin_max']:.3f}   "
        f"VCL range {rp['vcl_min']:.1f}-{rp['vcl_max']:.1f} um/s"
    )
    print(
        f"  non-progressive LIN  max {npg['lin_max']:.3f}   mean VCL {npg['vcl']:.2f} um/s   "
        f"mean VSL {npg['vsl']:.2f} um/s"
    )
    print(f"  immotile             max VSL {imm['vsl_max']:.4f} um/s   mean VCL {imm['vcl']:.4f} um/s")
