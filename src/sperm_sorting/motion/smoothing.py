"""Trajectory smoothing and path geometry.

The *average path* is the backbone of half of CASA: VAP is measured along it,
and STR, WOB, ALH and BCF are all defined relative to it. It is therefore worth
being explicit about something the CASA literature repeats and users still
forget:

    **The average path is an algorithm, not a measurement.** Its smoothing
    method and window length are part of the definition of every quantity
    derived from it, so VAP/STR/WOB/ALH/BCF from two CASA systems are not
    comparable unless both systems used the same smoother.

For that reason :class:`~sperm_sorting.config.MotionConfig` carries
``vap_window_ms``, ``smoothing`` and ``savgol_polyorder``, and
:mod:`.features` stamps the *resolved* window into the profile version of
every audit record.

Note the units: the window these functions take is a frame count, but the
configuration specifies a **duration**, converted per track against the frame
rate measured from that track's own timestamps
(:meth:`MotionConfig.vap_window_frames`). A fixed frame count is what Mortimer,
van der Horst & Mortimer (Asian J Androl 2015;17:545-53) identify as a source
of "widely aberrant ALH values": five frames is 100 ms at 50 FPS but 31 ms at
160 FPS, so the same nominal setting smooths a third as much trajectory on the
faster camera.

Endpoint handling
-----------------
A centred moving average is undefined within half a window of each end of the
track, and the three usual repairs behave very differently:

* **zero padding** -- treats the frames before the track as if the sperm were
  at the image origin ``(0, 0)``. That drags the first and last few points of
  the average path hundreds of pixels toward the top-left corner, inflating VAP
  and ALH enormously. Never acceptable.
* **edge replication** -- flattens the ends of the average path and biases the
  endpoints toward the raw endpoint value, shortening VAP.
* **symmetric shrinking window** (used here) -- at index ``i`` the half-width
  is ``min(half, i, n - 1 - i)``, so the window is always centred on the point
  and never reads outside the measured data.

Symmetric shrinking has a property the other two lack: the first and last
smoothed points are exactly the first and last measured points, so the average
path and the straight-line path share their endpoints. VSL is therefore
naturally bounded by VAP, and STR = VSL/VAP stays in ``[0, 1]`` instead of
drifting above 1 through an artefact of the smoother.

The Savitzky-Golay alternative uses SciPy's ``mode="interp"``, which fits the
edge polynomial to the last ``window`` real samples rather than padding, for
exactly the same reason.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

__all__ = [
    "SmoothingMethod",
    "as_points_array",
    "moving_average_path",
    "net_displacement",
    "path_length",
    "savgol_path",
    "smooth_path",
    "step_lengths",
]

#: Mirrors ``MotionConfig.smoothing`` so the dispatcher and the config cannot
#: drift apart without a type error.
SmoothingMethod = Literal["none", "moving_average", "savgol"]


def as_points_array(points_xy: ArrayLike) -> NDArray[np.float64]:
    """Coerce a trajectory to a contiguous ``(N, 2)`` float64 array.

    Kinematics are ratios of small differences of large pixel coordinates, so
    the whole module works in float64: at 1920 px and float32 the spacing is
    ~1e-4 px, which is a measurable fraction of a per-frame step for a slow
    sperm.
    """
    arr = np.ascontiguousarray(np.asarray(points_xy, dtype=np.float64))
    if arr.ndim == 1 and arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"trajectory must have shape (N, 2), got {arr.shape}"
        )
    return arr


def moving_average_path(points_xy: ArrayLike, window: int) -> NDArray[np.float64]:
    """CASA-standard average path: a centred moving average of the trajectory.

    Parameters
    ----------
    points_xy
        ``(N, 2)`` measured positions in pixels, in time order.
    window
        Window length in *frames*, normally the output of
        :meth:`MotionConfig.vap_window_frames` for the measured frame rate. An
        even window cannot be centred on a sample, so it is widened to the next
        odd length (``window`` 4 behaves as 5); this is recorded rather than
        hidden because the window is part of the definition of VAP.

    Returns
    -------
    ndarray
        ``(N, 2)`` average path, same length as the input.

    Notes
    -----
    The window shrinks symmetrically at the endpoints -- see the module
    docstring for why zero padding and edge replication are both rejected. A
    window of 0 or 1 (or a track shorter than 3 points) is a no-op and returns
    a copy, so callers never have to special-case short tracks.
    """
    pts = as_points_array(points_xy)
    n = pts.shape[0]
    half = max(0, int(window) // 2)
    if n < 3 or half == 0:
        return pts.copy()

    idx = np.arange(n)
    # Symmetric shrinking half-width: never reads outside the measured data.
    h = np.minimum(half, np.minimum(idx, n - 1 - idx))
    lo = idx - h
    hi = idx + h + 1

    # Prefix sums give an O(N) windowed mean regardless of window length,
    # which matters because this runs per track at up to 160 frames/second.
    cumsum = np.zeros((n + 1, 2), dtype=np.float64)
    np.cumsum(pts, axis=0, out=cumsum[1:])
    counts = (hi - lo).astype(np.float64)[:, None]
    return (cumsum[hi] - cumsum[lo]) / counts


def savgol_path(
    points_xy: ArrayLike, window: int, polyorder: int
) -> NDArray[np.float64]:
    """Savitzky-Golay smoothed trajectory.

    A local least-squares polynomial fit preserves the amplitude of curvature
    better than a boxcar mean, so it distorts a genuinely helical/circular
    swim path less. It is offered as an alternative average-path definition;
    which one is in force is recorded in the motility profile version.

    ``window`` is forced odd and clipped to the track length, and ``polyorder``
    is clipped to ``window - 1``, because SciPy raises on those combinations
    and a short track is a normal occurrence, not an error. ``mode="interp"``
    fits the edge polynomial to real samples instead of padding.
    """
    pts = as_points_array(points_xy)
    n = pts.shape[0]
    if n < 3:
        return pts.copy()

    win = int(window)
    if win % 2 == 0:
        win += 1
    max_win = n if n % 2 == 1 else n - 1
    win = min(win, max_win)
    order = min(int(polyorder), win - 1)
    if win < 3 or order < 1:
        # Nothing to fit: a first-order fit through <3 points is the data.
        return pts.copy()

    smoothed = savgol_filter(
        pts, window_length=win, polyorder=order, axis=0, mode="interp"
    )
    return np.ascontiguousarray(smoothed, dtype=np.float64)


def smooth_path(
    points_xy: ArrayLike,
    method: SmoothingMethod | str,
    window: int,
    polyorder: int = 2,
) -> NDArray[np.float64]:
    """Dispatch to the configured average-path smoother.

    ``method="none"`` returns the raw trajectory, which makes VAP identical to
    VCL and WOB identically 1. That is a legitimate configuration (it removes
    the algorithm dependence at the cost of the whole VAP family being
    uninformative), so it is supported rather than rejected.
    """
    if method == "none":
        return as_points_array(points_xy).copy()
    if method == "moving_average":
        return moving_average_path(points_xy, window)
    if method == "savgol":
        return savgol_path(points_xy, window, polyorder)
    raise ValueError(
        f"unknown smoothing method {method!r}; expected 'none', "
        "'moving_average' or 'savgol'"
    )


def step_lengths(points_xy: ArrayLike) -> NDArray[np.float64]:
    """Euclidean length of every point-to-point step, shape ``(N - 1,)``."""
    pts = as_points_array(points_xy)
    if pts.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    return np.hypot(deltas[:, 0], deltas[:, 1])


def path_length(points_xy: ArrayLike) -> float:
    """Total point-to-point path length in the units of ``points_xy``.

    This is the numerator of VCL when applied to the measured track and of VAP
    when applied to the average path.
    """
    return float(step_lengths(points_xy).sum())


def net_displacement(points_xy: ArrayLike) -> float:
    """Straight-line distance from the first point to the last.

    The numerator of VSL. Note that this ignores everything in between by
    design: a sperm that swims a perfect circle back to its origin has a large
    path length and zero net displacement, which is precisely the distinction
    LIN exists to express.
    """
    pts = as_points_array(points_xy)
    if pts.shape[0] < 2:
        return 0.0
    delta = pts[-1] - pts[0]
    return float(np.hypot(delta[0], delta[1]))
