"""Removal of bulk fluid motion from observed trajectories.

Why this module exists
----------------------
What the camera sees is not swimming. It is

    observed motion = self-propulsion + bulk transport by the fluid

and in a microfluidic channel the second term is usually the larger one. A
dead, entirely immotile sperm carried along at 120 px/s by the flow traces a
long, arrow-straight track: high VSL, LIN close to 1. Fed to a progressive
motility rule uncorrected, it is graded *rapid progressive* -- the single most
consequential silent error available in this pipeline, because it would push
non-viable cells into the accepted fraction.

Every progressive-motility decision must therefore be taken on flow-corrected
kinematics. The raw values are kept alongside so that a reviewer can see how
large the correction was and challenge it.

Estimating the flow
-------------------
Four strategies, matching :class:`~sperm_sorting.schemas.enums.FlowCorrectionMode`:

``DISABLED``
    Subtract nothing. Correct only for still-fluid bench recordings, where
    applying a spurious correction would be worse than applying none.
``FIXED_VECTOR``
    One calibrated ``(vx, vy)`` everywhere. Cheap and stable; a poor model near
    the channel walls, where Poiseuille flow is much slower than mid-channel.
``FLOW_MAP``
    A calibrated ``(H, W, 2)`` velocity field, bilinearly sampled at each
    track point. This is the model that respects the parabolic cross-channel
    profile.
``ROBUST_ESTIMATE``
    Measured live from the population: the slowest fraction of tracks is
    assumed to be passively transported debris and non-motile cells, and their
    *median* velocity is the bulk flow. A median rather than a mean, so that a
    handful of fast swimmers accidentally included in the subset cannot drag
    the estimate; and a hard minimum track count, below which the estimator
    reports ``None`` instead of guessing from a sample too small to be robust.

Applying the correction
-----------------------
:func:`apply_flow_correction` subtracts the *cumulative displacement* the flow
would have caused by each timestamp, not a per-point velocity. Positions are
positions: to remove a transport velocity from a position series you must
integrate it over elapsed time. Subtracting a velocity from a coordinate is a
unit error that happens to look plausible.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import FlowCorrectionConfig
from ..constants import EPS
from ..errors import CalibrationError
from ..schemas.enums import FlowCorrectionMode
from ..schemas.frame import FramePacket
from ..schemas.track import TrackRecord
from .smoothing import as_points_array

__all__ = [
    "DisabledFlow",
    "FixedVectorFlow",
    "FlowEstimator",
    "FlowMapFlow",
    "RobustFlowEstimator",
    "apply_flow_correction",
    "build_flow_estimator",
]


# ==========================================================================
# Interface
# ==========================================================================


class FlowEstimator(ABC):
    """Estimates the bulk fluid velocity in pixels per second.

    Implementations must:

    * return ``None`` -- never a fabricated zero -- when they cannot produce a
      trustworthy estimate. ``(0.0, 0.0)`` means "measured, and it is zero";
      ``None`` means "unknown", and the caller must decide whether classifying
      uncorrected kinematics is acceptable (usually it is not);
    * be safe to call once per frame from one thread;
    * keep :meth:`sample_points` consistent with the last :meth:`estimate`.
    """

    #: Which correction this estimator implements; copied into the audit record.
    mode: FlowCorrectionMode = FlowCorrectionMode.DISABLED
    #: Human-readable identifier for the audit log header.
    name: str = "flow_estimator"

    def __init__(self) -> None:
        self._last_vector: tuple[float, float] | None = None

    @abstractmethod
    def estimate(
        self, tracks: Sequence[TrackRecord], frame: FramePacket | None = None
    ) -> tuple[float, float] | None:
        """Return the bulk flow ``(vx, vy)`` in px/s, or ``None`` if unknown.

        Parameters
        ----------
        tracks
            Tracks currently active. Estimators that measure the flow from the
            population read their velocities from here; the others ignore it.
        frame
            The frame being processed, when one is available. Population-based
            estimators use it to restrict themselves to tracks that were
            actually observed on this frame, which is what makes the estimate
            *per-frame* rather than an average over the tracks' lifetimes.
        """

    def current_vector(self) -> tuple[float, float] | None:
        """The last vector :meth:`estimate` returned, or ``None``."""
        return self._last_vector

    def sample_points(self, points_xy: ArrayLike) -> NDArray[np.float64]:
        """Per-point flow velocity, shape ``(N, 2)``, px/s.

        The default implementation broadcasts the single current vector, which
        is exact for the spatially-uniform estimators.
        :class:`FlowMapFlow` overrides it with a real spatial lookup.

        An unknown flow samples as zero here; callers that must not silently
        skip the correction check :meth:`current_vector` for ``None`` first.
        """
        pts = as_points_array(points_xy)
        vector = self._last_vector or (0.0, 0.0)
        out = np.empty_like(pts)
        out[:, 0] = vector[0]
        out[:, 1] = vector[1]
        return out

    def reset(self) -> None:
        """Clear any temporal state. Only valid between sessions."""
        self._last_vector = None

    def describe(self) -> dict[str, Any]:
        """Metadata stamped into the audit log header."""
        return {"name": self.name, "mode": str(self.mode)}

    def _remember(self, vector: tuple[float, float] | None) -> None:
        """Cache the latest estimate so :meth:`sample_points` stays consistent."""
        self._last_vector = vector


# ==========================================================================
# Spatially uniform estimators
# ==========================================================================


class DisabledFlow(FlowEstimator):
    """No correction: reports a measured, exactly-zero flow.

    For controlled still-fluid test recordings only. It returns ``(0.0, 0.0)``
    rather than ``None`` because in that setting zero *is* the measurement, and
    downstream code is entitled to treat the kinematics as corrected.
    """

    mode = FlowCorrectionMode.DISABLED
    name = "disabled_flow"

    def estimate(
        self, tracks: Sequence[TrackRecord], frame: FramePacket | None = None
    ) -> tuple[float, float]:
        self._remember((0.0, 0.0))
        return (0.0, 0.0)


class FixedVectorFlow(FlowEstimator):
    """One calibrated bulk-flow vector, subtracted everywhere.

    Raises :class:`~sperm_sorting.errors.CalibrationError` when constructed
    without a measured vector: an uncalibrated "fixed" flow of zero would be
    indistinguishable from a real still-fluid measurement, which is exactly the
    confusion this system is not allowed to create.
    """

    mode = FlowCorrectionMode.FIXED_VECTOR
    name = "fixed_vector_flow"

    def __init__(self, vx_px_s: float | None, vy_px_s: float | None) -> None:
        super().__init__()
        if vx_px_s is None or vy_px_s is None:
            raise CalibrationError(
                "fixed-vector flow correction requires a measured bulk flow: "
                "flow_correction.fixed_vx_px_s / fixed_vy_px_s are unset. "
                "Measure them (e.g. from tracked debris in still-sample "
                "footage), or select a different flow_correction.mode."
            )
        if not (math.isfinite(vx_px_s) and math.isfinite(vy_px_s)):
            raise CalibrationError(
                f"fixed-vector flow must be finite, got ({vx_px_s}, {vy_px_s})"
            )
        self.vx_px_s = float(vx_px_s)
        self.vy_px_s = float(vy_px_s)
        self._last_vector = (self.vx_px_s, self.vy_px_s)

    def estimate(
        self, tracks: Sequence[TrackRecord], frame: FramePacket | None = None
    ) -> tuple[float, float]:
        return (self.vx_px_s, self.vy_px_s)

    def reset(self) -> None:
        # The calibrated vector is not session state; keep it.
        self._last_vector = (self.vx_px_s, self.vy_px_s)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"vx_px_s": self.vx_px_s, "vy_px_s": self.vy_px_s})
        return out


# ==========================================================================
# Position-dependent estimator
# ==========================================================================


class FlowMapFlow(FlowEstimator):
    """Position-dependent flow sampled from a calibrated ``(H, W, 2)`` field.

    Pressure-driven flow in a microchannel is parabolic across the channel
    (Poiseuille): fastest mid-channel, zero at the walls. A single vector
    therefore over-corrects cells near the walls and under-corrects cells in
    the centre, and both errors land directly on VSL, which is what the
    progressive grade is read from. A calibrated map removes that bias.

    The map holds ``[..., 0] = vx`` and ``[..., 1] = vy`` in pixels/second,
    indexed ``[row, col] = [y, x]`` to match image convention.
    """

    mode = FlowCorrectionMode.FLOW_MAP
    name = "flow_map_flow"

    def __init__(
        self,
        flow_map: ArrayLike | None = None,
        path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if flow_map is None and path is None:
            raise CalibrationError(
                "flow-map correction requires either an in-memory field or "
                "flow_correction.flow_map_path pointing at a saved .npy"
            )
        if flow_map is None:
            flow_map = self._load(Path(str(path)))
        field = np.asarray(flow_map, dtype=np.float64)
        if field.ndim != 3 or field.shape[2] != 2:
            raise CalibrationError(
                f"flow map must have shape (H, W, 2), got {field.shape}"
            )
        if field.shape[0] < 1 or field.shape[1] < 1:
            raise CalibrationError(f"flow map is empty: shape {field.shape}")
        if not np.all(np.isfinite(field)):
            raise CalibrationError(
                "flow map contains non-finite values; a NaN in the field would "
                "propagate silently into every corrected position"
            )
        self.field = field
        self.source_path = Path(str(path)) if path is not None else None
        self._last_vector = self._global_mean()

    @staticmethod
    def _load(path: Path) -> NDArray[np.float64]:
        if not path.exists():
            raise CalibrationError(f"flow map file not found: {path}")
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:  # malformed or non-array .npy
            raise CalibrationError(f"could not read flow map {path}: {exc}") from exc
        return np.asarray(data, dtype=np.float64)

    @property
    def height(self) -> int:
        return int(self.field.shape[0])

    @property
    def width(self) -> int:
        return int(self.field.shape[1])

    def _global_mean(self) -> tuple[float, float]:
        mean = self.field.reshape(-1, 2).mean(axis=0)
        return (float(mean[0]), float(mean[1]))

    def sample_points(self, points_xy: ArrayLike) -> NDArray[np.float64]:
        """Bilinearly sample the field at each ``(x, y)``.

        Positions outside the calibrated field are clamped to its border rather
        than extrapolated or zeroed: the nearest calibrated value is the best
        available estimate, whereas zero would silently disable the correction
        for any track that strayed a pixel outside the ROI.
        """
        pts = as_points_array(points_xy)
        if pts.shape[0] == 0:
            return pts.copy()

        h, w = self.height, self.width
        x = np.clip(pts[:, 0], 0.0, float(w - 1))
        y = np.clip(pts[:, 1], 0.0, float(h - 1))
        x0 = np.floor(x).astype(np.intp)
        y0 = np.floor(y).astype(np.intp)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]

        top = self.field[y0, x0] * (1.0 - fx) + self.field[y0, x1] * fx
        bottom = self.field[y1, x0] * (1.0 - fx) + self.field[y1, x1] * fx
        return np.ascontiguousarray(top * (1.0 - fy) + bottom * fy)

    def estimate(
        self, tracks: Sequence[TrackRecord], frame: FramePacket | None = None
    ) -> tuple[float, float]:
        """Representative single vector: the field at the current track heads.

        This scalar summary exists for the audit record and for callers that
        want one number; the real correction goes through
        :meth:`sample_points`, which is position-dependent. With no tracks to
        sample at, the field's global mean is reported.
        """
        heads = [(t.points[-1].x, t.points[-1].y) for t in tracks if t.points]
        if not heads:
            vector = self._global_mean()
        else:
            mean = self.sample_points(np.asarray(heads, dtype=np.float64)).mean(axis=0)
            vector = (float(mean[0]), float(mean[1]))
        self._remember(vector)
        return vector

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "shape": [self.height, self.width],
                "path": str(self.source_path) if self.source_path else None,
                "mean_vx_px_s": self._global_mean()[0],
                "mean_vy_px_s": self._global_mean()[1],
            }
        )
        return out


# ==========================================================================
# Population-based estimator
# ==========================================================================


class RobustFlowEstimator(FlowEstimator):
    """Per-frame bulk flow measured from the slowest fraction of the tracks.

    Debris and immotile cells are pure flow tracers: whatever they do, the
    fluid did. They cannot be identified directly, so the slowest
    ``quantile`` fraction of tracks is used as a proxy for them.

    Three deliberate choices:

    * **median, not mean.** The slow subset will occasionally contain a
      genuinely swimming cell that happened to be heading upstream, or a track
      whose association flickered. A mean lets one such outlier move the
      estimate; the component-wise median does not.
    * **hard minimum track count.** Below ``min_tracks`` the quantile is being
      taken over a handful of samples and is not robust in any useful sense, so
      the estimator returns ``None`` and lets the caller refuse to classify,
      rather than emitting a confident-looking guess.
    * **exponential smoothing across frames.** Bulk flow is a property of the
      pump and the channel; it changes on the timescale of seconds, while the
      per-frame estimate is noisy at the timescale of milliseconds. Smoothing
      with ``smoothing`` as the weight of the new sample (0.15 by default, i.e.
      an effective memory of roughly seven frames) suppresses that noise
      without lagging a real change in the flow.
    """

    mode = FlowCorrectionMode.ROBUST_ESTIMATE
    name = "robust_flow_estimator"

    def __init__(
        self,
        quantile: float = 0.25,
        min_tracks: int = 8,
        smoothing: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"robust_quantile must lie in (0, 1], got {quantile}")
        if min_tracks < 1:
            raise ValueError(f"robust_min_tracks must be >= 1, got {min_tracks}")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"robust_smoothing must lie in (0, 1], got {smoothing}")
        self.quantile = float(quantile)
        self.min_tracks = int(min_tracks)
        self.smoothing = float(smoothing)
        self._state: tuple[float, float] | None = None

    @staticmethod
    def _instantaneous_velocity(
        track: TrackRecord, frame: FramePacket | None
    ) -> tuple[float, float] | None:
        """Velocity from the track's last two *observed* points, or ``None``.

        Predicted points are the motion model's opinion, not a measurement of
        the fluid, so they are excluded. Two-point differencing is noisy, but
        the noise is zero-mean across the population and is removed by the
        median and the temporal smoothing; averaging over a longer window here
        would instead lag a changing flow.
        """
        observed = track.observed_points
        if len(observed) < 2:
            return None
        last, prev = observed[-1], observed[-2]
        if frame is not None and last.frame_id != frame.frame_id:
            # Not measured on this frame: it says nothing about the flow *now*.
            return None
        dt = last.capture_time_s - prev.capture_time_s
        if dt <= EPS:
            return None
        return ((last.x - prev.x) / dt, (last.y - prev.y) / dt)

    def estimate(
        self, tracks: Sequence[TrackRecord], frame: FramePacket | None = None
    ) -> tuple[float, float] | None:
        velocities = [
            v
            for v in (self._instantaneous_velocity(t, frame) for t in tracks)
            if v is not None
        ]
        if len(velocities) < self.min_tracks:
            # Not enough tracers. Report unknown; do not disturb the state.
            self._remember(None)
            return None

        vel = np.asarray(velocities, dtype=np.float64)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        k = max(1, math.ceil(self.quantile * vel.shape[0]))
        # Stable sort so the same input always yields the same subset.
        slowest = np.argsort(speed, kind="stable")[:k]
        median = np.median(vel[slowest], axis=0)
        sample = (float(median[0]), float(median[1]))

        if self._state is None:
            # Seed with the first real measurement. Smoothing from an assumed
            # zero would bias the first several frames toward no correction.
            self._state = sample
        else:
            alpha = self.smoothing
            self._state = (
                alpha * sample[0] + (1.0 - alpha) * self._state[0],
                alpha * sample[1] + (1.0 - alpha) * self._state[1],
            )
        self._remember(self._state)
        return self._state

    def reset(self) -> None:
        super().reset()
        self._state = None

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "quantile": self.quantile,
                "min_tracks": self.min_tracks,
                "smoothing": self.smoothing,
            }
        )
        return out


# ==========================================================================
# Factory
# ==========================================================================


def build_flow_estimator(cfg: FlowCorrectionConfig) -> FlowEstimator:
    """Construct the estimator named by ``cfg.mode``.

    Configuration validation already refuses ``fixed_vector`` without a vector
    and ``flow_map`` without a path, but the estimators re-check, because they
    are public API and may be constructed directly.
    """
    if cfg.mode is FlowCorrectionMode.DISABLED:
        return DisabledFlow()
    if cfg.mode is FlowCorrectionMode.FIXED_VECTOR:
        return FixedVectorFlow(cfg.fixed_vx_px_s, cfg.fixed_vy_px_s)
    if cfg.mode is FlowCorrectionMode.FLOW_MAP:
        return FlowMapFlow(path=cfg.flow_map_path)
    if cfg.mode is FlowCorrectionMode.ROBUST_ESTIMATE:
        return RobustFlowEstimator(
            quantile=cfg.robust_quantile,
            min_tracks=cfg.robust_min_tracks,
            smoothing=cfg.robust_smoothing,
        )
    raise ValueError(f"unhandled flow correction mode: {cfg.mode!r}")


# ==========================================================================
# Correction
# ==========================================================================


def apply_flow_correction(
    points_xy: ArrayLike,
    times: ArrayLike,
    flow_vx: float | ArrayLike,
    flow_vy: float | ArrayLike,
) -> NDArray[np.float64]:
    """Remove bulk transport from a position series.

    Parameters
    ----------
    points_xy
        ``(N, 2)`` observed positions in pixels.
    times
        ``(N,)`` capture times in seconds, non-decreasing. Real timestamps --
        never ``frame_index / nominal_fps``.
    flow_vx, flow_vy
        Flow velocity in px/s. Either a scalar (spatially uniform flow) or an
        ``(N,)`` array giving the flow sampled at each point (flow map).

    Returns
    -------
    ndarray
        ``(N, 2)`` corrected positions, anchored so that the first point is
        unchanged. Only *relative* motion is physical here -- the absolute
        origin of the corrected frame is arbitrary -- and anchoring at the
        first point keeps the corrected track near the observed one, which
        makes debugging overlays legible.

    Notes
    -----
    The subtracted quantity is the cumulative displacement the flow would have
    produced by each timestamp,

    .. math:: \\Delta(t_i) = \\int_{t_0}^{t_i} v_{flow}(t)\\,dt

    evaluated by the trapezoidal rule over the actual (possibly irregular)
    timestamps. For a constant flow this reduces exactly to
    :math:`v \\cdot (t_i - t_0)`.

    Subtracting a velocity from a position instead would be dimensionally
    wrong, and -- worse than merely wrong -- would produce a plausible-looking
    small offset that leaves most of the drift in place.
    """
    pts = as_points_array(points_xy)
    n = pts.shape[0]
    t = np.asarray(times, dtype=np.float64).ravel()
    if t.shape[0] != n:
        raise ValueError(
            f"times has {t.shape[0]} entries but the trajectory has {n} points"
        )
    if n == 0:
        return pts.copy()

    vx = _as_per_point(flow_vx, n, "flow_vx")
    vy = _as_per_point(flow_vy, n, "flow_vy")

    corrected = pts.copy()
    if n == 1:
        return corrected

    dt = np.diff(t)
    if np.any(dt < 0.0):
        raise ValueError(
            "times must be non-decreasing; unsorted timestamps would integrate "
            "the flow backwards for part of the track"
        )

    # Trapezoidal cumulative integral, prefixed with 0 for the anchor point.
    drift_x = np.concatenate(([0.0], np.cumsum(0.5 * (vx[:-1] + vx[1:]) * dt)))
    drift_y = np.concatenate(([0.0], np.cumsum(0.5 * (vy[:-1] + vy[1:]) * dt)))
    corrected[:, 0] -= drift_x
    corrected[:, 1] -= drift_y
    return corrected


def _as_per_point(value: float | ArrayLike, n: int, label: str) -> NDArray[np.float64]:
    """Broadcast a scalar or validate an ``(N,)`` flow component."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    arr = arr.ravel()
    if arr.shape[0] != n:
        raise ValueError(
            f"{label} must be a scalar or have {n} entries, got {arr.shape[0]}"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)
