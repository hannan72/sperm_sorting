"""Constant-velocity Kalman filter for boxes, in pure numpy.

The state is the eight-vector

``[cx, cy, a, h, vcx, vcy, va, vh]``

-- centre, aspect ratio ``a = w / h``, height, and the time derivative of each
-- and the measurement is the first four components. This is the SORT /
DeepSORT / ByteTrack parameterisation, kept here so that a reader who knows
those papers recognises the filter immediately.

Two properties of this parameterisation matter for sperm:

* **Aspect ratio, not width.** A sperm head is small and nearly isotropic, so
  its measured width is noisy; making the noisy quantity a *ratio* keeps that
  noise out of the position and height channels, where it would leak into the
  velocity estimate.
* **Noise proportional to height.** Every standard deviation below is scaled
  by the current height, so the filter behaves identically for a head six
  pixels across and one sixteen pixels across. ``TrackingConfig`` therefore
  carries a single dimensionless pair of scales rather than pixel variances.

The time step is one *frame*, not one second: the filter never sees a clock.
Real timing lives on :class:`~sperm_sorting.schemas.track.TrackPoint`
(``capture_time_s``), and physical velocity is computed downstream from those
timestamps. Keeping seconds out of the filter means a dropped frame cannot
silently rescale a velocity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import scipy.linalg

from ..schemas.detection import BoundingBox

#: Dimension of the state vector ``[cx, cy, a, h, vcx, vcy, va, vh]``.
STATE_DIM: Final[int] = 8
#: Dimension of the measurement vector ``[cx, cy, a, h]``.
MEASUREMENT_DIM: Final[int] = 4

#: Smallest height / width the filter will report. Prevents a degenerate box
#: (which :class:`BoundingBox` would accept) and division by zero in ``a``.
_MIN_EXTENT: Final[float] = 1e-3

#: Fixed standard deviations for the aspect-ratio channel. Aspect ratio is
#: dimensionless, so unlike every other channel it is *not* scaled by height.
_STD_ASPECT_POSITION: Final[float] = 1e-2
_STD_ASPECT_VELOCITY: Final[float] = 1e-5
_STD_ASPECT_MEASUREMENT: Final[float] = 1e-1

#: 0.95 quantiles of the chi-square distribution with N degrees of freedom,
#: for gating on the squared Mahalanobis distances returned by
#: :meth:`KalmanBoxTracker.mahalanobis_distance`.
CHI2_INV95: Final[dict[int, float]] = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


def box_to_measurement(box: BoundingBox) -> np.ndarray:
    """Convert a box to the filter's measurement space ``[cx, cy, a, h]``."""
    height = max(float(box.height), _MIN_EXTENT)
    width = max(float(box.width), _MIN_EXTENT)
    return np.array([box.cx, box.cy, width / height, height], dtype=np.float64)


def boxes_to_measurements(
    boxes: Sequence[BoundingBox] | np.ndarray,
) -> np.ndarray:
    """Convert many boxes to an ``(N, 4)`` measurement array.

    Accepts either a sequence of :class:`BoundingBox` or an ``(N, 4+)`` array
    of ``[x1, y1, x2, y2, ...]`` rows, so callers that already hold the array
    form produced by ``detections_to_array`` need not rebuild objects.
    """
    if isinstance(boxes, np.ndarray):
        array = np.asarray(boxes, dtype=np.float64).reshape(-1, boxes.shape[-1])
        if array.shape[0] == 0:
            return np.zeros((0, MEASUREMENT_DIM), dtype=np.float64)
        widths = np.maximum(array[:, 2] - array[:, 0], _MIN_EXTENT)
        heights = np.maximum(array[:, 3] - array[:, 1], _MIN_EXTENT)
        return np.stack(
            [
                0.5 * (array[:, 0] + array[:, 2]),
                0.5 * (array[:, 1] + array[:, 3]),
                widths / heights,
                heights,
            ],
            axis=1,
        )
    if len(boxes) == 0:
        return np.zeros((0, MEASUREMENT_DIM), dtype=np.float64)
    return np.stack([box_to_measurement(b) for b in boxes], axis=0)


def measurement_to_box(measurement: np.ndarray) -> BoundingBox:
    """Inverse of :func:`box_to_measurement`, clamped to a non-degenerate box."""
    cx, cy, aspect, height = (float(v) for v in measurement[:MEASUREMENT_DIM])
    height = max(height, _MIN_EXTENT)
    width = max(aspect * height, _MIN_EXTENT)
    return BoundingBox(
        cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height
    )


def virtual_trajectory(
    start: BoundingBox, end: BoundingBox, steps: int
) -> list[BoundingBox]:
    """Linearly interpolate ``steps`` boxes from just after ``start`` to ``end``.

    The returned list has length ``steps``; its last element equals ``end``.
    Used by OC-SORT's observation-centric re-update, which needs a plausible
    path between two real observations separated by a gap of missed frames.
    """
    if steps < 1:
        return [end]
    a = np.array(start.as_xyxy(), dtype=np.float64)
    b = np.array(end.as_xyxy(), dtype=np.float64)
    out: list[BoundingBox] = []
    for i in range(1, steps + 1):
        point = a + (b - a) * (i / steps)
        out.append(BoundingBox(*(float(v) for v in point)))
    return out


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over one box.

    The object owns *only* the filter. Track lifecycle (hits, age, state, the
    growing :class:`~sperm_sorting.schemas.track.TrackRecord`) belongs to the
    tracker, so that the same filter serves all three association algorithms.
    """

    __slots__ = (
        "_covariance",
        "_mean",
        "_motion_mat",
        "_std_position",
        "_std_velocity",
    )

    def __init__(
        self,
        box: BoundingBox,
        *,
        std_position: float = 0.05,
        std_velocity: float = 0.008,
        dt: float = 1.0,
    ) -> None:
        self._std_position = float(std_position)
        self._std_velocity = float(std_velocity)

        self._motion_mat = np.eye(STATE_DIM, dtype=np.float64)
        for i in range(MEASUREMENT_DIM):
            self._motion_mat[i, MEASUREMENT_DIM + i] = float(dt)

        measurement = box_to_measurement(box)
        self._mean = np.zeros(STATE_DIM, dtype=np.float64)
        self._mean[:MEASUREMENT_DIM] = measurement

        height = max(float(measurement[3]), _MIN_EXTENT)
        std = np.array(
            [
                2.0 * self._std_position * height,
                2.0 * self._std_position * height,
                _STD_ASPECT_POSITION,
                2.0 * self._std_position * height,
                10.0 * self._std_velocity * height,
                10.0 * self._std_velocity * height,
                _STD_ASPECT_VELOCITY,
                10.0 * self._std_velocity * height,
            ],
            dtype=np.float64,
        )
        self._covariance = np.diag(np.square(std))

    # ------------------------------------------------------------------ state

    @property
    def mean(self) -> np.ndarray:
        """Current state estimate. A copy; the filter owns the original."""
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance. A copy."""
        return self._covariance.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Centre velocity ``(vcx, vcy)`` in pixels per frame."""
        return self._mean[4:6].copy()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a restorable copy of the filter state."""
        return (self._mean.copy(), self._covariance.copy())

    def restore(self, state: tuple[np.ndarray, np.ndarray]) -> None:
        """Restore a state previously returned by :meth:`snapshot`."""
        mean, covariance = state
        self._mean = np.asarray(mean, dtype=np.float64).copy()
        self._covariance = np.asarray(covariance, dtype=np.float64).copy()

    # ----------------------------------------------------------------- filter

    def _process_noise_diagonal(self) -> np.ndarray:
        """Diagonal of Q. Returned as a vector: Q is diagonal by construction,
        and materialising the 8x8 would allocate on every predict."""
        height = max(float(self._mean[3]), _MIN_EXTENT)
        std = np.array(
            [
                self._std_position * height,
                self._std_position * height,
                _STD_ASPECT_POSITION,
                self._std_position * height,
                self._std_velocity * height,
                self._std_velocity * height,
                _STD_ASPECT_VELOCITY,
                self._std_velocity * height,
            ],
            dtype=np.float64,
        )
        return np.square(std)

    def _measurement_noise_diagonal(self) -> np.ndarray:
        """Diagonal of R, as a vector. See :meth:`_process_noise_diagonal`."""
        height = max(float(self._mean[3]), _MIN_EXTENT)
        std = np.array(
            [
                self._std_position * height,
                self._std_position * height,
                _STD_ASPECT_MEASUREMENT,
                self._std_position * height,
            ],
            dtype=np.float64,
        )
        return np.square(std)

    def predict(self) -> BoundingBox:
        """Advance the state one frame and return the predicted box."""
        self._mean = self._motion_mat @ self._mean
        self._covariance = self._motion_mat @ self._covariance @ self._motion_mat.T
        self._covariance.flat[:: STATE_DIM + 1] += self._process_noise_diagonal()
        self._mean[3] = max(float(self._mean[3]), _MIN_EXTENT)
        return self.to_box()

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        """Project the state into measurement space: ``(mean, covariance)``.

        The measurement matrix is ``H = [I_4 | 0]``, so ``H x`` is a slice of
        the state and ``H P H^T`` is the leading 4x4 block of the covariance.
        Written as slices rather than matmuls because this runs at least twice
        per track per frame, and at 160 FPS that is the difference between a
        fraction of the latency budget and a meaningful share of it.
        """
        mean = self._mean[:MEASUREMENT_DIM].copy()
        covariance = np.array(
            self._covariance[:MEASUREMENT_DIM, :MEASUREMENT_DIM], copy=True
        )
        covariance.flat[:: MEASUREMENT_DIM + 1] += self._measurement_noise_diagonal()
        return mean, covariance

    def update(self, box: BoundingBox) -> BoundingBox:
        """Correct the state with a measured box and return the posterior box."""
        measurement = box_to_measurement(box)
        projected_mean, projected_cov = self.project()

        # Kalman gain K = P H^T S^-1, computed as a solve rather than an
        # inverse. S is a symmetric 4x4; a direct solve matches a Cholesky
        # factorisation to ~1e-16 here and costs a third as much, because at
        # this size the work is dominated by call overhead, not arithmetic.
        cross_covariance = self._covariance[:, :MEASUREMENT_DIM]  # P H^T
        kalman_gain = np.linalg.solve(projected_cov, cross_covariance.T).T
        innovation = measurement - projected_mean

        self._mean = self._mean + kalman_gain @ innovation
        self._covariance = self._covariance - kalman_gain @ projected_cov @ kalman_gain.T
        # Symmetrise: repeated rank-4 downdates accumulate asymmetry, and an
        # asymmetric covariance eventually breaks the Cholesky factorisation
        # used by :meth:`mahalanobis_distance`.
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._mean[3] = max(float(self._mean[3]), _MIN_EXTENT)
        return self.to_box()

    # ------------------------------------------------------------------ query

    def to_box(self) -> BoundingBox:
        """Current estimate as a :class:`BoundingBox`."""
        return measurement_to_box(self._mean)

    def mahalanobis_distance(
        self,
        boxes: Sequence[BoundingBox] | np.ndarray,
        *,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from this state to each box.

        Compare against :data:`CHI2_INV95` with 2 degrees of freedom when
        ``only_position`` is set and 4 otherwise. Gating on this rather than on
        IoU alone is what stops a fast sperm from being stolen by a stationary
        piece of debris that happens to overlap its predicted box.
        """
        measurements = boxes_to_measurements(boxes)
        if measurements.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        mean, covariance = self.project()
        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky = np.linalg.cholesky(covariance)
        delta = (measurements - mean).T
        solved = scipy.linalg.solve_triangular(
            cholesky, delta, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(solved * solved, axis=0)

    # ------------------------------------------------- external state warping

    def apply_affine(self, warp: np.ndarray) -> None:
        """Warp the state by a 2x3 similarity transform (camera-motion comp.).

        The transform is decomposed rather than applied blindly: the rotation
        part acts on the centre and the centre velocity, the isotropic scale
        acts on height and height rate, and the aspect ratio -- invariant under
        a similarity -- is left alone. Applying the raw 2x2 block to the
        ``(a, h)`` pair, as some reference implementations do, would rotate a
        ratio into a length and is only harmless because the rotation is
        usually tiny.
        """
        warp = np.asarray(warp, dtype=np.float64)
        if warp.shape != (2, 3):
            raise ValueError(f"expected a 2x3 affine warp, got shape {warp.shape}")
        rotation = warp[:2, :2]
        translation = warp[:2, 2]
        scale = float(np.sqrt(abs(np.linalg.det(rotation))))

        transform = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        transform[0:2, 0:2] = rotation
        transform[2, 2] = 1.0
        transform[3, 3] = scale
        transform[4:6, 4:6] = rotation
        transform[6, 6] = 1.0
        transform[7, 7] = scale

        self._mean = transform @ self._mean
        self._mean[0:2] += translation
        self._covariance = transform @ self._covariance @ transform.T
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._mean[3] = max(float(self._mean[3]), _MIN_EXTENT)


class OCRKalmanBoxTracker(KalmanBoxTracker):
    """Kalman filter with OC-SORT's observation-centric re-update.

    A plain Kalman filter that is fed nothing but its own predictions for many
    frames drifts, and -- worse for this product -- its *velocity* channel
    inflates, because each prediction step adds process noise without any
    measurement to pull it back. When the object is finally re-detected, a
    normal update blends the new measurement into that corrupted state and the
    track carries the error forward for many more frames.

    OC-SORT's answer is to throw the corrupted stretch away. On
    re-association, :meth:`observation_centric_reupdate` rewinds the filter to
    the state it had at the last *real* observation and re-runs it along the
    straight line between that observation and the new one. The result depends
    only on measurements, never on the filter's own guesses.
    """

    __slots__ = ("_frozen",)

    def __init__(
        self,
        box: BoundingBox,
        *,
        std_position: float = 0.05,
        std_velocity: float = 0.008,
        dt: float = 1.0,
    ) -> None:
        super().__init__(box, std_position=std_position, std_velocity=std_velocity, dt=dt)
        self._frozen: tuple[np.ndarray, np.ndarray] = self.snapshot()

    def update(self, box: BoundingBox) -> BoundingBox:
        result = super().update(box)
        self.freeze()
        return result

    def freeze(self) -> None:
        """Remember the current state as the last measurement-backed one."""
        self._frozen = self.snapshot()

    def observation_centric_reupdate(
        self, last_observation: BoundingBox, new_observation: BoundingBox, gap: int
    ) -> BoundingBox:
        """Rewind to the last observation and re-run along the virtual path.

        ``gap`` is the number of frames between the two observations, so
        ``gap == 1`` means they are consecutive and this degenerates to an
        ordinary update.
        """
        steps = max(int(gap), 1)
        self.restore(self._frozen)
        path = virtual_trajectory(last_observation, new_observation, steps)
        box = self.to_box()
        for virtual_box in path:
            super().predict()
            box = super().update(virtual_box)
        self.freeze()
        return box
