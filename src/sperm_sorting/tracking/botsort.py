"""BoT-SORT: ByteTrack plus camera-motion compensation and appearance.

BoT-SORT keeps ByteTrack's two-band association and adds two things: a global
motion estimate that warps track states into the current frame's coordinates,
and an appearance branch fused with IoU. Both are off by default here, and
each is off for its own reason.

Camera-motion compensation: off, deliberately
---------------------------------------------
``botsort_use_cmc`` defaults to ``False`` **because the camera is rigidly
mounted**. In a hand-held or vehicle-mounted scene, global image motion is the
camera moving and removing it is pure gain. On this instrument the camera
cannot move, so whatever global image motion exists is *the fluid moving* --
bulk transport of the sample through the field of view.

That distinction is the whole product. This system measures how a sperm swims,
which means separating its own propulsion from the flow carrying it. Flow is
handled explicitly and visibly downstream by
:class:`~sperm_sorting.config.FlowCorrectionConfig`, which estimates the bulk
vector, subtracts it, and *records what it subtracted* in
``MotionFeatures.flow_vx_px_s`` so a reviewer can see how large the correction
was. Enabling CMC would have the tracker quietly absorb that same motion into
its state transitions first: trajectories would come out pre-corrected by an
unrecorded amount, the downstream estimator would then find little flow left
to remove and subtract a second, wrong correction, and the resulting velocity
would be neither raw nor corrected -- it would be uninterpretable. Progressive
motility is a velocity threshold, so an uninterpretable velocity is an
uninterpretable FIELD_ON/FIELD_OFF decision.

Turn CMC on only if the optical path is ever changed to one where the sensor
really can move relative to the sample -- a scanning stage, say -- and then say
so in the audit log, because it changes what the reported velocities mean.

Appearance / ReID: off, because there is no model
-------------------------------------------------
``botsort_use_reid`` defaults to ``False`` and this repository ships no
embedding model. The fusion is implemented in full anyway, against the
:class:`ReIDEmbedder` protocol, and the no-embedder path is *exact* rather
than approximate: BoT-SORT's rule is ``C = min(d_iou, d_hat_cos)`` where
``d_hat_cos`` is forced to 1 whenever appearance is not trusted. With no
embeddings every entry of ``d_hat_cos`` is 1, and since ``d_iou`` never
exceeds 1, the minimum is ``d_iou`` identically. Disabling ReID therefore
reduces to plain ByteTrack by construction, not by a special case.

Whether appearance can help at all here is an open question worth stating: a
sperm head is a few dozen pixels of low-contrast monochrome, and two of them
look far more alike than two pedestrians do. The plumbing exists so the
question can be answered with a real embedder rather than argued about.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from ..config import TrackingConfig
from ..schemas.detection import BoundingBox, Detection
from ..schemas.frame import FramePacket
from ..schemas.track import TrackRecord
from ._common import ManagedTrack
from .assignment import cosine_distance, iou_distance
from .bytetrack import ByteTracker

#: Key under which a detection may carry a precomputed ReID embedding in its
#: ``meta``. Lets a detector that already ran a backbone hand the feature over
#: instead of paying for a second forward pass.
REID_FEATURE_KEY = "reid_feature"

#: IoU distance above which appearance is not trusted at all. Two boxes that
#: barely overlap should not be joined on looks alone.
DEFAULT_PROXIMITY_THRESHOLD: float = 0.5
#: Cosine distance above which an appearance match is rejected.
DEFAULT_APPEARANCE_THRESHOLD: float = 0.25
#: Momentum of the per-track exponential moving average of features.
DEFAULT_FEATURE_MOMENTUM: float = 0.9


@runtime_checkable
class ReIDEmbedder(Protocol):
    """Appearance embedding model.

    Implementations return one L2-normalised row per box. They are called once
    per frame with every box that needs a feature, never once per box, because
    at 160 FPS the per-call overhead of a batched model dominates.
    """

    @property
    def embedding_dim(self) -> int:
        """Length of one embedding row."""

    def embed(self, image: np.ndarray, boxes: Sequence[BoundingBox]) -> np.ndarray:
        """Return an ``(len(boxes), embedding_dim)`` array of unit-norm rows."""


def fuse_iou_reid(
    iou_dist: np.ndarray,
    embedding_dist: np.ndarray | None = None,
    *,
    proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD,
    appearance_threshold: float = DEFAULT_APPEARANCE_THRESHOLD,
) -> np.ndarray:
    """BoT-SORT's cost fusion: ``min(d_iou, d_hat_cos)``.

    ``d_hat_cos`` is the appearance distance, forced to 1 wherever appearance
    is not trustworthy -- either because the look is too different
    (``> appearance_threshold``) or because the boxes are too far apart for
    appearance to be evidence of anything (``d_iou > proximity_threshold``).

    With ``embedding_dist=None`` the result equals ``iou_dist`` **exactly**:
    every ``d_hat_cos`` is 1 and ``d_iou`` is bounded by 1, so the elementwise
    minimum is ``d_iou``. That is the algorithm's own limit, not a stub.

    The return value is always a fresh array, never an alias of an argument,
    so a caller may gate or mask it in place.
    """
    iou = np.asarray(iou_dist, dtype=np.float64)
    if embedding_dist is None:
        return np.array(iou, dtype=np.float64, copy=True)
    fused = np.array(embedding_dist, dtype=np.float64, copy=True)
    fused[~np.isfinite(fused)] = 1.0
    fused[fused > appearance_threshold] = 1.0
    fused[iou > proximity_threshold] = 1.0
    return np.minimum(iou, fused)


class CameraMotionCompensator:
    """Estimates the frame-to-frame global affine warp.

    Two estimators, both from OpenCV:

    * ``"sparse_optical_flow"`` -- corners tracked with Lucas-Kanade, then a
      partial-affine (similarity) fit under RANSAC. Fast and robust to a few
      independently-moving objects, which is why BoT-SORT prefers it.
    * ``"ecc"`` -- direct intensity alignment. More accurate on low-texture
      images, considerably slower, and it converges poorly when the moving
      objects are a large fraction of the image.

    The returned warp maps *previous*-frame coordinates to *current*-frame
    coordinates, which is the direction a track state needs in order to be
    carried forward. ``None`` means "no usable estimate" -- first frame,
    too few tracked corners, or a failed fit -- and callers must treat that as
    identity rather than guessing.
    """

    def __init__(
        self,
        *,
        method: str = "sparse_optical_flow",
        downscale: int = 2,
        max_corners: int = 1000,
        quality_level: float = 0.01,
        min_distance: float = 1.0,
        ecc_iterations: int = 100,
        ecc_epsilon: float = 1e-5,
    ) -> None:
        if method not in ("sparse_optical_flow", "ecc"):
            raise ValueError(
                f"unknown camera-motion method {method!r}; expected "
                "'sparse_optical_flow' or 'ecc'"
            )
        self.method = method
        self.downscale = max(int(downscale), 1)
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = float(min_distance)
        self.ecc_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(ecc_iterations),
            float(ecc_epsilon),
        )
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the previous frame. The next call returns ``None``."""
        self._previous = None

    def estimate(self, image: np.ndarray) -> np.ndarray | None:
        """Return a 2x3 warp from the previous frame to this one, or ``None``."""
        current = self._prepare(image)
        previous, self._previous = self._previous, current
        if previous is None or previous.shape != current.shape:
            return None

        warp = (
            self._estimate_sparse(previous, current)
            if self.method == "sparse_optical_flow"
            else self._estimate_ecc(previous, current)
        )
        if warp is None:
            return None
        warp = np.asarray(warp, dtype=np.float64).reshape(2, 3)
        # Translation was measured in downscaled pixels; the rotation and
        # scale parts are dimensionless and need no correction.
        warp[:, 2] *= float(self.downscale)
        if not np.isfinite(warp).all():
            return None
        return warp

    # ------------------------------------------------------------- internals

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.dtype == np.uint16:
            frame = (frame >> 8).astype(np.uint8)
        elif frame.dtype != np.uint8:
            finite = frame[np.isfinite(frame)] if frame.size else frame
            peak = float(finite.max()) if finite.size else 0.0
            scale = 255.0 / peak if peak > 0.0 else 1.0
            frame = np.clip(frame.astype(np.float64) * scale, 0, 255).astype(np.uint8)
        if self.downscale > 1:
            frame = cv2.resize(
                frame,
                (
                    max(frame.shape[1] // self.downscale, 1),
                    max(frame.shape[0] // self.downscale, 1),
                ),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(frame)

    def _estimate_sparse(
        self, previous: np.ndarray, current: np.ndarray
    ) -> np.ndarray | None:
        corners = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=3,
        )
        if corners is None or len(corners) < 4:
            return None
        source_points = corners.astype(np.float32)
        # An explicit output buffer rather than ``None``: OpenCV treats
        # ``nextPts`` as an InputOutputArray and ignores its contents unless
        # OPTFLOW_USE_INITIAL_FLOW is set, which it is not.
        target_points = np.zeros_like(source_points)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, source_points, target_points
        )
        if tracked is None or status is None:
            return None
        keep = status.reshape(-1).astype(bool)
        source = corners.reshape(-1, 2)[keep]
        target = tracked.reshape(-1, 2)[keep]
        if len(source) < 4:
            return None
        warp, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
        return warp

    def _estimate_ecc(
        self, previous: np.ndarray, current: np.ndarray
    ) -> np.ndarray | None:
        initial = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                previous, current, initial, cv2.MOTION_EUCLIDEAN, self.ecc_criteria
            )
        except cv2.error:
            # ECC raises rather than returning a failure code when it cannot
            # converge, which on a near-featureless microscopy frame is a
            # normal outcome, not an error worth propagating.
            return None
        return warp


class BoTSortTrack(ManagedTrack):
    """A ByteTrack track that also carries a smoothed appearance embedding."""

    __slots__ = ("feature_momentum", "smooth_feature")

    def __init__(
        self,
        record: TrackRecord,
        detection: Detection,
        frame: FramePacket,
        *,
        std_position: float,
        std_velocity: float,
    ) -> None:
        self.smooth_feature: np.ndarray | None = None
        self.feature_momentum: float = DEFAULT_FEATURE_MOMENTUM
        super().__init__(
            record,
            detection,
            frame,
            std_position=std_position,
            std_velocity=std_velocity,
        )
        self.absorb_feature(detection.meta.get(REID_FEATURE_KEY))

    def absorb_feature(self, feature: np.ndarray | None) -> None:
        """Fold a new embedding into the track's exponential moving average.

        A single frame's embedding is noisy -- the head rotates, the focus
        drifts -- so BoT-SORT matches against a smoothed bank rather than the
        most recent observation.
        """
        if feature is None:
            return
        vector = np.asarray(feature, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-12:
            return
        vector = vector / norm
        if self.smooth_feature is None or self.smooth_feature.shape != vector.shape:
            self.smooth_feature = vector
            return
        blended = (
            self.feature_momentum * self.smooth_feature
            + (1.0 - self.feature_momentum) * vector
        )
        self.smooth_feature = blended / max(float(np.linalg.norm(blended)), 1e-12)

    def _on_matched(self, detection: Detection, frame: FramePacket) -> None:
        del frame
        self.absorb_feature(detection.meta.get(REID_FEATURE_KEY))


class BoTSortTracker(ByteTracker):
    """ByteTrack plus optional camera-motion compensation and appearance fusion.

    See the module docstring for why both extensions default to off. With the
    shipped defaults this tracker is exactly ByteTrack -- same associations,
    same IDs -- which is the point: enabling either extension is then a change
    a reviewer can attribute.
    """

    name = "botsort"
    track_class = BoTSortTrack

    def __init__(
        self,
        config: TrackingConfig | None = None,
        *,
        embedder: ReIDEmbedder | None = None,
        motion_compensator: CameraMotionCompensator | None = None,
        proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD,
        appearance_threshold: float = DEFAULT_APPEARANCE_THRESHOLD,
        feature_momentum: float = DEFAULT_FEATURE_MOMENTUM,
    ) -> None:
        super().__init__(config)
        self.embedder = embedder
        self.proximity_threshold = float(proximity_threshold)
        self.appearance_threshold = float(appearance_threshold)
        self.feature_momentum = float(feature_momentum)
        self._cmc: CameraMotionCompensator | None = None
        if self.config.botsort_use_cmc:
            self._cmc = (
                motion_compensator
                if motion_compensator is not None
                else CameraMotionCompensator()
            )
        elif motion_compensator is not None:
            raise ValueError(
                "a motion_compensator was supplied but tracking.botsort_use_cmc "
                "is False; enable it in the config rather than passing one in, "
                "so that the audit log records that global motion was absorbed "
                "by the tracker"
            )

    # ------------------------------------------------------------------- API

    @property
    def uses_reid(self) -> bool:
        """True when appearance actually participates in association."""
        return self.config.botsort_use_reid and self.embedder is not None

    def update(
        self, detections: list[Detection], frame: FramePacket
    ) -> list[TrackRecord]:
        if self.uses_reid:
            self._embed_detections(detections, frame)
        return super().update(detections, frame)

    def reset(self) -> None:
        super().reset()
        if self._cmc is not None:
            self._cmc.reset()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["botsort_use_cmc"] = self.config.botsort_use_cmc
        info["botsort_use_reid"] = self.config.botsort_use_reid
        info["reid_embedder"] = type(self.embedder).__name__ if self.embedder else None
        info["cmc_method"] = self._cmc.method if self._cmc else None
        info["proximity_threshold"] = self.proximity_threshold
        info["appearance_threshold"] = self.appearance_threshold
        return info

    # ------------------------------------------------------------- internals

    def _spawn(self, detection: Detection, frame: FramePacket) -> ManagedTrack:
        track = super()._spawn(detection, frame)
        if isinstance(track, BoTSortTrack):
            track.feature_momentum = self.feature_momentum
        return track

    def _embed_detections(self, detections: list[Detection], frame: FramePacket) -> None:
        """Fill in ``meta[REID_FEATURE_KEY]`` for detections that lack it."""
        if self.embedder is None:
            return
        pending = [d for d in detections if d.meta.get(REID_FEATURE_KEY) is None]
        if not pending:
            return
        features = np.asarray(
            self.embedder.embed(frame.image, [d.box for d in pending]),
            dtype=np.float64,
        ).reshape(len(pending), -1)
        for detection, feature in zip(pending, features, strict=True):
            detection.meta[REID_FEATURE_KEY] = feature

    def _compensate_camera_motion(self, frame: FramePacket) -> None:
        """Warp every track state into this frame's coordinates.

        Only runs when ``botsort_use_cmc`` is enabled, which it is not by
        default: on a rigidly-mounted camera the global motion this would
        remove is the sample's bulk fluid transport, and removing it here
        would silently subtract an unrecorded amount from every velocity the
        product reports. See the module docstring.
        """
        if self._cmc is None or not self.config.botsort_use_cmc:
            return
        warp = self._cmc.estimate(frame.image)
        if warp is None:
            return
        for track in self._tracks:
            track.kf.apply_affine(warp)
            # The association cost reads ``predicted_box``; refresh it so the
            # warp is actually seen by this frame's matching.
            track.last_box = track.kf.to_box()

    def _association_cost(
        self,
        tracks: list[ManagedTrack],
        detections: list[Detection],
        frame: FramePacket,
    ) -> np.ndarray:
        del frame
        iou_dist = iou_distance(
            [t.predicted_box for t in tracks], [d.box for d in detections]
        )
        if not self.uses_reid or not tracks or not detections:
            return iou_dist

        track_features = [
            t.smooth_feature if isinstance(t, BoTSortTrack) else None for t in tracks
        ]
        det_features = [d.meta.get(REID_FEATURE_KEY) for d in detections]
        track_ready = np.array([f is not None for f in track_features], dtype=bool)
        det_ready = np.array([f is not None for f in det_features], dtype=bool)
        if not track_ready.any() or not det_ready.any():
            return iou_dist

        dim = int(
            np.asarray(
                next(f for f in track_features if f is not None), dtype=np.float64
            ).size
        )
        track_matrix = np.zeros((len(tracks), dim), dtype=np.float64)
        det_matrix = np.zeros((len(detections), dim), dtype=np.float64)
        for i, feature in enumerate(track_features):
            if feature is not None:
                track_matrix[i] = np.asarray(feature, dtype=np.float64).reshape(-1)[:dim]
        for j, feature in enumerate(det_features):
            if feature is not None:
                det_matrix[j] = np.asarray(feature, dtype=np.float64).reshape(-1)[:dim]

        embedding_dist = cosine_distance(track_matrix, det_matrix)
        # A missing feature is "no appearance evidence", never "different
        # appearance": force those entries to 1 so the fusion falls back to IoU.
        embedding_dist[~track_ready, :] = 1.0
        embedding_dist[:, ~det_ready] = 1.0

        return fuse_iou_reid(
            iou_dist,
            embedding_dist,
            proximity_threshold=self.proximity_threshold,
            appearance_threshold=self.appearance_threshold,
        )
