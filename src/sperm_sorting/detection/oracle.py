"""Ground-truth detector with controllable, reproducible noise.

Why this exists
---------------
Every accuracy figure this pipeline produces -- track continuity, motility
grades, the shot ratio, the accept/reject decision -- is downstream of
detection. Measured with a real detector, a change in the final number could
have come from the tracker, the motion model, the gate, or from the detector
having a slightly different day. That makes the rest of the pipeline
effectively untestable.

:class:`OracleDetector` breaks the coupling. It reads the simulator's own
ground truth out of ``frame.meta["gt_detections"]`` and returns it as ordinary
detections, so a test can pin detection quality to *exactly* known and then
attribute any downstream error to the stage under test. Turning the three noise
knobs up sweeps detector quality as an independent variable -- "at what miss
rate does track identity start to break" is a question with an answer, and this
is what answers it.

The hard rule
-------------
``track_id`` from the ground truth is **never** written into
:attr:`Detection.track_id`. Doing so would hand the tracker the answer, and
every tracking test would then pass regardless of whether the tracker works. It
goes into ``Detection.meta["gt_track_id"]``, which evaluation code may read and
the tracker must not.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..config import DetectionConfig
from ..schemas.detection import BoundingBox, Detection
from ..schemas.frame import FramePacket
from .base import Detector
from .postprocess import clip_boxes, filter_boxes_by_size, top_k_detections

__all__ = ["GT_META_KEY", "OracleDetector"]

logger = logging.getLogger(__name__)

#: Key the synthetic source writes its per-frame ground truth under.
GT_META_KEY = "gt_detections"

#: Default seed, matching ``RunConfig.seed``, so an oracle built without an
#: explicit seed still reproduces the default run.
_DEFAULT_SEED = 1234


class OracleDetector(Detector):
    """Returns simulator ground truth, degraded by configurable noise.

    Parameters
    ----------
    cfg
        ``oracle_miss_rate``, ``oracle_false_positive_rate`` and
        ``oracle_box_jitter_px`` drive the noise; ``min_box_size_px``,
        ``max_box_size_px`` and ``max_detections`` are applied exactly as they
        are for a learned detector, so switching detectors does not silently
        change the box population the tracker sees.
    seed
        Base seed. Combined with each frame's ``frame_id`` to seed a fresh
        generator per frame -- see :meth:`_frame_rng`.
    """

    name = "oracle"

    def __init__(
        self, cfg: DetectionConfig | None = None, seed: int = _DEFAULT_SEED
    ) -> None:
        self.cfg = cfg if cfg is not None else DetectionConfig(architecture="oracle")
        self.class_names = tuple(self.cfg.class_names)
        self.seed = int(seed)
        self._warned_missing_gt = False

        for field in ("oracle_miss_rate", "oracle_false_positive_rate"):
            value = float(getattr(self.cfg, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"detection.{field} must lie in [0, 1], got {value}")
        if self.cfg.oracle_box_jitter_px < 0.0:
            raise ValueError("detection.oracle_box_jitter_px must be non-negative")

    # ------------------------------------------------------------------ rng

    def _frame_rng(self, frame_id: int) -> np.random.Generator:
        """A generator determined by ``(seed, frame_id)`` and nothing else.

        Deliberately *not* one long-lived generator advanced per call. A single
        stream makes the noise on frame 500 depend on how many frames were
        processed before it, so a test that starts mid-sequence, replays a
        subset, or retries a frame gets different noise -- and replay
        determinism is a hard requirement here, not a nicety. Seeding per frame
        makes the noise a pure function of the frame's identity.
        """
        return np.random.default_rng([self.seed, int(frame_id)])

    # -------------------------------------------------------------- detection

    def detect(self, frame: FramePacket) -> list[Detection]:
        records = self._ground_truth(frame)
        height, width = frame.height, frame.width
        if not records or height < 1 or width < 1:
            return []

        boxes = np.empty((len(records), 4), dtype=np.float32)
        class_ids = np.empty((len(records),), dtype=np.int64)
        track_ids: list[int | None] = []
        for index, record in enumerate(records):
            boxes[index] = self._parse_box(record, index)
            class_ids[index] = int(record.get("class_id", 0))
            raw_track = record.get("track_id")
            track_ids.append(None if raw_track is None else int(raw_track))

        rng = self._frame_rng(frame.frame_id)
        n_gt = boxes.shape[0]

        # Every random draw happens here, in a fixed order and at a fixed size,
        # *before* any of them is used. That makes each knob independent: raising
        # the miss rate does not shift the jitter stream, so a sweep over one
        # parameter is not silently a sweep over all three.
        miss_draw = rng.random(n_gt)
        jitter = rng.standard_normal((n_gt, 4)) * float(self.cfg.oracle_box_jitter_px)
        n_false_positive = int(
            rng.binomial(n_gt, float(self.cfg.oracle_false_positive_rate))
        )
        fp_source = rng.integers(0, n_gt, size=n_false_positive)
        fp_position = rng.random((n_false_positive, 2))
        # False positives are drawn across the *whole* admissible score band,
        # not a low one. A spurious detection that is always low-confidence
        # would be removable by a threshold, which would make the oracle
        # easier than reality rather than merely noisier.
        fp_score = rng.uniform(
            min(float(self.cfg.score_threshold), 0.999), 1.0, size=n_false_positive
        )

        kept = miss_draw >= float(self.cfg.oracle_miss_rate)
        true_boxes = self._jitter(boxes[kept], jitter[kept])
        true_classes = class_ids[kept]
        true_tracks = [t for t, keep in zip(track_ids, kept, strict=True) if keep]
        true_scores = np.ones((true_boxes.shape[0],), dtype=np.float32)

        fp_boxes = self._false_positive_boxes(
            boxes, fp_source, fp_position, (height, width)
        )
        fp_classes = class_ids[fp_source]

        all_boxes = np.concatenate([true_boxes, fp_boxes], axis=0)
        all_scores = np.concatenate(
            [true_scores, fp_score.astype(np.float32)], axis=0
        )
        all_classes = np.concatenate([true_classes, fp_classes], axis=0)
        all_tracks: list[int | None] = [*true_tracks, *([None] * n_false_positive)]

        # Same clip/size/cap treatment a learned detector gets, so that a test
        # comparing the two is comparing detectors and not post-processing.
        all_boxes = clip_boxes(all_boxes, (height, width))
        all_boxes, all_scores, keep = filter_boxes_by_size(
            all_boxes, all_scores, self.cfg.min_box_size_px, self.cfg.max_box_size_px
        )
        all_classes = all_classes[keep]
        all_tracks = [all_tracks[i] for i in keep]

        order = top_k_detections(all_scores, self.cfg.max_detections)
        n_true = int(true_boxes.shape[0])
        original_index = keep[order]

        detections: list[Detection] = []
        for position, source_index in zip(order, original_index, strict=True):
            class_id = int(all_classes[position])
            detections.append(
                Detection(
                    frame_id=frame.frame_id,
                    box=BoundingBox.from_xyxy(
                        *(float(v) for v in all_boxes[position])
                    ),
                    score=float(all_scores[position]),
                    class_id=class_id,
                    class_name=(
                        self.class_names[class_id]
                        if 0 <= class_id < len(self.class_names)
                        else f"class_{class_id}"
                    ),
                    capture_time_s=frame.capture_time_s,
                    track_id=None,  # see the module docstring; this is load-bearing
                    meta={
                        "gt_track_id": all_tracks[position],
                        "oracle_false_positive": bool(source_index >= n_true),
                    },
                )
            )
        return detections

    # ------------------------------------------------------------- internals

    def _ground_truth(self, frame: FramePacket) -> list[Mapping[str, Any]]:
        """Pull the ground-truth list out of the frame, tolerating its absence.

        A missing key is not an error -- the oracle may legitimately be pointed
        at a live or replayed frame that has no ground truth -- but it is worth
        saying once, because silently returning nothing looks identical to a
        detector that finds nothing.
        """
        records = frame.meta.get(GT_META_KEY)
        if records is None:
            if not self._warned_missing_gt:
                self._warned_missing_gt = True
                logger.warning(
                    "%s: frame %d has no meta['%s']; the oracle detector only "
                    "works on a source that publishes ground truth (the "
                    "synthetic simulator does). Returning no detections.",
                    self.name,
                    frame.frame_id,
                    GT_META_KEY,
                )
            return []
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError(
                f"frame.meta['{GT_META_KEY}'] must be a sequence of mappings, "
                f"got {type(records).__name__}"
            )
        return [r for r in records if isinstance(r, Mapping)]

    @staticmethod
    def _parse_box(record: Mapping[str, Any], index: int) -> np.ndarray:
        box = record.get("box_xyxy")
        if box is None:
            raise KeyError(
                f"ground-truth record {index} has no 'box_xyxy' key; got "
                f"{sorted(record)}"
            )
        values = np.asarray(box, dtype=np.float32).reshape(-1)
        if values.size != 4:
            raise ValueError(
                f"ground-truth record {index} 'box_xyxy' must have 4 values, "
                f"got {values.size}"
            )
        return values

    @staticmethod
    def _jitter(boxes: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Perturb each edge independently, then re-order so the box stays valid.

        Per-edge noise rather than a whole-box translation, because that is what
        a real detector's localisation error looks like: the centre and the
        extent are estimated separately and both wobble. Re-ordering after the
        fact (rather than clamping) keeps the box's area distribution symmetric
        instead of biasing it upward.
        """
        if boxes.shape[0] == 0:
            return boxes.astype(np.float32)
        jittered = boxes.astype(np.float32) + noise.astype(np.float32)
        x_pair = np.sort(jittered[:, [0, 2]], axis=1)
        y_pair = np.sort(jittered[:, [1, 3]], axis=1)
        return np.stack(
            [x_pair[:, 0], y_pair[:, 0], x_pair[:, 1], y_pair[:, 1]], axis=1
        ).astype(np.float32)

    @staticmethod
    def _false_positive_boxes(
        boxes: np.ndarray,
        source: np.ndarray,
        position: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Place spurious boxes of realistic size at uniformly random locations.

        Sizes are copied from real objects in the same frame so a false positive
        cannot be separated from a true one by size alone -- an oracle whose
        false positives were trivially filterable would understate how hard the
        downstream problem is.

        The count is drawn as ``Binomial(n_gt, rate)``, i.e. the rate is *per
        true object*, so false positives scale with scene density the way real
        ones do. The documented consequence: a frame with no ground truth gets
        no false positives.
        """
        n_fp = int(source.shape[0])
        if n_fp == 0:
            return np.zeros((0, 4), dtype=np.float32)
        height, width = float(shape[0]), float(shape[1])
        widths = np.clip(boxes[source, 2] - boxes[source, 0], 1.0, width)
        heights = np.clip(boxes[source, 3] - boxes[source, 1], 1.0, height)
        x1 = position[:, 0] * np.maximum(width - widths, 0.0)
        y1 = position[:, 1] * np.maximum(height - heights, 0.0)
        return np.stack([x1, y1, x1 + widths, y1 + heights], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ admin

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "architecture": "oracle",
                "backend": "none",
                "seed": self.seed,
                "oracle_miss_rate": self.cfg.oracle_miss_rate,
                "oracle_false_positive_rate": self.cfg.oracle_false_positive_rate,
                "oracle_box_jitter_px": self.cfg.oracle_box_jitter_px,
                "ground_truth_key": GT_META_KEY,
                "leaks_track_id": False,
            }
        )
        return info
