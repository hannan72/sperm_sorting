#!/usr/bin/env python3
"""Detection metrics, implemented here rather than borrowed.

    # score a trained checkpoint on held-out clips
    python training/eval_detector.py --checkpoint runs/det/best.pt \\
        --source synthetic --split test -o runs/det/eval

    # score hand-built or externally-produced boxes, no model involved
    python training/eval_detector.py --ground-truth gt.json \\
        --predictions pred.json -o runs/det/eval_json

What is reported
----------------
``AP50``, ``mAP50-95``, precision, recall and F1 at a fixed score threshold,
plus three numbers that are specific to this product and that no standard
detection benchmark provides:

**small-object recall**
    Recall restricted to ground-truth boxes below ``--small-area-px``. The
    default, 1024 px^2, is COCO's "small" definition (32x32) -- it is a
    convention, not a measurement. This matters here because the entire
    argument for both detector architectures is that they keep resolution for
    tiny objects; aggregate AP hides a head that only finds the big ones.

**debris false-positive rate**
    Fraction of predictions that land on an annotated non-sperm particle. A
    detector that scores well on AP while calling every speck of debris a sperm
    inflates the shot denominator, which directly moves the 60% decision. When
    the ground truth carries no debris annotations this is reported as
    *unavailable with a reason*, never as zero.

**counting error**
    Predicted minus true object count, per frame. The pipeline's denominator is
    a count, so a systematic over- or under-count is a systematic bias in the
    accept ratio -- and it is invisible in AP, which is indifferent to a
    constant offset that ranks correctly.

Plus latency percentiles and peak memory, measured on the machine that runs
this script.

The AP definition, stated exactly
---------------------------------
COCO-style, implemented from scratch; ``pycocotools`` is not required and is
not used.

1. Predictions are sorted by descending score **globally**, across all frames.
2. Each prediction is matched, within its own frame, to the highest-IoU
   *unmatched* ground-truth box of the same class whose IoU meets the
   threshold. Matched ground truth is consumed, so two predictions on one
   object give one true positive and one false positive -- which is the
   behaviour that makes duplicate detections costly, and duplicates are exactly
   what inflates a count.
3. Cumulative precision and recall are computed over that ranking.
4. AP is the mean of the precision envelope sampled at the **101 recall points**
   ``0.00, 0.01, ..., 1.00`` (``max_{r' >= r} p(r')``), which is COCO's
   ``recThrs`` interpolation.
5. ``mAP50-95`` is the unweighted mean of AP over IoU thresholds
   ``0.50, 0.55, ..., 0.95``.

A perfect prediction set therefore gives exactly ``AP50 = 1.0``; that is
asserted in ``training/README.md`` against a run, not assumed.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.bootstrap import ensure_importable  # noqa: E402

ensure_importable()

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from sperm_sorting.errors import SpermSortingError  # noqa: E402
from sperm_sorting.schemas.detection import BoundingBox, Detection  # noqa: E402
from training.common.args import (  # noqa: E402
    build_parser,
    describe_device,
    dump_json,
    resolve_config,
    resolve_device,
)
from training.common.experiment import ExperimentRecord  # noqa: E402
from training.common.logging_utils import console_table, print_block  # noqa: E402
from training.common.seeding import seed_everything  # noqa: E402

__all__ = [
    "COCO_IOU_THRESHOLDS",
    "DEFAULT_SMALL_AREA_PX",
    "FrameAnnotations",
    "average_precision",
    "counting_error",
    "debris_false_positive_rate",
    "evaluate_detections",
    "iou_matrix",
    "match_frame",
    "precision_recall_f1",
    "small_object_recall",
]

#: COCO's IoU sweep for mAP50-95.
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

#: COCO's 101-point recall grid.
_RECALL_POINTS: np.ndarray = np.linspace(0.0, 1.0, 101)

#: COCO's "small object" area, 32x32 px. A convention borrowed for
#: comparability, not a measurement of anything in this dataset.
DEFAULT_SMALL_AREA_PX: float = 1024.0

#: Class id treated as "sperm". Everything else in the ground truth is a
#: non-sperm particle for the purposes of the debris false-positive rate.
SPERM_CLASS_ID: int = 0


# ==========================================================================
# Data containers
# ==========================================================================


@dataclass(slots=True)
class FrameAnnotations:
    """Ground truth or predictions for one frame, in the repo's box convention.

    ``boxes`` is ``(N, 4)`` xyxy, matching
    :class:`sperm_sorting.schemas.detection.BoundingBox`. Scores are ``None``
    for ground truth and required for predictions -- AP is a ranking metric and
    an unranked prediction set has no AP.
    """

    frame_id: int
    boxes: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        if self.class_ids.size != self.boxes.shape[0]:
            raise SpermSortingError(
                f"frame {self.frame_id}: {self.boxes.shape[0]} boxes but "
                f"{self.class_ids.size} class ids"
            )
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
            if self.scores.size != self.boxes.shape[0]:
                raise SpermSortingError(
                    f"frame {self.frame_id}: {self.boxes.shape[0]} boxes but "
                    f"{self.scores.size} scores"
                )

    @property
    def areas(self) -> np.ndarray:
        widths = np.maximum(self.boxes[:, 2] - self.boxes[:, 0], 0.0)
        heights = np.maximum(self.boxes[:, 3] - self.boxes[:, 1], 0.0)
        return widths * heights

    def sperm_mask(self) -> np.ndarray:
        return self.class_ids == SPERM_CLASS_ID

    @classmethod
    def from_detections(cls, frame_id: int, detections: Sequence[Detection]) -> FrameAnnotations:
        """Build from the unified :class:`Detection` format the runtime emits."""
        if not detections:
            return cls(frame_id, np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0))
        return cls(
            frame_id=frame_id,
            boxes=np.array([d.box.as_xyxy() for d in detections], dtype=np.float64),
            class_ids=np.array([d.class_id for d in detections], dtype=np.int64),
            scores=np.array([d.score for d in detections], dtype=np.float64),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame_id": int(self.frame_id),
            "boxes_xyxy": self.boxes.tolist(),
            "class_ids": self.class_ids.tolist(),
        }
        if self.scores is not None:
            out["scores"] = self.scores.tolist()
        return out


# ==========================================================================
# Geometry
# ==========================================================================


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy box sets, ``(len(a), len(b))``.

    Written out rather than imported from torchvision so that every metric in
    this file runs with numpy alone -- an evaluation harness that needs a deep
    learning framework to score a JSON file of boxes is harder to trust and
    harder to reuse.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0.0, intersection / union, 0.0)


def match_frame(
    prediction_boxes: np.ndarray,
    prediction_order: np.ndarray,
    truth_boxes: np.ndarray,
    iou_threshold: float,
    truth_taken: np.ndarray,
) -> np.ndarray:
    """Greedily match ranked predictions in one frame to unmatched truth.

    ``prediction_order`` lists prediction indices in descending score order.
    ``truth_taken`` is mutated: a matched ground-truth box is consumed, so a
    second prediction on the same object becomes a false positive. That is the
    COCO rule and it is what makes duplicate detections cost something -- which
    matters more here than in a generic benchmark, because the pipeline's shot
    denominator is a count of objects.

    Returns, for each prediction index, the truth index it matched or ``-1``.
    """
    matches = np.full(prediction_boxes.shape[0], -1, dtype=np.int64)
    if prediction_boxes.size == 0 or truth_boxes.size == 0:
        return matches

    ious = iou_matrix(prediction_boxes, truth_boxes)
    for prediction in prediction_order:
        row = ious[prediction].copy()
        row[truth_taken] = -1.0
        best = int(np.argmax(row))
        if row[best] >= iou_threshold:
            matches[prediction] = best
            truth_taken[best] = True
    return matches


# ==========================================================================
# Average precision
# ==========================================================================


def average_precision(
    truth: Sequence[FrameAnnotations],
    predictions: Sequence[FrameAnnotations],
    iou_threshold: float = 0.5,
    *,
    class_id: int = SPERM_CLASS_ID,
) -> dict[str, Any]:
    """COCO-style AP at one IoU threshold. See the module docstring for the rule.

    Returns AP together with the ingredients (``n_truth``, ``n_predictions``,
    the final cumulative precision and recall) so that a surprising AP can be
    diagnosed without re-running anything. An AP of 0.0 with ``n_truth = 0`` and
    an AP of 0.0 with ``n_truth = 500`` are very different situations and the
    scalar cannot distinguish them.
    """
    truth_by_frame = {frame.frame_id: frame for frame in truth}
    n_truth = int(sum(int(np.sum(frame.class_ids == class_id)) for frame in truth))

    scores: list[float] = []
    is_true_positive: list[bool] = []

    for frame in predictions:
        gt = truth_by_frame.get(frame.frame_id)
        keep = frame.class_ids == class_id
        boxes = frame.boxes[keep]
        frame_scores = (
            frame.scores[keep]
            if frame.scores is not None
            else np.ones(int(np.sum(keep)), dtype=np.float64)
        )
        if boxes.shape[0] == 0:
            continue

        order = np.argsort(-frame_scores, kind="stable")
        if gt is None:
            matches = np.full(boxes.shape[0], -1, dtype=np.int64)
        else:
            gt_keep = gt.class_ids == class_id
            gt_boxes = gt.boxes[gt_keep]
            taken = np.zeros(gt_boxes.shape[0], dtype=bool)
            matches = match_frame(boxes, order, gt_boxes, iou_threshold, taken)

        for index in order:
            scores.append(float(frame_scores[index]))
            is_true_positive.append(bool(matches[index] >= 0))

    if n_truth == 0:
        return {
            "ap": float("nan"),
            "iou_threshold": float(iou_threshold),
            "n_truth": 0,
            "n_predictions": len(scores),
            "note": "no ground-truth objects of this class; AP is undefined",
        }
    if not scores:
        return {
            "ap": 0.0,
            "iou_threshold": float(iou_threshold),
            "n_truth": n_truth,
            "n_predictions": 0,
            "note": "no predictions were made; recall is 0 at every rank",
        }

    order = np.argsort(-np.asarray(scores), kind="stable")
    flags = np.asarray(is_true_positive, dtype=np.float64)[order]
    tp_cumulative = np.cumsum(flags)
    fp_cumulative = np.cumsum(1.0 - flags)
    recall = tp_cumulative / float(n_truth)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)

    # Precision envelope: replace each precision by the maximum at any higher
    # recall. Without it AP would be sensitive to the sawtooth that a single
    # false positive introduces, which is noise rather than signal.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, _RECALL_POINTS, side="left")
    sampled = np.where(indices < envelope.size, envelope[np.minimum(indices, envelope.size - 1)], 0.0)
    sampled[indices >= envelope.size] = 0.0

    return {
        "ap": float(np.mean(sampled)),
        "iou_threshold": float(iou_threshold),
        "n_truth": n_truth,
        "n_predictions": int(flags.size),
        "n_true_positive": int(tp_cumulative[-1]),
        "n_false_positive": int(fp_cumulative[-1]),
        "final_precision": float(precision[-1]),
        "final_recall": float(recall[-1]),
        "interpolation": "COCO 101-point recall grid with a precision envelope",
    }


def precision_recall_f1(
    truth: Sequence[FrameAnnotations],
    predictions: Sequence[FrameAnnotations],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    class_id: int = SPERM_CLASS_ID,
) -> dict[str, Any]:
    """Operating-point precision, recall and F1 at a fixed score threshold.

    AP summarises the whole ranking; this describes the single point the
    pipeline will actually run at, which is the one that determines what the
    tracker sees. Both are reported because a model can be better on one and
    worse on the other.
    """
    truth_by_frame = {frame.frame_id: frame for frame in truth}
    tp = fp = fn = 0

    seen = set()
    for frame in predictions:
        seen.add(frame.frame_id)
        keep = frame.class_ids == class_id
        if frame.scores is not None:
            keep &= frame.scores >= score_threshold
        boxes = frame.boxes[keep]
        scores = (
            frame.scores[keep]
            if frame.scores is not None
            else np.ones(boxes.shape[0], dtype=np.float64)
        )
        gt = truth_by_frame.get(frame.frame_id)
        gt_boxes = gt.boxes[gt.class_ids == class_id] if gt is not None else np.zeros((0, 4))

        taken = np.zeros(gt_boxes.shape[0], dtype=bool)
        order = np.argsort(-scores, kind="stable")
        matches = match_frame(boxes, order, gt_boxes, iou_threshold, taken)
        tp += int(np.sum(matches >= 0))
        fp += int(np.sum(matches < 0))
        fn += int(np.sum(~taken))

    # Frames with truth but no prediction entry at all are pure false negatives.
    for frame in truth:
        if frame.frame_id not in seen:
            fn += int(np.sum(frame.class_ids == class_id))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    return {
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def small_object_recall(
    truth: Sequence[FrameAnnotations],
    predictions: Sequence[FrameAnnotations],
    *,
    small_area_px: float = DEFAULT_SMALL_AREA_PX,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    class_id: int = SPERM_CLASS_ID,
) -> dict[str, Any]:
    """Recall split by ground-truth box area.

    The whole design argument for both detector architectures is that they hold
    resolution for tiny objects (see ``detection/p2net.py`` and
    ``detection/todcnn.py``). Aggregate recall cannot test that claim, because
    a handful of large clumps can carry it. This splits the ground truth at
    ``small_area_px`` and reports both halves with their counts, so a recall of
    0.9 on four large objects cannot be mistaken for evidence.
    """
    truth_by_frame = {frame.frame_id: frame for frame in truth}
    small_total = small_hit = large_total = large_hit = 0

    predictions_by_frame = {frame.frame_id: frame for frame in predictions}
    for frame_id, gt in truth_by_frame.items():
        gt_keep = gt.class_ids == class_id
        gt_boxes = gt.boxes[gt_keep]
        gt_areas = gt.areas[gt_keep]
        if gt_boxes.shape[0] == 0:
            continue

        prediction = predictions_by_frame.get(frame_id)
        if prediction is None:
            small_total += int(np.sum(gt_areas < small_area_px))
            large_total += int(np.sum(gt_areas >= small_area_px))
            continue

        keep = prediction.class_ids == class_id
        if prediction.scores is not None:
            keep &= prediction.scores >= score_threshold
        boxes = prediction.boxes[keep]
        scores = (
            prediction.scores[keep]
            if prediction.scores is not None
            else np.ones(boxes.shape[0], dtype=np.float64)
        )
        taken = np.zeros(gt_boxes.shape[0], dtype=bool)
        match_frame(boxes, np.argsort(-scores, kind="stable"), gt_boxes, iou_threshold, taken)

        is_small = gt_areas < small_area_px
        small_total += int(np.sum(is_small))
        large_total += int(np.sum(~is_small))
        small_hit += int(np.sum(taken & is_small))
        large_hit += int(np.sum(taken & ~is_small))

    return {
        "small_area_px": float(small_area_px),
        "small_area_definition": "COCO's 32x32 convention; a threshold, not a measurement",
        "n_small": small_total,
        "n_large": large_total,
        "recall_small": float(small_hit / small_total) if small_total else float("nan"),
        "recall_large": float(large_hit / large_total) if large_total else float("nan"),
        "note": (
            "recall_small is the number the architecture choice has to justify; "
            "aggregate recall can be carried by a few large clumps"
        ),
    }


def debris_false_positive_rate(
    truth: Sequence[FrameAnnotations],
    predictions: Sequence[FrameAnnotations],
    *,
    iou_threshold: float = 0.5,
    debris_iou_threshold: float = 0.30,
    score_threshold: float = 0.0,
    class_id: int = SPERM_CLASS_ID,
) -> dict[str, Any]:
    """Fraction of predicted sperm that actually land on annotated debris.

    Two thresholds, deliberately different. A prediction counts as a sperm
    detection at ``iou_threshold`` (0.5, the usual bar). It counts as a *debris*
    false positive at ``debris_iou_threshold`` (0.30, looser), because a
    detector firing on a speck of debris typically boxes it sloppily -- and the
    question being asked is "did this fire on debris", not "did it localise the
    debris well".

    When the ground truth carries no non-sperm annotations, this returns
    ``available = False`` with a reason rather than 0.0. Zero would read as "no
    debris false positives", which is a claim; "no debris was annotated" is the
    fact.
    """
    truth_by_frame = {frame.frame_id: frame for frame in truth}
    n_debris_annotations = int(
        sum(int(np.sum(frame.class_ids != class_id)) for frame in truth)
    )
    if n_debris_annotations == 0:
        return {
            "available": False,
            "reason": (
                "the ground truth contains no non-sperm annotations, so a debris "
                "false positive cannot be distinguished from a background one. "
                "Reporting 0.0 here would be a claim rather than a measurement."
            ),
            "n_debris_annotations": 0,
        }

    n_predictions = 0
    n_on_debris = 0
    n_background = 0

    for frame in predictions:
        keep = frame.class_ids == class_id
        if frame.scores is not None:
            keep &= frame.scores >= score_threshold
        boxes = frame.boxes[keep]
        if boxes.shape[0] == 0:
            continue
        scores = (
            frame.scores[keep]
            if frame.scores is not None
            else np.ones(boxes.shape[0], dtype=np.float64)
        )
        n_predictions += boxes.shape[0]

        gt = truth_by_frame.get(frame.frame_id)
        if gt is None:
            n_background += boxes.shape[0]
            continue

        sperm_boxes = gt.boxes[gt.class_ids == class_id]
        debris_boxes = gt.boxes[gt.class_ids != class_id]
        taken = np.zeros(sperm_boxes.shape[0], dtype=bool)
        matches = match_frame(
            boxes, np.argsort(-scores, kind="stable"), sperm_boxes, iou_threshold, taken
        )
        unmatched = matches < 0
        if not np.any(unmatched):
            continue

        debris_ious = iou_matrix(boxes[unmatched], debris_boxes)
        hits = (
            debris_ious.max(axis=1) >= debris_iou_threshold
            if debris_boxes.shape[0]
            else np.zeros(int(np.sum(unmatched)), dtype=bool)
        )
        n_on_debris += int(np.sum(hits))
        n_background += int(np.sum(~hits))

    return {
        "available": True,
        "n_debris_annotations": n_debris_annotations,
        "n_predictions": n_predictions,
        "n_false_positive_on_debris": n_on_debris,
        "n_false_positive_on_background": n_background,
        "debris_false_positive_rate": (
            float(n_on_debris / n_predictions) if n_predictions else float("nan")
        ),
        "debris_detection_rate": (
            float(n_on_debris / n_debris_annotations) if n_debris_annotations else float("nan")
        ),
        "iou_threshold": float(iou_threshold),
        "debris_iou_threshold": float(debris_iou_threshold),
        "note": (
            "a debris false positive enters the tracker and can be gated into a "
            "shot, inflating the denominator of the 60% rule"
        ),
    }


def counting_error(
    truth: Sequence[FrameAnnotations],
    predictions: Sequence[FrameAnnotations],
    *,
    score_threshold: float = 0.0,
    class_id: int = SPERM_CLASS_ID,
) -> dict[str, Any]:
    """Per-frame predicted-minus-true object count.

    The signed mean is the important one: AP is indifferent to a constant
    offset that ranks correctly, but the shot denominator is a count, so a
    detector that finds 1.3 extra objects per frame biases every accept ratio
    in the same direction. The absolute mean and RMSE are reported alongside so
    a large unbiased scatter is not mistaken for accuracy.
    """
    truth_by_frame = {frame.frame_id: int(np.sum(frame.class_ids == class_id)) for frame in truth}
    predicted_by_frame: dict[int, int] = {}
    for frame in predictions:
        keep = frame.class_ids == class_id
        if frame.scores is not None:
            keep &= frame.scores >= score_threshold
        predicted_by_frame[frame.frame_id] = int(np.sum(keep))

    frame_ids = sorted(set(truth_by_frame) | set(predicted_by_frame))
    if not frame_ids:
        return {"n_frames": 0, "note": "no frames to compare"}

    true_counts = np.array([truth_by_frame.get(f, 0) for f in frame_ids], dtype=np.float64)
    predicted_counts = np.array(
        [predicted_by_frame.get(f, 0) for f in frame_ids], dtype=np.float64
    )
    errors = predicted_counts - true_counts

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(true_counts > 0, np.abs(errors) / true_counts, np.nan)

    return {
        "n_frames": len(frame_ids),
        "mean_true_count": float(np.mean(true_counts)),
        "mean_predicted_count": float(np.mean(predicted_counts)),
        "mean_signed_error": float(np.mean(errors)),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_absolute_error": float(np.max(np.abs(errors))) if errors.size else float("nan"),
        "mean_absolute_percentage_error": (
            float(np.nanmean(relative)) if np.any(np.isfinite(relative)) else float("nan")
        ),
        "frames_exact": int(np.sum(errors == 0)),
        "note": (
            "mean_signed_error is the one that matters: the shot denominator is a "
            "count, so a systematic offset biases every accept ratio the same way"
        ),
    }


def evaluate_detections(
    truth: Sequence[FrameAnnotations],
    predictions: Sequence[FrameAnnotations],
    *,
    score_threshold: float = 0.0,
    small_area_px: float = DEFAULT_SMALL_AREA_PX,
    class_id: int = SPERM_CLASS_ID,
) -> dict[str, Any]:
    """Every detection metric this project reports, in one dict."""
    per_iou = {
        f"{threshold:.2f}": average_precision(truth, predictions, threshold, class_id=class_id)
        for threshold in COCO_IOU_THRESHOLDS
    }
    aps = [row["ap"] for row in per_iou.values() if row["ap"] == row["ap"]]

    return {
        "ap50": per_iou["0.50"]["ap"],
        "ap75": per_iou["0.75"]["ap"],
        "map50_95": float(np.mean(aps)) if aps else float("nan"),
        "ap_per_iou": per_iou,
        "ap_definition": (
            "COCO-style: global score ranking, per-frame greedy IoU matching with "
            "ground truth consumed on match, 101-point recall grid with a precision "
            "envelope. pycocotools is not used."
        ),
        "operating_point": precision_recall_f1(
            truth, predictions, score_threshold=score_threshold, class_id=class_id
        ),
        "small_objects": small_object_recall(
            truth,
            predictions,
            small_area_px=small_area_px,
            score_threshold=score_threshold,
            class_id=class_id,
        ),
        "debris": debris_false_positive_rate(
            truth, predictions, score_threshold=score_threshold, class_id=class_id
        ),
        "counting": counting_error(
            truth, predictions, score_threshold=score_threshold, class_id=class_id
        ),
    }


# ==========================================================================
# Latency and memory
# ==========================================================================


def summarise_latency(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Percentiles of a latency sample.

    Percentiles rather than a mean, because a real-time pipeline is killed by
    the tail: at 160 FPS the budget is 6.25 ms per frame, and a p99 of 40 ms
    means one frame in a hundred is dropped no matter how good the mean is.
    """
    values = np.asarray([float(v) for v in samples_ms], dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "note": "no timing samples were collected"}
    return {
        "n": int(values.size),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "min_ms": float(np.min(values)),
        "note": (
            "measured on the machine named in experiment.json['hardware']; a "
            "latency figure is a property of that machine, not of the model"
        ),
    }


def peak_memory(device: Any) -> dict[str, Any]:
    """Peak memory, reported per source rather than as one blended number.

    Host RSS and CUDA allocator peak measure different things and neither
    substitutes for the other; collapsing them into "peak memory" would hide
    which one is the constraint.
    """
    out: dict[str, Any] = {}
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is kibibytes on Linux and bytes on macOS.
        divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        out["host_peak_rss_mb"] = round(usage.ru_maxrss / divisor, 1)
    except (ImportError, AttributeError):  # pragma: no cover - platform dependent
        out["host_peak_rss_mb"] = None
        out["host_peak_rss_note"] = "resource.getrusage unavailable on this platform"

    if getattr(device, "type", "cpu") == "cuda":
        import torch

        index = 0 if device.index is None else int(device.index)
        out["cuda_peak_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(index) / 1024**2, 1
        )
        out["cuda_peak_reserved_mb"] = round(
            torch.cuda.max_memory_reserved(index) / 1024**2, 1
        )
    else:
        out["cuda_peak_allocated_mb"] = None
        out["cuda_peak_note"] = "not a CUDA device; the allocator peak does not apply"
    return out


# ==========================================================================
# JSON I/O
# ==========================================================================


def load_annotations_json(path: str | Path, *, require_scores: bool) -> list[FrameAnnotations]:
    """Read frames from a JSON file.

    Two shapes are accepted, because both occur naturally::

        {"frames": [{"frame_id": 0,
                     "boxes_xyxy": [[...]], "class_ids": [0], "scores": [0.9]}]}

        {"frames": [{"frame_id": 0,
                     "detections": [{"box_xyxy": [...], "score": 0.9, "class_id": 0}]}]}

    The second is the shape :meth:`Detection.to_json_dict` produces, so a
    pipeline run's own output can be scored without conversion.
    """
    path = Path(path)
    if not path.exists():
        raise SpermSortingError(f"annotation file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpermSortingError(f"could not parse {path}: {exc}") from exc

    raw_frames = data.get("frames") if isinstance(data, dict) else data
    if not isinstance(raw_frames, list):
        raise SpermSortingError(
            f"{path} must hold a list of frames, or a dict with a 'frames' list"
        )

    frames: list[FrameAnnotations] = []
    for position, raw in enumerate(raw_frames):
        frame_id = int(raw.get("frame_id", position))
        if "detections" in raw:
            records = raw["detections"] or []
            boxes = np.array([r["box_xyxy"] for r in records], dtype=np.float64).reshape(-1, 4)
            class_ids = np.array([int(r.get("class_id", 0)) for r in records], dtype=np.int64)
            scores_list = [r.get("score") for r in records]
            scores = (
                np.array([float(s) for s in scores_list], dtype=np.float64)
                if all(s is not None for s in scores_list) and records
                else None
            )
        else:
            boxes = np.array(raw.get("boxes_xyxy", []), dtype=np.float64).reshape(-1, 4)
            class_ids = np.array(
                raw.get("class_ids", [0] * boxes.shape[0]), dtype=np.int64
            ).reshape(-1)
            raw_scores = raw.get("scores")
            scores = np.array(raw_scores, dtype=np.float64) if raw_scores is not None else None

        if require_scores and scores is None and boxes.shape[0] > 0:
            raise SpermSortingError(
                f"{path} frame {frame_id} has no scores. AP is a ranking metric; an "
                "unranked prediction set has no AP. Supply a score per box (use 1.0 "
                "if the detector genuinely produces none)."
            )
        frames.append(FrameAnnotations(frame_id, boxes, class_ids, scores))
    return frames


# ==========================================================================
# Running a checkpoint
# ==========================================================================


def _build_detector(cfg: Any, checkpoint: Path | None, device: Any) -> Any:
    """Construct the detector, rebuilding the architecture the checkpoint names.

    A checkpoint written by ``train_detector.py`` carries ``architecture`` and
    ``arch_kwargs``, and those win over the configuration's defaults. They have
    to: a network trained at ``width=8`` cannot be loaded into the default
    ``width=32``, and the resulting shape errors name thirty tensors without
    once saying "you built the wrong size". Reading the geometry back from the
    artefact makes a checkpoint self-describing, which is the only way a
    weights file survives being moved away from the command that produced it.

    The configuration still governs everything about *inference* -- score and
    NMS thresholds, box-size limits, tiling, device -- because those are
    deployment choices, not properties of the weights.
    """
    detection_cfg = cfg.detection.model_copy(
        update={"backend": cfg.detection.backend.model_copy(update={"device": str(device)})}
    )

    if checkpoint is None:
        from sperm_sorting.detection.factory import build_detector

        return build_detector(detection_cfg)

    from training.common.checkpoints import read_checkpoint

    payload = read_checkpoint(checkpoint)
    architecture = str(payload.get("architecture") or detection_cfg.architecture)
    kwargs = dict(payload.get("arch_kwargs") or {})

    if architecture == "p2net":
        from sperm_sorting.detection.p2net import P2Net, P2NetDetector

        if kwargs:
            net = P2Net(**kwargs)
            detector = P2NetDetector(detection_cfg, net=net)
        else:
            detector = P2NetDetector(detection_cfg)
    elif architecture == "todcnn":
        from sperm_sorting.detection.todcnn import TodCnnDetector, TodCnnNet

        if kwargs:
            net = TodCnnNet(**kwargs)
            detector = TodCnnDetector(detection_cfg, net=net, stride=int(kwargs.get("stride", 4)))
        else:
            detector = TodCnnDetector(detection_cfg)
    else:
        from sperm_sorting.detection.factory import build_detector

        detector = build_detector(detection_cfg)

    detector.load_weights(checkpoint, strict=True)
    return detector


def _run_detector_over_frames(
    detector: Any, frames: Sequence[Any], device: Any
) -> tuple[list[FrameAnnotations], list[FrameAnnotations], list[float]]:
    """Run the detector frame by frame, timing each call.

    Frames are processed one at a time rather than batched, because that is how
    the runtime does it: one camera frame in, one detection list out. A batched
    throughput figure would not describe the latency the scheduler has to live
    with.
    """
    import time as _time

    from sperm_sorting.schemas.enums import SourceKind, TimestampSource
    from sperm_sorting.schemas.frame import FramePacket

    truth: list[FrameAnnotations] = []
    predictions: list[FrameAnnotations] = []
    latencies: list[float] = []

    for index, frame in enumerate(frames):
        packet = FramePacket(
            frame_id=index,
            image=np.ascontiguousarray(frame.image),
            capture_time_s=float(index) / 160.0,
            timestamp_source=TimestampSource.SYNTHETIC,
            source_kind=SourceKind.SYNTHETIC,
        )
        started = _time.perf_counter()
        detections = detector.detect(packet)
        latencies.append((_time.perf_counter() - started) * 1000.0)

        truth.append(FrameAnnotations(index, frame.boxes, frame.class_ids))
        predictions.append(FrameAnnotations.from_detections(index, detections))

    del device
    return truth, predictions, latencies


# ==========================================================================
# CLI
# ==========================================================================


def build_argument_parser() -> Any:
    parser = build_parser(
        description="Evaluate a detector: AP, small-object recall, debris FP, counting, latency.",
        epilog=(
            "Examples:\n"
            "  python training/eval_detector.py --checkpoint runs/det/best.pt \\\n"
            "      --source synthetic --split test -o runs/det/eval\n"
            "  python training/eval_detector.py --ground-truth gt.json \\\n"
            "      --predictions pred.json -o runs/det/eval_json\n"
        ),
    )
    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", type=Path, default=None, help="Detector checkpoint.")

    data = parser.add_argument_group("data")
    data.add_argument("--source", choices=("visem", "synthetic"), default="synthetic")
    data.add_argument("--data-root", type=Path, default=None)
    data.add_argument("--split", choices=("train", "valid", "test"), default="test")
    data.add_argument("--n-clips", type=int, default=6)
    data.add_argument("--frames-per-clip", type=int, default=8)
    data.add_argument("--frame-width", type=int, default=320)
    data.add_argument("--frame-height", type=int, default=256)
    data.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Score boxes from JSON instead of running a model. Requires --predictions.",
    )
    data.add_argument("--predictions", type=Path, default=None)

    metric = parser.add_argument_group("metrics")
    metric.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Operating point for precision/recall/F1. Defaults to detection.score_threshold.",
    )
    metric.add_argument(
        "--small-area-px",
        type=float,
        default=DEFAULT_SMALL_AREA_PX,
        help="Ground-truth boxes below this area count as small objects (COCO uses 1024).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if (args.ground_truth is None) != (args.predictions is None):
        print(
            "error: --ground-truth and --predictions must be given together",
            file=sys.stderr,
        )
        return 2

    try:
        common = resolve_config(args)
    except SpermSortingError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    out_dir = common.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentRecord(script="eval_detector", out_dir=out_dir)
    record.args = {
        **common.to_json_dict(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "source": args.source,
        "split": args.split,
        "ground_truth": str(args.ground_truth) if args.ground_truth else None,
        "predictions": str(args.predictions) if args.predictions else None,
        "small_area_px": args.small_area_px,
    }

    with record:
        try:
            _run(args, common, record)
        except SpermSortingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            record.finish("failed", str(exc))
            record.save()
            return 1
    return 0


def _run(args: Any, common: Any, record: ExperimentRecord) -> None:
    cfg = common.cfg
    out_dir = common.out_dir

    record.determinism = seed_everything(cfg.run.seed, cfg.run.deterministic)
    record.set_config(cfg)
    device = resolve_device(common.device)
    record.hardware = describe_device(device)

    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(cfg.detection.score_threshold)
    )

    latency: dict[str, Any]
    if args.ground_truth is not None:
        truth = load_annotations_json(args.ground_truth, require_scores=False)
        predictions = load_annotations_json(args.predictions, require_scores=True)
        latency = {
            "n": 0,
            "note": "no model was run (--ground-truth/--predictions mode); nothing to time",
        }
        record.set_dataset(
            name="hand-built / external JSON",
            licence="n/a",
            splits={"scored": len(truth)},
            source=str(args.ground_truth),
        )
    else:
        from training.common.detection_data import load_detection_source

        source = load_detection_source(
            args.source,
            root=args.data_root,
            seed=cfg.run.seed,
            n_clips=args.n_clips,
            frames_per_clip=args.frames_per_clip,
            width=args.frame_width,
            height=args.frame_height,
        )
        frames = source.splits[args.split]
        reserved = {"name", "licence", "source", "splits"}
        record.set_dataset(
            name=str(source.info.get("name", args.source)),
            licence=str(source.info.get("licence", "unrecorded")),
            splits={k: len(v) for k, v in source.splits.items()},
            source=str(source.info.get("source", "")),
            evaluated_split=args.split,
            **{k: v for k, v in source.to_json_dict().items() if k not in reserved},
        )

        detector = _build_detector(cfg, args.checkpoint, device)
        record.model = {
            **detector.describe(),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        }
        if args.checkpoint is None:
            record.note(
                "No --checkpoint was given, so the detector ran with randomly "
                "initialised weights. Every number below describes an untrained "
                "network and must not be reported as detector performance."
            )
        # Warm up so the first frame's kernel selection is not counted as latency.
        height, width = frames[0].shape
        detector.warmup(height, width, iterations=2)
        truth, predictions, samples = _run_detector_over_frames(detector, frames, device)
        latency = summarise_latency(samples)
        detector.close()

    metrics = evaluate_detections(
        truth,
        predictions,
        score_threshold=score_threshold,
        small_area_px=float(args.small_area_px),
    )
    metrics["latency"] = latency
    metrics["peak_memory"] = peak_memory(device)
    metrics["n_frames"] = len(truth)

    metrics_path = dump_json(out_dir / "eval_detector.json", metrics)
    record.metrics = metrics
    record.artifact("metrics", metrics_path)

    summary_rows = [
        {"metric": "AP50", "value": metrics["ap50"]},
        {"metric": "AP75", "value": metrics["ap75"]},
        {"metric": "mAP50-95", "value": metrics["map50_95"]},
        {"metric": "precision", "value": metrics["operating_point"]["precision"]},
        {"metric": "recall", "value": metrics["operating_point"]["recall"]},
        {"metric": "F1", "value": metrics["operating_point"]["f1"]},
        {"metric": "recall (small objects)", "value": metrics["small_objects"]["recall_small"]},
        {"metric": "recall (large objects)", "value": metrics["small_objects"]["recall_large"]},
        {
            "metric": "debris FP rate",
            "value": (
                metrics["debris"].get("debris_false_positive_rate")
                if metrics["debris"].get("available")
                else None
            ),
        },
        {"metric": "count error (signed mean)", "value": metrics["counting"].get("mean_signed_error")},
        {"metric": "count error (abs mean)", "value": metrics["counting"].get("mean_absolute_error")},
        {"metric": "latency p50 (ms)", "value": latency.get("p50_ms")},
        {"metric": "latency p95 (ms)", "value": latency.get("p95_ms")},
        {"metric": "latency p99 (ms)", "value": latency.get("p99_ms")},
        {"metric": "host peak RSS (MB)", "value": metrics["peak_memory"].get("host_peak_rss_mb")},
    ]

    lines = [
        "",
        console_table(
            summary_rows,
            ("value",),
            index_column="metric",
            title=(
                f"detection metrics over {metrics['n_frames']} frame(s), "
                f"score threshold {score_threshold:.2f}"
            ),
            footer=metrics["ap_definition"],
        ),
    ]
    if not metrics["debris"].get("available", False):
        lines += ["", f"debris FP rate unavailable: {metrics['debris']['reason']}"]
    lines += ["", f"wrote {metrics_path}"]
    print_block(lines)


if __name__ == "__main__":
    raise SystemExit(main())
