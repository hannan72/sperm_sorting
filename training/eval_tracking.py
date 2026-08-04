#!/usr/bin/env python3
"""Tracking metrics: HOTA, IDF1, MOTA, MOTP, ID switches, fragmentation.

    # hand-built or exported tracks
    python training/eval_tracking.py --ground-truth gt_tracks.json \\
        --predictions pred_tracks.json -o runs/track/eval

    # run the configured detector + tracker over synthetic clips
    python training/eval_tracking.py --source synthetic --split test \\
        -c configs/synthetic.yaml -o runs/track/eval_synth

Plus two metrics that exist because of what this particular product does with a
track, and which no standard MOT benchmark reports:

**duplicate-count rate**
    One physical sperm counted more than once. The shot denominator is
    ``len(ShotRecord.track_ids)``, so a sperm that acquires two track IDs is
    counted twice and shifts the 60% accept ratio. ``ShotRecord.add_track``
    refuses a duplicate *ID*, but it cannot see that two different IDs are the
    same cell -- only this comparison against ground truth can.

**track survival length**
    How long a track lives, in frames and seconds. It sets an upper bound on
    everything downstream: below
    ``motion.min_points_for_kinematics`` there are no kinematics, below
    ``track_quality.min_observed_points`` the track is excluded from the shot
    entirely, and below ``motion.min_points_for_alh_bcf`` ALH and BCF are
    refused. A mean is not enough -- the distribution is what says how much of
    the population clears those bars.

Which HOTA this is
------------------
**The formulation implemented here is Luiten et al., "HOTA: A Higher Order
Metric for Evaluating Multi-Object Tracking" (IJCV 2021), as specified in the
paper and realised in TrackEval's ``HOTA`` class.** Concretely:

1. For each pair (ground-truth id ``g``, predicted id ``p``) a *global
   alignment score* is precomputed::

       A(g, p) = TPA_soft(g, p) / (|g| + |p| - TPA_soft(g, p))

   where ``TPA_soft`` accumulates, over every frame in which both appear, the
   Jaccard-normalised similarity
   ``s / (sum_row s + sum_col s - s)`` and ``|g|``, ``|p|`` are the detection
   counts of each id.
2. For each alpha in ``0.05, 0.10, ..., 0.95``, per-frame matching is a
   Hungarian assignment maximising ``A(g, p) * IoU(g, p)`` -- the alignment
   score *biases* the assignment towards globally consistent identities, which
   is what makes HOTA sensitive to association rather than only to detection.
   A pair is kept only if its raw ``IoU >= alpha``.
3. ``DetA_alpha = TP / (TP + FN + FP)``.
4. ``AssA_alpha = (1 / TP) * sum over matched detections c of A(c)``, where
   ``A(c) = TPA(c) / (TPA(c) + FNA(c) + FPA(c))`` uses the *actual* match
   counts at this alpha, not the soft ones from step 1.
5. ``HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)``, and
   ``HOTA = mean over alpha of HOTA_alpha``. ``DetA``, ``AssA`` and ``LocA``
   are likewise reported as means over alpha.

Note that ``HOTA`` is the mean of the per-alpha geometric means, **not** the
geometric mean of the averaged ``DetA`` and ``AssA``; the two differ, and the
former is what the paper defines. Per-alpha values are in the JSON output so
the aggregation can be checked.

``motmetrics`` and ``TrackEval`` are **not** required. If ``motmetrics`` is
installed, ``--cross-check`` will use it to recompute MOTA/IDF1 and report both
numbers side by side, which is a check on this implementation rather than a
substitute for it.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.bootstrap import ensure_importable  # noqa: E402

ensure_importable()

import json  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from sperm_sorting.errors import SpermSortingError  # noqa: E402
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
    "HOTA_ALPHAS",
    "TrackSet",
    "clear_mot",
    "duplicate_count_metrics",
    "evaluate_tracking",
    "hota",
    "identity_f1",
    "load_tracks_json",
    "track_survival",
]

#: HOTA's localisation-threshold sweep, exactly as in the paper.
HOTA_ALPHAS: tuple[float, ...] = tuple(round(0.05 + 0.05 * i, 2) for i in range(19))

_EPS = 1e-10


# ==========================================================================
# Track container
# ==========================================================================


@dataclass
class TrackSet:
    """Observations of many tracks over many frames.

    Stored as ``{frame_id: {track_id: (x1, y1, x2, y2)}}`` because every metric
    below is either a per-frame association (which needs the frame slice) or a
    per-identity aggregate (which needs the id index), and this shape gives the
    first directly and the second in one pass.
    """

    observations: dict[int, dict[int, tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    #: Frames per second, when known. Only used to express survival lengths in
    #: seconds as well as frames; every metric is frame-based.
    fps: float | None = None

    def add(self, frame_id: int, track_id: int, box: Sequence[float]) -> None:
        """Record one observation. A repeated ``(frame, id)`` pair is an error.

        Two boxes for one id in one frame is not a tracking result, it is a
        malformed one -- and silently keeping the last would quietly change
        every count below.
        """
        frame = self.observations.setdefault(int(frame_id), {})
        key = int(track_id)
        if key in frame:
            raise SpermSortingError(
                f"track {key} already has a box in frame {frame_id}; one track "
                "cannot occupy two places at once"
            )
        values = tuple(float(v) for v in box)
        if len(values) != 4:
            raise SpermSortingError(
                f"track {key} frame {frame_id}: box must be 4 values (xyxy), got {values}"
            )
        frame[key] = values  # type: ignore[assignment]

    @property
    def frame_ids(self) -> list[int]:
        return sorted(self.observations)

    @property
    def track_ids(self) -> list[int]:
        return sorted({tid for frame in self.observations.values() for tid in frame})

    @property
    def n_detections(self) -> int:
        return int(sum(len(frame) for frame in self.observations.values()))

    def frames_of(self, track_id: int) -> list[int]:
        """Frames in which ``track_id`` appears, in order."""
        return sorted(f for f, boxes in self.observations.items() if track_id in boxes)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "frames": [
                {
                    "frame_id": frame_id,
                    "tracks": [
                        {"track_id": tid, "box_xyxy": list(box)}
                        for tid, box in sorted(self.observations[frame_id].items())
                    ],
                }
                for frame_id in self.frame_ids
            ],
        }


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _similarity(
    truth: TrackSet, predicted: TrackSet, frame_id: int
) -> tuple[list[int], list[int], np.ndarray]:
    """``(gt_ids, pred_ids, IoU matrix)`` for one frame."""
    gt_frame = truth.observations.get(frame_id, {})
    pred_frame = predicted.observations.get(frame_id, {})
    gt_ids = sorted(gt_frame)
    pred_ids = sorted(pred_frame)
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for i, g in enumerate(gt_ids):
        for j, p in enumerate(pred_ids):
            matrix[i, j] = _iou(gt_frame[g], pred_frame[p])
    return gt_ids, pred_ids, matrix


def _assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximising assignment on ``cost``, via scipy when present.

    The fallback is a greedy descent over the score matrix. It is not
    equivalent to the optimal assignment in general, so it is *reported* in the
    output rather than used silently: a HOTA computed with a greedy assignment
    is a different number from one computed with Hungarian, and nobody should
    have to guess which they are reading.
    """
    if cost.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-cost)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        work = cost.copy()
        rows: list[int] = []
        cols: list[int] = []
        for _ in range(min(work.shape)):
            index = int(np.argmax(work))
            r, c = divmod(index, work.shape[1])
            if work[r, c] <= -np.inf:
                break
            rows.append(r)
            cols.append(c)
            work[r, :] = -np.inf
            work[:, c] = -np.inf
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _assignment_solver() -> str:
    try:
        import scipy.optimize  # noqa: F401

        return "scipy.optimize.linear_sum_assignment (Hungarian, optimal)"
    except ImportError:  # pragma: no cover
        return "greedy fallback (scipy absent) -- NOT the optimal assignment"


# ==========================================================================
# CLEAR MOT
# ==========================================================================


def clear_mot(
    truth: TrackSet, predicted: TrackSet, *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """MOTA, MOTP, ID switches and fragmentation (Bernardin & Stiefelhagen 2008).

    The matching is *sticky*, as the original specifies: a ground-truth track
    already matched to a predicted id keeps that pairing whenever their IoU
    still clears the threshold, and only the leftovers go to the Hungarian
    step. Without stickiness, two equally good candidates would swap
    arbitrarily from frame to frame and the ID-switch count would measure the
    tie-breaking rule instead of the tracker.

    ``MOTP`` is reported as the **mean IoU of matched pairs**, so higher is
    better. The original paper defines MOTP as a mean *distance* (lower
    better); the IoU form is the near-universal convention for box tracking and
    the direction is stated here because the two are trivially confusable.

    ``fragmentation`` counts, over the frames in which a ground-truth track is
    present, the number of transitions from *tracked* to *not tracked*. A track
    simply ending is not a fragmentation.
    """
    frame_ids = sorted(set(truth.frame_ids) | set(predicted.frame_ids))
    last_match: dict[int, int] = {}
    tracked_now: dict[int, bool] = {}

    total_gt = 0
    false_negatives = 0
    false_positives = 0
    id_switches = 0
    fragmentations = 0
    matched_ious: list[float] = []
    matched_pairs: dict[int, set[int]] = {}

    for frame_id in frame_ids:
        gt_ids, pred_ids, similarity = _similarity(truth, predicted, frame_id)
        total_gt += len(gt_ids)

        gt_index = {g: i for i, g in enumerate(gt_ids)}
        pred_index = {p: j for j, p in enumerate(pred_ids)}
        gt_taken = np.zeros(len(gt_ids), dtype=bool)
        pred_taken = np.zeros(len(pred_ids), dtype=bool)
        frame_matches: dict[int, int] = {}

        # 1. Preserve existing pairings that still hold.
        for g, p in last_match.items():
            if g in gt_index and p in pred_index:
                i, j = gt_index[g], pred_index[p]
                if similarity[i, j] >= iou_threshold:
                    gt_taken[i] = pred_taken[j] = True
                    frame_matches[g] = p
                    matched_ious.append(float(similarity[i, j]))

        # 2. Hungarian over what is left.
        free_gt = [i for i in range(len(gt_ids)) if not gt_taken[i]]
        free_pred = [j for j in range(len(pred_ids)) if not pred_taken[j]]
        if free_gt and free_pred:
            sub = similarity[np.ix_(free_gt, free_pred)]
            rows, cols = _assign(sub)
            for r, c in zip(rows, cols, strict=True):
                if sub[r, c] < iou_threshold:
                    continue
                i, j = free_gt[r], free_pred[c]
                gt_taken[i] = pred_taken[j] = True
                frame_matches[gt_ids[i]] = pred_ids[j]
                matched_ious.append(float(sub[r, c]))

        # 3. Bookkeeping.
        for g, p in frame_matches.items():
            matched_pairs.setdefault(g, set()).add(p)
            previous = last_match.get(g)
            if previous is not None and previous != p:
                id_switches += 1
            last_match[g] = p

        for g in gt_ids:
            was_tracked = tracked_now.get(g, False)
            is_tracked = g in frame_matches
            if was_tracked and not is_tracked:
                fragmentations += 1
            tracked_now[g] = is_tracked

        false_negatives += int(np.sum(~gt_taken))
        false_positives += int(np.sum(~pred_taken))

    mota = (
        1.0 - (false_negatives + false_positives + id_switches) / total_gt
        if total_gt
        else float("nan")
    )
    return {
        "mota": float(mota),
        "motp_mean_iou": float(np.mean(matched_ious)) if matched_ious else float("nan"),
        "motp_convention": "mean IoU of matched pairs; HIGHER is better",
        "id_switches": int(id_switches),
        "fragmentations": int(fragmentations),
        "fragmentation_definition": (
            "transitions from tracked to not-tracked while the ground-truth track "
            "is still present; a track simply ending is not counted"
        ),
        "n_gt_detections": int(total_gt),
        "n_pred_detections": int(predicted.n_detections),
        "false_negatives": int(false_negatives),
        "false_positives": int(false_positives),
        "n_matches": int(len(matched_ious)),
        "iou_threshold": float(iou_threshold),
        "matched_pred_ids_per_gt": {int(g): sorted(p) for g, p in matched_pairs.items()},
    }


# ==========================================================================
# IDF1
# ==========================================================================


def identity_f1(
    truth: TrackSet, predicted: TrackSet, *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """IDF1 (Ristani et al., ECCV 2016 workshops).

    A single **global** one-to-one assignment between ground-truth and
    predicted identities, chosen to maximise the total number of correctly
    associated detections, and *then* scored. That is the whole point of IDF1
    and the reason it complements MOTA: MOTA's matching is per frame, so a
    tracker that swaps two identities halfway through pays exactly one ID
    switch, while IDF1 charges it for every frame after the swap.

    ``IDTP`` is the size of the best assignment, ``IDFN = |GT| - IDTP``,
    ``IDFP = |pred| - IDTP``, and
    ``IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)``.
    """
    gt_ids = truth.track_ids
    pred_ids = predicted.track_ids
    n_gt_detections = truth.n_detections
    n_pred_detections = predicted.n_detections

    if not gt_ids or not pred_ids:
        return {
            "idf1": 0.0 if n_gt_detections or n_pred_detections else float("nan"),
            "idp": float("nan"),
            "idr": float("nan"),
            "idtp": 0,
            "idfp": n_pred_detections,
            "idfn": n_gt_detections,
            "n_gt_ids": len(gt_ids),
            "n_pred_ids": len(pred_ids),
            "iou_threshold": float(iou_threshold),
            "note": "one side has no tracks at all",
        }

    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: j for j, p in enumerate(pred_ids)}
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)

    for frame_id in sorted(set(truth.frame_ids) & set(predicted.frame_ids)):
        frame_gt, frame_pred, similarity = _similarity(truth, predicted, frame_id)
        for i, g in enumerate(frame_gt):
            for j, p in enumerate(frame_pred):
                if similarity[i, j] >= iou_threshold:
                    counts[gt_index[g], pred_index[p]] += 1.0

    rows, cols = _assign(counts)
    idtp = int(sum(counts[r, c] for r, c in zip(rows, cols, strict=True)))
    idfn = int(n_gt_detections - idtp)
    idfp = int(n_pred_detections - idtp)

    denominator = 2 * idtp + idfp + idfn
    idf1 = float(2 * idtp / denominator) if denominator else float("nan")
    idp = float(idtp / n_pred_detections) if n_pred_detections else float("nan")
    idr = float(idtp / n_gt_detections) if n_gt_detections else float("nan")

    return {
        "idf1": idf1,
        "idp": idp,
        "idr": idr,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "n_gt_ids": len(gt_ids),
        "n_pred_ids": len(pred_ids),
        "iou_threshold": float(iou_threshold),
        "assignment": {
            int(gt_ids[r]): int(pred_ids[c])
            for r, c in zip(rows, cols, strict=True)
            if counts[r, c] > 0
        },
    }


# ==========================================================================
# HOTA
# ==========================================================================


def hota(truth: TrackSet, predicted: TrackSet) -> dict[str, Any]:
    """HOTA and its components. See the module docstring for the formulation.

    Returns the alpha-averaged ``hota``, ``det_a``, ``ass_a`` and ``loc_a``,
    plus the full per-alpha breakdown so the aggregation can be checked by
    hand. ``hota`` is the mean of ``sqrt(DetA_alpha * AssA_alpha)`` -- *not*
    ``sqrt(mean DetA * mean AssA)``, which is a different number.
    """
    gt_ids = truth.track_ids
    pred_ids = predicted.track_ids
    frame_ids = sorted(set(truth.frame_ids) | set(predicted.frame_ids))

    if not gt_ids and not pred_ids:
        return {
            "hota": float("nan"),
            "det_a": float("nan"),
            "ass_a": float("nan"),
            "loc_a": float("nan"),
            "note": "both track sets are empty",
            "formulation": _HOTA_FORMULATION,
        }
    if not gt_ids or not pred_ids:
        return {
            "hota": 0.0,
            "det_a": 0.0,
            "ass_a": 0.0,
            "loc_a": float("nan"),
            "note": "one side has no tracks; every alpha scores zero",
            "formulation": _HOTA_FORMULATION,
        }

    gt_index = {g: i for i, g in enumerate(gt_ids)}
    pred_index = {p: j for j, p in enumerate(pred_ids)}
    n_gt, n_pred = len(gt_ids), len(pred_ids)

    # --- step 1: global alignment ---------------------------------------
    potential = np.zeros((n_gt, n_pred), dtype=np.float64)
    gt_count = np.zeros((n_gt, 1), dtype=np.float64)
    pred_count = np.zeros((1, n_pred), dtype=np.float64)
    per_frame: list[tuple[list[int], list[int], np.ndarray]] = []

    for frame_id in frame_ids:
        frame_gt, frame_pred, similarity = _similarity(truth, predicted, frame_id)
        per_frame.append((frame_gt, frame_pred, similarity))
        if frame_gt:
            gt_count[[gt_index[g] for g in frame_gt], 0] += 1.0
        if frame_pred:
            pred_count[0, [pred_index[p] for p in frame_pred]] += 1.0
        if not frame_gt or not frame_pred:
            continue
        # Jaccard-normalised similarity, exactly as TrackEval computes it.
        denominator = (
            similarity.sum(axis=0)[None, :] + similarity.sum(axis=1)[:, None] - similarity
        )
        normalised = np.zeros_like(similarity)
        usable = denominator > _EPS
        normalised[usable] = similarity[usable] / denominator[usable]
        rows = np.array([gt_index[g] for g in frame_gt], dtype=np.int64)
        cols = np.array([pred_index[p] for p in frame_pred], dtype=np.int64)
        potential[rows[:, None], cols[None, :]] += normalised

    alignment = potential / np.maximum(gt_count + pred_count - potential, _EPS)

    # --- steps 2-5: per-alpha matching and scoring ------------------------
    per_alpha: list[dict[str, Any]] = []
    for alpha in HOTA_ALPHAS:
        matches = np.zeros((n_gt, n_pred), dtype=np.float64)
        tp = fn = fp = 0
        localisation = 0.0

        for frame_gt, frame_pred, similarity in per_frame:
            if not frame_gt and not frame_pred:
                continue
            if not frame_gt:
                fp += len(frame_pred)
                continue
            if not frame_pred:
                fn += len(frame_gt)
                continue

            rows_idx = np.array([gt_index[g] for g in frame_gt], dtype=np.int64)
            cols_idx = np.array([pred_index[p] for p in frame_pred], dtype=np.int64)
            score = alignment[rows_idx[:, None], cols_idx[None, :]] * similarity
            rows, cols = _assign(score)
            keep = similarity[rows, cols] >= alpha - _EPS
            rows, cols = rows[keep], cols[keep]

            n_matched = int(rows.size)
            tp += n_matched
            fn += len(frame_gt) - n_matched
            fp += len(frame_pred) - n_matched
            if n_matched:
                localisation += float(np.sum(similarity[rows, cols]))
                matches[rows_idx[rows], cols_idx[cols]] += 1.0

        det_a = float(tp / (tp + fn + fp)) if (tp + fn + fp) else float("nan")
        # A(c) for every matched pair, weighted by how many detections that pair
        # contributed. This is the "association accuracy" of the paper.
        pair_alignment = matches / np.maximum(gt_count + pred_count - matches, 1.0)
        ass_a = float(np.sum(matches * pair_alignment) / max(tp, 1)) if tp else 0.0
        loc_a = float(localisation / tp) if tp else float("nan")
        per_alpha.append(
            {
                "alpha": float(alpha),
                "tp": int(tp),
                "fn": int(fn),
                "fp": int(fp),
                "det_a": det_a,
                "ass_a": ass_a,
                "loc_a": loc_a,
                "hota": float(np.sqrt(max(det_a, 0.0) * max(ass_a, 0.0)))
                if det_a == det_a
                else float("nan"),
            }
        )

    def mean_of(key: str) -> float:
        values = [row[key] for row in per_alpha if row[key] == row[key]]
        return float(np.mean(values)) if values else float("nan")

    return {
        "hota": mean_of("hota"),
        "det_a": mean_of("det_a"),
        "ass_a": mean_of("ass_a"),
        "loc_a": mean_of("loc_a"),
        "alphas": list(HOTA_ALPHAS),
        "per_alpha": per_alpha,
        "aggregation": (
            "hota = mean over alpha of sqrt(DetA_alpha * AssA_alpha); this is NOT "
            "sqrt(mean DetA * mean AssA)"
        ),
        "assignment_solver": _assignment_solver(),
        "formulation": _HOTA_FORMULATION,
    }


_HOTA_FORMULATION: str = (
    "Luiten et al., 'HOTA: A Higher Order Metric for Evaluating Multi-Object "
    "Tracking', IJCV 2021, as realised in TrackEval's HOTA class: global "
    "alignment score A(g,p) from Jaccard-normalised per-frame similarity, "
    "per-frame Hungarian assignment on A(g,p) * IoU with an IoU >= alpha "
    "acceptance test, DetA = TP/(TP+FN+FP), AssA = mean over matched "
    "detections of TPA/(TPA+FNA+FPA), HOTA_alpha = sqrt(DetA*AssA), alphas "
    "0.05:0.05:0.95. Implemented from the specification; TrackEval is not a "
    "dependency."
)


# ==========================================================================
# Product-specific metrics
# ==========================================================================


def duplicate_count_metrics(
    truth: TrackSet, predicted: TrackSet, *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """How often one physical sperm is counted more than once.

    This is the failure mode the shot denominator cannot defend itself against.
    :meth:`sperm_sorting.schemas.shot.ShotRecord.add_track` refuses a repeated
    track *id*, but two different ids on one cell are, to it, two cells -- and
    the 60% rule is a ratio of counts.

    Three numbers, because they answer different questions:

    ``duplicate_track_rate``
        Fraction of ground-truth tracks covered by more than one predicted id.
        "How often does this happen at all."
    ``excess_id_ratio``
        ``(distinct predicted ids matched to GT - matched GT tracks) /
        matched GT tracks``. This is the *count inflation* the denominator
        actually sees, which is the quantity that moves the accept ratio.
    ``phantom_track_rate``
        Fraction of predicted tracks that matched no ground truth at all. These
        are not duplicates -- they are objects that do not exist -- and they
        inflate the denominator just as effectively, so they are reported
        beside rather than inside the duplicate figure.
    """
    clear = clear_mot(truth, predicted, iou_threshold=iou_threshold)
    matched: dict[int, set[int]] = {
        int(g): {int(p) for p in ids}
        for g, ids in clear["matched_pred_ids_per_gt"].items()
    }

    n_gt_tracks = len(truth.track_ids)
    n_pred_tracks = len(predicted.track_ids)
    matched_gt = [g for g, ids in matched.items() if ids]
    used_pred_ids = {p for ids in matched.values() for p in ids}

    n_matched_gt = len(matched_gt)
    n_duplicated = sum(1 for g in matched_gt if len(matched[g]) > 1)
    excess = len(used_pred_ids) - n_matched_gt
    phantoms = n_pred_tracks - len(used_pred_ids)

    return {
        "n_gt_tracks": n_gt_tracks,
        "n_pred_tracks": n_pred_tracks,
        "n_matched_gt_tracks": n_matched_gt,
        "n_gt_tracks_with_multiple_ids": n_duplicated,
        "duplicate_track_rate": float(n_duplicated / n_matched_gt) if n_matched_gt else float("nan"),
        "excess_ids": int(excess),
        "excess_id_ratio": float(excess / n_matched_gt) if n_matched_gt else float("nan"),
        "n_phantom_tracks": int(phantoms),
        "phantom_track_rate": float(phantoms / n_pred_tracks) if n_pred_tracks else float("nan"),
        "ids_per_gt_track": {int(g): len(ids) for g, ids in sorted(matched.items())},
        "iou_threshold": float(iou_threshold),
        "why_this_matters": (
            "the shot denominator is len(ShotRecord.track_ids); one sperm with two "
            "ids is counted twice, which shifts the 60% accept ratio"
        ),
    }


def track_survival(
    truth: TrackSet, predicted: TrackSet, *, thresholds: Sequence[int] = (5, 6, 15)
) -> dict[str, Any]:
    """Distribution of track lifetimes, for both sides.

    ``thresholds`` default to the three bars the runtime actually applies:
    ``motion.min_points_for_kinematics`` (5), ``track_quality.min_observed_points``
    (6) and ``motion.min_points_for_alh_bcf`` (15). Reporting the fraction of
    tracks clearing each is far more use than a mean, because the mean says
    nothing about how much of the population is excluded from the shot
    entirely.
    """

    def describe(track_set: TrackSet, label: str) -> dict[str, Any]:
        lengths = np.array(
            [len(track_set.frames_of(tid)) for tid in track_set.track_ids], dtype=np.float64
        )
        if lengths.size == 0:
            return {"label": label, "n_tracks": 0, "note": "no tracks"}
        out: dict[str, Any] = {
            "label": label,
            "n_tracks": int(lengths.size),
            "mean_frames": float(np.mean(lengths)),
            "median_frames": float(np.median(lengths)),
            "p10_frames": float(np.percentile(lengths, 10)),
            "p25_frames": float(np.percentile(lengths, 25)),
            "p75_frames": float(np.percentile(lengths, 75)),
            "p90_frames": float(np.percentile(lengths, 90)),
            "min_frames": int(np.min(lengths)),
            "max_frames": int(np.max(lengths)),
        }
        if track_set.fps:
            out["mean_seconds"] = float(np.mean(lengths) / track_set.fps)
            out["median_seconds"] = float(np.median(lengths) / track_set.fps)
        for threshold in thresholds:
            out[f"fraction_at_least_{threshold}_frames"] = float(
                np.mean(lengths >= threshold)
            )
        out["histogram_frames"] = {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(lengths, return_counts=True), strict=True)
        }
        return out

    return {
        "ground_truth": describe(truth, "ground_truth"),
        "predicted": describe(predicted, "predicted"),
        "thresholds_meaning": {
            "5": "motion.min_points_for_kinematics -- below this there are no kinematics",
            "6": "track_quality.min_observed_points -- below this the track leaves the shot entirely",
            "15": "motion.min_points_for_alh_bcf -- below this ALH and BCF are refused",
        },
    }


# ==========================================================================
# Aggregate
# ==========================================================================


def evaluate_tracking(
    truth: TrackSet, predicted: TrackSet, *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """Every tracking metric this project reports."""
    return {
        "hota": hota(truth, predicted),
        "identity": identity_f1(truth, predicted, iou_threshold=iou_threshold),
        "clear": clear_mot(truth, predicted, iou_threshold=iou_threshold),
        "duplicates": duplicate_count_metrics(truth, predicted, iou_threshold=iou_threshold),
        "survival": track_survival(truth, predicted),
        "n_frames": len(set(truth.frame_ids) | set(predicted.frame_ids)),
    }


def cross_check_with_motmetrics(
    truth: TrackSet, predicted: TrackSet, *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """Recompute MOTA/IDF1 with ``motmetrics``, if it happens to be installed.

    A cross-check, never a substitute: this project does not depend on
    ``motmetrics`` and must produce every number without it. When the package
    is present, disagreement is worth knowing about, so both values are
    reported rather than one silently replacing the other.
    """
    try:
        import motmetrics as mm
    except ImportError:
        return {"available": False, "reason": "motmetrics is not installed"}

    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame_id in sorted(set(truth.frame_ids) | set(predicted.frame_ids)):
        gt_frame = truth.observations.get(frame_id, {})
        pred_frame = predicted.observations.get(frame_id, {})
        gt_ids = sorted(gt_frame)
        pred_ids = sorted(pred_frame)
        gt_boxes = np.array(
            [[gt_frame[g][0], gt_frame[g][1],
              gt_frame[g][2] - gt_frame[g][0], gt_frame[g][3] - gt_frame[g][1]]
             for g in gt_ids], dtype=np.float64
        ).reshape(-1, 4)
        pred_boxes = np.array(
            [[pred_frame[p][0], pred_frame[p][1],
              pred_frame[p][2] - pred_frame[p][0], pred_frame[p][3] - pred_frame[p][1]]
             for p in pred_ids], dtype=np.float64
        ).reshape(-1, 4)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0 - iou_threshold)
        accumulator.update(gt_ids, pred_ids, distances, frameid=frame_id)

    metrics = mm.metrics.create()
    summary = metrics.compute(
        accumulator,
        metrics=["mota", "motp", "idf1", "num_switches", "num_fragmentations"],
        name="cross_check",
    )
    row = summary.loc["cross_check"]
    return {
        "available": True,
        "mota": float(row["mota"]),
        "motp_distance": float(row["motp"]),
        "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]),
        "fragmentations": int(row["num_fragmentations"]),
        "note": (
            "motmetrics reports MOTP as a distance (1 - IoU); this harness reports "
            "it as a mean IoU, so the two are complementary, not equal"
        ),
    }


# ==========================================================================
# I/O
# ==========================================================================


def load_tracks_json(path: str | Path) -> TrackSet:
    """Read a track set from JSON.

    Two shapes, both accepted::

        {"fps": 160, "frames": [{"frame_id": 0,
                                 "tracks": [{"track_id": 1, "box_xyxy": [...]}]}]}

        {"fps": 160, "observations": [{"frame_id": 0, "track_id": 1,
                                       "box_xyxy": [...]}]}

    The flat form is what falls out of a CSV or an audit log; the nested form is
    what falls out of a pipeline run. Supporting both costs ten lines and saves
    everyone a conversion script.
    """
    path = Path(path)
    if not path.exists():
        raise SpermSortingError(f"track file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpermSortingError(f"could not parse {path}: {exc}") from exc

    tracks = TrackSet(fps=data.get("fps") if isinstance(data, dict) else None)
    if isinstance(data, dict) and "observations" in data:
        for record in data["observations"]:
            tracks.add(record["frame_id"], record["track_id"], record["box_xyxy"])
        return tracks

    frames = data.get("frames") if isinstance(data, dict) else data
    if not isinstance(frames, list):
        raise SpermSortingError(
            f"{path} must hold a 'frames' list or an 'observations' list"
        )
    for position, frame in enumerate(frames):
        frame_id = int(frame.get("frame_id", position))
        for record in frame.get("tracks", []) or []:
            tracks.add(frame_id, record["track_id"], record["box_xyxy"])
    return tracks


def tracks_from_detection_frames(frames: Sequence[Any]) -> TrackSet:
    """Ground-truth :class:`TrackSet` from annotated detection frames.

    Frames without ``track_ids`` are refused rather than skipped: a partially
    identified ground truth silently under-counts the denominator of every
    metric here, and the result would look like a well-performing tracker.
    """
    tracks = TrackSet()
    for index, frame in enumerate(frames):
        if frame.track_ids is None:
            raise SpermSortingError(
                f"frame {index} of video '{frame.video_id}' has no ground-truth "
                "track ids, so tracking cannot be scored against it"
            )
        for box, track_id in zip(frame.boxes, frame.track_ids, strict=True):
            if int(track_id) < 0:
                continue
            tracks.add(index, int(track_id), box)
    return tracks


def run_tracker_over_frames(cfg: Any, frames: Sequence[Any], device: Any) -> TrackSet:
    """Run the configured detector and tracker over annotated frames.

    Ground truth is placed in ``FramePacket.meta['gt_detections']`` in the shape
    :class:`~sperm_sorting.detection.oracle.OracleDetector` expects, so
    ``detection.architecture=oracle`` measures the *tracker* with detection
    quality pinned to a known value. Any other architecture measures the pair.
    """
    from sperm_sorting.detection.factory import build_detector
    from sperm_sorting.schemas.enums import SourceKind, TimestampSource
    from sperm_sorting.schemas.frame import FramePacket
    from sperm_sorting.tracking.factory import build_tracker

    detection_cfg = cfg.detection.model_copy(
        update={"backend": cfg.detection.backend.model_copy(update={"device": str(device)})}
    )
    detector = build_detector(detection_cfg)
    tracker = build_tracker(cfg.tracking)

    fps = float(cfg.acquisition.synthetic.fps) or 160.0
    tracks = TrackSet(fps=fps)
    try:
        for index, frame in enumerate(frames):
            gt_records = [
                {
                    "box_xyxy": [float(v) for v in box],
                    "class_id": int(class_id),
                    "track_id": int(track_id) if frame.track_ids is not None else None,
                }
                for box, class_id, track_id in zip(
                    frame.boxes,
                    frame.class_ids,
                    frame.track_ids
                    if frame.track_ids is not None
                    else [-1] * len(frame.boxes),
                    strict=True,
                )
            ]
            packet = FramePacket(
                frame_id=index,
                image=np.ascontiguousarray(frame.image),
                capture_time_s=index / fps,
                timestamp_source=TimestampSource.SYNTHETIC,
                source_kind=SourceKind.SYNTHETIC,
                meta={"gt_detections": gt_records},
            )
            detections = detector.detect(packet)
            active = tracker.update(detections, packet)
            for track in active:
                point = track.points[-1] if track.points else None
                if point is None or not point.observed or point.frame_id != index:
                    continue
                tracks.add(index, track.track_id, point.box.as_xyxy())
    finally:
        detector.close()
    return tracks


# ==========================================================================
# CLI
# ==========================================================================


def build_argument_parser() -> Any:
    parser = build_parser(
        description="Evaluate tracking: HOTA, IDF1, MOTA, MOTP, IDSW, duplicates, survival.",
        epilog=(
            "Examples:\n"
            "  python training/eval_tracking.py --ground-truth gt.json --predictions pred.json\n"
            "  python training/eval_tracking.py --source synthetic --split test \\\n"
            "      -c configs/synthetic.yaml -o runs/track/eval\n"
        ),
    )
    data = parser.add_argument_group("data")
    data.add_argument("--ground-truth", type=Path, default=None)
    data.add_argument("--predictions", type=Path, default=None)
    data.add_argument("--source", choices=("visem", "synthetic"), default="synthetic")
    data.add_argument("--data-root", type=Path, default=None)
    data.add_argument("--split", choices=("train", "valid", "test"), default="test")
    data.add_argument("--n-clips", type=int, default=6)
    data.add_argument("--frames-per-clip", type=int, default=12)
    data.add_argument("--frame-width", type=int, default=320)
    data.add_argument("--frame-height", type=int, default=256)

    metric = parser.add_argument_group("metrics")
    metric.add_argument("--iou-threshold", type=float, default=0.5)
    metric.add_argument(
        "--cross-check",
        action="store_true",
        help="Also compute MOTA/IDF1 with motmetrics, if installed, and report both.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if (args.ground_truth is None) != (args.predictions is None):
        print("error: --ground-truth and --predictions must be given together", file=sys.stderr)
        return 2

    try:
        common = resolve_config(args)
    except SpermSortingError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    out_dir = common.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentRecord(script="eval_tracking", out_dir=out_dir)
    record.args = {
        **common.to_json_dict(),
        "ground_truth": str(args.ground_truth) if args.ground_truth else None,
        "predictions": str(args.predictions) if args.predictions else None,
        "source": args.source,
        "split": args.split,
        "iou_threshold": args.iou_threshold,
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

    if args.ground_truth is not None:
        truth = load_tracks_json(args.ground_truth)
        predicted = load_tracks_json(args.predictions)
        record.set_dataset(
            name="hand-built / exported track JSON",
            licence="n/a",
            splits={"frames": len(set(truth.frame_ids) | set(predicted.frame_ids))},
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
        truth = tracks_from_detection_frames(frames)
        truth.fps = float(cfg.acquisition.synthetic.fps)
        predicted = run_tracker_over_frames(cfg, frames, device)
        record.model = {
            "detector": cfg.detection.architecture,
            "tracker": cfg.tracking.algorithm,
            "note": (
                "with detection.architecture=oracle this measures the tracker with "
                "detection quality pinned; with any other architecture it measures "
                "the detector and tracker together"
            ),
        }

    metrics = evaluate_tracking(truth, predicted, iou_threshold=float(args.iou_threshold))
    if args.cross_check:
        metrics["motmetrics_cross_check"] = cross_check_with_motmetrics(
            truth, predicted, iou_threshold=float(args.iou_threshold)
        )

    metrics_path = dump_json(out_dir / "eval_tracking.json", metrics)
    record.metrics = metrics
    record.artifact("metrics", metrics_path)

    rows = [
        {"metric": "HOTA", "value": metrics["hota"]["hota"]},
        {"metric": "  DetA", "value": metrics["hota"]["det_a"]},
        {"metric": "  AssA", "value": metrics["hota"]["ass_a"]},
        {"metric": "  LocA", "value": metrics["hota"]["loc_a"]},
        {"metric": "IDF1", "value": metrics["identity"]["idf1"]},
        {"metric": "  IDP", "value": metrics["identity"]["idp"]},
        {"metric": "  IDR", "value": metrics["identity"]["idr"]},
        {"metric": "MOTA", "value": metrics["clear"]["mota"]},
        {"metric": "MOTP (mean IoU, higher better)", "value": metrics["clear"]["motp_mean_iou"]},
        {"metric": "ID switches", "value": metrics["clear"]["id_switches"]},
        {"metric": "fragmentations", "value": metrics["clear"]["fragmentations"]},
        {"metric": "duplicate-count rate", "value": metrics["duplicates"]["duplicate_track_rate"]},
        {"metric": "excess-id ratio", "value": metrics["duplicates"]["excess_id_ratio"]},
        {"metric": "phantom-track rate", "value": metrics["duplicates"]["phantom_track_rate"]},
        {
            "metric": "track survival, median frames",
            "value": metrics["survival"]["predicted"].get("median_frames"),
        },
        {
            "metric": "  fraction >= 6 frames (shot bar)",
            "value": metrics["survival"]["predicted"].get("fraction_at_least_6_frames"),
        },
    ]

    lines = [
        "",
        console_table(
            rows,
            ("value",),
            index_column="metric",
            title=(
                f"tracking metrics over {metrics['n_frames']} frame(s), "
                f"IoU threshold {args.iou_threshold:.2f}"
            ),
            footer=metrics["hota"].get("formulation", ""),
        ),
    ]
    cross = metrics.get("motmetrics_cross_check")
    if cross is not None:
        if cross.get("available"):
            lines += [
                "",
                "motmetrics cross-check: "
                f"MOTA {cross['mota']:.6f} (this harness {metrics['clear']['mota']:.6f}), "
                f"IDF1 {cross['idf1']:.6f} (this harness {metrics['identity']['idf1']:.6f}), "
                f"IDSW {cross['id_switches']} (this harness {metrics['clear']['id_switches']})",
            ]
        else:
            lines += ["", f"motmetrics cross-check skipped: {cross['reason']}"]
    lines += ["", f"wrote {metrics_path}"]
    print_block(lines)


if __name__ == "__main__":
    raise SystemExit(main())
