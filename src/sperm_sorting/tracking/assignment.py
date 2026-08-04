"""Association costs and optimal assignment.

Everything here is a pure function over numpy arrays: no tracker state, no
configuration objects. That is deliberate -- association is the part of a
tracker most worth testing in isolation, and it is the part most often broken
by a silent shape or convention mismatch.

Conventions, fixed once for the whole module:

* Boxes are ``(N, 4)`` arrays of ``[x1, y1, x2, y2]`` in pixels, matching
  :class:`~sperm_sorting.schemas.detection.BoundingBox`. Extra columns (a
  score, say, as produced by ``detections_to_array``) are ignored.
* A *cost* is something to be minimised and a *distance* is a cost in
  ``[0, 1]``. Similarities (IoU, GIoU) are never passed to the assignment
  solver directly.
* ``threshold`` always means "reject a pair whose cost exceeds this", i.e.
  pairs with ``cost <= threshold`` survive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..schemas.detection import BoundingBox

#: Weight of OC-SORT's direction-consistency term. The paper's value.
DEFAULT_VDC_WEIGHT: Final[float] = 0.2

#: Finite stand-in for a forbidden pair. ``linear_sum_assignment`` raises on a
#: matrix containing infinities, so gates are expressed with this instead.
FORBIDDEN_COST: Final[float] = 1e6

_EMPTY_MATCHES: Final[np.ndarray] = np.empty((0, 2), dtype=np.int64)


def as_box_array(boxes: Sequence[BoundingBox] | np.ndarray) -> np.ndarray:
    """Coerce boxes to a contiguous ``(N, 4)`` float array of ``xyxy`` rows."""
    if isinstance(boxes, np.ndarray):
        array = np.asarray(boxes, dtype=np.float64)
        if array.size == 0:
            return np.zeros((0, 4), dtype=np.float64)
        array = array.reshape(-1, array.shape[-1])
        return np.ascontiguousarray(array[:, :4])
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array([b.as_xyxy() for b in boxes], dtype=np.float64)


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def iou_batch(
    boxes_a: Sequence[BoundingBox] | np.ndarray,
    boxes_b: Sequence[BoundingBox] | np.ndarray,
) -> np.ndarray:
    """Pairwise intersection-over-union, shape ``(len(a), len(b))``.

    Fully vectorised: at 160 FPS with a few dozen sperm per frame this is
    called several times per frame, and a Python loop here would show up in
    the latency budget.
    """
    a = as_box_array(boxes_a)
    b = as_box_array(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    intersection = inter_w * inter_h

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0.0, intersection / np.maximum(union, 1e-12), 0.0)


def giou_batch(
    boxes_a: Sequence[BoundingBox] | np.ndarray,
    boxes_b: Sequence[BoundingBox] | np.ndarray,
) -> np.ndarray:
    """Pairwise generalised IoU in ``[-1, 1]``, shape ``(len(a), len(b))``.

    GIoU keeps a usable gradient when boxes do not overlap at all, which
    matters here: a sperm at 160 FPS can move most of its own body length
    between frames, and plain IoU is exactly zero -- and therefore
    uninformative about *which* candidate is closest -- as soon as that
    happens.
    """
    a = as_box_array(boxes_a)
    b = as_box_array(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    iou = iou_batch(a, b)

    enclose_x1 = np.minimum(a[:, None, 0], b[None, :, 0])
    enclose_y1 = np.minimum(a[:, None, 1], b[None, :, 1])
    enclose_x2 = np.maximum(a[:, None, 2], b[None, :, 2])
    enclose_y2 = np.maximum(a[:, None, 3], b[None, :, 3])
    enclose_area = np.clip(enclose_x2 - enclose_x1, 0.0, None) * np.clip(
        enclose_y2 - enclose_y1, 0.0, None
    )

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(inter_x2 - inter_x1, 0.0, None) * np.clip(
        inter_y2 - inter_y1, 0.0, None
    )
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection

    correction = np.where(
        enclose_area > 0.0,
        (enclose_area - union) / np.maximum(enclose_area, 1e-12),
        0.0,
    )
    return iou - correction


def iou_distance(
    boxes_a: Sequence[BoundingBox] | np.ndarray,
    boxes_b: Sequence[BoundingBox] | np.ndarray,
) -> np.ndarray:
    """``1 - IoU``: a distance in ``[0, 1]``, ready for the solver."""
    return 1.0 - iou_batch(boxes_a, boxes_b)


def giou_distance(
    boxes_a: Sequence[BoundingBox] | np.ndarray,
    boxes_b: Sequence[BoundingBox] | np.ndarray,
) -> np.ndarray:
    """``(1 - GIoU) / 2``: GIoU rescaled to a distance in ``[0, 1]``."""
    return 0.5 * (1.0 - giou_batch(boxes_a, boxes_b))


# --------------------------------------------------------------------------
# Appearance
# --------------------------------------------------------------------------


def cosine_distance(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance ``(1 - cos) / 2`` in ``[0, 1]``.

    Inputs are ``(N, D)`` and ``(M, D)``; rows are L2-normalised here so a
    caller cannot silently pass unnormalised embeddings.
    """
    a = np.asarray(features_a, dtype=np.float64).reshape(len(features_a), -1)
    b = np.asarray(features_b, dtype=np.float64).reshape(len(features_b), -1)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 0.5 * (1.0 - a @ b.T)


# --------------------------------------------------------------------------
# Direction consistency (OC-SORT)
# --------------------------------------------------------------------------


def speed_direction(from_box: np.ndarray, to_box: np.ndarray) -> np.ndarray:
    """Unit vector ``(dx, dy)`` from one box centre to another.

    Returns a zero vector when the two centres coincide, which callers read as
    "no usable direction" rather than "no motion".
    """
    a = np.asarray(from_box, dtype=np.float64)
    b = np.asarray(to_box, dtype=np.float64)
    dx = 0.5 * (b[0] + b[2]) - 0.5 * (a[0] + a[2])
    dy = 0.5 * (b[1] + b[3]) - 0.5 * (a[1] + a[3])
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        return np.zeros(2, dtype=np.float64)
    return np.array([dx / norm, dy / norm], dtype=np.float64)


def velocity_direction_cost(
    tracks: Sequence[BoundingBox] | np.ndarray,
    detections: Sequence[BoundingBox] | np.ndarray,
    velocities: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    *,
    weight: float = DEFAULT_VDC_WEIGHT,
) -> np.ndarray:
    """OC-SORT's direction-consistency term, shape ``(len(tracks), len(dets))``.

    ``tracks`` are the tracks' **last observed** boxes (not their Kalman
    predictions -- that is the whole point of "observation-centric") and
    ``velocities`` is the matching ``(N, 2)`` array of unit direction vectors
    from :func:`speed_direction`. For each track/detection pair the angle
    between the track's established heading and the direction from its last
    observation to that detection is measured, and pairs that agree are
    rewarded:

    * perfectly aligned -> ``-weight / 2``
    * perpendicular     -> ``0``
    * reversed          -> ``+weight / 2``

    Add the result to an IoU distance. A track with no established heading
    (``velocities`` omitted, or a zero / non-finite row) contributes exactly
    zero, so an unproven direction never biases an assignment.

    ``scores`` optionally weights each column by detector confidence, as in the
    reference implementation, so a marginal detection cannot pull a track off
    course on direction agreement alone.
    """
    track_boxes = as_box_array(tracks)
    det_boxes = as_box_array(detections)
    n_tracks, n_dets = track_boxes.shape[0], det_boxes.shape[0]
    cost = np.zeros((n_tracks, n_dets), dtype=np.float64)
    if n_tracks == 0 or n_dets == 0 or velocities is None:
        return cost

    velocity = np.asarray(velocities, dtype=np.float64).reshape(n_tracks, 2)
    valid = np.isfinite(velocity).all(axis=1) & (np.linalg.norm(velocity, axis=1) > 1e-9)
    valid &= np.isfinite(track_boxes).all(axis=1)
    if not valid.any():
        return cost

    track_cx = 0.5 * (track_boxes[:, 0] + track_boxes[:, 2])
    track_cy = 0.5 * (track_boxes[:, 1] + track_boxes[:, 3])
    det_cx = 0.5 * (det_boxes[:, 0] + det_boxes[:, 2])
    det_cy = 0.5 * (det_boxes[:, 1] + det_boxes[:, 3])

    dx = det_cx[None, :] - track_cx[:, None]
    dy = det_cy[None, :] - track_cy[:, None]
    norm = np.maximum(np.hypot(dx, dy), 1e-9)
    dx /= norm
    dy /= norm

    cos_angle = np.clip(velocity[:, 0:1] * dx + velocity[:, 1:2] * dy, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    # 0 rad -> -0.5, pi/2 -> 0, pi -> +0.5.
    cost = weight * (angle - 0.5 * np.pi) / np.pi
    cost[~valid, :] = 0.0
    cost[~np.isfinite(cost)] = 0.0

    if scores is not None:
        score_row = np.asarray(scores, dtype=np.float64).reshape(1, n_dets)
        cost = cost * np.clip(score_row, 0.0, 1.0)
    return cost


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Globally optimal assignment, with a per-pair cost gate.

    Returns ``(matches, unmatched_a, unmatched_b)`` where ``matches`` is an
    ``(M, 2)`` array of ``[row, column]`` index pairs and the other two are
    1-D index arrays. A pair whose cost *exceeds* ``threshold`` is rejected and
    both of its members reappear as unmatched.

    The Hungarian algorithm is used rather than greedy nearest-neighbour
    because greedy matching is order-dependent, and order-dependence here means
    the same video can produce different track IDs on different runs -- which
    would break the once-only counting rule this product is built on.
    Ties are resolved by ``scipy``'s deterministic pivoting, so identical input
    always yields identical output.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.size == 0:
        rows = cost.shape[0] if cost.ndim == 2 else 0
        cols = cost.shape[1] if cost.ndim == 2 else 0
        return (
            _EMPTY_MATCHES.copy(),
            np.arange(rows, dtype=np.int64),
            np.arange(cols, dtype=np.int64),
        )

    solvable = np.where(np.isfinite(cost), cost, FORBIDDEN_COST)
    row_idx, col_idx = linear_sum_assignment(solvable)

    keep = cost[row_idx, col_idx] <= threshold
    keep &= np.isfinite(cost[row_idx, col_idx])
    matched_rows = row_idx[keep]
    matched_cols = col_idx[keep]

    matches = (
        np.stack([matched_rows, matched_cols], axis=1).astype(np.int64)
        if matched_rows.size
        else _EMPTY_MATCHES.copy()
    )
    unmatched_a = np.setdiff1d(
        np.arange(cost.shape[0], dtype=np.int64), matched_rows.astype(np.int64)
    )
    unmatched_b = np.setdiff1d(
        np.arange(cost.shape[1], dtype=np.int64), matched_cols.astype(np.int64)
    )
    return matches, unmatched_a, unmatched_b


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    mahalanobis: np.ndarray,
    *,
    gating_threshold: float,
    gated_value: float = FORBIDDEN_COST,
) -> np.ndarray:
    """Forbid pairs whose squared Mahalanobis distance exceeds the gate.

    Returns a new matrix; the input is not modified.
    """
    gated = np.array(cost_matrix, dtype=np.float64, copy=True)
    gated[np.asarray(mahalanobis, dtype=np.float64) > gating_threshold] = gated_value
    return gated
