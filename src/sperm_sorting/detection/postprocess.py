"""Decoding, suppression and coordinate-space bookkeeping for detectors.

Everything in this module is deliberately **pure numpy**. Three reasons:

1. The ONNX and (future) TensorRT backends never hold a torch tensor, so a
   torch-based NMS would force a dependency the deployment target does not
   need. ``torchvision.ops.nms`` in particular is a common source of
   "works on my machine" breakage when torch and torchvision versions drift.
2. Post-processing runs once per frame at up to 160 Hz on a handful of boxes;
   the GPU round-trip costs more than the arithmetic saves.
3. Pure numpy is trivially deterministic, and replay-determinism is a hard
   requirement of this pipeline. Every ordering here is settled with a
   ``kind="stable"`` sort so that equal scores break ties by index rather than
   by whatever the sorting implementation happens to do today.

The one invariant that the rest of the package relies on: a box that leaves
this module is in the pixel space its caller says it is in, and the caller is
responsible for calling :func:`scale_boxes` before handing it to anything
downstream. Detectors undo their own geometry; nothing else knows about it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..constants import EPS
from ..schemas.detection import BoundingBox, Detection
from ..schemas.frame import FramePacket

__all__ = [
    "arrays_to_detections",
    "batched_nms",
    "clip_boxes",
    "compute_tile_grid",
    "decode_centernet_heatmap",
    "filter_boxes_by_size",
    "finalise_boxes",
    "merge_tiled_detections",
    "nms",
    "scale_boxes",
    "top_k_detections",
]

# Index arrays are int64 everywhere so that they can be used directly as numpy
# fancy indices without a silent upcast on every call.
_EMPTY_INDEX = np.zeros((0,), dtype=np.int64)


# ==========================================================================
# Suppression
# ==========================================================================


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy non-maximum suppression, returning the indices that survive.

    The loop is over *kept* boxes rather than over all pairs, and the IoU of
    one kept box against every remaining candidate is computed in a single
    vectorised step. For the box counts this pipeline sees (tens to a few
    hundred) that is far cheaper than materialising the full N x N IoU matrix,
    and it degrades gracefully if a mis-thresholded frame produces thousands.

    Ties in ``scores`` are broken by ascending index via a stable sort, which
    is what makes repeated runs bit-identical.

    Parameters
    ----------
    boxes
        ``(N, 4)`` array of ``x1, y1, x2, y2``.
    scores
        ``(N,)`` confidence per box.
    iou_threshold
        Candidates overlapping a kept box by more than this are dropped. A
        non-positive threshold keeps only the single best box per cluster of
        *any* overlap; a threshold >= 1 disables suppression entirely.

    Returns
    -------
    np.ndarray
        ``int64`` indices into ``boxes``, ordered by descending score.
    """
    # float64 internally: batched_nms shifts coordinates by a per-class offset
    # that can be large enough to eat float32's mantissa on a 4K frame.
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return _EMPTY_INDEX.copy()
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"boxes/scores length mismatch: {boxes.shape[0]} vs {scores.shape[0]}"
        )
    if iou_threshold >= 1.0:
        return np.argsort(-scores, kind="stable").astype(np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[current], x1[rest])
        iy1 = np.maximum(y1[current], y1[rest])
        ix2 = np.minimum(x2[current], x2[rest])
        iy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[current] + areas[rest] - inter
        # Two zero-area boxes have an undefined IoU; calling it 0 keeps both,
        # which is the conservative choice -- the size filter removes them.
        iou = np.where(union > EPS, inter / np.maximum(union, EPS), 0.0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Class-aware NMS: boxes of different classes never suppress each other.

    This matters here specifically because the debris/impurity class is
    *supposed* to sit on top of, or immediately beside, a sperm detection.
    Suppressing across classes would silently delete the very false positives
    the evaluation is trying to count.

    Implemented with the standard coordinate-offset trick -- each class is
    translated into its own disjoint region of the plane, so a single global
    NMS pass cannot produce a cross-class overlap -- rather than looping per
    class, which keeps the cost independent of the number of classes.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    if boxes.shape[0] == 0:
        return _EMPTY_INDEX.copy()
    if not (boxes.shape[0] == scores.shape[0] == class_ids.shape[0]):
        raise ValueError(
            "boxes/scores/class_ids length mismatch: "
            f"{boxes.shape[0]}/{scores.shape[0]}/{class_ids.shape[0]}"
        )

    span = float(np.max(boxes)) - float(np.min(boxes))
    stride = span + 1.0
    offsets = class_ids.astype(np.float64) * stride
    shifted = boxes + offsets[:, None]
    return nms(shifted, scores, iou_threshold)


def top_k_detections(
    scores: np.ndarray, max_detections: int | None
) -> np.ndarray:
    """Indices of the ``max_detections`` highest scores, descending.

    Kept separate from :func:`nms` because the cap is a *resource* limit (it
    bounds tracker input and therefore per-frame latency), not a quality
    filter, and the two must be tunable independently.
    """
    scores = np.asarray(scores).reshape(-1)
    order = np.argsort(-scores, kind="stable").astype(np.int64)
    if max_detections is not None and 0 <= max_detections < order.size:
        order = order[:max_detections]
    return order


# ==========================================================================
# CenterNet-style decoding
# ==========================================================================


def _local_maximum_mask(heatmap: np.ndarray, kernel: int) -> np.ndarray:
    """``pooled == heatmap`` mask from a stride-1 'same' max-pool.

    A sliding-window view is used instead of ``scipy.ndimage.maximum_filter``
    or ``cv2.dilate`` so that this module keeps working in a numpy-only
    deployment, and so the semantics are exactly torch's ``max_pool2d`` with
    ``stride=1`` -- which is what the reference CenterNet decode uses and what
    an exported ONNX graph would embed.
    """
    if kernel <= 1:
        return np.ones_like(heatmap, dtype=bool)
    if kernel % 2 == 0:
        raise ValueError(f"nms_kernel must be odd, got {kernel}")
    pad = kernel // 2
    # -inf padding cannot win a max against a real score, so border cells are
    # judged only against their in-image neighbours.
    padded = np.pad(
        heatmap,
        ((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=-np.inf,
    )
    # A full-rank window shape (1 on the channel axis) rather than `axis=(1, 2)`:
    # identical result, but it stays within numpy's typed overloads.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (1, kernel, kernel))
    pooled = windows.max(axis=(-3, -2, -1))
    # `pooled >= heatmap` always holds, so `<=` is an exact equality test that
    # also keeps every member of a flat plateau. The NMS pass that follows in
    # the detector collapses plateaus; doing it here would need a tie-break
    # rule that the training targets do not define.
    return pooled <= heatmap


def decode_centernet_heatmap(
    heatmap: np.ndarray,
    size_map: np.ndarray,
    offset_map: np.ndarray,
    stride: float,
    score_threshold: float,
    max_detections: int,
    nms_kernel: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn dense anchor-free head outputs into boxes.

    The peak-picking max-pool is the whole point of an anchor-free head: it
    replaces box-level NMS with a 3x3 "am I the brightest cell in my
    neighbourhood" test, which costs one convolution instead of a quadratic
    pairwise loop. It is *not* a substitute for NMS across tile seams or
    across duplicated peaks on a large object, which is why the detectors
    still run :func:`batched_nms` afterwards.

    Parameters
    ----------
    heatmap
        ``(C, H, W)`` (or ``(H, W)`` for a single class) of per-class centre
        probabilities, **already sigmoid-activated**.
    size_map
        ``(2, H, W)`` predicted ``w, h`` in **input-image pixels**. Predicting
        pixels rather than feature cells means the numbers stay interpretable
        against ``min_box_size_px`` without knowing the stride.
    offset_map
        ``(2, H, W)`` sub-cell centre refinement in **feature-cell units**,
        i.e. the fractional part discarded when the true centre was quantised
        to a cell. Without it the centre of every object snaps to a
        ``stride``-pixel lattice, which for a 10 px sperm head is a 40%
        positional error at stride 4.
    stride
        Input pixels per feature cell.
    score_threshold
        Peaks below this are discarded before the top-k cut.
    max_detections
        Hard cap on returned boxes. Applied after sorting by score, so it
        keeps the best ones.
    nms_kernel
        Side of the odd square max-pool window used for peak-picking.

    Returns
    -------
    tuple
        ``(boxes, scores, class_ids)`` with boxes ``(N, 4)`` xyxy in
        **input-image pixels**, scores ``(N,)`` float32, class ids ``(N,)``
        int64, all sorted by descending score.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 2:
        heatmap = heatmap[None, :, :]
    if heatmap.ndim != 3:
        raise ValueError(f"heatmap must be (C, H, W) or (H, W), got {heatmap.shape}")
    size_map = np.asarray(size_map, dtype=np.float32).reshape(2, *heatmap.shape[1:])
    offset_map = np.asarray(offset_map, dtype=np.float32).reshape(
        2, *heatmap.shape[1:]
    )

    peaks = _local_maximum_mask(heatmap, nms_kernel) & (heatmap >= score_threshold)
    cls_idx, ys, xs = np.nonzero(peaks)
    if cls_idx.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    scores = heatmap[cls_idx, ys, xs]
    order = top_k_detections(scores, max_detections)
    cls_idx, ys, xs, scores = cls_idx[order], ys[order], xs[order], scores[order]

    # Centre in input pixels: integer cell + predicted sub-cell offset, scaled
    # up by the stride. This is the exact inverse of the quantisation in
    # `heads.build_centernet_targets`; the two must be changed together.
    cx = (xs.astype(np.float32) + offset_map[0, ys, xs]) * float(stride)
    cy = (ys.astype(np.float32) + offset_map[1, ys, xs]) * float(stride)
    half_w = np.maximum(size_map[0, ys, xs], 0.0) * 0.5
    half_h = np.maximum(size_map[1, ys, xs], 0.0) * 0.5

    boxes = np.stack(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1
    ).astype(np.float32)
    return boxes, scores.astype(np.float32), cls_idx.astype(np.int64)


# ==========================================================================
# Coordinate spaces
# ==========================================================================


def clip_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Clamp boxes into ``[0, W] x [0, H]`` for ``shape = (H, W)``.

    Returns a new array; nothing in the pipeline mutates a caller's boxes.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes.copy()
    height, width = float(shape[0]), float(shape[1])
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, width)
    out[:, 1] = np.clip(out[:, 1], 0.0, height)
    out[:, 2] = np.clip(out[:, 2], 0.0, width)
    out[:, 3] = np.clip(out[:, 3], 0.0, height)
    # Clipping can invert a box that lay entirely outside the frame. Collapse
    # rather than flip, because `BoundingBox` rejects x2 < x1 outright.
    out[:, 2] = np.maximum(out[:, 2], out[:, 0])
    out[:, 3] = np.maximum(out[:, 3], out[:, 1])
    return out


def scale_boxes(
    boxes: np.ndarray,
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
    letterbox_pad: tuple[float, float] | None = None,
) -> np.ndarray:
    """Map boxes out of the network's input space back to the source frame.

    Two padding conventions are supported, because the detectors in this
    package use one and imported ONNX models routinely use the other:

    * ``letterbox_pad=None`` -- ``from_shape`` is the shape of the *content*
      (the resized image), and any padding the network needed was appended on
      the right/bottom. Bottom-right padding never shifts a coordinate, so
      there is nothing to subtract; this is why the torch detectors here pad
      that way. Callers must pass the pre-padding shape.
    * ``letterbox_pad=(pad_x, pad_y)`` -- classic centred letterbox, where
      ``pad_x`` was added on **both** left and right and ``pad_y`` on both top
      and bottom of ``from_shape``. The offset is removed first, then the
      remaining content is scaled.

    Parameters
    ----------
    boxes
        ``(N, 4)`` xyxy in ``from_shape`` pixels.
    from_shape
        ``(H, W)`` of the space the boxes are currently in.
    to_shape
        ``(H, W)`` of the space they should end up in.

    Returns
    -------
    np.ndarray
        ``(N, 4)`` float32 in ``to_shape`` pixels. Not clipped -- clipping is
        a separate decision, and a partly out-of-frame box is still evidence.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return boxes.copy()

    from_h, from_w = float(from_shape[0]), float(from_shape[1])
    to_h, to_w = float(to_shape[0]), float(to_shape[1])
    out = boxes.copy()

    if letterbox_pad is not None:
        pad_x, pad_y = float(letterbox_pad[0]), float(letterbox_pad[1])
        out[:, [0, 2]] -= pad_x
        out[:, [1, 3]] -= pad_y
        from_w -= 2.0 * pad_x
        from_h -= 2.0 * pad_y

    if from_w <= 0.0 or from_h <= 0.0:
        raise ValueError(
            f"degenerate source shape after removing padding: {(from_h, from_w)}"
        )

    out[:, [0, 2]] *= to_w / from_w
    out[:, [1, 3]] *= to_h / from_h
    return out


def filter_boxes_by_size(
    boxes: np.ndarray,
    scores: np.ndarray,
    min_size: float,
    max_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop boxes whose extent is implausible for the imaged object.

    The test is on the **short** side against ``min_size`` and the **long**
    side against ``max_size``. Using the short side as the floor is what makes
    this a noise filter rather than an aspect-ratio filter: a sperm imaged
    near the edge of the depth of field can be genuinely elongated, but it can
    never be one pixel thin.

    Returns ``(boxes, scores, keep)`` where ``keep`` indexes the *input*
    arrays, so a caller holding parallel class ids or per-box metadata can
    apply the same selection without re-deriving it.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return boxes.copy(), scores.copy(), _EMPTY_INDEX.copy()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    short_side = np.minimum(widths, heights)
    long_side = np.maximum(widths, heights)
    mask = (short_side >= float(min_size)) & (long_side <= float(max_size))
    keep = np.nonzero(mask)[0].astype(np.int64)
    return boxes[keep], scores[keep], keep


# ==========================================================================
# Tiling
# ==========================================================================


def compute_tile_grid(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Cover ``(height, width)`` with overlapping ``tile_size`` windows.

    The final tile on each axis is pushed flush against the far edge rather
    than left partial. A partial tile would have to be padded, and padding a
    tile changes the local statistics a detector sees -- objects near the
    frame edge would then be scored under different conditions from objects in
    the middle, which shows up as a systematic edge bias in recall.

    ``overlap`` must exceed the largest expected object so that every object
    is wholly inside at least one tile; an object cut by every tile boundary
    it touches can only ever be detected as two fragments.

    Returns a list of ``(x0, y0, x1, y1)`` in source pixels, row-major, so the
    order is deterministic.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(
            f"tiling overlap ({overlap}) must be smaller than tile_size "
            f"({tile_size}); otherwise the grid never advances"
        )

    def starts(extent: int) -> list[int]:
        if extent <= tile_size:
            return [0]
        values = list(range(0, extent - tile_size + 1, step))
        if values[-1] + tile_size < extent:
            values.append(extent - tile_size)
        return values

    tiles: list[tuple[int, int, int, int]] = []
    for y0 in starts(height):
        for x0 in starts(width):
            tiles.append(
                (x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height))
            )
    return tiles


def finalise_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    source_shape: tuple[int, int],
    min_box_size: float,
    max_box_size: float,
    iou_threshold: float,
    max_detections: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip, size-filter, suppress and cap -- in that order.

    Shared by every backend so that swapping torch for ONNX cannot change the
    box population the tracker sees. The order is not arbitrary:

    * **clip first**, so a box hanging off the frame edge is measured by its
      *visible* extent against the size filter rather than by an extent that is
      partly imaginary;
    * **filter before suppressing**, because NMS cost grows with the box count
      and noise boxes are the cheapest thing to remove;
    * **cap last**, so the detection budget is spent on survivors rather than
      on duplicates that suppression was about to delete anyway.
    """
    boxes = clip_boxes(boxes, source_shape)
    boxes, scores, keep = filter_boxes_by_size(
        boxes, scores, min_box_size, max_box_size
    )
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)[keep]
    if boxes.shape[0] == 0:
        return boxes, scores, class_ids

    selected = batched_nms(boxes, scores, class_ids, iou_threshold)
    boxes, scores, class_ids = boxes[selected], scores[selected], class_ids[selected]
    capped = top_k_detections(scores, max_detections)
    return boxes[capped], scores[capped], class_ids[capped]


def arrays_to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    frame: FramePacket,
    class_names: Sequence[str],
) -> list[Detection]:
    """Wrap final arrays as :class:`Detection` records bound to ``frame``.

    ``track_id`` is left ``None`` unconditionally. Association is the tracker's
    job; a detector that guessed here would make every tracking metric
    downstream meaningless, which is exactly the failure the oracle detector's
    ``gt_track_id`` convention exists to prevent.
    """
    detections: list[Detection] = []
    for box, score, class_id in zip(boxes, scores, class_ids, strict=True):
        index = int(class_id)
        detections.append(
            Detection(
                frame_id=frame.frame_id,
                box=BoundingBox.from_xyxy(*(float(v) for v in box)),
                score=float(score),
                class_id=index,
                class_name=(
                    class_names[index]
                    if 0 <= index < len(class_names)
                    else f"class_{index}"
                ),
                capture_time_s=frame.capture_time_s,
            )
        )
    return detections


def merge_tiled_detections(
    tile_boxes: Sequence[np.ndarray],
    tile_scores: Sequence[np.ndarray],
    tile_class_ids: Sequence[np.ndarray],
    tile_origins: Sequence[tuple[int, int]],
    iou_threshold: float,
    source_shape: tuple[int, int] | None = None,
    max_detections: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch per-tile detections into one source-frame detection set.

    Each tile's boxes are translated by that tile's origin and then a single
    class-aware NMS is run over the union. The NMS is not optional bookkeeping:
    tiles overlap by design, so every object in an overlap region is detected
    once per tile that contains it, and without this step the tracker would see
    duplicate observations of one sperm and could split it into two tracks --
    which corrupts the shot denominator, not merely the box list.

    Parameters
    ----------
    tile_boxes, tile_scores, tile_class_ids
        Per-tile arrays; boxes are ``(N_i, 4)`` xyxy in that tile's own pixel
        coordinates.
    tile_origins
        Per-tile ``(x0, y0)`` in source pixels.
    iou_threshold
        Cross-tile suppression threshold.
    source_shape
        ``(H, W)``; when given, merged boxes are clipped to the frame.
    max_detections
        Optional cap applied after suppression.

    Returns
    -------
    tuple
        ``(boxes, scores, class_ids)`` in source-frame pixels, descending
        score.
    """
    if not (len(tile_boxes) == len(tile_scores) == len(tile_class_ids) == len(tile_origins)):
        raise ValueError("merge_tiled_detections requires four equal-length sequences")

    collected_boxes: list[np.ndarray] = []
    collected_scores: list[np.ndarray] = []
    collected_classes: list[np.ndarray] = []
    for boxes, scores, class_ids, (x0, y0) in zip(
        tile_boxes, tile_scores, tile_class_ids, tile_origins, strict=True
    ):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0] == 0:
            continue
        shifted = boxes.copy()
        shifted[:, [0, 2]] += float(x0)
        shifted[:, [1, 3]] += float(y0)
        collected_boxes.append(shifted)
        collected_scores.append(np.asarray(scores, dtype=np.float32).reshape(-1))
        collected_classes.append(np.asarray(class_ids, dtype=np.int64).reshape(-1))

    if not collected_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes_all = np.concatenate(collected_boxes, axis=0)
    scores_all = np.concatenate(collected_scores, axis=0)
    classes_all = np.concatenate(collected_classes, axis=0)

    if source_shape is not None:
        boxes_all = clip_boxes(boxes_all, source_shape)

    keep = batched_nms(boxes_all, scores_all, classes_all, iou_threshold)
    if max_detections is not None and 0 <= max_detections < keep.size:
        keep = keep[:max_detections]
    return boxes_all[keep], scores_all[keep], classes_all[keep]
