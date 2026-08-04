"""Anchor-free detection head, shared by every torch detector here.

Why anchor-free at all, for this problem specifically: a sperm head at the
configured optics is a small, near-isotropic blob with almost no scale
variation and no meaningful aspect-ratio variation. Anchors exist to cover a
wide, unknown scale/ratio space; here that space is a point, so anchors buy
nothing and cost an assignment rule, an anchor-matching IoU threshold, and a
per-anchor duplication of every head channel. A CenterNet-style head
(Zhou et al., "Objects as Points", arXiv 1904.07850) reduces detection to
"find the local maxima of a centre heatmap", which is both cheaper and easier
to keep deterministic.

The head is *shared* between :mod:`todcnn` and :mod:`p2net` on purpose. It
means the two architectures differ only in their feature extractor, so a
comparison between them measures the backbone and nothing else -- if each had
its own head, any measured difference would be uninterpretable.

Training lives in ``training/``; the losses and the target builder live here
because they are part of the head's contract. ``build_centernet_targets`` is
the exact inverse of ``postprocess.decode_centernet_heatmap`` and the two must
always be edited together -- a silent disagreement between them produces a
model that trains to a good loss and detects nothing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "CenterNetHead",
    "ConvNormAct",
    "build_centernet_targets",
    "centernet_focal_loss",
    "centernet_loss",
    "draw_gaussian",
    "gaussian_radius",
    "group_count",
    "head_output_signature",
    "masked_l1_loss",
]

#: Probabilities are clamped this far away from 0 and 1 before entering a log.
#: float32 ``log(0)`` is ``-inf`` and one such cell poisons the whole batch
#: gradient, which is the single most common way a CenterNet run dies.
_PROB_EPS = 1e-4


def group_count(channels: int, preferred: int = 8) -> int:
    """Largest divisor of ``channels`` not exceeding ``preferred``.

    ``nn.GroupNorm`` requires the group count to divide the channel count
    exactly, so a configurable-width backbone cannot hard-code 8 groups without
    crashing on, say, width 12. Falling back to the gcd keeps every width legal.
    """
    return math.gcd(int(preferred), int(channels)) or 1


class ConvNormAct(nn.Sequential):
    """Conv -> GroupNorm -> ReLU, the only building block these backbones use.

    GroupNorm throughout, for the same reason as in :class:`CenterNetHead`:
    inference is batch-size 1 and tiled inference batches correlated crops of
    one frame, so BatchNorm's statistics are either stale or wrong depending on
    the mode. GroupNorm is identical in train and eval, which also removes a
    whole class of "trains fine, deploys badly" failures.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 8,
        activation: bool = True,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels, groups), out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class CenterNetHead(nn.Module):
    """Three sibling predictions on one feature map.

    ``heatmap``
        One channel per class, sigmoid. Peak = object centre.
    ``size``
        Two channels, ``w`` and ``h`` **in input-image pixels**. Pixels rather
        than feature cells or log-space so that the number is directly
        comparable to ``DetectionConfig.min_box_size_px`` without the reader
        having to know the stride of whichever backbone produced it.
    ``offset``
        Two channels of sub-cell centre refinement, sigmoid-constrained to
        ``[0, 1)``.

    Two deliberate deviations from the reference implementation:

    * ``size`` goes through ``softplus``. A raw linear size head can predict a
      negative width, which produces an inverted box that ``BoundingBox``
      rejects outright -- a crash, on an untrained or merely unlucky model.
      ``softplus`` makes non-negativity structural instead of hoping the loss
      enforces it.
    * ``offset`` goes through ``sigmoid``. The target is by construction the
      fractional part of a coordinate, so it lies in ``[0, 1)``; an
      unconstrained head can predict 3.5 and move the decoded centre into a
      different cell than the one that fired, which reads as a systematic
      localisation bias rather than as the range violation it is.

    The three heads share a single 3x3 trunk. They are all conditioned on the
    same question -- "is an object centred in this cell" -- and at 160 Hz on a
    high-resolution stride-4 map, three separate 3x3 trunks is a cost the
    latency budget cannot absorb.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        hidden_channels: int = 64,
        prior_prob: float = 0.01,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 < prior_prob < 1.0:
            raise ValueError(f"prior_prob must lie in (0, 1), got {prior_prob}")

        self.num_classes = int(num_classes)
        self.prior_prob = float(prior_prob)

        # GroupNorm, not BatchNorm: inference runs on batch size 1 (one frame),
        # and tiled inference runs on a batch whose members are crops of the
        # same frame and therefore not independent. BatchNorm's running stats
        # are fine in both cases, but GroupNorm removes the train/eval
        # discrepancy entirely and is batch-size invariant by construction.
        self.trunk = ConvNormAct(
            in_channels, hidden_channels, kernel_size=3, groups=num_groups
        )
        self.heatmap_conv = nn.Conv2d(hidden_channels, self.num_classes, 1)
        self.size_conv = nn.Conv2d(hidden_channels, 2, 1)
        self.offset_conv = nn.Conv2d(hidden_channels, 2, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Focal-loss prior: start out predicting "almost certainly background"
        # everywhere. Without this the first iterations are dominated by the
        # ~10^4 negative cells per positive one and the loss diverges.
        if self.heatmap_conv.bias is not None:
            bias = -math.log((1.0 - self.prior_prob) / self.prior_prob)
            nn.init.constant_(self.heatmap_conv.bias, bias)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """Return ``heatmap``, ``heatmap_logits``, ``size`` and ``offset``.

        Both the activated heatmap and its logits are returned: the decode
        path wants probabilities, and a numerically-stable BCE-style loss (or
        an ONNX export that folds the sigmoid) wants the logits.
        """
        shared = self.trunk(features)
        logits = self.heatmap_conv(shared)
        return {
            "heatmap_logits": logits,
            "heatmap": torch.sigmoid(logits).clamp(_PROB_EPS, 1.0 - _PROB_EPS),
            "size": F.softplus(self.size_conv(shared)),
            "offset": torch.sigmoid(self.offset_conv(shared)),
        }


# ==========================================================================
# Losses
# ==========================================================================


def centernet_focal_loss(
    pred: Tensor,
    target: Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    reduction: str = "sum_over_positives",
) -> Tensor:
    """Penalty-reduced focal loss for a gaussian-splatted heatmap.

    This is the CornerNet/CenterNet variant, not the RetinaNet one. The
    difference is ``beta``: a cell near a true centre carries a target of, say,
    0.9, and predicting 0.9 there should not be punished like predicting 0.9 on
    empty background. ``(1 - target) ** beta`` down-weights exactly those
    near-miss negatives. For objects a handful of pixels across -- where the
    gaussian covers most of the object and neighbouring cells are genuinely
    ambiguous -- this is the difference between a head that localises and one
    that smears.

    Parameters
    ----------
    pred
        ``(B, C, H, W)`` sigmoid-activated predictions.
    target
        ``(B, C, H, W)`` gaussian-splatted targets in ``[0, 1]``; exactly 1.0
        marks a true centre.
    alpha
        Focusing exponent on the prediction error.
    beta
        Down-weighting exponent on the negative cells.
    reduction
        ``"sum_over_positives"`` (the reference behaviour: total loss divided
        by the number of true centres, so the value is per-object and does not
        depend on image size), ``"sum"`` or ``"mean"``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    pred = pred.clamp(_PROB_EPS, 1.0 - _PROB_EPS)

    positive_mask = target.ge(1.0).float()
    negative_mask = 1.0 - positive_mask

    positive_loss = (
        torch.log(pred) * torch.pow(1.0 - pred, alpha) * positive_mask
    )
    negative_loss = (
        torch.log(1.0 - pred)
        * torch.pow(pred, alpha)
        * torch.pow(1.0 - target, beta)
        * negative_mask
    )
    total = -(positive_loss.sum() + negative_loss.sum())

    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / max(pred.numel(), 1)
    if reduction == "sum_over_positives":
        num_positive = positive_mask.sum()
        # An image with no objects still contributes its negative loss; the
        # guard only prevents a division by zero.
        return total / torch.clamp(num_positive, min=1.0)
    raise ValueError(f"unknown reduction '{reduction}'")


def masked_l1_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L1 over the cells the mask selects, normalised by their count.

    Size and offset are only defined where an object is centred; the other
    ~99.99% of cells have no target at all. Averaging over the whole map would
    therefore divide a real gradient by the number of *background* cells, which
    scales the size loss with image resolution -- so a model tuned at 640 px
    silently mistrains at 1920 px. Normalising by the positive count keeps the
    loss per-object and resolution-independent.

    ``mask`` may be ``(B, 1, H, W)`` or ``(B, H, W)``; it is broadcast over the
    channel axis so one mask governs both ``w`` and ``h``.
    """
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(pred).float()
    loss = torch.abs(pred - target) * mask
    return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def centernet_loss(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    size_weight: float = 0.1,
    offset_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Combine the three head losses into the total optimised by training.

    ``size_weight`` defaults to 0.1 following the reference implementation: the
    size loss is in pixels and the heatmap loss is dimensionless, so without
    down-weighting a 100 px object's size error dominates the gradient and the
    heatmap never sharpens.

    Returns each term separately as well as the total, because a run where the
    total is falling only because the size term is falling is a run that is not
    learning to detect, and that is invisible from the total alone.
    """
    heatmap_loss = centernet_focal_loss(outputs["heatmap"], targets["heatmap"])
    size_loss = masked_l1_loss(outputs["size"], targets["size"], targets["mask"])
    offset_loss = masked_l1_loss(outputs["offset"], targets["offset"], targets["mask"])
    total = heatmap_loss + size_weight * size_loss + offset_weight * offset_loss
    return {
        "loss": total,
        "loss_heatmap": heatmap_loss,
        "loss_size": size_loss,
        "loss_offset": offset_loss,
    }


# ==========================================================================
# Target construction
# ==========================================================================


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Radius at which a shifted box of this size still overlaps by ``min_overlap``.

    The CornerNet formulation: solve, for each of the three ways a predicted
    box can be displaced relative to the true one (both corners inward, both
    outward, one of each), the largest displacement that still yields the
    required IoU, and take the tightest of the three. The point is that the
    splat is scaled to the object -- a near-miss on a 5 px sperm head is a
    different error from a near-miss on a 50 px clump, and a fixed-radius
    gaussian would treat them identically.

    Note on the formula: the widely-copied reference code divides all three
    roots by 2, which is only correct for the first case (where the quadratic's
    leading coefficient happens to be 1). Cases 2 and 3 have leading
    coefficients of 4 and ``4 * min_overlap``, so this implementation divides
    each root by ``2a``. The corrected roots are smaller, giving slightly
    tighter splats. This is intentional; do not "fix" it back.
    """
    if height <= 0.0 or width <= 0.0:
        return 0.0
    if not 0.0 < min_overlap < 1.0:
        raise ValueError(f"min_overlap must lie in (0, 1), got {min_overlap}")

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 * b1 - 4.0 * a1 * c1, 0.0))) / (2.0 * a1)

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(max(b2 * b2 - 4.0 * a2 * c2, 0.0))) / (2.0 * a2)

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    r3 = (-b3 + math.sqrt(max(b3 * b3 - 4.0 * a3 * c3, 0.0))) / (2.0 * a3)

    return max(0.0, min(r1, r2, r3))


def draw_gaussian(
    heatmap: np.ndarray, center: tuple[int, int], radius: int, k: float = 1.0
) -> np.ndarray:
    """Splat an unnormalised 2-D gaussian into ``heatmap``, merging by maximum.

    Merging by maximum rather than by addition is what keeps the target in
    ``[0, 1]`` when two sperm swim close enough for their splats to overlap --
    and at the densities this system targets (~28 objects in frame) that
    happens constantly. Addition would create target values above 1.0, which
    the focal loss reads as "more than a centre" and which have no meaning.

    ``heatmap`` is modified in place and also returned for convenience.
    """
    radius = int(max(radius, 0))
    diameter = 2 * radius + 1
    # sigma = diameter / 6 puts the object's extent at +/-3 sigma, i.e. the
    # splat has effectively died out exactly at the radius.
    sigma = diameter / 6.0
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    gaussian = np.exp(-(xs * xs + ys * ys) / (2.0 * sigma * sigma + 1e-12))
    gaussian[gaussian < np.finfo(np.float32).eps * gaussian.max()] = 0.0

    cx, cy = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return heatmap

    masked_heatmap = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def build_centernet_targets(
    boxes: np.ndarray | Tensor,
    class_ids: np.ndarray | Tensor,
    out_h: int,
    out_w: int,
    stride: float,
    num_classes: int,
    min_overlap: float = 0.7,
) -> dict[str, Tensor]:
    """Build the dense training targets for one image.

    This is the exact inverse of :func:`postprocess.decode_centernet_heatmap`.
    The quantisation contract, stated once so it can be checked:

        ``centre_in_cells = centre_in_input_pixels / stride``
        ``cell = floor(centre_in_cells)``  -> which cell must fire
        ``offset = centre_in_cells - cell`` -> in ``[0, 1)``, the offset target
        ``size = (w, h)`` in **input pixels**, unscaled

    Parameters
    ----------
    boxes
        ``(N, 4)`` xyxy in **input-image pixels** (i.e. after whatever resize
        the dataloader applied, not in original-frame pixels).
    class_ids
        ``(N,)`` integer class index per box.
    out_h, out_w
        Spatial size of the head's output map.
    stride
        Input pixels per output cell.
    num_classes
        Channel count of the heatmap.
    min_overlap
        Passed to :func:`gaussian_radius`.

    Returns
    -------
    dict of Tensor
        ``heatmap`` ``(num_classes, out_h, out_w)``, ``size`` ``(2, out_h,
        out_w)``, ``offset`` ``(2, out_h, out_w)``, ``mask`` ``(1, out_h,
        out_w)`` marking the cells where size/offset are defined.
    """
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"output map must be non-empty, got {(out_h, out_w)}")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    if isinstance(boxes, Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(class_ids, Tensor):
        class_ids = class_ids.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if boxes.shape[0] != class_ids.shape[0]:
        raise ValueError(
            f"boxes/class_ids length mismatch: {boxes.shape[0]} vs {class_ids.shape[0]}"
        )

    heatmap = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    size = np.zeros((2, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((1, out_h, out_w), dtype=np.float32)

    for box, class_id in zip(boxes, class_ids, strict=True):
        x1, y1, x2, y2 = (float(v) for v in box)
        width, height = x2 - x1, y2 - y1
        if width <= 0.0 or height <= 0.0:
            # A degenerate annotation cannot be splatted and must not silently
            # become a centre at (0, 0).
            continue
        if not 0 <= int(class_id) < num_classes:
            raise ValueError(
                f"class_id {int(class_id)} outside [0, {num_classes})"
            )

        centre_cells_x = (x1 + x2) * 0.5 / stride
        centre_cells_y = (y1 + y2) * 0.5 / stride
        cell_x, cell_y = math.floor(centre_cells_x), math.floor(centre_cells_y)
        if not (0 <= cell_x < out_w and 0 <= cell_y < out_h):
            # Centre outside the map: the object is in the padding region or
            # has been cropped away. Dropping it is correct -- there is no cell
            # that could learn to fire for it.
            continue

        radius = int(max(0, math.floor(gaussian_radius(height / stride, width / stride, min_overlap))))
        draw_gaussian(heatmap[int(class_id)], (cell_x, cell_y), radius)

        size[0, cell_y, cell_x] = width
        size[1, cell_y, cell_x] = height
        offset[0, cell_y, cell_x] = centre_cells_x - cell_x
        offset[1, cell_y, cell_x] = centre_cells_y - cell_y
        mask[0, cell_y, cell_x] = 1.0
        # The peak must read exactly 1.0 so the focal loss can identify it as
        # a positive; the splat itself only reaches 1.0 at its own centre, but
        # a max-merge from a neighbour could otherwise leave it below.
        heatmap[int(class_id), cell_y, cell_x] = 1.0

    return {
        "heatmap": torch.from_numpy(heatmap),
        "size": torch.from_numpy(size),
        "offset": torch.from_numpy(offset),
        "mask": torch.from_numpy(mask),
    }


def head_output_signature(num_classes: int) -> dict[str, Any]:
    """Describe the head's outputs, for ONNX export and audit metadata.

    Kept next to the head so that an exported graph's output names can never
    drift from what :class:`~.onnx_detector.OnnxDetector` looks for.
    """
    return {
        "outputs": ["heatmap", "size", "offset"],
        "heatmap_channels": int(num_classes),
        "size_channels": 2,
        "offset_channels": 2,
        "size_units": "input_image_pixels",
        "offset_units": "output_cells",
    }
