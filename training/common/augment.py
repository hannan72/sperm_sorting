"""Augmentation, constrained by what the labels actually assert.

The governing rule for this dataset is unusual and worth stating before any
code: **the label is a claim about geometry and texture that must remain
visible in the pixels.** MHSMA's four aspects are WHO strict-criteria
judgements --

* ``head``     : head length 4.0-5.0 um, width 2.5-3.5 um, length:width
                 1.50-1.75;
* ``acrosome`` : acrosomal cap covering 40-70% of the head **area**;
* ``vacuole``  : no vacuole larger than ~20% of the head area;
* ``tail``     : uncoiled, ~45 um, thinner than the midpiece.

Every one of those is a measurement of the very thing a strong augmentation
would change. So the set below is short, and each entry carries the argument
for why it is label-preserving. The excluded set is longer, and each exclusion
carries the argument for why it is not.

Included
--------
``small rotation`` (default +/- 15 deg)
    Orientation is a nuisance variable: a cell settles on the coverslip at an
    arbitrary angle and the microscope has no preferred axis, so a rotated
    image of a normal head is still a normal head. Kept small only because
    MHSMA crops are head-centred with the tail trailing, so a large rotation
    pushes the tail out of the frame and turns a "normal tail" into an image
    with no tail evidence in it -- which would be a label the pixels no longer
    support.
``horizontal and vertical flip``
    A mirrored image is the same cell viewed from the other side of the
    coverslip. None of the four WHO criteria is chiral -- they are all
    lengths, ratios and area fractions -- so a flip cannot change any of them.
    This is the one free 4x on this dataset.
``mild brightness and contrast`` (default +/- 10%)
    Illumination level, condenser setting and exposure vary between fields and
    between sessions, so absolute grey level carries no morphological
    information. Bounded at 10% because the acrosome judgement *is* a contrast
    judgement -- the cap is rendered lighter than the post-acrosomal region --
    and a large contrast change moves the apparent cap boundary, i.e. the
    40-70% area fraction the label encodes.
``slight blur`` (default sigma <= 0.6 px, applied to a minority of samples)
    Focus varies within the depth of field and the quality gate deliberately
    lets marginally-defocused crops through. Bounded hard because a vacuole is
    a small, low-contrast void: blur it enough and a genuine ``vacuole=1``
    becomes an image in which no vacuole is resolvable, which trains the model
    to answer "abnormal" from no evidence.

Excluded, deliberately
----------------------
``elastic deformation``
    It changes head length, width and therefore the axis ratio -- which is
    exactly the ``head`` label. A deformation that pushes 1.60 out to 1.85
    produces a tapered head still labelled normal. This is the single most
    damaging augmentation available for this task and it is a common default.
``aggressive scaling / random resized crop``
    Head *size* in micrometres is a criterion (macrocephaly, microcephaly), and
    the crop stage fixes the scale precisely so that size is measurable. Rescaling
    breaks that, and it also breaks the acrosome area fraction near the border.
``cutout / random erasing over the head``
    It can erase the acrosomal cap or the vacuole, i.e. the sole evidence for
    two of the four labels, while the label continues to assert them. What the
    model learns from that is to guess confidently when the evidence is absent,
    which is the exact failure this product cannot tolerate.
``colour jitter, hue, saturation, channel shuffle``
    The sensor is monochrome (``Mono8``). There is no colour to jitter.
``mixup / cutmix``
    They produce fractional labels for a conjunctive rule
    (``all_four_normal``) whose semantics are not defined on fractions.

Everything operates on ``(C, H, W)`` float tensors in ``[0, 1]`` and is driven
by an explicit ``torch.Generator``, so an augmented epoch is reproducible from
the seed alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DetectionAugmentation",
    "MorphologyAugmentation",
    "rotate_boxes",
]


def _uniform(generator: Any, low: float, high: float) -> float:
    """One draw from ``U(low, high)`` on ``generator``'s stream."""
    import torch

    if high <= low:
        return float(low)
    value = torch.rand(1, generator=generator).item()
    return float(low + (high - low) * value)


def _bernoulli(generator: Any, p: float) -> bool:
    import torch

    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(torch.rand(1, generator=generator).item() < p)


# ==========================================================================
# Morphology crops
# ==========================================================================


@dataclass
class MorphologyAugmentation:
    """Label-preserving augmentation for a 128x128 (or 64x64) morphology crop.

    Every default is justified in the module docstring. The parameters exist so
    that the bounds can be *tightened* for a specific dataset, not so that they
    can be casually loosened -- loosening them past the stated limits breaks the
    label/pixel correspondence the whole dataset rests on.

    Parameters
    ----------
    enabled
        Master switch. Validation and test passes must construct this with
        ``enabled=False``: augmenting an evaluation set makes the reported
        metric a measurement of the augmentation.
    max_rotation_deg
        Half-width of the rotation range.
    flip_horizontal_p, flip_vertical_p
        Flip probabilities. 0.5 each gives the full mirror group.
    brightness, contrast
        Half-width of the multiplicative jitter, as a fraction.
    blur_p, max_blur_sigma
        Probability of applying a Gaussian blur, and its maximum sigma in
        output pixels.
    fill
        Value written into the corners a rotation exposes. The crops are
        brightfield -- objects dark on a light background -- so the corners are
        filled with the crop's own border median rather than with 0, which
        would paint four black wedges the model can trivially key on.
    """

    enabled: bool = True
    max_rotation_deg: float = 15.0
    flip_horizontal_p: float = 0.5
    flip_vertical_p: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    blur_p: float = 0.25
    max_blur_sigma: float = 0.6
    fill: str = "border_median"

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_rotation_deg <= 45.0:
            raise ValueError(
                f"max_rotation_deg must lie in [0, 45], got {self.max_rotation_deg}: "
                "beyond 45 degrees a head-centred crop loses its trailing tail, "
                "and the tail label is then unsupported by the image"
            )
        for name in ("brightness", "contrast"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 0.35:
                raise ValueError(
                    f"{name} jitter must lie in [0, 0.35], got {value}: a larger "
                    "photometric change moves the apparent acrosomal cap boundary, "
                    "which is the acrosome label"
                )
        if not 0.0 <= self.max_blur_sigma <= 1.5:
            raise ValueError(
                f"max_blur_sigma must lie in [0, 1.5] px, got {self.max_blur_sigma}: "
                "heavier blur erases small vacuoles, which is the vacuole label"
            )
        for name in ("flip_horizontal_p", "flip_vertical_p", "blur_p"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")

    def __call__(self, image: Any, generator: Any) -> Any:
        """Augment one ``(C, H, W)`` float tensor in ``[0, 1]``."""
        import torch
        from torchvision.transforms import functional as TF

        if not self.enabled:
            return image
        if image.ndim != 3:
            raise ValueError(f"expected a (C, H, W) tensor, got shape {tuple(image.shape)}")

        out = image

        if _bernoulli(generator, self.flip_horizontal_p):
            out = TF.hflip(out)
        if _bernoulli(generator, self.flip_vertical_p):
            out = TF.vflip(out)

        if self.max_rotation_deg > 0.0:
            angle = _uniform(generator, -self.max_rotation_deg, self.max_rotation_deg)
            if abs(angle) > 1e-3:
                out = TF.rotate(
                    out,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=[self._fill_value(out)],
                )

        if self.brightness > 0.0:
            factor = _uniform(generator, 1.0 - self.brightness, 1.0 + self.brightness)
            out = TF.adjust_brightness(out, factor)
        if self.contrast > 0.0:
            factor = _uniform(generator, 1.0 - self.contrast, 1.0 + self.contrast)
            out = TF.adjust_contrast(out, factor)

        if self.max_blur_sigma > 0.0 and _bernoulli(generator, self.blur_p):
            sigma = _uniform(generator, 1e-3, self.max_blur_sigma)
            # Kernel wide enough to contain +/-3 sigma, and odd as the op requires.
            radius = max(1, int(math.ceil(3.0 * sigma)))
            out = TF.gaussian_blur(out, kernel_size=2 * radius + 1, sigma=sigma)

        return torch.clamp(out, 0.0, 1.0)

    def _fill_value(self, image: Any) -> float:
        """Grey level for rotation corners.

        The border median of this very crop, so the exposed wedges match the
        illumination of the field the crop came from. A constant fill would
        create a sharp, orientation-dependent edge that a convolutional model
        learns far faster than it learns acrosome morphology.
        """
        import torch

        if self.fill != "border_median":
            return float(self.fill)
        border = torch.cat(
            [
                image[:, 0, :].reshape(-1),
                image[:, -1, :].reshape(-1),
                image[:, :, 0].reshape(-1),
                image[:, :, -1].reshape(-1),
            ]
        )
        return float(border.median())

    def to_json_dict(self) -> dict[str, Any]:
        """Exact augmentation settings, for the experiment record."""
        return {
            "enabled": self.enabled,
            "max_rotation_deg": self.max_rotation_deg,
            "flip_horizontal_p": self.flip_horizontal_p,
            "flip_vertical_p": self.flip_vertical_p,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "blur_p": self.blur_p,
            "max_blur_sigma": self.max_blur_sigma,
            "excluded": [
                "elastic_deformation (changes head axis ratio = the head label)",
                "aggressive_scaling (changes head size = the head label)",
                "cutout_over_head (erases acrosome/vacuole evidence)",
                "colour_jitter (monochrome sensor)",
                "mixup/cutmix (undefined for the conjunctive all-four-normal rule)",
            ],
        }


# ==========================================================================
# Detection frames
# ==========================================================================


def rotate_boxes(
    boxes: np.ndarray, angle_deg: float, width: float, height: float
) -> np.ndarray:
    """Rotate xyxy boxes about the image centre and re-fit axis-aligned boxes.

    The re-fit is the honest part: rotating an axis-aligned box gives a
    quadrilateral, and the smallest axis-aligned box containing it is larger
    than the original by up to ``|cos t| + |sin t|`` on each side. At the 10
    degree default that is 1.2%, which for a near-isotropic sperm head is
    negligible; it is stated here so that anyone raising the angle knows the
    boxes inflate with it.

    Boxes are clipped to the frame afterwards and degenerate results are left
    for the caller to drop -- silently discarding them here would change the
    object count without saying so.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.astype(np.float32)

    theta = math.radians(float(angle_deg))
    # torchvision rotates counter-clockwise for a positive angle in image
    # coordinates (y down), so the point transform uses the matching sign.
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = width / 2.0, height / 2.0

    corners_x = boxes[:, [0, 2, 0, 2]] - cx
    corners_y = boxes[:, [1, 1, 3, 3]] - cy
    rotated_x = corners_x * cos_t + corners_y * sin_t + cx
    rotated_y = -corners_x * sin_t + corners_y * cos_t + cy

    out = np.stack(
        [
            rotated_x.min(axis=1),
            rotated_y.min(axis=1),
            rotated_x.max(axis=1),
            rotated_y.max(axis=1),
        ],
        axis=1,
    )
    out[:, 0] = np.clip(out[:, 0], 0.0, width)
    out[:, 2] = np.clip(out[:, 2], 0.0, width)
    out[:, 1] = np.clip(out[:, 1], 0.0, height)
    out[:, 3] = np.clip(out[:, 3], 0.0, height)
    return out.astype(np.float32)


@dataclass
class DetectionAugmentation:
    """Label-preserving augmentation for a monochrome detection frame.

    Parameters mirror :class:`MorphologyAugmentation` where the argument is the
    same, and differ where it is not:

    ``flips``
        Free. A detector is a per-frame appearance model with no temporal
        context, so mirroring the frame does not contradict the flow direction
        -- nothing downstream of the detector sees the frame.
    ``small rotation`` (default +/- 10 deg)
        Same nuisance-variable argument as for crops, tightened because boxes
        inflate under rotation (see :func:`rotate_boxes`).
    ``brightness / contrast``
        Illumination varies across the field; the simulator models a Koehler
        gradient for exactly this reason.
    **No colour jitter.** The sensor is ``Mono8``. There is nothing to jitter,
    and applying a three-channel transform to a replicated grey image is a
    slow no-op that looks like augmentation in a config file.
    **No scale augmentation.** The premise of both detector architectures is
    that the object scale distribution is effectively a point (see
    ``detection/heads.py``). Teaching the head a scale range that the optics
    cannot produce spends capacity on a distribution that will never be seen.
    """

    enabled: bool = True
    max_rotation_deg: float = 10.0
    flip_horizontal_p: float = 0.5
    flip_vertical_p: float = 0.5
    brightness: float = 0.12
    contrast: float = 0.10
    #: Boxes whose shorter side falls below this after clipping are dropped, so
    #: a box rotated mostly out of frame does not become a 1-pixel target the
    #: head cannot possibly hit.
    min_box_side_px: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_rotation_deg <= 30.0:
            raise ValueError(
                f"max_rotation_deg must lie in [0, 30], got {self.max_rotation_deg}: "
                "axis-aligned boxes inflate by |cos|+|sin| under rotation, which "
                "at 30 degrees is already 37%"
            )
        for name in ("brightness", "contrast"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 0.35:
                raise ValueError(f"{name} jitter must lie in [0, 0.35], got {value}")

    def __call__(
        self, image: Any, boxes: np.ndarray, generator: Any
    ) -> tuple[Any, np.ndarray]:
        """Augment a ``(1, H, W)`` float frame and its ``(N, 4)`` xyxy boxes."""
        import torch
        from torchvision.transforms import functional as TF

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if not self.enabled:
            return image, boxes
        if image.ndim != 3:
            raise ValueError(f"expected a (C, H, W) tensor, got shape {tuple(image.shape)}")

        height, width = int(image.shape[-2]), int(image.shape[-1])
        out = image

        if _bernoulli(generator, self.flip_horizontal_p):
            out = TF.hflip(out)
            if boxes.size:
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2
        if _bernoulli(generator, self.flip_vertical_p):
            out = TF.vflip(out)
            if boxes.size:
                y1 = height - boxes[:, 3]
                y2 = height - boxes[:, 1]
                boxes[:, 1], boxes[:, 3] = y1, y2

        if self.max_rotation_deg > 0.0:
            angle = _uniform(generator, -self.max_rotation_deg, self.max_rotation_deg)
            if abs(angle) > 1e-3:
                fill = float(out.median())
                out = TF.rotate(
                    out,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=[fill],
                )
                boxes = rotate_boxes(boxes, angle, width, height)

        if self.brightness > 0.0:
            out = TF.adjust_brightness(
                out, _uniform(generator, 1.0 - self.brightness, 1.0 + self.brightness)
            )
        if self.contrast > 0.0:
            out = TF.adjust_contrast(
                out, _uniform(generator, 1.0 - self.contrast, 1.0 + self.contrast)
            )

        if boxes.size:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = np.minimum(widths, heights) >= float(self.min_box_side_px)
            boxes = boxes[keep]

        return torch.clamp(out, 0.0, 1.0), boxes.astype(np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_rotation_deg": self.max_rotation_deg,
            "flip_horizontal_p": self.flip_horizontal_p,
            "flip_vertical_p": self.flip_vertical_p,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "min_box_side_px": self.min_box_side_px,
            "excluded": [
                "colour_jitter (monochrome Mono8 sensor)",
                "scale_augmentation (object scale distribution is a point)",
                "mosaic/copy-paste (fabricates densities the optics cannot produce)",
            ],
        }
