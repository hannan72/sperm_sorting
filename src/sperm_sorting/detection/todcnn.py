"""TOD-CNN-inspired tiny-object detector.

Provenance, stated plainly
--------------------------
This is an **independent reimplementation of a published concept, not of a
published model**. The original TOD-CNN (Chen et al., *TOD-CNN: An effective
convolutional neural network for tiny object detection in sperm videos*,
arXiv:2204.08166) is a Keras/TensorFlow-1 anchor-based network trained on the
MIaMIA-SVDS dataset with two classes (sperm and impurity) and an anchor set
whose every box is under 20 pixels.

Nothing from that work is reused here: no weights, no exported graph, no
layer-by-layer transcription. What is borrowed is the *design argument*, which
is the part that generalises:

* Tiny objects die in the downsampling stack. A 10 px sperm head at stride 32
  occupies a third of a feature cell -- there is no spatial evidence left for a
  head to localise. So the backbone here stops downsampling at stride 4 and
  never goes deeper.
* Receptive field still has to grow, or the network cannot tell a sperm head
  from any other small dark blob. Dilated convolutions buy that context at
  constant resolution, which is the trade the original network makes with its
  shallow multi-scale stack.
* Depth is spent on width and context at high resolution rather than on more
  stages.

The anchor set is dropped in favour of the shared anchor-free head from
:mod:`.heads` -- see that module for why anchors are the wrong tool when the
object's scale distribution is effectively a point. That makes this network
directly comparable with :mod:`.p2net`, which is the reason both exist.

Any accuracy number produced by this class is a property of *this* code and its
training run, and must never be reported as a reproduction of the published
TOD-CNN results.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..config import DetectionConfig
from .heads import CenterNetHead, ConvNormAct
from .torch_base import TorchDetectorBase

__all__ = ["TodCnnDetector", "TodCnnNet"]


class TodCnnNet(nn.Module):
    """Shallow, high-resolution backbone plus the shared anchor-free head.

    Layout at the default ``stride=4``::

        input   1 x H x W
        stem    w x H x W                    (two 3x3, no downsampling)
        down1   2w x H/2 x W/2
        down2   4w x H/4 x W/4
        context 4w x H/4 x W/4               (dilations 1, 2, 4)
        fuse    4w x H/4 x W/4               (context + down1, concatenated)
        head    (C + 2 + 2) x H/4 x W/4

    The lateral path from ``down1`` is not decoration: the stride-2 map is the
    last place a 10 px object still has ~5 px of spatial support, and merging it
    back in gives the head evidence that the dilated stack has already blurred.

    Parameters
    ----------
    in_channels
        1 for the native monochrome frame. Kept configurable only so that a
        pre-stacked temporal input (N frames as N channels) can be tried
        without touching this file; it is **not** an RGB switch.
    width
        Base channel count. 32 is roughly the smallest that trains stably on
        the synthetic source; 16 runs comfortably on CPU for tests.
    num_classes
        Heatmap channels. Two (sperm, impurity) mirrors the original dataset's
        label set; the default here is one because the rest of this pipeline
        counts sperm and treats debris as a false positive to be measured.
    stride
        Total downsampling, 2 or 4. Never more -- that is the entire premise.
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        num_classes: int = 1,
        stride: int = 4,
        head_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4),
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        if stride not in (2, 4):
            raise ValueError(
                f"TodCnnNet stride must be 2 or 4, got {stride}: aggressive "
                "downsampling is the failure mode this architecture exists to "
                "avoid"
            )
        if width <= 0 or head_channels <= 0:
            raise ValueError("width and head_channels must be positive")
        if not dilations:
            raise ValueError("at least one dilation is required")

        self.stride = int(stride)
        self.num_classes = int(num_classes)
        self.width = int(width)

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, width, 3),
            ConvNormAct(width, width, 3),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvNormAct(width, 2 * width, 3),
            ConvNormAct(2 * width, 2 * width, 3),
        )
        if stride == 4:
            self.down2 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvNormAct(2 * width, 4 * width, 3),
                ConvNormAct(4 * width, 4 * width, 3),
            )
            # The lateral has to be brought to the head's resolution; average
            # pooling rather than max preserves the mean intensity of a faint
            # object instead of latching onto whichever pixel is brightest.
            self.lateral = nn.Sequential(
                nn.AvgPool2d(2, 2), ConvNormAct(2 * width, 2 * width, 1)
            )
        else:
            self.down2 = nn.Sequential(
                ConvNormAct(2 * width, 4 * width, 3),
                ConvNormAct(4 * width, 4 * width, 3),
            )
            self.lateral = ConvNormAct(2 * width, 2 * width, 1)

        # Context stack at constant resolution. Each dilation roughly doubles
        # the receptive field for the cost of one 3x3, which is how a stride-4
        # network sees enough of the field to reject debris.
        self.context = nn.Sequential(
            *[
                ConvNormAct(4 * width, 4 * width, 3, dilation=d)
                for d in dilations
            ]
        )
        self.fuse = ConvNormAct(6 * width, 4 * width, 1)
        self.head = CenterNetHead(
            in_channels=4 * width,
            num_classes=self.num_classes,
            hidden_channels=head_channels,
            prior_prob=prior_prob,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        stem = self.stem(x)
        level2 = self.down1(stem)
        level4 = self.down2(level2)
        context = self.context(level4)
        fused = self.fuse(torch.cat([context, self.lateral(level2)], dim=1))
        return self.head(fused)


class TodCnnDetector(TorchDetectorBase):
    """:class:`~.base.Detector` wrapper around :class:`TodCnnNet`.

    All geometry handling -- resize, padding, tiling, mapping boxes back to
    source-frame pixels -- comes from :class:`~.torch_base.TorchDetectorBase`,
    so this class is only architecture selection and sizing.
    """

    def __init__(
        self,
        cfg: DetectionConfig | None = None,
        *,
        width: int = 32,
        stride: int = 4,
        head_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4),
        in_channels: int = 1,
        net: nn.Module | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else DetectionConfig(architecture="todcnn")
        network = net if net is not None else TodCnnNet(
            in_channels=in_channels,
            width=width,
            num_classes=len(cfg.class_names),
            stride=stride,
            head_channels=head_channels,
            dilations=dilations,
        )
        super().__init__(
            network,
            cfg,
            name="todcnn",
            stride=stride,
            # Only the two max-pools constrain the input size, so a stride-4
            # network needs a multiple of 4 and nothing more. Padding to 32
            # "to be safe" would add up to 31 wasted rows on every frame.
            size_divisor=stride,
        )
        self.width = width
        self.head_channels = head_channels
        self.dilations = tuple(dilations)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "width": self.width,
                "head_channels": self.head_channels,
                "dilations": list(self.dilations),
                "reimplementation_of": "arXiv:2204.08166 (concept only)",
                "original_weights_used": False,
            }
        )
        return info
