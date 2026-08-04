"""P2Net: an FPN detector whose finest -- and only -- prediction level is P2.

The architecture is one decision, and this is it: **the head lives at stride 4
and nowhere else.**

A conventional FPN detector predicts on P3-P7 (strides 8 to 128) because it is
built for objects spanning two orders of magnitude of scale. This problem has
the opposite shape. A sperm head at the configured optics is a roughly 10-20 px
blob, and the scale distribution is narrow enough that every object in every
frame falls into a single FPN level. Predicting on P3-P7 would therefore spend
five heads to have four of them see nothing, while the one level that matters
is the one a standard configuration omits.

So the pyramid is built downward to stride 32 -- the deep, coarse levels are
still needed, because that is where the context that separates a sperm head
from a debris speck lives -- but the top-down path carries that context back up
to P2 and the head is attached there, at the resolution where the objects
actually are.

Compared with :mod:`.todcnn`, which reaches high resolution by *never*
downsampling, this reaches it by downsampling and then coming back. That gives
a much larger receptive field per unit of compute at the cost of a lossy
round-trip. Which trade wins is an empirical question, which is why both are
implemented against the identical head and the identical pre/post-processing.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from ..config import DetectionConfig
from .heads import CenterNetHead, ConvNormAct, group_count
from .torch_base import TorchDetectorBase

__all__ = ["BasicBlock", "MobileBlock", "P2Net", "P2NetDetector"]


class BasicBlock(nn.Module):
    """ResNet basic block with GroupNorm.

    The residual connection matters more than usual here: the top-down path has
    to preserve a signal that is only a few pixels wide across four stages of
    downsampling, and a plain conv stack attenuates exactly that.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, stride=stride, groups=groups)
        self.conv2 = ConvNormAct(
            out_channels, out_channels, 3, groups=groups, activation=False
        )
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvNormAct(
                in_channels, out_channels, 1, stride=stride, groups=groups, activation=False
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.act(self.conv2(self.conv1(x)) + identity)


class MobileBlock(nn.Module):
    """Inverted-residual block: expand 1x1, depthwise 3x3, project 1x1.

    Offered as an alternative to :class:`BasicBlock` because the CPU inference
    path in the tests and on a developer machine is dominated by the stride-4
    tensors, where a depthwise convolution is several times cheaper than a dense
    one for a similar receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        groups: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(out_channels, in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = ConvNormAct(in_channels, hidden, 1, groups=groups)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False
            ),
            nn.GroupNorm(group_count(hidden, groups), hidden),
            nn.ReLU(inplace=True),
        )
        self.project = ConvNormAct(hidden, out_channels, 1, groups=groups, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.project(self.depthwise(self.expand(x)))
        return out + x if self.use_residual else out


_BLOCKS: dict[str, type[nn.Module]] = {"basic": BasicBlock, "mobile": MobileBlock}


class P2Net(nn.Module):
    """Encoder + top-down FPN + one anchor-free head at P2 (stride 4).

    Parameters
    ----------
    in_channels
        1 by default: the frame is monochrome and stays monochrome. There is no
        ImageNet stem here to satisfy, so nothing is replicated to three
        channels anywhere in this file.
    width
        Base channel count; stage widths are ``width * (1, 2, 4, 8)`` truncated
        to ``num_stages``.
    num_stages
        Encoder stages after the stride-2 stem, so the coarsest level is stride
        ``2 ** (num_stages + 1)``. Four gives the usual C2-C5.
    blocks_per_stage
        Depth per stage. ``(1, 1, 1, 1)`` is a genuinely small network that runs
        on CPU in a unit test; ``(2, 2, 2, 2)`` is a sensible training default.
    fpn_channels
        Channel count every pyramid level is projected to.
    block
        ``"basic"`` (ResNet-ish) or ``"mobile"`` (depthwise-separable).
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        num_classes: int = 1,
        num_stages: int = 4,
        blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1),
        fpn_channels: int = 64,
        head_channels: int = 64,
        block: str = "basic",
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        if not 2 <= num_stages <= 4:
            raise ValueError(f"num_stages must lie in [2, 4], got {num_stages}")
        if width <= 0 or fpn_channels <= 0 or head_channels <= 0:
            raise ValueError("channel counts must be positive")
        if block not in _BLOCKS:
            raise ValueError(f"unknown block '{block}', expected one of {sorted(_BLOCKS)}")
        blocks_per_stage = tuple(blocks_per_stage)[:num_stages]
        if len(blocks_per_stage) != num_stages or any(b < 1 for b in blocks_per_stage):
            raise ValueError(
                f"blocks_per_stage must give a positive depth for each of the "
                f"{num_stages} stages, got {blocks_per_stage}"
            )

        block_cls = _BLOCKS[block]
        self.num_stages = int(num_stages)
        self.num_classes = int(num_classes)
        #: Stride of the level the head is attached to. Fixed at 4 by design.
        self.head_stride = 4
        #: Input must be a multiple of this for the top-down adds to line up.
        self.size_divisor = 2 ** (num_stages + 1)

        # Stem takes the input to stride 2. Two 3x3s rather than one 7x7: the
        # receptive field is comparable, the parameter count is lower, and the
        # first layer never sees more than a 3x3 neighbourhood of a 10 px
        # object, which is all the detail there is.
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, width, 3, stride=2),
            ConvNormAct(width, width, 3),
        )

        stage_widths = [width * (2**i) for i in range(num_stages)]
        self.stages = nn.ModuleList()
        prev = width
        for stage_index, (out_ch, depth) in enumerate(
            zip(stage_widths, blocks_per_stage, strict=True)
        ):
            layers: list[nn.Module] = [block_cls(prev, out_ch, stride=2)]
            layers.extend(block_cls(out_ch, out_ch, stride=1) for _ in range(depth - 1))
            self.stages.append(nn.Sequential(*layers))
            prev = out_ch
            del stage_index

        # 1x1 laterals project every level to a common width so the top-down
        # additions are well defined.
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, fpn_channels, 1) for c in stage_widths]
        )
        # One 3x3 smoothing conv per level; on P2 it also removes the aliasing
        # the nearest-neighbour upsample introduces, which otherwise shows up
        # as a checkerboard in the heatmap.
        self.smooth = nn.ModuleList(
            [ConvNormAct(fpn_channels, fpn_channels, 3) for _ in stage_widths]
        )
        self.head = CenterNetHead(
            in_channels=fpn_channels,
            num_classes=self.num_classes,
            hidden_channels=head_channels,
            prior_prob=prior_prob,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features: list[Tensor] = []
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
            features.append(out)

        # Top-down: start at the coarsest level and walk back up to P2.
        laterals = [lateral(f) for lateral, f in zip(self.laterals, features, strict=True)]
        for index in range(len(laterals) - 2, -1, -1):
            # Explicit `size=` rather than `scale_factor=2`: an odd spatial dim
            # cannot happen given `size_divisor`, but an explicit size makes the
            # add safe even if that guarantee is ever loosened.
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1], size=laterals[index].shape[-2:], mode="nearest"
            )
        p2 = self.smooth[0](laterals[0])
        return self.head(p2)


class P2NetDetector(TorchDetectorBase):
    """:class:`~.base.Detector` wrapper around :class:`P2Net`."""

    def __init__(
        self,
        cfg: DetectionConfig | None = None,
        *,
        width: int = 32,
        num_stages: int = 4,
        blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1),
        fpn_channels: int = 64,
        head_channels: int = 64,
        block: str = "basic",
        in_channels: int = 1,
        net: nn.Module | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else DetectionConfig(architecture="p2net")
        network = net if net is not None else P2Net(
            in_channels=in_channels,
            width=width,
            num_classes=len(cfg.class_names),
            num_stages=num_stages,
            blocks_per_stage=blocks_per_stage,
            fpn_channels=fpn_channels,
            head_channels=head_channels,
            block=block,
        )
        super().__init__(
            network,
            cfg,
            name="p2net",
            stride=int(getattr(network, "head_stride", 4)),
            size_divisor=int(getattr(network, "size_divisor", 2 ** (num_stages + 1))),
        )
        self.width = width
        self.num_stages = num_stages
        self.blocks_per_stage = tuple(blocks_per_stage)[:num_stages]
        self.fpn_channels = fpn_channels
        self.block = block

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "width": self.width,
                "num_stages": self.num_stages,
                "blocks_per_stage": list(self.blocks_per_stage),
                "fpn_channels": self.fpn_channels,
                "block": self.block,
                "prediction_level": "P2",
            }
        )
        return info
