"""Grayscale-native feature extractors for the morphology network.

Every builder returns ``(module, feature_dim)`` where ``module`` maps a
``(B, C, H, W)`` float tensor to a pooled ``(B, feature_dim)`` feature vector.
Pooling is done inside the backbone (global average pooling) so that the heads
in :mod:`.model` never need to know the spatial geometry of the trunk, and so
that a crop size other than the configured one still produces a valid feature
vector instead of a shape error deep in a linear layer.

Grayscale, not RGB
------------------
The microscope is monochrome (``Mono8`` on the Basler; see
:class:`~sperm_sorting.config.BaslerConfig`). Feeding a replicated 3-channel
image to an ImageNet stem wastes two thirds of the first convolution's FLOPs on
identical data at every frame, at 160 FPS. Every builder therefore takes
``in_channels`` and rebuilds the stem.

When ``pretrained=True`` the RGB kernel of the first convolution is **summed**
across the input-channel axis rather than re-initialised or averaged:

    ``W_gray[o, 0, i, j] = W_rgb[o, 0, i, j] + W_rgb[o, 1, i, j] + W_rgb[o, 2, i, j]``

Why summing is the right move. For a grayscale image ``g`` presented as
``R = G = B = g``, the pretrained convolution computes
``sum_c W_c * g == (sum_c W_c) * g``. Summing therefore reproduces the exact
response the pretrained filter would have given on a gray image, so every
downstream BatchNorm running statistic and every learned filter further up the
trunk remains valid. Averaging instead scales every activation by 1/3, which
puts the first BatchNorm's inputs a factor of three below the statistics it was
trained with; re-initialising throws the pretraining away entirely, which is
the one thing ``pretrained=True`` was asked to avoid.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from ..errors import ConfigurationError

__all__ = [
    "BACKBONE_BUILDERS",
    "GlobalPoolBackbone",
    "SimpleCNN",
    "adapt_conv_in_channels",
    "available_backbones",
    "build_backbone",
    "build_efficientnet_b0",
    "build_mobilenetv3_small",
    "build_simplecnn",
]


# ==========================================================================
# First-convolution surgery
# ==========================================================================


def adapt_conv_in_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Return a copy of ``conv`` accepting ``in_channels`` inputs.

    The weight is carried over by summing the original input-channel axis and
    redistributing it across the new one (see the module docstring for why
    summing rather than averaging). For ``in_channels == 1`` this is a plain
    sum; for other counts the summed kernel is replicated and divided by the
    new channel count so that a constant-across-channels input still produces
    the original response.

    ``conv.bias`` (usually ``None`` in these stems) is copied verbatim, since it
    does not depend on the input channel count.
    """
    if conv.in_channels == in_channels:
        return conv

    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        dilation=conv.dilation,  # type: ignore[arg-type]
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        summed = conv.weight.sum(dim=1, keepdim=True)  # (O, 1, kH, kW)
        if in_channels == 1:
            new.weight.copy_(summed)
        else:
            new.weight.copy_(summed.repeat(1, in_channels, 1, 1) / float(in_channels))
        if conv.bias is not None and new.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _first_conv(net: nn.Module) -> tuple[str, nn.Conv2d]:
    """Locate the stem convolution by module-traversal order.

    Done by traversal rather than by hard-coded path (``features[0][0]``) so
    that a torchvision release which reshuffles its stem does not silently
    leave a 3-channel convolution in place -- it would raise here instead.
    """
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module
    raise ConfigurationError("backbone contains no Conv2d; cannot adapt its stem")


def _replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    """Set ``root.<qualified_name> = replacement``, including inside Sequentials."""
    parent_name, _, leaf = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, replacement)


def _retarget_stem(net: nn.Module, in_channels: int) -> None:
    """Rewrite the first convolution of ``net`` in place for ``in_channels``."""
    name, conv = _first_conv(net)
    _replace_module(net, name, adapt_conv_in_channels(conv, in_channels))


# ==========================================================================
# Wrappers
# ==========================================================================


class GlobalPoolBackbone(nn.Module):
    """Wrap a convolutional trunk so it emits a pooled feature vector.

    Global average pooling (rather than a flatten of a fixed grid) means the
    contract with :mod:`.model` is ``(B, feature_dim)`` for *any* input size, so
    changing ``crop.output_size`` cannot produce a shape error in the heads.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.feature_dim = int(feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, H, W)`` in, pooled ``(B, feature_dim)`` out."""
        features = self.trunk(x)
        if features.ndim != 4:
            raise ValueError(
                f"backbone trunk must emit (B, C, H, W); got shape {tuple(features.shape)}"
            )
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        return torch.flatten(pooled, 1)


class SimpleCNN(nn.Module):
    """Small from-scratch trunk: four stride-2 conv-BN-ReLU blocks.

    Exists so that unit tests, CI and the synthetic end-to-end run never touch
    the network to fetch torchvision weights, and so that a full forward pass on
    a 128x128 crop costs well under a millisecond on a CPU. It is a real, if
    modest, model -- not a stub -- and is a reasonable starting point for a
    128x128 single-channel input where ImageNet features transfer poorly
    anyway (phase-contrast microscopy looks nothing like ImageNet).
    """

    def __init__(self, in_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        if width < 1:
            raise ConfigurationError(f"simplecnn width must be >= 1, got {width}")
        widths = (width, width * 2, width * 4, width * 8)
        layers: list[nn.Module] = []
        prev = in_channels
        for out in widths:
            layers += [
                nn.Conv2d(prev, out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
                nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            prev = out
        self.blocks = nn.Sequential(*layers)
        self.feature_dim = widths[-1]

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, H, W)`` in, a ``(B, width*8, H/16, W/16)`` feature map out."""
        return self.blocks(x)


# ==========================================================================
# Builders
# ==========================================================================


def build_simplecnn(
    in_channels: int = 1, width: int = 32
) -> tuple[nn.Module, int]:
    """Build the from-scratch CPU-friendly trunk.

    Returns ``(module, feature_dim)`` with ``feature_dim == width * 8``.
    """
    trunk = SimpleCNN(in_channels=in_channels, width=width)
    return GlobalPoolBackbone(trunk, trunk.feature_dim), trunk.feature_dim


def _torchvision_features(
    factory_name: str, in_channels: int, pretrained: bool
) -> tuple[nn.Module, int]:
    """Shared body for the torchvision builders.

    torchvision is an *optional* dependency of this project (see the ``torch``
    extra in ``pyproject.toml``), so the import is local and its absence is
    reported as a configuration problem rather than an ``ImportError`` from the
    middle of model construction.
    """
    try:
        from torchvision import models as tv_models
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigurationError(
            f"backbone '{factory_name}' requires torchvision; install the "
            "'torch' extra, or use backbone='simplecnn' which has no such "
            "dependency"
        ) from exc

    factory: Callable[..., nn.Module] = getattr(tv_models, factory_name)
    # "DEFAULT" resolves to the current best ImageNet weights for the model and
    # triggers a download on first use; tests and CI always pass pretrained=False.
    net = factory(weights="DEFAULT" if pretrained else None)

    # Both mobilenetv3 and efficientnet expose their convolutional trunk here.
    trunk = getattr(net, "features", None)
    if not isinstance(trunk, nn.Module):
        raise ConfigurationError(
            f"torchvision model '{factory_name}' has no .features trunk; this "
            "build of torchvision is not supported"
        )
    feature_dim = _infer_feature_dim(net)
    _retarget_stem(trunk, in_channels)
    return GlobalPoolBackbone(trunk, feature_dim), feature_dim


def _infer_feature_dim(net: nn.Module) -> int:
    """Read the trunk's output width off the classifier torchvision attached.

    Reading it from the discarded classifier is more robust than hard-coding
    576 / 1280, which change between torchvision releases and between model
    variants.
    """
    classifier = getattr(net, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return int(classifier.in_features)
    if isinstance(classifier, nn.Sequential):
        for layer in classifier:
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
    raise ConfigurationError(
        "cannot determine the feature width of the torchvision backbone: no "
        "Linear layer found in its classifier"
    )


def build_mobilenetv3_small(
    in_channels: int = 1, pretrained: bool = False
) -> tuple[nn.Module, int]:
    """MobileNetV3-Small trunk, grayscale stem. ``feature_dim`` is 576.

    The default morphology backbone: it is the cheapest torchvision model that
    still carries useful ImageNet priors, which matters because morphology runs
    once per *track*, inside a hard per-track deadline
    (:attr:`~sperm_sorting.config.MorphologyConfig.deadline_ms`).
    """
    return _torchvision_features("mobilenet_v3_small", in_channels, pretrained)


def build_efficientnet_b0(
    in_channels: int = 1, pretrained: bool = False
) -> tuple[nn.Module, int]:
    """EfficientNet-B0 trunk, grayscale stem. ``feature_dim`` is 1280.

    Roughly 4x the FLOPs of MobileNetV3-Small; offered for offline evaluation
    of how much accuracy the cheap backbone gives up, not as a default.
    """
    return _torchvision_features("efficientnet_b0", in_channels, pretrained)


#: Name -> builder. The keys are exactly the literals accepted by
#: :attr:`sperm_sorting.config.MorphologyConfig.backbone`; a mismatch between
#: the two is caught by :func:`build_backbone` at startup rather than by a
#: ``KeyError`` at inference time.
BACKBONE_BUILDERS: dict[str, Callable[..., tuple[nn.Module, int]]] = {
    "mobilenetv3_small": build_mobilenetv3_small,
    "efficientnet_b0": build_efficientnet_b0,
    "simplecnn": build_simplecnn,
}


def available_backbones() -> tuple[str, ...]:
    """Backbone names this build can actually construct."""
    return tuple(BACKBONE_BUILDERS)


def build_backbone(
    name: str,
    in_channels: int = 1,
    *,
    pretrained: bool = False,
    width: int = 32,
) -> tuple[nn.Module, int]:
    """Dispatch to a builder by name.

    Parameters
    ----------
    name
        One of :func:`available_backbones`.
    in_channels
        Input channel count; 1 for the monochrome microscope.
    pretrained
        Load ImageNet weights (torchvision backbones only). Ignored by
        ``simplecnn``, which has no pretrained form -- passing ``True`` there is
        an error rather than a silent no-op, because silently getting random
        weights when you asked for pretrained ones is precisely the kind of
        thing that shows up as an unexplained accuracy drop weeks later.
    width
        Base channel count for ``simplecnn``.

    Returns
    -------
    ``(module, feature_dim)``.
    """
    if name not in BACKBONE_BUILDERS:
        raise ConfigurationError(
            f"unknown morphology backbone '{name}'; available: "
            f"{', '.join(available_backbones())}"
        )
    if in_channels < 1:
        raise ConfigurationError(f"in_channels must be >= 1, got {in_channels}")

    if name == "simplecnn":
        if pretrained:
            raise ConfigurationError(
                "backbone 'simplecnn' has no pretrained weights; either set "
                "pretrained=False or choose 'mobilenetv3_small'"
            )
        return build_simplecnn(in_channels=in_channels, width=width)
    return BACKBONE_BUILDERS[name](in_channels=in_channels, pretrained=pretrained)
