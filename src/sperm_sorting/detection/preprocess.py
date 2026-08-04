"""Input geometry: frame in, network-ready batch out.

Split out of any one detector because all three inference backends -- torch,
ONNX Runtime and (eventually) TensorRT -- must agree on it exactly. If the
torch path normalised by 255 and the ONNX path normalised by the frame maximum,
an exported model would score differently from the model it was exported from,
and the discrepancy would look like an export bug rather than a preprocessing
one.

These functions are the forward half of a pair; :func:`postprocess.scale_boxes`
is the inverse. Whenever one changes, check the other.

No torch import here on purpose: the ONNX deployment target installs
``onnxruntime`` without ``torch``, and this module has to work there.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..errors import InferenceError

__all__ = [
    "pad_to_divisor",
    "prepare_input",
    "resize_long_side",
    "resize_to",
    "round_up",
    "to_float_gray",
]


def round_up(value: int, divisor: int) -> int:
    """Smallest multiple of ``divisor`` that is >= ``value``."""
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return int(-(-int(value) // int(divisor)) * int(divisor))


def to_float_gray(image: np.ndarray) -> np.ndarray:
    """Normalise any accepted input layout to float32 ``(H, W)`` in ``[0, 1]``.

    Integer inputs are divided by the full range of their dtype, never by their
    own maximum. Per-frame max-normalisation would make the network's input
    depend on whether a single bright speck happened to be in view this frame --
    a hidden, frame-to-frame gain change that shows up downstream as
    unexplained score drift.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] == 3:
            # The target camera is monochrome and has no colour filter array, so
            # a 3-channel frame can only be a replicated or false-coloured mono
            # image. It is collapsed rather than treated as carrying colour.
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        else:
            raise InferenceError(
                f"unsupported image with {array.shape[2]} channels; this "
                "pipeline is monochrome"
            )
    elif array.ndim != 2:
        raise InferenceError(f"expected a 2-D frame, got shape {array.shape}")

    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if array.dtype == bool:
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return (array.astype(np.float32) - info.min) / float(info.max - info.min)
    # Floating input is assumed already normalised; the clip guards against a
    # caller that handed over raw sensor floats.
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def _interpolation(scale: float) -> int:
    """``INTER_AREA`` when shrinking, ``INTER_LINEAR`` when growing.

    ``INTER_AREA`` integrates over the source pixels instead of point-sampling
    them, so a 10 px object being downscaled is dimmed smoothly rather than
    being randomly kept or dropped depending on where the sample lands. For
    objects this small that is the difference between a detectable blob and
    nothing.
    """
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR


def resize_long_side(
    gray: np.ndarray, input_size: int | None
) -> tuple[np.ndarray, tuple[int, int]]:
    """Scale so the longer side equals ``input_size``, preserving aspect ratio.

    ``input_size=None`` -- the pipeline default -- keeps native resolution,
    which is the right choice for objects a few pixels across: a 2x downscale of
    a 10 px sperm head throws away most of the evidence the head is trained on.

    Returns ``(resized, (H, W))`` where the shape is the *content* shape, i.e.
    before any padding.
    """
    height, width = gray.shape[:2]
    if input_size is None or max(height, width) == int(input_size):
        return gray, (height, width)
    if int(input_size) <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")

    scale = float(input_size) / float(max(height, width))
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=_interpolation(scale))
    return resized, (new_h, new_w)


def resize_to(
    gray: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to an exact ``(H, W)``, distorting the aspect ratio if required.

    Only used when an imported ONNX graph declares static spatial dimensions and
    leaves no choice. The distortion is safe because
    :func:`postprocess.scale_boxes` applies independent x and y factors, so the
    inverse is exact -- but it does mean a circular sperm head becomes elliptical
    in the network's view, which is worth knowing when such a model
    underperforms.
    """
    target_h, target_w = int(shape[0]), int(shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target shape must be positive, got {shape}")
    height, width = gray.shape[:2]
    if (height, width) == (target_h, target_w):
        return gray, (height, width)
    scale = min(target_h / height, target_w / width)
    resized = cv2.resize(
        gray, (target_w, target_h), interpolation=_interpolation(scale)
    )
    return resized, (target_h, target_w)


def pad_to_divisor(gray: np.ndarray, divisor: int) -> np.ndarray:
    """Pad the right and bottom edges so the shape divides evenly by ``divisor``.

    Padding goes on the right/bottom only, never centred. A bottom-right pad
    leaves every source coordinate unchanged, so there is no offset to subtract
    when mapping boxes back -- an entire class of off-by-pad bugs simply cannot
    occur.

    ``BORDER_REPLICATE`` rather than a constant fill: zero-padding a brightfield
    frame (dark objects on a bright field) manufactures a hard bright/dark edge,
    and a centre-heatmap head will happily fire on it.
    """
    if divisor <= 1:
        return gray
    height, width = gray.shape[:2]
    pad_h = round_up(height, divisor) - height
    pad_w = round_up(width, divisor) - width
    if pad_h == 0 and pad_w == 0:
        return gray
    return cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)


def prepare_input(
    image: np.ndarray,
    input_size: int | None = None,
    size_divisor: int = 1,
    channels: int = 1,
    target_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Full frame-to-batch conversion.

    Parameters
    ----------
    image
        Source frame, any accepted layout/dtype.
    input_size
        Long-side resize target, or ``None`` for native resolution. Ignored when
        ``target_shape`` is given.
    size_divisor
        Pad the result up to a multiple of this.
    channels
        Network input channels. 1 is native. **3 replicates the single grey
        channel three times** -- this is only for reusing an ImageNet-stemmed
        model and adds no information; it is done explicitly here rather than
        being hidden in a backbone so that the cost is visible.
    target_shape
        Exact ``(H, W)`` to resize to, for graphs with static input dimensions.

    Returns
    -------
    tuple
        ``(batch, content_shape, source_shape)`` where ``batch`` is
        ``(1, channels, H, W)`` float32, ``content_shape`` is the pre-padding
        ``(H, W)`` to map boxes out of, and ``source_shape`` is the original
        frame's ``(H, W)``.
    """
    gray = to_float_gray(image)
    source_shape = (gray.shape[0], gray.shape[1])
    if target_shape is not None:
        resized, content_shape = resize_to(gray, target_shape)
    else:
        resized, content_shape = resize_long_side(gray, input_size)
    padded = pad_to_divisor(resized, size_divisor)

    batch = np.ascontiguousarray(padded, dtype=np.float32)[None, None, :, :]
    if channels == 3:
        batch = np.repeat(batch, 3, axis=1)
    elif channels != 1:
        raise InferenceError(
            f"model expects {channels} input channels; only 1 (native "
            "monochrome) and 3 (explicitly replicated) are supported"
        )
    return batch, content_shape, source_shape
