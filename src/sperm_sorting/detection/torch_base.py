"""Shared machinery for the torch-backed detectors.

Both :class:`~.todcnn.TodCnnDetector` and :class:`~.p2net.P2NetDetector` differ
only in their feature extractor. Everything else -- greyscale normalisation,
optional resize, padding to the backbone's size divisor, tiled inference, the
latency guard, undoing all of that geometry on the way out, weight loading and
backend selection -- is identical and lives here exactly once. Duplicating it
would make the two architectures incomparable: any measured difference could
be a difference in preprocessing rather than in the model.

The coordinate contract is enforced in one place, :meth:`TorchDetectorBase.detect`:
boxes enter the world in padded network-input pixels and leave it in
source-frame pixels, and no subclass is given the opportunity to get that
wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from ..config import DetectionConfig
from ..errors import InferenceError
from ..schemas.detection import Detection
from ..schemas.frame import FramePacket
from .base import Detector
from .postprocess import (
    arrays_to_detections,
    clip_boxes,
    compute_tile_grid,
    decode_centernet_heatmap,
    filter_boxes_by_size,
    finalise_boxes,
    merge_tiled_detections,
    scale_boxes,
)
from .preprocess import pad_to_divisor, resize_long_side, round_up, to_float_gray

__all__ = ["TorchDetectorBase", "load_state_dict_from_checkpoint", "resolve_device"]

logger = logging.getLogger(__name__)

#: Tiles are pushed through the network in chunks of this many at a time.
#: A 1920x1200 frame at 640/96 tiling is 12 tiles; running all twelve as one
#: batch on a stride-4, 64-channel feature map allocates well over a gigabyte,
#: which on the target box competes with the camera's DMA buffers.
_DEFAULT_TILE_BATCH = 4


def resolve_device(spec: str) -> torch.device:
    """Turn a config device string into a device that actually exists.

    Silently falling back from CUDA to CPU would turn a 2 ms inference into a
    200 ms one with no visible cause, so the fallback is logged loudly. It is
    still a fallback rather than an error because a CPU-only developer machine
    must be able to run the same config as the device.
    """
    try:
        device = torch.device(spec)
    except (RuntimeError, ValueError) as exc:
        raise InferenceError(
            f"backend.device '{spec}' is not a valid torch device: {exc}"
        ) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "backend.device='%s' requested but no CUDA device is available; "
            "falling back to CPU. Inference will be far slower than the "
            "real-time budget assumes.",
            spec,
        )
        return torch.device("cpu")
    return device


def load_state_dict_from_checkpoint(
    module: nn.Module, path: str | Path, strict: bool = True
) -> dict[str, Any]:
    """Load weights into ``module``, accepting the usual checkpoint layouts.

    ``weights_only=True`` is passed to :func:`torch.load` deliberately: a
    checkpoint is data, and a pickle that can execute arbitrary code on load is
    not something a device that ingests operator-supplied model files should
    accept.

    Returns whatever non-tensor metadata the checkpoint carried (epoch, config
    hash, provenance tag), so the caller can stamp it into the audit log.
    """
    path = Path(path)
    if not path.exists():
        raise InferenceError(f"detection.weights file not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InferenceError(f"could not read checkpoint {path}: {exc}") from exc

    metadata: dict[str, Any] = {}
    state: Any = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                metadata = {
                    k: v
                    for k, v in payload.items()
                    if k != key and not isinstance(v, (dict, torch.Tensor))
                }
                break
    if not isinstance(state, dict):
        raise InferenceError(
            f"checkpoint {path} does not contain a state dict "
            f"(got {type(state).__name__})"
        )

    # DistributedDataParallel prefixes every key; stripping it here means a
    # multi-GPU training run's checkpoint loads on the single-GPU device
    # without a conversion step that someone will forget.
    cleaned = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    try:
        module.load_state_dict(cleaned, strict=strict)
    except RuntimeError as exc:
        raise InferenceError(
            f"checkpoint {path} does not match this architecture: {exc}"
        ) from exc
    return metadata


class TorchDetectorBase(Detector):
    """Common inference path for the in-repo torch detectors.

    Subclasses supply a network whose ``forward`` returns the
    :class:`~.heads.CenterNetHead` output dict, plus the network's output
    ``stride`` and the ``size_divisor`` its downsampling requires.
    """

    def __init__(
        self,
        net: nn.Module,
        cfg: DetectionConfig,
        *,
        name: str,
        stride: int,
        size_divisor: int,
        tile_batch: int = _DEFAULT_TILE_BATCH,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        if size_divisor <= 0:
            raise ValueError("size_divisor must be positive")

        self.name = name
        self.cfg = cfg
        self.class_names = tuple(cfg.class_names)
        self.stride = int(stride)
        self.size_divisor = int(size_divisor)
        self.tile_batch = max(1, int(tile_batch))

        backend = cfg.backend
        self.device = resolve_device(backend.device)
        if backend.num_threads is not None and self.device.type == "cpu":
            # Process-global by nature. Set here rather than at import time so
            # that a config that never builds a torch detector never touches
            # the ambient thread pool.
            torch.set_num_threads(int(backend.num_threads))

        self.net = net.to(self.device).eval()
        self.fp16 = False
        if backend.fp16:
            if self.device.type == "cuda":
                self.net = self.net.half()
                self.fp16 = True
            else:
                logger.warning(
                    "backend.fp16=true is ignored on device '%s': half-precision "
                    "convolution on CPU is slower than fp32, not faster.",
                    self.device,
                )

        self._checkpoint_metadata: dict[str, Any] = {}
        self._tiling_disabled = False
        self._tiling_warned = False
        self._latency_guard_enabled = True
        self._closed = False

    # ------------------------------------------------------------------ setup

    def load_weights(self, path: str | Path, strict: bool = True) -> None:
        """Load a checkpoint into the network and move it back to the device."""
        self._checkpoint_metadata = load_state_dict_from_checkpoint(
            self.net, path, strict=strict
        )
        self.net = self.net.to(self.device).eval()
        if self.fp16:
            self.net = self.net.half()

    def reset_tiling(self) -> None:
        """Re-enable tiling after the latency guard tripped.

        Exposed for tests and for an operator-triggered re-arm; the guard is
        otherwise one-way within a session by design, because a detector that
        oscillates between tiled and whole-frame inference produces two
        different recall regimes inside one shot.
        """
        self._tiling_disabled = False
        self._tiling_warned = False

    def warmup(self, height: int, width: int, iterations: int = 3) -> None:
        """Warm the kernels without letting cold-start latency kill tiling.

        The first inference pays for lazy allocator growth and, on CUDA, for
        context creation and autotuning. Judging the tiling budget on that
        measurement would disable tiling on every run, permanently, for a cost
        that never recurs.
        """
        self._latency_guard_enabled = False
        try:
            super().warmup(height, width, iterations)
        finally:
            self._latency_guard_enabled = True

    # -------------------------------------------------------------- inference

    def detect(self, frame: FramePacket) -> list[Detection]:
        """Detect in one frame, returning boxes in source-frame pixels."""
        if self._closed:
            raise InferenceError(f"{self.name}: detect() called after close()")

        image = frame.image
        if image is None or getattr(image, "size", 0) == 0:
            return []
        gray = to_float_gray(image)
        source_shape = (gray.shape[0], gray.shape[1])
        if source_shape[0] < 1 or source_shape[1] < 1:
            return []

        tiles = self._tile_plan(source_shape)
        if tiles is None:
            boxes, scores, class_ids = self._detect_whole_frame(gray, source_shape)
        else:
            started = perf_counter()
            boxes, scores, class_ids = self._detect_tiled(gray, source_shape, tiles)
            self._check_tiling_budget((perf_counter() - started) * 1000.0, len(tiles))

        return self._to_detections(boxes, scores, class_ids, frame)

    def _tile_plan(
        self, source_shape: tuple[int, int]
    ) -> list[tuple[int, int, int, int]] | None:
        """Tile rectangles for this frame, or ``None`` to run whole-frame."""
        tiling = self.cfg.tiling
        if not tiling.enabled or self._tiling_disabled:
            return None
        height, width = source_shape
        if height <= tiling.tile_size and width <= tiling.tile_size:
            # A single tile is whole-frame inference with extra padding and an
            # extra merge; skip the ceremony.
            return None
        tiles = compute_tile_grid(height, width, tiling.tile_size, tiling.overlap)
        return tiles if len(tiles) > 1 else None

    def _check_tiling_budget(self, elapsed_ms: float, n_tiles: int) -> None:
        """Disable tiling once it demonstrably cannot meet the frame budget.

        The measurement can only be taken *after* the fact, so the frame that
        trips the guard has already paid the cost; re-running it whole-frame
        would double that. The guard therefore protects every subsequent frame
        and accepts one overrun, which is the cheapest correct behaviour.
        """
        if not self._latency_guard_enabled or self._tiling_disabled:
            return
        budget = self.cfg.tiling.max_latency_ms
        if budget <= 0.0 or elapsed_ms <= budget:
            return
        self._tiling_disabled = True
        if not self._tiling_warned:
            self._tiling_warned = True
            logger.warning(
                "%s: tiled inference took %.1f ms over %d tiles, exceeding the "
                "%.1f ms budget in detection.tiling.max_latency_ms; falling back "
                "to whole-frame inference for the rest of this session. Small "
                "objects may be missed -- lower detection.tiling.tile_size, "
                "raise the budget, or move to a faster backend.",
                self.name,
                elapsed_ms,
                n_tiles,
                budget,
            )

    def _detect_whole_frame(
        self, gray: np.ndarray, source_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resized, content_shape = resize_long_side(gray, self.cfg.input_size)
        padded = pad_to_divisor(resized, self.size_divisor)
        # (H, W) -> (1, 1, H, W): batch of one, single grey channel.
        outputs = self._forward(padded[None, None, ...])

        boxes, scores, class_ids = decode_centernet_heatmap(
            outputs["heatmap"][0],
            outputs["size"][0],
            outputs["offset"][0],
            stride=self.stride,
            score_threshold=self.cfg.score_threshold,
            max_detections=self.cfg.max_detections,
        )
        # Clip to the *content* area first: anything beyond it was invented by
        # the padding and has no source-frame preimage.
        boxes = clip_boxes(boxes, content_shape)
        boxes = scale_boxes(boxes, content_shape, source_shape)
        return self._finalise(boxes, scores, class_ids, source_shape)

    def _detect_tiled(
        self,
        gray: np.ndarray,
        source_shape: tuple[int, int],
        tiles: list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the network on overlapping native-resolution crops.

        Note that tiling deliberately ignores ``input_size``. Tiling exists to
        keep a few-pixel object at its native scale; downscaling the frame and
        then tiling it would defeat the only reason to pay for tiling.
        """
        # Every tile is padded to one common shape so the whole grid can go
        # through the network as batches rather than one forward per tile.
        max_tile_h = max(y1 - y0 for _, y0, _, y1 in tiles)
        max_tile_w = max(x1 - x0 for x0, _, x1, _ in tiles)
        pad_h = round_up(max_tile_h, self.size_divisor)
        pad_w = round_up(max_tile_w, self.size_divisor)

        crops: list[np.ndarray] = []
        contents: list[tuple[int, int]] = []
        origins: list[tuple[int, int]] = []
        for x0, y0, x1, y1 in tiles:
            crop = gray[y0:y1, x0:x1]
            contents.append((crop.shape[0], crop.shape[1]))
            origins.append((x0, y0))
            crops.append(
                cv2.copyMakeBorder(
                    crop,
                    0,
                    pad_h - crop.shape[0],
                    0,
                    pad_w - crop.shape[1],
                    cv2.BORDER_REPLICATE,
                )
                if (crop.shape[0] != pad_h or crop.shape[1] != pad_w)
                else crop
            )

        tile_boxes: list[np.ndarray] = []
        tile_scores: list[np.ndarray] = []
        tile_classes: list[np.ndarray] = []
        for start in range(0, len(crops), self.tile_batch):
            chunk = crops[start : start + self.tile_batch]
            batch = np.stack([np.ascontiguousarray(c) for c in chunk], axis=0)[:, None]
            outputs = self._forward(batch)
            for local_index in range(batch.shape[0]):
                boxes, scores, class_ids = decode_centernet_heatmap(
                    outputs["heatmap"][local_index],
                    outputs["size"][local_index],
                    outputs["offset"][local_index],
                    stride=self.stride,
                    score_threshold=self.cfg.score_threshold,
                    max_detections=self.cfg.max_detections,
                )
                content = contents[start + local_index]
                boxes = clip_boxes(boxes, content)
                # Filter per tile so that obvious noise never reaches the
                # cross-seam suppression, which is quadratic in box count.
                boxes, scores, keep = filter_boxes_by_size(
                    boxes, scores, self.cfg.min_box_size_px, self.cfg.max_box_size_px
                )
                tile_boxes.append(boxes)
                tile_scores.append(scores)
                tile_classes.append(class_ids[keep])

        return merge_tiled_detections(
            tile_boxes,
            tile_scores,
            tile_classes,
            origins,
            iou_threshold=self.cfg.nms_iou_threshold,
            source_shape=source_shape,
            max_detections=self.cfg.max_detections,
        )

    def _forward(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        """Run the network on an ``(B, 1, H, W)`` float32 batch."""
        tensor: Tensor = torch.from_numpy(np.ascontiguousarray(batch))
        tensor = tensor.to(self.device)
        if self.fp16:
            tensor = tensor.half()
        with torch.inference_mode():
            outputs = self.net(tensor)
        return {
            key: outputs[key].detach().float().cpu().numpy()
            for key in ("heatmap", "size", "offset")
        }

    # ------------------------------------------------------------ finalising

    def _finalise(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        source_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return finalise_boxes(
            boxes,
            scores,
            class_ids,
            source_shape,
            min_box_size=self.cfg.min_box_size_px,
            max_box_size=self.cfg.max_box_size_px,
            iou_threshold=self.cfg.nms_iou_threshold,
            max_detections=self.cfg.max_detections,
        )

    def _to_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        frame: FramePacket,
    ) -> list[Detection]:
        return arrays_to_detections(boxes, scores, class_ids, frame, self.class_names)

    # ------------------------------------------------------------------ admin

    def close(self) -> None:
        """Drop the network and free any CUDA cache it held."""
        if self._closed:
            return
        self._closed = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "architecture": self.cfg.architecture,
                "backend": "torch",
                "device": str(self.device),
                "fp16": self.fp16,
                "stride": self.stride,
                "input_size": self.cfg.input_size,
                "score_threshold": self.cfg.score_threshold,
                "nms_iou_threshold": self.cfg.nms_iou_threshold,
                "tiling_enabled": bool(self.cfg.tiling.enabled),
                "tiling_disabled_by_latency": self._tiling_disabled,
                "parameters": int(sum(p.numel() for p in self.net.parameters())),
                "checkpoint": self._checkpoint_metadata or None,
            }
        )
        return info
