"""ONNX Runtime detector.

This is the deployment path. Torch is a training dependency; the device is
expected to ship an exported graph and ``onnxruntime`` and nothing else, which
is why this module imports neither :mod:`torch` nor anything that does.

Two output signatures are supported, and the choice is made from the model's
own metadata rather than from configuration:

``raw heads``
    Three tensors -- ``heatmap``, ``size``, ``offset`` -- exactly as
    :class:`~.heads.CenterNetHead` produces them. Decoding happens here, using
    the same :mod:`.postprocess` helpers the torch path uses, so an exported
    model and its source produce identical boxes.

``(N, 6) boxes``
    One tensor of ``x1, y1, x2, y2, score, class``. This is what every
    off-the-shelf exporter emits when NMS is folded into the graph, and
    supporting it is what lets a third-party detector be dropped in for
    comparison without writing a new class.

Guessing wrong here is silent and catastrophic -- a (N, 6) tensor read as a
heatmap decodes to garbage boxes with plausible-looking scores -- so the
signature detection below is deliberately conservative and raises rather than
falling back when it cannot be sure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DetectionConfig
from ..errors import BackendUnavailableError, InferenceError
from ..schemas.detection import Detection
from ..schemas.frame import FramePacket
from .base import Detector
from .postprocess import (
    arrays_to_detections,
    clip_boxes,
    decode_centernet_heatmap,
    finalise_boxes,
    scale_boxes,
)
from .preprocess import prepare_input

__all__ = ["OnnxDetector"]

logger = logging.getLogger(__name__)

#: Substrings that identify each raw head in an exported graph's output names.
#: Ordered most- to least-specific within each group.
_HEATMAP_HINTS = ("heatmap", "hm", "center", "centre", "cls", "score_map")
_SIZE_HINTS = ("size", "wh", "extent")
_OFFSET_HINTS = ("offset", "off", "reg")

#: Padding multiple used when the graph declares dynamic spatial dimensions.
#: 32 covers every common backbone's downsampling; the cost is at most 31 rows
#: and columns of replicated border.
_DEFAULT_SIZE_DIVISOR = 32

#: Boxes whose largest coordinate is below this are treated as normalised to
#: ``[0, 1]``. A real pixel box on any frame this pipeline handles is far
#: larger, so the test cannot misfire on a genuine pixel-space output -- except
#: for the degenerate case of a single sub-pixel box, which the size filter
#: would discard regardless.
_NORMALISED_COORD_LIMIT = 1.5


def _import_onnxruntime() -> Any:
    """Import ``onnxruntime`` or explain precisely how to get it."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise BackendUnavailableError(
            "detection.architecture='onnx' requires the onnxruntime package, "
            "which is not installed. Install it with "
            "`pip install 'sperm-sorting-ai[onnx]'`, or "
            "`pip install onnxruntime-gpu` for CUDA execution. To run without "
            "it, set detection.architecture to 'p2net', 'todcnn' or 'oracle'."
        ) from exc
    return onnxruntime


class OnnxDetector(Detector):
    """Runs an exported detection graph through ONNX Runtime.

    Parameters
    ----------
    cfg
        Thresholds, tiling-independent geometry and ``backend.onnx_providers``.
    weights
        Path to the ``.onnx`` file; defaults to ``cfg.weights``.
    size_divisor
        Padding multiple for graphs with dynamic spatial dimensions.
    """

    name = "onnx"

    def __init__(
        self,
        cfg: DetectionConfig | None = None,
        weights: str | Path | None = None,
        size_divisor: int = _DEFAULT_SIZE_DIVISOR,
    ) -> None:
        self.cfg = cfg if cfg is not None else DetectionConfig(architecture="onnx")
        self.class_names = tuple(self.cfg.class_names)
        self.size_divisor = int(size_divisor)

        path = Path(weights) if weights is not None else self.cfg.weights
        if path is None:
            raise InferenceError(
                "detection.architecture='onnx' requires detection.weights to "
                "point at an exported .onnx file"
            )
        self.weights_path = Path(path)
        if not self.weights_path.exists():
            raise InferenceError(f"ONNX model not found: {self.weights_path}")

        ort = _import_onnxruntime()
        self._ort = ort
        self._session = self._create_session(ort)
        self._closed = False

        self._input = self._session.get_inputs()[0]
        self._outputs = self._session.get_outputs()
        self._input_channels, self._static_hw = self._describe_input()
        self._signature, self._output_order = self._describe_outputs()
        logger.info(
            "%s: loaded %s (signature=%s, providers=%s)",
            self.name,
            self.weights_path.name,
            self._signature,
            self._session.get_providers(),
        )

    # ------------------------------------------------------------------ setup

    def _create_session(self, ort: Any) -> Any:
        """Build the inference session with the configured providers.

        A requested provider that is not built into the installed runtime is a
        warning rather than an error: the same config must run on the device
        (CUDA/TensorRT) and on a developer laptop (CPU). But it is a *loud*
        warning, because silently dropping to CPU turns a 2 ms inference into a
        200 ms one with no other symptom.
        """
        available = list(ort.get_available_providers())
        requested = list(self.cfg.backend.onnx_providers)
        usable = [p for p in requested if p in available]
        if not usable:
            fallback = (
                "CPUExecutionProvider"
                if "CPUExecutionProvider" in available
                else (available[0] if available else None)
            )
            if fallback is None:
                raise BackendUnavailableError(
                    "onnxruntime reports no execution providers at all; the "
                    "installation is broken. Reinstall onnxruntime."
                )
            logger.warning(
                "%s: none of the requested providers %s are available (this "
                "build offers %s); falling back to %s.",
                self.name,
                requested,
                available,
                fallback,
            )
            usable = [fallback]

        options = ort.SessionOptions()
        if self.cfg.backend.num_threads is not None:
            options.intra_op_num_threads = int(self.cfg.backend.num_threads)
            options.inter_op_num_threads = 1
        # Deterministic execution: the parallel executor can reorder reductions
        # between runs, and replay-determinism is a hard requirement here.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            return ort.InferenceSession(
                str(self.weights_path), sess_options=options, providers=usable
            )
        except Exception as exc:
            raise InferenceError(
                f"could not create an ONNX Runtime session for "
                f"{self.weights_path}: {exc}"
            ) from exc

    def _describe_input(self) -> tuple[int, tuple[int, int] | None]:
        """Read channel count and any static spatial dimensions from the graph.

        NCHW is assumed. Every exporter in this project's toolchain emits NCHW,
        and an NHWC graph would be caught immediately by a channel count of
        ``H``, which is not a plausible channel count.
        """
        shape = list(self._input.shape)
        if len(shape) != 4:
            raise InferenceError(
                f"expected a 4-D NCHW input, got shape {shape} for input "
                f"'{self._input.name}'"
            )
        channels = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 1
        if channels not in (1, 3):
            raise InferenceError(
                f"model input '{self._input.name}' expects {channels} channels; "
                "this pipeline can supply 1 (native monochrome) or 3 "
                "(explicitly replicated monochrome)"
            )
        # A dimension is only static if it is a positive int; exporters use a
        # string symbol (or a non-positive placeholder) for a dynamic axis.
        static_hw: tuple[int, int] | None = None
        if (
            isinstance(shape[2], int)
            and isinstance(shape[3], int)
            and shape[2] > 0
            and shape[3] > 0
        ):
            static_hw = (int(shape[2]), int(shape[3]))
        return int(channels), static_hw

    def _describe_outputs(self) -> tuple[str, tuple[int, ...]]:
        """Decide between the raw-head and the (N, 6) box signature."""
        names = [str(o.name).lower() for o in self._outputs]

        def match(hints: tuple[str, ...], taken: set[int]) -> int | None:
            for hint in hints:
                for index, name in enumerate(names):
                    if index not in taken and hint in name:
                        return index
            return None

        if len(self._outputs) >= 3:
            taken: set[int] = set()
            heat = match(_HEATMAP_HINTS, taken)
            if heat is not None:
                taken.add(heat)
            size = match(_SIZE_HINTS, taken)
            if size is not None:
                taken.add(size)
            offset = match(_OFFSET_HINTS, taken)
            if offset is not None:
                taken.add(offset)
            if heat is not None and size is not None and offset is not None:
                return "heads", (heat, size, offset)

            # Names were unhelpful; fall back to the channel counts, which are
            # unambiguous for this head: exactly two of the three outputs have
            # two channels.
            two_channel = [
                i
                for i, o in enumerate(self._outputs)
                if len(o.shape) == 4 and o.shape[1] == 2
            ]
            others = [
                i
                for i, o in enumerate(self._outputs)
                if len(o.shape) == 4 and i not in two_channel
            ]
            if len(two_channel) == 2 and len(others) == 1:
                logger.warning(
                    "%s: output names %s did not identify the heads; inferring "
                    "the order from channel counts. Re-export with names "
                    "'heatmap', 'size', 'offset' to remove the guess.",
                    self.name,
                    names,
                )
                return "heads", (others[0], two_channel[0], two_channel[1])

        if len(self._outputs) >= 1:
            shape = list(self._outputs[0].shape)
            trailing = shape[-1] if shape else None
            if (trailing == 6) or (not isinstance(trailing, int) and len(shape) in (2, 3)):
                return "boxes", (0,)

        raise InferenceError(
            f"cannot interpret the outputs of {self.weights_path.name}: "
            f"names={names}, shapes={[list(o.shape) for o in self._outputs]}. "
            "Supported signatures are three tensors named heatmap/size/offset, "
            "or a single (N, 6) tensor of x1,y1,x2,y2,score,class."
        )

    # -------------------------------------------------------------- inference

    def detect(self, frame: FramePacket) -> list[Detection]:
        if self._closed:
            raise InferenceError(f"{self.name}: detect() called after close()")
        image = frame.image
        if image is None or getattr(image, "size", 0) == 0:
            return []

        batch, content_shape, source_shape = prepare_input(
            image,
            input_size=self.cfg.input_size,
            size_divisor=1 if self._static_hw is not None else self.size_divisor,
            channels=self._input_channels,
            target_shape=self._static_hw,
        )
        if source_shape[0] < 1 or source_shape[1] < 1:
            return []

        try:
            raw = self._session.run(
                [o.name for o in self._outputs], {self._input.name: batch}
            )
        except Exception as exc:
            raise InferenceError(
                f"{self.name}: inference failed on frame {frame.frame_id}: {exc}"
            ) from exc

        input_shape = (int(batch.shape[2]), int(batch.shape[3]))
        if self._signature == "heads":
            boxes, scores, class_ids = self._decode_heads(raw, input_shape)
        else:
            boxes, scores, class_ids = self._decode_boxes(raw, input_shape)

        # Boxes are in padded-input pixels: clip to the content area first (the
        # padding has no source-frame preimage), then undo the resize.
        boxes = clip_boxes(boxes, content_shape)
        boxes = scale_boxes(boxes, content_shape, source_shape)
        boxes, scores, class_ids = finalise_boxes(
            boxes,
            scores,
            class_ids,
            source_shape,
            min_box_size=self.cfg.min_box_size_px,
            max_box_size=self.cfg.max_box_size_px,
            iou_threshold=self.cfg.nms_iou_threshold,
            max_detections=self.cfg.max_detections,
        )
        return arrays_to_detections(
            boxes, scores, class_ids, frame, self.class_names
        )

    def _decode_heads(
        self, raw: list[np.ndarray], input_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        heat_idx, size_idx, offset_idx = self._output_order
        heatmap = np.asarray(raw[heat_idx], dtype=np.float32)[0]
        size_map = np.asarray(raw[size_idx], dtype=np.float32)[0]
        offset_map = np.asarray(raw[offset_idx], dtype=np.float32)[0]

        # The stride is a property of the exported graph, not of config; taking
        # it from the actual output geometry means a re-export at a different
        # stride needs no config change and cannot silently mismatch.
        stride = float(input_shape[0]) / float(heatmap.shape[1])

        # An exporter that folded the sigmoid away leaves logits here. Detecting
        # that from the value range is a heuristic, but the alternative -- a
        # config flag -- would be set wrong exactly once and then produce a
        # detector that finds nothing, with no error.
        if heatmap.min() < 0.0 or heatmap.max() > 1.0:
            logger.debug(
                "%s: heatmap range [%.3f, %.3f] is outside [0, 1]; applying a "
                "sigmoid, assuming the export emitted logits.",
                self.name,
                float(heatmap.min()),
                float(heatmap.max()),
            )
            heatmap = 1.0 / (1.0 + np.exp(-np.clip(heatmap, -30.0, 30.0)))

        return decode_centernet_heatmap(
            heatmap,
            size_map,
            offset_map,
            stride=stride,
            score_threshold=self.cfg.score_threshold,
            max_detections=self.cfg.max_detections,
        )

    def _decode_boxes(
        self, raw: list[np.ndarray], input_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = np.asarray(raw[self._output_order[0]], dtype=np.float32)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise InferenceError(
                    f"{self.name}: expected batch size 1 from the box output, "
                    f"got shape {array.shape}"
                )
            array = array[0]
        if array.ndim != 2 or array.shape[1] < 6:
            raise InferenceError(
                f"{self.name}: box output must be (N, 6) "
                f"x1,y1,x2,y2,score,class; got shape {array.shape}"
            )
        if array.shape[0] == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        boxes = array[:, :4].copy()
        scores = array[:, 4].astype(np.float32)
        class_ids = array[:, 5].astype(np.int64)

        # Some exporters emit coordinates normalised to the input size. Scaling
        # them up here rather than requiring a config flag keeps a third-party
        # model drop-in; see _NORMALISED_COORD_LIMIT for why the test is safe.
        if boxes.size and float(np.max(np.abs(boxes))) <= _NORMALISED_COORD_LIMIT:
            boxes[:, [0, 2]] *= float(input_shape[1])
            boxes[:, [1, 3]] *= float(input_shape[0])

        # A graph with folded NMS has already thresholded, but not necessarily
        # at *our* threshold, and config must remain the single source of truth.
        keep = scores >= float(self.cfg.score_threshold)
        return boxes[keep], scores[keep], class_ids[keep]

    # ------------------------------------------------------------------ admin

    def close(self) -> None:
        """Drop the session. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._session = None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "architecture": "onnx",
                "backend": "onnxruntime",
                "weights": str(self.weights_path),
                "output_signature": self._signature,
                "input_channels": self._input_channels,
                "static_input_shape": (
                    list(self._static_hw) if self._static_hw else None
                ),
                "providers": (
                    list(self._session.get_providers()) if self._session else []
                ),
                "onnxruntime_version": getattr(self._ort, "__version__", "unknown"),
            }
        )
        return info
