"""Runtime morphology inference: the adapter between the network and the schema.

This module owns **the one polarity flip in the codebase**. The network emits
logits for ``P(abnormal)`` (see :mod:`.polarity`); the product schema wants
:attr:`~sperm_sorting.schemas.morphology.AspectResult.p_normal`. Both the
probability and the threshold are converted here, by calling
:func:`~.polarity.flip_polarity`, and nowhere else.

Failure is never "normal"
-------------------------
Three things can go wrong -- the deadline can pass, there can be no usable
crop, and the backend can raise -- and all three must be visibly distinct from
"this sperm is fine". Every failure path returns
:meth:`MorphologyResult.failed` with the matching
:class:`~sperm_sorting.schemas.enums.MorphologyStatus`, which leaves all four
aspects ``None``, which makes
:attr:`~sperm_sorting.schemas.morphology.MorphologyResult.is_complete` false,
which makes ``all_four_normal`` false. There is no code path in this module
that constructs an :class:`AspectResult` without a real model output behind it.

Deadlines are checked against :func:`time.monotonic`, never the wall clock: the
whole point is elapsed real time on a machine whose clock may step.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import MorphologyConfig
from ..constants import MORPHOLOGY_ASPECTS
from ..errors import BackendUnavailableError, InferenceError
from ..schemas.enums import MorphologyStatus
from ..schemas.morphology import AspectResult, MorphologyResult
from ..schemas.track import TrackRecord
from .calibration import CalibrationBundle, sigmoid
from .polarity import (
    POLARITY_CONVENTION,
    p_normal_from_p_abnormal,
    p_normal_threshold_from_p_abnormal_threshold,
)

__all__ = [
    "BaseMorphologyEngine",
    "MorphologyEngine",
    "RandomMorphologyEngine",
    "preprocess_crops",
]

_log = logging.getLogger(__name__)

#: Thresholds are kept this far from the closed ends of the unit interval so
#: that they satisfy the strict ``0 < t < 1`` of ``MorphologyConfig``.
_THRESHOLD_EPS = 1e-6


# ==========================================================================
# Preprocessing
# ==========================================================================


def _to_grayscale(crop: np.ndarray) -> np.ndarray:
    """Reduce an ``(H, W)``/``(H, W, 1)``/``(H, W, 3)`` array to ``(H, W)``."""
    if crop.ndim == 2:
        return crop
    if crop.ndim == 3 and crop.shape[2] == 1:
        return crop[:, :, 0]
    if crop.ndim == 3 and crop.shape[2] == 3:
        # The source is monochrome; a 3-channel crop means something upstream
        # replicated it. Averaging inverts that replication exactly.
        return crop.mean(axis=2)
    raise InferenceError(
        f"morphology crop must be (H, W), (H, W, 1) or (H, W, 3); got shape {crop.shape}"
    )


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to ``(h, w)``. Safety net only -- the crop stage owns geometry.

    :class:`sperm_sorting.config.AppConfig` already validates that
    ``crop.output_size == morphology.input_size``, so reaching this function
    means something upstream produced an unexpected size. It is handled rather
    than raised on (a mis-sized crop is still better evidence than no evidence)
    but it is logged, because a silent resize would hide a real geometry bug.
    """
    if image.shape[:2] == size:
        return image
    _log.warning(
        "morphology crop arrived at %s but the model expects %s; resizing. Check "
        "that crop.output_size matches morphology.input_size.",
        image.shape[:2],
        size,
    )
    try:
        import cv2

        return cv2.resize(
            image.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_AREA
        )
    except ImportError:  # pragma: no cover - opencv is a base dependency
        rows = (np.arange(size[0]) * image.shape[0] / size[0]).astype(np.int64)
        cols = (np.arange(size[1]) * image.shape[1] / size[1]).astype(np.int64)
        return image[np.ix_(rows, cols)]


def preprocess_crops(
    crops: Sequence[np.ndarray],
    input_size: tuple[int, int],
    in_channels: int = 1,
) -> np.ndarray:
    """Stack crops into a ``(B, C, H, W)`` float32 batch in ``[0, 1]``.

    Only dtype, shape and scale are handled here. *Photometric* normalisation
    (min-max, z-score) belongs to the crop stage and is configured by
    :attr:`sperm_sorting.config.CropConfig.normalize`; doing it in two places
    would mean a config change silently normalising twice.

    Integer inputs are divided by the dtype maximum (255 for ``uint8``, 65535
    for ``uint16``), so a 12-bit camera stream promoted to ``uint16`` lands in
    the same range as an 8-bit one. Floating-point inputs are assumed to be
    normalised already and are passed through, only clipped to ``[0, 1]``.
    """
    if not crops:
        return np.zeros((0, in_channels, input_size[0], input_size[1]), dtype=np.float32)

    batch = np.empty((len(crops), in_channels, input_size[0], input_size[1]), np.float32)
    for index, raw in enumerate(crops):
        if raw is None or getattr(raw, "size", 0) == 0:
            raise InferenceError(f"morphology crop {index} is empty")
        source = np.asarray(raw)
        # The integer/float decision is made on the *source* dtype, before the
        # channel reduction: averaging three uint8 channels yields a float64
        # array whose values are still 0-255, and testing the reduced array's
        # dtype would send it down the "already normalised" path and clip every
        # pixel to 1.0.
        integral = np.issubdtype(source.dtype, np.integer)
        gray = _to_grayscale(source).astype(np.float32)
        if integral:
            gray /= float(np.iinfo(source.dtype).max)
        else:
            gray = np.clip(gray, 0.0, 1.0)
        gray = _resize(gray, input_size)
        # A monochrome sensor feeding a >1-channel model means the model was
        # built for replicated input; broadcasting is the correct inverse.
        batch[index] = np.broadcast_to(gray, (in_channels, *input_size))
    return batch


# ==========================================================================
# Backends
# ==========================================================================


class _Runner(ABC):
    """Turns a preprocessed batch into per-aspect ``P(abnormal)`` **logits**."""

    aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS

    @abstractmethod
    def run(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        """``(B, C, H, W)`` float32 in, ``{aspect: (B,) logits}`` out."""

    def close(self) -> None:
        """Release backend resources. Safe to call more than once."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Backend metadata for the audit-log header."""


class _TorchRunner(_Runner):
    """PyTorch backend. Holds the model in ``eval`` mode under ``inference_mode``."""

    def __init__(
        self,
        cfg: MorphologyConfig,
        model: Any | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is an optional extra
            raise BackendUnavailableError(
                "morphology backend 'torch' requires PyTorch; install the 'torch' "
                "extra or set morphology.backend.kind='onnxruntime'"
            ) from exc
        self._torch = torch

        if cfg.backend.num_threads:
            torch.set_num_threads(int(cfg.backend.num_threads))

        self.untrained = False
        info: dict[str, Any] = {}
        if model is not None:
            self.model = model
        elif cfg.weights is not None:
            from .model import load_checkpoint

            self.model, info = load_checkpoint(cfg.weights, map_location=cfg.backend.device)
        else:
            from .model import MultiTaskMorphologyNet

            # Explicitly allowed so the pipeline can be exercised end to end
            # before weights exist. Loud, because an untrained morphology model
            # produces confident-looking nonsense.
            _log.warning(
                "morphology.weights is not set: constructing an UNTRAINED %s. Its "
                "outputs are meaningless and must not be used for any decision.",
                cfg.backbone,
            )
            self.model = MultiTaskMorphologyNet(
                backbone=cfg.backbone, in_channels=1, input_size=cfg.input_size
            )
            self.untrained = True

        self.checkpoint_info = info
        self.aspects = tuple(getattr(self.model, "aspects", MORPHOLOGY_ASPECTS))
        self.device = torch.device(cfg.backend.device)
        self.model = self.model.to(self.device).eval()
        self._fp16 = bool(cfg.backend.fp16) and self.device.type == "cuda"
        if self._fp16:
            self.model = self.model.half()

    def run(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        """Forward the batch under ``inference_mode``; returns raw logits."""
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        if self._fp16:
            tensor = tensor.half()
        with torch.inference_mode():
            logits = self.model(tensor)
        return {
            name: value.detach().float().cpu().numpy().reshape(-1)
            for name, value in logits.items()
        }

    def describe(self) -> dict[str, Any]:
        """Torch backend metadata, including the checkpoint provenance keys."""
        described = self.model.describe() if hasattr(self.model, "describe") else {}
        return {
            "backend": "torch",
            "device": str(self.device),
            "fp16": self._fp16,
            "untrained": self.untrained,
            "model": described,
            "checkpoint": {
                k: v
                for k, v in self.checkpoint_info.items()
                if k in ("model_id", "weights_provenance", "created_utc", "label_polarity")
            },
        }


class _OnnxRunner(_Runner):
    """ONNX Runtime backend.

    Accepts either the packed single output written by
    :func:`~.model.export_onnx` -- a ``(B, n_aspects)`` tensor whose columns
    follow :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS` -- or one named
    output per aspect. The packed form is preferred because column order is then
    fixed by the exporter rather than by ONNX Runtime's output ordering.

    The graph is expected to emit **logits for P(abnormal)**, identically to the
    torch path, so the single flip downstream serves both backends. A graph that
    emitted probabilities would be silently mis-read, which is why
    :func:`~.model.export_onnx` is the only supported way to produce one.
    """

    def __init__(self, cfg: MorphologyConfig) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - onnx is an optional extra
            raise BackendUnavailableError(
                "morphology backend 'onnxruntime' requires onnxruntime; install "
                "the 'onnx' extra"
            ) from exc
        if cfg.weights is None:
            raise InferenceError(
                "morphology.backend.kind='onnxruntime' requires morphology.weights "
                "to point at an exported .onnx graph"
            )
        path = Path(cfg.weights)
        if not path.exists():
            raise InferenceError(f"morphology ONNX graph not found: {path}")

        options = ort.SessionOptions()
        if cfg.backend.num_threads:
            options.intra_op_num_threads = int(cfg.backend.num_threads)
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=list(cfg.backend.onnx_providers)
            )
        except Exception as exc:
            raise BackendUnavailableError(
                f"could not create an ONNX Runtime session for {path}: {exc}"
            ) from exc

        self.aspects = tuple(MORPHOLOGY_ASPECTS)
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._path = path

    def run(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        """Run the graph and unpack its outputs into per-aspect logits."""
        outputs = self.session.run(self._output_names, {self._input_name: batch})
        if len(outputs) == 1:
            packed = np.asarray(outputs[0], dtype=np.float64).reshape(batch.shape[0], -1)
            if packed.shape[1] != len(self.aspects):
                raise InferenceError(
                    f"ONNX graph {self._path} emitted {packed.shape[1]} columns; "
                    f"expected {len(self.aspects)} (one per aspect, in the order "
                    f"{self.aspects})"
                )
            return {name: packed[:, i] for i, name in enumerate(self.aspects)}

        named = {
            name: np.asarray(value, dtype=np.float64).reshape(-1)
            for name, value in zip(self._output_names, outputs, strict=True)
        }
        missing = [a for a in self.aspects if a not in named]
        if missing:
            raise InferenceError(
                f"ONNX graph {self._path} has outputs {self._output_names}; expected "
                f"either one packed output or one per aspect. Missing: {missing}"
            )
        return {name: named[name] for name in self.aspects}

    def describe(self) -> dict[str, Any]:
        """ONNX Runtime metadata, including the providers actually selected."""
        return {
            "backend": "onnxruntime",
            "graph": str(self._path),
            "providers": list(self.session.get_providers()),
            "outputs": list(self._output_names),
            "untrained": False,
        }


# ==========================================================================
# Engine interface
# ==========================================================================


class BaseMorphologyEngine(ABC):
    """Common interface for anything that can judge a crop's morphology.

    The runtime never knows which implementation it holds, so the deadline and
    failure semantics documented on :meth:`evaluate_track` are part of the
    contract, not of one implementation.
    """

    #: Written into every :class:`MorphologyResult` for traceability.
    model_id: str = ""
    weights_provenance: str = ""
    aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS

    @abstractmethod
    def infer_batch(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        """``P(normal)`` per aspect for each crop, in input order."""

    @abstractmethod
    def evaluate_track(
        self,
        track: TrackRecord,
        crop_image: np.ndarray | None,
        deadline_s: float | None = None,
    ) -> MorphologyResult:
        """Judge one track's crop, honouring its monotonic deadline."""

    def evaluate_batch(
        self,
        tracks: Sequence[TrackRecord],
        crops: Sequence[np.ndarray | None],
        deadline_s: float | None = None,
    ) -> list[MorphologyResult]:
        """Judge several tracks. Default implementation is a loop."""
        if len(tracks) != len(crops):
            raise ValueError(
                f"evaluate_batch got {len(tracks)} tracks and {len(crops)} crops"
            )
        return [
            self.evaluate_track(track, crop, deadline_s)
            for track, crop in zip(tracks, crops, strict=True)
        ]

    def warmup(self, iterations: int = 2) -> None:
        """Run dummy inferences so the first real crop is not the slow one."""

    def close(self) -> None:
        """Release backend resources. Safe to call more than once."""

    def describe(self) -> dict[str, Any]:
        """Engine metadata for the audit-log header."""
        return {
            "model_id": self.model_id,
            "weights_provenance": self.weights_provenance,
            "aspects": list(self.aspects),
            "label_polarity": POLARITY_CONVENTION,
        }

    def __enter__(self) -> BaseMorphologyEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ==========================================================================
# Real engine
# ==========================================================================


class MorphologyEngine(BaseMorphologyEngine):
    """Production morphology engine.

    Parameters
    ----------
    cfg
        The ``morphology`` config section. Selects the backbone, weights,
        backend, batch size, per-aspect thresholds and temperatures.
    model
        Pre-built torch module, bypassing ``cfg.weights``. For tests and for
        the training loop's own validation pass; production loads a checkpoint.
    calibration
        Explicit :class:`~.calibration.CalibrationBundle`. When ``None`` the
        engine looks for a sidecar next to the weights (see
        :meth:`find_calibration_sidecar`); when there is none it falls back to
        ``cfg.thresholds`` / ``cfg.temperatures``.
    strict_deadline
        Also drop a result that *finished* after its deadline. On by default: a
        verdict that arrives after the fluid segment has passed the magnet
        cannot be acted on, so reporting it as ``COMPLETE`` would overstate what
        the system actually did. The computed probabilities are preserved in
        ``result.meta`` so nothing is lost for offline analysis.

    Threshold and temperature precedence
    ------------------------------------
    A calibration bundle wins over the config, because the config defaults are
    the uncalibrated placeholders 0.5 / 1.0 and a fitted bundle is by definition
    better evidence. Which source supplied each number is reported by
    :meth:`describe`, so the audit header always says where the operating point
    came from.
    """

    def __init__(
        self,
        cfg: MorphologyConfig,
        *,
        model: Any | None = None,
        calibration: CalibrationBundle | None = None,
        strict_deadline: bool = True,
        warmup_iterations: int = 2,
    ) -> None:
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.weights_provenance = cfg.weights_provenance
        self.strict_deadline = bool(strict_deadline)
        self.input_size = (int(cfg.input_size[0]), int(cfg.input_size[1]))
        self.batch_size = max(1, int(cfg.batch_size))

        kind = cfg.backend.kind
        if kind == "torch":
            self._runner: _Runner = _TorchRunner(cfg, model=model)
        elif kind == "onnxruntime":
            if model is not None:
                raise InferenceError(
                    "an explicit torch model cannot be used with "
                    "morphology.backend.kind='onnxruntime'"
                )
            self._runner = _OnnxRunner(cfg)
        else:
            raise BackendUnavailableError(
                f"morphology backend '{kind}' is not implemented. Supported: "
                "'torch', 'onnxruntime'. TensorRT export exists for detection "
                "only; morphology runs once per track, not per frame, so it has "
                "never needed it."
            )

        self.aspects = tuple(self._runner.aspects)
        if self.aspects != MORPHOLOGY_ASPECTS:
            # MorphologyResult has exactly four named slots. A model with a
            # different head set cannot fill them, and quietly dropping or
            # reordering heads is the ordering bug this package is built to
            # avoid, so refuse at construction time instead.
            raise InferenceError(
                f"morphology model exposes aspects {self.aspects}, but the schema "
                f"requires exactly {MORPHOLOGY_ASPECTS} in that order"
            )
        self.in_channels = int(getattr(getattr(self._runner, "model", None), "in_channels", 1))

        self.calibration = calibration or self._load_sidecar(cfg)
        self._temperatures, self._temperature_source = self._resolve_temperatures()
        self._thresholds, self._threshold_source = self._resolve_thresholds()

        if warmup_iterations > 0:
            self.warmup(warmup_iterations)

    # ------------------------------------------------------------- calibration

    @staticmethod
    def find_calibration_sidecar(weights: Path | None) -> Path | None:
        """Locate a calibration bundle sitting beside the weights.

        Search order: ``<weights>.calibration.json`` (e.g.
        ``morphology_v3.calibration.json`` next to ``morphology_v3.pt``), then
        ``<weights_dir>/calibration.json``. Kept as a filename convention rather
        than a config field because :class:`~sperm_sorting.config.MorphologyConfig`
        is a fixed contract this module may not extend.
        """
        if weights is None:
            return None
        weights = Path(weights)
        candidates = [
            weights.with_suffix(".calibration.json"),
            weights.parent / "calibration.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_sidecar(self, cfg: MorphologyConfig) -> CalibrationBundle | None:
        path = self.find_calibration_sidecar(cfg.weights)
        if path is None:
            return None
        bundle = CalibrationBundle.load_json(path)
        _log.info("loaded morphology calibration bundle from %s", path)
        if bundle.model_id and cfg.model_id not in ("", "unset") and bundle.model_id != cfg.model_id:
            _log.warning(
                "calibration bundle %s was fitted for model_id=%r but this engine "
                "runs model_id=%r; the operating point may not transfer",
                path,
                bundle.model_id,
                cfg.model_id,
            )
        return bundle

    def _resolve_temperatures(self) -> tuple[dict[str, float], dict[str, str]]:
        temperatures: dict[str, float] = {}
        source: dict[str, str] = {}
        for name in self.aspects:
            if self.calibration is not None and name in self.calibration.temperatures:
                temperatures[name] = float(self.calibration.temperatures[name])
                source[name] = "calibration_bundle"
            else:
                temperatures[name] = float(self.cfg.temperatures.get(name, 1.0))
                source[name] = "config"
        return temperatures, source

    def _resolve_thresholds(self) -> tuple[dict[str, float], dict[str, str]]:
        """Per-aspect thresholds in **``P(normal)`` space**, ready for the schema.

        This is one of exactly two places the polarity flip is applied, and it
        goes through :func:`~.polarity.flip_polarity` like the other one. A
        calibration bundle stores its thresholds in ``P(abnormal)`` space
        because that is where they were fitted.
        """
        thresholds: dict[str, float] = {}
        source: dict[str, str] = {}
        for name in self.aspects:
            if self.calibration is not None and name in self.calibration.thresholds:
                abnormal_threshold = float(self.calibration.thresholds[name])
                value = p_normal_threshold_from_p_abnormal_threshold(abnormal_threshold)
                source[name] = "calibration_bundle"
            else:
                value = float(self.cfg.thresholds.get(name, 0.5))
                source[name] = "config"
            thresholds[name] = float(
                np.clip(value, _THRESHOLD_EPS, 1.0 - _THRESHOLD_EPS)
            )
        return thresholds, source

    @property
    def thresholds(self) -> dict[str, float]:
        """Per-aspect decision thresholds on ``p_normal``, as the schema uses them."""
        return dict(self._thresholds)

    # -------------------------------------------------------------- inference

    def _logits(self, crops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        """Raw ``P(abnormal)`` logits for every crop, batched by ``cfg.batch_size``."""
        chunks: list[dict[str, np.ndarray]] = []
        for start in range(0, len(crops), self.batch_size):
            batch = preprocess_crops(
                crops[start : start + self.batch_size], self.input_size, self.in_channels
            )
            chunks.append(self._runner.run(batch))
        if not chunks:
            return {name: np.zeros(0, dtype=np.float64) for name in self.aspects}
        return {
            name: np.concatenate([chunk[name] for chunk in chunks]) for name in self.aspects
        }

    def _p_normal(self, crops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        """Calibrated ``P(normal)`` per aspect. **The flip happens here.**"""
        logits = self._logits(crops)
        out: dict[str, np.ndarray] = {}
        for name in self.aspects:
            scaled = np.asarray(logits[name], dtype=np.float64) / self._temperatures[name]
            p_abnormal = sigmoid(scaled)
            # The one flip: network space (abnormal-positive) -> schema space.
            out[name] = p_normal_from_p_abnormal(p_abnormal)
        return out

    def infer_batch(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        """``P(normal)`` per aspect for each crop, in input order.

        Returns one dict per crop keyed by aspect name. The values are
        ``P(normal)``, i.e. already flipped out of the network's
        abnormal-positive space, and already temperature-scaled.
        """
        if not crops:
            return []
        probabilities = self._p_normal(crops)
        return [
            {name: float(probabilities[name][i]) for name in self.aspects}
            for i in range(len(crops))
        ]

    # ---------------------------------------------------------------- warm-up

    def warmup(self, iterations: int = 2) -> None:
        """Run dummy batches so lazy kernel selection does not land on frame one."""
        dummy = np.zeros(self.input_size, dtype=np.uint8)
        for _ in range(max(0, iterations)):
            try:
                self._p_normal([dummy])
            except Exception as exc:  # pragma: no cover - depends on the backend
                _log.warning("morphology warm-up failed: %s", exc)
                return

    # -------------------------------------------------------------- evaluation

    def _aspect_results(self, probabilities: dict[str, float]) -> dict[str, AspectResult]:
        return {
            name: AspectResult(
                name=name,
                p_normal=float(probabilities[name]),
                threshold=self._thresholds[name],
            )
            for name in self.aspects
        }

    def _complete(
        self,
        track_id: int,
        probabilities: dict[str, float],
        frame_id: int | None,
        latency_ms: float,
    ) -> MorphologyResult:
        aspects = self._aspect_results(probabilities)
        return MorphologyResult(
            track_id=track_id,
            status=MorphologyStatus.COMPLETE,
            head=aspects["head"],
            acrosome=aspects["acrosome"],
            vacuole=aspects["vacuole"],
            tail=aspects["tail"],
            frame_id=frame_id,
            latency_ms=latency_ms,
            model_id=self.model_id,
            weights_provenance=self.weights_provenance,
            meta={"label_polarity": POLARITY_CONVENTION},
        )

    @staticmethod
    def _frame_id(track: TrackRecord) -> int | None:
        """Frame the crop came from, duplicated onto the result for auditability."""
        return track.crop.frame_id if track.crop is not None else None

    def evaluate_track(
        self,
        track: TrackRecord,
        crop_image: np.ndarray | None,
        deadline_s: float | None = None,
    ) -> MorphologyResult:
        """Judge one track's crop.

        Ordering of the checks is the contract:

        1. **Deadline first.** If :func:`time.monotonic` is already at or past
           ``deadline_s``, return ``DEADLINE_MISSED`` **without running the
           model**. Spending the GPU/CPU on a result nobody can use is exactly
           the wrong thing to do when the pipeline is already late, and it makes
           the backlog worse.
        2. **Crop next.** A missing or empty crop is ``NO_VALID_CROP``.
        3. **Then inference.** Anything the backend raises becomes
           ``INFERENCE_FAILED`` with the exception text as the reason.
        4. **Deadline again**, when ``strict_deadline`` is set.

        Every non-``COMPLETE`` return leaves all four aspects ``None``, so
        ``all_four_normal`` is ``False``. A failure can never read as normal.
        """
        frame_id = self._frame_id(track)

        if deadline_s is not None:
            now = time.monotonic()
            if now >= deadline_s:
                return MorphologyResult.failed(
                    track.track_id,
                    MorphologyStatus.DEADLINE_MISSED,
                    f"deadline passed {(now - deadline_s) * 1000.0:.2f} ms before "
                    "morphology started; model not run",
                    frame_id=frame_id,
                )

        if crop_image is None or getattr(crop_image, "size", 0) == 0:
            return MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.NO_VALID_CROP,
                "no crop was supplied for this track",
                frame_id=frame_id,
            )

        started = time.perf_counter()
        try:
            probabilities = self.infer_batch([crop_image])[0]
        except Exception as exc:
            return MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.INFERENCE_FAILED,
                f"{type(exc).__name__}: {exc}",
                frame_id=frame_id,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if self.strict_deadline and deadline_s is not None and time.monotonic() >= deadline_s:
            return self._overrun(track.track_id, probabilities, frame_id, latency_ms, deadline_s)

        return self._complete(track.track_id, probabilities, frame_id, latency_ms)

    def _overrun(
        self,
        track_id: int,
        probabilities: dict[str, float],
        frame_id: int | None,
        latency_ms: float,
        deadline_s: float,
    ) -> MorphologyResult:
        """Result that finished, but too late to act on.

        The probabilities are attached to ``meta`` rather than discarded: they
        are useless to the actuator but valuable when working out why the
        deadline was blown.
        """
        overrun_ms = (time.monotonic() - deadline_s) * 1000.0
        result = MorphologyResult.failed(
            track_id,
            MorphologyStatus.DEADLINE_MISSED,
            f"morphology finished {overrun_ms:.2f} ms after its deadline; result discarded",
            frame_id=frame_id,
        )
        result.latency_ms = latency_ms
        result.model_id = self.model_id
        result.weights_provenance = self.weights_provenance
        result.meta = {
            "p_normal_computed_after_deadline": {k: float(v) for k, v in probabilities.items()},
            "overrun_ms": overrun_ms,
            "label_polarity": POLARITY_CONVENTION,
        }
        return result

    def evaluate_batch(
        self,
        tracks: Sequence[TrackRecord],
        crops: Sequence[np.ndarray | None],
        deadline_s: float | None = None,
    ) -> list[MorphologyResult]:
        """Judge several tracks with a single batched forward pass.

        Batching is what makes morphology affordable: the per-call overhead of a
        torch or ONNX Runtime invocation dominates a 128x128 MobileNet forward,
        so 16 crops in one call cost far less than 16 calls.

        Tracks whose crop is missing are answered ``NO_VALID_CROP`` and excluded
        from the batch rather than padded with zeros -- a zero crop would
        produce a real-looking probability, which is precisely the failure mode
        this module exists to prevent. The deadline is evaluated once, before the
        batch, so a late batch costs nothing.
        """
        if len(tracks) != len(crops):
            raise ValueError(
                f"evaluate_batch got {len(tracks)} tracks and {len(crops)} crops"
            )
        if not tracks:
            return []

        if deadline_s is not None:
            now = time.monotonic()
            if now >= deadline_s:
                overrun_ms = (now - deadline_s) * 1000.0
                return [
                    MorphologyResult.failed(
                        track.track_id,
                        MorphologyStatus.DEADLINE_MISSED,
                        f"deadline passed {overrun_ms:.2f} ms before the morphology "
                        "batch started; model not run",
                        frame_id=self._frame_id(track),
                    )
                    for track in tracks
                ]

        results: list[MorphologyResult | None] = [None] * len(tracks)
        usable: list[int] = []
        for index, (track, crop) in enumerate(zip(tracks, crops, strict=True)):
            if crop is None or getattr(crop, "size", 0) == 0:
                results[index] = MorphologyResult.failed(
                    track.track_id,
                    MorphologyStatus.NO_VALID_CROP,
                    "no crop was supplied for this track",
                    frame_id=self._frame_id(track),
                )
            else:
                usable.append(index)

        if usable:
            started = time.perf_counter()
            try:
                batch = self.infer_batch([np.asarray(crops[i]) for i in usable])
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for index in usable:
                    results[index] = MorphologyResult.failed(
                        tracks[index].track_id,
                        MorphologyStatus.INFERENCE_FAILED,
                        reason,
                        frame_id=self._frame_id(tracks[index]),
                    )
            else:
                # Amortised per-track latency: the batch is one indivisible
                # operation, so attributing the whole wall time to each track
                # would overstate the cost by the batch size.
                latency_ms = (time.perf_counter() - started) * 1000.0 / len(usable)
                overran = (
                    deadline_s
                    if self.strict_deadline
                    and deadline_s is not None
                    and time.monotonic() >= deadline_s
                    else None
                )
                for slot, index in enumerate(usable):
                    track_id = tracks[index].track_id
                    frame_id = self._frame_id(tracks[index])
                    results[index] = (
                        self._overrun(track_id, batch[slot], frame_id, latency_ms, overran)
                        if overran is not None
                        else self._complete(track_id, batch[slot], frame_id, latency_ms)
                    )

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------ admin

    def close(self) -> None:
        """Release the backend session/model. Safe to call more than once."""
        self._runner.close()

    def describe(self) -> dict[str, Any]:
        """Full engine metadata: backend, operating point, and where it came from."""
        return {
            **super().describe(),
            "backbone": self.cfg.backbone,
            "weights": str(self.cfg.weights) if self.cfg.weights else None,
            "input_size": list(self.input_size),
            "batch_size": self.batch_size,
            "strict_deadline": self.strict_deadline,
            "deadline_ms": self.cfg.deadline_ms,
            "thresholds_p_normal": dict(self._thresholds),
            "threshold_source": dict(self._threshold_source),
            "temperatures": dict(self._temperatures),
            "temperature_source": dict(self._temperature_source),
            "calibrated": self.calibration is not None,
            "runner": self._runner.describe(),
        }


# ==========================================================================
# Test double
# ==========================================================================


class RandomMorphologyEngine(BaseMorphologyEngine):
    """**TEST ONLY.** Seeded random verdicts; never a production engine.

    Exists so the tracking / shot / decision / scheduling stages can be
    exercised end to end before any morphology weights exist, and so the
    synthetic-source accuracy tests can dial the eligible fraction directly.

    Three deliberate safeguards against this ever reaching a real run:

    * :func:`~.factory.build_morphology_engine` has no path to it.
      :attr:`sperm_sorting.config.BackendConfig.kind` is a ``Literal`` over
      ``torch``/``onnxruntime``/``tensorrt``, so no YAML file can name it.
    * :attr:`weights_provenance` is hard-coded to ``"random-test-engine"``, so
      any audit log produced with it is self-evidently not a real evaluation.
      That string is not one of the ``WEIGHTS_PROVENANCE_*`` constants, on
      purpose.
    * It must be constructed explicitly in Python, by a test.

    Parameters
    ----------
    p_normal_rate
        Per-aspect probability that an aspect comes out normal. The defaults
        give an ``all_four_normal`` rate of roughly 0.55, matching
        :attr:`sperm_sorting.config.SyntheticSourceConfig.normal_morphology_rate`.
    """

    #: Deliberately not a WEIGHTS_PROVENANCE_* constant.
    _PROVENANCE = "random-test-engine"

    def __init__(
        self,
        seed: int = 1234,
        *,
        p_normal_rate: dict[str, float] | float = 0.87,
        thresholds: dict[str, float] | None = None,
        aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS,
        latency_ms: float = 0.0,
        model_id: str = "random-test-engine",
    ) -> None:
        self.aspects = tuple(aspects)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.model_id = model_id
        self.weights_provenance = self._PROVENANCE
        self.latency_ms = float(latency_ms)
        self.thresholds = {
            name: float((thresholds or {}).get(name, 0.5)) for name in self.aspects
        }
        rates = (
            dict.fromkeys(self.aspects, float(p_normal_rate))
            if isinstance(p_normal_rate, (int, float))
            else {name: float(p_normal_rate.get(name, 0.87)) for name in self.aspects}
        )
        for name, rate in rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"p_normal_rate['{name}'] must lie in [0, 1], got {rate}")
        self.p_normal_rate = rates

    def _draw(self, name: str) -> float:
        """Draw a ``p_normal`` on the correct side of the aspect's threshold.

        Drawing the *side* first and the magnitude second makes the resulting
        normal rate exactly ``p_normal_rate``, instead of whatever a uniform
        draw happens to produce, so a test can assert on the eligible fraction.
        """
        threshold = self.thresholds[name]
        if self.rng.random() < self.p_normal_rate[name]:
            return float(self.rng.uniform(threshold, 1.0))
        return float(self.rng.uniform(0.0, threshold))

    def infer_batch(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        """``P(normal)`` per aspect; the crop content is ignored by design."""
        return [{name: self._draw(name) for name in self.aspects} for _ in crops]

    def evaluate_track(
        self,
        track: TrackRecord,
        crop_image: np.ndarray | None,
        deadline_s: float | None = None,
    ) -> MorphologyResult:
        """Same failure semantics as :meth:`MorphologyEngine.evaluate_track`.

        The deadline and missing-crop paths are reproduced faithfully rather
        than short-circuited, because pipeline tests exist precisely to exercise
        them.
        """
        frame_id = track.crop.frame_id if track.crop is not None else None
        if deadline_s is not None and time.monotonic() >= deadline_s:
            return MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.DEADLINE_MISSED,
                "deadline already passed; random engine did not draw",
                frame_id=frame_id,
            )
        if crop_image is None or getattr(crop_image, "size", 0) == 0:
            return MorphologyResult.failed(
                track.track_id,
                MorphologyStatus.NO_VALID_CROP,
                "no crop was supplied for this track",
                frame_id=frame_id,
            )
        probabilities = self.infer_batch([crop_image])[0]
        aspects = {
            name: AspectResult(
                name=name,
                p_normal=probabilities[name],
                threshold=self.thresholds[name],
            )
            for name in self.aspects
        }
        return MorphologyResult(
            track_id=track.track_id,
            status=MorphologyStatus.COMPLETE,
            head=aspects["head"],
            acrosome=aspects["acrosome"],
            vacuole=aspects["vacuole"],
            tail=aspects["tail"],
            frame_id=frame_id,
            latency_ms=self.latency_ms,
            model_id=self.model_id,
            weights_provenance=self.weights_provenance,
            meta={"synthetic": True, "seed": self.seed},
        )

    def describe(self) -> dict[str, Any]:
        """Metadata that makes the test-only nature of this engine explicit."""
        return {
            **super().describe(),
            "test_only": True,
            "seed": self.seed,
            "p_normal_rate": dict(self.p_normal_rate),
            "thresholds": dict(self.thresholds),
        }
