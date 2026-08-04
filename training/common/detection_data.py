"""Detection training data: VISEM-Tracking, or synthetic clips.

The one rule this module exists to enforce
------------------------------------------
**Detection splits are made by video, never by frame.** At 30-160 FPS,
consecutive frames of one recording are near-duplicates: the same cells, the
same debris, the same illumination gradient, displaced by a few pixels. A
random frame-level split therefore puts near-copies of the validation set into
the training set, and the resulting validation AP measures memorisation. On
VISEM-Tracking, with 20 recordings of a few thousand frames each, the effect is
not subtle -- it is the difference between a number that predicts field
performance and one that does not.

:func:`assert_no_video_leakage` is called by the trainer before a single step
is taken and **raises** on any overlap. It does not warn. A warning in a log
that nobody reads is indistinguishable from no check at all, and the failure it
guards against produces a *better*-looking metric, so nothing downstream will
ever flag it.

Sources
-------
``visem``
    Delegates to ``datasets.adapters.visem``, coded against the narrow
    :class:`DetectionDatasetAdapter` protocol below so that the adapter and
    this harness can land in either order.

``synthetic``
    Generates clips from the in-repo simulator. Preferred path: the runtime
    :class:`~sperm_sorting.acquisition.synthetic.SyntheticFrameSource`, one
    instance per clip with its own seed, which publishes ground-truth boxes in
    ``FramePacket.meta["gt_detections"]`` -- exactly the format the oracle
    detector and the evaluation harness already read. If that source is not
    available (its scene generator is a separate module), a lean local renderer
    built directly on :mod:`sperm_sorting.simulator.render` produces
    equivalent still frames instead, so the detector remains trainable.

    Either way a *clip* is the unit of splitting, and clips are what the
    leakage validator sees. Frames within a clip share their cell population,
    which is precisely the correlation the by-video rule exists to respect.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sperm_sorting.errors import ConfigurationError

__all__ = [
    "DETECTION_SOURCE_KINDS",
    "DetectionClipDataset",
    "DetectionDatasetAdapter",
    "DetectionFrame",
    "DetectionSource",
    "assert_no_video_leakage",
    "build_synthetic_clips",
    "collate_detection_batch",
    "load_detection_source",
    "split_by_video",
]

#: Sources ``--source`` accepts.
DETECTION_SOURCE_KINDS: tuple[str, ...] = ("visem", "synthetic")

#: Reported when the VISEM adapter does not supply its own licence string.
VISEM_LICENCE: str = (
    "see the VISEM-Tracking dataset's own licence and citation terms "
    "(Thambawita et al., 'VISEM-Tracking: a human spermatozoa tracking "
    "dataset', Scientific Data 2023); the adapter reports the authoritative "
    "string when it exposes one"
)


# ==========================================================================
# Frame record
# ==========================================================================


@dataclass(slots=True)
class DetectionFrame:
    """One annotated frame, in the coordinate convention the whole repo uses.

    ``boxes`` is ``(N, 4)`` xyxy in **pixels of this image**, matching
    :class:`sperm_sorting.schemas.detection.BoundingBox`. ``video_id`` is the
    grouping key the leakage validator works on and is the only thing standing
    between this dataset and a frame-level split.
    """

    image: np.ndarray
    boxes: np.ndarray
    class_ids: np.ndarray
    video_id: str
    frame_index: int
    #: Ground-truth object identity, when the source has it. Never consumed by
    #: the detector -- it is carried so that a detection run can be scored
    #: against tracking ground truth without a second pass over the data.
    track_ids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        if self.image.ndim == 3 and self.image.shape[-1] == 1:
            self.image = self.image[..., 0]
        if self.image.ndim != 2:
            raise ConfigurationError(
                f"detection frames must be monochrome (H, W); got shape {self.image.shape}. "
                "The target camera is Mono8 and nothing in this pipeline assumes "
                "three channels."
            )
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        if self.boxes.shape[0] != self.class_ids.shape[0]:
            raise ConfigurationError(
                f"{self.video_id}/{self.frame_index}: {self.boxes.shape[0]} boxes but "
                f"{self.class_ids.shape[0]} class ids"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.image.shape[0]), int(self.image.shape[1]))


# ==========================================================================
# Protocol
# ==========================================================================


@runtime_checkable
class DetectionDatasetAdapter(Protocol):
    """The minimum this harness needs from a detection dataset adapter.

    Frames are requested *by video* rather than as one flat sequence. That is
    not a convenience: an adapter that can only hand back a flat frame list
    makes a by-video split impossible to express, so the shape of this protocol
    is itself the enforcement mechanism.
    """

    name: str
    licence: str

    def video_ids(self) -> Sequence[str]:
        """Every recording in the dataset."""
        ...

    def load_video(self, video_id: str) -> Sequence[DetectionFrame]:
        """Annotated frames of one recording, in capture order."""
        ...


# ==========================================================================
# Leakage validation
# ==========================================================================


def assert_no_video_leakage(splits: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Raise unless every video appears in at most one split.

    Parameters
    ----------
    splits
        Split name to the video ids assigned to it.

    Returns
    -------
    dict
        A record of what was checked, written into ``experiment.json`` so that
        the check is visible in the result rather than only in the code.

    Raises
    ------
    ConfigurationError
        On any overlap, on a duplicate within one split, or on an empty split.
        All three are hard failures. A duplicated video inside one split
        silently doubles its weight; an empty validation split makes every
        epoch's metric NaN, which the early stopper will then never improve on.

    Notes
    -----
    This first tries ``datasets.validators``, so that if the dataset package
    ships an authoritative validator it is the one that runs and the two can
    never disagree. The implementation below is the fallback, and it is a real
    check rather than a stub.
    """
    record: dict[str, Any] = {
        "checker": "training.common.detection_data.assert_no_video_leakage",
        "splits": {name: sorted(set(ids)) for name, ids in splits.items()},
        "counts": {name: len(list(ids)) for name, ids in splits.items()},
    }

    external = _external_leakage_validator()
    if external is not None:
        external(splits)
        record["checker"] = f"{external.__module__}.{external.__qualname__}"

    for name, ids in splits.items():
        listed = list(ids)
        if not listed:
            raise ConfigurationError(
                f"split '{name}' has no videos. An empty split cannot be trained on "
                "or validated against; check the --split-fractions and the number of "
                "available recordings."
            )
        if len(listed) != len(set(listed)):
            duplicated = sorted({v for v in listed if listed.count(v) > 1})
            raise ConfigurationError(
                f"split '{name}' lists {duplicated} more than once, which would "
                "silently double their weight."
            )

    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = sorted(set(splits[left]) & set(splits[right]))
            if shared:
                raise ConfigurationError(
                    f"video-level leakage between '{left}' and '{right}': {shared}. "
                    "Frames of one recording are near-duplicates of each other, so a "
                    "video appearing on both sides makes the validation metric a "
                    "measurement of memorisation. This is refused, not warned about, "
                    "because the failure makes the metric look BETTER and nothing "
                    "downstream would ever flag it."
                )

    record["passed"] = True
    return record


def _external_leakage_validator() -> Any:
    """Find ``datasets.validators``' leakage checker, if it exists yet.

    Looked up by a short list of plausible names rather than one, because the
    adapters are being written in parallel; a naming mismatch should fall back
    to the local check, not disable checking.
    """
    try:
        module = importlib.import_module("datasets.validators")
    except ImportError:
        return None
    for attribute in (
        "assert_no_video_leakage",
        "validate_no_video_leakage",
        "check_video_leakage",
    ):
        candidate = getattr(module, attribute, None)
        if callable(candidate):
            return candidate
    return None


def split_by_video(
    video_ids: Sequence[str],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 1234,
) -> dict[str, list[str]]:
    """Assign whole videos to train/valid/test.

    The shuffle is seeded and the assignment is by *sorted* video id, so the
    same seed and the same set of recordings always produce the same split --
    including when the adapter changes the order it lists them in, which it is
    free to do.

    At least one video is guaranteed to each split (the split is refused if
    there are fewer than three recordings), because a validation split of zero
    videos silently disables early stopping and best-checkpoint selection.
    """
    unique = sorted({str(v) for v in video_ids})
    if len(unique) < 3:
        raise ConfigurationError(
            f"a by-video split needs at least 3 recordings, got {len(unique)}: "
            f"{unique}. Frame-level splitting is not offered as an alternative -- "
            "see the module docstring."
        )
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ConfigurationError(f"split fractions must sum to 1.0, got {fractions}")
    if any(f <= 0.0 for f in fractions):
        raise ConfigurationError(f"every split fraction must be positive, got {fractions}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = [unique[i] for i in order]

    total = len(shuffled)
    n_train = max(1, int(round(fractions[0] * total)))
    n_valid = max(1, int(round(fractions[1] * total)))
    # The test split takes the remainder; clamp so train never eats it all.
    while n_train + n_valid > total - 1:
        if n_train > 1:
            n_train -= 1
        else:
            n_valid -= 1

    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train : n_train + n_valid],
        "test": shuffled[n_train + n_valid :],
    }


# ==========================================================================
# Synthetic clips
# ==========================================================================


def build_synthetic_clips(
    n_clips: int,
    frames_per_clip: int,
    *,
    seed: int,
    width: int = 320,
    height: int = 256,
    density: float = 6.0,
    debris_density: float = 3.0,
    um_per_px: float = 0.35,
    flow_vx_px_s: float = 40.0,
    fps: float = 30.0,
) -> list[DetectionFrame]:
    """Generate ``n_clips`` clips of ``frames_per_clip`` annotated frames.

    Each clip is one "video": its cells are sampled once, given a pose, and
    advanced by the bulk flow across the clip, so frames within a clip are
    correlated exactly the way frames of a real recording are. That is what
    makes the by-video split meaningful on synthetic data too.

    The runtime :class:`~sperm_sorting.acquisition.synthetic.SyntheticFrameSource`
    is used when its scene generator is importable, because then the frames and
    their ``gt_detections`` are produced by the very code the runtime uses. The
    local fallback below exists so that detector training is not blocked on
    that module; it renders the same cells with the same renderer and computes
    boxes from the same ``render_sperm_on_canvas`` return value, so the two
    agree on geometry.

    The defaults are small (320x256, 30 FPS) because this is bootstrap data for
    a CPU smoke test, not a substitute for VISEM-Tracking. They are overridable.
    """
    if n_clips < 1 or frames_per_clip < 1:
        raise ConfigurationError(
            f"need at least one clip and one frame, got {n_clips} x {frames_per_clip}"
        )

    frames = _clips_from_runtime_source(
        n_clips, frames_per_clip, seed=seed, width=width, height=height,
        density=density, debris_density=debris_density, um_per_px=um_per_px,
        flow_vx_px_s=flow_vx_px_s, fps=fps,
    )
    if frames is not None:
        return frames
    return _clips_from_renderer(
        n_clips, frames_per_clip, seed=seed, width=width, height=height,
        density=density, debris_density=debris_density, um_per_px=um_per_px,
        flow_vx_px_s=flow_vx_px_s, fps=fps,
    )


def _clips_from_runtime_source(
    n_clips: int,
    frames_per_clip: int,
    *,
    seed: int,
    width: int,
    height: int,
    density: float,
    debris_density: float,
    um_per_px: float,
    flow_vx_px_s: float,
    fps: float,
) -> list[DetectionFrame] | None:
    """Use the runtime synthetic source, or return ``None`` if unavailable.

    ``None`` rather than an exception: the scene generator is a separate module
    that may not exist yet, and a detector training run should fall back to the
    local renderer rather than fail.
    """
    try:
        from sperm_sorting.acquisition.synthetic import SyntheticFrameSource
        from sperm_sorting.config import SyntheticSourceConfig
    except ImportError:
        return None

    frames: list[DetectionFrame] = []
    for clip in range(n_clips):
        cfg = SyntheticSourceConfig(
            width=width,
            height=height,
            fps=fps,
            n_frames=frames_per_clip,
            um_per_px=um_per_px,
            density=density,
            debris_density=debris_density,
            flow_vx_px_s=flow_vx_px_s,
            flow_vy_px_s=0.0,
            seed=int(seed) + 7919 * clip,
        )
        source = SyntheticFrameSource(cfg)
        try:
            source.open()
        except Exception:
            # The scene generator is not importable/usable in this checkout.
            return None
        try:
            for index in range(frames_per_clip):
                packet = source.read()
                if packet is None:
                    break
                records = packet.meta.get("gt_detections") or []
                boxes = np.array(
                    [list(r["box_xyxy"]) for r in records], dtype=np.float32
                ).reshape(-1, 4)
                class_ids = np.array(
                    [int(r.get("class_id", 0)) for r in records], dtype=np.int64
                )
                track_ids = np.array(
                    [int(r["track_id"]) if r.get("track_id") is not None else -1 for r in records],
                    dtype=np.int64,
                )
                frames.append(
                    DetectionFrame(
                        image=packet.image,
                        boxes=boxes,
                        class_ids=class_ids,
                        video_id=f"synthclip{clip:03d}",
                        frame_index=index,
                        track_ids=track_ids,
                        meta={"generator": "sperm_sorting.acquisition.synthetic"},
                    )
                )
        finally:
            source.close()

    return frames if frames else None


def _clips_from_renderer(
    n_clips: int,
    frames_per_clip: int,
    *,
    seed: int,
    width: int,
    height: int,
    density: float,
    debris_density: float,
    um_per_px: float,
    flow_vx_px_s: float,
    fps: float,
) -> list[DetectionFrame]:
    """Fallback clip generator built straight on the renderer.

    Deliberately minimal: cells are sampled once per clip, given a fixed
    :class:`~sperm_sorting.simulator.render.CellPose` so they do not flicker,
    and translated by the bulk flow plus a small per-cell swimming velocity.
    There is no motility grading and no per-sperm health state in the output,
    because a detector needs neither -- it needs images and boxes, and
    manufacturing the rest here would duplicate the runtime scene generator
    rather than complement it.
    """
    from sperm_sorting.simulator.params import sample_health_state
    from sperm_sorting.simulator.render import (
        CellPose,
        RenderConfig,
        finish_image,
        illumination_field,
        render_debris_on_canvas,
        render_sperm_on_canvas,
    )

    render_cfg = RenderConfig()
    dt = 1.0 / max(fps, 1e-6)
    frames: list[DetectionFrame] = []

    for clip in range(n_clips):
        rng = np.random.default_rng(int(seed) + 7919 * clip)
        n_cells = max(1, int(rng.poisson(max(density, 0.1))))
        n_debris = int(rng.poisson(max(debris_density, 0.0)))

        states = [sample_health_state(rng, None, 0.6) for _ in range(n_cells)]
        poses = [CellPose.sample(rng) for _ in range(n_cells)]
        positions = np.stack(
            [rng.uniform(0, width, n_cells), rng.uniform(0, height, n_cells)], axis=1
        )
        angles = rng.uniform(0.0, 2.0 * math.pi, n_cells)
        # Swimming velocity on top of the bulk flow, in px/s. Modest, so a cell
        # stays in frame for most of the clip and the frames stay correlated.
        swim = rng.normal(0.0, 12.0, size=(n_cells, 2))

        debris_pos = np.stack(
            [rng.uniform(0, width, n_debris), rng.uniform(0, height, n_debris)], axis=1
        )
        debris_kind = ["blob" if rng.random() < 0.6 else "streak" for _ in range(n_debris)]
        debris_size = rng.uniform(2.0, 6.0, n_debris)
        debris_elong = rng.uniform(1.0, 4.0, n_debris)
        debris_angle = rng.uniform(0.0, math.pi, n_debris)

        for index in range(frames_per_clip):
            canvas = illumination_field((height, width), render_cfg, rng)
            boxes: list[tuple[float, float, float, float]] = []
            track_ids: list[int] = []

            for cell in range(n_cells):
                # Wrap rather than clip, so the population stays constant across
                # the clip and the frame count per clip is uniform.
                cx = float((positions[cell, 0] + (flow_vx_px_s + swim[cell, 0]) * dt * index) % width)
                cy = float((positions[cell, 1] + swim[cell, 1] * dt * index) % height)
                box = render_sperm_on_canvas(
                    canvas,
                    states[cell],
                    cx,
                    cy,
                    float(angles[cell]),
                    um_per_px,
                    render_cfg,
                    None,
                    poses[cell],
                )
                if box is not None:
                    boxes.append(box)
                    track_ids.append(cell)

            for particle in range(n_debris):
                dx = float((debris_pos[particle, 0] + flow_vx_px_s * dt * index) % width)
                render_debris_on_canvas(
                    canvas,
                    debris_kind[particle],
                    dx,
                    float(debris_pos[particle, 1]),
                    float(debris_angle[particle]),
                    float(debris_size[particle]),
                    float(debris_elong[particle]),
                    0.55,
                    render_cfg,
                )

            image = finish_image(canvas, render_cfg, rng)
            frames.append(
                DetectionFrame(
                    image=image,
                    boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                    class_ids=np.zeros(len(boxes), dtype=np.int64),
                    video_id=f"synthclip{clip:03d}",
                    frame_index=index,
                    track_ids=np.array(track_ids, dtype=np.int64),
                    meta={
                        "generator": "training.common.detection_data (fallback renderer)",
                        "n_debris": n_debris,
                    },
                )
            )

    return frames


# ==========================================================================
# torch Dataset
# ==========================================================================


class DetectionClipDataset:
    """``torch.utils.data.Dataset`` producing CenterNet targets per frame.

    The targets come from
    :func:`sperm_sorting.detection.heads.build_centernet_targets`, which is the
    exact inverse of the decoder used at inference. Building them here rather
    than in the training loop means they are produced in the DataLoader
    workers, off the critical path, and means the augmented boxes -- not the
    originals -- are what the targets encode.

    Frames are **not** resized. The whole premise of both detector
    architectures is that a sperm head is a small blob whose scale distribution
    is effectively a point; resizing would either destroy it or invent scale
    variance that the optics cannot produce. Frames are padded to the network's
    ``size_divisor`` instead, with the image's own median so the padding does
    not read as an object.
    """

    def __init__(
        self,
        frames: Sequence[DetectionFrame],
        *,
        stride: int,
        size_divisor: int,
        num_classes: int = 1,
        augmentation: Any | None = None,
        base_seed: int = 0,
        min_overlap: float = 0.7,
    ) -> None:
        if not frames:
            raise ConfigurationError("DetectionClipDataset was given no frames")
        self.frames = list(frames)
        self.stride = int(stride)
        self.size_divisor = int(size_divisor)
        self.num_classes = int(num_classes)
        self.augmentation = augmentation
        self.base_seed = int(base_seed)
        self.min_overlap = float(min_overlap)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frames)

    def video_ids(self) -> list[str]:
        """Distinct recordings represented, for the leakage record."""
        return sorted({frame.video_id for frame in self.frames})

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch

        from sperm_sorting.detection.heads import build_centernet_targets

        frame = self.frames[index]
        raw = frame.image
        image = torch.from_numpy(np.ascontiguousarray(raw)).to(torch.float32)
        if raw.dtype == np.uint8:
            image = image / 255.0
        elif raw.dtype == np.uint16:
            image = image / 65535.0
        else:
            image = torch.clamp(image, 0.0, 1.0)
        image = image.unsqueeze(0)
        boxes = frame.boxes.copy()

        if self.augmentation is not None:
            generator = torch.Generator()
            generator.manual_seed(
                (self.base_seed * 1_000_003 + self.epoch * 9973 + int(index)) % (2**63 - 1)
            )
            image, boxes = self.augmentation(image, boxes, generator)

        image, pad_h, pad_w = _pad_to_divisor(image, self.size_divisor)
        height, width = int(image.shape[-2]), int(image.shape[-1])
        out_h, out_w = height // self.stride, width // self.stride

        class_ids = np.zeros(boxes.shape[0], dtype=np.int64)
        targets = build_centernet_targets(
            boxes,
            class_ids,
            out_h,
            out_w,
            float(self.stride),
            self.num_classes,
            self.min_overlap,
        )
        targets["n_objects"] = torch.tensor(int(boxes.shape[0]), dtype=torch.int64)
        targets["padding"] = torch.tensor([pad_h, pad_w], dtype=torch.int64)
        return image, targets


def _pad_to_divisor(image: Any, divisor: int) -> tuple[Any, int, int]:
    """Pad bottom/right so both dimensions divide by ``divisor``.

    Bottom/right rather than symmetric, so box coordinates need no shift --
    every box in the original image keeps its coordinates in the padded one.
    The padding value is the image's own median so a large constant block does
    not become the strongest feature in the frame.
    """
    import torch
    from torch.nn import functional as F

    height, width = int(image.shape[-2]), int(image.shape[-1])
    pad_h = (-height) % divisor
    pad_w = (-width) % divisor
    if pad_h == 0 and pad_w == 0:
        return image, 0, 0
    value = float(torch.median(image))
    return F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=value), pad_h, pad_w


def collate_detection_batch(batch: Sequence[tuple[Any, dict[str, Any]]]) -> tuple[Any, dict[str, Any]]:
    """Stack images and per-frame target maps into one batch.

    A plain ``default_collate`` would work only when every frame in the batch
    has the same spatial size. It does here (all frames of a dataset share a
    resolution), but the check is explicit and the error message says which
    shapes disagreed, because "stack expects each tensor to be equal size" is
    not an actionable message at 2 a.m.
    """
    import torch

    images = [item[0] for item in batch]
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) != 1:
        raise ConfigurationError(
            f"a detection batch mixes image shapes {sorted(shapes)}. Frames are "
            "never resized in this pipeline, so a batch may only contain frames of "
            "one resolution -- group by recording, or use batch size 1."
        )

    stacked = torch.stack(images, dim=0)
    keys = ("heatmap", "size", "offset", "mask")
    targets: dict[str, Any] = {
        key: torch.stack([item[1][key] for item in batch], dim=0) for key in keys
    }
    targets["n_objects"] = torch.stack([item[1]["n_objects"] for item in batch], dim=0)
    return stacked, targets


# ==========================================================================
# Entry point
# ==========================================================================


@dataclass
class DetectionSource:
    """Frames grouped by split, plus the record of how they were split."""

    splits: dict[str, list[DetectionFrame]]
    video_splits: dict[str, list[str]]
    leakage_check: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **self.info,
            "video_splits": {k: list(v) for k, v in self.video_splits.items()},
            "leakage_check": self.leakage_check,
            "splits": {
                name: {
                    "n_frames": len(frames),
                    "n_videos": len({f.video_id for f in frames}),
                    "n_objects": int(sum(f.boxes.shape[0] for f in frames)),
                }
                for name, frames in self.splits.items()
            },
        }


def load_detection_source(
    source: str,
    *,
    root: Path | None = None,
    seed: int = 1234,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    n_clips: int = 6,
    frames_per_clip: int = 8,
    width: int = 320,
    height: int = 256,
) -> DetectionSource:
    """Load frames and split them **by video**, refusing any leakage."""
    if source not in DETECTION_SOURCE_KINDS:
        raise ConfigurationError(
            f"unknown --source '{source}'; available: {', '.join(DETECTION_SOURCE_KINDS)}"
        )

    if source == "visem":
        adapter = _load_visem_adapter(root)
        video_ids = list(adapter.video_ids())
        video_splits = split_by_video(video_ids, fractions=fractions, seed=seed)
        check = assert_no_video_leakage(video_splits)
        splits = {
            name: [frame for vid in ids for frame in adapter.load_video(vid)]
            for name, ids in video_splits.items()
        }
        info = {
            "name": str(getattr(adapter, "name", "VISEM-Tracking")),
            "licence": str(getattr(adapter, "licence", VISEM_LICENCE)),
            "source": "datasets.adapters.visem",
            "root": str(root) if root else None,
            "split_unit": "video",
        }
        return DetectionSource(splits, video_splits, check, info)

    frames = build_synthetic_clips(
        n_clips, frames_per_clip, seed=seed, width=width, height=height
    )
    video_ids = sorted({frame.video_id for frame in frames})
    video_splits = split_by_video(video_ids, fractions=fractions, seed=seed)
    check = assert_no_video_leakage(video_splits)
    by_video: dict[str, list[DetectionFrame]] = {}
    for frame in frames:
        by_video.setdefault(frame.video_id, []).append(frame)
    splits = {
        name: [frame for vid in ids for frame in by_video[vid]]
        for name, ids in video_splits.items()
    }
    info = {
        "name": "sperm_sorting simulator (procedural clips)",
        "licence": "generated in-repo; no third-party terms apply",
        "source": frames[0].meta.get("generator", "unknown"),
        "split_unit": "clip (one clip == one video)",
        "seed": int(seed),
        "n_clips": n_clips,
        "frames_per_clip": frames_per_clip,
        "frame_size": [height, width],
    }
    return DetectionSource(splits, video_splits, check, info)


def _load_visem_adapter(root: Path | None) -> DetectionDatasetAdapter:
    """Import and construct the VISEM-Tracking adapter, or explain what is missing."""
    try:
        module = importlib.import_module("datasets.adapters.visem")
    except ImportError as exc:
        raise ConfigurationError(
            "--source visem requires 'datasets.adapters.visem', which is not "
            f"importable ({exc}). Use '--source synthetic' to bootstrap against "
            "the in-repo simulator until the adapter lands."
        ) from exc

    adapter_cls = None
    for name in ("VisemTrackingAdapter", "VisemAdapter", "VISEMTrackingAdapter"):
        adapter_cls = getattr(module, name, None)
        if adapter_cls is not None:
            break
    if adapter_cls is None:
        available = [n for n in dir(module) if not n.startswith("_")]
        raise ConfigurationError(
            "datasets.adapters.visem exposes no VISEM adapter class (looked for "
            "VisemTrackingAdapter, VisemAdapter, VISEMTrackingAdapter); found: "
            f"{', '.join(available) or '(nothing public)'}"
        )

    try:
        adapter = adapter_cls(root) if root is not None else adapter_cls()
    except TypeError:
        adapter = adapter_cls(root=root) if root is not None else adapter_cls()

    missing = [
        member
        for member in ("video_ids", "load_video")
        if not callable(getattr(adapter, member, None))
    ]
    if missing:
        raise ConfigurationError(
            f"{adapter_cls.__name__} does not satisfy the DetectionDatasetAdapter "
            f"protocol: missing callable(s) {', '.join(missing)}. The harness needs "
            "video_ids() -> Sequence[str] and load_video(id) -> Sequence[DetectionFrame], "
            "because a by-video split cannot be expressed over a flat frame list."
        )
    return adapter  # type: ignore[return-value]
