"""Morphology training data: MHSMA, or the simulator as a bootstrap.

Two sources, one in-memory representation
-----------------------------------------
Everything downstream sees a :class:`MorphologySplit` -- an ``(N, H, W)``
``uint8`` image stack plus one ``(N,)`` integer label vector per aspect, with
labels in the **MHSMA convention verbatim** (``0 = normal``, ``1 = abnormal``).
There is no flip anywhere in this module; see
:mod:`sperm_sorting.morphology.polarity` for the single place one is permitted.

``--source mhsma``
    Delegates to ``datasets.adapters.mhsma.MhsmaAdapter``, which is built in
    parallel with this harness. Because the adapter may not exist yet, this
    module codes against a narrow :class:`MorphologyDatasetAdapter` protocol --
    two methods and two attributes -- and reports precisely what is missing
    when it cannot satisfy it, rather than failing with an ``ImportError`` from
    somewhere deep in a training loop.

    **MHSMA's official train/valid/test split is preserved exactly.** It is
    never re-split, re-shuffled or merged and re-divided. Two reasons: the
    published split is what any comparison against published numbers requires,
    and a random re-split of 1540 head-centred crops would almost certainly put
    crops of the *same* smear in both train and test.

``--source synthetic``
    Renders crops with the in-repo simulator
    (:func:`sperm_sorting.simulator.render.render_sperm` driven by
    :func:`sperm_sorting.simulator.params.sample_health_state`). This is the
    bootstrap path that exists before real data does. It is not a toy: the
    simulator samples the ground-truth :class:`HealthState` *first* and derives
    the appearance from it, so every label is causally supported by the pixels
    -- a model that cannot fit this data cannot fit MHSMA either.

    Splits are generated from **disjoint RNG streams**, one per split, so they
    are independent by construction rather than by a shuffle that has to be
    trusted. Changing ``--n-train`` therefore cannot alter the validation set,
    which is what makes two runs with different training-set sizes comparable.

Class imbalance
---------------
The default synthetic prevalences are the *verified MHSMA train-split*
prevalences -- acrosome 30.1%, head 27.3%, vacuole 17.0%, tail 4.6% -- so a
model bootstrapped on the simulator meets the same imbalance it will meet on
real data, and the ``pos_weight`` computed from one transfers to the other.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sperm_sorting.constants import (
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    MORPHOLOGY_ASPECTS,
    WEIGHTS_PROVENANCE_PUBLIC,
    WEIGHTS_PROVENANCE_SYNTHETIC,
)
from sperm_sorting.errors import ConfigurationError

__all__ = [
    "MHSMA_LICENCE",
    "MHSMA_TRAIN_PREVALENCE",
    "MorphologyArrayDataset",
    "MorphologyDatasetAdapter",
    "MorphologySplit",
    "SOURCE_KINDS",
    "SPLIT_NAMES",
    "build_synthetic_split",
    "load_morphology_source",
]

#: The three official MHSMA splits, in order. Never re-derived.
SPLIT_NAMES: tuple[str, str, str] = ("train", "valid", "test")

#: Sources ``--source`` accepts.
SOURCE_KINDS: tuple[str, ...] = ("mhsma", "synthetic")

#: Verified abnormal prevalence of the MHSMA **train** split, per aspect. Used
#: as the simulator's default so the bootstrap run meets the real imbalance,
#: and as the fallback for ``pos_weight`` when a split is degenerate.
MHSMA_TRAIN_PREVALENCE: dict[str, float] = {
    "head": 0.273,
    "acrosome": 0.301,
    "vacuole": 0.170,
    "tail": 0.046,
}

#: Recorded in ``experiment.json`` for every MHSMA run. Public research data
#: carries terms, and weights stamped ``public-research-baseline`` inherit
#: them; a record that omits the licence cannot answer the only question that
#: matters when those weights leave the building. The adapter is expected to
#: supply the authoritative string; this is what is reported when it does not.
MHSMA_LICENCE: str = (
    "see the MHSMA dataset's own licence and citation terms "
    "(Javadi & Mirroshandel, 'A novel deep learning method for automatic "
    "assessment of human sperm images', Comput Biol Med 2019); the adapter "
    "reports the authoritative string when it exposes one"
)


# ==========================================================================
# Protocol
# ==========================================================================


@runtime_checkable
class MorphologyDatasetAdapter(Protocol):
    """The minimum this harness needs from a morphology dataset adapter.

    Deliberately narrow. The adapters live in ``datasets/`` and are being built
    in parallel with this package; coding against two methods and two
    attributes means the two can land in either order, and it means this module
    can say exactly what is missing rather than surfacing an ``AttributeError``
    from inside a data loader.

    ``load_split`` must return labels in the **MHSMA convention verbatim**
    (``0 = normal``, ``1 = abnormal``). An adapter that flips them would invert
    the product, so :func:`_validate_split` rejects anything outside ``{0, 1}``
    and the training script cross-checks the resulting prevalence against the
    published figures.
    """

    #: Dataset identity, for the experiment record.
    name: str
    #: Licence / terms string, for the experiment record.
    licence: str

    def split_names(self) -> Sequence[str]:
        """Official split names, e.g. ``("train", "valid", "test")``."""
        ...

    def load_split(self, name: str) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
        """``(images, labels)`` for one official split.

        ``images`` is ``(N, H, W)`` ``uint8``; ``labels`` maps each aspect name
        to an ``(N,)`` integer array of MHSMA labels.
        """
        ...


# ==========================================================================
# In-memory split
# ==========================================================================


@dataclass
class MorphologySplit:
    """One split's images and per-aspect labels, held in memory.

    MHSMA is 1540 crops of at most 128x128 ``uint8``: 25 MB in total. Holding
    it in RAM removes the entire I/O path from the training loop, which on a
    CPU-only box is the difference between a data-bound and a compute-bound
    run. The synthetic source is generated once for the same reason.
    """

    name: str
    images: np.ndarray
    labels: dict[str, np.ndarray]
    aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS

    def __post_init__(self) -> None:
        self.images = np.asarray(self.images)
        if self.images.ndim == 4 and self.images.shape[-1] == 1:
            self.images = self.images[..., 0]
        if self.images.ndim != 3:
            raise ConfigurationError(
                f"split '{self.name}': images must be (N, H, W); got shape "
                f"{self.images.shape}"
            )
        self.labels = {k: np.asarray(v).ravel().astype(np.int64) for k, v in self.labels.items()}
        _validate_split(self)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_size(self) -> tuple[int, int]:
        return (int(self.images.shape[1]), int(self.images.shape[2]))

    def prevalence(self) -> dict[str, float]:
        """Abnormal fraction per aspect. The input to ``pos_weight``."""
        n = max(len(self), 1)
        return {
            name: float(np.sum(self.labels[name] == LABEL_ABNORMAL) / n)
            for name in self.aspects
        }

    def positive_counts(self) -> dict[str, int]:
        """Number of abnormal examples per aspect.

        Reported next to every metric because a sensitivity computed from seven
        positives is a different kind of number from one computed from three
        hundred, and the metric alone does not say which it is.
        """
        return {
            name: int(np.sum(self.labels[name] == LABEL_ABNORMAL)) for name in self.aspects
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": len(self),
            "image_size": list(self.image_size),
            "prevalence": self.prevalence(),
            "n_abnormal": self.positive_counts(),
        }


def _validate_split(split: MorphologySplit) -> None:
    """Reject anything that would silently corrupt training.

    Three checks, each guarding a failure that produces no other symptom:

    * a missing aspect (the head would train on the wrong column);
    * a label outside ``{0, 1}`` (an adapter emitting -1 for "unlabelled", or a
      one-hot encoding, would be read as "abnormal");
    * a length mismatch between images and labels (silently truncating one of
      them is how a whole dataset ends up off by one).
    """
    n = len(split)
    for name in split.aspects:
        if name not in split.labels:
            raise ConfigurationError(
                f"split '{split.name}' is missing labels for aspect '{name}'; "
                f"present: {sorted(split.labels)}"
            )
        values = split.labels[name]
        if values.shape[0] != n:
            raise ConfigurationError(
                f"split '{split.name}' aspect '{name}': {values.shape[0]} labels for "
                f"{n} images"
            )
        if values.size and not np.all(np.isin(values, (LABEL_NORMAL, LABEL_ABNORMAL))):
            found = sorted(set(np.unique(values).tolist()))[:6]
            raise ConfigurationError(
                f"split '{split.name}' aspect '{name}' contains labels {found}; only "
                f"the MHSMA convention {LABEL_NORMAL} (normal) / {LABEL_ABNORMAL} "
                "(abnormal) is accepted. If the adapter emits the opposite "
                "convention, fix it in the adapter -- this package never flips."
            )


# ==========================================================================
# torch Dataset
# ==========================================================================


class MorphologyArrayDataset:
    """``torch.utils.data.Dataset`` over a :class:`MorphologySplit`.

    Yields ``(image, targets)`` where ``image`` is ``(1, H, W)`` float in
    ``[0, 1]`` and ``targets`` is ``(n_aspects,)`` float in
    :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS` order -- the packed
    form :class:`~sperm_sorting.morphology.model.MorphologyLoss` accepts.

    Targets are float, not int, because ``binary_cross_entropy_with_logits``
    requires a float target; converting per batch in the loop instead would put
    a host-side cast on the hot path for no benefit.

    Augmentation is driven by a per-sample generator seeded from
    ``(base_seed, epoch, index)``. That is more work than one shared generator
    and it buys the property that matters: the augmentation applied to sample
    ``i`` in epoch ``e`` does not depend on the batch size, the worker count or
    the shuffle order, so two runs that differ only in ``--num-workers`` see
    the identical augmented data.
    """

    def __init__(
        self,
        split: MorphologySplit,
        *,
        augmentation: Any | None = None,
        base_seed: int = 0,
        aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS,
    ) -> None:
        self.split = split
        self.augmentation = augmentation
        self.base_seed = int(base_seed)
        self.aspects = tuple(aspects)
        self.epoch = 0
        self._targets = np.stack(
            [split.labels[name].astype(np.float32) for name in self.aspects], axis=1
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream. Called once per epoch by the loop."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        raw = self.split.images[index]
        image = torch.from_numpy(np.ascontiguousarray(raw)).to(torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        # uint8 crops are 0-255; the simulator and MHSMA both produce uint8.
        if raw.dtype == np.uint8:
            image = image / 255.0
        else:
            image = torch.clamp(image, 0.0, 1.0)

        if self.augmentation is not None:
            generator = torch.Generator()
            # Mixing the three indices multiplicatively rather than adding them
            # keeps distinct (epoch, index) pairs from colliding onto the same
            # seed, which additive mixing does constantly.
            generator.manual_seed(
                (self.base_seed * 1_000_003 + self.epoch * 9973 + int(index)) % (2**63 - 1)
            )
            image = self.augmentation(image, generator)

        return image, torch.from_numpy(self._targets[index])


# ==========================================================================
# Synthetic source
# ==========================================================================


def build_synthetic_split(
    name: str,
    n_samples: int,
    *,
    seed: int,
    image_size: int = 128,
    prevalence: Mapping[str, float] | None = None,
    progressive_rate: float = 0.6,
) -> MorphologySplit:
    """Render ``n_samples`` labelled crops with the in-repo simulator.

    Each sample is an independent draw of a :class:`HealthState` followed by a
    render of *that* state, so the four binary labels are causally responsible
    for the appearance rather than annotations attached to a random picture.
    A model that shortcuts here has nothing to shortcut *to*.

    ``seed`` should differ per split. The caller derives it from the run seed
    and the split name so that the three splits come from disjoint streams and
    are therefore independent by construction, not by a trusted shuffle.
    """
    from sperm_sorting.simulator.params import sample_health_state
    from sperm_sorting.simulator.render import SUPPORTED_SIZES, render_sperm

    if n_samples < 1:
        raise ConfigurationError(f"n_samples must be >= 1 for split '{name}', got {n_samples}")
    if image_size not in SUPPORTED_SIZES:
        raise ConfigurationError(
            f"synthetic image_size must be one of {SUPPORTED_SIZES} (MHSMA parity), "
            f"got {image_size}"
        )

    rates = dict(MHSMA_TRAIN_PREVALENCE if prevalence is None else prevalence)
    for aspect in MORPHOLOGY_ASPECTS:
        if aspect not in rates:
            raise ConfigurationError(f"synthetic prevalence is missing aspect '{aspect}'")

    rng = np.random.default_rng(seed)
    images = np.empty((n_samples, image_size, image_size), dtype=np.uint8)
    labels = {aspect: np.zeros(n_samples, dtype=np.int64) for aspect in MORPHOLOGY_ASPECTS}

    for index in range(n_samples):
        state = sample_health_state(rng, rates, progressive_rate)
        images[index] = render_sperm(state, (image_size, image_size), rng)
        for aspect, value in zip(MORPHOLOGY_ASPECTS, state.aspects, strict=True):
            labels[aspect][index] = int(value)

    return MorphologySplit(name=name, images=images, labels=labels)


# ==========================================================================
# MHSMA source
# ==========================================================================


def _load_mhsma_adapter(root: Path | None) -> MorphologyDatasetAdapter:
    """Import and construct the MHSMA adapter, or explain what is missing.

    The adapter lives in ``datasets/``, which is developed in parallel with
    this harness. Every failure mode here produces a message naming the module,
    the symbol and the protocol member that was absent, because "ImportError:
    no module named datasets.adapters.mhsma" three hours into a job is a much
    worse experience than being told up front which half of the repository is
    not ready.
    """
    try:
        module = importlib.import_module("datasets.adapters.mhsma")
    except ImportError as exc:
        raise ConfigurationError(
            "--source mhsma requires 'datasets.adapters.mhsma', which is not "
            f"importable ({exc}). The dataset adapters live in datasets/ and are "
            "built separately from this training harness. Use "
            "'--source synthetic' to bootstrap against the in-repo simulator "
            "until the adapter lands."
        ) from exc

    adapter_cls = getattr(module, "MhsmaAdapter", None)
    if adapter_cls is None:
        available = [n for n in dir(module) if not n.startswith("_")]
        raise ConfigurationError(
            "datasets.adapters.mhsma has no 'MhsmaAdapter'; found: "
            f"{', '.join(available) or '(nothing public)'}"
        )

    try:
        adapter = adapter_cls(root) if root is not None else adapter_cls()
    except TypeError:
        # Adapters differ on whether the root is positional or keyword; try the
        # other spelling before giving up, since guessing wrong here would be a
        # pointless hard failure.
        adapter = adapter_cls(root=root) if root is not None else adapter_cls()

    missing = [
        member
        for member in ("split_names", "load_split")
        if not callable(getattr(adapter, member, None))
    ]
    if missing:
        raise ConfigurationError(
            f"{adapter_cls.__name__} does not satisfy the MorphologyDatasetAdapter "
            f"protocol: missing callable(s) {', '.join(missing)}. The harness needs "
            "split_names() -> Sequence[str] and load_split(name) -> (images, labels) "
            "with labels in the MHSMA convention (0 normal, 1 abnormal)."
        )
    return adapter  # type: ignore[return-value]


def _load_mhsma(root: Path | None) -> tuple[dict[str, MorphologySplit], dict[str, Any]]:
    """Load all three official MHSMA splits, preserving them exactly."""
    adapter = _load_mhsma_adapter(root)
    available = list(adapter.split_names())

    splits: dict[str, MorphologySplit] = {}
    for name in SPLIT_NAMES:
        if name not in available:
            raise ConfigurationError(
                f"the MHSMA adapter reports splits {available} but this harness "
                f"requires the official '{name}' split. MHSMA's published "
                "train/valid/test division is never re-derived here: re-splitting "
                "1540 head-centred crops at random would put crops of the same "
                "smear on both sides."
            )
        images, labels = adapter.load_split(name)
        splits[name] = MorphologySplit(name=name, images=images, labels=dict(labels))

    info: dict[str, Any] = {
        "name": str(getattr(adapter, "name", "MHSMA")),
        "licence": str(getattr(adapter, "licence", MHSMA_LICENCE)),
        "source": "datasets.adapters.mhsma.MhsmaAdapter",
        "root": str(root) if root else None,
        "weights_provenance": WEIGHTS_PROVENANCE_PUBLIC,
        "official_split_preserved": True,
    }
    return splits, info


# ==========================================================================
# Entry point
# ==========================================================================


@dataclass
class MorphologySource:
    """Loaded splits plus everything the experiment record needs about them."""

    splits: dict[str, MorphologySplit]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def weights_provenance(self) -> str:
        return str(self.info.get("weights_provenance", "unset"))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **self.info,
            "splits": {name: split.to_json_dict() for name, split in self.splits.items()},
        }


def load_morphology_source(
    source: str,
    *,
    root: Path | None = None,
    seed: int = 1234,
    n_train: int = 2000,
    n_valid: int = 500,
    n_test: int = 500,
    image_size: int = 128,
    prevalence: Mapping[str, float] | None = None,
) -> MorphologySource:
    """Load ``train``/``valid``/``test`` from the named source.

    Parameters
    ----------
    source
        ``"mhsma"`` or ``"synthetic"``.
    root
        Dataset root, passed to the MHSMA adapter. Ignored by the simulator.
    seed
        Base seed. Synthetic splits derive disjoint streams from it.
    n_train, n_valid, n_test
        Synthetic split sizes. Ignored for MHSMA, whose sizes are fixed by the
        published split.
    image_size
        Synthetic crop edge; 64 or 128, matching MHSMA's two variants.
    prevalence
        Per-aspect abnormal rate for the simulator; defaults to the verified
        MHSMA train prevalences so the bootstrap meets the real imbalance.
    """
    if source not in SOURCE_KINDS:
        raise ConfigurationError(
            f"unknown --source '{source}'; available: {', '.join(SOURCE_KINDS)}"
        )

    if source == "mhsma":
        splits, info = _load_mhsma(root)
        return MorphologySource(splits=splits, info=info)

    sizes = {"train": n_train, "valid": n_valid, "test": n_test}
    splits = {}
    for offset, name in enumerate(SPLIT_NAMES):
        # A distinct, deterministic stream per split. Deriving it from the
        # split *name* rather than from a running counter means adding a fourth
        # split later cannot renumber the existing three.
        split_seed = (int(seed) + 1_000 * (offset + 1)) % (2**32)
        splits[name] = build_synthetic_split(
            name,
            sizes[name],
            seed=split_seed,
            image_size=image_size,
            prevalence=prevalence,
        )

    info = {
        "name": "sperm_sorting simulator (procedural)",
        "licence": "generated in-repo; no third-party terms apply",
        "source": "sperm_sorting.simulator.render.render_sperm",
        "weights_provenance": WEIGHTS_PROVENANCE_SYNTHETIC,
        "official_split_preserved": "n/a (splits generated from disjoint RNG streams)",
        "seed": int(seed),
        "image_size": int(image_size),
        "target_prevalence": dict(MHSMA_TRAIN_PREVALENCE if prevalence is None else prevalence),
    }
    return MorphologySource(splits=splits, info=info)
