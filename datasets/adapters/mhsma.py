"""MHSMA: Modified Human Sperm Morphology Analysis dataset.

Source: https://github.com/soroushj/mhsma-dataset (the ``.npy`` files are
committed *in that repository*; MHSMA is not on Zenodo).

What it is
----------
1540 grayscale crops of single sperm -- 1000 train / 240 valid / 300 test --
from 235 male-factor-infertility patients, at two sizes (128x128 and 64x64),
each labelled independently for four morphology aspects: head, acrosome,
vacuole and tail. Derived from HSMA-DS (Ghasemian et al., 2015), captured at
x400 and x600. The head is roughly centred; **the tail is not entirely
visible**, which is why the ``tail`` aspect is the weakest supervision in the
set and why a tail verdict from a public-weights model should be treated with
suspicion. Eighteen files in total: 6 image arrays and 12 label arrays, all
``uint8``.

The label-polarity trap
-----------------------
**0 = normal, 1 = abnormal**, for every aspect. The upstream README calls the
*normal* class "positive", so its "% Positive" column is the percentage of
**normal** cells, not the abnormality prevalence. Read that table the natural
way and every number inverts: acrosome becomes 69.9% abnormal instead of
30.1%, and a model trained on the flipped target sorts *for* the defect it was
built to reject. In a device that physically separates cells, that is not a
metrics bug -- it is the product doing the opposite of its purpose, with a
confusion matrix that looks entirely healthy because it is symmetric.

So :meth:`MhsmaAdapter.validate` does not trust the convention, it **measures**
it: the abnormal prevalence of every aspect in every split is compared against
the published figures, and a copy whose prevalences come out near ``100 - p``
raises :class:`~sperm_sorting.errors.DatasetValidationError` naming the affected
aspects. This is the single check in this package that stops an inverted product
from shipping, so it raises rather than returning a report that a caller might
not read. See :mod:`sperm_sorting.morphology.polarity` for the corresponding
runtime contract: labels are consumed *verbatim*, the network's logit is
``P(abnormal)``, and the only flip in the codebase lives in the inference
adapter.

Split naming
------------
Upstream uses ``valid``, not ``val``. Both are accepted here and normalised to
``valid`` -- getting a ``FileNotFoundError`` for ``y_head_val.npy`` is a
five-minute detour that adds nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from sperm_sorting.constants import (
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    MORPHOLOGY_ASPECTS,
)
from sperm_sorting.errors import DatasetValidationError

from ..validators.integrity import (
    CheckStatus,
    ValidationReport,
    check_file_present,
    check_label_range,
    check_npy_header,
)
from ..validators.leakage import mhsma_split_leakage_note
from .base import CaptureConditions, DatasetAdapter, DatasetInfo

__all__ = [
    "MHSMA_SPLITS",
    "PUBLISHED_ABNORMAL_PREVALENCE_PCT",
    "PUBLISHED_SPLIT_SIZES",
    "MhsmaAdapter",
    "MhsmaAugmentation",
    "MhsmaDataset",
    "normalize_split",
]

# --------------------------------------------------------------------------
# Optional torch. The adapter's array API must work without it; only
# MhsmaDataset needs a tensor library.
# --------------------------------------------------------------------------
try:  # pragma: no cover - trivially environment-dependent
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = str(exc)

    class _TorchDataset:  # type: ignore[no-redef]
        """Stand-in so the class body below still builds without torch."""


#: Canonical split names, in the order a report should list them.
MHSMA_SPLITS: Final[tuple[str, str, str]] = ("train", "valid", "test")

#: Accepted aliases -> canonical name. Upstream says ``valid``.
_SPLIT_ALIASES: Final[dict[str, str]] = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
    "testing": "test",
}

#: Aspects as named in the *filenames*. Note this is alphabetical and is **not**
#: the model's head order: that is ``constants.MORPHOLOGY_ASPECTS``
#: (head, acrosome, vacuole, tail). Keeping the two apart on purpose -- silently
#: reusing the filename order as the tensor order is how an acrosome logit ends
#: up scored against head labels.
_FILE_ASPECTS: Final[tuple[str, ...]] = ("acrosome", "head", "vacuole", "tail")

#: Image side lengths shipped upstream.
_SIZES: Final[tuple[int, int]] = (64, 128)

#: Official split sizes.
PUBLISHED_SPLIT_SIZES: Final[dict[str, int]] = {"train": 1000, "valid": 240, "test": 300}

#: Published **abnormal** prevalence, in percent, per split and aspect.
#:
#: These are the upstream figures re-expressed in the correct polarity (the
#: dataset README publishes the complement and calls it "% Positive"). They
#: correspond to exact counts -- e.g. valid/tail 2.92% is 7 abnormal tails out
#: of 240, the smallest positive class in the dataset by a wide margin, which is
#: why a tail metric on the validation split has an enormous confidence interval
#: and should never be reported without one.
PUBLISHED_ABNORMAL_PREVALENCE_PCT: Final[dict[str, dict[str, float]]] = {
    "train": {"acrosome": 30.10, "head": 27.30, "vacuole": 17.00, "tail": 4.60},
    "valid": {"acrosome": 27.50, "head": 26.67, "vacuole": 12.92, "tail": 2.92},
    "test": {"acrosome": 29.00, "head": 27.00, "vacuole": 12.67, "tail": 5.33},
}

#: Default tolerance, in percentage points, for "matches the published figure".
#: 2.0 points is loose enough for a legitimately reduced copy and far tighter
#: than the ~40-95 point gap an inversion produces.
DEFAULT_PREVALENCE_TOLERANCE_PCT: Final[float] = 2.0


def normalize_split(split: str) -> str:
    """Canonicalise a split name; ``val`` -> ``valid``.

    Raises
    ------
    ValueError
        On an unknown name, listing what is accepted.
    """
    key = str(split).strip().lower()
    try:
        return _SPLIT_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown MHSMA split {split!r}. Accepted: "
            f"{sorted(_SPLIT_ALIASES)} (all normalise to one of {list(MHSMA_SPLITS)})"
        ) from None


def _normalize_aspect(aspect: str) -> str:
    key = str(aspect).strip().lower()
    if key not in _FILE_ASPECTS:
        raise ValueError(
            f"unknown MHSMA aspect {aspect!r}; expected one of {list(_FILE_ASPECTS)}"
        )
    return key


# ==========================================================================
# Augmentation
# ==========================================================================


@dataclass(frozen=True, slots=True)
class MhsmaAugmentation:
    """Label-preserving augmentation for morphology crops.

    What is here and what is deliberately not:

    * **Flips and 90-degree rotations** -- included. A sperm has no canonical
      orientation in the field, and none of the four aspects is defined by
      handedness, so these are exactly label-preserving.
    * **Small intensity jitter** -- included. Illumination and exposure differ
      between microscopes, and this is the cheapest stand-in for that shift.
    * **Small integer translations** -- included, at up to a few pixels. The
      head is roughly centred upstream but will not be perfectly centred by a
      real detector's box.
    * **Scaling, shear and aspect-ratio changes** -- excluded, on purpose.
      Head morphology is largely a length-to-width judgement; anisotropic
      warping changes the exact quantity the model is being asked about, so it
      does not augment the label, it corrupts it. The same reasoning drives the
      letterboxing rule in :mod:`sperm_sorting.cropping.extractor`.
    * **Elastic/cutout-style distortion** -- excluded. Vacuoles are small
      intensity depressions inside the head; a transform that can invent or
      erase one is not label-preserving for the ``vacuole`` aspect.

    Randomness is drawn from a caller-supplied :class:`numpy.random.Generator`
    seeded from ``(seed, index, epoch)``, so augmentation is reproducible and
    identical whether the loader runs with 0 or 8 workers.
    """

    horizontal_flip: bool = True
    vertical_flip: bool = True
    rot90: bool = True
    #: Maximum multiplicative contrast jitter, e.g. 0.1 -> factor in [0.9, 1.1].
    contrast_jitter: float = 0.10
    #: Maximum additive brightness jitter in 8-bit grey levels.
    brightness_jitter: float = 12.0
    #: Maximum absolute integer shift in pixels, applied independently per axis.
    max_translate_px: int = 4

    def __post_init__(self) -> None:
        if self.contrast_jitter < 0 or self.brightness_jitter < 0:
            raise ValueError("jitter magnitudes must be non-negative")
        if self.max_translate_px < 0:
            raise ValueError("max_translate_px must be non-negative")

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Augment a 2-D ``uint8`` image, returning a new ``uint8`` array."""
        out = np.asarray(image)
        if out.ndim != 2:
            raise ValueError(f"expected a 2-D grayscale crop, got shape {out.shape}")
        if self.horizontal_flip and rng.random() < 0.5:
            out = out[:, ::-1]
        if self.vertical_flip and rng.random() < 0.5:
            out = out[::-1, :]
        if self.rot90:
            k = int(rng.integers(0, 4))
            if k:
                out = np.rot90(out, k)
        if self.max_translate_px:
            dy = int(rng.integers(-self.max_translate_px, self.max_translate_px + 1))
            dx = int(rng.integers(-self.max_translate_px, self.max_translate_px + 1))
            if dy or dx:
                # Edge padding rather than zero fill: a black band at the border
                # is a high-contrast edge the model can latch onto, and it does
                # not resemble anything a real crop contains.
                out = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = np.ascontiguousarray(out)

        if self.contrast_jitter or self.brightness_jitter:
            gain = 1.0 + float(rng.uniform(-self.contrast_jitter, self.contrast_jitter))
            bias = float(rng.uniform(-self.brightness_jitter, self.brightness_jitter))
            mean = float(out.mean())
            # Jitter contrast about the crop's own mean so brightness and
            # contrast stay separable knobs rather than one confounded one.
            adjusted = (out.astype(np.float32) - mean) * gain + mean + bias
            out = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)
        return out


# ==========================================================================
# Adapter
# ==========================================================================


class MhsmaAdapter(DatasetAdapter):
    """Reader for the 18 MHSMA ``.npy`` files.

    Parameters
    ----------
    root
        Either the cloned repository (which contains an ``mhsma/`` folder) or
        the ``mhsma/`` folder itself. Both are accepted; see
        :meth:`_resolve_root`.
    require_present
        See :class:`~datasets.adapters.base.DatasetAdapter`.
    mmap
        Memory-map the image arrays instead of reading them into RAM. On by
        default: the six image arrays total ~28 MB, which is not a problem in
        itself, but memory-mapped arrays are read-only, and a read-only training
        set is one fewer way for an augmentation to silently mutate the corpus
        under a second epoch.
    """

    info = DatasetInfo(
        name="mhsma",
        title="MHSMA: Modified Human Sperm Morphology Analysis dataset",
        url="https://github.com/soroushj/mhsma-dataset",
        license_key="mhsma",
        annotation_level="per-image, 4 independent binary morphology aspects",
        approximate_size="~30 MB (18 .npy files, committed in the git repository)",
        capture=CaptureConditions(
            objective_magnification=None,
            total_magnification=None,
            contrast_mode="brightfield (stained smear)",
            stained=True,
            camera=None,
            fps_range=None,
            fps_uniform=None,
            resolution=(128, 128),
            um_per_px=None,
            notes=(
                "Derived from HSMA-DS (Ghasemian et al. 2015), captured at x400 and "
                "x600 -- two magnifications mixed within one corpus, with no per-image "
                "record of which. Crops are single, roughly head-centred sperm; the "
                "tail is not entirely visible. 235 male-factor-infertility patients."
            ),
        ),
        domain_shift_notes=[
            "Stained, fixed smears versus the device's live, unstained wet "
            "preparation: head contrast comes from dye uptake here and from "
            "refractive index there, so the intensity statistics the model keys on "
            "are not the ones it will meet.",
            "Two magnifications (x400 and x600) are mixed without per-image "
            "provenance, so the pixels-per-micrometre of any given crop is unknown; "
            "absolute size cannot be learned from this dataset, only shape.",
            "Crops are pre-centred on the head. Device crops come from a detector "
            "box, so head position and padding will vary; train with translation "
            "augmentation or the model will rely on centring as a feature.",
            "The tail is not fully visible in the source images, so the 'tail' label "
            "is supervision on a fragment. A tail verdict from public weights is the "
            "least transferable of the four aspects.",
            "Single, isolated cells only. Device fields are crowded, so crops will "
            "contain neighbouring cells and debris this dataset never shows.",
            "Selection is from male-factor-infertility patients, so the abnormality "
            "prevalence here is not the prevalence of any sample the device sees; "
            "any calibrated threshold must be re-fitted on device data.",
        ],
        expected_layout=(
            "  <root>/mhsma/x_128_train.npy   (1000, 128, 128) uint8\n"
            "  <root>/mhsma/x_128_valid.npy   ( 240, 128, 128) uint8\n"
            "  <root>/mhsma/x_128_test.npy    ( 300, 128, 128) uint8\n"
            "  <root>/mhsma/x_64_{train,valid,test}.npy  same counts, 64x64\n"
            "  <root>/mhsma/y_{acrosome,head,vacuole,tail}_{train,valid,test}.npy\n"
            "  (18 files total, every one uint8; <root> may also be the mhsma/ folder)"
        ),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        require_present: bool = True,
        mmap: bool = True,
    ) -> None:
        self._mmap = bool(mmap)
        self._image_cache: dict[tuple[str, int], np.ndarray] = {}
        self._label_cache: dict[tuple[str, str], np.ndarray] = {}
        super().__init__(root, require_present=require_present)

    # ------------------------------------------------------------ discovery

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """Accept the repository root, the ``mhsma/`` folder, or a clone parent.

        People clone ``mhsma-dataset`` and point at the clone; people also
        download the folder and point at it directly. Both are obviously
        correct from where the user is standing, so both work.
        """
        candidates = [
            given / "mhsma",
            given,
            given / "mhsma-dataset" / "mhsma",
            given / "mhsma-dataset",
        ]
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "x_128_train.npy").is_file():
                return candidate
        # A directory holding *some* of the 18 files is an incomplete copy, and
        # the useful error there enumerates what is missing (validate() does).
        # A directory holding none of them is not an MHSMA copy at all, so
        # ``None`` here makes the constructor raise DatasetNotFoundError with
        # the download URL -- which is the more useful answer for an empty or
        # wrong path.
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("x_*.npy")):
                return candidate
        return None

    def image_path(self, split: str, size: int) -> Path:
        """Path of the image array for ``split`` at ``size`` (64 or 128)."""
        split = normalize_split(split)
        if int(size) not in _SIZES:
            raise ValueError(f"MHSMA ships {list(_SIZES)} pixel crops, not {size}")
        return self.root / f"x_{int(size)}_{split}.npy"

    def label_path(self, split: str, aspect: str) -> Path:
        """Path of the label array for ``split`` and ``aspect``."""
        return self.root / f"y_{_normalize_aspect(aspect)}_{normalize_split(split)}.npy"

    def expected_files(self) -> list[Path]:
        """All 18 files this dataset must contain, in a stable order."""
        root = self._root or self._given_root
        files = [root / f"x_{size}_{split}.npy" for size in _SIZES for split in MHSMA_SPLITS]
        files += [
            root / f"y_{aspect}_{split}.npy"
            for aspect in _FILE_ASPECTS
            for split in MHSMA_SPLITS
        ]
        return files

    # ----------------------------------------------------------------- data

    def splits(self) -> list[str]:
        return list(MHSMA_SPLITS)

    def images(self, split: str, size: int = 128) -> np.ndarray:
        """``(N, size, size)`` ``uint8`` images for ``split``.

        The array is cached and, with ``mmap=True``, read-only. Copy before
        mutating.
        """
        split = normalize_split(split)
        key = (split, int(size))
        if key not in self._image_cache:
            path = self.require_path(self.image_path(split, size), f"images for {split}@{size}")
            self._image_cache[key] = np.load(path, mmap_mode="r" if self._mmap else None)
        return self._image_cache[key]

    def labels(self, split: str, aspect: str) -> np.ndarray:
        """``(N,)`` ``uint8`` labels for one aspect: 0 = normal, 1 = abnormal.

        Returned **verbatim**. There is no flip here and there must never be
        one: see :mod:`sperm_sorting.morphology.polarity`.
        """
        split = normalize_split(split)
        aspect = _normalize_aspect(aspect)
        key = (split, aspect)
        if key not in self._label_cache:
            path = self.require_path(
                self.label_path(split, aspect), f"{aspect} labels for {split}"
            )
            self._label_cache[key] = np.load(path)
        return self._label_cache[key]

    def label_matrix(self, split: str) -> np.ndarray:
        """``(N, 4)`` labels in :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS`
        order -- head, acrosome, vacuole, tail.

        Column order follows the model's head order, not the alphabetical file
        order, so that a column index means the same thing here, in the loss and
        in the metrics report.
        """
        columns = [self.labels(split, aspect) for aspect in MORPHOLOGY_ASPECTS]
        lengths = {len(c) for c in columns}
        if len(lengths) != 1:
            raise DatasetValidationError(
                f"MHSMA {split}: aspect label arrays have differing lengths "
                f"{dict(zip(MORPHOLOGY_ASPECTS, (len(c) for c in columns), strict=True))}; the four "
                "aspects must label the same images."
            )
        return np.stack(columns, axis=1)

    def __len__(self) -> int:
        """Total images across the three splits (at 128 px; sizes agree)."""
        return sum(int(self.images(split, 128).shape[0]) for split in MHSMA_SPLITS)

    def split_size(self, split: str) -> int:
        return int(self.images(split, 128).shape[0])

    # ----------------------------------------------------------- prevalence

    def prevalence(self, split: str) -> dict[str, float]:
        """**Abnormal** prevalence per aspect as a fraction in ``[0, 1]``.

        Keys follow :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS`
        (head, acrosome, vacuole, tail). The value is the fraction of images
        whose label equals :data:`~sperm_sorting.constants.LABEL_ABNORMAL`, i.e.
        the *complement* of the upstream README's "% Positive" column. See
        :func:`prevalence_percent` for the same numbers in the published units.
        """
        out: dict[str, float] = {}
        for aspect in MORPHOLOGY_ASPECTS:
            labels = np.asarray(self.labels(split, aspect))
            if labels.size == 0:
                raise DatasetValidationError(
                    f"MHSMA {split}/{aspect}: label array is empty; prevalence is undefined"
                )
            out[aspect] = float(np.count_nonzero(labels == LABEL_ABNORMAL) / labels.size)
        return out

    def prevalence_percent(self, split: str) -> dict[str, float]:
        """:meth:`prevalence` in percent, directly comparable with
        :data:`PUBLISHED_ABNORMAL_PREVALENCE_PCT`."""
        return {k: 100.0 * v for k, v in self.prevalence(split).items()}

    def class_counts(self, split: str) -> dict[str, dict[str, int]]:
        """Per-aspect ``{"normal": n, "abnormal": m}`` counts.

        Useful for a loss ``pos_weight``: with 7 abnormal tails in 240
        validation images, an unweighted BCE simply learns "always normal" and
        reports 97% accuracy.
        """
        out: dict[str, dict[str, int]] = {}
        for aspect in MORPHOLOGY_ASPECTS:
            labels = np.asarray(self.labels(split, aspect))
            n_abnormal = int(np.count_nonzero(labels == LABEL_ABNORMAL))
            out[aspect] = {
                "normal": int(labels.size) - n_abnormal,
                "abnormal": n_abnormal,
            }
        return out

    def pos_weight(self, split: str = "train") -> dict[str, float]:
        """``n_normal / n_abnormal`` per aspect, for ``BCEWithLogitsLoss``.

        Positive class is *abnormal*, matching
        :data:`sperm_sorting.morphology.polarity.POSITIVE_CLASS`. Returns
        ``inf`` for an aspect with no abnormal example rather than a silently
        clamped value -- an aspect with no positives cannot be trained and the
        caller must decide what to do about it.
        """
        out: dict[str, float] = {}
        for aspect, counts in self.class_counts(split).items():
            out[aspect] = (
                float("inf")
                if counts["abnormal"] == 0
                else counts["normal"] / counts["abnormal"]
            )
        return out

    # ------------------------------------------------------------ validation

    def validate(
        self,
        *,
        tolerance_pct: float = DEFAULT_PREVALENCE_TOLERANCE_PCT,
        check_prevalence: bool = True,
    ) -> ValidationReport:
        """Structural checks plus the polarity measurement.

        Ordinary problems (a missing file, a wrong dtype, a prevalence that is
        merely off) are collected into the returned report with
        :attr:`~datasets.validators.integrity.ValidationReport.ok` set to False.

        **Label-polarity inversion raises immediately.** It is the one failure
        mode where continuing produces a model that is confidently, symmetrically
        wrong -- every metric looks normal, and the device sorts out exactly the
        cells it was built to keep. A returned report can be ignored; an
        exception cannot.

        Parameters
        ----------
        tolerance_pct
            Allowed absolute deviation, in percentage points, from the published
            abnormal prevalence.
        check_prevalence
            Set False only to inspect a deliberately-subsetted copy. Doing so
            disables the inversion check, which is why it is a keyword argument
            with a loud name rather than a default.

        Raises
        ------
        DatasetValidationError
            When one or more aspects look inverted.
        """
        report = self._new_report()
        report.context["mmap"] = self._mmap
        root = self._root or self._given_root

        # -- 1. presence. This is the only failure that stops everything else:
        #       nothing can be measured on a file that is not there.
        presence = [
            check_file_present(root / f"x_{size}_{split}.npy", name=f"file:x_{size}_{split}")
            for size in _SIZES
            for split in MHSMA_SPLITS
        ] + [
            check_file_present(root / f"y_{aspect}_{split}.npy", name=f"file:y_{aspect}_{split}")
            for aspect in _FILE_ASPECTS
            for split in MHSMA_SPLITS
        ]
        report.extend([c for c in presence if c.failed])
        if any(c.failed for c in presence):
            report.add(
                "structure",
                CheckStatus.FAIL,
                f"{sum(1 for c in presence if c.failed)} of the 18 MHSMA .npy files are "
                "missing or empty; content and polarity cannot be checked. Re-download "
                f"from {self.info.url}.",
            )
            report.checks.append(mhsma_split_leakage_note())
            return report
        report.add("structure", CheckStatus.PASS, "all 18 .npy files present and non-empty")

        # -- 2. shapes and dtypes. Recorded, but deliberately NOT fatal to the
        #       polarity check below: a copy with an unexpected split size is
        #       exactly the copy whose polarity most needs measuring, and an
        #       early return here would skip the one check that matters.
        for size in _SIZES:
            for split in MHSMA_SPLITS:
                report.checks.append(
                    check_npy_header(
                        root / f"x_{size}_{split}.npy",
                        expected_shape=(PUBLISHED_SPLIT_SIZES[split], size, size),
                        expected_dtype="uint8",
                        name=f"shape:x_{size}_{split}",
                    )
                )
        for aspect in _FILE_ASPECTS:
            for split in MHSMA_SPLITS:
                report.checks.append(
                    check_npy_header(
                        root / f"y_{aspect}_{split}.npy",
                        expected_shape=(PUBLISHED_SPLIT_SIZES[split],),
                        expected_dtype="uint8",
                        name=f"shape:y_{aspect}_{split}",
                    )
                )

        try:
            for split in MHSMA_SPLITS:
                self.images(split, 128)
                self.images(split, 64)
                for aspect in _FILE_ASPECTS:
                    self.labels(split, aspect)
        except (ValueError, OSError) as exc:
            report.add(
                "structure:loadable",
                CheckStatus.FAIL,
                f"an MHSMA array is present but cannot be loaded ({exc!r}); content and "
                "polarity cannot be checked",
            )
            report.checks.append(mhsma_split_leakage_note())
            return report

        # -- 2. content: label range, agreement between sizes ----------------
        for split in MHSMA_SPLITS:
            n_128 = int(self.images(split, 128).shape[0])
            n_64 = int(self.images(split, 64).shape[0])
            if n_128 != n_64:
                report.add(
                    f"content:size_agreement:{split}",
                    CheckStatus.FAIL,
                    f"{split}: 128px array has {n_128} images but 64px array has {n_64}; "
                    "the two resolutions must be the same images",
                )
            else:
                report.add(
                    f"content:size_agreement:{split}",
                    CheckStatus.PASS,
                    f"{split}: 64px and 128px arrays agree at {n_128} images",
                    n_images=n_128,
                )
            for aspect in _FILE_ASPECTS:
                labels = np.asarray(self.labels(split, aspect))
                report.checks.append(
                    check_label_range(
                        labels,
                        allowed=(LABEL_NORMAL, LABEL_ABNORMAL),
                        name=f"labels:{aspect}:{split}",
                    )
                )
                if labels.size != n_128:
                    report.add(
                        f"labels:length:{aspect}:{split}",
                        CheckStatus.FAIL,
                        f"{split}/{aspect}: {labels.size} labels for {n_128} images",
                    )

            images = np.asarray(self.images(split, 128))
            spread = int(images.max()) - int(images.min())
            if spread == 0:
                report.add(
                    f"content:image_variation:{split}",
                    CheckStatus.FAIL,
                    f"{split}: every pixel of every 128px image has the same value "
                    f"({int(images.min())}); this copy is not the dataset",
                )
            else:
                report.add(
                    f"content:image_variation:{split}",
                    CheckStatus.PASS,
                    f"{split}: 128px intensity range {int(images.min())}-{int(images.max())}",
                )

        # -- 3. the check that matters: polarity ----------------------------
        if check_prevalence:
            self._check_polarity(report, tolerance_pct)
        else:
            report.add(
                "polarity",
                CheckStatus.UNVERIFIABLE,
                "prevalence checking was disabled by the caller (check_prevalence=False), "
                "so label polarity was NOT verified for this copy. 0=normal/1=abnormal is "
                "assumed, not measured.",
            )

        # -- 4. the risk that cannot be measured ----------------------------
        report.checks.append(mhsma_split_leakage_note())
        return report

    def _check_polarity(self, report: ValidationReport, tolerance_pct: float) -> None:
        """Measure abnormal prevalence and compare with the published table.

        Three outcomes per (split, aspect):

        ``PASS``
            Within ``tolerance_pct`` of the published figure.
        *inverted*
            Within tolerance of ``100 - published``, or on the opposite side of
            the 40/60 band from the published value. Collected and raised.
        ``FAIL``
            Neither -- the copy is some other data, or has been resampled.
        """
        inverted: list[str] = []
        measured: dict[str, dict[str, float]] = {}

        for split in MHSMA_SPLITS:
            observed = self.prevalence_percent(split)
            measured[split] = observed
            for aspect in MORPHOLOGY_ASPECTS:
                published = PUBLISHED_ABNORMAL_PREVALENCE_PCT[split][aspect]
                value = observed[aspect]
                name = f"polarity:{aspect}:{split}"
                details = {
                    "measured_abnormal_pct": round(value, 4),
                    "published_abnormal_pct": published,
                    "tolerance_pct": tolerance_pct,
                }
                if abs(value - published) <= tolerance_pct:
                    report.add(
                        name,
                        CheckStatus.PASS,
                        f"{split}/{aspect}: {value:.2f}% abnormal (published {published:.2f}%)",
                        **details,
                    )
                    continue
                if self._looks_inverted(value, published, tolerance_pct):
                    inverted.append(
                        f"{split}/{aspect}: measured {value:.2f}% abnormal, published "
                        f"{published:.2f}% (complement {100.0 - published:.2f}%)"
                    )
                    report.add(
                        name,
                        CheckStatus.FAIL,
                        f"{split}/{aspect}: LABELS LOOK INVERTED -- {value:.2f}% abnormal "
                        f"where {published:.2f}% is published",
                        **details,
                    )
                    continue
                report.add(
                    name,
                    CheckStatus.FAIL,
                    f"{split}/{aspect}: {value:.2f}% abnormal, published {published:.2f}% "
                    f"(tolerance {tolerance_pct} points). Not an inversion either -- this "
                    "copy is not the official MHSMA release.",
                    **details,
                )

        report.context["measured_abnormal_pct"] = measured
        report.context["published_abnormal_pct"] = PUBLISHED_ABNORMAL_PREVALENCE_PCT

        if inverted:
            aspects = sorted({entry.split("/")[1].split(":")[0] for entry in inverted})
            raise DatasetValidationError(
                "MHSMA label polarity is INVERTED for aspect(s): "
                f"{', '.join(aspects)}.\n  "
                + "\n  ".join(inverted)
                + "\n\nMHSMA encodes 0 = normal and 1 = abnormal. The upstream README "
                "calls the NORMAL class 'positive', so its '% Positive' column is the "
                "percentage of normal cells; reading it as abnormality prevalence "
                "inverts every figure. Training on inverted labels produces a model "
                "that sorts FOR the defects it is supposed to reject, and every metric "
                "will still look healthy because the confusion matrix is symmetric. "
                "Do not proceed: re-download from "
                f"{self.info.url} and do not apply any 1-y transform to the labels "
                "(see sperm_sorting.morphology.polarity)."
            )

    @staticmethod
    def _looks_inverted(measured: float, published: float, tolerance_pct: float) -> bool:
        """Whether ``measured`` is the complement of ``published``.

        Two signals, either sufficient: a near-exact match against
        ``100 - published`` (a straight inversion of the same data), or the
        measured value sitting on the far side of the 40/60 band from the
        published one (an inversion of a resampled copy, where the exact match
        is gone but "most cells are abnormal" is still nonsense for a dataset
        whose worst aspect is 30% abnormal).
        """
        if abs(measured - (100.0 - published)) <= tolerance_pct:
            return True
        if published < 40.0 and measured > 60.0:
            return True
        return published > 60.0 and measured < 40.0

    # ------------------------------------------------------------- torch API

    def torch_dataset(
        self,
        split: str,
        *,
        size: int = 128,
        augment: MhsmaAugmentation | Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
        normalize: Literal["none", "unit", "zscore"] = "unit",
        seed: int = 0,
    ) -> MhsmaDataset:
        """Build a :class:`MhsmaDataset` over one split. See that class."""
        return MhsmaDataset(
            self, split, size=size, augment=augment, normalize=normalize, seed=seed
        )

    def summary(self) -> dict[str, Any]:
        """Counts and prevalences for a run header or a dataset card."""
        return {
            "info": self.info.to_json_dict(),
            "splits": {
                split: {
                    "n_images": self.split_size(split),
                    "abnormal_prevalence_pct": {
                        k: round(v, 3) for k, v in self.prevalence_percent(split).items()
                    },
                    "class_counts": self.class_counts(split),
                }
                for split in MHSMA_SPLITS
            },
        }


# ==========================================================================
# torch Dataset
# ==========================================================================


class MhsmaDataset(_TorchDataset):
    """``(image[1, S, S], labels[4])`` pairs for the multi-aspect morphology head.

    * ``image`` is a single-channel float tensor (or ``uint8`` when
      ``normalize="none"``). One channel, not three: the source is grayscale and
      replicating it to RGB triples the first convolution's cost to convey
      nothing.
    * ``labels`` is a length-4 ``float32`` tensor in
      :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS` order --
      **head, acrosome, vacuole, tail** -- carrying the MHSMA integers
      *verbatim*, so ``1.0`` means abnormal. ``float32`` because
      ``BCEWithLogitsLoss`` wants float targets, not because anything is
      continuous.

    **There is no ``1 - y`` here, and there must never be one.** The polarity
    contract in :mod:`sperm_sorting.morphology.polarity` puts the single flip in
    the inference adapter precisely so that every dataset, sampler and loss can
    do the obvious thing and be right.

    Augmentation determinism: the RNG for item ``i`` is seeded from
    ``(seed, epoch, i)``, so a sample is augmented identically regardless of how
    many DataLoader workers are running, and a run is reproducible from the seed
    alone. Call :meth:`set_epoch` between epochs to vary the augmentation --
    without it every epoch sees identical augmented copies, which is a subtle
    way to get no augmentation at all.
    """

    def __init__(
        self,
        adapter: MhsmaAdapter,
        split: str,
        *,
        size: int = 128,
        augment: MhsmaAugmentation | Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
        normalize: Literal["none", "unit", "zscore"] = "unit",
        seed: int = 0,
    ) -> None:
        if torch is None:  # pragma: no cover - environment-dependent
            raise ImportError(
                "MhsmaDataset needs PyTorch, which is not installed "
                f"({_TORCH_IMPORT_ERROR}). Install it with "
                "`pip install 'sperm-sorting-ai[torch]'`, or use "
                "MhsmaAdapter.images()/labels() for the plain numpy arrays."
            )
        if normalize not in ("none", "unit", "zscore"):
            raise ValueError(f"unknown normalize mode {normalize!r}")

        self.adapter = adapter
        self.split = normalize_split(split)
        self.size = int(size)
        self.augment = augment
        self.normalize = normalize
        self.seed = int(seed)
        self.epoch = 0

        self._images = adapter.images(self.split, self.size)
        self._labels = adapter.label_matrix(self.split)
        if len(self._images) != len(self._labels):
            raise DatasetValidationError(
                f"MHSMA {self.split}: {len(self._images)} images but "
                f"{len(self._labels)} label rows"
            )

    # ------------------------------------------------------------------ api

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream. Call once per epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image = np.array(self._images[index], dtype=np.uint8, copy=True)
        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self.epoch, int(index)))
            image = self.augment(image, rng)
            if image.dtype != np.uint8:
                raise TypeError(
                    f"augmentation returned dtype {image.dtype}; it must return uint8 so "
                    "that normalisation stays the dataset's single responsibility"
                )

        tensor = self._to_tensor(image)
        labels = torch.from_numpy(np.asarray(self._labels[index], dtype=np.float32))
        return tensor, labels

    @property
    def aspects(self) -> tuple[str, ...]:
        """Label column order. Always ``MORPHOLOGY_ASPECTS``."""
        return tuple(MORPHOLOGY_ASPECTS)

    def pos_weight_tensor(self) -> Any:
        """``pos_weight`` for ``BCEWithLogitsLoss``, in label-column order.

        Infinite weights (an aspect with no abnormal example in this split) are
        replaced by the split size, which is the weight the aspect would have at
        exactly one positive -- finite, obviously large, and clearly wrong to
        anyone reading the log, which beats an ``inf`` that turns the whole loss
        into ``nan`` on the first batch.
        """
        weights = self.adapter.pos_weight(self.split)
        n = float(len(self))
        values = [w if np.isfinite(w) else n for w in (weights[a] for a in MORPHOLOGY_ASPECTS)]
        return torch.tensor(values, dtype=torch.float32)

    # -------------------------------------------------------------- internal

    def _to_tensor(self, image: np.ndarray) -> Any:
        if self.normalize == "none":
            return torch.from_numpy(image).unsqueeze(0)
        array = image.astype(np.float32)
        if self.normalize == "unit":
            array /= 255.0
        else:  # zscore, per crop
            std = float(array.std())
            # A constant crop has no information; returning zeros is honest and
            # deterministic, where dividing by ~0 would emit inf into the model.
            array = (array - float(array.mean())) / std if std > 1e-6 else np.zeros_like(array)
        return torch.from_numpy(array).unsqueeze(0)
