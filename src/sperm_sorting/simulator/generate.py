"""Dataset builder: turn the simulator into an on-disk training set.

Why this exists
---------------
The morphology model needs training data before the device exists, and no
public dataset carries morphology *and* motility for the same cell. This
module writes the bootstrap set: for every sample it draws one ground-truth
:class:`~.params.HealthState`, renders it, simulates its trajectory, and stores
the image, the CASA feature vector and all four label sets together. Because
they come from one state, the image and the trajectory describe the same
virtual sperm -- which is the whole point.

Weights trained on this data are ``WEIGHTS_PROVENANCE_SYNTHETIC`` and must
never be presented as device-validated.

Layout
------
``<out>/<split>/`` contains::

    images.npy      uint8   [N, S, S]
    feats.npy       float32 [N, 8]      CASA features, normalised
    y_overall.npy   uint8   [N]         0 healthy / 1 unhealthy
    y_aspects.npy   uint8   [N, 4]      head, acrosome, vacuole, tail
    y_motility.npy  uint8   [N]         0 progressive / 1 non-prog / 2 immotile
    meta.json                           everything needed to reproduce it

``meta.json`` deliberately carries **no wall-clock timestamp**: the same
arguments must produce byte-identical output, and a build time would break
that for no benefit. Provenance lives in the seed and the generator version.

CLI::

    python -m sperm_sorting.simulator.generate --n 20000 --out data/ --image-size 128

``argparse`` rather than the project's ``typer``, because the simulator's
dependency contract is numpy and Pillow only and this entry point must keep
working in a stripped environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..constants import (
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    MORPHOLOGY_ASPECTS,
    WEIGHTS_PROVENANCE_SYNTHETIC,
)
from ..schemas.enums import MotilityClass
from . import label as label_mod
from .label import aspect_labels, motility_label, overall_label
from .motility import (
    FEATURE_NAMES,
    features_for_state,
)
from .motility import (
    describe as motility_describe,
)
from .params import (
    RAPID_SHARE_OF_PROGRESSIVE,
    HealthState,
    Prevalences,
    sample_health_state,
    sample_motility,
)
from .render import (
    DEFAULT_SCENE_UM_PER_PX,
    SUPPORTED_SIZES,
    RenderConfig,
    render_sperm,
)

#: Bumped whenever the generative model changes in a way that makes previously
#: written datasets non-comparable. Stamped into every ``meta.json`` so a
#: checkpoint can always be traced to the data that produced it.
GENERATOR_VERSION: Final[str] = "1.0.0"

#: Default split fractions.
DEFAULT_SPLITS: Final[dict[str, float]] = {"train": 0.8, "val": 0.1, "test": 0.1}

#: Trajectory sampling used for the stored CASA features. Fixed here rather
#: than passed in, because the features are only comparable across samples if
#: every one was measured over the same duration at the same rate -- VCL and
#: LIN both depend on track length.
TRAJECTORY_POINTS: Final[int] = 96
TRAJECTORY_FPS: Final[float] = 160.0
TRAJECTORY_UM_PER_PX: Final[float] = DEFAULT_SCENE_UM_PER_PX

#: Target fraction of healthy (label 0) samples. Balanced at build time rather
#: than by a loss weight: under the default prevalences only ~16% of naturally
#: drawn cells are healthy, and a 6:1 imbalance costs far more accuracy on the
#: minority class than the small distributional distortion of balancing costs
#: overall. The achieved balance is reported and recorded either way.
TARGET_HEALTHY_FRACTION: Final[float] = 0.5


def sample_balanced_state(
    rng: np.random.Generator,
    want_healthy: bool,
    prevalences: Prevalences,
    progressive_rate: float,
) -> HealthState:
    """Draw a state conditioned on the desired overall label.

    Healthy is a conjunction (all four aspects normal *and* progressive), so it
    can be constructed directly. Unhealthy is a disjunction, so it is drawn
    from the natural distribution and rejected when it happens to come out
    healthy -- acceptance is ~84% at the defaults, and rejection keeps the
    unhealthy half's *shape* (which defect, how severe) faithful to the
    prevalences instead of imposing an artificial one.
    """
    if want_healthy:
        grade = (
            MotilityClass.RAPID_PROGRESSIVE
            if rng.random() < RAPID_SHARE_OF_PROGRESSIVE
            else MotilityClass.SLOW_PROGRESSIVE
        )
        return sample_health_state(
            rng, prevalences, progressive_rate, aspects=(0, 0, 0, 0), motility=grade
        )
    for _ in range(256):
        state = sample_health_state(rng, prevalences, progressive_rate)
        if overall_label(state) == LABEL_ABNORMAL:
            return state
    # Only reachable if every prevalence is 0 and progressive_rate is 1, i.e.
    # the caller asked for an unhealthy cell from a distribution that cannot
    # produce one. Force a defect rather than mislabel a healthy cell.
    state = sample_health_state(
        rng, prevalences, progressive_rate, motility=sample_motility(rng, 0.0)
    )
    return state


def _build_split(
    n: int,
    image_size: int,
    rng: np.random.Generator,
    prevalences: Prevalences,
    progressive_rate: float,
    render_cfg: RenderConfig,
    healthy_fraction: float,
) -> dict[str, np.ndarray]:
    """Generate one split's arrays."""
    dt_s = 1.0 / TRAJECTORY_FPS
    n_healthy = round(n * healthy_fraction)
    wants = np.zeros(n, dtype=bool)
    wants[:n_healthy] = True
    # Shuffle so that a consumer reading the file sequentially without
    # shuffling still sees a mixed stream; a sorted file silently breaks
    # anything that batches in order.
    rng.shuffle(wants)

    images = np.empty((n, image_size, image_size), dtype=np.uint8)
    feats = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    y_overall = np.empty(n, dtype=np.uint8)
    y_aspects = np.empty((n, len(MORPHOLOGY_ASPECTS)), dtype=np.uint8)
    y_motility = np.empty(n, dtype=np.uint8)

    for i in range(n):
        state = sample_balanced_state(
            rng, bool(wants[i]), prevalences, progressive_rate
        )
        images[i] = render_sperm(
            state,
            size=(image_size, image_size),
            rng=rng,
            cfg=render_cfg,
            angle=float(rng.uniform(-0.25, 0.25)),
        )
        _, feats[i] = features_for_state(
            state,
            rng,
            n_points=TRAJECTORY_POINTS,
            dt_s=dt_s,
            um_per_px=TRAJECTORY_UM_PER_PX,
        )
        y_overall[i] = overall_label(state)
        y_aspects[i] = aspect_labels(state)
        y_motility[i] = motility_label(state)

    return {
        "images": images,
        "feats": feats,
        "y_overall": y_overall,
        "y_aspects": y_aspects,
        "y_motility": y_motility,
    }


def _split_counts(n: int, splits: Mapping[str, float]) -> dict[str, int]:
    """Integer sample counts that sum exactly to ``n``.

    Largest-remainder apportionment: rounding each fraction independently
    loses or gains samples, and a dataset whose parts do not sum to the
    requested total is a bug that surfaces much later as a confusing metric.
    """
    total = float(sum(splits.values()))
    if total <= 0.0:
        raise ValueError(f"split fractions must sum to a positive value, got {total}")
    exact = {k: n * v / total for k, v in splits.items()}
    counts = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(counts.values())
    order = sorted(splits, key=lambda k: (-(exact[k] - counts[k]), k))
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def build_dataset(
    n: int,
    out_dir: str | Path,
    splits: Mapping[str, float] | None = None,
    image_size: int = 128,
    seed: int = 1234,
    prevalences: Prevalences | Mapping[str, float] | None = None,
    progressive_rate: float = 0.6,
    *,
    healthy_fraction: float = TARGET_HEALTHY_FRACTION,
    render_cfg: RenderConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build and write the dataset. Returns the ``meta.json`` contents.

    Parameters
    ----------
    n
        Total samples across all splits.
    out_dir
        Root directory; one subdirectory per split is created.
    splits
        Name to fraction; fractions are normalised. Defaults to 80/10/10.
    image_size
        Square edge; must be one of :data:`~.render.SUPPORTED_SIZES`.
    seed
        Master seed. Each split draws from its own
        :class:`numpy.random.SeedSequence` child, so the splits are
        statistically independent *and* changing one split's size cannot
        perturb another's contents.
    prevalences, progressive_rate
        Shape of the abnormal population; see :func:`sample_balanced_state`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if image_size not in SUPPORTED_SIZES:
        raise ValueError(
            f"image_size must be one of {SUPPORTED_SIZES} (MHSMA parity), got {image_size}"
        )
    if not 0.0 <= healthy_fraction <= 1.0:
        raise ValueError(f"healthy_fraction must lie in [0, 1], got {healthy_fraction}")
    splits = dict(splits) if splits else dict(DEFAULT_SPLITS)
    prev = Prevalences.coerce(prevalences)
    render_cfg = render_cfg or RenderConfig()
    root = Path(out_dir)
    counts = _split_counts(n, splits)

    # One child seed sequence per split, in sorted name order so the mapping
    # from split name to stream does not depend on dict ordering.
    names = sorted(counts)
    children = np.random.SeedSequence(seed).spawn(len(names))

    per_split: dict[str, Any] = {}
    for name, child in zip(names, children, strict=True):
        count = counts[name]
        if count < 1:
            raise ValueError(f"split '{name}' would receive 0 of {n} samples")
        if verbose:
            print(f"[generate] {name}: {count} samples ...", flush=True)
        arrays = _build_split(
            count,
            image_size,
            np.random.default_rng(child),
            prev,
            progressive_rate,
            render_cfg,
            healthy_fraction,
        )
        split_dir = root / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for key, arr in arrays.items():
            np.save(split_dir / f"{key}.npy", arr)

        overall = arrays["y_overall"]
        aspects = arrays["y_aspects"]
        motility = arrays["y_motility"]
        per_split[name] = {
            "n": int(count),
            "healthy_fraction": float(np.mean(overall == LABEL_NORMAL)),
            "aspect_abnormal_fraction": {
                a: float(np.mean(aspects[:, i] == LABEL_ABNORMAL))
                for i, a in enumerate(MORPHOLOGY_ASPECTS)
            },
            "motility_fraction": {
                label_mod.MOTILITY_LABEL_NAMES[k]: float(np.mean(motility == k))
                for k in range(len(label_mod.MOTILITY_LABEL_NAMES))
            },
            "feature_mean": [float(v) for v in arrays["feats"].mean(axis=0)],
        }

    meta: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "generator": "sperm_sorting.simulator.generate.build_dataset",
        "weights_provenance": WEIGHTS_PROVENANCE_SYNTHETIC,
        "seed": int(seed),
        "n": int(n),
        "image_size": int(image_size),
        "splits": {k: int(v) for k, v in counts.items()},
        "prevalences": prev.as_dict(),
        "progressive_rate": float(progressive_rate),
        "target_healthy_fraction": float(healthy_fraction),
        "aspect_names": list(MORPHOLOGY_ASPECTS),
        "feature_names": list(FEATURE_NAMES),
        "motility_label_names": list(label_mod.MOTILITY_LABEL_NAMES),
        "overall_label_names": list(label_mod.OVERALL_LABEL_NAMES),
        "label_convention": {"normal": LABEL_NORMAL, "abnormal": LABEL_ABNORMAL},
        "trajectory": {
            "n_points": TRAJECTORY_POINTS,
            "fps": TRAJECTORY_FPS,
            "um_per_px": TRAJECTORY_UM_PER_PX,
        },
        "casa": motility_describe(),
        "achieved": per_split,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for name in names:
        (root / name / "meta.json").write_text(
            json.dumps({**meta, "split": name}, indent=2) + "\n", encoding="utf-8"
        )

    if verbose:
        print(f"[generate] wrote {n} samples to {root}")
        for name in names:
            info = per_split[name]
            asp = ", ".join(
                f"{a}={info['aspect_abnormal_fraction'][a]:.3f}" for a in MORPHOLOGY_ASPECTS
            )
            mot = ", ".join(f"{k}={v:.3f}" for k, v in info["motility_fraction"].items())
            print(
                f"[generate]   {name:<6} n={info['n']:<7} "
                f"healthy={info['healthy_fraction']:.3f}  abnormal[{asp}]  motility[{mot}]"
            )
    return meta


def load_split(out_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Read one split back. Small, but it keeps the file names in one place."""
    d = Path(out_dir) / split
    return {
        key: np.load(d / f"{key}.npy")
        for key in ("images", "feats", "y_overall", "y_aspects", "y_motility")
    }


def _parse_prevalences(text: str | None) -> Prevalences | None:
    """Parse ``head=0.2,tail=0.4`` into a :class:`Prevalences`."""
    if not text:
        return None
    out: dict[str, float] = {}
    for item in text.split(","):
        if "=" not in item:
            raise ValueError(f"prevalence '{item}' is not of the form aspect=value")
        key, _, value = item.partition("=")
        out[key.strip()] = float(value)
    return Prevalences.coerce(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sperm_sorting.simulator.generate",
        description="Build the synthetic morphology + motility training set.",
    )
    parser.add_argument("--n", type=int, default=20000, help="total samples")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--image-size", type=int, default=128, choices=list(SUPPORTED_SIZES)
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--val-frac", type=float, default=DEFAULT_SPLITS["val"])
    parser.add_argument("--test-frac", type=float, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--progressive-rate", type=float, default=0.6)
    parser.add_argument(
        "--prevalences", type=str, default=None, help="e.g. head=0.25,tail=0.30"
    )
    parser.add_argument(
        "--healthy-fraction", type=float, default=TARGET_HEALTHY_FRACTION
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the module's assertions instead of building (default when --out is absent)",
    )
    args = parser.parse_args(argv)

    if args.self_check or args.out is None:
        _self_check()
        return 0

    build_dataset(
        n=args.n,
        out_dir=args.out,
        splits={
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
        image_size=args.image_size,
        seed=args.seed,
        prevalences=_parse_prevalences(args.prevalences),
        progressive_rate=args.progressive_rate,
        healthy_fraction=args.healthy_fraction,
        verbose=not args.quiet,
    )
    return 0


def _self_check() -> None:  # pragma: no cover - runnable self-check
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sperm-sim-"))
    try:
        meta = build_dataset(
            n=200, out_dir=tmp, image_size=64, seed=11, verbose=False
        )

        # -- shapes, dtypes and cross-array consistency --------------------
        assert sum(meta["splits"].values()) == 200, meta["splits"]
        for name, count in meta["splits"].items():
            d = load_split(tmp, name)
            assert d["images"].shape == (count, 64, 64), d["images"].shape
            assert d["images"].dtype == np.uint8
            assert d["feats"].shape == (count, len(FEATURE_NAMES))
            assert d["feats"].dtype == np.float32
            assert np.isfinite(d["feats"]).all(), "features must be finite"
            assert d["y_overall"].shape == (count,) and d["y_overall"].dtype == np.uint8
            assert d["y_aspects"].shape == (count, 4) and d["y_aspects"].dtype == np.uint8
            assert d["y_motility"].shape == (count,) and d["y_motility"].dtype == np.uint8
            assert set(np.unique(d["y_overall"])) <= {0, 1}
            assert set(np.unique(d["y_aspects"])) <= {0, 1}
            assert set(np.unique(d["y_motility"])) <= {0, 1, 2}
            # The health rule must hold in the written arrays, not just in
            # label.py: healthy iff all aspects normal and motility progressive.
            healthy = d["y_overall"] == LABEL_NORMAL
            derived = (d["y_aspects"].sum(axis=1) == 0) & (d["y_motility"] == 0)
            assert np.array_equal(healthy, derived), f"{name}: labels are inconsistent"
            assert (d["images"].std(axis=(1, 2)) > 1.0).all(), "a blank image was written"

        # -- balance -------------------------------------------------------
        train = load_split(tmp, "train")
        frac = float(np.mean(train["y_overall"] == LABEL_NORMAL))
        assert abs(frac - 0.5) < 0.02, f"train healthy fraction {frac:.3f} is unbalanced"

        # -- meta.json is complete and reproducible ------------------------
        on_disk = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
        for key in (
            "generator_version", "seed", "prevalences", "feature_names",
            "aspect_names", "image_size", "splits", "casa", "achieved",
        ):
            assert key in on_disk, f"meta.json is missing '{key}'"
        assert on_disk["feature_names"] == list(FEATURE_NAMES)
        assert "timestamp" not in on_disk and "built_at" not in on_disk

        # -- determinism ---------------------------------------------------
        tmp2 = Path(tempfile.mkdtemp(prefix="sperm-sim-"))
        try:
            build_dataset(n=200, out_dir=tmp2, image_size=64, seed=11, verbose=False)
            again = load_split(tmp2, "train")
            for key in ("images", "feats", "y_overall", "y_aspects", "y_motility"):
                assert np.array_equal(train[key], again[key]), f"{key} is not deterministic"
            assert (tmp2 / "meta.json").read_text(encoding="utf-8") == (
                tmp / "meta.json"
            ).read_text(encoding="utf-8"), "meta.json must be byte-identical"

            build_dataset(n=200, out_dir=tmp2, image_size=64, seed=12, verbose=False)
            other = load_split(tmp2, "train")
            assert not np.array_equal(train["images"], other["images"])
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        # -- split apportionment ------------------------------------------
        assert _split_counts(100, {"train": 0.8, "val": 0.1, "test": 0.1}) == {
            "train": 80, "val": 10, "test": 10
        }
        assert sum(_split_counts(7, {"a": 1.0, "b": 1.0, "c": 1.0}).values()) == 7
        assert sum(_split_counts(1001, DEFAULT_SPLITS).values()) == 1001

        # -- argument validation -------------------------------------------
        for bad in (
            lambda: build_dataset(0, tmp, verbose=False),
            lambda: build_dataset(10, tmp, image_size=100, verbose=False),
            lambda: build_dataset(10, tmp, healthy_fraction=2.0, verbose=False),
            lambda: build_dataset(10, tmp, splits={"train": 0.0}, verbose=False),
        ):
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("build_dataset must validate its arguments")

        parsed = _parse_prevalences("head=0.1,tail=0.9")
        assert parsed is not None and parsed.head == 0.1
        assert _parse_prevalences(None) is None

        print("generate.py self-check OK")
        print(f"  splits: {meta['splits']}")
        for name, info in meta["achieved"].items():
            print(
                f"  {name:<6} healthy={info['healthy_fraction']:.3f}  "
                f"motility={ {k: round(v, 3) for k, v in info['motility_fraction'].items()} }"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - CLI / self-check
    sys.exit(main())
