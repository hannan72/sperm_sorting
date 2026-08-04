"""Procedural synthetic-data simulator.

This is the only data source in the project for which **per-sperm ground truth
exists**. The public datasets give either boxes and tracks
(VISEM-Tracking) or morphology crops (MHSMA), never both for the same cell.
Here a single :class:`~.params.HealthState` is sampled first and *everything
observable is derived from it*, so one sample yields a rendered image and a
trajectory that are jointly labelled for morphology and motility at no
annotation cost.

Three consumers:

1. :mod:`~.scene` is the synthetic frame source that lets the whole real-time
   pipeline be scored end to end against known truth;
2. :mod:`~.generate` writes the bootstrap training set for the morphology model
   before any device data exists;
3. the live demo uses both.

Contract
--------
* **numpy and Pillow only.** No torch, no sklearn, no OpenCV. This package must
  import in a stripped environment.
* **Deterministic given a seed.** Every stochastic path takes an explicit
  :class:`numpy.random.Generator`; the global numpy random state is never
  touched. Same seed, byte-identical output.
* **The label causes the appearance.** A morphology flag is never an
  independent annotation: it forces the corresponding continuous knob out of
  its normal band, and the renderer reads only those knobs. See
  :func:`~.params.sample_health_state`.
* **Weights trained here are** ``WEIGHTS_PROVENANCE_SYNTHETIC`` and are never
  device-validated.

Every module runs its own assertions under ``python -m``::

    python -m sperm_sorting.simulator.params
    python -m sperm_sorting.simulator.render
    python -m sperm_sorting.simulator.motility
    python -m sperm_sorting.simulator.label
    python -m sperm_sorting.simulator.scene
    python -m sperm_sorting.simulator.generate --self-check
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Submodules are imported lazily (PEP 562) for two reasons. Importing them
# eagerly here makes ``python -m sperm_sorting.simulator.<module>`` emit a
# RuntimeWarning, because runpy finds the module already in ``sys.modules``
# before it executes it -- and every module in this package is meant to be run
# that way for its self-checks. It also keeps ``import sperm_sorting.simulator``
# cheap for a caller that only wants one piece.
_EXPORTS: dict[str, str] = {
    # params
    "HealthState": "params",
    "Prevalences": "params",
    "DEFAULT_PREVALENCES": "params",
    "SIMULATED_MOTILITY_CLASSES": "params",
    "sample_health_state": "params",
    "sample_motility": "params",
    "normal_state": "params",
    "abnormal_state": "params",
    # render
    "RenderConfig": "render",
    "CellGeometry": "render",
    "CellPose": "render",
    "SUPPORTED_SIZES": "render",
    "CROP_FIELD_UM": "render",
    "DEFAULT_CROP_UM_PER_PX": "render",
    "DEFAULT_SCENE_UM_PER_PX": "render",
    "render_sperm": "render",
    "render_sperm_on_canvas": "render",
    "cell_geometry": "render",
    "cell_ink": "render",
    "composite_ink": "render",
    # motility
    "FEATURE_NAMES": "motility",
    "FEATURE_SCALES": "motility",
    "VAP_WINDOW": "motility",
    "simulate_trajectory": "motility",
    "casa_features": "motility",
    "normalize_features": "motility",
    "features_for_state": "motility",
    "vap_window_for_fps": "motility",
    # label
    "overall_label": "label",
    "aspect_labels": "label",
    "motility_label": "label",
    "morphology_label": "label",
    "truth_table": "label",
    "MOTILITY_LABEL_NAMES": "label",
    "OVERALL_LABEL_NAMES": "label",
    # scene
    "SceneConfig": "scene",
    "SceneGenerator": "scene",
    "SpermAgent": "scene",
    "DebrisAgent": "scene",
    "SPERM_CLASS_ID": "scene",
    # generate
    "build_dataset": "generate",
    "load_split": "generate",
    "GENERATOR_VERSION": "generate",
}

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from .generate import GENERATOR_VERSION, build_dataset, load_split
    from .label import (
        MOTILITY_LABEL_NAMES,
        OVERALL_LABEL_NAMES,
        aspect_labels,
        morphology_label,
        motility_label,
        overall_label,
        truth_table,
    )
    from .motility import (
        FEATURE_NAMES,
        FEATURE_SCALES,
        VAP_WINDOW,
        casa_features,
        features_for_state,
        normalize_features,
        simulate_trajectory,
        vap_window_for_fps,
    )
    from .params import (
        DEFAULT_PREVALENCES,
        SIMULATED_MOTILITY_CLASSES,
        HealthState,
        Prevalences,
        abnormal_state,
        normal_state,
        sample_health_state,
        sample_motility,
    )
    from .render import (
        CROP_FIELD_UM,
        DEFAULT_CROP_UM_PER_PX,
        DEFAULT_SCENE_UM_PER_PX,
        SUPPORTED_SIZES,
        CellGeometry,
        CellPose,
        RenderConfig,
        cell_geometry,
        cell_ink,
        composite_ink,
        render_sperm,
        render_sperm_on_canvas,
    )
    from .scene import (
        SPERM_CLASS_ID,
        DebrisAgent,
        SceneConfig,
        SceneGenerator,
        SpermAgent,
    )


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CROP_FIELD_UM",
    "DEFAULT_CROP_UM_PER_PX",
    "DEFAULT_PREVALENCES",
    "DEFAULT_SCENE_UM_PER_PX",
    # motility
    "FEATURE_NAMES",
    "FEATURE_SCALES",
    "GENERATOR_VERSION",
    "MOTILITY_LABEL_NAMES",
    "OVERALL_LABEL_NAMES",
    "SIMULATED_MOTILITY_CLASSES",
    "SPERM_CLASS_ID",
    "SUPPORTED_SIZES",
    "VAP_WINDOW",
    "CellGeometry",
    "CellPose",
    "DebrisAgent",
    # params
    "HealthState",
    "Prevalences",
    # render
    "RenderConfig",
    # scene
    "SceneConfig",
    "SceneGenerator",
    "SpermAgent",
    "abnormal_state",
    "aspect_labels",
    # generate
    "build_dataset",
    "casa_features",
    "cell_geometry",
    "cell_ink",
    "composite_ink",
    "features_for_state",
    "load_split",
    "morphology_label",
    "motility_label",
    "normal_state",
    "normalize_features",
    # label
    "overall_label",
    "render_sperm",
    "render_sperm_on_canvas",
    "sample_health_state",
    "sample_motility",
    "simulate_trajectory",
    "truth_table",
    "vap_window_for_fps",
]
