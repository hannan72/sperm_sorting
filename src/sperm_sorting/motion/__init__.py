"""Motion and kinematics: CASA velocities, flow correction, motility grading.

The three responsibilities are deliberately separate files, because they fail
in different ways and are calibrated by different people:

* :mod:`.smoothing` -- the average-path algorithm. Pure geometry. Everything
  derived from the average path (VAP, STR, WOB, ALH, BCF) inherits its window
  and is therefore *not* comparable across CASA systems.
* :mod:`.flow` -- removal of bulk fluid transport. Without it a dead sperm
  carried by the flow is straight, fast, and graded rapid progressive.
* :mod:`.features` -- the kinematic record itself, from measured points and
  real timestamps only, in pixels always and in micrometres only when the
  optics have been calibrated.
* :mod:`.classifier` -- the motility grade and the track-quality bar, both of
  which refuse to answer rather than answer wrongly when a prerequisite
  (calibration, flow estimate, enough points) is missing.

Typical use::

    estimator = build_flow_estimator(cfg.motion.flow_correction)
    classifier = ProgressiveMotilityClassifier(
        cfg.motion, cfg.calibration.optical, frame.timestamp_source
    )
    # then, once per frame:
    flow = estimator.estimate(active_tracks, frame)      # may be None
    for track in finished_tracks:
        assess_track_quality(track, cfg.track_quality)
        if track.track_quality_pass:
            classifier.classify(track, flow)             # fills track.motion
"""

from __future__ import annotations

from .classifier import (
    ProgressiveMotilityClassifier,
    assess_track_quality,
    classify_motility,
)
from .features import FlowSampler, compute_motion_features, lateral_deviations
from .flow import (
    DisabledFlow,
    FixedVectorFlow,
    FlowEstimator,
    FlowMapFlow,
    RobustFlowEstimator,
    apply_flow_correction,
    build_flow_estimator,
)
from .smoothing import (
    SmoothingMethod,
    as_points_array,
    moving_average_path,
    net_displacement,
    path_length,
    savgol_path,
    smooth_path,
    step_lengths,
)

__all__ = [
    "DisabledFlow",
    "FixedVectorFlow",
    "FlowEstimator",
    "FlowMapFlow",
    "FlowSampler",
    "ProgressiveMotilityClassifier",
    "RobustFlowEstimator",
    "SmoothingMethod",
    "apply_flow_correction",
    "as_points_array",
    "assess_track_quality",
    "build_flow_estimator",
    "classify_motility",
    "compute_motion_features",
    "lateral_deviations",
    "moving_average_path",
    "net_displacement",
    "path_length",
    "savgol_path",
    "smooth_path",
    "step_lengths",
]
