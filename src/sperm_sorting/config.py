"""Configuration models and loader.

Configuration is Pydantic (v2) rather than dataclasses because it is built once
per process from untrusted YAML, where per-field validation is cheap and
catching a typo at startup is worth far more than the microseconds it costs.

Three rules govern this module:

1. **No physical constant is hard-coded.** Transport delay, field rise time,
   micrometres-per-pixel and the flow vector are all ``None`` until measured.
   The models below refuse to pretend otherwise: see
   :class:`OpticalCalibration.require_calibrated`.
2. **Thresholds are versioned.** :class:`MotilityConfig.profile_version` and
   :class:`DecisionConfig.threshold` are copied into every audit record, so a
   log written last month remains interpretable after a config change.
3. **The decision rule is validated, not merely typed.**
   :class:`DecisionConfig` rejects a threshold outside ``(0, 1)`` and a minimum
   count that would contradict the shot sizing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import (
    ACCEPT_RATIO_THRESHOLD,
    DEFAULT_RAPID_PROGRESSIVE_VSL_UM_S,
    DEFAULT_SLOW_PROGRESSIVE_VSL_UM_S,
    MAXIMUM_SHOT_DURATION_S,
    MAXIMUM_TRACKABLE_SPERM,
    MINIMUM_TRACKABLE_SPERM,
    MOTILITY_PROFILE_VERSION,
    TARGET_TRACKABLE_SPERM,
)
from .errors import CalibrationError, ConfigurationError
from .schemas.enums import FlowCorrectionMode, SourceKind


class _Base(BaseModel):
    """Strict base: unknown keys are an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ==========================================================================
# Run
# ==========================================================================


class RunConfig(_Base):
    """Top-level run identity and determinism controls."""

    name: str = "default"
    mode: Literal["live", "replay", "synthetic"] = "synthetic"
    seed: int = 1234
    #: When true, every stochastic component is seeded and cudnn is made
    #: deterministic. Costs throughput; required for replay-determinism tests.
    deterministic: bool = True
    #: Wall-clock limit for a run; ``None`` runs until the source is exhausted.
    max_duration_s: float | None = None
    #: Frame limit; ``None`` means unlimited.
    max_frames: int | None = None


class BackendConfig(_Base):
    """Inference backend selection, shared by detector and morphology."""

    kind: Literal["torch", "onnxruntime", "tensorrt"] = "torch"
    device: str = "cpu"
    #: Half precision where the backend supports it.
    fp16: bool = False
    #: Threads for CPU inference; ``None`` leaves the runtime default.
    num_threads: int | None = None
    #: ONNX Runtime execution providers, most-preferred first.
    onnx_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    #: TensorRT workspace in megabytes.
    trt_workspace_mb: int = 1024


# ==========================================================================
# Acquisition
# ==========================================================================


class BaslerConfig(_Base):
    """Basler a2A1920-160umPRO settings.

    Defaults reflect the documented sensor geometry (Sony IMX392LLR-C, mono,
    global shutter, 1936 x 1216 full / 1920 x 1200 default, 3.45 um pixels).
    Exposure and gain are scene-dependent and must be set for the actual
    illumination; they are not physical constants of the device.

    Bandwidth note: 1920 x 1200 x Mono8 at 164 fps is ~378 MB/s, which is at
    the practical ceiling of USB 3.0. Mono12 at full frame rate is not
    achievable. If the link cannot sustain it, reduce the ROI height (the
    dominant term for readout) or enable :attr:`binning`.
    """

    serial_number: str | None = None
    #: Full sensor is 1936 x 1216; the runtime default crops to 1920 x 1200.
    width: int = 1920
    height: int = 1200
    offset_x: int = 8
    offset_y: int = 8
    #: pylon enum values have no space: "Mono8", not "Mono 8".
    pixel_format: Literal["Mono8", "Mono10", "Mono10p", "Mono12", "Mono12p"] = "Mono8"
    #: Microseconds. The camera's Common exposure mode accepts 19 us and up
    #: (Ultra Short mode reaches 1 us). At 0.0345 um/px a sperm swimming at
    #: 100 um/s smears 2.9 px in 1 ms but only 0.055 px in 19 us, so the
    #: exposure must stay well under a millisecond to keep motion blur
    #: sub-pixel. 200 us is a starting point that needs bright illumination;
    #: see docs/calibration.md for the full blur budget.
    exposure_time_us: float = 200.0
    gain_db: float = 0.0
    #: Symmetric binning. 2x2 gives 0.069 um/px, which is still ~1.6x above
    #: the Rayleigh-limited Nyquist requirement, while cutting the data rate
    #: fourfold and improving SNR. Off by default so the native geometry is
    #: the baseline, but it is the cheapest fix if USB bandwidth binds.
    binning: Literal[1, 2, 4] = 1
    #: Documented maximum is 164 fps "at default settings"; the model number
    #: and shop page say 160. 160 is used as the conservative design figure.
    #: ``None`` lets the camera free-run at its maximum for the current ROI.
    acquisition_frame_rate: float | None = 160.0
    #: pypylon grab strategy.
    #:
    #: ``OneByOne`` delivers every frame in arrival order and is the only
    #: correct choice here: tracking reconstructs a trajectory from
    #: *consecutive* observations, and ``LatestImageOnly`` silently discards
    #: frames under load, which fragments tracks and corrupts every velocity
    #: derived from them. A dropped frame must be visible to the pipeline as a
    #: reported drop, not hidden inside the driver. ``LatestImageOnly`` is
    #: retained only for a live-display path that does no measurement.
    grab_strategy: Literal["OneByOne", "LatestImageOnly"] = "OneByOne"
    #: Number of driver-side buffers. Deeper queues absorb host scheduling
    #: jitter at 160 FPS; each 1920x1200 Mono8 buffer is ~2.3 MB.
    max_num_buffer: int = 32
    #: Enable chunk data so the camera's own timestamp travels with the frame.
    enable_chunk_timestamp: bool = True
    #: Chunk parameter carrying the timestamp. On ace 2 Basler recommends
    #: ``BslChunkTimestampValue``; ``ChunkTimestamp`` is the legacy name and
    #: is still present. The acquisition layer probes for both.
    chunk_timestamp_parameter: Literal["BslChunkTimestampValue", "ChunkTimestamp"] = (
        "BslChunkTimestampValue"
    )
    #: Which event the chunk timestamp refers to.
    chunk_timestamp_selector: Literal[
        "FrameStart", "ExposureStart", "ExposureEnd"
    ] = "ExposureStart"
    #: Camera tick frequency in Hz. ace 2 runs a 1 GHz device clock, so one
    #: tick is 1 ns. ``None`` triggers auto-detection at open time and falls
    #: back to this value.
    timestamp_tick_frequency_hz: float | None = 1e9
    #: Milliseconds to wait for a frame before declaring a timeout.
    grab_timeout_ms: int = 1000


class VideoSourceConfig(_Base):
    """Replay from a pre-recorded file."""

    path: Path | None = None
    #: Replay as fast as possible (``False``) or at the recorded rate (``True``).
    realtime: bool = False
    loop: bool = False
    start_frame: int = 0
    #: Fallback FPS when the container has no usable rate. Timestamps derived
    #: this way are marked ``CONTAINER_PTS`` so motion code knows they are
    #: nominal rather than hardware-measured.
    fallback_fps: float = 160.0


class SyntheticSourceConfig(_Base):
    """Procedural generator used as the software-test frame source.

    This is not a toy: it is the only source for which per-sperm ground truth
    exists, so it is what the end-to-end accuracy tests measure against.

    The defaults describe a *physically coherent* operating point for the
    reference build rather than round numbers. Working through it:

    * sampling 0.0345 um/px puts the 1920 x 1200 field at 66.2 x 41.4 um;
    * a shot needs 25 sperm within 1 s, so if each sperm dwells ``r`` seconds
      the field must hold ``25 * r`` of them at any instant;
    * ``r = 0.2 s`` gives 32 frames of evidence per sperm at 160 FPS -- ample
      for a velocity estimate -- and needs ~5 simultaneously visible;
    * a 0.2 s dwell across a 66.2 um field means a bulk flow of ~331 um/s,
      which is ``331 / 0.0345 = 9600`` px/s;
    * 5 sperm in 66.2 x 41.4 x 20 um of fluid is ~91 million/mL, a plausible
      concentration for a good sample (the WHO 6th-edition 5th centile is 16
      million/mL).

    ``scripts/check_feasibility.py`` recomputes this from any configuration.
    """

    width: int = 1920
    height: int = 1200
    fps: float = 160.0
    n_frames: int = 800
    #: Sample-plane sampling the scene is rendered at. Kept here so the
    #: simulator draws sperm at their true physical size rather than at an
    #: arbitrary one; a head is 4.1 um, so ~119 px at the reference scale.
    um_per_px: float = 0.0345
    #: Mean number of simultaneously visible sperm. See the derivation above.
    density: float = 5.0
    #: Fraction of generated sperm whose morphology is entirely normal.
    normal_morphology_rate: float = 0.55
    #: Fraction of generated sperm that are progressively motile.
    progressive_rate: float = 0.6
    #: Bulk flow in pixels/second, added to every object's motion. 9600 px/s
    #: is ~331 um/s at the reference sampling: fast enough to deliver a shot
    #: per second, slow enough to leave 32 frames of evidence per sperm.
    flow_vx_px_s: float = 9600.0
    flow_vy_px_s: float = 0.0
    #: Debris particles per frame; they move with the flow and must not be
    #: detected as sperm. This is how debris false positives get measured.
    debris_density: float = 4.0
    noise_sigma: float = 4.0
    background_level: int = 200
    #: Fraction of sperm rendered defocused, to exercise the quality gate.
    defocus_rate: float = 0.15
    seed: int = 1234


class AcquisitionConfig(_Base):
    kind: SourceKind = SourceKind.SYNTHETIC
    basler: BaslerConfig = Field(default_factory=BaslerConfig)
    video: VideoSourceConfig = Field(default_factory=VideoSourceConfig)
    synthetic: SyntheticSourceConfig = Field(default_factory=SyntheticSourceConfig)

    @model_validator(mode="after")
    def _check_source(self) -> AcquisitionConfig:
        if self.kind is SourceKind.VIDEO and self.video.path is None:
            raise ValueError("acquisition.kind=video requires acquisition.video.path")
        return self


# ==========================================================================
# Preprocessing and quality
# ==========================================================================


class PreprocessConfig(_Base):
    """Region of interest and intensity normalisation."""

    #: ``(x, y, w, h)`` in sensor pixels; ``None`` uses the full frame.
    roi: tuple[int, int, int, int] | None = None
    #: Per-frame intensity normalisation strategy.
    normalize: Literal["none", "minmax", "zscore", "clahe"] = "none"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    #: Optional background subtraction using a rolling median of N frames.
    background_subtraction: bool = False
    background_window: int = 64
    #: Invert intensity so sperm are bright on a dark field. Brightfield
    #: microscopy gives dark objects on a bright field; some detectors train
    #: better on the inverted convention.
    invert: bool = False


class QualityGateConfig(_Base):
    """Whole-frame acceptance thresholds.

    Values are in the units produced by :mod:`sperm_sorting.preprocessing`;
    they are illumination-dependent and should be re-tuned on device data.
    """

    enabled: bool = True
    #: Variance of the Laplacian below which a frame is considered defocused.
    min_focus_score: float = 8.0
    #: Below this, a frame is degraded rather than rejected outright.
    degraded_focus_score: float = 20.0
    min_mean_intensity: float = 0.05
    max_mean_intensity: float = 0.95
    min_contrast: float = 0.010
    max_saturated_fraction: float = 0.05
    max_underexposed_fraction: float = 0.30
    #: A degraded frame still feeds tracking (continuity matters) but is never
    #: eligible for a morphology crop.
    degraded_frames_feed_tracking: bool = True


# ==========================================================================
# Detection
# ==========================================================================


class TilingConfig(_Base):
    """Optional sliced inference for tiny objects on a large frame."""

    enabled: bool = False
    tile_size: int = 640
    overlap: int = 96
    #: Abort tiling if the measured per-frame latency exceeds this budget.
    max_latency_ms: float = 8.0


class DetectionConfig(_Base):
    """Detector selection and post-processing.

    **The detection target is the sperm head, not the whole cell.** At the
    reference sampling of 0.0345 um/px the field of view is 66.2 x 41.4 um,
    while a whole spermatozoon is 50-60 um long: it does not fit across the
    frame and only fits along it when favourably oriented. A head, at
    4.1 x 2.8 um, is ~119 x 81 px and always fits.

    This is not a compromise forced on us -- it is what CASA does anyway,
    since kinematics are defined on the head centroid. It also lines up with
    MHSMA, whose crops are described as head-centred with the tail not
    entirely visible. The consequence to keep in mind is that tail morphology
    is judged from a partial tail, which is one reason the tail aspect is the
    least reliable of the four (the other being its 4.6% abnormal prevalence).
    """

    architecture: Literal["todcnn", "p2net", "onnx", "oracle"] = "p2net"
    weights: Path | None = None
    backend: BackendConfig = Field(default_factory=BackendConfig)
    #: Long side the frame is resized to before inference; ``None`` keeps
    #: native resolution, which matters for objects a few pixels across.
    input_size: int | None = None
    score_threshold: float = 0.30
    nms_iou_threshold: float = 0.50
    max_detections: int = 512
    #: Boxes smaller than this many pixels on their short side are discarded
    #: as noise. At 0.0345 um/px a sperm head is ~100 px across, so this is a
    #: very permissive floor; see ``docs/calibration.md``.
    min_box_size_px: float = 3.0
    max_box_size_px: float = 4096.0
    class_names: list[str] = Field(default_factory=lambda: ["sperm"])
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    #: For the oracle detector only: probability of dropping a true object and
    #: of emitting a spurious one, so the pipeline can be tested under
    #: realistic detector noise without a trained model.
    oracle_miss_rate: float = 0.0
    oracle_false_positive_rate: float = 0.0
    oracle_box_jitter_px: float = 0.0


# ==========================================================================
# Tracking
# ==========================================================================


class TrackingConfig(_Base):
    algorithm: Literal["bytetrack", "ocsort", "botsort"] = "bytetrack"
    #: Detections at or above this score take part in the first association.
    high_score_threshold: float = 0.55
    #: Detections between ``low`` and ``high`` take part in ByteTrack's second
    #: association pass; this is what keeps dim, partly-occluded sperm alive.
    low_score_threshold: float = 0.15
    #: IoU below which an association is refused.
    match_iou_threshold: float = 0.20
    #: Second-pass IoU, deliberately looser.
    second_match_iou_threshold: float = 0.35
    #: Frames a confirmed track may go unmatched before removal. At 160 FPS,
    #: 30 frames is ~190 ms of occlusion tolerance.
    max_age: int = 30
    #: Matches required before a tentative track is confirmed.
    min_hits: int = 3
    #: OC-SORT observation-centric re-update window.
    ocsort_delta_t: int = 3
    ocsort_use_byte: bool = True
    #: BoT-SORT camera-motion compensation. Off by default: the camera is
    #: rigidly mounted, so global motion is fluid flow, which must be handled
    #: by flow correction rather than absorbed into the tracker.
    botsort_use_cmc: bool = False
    botsort_use_reid: bool = False
    #: Kalman process/measurement noise scales.
    kalman_std_position: float = 0.05
    kalman_std_velocity: float = 0.008
    #: Track IDs are never reused within a session. Guarded by a test.
    reuse_track_ids: Literal[False] = False


class TrackQualityConfig(_Base):
    """Bar a track must clear to be counted at all.

    A track failing this is excluded from the shot entirely -- numerator *and*
    denominator -- because it is not a trustworthy observation of one sperm.
    """

    #: Minimum measured (not predicted) observations.
    min_observed_points: int = 6
    #: Minimum observed lifetime in seconds.
    min_duration_s: float = 0.05
    #: Maximum fraction of the track's points that may be interpolated.
    max_interpolated_fraction: float = 0.5
    #: Minimum mean detector score across observed points.
    min_mean_score: float = 0.25


# ==========================================================================
# Motion
# ==========================================================================


class FlowCorrectionConfig(_Base):
    mode: FlowCorrectionMode = FlowCorrectionMode.ROBUST_ESTIMATE
    #: Fixed-vector mode: calibrated bulk flow in pixels/second. ``None``
    #: until measured; selecting ``FIXED_VECTOR`` without these is an error.
    fixed_vx_px_s: float | None = None
    fixed_vy_px_s: float | None = None
    #: Flow-map mode: path to a saved ``(H, W, 2)`` velocity field.
    flow_map_path: Path | None = None
    #: Robust mode: fraction of slowest tracks assumed to be passively
    #: transported and used to estimate the bulk vector.
    robust_quantile: float = 0.25
    #: Minimum number of tracks required for a robust estimate; below this the
    #: estimator reports unavailable rather than guessing.
    robust_min_tracks: int = 8
    #: Exponential smoothing factor for the running flow estimate.
    robust_smoothing: float = 0.15

    @model_validator(mode="after")
    def _check_mode(self) -> FlowCorrectionConfig:
        if self.mode is FlowCorrectionMode.FIXED_VECTOR and (
            self.fixed_vx_px_s is None or self.fixed_vy_px_s is None
        ):
            raise ValueError(
                "flow_correction.mode=fixed_vector requires both fixed_vx_px_s "
                "and fixed_vy_px_s to be calibrated (they are None)"
            )
        if self.mode is FlowCorrectionMode.FLOW_MAP and self.flow_map_path is None:
            raise ValueError(
                "flow_correction.mode=flow_map requires flow_map_path"
            )
        return self


class MotilityThresholds(_Base):
    """Velocity cut-points, versioned so old logs stay interpretable.

    These are not invented numbers. The WHO laboratory manual, 6th edition
    (2021), section 2.4.6.1, reinstated a four-category motility grading with
    explicit approximate velocity limits -- rapidly progressive at or above
    25 um/s, slowly progressive from 5 to under 25 um/s, non-progressive below
    5 um/s, and immotile with no active tail movement. The four members of
    :class:`MotilityClass` are exactly those categories, and the two defaults
    below are exactly those limits. (The 5th edition had collapsed grading to
    PR/NP/IM; the 6th brought the four-way split back because the presence of
    rapidly progressive sperm is clinically informative. WHO's PR is a+b, NP
    is c, IM is d.)

    Two caveats that the manual is explicit about and that this code cannot
    fix for the user:

    * The velocity limits are *approximate* and are stated alongside two other
      formulations -- half a tail length per second, and one head length per
      second -- that agree with them only roughly. WHO's own editorial board
      notes that a human grader is not expected to measure velocity at all,
      and that per-cell velocity is "only possible by CASA".
    * **Velocity is temperature-dependent and WHO requires 37 C.** Applying
      these thresholds to a sample at uncontrolled room temperature does not
      produce a WHO-comparable grading. See :attr:`sample_temperature_c`.
    """

    profile_version: str = MOTILITY_PROFILE_VERSION
    rapid_progressive_vsl_um_s: float = DEFAULT_RAPID_PROGRESSIVE_VSL_UM_S
    slow_progressive_vsl_um_s: float = DEFAULT_SLOW_PROGRESSIVE_VSL_UM_S
    #: A track below the slow-progressive cut is non-progressive if it still
    #: moves more than this (local movement), otherwise immotile.
    immotile_vcl_um_s: float = 2.0
    #: Additional linearity requirement for progressive grades. WHO's own
    #: definition admits motion "either linearly or in a large circle", so a
    #: LIN floor is *stricter* than WHO and is this implementation's choice,
    #: not a WHO criterion. It exists to exclude cells that are fast but
    #: circling tightly. Set to 0.0 to follow WHO's wording literally.
    min_lin_for_progressive: float = 0.35
    #: Temperature the sample was actually held at. WHO section 2.4.6 requires
    #: 37 C because velocity is temperature-dependent; a run at any other
    #: temperature is flagged in the audit log as non-WHO-comparable.
    sample_temperature_c: float = 37.0
    #: Tolerance before the run is flagged as thermally out of specification.
    temperature_tolerance_c: float = 0.5

    @property
    def temperature_in_spec(self) -> bool:
        return abs(self.sample_temperature_c - 37.0) <= self.temperature_tolerance_c

    @model_validator(mode="after")
    def _ordered(self) -> MotilityThresholds:
        if self.rapid_progressive_vsl_um_s <= self.slow_progressive_vsl_um_s:
            raise ValueError(
                "rapid_progressive_vsl_um_s must exceed slow_progressive_vsl_um_s"
            )
        if self.slow_progressive_vsl_um_s <= 0:
            raise ValueError("slow_progressive_vsl_um_s must be positive")
        return self


class MotionConfig(_Base):
    flow_correction: FlowCorrectionConfig = Field(default_factory=FlowCorrectionConfig)
    thresholds: MotilityThresholds = Field(default_factory=MotilityThresholds)
    #: Points required before kinematics are computed at all.
    min_points_for_kinematics: int = 5

    #: Duration of the moving-average window that defines the average path,
    #: and therefore VAP and everything derived from it (STR, WOB, ALH, BCF).
    #:
    #: Specified in *milliseconds*, not frames, deliberately. Older CASA
    #: systems used a fixed five-point smoother, and Mortimer, van der Horst &
    #: Mortimer (Asian J Androl 2015;17:545-53) show that a fixed point count
    #: "will provide inadequate smoothing and hence widely aberrant ALH
    #: values" once the frame rate changes: five frames is 100 ms at 50 FPS
    #: but only 31 ms at 160 FPS, so the same nominal setting smooths over
    #: three times less of the trajectory. Fixing the window in time keeps the
    #: average path comparable across acquisition rates.
    vap_window_ms: float = 100.0
    #: Bounds on the derived frame count, so an extreme frame rate cannot
    #: produce a degenerate or absurdly long window.
    vap_window_min_frames: int = 3
    vap_window_max_frames: int = 51

    #: Savitzky-Golay smoothing for the displayed/analysed trajectory.
    smoothing: Literal["none", "moving_average", "savgol"] = "moving_average"
    savgol_polyorder: int = 2

    #: Minimum sampling rate below which ALH and BCF are refused.
    #:
    #: Note the attribution: the WHO 6th-edition manual specifies *no* numeric
    #: minimum frame rate -- it says only that acquisition parameters affect
    #: comparability, and that one second of observation suffices for basic
    #: measurements. The ~50-60 Hz figure comes from the CASA methodology
    #: literature; Mortimer et al. (2015) call 60 images/s "really the minimum
    #: imaging frequency required for reliable human sperm track analysis".
    #: Below it, VCL is under-estimated and LIN and ALH are over-estimated, so
    #: reporting them anyway would bias the progressive classification toward
    #: false positives. They are reported unavailable instead.
    min_fps_for_alh_bcf: float = 50.0
    #: Minimum number of points before ALH/BCF are attempted.
    min_points_for_alh_bcf: int = 15
    #: Velocity below which motion is indistinguishable from tracker jitter.
    #: Estimated from the tracker's positional noise; see calibration docs.
    noise_floor_px_s: float = 1.5

    def vap_window_frames(self, fps: float) -> int:
        """Moving-average window in frames for a measured frame rate.

        Derived from :attr:`vap_window_ms` and clamped to the configured
        bounds. Always odd, so the window is centred on its sample.
        """
        if fps <= 0:
            return self.vap_window_min_frames
        frames = int(round(self.vap_window_ms * 1e-3 * fps))
        frames = max(self.vap_window_min_frames, min(frames, self.vap_window_max_frames))
        if frames % 2 == 0:
            frames += 1
        return frames


# ==========================================================================
# Best frame, cropping, morphology
# ==========================================================================


class BestFrameConfig(_Base):
    """Weights for the composite crop-quality score.

    Detector confidence is deliberately only one term among many and is
    deliberately not dominant: a confidently-detected but motion-blurred sperm
    is a bad morphology input.
    """

    top_k_frames: int = 1
    w_focus: float = 0.25
    w_motion_blur: float = 0.20
    w_local_contrast: float = 0.10
    w_exposure: float = 0.10
    w_overlap: float = 0.15
    w_truncation: float = 0.10
    w_detector_score: float = 0.05
    w_track_confidence: float = 0.05
    #: Frames whose composite score is below this are never used.
    min_quality_score: float = 0.20
    #: Reject a crop whose IoU with a neighbouring detection exceeds this.
    max_overlap_iou: float = 0.35
    #: Reject a crop with less than this fraction of its padded box visible.
    min_visible_fraction: float = 0.85
    #: Never consider frames the quality gate marked degraded.
    require_frame_quality_pass: bool = True

    @model_validator(mode="after")
    def _weights_sum(self) -> BestFrameConfig:
        total = (
            self.w_focus
            + self.w_motion_blur
            + self.w_local_contrast
            + self.w_exposure
            + self.w_overlap
            + self.w_truncation
            + self.w_detector_score
            + self.w_track_confidence
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"best_frame weights must sum to 1.0, got {total:.6f}"
            )
        if self.w_detector_score >= 0.5:
            raise ValueError(
                "detector score may not dominate best-frame selection "
                "(w_detector_score must be < 0.5)"
            )
        return self


class CropConfig(_Base):
    #: Padding as a fraction of the box's larger side.
    padding_fraction: float = 0.35
    #: Absolute minimum padding in pixels.
    min_padding_px: float = 4.0
    #: Output crop size fed to the morphology model, ``(h, w)``.
    output_size: tuple[int, int] = (128, 128)
    #: Keep the source aspect ratio and letterbox rather than squashing.
    preserve_aspect_ratio: bool = True
    letterbox_value: int = 0
    #: Per-crop intensity normalisation.
    normalize: Literal["none", "minmax", "zscore"] = "minmax"
    #: Fraction of the crop's border that must be free of the object for the
    #: tail to be judged complete.
    tail_margin_fraction: float = 0.06


class MorphologyConfig(_Base):
    backbone: Literal["mobilenetv3_small", "efficientnet_b0", "simplecnn"] = (
        "mobilenetv3_small"
    )
    weights: Path | None = None
    backend: BackendConfig = Field(default_factory=BackendConfig)
    input_size: tuple[int, int] = (128, 128)
    #: Per-aspect decision thresholds on ``P(normal)``, calibrated
    #: independently. Order follows ``constants.MORPHOLOGY_ASPECTS``.
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "head": 0.5,
            "acrosome": 0.5,
            "vacuole": 0.5,
            "tail": 0.5,
        }
    )
    #: Per-aspect temperature for probability calibration; 1.0 = uncalibrated.
    temperatures: dict[str, float] = Field(
        default_factory=lambda: {
            "head": 1.0,
            "acrosome": 1.0,
            "vacuole": 1.0,
            "tail": 1.0,
        }
    )
    batch_size: int = 16
    #: Milliseconds a track's morphology may take after its shot closes.
    #: Exceeding it marks the track ``DEADLINE_MISSED`` -- kept in the
    #: denominator, excluded from the numerator.
    deadline_ms: float = 250.0
    model_id: str = "unset"
    weights_provenance: str = "unset"

    @model_validator(mode="after")
    def _aspects_present(self) -> MorphologyConfig:
        from .constants import MORPHOLOGY_ASPECTS

        for name in MORPHOLOGY_ASPECTS:
            if name not in self.thresholds:
                raise ValueError(f"morphology.thresholds missing aspect '{name}'")
            if not 0.0 < self.thresholds[name] < 1.0:
                raise ValueError(
                    f"morphology.thresholds['{name}'] must lie in (0, 1)"
                )
            if name not in self.temperatures:
                raise ValueError(f"morphology.temperatures missing aspect '{name}'")
            if self.temperatures[name] <= 0.0:
                raise ValueError(
                    f"morphology.temperatures['{name}'] must be positive"
                )
        return self


# ==========================================================================
# Shots and decision
# ==========================================================================


class CountingGateConfig(_Base):
    """Virtual line a track must cross to be assigned to a shot.

    Placed near the downstream end of the ROI so that a track has been
    observed for as long as possible before it is committed to a shot.
    """

    #: Which axis the flow runs along.
    axis: Literal["x", "y"] = "x"
    #: Gate position as a fraction of the ROI extent along ``axis``.
    position_fraction: float = 0.85
    #: Required sign of travel: +1 means increasing coordinate.
    direction: Literal[-1, 1] = 1
    #: Crossings are only counted when the track's displacement along the flow
    #: axis over its lifetime exceeds this, to reject jitter across the line.
    min_axis_displacement_px: float = 5.0


class ShotConfig(_Base):
    target_trackable_sperm: int = TARGET_TRACKABLE_SPERM
    minimum_trackable_sperm: int = MINIMUM_TRACKABLE_SPERM
    maximum_trackable_sperm: int = MAXIMUM_TRACKABLE_SPERM
    maximum_shot_duration_seconds: float = MAXIMUM_SHOT_DURATION_S
    gate: CountingGateConfig = Field(default_factory=CountingGateConfig)
    #: Flush a partially-filled shot on shutdown rather than discarding it.
    flush_on_shutdown: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> ShotConfig:
        if not (
            self.minimum_trackable_sperm
            <= self.target_trackable_sperm
            <= self.maximum_trackable_sperm
        ):
            raise ValueError(
                "shot counts must satisfy minimum <= target <= maximum, got "
                f"{self.minimum_trackable_sperm} <= "
                f"{self.target_trackable_sperm} <= "
                f"{self.maximum_trackable_sperm}"
            )
        if self.maximum_shot_duration_seconds <= 0:
            raise ValueError("maximum_shot_duration_seconds must be positive")
        return self


class DecisionConfig(_Base):
    """The 60% rule.

    ``threshold`` is compared with a strict ``>``: a ratio of exactly the
    threshold is a REJECT, which drives FIELD_ON. This is the product
    definition, not an off-by-one; see ``docs/pipeline.md``.
    """

    threshold: float = ACCEPT_RATIO_THRESHOLD
    minimum_trackable_sperm: int = MINIMUM_TRACKABLE_SPERM

    @model_validator(mode="after")
    def _sane(self) -> DecisionConfig:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"decision.threshold must lie in (0, 1), got {self.threshold}")
        if self.minimum_trackable_sperm < 1:
            raise ValueError("decision.minimum_trackable_sperm must be >= 1")
        return self


# ==========================================================================
# Scheduling and actuation
# ==========================================================================


class SchedulingConfig(_Base):
    """Timing between the imaging region and the magnetic region.

    Every value here is a property of the physical kit and **must be measured**
    with ``scripts/calibrate_transport_delay.py``. The defaults are explicitly
    marked uncalibrated: :attr:`calibrated` gates whether the pipeline is
    allowed to actuate.
    """

    calibrated: bool = False
    calibration_id: str = ""
    #: Time for a fluid element to travel imaging region -> magnetic region.
    transport_delay_ms: float = 0.0
    #: Measured spread of the transport delay; widens the activation window.
    transport_delay_std_ms: float = 0.0
    #: Time for the field to reach its commanded on-state.
    field_rise_time_ms: float = 0.0
    #: Time for the field to collapse.
    field_fall_time_ms: float = 0.0
    #: How long a FIELD_ON pulse is held; ``None`` holds until superseded.
    pulse_duration_ms: float | None = None
    #: Extra lead time before the nominal activation instant.
    pre_activation_margin_ms: float = 0.0
    #: Extra hold time after the nominal deactivation instant.
    post_activation_margin_ms: float = 0.0
    #: Watchdog must be fed within this or the field is forced off.
    watchdog_timeout_ms: float = 500.0
    #: Dispatching later than this past the intended instant marks a command
    #: LATE and raises a metric; it does not by itself stop the run.
    late_tolerance_ms: float = 5.0
    #: Commands whose dispatch time is already this far in the past are
    #: dropped rather than fired, because firing them would gate the wrong
    #: fluid segment.
    drop_if_late_by_ms: float = 50.0
    #: Refuse to actuate when uncalibrated. Set false only for bench tests.
    require_calibration_to_actuate: bool = True

    @property
    def lead_time_s(self) -> float:
        """How far before the activation instant a command must be dispatched.

        The field needs its rise time to physically reach the on-state, and
        the pre-activation margin absorbs transport-delay jitter.
        """
        return (self.field_rise_time_ms + self.pre_activation_margin_ms) / 1000.0

    def require_calibrated(self) -> None:
        if self.require_calibration_to_actuate and not self.calibrated:
            raise CalibrationError(
                "scheduling is not calibrated: transport delay and field "
                "rise/fall times are unmeasured. Run "
                "scripts/calibrate_transport_delay.py, or set "
                "scheduling.require_calibration_to_actuate=false for a bench "
                "test with a mock actuator."
            )


class ActuationConfig(_Base):
    kind: Literal["mock", "gpio", "serial"] = "mock"
    #: GPIO backend: BCM pin driving the field driver's enable line.
    gpio_pin: int | None = None
    gpio_chip: str = "/dev/gpiochip0"
    gpio_active_high: bool = True
    #: Serial backend.
    serial_port: str | None = None
    serial_baudrate: int = 115200
    serial_timeout_s: float = 0.05
    on_command: str = "FIELD_ON\n"
    off_command: str = "FIELD_OFF\n"
    #: Require the actuator to acknowledge each command.
    require_acknowledgement: bool = True
    acknowledgement_timeout_ms: float = 20.0
    #: State the field is driven to at startup, shutdown and on any fault.
    #: FIELD_OFF is the safe default: it lets the sample flow to collection
    #: unsorted rather than diverting everything to waste.
    safe_state: Literal["FIELD_OFF"] = "FIELD_OFF"


# ==========================================================================
# Calibration
# ==========================================================================


class OpticsConfig(_Base):
    """Physical optical train, used to derive a *nominal* sampling scale.

    Nominal is not measured. The effective magnification of a real system
    departs from the catalogue figure because of the tube lens, the C-mount
    adapter and manufacturing tolerance, and a 0.5x or 0.63x coupler -- very
    common, and easy to overlook -- changes the scale by a factor of two.
    These values therefore exist only to cross-check a measurement and to warn
    when a measurement is implausible, never to substitute for one.
    """

    #: Camera pixel pitch. 3.45 um for the a2A1920-160umPRO.
    pixel_pitch_um: float = 3.45
    #: Objective magnification. 100x for the Olympus PLN 100X oil.
    objective_magnification: float = 100.0
    #: Extra magnification between objective and sensor: 1.0 for a direct
    #: C-mount, 0.5 or 0.63 for a reducing coupler.
    coupler_magnification: float = 1.0
    #: Objective numerical aperture. 1.25 for the PLN 100X oil.
    numerical_aperture: float = 1.25
    #: Illumination wavelength used for the diffraction-limit calculation.
    wavelength_nm: float = 550.0
    #: Objective field number in millimetres; 22 mm for the PLN 100X.
    field_number_mm: float = 22.0

    @property
    def nominal_um_per_px(self) -> float:
        """Sample-plane sampling implied by the optical train.

        For the reference build: 3.45 / (100 x 1.0) = 0.0345 um/px.
        """
        total_mag = self.objective_magnification * self.coupler_magnification
        if total_mag <= 0:
            raise ValueError("total magnification must be positive")
        return self.pixel_pitch_um / total_mag

    @property
    def rayleigh_limit_um(self) -> float:
        """Rayleigh resolution limit, ``0.61 * lambda / NA``.

        At 550 nm and NA 1.25 this is 0.268 um, matching the objective's
        published 0.27 um resolving power.
        """
        return 0.61 * (self.wavelength_nm * 1e-3) / self.numerical_aperture

    @property
    def abbe_limit_um(self) -> float:
        """Abbe diffraction limit, ``lambda / (2 * NA)``: 0.220 um for the
        reference build."""
        return (self.wavelength_nm * 1e-3) / (2.0 * self.numerical_aperture)

    @property
    def nyquist_oversampling(self) -> float:
        """How far above the Nyquist requirement the sampling sits.

        Values above 1.0 mean the sensor resolves finer than the optics can,
        which is the reference build's situation (~3.9x against Rayleigh).
        That headroom is real and spendable: 2x2 binning still leaves ~1.9x.
        """
        return (self.rayleigh_limit_um / 2.0) / self.nominal_um_per_px

    def field_of_view_um(self, width_px: int, height_px: int) -> tuple[float, float]:
        """Imaged area at the sample plane, in micrometres.

        For 1920 x 1200 on the reference build: 66.24 x 41.40 um. A whole
        spermatozoon is roughly 50-60 um long, so it fits across the frame
        but *not* down it -- see ``docs/assumptions.md``, which works through
        what that means for shot sizing.
        """
        scale = self.nominal_um_per_px
        return (width_px * scale, height_px * scale)


class OpticalCalibration(_Base):
    """Measured pixel-to-micrometre scale.

    :attr:`calibrated` stays false until a stage micrometer or graticule has
    actually been imaged with ``scripts/calibrate_optics.py``. Until then every
    velocity stays in pixels per second, because reporting micrometres from an
    uncalibrated system would be a fabricated physical measurement.
    """

    calibrated: bool = False
    calibration_id: str = ""
    um_per_px: float | None = None
    #: Optical train used to derive the nominal cross-check.
    optics: OpticsConfig = Field(default_factory=OpticsConfig)
    #: Estimated relative uncertainty of the measurement.
    relative_uncertainty: float | None = None
    #: Reject a measurement differing from nominal by more than this factor.
    #: A 2x discrepancy almost always means an unnoticed reducing coupler.
    max_nominal_discrepancy: float = 1.5

    @property
    def nominal_um_per_px(self) -> float:
        return self.optics.nominal_um_per_px

    @model_validator(mode="after")
    def _plausible(self) -> OpticalCalibration:
        if self.calibrated and self.um_per_px is None:
            raise ValueError(
                "optical calibration is marked calibrated but um_per_px is None"
            )
        if self.calibrated and self.um_per_px is not None:
            nominal = self.optics.nominal_um_per_px
            ratio = self.um_per_px / nominal
            if not (1.0 / self.max_nominal_discrepancy <= ratio <= self.max_nominal_discrepancy):
                raise ValueError(
                    f"measured um_per_px ({self.um_per_px:.5f}) differs from the "
                    f"nominal value implied by the optical train "
                    f"({nominal:.5f}) by {ratio:.2f}x, beyond the allowed "
                    f"{self.max_nominal_discrepancy}x. Check for a reducing "
                    "C-mount coupler, the wrong objective, or a mis-measured "
                    "graticule."
                )
        return self

    def require_calibrated(self) -> float:
        """Return ``um_per_px``, or refuse.

        Reporting micrometres per second from an uncalibrated system would be
        a fabricated physical measurement, so this raises instead.
        """
        if not self.calibrated or self.um_per_px is None:
            raise CalibrationError(
                "optical calibration is missing: cannot convert pixels to "
                "micrometres. Run scripts/calibrate_optics.py with an imaged "
                "stage micrometer, or keep velocities in pixel units."
            )
        return self.um_per_px


class CalibrationConfig(_Base):
    optical: OpticalCalibration = Field(default_factory=OpticalCalibration)
    #: Where calibration artefacts are written and read from.
    directory: Path = Path("models/calibration")


# ==========================================================================
# Runtime, monitoring
# ==========================================================================


class RuntimeConfig(_Base):
    """Queue sizing and back-pressure policy.

    Every inter-stage queue is bounded. When a queue fills, the policy decides
    whether to block the producer (correct for replay: nothing may be lost)
    or to drop the oldest frame (correct for live: a stale frame is worthless
    and blocking the camera thread causes driver-level overruns).
    """

    queue_size: int = 8
    #: ``drop_oldest`` keeps the pipeline current under load; ``block``
    #: guarantees no frame is skipped. Replay must use ``block`` for the
    #: determinism test to hold.
    backpressure: Literal["drop_oldest", "block"] = "drop_oldest"
    #: Seconds a worker waits on a queue before checking for cancellation.
    poll_interval_s: float = 0.05
    #: Seconds to allow for graceful shutdown before cancelling hard.
    shutdown_grace_s: float = 5.0
    #: Run detection/morphology in a thread pool so the event loop keeps
    #: servicing acquisition. 0 means "run inline" (deterministic, for tests).
    inference_threads: int = 0


class MonitoringConfig(_Base):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    #: JSON lines to stdout instead of human-readable text.
    json_logs: bool = False
    #: Directory for audit logs; ``None`` disables audit logging entirely,
    #: which also disables the replay-determinism guarantees.
    audit_dir: Path | None = Path("runs")
    #: Persist per-track records as well as per-shot records. Larger logs,
    #: but required to reconstruct a decision sperm by sperm.
    audit_tracks: bool = True
    #: Persist per-track point lists. Large; off by default.
    audit_track_points: bool = False
    #: Interval at which runtime metrics are emitted.
    metrics_interval_s: float = 1.0
    #: Warn when a stage's queue stays above this fraction of capacity.
    queue_high_water: float = 0.8
    #: Health monitor declares the source dead after this long with no frame.
    frame_timeout_s: float = 2.0


# ==========================================================================
# Root
# ==========================================================================


class AppConfig(_Base):
    """The whole configuration tree."""

    run: RunConfig = Field(default_factory=RunConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    track_quality: TrackQualityConfig = Field(default_factory=TrackQualityConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    best_frame: BestFrameConfig = Field(default_factory=BestFrameConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    shots: ShotConfig = Field(default_factory=ShotConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    actuation: ActuationConfig = Field(default_factory=ActuationConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def _cross_section(self) -> AppConfig:
        # The decision rule and the shot sizing must agree on the minimum, or
        # a shot could close "normally" at a size the decision engine calls
        # INDETERMINATE.
        if self.decision.minimum_trackable_sperm != self.shots.minimum_trackable_sperm:
            raise ValueError(
                "decision.minimum_trackable_sperm "
                f"({self.decision.minimum_trackable_sperm}) must equal "
                "shots.minimum_trackable_sperm "
                f"({self.shots.minimum_trackable_sperm})"
            )
        if self.crop.output_size != self.morphology.input_size:
            raise ValueError(
                f"crop.output_size {self.crop.output_size} must equal "
                f"morphology.input_size {self.morphology.input_size}"
            )
        if self.run.mode == "replay" and self.runtime.backpressure != "block":
            raise ValueError(
                "replay mode requires runtime.backpressure='block' so that no "
                "frame is dropped; otherwise replay is not deterministic"
            )
        return self

    # ------------------------------------------------------------------ util

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            _jsonable(self.model_dump(mode="json")), sort_keys=False, indent=2
        )

    def summary(self) -> dict[str, Any]:
        """Compact dict stamped into every audit log header."""
        return {
            "run_name": self.run.name,
            "mode": self.run.mode,
            "seed": self.run.seed,
            "source": str(self.acquisition.kind),
            "detector": self.detection.architecture,
            "tracker": self.tracking.algorithm,
            "morphology_backbone": self.morphology.backbone,
            "morphology_model_id": self.morphology.model_id,
            "weights_provenance": self.morphology.weights_provenance,
            "decision_threshold": self.decision.threshold,
            "minimum_trackable": self.decision.minimum_trackable_sperm,
            "motility_profile": self.motion.thresholds.profile_version,
            "flow_correction": str(self.motion.flow_correction.mode),
            "optically_calibrated": self.calibration.optical.calibrated,
            "scheduling_calibrated": self.scheduling.calibrated,
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ==========================================================================
# Loading
# ==========================================================================

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` inside string leaves."""
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings onto a nested dict.

    Used by every training and runtime entry point so that any configuration
    field can be changed from the command line without editing YAML.
    """
    out = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigurationError(
                f"override '{item}' is not of the form key.path=value"
            )
        path, _, raw = item.partition("=")
        keys = path.strip().split(".")
        node = out
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = _coerce_scalar(raw)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file, resolving an optional ``extends:`` parent.

    ``extends`` is relative to the child file, so config trees can be moved
    without rewriting their internals.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"config root must be a mapping: {path}")
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / str(parent)).resolve()
        data = deep_merge(load_yaml(parent_path), data)
    return data


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    **kwargs: Any,
) -> AppConfig:
    """Build an :class:`AppConfig` from YAML plus CLI overrides.

    Parameters
    ----------
    path
        YAML file; ``None`` yields the built-in defaults.
    overrides
        ``a.b.c=value`` strings applied after the file.
    kwargs
        Extra top-level sections merged last, for programmatic construction.
    """
    data: dict[str, Any] = load_yaml(path) if path is not None else {}
    if kwargs:
        data = deep_merge(data, kwargs)
    if overrides:
        data = apply_overrides(data, list(overrides))
    data = _expand_env(data)
    try:
        return AppConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
