# Public Python API

Everything a caller needs to build a configuration, run a pipeline, plug in their
own detector, tracker or morphology model, and read the audit log.

Package: `sperm_sorting`, importable from `src/` (`pyproject.toml` sets
`package-dir = {"" = "src"}`). Console script: `sperm-sorting`.

Architecture and rationale are in `docs/architecture.md`; processing order and
invariants are in `docs/pipeline.md`.

---

## 1. Configuration

### 1.1 Building an `AppConfig`

`AppConfig` is the whole configuration tree, a Pydantic v2 model with
`extra="forbid"` -- an unrecognised key is an error, not a silent default.

```python
from sperm_sorting.config import AppConfig, load_config

# 1. built-in defaults
cfg = AppConfig()

# 2. from YAML (resolves an optional `extends:` parent, relative to the child)
cfg = load_config("configs/default.yaml")

# 3. YAML plus dotted-path overrides, applied after the file
cfg = load_config(
    "configs/device_v1.yaml",
    overrides=["decision.threshold=0.6", "run.max_frames=2000"],
)

# 4. programmatic, merged last
cfg = load_config("configs/default.yaml", run={"name": "experiment-7", "seed": 99})
```

`load_config` also expands `${VAR}` and `${VAR:-default}` inside string leaves,
and raises `ConfigurationError` -- never a bare `ValidationError` -- on any
failure.

```python
def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    **kwargs: Any,
) -> AppConfig: ...
```

Supporting helpers: `load_yaml(path)`, `deep_merge(base, override)`,
`apply_overrides(data, ["a.b.c=value", ...])`.

### 1.2 Sections

| Attribute | Model | Covers |
|---|---|---|
| `run` | `RunConfig` | name, mode (`live`/`replay`/`synthetic`), seed, determinism, frame/duration limits |
| `acquisition` | `AcquisitionConfig` | `kind` plus `basler`, `video`, `synthetic` sub-models |
| `preprocess` | `PreprocessConfig` | ROI, normalisation, inversion, background subtraction |
| `quality_gate` | `QualityGateConfig` | focus, exposure and contrast thresholds |
| `detection` | `DetectionConfig` | architecture, weights, backend, score/NMS thresholds, tiling, oracle noise |
| `tracking` | `TrackingConfig` | algorithm, association thresholds, `max_age`, `min_hits`, Kalman scales |
| `track_quality` | `TrackQualityConfig` | the bar a track must clear to be counted at all |
| `motion` | `MotionConfig` | flow correction, motility thresholds, VAP window, ALH/BCF refusal rules |
| `best_frame` | `BestFrameConfig` | eight composite-score weights (validated to sum to 1.0) |
| `crop` | `CropConfig` | padding, output size, letterboxing, normalisation |
| `morphology` | `MorphologyConfig` | backbone, weights, per-aspect thresholds and temperatures, deadline, provenance |
| `shots` | `ShotConfig` | target/minimum/maximum counts, duration, counting gate |
| `decision` | `DecisionConfig` | threshold, minimum trackable |
| `scheduling` | `SchedulingConfig` | transport delay, field rise/fall, margins, watchdog, `calibrated` |
| `actuation` | `ActuationConfig` | mock/gpio/serial, acknowledgement, safe state |
| `calibration` | `CalibrationConfig` | `optical` (measured `um_per_px`) and the artefact directory |
| `runtime` | `RuntimeConfig` | queue size, back-pressure policy, poll interval, shutdown grace |
| `monitoring` | `MonitoringConfig` | log level, JSON logs, audit directory, metrics interval |

### 1.3 Cross-section validation

`AppConfig` refuses to build when any of these hold:

- `decision.minimum_trackable_sperm != shots.minimum_trackable_sperm`
- `crop.output_size != morphology.input_size`
- `run.mode == "replay"` and `runtime.backpressure != "block"`

Per-model rules that also raise: `best_frame` weights not summing to 1.0 (or
`w_detector_score >= 0.5`); motility thresholds out of order;
`flow_correction.mode=fixed_vector` without both components; a `morphology`
threshold outside `(0, 1)` or a missing aspect; `shots` counts not satisfying
`minimum <= target <= maximum`; a measured `um_per_px` more than
`max_nominal_discrepancy` from the nominal one.

### 1.4 Calibration accessors

```python
opt = cfg.calibration.optical

opt.nominal_um_per_px          # 0.0345 for the reference build -- derived, not measured
opt.optics.rayleigh_limit_um   # 0.2684
opt.optics.abbe_limit_um       # 0.2200
opt.optics.nyquist_oversampling  # 3.89 (against Rayleigh)
opt.optics.field_of_view_um(1920, 1200)   # (66.24, 41.40)

opt.require_calibrated()       # returns um_per_px, or raises CalibrationError

cfg.scheduling.lead_time_s     # (rise + pre-margin) / 1000
cfg.scheduling.require_calibrated()   # raises unless measured

cfg.motion.vap_window_frames(160.0)   # 17 ; at 50 Hz -> 5
cfg.motion.thresholds.temperature_in_spec   # |T - 37| <= tolerance
```

### 1.5 Serialisation

```python
cfg.to_yaml()      # round-trippable YAML
cfg.summary()      # the compact dict stamped into every audit manifest
```

`summary()` returns: `run_name`, `mode`, `seed`, `source`, `detector`, `tracker`,
`morphology_backbone`, `morphology_model_id`, `weights_provenance`,
`decision_threshold`, `minimum_trackable`, `motility_profile`,
`flow_correction`, `optically_calibrated`, `scheduling_calibrated`.

---

## 2. Running

### 2.1 The easy path: `Application`

```python
from sperm_sorting.app import Application, run_app
from sperm_sorting.config import load_config

cfg = load_config("configs/synthetic.yaml")

with Application(cfg) as app:          # __enter__ calls setup()
    app.install_signal_handlers()      # Ctrl-C becomes a graceful stop
    frames = app.run()
    print(app.format_report())
    decisions = app.decisions          # list[Decision]
    commands  = app.commands           # list[FieldCommand]
# __exit__ calls close(): field off first, then source, detector, audit

# or, in one call
app = run_app(cfg)
```

`Application.setup()` builds everything in a deliberate order -- seed and
logging, audit log, actuator driven to FIELD_OFF, models and source, scheduler
**armed last** and only if calibrated. `close()` is idempotent and never raises,
because it runs on the path taken after another exception.

### 2.2 Building a `Pipeline` directly

For tests, notebooks, or embedding.

```python
from sperm_sorting.acquisition.factory import build_frame_source
from sperm_sorting.detection.factory import build_detector
from sperm_sorting.tracking.factory import build_tracker
from sperm_sorting.morphology.factory import build_morphology_engine
from sperm_sorting.runtime.pipeline import Pipeline
from sperm_sorting.scheduling.clock import ManualClock
from sperm_sorting.scheduling.scheduler import ActuationScheduler

clock = ManualClock()
scheduler = ActuationScheduler(cfg.scheduling, clock=clock)   # dispatch=None -> no hardware

pipeline = Pipeline(
    cfg,
    detector=build_detector(cfg.detection),
    tracker=build_tracker(cfg.tracking),
    morphology=build_morphology_engine(cfg.morphology),
    scheduler=scheduler,
    clock=clock,
    audit=None,        # AuditLogger | None
    metrics=None,      # RuntimeMetrics | None -> constructed
    health=None,       # HealthMonitor  | None -> constructed
    frame_width=1920,  # only needed when preprocess.roi is None and the
    frame_height=1200, # source is not synthetic
)

source = build_frame_source(cfg.acquisition)
source.open()
try:
    for packet in source.frames(max_frames=600):
        result = pipeline.process_frame(packet)
        for decision in result.decisions:
            print(decision.status, decision.rationale)
finally:
    source.close()

final_decisions = pipeline.flush()   # close and decide everything outstanding
print(pipeline.summary())
```

`Pipeline.process_frame` is **synchronous and deterministic**: one call advances
everything by one frame. That is what makes replay reproducible and lets a test
drive a whole run in milliseconds.

### 2.3 `FrameResult`

```python
@dataclass(slots=True)
class FrameResult:
    frame_id: int
    accepted: bool                      # False if the quality gate rejected it
    n_detections: int = 0
    n_active_tracks: int = 0
    gated_track_ids: list[int] = ...    # crossed the counting gate on this frame
    finished_track_ids: list[int] = ...
    closed_shot_ids: list[int] = ...
    decisions: list[Decision] = ...
    commands: list[FieldCommand] = ...
    reject_reason: str = ""
```

### 2.4 `PipelineRunner`

```python
from sperm_sorting.runtime.workers import PipelineRunner

runner = PipelineRunner(cfg, source, pipeline, on_frame=lambda r: None)
runner.run()              # picks the topology from cfg.run.mode
runner.run_synchronous()  # one thread, deterministic
runner.run_threaded()     # acquisition decoupled by a bounded queue
runner.stop()             # graceful; safe from a signal handler
runner.queue_stats(); runner.worker_status()
```

### 2.5 The decision rule as a pure function

```python
from sperm_sorting.decision.engine import decide
from sperm_sorting.schemas.shot import exceeds_threshold

d = decide(15, 25, threshold=0.60, minimum_trackable=20, shot_id=7)
d.status         # ShotStatus.REJECT   -- exactly 60% is a REJECT
d.field_command  # FieldCommandKind.FIELD_ON   -- FIELD_ON is the rejection

exceeds_threshold(15, 25, 0.60)   # False
exceeds_threshold(16, 25, 0.60)   # True
exceeds_threshold(12, 20, 0.60)   # False
exceeds_threshold(13, 20, 0.60)   # True
```

`decide` raises `ValueError` if either count is negative or if
`ai_eligible_count > trackable_count`.

### 2.6 CLI

```bash
sperm-sorting run -c configs/synthetic.yaml -n 600
sperm-sorting run -c configs/replay.yaml --video clip.mp4      # implies replay + block
sperm-sorting run -c configs/default.yaml -s decision.threshold=0.6
sperm-sorting show-config -c configs/device_v1.yaml [--json]
sperm-sorting feasibility -c configs/default.yaml --chamber-depth-um 20
sperm-sorting doctor -c configs/device_v1.yaml
sperm-sorting version
```

`feasibility` exits non-zero when the configuration cannot deliver the configured
shot rate.

---

## 3. Plugging in your own model

Three abstract interfaces. Implement one and the runtime never knows the
difference.

### 3.1 `Detector`

```python
from sperm_sorting.detection.base import Detector
from sperm_sorting.schemas.detection import BoundingBox, Detection
from sperm_sorting.schemas.frame import FramePacket

class MyDetector(Detector):
    name = "my-detector"
    class_names = ("sperm",)

    def detect(self, frame: FramePacket) -> list[Detection]:
        ...
```

**Contract.**

| Requirement | Why |
|---|---|
| Accept a 2-D `uint8`/`uint16` array or a `FramePacket` | the source is monochrome; nothing assumes three channels |
| Return boxes in **source-frame pixel coordinates**, undoing any internal resize, padding or tiling yourself | nothing downstream knows your internal geometry |
| Be safe to call repeatedly from one thread | the analysis stage is single-threaded |
| Never raise on an empty frame -- return `[]` | an empty field is normal |

Provided for free: `detect_array(image, frame_id, capture_time_s)`,
`warmup(height, width, iterations)` (so lazy CUDA context creation and cuDNN
autotuning do not land on the first live frame), `close()`, `describe()`, and
context-manager support.

Built-ins: `p2net`, `todcnn`, `onnx`, `oracle`, via
`build_detector(cfg.detection)`.

### 3.2 `Tracker`

```python
from sperm_sorting.tracking.base import Tracker
from sperm_sorting.schemas.track import TrackRecord

class MyTracker(Tracker):
    name = "my-tracker"

    def update(self, detections: list[Detection], frame: FramePacket) -> list[TrackRecord]: ...
    def all_tracks(self) -> list[TrackRecord]: ...
    def finished_tracks(self) -> list[TrackRecord]: ...
```

**Contract -- three guarantees the rest of the product rests on.**

1. **Unique IDs.** Every ID identifies one track for the whole session and is
   **never recycled**, even after removal. `TrackingConfig.reuse_track_ids` is
   typed `Literal[False]`, so no configuration can turn it on.
2. **Idempotent records.** `update()` returns the *same* `TrackRecord` object for
   a given ID on every call, so a caller may hold a reference and watch it grow.
3. **Explicit interpolation.** A point produced by the motion model rather than a
   measurement is appended with `observed=False`, so motion analysis can exclude
   it.

`update()` returns the **currently active** tracks. `finished_tracks()` **drains**
tracks removed since the last call -- a track is only ready for final motion
analysis and morphology once it can no longer grow, so that is the hand-off point
into the rest of the pipeline.

Built-ins: `bytetrack` (default), `ocsort`, `botsort`, via
`build_tracker(cfg.tracking)`.

### 3.3 `BaseMorphologyEngine`

```python
from sperm_sorting.morphology.inference import BaseMorphologyEngine
from sperm_sorting.schemas.morphology import MorphologyResult

class MyMorphologyEngine(BaseMorphologyEngine):
    model_id = "my-model-v1"
    weights_provenance = "device-finetuned"

    def infer_batch(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        """P(normal) per aspect for each crop, in input order."""

    def evaluate_track(
        self,
        track: TrackRecord,
        crop_image: np.ndarray | None,
        deadline_s: float | None = None,
    ) -> MorphologyResult: ...
```

**Contract.**

- `infer_batch` returns **`P(normal)`**, keyed by aspect name, in the canonical
  order `("head", "acrosome", "vacuole", "tail")`.
- **Polarity.** Internally, every logit in this package is a logit for
  `P(abnormal)` -- MHSMA labels `0 = normal`, so the training target is the
  dataset label verbatim. The single flip to `p_normal` happens in the inference
  adapter, through `polarity.flip_polarity`, and nowhere else. If your model
  emits `P(abnormal)`, flip it once, in your engine, and say so in `model_id`.
- **Deadlines are honoured against `time.monotonic`**, never the wall clock.
  Exceeding one returns `MorphologyStatus.DEADLINE_MISSED`.
- **Failure is never "normal".** Every failure path must return
  `MorphologyResult.failed(...)`, which leaves all four aspects `None`, which
  makes `is_complete` false, which makes `all_four_normal` false. Never construct
  an `AspectResult` without a real model output behind it.

Provided: `evaluate_batch` (a loop by default), `warmup`, `close`, `describe`
(which reports `model_id`, `weights_provenance`, `aspects` and the polarity
convention into the audit header).

Built-ins: `MorphologyEngine` (torch or ONNX Runtime, selected by
`cfg.backend.kind`) and `RandomMorphologyEngine` (a seeded test double). Note
that there is **deliberately no path from configuration to the test double** --
`BackendConfig.kind` is a `Literal` over `torch`/`onnxruntime`/`tensorrt`, so no
YAML can name it.

### 3.4 `FrameSource`

```python
from sperm_sorting.acquisition.base import FrameSource
from sperm_sorting.schemas.enums import SourceKind

class MySource(FrameSource):
    kind = SourceKind.VIDEO

    def open(self) -> None: ...
    def close(self) -> None: ...        # idempotent; must not raise
    def read(self) -> FramePacket | None: ...   # None at end of stream
    def describe(self) -> dict[str, Any]: ...
```

**The one thing a source must be honest about is time.** Motion analysis divides
by elapsed time, so a source that invents plausible-looking timestamps produces
plausible-looking velocities that are wrong. Set `timestamp_source` truthfully,
and report drops in `dropped_before` rather than leaving them to be inferred from
a gap in frame IDs -- `frame_id` is strictly increasing and gap-free by contract.

---

## 4. Schema reference

All in `sperm_sorting.schemas`; all `slots` dataclasses; all carry
`schema_version` and `to_json_dict()`.

### 4.1 `BoundingBox` (frozen)

`x1, y1, x2, y2` in pixels, `x2`/`y2` exclusive. Properties `width`, `height`,
`area`, `center`, `cx`, `cy`. Converters `as_xyxy`, `as_xywh`, `as_cxcywh`,
`as_array`, and constructors `from_xyxy`, `from_xywh`, `from_cxcywh`. Operations
`clipped(w, h)`, `expanded(px, py)`, `iou(other)`. Raises `ValueError` on a
degenerate box.

### 4.2 `Detection`

`frame_id`, `box`, `score`, `class_id`, `class_name`, `capture_time_s`,
`track_id` (`None` until associated), `meta`.
`detections_to_array(list)` stacks to `(N, 5)` `[x1, y1, x2, y2, score]`, with a
correctly-shaped empty array for an empty input.

### 4.3 `FramePacket` / `FrameQuality`

`FramePacket`: `frame_id`, `image` (2-D), `capture_time_s` (**monotonic**, the
only timestamp motion analysis may use), `timestamp_source`, `source_kind`,
`received_time_s` (latency accounting only, never velocity), `dropped_before`,
`session_id`, `quality`, `roi`, `meta`. Properties `height`, `width`, `shape`.

`FrameQuality`: `verdict`, `focus_score` (variance of Laplacian, in
8-bit-equivalent grey levels), `mean_intensity`, `contrast`,
`saturated_fraction`, `underexposed_fraction`, `reason`.

### 4.4 `TrackPoint`, `MotionFeatures`, `CropRecord`, `TrackRecord`

`TrackPoint`: `frame_id`, `capture_time_s`, `box`, `score`, `observed`; `x`/`y`
are the box centre.

`MotionFeatures` -- provenance (`n_points`, `n_observed_points`, `duration_s`,
`mean_frame_interval_s`, `timestamp_source`, `flow_correction_mode`,
`profile_version`, `optically_calibrated`, `um_per_px`); raw px/s
(`vcl_px_s`, `vsl_px_s`, `vap_px_s`); corrected px/s (`*_corrected_px_s`);
corrected um/s (`vcl_um_s`, `vsl_um_s`, `vap_um_s` -- **`None` without
calibration**); ratios (`lin`, `str_`, `wob`); `alh_um` and `bcf_hz` with
`alh_unavailable_reason` / `bcf_unavailable_reason`; geometry
(`net_displacement_px`, `path_length_px`, `direction_rad`,
`direction_stability`); the flow actually subtracted (`flow_vx_px_s`,
`flow_vy_px_s`); and `motility_class` with `motility_reason`.

*Note the JSON key for `str_` is `"str"` -- the trailing underscore avoids the
Python builtin.*

`CropRecord`: `track_id` (**duplicated on purpose**, so the crop-to-track binding
is checkable), `frame_id`, `capture_time_s`, `source_box`, `output_size`,
`quality_score` plus a per-term `quality_terms` breakdown, `truncated`,
`visible_fraction`, `tail_complete`, `max_overlap_iou`, `detector_score`,
`track_confidence`.

`TrackRecord`: `track_id`, `state`, `points`, frame/time span, `hit_count`,
`time_since_update`, `mean_score`, `motion`, `crop`, `morphology`, `shot_id`,
gate crossing, `track_quality_pass`/`_reason`, `evaluation_complete`,
`evaluation_deadline_s`, `ai_eligible`, `ineligibility_reason`. Methods
`add_point`, `observed_points`, `n_observed`, `duration_s`, `last_box`,
`is_progressive`, `all_four_normal`, **`compute_eligibility()`**.

### 4.5 `AspectResult` / `MorphologyResult`

`AspectResult`: `name`, `p_normal`, `threshold`; derived `normal`
(`p_normal >= threshold`), `label` (0 normal / 1 abnormal), `margin`.

`MorphologyResult`: `track_id`, `status`, the four aspects, `frame_id`,
`latency_ms`, `model_id`, `weights_provenance`, `failure_reason`. Derived
`is_complete`, `aspects` (canonical order), **`all_four_normal`** (conjunctive; a
missing aspect is *not* normal), `first_abnormal_aspect()`, `probabilities()`,
`labels()`. Constructor `MorphologyResult.failed(track_id, status, reason)`.

### 4.6 `ShotRecord`

`shot_id`, `opened_at_s`/`opened_frame_id`, **`track_ids`** (the denominator, in
gate-crossing order), **`eligible_track_ids`** (the numerator), close fields,
`first_gate_time_s`/`last_gate_time_s` (which delimit the *fluid segment*),
`status`, `ai_eligible_ratio`, `threshold_applied`,
`minimum_trackable_applied`, `ineligibility_histogram`, `rejected_track_count`.
Methods `add_track` (returns `False` on a duplicate), `trackable_count`,
`ai_eligible_count`, `gate_span_s`, `note_gate_crossing`, `compute_ratio`,
`record_ineligibility`.

### 4.7 `FieldCommand`

`command_id`, `kind`, `origin`, `activate_at_s`, `dispatch_at_s`, `deadline_s`,
`duration_s`, `shot_id`, `created_at_s`, `outcome`, `dispatched_at_s`,
`acknowledged_at_s`, `timing_error_s`, `failure_reason`. Ordered by
`dispatch_at_s` then `command_id`, so the scheduler's heap is deterministic.

### 4.8 Enums

| Enum | Members |
|---|---|
| `SourceKind` | `basler`, `video`, `synthetic` |
| `TimestampSource` | `hardware`, `host_monotonic`, `container_pts`, `synthetic` |
| `QualityVerdict` | `pass`, `degraded`, `reject` |
| `TrackState` | `tentative`, `confirmed`, `lost`, `removed` |
| `MotilityClass` | `rapid_progressive`, `slow_progressive`, `non_progressive`, `immotile`, `undetermined`; property `is_progressive` |
| `FlowCorrectionMode` | `disabled`, `fixed_vector`, `flow_map`, `robust_estimate` |
| `MorphologyStatus` | `complete`, `deadline_missed`, `no_valid_crop`, `inference_failed`, `not_required` |
| `ShotCloseReason` | `target_reached`, `hard_maximum`, `timeout`, `shutdown` |
| `ShotStatus` | `accept`, `reject`, `indeterminate` |
| `FieldCommandKind` | `FIELD_ON`, `FIELD_OFF` |
| `CommandOrigin` | `decision`, `safe_default`, `watchdog`, `manual` |
| `CommandOutcome` | `pending`, `dispatched`, `acknowledged`, `late`, `superseded`, `failed` |
| `IneligibilityReason` | `none`, `track_quality_fail`, `not_progressive`, `motility_undetermined`, `abnormal_head`, `abnormal_acrosome`, `abnormal_vacuole`, `abnormal_tail`, `morphology_incomplete`, `deadline_missed` |

All derive from `str`, so they serialise to readable JSON without a custom
encoder and compare equal to their wire value.

---

## 5. Calibration API

```python
from sperm_sorting.calibration import (
    calibrate_from_known_distance, calibrate_from_graticule,
    save_calibration, load_calibration, px_s_to_um_s,
    calibrate_fixed_vector, calibrate_flow_map, save_flow_map, load_flow_map,
    estimate_from_tracer, estimate_from_geometry, estimate_field_switching,
    save_transport_calibration, load_transport_calibration,
)
```

Operator procedures, acceptance criteria and failure modes: `docs/calibration.md`.

Feasibility budget:

```python
from sperm_sorting.shots.feasibility import assess_feasibility

report = assess_feasibility(cfg, chamber_depth_um=20.0)
print(report.format_report())
report.feasible          # False if there are any warnings
report.to_json_dict()    # written into the audit manifest
```

Operator scripts, which wrap the same functions:

```bash
python scripts/calibrate_optics.py            # stage micrometer -> um_per_px
python scripts/calibrate_transport_delay.py   # tracer bolus -> transport delay
python scripts/check_feasibility.py           # throughput / observation / latency budgets
```

---

## 5A. Datasets and training

Outside `sperm_sorting` itself, but part of the public surface.

**Dataset adapters** (`datasets.adapters`, lazily imported per PEP 562) --
`mhsma`, `visem_tracking`, `visem`, `visem_graphs`, `detection_sperm` and
`device`. All present the same internal format: boxes as absolute-pixel
`(x1, y1, x2, y2)`, morphology labels in MHSMA polarity (`0 = normal`,
`1 = abnormal`), and a `DatasetInfo` carrying `CaptureConditions` so that the
optical mismatch described in `docs/domain_shift.md` travels with the data.

`DeviceDatasetAdapter` (`datasets/adapters/device.py`) reads the JSON Lines
capture format written by `DeviceAnnotationWriter`. A morphology aspect that was
**not assessed** must be `None` or absent, never `0` -- "nobody looked" and
"looked, and it was normal" are different facts.

**Validators** (`datasets.validators`) -- `integrity` (returns a
`ValidationReport` rather than raising on the first problem), `leakage`
(`assert_no_frame_leakage`, `check_adjacent_frames`, `patient_level_split`; it
**raises**, because the failure it guards against makes metrics look better) and
`licenses` (`get_license`, `check_commercial_use`, `check_share_alike`,
`strictest_terms`).

**Converters** (`datasets.converters`) -- `to_detection_format`
(YOLO / Pascal VOC / COCO to and from the internal `Detection`), `to_mot_format`
(MOTChallenge text, so standard HOTA/IDF1 tooling can score this project), and
`to_crops` (which delegates to the real `CropExtractor` rather than
reimplementing the crop rule).

**Training and evaluation entry points**:

```bash
python training/train_morphology.py --source synthetic --epochs 2 -o runs/morph_smoke
python training/train_detector.py   --source synthetic --epochs 2 -o runs/det_smoke
python training/eval_morphology.py  --checkpoint ... --calibration ... --split test
python training/eval_detector.py    --checkpoint ... --split test
python training/eval_tracking.py    --source synthetic --split test   # HOTA, IDF1, MOTA, MOTP, ID switches, fragmentation
python -m sperm_sorting.cli generate-data --n 20000 --out data --image-size 128
```

**No training run has been executed and `models/` contains no weights.** There
are no performance figures anywhere in this repository, and there must be none
until there are measurements to report -- see `docs/safety_and_claims.md`
section 7.2.

---

## 6. Audit-log format

One directory per run, `<audit_dir>/<YYYYmmdd-HHMMSS>-<run_name>/`, containing
five files. Everything is written at **shot rate (~1 Hz)**, never frame rate, and
every stream is flushed after every record -- a syscall per shot, so that a power
loss leaves a complete log up to the last decision rather than an empty buffer.

```
runs/20260804-141207-device-v1/
  manifest.json     configuration, versions, environment, calibration state, git commit
  events.jsonl      shots, decisions, commands, faults
  tracks.jsonl      per-sperm records (optional; large)
  metrics.jsonl     periodic runtime metrics
  summary.json      end-of-run summary, written at close
```

```python
from sperm_sorting.monitoring.audit import AuditLogger, read_events

with AuditLogger("runs", run_name="device-v1", audit_tracks=True) as audit:
    audit.write_manifest(cfg.summary(), environment=..., feasibility=...)
    audit.shot_decided(shot, decision)
    audit.command_dispatched(command)
    audit.track(track)
    audit.metrics(metrics.snapshot())
    audit.fault("acquisition", "camera disconnected")
    audit.close(summary=pipeline.summary())

events = read_events("runs/20260804-141207-device-v1/events.jsonl")
```

`read_events` skips malformed trailing lines rather than raising: a log truncated
by a power loss should still be readable up to the cut.

Setting `monitoring.audit_dir: null` disables audit logging entirely -- and logs
a warning saying that decisions will not be reconstructable, which also voids the
replay-determinism guarantees.

### 6.1 `manifest.json`

```json
{
  "schema_version": "1.0.0",
  "package_version": "0.1.0",
  "written_at": "2026-08-04T14:12:07+0000",
  "git_commit": "d01ba63f1c2e4a5b6789012345678901234abcde",
  "python": "3.11.9 (main, Apr  2 2026, 09:12:33) [GCC 13.2.0]",
  "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
  "hostname": "bench-01",
  "pid": 48213,
  "config": {
    "run_name": "device-v1",
    "mode": "live",
    "seed": 1234,
    "source": "basler",
    "detector": "p2net",
    "tracker": "bytetrack",
    "morphology_backbone": "mobilenetv3_small",
    "morphology_model_id": "unset",
    "weights_provenance": "public-research-baseline",
    "decision_threshold": 0.6,
    "minimum_trackable": 20,
    "motility_profile": "who6-2021-s2.4.6.1-v1",
    "flow_correction": "robust_estimate",
    "optically_calibrated": false,
    "scheduling_calibrated": false
  },
  "environment": {
    "torch": "2.3.1", "cuda_available": true, "cuda_devices": ["NVIDIA RTX A2000"],
    "onnxruntime": "1.18.0", "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": null
  },
  "feasibility": {
    "um_per_px": 0.0345,
    "um_per_px_is_measured": false,
    "field_width_um": 66.24, "field_height_um": 41.4, "field_length_um": 66.24,
    "flow_speed_um_s": 331.2,
    "residence_time_s": 0.2, "frames_per_transit": 32.0, "min_frames_required": 6,
    "required_visible_sperm": 5.0,
    "required_concentration_per_ml": 91159000.0,
    "chamber_depth_um": 20.0,
    "whole_sperm_fits_along_flow": true,
    "whole_sperm_fits_across_flow": false,
    "feasible": false,
    "warnings": ["a whole 53 um spermatozoon does not fit across the 41.4 um field dimension, ..."]
  },
  "full_config": { "...": "the complete AppConfig, model_dump(mode='json')" }
}
```

Two fields do most of the work when a run is later questioned:
`weights_provenance` (a model that never declared its lineage reads `"unset"`)
and the two `*_calibrated` booleans.

### 6.2 `events.jsonl`

One self-describing record per line. Every record has `t` (monotonic) and `kind`.

**`shot_decided`** -- the central record:

```json
{"t": 1183.4471, "kind": "shot_decided",
 "shot": {"shot_id": 12, "opened_at_s": 41.2153, "opened_frame_id": 6595,
          "closed_at_s": 41.7106, "closed_frame_id": 6674, "close_reason": "target_reached",
          "duration_s": 0.4953, "first_gate_time_s": 41.2211, "last_gate_time_s": 41.7098,
          "gate_span_s": 0.4887, "trackable_count": 25, "ai_eligible_count": 15,
          "ai_eligible_ratio": 0.6, "status": "reject",
          "threshold_applied": 0.6, "minimum_trackable_applied": 20,
          "track_ids": [418,419,421,422,424,425,427,428,430,431,433,434,436,437,
                        439,440,442,443,445,446,448,449,451,452,454],
          "eligible_track_ids": [418,421,424,427,430,433,436,439,442,445,448,451,454,419,422],
          "ineligibility_histogram": {"not_progressive": 6, "abnormal_head": 2,
                                      "abnormal_acrosome": 1, "deadline_missed": 1},
          "rejected_track_count": 3, "schema_version": "1.0.0"},
 "decision": {"shot_id": 12, "status": "reject", "field_command": "FIELD_ON",
              "ai_eligible_count": 15, "trackable_count": 25, "ratio": 0.6,
              "threshold": 0.6, "minimum_trackable": 20,
              "rationale": "15/25 = 0.6000 does not exceed 0.60; segment rejected and the field is energised to divert it"}}
```

*(That is boundary case 15/25: exactly 60%, which **rejects**, and the rejection
is `FIELD_ON`. See `docs/pipeline.md` section 4.)*

**`command`**:

```json
{"t": 1183.4479, "kind": "command",
 "command": {"command_id": 24, "kind": "FIELD_ON", "origin": "decision",
             "activate_at_s": 41.3411, "dispatch_at_s": 41.3211, "deadline_s": 41.3261,
             "duration_s": null, "shot_id": 12, "created_at_s": 41.7112,
             "outcome": "dispatched", "dispatched_at_s": 41.3215,
             "acknowledged_at_s": 41.3221, "timing_error_s": 0.0004,
             "failure_reason": "", "schema_version": "1.0.0"}}
```

**`fault`**:

```json
{"t": 1421.8830, "kind": "fault", "source": "frame_buffer",
 "message": "frames are ageing out of the buffer before morphology runs; increase the buffer capacity or shorten track lifetimes"}
```

### 6.3 `tracks.jsonl`

One record per sperm, written when the track resolves. Enabled by
`monitoring.audit_tracks`; per-point lists are additionally gated by
`audit_track_points` because they are large.

```json
{"track_id": 431, "state": "removed",
 "first_frame_id": 6602, "last_frame_id": 6634,
 "first_time_s": 41.2597, "last_time_s": 41.4597, "duration_s": 0.2,
 "n_points": 33, "n_observed": 31, "mean_score": 0.7412,
 "shot_id": 12, "gate_crossing_frame_id": 6633, "gate_crossing_time_s": 41.4535,
 "track_quality_pass": true, "track_quality_reason": "",
 "evaluation_complete": true, "evaluation_deadline_s": 41.7097,
 "ai_eligible": false, "ineligibility_reason": "abnormal_head",
 "motion": {"n_points": 33, "n_observed_points": 31, "duration_s": 0.2,
            "mean_frame_interval_s": 0.00625, "timestamp_source": "hardware",
            "flow_correction_mode": "robust_estimate",
            "profile_version": "who6-2021-s2.4.6.1-v1|ma|w17",
            "optically_calibrated": true, "um_per_px": 0.03452,
            "vcl_px_s": 9714.2, "vsl_px_s": 9603.8, "vap_px_s": 9660.1,
            "vcl_corrected_px_s": 1131.4, "vsl_corrected_px_s": 903.7,
            "vap_corrected_px_s": 1012.6,
            "vcl_um_s": 39.06, "vsl_um_s": 31.19, "vap_um_s": 34.96,
            "lin": 0.7987, "str": 0.8923, "wob": 0.8951,
            "alh_um": 1.42, "alh_unavailable_reason": "",
            "bcf_hz": 12.8, "bcf_unavailable_reason": "",
            "net_displacement_px": 180.74, "path_length_px": 226.28,
            "direction_rad": 0.0412, "direction_stability": 0.1123,
            "flow_vx_px_s": 9598.4, "flow_vy_px_s": -3.1,
            "motility_class": "rapid_progressive",
            "motility_reason": "rapid progressive: corrected VSL 31.2 um/s >= 25.0 um/s with LIN 0.80 >= 0.35"}, 
 "crop": {"track_id": 431, "frame_id": 6621, "capture_time_s": 41.3785,
          "source_box_xyxy": [812.4, 553.1, 941.8, 645.7], "output_size": [128, 128],
          "quality_score": 0.7314,
          "quality_terms": {"focus": 0.81, "motion_blur": 0.92, "local_contrast": 0.55,
                            "exposure": 0.88, "overlap": 1.0, "truncation": 1.0,
                            "detector_score": 0.79, "track_confidence": 0.74},
          "truncated": false, "visible_fraction": 1.0, "tail_complete": false,
          "max_overlap_iou": 0.0, "detector_score": 0.7912, "track_confidence": 0.7412},
 "morphology": {"track_id": 431, "status": "complete", "frame_id": 6621,
                "head":     {"name": "head",     "p_normal": 0.3120, "threshold": 0.5, "normal": false, "label": 1, "margin": 0.188},
                "acrosome": {"name": "acrosome", "p_normal": 0.7410, "threshold": 0.5, "normal": true,  "label": 0, "margin": 0.241},
                "vacuole":  {"name": "vacuole",  "p_normal": 0.8802, "threshold": 0.5, "normal": true,  "label": 0, "margin": 0.380},
                "tail":     {"name": "tail",     "p_normal": 0.6531, "threshold": 0.5, "normal": true,  "label": 0, "margin": 0.153},
                "all_four_normal": false, "latency_ms": 6.42,
                "model_id": "unset", "weights_provenance": "public-research-baseline",
                "failure_reason": "", "schema_version": "1.0.0"},
 "schema_version": "1.0.0"}
```

Note `tail_complete: false` -- expected routinely, because a whole spermatozoon
does not fit across the field (`docs/assumptions.md` section 1.6). And note that
the four aspect probabilities are never averaged: `all_four_normal` is a
conjunction, and `ineligibility_reason` names the **first** failing aspect in
canonical order.

### 6.4 `metrics.jsonl`

```json
{"t": 1183.0021, "elapsed_s": 41.712, "acquisition_fps": 159.87, "processed_fps": 159.34,
 "frames_acquired": 6674, "frames_processed": 6652,
 "frames_dropped_source": 0, "frames_dropped_quality": 22, "frames_dropped_backpressure": 0,
 "drop_rate": 0.0033,
 "detections_total": 30184, "tracks_created": 457, "tracks_gated": 331,
 "crops_extracted": 198, "morphology_completed": 195,
 "morphology_failed": 1, "morphology_deadline_missed": 2,
 "queue_high_water": {"frames": 0.375},
 "latency": {
   "preprocess":   {"count": 6652, "p50_ms": 0.41,  "p95_ms": 0.63,  "p99_ms": 1.02, "max_ms": 2.11},
   "quality_gate": {"count": 6674, "p50_ms": 0.28,  "p95_ms": 0.44,  "p99_ms": 0.71, "max_ms": 1.55},
   "detect":       {"count": 6652, "p50_ms": 3.12,  "p95_ms": 4.08,  "p99_ms": 5.44, "max_ms": 9.87},
   "track":        {"count": 6652, "p50_ms": 0.52,  "p95_ms": 0.91,  "p99_ms": 1.44, "max_ms": 3.02},
   "flow":         {"count": 6652, "p50_ms": 0.09,  "p95_ms": 0.16,  "p99_ms": 0.27, "max_ms": 0.61},
   "gate":         {"count": 6652, "p50_ms": 0.04,  "p95_ms": 0.08,  "p99_ms": 0.13, "max_ms": 0.30},
   "motion":       {"count": 457,  "p50_ms": 0.77,  "p95_ms": 1.21,  "p99_ms": 1.88, "max_ms": 3.44},
   "best_frame":   {"count": 209,  "p50_ms": 2.03,  "p95_ms": 3.14,  "p99_ms": 4.22, "max_ms": 7.10},
   "crop":         {"count": 198,  "p50_ms": 0.66,  "p95_ms": 1.02,  "p99_ms": 1.51, "max_ms": 2.80},
   "morphology":   {"count": 198,  "p50_ms": 6.11,  "p95_ms": 8.44,  "p99_ms": 11.20, "max_ms": 18.33},
   "decision":     {"count": 13,   "p50_ms": 0.02,  "p95_ms": 0.03,  "p99_ms": 0.05, "max_ms": 0.09}}}
```

**Every number in this example is illustrative, not measured.** No model in this
repository has been trained and no benchmark has been run; the values show the
shape of the record, nothing more.

Latency is reported as **percentiles, never a mean**: a pipeline that meets its
deadline 99% of the time and misses catastrophically 1% of the time has a fine
mean and is unusable. The three drop counters are kept separate because they have
three different remedies.

### 6.5 `summary.json`

Written by `audit.close(summary=...)`, from `Application.summary()`.

```json
{"frames_processed": 6652, "tracks_created": 457, "tracks_gated": 331,
 "shots": 13, "accept": 4, "reject": 8, "indeterminate": 1,
 "mean_trackable_per_shot": 24.2,
 "commands_dispatched": 16, "commands_late": 0, "commands_dropped_late": 0,
 "watchdog": {"timeout_s": 0.5, "tripped": false, "n_trips": 0, "time_since_fed_s": 0.0061},
 "actuator": {"name": "serial", "state": "FIELD_OFF", "open": false,
              "n_commands": 16, "n_ack_failures": 0, "can_acknowledge": true},
 "queues": {"frames": {"name": "frames", "capacity": 16, "put_count": 6674,
                       "get_count": 6674, "dropped_count": 0, "blocked_count": 0,
                       "high_water": 6}},
 "workers": [{"name": "acquisition", "started": true, "finished": true, "alive": false,
              "iterations": 6675, "error": null},
             {"name": "analysis", "started": true, "finished": true, "alive": false,
              "iterations": 6689, "error": null}]}
```

---

## 7. Exceptions

```python
from sperm_sorting.errors import (
    SpermSortingError,          # base
    ConfigurationError,         # refuse to start
    CalibrationError,           # start, but refuse physical units / actuation
    HardwareError, CameraError, ActuatorError, WatchdogTimeout,
    InferenceError, BackendUnavailableError,
    DatasetError, DatasetNotFoundError, DatasetValidationError, LeakageError,
    SchedulingError, LateCommandError, DeadlineMissed,
)
```

Plus `CropIdentityError` (`cropping/extractor.py`) and
`BestFrameOrderingError` (`quality/selector.py`), which enforce invariants I2 and
I3 from `docs/pipeline.md`.

`DeadlineMissed` is an **expected, logged condition**, not a crash: a track whose
morphology did not finish in time stays in the shot denominator and is excluded
from the numerator.

---

## 8. Constants

```python
from sperm_sorting.constants import (
    ACCEPT_RATIO_THRESHOLD,               # 0.60 -- exactly at it REJECTS
    MINIMUM_TRACKABLE_SPERM,              # 20
    TARGET_TRACKABLE_SPERM,               # 25
    MAXIMUM_TRACKABLE_SPERM,              # 30
    MAXIMUM_SHOT_DURATION_S,              # 1.0
    LABEL_NORMAL, LABEL_ABNORMAL,         # 0, 1  (MHSMA convention)
    MORPHOLOGY_ASPECTS,                   # ("head", "acrosome", "vacuole", "tail")
    DEFAULT_RAPID_PROGRESSIVE_VSL_UM_S,   # 25.0  (WHO 6th ed. 2.4.6.1)
    DEFAULT_SLOW_PROGRESSIVE_VSL_UM_S,    # 5.0   (WHO 6th ed. 2.4.6.1)
    MOTILITY_PROFILE_VERSION,             # "who6-2021-s2.4.6.1-v1"
    SCHEMA_VERSION,                       # "1.0.0"
    WEIGHTS_PROVENANCE_PUBLIC,            # "public-research-baseline"
    WEIGHTS_PROVENANCE_DEVICE,            # "device-finetuned"
    WEIGHTS_PROVENANCE_SYNTHETIC,         # "synthetic-bootstrap"
    EPS, UNAVAILABLE,
)
```

Physical values that depend on the built device -- transport delay, field rise
time, micrometres per pixel, the flow vector -- deliberately do **not** live here.
They are configuration values that must be measured; see `docs/calibration.md`.
