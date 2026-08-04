# Developer guide

Everything needed to pick this repository up and keep building it: what each
file does, why it is shaped the way it is, how the pieces connect, and which
things you must not break.

Read [§2 The mental model](#2-the-mental-model) and
[§4 The invariants](#4-the-invariants) before changing anything. The rest is
reference.

---

## Contents

1. [Getting running in five minutes](#1-getting-running-in-five-minutes)
2. [The mental model](#2-the-mental-model)
3. [Data flow, with real types](#3-data-flow-with-real-types)
4. [The invariants](#4-the-invariants)
5. [Repository map](#5-repository-map)
6. [`src/sperm_sorting/` — module by module](#6-srcsperm_sorting--module-by-module)
7. [The configuration system](#7-the-configuration-system)
8. [`datasets/`](#8-datasets)
9. [`training/`](#9-training)
10. [`web/`](#10-web)
11. [`tests/`](#11-tests)
12. [`scripts/` and the CLI](#12-scripts-and-the-cli)
13. [Recipes: how to do common things](#13-recipes-how-to-do-common-things)
14. [Traps](#14-traps)
15. [Where to continue](#15-where-to-continue)
16. [Glossary](#16-glossary)

---

## 1. Getting running in five minutes

```bash
git clone git@github.com:hannan72/sperm_sorting.git
cd sperm_sorting
python -m venv .venv && source .venv/bin/activate

# CPU torch is enough for everything except real detector training
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[all]"

sperm-sorting doctor                                  # what is installed, what is calibrated
sperm-sorting feasibility -c configs/synthetic.yaml   # can the optics deliver 25 sperm/s?
sperm-sorting run -c configs/synthetic.yaml -n 500    # the whole pipeline, on the simulator
pytest -q                                             # 176 tests, ~4 minutes on CPU
pytest -q -m "not slow"                               # 116 invariant tests, ~2 seconds
```

`make help` lists shortcuts. Everything below assumes `.venv` is active.

**Three commands worth internalising.** `doctor` tells you what the process can
actually do right now. `feasibility` tells you whether the *physics* of your
configuration closes — it catches problems that otherwise appear weeks later as
a stream of `INDETERMINATE` shots. `run -c configs/synthetic.yaml` exercises
every stage end to end against known ground truth in about 90 seconds.

---

## 2. The mental model

A semen sample flows continuously through a microfluidic kit. Before entering,
it is incubated with Annexin V-conjugated magnetic microbeads. Downstream of
the microscope sits an electromagnet.

Two mechanisms act on the sample, and they are **independent**:

- **The magnetic layer** separates on Annexin V binding — a biochemical
  property. This software cannot see it and must never claim to.
- **The AI layer** — this repository — watches the flow through the microscope
  and decides, segment by segment, whether to energise the field.

The software's entire output is one bit, updated roughly once per second:

```
FIELD_ON   → energise the magnet → this segment is diverted to waste
FIELD_OFF  → magnet off          → this segment passes to collection
```

To produce that bit it cuts the continuous flow into **shots** — software-
defined segments of ~25 uniquely trackable sperm, closed after at most one
second — counts how many sperm in each shot satisfy a combined
motility-and-morphology rule, and compares the ratio against 60%.

### The one paragraph that explains the whole design

Everything hangs off **track identity**. One physical sperm must get one
persistent ID, be counted exactly once, have its velocity measured from its own
trajectory, and have its morphology judged from a crop of *itself*. Every
subsystem here exists to protect that binding, and every subtle bug in the
project so far has been a violation of it. When you are unsure whether a change
is safe, ask what it does to track identity.

### What FIELD_ON means

`FIELD_ON` is the **rejection**. Reading it as "good" inverts the product. It
is also *not* the safe state: `FIELD_OFF` is, because an unsorted sample is a
degraded outcome while a stuck-on field silently diverts everything to waste.
Every fault path, watchdog expiry and shutdown ends in `FIELD_OFF`.

---

## 3. Data flow, with real types

One frame's journey. Types are the real ones from `src/sperm_sorting/schemas/`.

```
FrameSource.read()                        -> FramePacket
  FramePreprocessor.process(packet)       -> FramePacket   (ROI, normalise)
  ImageQualityGate.apply(packet)          -> FramePacket | None   (None = dropped)
  FrameBuffer.put(packet)                                  (bounded ring buffer)
  Detector.detect(packet)                 -> list[Detection]
  Tracker.update(dets, packet)            -> list[TrackRecord]   (same objects each call)
  FlowEstimator.estimate(tracks, packet)  -> (vx, vy) | None
  CountingGate.update(track)              -> GateCrossing | None
    on crossing:
      assess_track_quality(track, cfg)                     (final; sets track_quality_pass)
      ShotManager.add_track(track, t, fid) -> ShotRecord | None   (returns a CLOSED shot)
  Tracker.finished_tracks()               -> list[TrackRecord]   (drained once, ever)
    for each finished track that was gated:
      ProgressiveMotilityClassifier.classify(track, flow)  -> MotilityClass
      if progressive:
        BestFrameSelector.select(track, frames, dets)      -> list[CandidateFrame]
        CropExtractor.extract(track, cand, frame)          -> (np.ndarray, CropRecord)
        MorphologyEngine.evaluate_track(track, crop, deadline) -> MorphologyResult
      track.compute_eligibility()                          -> bool  (sets ai_eligible)
      ShotManager.notify_track_resolved(track_id)
  ShotManager.ready_shots(t, tracks_by_id) -> list[ShotRecord]
    DecisionEngine.evaluate(shot)          -> Decision
    ActuationScheduler.submit(kind, gate_time_s=...)       -> FieldCommand
  ActuationScheduler.poll(t)               -> list[FieldCommand]   (dispatches due ones)
    MagneticActuator.apply(command)        -> bool
```

All of that is one method: `Pipeline.process_frame(packet) -> FrameResult`
([`runtime/pipeline.py`](src/sperm_sorting/runtime/pipeline.py)). It is
synchronous and deterministic. `PipelineRunner` wraps it for live capture,
where acquisition needs its own thread.

### The two timelines

There are two, and confusing them is the most common source of subtle bugs.

| Timeline | Field | Used for |
|---|---|---|
| **Capture time** | `FramePacket.capture_time_s` | Velocity, shot duration, gate crossings, scheduling. Comes from the camera's own clock where available. |
| **Host time** | `FramePacket.received_time_s`, `Clock.now()` | Latency accounting, morphology deadlines, the watchdog. |

Motion analysis must use capture time, or every velocity picks up USB and
scheduling jitter. `FramePacket.timestamp_source` records which clock a
timestamp actually came from, so a reader can tell a hardware-timed measurement
from a software-timed approximation.

---

## 4. The invariants

These are the acceptance criteria. Each has tests whose failure means the
product is wrong, not that a test is flaky.

### 4.1 One sperm is counted exactly once

Track IDs are globally unique per tracker instance and **never reused**, even
after a track is removed (`TrackingConfig.reuse_track_ids` is typed
`Literal[False]` to make the intent unmissable). `CountingGate` keeps a
permanent `_crossed` set so a sperm loitering on the gate line cannot be counted
twice. `ShotManager` keeps a permanent `_assigned` set as a second barrier.
`ShotRecord.add_track` returns `False` on a duplicate rather than silently
deduplicating.

*Tested by:* `tests/test_shots.py`, `tests/test_pipeline_integration.py::test_every_sperm_is_counted_at_most_once`.

### 4.2 Exactly 60% REJECTS

The comparison is a strict `>`, evaluated with **exact rational arithmetic**:

```python
Fraction(numerator, denominator) > Fraction(str(threshold))
```

`Fraction(str(0.60))` is exactly `3/5`. `Fraction(0.60)` is the binary
approximation, which is slightly *less* than 3/5 and would make exactly-60%
read as above threshold. That single detail is the difference between a correct
and an inverted boundary.

*Tested by:* `tests/test_decision_rule.py` — the five mandated cases plus every
exactly-60% pair from denominators 20 to 200.

### 4.3 The denominator never shrinks

`ai_eligible_ratio = ai_eligible_count / trackable_count`, where the
denominator is **every** valid trackable sperm assigned to the shot. Not just
the progressive ones. Not with abnormal sperm removed. Not with deadline
misses dropped.

Track quality is assessed **once, at the gate**, and the verdict is final. If it
could be revisited later, the denominator could shrink after the numerator was
known — which is precisely the manipulation that would make a bad sample look
good.

### 4.4 The all-four morphology rule is conjunctive

```python
all_four_normal = head_normal and acrosome_normal and vacuole_normal and tail_normal
```

Never an average. A missing or failed aspect is **not** normal.
`MorphologyResult.all_four_normal` is the only implementation; nothing else may
decide this. Probabilities of 0.99/0.99/0.99/0.02 average to 0.75 and would
pass any mean-based rule — `tests/test_eligibility_rule.py::test_all_four_is_not_an_average`
asserts they do not pass here.

### 4.5 Morphology never runs before tracking

A crop is only meaningful if it can be bound to the motion measurement of the
*same* cell, and that binding does not exist until the cell has an ID and a
trajectory. `BestFrameSelector.select` raises `BestFrameOrderingError` if handed
a track with no motion features or one that is not progressive.
`CropRecord.track_id` must equal `TrackRecord.track_id`, asserted inside
`CropExtractor.extract` (raising `CropIdentityError`) and again in the
integration tests.

### 4.6 FIELD_OFF is the fail-safe

Every one of these ends with the magnet off: startup, shutdown, inference
failure, camera disconnect, actuator write failure, acknowledgement mismatch,
watchdog expiry, `INDETERMINATE` shot, Ctrl-C.

The watchdog exists for the case a crash does *not* cover: a **hang**. A stuck
inference call or a deadlocked queue leaves the process alive with nothing
raised, and without a watchdog the field would stay wherever the last decision
left it.

*Tested by:* `tests/test_scheduling_and_actuation.py`,
`tests/test_pipeline_integration.py::test_field_ends_off_after_*`.

### 4.7 Nothing physical is guessed

Transport delay, field rise/fall times, µm/px and the bulk flow vector have no
defaults. They are `None` or `calibrated: false` until measured, and the system
**refuses** rather than substituting a plausible number:

- `OpticalCalibration.require_calibrated()` raises; velocities stay in px/s and
  motility grading returns `UNDETERMINED` rather than comparing µm/s thresholds
  against pixel values.
- `SchedulingConfig.require_calibrated()` raises; `ActuationScheduler.arm()`
  fails and no command is driven.

Both failures would otherwise be silent: a wrong transport delay gates the wrong
fluid, a wrong µm/px rescales every velocity across the WHO boundaries, and
neither raises anything on its own.

### 4.8 Label polarity: 0 = normal, 1 = abnormal

MHSMA's convention, used verbatim throughout. The network emits a logit for
`P(abnormal)`, so the training target is the dataset label with **no flip
anywhere in training**. The single flip to `P(normal)` lives in
[`morphology/polarity.py:flip_polarity`](src/sperm_sorting/morphology/polarity.py)
and is called only from the inference adapter. `polarity.self_check()` runs at
import time; `load_checkpoint` and `CalibrationBundle.load_json` both refuse an
artefact whose recorded polarity string differs — or is absent, since a
pre-convention artefact is exactly the one that might be inverted.

> The MHSMA README calls the *normal* class "positive", so its "% Positive"
> column is percent **normal**. Reading that table as abnormality prevalence
> inverts every number.

---

## 5. Repository map

```
sperm_sorting/
├── src/sperm_sorting/     the product              98 files, ~27.6k lines
├── datasets/              adapters, converters, validators   18 files, ~8.7k
├── training/              training and evaluation  23 files, ~10.3k
├── web/                   FastAPI demo              7 files, ~4.0k
├── tests/                 unit + integration        8 files, ~2.2k
├── docs/                  audits and design docs   11 files, ~5.6k
├── configs/               YAML configurations       6 files
├── scripts/               operator utilities        3 files
├── models/                weights + calibration (gitignored, README only)
├── pyproject.toml         packaging, ruff, mypy, pytest config
├── requirements.lock      frozen environment
├── Makefile               shortcuts (make help)
├── Dockerfile             CPU runtime image
├── .github/workflows/ci.yml
├── README.md / README_FA.md
├── DEVELOPMENT.md         this file
├── LICENSE                Apache-2.0 (code only)
├── THIRD_PARTY_NOTICES.md
└── CITATION.cff
```

Read the docs in this order if you are new:
[`docs/pipeline.md`](docs/pipeline.md) →
[`docs/assumptions.md`](docs/assumptions.md) →
[`docs/architecture.md`](docs/architecture.md) →
[`docs/safety_and_claims.md`](docs/safety_and_claims.md).
[`docs/engineering_report.md`](docs/engineering_report.md) is the honest status
account — what is measured, what is not done, and why.

---

## 6. `src/sperm_sorting/` — module by module

### 6.1 The spine

These four files define contracts everything else depends on. Change them
carefully.

| File | Lines | What it is |
|---|---|---|
| `constants.py` | 107 | Specification-fixed values: the 60% threshold, shot sizes, `LABEL_NORMAL = 0`, `MORPHOLOGY_ASPECTS`, the WHO velocity limits, `SCHEMA_VERSION`. Nothing device-specific lives here. |
| `errors.py` | 112 | Exception hierarchy in three classes: startup (`ConfigurationError`, `CalibrationError`), runtime (`HardwareError`, `InferenceError`), data (`DatasetError`, `LeakageError`). The split matters because each drives a different response in the pipeline. |
| `config.py` | 1226 | The whole configuration surface. Pydantic v2, `extra="forbid"`. Heavily commented — it doubles as the design-rationale document. |
| `schemas/` | ~1200 | Every type that flows between stages. |

#### `schemas/`

`slots` dataclasses, not Pydantic — one is constructed per frame at up to
164 Hz and carries a large numpy buffer, so per-field validation on that path
would be wasted work. Configuration is Pydantic because it is built once from
untrusted YAML.

| File | Key types |
|---|---|
| `enums.py` | `SourceKind`, `TimestampSource`, `QualityVerdict`, `TrackState`, `MotilityClass`, `FlowCorrectionMode`, `MorphologyStatus`, `ShotCloseReason`, `ShotStatus`, `FieldCommandKind`, `CommandOrigin`, `CommandOutcome`, `IneligibilityReason`. All `str`-valued, so they serialise readably without a custom encoder. |
| `frame.py` | `FramePacket` (image + capture time + provenance + drop count), `FrameQuality`. |
| `detection.py` | `BoundingBox` (xyxy, with `from_xywh`/`from_cxcywh`/`iou`/`clipped`/`expanded`), `Detection`, `detections_to_array`. |
| `track.py` | `TrackPoint`, `MotionFeatures` (the full CASA record), `CropRecord`, **`TrackRecord`** — the central accounting unit, and the home of `compute_eligibility()`. |
| `morphology.py` | `AspectResult`, `MorphologyResult` — home of `all_four_normal` and `first_abnormal_aspect()`. |
| `shot.py` | `exceeds_threshold()` (the exact-rational comparison), `ShotRecord`. |
| `command.py` | `FieldCommand` — a *future-dated* instruction with `activate_at_s`, `dispatch_at_s`, `deadline_s`. |

`MotionFeatures` deserves a note: it carries velocities in **two unit systems**
(`*_px_s` always, `*_um_s` only when calibrated) and **two frames of reference**
(raw and flow-corrected). Progressive classification uses the corrected values;
the raw ones are kept so a reviewer can see how large the correction was.

### 6.2 `acquisition/` — where frames come from

| File | What |
|---|---|
| `base.py` | `FrameSource` ABC. Three guarantees: honest timestamps, explicitly reported drops (never inferred from a frame-id gap), and identical downstream behaviour whichever source is running. |
| `basler.py` | Live pypylon acquisition. **Read the module docstring** — three things here are easy to get wrong: grab strategy (`OneByOne`, never `LatestImageOnly`), timestamps (chunk `BslChunkTimestampValue` preferred, 1 GHz tick on ace 2), and USB bandwidth (~378 MB/s at full rate, at the practical ceiling). |
| `video.py` | Replay via OpenCV. Uses container PTS where usable; logs loudly when it falls back to a nominal FPS, because every velocity in the run then depends on that number. |
| `synthetic.py` | Wraps the simulator. Bridges `SyntheticSourceConfig` → `SceneConfig`. Ground truth arrives in `FramePacket.meta["gt_detections"]` and `["gt_states"]`. |
| `factory.py` | `build_frame_source(cfg)`. Hardware backends imported lazily. |

### 6.3 `preprocessing/` — ROI, normalisation, the quality gate

- **`preprocessor.py`** (718 lines) — ROI crop, optional inversion, rolling-median
  background subtraction (bounded ring buffer; the median refreshes every
  `window // 8` frames because a full 64-deep median does not fit a 6 ms
  budget), and normalisation. Critically, it **translates ground-truth boxes
  into ROI coordinates** — without that, configuring an ROI silently breaks the
  oracle detector and every synthetic accuracy test.
- **`quality_gate.py`** — whole-frame verdict: `PASS` / `DEGRADED` / `REJECT`.
  A degraded frame still feeds tracking (continuity matters) but is never
  eligible for a morphology crop. All measurements are taken on a normalised
  0–1 view so thresholds mean the same thing for `uint8` and `float32` input,
  then scaled ×255 so the configured focus threshold has a familiar magnitude.

### 6.4 `detection/` — finding sperm heads

**The detection target is the head, not the whole cell.** At 0.0345 µm/px the
field of view is 66.2 × 41.4 µm and a whole spermatozoon is 50–60 µm; a head
(4.1 × 2.8 µm → 119 × 81 px) always fits. This also matches how CASA defines
kinematics and how MHSMA frames its crops.

| File | What |
|---|---|
| `base.py` | `Detector` ABC. Boxes must come back in **source-frame pixels**, with any internal resize/letterbox/tiling undone by the detector itself. |
| `preprocess.py` | Frame → network batch. Right/bottom padding with `BORDER_REPLICATE`, so no coordinate shifts and no manufactured hard edge. |
| `postprocess.py` | `nms`, `batched_nms`, `decode_centernet_heatmap`, `scale_boxes`, `merge_tiled_detections`, `finalise_boxes`. The inverse of `preprocess.py`. |
| `heads.py` | The shared anchor-free head: heatmap + size + offset, plus `centernet_focal_loss`, `masked_l1_loss`, `build_centernet_targets`. |
| `torch_base.py` | `TorchDetectorBase` — device/fp16/threads, checkpoint loading, tiling with a latency guard. Shared so that comparing `p2net` and `todcnn` measures the *backbone* and nothing else. |
| `p2net.py` | `P2Net` — encoder + top-down FPN whose finest and only prediction level is P2 (stride 4). The production candidate. |
| `todcnn.py` | `TodCnnNet` — an independent PyTorch reimplementation of the *concept* published as TOD-CNN (arXiv 2204.08166), not its weights. Never downsamples below stride 4, and is ~4.4× slower for it. |
| `onnx_detector.py` | ONNX Runtime backend. Handles both the raw-heads and the `(N,6)` output signatures. |
| `oracle.py` | Reads ground truth from `frame.meta`, degraded by configurable miss/false-positive/jitter rates. **Exists so a pipeline test fails when the pipeline is wrong, not when an untrained detector finds nothing.** It deliberately does *not* leak `track_id` into `Detection.track_id` — that would bypass the tracker and invalidate every tracking test. |

Two implementation notes worth knowing: the size head uses `softplus` (a raw
linear head can predict a negative width, producing an inverted box that
`BoundingBox` rejects — a crash, not a bad detection), and the gaussian radius
divides by `2a` rather than the `2` in the widely-copied CornerNet code.

### 6.5 `tracking/` — persistent identity

| File | What |
|---|---|
| `base.py` | `Tracker` ABC. Three guarantees: unique never-reused IDs, the *same* `TrackRecord` object returned for a given ID on every call, and predicted (unmatched) points appended with `observed=False`. |
| `_common.py` | `TrackStore`, `ManagedTrack`, `TrackerBase`. All three trackers share this, so the identity bookkeeping — which *is* the tested contract — exists once. |
| `kalman.py` | Constant-velocity filter on `[cx, cy, a, h, ...]`. Pure numpy. |
| `assignment.py` | `iou_batch`, `giou_batch`, `linear_assignment` (scipy Hungarian with a cost gate), `velocity_direction_cost`. |
| `bytetrack.py` | Three association passes. The second pass against *low-score* detections is the whole idea and is what keeps dim, partly-occluded sperm alive. |
| `ocsort.py` | Observation-Centric Momentum, Re-Update and Recovery. |
| `botsort.py` | ByteTrack + optional camera-motion compensation and ReID fusion. |

Two things to know. **Trailing predicted points are trimmed when a track is
retired** — a track dies because `max_age` frames of prediction went
unconfirmed, and keeping that tail would push short tracks over
`max_interpolated_fraction` and drop real sperm out of the denominator. Interior
gaps are always kept. And **observed points carry the detector's box, not the
Kalman posterior**, so downstream smoothing does not smooth an already-smoothed
signal.

**CMC defaults off**, and that is deliberate: the camera is rigidly mounted, so
global image motion *is* fluid flow. Absorbing it into the tracker would destroy
the velocity measurement this product depends on. It must be handled downstream
by flow correction.

### 6.6 `motion/` — CASA kinematics and WHO grading

| File | What |
|---|---|
| `smoothing.py` | `moving_average_path` (the CASA average path), `savgol_path`, `path_length`, `net_displacement`. |
| `flow.py` | `DisabledFlow`, `FixedVectorFlow`, `FlowMapFlow`, `RobustFlowEstimator`, `apply_flow_correction`. |
| `features.py` | `compute_motion_features` — fills in every field of `MotionFeatures`. |
| `classifier.py` | `classify_motility`, `ProgressiveMotilityClassifier`, `assess_track_quality`. |

**Flow correction is not optional decoration.** Observed motion is swimming
*plus* bulk transport. Without correction, a dead sperm drifting at 300 µm/s
would be graded rapidly progressive and the shot ratio would measure the pump.
`RobustFlowEstimator` takes a **median** over the slowest quantile of tracks, so
a few fast swimmers cannot drag it, and returns `None` below `robust_min_tracks`
rather than guessing.

`MotionFeatures.flow_correction_mode` records what was **applied**, not what was
configured. A configured `ROBUST_ESTIMATE` that produced no estimate is recorded
as `DISABLED`, and the classifier returns `UNDETERMINED` on that mismatch. This
is the guard that stops a passively drifting cell being graded progressive when
the estimator is starved.

**The smoothing window is specified in milliseconds, not frames**
(`MotionConfig.vap_window_ms`, resolved by `vap_window_frames(fps)`). A fixed
frame count smooths over different durations at different rates — five frames is
100 ms at 50 FPS but 31 ms at 160 FPS — which Mortimer et al. (2015) show
produces aberrant ALH. Measured here: 15× difference on one trajectory.

The four `MotilityClass` members **are** WHO 6th ed. §2.4.6.1, and 25 / 5 µm/s
are WHO's own approximate limits. `min_lin_for_progressive` is *stricter* than
WHO (whose wording admits progression "in a large circle") and is documented as
this implementation's choice; set it to `0.0` to follow WHO literally.

ALH and BCF are **refused** below `min_fps_for_alh_bcf` and returned as `None`
with a reason. Note the attribution: WHO specifies no minimum frame rate; the
~60 Hz figure is Mortimer et al. (2015).

### 6.7 `quality/` and `cropping/` — choosing and cutting the crop

- **`quality/frame_score.py`** — `score_candidate` scores one sperm in one frame
  as a crop source across eight weighted terms: local focus, motion blur, local
  contrast, exposure, overlap, truncation, detector score, track confidence.
  Motion blur is estimated by **gradient structure-tensor coherence** — a linear
  smear destroys variation along the direction of travel and preserves it
  across, so gradients collapse onto one axis. Because a sperm is intrinsically
  elongated the coherence never reaches zero, so the term is valid for *ranking
  frames of the same cell* and not as an absolute blur measure.
- **`quality/selector.py`** — `BestFrameSelector`, `CandidateFrame`,
  `FrameBuffer`. The buffer is a fixed-capacity ring keyed by frame id: at
  164 FPS a 1920 × 1200 `uint8` frame is 2.3 MB, so an unbounded cache reaches a
  gigabyte in about seven seconds.
- **`cropping/extractor.py`** — pads by `padding_fraction`, **letterboxes rather
  than squashing** (squashing changes the head length:width ratio, which is
  exactly what the head classifier keys on), clips at borders while recording
  `truncated` and `visible_fraction`, and estimates `tail_complete` — returning
  `None` when it genuinely cannot tell rather than guessing.

Detector confidence must never dominate selection. The config forbids
`w_detector_score >= 0.5`, and `validate_weights` additionally refuses
`w_detector_score + w_track_confidence >= 0.5`, because `track_confidence` *is*
the mean detector score and would otherwise smuggle it back in.

### 6.8 `morphology/` — four aspects, four heads

| File | What |
|---|---|
| `polarity.py` | **The label-polarity contract.** One convention, one flip, one file. `self_check()` runs at import. |
| `backbones.py` | `simplecnn` (from scratch, CPU-fast), `mobilenetv3_small`, `efficientnet_b0`. Grayscale stems adapt a pretrained RGB conv by **summing** across the channel axis, which preserves the learned filter response for a grayscale input. |
| `model.py` | `MultiTaskMorphologyNet` — one trunk, four independent heads. `MorphologyLoss` with per-aspect `pos_weight`. `save_checkpoint`/`load_checkpoint` record and enforce the polarity string. `export_onnx`. |
| `calibration.py` | `TemperatureScaler`, `fit_thresholds` (Youden / F1 / balanced accuracy / MCC), `expected_calibration_error`, `reliability_curve`, `CalibrationBundle`. Pure numpy — no torch, so the ONNX deployment path can calibrate without PyTorch installed. |
| `metrics.py` | Per-aspect sensitivity, specificity, precision, NPV, macro-F1, balanced accuracy, MCC, ROC-AUC, PR-AUC, ECE. Every docstring states which class is positive, because that is exactly where sign errors hide. |
| `inference.py` | `MorphologyEngine` (torch or ONNX), `RandomMorphologyEngine` (**test only**), deadline handling. |
| `factory.py` | `build_morphology_engine(cfg)`. |

**Class imbalance is severe and differs per aspect.** Verified MHSMA train
prevalence: acrosome 30.1%, head 27.3%, vacuole 17.0%, **tail 4.6%** — and the
validation split holds only **7** abnormal tails out of 240. Consequences that
are baked in: never select on raw accuracy (it is deliberately not reported),
per-aspect thresholds are fitted independently, and the evaluator prints an
explicit warning next to any aspect with fewer than ~20 positives.

Calibration travels as a **sidecar** (`<weights>.calibration.json`), not as a
config field, so it can be re-fitted without retraining. The engine prefers it
over config defaults and reports `threshold_source` per aspect.

Deadline handling: if the clock is already past `deadline_s` before inference
starts, the engine returns `DEADLINE_MISSED` **without running the model**. A
failure never silently becomes "normal".

### 6.9 `shots/` — cutting continuous flow into decision units

- **`gate.py`** — `CountingGate`. A crossing counts only when the centre moves
  across the line, in the configured flow direction, *and* the track's lifetime
  displacement along that axis exceeds a floor. The gate sits downstream
  (`position_fraction: 0.85`) so a track has been observed for most of its
  transit before being committed.
- **`manager.py`** — `ShotManager`. Two-phase lifecycle: **assembly** (closes on
  target count / hard maximum / one-second timeout) then **finalisation** (waits
  for members' morphology, bounded by a deadline). `_resolved_early` handles the
  common case where a track resolves *before* its shot closes — see
  [§14 Traps](#14-traps).
- **`feasibility.py`** — `assess_feasibility(cfg)`. Computes three budgets that
  each fail silently: throughput, observation, and **decision latency vs
  transport delay**. Run it after any change to optics, flow, shot sizing or the
  morphology deadline.

### 6.10 `decision/engine.py` — the rule

Small, pure, and dependency-free by design so the rule can be exhaustively
tested without constructing a pipeline. `decide()` is a pure function;
`DecisionEngine` is a thin wrapper that writes the verdict onto the `ShotRecord`
and **stamps the threshold and minimum in force**, so a later config change
cannot retroactively reinterpret an old log.

### 6.11 `scheduling/` — future-dated commands

- **`clock.py`** — `MonotonicClock` (production), `ManualClock` (tests and
  deterministic replay: `sleep` advances instead of blocking, so a one-second
  shot timeout is exercised in microseconds), `ScaledClock` (faster/slower
  replay).
- **`scheduler.py`** — `ActuationScheduler`. Timeline for one decision:

```
t_gate      the shot's fluid segment passes the counting gate
t_activate  = t_gate + transport_delay          field must be in state
t_dispatch  = t_activate - settle - margin      command must leave here
t_deadline  = t_dispatch + late_tolerance       after this it is LATE
```

Rising and falling edges use their **own** settle times (`field_rise_time_ms`
vs `field_fall_time_ms`); using one for both biases every command in one
direction. A command later than `drop_if_late_by_ms` is **dropped**, because
firing it would gate the wrong fluid — acting on the wrong segment is worse than
not acting.

### 6.12 `actuation/` — the magnet

`MagneticActuator` ABC + `MockActuator` (records every transition; what the
integration tests assert against), `GpioActuator` (libgpiod v2 — polarity
matters, since a line that floats on exit could energise the magnet), and
`SerialActuator` (line protocol to an MCU; a read timeout reports *unknown*
rather than assuming success, which would defeat the point of acknowledgement).

`Watchdog` forces `FIELD_OFF` when the pipeline stops feeding it.

### 6.13 `calibration/` — measuring the instrument

- **`optics.py`** — `calibrate_from_graticule` (FFT of the ruling period; uses
  every ruling in the field) and `calibrate_from_known_distance` (two-point
  fallback). Both **cross-check against the nominal scale implied by the optical
  train** and refuse a result more than 1.5× away, because a 0.5× reducing
  C-mount coupler is easy to overlook and puts every velocity out by exactly a
  factor of two. Measured accuracy on a synthetic graticule: 0.66%.
- **`flow.py`** — fixed vector (median of the slowest quantile) and flow map
  (coarse grid, nearest-neighbour fill, smoothing, bilinear resample — because
  pressure-driven flow in a microchannel is parabolic across the section, so a
  single vector over-corrects at the walls).
- **`transport.py`** — tracer-bolus measurement (**minimum three trials**; the
  spread sets the activation margin, so one trial gives no way to size it), plus
  a plug-flow geometry estimate that is explicitly a cross-check and not a
  substitute.

### 6.14 `runtime/`, `monitoring/`, `backends/`

- **`runtime/pipeline.py`** — `Pipeline.process_frame`. Synchronous and
  deterministic. The mandated order lives here.
- **`runtime/queues.py`** — `BoundedQueue` with an explicit overflow policy.
  `block` for replay (nothing may be lost, or the determinism guarantee is
  void); `drop_oldest` for live (a stale frame is worthless — by the time it is
  processed the fluid it shows has passed the magnet). Drops are always counted.
- **`runtime/workers.py`** — `PipelineRunner`. Threads rather than asyncio,
  because every stage is CPU-bound or blocks in a C extension. Acquisition gets
  its own thread; everything else shares one. Sixteen threads for sixteen named
  stages would add sixteen hand-offs of a 2.3 MB frame to parallelise
  sub-millisecond work.
- **`monitoring/audit.py`** — JSON Lines. A run directory holds
  `manifest.json`, `events.jsonl`, `tracks.jsonl`, `metrics.jsonl`,
  `summary.json`. Flushed after every record: a power loss then leaves a
  complete log up to the last decision rather than an empty buffer.
- **`monitoring/metrics.py`** — `RuntimeMetrics`, `LatencyTracker`. Latency is
  reported as **percentiles, never a mean** — a mean hides exactly the tail that
  breaks a real-time system.
- **`monitoring/health.py`** — `HealthMonitor`. Distinguishes `DEGRADED` (keep
  running, tell someone) from `FAILED` (stop, safe state).
- **`backends/runtime_backend.py`** — resolves torch/ONNX/TensorRT and **fails
  with an actionable message** rather than silently falling back to CPU, which
  would turn a deployment mistake into a mysterious latency regression.

### 6.15 `simulator/` — the only jointly-labelled data that exists

The public datasets do not overlap: MHSMA has morphology but no video,
VISEM-Tracking has boxes and track IDs but no morphology, VISEM has only
sample-level percentages. **Nothing public can validate the combined rule** —
only its two halves, separately. The simulator closes that gap by sampling one
ground-truth health state and emitting *both* a rendered image *and* a
trajectory for the same virtual cell.

| File | What |
|---|---|
| `params.py` | `HealthState`, `Prevalences`, `sample_health_state`. Each abnormal flag **causes** its knob to move outside the WHO band with a margin — the label is the cause of the appearance, not an independent annotation, or the classifier would have nothing learnable. |
| `render.py` | `render_sperm`, `render_sperm_on_canvas`, `render_debris_on_canvas`. Absorbance model; brightfield convention (objects darker than background). |
| `motility.py` | `simulate_trajectory`, `casa_features` — a **second, independent** CASA implementation, kept separate so the production one can be cross-checked against it. |
| `label.py` | The health rule in exactly one place, plus a `truth_table()` self-check over all 2⁴ × 4 combinations. |
| `scene.py` | `SceneGenerator` — multi-sperm frames with debris, flow, noise, defocus, and per-agent persistent `gt_track_id`. |
| `generate.py` | `build_dataset` — writes `images.npy`, `feats.npy`, `y_*.npy`, `meta.json` per split. |

Deterministic given a seed; `numpy.random.Generator` threaded through
explicitly, never global state. Every module has a runnable
`if __name__ == "__main__"` self-check.

Balancing distorts the marginals by design: 50/50 on the overall label makes the
healthy half all-normal-and-progressive, so per-aspect abnormal rates land near
half the configured prevalence. Every achieved rate is recorded in `meta.json`.

---

## 7. The configuration system

One `AppConfig` tree, built from YAML plus CLI overrides. `extra="forbid"`
everywhere, so a typo is a startup error rather than a silent no-op.

```bash
sperm-sorting run -c configs/synthetic.yaml \
  -s decision.threshold=0.65 \
  -s motion.thresholds.rapid_progressive_vsl_um_s=30 \
  -s acquisition.synthetic.density=12
```

Files use `extends:` for inheritance (relative to the child):

| Config | Purpose |
|---|---|
| `default.yaml` | Baseline. Uncalibrated by design — it will not actuate. |
| `synthetic.yaml` | Self-contained simulator run. Calibration is *asserted* (the simulator's geometry is known exactly), and the `calibration_id` says `not-a-real-instrument` so no audit log can be mistaken for one. |
| `replay.yaml` | Replay semantics: `backpressure: block`, deterministic seeding. The video comes from `--video`, because a video source with no path is not a valid configuration. |
| `device_v1.yaml` | The live instrument. **Refuses to actuate** until the two calibration blocks at the bottom are filled in from measurements. |
| `configs/training/morphology.yaml`, `configs/training/detector.yaml` | Model and operating-point settings for training. |

### Cross-section validators

`AppConfig` enforces relationships single fields cannot express:

- `decision.minimum_trackable_sperm` must equal `shots.minimum_trackable_sperm`
  — otherwise a shot could close "normally" at a size the decision engine calls
  `INDETERMINATE`.
- `crop.output_size` must equal `morphology.input_size` — otherwise every crop
  is silently resized at serving time.
- `run.mode == "replay"` requires `runtime.backpressure == "block"`.
- `BestFrameConfig` weights must sum to 1.0 and detector score may not dominate.
- `OpticalCalibration` rejects a measurement more than
  `max_nominal_discrepancy` from the optical train's nominal.

### Deriving nominal optics

`OpticsConfig` computes rather than stores: `nominal_um_per_px`,
`rayleigh_limit_um`, `abbe_limit_um`, `nyquist_oversampling`,
`field_of_view_um(w, h)`. For the reference build (3.45 µm pixel, 100×, 1×
coupler): 0.0345 µm/px, 0.268 µm Rayleigh, 3.89× oversampled, 66.24 × 41.40 µm.

---

## 8. `datasets/`

**No dataset is redistributed.** Every adapter works — imports, constructs, and
gives an actionable error naming the expected path and download URL — with the
data absent.

| Adapter | Dataset | Licence | Gives you |
|---|---|---|---|
| `mhsma.py` | MHSMA | CC BY-NC-SA 4.0 | 1540 crops, four binary aspect labels. No video. |
| `visem_tracking.py` | VISEM-Tracking | CC BY 4.0 | 20 videos, boxes + track IDs, 3 classes. No morphology. |
| `visem.py` | VISEM | CC BY-NC 4.0 | Sample-level WHO percentages only. **Exposes no per-sperm interface at all** (`sample_level_only = True`). |
| `visem_graphs.py` | VISEM-Tracking-graphs | CC BY 4.0 | Optional GNN extension. Includes a corrected video-graph rebuild. |
| `detection_sperm.py` | MIaMIA-SVDS | **UNCLEAR** | On-disk format unverified; sniffs VOC/YOLO/COCO. |
| `device.py` | Your captures | yours | The JSONL annotation schema for instrument data. |

`converters/` routes everything through the internal `Detection` type:
`to_detection_format.py` (YOLO ↔ VOC ↔ COCO), `to_mot_format.py` (so standard
HOTA/IDF1 tooling can score us), `to_crops.py` (reuses `CropExtractor` and
`score_candidate` from `src/`, so training crops are cut by the pipeline's own
rules).

`validators/leakage.py` is the most consequential file here. Adjacent frames of
one video are near-duplicates, so a frame-level split reports a memorisation
score as a validation score. It raises `LeakageError` rather than warning.

**`MhsmaAdapter.validate()` checks the polarity claim rather than trusting it**
— it asserts measured prevalence against the published figures and raises
`DatasetValidationError` naming the aspect if the labels look inverted. This is
the single check that prevents shipping an inverted product.

> Note: the package name `datasets` shadows Hugging Face `datasets` when
> importing from the repo root.

---

## 9. `training/`

Every script shares `--config` / `-s key=value` / `--out` / `--resume` /
`--device` / `--seed`.

| Script | What |
|---|---|
| `train_morphology.py` | Four-head network on MHSMA (`--source mhsma`) or simulator crops (`--source synthetic`). Preserves MHSMA's official split. Fits calibration on **validation**, never test, and writes a `CalibrationBundle` sidecar. |
| `train_detector.py` | P2Net or TOD-CNN with the shared CenterNet head. **Splits by video and fails loudly** on leakage. |
| `eval_morphology.py` | Full metric set at the shipped operating point, plus the all-four-normal joint accuracy — the quantity the product depends on, which is *not* the average of the four. Warns on low positive counts. |
| `eval_detector.py` | AP50, mAP50-95, small-object recall, debris FP rate, counting error, latency. AP implemented in-repo; no pycocotools dependency. |
| `eval_tracking.py` | HOTA, IDF1, MOTA, MOTP, ID switches, fragmentation, plus duplicate-count rate and track-survival distribution. HOTA follows Luiten et al. (2021) as realised in TrackEval, with `HOTA = mean_α HOTA_α`; per-α values are in the JSON so the aggregation is checkable. |
| `eval_pipeline.py` | **The one that matters.** Runs the whole product against simulator ground truth: per-sperm eligibility agreement, shot-ratio error, decision confusion matrix, command-alignment error. |

`common/` holds the shared machinery: `args`, `seeding`, `checkpoints` (which
genuinely resume — optimizer and scheduler state, not just weights), `earlystop`,
`logging_utils`, `experiment` (writes `experiment.json`: git commit, resolved
config, versions, dataset name and licence, split sizes, seed, hardware),
`plots`, `schedules`, `amp`, `augment`, and the two data loaders.

**Augmentation is constrained by what the labels assert.** Small rotations,
flips, mild brightness/contrast and slight blur are fine. No elastic
deformation, no aggressive scaling, no cutout over the head — those change the
very features (head shape, acrosome fraction, vacuole presence) being
classified. Each inclusion and exclusion is argued in `common/augment.py`.

Checkpoints are **supersets of the deployable format**, so
`morphology.model.load_checkpoint` reads `best.pt` directly with its polarity
guard and there is no export step to forget.

---

## 10. `web/`

FastAPI + vanilla HTML/JS/canvas. No build step, no framework, no CDN —
everything served locally, because the device runs offline.

`GET /` `/health` `/config` `/aspects`, `POST /generate` `/classify` `/decide`.

Three things the demo makes tangible: the generator as ground truth (predicted
vs true per aspect), the decision rule (with the five mandated cases as
one-click presets), and the optical budget.

`/decide` calls `sperm_sorting.decision.engine.decide` directly — **the rule is
never reimplemented in the API or in JavaScript.** With no trained weights the
service falls back to `RandomMorphologyEngine` and the UI displays the
provenance prominently, so the demo can never look like a working classifier
when it is not. Motility prediction *is* the real rule applied to simulated
kinematics; only morphology is untrained, and the response distinguishes them
per field.

```bash
uvicorn web.app:app --reload      # then open http://127.0.0.1:8000
pytest -q web/test_api.py         # 43 tests
```

---

## 11. `tests/`

176 tests, 1 skipped. `pytest -q -m "not slow"` runs the 116 invariant tests in
under two seconds; the full suite is a few minutes on CPU, dominated by the
end-to-end runs in `test_pipeline_integration.py`.

| File | Covers |
|---|---|
| `test_decision_rule.py` | The five mandated cases; every exactly-60% pair from d=20..200; that `Fraction(str(0.60)) == 3/5` while `Fraction(0.60) < 3/5`; FIELD_ON as the rejection. |
| `test_eligibility_rule.py` | The all-four rule (including that it is not an average); each of the five eligibility conditions individually; MHSMA polarity; per-aspect thresholds. |
| `test_shots.py` | Gate crossings counted once; wrong-direction and jitter rejection; all three closure conditions; the denominator invariants; deadline handling. |
| `test_scheduling_and_actuation.py` | Clocks; arming refusal; transport delay and lead time; lateness and dropping; the mandated `OFF, ON, OFF, OFF, ON` sequence; every actuator fault path; the watchdog. |
| `test_calibration.py` | The refusals; nominal optics; graticule recovery; the reducing-coupler trap; flow and transport estimation. |
| `test_pipeline_integration.py` | The whole chain; once-only counting; crop-track binding; ordering; replay determinism; camera disconnect; inference failure; frame drops. Marked `slow`. |

`conftest.py` holds fixtures; `builders.py` holds plain object builders
(`make_frame`, `make_detection`, `make_track`) imported directly — `tests/` has
no `__init__.py`, so pytest puts the directory on `sys.path`.

Markers: `slow`, `torch`, `web`, `hardware`, `dataset`. `--strict-markers` is
on, so a typo'd marker is an error.

**Fixtures use the product's own factories** wherever possible, so a test
exercises the real construction path. A fixture that quietly diverges from
production is a test that passes while the product is broken.

---

## 12. `scripts/` and the CLI

```bash
sperm-sorting run          -c CONFIG [-n N] [--video FILE] [--mode MODE]
sperm-sorting doctor       -c CONFIG      # environment, calibration, models, rule
sperm-sorting feasibility  -c CONFIG      # exit 1 if the budget does not close
sperm-sorting show-config  -c CONFIG [--json]
sperm-sorting generate-data --n 20000 --out data/
```

```bash
python scripts/calibrate_optics.py graticule.png --pitch-um 10 --coupler 1.0
python scripts/calibrate_transport_delay.py --imaging 0 1 2 3 --magnet 1.61 2.59 3.60 4.60 \
    --rise 8.2 --fall 6.4 --id kit-A-2026-08-04
python scripts/check_feasibility.py -c configs/device_v1.yaml --sweep-flow
```

Both calibration scripts print a **config block ready to paste**. The
`--sweep-flow` option shows the throughput/observation trade-off across flow
speeds and is the fastest way to find a workable operating point.

---

## 13. Recipes: how to do common things

### Add a detector

1. Subclass `Detector` in `detection/`, or `TorchDetectorBase` for a torch model
   (you then inherit device handling, tiling and checkpoint loading).
2. Return boxes in **source-frame pixels** — undo any internal resize yourself.
   Reuse `detection/preprocess.py` and `detection/postprocess.py`.
3. Register it in `detection/factory.py` and add the name to
   `DetectionConfig.architecture`'s `Literal`.
4. Verify: empty frame returns `[]`; boxes lie inside the frame; two builds of
   the same config give identical output.

### Add a tracker

Subclass `TrackerBase` in `tracking/_common.py` — that gives you the identity
guarantees for free. Implement only the association step. Register in
`tracking/factory.py`. Then run the six checks the existing trackers pass
(identity, no ID reuse, occlusion survival, crossings, empty frames, same-object
reference).

### Swap the morphology backbone

Add a builder to `morphology/backbones.py` returning `(module, feature_dim)`,
list it in `available_backbones()`, and extend
`MorphologyConfig.backbone`'s `Literal`. The four heads and the loss are
backbone-agnostic.

### Train a model

```bash
python training/train_morphology.py --config configs/training/morphology.yaml \
    --source mhsma --data-root data/mhsma --epochs 60
python training/eval_morphology.py --checkpoint runs/.../best.pt --split test
```

Point `morphology.weights` at `best.pt` in your runtime config. The
`calibration.json` sidecar next to it is picked up automatically.

### Calibrate a real instrument

1. `scripts/calibrate_optics.py` against an imaged stage micrometer — **with the
   coupler that will actually be used**.
2. `scripts/calibrate_transport_delay.py` with ≥3 tracer trials at the real
   flow rate.
3. Paste both config blocks into `configs/device_v1.yaml`.
4. `sperm-sorting feasibility -c configs/device_v1.yaml` — the decision latency
   must be **shorter** than the transport delay.
5. `sperm-sorting doctor -c configs/device_v1.yaml` should show both calibrated.

### Add a dataset

Subclass `DatasetAdapter` in `datasets/adapters/base.py`. Fill in `DatasetInfo`
with the licence and the **capture conditions** (magnification, contrast mode,
camera, FPS, resolution) — that metadata is what makes public weights honest.
Implement `validate()` returning a `ValidationReport`, and register in
`datasets/__init__.py`. For training, satisfy the narrow protocol in
`training/common/morphology_data.py` or `detection_data.py`.

### Change a threshold safely

Thresholds are versioned. Bump `MOTILITY_PROFILE_VERSION` in `constants.py` when
changing motility cut-points, so old audit logs stay interpretable. The decision
threshold and minimum are stamped onto every `ShotRecord` at decision time, so
they need no bump — but changing `ACCEPT_RATIO_THRESHOLD` changes the product,
and `tests/test_decision_rule.py` will tell you so.

### Read an audit log

```python
from sperm_sorting.monitoring.audit import read_events
events = read_events("runs/20260804-132916-synthetic/events.jsonl")
shots = [e for e in events if e["kind"] == "shot_decided"]
print(shots[0]["decision"]["rationale"])
print(shots[0]["shot"]["ineligibility_histogram"])
```

Every non-eligible member carries exactly one recorded reason, so any decision
can be reconstructed sperm by sperm.

---

## 14. Traps

Things that have already bitten, or that are one careless edit away.

**`LatestImageOnly` looks like the right grab strategy and is not.** It discards
frames under load. Tracking reconstructs trajectories from *consecutive*
observations, so a silently dropped frame fragments tracks and corrupts every
velocity derived from them. `OneByOne` is correct; a dropped frame must be
visible as a reported drop, not hidden in the driver.

**A fixed frame-count smoothing window is wrong.** Five frames is 100 ms at
50 FPS and 31 ms at 160 FPS. Use `vap_window_ms` and `vap_window_frames(fps)`.

**Resolutions arrive before their shot closes.** The gate sits at 85% of the
ROI, so a track crosses and exits while its shot is still filling. The first
implementation dropped those notifications, which would have emptied the
numerator of nearly every shot while leaving the denominator full — a systematic
bias toward REJECT with nothing raised. `ShotManager._resolved_early` handles it.

**Analysis latency must beat the transport delay.** If it does not, every
command is dropped as late and *nothing raises* — each component behaved
correctly in isolation. Run `sperm-sorting feasibility` after any change to shot
duration, morphology deadline, flow speed or magnet placement.

**A 0.5× reducing C-mount coupler halves µm/px.** Every velocity is then out by
exactly a factor of two, the images look fine, and nothing complains. The
optical calibration guard catches it; do not disable that check.

**Do not import `training/` from `datasets/`.** Training depends on datasets,
not the reverse — otherwise the adapters become unusable outside this
repository. When the two interfaces disagree, write the shim on the training
side (see `_VisemTrackingProtocolAdapter`).

**`datasets.adapters.visem` is not the detection dataset.** It is sample-level
VISEM with no bounding boxes. The detection dataset is `visem_tracking`. These
are different releases with different licences.

**`labels_ftid` puts the track id first, then the class.** Reversing them
silently swaps class for identity.

**Never flip label polarity a second time.** The one flip is
`polarity.flip_polarity`. If you find yourself writing `1 - p` anywhere near
morphology, stop.

**MHSMA's tail aspect has 7 abnormal examples in validation.** Any tail metric
computed there moves by 0.14 when one example changes side. Report the count
alongside the number, or do not report the number.

**The GPIO line's polarity matters for safety.** If the power stage is enabled
by a *low* level, a line that floats or is released on exit energises the
magnet. `gpio_active_high` must match the hardware.

---

## 15. Where to continue

In dependency order. The full account is in
[`docs/engineering_report.md`](docs/engineering_report.md).

1. **Calibrate a real instrument.** Nothing downstream is trustworthy until
   optics and transport delay are measured, and the software refuses to actuate
   regardless.
2. **Check the timing budget on the real geometry.** If decision latency exceeds
   transport delay, that is a mechanical problem — no model work fixes it.
3. **Move to a GPU and re-measure.** Detection is 97× over budget on CPU
   (603 ms vs 6.25 ms at 1920 × 1200). Optimise the **quality gate** (12.6 ms
   p50) and **best-frame selection** (20.9 ms p50) first — they cost more than
   the models do, and both are straightforwardly optimisable (downsampled focus
   metrics, incremental scoring as frames arrive, scanning the box rather than
   the frame).
4. **Train the detector** on VISEM-Tracking, split by video. Expect low absolute
   numbers: the published YOLOv5l baseline reaches mAP@0.5 = 0.2231.
5. **Train morphology** on MHSMA, preserving the official split, thresholds by
   Youden's J on validation.
6. **Capture and annotate device video.** This matters most. Every public
   dataset is 400× phase contrast or a stained smear; none resembles a 100× oil
   brightfield frame. Public weights are a starting point for fine-tuning and
   nothing more.
7. **Measure the product, not the parts.** Run `training/eval_pipeline.py`
   before and after every model change. A detector that improves AP while
   degrading the shot decision is not an improvement.
8. **Bench the hardware paths** — camera, GPIO, serial — against the mock
   actuator's recorded state sequences.

### Known gaps

- No model is trained; no accuracy figure appears anywhere in this repository.
- `BaslerFrameSource`, `GpioActuator`, `SerialActuator` are written against
  verified APIs but have never touched hardware.
- No public dataset has been downloaded; adapters are verified against synthetic
  fixtures shaped to the documented formats.
- BoT-SORT with shipped defaults *is* ByteTrack, because no ReID model exists.
  The fusion path is real and was verified with a dummy embedder.
- `top_k_frames > 1` has its data structures in place but the pipeline uses k=1.
- Self-supervised pretraining on unlabelled device video is designed for in the
  adapter layer but not implemented.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **ai_eligible** | A sperm satisfying the full combined rule. Deliberately *not* "healthy" — the system observes phenotype only. |
| **Shot** | A software-defined segment of continuous flow, ~25 trackable sperm, ≤1 s. One independent decision unit. |
| **Gate** | The virtual line, downstream in the ROI, that a track crosses exactly once to join a shot. |
| **Trackable count** | The denominator: every valid trackable sperm in the shot. |
| **Track quality** | The bar a track must clear to be counted at all. Judged once, at the gate, and final. |
| **VCL / VSL / VAP** | Curvilinear / straight-line / average-path velocity (CASA). |
| **LIN / STR / WOB** | VSL/VCL, VSL/VAP, VAP/VCL. |
| **ALH / BCF** | Lateral head amplitude; path-crossing rate. Both algorithm-dependent; BCF does **not** correlate with flagellar beat frequency (WHO §4.5.1.4). |
| **PR / NP / IM** | WHO progressive (rapid + slow) / non-progressive / immotile. |
| **Transport delay** | Time for fluid to travel from the imaging region to the magnet. |
| **Provenance** | `public-research-baseline` / `synthetic-bootstrap` / `device-finetuned`. Stamped into every audit log. |
| **Oracle detector** | Returns simulator ground truth with controllable noise, so pipeline tests fail on pipeline bugs rather than on untrained models. |

---

## A closing note on claims

This is a research prototype. It analyses visible phenotype and nothing else. It
does **not** measure DNA integrity, phosphatidylserine exposure, Annexin V
binding, apoptosis, magnetic labelling, fertility potential or pregnancy rate.

Those are not merely unmeasured — several are **unmeasurable by this method**.
The correlations between morphology/motility and DNA fragmentation are weak
(|r| ≈ 0.3), they are *between men* rather than per-cell, and every confirmatory
assay is destructive or label-dependent, so the cell you assay is never the cell
you use. A per-cell visual model cannot even be ground-truthed against
Annexin-V on the same spermatozoon.

Before writing anything user-facing — a README, a paper, a slide, a product page
— read [`docs/safety_and_claims.md`](docs/safety_and_claims.md). The regulatory
position turns on *claims*, not on technology, and a research-use-only label
does not protect a project whose documentation asserts a diagnostic claim.
