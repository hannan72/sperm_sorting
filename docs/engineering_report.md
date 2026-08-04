# Engineering report

What was built, what was measured, what is not done, and what should happen
next. Every number here comes from a run on the machine described below. No
figure in this document is estimated, extrapolated or aspirational — where a
quantity has not been measured, it says so.

**Date:** 2026-08-04
**Machine:** Linux 7.0.0-28-generic, x86-64, Python 3.13.5, PyTorch 2.13.0+cpu,
OpenCV 5.0.0, **no CUDA**. An RTX 2070 (8 GB) is present in the host but the
CPU-only PyTorch build was used, so every latency below is a CPU figure.

---

## 1. What is complete

| Subsystem | State | Verified by |
|---|---|---|
| Schemas, config, errors | complete | 176 tests; all four configs load and cross-validate |
| Acquisition (Basler, video, synthetic) | complete | synthetic and video exercised end to end; Basler is **unexercised — no camera** |
| Preprocessing and quality gate | complete | ROI translation, normalisation modes, verdict logic, bounded memory over 2000 frames |
| Detection (TOD-CNN-inspired, P2Net, ONNX, oracle) | complete, **untrained** | target↔decode round-trip to 1e-4; ONNX matches torch to 1e-2 px |
| Tracking (ByteTrack, OC-SORT, BoT-SORT) | complete | identity, no-ID-reuse, occlusion, crossing, empty frames — six checks × three trackers |
| Motion (CASA kinematics, flow correction, WHO grading) | complete | known-velocity, flow-cancellation, circular swimmer, threshold boundaries, ALH/BCF gating |
| Best-frame selection and cropping | complete | sharp-beats-blurred, isolated-beats-overlapped, aspect preservation, track-identity binding |
| Morphology (shared backbone, four heads, calibration) | complete, **untrained** | polarity round-trip, threshold fitting on 5%-positive data, temperature scaling, checkpoint guards |
| Shot assembly and the counting gate | complete | once-only counting, all three closure conditions, denominator invariants |
| Decision rule | complete | all five mandated cases; every exactly-60% pair from d=20..200 |
| Scheduling and actuation | complete | transport delay, lead time, lateness, dropping, the mandated OFF/ON/OFF/OFF/ON sequence |
| Watchdog and fail-safe | complete | trips, recovers, forces FIELD_OFF; verified across inference failure and camera disconnect |
| Calibration (optics, flow, transport) | complete | 0.66% error recovering 0.0345 µm/px from a synthetic graticule; reducing-coupler trap caught |
| Simulator | complete | determinism, causal labels, motility contract, scene ground truth |
| Audit logging and metrics | complete | manifest, events, tracks, metrics, summary; replay determinism |
| Web demo | complete | 43 API tests; all five mandated cases served from the real rule |
| Dataset adapters | complete, **no data present** | synthetic fixtures; the MHSMA polarity check demonstrated on deliberately inverted labels |
| Training and evaluation scripts | complete, **no training run to convergence** | smoke-trained on synthetic data only |

**Test suite: 176 passing, 1 skipped, 0 failing.**

---

## 2. What was measured

### 2.1 Component latency (CPU, no CUDA)

At 160 FPS the per-frame budget is **6.25 ms**.

| Component | Resolution | Mean | p95 | vs budget |
|---|---|---|---|---|
| Detector P2Net | 640 × 400 | 54.2 ms | 57.2 ms | 8.7× over |
| Detector P2Net | 1920 × 1200 | 603.4 ms | 615.5 ms | **97× over** |
| Detector TOD-CNN-inspired | 640 × 400 | 232.6 ms | 250.2 ms | 37× over |
| Detector TOD-CNN-inspired | 1920 × 1200 | 2644.6 ms | 2641.3 ms | **423× over** |
| Morphology, single crop | 128 × 128 | 5.2 ms | 5.8 ms | within budget |
| Morphology, batch of 16 | 128 × 128 | 17.3 ms | 18.6 ms | 1.08 ms/crop |
| Tracker, 30 objects | — | 2.15 ms | 2.47 ms | **within budget** |

The TOD-CNN-inspired detector is ~4.4× slower than P2Net despite having fewer
parameters. That is inherent to the architecture rather than a defect: it never
downsamples below stride 4, so every convolution runs at high resolution. It is
retained as the specified baseline for comparison, not as a production
candidate.

### 2.2 End-to-end pipeline (synthetic source, oracle detector)

640 × 400, 240 frames, density 10, flow 6400 px/s:

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| preprocess | 0.01 ms | 0.01 ms | 0.01 ms |
| quality gate | 12.64 ms | 15.83 ms | 20.77 ms |
| detect (oracle) | 0.35 ms | 0.50 ms | 0.59 ms |
| track | 0.93 ms | 1.39 ms | 1.59 ms |
| flow estimate | <0.01 ms | <0.01 ms | <0.01 ms |
| gate | 0.01 ms | 0.04 ms | 0.05 ms |
| motion | 0.37 ms | 0.55 ms | 0.58 ms |
| best frame | 20.85 ms | 37.29 ms | 57.68 ms |
| crop | 1.51 ms | 2.75 ms | 2.79 ms |
| morphology | 5.53 ms | 6.00 ms | 11.69 ms |
| decision | 0.05 ms | 0.10 ms | 0.10 ms |

Achieved throughput at 1920 × 1200 was **~6 FPS** against a 160 FPS target.

One consequence of that shortfall is visible in the actuation figures and is
worth naming so it is not mistaken for a scheduling defect. Commands are
dispatched from the per-frame poll, so the dispatch granularity is the frame
period. At the achieved ~5.5 FPS that period is ~180 ms, far beyond the 5 ms
late tolerance, so `eval_pipeline.py` reports a handful of commands as LATE
while dropping none. On hardware running at 160 FPS the poll interval is
6.25 ms and the effect disappears. It is a symptom of CPU throughput, not of
the scheduler.

A second figure needs the same caveat: `eval_pipeline.py` reports a few
`FIELD_ON not delivered` on any finite run. Those are the last shots, whose
activation instants fall past the end of the recording — with a 1600 ms
transport delay, roughly the final 1.6 s of decisions have nowhere to land.
Over a 700-frame run (4.4 s, 10 shots) it was 2; over 240 frames it was 3. The
count should fall to zero on a continuous run and is worth watching as a
regression signal, but it is expected behaviour at the tail.

The two dominant costs are not the models. **Best-frame selection (20.9 ms p50,
57.7 ms p99)** and **the quality gate (12.6 ms p50)** together account for more
of the budget than detection and morphology combined. Both are per-pixel
OpenCV work on full frames, and both are straightforwardly optimisable —
downsampled focus metrics, incremental scoring as frames arrive rather than a
sweep at track end, and restricting the crop-quality scan to the box rather
than the frame. Neither has been optimised, because correctness came first and
optimising before a GPU is in the loop would be optimising the wrong thing.

### 2.3 Optical budget (computed from verified specifications)

| Quantity | Value |
|---|---|
| Sample-plane sampling | 0.0345 µm/px |
| Abbe limit at NA 1.25, 550 nm | 0.220 µm |
| Rayleigh limit | 0.268 µm (matches the objective's published 0.27 µm) |
| Nyquist oversampling | 3.89× |
| Field of view at 1920 × 1200 | 66.24 × 41.40 µm |
| Sperm head (4.1 × 2.8 µm) | 119 × 81 px |
| Whole spermatozoon (~53 µm) | 1539 px |
| Motion blur, 100 µm/s at 19 µs | 0.055 px |
| Motion blur, 100 µm/s at 1 ms | 2.90 px |
| Data rate, Mono8 at 164 FPS | ~378 MB/s |

### 2.4 Calibration accuracy

Optical calibration recovered 0.03473 µm/px from a synthetic 10 µm graticule
against a true 0.0345 — **0.66% error**, with 6 rulings in the field. The
reducing-coupler guard correctly rejected a measurement 2.00× from nominal.

---

## 3. Three findings that changed the design

These were not anticipated from the specification; they emerged from doing the
arithmetic and from running the system.

### 3.1 A whole sperm does not fit in the field of view

At 100× with a 3.45 µm pixel, the field is 66.2 × 41.4 µm while a spermatozoon
is 50–60 µm long. It fits along the frame only when favourably oriented and
never fits across it.

**Consequence:** the detection target is the sperm *head*, not the whole cell.
This turns out to align with both CASA (kinematics are defined on the head
centroid) and MHSMA (whose crops are head-centred with the tail not entirely
visible), so it is not a compromise so much as a correction. The residual cost
is that tail morphology is judged from a partial tail; the code reports
`tail_complete=False` rather than concealing it, and the tail aspect should be
expected to be the least reliable of the four — it is also the rarest, at 4.6%
abnormal prevalence in MHSMA, with only 7 abnormal examples in the entire
validation split.

### 3.2 Analysis latency must beat the transport delay

The first full pipeline run produced correct decisions and **dispatched none of
them**: all four field commands were dropped as late. Nothing raised. Every
component had behaved correctly in isolation.

The worst-case interval from a sperm crossing the counting gate to its shot's
decision being available is

```
shot assembly (up to 1000 ms, the duration limit)
+ field transit  (one residence time, 200 ms at the reference flow)
+ morphology     (250 ms finalisation deadline)
= 1450 ms
```

and the transport delay must exceed it. The configuration under test had 120 ms.

**Consequence:** `assess_feasibility` now computes this budget and reports it,
`sperm-sorting feasibility` exposes it, and `tests/test_pipeline_integration.py`
asserts that an infeasible timing is detected. The synthetic configuration was
changed to a 1600 ms transport delay, which at the reference flow corresponds to
roughly 530 µm of channel between the imaging region and the magnet.

This is a **hard constraint on the physical build**, not a software parameter.
If the magnet cannot be placed far enough downstream, then either the shot
duration limit or the morphology deadline has to come down, and both trade
against decision quality.

### 3.3 The training harness reached for the wrong VISEM

`training/common/detection_data.py` imported `datasets.adapters.visem` to load
detector training data. That is the **sample-level** VISEM adapter, which
carries WHO percentages per participant and no bounding boxes at all — training
a detector on it is not merely unsupported but meaningless. The detection
dataset is `visem_tracking`, a different release with a different licence.

The two packages were written against each other's documented interfaces before
either existed, which is what let the mismatch survive: the harness's protocol
(`video_ids()`, `load_video()`) and the adapter's API (`videos()`,
`iter_frames()`) were both reasonable and simply different. Fixed by pointing
at the right module and adding `_VisemTrackingProtocolAdapter`, a shim on the
*training* side — training depends on datasets, and putting the translation the
other way would have made the dataset adapters unusable outside this repository.
`MhsmaAdapter` gained the two protocol methods directly, expressed in terms of
its existing accessors.

Both paths are now exercised end to end against fixtures shaped to the
documented on-disk formats. The YOLO-to-pixel conversion is exact and
`labels_ftid` parses track id first and class second, which is the field order
that would silently swap class for identity if reversed.

### 3.4 Resolutions arrive before their shot closes

The counting gate sits at 85% of the ROI, so a track crosses it and leaves the
field shortly afterwards — while the shot it joined is usually still filling
toward 25 members. Morphology resolutions therefore arrive *before* there is a
pending shot to record them against.

The first implementation dropped those notifications. Every such member would
have been counted as a deadline miss, emptying the numerator of nearly every
shot while leaving the denominator full — a systematic bias toward REJECT, with
no error anywhere. Caught by `test_shots.py`; fixed by holding early
resolutions until the shot closes.

---

## 4. What is not done

### 4.1 No model is trained

The detectors and the morphology network are implemented and verified for
shape, geometry, determinism and numerical correctness. **No weights exist.**
No accuracy, AP, HOTA, sensitivity or specificity figure appears anywhere in
this repository, because any such figure would be invented.

An untrained detector at the default score threshold returns zero detections —
correct behaviour, since the head is initialised to the focal-loss prior of
0.01. Every pipeline test therefore uses the oracle detector, so that a failure
indicates a broken *pipeline* rather than an untrained model. Both the detector
and morphology constructors emit a warning when no weights are configured.

### 4.2 CPU cannot run this in real time

Detection is 97× over budget at full resolution on CPU. This is a GPU workload;
the architectures assume one. Nothing here has been profiled or optimised on a
GPU, and no TensorRT engine has been built or measured.

### 4.3 Nothing physical is calibrated

Transport delay, field rise and fall times, micrometres-per-pixel, the bulk
flow vector and the chamber depth are all unmeasured. They are `None` or
`calibrated: false`, the scheduler refuses to arm, and motility grading returns
`UNDETERMINED` rather than comparing µm/s thresholds against pixel values. The
calibration utilities are implemented and tested against synthetic inputs;
none has been run against a real instrument.

### 4.4 The hardware paths are unexercised

`BaslerFrameSource`, `GpioActuator` and `SerialActuator` are written against
the documented APIs — pypylon's grab-result and chunk-data interfaces were
verified by inspecting the actual wheels rather than from memory — but no
camera, GPIO line or serial device was available. They are excluded from
coverage and should be treated as unproven until run on the bench.

Two version hazards are worth flagging: `pylon.FirstFound` exists in pypylon
26.x but **not** in 4.2.0, so the portable `CreateFirstDevice()` form is used;
and `GrabResult.GetTimeStamp()` returns zero on cameras without the feature, so
zero is treated as "unsupported" rather than as time zero.

### 4.5 No real data has been touched

None of the five public datasets was downloaded. The adapters are verified
against synthetic fixtures shaped to the documented formats. The MHSMA polarity
guard was demonstrated by feeding it deliberately inverted labels and
confirming it raises.

### 4.6 Known gaps in scope

* The VISEM-Tracking-graphs GNN extension is implemented as an optional
  adapter, including a corrected video-graph rebuild, but no graph-based model
  was trained. The MVP does not depend on it, as specified.
* Self-supervised pretraining on unlabelled device video is designed for in the
  adapter layer but not implemented.
* Multi-frame morphology aggregation (`top_k_frames > 1`) has its data
  structures in place but the pipeline uses k=1.
* BoT-SORT with the shipped defaults is exactly ByteTrack, because no ReID
  model exists. The fusion path is real and was verified with a dummy
  embedder, but there is nothing to embed with.

---

## 5. Honest notes on the specification

Three points where this implementation deviates from, or adds to, the brief —
each deliberate and each documented in place.

1. **The ~60 Hz CASA frame-rate requirement is not a WHO requirement.** The WHO
   6th-edition manual specifies no numeric minimum frame rate; the figure comes
   from Mortimer et al. (2015). The code attributes it correctly.

2. **The VAP smoothing window is specified in milliseconds, not frames.** A
   fixed frame count smooths over different durations at different frame rates,
   which Mortimer et al. show produces aberrant ALH. On one synthetic
   trajectory here, a 100 ms window and a nominal five-frame window differed by
   **15×** in ALH against a known true amplitude. Five frames is 100 ms at 50
   FPS but 31 ms at 160 FPS.

3. **`min_lin_for_progressive` is stricter than WHO**, whose wording admits
   progression "either linearly or in a large circle". It is documented as this
   implementation's choice and can be set to 0.0 to follow WHO literally.

One correction to a widely-held assumption, since it affects anyone reading the
same sources: **the WHO 6th edition reinstated the four-category a/b/c/d
motility grading.** PR/NP/IM was the 5th edition. The four `MotilityClass`
members here *are* WHO 6th ed. §2.4.6.1, and 25 and 5 µm/s are WHO's own
approximate limits.

---

## 6. Next steps, in dependency order

1. **Calibrate the instrument.** Optics against a stage micrometer; transport
   delay against a tracer bolus with at least three trials; field rise and fall
   with a Hall probe. Nothing downstream is trustworthy until this is done, and
   the software will refuse to actuate regardless.
2. **Check the timing budget on the real geometry.** Run
   `sperm-sorting feasibility` with the measured values. If the decision
   latency exceeds the transport delay, that is a mechanical problem and no
   amount of model work will fix it.
3. **Move to a GPU and re-measure.** Everything in §2.1 should be repeated with
   CUDA, then with ONNX Runtime, then with TensorRT. Optimise the quality gate
   and best-frame selection first — they cost more than the models do.
4. **Train the detector on VISEM-Tracking**, split by video, and evaluate with
   `training/eval_detector.py`. Expect low absolute numbers: the published
   YOLOv5l baseline on this dataset reaches mAP@0.5 = 0.2231.
5. **Train morphology on MHSMA**, preserving the official split, and fit
   per-aspect thresholds by Youden's J on the validation split. Report the tail
   aspect with its positive count alongside, because n=7 in validation cannot
   support a confident number.
6. **Capture and annotate device video.** This is the step that matters most.
   Every public dataset is 400× phase contrast or a stained smear; none
   resembles a 100× oil brightfield frame. Public weights are a starting point
   for fine-tuning and nothing more.
7. **Measure the product, not the parts.** `training/eval_pipeline.py` scores
   end-to-end eligibility agreement, shot-ratio error and the decision
   confusion matrix against simulator ground truth. Run it before and after
   every model change; a detector that improves AP while degrading the shot
   decision is not an improvement.
8. **Bench the hardware paths** — camera, GPIO and serial — with the mock
   actuator's recorded state sequences as the reference.

---

## 7. Reproducing everything in this report

```bash
pip install -e ".[all]"
pytest -q                                              # 176 passed, 1 skipped
python -m sperm_sorting.cli doctor                     # environment and calibration state
python scripts/check_feasibility.py -c configs/synthetic.yaml --sweep-flow
python -m sperm_sorting.cli run -c configs/synthetic.yaml -n 500
```

The latency table in §2.1 comes from timing `build_detector(...).detect(frame)`
and `build_morphology_engine(...).infer_batch(crops)` directly, with two warmup
iterations discarded. The stage table in §2.2 is `RuntimeMetrics.format_summary()`
from a real run, not a separate measurement.
