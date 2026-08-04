# `training/` — training and evaluation

Six entry points and the machinery they share. Everything here imports **from**
`src/sperm_sorting` and nothing in `src/` imports from here: the runtime package
must stay installable on a device that will never train anything.

Two rules govern this directory:

1. **A result that cannot be reproduced is not a result.** Every script writes
   `experiment.json` with the git commit, the fully-resolved config, package
   versions, dataset name + licence + split sizes, the seed and what it actually
   set, the hardware, start/end times and the final metrics. It is a mandatory
   output, written even on the failure path.
2. **No number in this file is invented.** Every figure below came out of a run
   on the machine described in the corresponding `experiment.json`. Where a
   quantity has not been measured, this document says so instead of guessing.

---

## Running the scripts

Both invocation styles work:

```bash
python training/train_morphology.py --help
python -m training.train_morphology --help
```

Every script takes the same configuration pair, identical to
`sperm_sorting.cli`:

| flag | meaning |
|---|---|
| `-c, --config FILE` | YAML config; omit for the built-in defaults |
| `-s, --set a.b.c=v` | override one config field; repeatable, later wins |
| `-o, --out DIR` | where checkpoints, metrics, plots and `experiment.json` go |
| `--resume CKPT` | resume a training run (model + optimiser + scheduler + scaler + epoch + best-metric + patience) |
| `--device auto\|cpu\|cuda[:N]` | `auto` picks CUDA when present; naming an absent CUDA device is an **error**, not a silent fallback |
| `--seed N` | overrides `run.seed` |
| `--non-deterministic` | allow cuDNN autotuning (faster on CUDA, no longer bit-reproducible) |

Everything runs on CPU. AMP is requested with `--amp` and is honoured **only on
CUDA**; on CPU it reports itself as ignored rather than pretending, because a
latency figure recorded under `amp=true` that was actually fp32 is a fabricated
measurement.

---

## `train_morphology.py`

Trains `MultiTaskMorphologyNet` — one trunk, four independent binary heads.

**Needs:** MHSMA via `datasets.adapters.mhsma` (`--source mhsma --data-root …`),
or nothing at all (`--source synthetic`, which renders labelled crops with the
in-repo simulator).

**Produces:** `best.pt`, `last.pt`, `calibration.json`, `metrics.json`,
`metrics.jsonl`, `experiment.json`, `plots/`, `tensorboard/`.

```bash
# bootstrap smoke test, CPU
python training/train_morphology.py --source synthetic --epochs 2 \
    --n-train 240 --n-valid 120 --n-test 120 --image-size 64 --batch-size 16 \
    -s morphology.backbone=simplecnn \
    -s crop.output_size='[64,64]' -s morphology.input_size='[64,64]' \
    -o runs/morph_smoke

# real data, MHSMA official split preserved exactly
python training/train_morphology.py --source mhsma --data-root data/mhsma \
    --epochs 60 --batch-size 32 -o runs/morph_mhsma

# resume
python training/train_morphology.py --source mhsma --data-root data/mhsma \
    --epochs 60 -o runs/morph_mhsma --resume runs/morph_mhsma/last.pt
```

**Measured runtime.** The smoke command above: **12 s wall clock** total on
this repo's CPU box (2 epochs, 240 training crops, 64 px, `simplecnn`), of which
~3 s is training and the rest is data generation, calibration and plotting.
Runtime scales with (crops × epochs × backbone cost); no MHSMA timing is quoted
here because no MHSMA run has been done in this checkout.

**What it is careful about**

- **Polarity.** The network emits a logit for `P(abnormal)`, so the MHSMA label
  is the training target verbatim. There is no `1 - y` in the file. The one
  permitted flip lives in `morphology/polarity.py` and is applied only by the
  inference adapter. `save_checkpoint` stamps the convention string and
  `load_checkpoint` refuses a mismatch.
- **Per-aspect `pos_weight`,** computed from the training split's own
  prevalence. Verified MHSMA train prevalences are acrosome 30.1 %, head 27.3 %,
  vacuole 17.0 %, tail **4.6 %**, so the tail head needs a weight near 20 while
  acrosome needs about 2.3. One shared weight would either ignore the tail or
  drown the acrosome.
- **Raw accuracy is never computed, logged, or selected on.** A model that calls
  every tail normal scores 95.4 % and is worthless. Selection is macro-F1
  (default), balanced accuracy, MCC, ROC-AUC, PR-AUC or validation loss.
- **Validation thresholds are re-fitted each epoch by Youden's J**, not left at
  0.5 — at 4.6 % prevalence a 0.5 threshold predicts "normal" for every tail and
  turns macro-F1 into a constant. This is mildly optimistic (the threshold sees
  the split it is scored on); it is stated in the output, and the threshold-free
  ROC-AUC and PR-AUC are printed beside it as a check.
- **Calibration is fitted on validation, never on test.** After training, the
  best checkpoint is reloaded, validation logits recomputed, and one temperature
  + one threshold per aspect are fitted and written to `calibration.json` — the
  filename `MorphologyEngine.find_calibration_sidecar` looks for.
- **Provenance** is stamped from `constants.py`: `public-research-baseline` for
  MHSMA, `synthetic-bootstrap` for simulator data.
- **Augmentation is label-preserving only** — small rotations (±15°), flips,
  mild brightness/contrast (±10 %), slight blur (σ ≤ 0.6 px). Each justification
  is in `training/common/augment.py`. Excluded on purpose: elastic deformation
  (changes head axis ratio = the head label), aggressive scaling (changes head
  size = the head label), cutout over the head (erases the acrosome/vacuole
  evidence the label asserts), colour jitter (mono sensor), mixup/cutmix
  (undefined for a conjunctive rule).

**Low-positive warning.** Any aspect with fewer than 20 positives in the
evaluated split is flagged by name and count, on the console and in the JSON.
The MHSMA validation split has 7 abnormal tails out of 240; the smoke run above
had 4 out of 120 and printed:

```
WARNING: aspect 'tail' has only 4 abnormal example(s) out of 120 in the
validation split. ... one example changing side moves sensitivity by 0.25.
```

---

## `train_detector.py`

Trains `P2Net` or `TodCnnNet` against the CenterNet targets from
`detection/heads.py`.

**Needs:** VISEM-Tracking via `datasets.adapters.visem` (`--source visem`), or
nothing (`--source synthetic`).

**Produces:** `best.pt`, `last.pt`, `metrics.json`, `metrics.jsonl`,
`experiment.json`, `plots/`, `tensorboard/`.

```bash
python training/train_detector.py --source synthetic --epochs 2 \
    --n-clips 6 --frames-per-clip 4 --frame-width 192 --frame-height 160 \
    --batch-size 2 --width 8 --head-channels 16 --fpn-channels 16 \
    --warmup-steps 5 -s detection.architecture=todcnn -o runs/det_smoke
```

**Measured runtime.** The command above: **6 s wall clock** on this CPU box
(2 epochs, 16 training frames at 192×160, `width=8`). Nothing larger has been
timed in this checkout.

**What it is careful about**

- **Splits are by video, and it is enforced.**
  `assert_no_video_leakage` runs before the first optimiser step and **raises**
  on any overlap, duplicate, or empty split. It does not warn: the failure it
  guards makes the metric look *better*, so nothing downstream would flag it.
  It defers to `datasets.validators` when that module ships a validator, and
  falls back to its own real check otherwise. Synthetic clips are one clip = one
  video.
- **Warmup is not decoration.** CenterNet's penalty-reduced focal loss divides
  by the number of true centres; in the first few hundred steps the heatmap head
  is still at its `prior_prob` bias and the softplus size head predicts
  near-zero widths, and at a normal LR the run diverges inside the first epoch.
  `--warmup-steps` defaults to 500. Gradient clipping (`--clip-grad-norm 5.0`)
  is on for the same reason.
- **Grayscale in, no resize.** Frames are padded to the network's
  `size_divisor` with their own median; box coordinates are untouched. A sperm
  head is a handful of pixels, so resizing either destroys it or fabricates
  scale variation the optics cannot produce.
- **Augmentation:** flips, ≤10° rotation, mild brightness/contrast. **No colour
  jitter** (mono sensor), **no scale augmentation** (the object scale
  distribution is effectively a point — that is the premise of both
  architectures).
- **Validation goes through the deployment path** — the same decode, NMS and
  box-size filtering the runtime uses — so a target/decode disagreement shows up
  as bad AP rather than hiding behind a good loss.
- Checkpoints are **self-describing**: `architecture` and `arch_kwargs` travel
  with the weights, and `eval_detector.py` rebuilds the exact geometry from them
  rather than from config defaults.

---

## `eval_morphology.py`

Loads a checkpoint plus its calibration bundle and evaluates one split.

```bash
python training/eval_morphology.py --checkpoint runs/morph_smoke/best.pt \
    --split test --source synthetic --n-test 120 --image-size 64 \
    -s morphology.backbone=simplecnn \
    -s crop.output_size='[64,64]' -s morphology.input_size='[64,64]' \
    -o runs/morph_smoke/eval_test
```

**Produces:** `eval_<split>.json`, `experiment.json`, `plots/`. **Runtime:**
seconds for a few hundred crops on CPU.

Reports per aspect: sensitivity, specificity, precision, NPV, macro-F1, balanced
accuracy, MCC, ROC-AUC, PR-AUC, ECE, MCE and the 2×2 confusion matrix; plus a
macro row; plus the **all-four-normal joint accuracy** printed in its own table
next to the mean per-aspect accuracy, because the runtime rule
(`MorphologyResult.all_four_normal`) is a conjunction and the four error rates
compound — the joint number is *not* the average of the four.

Without a calibration bundle it falls back to the raw sigmoid at 0.5 and says so
loudly in both the console output and the JSON; that is not a shippable
operating point on the low-prevalence aspects.

---

## `eval_detector.py`

```bash
# score a checkpoint
python training/eval_detector.py --checkpoint runs/det_smoke/best.pt \
    --source synthetic --split test -s detection.architecture=todcnn \
    -o runs/det_smoke/eval

# score hand-built or externally produced boxes, no model involved
python training/eval_detector.py --ground-truth gt.json --predictions pred.json \
    -o runs/det/eval_json
```

**Produces:** `eval_detector.json`, `experiment.json`. **Runtime:** dominated by
detector inference; the JSON-scoring mode is instantaneous.

Reports AP50, AP75, mAP50-95, precision/recall/F1 at a fixed score threshold,
**small-object recall** (`--small-area-px`, default 1024 px² = COCO's 32×32
convention), **debris false-positive rate**, **counting error** (predicted minus
true object count per frame — signed mean first, because the shot denominator is
a count and a constant offset biases every accept ratio the same way), latency
percentiles and peak memory.

**AP is implemented here; `pycocotools` is not required and not used.** Global
score ranking, per-frame greedy IoU matching with ground truth consumed on
match, 101-point recall grid with a precision envelope, mAP over IoU
0.50:0.05:0.95.

The debris FP rate reports `available: false` **with a reason** when the ground
truth carries no non-sperm annotations. Zero would be a claim; "no debris was
annotated" is the fact.

---

## `eval_tracking.py`

```bash
python training/eval_tracking.py --ground-truth gt_tracks.json \
    --predictions pred_tracks.json -o runs/track/eval

# or run the configured detector + tracker over synthetic clips
python training/eval_tracking.py --source synthetic --split test \
    -c configs/synthetic.yaml -o runs/track/eval_synth
```

**Produces:** `eval_tracking.json`, `experiment.json`. **Runtime:**
sub-second for the hand-built fixtures below; the HOTA sweep is 19 α values ×
frames, so it is linear in run length.

Reports HOTA (with DetA / AssA / LocA and the full per-α breakdown), IDF1 /
IDP / IDR, MOTA, MOTP, ID switches, fragmentation, plus **duplicate-count rate**
and **track survival length**.

### Which HOTA

**Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object
Tracking", IJCV 2021, as specified in the paper and realised in TrackEval's
`HOTA` class.** A global alignment score `A(g,p)` is built from the
Jaccard-normalised per-frame similarity; per-frame matching is a Hungarian
assignment maximising `A(g,p) · IoU` with an `IoU ≥ α` acceptance test;
`DetA = TP/(TP+FN+FP)`; `AssA = mean over matched detections of
TPA/(TPA+FNA+FPA)`; `HOTA_α = √(DetA_α · AssA_α)`; α ∈ {0.05, 0.10, …, 0.95};
`HOTA = mean_α HOTA_α` — **not** `√(mean DetA · mean AssA)`, which is a
different number. `TrackEval` is not a dependency. `motmetrics` is not either;
`--cross-check` will use it if installed and print both values side by side.

### Duplicate-count rate

`ShotRecord.add_track` refuses a repeated track *id*, but it cannot see that two
ids are the same cell — and the 60 % rule is a ratio of counts. Three numbers:
`duplicate_track_rate` (fraction of GT tracks covered by >1 predicted id),
`excess_id_ratio` (the count inflation the denominator actually sees) and
`phantom_track_rate` (predicted tracks matching no GT at all).

---

## `eval_pipeline.py`

**The one that answers "does the product work", as opposed to "does each model
work".**

```bash
python training/eval_pipeline.py -c configs/synthetic.yaml \
    -s run.max_frames=400 -o runs/pipeline_eval
```

**Needs:** a frame source publishing `FramePacket.meta["gt_detections"]` and
`meta["gt_states"]` — i.e. the synthetic simulator, the only source where
per-sperm ground truth exists (VISEM-Tracking has boxes and identities but no
morphology; MHSMA has morphology but no video; VISEM has only sample-level
percentages).

**Produces:** `eval_pipeline.json`, `experiment.json`.

**Measured runtime.** 400 frames at 640×400 with the oracle detector,
ByteTrack and an untrained morphology net: **34 s wall clock** on this CPU box.
Runtime is roughly linear in frames and superlinear in scene density.

Reports:

- **per-sperm eligibility agreement** — predicted `TrackRecord.ai_eligible` vs
  truth, for every track actually gated into a shot, with the confusion matrix,
  MCC, and the breakdown of `IneligibilityReason` against the disagreements;
- **shot-ratio error** — predicted `ai_eligible_ratio` vs the ratio computed
  from the true states of that shot's own members (signed mean first);
- **shot-decision confusion matrix** — ACCEPT / REJECT / INDETERMINATE against
  the decision the same rule gives on the true ratio, using
  `shot.exceeds_threshold` so the reference decision matches the product's on
  the exactly-60 % boundary;
- **indeterminate rate**, predicted and true (they have different fixes: a high
  true rate means the optics and flow cannot deliver 20 trackable sperm; a high
  predicted rate with a low true one means the pipeline is losing tracks);
- **command-alignment error** — did the field command match what the
  truth-derived decision required, *and* was it delivered. These are counted
  separately: a correct decision that reaches the actuator after the fluid has
  passed the magnet has the same physical outcome as a wrong one.

It builds on `runtime/pipeline.py` and `app.py` — `Application.setup()` wires
every real component — but drives `Pipeline.process_frame` directly instead of
using `PipelineRunner`, so the run is deterministic and the ground truth can be
read off each packet as it goes past.

Reminder printed by the script: **FIELD_ON is the rejection.** Energising the
magnet diverts the labelled population toward waste.

---

## `training/common/`

| module | what it owns |
|---|---|
| `args.py` | the `--config` / `--set` / `--out` / `--resume` / `--device` / `--seed` pattern; device resolution; atomic JSON writes |
| `seeding.py` | `seed_everything(seed, deterministic)` → a dict of what it actually set; DataLoader worker seeding |
| `checkpoints.py` | `last.pt` / `best.pt` as supersets of the deployable format; genuine resume of optimiser + scheduler + scaler + epoch + best-metric + patience |
| `earlystop.py` | `EarlyStopping(patience, mode, min_delta)`, sharing one comparison with the checkpoint manager so "best epoch" is never ambiguous |
| `logging_utils.py` | TensorBoard wrapper that degrades to a recorded no-op, JSONL metric log, fixed-width console table |
| `experiment.py` | `experiment.json` — git, config, versions, dataset + licence + splits, seed, hardware, timings, metrics |
| `plots.py` | confusion matrix, reliability curve (with bin populations), PR, ROC, training curves; every function is a no-op with a reason when matplotlib is absent |
| `schedules.py` | per-step warmup + cosine / step / constant, as a pure function of the step index so `--resume` restores the right LR |
| `amp.py` | autocast + grad scaler, CUDA-only, with an identical call sequence when disabled |
| `augment.py` | the label-preserving augmentation sets, with the argument for every inclusion and exclusion |
| `morphology_data.py` | MHSMA adapter protocol + the synthetic crop source; official split preserved, never re-derived |
| `morphology_report.py` | low-positive warnings, all-four-normal joint accuracy, the shared plot set |
| `detection_data.py` | detection adapter protocol, synthetic clips, `assert_no_video_leakage`, CenterNet target construction |

---

## Verification runs

Every number in this section came out of the command above it, on this
repository's CPU box. They are *properties* the implementation must have, not
performance claims.

### Seeding — two runs with the same seed agree, different seeds do not

One epoch, 160 synthetic crops, `simplecnn`, first-epoch losses:

| seed | run | `train_loss` | `val_loss` |
|---|---|---|---|
| 1234 | a | 4.7915260077 | 4.3753796577 |
| 1234 | b | 4.7915260077 | 4.3753796577 |
| 4321 | a | 4.6519056082 | 4.5926966667 |

### `--resume` genuinely continues

`runs/morph_smoke/last.pt` before and after resuming a 2-epoch run to 4 epochs:

| | before | after |
|---|---|---|
| `training_state.epoch` | 2 | 4 |
| `training_state.global_step` | 30 | 60 |
| Adam `step` on parameter 0 | 30.0 | 60.0 |
| Adam `exp_avg` norm, parameter 0 | 0.0756 | 0.1321 |
| scheduler `last_epoch` | 30 | 60 |
| `history` length | 2 | 4 |

The Adam step counter continuing from 30 rather than restarting is the check
that matters: it is what distinguishes a resume from a warm restart.

### `eval_detector.py` on hand-built boxes

5 frames × 4 sperm boxes (20 ground-truth objects) + one debris box per frame.

| prediction set | AP50 | precision | recall | F1 | debris FP rate | count error (signed) |
|---|---|---|---|---|---|---|
| identical to ground truth | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0 |
| one of four dropped per frame (15/20) | 0.7525 | 1.0000 | **0.7500** | 0.8571 | 0.0000 | −1 |
| perfect + one FP on the debris particle | 1.0000 | 0.8000 | 1.0000 | 0.8889 | **0.2000** | +1 |

Recall of exactly 0.7500 is the hand-computed 15/20. AP50 of 0.7525 is
76/101 — the COCO 101-point grid has 76 recall points at or below 0.75 and the
precision envelope is 1.0 across all of them, which is the correct behaviour of
that interpolation, not an approximation of 0.75. The debris FP rate of 0.2000
is 5 debris hits out of 25 predictions.

### `eval_tracking.py` on hand-built tracks

3 tracks × 20 frames = 60 ground-truth detections. The "swap" case reports
ground-truth track 1 under id `1` for frames 0–9 and under id `99` for frames
10–19; the geometry is identical to the perfect case.

| | perfect | single ID swap mid-track |
|---|---|---|
| HOTA | **1.0000** | 0.9129 |
| DetA | 1.0000 | 1.0000 |
| AssA | 1.0000 | 0.8333 |
| LocA | 1.0000 | 1.0000 |
| **IDF1** | **1.0000** | **0.8333** |
| MOTA | 1.0000 | 0.9833 |
| MOTP (mean IoU) | 1.0000 | 1.0000 |
| **ID switches** | **0** | **1** |
| fragmentations | 0 | 0 |
| duplicate-count rate | 0.0000 | 0.3333 |
| excess-id ratio | 0.0000 | 0.3333 |

Every swap-case number is checkable by hand:
IDTP = 20 + 10 + 20 = 50, IDFN = IDFP = 10, so IDF1 = 100/120 = 0.8333.
MOTA = 1 − (0 + 0 + 1)/60 = 59/60 = 0.9833.
AssA = (20·1 + 20·1 + 10·0.5 + 10·0.5)/60 = 50/60 = 0.8333, since the split
identity gives A = 10/(20 + 10 − 10) = 0.5 for each half.
HOTA = √(1 × 0.8333) = 0.9129.
Duplicate rate = 1 of 3 ground-truth tracks acquired two ids.

### `eval_pipeline.py` end to end

`configs/synthetic.yaml`, 400 frames at 640×400, oracle detector, ByteTrack,
**morphology weights unset (an untrained network)**. The eligibility numbers
below therefore measure an untrained model and are reported only to show the
harness works end to end — they are not a statement about the product:

```
per-sperm eligibility agreement    0.5385   (78 gated sperm scored)
  sensitivity (eligible)           0.0000   <- untrained head calls nothing eligible
  specificity (eligible)           1.0000
shot-ratio mean signed error      -0.5067
shot-decision agreement            1.0000   (3 REJECT + 1 INDETERMINATE, all correct)
indeterminate rate (pred / true)   0.2500 / 0.2500
command-alignment error            0.0000
  FIELD_ON not delivered           2        <- run ended inside the 1600 ms transport delay
```

The `FIELD_ON not delivered` count is the point of separating decision error
from delivery error: all four decisions were right, but two of the resulting
commands were still queued when the bounded run ended.

---

## Known gaps in this checkout

- **`datasets/adapters/` is empty.** `--source mhsma` and `--source visem` are
  implemented against the narrow `MorphologyDatasetAdapter` and
  `DetectionDatasetAdapter` protocols in `training/common/`, and report exactly
  which module, class or method was missing rather than failing with a bare
  `ImportError`. Neither has been exercised against a real adapter, so no MHSMA
  or VISEM-Tracking numbers appear anywhere in this directory.
- **No trained weights are shipped.** Every checkpoint referenced above came
  from a 2-epoch smoke run and exists to prove the plumbing, not the model.
- **`datasets/validators/` is empty**, so `assert_no_video_leakage` currently
  runs its own implementation. It will defer to `datasets.validators` the moment
  that module exposes `assert_no_video_leakage`, `validate_no_video_leakage` or
  `check_video_leakage`.
