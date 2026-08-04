# Domain shift

Why weights trained on public datasets are **baseline research weights only**,
how far the domains actually are apart, and what the mitigation path looks like.

The short version: every public dataset in this project was captured on a
different optical setup, at a different magnification, with a different contrast
mechanism, at a different frame rate, from a different population. None of those
differences is marginal, and several of them remove information the model is
being asked to use.

Nothing here reports a performance number, because **no model in this repository
has been trained**. This document is about the size of the gap and the plan for
closing it, not about results.

---

## 1. The gap, quantified

| Axis | VISEM / VISEM-Tracking | MHSMA / HSMA-DS | MIaMIA-SVDS | **This instrument** | Ratio |
|---|---|---|---|---|---|
| Magnification | 400x | x400 and x600 | 20x obj + 20x eyepiece | **100x oil, NA 1.25** | **4-5x lower** in the public sets |
| Contrast method | **phase contrast**, unstained wet preparation | **fixed, stained smear** (stain UNVERIFIED) | CASA-system optics | **brightfield on oil** | resembles neither |
| Frame resolution | 640 x 480 | 128x128 / 64x64 crops | native UNVERIFIED (416x416 is a resize target) | 1920 x 1200 | — |
| Sperm head span | **tens of px** | crop-filling, head-centred | anchors 7-19 px | **~119 x 81 px** | ~4-10x |
| Frame rate | 45-50 (non-uniform) / 50 Hz | n/a (still images) | 30 Hz | **160 Hz** | **3.2-5.3x** |
| Numerical aperture | 400x dry, comparatively deep DOF | dry | dry | **NA 1.25, far shallower DOF** | — |
| Shutter | UEye UI-2210C (rolling vs global **UNVERIFIED**) | n/a | n/a | **global** | — |
| Colour | UI-2210C (mono vs colour **UNVERIFIED**) | grayscale | — | **monochrome** | — |
| Temperature | 37 C heated stage | n/a (fixed smear) | — | 37 C required, **UNMEASURED** | — |
| Population | 85 and 20 participants | **235 male-factor-infertility patients** | — | unspecified | — |

### 1.1 Magnification: information that is not there

Four to five times lower magnification is not a scale factor that a resize can
undo. **Vacuole and acrosome detail is literally absent** from the public
morphology sources at the resolution this instrument works at. A vacuole
occupying one fifth of a head -- WHO's own criterion -- spans roughly 24 px here
and a handful of pixels in a 400x image. A model trained on the latter has never
seen the feature it is being asked to judge; fine-tuning it on device data is
therefore not adaptation but genuinely new learning.

The converse also matters. At ~119 px per head this instrument is heavily
**over**-resolved for head-centroid CASA, so a detector trained on tiny objects
(MIaMIA's anchors are all under 20 px) is solving a different problem: the
tiny-object regime is about *preserving* spatial evidence through a downsampling
stack, and here there is evidence to spare.

### 1.2 Contrast: three mutually unlike appearances

- **Phase contrast on unstained wet preparations** (VISEM, VISEM-Tracking)
  produces bright halos around edges and derives contrast from optical path
  length, not absorption. The halo is an artefact of the method and is a strong,
  learnable cue that does not exist in brightfield.
- **Fixed, stained smears** (MHSMA via HSMA-DS) produce absorption contrast on a
  dead, flattened, air-dried cell. The staining protocol is **UNVERIFIED**, which
  matters because WHO notes that "each stain provides quite different results
  down to the level of sperm sizes".
- **Brightfield on oil**, which this instrument uses, resembles neither. Unstained
  live sperm are near-transparent phase objects with poor brightfield contrast,
  which is itself flagged as the second most likely wrong assumption in
  `docs/assumptions.md`.

The practical consequence for transfer is that low-level filters -- the first
convolution, which is where most of the transferable signal in a small model
lives -- are tuned to the wrong image formation model.

### 1.3 Frame rate: a bias with a known direction

30 / 45-50 / 50 Hz against 160 Hz. Frame rate is not a neutral acquisition
parameter for kinematics:

- Higher fps **raises VCL** and **lowers LIN and ALH**, with VSL roughly
  unchanged (Castellini et al., *Fertil Steril* 2011 -- **PARTIALLY VERIFIED**,
  the primary PDF was 403-blocked, so the *direction* is used and no magnitude
  is).
- So a model or a threshold set calibrated at 50 Hz, applied at 160 Hz, sees
  systematically higher VCL and lower LIN than it was tuned for -- which, with a
  LIN floor in the progressive rule, biases toward **demotion**.
- The VISEM-Tracking frame rate is additionally **non-uniform across videos**,
  so kinematics must be normalised by the true per-video frame interval rather
  than a single assumed rate.

This is why `MotionConfig.vap_window_ms` is a **duration**, converted per track
against that track's own measured frame rate. A fixed frame count is what
Mortimer et al. (2015) identify as producing "widely aberrant ALH values" when
the rate changes.

### 1.4 Depth of field: far more out-of-focus cells

At NA 1.25 the depth of field is much shallower than on a 400x dry setup. In a
chamber of any depth, a far larger fraction of cells will be out of focus here
than in the public recordings. Two effects:

- The **detector** sees a defocus distribution the public data does not contain.
- The **feasibility budget** is affected: the 20 um optical section assumed in
  `docs/assumptions.md` section 2 almost certainly over-counts, which makes the
  91 million/mL requirement an under-estimate.

Numeric depth-of-field values are deliberately not quoted -- Edmund's figures
were retrieved once and could not be re-fetched, so they are UNVERIFIED.

### 1.5 Population: MHSMA is not a general sample

MHSMA's 1540 crops come from **235 male-factor-infertility patients**. Its
abnormality prevalences -- acrosome 30.1%, head 27.3%, vacuole 17.0%, tail 4.6%
in the train split -- are prevalences *in that population*.

Two consequences that survive any amount of fine-tuning:

- A model's **prior** is set by the training prevalence. Deployed on a different
  population, its calibrated probabilities are wrong even if its ranking is
  right. This is exactly what `morphology/calibration.py` exists to correct,
  and why the calibration bundle is a small JSON file that travels next to the
  checkpoint and can be re-fitted on new device data in seconds without touching
  the network.
- The **tail aspect has almost no signal to transfer**: 4.6% prevalence in train,
  and **only 7 abnormal tails in the entire validation split**. Any threshold
  fitted there has enormous variance.

### 1.6 The gap that no amount of adaptation closes

Public data does not contain the label this product needs.

**No public dataset provides per-sperm bounding boxes and per-sperm morphology
for the same cell** (`docs/dataset_audit.md` section 6). MHSMA has morphology and
no boxes; VISEM-Tracking has boxes and track identity and no morphology; VISEM
has only ejaculate-level percentages. The product rule is a *conjunction over one
cell*:

```
ai_eligible = progressive motility AND all four morphology aspects normal
```

Each half can be trained from public data. **The conjunction cannot be measured
from it.** That is a labelling gap, not a domain gap, and it is why the built-in
simulator exists -- it is the only source in this project for which per-cell
motion and per-cell morphology ground truth exist for the same virtual cell.

---

## 2. What this means operationally

Public-dataset weights are stamped `WEIGHTS_PROVENANCE_PUBLIC =
"public-research-baseline"` (`constants.py`). That string is copied onto every
`MorphologyResult`, echoed by `AppConfig.summary()`, and therefore written into
the manifest of every audit log. `configs/device_v1.yaml` says it in the config
file itself:

```yaml
morphology:
  weights: models/morphology/mhsma_mobilenetv3.pt
  # Public-dataset weights are BASELINE RESEARCH WEIGHTS. They were trained on
  # 400x phase-contrast images of stained smears, not on this instrument's
  # 100x oil brightfield frames. Fine-tune on device data before trusting them.
  weights_provenance: public-research-baseline
```

The default value of `weights_provenance` is the literal string `"unset"`, which
is deliberately not a valid provenance: a run that never declared where its
weights came from says so in its own log.

There is a licensing dimension to the same decision. MHSMA is **CC BY-NC-SA
4.0**, and the safe reading is that MHSMA-derived weights inherit the restriction
-- so MHSMA-derived weights are not only scientifically unsuited to the device,
they are commercially unusable. See `docs/license_audit.md` section 4.

---

## 3. The mitigation path

Five components. **Status is stated per component**, because part of the point of
this document is that nothing is claimed to work that has not been built.

| # | Component | Where it lives | Status in this tree |
|---|---|---|---|
| 1 | `DeviceDatasetAdapter` | `datasets/adapters/device.py` | **implemented** |
| 2 | Detector fine-tuning | `training/train_detector.py` | **implemented**; not yet run |
| 3 | Morphology fine-tuning + recalibration | `training/train_morphology.py`, `morphology/calibration.py` | **implemented**; not yet run |
| 4 | Domain-specific normalisation | `preprocessing/preprocessor.py`, `cropping/extractor.py`, `detection/preprocess.py`, `training/common/augment.py` | **implemented** |
| 5 | Self-supervised pretraining on unlabelled device video | — | **specified here, optional, not implemented** |

"Implemented" means the code exists. **No training run has been executed and no
model exists**, so there are no results to report anywhere in this repository.

### 3.1 `DeviceDatasetAdapter` (`datasets/adapters/device.py`)

The single reader for data recorded on the instrument -- "the only corpus that is
actually in-domain, and the only one with no third-party licence constraint",
in the module's own words. It presents device captures in the same in-memory form
every public-dataset adapter produces, so a training run does not care where its
data came from.

**Format: JSON Lines, header first.** One file per capture session:

```
{"record_type": "header", "schema_version": "1.0.0", "capture": {...}, ...}
{"record_type": "frame", "frame_id": 0, "boxes": [...], ...}
{"record_type": "frame", "frame_id": 1, "boxes": [...], ...}
```

JSONL rather than one JSON array because a capture is appended to while it runs:
a partially-written JSONL file is readable up to its last complete line, whereas
a truncated JSON array is not readable at all. Session metadata lives in the
**header**, not on every frame, because `um_per_px`, the operator and the sample
id are properties of the session -- repeating them 30,000 times invites drift,
and a file where frame 400 claims a different scale from frame 399 is one nobody
can interpret. The three fields that genuinely can drift within a session
(`exposure_us`, `gain_db`, `temperature_c`) may be overridden per frame, and
`DeviceCapture.effective_capture` resolves the override, so a reader never has to
know which it was.

**Required capture metadata** (`REQUIRED_CAPTURE_FIELDS`), each earning its
place against the failure modes this document describes:

| Field | Why it is required |
|---|---|
| `sample_id`, `operator` | Provenance. Without them a capture cannot be grouped for a patient-level split, and cannot be excluded when consent is withdrawn. |
| `um_per_px` | Without it no velocity can be expressed in physical units, and the WHO thresholds this product decides on are in um/s. Mirrors `CalibrationError`: never silently substitute pixel units for physical ones. |
| `frame_rate_hz` | Every kinematic quantity divides by the frame interval -- and section 1.3 is why. |
| `exposure_us`, `gain_db` | The two knobs that change image appearance most, and therefore the two you need when a model trained on Tuesday's captures underperforms on Friday's. |

`temperature_c` is **optional but warned about**, because not every rig measures
it and a motility comparison across captures at unknown temperatures is not a
comparison (section 5.4 of `docs/assumptions.md`).

**Contracts it enforces**, all of which follow from constraints elsewhere in the
codebase:

- **Boxes in `(x1, y1, x2, y2)` absolute source-frame pixels** -- the one format
  `schemas/detection.py` accepts, so no module ever has to guess whether it is
  holding `xywh` or `xyxy`. VISEM-Tracking's normalised centre-form is converted
  *into* it, never the reverse.
- **Labels in MHSMA polarity**, `0 = normal` / `1 = abnormal`, per aspect, over
  `constants.MORPHOLOGY_ASPECTS`. `ObjectRecord.__post_init__` rejects an unknown
  aspect name and any value other than 0, 1 or `None`.
- **"Not assessed" is `None` or absent -- never 0.** From the module: "Nobody
  looked" and "looked, and it was normal" are different facts, and collapsing
  them silently biases every prevalence computed from this data toward normal.
- **Splits by video and by patient, never by frame**, enforced by
  `datasets/validators/leakage.py`, which raises rather than warns. Its own
  reasoning is the sharpest statement of the problem in the repository: split
  VISEM-Tracking's frames at random and 80% of validation frames have a
  near-identical twin in training, so the reported mAP measures memorisation of
  20 fields of view -- and the published YOLOv5l baseline of 0.2231 "is what an
  honest per-video split looks like".
- **Validation at load**, raising `DatasetValidationError`.

What it buys is that the *same* training and evaluation code runs on public,
device and synthetic data, so a comparison between them measures the data and
not the loader.

### 3.2 Detector fine-tuning

The detector is the more tractable half, because VISEM-Tracking is **CC BY 4.0**
(commercially clean) and supplies real boxes and real track identity.

The plan:

1. Pretrain on VISEM-Tracking, respecting the official 16/4 video split with
   validation videos **82, 60, 54, 52**. Expect a hard problem -- the published
   YOLOv5l baseline is mAP@0.5 = 0.2231.
2. Randomise position during augmentation, because VISEM-Tracking's boxes
   concentrate in the **upper-left** of the frame and a detector will otherwise
   learn the prior instead of the object.
3. Fine-tune on device boxes at the device's scale.

`training/train_detector.py` implements this and calls
`assert_no_video_leakage` **before the first optimiser step**, raising rather
than warning. Its reasoning is worth repeating because it generalises: the
failure it guards against makes the metric look *better*, so nothing downstream
would ever flag it, and a warning in a log nobody reads is indistinguishable from
no check at all.

Two architectural notes that reduce how much has to transfer. The head is
**anchor-free** (CenterNet-style), so there is no anchor set tuned to the source
domain's object size to carry over -- which matters precisely because the source
domains are 4-10x smaller in pixels. And the same head is shared by `todcnn` and
`p2net`, so a comparison between them measures the backbone and nothing else.

### 3.3 Morphology fine-tuning and recalibration

The harder half, and the one with the licensing problem.

1. **Pretrain** on MHSMA at 128x128 (its crops match `crop.output_size` exactly,
   which is why 128 is the default). Note what this is and is not: it is a
   feature initialiser trained on stained smears at 400x/600x.
2. **Fine-tune** on device crops cut by the real `CropExtractor`, so that the
   letterboxing, padding fraction and normalisation the model sees at training
   time are byte-identical to what it sees at inference.
3. **Re-fit the calibration bundle** on device data. This is the cheap, high-value
   step, and it is already implemented:
   - `TemperatureScaler` divides each aspect's logits by one learned constant.
     It cannot change the ranking, so it cannot change AUC -- but it makes the
     numbers mean what they say, which matters because the audit log records
     `p_normal` and a human reads it.
   - `fit_thresholds` chooses a **per-aspect** threshold on a validation split,
     against a criterion that is not fooled by prevalence (Youden's J by
     default). With 4.6% of tails abnormal, the 0.5 default classifies everything
     as normal and scores 95% accuracy while catching none of them.

   Both artefacts are deliberately kept **out of the weights**: a
   `CalibrationBundle` is a small JSON file next to the checkpoint that can be
   re-fitted on new device data in seconds. It carries the polarity string, and
   refuses to load under the other convention.
4. **Report the right metrics.** `morphology/metrics.py` deliberately does not
   report raw accuracy. Balanced accuracy, MCC and per-class sensitivity and
   specificity all collapse to chance for the all-normal predictor; accuracy does
   not. `training/eval_morphology.py` reports, per aspect: sensitivity,
   specificity, precision, NPV, macro-F1, balanced accuracy, MCC, ROC-AUC, PR-AUC
   and expected calibration error -- at the checkpoint's *shipped* operating
   point, i.e. with its calibration bundle applied.

`training/train_morphology.py` states the same polarity contract this document
does, and enforces it structurally: the training target is the MHSMA integer
verbatim and there is no `1 - y` anywhere in the file. It can also train on
simulator crops (`--source synthetic`), which is the clean-lineage bootstrap of
`docs/license_audit.md` section 5.1.

### 3.4 Domain-specific normalisation

Implemented, and the cheapest partial mitigation available, because it removes
the *photometric* component of the shift before a model ever sees the image.

| Knob | Where | What it addresses |
|---|---|---|
| `PreprocessConfig.invert` | frame | Brightfield gives dark objects on a bright field; phase contrast and some detectors expect the opposite. The simulator states the convention once (`RenderConfig.dark_objects`) and implements it once, because getting the sign backwards silently would poison every model trained on it. |
| `PreprocessConfig.normalize` (`minmax` / `zscore` / `clahe`) | frame | Global illumination differences between rigs. CLAHE additionally raises local contrast, at the cost of amplifying noise in flat regions. |
| `PreprocessConfig.background_subtraction` | frame | A rolling median over `background_window` frames removes fixed pattern and slow illumination drift, which differ per rig. |
| `CropConfig.normalize` | crop | Per-crop `minmax` or `zscore`, applied after the frame-level pass. Note the deliberate asymmetry: the frame-level `zscore` squashes into `[0, 1]` because it must produce a viewable frame, while the crop-level one is standardised for a CNN and clipped to ±4 sigma. |
| `CropConfig.preserve_aspect_ratio` | crop | Letterboxing rather than squashing, so head length:width -- the single feature the head classifier keys on -- is not distorted by exactly the amount the box was non-square. |
| Grayscale-native backbones | model | The microscope is monochrome. `morphology/backbones.py` rebuilds the stem for `in_channels=1` and, when starting from pretrained RGB weights, **sums** the first convolution's kernel across the input axis rather than averaging or re-initialising -- so that for `R = G = B = g`, `sum_c W_c * g == (sum_c W_c) * g` reproduces exactly the response the pretrained filter would have given, leaving every downstream BatchNorm statistic valid. Averaging would scale every activation by 1/3. |

`preprocess.py` in `detection/` is shared by the torch and ONNX paths for the
same reason: if one normalised by 255 and the other by the frame maximum, an
exported model would score differently from the model it was exported from, and
the discrepancy would look like an export bug.

### 3.5 Optional: self-supervised pretraining on unlabelled device video

The device produces vastly more unlabelled video than anyone will annotate. A
self-supervised objective -- contrastive, masked-image-modelling, or a temporal
pretext task -- can pretrain a backbone on the *device's own* image statistics,
after which supervised fine-tuning needs far fewer labels.

Why it is a good fit here specifically:

- The domain gap is largely **low-level** -- contrast mechanism, noise, scale,
  defocus -- and that is exactly what self-supervision on in-domain data fixes.
- Unlabelled device video is free and unlimited, whereas per-aspect morphology
  annotation is expert-intensive.
- **VISEM** is 85 videos of 2-7 minutes with only ejaculate-level labels, i.e.
  the largest volume of real human sperm microscopy available and almost no
  supervision. It is a natural self-supervised corpus -- with the caveat that it
  is 400x phase contrast, so it narrows the *biological* gap while leaving the
  *optical* one, and it is **CC BY-NC 4.0**, so anything derived from it is
  non-commercial.
- A synthetic-then-device path is cleaner still:
  `WEIGHTS_PROVENANCE_SYNTHETIC` weights come from this project's own simulator
  and carry no third-party restriction at all.

Marked optional because it is a large piece of work whose payoff depends on how
much device annotation turns out to be affordable.

---

## 4. How to tell whether it worked

Without inventing a target number, the checks that would demonstrate transfer:

| Check | What it shows |
|---|---|
| Detector recall and precision on **held-out device video**, split by patient | The only measurement that matters for the detector. Public-set metrics are a sanity check, not evidence. |
| Track fragmentation rate on device video: tracks per true sperm | Directly tests invariant I1. A fragmentation rate above 1.0 means sperm are being counted twice in the denominator. |
| Per-aspect **balanced accuracy, MCC, sensitivity and specificity** on device crops | Never raw accuracy. The tail aspect must be reported separately, with its own count of abnormal examples. |
| **Expected calibration error** before and after re-fitting the bundle | Whether `p_normal` in the audit log means what it says. |
| Distribution of `IneligibilityReason` on device runs versus synthetic runs | A large shift in the histogram shape localises which stage the domain gap is hurting. |
| Fraction of shots reaching `INDETERMINATE` | Distinguishes a throughput problem (`docs/assumptions.md` section 2) from a model problem. |
| End-to-end `ai_eligible` accuracy | **Only measurable on synthetic data**, because it is the only source with per-cell ground truth for both halves of the conjunction. |

And the standing caveat, from `acquisition/synthetic.py`: synthetic timestamps
are exact by construction, so velocity estimates are *better* there than on
hardware. **A model that only works on synthetic data should be assumed not to
work on a camera.**
