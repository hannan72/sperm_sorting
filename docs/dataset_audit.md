# Dataset audit

Exact layout, shapes, dtypes, annotation formats, label semantics, splits,
balance, quirks, capture conditions and licence for every public dataset this
project touches. Provenance and what could not be verified are in
`docs/source_audit.md`; licensing analysis is in `docs/license_audit.md`.

The single most important finding is in section 6: **no public dataset provides
per-sperm bounding boxes and per-sperm morphology for the same cell.** That gap
is not incidental -- it is why the built-in simulator exists.

---

## 1. MHSMA (Modified Human Sperm Morphology Analysis)

**Source.** https://github.com/soroushj/mhsma-dataset -- the arrays are committed
into the repository itself, under `mhsma/`. It is **not on Zenodo**; Zenodo
record 2640506 is VISEM.

**Provenance.** Derived from HSMA-DS (Ghasemian 2015,
doi:10.1016/j.cmpb.2015.08.013), captured at x400 and x600. The staining
protocol and microscope model are **UNVERIFIED**.

### 1.1 File layout

Eighteen `.npy` files, **all dtype `uint8`**.

| File | Shape | dtype |
|---|---|---|
| `x_128_train.npy` | `(1000, 128, 128)` | `uint8` |
| `x_128_valid.npy` | `(240, 128, 128)` | `uint8` |
| `x_128_test.npy` | `(300, 128, 128)` | `uint8` |
| `x_64_train.npy` | `(1000, 64, 64)` | `uint8` |
| `x_64_valid.npy` | `(240, 64, 64)` | `uint8` |
| `x_64_test.npy` | `(300, 64, 64)` | `uint8` |
| `y_acrosome_{train,valid,test}.npy` | `(1000,)` / `(240,)` / `(300,)` | `uint8` |
| `y_head_{train,valid,test}.npy` | `(1000,)` / `(240,)` / `(300,)` | `uint8` |
| `y_vacuole_{train,valid,test}.npy` | `(1000,)` / `(240,)` / `(300,)` | `uint8` |
| `y_tail_{train,valid,test}.npy` | `(1000,)` / `(240,)` / `(300,)` | `uint8` |

Images are 2-D and grayscale -- there is no channel axis. A loader that assumes
`(N, H, W, C)` fails.

**The split word is `valid`, not `val`.** A path built from memory silently
resolves to nothing.

### 1.2 Label semantics

`0 = normal`, `1 = abnormal`, for every one of the four aspects independently.

**The README calls the *normal* class "positive".** Its "% Positive" column is
therefore the percentage of *normal* cells. Reading that table as abnormality
prevalence inverts every number in it, and an inverted morphology model is
indistinguishable from a working one by accuracy alone.

This repository encodes the convention once, in
`src/sperm_sorting/constants.py`:

```python
LABEL_NORMAL: Final[int] = 0
LABEL_ABNORMAL: Final[int] = 1
```

and fixes a single polarity contract in `src/sperm_sorting/morphology/polarity.py`:
every logit the network emits is a logit for `P(abnormal)`, the training target
is the MHSMA integer verbatim, and the one flip to the schema's `p_normal`
happens in the inference adapter and nowhere else.

The canonical aspect order, used for model heads, threshold vectors and metric
tables alike, is `constants.MORPHOLOGY_ASPECTS = ("head", "acrosome", "vacuole", "tail")`.

### 1.3 Official split and population

1000 train / 240 valid / 300 test = **1540 images from 235 male-factor-infertility
patients**. Split by the dataset authors; this project uses it unchanged.

Note the population: these are patients presenting with male-factor infertility,
not a general or a fertile-population sample. Prevalences below are prevalences
*in that population*.

### 1.4 Class balance -- abnormal prevalence, percent

Derived by inverting the README's "% Positive" column (see 1.2).

| Split | acrosome | head | vacuole | tail |
|---|---|---|---|---|
| whole | 29.48 | 27.14 | 15.52 | 4.48 |
| train | 30.10 | 27.30 | 17.00 | 4.60 |
| valid | 27.50 | 26.67 | 12.92 | **2.92** |
| test | 29.00 | 27.00 | 12.67 | 5.33 |

Two consequences the code acts on:

**A single shared decision threshold is meaningless here.** The aspects differ
in prevalence by roughly an order of magnitude, from 30.1% (acrosome, train) to
4.6% (tail, train). A model that predicts "normal" for every tail scores 95.4%
accuracy on the train distribution and has zero value. This is why
`MorphologyConfig.thresholds` and `MorphologyConfig.temperatures` are
**per-aspect** dictionaries, why the shipped 0.5 values are labelled a
placeholder rather than a calibration, and why
`src/sperm_sorting/morphology/metrics.py` deliberately does not report raw
accuracy at all -- the headline numbers there are balanced accuracy, MCC, and
per-class sensitivity and specificity, all of which collapse to chance for the
all-normal predictor.

**The validation split holds only 7 abnormal tails.** 2.92% of 240 is 7.008, so
the tail aspect's entire validation signal is seven images. Any tail threshold
fitted on that split has an enormous variance; a single relabelled image moves
sensitivity by 14 percentage points. Threshold fitting for the tail aspect
should be treated as unreliable on the official split, and either pooled
across splits (accepting the leakage that implies) or deferred to device data.

### 1.5 Known quirks

- Grayscale, one sperm per image, head roughly centred.
- **The tail is not entirely visible** in the crops. The tail aspect is therefore
  judged from a partial tail -- which, together with its 4.6% prevalence, makes
  it the least reliable of the four. `DetectionConfig`'s docstring records this
  explicitly, and the crop stage carries a `tail_complete` flag on every
  `CropRecord` for the same reason.
- No video, no bounding boxes, no track identity, no kinematics.

### 1.6 Capture conditions and licence

x400 and x600, from fixed and stained smears (HSMA-DS). Staining protocol
**UNVERIFIED**. Licence **CC BY-NC-SA 4.0** -- non-commercial *and* share-alike.

---

## 2. VISEM-Tracking

**Source.** Zenodo record 7293726; *Sci Data* 2023; preprint arXiv:2212.02842.

### 2.1 File layout

```
VISEM_Tracking_Train_v4/
  Train/
    <video_id>/
      <frames directory>
      labels/            # bounding boxes only
      labels_ftid/       # bounding boxes plus tracking id
      <video_id>.mp4
```

### 2.2 Annotation format -- exact field order

`labels/` uses standard YOLO, normalised to `[0, 1]`:

```
class  x_center  y_center  width  height
```

`labels_ftid/` prefixes the track identity, so the field order is **not** the
YOLO order:

```
sperm_id  class  x_center  y_center  width  height
```

A parser that assumes the first field is the class will read every track id as a
class index. Both files use normalised centre-form coordinates; this project's
internal representation is absolute-pixel `(x1, y1, x2, y2)`
(`schemas/detection.py`), so a conversion is always required.

### 2.3 Classes

| index | name |
|---|---|
| 0 | sperm |
| 1 | cluster |
| 2 | small or pinhead |

### 2.4 Scale

| Quantity | Value |
|---|---|
| Annotated clips | 20, of 30 s each |
| Patients | 20 |
| Annotated frames | 29,196 |
| Bounding boxes | 656,334 |
| Unique sperm track IDs | 1,121 |
| Cluster IDs | 20 |
| Pinhead IDs | 35 |

### 2.5 Official split

16 train / 4 validation, **split by video**. The validation video IDs are:

> **82, 60, 54, 52**

There is **no official test split**. Anything reported as a test result on this
dataset is a split someone invented; say which.

Splitting by video is not optional. Consecutive frames of the same clip are
near-duplicates, and a frame-level shuffle puts the same sperm on both sides of
the boundary. `errors.LeakageError` exists for exactly this failure.

### 2.6 Known quirks

| Quirk | Detail | Consequence |
|---|---|---|
| Empty frames | `video_23` contains **174 frames with no sperm at all** | A loader that assumes every frame has at least one box will crash or silently skip; a metric that divides by object count will produce NaN. |
| Non-uniform lengths | `video_35` and `video_52` have 1440 frames; `video_82` has 1500 | Note that `video_52` and `video_82` are both *validation* videos. |
| Non-uniform frame rate | 45-50 FPS, and **not uniform across videos** | Every kinematic quantity must be normalised by the true per-video frame interval, never by a single assumed rate. This is the same failure mode `TimestampSource.CONTAINER_PTS` exists to make visible in the runtime. |
| Spatial prior | Boxes concentrate in the **upper-left** of the frame | A detector trained without positional augmentation learns the prior rather than the object. |
| Difficulty | The published YOLOv5l baseline reaches **mAP@0.5 = 0.2231** | This is a hard dataset. Any substantially higher number should be assumed to come from a different split or a different metric until proven otherwise. |

### 2.7 Capture conditions and licence

Olympus CX31, 400x, **phase contrast**, 37 C heated stage, UEye UI-2210C camera.
Resolution 640x480 -- **inherited from VISEM; the VISEM-Tracking authors never
restate it for this dataset.** Whether the UI-2210C is monochrome or colour, and
rolling or global shutter, is **UNVERIFIED**.

Licence **CC BY 4.0** (commercial use permitted).

---

## 3. VISEM

**Source.** Zenodo record 2640506.

| Property | Value |
|---|---|
| Participants / videos | 85 / 85 |
| Duration | 2-7 minutes each |
| Resolution | 640x480 |
| Frame rate | 50 FPS |
| Container | AVI |
| Distribution | a single 35.2 GB zip |

**Annotations are sample- or video-level only.** Six CSVs carry a WHO semen
analysis, motility percentages (progressive / non-progressive / immotile),
concentration, fatty acids, sex hormones and demographics. There are **no
per-sperm labels and no bounding boxes**. The exact CSV filenames are
**UNVERIFIED** -- `datasets.simula.no` returned an SSL certificate error.

**Capture.** Olympus CX31, phase contrast, 37 C, UEye UI-2210C, 400x, 10 ul under
a 22x22 mm coverslip.

**Licence.** CC BY-NC 4.0 (non-commercial).

The practical role of VISEM here is as a source of *unlabelled* device-adjacent
video: it is the largest volume of real human sperm microscopy available, and
its only supervised signal is an ejaculate-level percentage. See section 6 for
why that cannot be turned into per-cell labels, and `docs/domain_shift.md` for
the self-supervised use it is suited to.

---

## 4. VISEM-Tracking-graphs

**Source.** Hugging Face, `SimulaMet-HOST/visem-tracking-graphs`. 3.26 GB,
GraphML, read with networkx.

### 4.1 Layout

```
spatial_threshold_0.1/ ... spatial_threshold_0.5/
  <video>/
    frame_graphs/frame_graph_{i}.graphml
    video_graph.graphml
```

### 4.2 Node and edge schema

| Element | Field | Meaning |
|---|---|---|
| Node id | `sperm_id` | Track identity from `labels_ftid` |
| Node attr | `frame_number` | Frame index |
| Node attr | `class_name` | The YOLO class **index as a string** -- `"0"`, `"1"`, `"2"`, not a word |
| Node attr | `x_center`, `y_center`, `width`, `height` | YOLO-normalised, 0-1 |
| Edge | spatial | `weight` = Euclidean distance between normalised centres; added when below the directory's threshold |
| Edge | temporal | `edge_type="temporal"` |

Frame graphs are **undirected**; the video graph is **directed**.

### 4.3 Defect: node collapse in `video_graph.graphml`

`video_graph.graphml` keys nodes by **`sperm_id` alone**, not by
`(sperm_id, frame)`. GraphML node ids are unique by definition, so every
observation of one sperm across the whole clip collapses onto a single node, and
the temporal-edge construction -- which intends to link observation *i* to
observation *i+1* of the same track -- degenerates into **self-loops** on that
one node. The per-video graphs therefore carry no usable temporal structure, and
the per-node `frame_number`, `x_center` and `y_center` attributes hold whichever
observation was written last.

The per-frame graphs (`frame_graphs/frame_graph_{i}.graphml`) appear sound,
because within one frame a `sperm_id` really is unique.

**The fix: key nodes by `(video_id, frame_id, track_id)`.** All three components
are needed -- `frame_id` to separate observations, `track_id` to keep them
associated, and `video_id` because track ids are only unique within a video, so
merging clips without it re-creates the same collapse at a larger scale. With
that identity, temporal edges become genuine `t -> t+1` arcs between distinct
nodes and spatial edges remain intra-frame.

Regenerating from `labels_ftid` directly is straightforward and is preferable to
patching the released graphs, because the released graphs have already lost the
information.

### 4.4 Defect: anisotropic spatial threshold

The spatial edge criterion compares **normalised** coordinates. On a 4:3 frame,
one unit of normalised x is 640 px and one unit of normalised y is 480 px, so a
fixed normalised threshold reaches 4/3 as far horizontally as vertically. The
resulting neighbourhoods are ellipses, not discs, and the graph's "proximity"
relation is orientation-dependent. Convert to pixels before thresholding, or
scale the axes.

### 4.5 Licence

Data **CC BY 4.0**; generator code **MIT**.

---

## 5. Detection-Sperm / TOD-CNN / MIaMIA-SVDS

**Sources.** https://github.com/Demozsj/Detection-Sperm (code); figshare record
15074253 (`Data Set.rar`, 1.42 GB); paper arXiv:2204.08166.

### 5.1 The repository is a model repository, not a dataset repository

Its own README describes the included data as a "simple example of a data set due
to github's limited data volume". The real dataset is the figshare archive.

Two files in `model_data/` are worth quoting because they characterise the
problem the original authors were solving:

```
model_data/sperm_classes.txt      ->  S
                                      Impurity

model_data/sperm_anchors.txt      ->  7,11  8,15  9,10  10,14  12,11  13,19
```

The shipped detector is therefore **two-class**, and **every anchor is under
20 px** -- the tiny-object regime. That is the opposite end of the scale space
from this project, where a sperm head at the reference optics spans ~119 x 81 px.

### 5.2 MIaMIA-SVDS (also called SVIA)

| Subset | Content | Task |
|---|---|---|
| A | >125,000 objects, box + category, 101 videos | detection |
| B | >26,000 sperms **segmented**, 10 videos | tracking ground truth |
| C | >125,000 cropped images | classification |

Total >278,000 annotated objects, annotated by 14 experts and verified by 6.
Object sizes ~5-50 um². Impurities include bacteria, protein clumps and bubbles
-- which is what makes Subset A useful as a *debris* reference even though its
optics do not match.

### 5.3 Splits

6:2:2 **by video**: 2125 train / 668 validation / 829 test images, from 21
videos.

### 5.4 Capture conditions

WLJY-9000 CASA system, 20x objective plus 20x electronic eyepiece, 30 FPS, clips
of 1-3 s. **Native video resolution is UNVERIFIED** -- the 416x416 that appears
in the paper is a network resize target, not a capture resolution.

### 5.5 Annotation format

Annotated with LabelImg, which implies VOC XML at annotation time. **The on-disk
format of the released archive is UNVERIFIED.**

### 5.6 Toolchain

Keras 2.1.5 on TensorFlow 1.13.1, Python 3.7 -- an end-of-life TF1 stack. This
is a reproduction hazard rather than a data problem, and is part of why
`src/sperm_sorting/detection/todcnn.py` is an independent reimplementation of the
published *design argument* rather than a port. See `THIRD_PARTY_NOTICES.md`.

### 5.7 Licence

**UNCLEAR -- a genuine conflict.** No LICENSE file on GitHub; the README says
non-commercial research use is welcome; the figshare metadata tags CC BY 4.0.
Both statements are reported in `docs/license_audit.md` and neither is adopted.

---

## 6. What each dataset can and cannot supply

| | MHSMA | VISEM-Tracking | VISEM | VT-graphs | MIaMIA-SVDS |
|---|---|---|---|---|---|
| Video | no | yes | yes | no (derived) | yes |
| Per-sperm bounding boxes | **no** | **yes** | no | yes (derived) | yes (Subset A) |
| Per-sperm track identity | no | **yes** | no | yes (defective in the video graph) | yes (Subset B) |
| Per-sperm segmentation | no | no | no | no | yes (Subset B) |
| Per-sperm morphology labels | **yes (4 aspects)** | **no** | no | no | class only (sperm / impurity) |
| Per-sperm motility grade | no | no | no | no | no |
| Sample-level WHO analysis | no | no | **yes** | no | no |
| Debris / impurity class | no | cluster + pinhead | no | yes | **yes, explicitly** |
| Frames | 1540 crops | 29,196 | ~85 x 2-7 min | 29,196 (as graphs) | 3622 (split total) |
| Magnification | x400 / x600 | 400x | 400x | 400x | 20x obj + 20x eyepiece |
| Contrast method | stained smear | phase contrast | phase contrast | phase contrast | CASA system optics |
| Frame rate | n/a | 45-50 (non-uniform) | 50 | n/a | 30 |
| Licence | CC BY-NC-SA 4.0 | CC BY 4.0 | CC BY-NC 4.0 | CC BY 4.0 / MIT | **unclear** |

### 6.1 The gap

Read the two bolded rows together:

- **MHSMA has morphology and no boxes.** It is a bag of pre-cropped single-cell
  images with no video, no positions and no time.
- **VISEM-Tracking has boxes and track identity and no morphology.** Its only
  per-object label is a three-way class (sperm / cluster / pinhead).
- **VISEM has neither**, only ejaculate-level percentages.

So **no public dataset provides per-sperm bounding boxes and per-sperm morphology
for the same cell.** The label spaces are disjoint.

This matters because the product's rule is a *conjunction over one cell*:

```
ai_eligible  =  progressive motility  AND  all four morphology aspects normal
```

Each half can be trained separately from public data. The **conjunction cannot
be measured** from public data, because measuring it requires knowing, for a
single spermatozoon, both how it swam and what it looked like. Training a
detector on VISEM-Tracking and a classifier on MHSMA and then reporting an
end-to-end accuracy figure would be reporting a number that no dataset supports.

It is also not fixable by pairing the datasets. Correlating a VISEM-Tracking
track with an MHSMA crop would join two different cells from two different
patients imaged on two different microscopes; and the deeper obstacle, discussed
in `docs/safety_and_claims.md`, is that WHO strict morphology requires a fixed,
stained, dead cell, so *the cell you assay is never the cell you use*.

### 6.2 Why the simulator exists

`src/sperm_sorting/simulator/` closes exactly this gap, and only this gap. It
samples a ground-truth `HealthState` first and derives everything observable
from it, so one sample is jointly labelled for morphology **and** motion at zero
annotation cost:

- `params.py` samples the state. The causality rule is that a binary morphology
  flag *causes* its continuous knob to leave the normal band -- if flags were
  sampled independently of appearance there would be nothing in the pixels to
  learn.
- `render.py` turns the state into pixels (brightfield convention: objects
  darker than the background, with anti-aliasing, at 64x64 and 128x128 so results
  are directly comparable with MHSMA's two crop variants).
- `motility.py` turns the same state into a trajectory, and carries a second,
  deliberately naive CASA implementation written from the definitions rather
  than from `sperm_sorting.motion`, so that a shared misunderstanding between
  labeller and estimator cannot hide.
- `label.py` holds the health rule in exactly one place, so the dataset builder,
  the demo and the evaluation harness cannot drift apart.
- `scene.py` is the multi-sperm frame source for the whole pipeline. It knows
  every sperm's health state, its true track identity and the true bulk flow, so
  detection, tracking, flow correction, motility grading, morphology, shot
  assembly and the decision rule can all be scored end to end against numbers
  correct by construction. One agent keeps one `track_id` from spawn to despawn
  and ids are never reused -- without that, no identity-switch or fragmentation
  metric would mean anything.
- `generate.py` writes the on-disk bootstrap set: per sample, one ground-truth
  `HealthState`, its rendered image, its CASA feature vector and all four label
  sets, stored together. Weights trained on it are
  `WEIGHTS_PROVENANCE_SYNTHETIC` and must never be presented as
  device-validated.

The simulator is therefore the only source against which the end-to-end
`ai_eligible` rule can be measured, which is why `configs/synthetic.yaml` pairs
it with the oracle detector: pinning detection quality to a known value is what
makes an error attributable to the stage under test.

Its limits are equally clear and are stated in
`acquisition/synthetic.py`: timestamps are exact by construction, so velocity
estimates are *better* on synthetic data than on hardware, and **a model that
only works on synthetic data should be assumed not to work on a camera**.

Its normal ranges are taken from WHO strict criteria where a WHO number exists,
and are **documented modelling choices where one does not** -- see the
`params.py` docstring, which marks each. It is a model of sperm appearance and
motion, not a measurement of it.

---

## 7. Cross-cutting domain shift

Summarised here; quantified and matched to mitigations in `docs/domain_shift.md`.

| Axis | Public data | This instrument |
|---|---|---|
| Magnification | 400x (VISEM family), 20x obj + 20x eyepiece (MIaMIA), x400/x600 (HSMA-DS) | 100x oil, NA 1.25 |
| Sampling | sperm heads tens of px across | ~119 x 81 px per head |
| Contrast | phase contrast on unstained wet preps; or fixed stained smears | brightfield on oil |
| Frame rate | 30 / 45-50 / 50 Hz | 160 |
| Depth of field | comparatively deep (dry 400x) | far shallower at NA 1.25 |
| Sensor | UEye UI-2210C (mono/colour UNVERIFIED) | Sony IMX392LLR-C, mono, global shutter |
| Population | MHSMA: 235 male-factor-infertility patients | unspecified |

Two consequences follow directly and are already reflected in the code. Four to
five times lower magnification means **vacuole and acrosome detail is literally
absent** from the public morphology sources at the resolution this instrument
works at; and brightfield-on-oil resembles neither phase contrast (with its
halos and absence of absorption contrast) nor a stained smear.
