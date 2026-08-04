# Datasets

> **No dataset is redistributed in this repository.**
>
> Nothing under `datasets/` contains image, video or annotation data, and nothing
> here downloads any on your behalf. Several of these corpora are licensed
> **non-commercial**, one has **contradictory** licence terms, and all of them
> are large (VISEM alone is 35 GB). You obtain each one yourself, from its own
> source, under its own licence. `.gitignore` excludes `datasets/raw/` and
> `datasets/cache/` so a downloaded copy cannot be committed by accident.
>
> Every adapter imports, constructs and fails *usefully* with no data present:
> a missing dataset raises `DatasetNotFoundError` naming the path searched, the
> download URL, the licence and the expected layout.

---

## Quick start

```python
from datasets import get_adapter, list_adapters

list_adapters()
# ['detection_sperm', 'device', 'mhsma', 'visem', 'visem_graphs', 'visem_tracking']

adapter = get_adapter("mhsma")("~/data/mhsma-dataset")
report = adapter.validate()          # raises DatasetValidationError if polarity is inverted
print(report.format_text())
print(adapter.prevalence("train"))   # ABNORMAL prevalence, as a fraction
```

Before training anything you intend to sell:

```python
from datasets.validators.licenses import check_commercial_use

blockers = check_commercial_use(["mhsma", "visem_tracking"])
if blockers:
    raise SystemExit("\n".join(blockers))
```

---

## The five things most likely to go wrong

1. **MHSMA labels are `0 = normal`, `1 = abnormal`.** The upstream README calls
   the *normal* class "positive", so its "% Positive" column is the percentage
   of **normal** cells. Read it the natural way and every figure inverts.
   `MhsmaAdapter.validate()` measures the prevalence and **raises** if it looks
   flipped — it does not merely report it.
2. **`labels_ftid/` in VISEM-Tracking puts the tracking ID first**:
   `sperm_id class x_center y_center width height`. Parsing it as plain YOLO
   swaps class and identity, and both fields are small non-negative integers, so
   nothing downstream notices.
3. **Splitting by frame instead of by video/patient leaks.** VISEM-Tracking's
   29,196 frames come from 20 videos; a random frame split puts a near-identical
   twin of most validation frames into training. Use
   `datasets.validators.leakage.patient_level_split`.
4. **VISEM has no per-sperm labels.** Its motility percentages are per sample.
   Assigning them to individual detections fabricates every label.
   `VisemAdapter.sample_level_only is True`, and the adapter has no per-sperm
   interface at all.
5. **Public weights are baseline research weights.** Every dataset here was
   captured on a different microscope from the device. See *Domain shift* below
   and `constants.WEIGHTS_PROVENANCE_PUBLIC`.

---

## The corpora

| name | supervision | size | licence | commercial use |
|---|---|---|---|---|
| `mhsma` | 4 binary morphology aspects per image | ~30 MB | CC BY-NC-SA 4.0 | **no** (+ share-alike) |
| `visem_tracking` | boxes + track IDs, 3 classes | ~35 GB | CC BY 4.0 | yes |
| `visem` | sample-level WHO / motility only | 35.2 GB | CC BY-NC 4.0 | **no** |
| `visem_graphs` | graphs derived from VISEM-Tracking | a few GB | CC BY 4.0 (data), MIT (code) | yes |
| `detection_sperm` | boxes, 2 classes (sperm / debris) | 1.42 GB | **contradictory** | **unclear — blocked** |
| `device` | boxes + tracks + morphology, with capture metadata | n/a | internal | yes |

The machine-readable version of this table lives in
`datasets/validators/licenses.py` and is the single source of truth;
`check_commercial_use()` fails closed on the unclear one.

---

### MHSMA — Modified Human Sperm Morphology Analysis

**Get it:** <https://github.com/soroushj/mhsma-dataset> — the `.npy` files are
committed in that repository, so `git clone` is the download. It is **not** on
Zenodo.

**Cite:** Javadi & Mirroshandel (2019), *A novel deep learning method for
automatic assessment of human sperm images*, Computers in Biology and Medicine
109:182–194.

1,540 grayscale crops of single sperm from 235 male-factor-infertility patients,
at 128×128 and 64×64, each labelled independently for head, acrosome, vacuole
and tail. Derived from HSMA-DS (Ghasemian et al., 2015), captured at ×400 and
×600. The head is roughly centred; **the tail is not entirely visible**.

**Layout** (`<root>` may be the clone *or* the `mhsma/` folder inside it):

```
<root>/mhsma/
    x_128_train.npy   (1000, 128, 128)  uint8
    x_128_valid.npy   ( 240, 128, 128)  uint8
    x_128_test.npy    ( 300, 128, 128)  uint8
    x_64_train.npy    (1000,  64,  64)  uint8
    x_64_valid.npy    ( 240,  64,  64)  uint8
    x_64_test.npy     ( 300,  64,  64)  uint8
    y_{acrosome,head,vacuole,tail}_train.npy   (1000,)  uint8
    y_{acrosome,head,vacuole,tail}_valid.npy   ( 240,)  uint8
    y_{acrosome,head,vacuole,tail}_test.npy    ( 300,)  uint8
```

Eighteen files, every one `uint8`. The split directory naming is **`valid`**,
not `val`; the adapter accepts both and normalises to `valid`.

**Published ABNORMAL prevalence (%)** — the numbers `validate()` checks against:

| split | acrosome | head | vacuole | tail |
|---|---|---|---|---|
| train (1000) | 30.10 | 27.30 | 17.00 | 4.60 |
| valid (240) | 27.50 | 26.67 | 12.92 | **2.92** |
| test (300) | 29.00 | 27.00 | 12.67 | 5.33 |

That 2.92% is **7 abnormal tails in the entire validation split**. Any tail
metric computed on it has an enormous confidence interval and must not be
reported without one.

**Leakage status: UNVERIFIABLE, not absent.** MHSMA publishes no patient, slide
or field identifier, so it is impossible to confirm from the released files that
the official 1000/240/300 split is patient-level. Every MHSMA validation report
carries an explicit `UNVERIFIABLE` check saying so. Treat MHSMA scores as
benchmark-comparable, not as evidence of generalisation to unseen patients — and
do not re-split these files, because the grouping key needed to do it safely is
not published.

**Licence: CC BY-NC-SA 4.0** — non-commercial **and** share-alike. The
share-alike term is the one people forget: a checkpoint trained on MHSMA is
arguably an adapted work.

---

### VISEM-Tracking

**Get it:** <https://zenodo.org/records/7293726>

**Cite:** Thambawita et al. (2023), *VISEM-Tracking: a human spermatozoa tracking
dataset*, Scientific Data 10:260 (arXiv:2212.02842).

20 annotated 30-second clips from 20 patients, 640×480, **45–50 FPS and not
uniform across videos**; 29,196 annotated frames, 656,334 boxes, 1,121 unique
sperm track IDs plus 20 cluster and 35 pinhead IDs. Three classes:
`0 = sperm`, `1 = cluster`, `2 = small or pinhead`.

**Layout** (`<root>` may be any of the three levels shown):

```
<root>/VISEM_Tracking_Train_v4/Train/<video_id>/
    <video_id>.mp4
    images/                  extracted frames (loose frames in the video folder
                             are also handled)
    labels/<frame>.txt       class x_center y_center width height
    labels_ftid/<frame>.txt  sperm_id class x_center y_center width height
```

Coordinates are **YOLO-normalised**; the adapter converts to absolute pixels
once, using each video's measured frame size.

> **`labels_ftid` has the tracking ID first, then the class.** This is encoded in
> exactly one function (`parse_labels_ftid_line`) and cross-checked by
> `validate()`, which fails if any parsed class falls outside `{0, 1, 2}`.

**Official split: 16 train / 4 validation, by video. Validation = 82, 60, 54, 52.**
There is **no official test split** — any test number on this dataset comes from
a split you invented and must be described as such. `official_split()` returns
the published constant; pass `restrict_to_present=True` to intersect it with
what is on your disk.

**Known quirks** (all measured by `quirk_report()` / `validate(deep=True)`):

- `video_23` has 174 frames with no sperm at all.
- Frame counts differ: `video_35` and `video_52` have 1,440; `video_82` has 1,500.
- Boxes concentrate in the **upper-left** of the frame — a real spatial prior a
  detector will overfit.

The published YOLOv5l baseline is **mAP@0.5 = 0.2231**. That is what tiny, dense,
low-contrast objects look like; treat a substantially higher number on a
self-made split as evidence of leakage until proven otherwise.

**Licence: CC BY 4.0** — commercial use permitted.

---

### VISEM (sample-level)

**Get it:** <https://zenodo.org/records/2640506>

**Cite:** Haugen et al. (2019), *VISEM: A Multimodal Video Dataset of Human
Spermatozoa*, MMSys'19, 261–266.

85 participants, one 2–7 minute video each, 640×480, 50 FPS, AVI, 35.2 GB.
Annotations are **sample/video level only** — WHO semen analysis, motility
percentages (progressive / non-progressive / immotile), concentration, fatty
acids, sex hormones and demographics, in six CSVs.
**No per-sperm labels, no bounding boxes.**

```
<root>/semen_analysis_data.csv
<root>/participant_related_data.csv
<root>/sex_hormones.csv
<root>/fatty_acids_spermatozoa.csv
<root>/fatty_acids_serum.csv
<root>/videos.csv
<root>/videos/*.avi        (optional; this is the 35 GB)
```

Exact filenames and column headers vary between downloads, so the adapter matches
them fuzzily and *reports what it found*; a column it cannot locate raises with
the list of headers that are present rather than guessing.

**Fabricating per-sperm labels from these percentages is not supported.** A
sample that is "62% progressive" says nothing about *which* sperm are
progressive. `VisemAdapter.sample_level_only is True`, and asking the adapter for
`.detections`, `.tracks`, `.crops` or `.labels` raises an `AttributeError` that
explains why. What VISEM *is* good for: comparing an aggregate the pipeline
produces (progressive fraction over a whole video) against a laboratory-measured
aggregate for the same sample.

**Licence: CC BY-NC 4.0** — non-commercial.

---

### VISEM-Tracking-graphs (optional extension)

**Get it:** <https://huggingface.co/datasets/SimulaMet-HOST/visem-tracking-graphs>

GraphML read with networkx (an **optional** dependency — the MVP does not need
it, and nothing in `sperm_sorting` imports this adapter).

```
<root>/spatial_threshold_{0.1,0.2,0.3,0.4,0.5}/
    <video_id>/frame_graphs/frame_graph_{i}.graphml
    <video_id>/video_graph.graphml
```

Node id is `sperm_id`; node attributes are `frame_number`, `class_name` (the YOLO
class index as a string), `x_center`, `y_center`, `width`, `height`, all
YOLO-normalised. Edges are spatial (`weight` = Euclidean distance between
normalised centres, added below the threshold) or temporal
(`edge_type="temporal"`). Frame graphs are undirected; the video graph is
directed.

> **Known upstream defect.** `video_graph.graphml` keys nodes by `sperm_id`
> alone rather than `(sperm_id, frame)`, so every frame of a track collapses onto
> one node and the temporal-edge loop produces **self-loops**. The per-frame
> graphs are sound. `VisemGraphsAdapter.regenerate_video_graph(video_id)`
> rebuilds the video graph correctly using `(video_id, frame_id, track_id)` as
> node identity; `inspect_upstream_video_graph(video_id)` measures the defect on
> your copy.

**Licence:** data CC BY 4.0, generator code MIT. Derived from VISEM-Tracking, so
it also inherits that dataset's attribution requirement.

---

### Detection-Sperm / TOD-CNN / MIaMIA-SVDS

**Model repository:** <https://github.com/Demozsj/Detection-Sperm> — this is a
*model* repo, not a dataset repo. Its `model_data/sperm_classes.txt` is exactly
two lines, `S` and `Impurity`, and its anchors are all under 20 px
(7,11 8,15 9,10 10,14 12,11 13,19).

**Data:** MIaMIA-SVDS on figshare, record 15074253, a 1.42 GB `Data Set.rar`:
Subset-A (>125,000 objects with boxes and categories, 101 videos), Subset-B
(>26,000 segmented sperms, 10 videos), Subset-C (>125,000 cropped classification
images).

Capture: WLJY-9000 CASA, 20× objective plus a 20× electronic eyepiece, 30 FPS,
clips of 1–3 s.

> **UNVERIFIED, and treated as such in code:**
> - the **on-disk annotation format**. Annotation used LabelImg, which implies
>   Pascal VOC XML, but the contents of the `.rar` have not been confirmed. The
>   adapter therefore **sniffs** the format (VOC XML / YOLO txt / COCO JSON) by
>   inspecting file contents and reports the evidence; an ambiguous layout raises
>   rather than picking silently.
> - the **native video resolution**. The 416×416 in the paper is a network input
>   size after resizing. For a YOLO release the adapter *requires* an explicit
>   `frame_size=(w, h)` rather than assuming one — a wrong size silently rescales
>   every box.

Class mapping: `S` → `sperm` (0), `Impurity` → `debris` (1). Keeping debris as an
explicit class is what makes a false-positive rate measurable instead of assumed.

> **LICENCE CONFLICT — treated as `UNCLEAR`, which blocks commercial use.**
> There is no LICENSE file in the GitHub repository; the README welcomes
> non-commercial research use; the figshare metadata carries CC BY 4.0. This
> repository does not pick a side. `check_commercial_use()` reports it as a
> blocker; resolve it with the authors in writing before any commercial use.

---

### Device captures

Not downloaded — produced by the instrument. This is the only in-domain corpus
and the only one with no third-party licence constraint, and it is what the
domain-adaptation path consumes.

**Format: JSON Lines, header first.**

```
<root>/*.jsonl                    (or <root>/annotations/*.jsonl)
<root>/frames/...                 images referenced by image_path
```

```jsonc
{"record_type":"header","schema_version":"1.0.0","session_id":"s001",
 "capture":{"sample_id":"S-001","operator":"hk","um_per_px":0.31,
            "frame_rate_hz":160.0,"exposure_us":800.0,"gain_db":6.0,
            "temperature_c":37.0,"camera_model":"acA1920-155um","objective":"20x"},
 "morphology_aspects":["head","acrosome","vacuole","tail"],
 "label_encoding":{"normal":0,"abnormal":1}}
{"record_type":"frame","frame_id":0,"capture_time_s":0.0,"width":1920,"height":1200,
 "image_path":"frames/000000.png",
 "boxes":[{"box_xyxy":[100.0,200.0,130.0,240.0],"class_id":0,"class_name":"sperm",
           "track_id":7,"score":1.0,
           "morphology":{"head":0,"acrosome":1,"vacuole":null,"tail":0}}]}
```

- JSON Lines because a capture is appended to while it runs: an interrupted file
  is still readable up to its last complete line.
- Capture metadata lives in the **header** because it is a property of the
  session; the three fields that can genuinely drift (`exposure_us`, `gain_db`,
  `temperature_c`) may be overridden per frame via `capture_overrides`.
- **Required** capture metadata: `sample_id`, `operator`, `um_per_px`,
  `frame_rate_hz`, `exposure_us`, `gain_db`. `validate()` fails without them.
  `um_per_px` is required because without it no velocity can be expressed in the
  µm/s the WHO motility thresholds are defined in.
- **Optional but warned about:** `temperature_c`. Motility is strongly
  temperature-dependent, so motility compared across captures at unknown
  temperatures is not a like-for-like comparison.
- A morphology aspect that was **not assessed** must be `null` or absent —
  never `0`. "Nobody looked" and "looked, and it was normal" are different facts.

Write with `DeviceAnnotationWriter`, read with `DeviceDatasetAdapter`.

---

## Domain shift: why public weights are *baseline research weights*

Every public corpus here was captured on somebody else's microscope. The
per-dataset `CaptureConditions` (magnification, contrast mode, staining, camera,
frame rate, resolution, µm/px) and `domain_shift_notes` record exactly how, and
`CaptureConditions.differences_from()` enumerates the gap against a device
session field by field. The headlines:

- **MHSMA** is *stained, fixed* smears at mixed ×400/×600 with no per-image
  record of which; the device images live, unstained cells. Head contrast comes
  from dye uptake there and refractive index here.
- **VISEM / VISEM-Tracking** are 640×480 — a sperm head is a handful of pixels,
  so a detector trained there is tuned for a blob, not a shape — at 45–50 FPS
  against a much faster device camera, on free-swimming cells in a static
  chamber rather than cells in a flow.
- **MIaMIA-SVDS** anchors are all under 20 px at 30 FPS through a two-stage
  20×+20× path.

Consequently, a checkpoint trained only on these is stamped
`WEIGHTS_PROVENANCE_PUBLIC` and must never be presented as device-validated.

---

## Building a split that does not leak

```python
from datasets.validators.leakage import (
    assert_no_frame_leakage, check_adjacent_frames, patient_level_split,
)

split = patient_level_split(items, ratios={"train": 0.8, "val": 0.2}, seed=0)
assert_no_frame_leakage(split["train"], split["val"])          # raises LeakageError

# Even a per-clip split leaks if clips came from one recording:
check_adjacent_frames(split["train"], split["val"], max_gap=50).raise_if_leaked()
```

`patient_level_split` groups first and splits *groups*, assigning largest-first
to whichever split has the biggest item deficit — so an uneven set of videos
still lands within one group's worth of the requested ratio. Its output is
self-checked for leakage before it is returned.

`max_gap` is a frame count, not a time. At 50 FPS, `max_gap=1` catches only
literal neighbours; ~50 is the realistic guard against near-duplicate content.

---

## Converters

```python
from datasets.converters import (
    yolo_to_detections, detections_to_yolo,     # YOLO  <-> Detection
    voc_to_detections, detections_to_voc,       # VOC   <-> Detection
    coco_to_detections, detections_to_coco,     # COCO  <-> Detection
    tracks_to_mot, mot_to_tracks,               # tracks <-> MOTChallenge
    CropDatasetBuilder,                         # boxes -> morphology crops
)
```

Everything routes through `sperm_sorting.schemas.detection.Detection`
(`x1 y1 x2 y2`, absolute pixels, `x2`/`y2` exclusive). Box round-trips are
lossless to double precision — which is why full float precision is written by
default; the conventional `%.6f` YOLO format loses ~0.3 px on a 640-wide frame,
a fifth of a sperm head.

Convention traps handled explicitly rather than assumed:

- **VOC** is 1-based and inclusive in the original specification; several modern
  tools write 0-based coordinates into the same tags, and nothing in the file
  says which. `voc_to_detections(..., origin=...)` makes the choice visible.
- **MOTChallenge** frames and IDs are **1-based**. `frame_offset` defaults to 1
  on both write and read, and `write_mot_file` drops a `.meta.json` sidecar
  recording the offsets and the class-name mapping so a file rereads correctly.
- **COCO** category IDs are arbitrary integers, so they are resolved through the
  `categories` table rather than used as class indices.

`CropDatasetBuilder` cuts crops with the pipeline's own
`sperm_sorting.cropping.extractor.CropExtractor` and scores frames with
`sperm_sorting.quality.frame_score.score_candidate` — imported, not
reimplemented, so a training crop is cut by the same padding and letterbox rules
as an inference crop. It **never fabricates a morphology label**: crops from a
detection-only dataset are written with `"morphology": null`.

---

## Validating a copy

```python
from sperm_sorting.errors import DatasetValidationError

report = adapter.validate()
print(report.format_text())
report.raise_on_failure(DatasetValidationError)   # opt in to hard failure
```

`ValidationReport` has five statuses, not two, because forcing an unknown into
PASS/FAIL is how "we could not check this" becomes "this is fine":

| status | meaning |
|---|---|
| `PASS` | checked, correct |
| `FAIL` | checked, wrong — the only status that makes `report.ok` false |
| `WARN` | legal but likely to bite (an incomplete copy, a nearly-empty class) |
| `UNVERIFIABLE` | matters, and **cannot** be checked from the published files |
| `SKIPPED` | not applicable to this copy |

`UNVERIFIABLE` is why MHSMA's patient-level split never reports as a pass.

The one exception to "report, don't raise" is MHSMA label polarity: it raises,
because a report can be ignored and an inverted morphology classifier is a
device that sorts out exactly the cells it was built to keep.
