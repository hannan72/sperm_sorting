# Licence audit

Three different things are licensed here and they are routinely conflated:
**the code**, **the datasets**, and **the trained weights**. They are kept in
separate sections below because a licence that is fine for one can be fatal for
another, and because the relationship between the second and the third is
genuinely unsettled law rather than a matter anyone can look up.

**This is not legal advice.** It is an engineering record of what each source
actually says, where two sources disagree, and which readings are safe. Anything
commercial needs a lawyer, and the questions to put to them are listed in
section 5.

---

## 1. Per-asset licence table

| Asset | Licence | Commercial use | Share-alike | Attribution | Source of the statement |
|---|---|---|---|---|---|
| **This repository's code** | Apache-2.0 | yes | no | yes (NOTICE) | `pyproject.toml`, `license = { text = "Apache-2.0" }` |
| **MHSMA** | CC BY-NC-SA 4.0 | **no** | **yes** | yes | LICENSE in `soroushj/mhsma-dataset` |
| **VISEM-Tracking** | CC BY 4.0 | yes | no | yes | Zenodo record 7293726 |
| **VISEM** | CC BY-NC 4.0 | **no** | no | yes | Zenodo record 2640506 |
| **VISEM-Tracking-graphs (data)** | CC BY 4.0 | yes | no | yes | Hugging Face `SimulaMet-HOST/visem-tracking-graphs` |
| **VISEM-Tracking-graphs (generator code)** | MIT | yes | no | yes | same repository |
| **Detection-Sperm / MIaMIA-SVDS** | **UNCLEAR -- conflict** | **unknown** | unknown | yes under either reading | see section 1.1 |

### 1.1 The Detection-Sperm conflict, reported and not resolved

Three statements exist and they do not agree:

1. The GitHub repository `Demozsj/Detection-Sperm` has **no LICENSE file at all**.
2. Its **README states that non-commercial research use is welcome** -- which is
   a permission for one kind of use, not a licence grant covering others.
3. The **figshare record 15074253 tags the data CC BY 4.0**, which would permit
   commercial use.

Neither reading is adopted here. The two are not reconcilable by inspection:
figshare's licence field is metadata selected by an uploader, and a README
sentence is a statement of intent; a court would look at both plus the deposit
terms, and this repository has no basis for guessing the outcome. **Treat the
asset as non-commercial and unclear until the authors are asked in writing.**

The practical consequence for this project is small, because nothing from
Detection-Sperm is redistributed: `src/sperm_sorting/detection/todcnn.py` is an
independent reimplementation of the published *design argument*, and no weights,
no exported graph and no layer-by-layer transcription were taken. See
`THIRD_PARTY_NOTICES.md`.

### 1.2 Reading the flags

- **Commercial use = no** (`NC`) is the flag that stops a product. It is a
  restriction on the *use*, not only on redistribution, so an internal
  commercial pipeline built on NC data is already outside the licence even if
  nothing is published.
- **Share-alike = yes** (`SA`) is the flag that spreads. It requires adaptations
  to be released under the same terms. MHSMA is the only asset here carrying
  both, and it is also the only source of morphology labels -- which is exactly
  the combination that makes section 4 the important section of this document.
- **Attribution = yes** applies to every asset without exception, including the
  permissive ones. Attribution text is in `THIRD_PARTY_NOTICES.md`.

---

## 2. Code licensing

### 2.1 This repository

Apache-2.0, declared in `pyproject.toml`. Apache-2.0 is permissive: commercial
use, modification, distribution and private use are all permitted, with
conditions of attribution and stating changes, and it carries an express patent
grant.

### 2.2 Runtime dependencies

The full list, with licences, is in `THIRD_PARTY_NOTICES.md`. The runtime set
declared in `pyproject.toml` -- numpy, scipy, opencv-python-headless, pydantic,
pyyaml, typer, rich, pandas, pillow -- is entirely permissive (BSD, MIT,
Apache-2.0, HPND) and imposes no copyleft obligation on this project.

One item deserves a note: **opencv-python-headless** ships OpenCV, which is
Apache-2.0 for version 4.5.0 and later. Older OpenCV was 3-clause BSD; some
contrib modules carry additional third-party terms and are not included in the
headless wheel. The declared floor here is `>=4.8`, so the Apache-2.0 terms
apply.

Optional extras (`torch`, `onnxruntime`, `pypylon`, `gpiod`, `pyserial`,
FastAPI/uvicorn) are also permissive, with one operational caveat:
**pypylon is a binding to Basler's pylon SDK**, and the SDK itself is a vendor
runtime with its own licence terms that must be accepted separately. The binding
being open source does not make the SDK redistributable.

### 2.3 Reimplemented algorithms

ByteTrack, OC-SORT, BoT-SORT, CenterNet/CornerNet, TOD-CNN, MobileNetV3 and
EfficientNet are implemented here from their published descriptions. **No
third-party source was copied and no third-party weights are bundled.** The
architectural ideas themselves are not copyrightable; the specific expression of
them in the original repositories is, and none of that expression is present
here. Per-item provenance statements are in `THIRD_PARTY_NOTICES.md`.

A separate point, which is not a copyright point: `torchvision`'s
`weights="DEFAULT"` path for MobileNetV3 and EfficientNet downloads ImageNet
pretrained weights at first use. Those weights carry their own terms.
`src/sperm_sorting/morphology/backbones.py` takes `pretrained` as an explicit
argument and tests and CI always pass `pretrained=False`, so nothing is fetched
implicitly -- but a deployment that turns it on has taken on a dependency it
should record.

---

## 3. Dataset licensing

### 3.1 What each licence permits for training

| Dataset | May train on it for research | May train on it for a commercial product | May redistribute the data | May redistribute derivatives |
|---|---|---|---|---|
| MHSMA | yes | **no** | yes, under CC BY-NC-SA 4.0 | **only under CC BY-NC-SA 4.0** |
| VISEM-Tracking | yes | yes | yes, under CC BY 4.0 | yes, with attribution |
| VISEM | yes | **no** | yes, under CC BY-NC 4.0 | yes, under CC BY-NC 4.0 |
| VISEM-Tracking-graphs | yes | yes | yes, under CC BY 4.0 | yes, with attribution |
| Detection-Sperm / MIaMIA-SVDS | yes | **unknown -- assume no** | unclear | unclear |

### 3.2 Datasets are not redistributed by this repository

`datasets/adapters/`, `datasets/converters/` and `datasets/validators/` contain
*readers* only. No dataset bytes are vendored into this repository, and none
should be. Each dataset must be downloaded by the user from its own distribution
point, under its own terms, which is also the only way the user actually sees the
licence they are accepting.

The licence terms in section 1 are additionally encoded as a machine-readable
registry in **`datasets/validators/licenses.py`** -- `LicenseRecord`,
`get_license`, `check_commercial_use`, `check_share_alike`, `strictest_terms`,
`describe_licenses`. That module exists as code rather than as a paragraph for
one reason: **a model trained on a mixture inherits the strictest terms in the
mixture**, and "which datasets went into this checkpoint" is a question that gets
asked long after the person who knows the answer has moved on. `strictest_terms`
is what answers it mechanically.

### 3.3 Derived artefacts that are still "the dataset"

Under CC, an adaptation carries the licence. The following are adaptations of
MHSMA and inherit CC BY-NC-SA 4.0:

- a re-encoded, resized, augmented or format-converted copy of the crops;
- a cached tensor file or a webdataset shard built from them;
- a merged corpus that includes them;
- a per-image annotation file keyed to them.

The following are *not* obviously adaptations, and this is where section 4
begins: model weights, calibration bundles, and reported metrics.

---

## 4. Trained-weight licensing

### 4.1 The question

Does a copyleft or non-commercial condition on training data reach the model
weights produced from it?

**There is no settled answer, and the honest position is that it is
jurisdiction- and interpretation-dependent.** The competing readings:

**The "weights are not an adaptation" reading.** A neural network's parameters
are not a derivative work of the training images in the copyright sense. They do
not contain, reproduce or transform any particular image; they are statistics
over a corpus, and facts and statistics are not protected. On this reading
CC BY-NC-SA's share-alike condition never attaches to the weights, because the
weights are not an adaptation of the licensed material. Several
text-and-data-mining exceptions -- the EU DSM Directive Articles 3 and 4, and
the US fair-use tradition -- point in a compatible direction, at least for
research.

**The "training is a use" reading.** CC's non-commercial condition restricts the
*exercise of licensed rights*, and training requires reproducing the dataset --
copying it to disk, into memory, into a shuffled batch. If that reproduction is
done "primarily for commercial advantage", the NC condition is breached at the
moment of training, regardless of what the weights legally are. On this reading
the weights are downstream of a licence breach, and the analysis of whether they
are themselves an adaptation never has to be reached.

The second reading does not depend on resolving the first, which is why it is the
one that governs practice. It is also the reading that a rights-holder is most
likely to advance.

### 4.2 The safe reading, adopted here

> **Weights trained on MHSMA inherit MHSMA's restriction.** Treat them as
> non-commercial and share-alike. Do not ship them in a product, do not license
> them to a customer, and if they are published, publish them under
> CC BY-NC-SA 4.0 with attribution to MHSMA.

The same logic, minus share-alike, applies to VISEM: weights trained on VISEM are
non-commercial.

Weights trained on VISEM-Tracking or VISEM-Tracking-graphs alone are clean for
commercial use, subject to attribution.

Weights trained on MIaMIA-SVDS are **unknown**, and should be treated as
non-commercial until the conflict in section 1.1 is resolved.

Note the asymmetry that follows: the *detector* can be trained from a
commercially clean source (VISEM-Tracking, CC BY 4.0), while the *morphology
classifier* cannot, because MHSMA is the only public source of per-aspect
morphology labels and it is NC-SA. The commercial blocker is entirely on the
morphology side.

### 4.3 How this is tracked in the code

Provenance is a field, not a convention. `src/sperm_sorting/constants.py`:

```python
WEIGHTS_PROVENANCE_PUBLIC: Final[str] = "public-research-baseline"
WEIGHTS_PROVENANCE_DEVICE: Final[str] = "device-finetuned"
WEIGHTS_PROVENANCE_SYNTHETIC: Final[str] = "synthetic-bootstrap"
```

`MorphologyConfig.weights_provenance` is stamped into every checkpoint, copied
onto every `MorphologyResult`, echoed by `AppConfig.summary()`, and therefore
written into the manifest of every audit log. A run whose weights were derived
from NC data is identifiable from its own log, after the fact, without anyone
having to remember.

The default value is the literal string `"unset"`, which is deliberately not a
valid provenance: a run that never declared where its weights came from says so.

### 4.4 Mixed training sets

A model fine-tuned from MHSMA-derived weights onto device data is still
MHSMA-derived. Initialisation carries the same problem as training: the
restricted material is in the starting point. The only way out is a clean
lineage all the way down, which is section 5.

---

## 5. A path to a commercially clean system

The restriction is a data restriction, so the remedy is a data remedy. There is
no licence-side trick.

### 5.1 The lineage

1. **Capture device data and own it.** Images produced on the instrument, from
   samples collected under an appropriate consent and ethics framework, with the
   rights assigned to the operator. This is the foundation; everything else is
   optional.
2. **Annotate it, or commission annotation under a work-for-hire or
   assignment agreement.** Boxes and track identity for the detector; four-aspect
   morphology for the classifier. The MHSMA aspect decomposition may be
   *followed* -- a scheme is not copyrightable -- but the labels must be produced
   independently, not transcribed.
3. **Alternatively, licence data commercially.** Several clinical CASA vendors
   and academic biobanks will licence annotated semen microscopy under negotiated
   terms. This is usually faster than annotating from scratch and gives a written
   grant to point at.
4. **Train end to end on that corpus**, from random initialisation or from a
   generically-licensed backbone (ImageNet weights carry their own terms, which
   are permissive but must be recorded). Never from an MHSMA-derived checkpoint.
5. **Bootstrap with the simulator, not with restricted data.** The synthetic
   source in `src/sperm_sorting/simulator/` is this project's own code producing
   its own images: `WEIGHTS_PROVENANCE_SYNTHETIC` weights are clean. Synthetic
   pretraining followed by device fine-tuning is a clean lineage end to end.

### 5.2 What the public sets are still good for

Under this plan the public datasets keep two legitimate roles, both of which are
compatible with a non-commercial licence because neither produces a shipped
artefact:

- **Architecture selection.** Deciding between P2Net and the TOD-CNN-style
  backbone, between ByteTrack and OC-SORT, between MobileNetV3 and EfficientNet,
  is a research question answered on research data. The *decision* is not a
  derivative work; the weights that informed it are discarded.
- **Sanity checking.** A detector that cannot find sperm in VISEM-Tracking has a
  problem worth knowing about before device data is spent on it. A morphology
  head whose MHSMA balanced accuracy is at chance is broken.

Both uses are research uses, and both must be kept clearly separate from the
shipped model -- which the `weights_provenance` field makes auditable.

### 5.3 Questions for a lawyer

Listed so they are not rediscovered:

1. In our jurisdictions of operation, does training on CC BY-NC-SA material for
   an eventual commercial product breach the NC condition at training time, even
   if the weights are never held to be an adaptation?
2. Do the EU DSM Articles 3/4 text-and-data-mining exceptions apply to us, and
   does Article 4's rights-reservation opt-out interact with a CC NC term?
3. Does share-alike attach to weights in any jurisdiction we ship into?
4. What is the status of a model initialised from NC-derived weights and then
   fine-tuned exclusively on owned data?
5. For Detection-Sperm, can a written clarification be obtained from the authors,
   and what does the figshare deposit agreement say about the licence field?
6. What consent and ethics framework does device-data capture require in each
   jurisdiction, and does it permit model training as a secondary use?

---

## 6. Attribution obligations

Every asset in section 1 requires attribution, including the permissive ones.
The attribution text lives in `THIRD_PARTY_NOTICES.md`; the citable references
live in `CITATION.cff`. A publication or a released artefact that uses any of
these datasets must cite them, and CC BY additionally requires indicating
whether changes were made -- which for a converted or re-split copy is always
yes.
