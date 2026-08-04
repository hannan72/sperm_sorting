# Third-party notices

`sperm-sorting-ai` is licensed under **Apache-2.0** (`pyproject.toml`). This file
records the third-party software, datasets and published work it depends on or
draws from.

Two statements that govern everything below:

> **No third-party model weights are bundled with this repository.**
>
> **The detector, tracker and classifier architectures here are independent
> reimplementations of published concepts. No third-party source code was
> copied.**

Licensing analysis -- including the unresolved conflict on Detection-Sperm and
the question of whether dataset restrictions reach trained weights -- is in
`docs/license_audit.md`. Provenance and verification status for every dataset and
hardware source is in `docs/source_audit.md`.

---

## 1. Python dependencies

Taken from `pyproject.toml`. Version and licence columns were read from the
package metadata of the environment in this repository (`.venv`) on 2026-08-04,
except where marked "not installed here".

### 1.1 Runtime (`[project.dependencies]`)

| Package | Version installed | Licence (from package metadata) |
|---|---|---|
| numpy | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.18.0 | BSD-3-Clause |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 |
| pydantic | 2.13.4 | MIT |
| PyYAML | 6.0.3 | MIT |
| typer | 0.27.1 | MIT |
| rich | 15.0.0 | MIT |
| pandas | 3.0.5 | BSD-3-Clause |
| pillow | 12.2.0 | MIT-CMU (HPND) |

### 1.2 Optional extras

| Package | Extra | Version installed | Licence |
|---|---|---|---|
| torch | `torch`, `train` | 2.13.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torchvision | `torch`, `train` | 0.28.0+cpu | BSD-3-Clause |
| onnx | `onnx` | 1.22.0 | Apache-2.0 |
| onnxruntime | `onnx` | 1.28.0 | MIT |
| scikit-learn | `train` | 1.9.0 | BSD-3-Clause |
| matplotlib | `train` | 3.11.1 | Matplotlib licence (PSF-based, BSD-compatible) |
| tensorboard | `train` | 2.21.0 | Apache-2.0 |
| tqdm | `train` | 4.70.0 | MPL-2.0 AND MIT |
| fastapi | `web` | 0.141.1 | MIT |
| uvicorn | `web` | 0.52.1 | BSD-3-Clause |
| python-multipart | `web` | 0.0.32 | Apache-2.0 |
| pytest | `dev` | 9.1.1 | MIT |
| pytest-cov | `dev` | 7.1.0 | MIT |
| hypothesis | `dev` | 6.165.0 | MPL-2.0 |
| httpx2 | `dev` | 2.9.1 | BSD-3-Clause |
| ruff | `dev` | 0.16.1 | MIT |
| mypy | `dev` | 2.3.0 | MIT |
| **pypylon** | `camera` | not installed here | declared BSD-3-Clause upstream -- **not verified from installed metadata** |
| **gpiod** (libgpiod Python bindings) | `gpio` | not installed here | declared LGPL-2.1-or-later upstream -- **not verified from installed metadata** |
| **pyserial** | `serial` | not installed here | declared BSD-3-Clause upstream -- **not verified from installed metadata** |

### 1.3 Notes on specific dependencies

**pypylon and the Basler pylon SDK.** `pypylon` is a Python binding. **The pylon
SDK it binds to is separate vendor software with its own licence terms, which
must be accepted separately and which are not covered by the binding's licence.**
The SDK is not redistributable simply because the binding is open source. The
device documentation this project relied on -- exposure limits, ROI parameters,
timestamp chunks -- is Basler's, cited in `docs/source_audit.md` section 3.1.

**gpiod / libgpiod.** libgpiod is **LGPL**, which carries obligations that the
permissive licences above do not: distribution of a combined work must permit the
recipient to relink against a modified libgpiod. The Python bindings are used
dynamically and only inside `actuation/gpio.py`'s `open()` method, and the
character-device interface was chosen over the deprecated sysfs interface and
over the Raspberry-Pi-specific `RPi.GPIO` because the target board is not fixed.
Anyone distributing a device image should confirm their LGPL position.

**opencv-python-headless.** The wheel packages OpenCV, which is Apache-2.0 from
version 4.5.0 onward (earlier releases were 3-clause BSD). Some `contrib` modules
carry additional third-party terms; they are not included in the headless wheel.

**tqdm and hypothesis** are MPL-2.0 (tqdm dual with MIT). MPL-2.0 is file-level
copyleft and imposes no obligation on this project's own files.

**httpx2.** The `dev` extra declares `httpx2`, not `httpx`. The installed
package's metadata is coherent -- version 2.9.1, BSD-3-Clause, author-email Tom
Christie, maintainer "Pydantic Services Inc.", homepage
`github.com/pydantic/httpx2` -- and the licence above was read from it. Note
however that the accompanying comment in `pyproject.toml`
("starlette.testclient requires httpx2, not httpx") was **not verified** during
this audit, and `[tool.pytest.ini_options]` still describes the `web` marker as
"requires FastAPI and httpx". Confirm which client Starlette's `TestClient`
actually requires before relying on either statement.

**torchvision pretrained weights.** `morphology/backbones.py` can initialise
MobileNetV3 or EfficientNet from torchvision's ImageNet weights, which are
downloaded at first use and **carry their own terms**. The parameter is explicit
(`pretrained: bool`), tests and CI always pass `pretrained=False`, and nothing is
fetched implicitly -- but a deployment that enables it has taken on a dependency
that belongs in its own notices.

---

## 2. Datasets

None of these is redistributed by this repository. Each must be obtained from its
own distribution point, under its own terms. Full technical detail is in
`docs/dataset_audit.md`; licensing analysis is in `docs/license_audit.md`.

| Dataset | Source | Licence | Commercial | Share-alike |
|---|---|---|---|---|
| **MHSMA** | https://github.com/soroushj/mhsma-dataset | CC BY-NC-SA 4.0 | **no** | **yes** |
| **VISEM-Tracking** | Zenodo 7293726; *Sci Data* 2023; arXiv:2212.02842 | CC BY 4.0 | yes | no |
| **VISEM** | Zenodo 2640506 | CC BY-NC 4.0 | **no** | no |
| **VISEM-Tracking-graphs** | Hugging Face `SimulaMet-HOST/visem-tracking-graphs` | data CC BY 4.0; generator code MIT | yes | no |
| **Detection-Sperm / MIaMIA-SVDS (SVIA)** | https://github.com/Demozsj/Detection-Sperm ; figshare 15074253 | **UNCLEAR -- conflict** | **unknown** | unknown |

**MHSMA** is derived from HSMA-DS (Ghasemian 2015,
doi:10.1016/j.cmpb.2015.08.013). Its CC BY-NC-SA licence is **non-commercial and
share-alike**, and the safe reading is that weights trained on it inherit both
restrictions. `constants.WEIGHTS_PROVENANCE_PUBLIC` exists so that any such model
is identifiable from its own audit log.

**Detection-Sperm** has no LICENSE file on GitHub; its README states that
non-commercial research use is welcome; the figshare record's metadata tags CC BY
4.0. The three statements do not agree, and this repository adopts none of them
-- see `docs/license_audit.md` section 1.1. Nothing from that repository is
redistributed here.

**Note on Zenodo record 2640506:** it is **VISEM**, not MHSMA. This is a common
and consequential mix-up; see `docs/source_audit.md` correction C1.

---

## 3. Algorithms and architectures reimplemented here

Every entry below is an **independent reimplementation of a published concept**.
For each: no weights, no exported graph, no layer-by-layer transcription, and no
source code from the original authors' repositories. The architectural *ideas*
are not copyrightable; the original authors' *expression* of them is, and none of
it is present here.

Any accuracy number this repository ever produces is a property of *this* code
and *its* training run, and **must never be reported as a reproduction of a
published result**.

> **Citation completeness.** Only two identifiers below are taken from a source
> verified in this repository: the CenterNet arXiv id, which is cited in
> `detection/heads.py`, and the TOD-CNN arXiv id, which is cited in
> `detection/todcnn.py` and confirmed in `docs/source_audit.md`. The remaining
> bibliographic details are marked **‡** and were not verified against a primary
> source during the audit. Verify them before publication.

### 3.1 Detection

**CenterNet / "Objects as Points"** -- Zhou, Wang & Krähenbühl, arXiv:1904.07850.
*Cited in `detection/heads.py`.*

The anchor-free head shared by both torch detectors. The design argument, as the
module records it: a sperm head at the configured optics is a small,
near-isotropic blob with almost no scale variation and no meaningful
aspect-ratio variation. Anchors exist to cover a wide, unknown scale/ratio space;
here that space is effectively a point, so anchors buy nothing and cost an
assignment rule, an anchor-matching IoU threshold and a per-anchor duplication of
every head channel. A centre-heatmap head reduces detection to "find the local
maxima", which is cheaper and easier to keep deterministic.

The head is deliberately *shared* between `todcnn` and `p2net`, so a comparison
between them measures the backbone and nothing else.

**CornerNet** -- Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints",
ECCV 2018, arXiv:1808.01244. ‡

The lineage of the keypoint-heatmap formulation and of the corner/centre pooling
and focal-style heatmap losses that CenterNet builds on. Acknowledged as the
origin of the approach; no CornerNet-specific structure is implemented here.

**TOD-CNN** -- Chen et al., "TOD-CNN: An effective convolutional neural network
for tiny object detection in sperm videos", arXiv:2204.08166.
*Cited in `detection/todcnn.py` and verified in `docs/source_audit.md`.*

`detection/todcnn.py` is an **independent reimplementation of the design
argument, not of the model**. The original is a Keras / TensorFlow-1 anchor-based
network trained on MIaMIA-SVDS with two classes (`S`, `Impurity`) and an anchor
set whose every box is under 20 px. What is borrowed is the reasoning that
generalises: tiny objects lose their spatial evidence in a downsampling stack, so
this backbone stops downsampling at stride 4 and never goes deeper; receptive
field is bought instead with dilated convolutions at constant resolution; depth
is spent on width and context rather than on more stages. The anchor set is
dropped entirely in favour of the shared anchor-free head.

**P2Net** -- no external source. An FPN whose finest and *only* prediction level
is P2 (stride 4), built downward to stride 32 so that the deep context which
separates a sperm head from a debris speck is available, then carried back up to
the resolution where the objects actually are. Implemented against the identical
head and pre/post-processing as `todcnn`, which is the reason both exist.

### 3.2 Tracking

**ByteTrack** -- Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022, arXiv:2110.06864. ‡

The default tracker. The contributed idea is small and specific: keep the
low-confidence detections other trackers discard, and give them a *second*
association pass against tracks nothing better claimed. A low-confidence box in
isolation is usually noise; a low-confidence box exactly where an established
track predicted it is almost always the object, dimmed or half-occluded -- which
is the case a sperm swimming under debris, tumbling edge-on, or drifting through
a dim patch of the field lives in.

**OC-SORT** -- Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
Multi-Object Tracking", CVPR 2023, arXiv:2203.14360. ‡

Observation-Centric Momentum, Re-Update and Recovery, all reimplemented from the
described method. Relevant here because a sperm is a small, low-contrast,
non-linearly moving object -- exactly the regime where a Kalman filter's own
predictions are least worth believing.

**BoT-SORT** -- Aharon, Orfaig & Bar-Hillel/Bobrovsky, "BoT-SORT: Robust
Associations Multi-Pedestrian Tracking", arXiv:2206.14651. ‡

ByteTrack plus camera-motion compensation and appearance fusion. **Both are off
by default here**, and camera-motion compensation should stay off: the camera is
rigidly mounted, so the global image motion CMC removes *is* the sample's fluid
flow, which belongs to the flow-correction stage where it is measured, subtracted
and recorded.

**SORT / DeepSORT** -- Bewley et al., arXiv:1602.00763 ‡; Wojke et al.,
arXiv:1703.07402 ‡.

`tracking/kalman.py` uses the SORT / DeepSORT / ByteTrack state parameterisation
`[cx, cy, a, h, vcx, vcy, va, vh]`, kept deliberately recognisable to a reader who
knows those papers. The filter is written from the description in pure numpy.

Assignment uses `scipy.optimize.linear_sum_assignment` (Jonker-Volgenant), from
SciPy, BSD-3-Clause.

### 3.3 Morphology backbones

**MobileNetV3** -- Howard et al., "Searching for MobileNetV3", ICCV 2019,
arXiv:1905.02244. ‡

**EfficientNet** -- Tan & Le, "EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks", ICML 2019, arXiv:1905.11946. ‡

Both are used via **torchvision's** implementations (BSD-3-Clause), with the stem
rebuilt for a single input channel because the microscope is monochrome. When
starting from pretrained RGB weights, the first convolution's kernel is **summed**
across the input-channel axis rather than averaged or re-initialised: for
`R = G = B = g`, `sum_c W_c * g == (sum_c W_c) * g`, so summing reproduces exactly
the response the pretrained filter would have given on a grey image and leaves
every downstream BatchNorm statistic valid.

A third backbone, `simplecnn`, is this project's own and has no pretrained form.

### 3.4 Methods, standards and literature relied on

Not code, but the sources the numeric behaviour is derived from. Full citations
are in `CITATION.cff`; verification status is in `docs/source_audit.md`.

- **WHO laboratory manual for the examination and processing of human semen, 6th
  edition (2021).** Source of the four-category motility grading and the 25 / 5
  um/s limits (section 2.4.6.1), the morphology criteria and morphometry (Table
  2.6), the CASA definitions (section 4.5.1.4), the reference limits (Table 8.3),
  the Annexin V MACS protocol (section 5.6.2), and the 37 C requirement.
- **Mortimer, van der Horst & Mortimer**, *Asian J Androl* 2015;17:545-53.
  Source of the ~60 images/s figure -- **which is theirs, not WHO's** -- and of the
  finding that a fixed-point-count smoother produces aberrant ALH when the frame
  rate changes. Directly responsible for `MotionConfig.vap_window_ms` being a
  duration.
- **Björndahl & Kirkman-Brown**, *Fertil Steril* 2022;117:246-51 (open access,
  CC BY). Quoted in `docs/safety_and_claims.md` section 8 on emerging
  technologies asserting equivalence to WHO core methods.
- **Gallagher et al.**, *Hum Reprod* 2019;34:1173-85. ‡ Cited *by WHO* for the
  finding that BCF does not correlate with flagellar beat frequency.
- **Castellini et al.**, *Fertil Steril* 2011. ‡ Direction of the frame-rate
  effect on VCL, LIN and ALH. **Partially verified only** -- the primary PDF was
  403-blocked -- so only the direction is used, never a magnitude.
- **Cochrane CD010461** (Garg et al., updated 2026) and **Martinez MG et al.**,
  *J Assist Reprod Genet* 2018;35(12):2215-21. Evidence status of MACS.
- **Reprod Sci** 2025, PMID 40312558. Meta-correlation of morphology against DFI.
- **Ghasemian 2015**, doi:10.1016/j.cmpb.2015.08.013. HSMA-DS, from which MHSMA
  derives.
- **MDCG 2020-16 rev.4**, EU IVDR classification guidance, and **21 CFR
  864.5220** with 510(k) records K071737, K220828, K242830, K183602.

---

## 4. Hardware documentation

Not licensed content used in this repository, but the primary sources the
configuration defaults were read from:

- **Basler** product documentation for the a2A1920-160umPRO
  (`docs.baslerweb.com/a2a1920-160umpro`, `/exposure-time`, `/image-roi`,
  `/timestamp`, `/data-chunks`). Basler and pylon are trademarks of Basler AG.
- **Edmund Optics** listing for the Olympus PLN 100X Oil objective (stock 29225).
  Olympus is a trademark of Olympus Corporation / Evident Corporation.

Neither vendor endorses this project. Trademarks are used nominatively, to
identify the hardware the software targets.

---

## 5. Attribution when publishing

Every dataset in section 2 requires attribution, including the permissively
licensed ones -- and CC BY additionally requires indicating whether changes were
made, which for a converted or re-split copy is always yes. `CITATION.cff`
carries the citable references. `docs/license_audit.md` section 6 sets out the
obligations.
