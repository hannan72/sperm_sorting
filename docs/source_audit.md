# Source audit

What was inspected, what a primary source actually said, and what could not be
verified. Every number that appears anywhere else in `docs/` is either traceable
to a row in this document or is computed in the code from one.

**Audit date.** The whole audit was compiled for the initial commit of this
repository, 2026-08-04. Per-URL retrieval timestamps were not recorded
individually, so every entry below carries the compilation date rather than a
fetch time of its own. Where a fetch failed, the failure mode is recorded,
because a failure is evidence too: it tells the next person what to try and what
not to trust in the meantime.

**Reading this document.** "Verified" means the statement was read out of the
primary source named in the row -- the vendor's own documentation page, the
official WHO PDF, the dataset repository's own files, the installed Python
wheel. "Unverified" means it was not, for the stated reason, and it is repeated
as unverified everywhere it is used. There is no third category: nothing here is
presented as established because it is widely believed.

---

## 1. Corrections first

These are the load-bearing part of the audit. Each is a belief that is common,
plausible, and wrong, and each would have propagated into the code or the
documentation if it had not been checked against a primary source.

| # | The common assumption | What the primary source says | What it would have cost |
|---|---|---|---|
| C1 | MHSMA is the dataset on Zenodo record 2640506. | **Zenodo 2640506 is VISEM**, not MHSMA. MHSMA is distributed as `.npy` files committed directly into the GitHub repository `soroushj/mhsma-dataset` under `mhsma/`; it is not on Zenodo at all. | A download script pointed at the wrong 35.2 GB archive, and a morphology "dataset" containing no morphology labels. |
| C2 | The WHO 6th edition uses the PR / NP / IM three-way motility split. | **The 6th edition reinstated the four-category a/b/c/d system.** Section 2.4.6.1, p. 24: "A four-category system for grading motility is recommended." Section 1.3 gives the reason: "because presence (or absence) of rapid progressive spermatozoa is clinically important." PR/NP/IM was the *5th* edition. PR is the aggregate a+b, NP is c, IM is d. | The `MotilityClass` enum would have had three members instead of four, and the 25 um/s rapid-progressive cut -- which is a literal 6th-edition number -- would have looked like an invention. |
| C3 | CASA requires ~50-60 frames per second, per WHO. | **The WHO 6th edition specifies no numeric minimum frame rate.** The whole manual was grepped. Frame rate appears twice: section 4.5.1.1 lists it among factors that "affect the performance of CASA instruments", and the VAP definition notes non-comparability "with different acquisition parameters such as framerate". The only temporal specification is that "at least 1 second is enough for basic CASA measurements". The ~60 Hz figure is **Mortimer, van der Horst & Mortimer, *Asian J Androl* 2015;17:545-53**, verbatim: "a 60 images/s video camera (which is really the minimum imaging frequency required for reliable human sperm track analysis)". | A configuration comment citing WHO for a number WHO does not give -- an attribution error that survives review because everyone already believes it. `MotionConfig.min_fps_for_alh_bcf` carries the correct attribution. |
| C4 | MHSMA's README "% Positive" column is the abnormality rate. | **MHSMA's README calls the *normal* class "positive".** The integer encoding is `0 = normal`, `1 = abnormal`. Reading the README table as prevalence of abnormality inverts every number in it. | Every per-aspect class weight, threshold and metric inverted -- a model that keeps exactly the sperm it was built to reject, at a plausible-looking accuracy. Guarded in code by `constants.LABEL_NORMAL` / `LABEL_ABNORMAL` and by the single-flip contract in `morphology/polarity.py`. |
| C5 | The Basler a2A1920-160umPRO runs at 164 fps. | The Basler documentation says 164 fps "at default settings"; the shop page **and the model number itself** say 160. The two are not reconciled anywhere in the vendor's material. | Nothing catastrophic, but a throughput budget built on the optimistic figure. 160 is used as the conservative design number throughout (`BaslerConfig.acquisition_frame_rate`), and 164 appears only where the worst-case bandwidth is being computed. |
| C6 | The Olympus PLN 100X is a 160 mm finite objective. | It is **infinity-corrected with a 180 mm Olympus tube lens**; 160 mm is the obsolete finite standard. Objective focal length 1.80 mm. | A 12.5% error in every derived magnification, and therefore in every velocity, if the nominal scale were ever used in place of a measurement. |
| C7 | A whole spermatozoon fits in the field of view at 100x. | It does not. At 0.0345 um/px the field is 66.24 x 41.40 um; a whole cell is ~53 um long and up to 60 um. It fits along the frame with 20% slack and **does not fit across it at all**. | Designing the detector around whole-cell boxes. The detection target is the head -- see `docs/assumptions.md` section 2 and `DetectionConfig`'s docstring. |
| C8 | Detection-Sperm / MIaMIA-SVDS is CC BY 4.0. | There is a genuine conflict. The GitHub repository has **no LICENSE file**; its README says non-commercial research use is welcome; the figshare record's metadata tags CC BY 4.0. Both statements are reported in `docs/license_audit.md` and neither is adopted. | Assuming commercial rights that may not exist. |
| C9 | Annexin-V status or DNA fragmentation can be predicted from a microscopy image. | No. WHO's own register is "only partially related" / "associated with" / "may also have"; the published effect sizes are |r| ~ 0.3, i.e. 9-15% of variance; and every correlation in the literature is between *per-ejaculate summary percentages across men*, not between per-cell states. See `docs/safety_and_claims.md`. | The single most serious claim error available to this project. |

---

## 2. Datasets

Full per-dataset detail is in `docs/dataset_audit.md`; licensing is in
`docs/license_audit.md`. This section records only what was inspected and how
far the inspection got.

### 2.1 MHSMA

| | |
|---|---|
| URL | https://github.com/soroushj/mhsma-dataset (files committed under `mhsma/`) |
| Verified from primary source | 18 `.npy` files, all `uint8`. Splits `train` / **`valid`** / `test` at 1000 / 240 / 300 = 1540 images from 235 male-factor-infertility patients. Two crop sizes, 128x128 and 64x64. Four label arrays per split: `y_{acrosome,head,vacuole,tail}_{train,valid,test}.npy`. Labels `0 = normal`, `1 = abnormal`. Grayscale, single sperm, head roughly centred, tail not entirely visible. Derived from HSMA-DS (Ghasemian 2015, doi 10.1016/j.cmpb.2015.08.013), captured at x400 and x600. Licence CC BY-NC-SA 4.0. |
| Correction | See C1 (not on Zenodo) and C4 (README's "positive" means normal). |
| Not verified | The HSMA-DS staining protocol and microscope model. Not stated in the MHSMA repository, and the upstream paper was not obtained. This matters: WHO section 2.5 notes that "each stain provides quite different results down to the level of sperm sizes", so an unknown stain is an unknown photometric domain. |
| Split-naming trap | The directory word is `valid`, not `val`. A loader written from memory silently finds nothing. |

### 2.2 VISEM-Tracking

| | |
|---|---|
| URL | Zenodo record 7293726; paper *Sci Data* 2023; preprint arXiv:2212.02842 |
| Verified from primary source | 20 annotated 30-second clips from 20 patients; 29,196 annotated frames; 656,334 bounding boxes; 1,121 unique sperm track IDs, 20 cluster IDs, 35 pinhead IDs. Three classes: `0 = sperm`, `1 = cluster`, `2 = small or pinhead`. YOLO-normalised `x_center y_center width height`. Per video: a frames directory, `labels` (boxes only), `labels_ftid` (boxes plus tracking id), and the `.mp4`. Official split 16 train / 4 validation **by video**, validation videos **82, 60, 54, 52**; no official test split. Distribution tree `VISEM_Tracking_Train_v4/Train/<video_id>/`. Frame rate 45-50 FPS and **not uniform across videos**. Published YOLOv5l baseline mAP@0.5 = 0.2231. Capture: Olympus CX31, 400x, phase contrast, 37 C heated stage, UEye UI-2210C camera. Licence CC BY 4.0. |
| Quirks found | `video_23` has 174 frames with no sperm at all. `video_35` and `video_52` have 1440 frames; `video_82` has 1500. Boxes concentrate in the upper-left of the frame -- a spatial prior a detector will learn if the crops are not randomised. |
| Not verified | Resolution is stated as 640x480 **inherited from VISEM**; the VISEM-Tracking authors never restate it for this dataset. Treated as 640x480 but flagged. |

### 2.3 VISEM

| | |
|---|---|
| URL | Zenodo record 2640506 |
| Verified from primary source | 85 participants, 85 videos, 2-7 minutes each, 640x480, 50 FPS, AVI, distributed as a single 35.2 GB zip. Capture: Olympus CX31, phase contrast, 37 C, UEye UI-2210C, 400x, 10 ul under a 22x22 mm coverslip. Annotations are **sample/video level only**: WHO semen analysis, motility percentages (progressive / non-progressive / immotile), concentration, fatty acids, sex hormones, demographics, across six CSVs. **No per-sperm labels. No bounding boxes.** Licence CC BY-NC 4.0. |
| Could not verify | The exact CSV filenames. `datasets.simula.no` returned an **SSL certificate error** and was not retried with verification disabled, because accepting an unverified certificate to establish a fact about provenance is self-defeating. The count (six CSVs) and their subject matter came from the Zenodo record itself. |

### 2.4 VISEM-Tracking-graphs

| | |
|---|---|
| URL | Hugging Face, `SimulaMet-HOST/visem-tracking-graphs` |
| Verified from primary source | GraphML read via networkx, 3.26 GB. Directories `spatial_threshold_{0.1..0.5}/`, and per video `frame_graphs/frame_graph_{i}.graphml` plus `video_graph.graphml`. Node id is the `sperm_id`. Node attributes: `frame_number`, `class_name` (the YOLO class index as a string), `x_center`, `y_center`, `width`, `height`, all YOLO-normalised to 0-1. Edges are spatial (weight = Euclidean distance between normalised centres, added when below the threshold) or temporal (`edge_type="temporal"`). Frame graphs are undirected; the video graph is directed. Licence: data CC BY 4.0, generator code MIT. |
| Defect found | `video_graph.graphml` keys nodes by `sperm_id` **alone**, not by `(sperm_id, frame)`. Every observation of one sperm therefore collapses onto a single node, and the temporal edge loop degenerates into self-loops. The per-frame graphs appear sound. Fix and regeneration recipe in `docs/dataset_audit.md` section 5. |
| Second defect found | The `spatial_threshold` compares **normalised** coordinates, so a fixed threshold is anisotropic in pixels on a 4:3 frame -- the same numeric threshold spans 4/3 as many pixels horizontally as vertically. |

### 2.5 Detection-Sperm / TOD-CNN / MIaMIA-SVDS

| | |
|---|---|
| URLs | https://github.com/Demozsj/Detection-Sperm (code); figshare record 15074253 (data, `Data Set.rar`, 1.42 GB); paper arXiv:2204.08166 |
| Verified from primary source | The GitHub repository is a **model repository, not a dataset repository** -- its own words describe the included data as a "simple example of a data set due to github's limited data volume". `model_data/sperm_classes.txt` contains exactly two lines, `S` and `Impurity`, so the shipped detector is two-class. `model_data/sperm_anchors.txt` is `7,11 8,15 9,10 10,14 12,11 13,19` -- every anchor under 20 px, which is the tiny-object regime. Dataset MIaMIA-SVDS (also called SVIA): Subset-A >125,000 objects with box and category across 101 videos (detection); Subset-B >26,000 segmented sperms across 10 videos (tracking ground truth); Subset-C >125,000 cropped images (classification); >278,000 annotated objects in total, annotated by 14 experts and verified by 6. Object sizes ~5-50 um². Impurities include bacteria, protein clumps and bubbles. Capture: WLJY-9000 CASA system, 20x objective plus 20x electronic eyepiece, 30 FPS, clips of 1-3 s. Splits 6:2:2 by video: 2125 train / 668 validation / 829 test images from 21 videos. Framework: Keras 2.1.5 on TensorFlow 1.13.1, Python 3.7 -- an end-of-life TF1 stack. |
| Could not verify | (a) The **native video resolution**. The 416x416 in the paper is a network resize target, not the capture resolution. (b) The **on-disk annotation format** of the released archive. The annotation tool is stated to be LabelImg, which implies VOC XML at annotation time, but the format actually shipped in `Data Set.rar` was not confirmed. |
| Licence | Genuine conflict; see C8 and `docs/license_audit.md`. |

---

## 3. Hardware and optics

### 3.1 Basler a2A1920-160umPRO

| | |
|---|---|
| URLs | `docs.baslerweb.com/a2a1920-160umpro`, `/exposure-time`, `/image-roi`, `/timestamp`, `/data-chunks` |
| Verified from primary source | Sensor Sony IMX392LLR-C, progressive-scan CMOS, **global shutter**, 1/2.3", 7.9 mm diagonal. 1936x1216 full, 1920x1200 default, 2.3 MP, 3.45 x 3.45 um pixels. **Monochrome only.** Pixel formats Mono8, Mono10, Mono10p, Mono12, Mono12p -- the API enum values contain **no space** ("Mono8", not "Mono 8"). USB 3.0 (5 Gbit/s nominal), USB3 Vision, C-mount, ace 2 R family, ~3 W at 5 VDC. Exposure, from the exact model row of the exposure-time page: Common mode 19 us to 10,000,000 us; Ultra Short mode 1 us to 14 us; the `ExposureTime` parameter is in **microseconds**; `BslEffectiveExposureTime` is read-only on ace 2. ROI: `Width` / `Height` / `OffsetX` / `OffsetY`, model minimums Width 4 and Height 1; reducing the ROI raises the maximum frame rate and **height is the dominant term** because readout is row-limited; read the achievable rate from `BslResultingAcquisitionFrameRate`; cap it with `AcquisitionFrameRateEnable` plus `AcquisitionFrameRate`. Timestamps: the ace 2 device clock is **1 GHz, so one tick is 1 ns**, and the counter resets on power cycle. |
| Correction | The legacy `ResultingFrameRate`, `CenterX` and `CenterY` parameters are **ace Classic only** and are not present on this camera. Sample code copied from an ace Classic example fails here. |
| Correction | On ace 2 the timestamp chunk is **`BslChunkTimestampValue`**. The legacy `ChunkTimestamp` is still present, but Basler explicitly recommends the `Bsl` one. `BslChunkTimestampSelector` chooses FrameStart / ExposureStart / ExposureEnd. USB cameras do **not** provide a Framecounter chunk. |
| Warning read from the docs | `TimestampLatch` / `TimestampLatchValue` exist, but the documentation warns of an "unspecified and variable delay". They must **not** be used for sub-millisecond host-to-camera alignment without first characterising that delay. |
| Performance note | `StaticChunkNodeMapPoolSize` should be pre-allocated to `MaxNumBuffer`, otherwise a node map is constructed per frame at 164 fps. |
| Not verified | (a) **Dynamic range 71.7 dB** comes from the shop page, not the documentation. (b) Chunk support was verified **at ace 2 USB family level, not per SKU**; probe the actual camera with `camera.ChunkSelector.GetSettableValues()` rather than assuming. (c) Sustained USB 3.0 throughput for this specific model against Basler's own bandwidth table. |
| Discrepancy left open | 164 fps (docs) versus 160 fps (shop page and model number). See C5. |

### 3.2 Olympus PLN 100X Oil (Edmund Optics stock 29225)

| | |
|---|---|
| Verified from primary source | Magnification 100X. NA 1.25. Working distance 0.15 mm. Oil immersion. Field number 22 mm. Parfocal length 45 mm. **Infinity-corrected, Olympus tube lens 180 mm** (see C6). Objective focal length 1.80 mm. Field of view at the sample = FN / mag = 22 / 100 = 0.22 mm = 220 um. Published resolving power 0.27 um. Entrance pupil 4.50 mm. RMS thread. Plan Achromat, UIS2. |
| Cross-check that passed | Rayleigh limit computed as 0.61 * 550 nm / 1.25 = 0.2684 um, which matches the manufacturer's published 0.27 um. |
| Not verified | (a) **Cover-glass thickness.** Edmund lists "N/A". Olympus oil objectives are normally 0.17 mm, but this was not confirmed and is not assumed. (b) Edmund's depth-of-field figure of 0.27 um and depth-of-focus figure of 1760 um were retrieved once and could not be re-fetched -- the page subsequently returned a bot block. They are therefore **not used** anywhere; `docs/assumptions.md` treats depth of field qualitatively. |

### 3.3 pypylon

Verified by **inspecting the installed wheels**, versions 4.2.0 and 26.7 -- not
by reading documentation, because the hazard here is precisely that the
documentation describes a version you may not have.

| | |
|---|---|
| Version hazard | `pylon.FirstFound` and `GrabResult.GetFirstImageDataComponent()` exist in 26.x and are **absent in 4.2.0**. The current GitHub README samples are written in 26.x style and fail on 4.x. The portable open is `camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())`. |
| API convention | Direct attribute assignment (`cam.Gain = 42`) is **deprecated**; use `.Value` -- `camera.PixelFormat.Value = "Mono8"`, `camera.Width.Value = ...`, `camera.ExposureTime.Value = ...`. Also available: `TrySetValue()`, `SetToMaximum()`, `TrySetToMaximum()`, `GetValueOrDefault()`, `WidthMax` / `HeightMax`. |
| Grab strategies (four verified) | `GrabStrategy_OneByOne`, `_LatestImageOnly`, `_LatestImages`, `_UpcomingImage`. **OneByOne (the default) delivers every image in arrival order and is the only correct choice for CASA tracking.** `LatestImageOnly` discards older frames, which fragments trajectories and corrupts every velocity derived from them -- it is right only for a live display that measures nothing. `UpcomingImage` is **not supported on USB devices**; the vendor's own sample guards it with `if not camera.IsUsb()`. |
| GrabResult surface | Properties `Array`, `Buffer`, `TimeStamp`, `BlockID`, `ID`, `ImageNumber`, `NumberOfSkippedImages`, `Width`, `Height`, `OffsetX`, `OffsetY`, `PixelType`, `PayloadType`, `ErrorCode`, `ErrorDescription`, `ChunkDataNodeMap`, `DataComponentCount`. Methods `GrabSucceeded()`, `IsValid()`, `HasCRC()`, `CheckCRC()`, `IsChunkDataAvailable()`, `GetArrayZeroCopy()`, `GetChunkNode(name)`, `Release()`. |
| Trap read from the docstring | `GetTimeStamp()`: "camera specific tick count ... describes when the image exposure was started. Cameras that do not support this feature return ZERO." A zero timestamp must therefore be treated as *absent*, not as t=0. |
| Three working routes to the chunk timestamp | `res.ChunkTimestamp.Value` (when `IsReadable()`); `pylon.IntegerParameter(res.ChunkDataNodeMap, "ChunkTimestamp").Value`; `pylon.IntegerParameter(res.ChunkDataNodeMap, "BslChunkTimestampValue").Value` (ace 2). |

---

## 4. WHO laboratory manual, 6th edition (2021)

Read from the **official PDF**. Section and page references below are to that
document.

| Topic | Verified from the primary source |
|---|---|
| Motility grading | Section 2.4.6.1, p. 24. Four categories reinstated (see C2), with explicit approximate velocity limits: rapidly progressive >= 25 um/s ("at least 25 um, or 1/2 tail length, in one second"); slowly progressive 5 to <25 um/s ("at least one head length to less than 1/2 tail length"); non-progressive <5 um/s ("the flagellar force displacing the head less than 5 um, one head length"); immotile, no active tail movements. These are net-displacement (VSL-like) thresholds. |
| Temperature | Section 2.4.6: "The velocity of motile spermatozoa is temperature dependent." **37 C is required.** A velocity-threshold classifier applied at uncontrolled room temperature does not produce a WHO-comparable grading. |
| History of the 25 um/s limit | 25 um/s at 37 C was WHO 4th ed. (1999) grade a. The 5th edition dropped it (Cooper & Yeung 2006: technicians cannot judge it without bias). The 6th reinstated it as an *approximate* limit. Note the sign flip: 4th ed. ">25", 6th ed. ">=25". |
| Reference limits | Table 8.3, p. 213, 5th centiles with 95% CI, from men in couples achieving natural conception within one year. Volume 1.4 mL (1.3-1.5), median 3.0. Concentration 16 x 10^6/mL (15-18), median 66, 95th centile **208** -- several vendor tables mis-transcribe this as 200. Total sperm 39 x 10^6 (35-40), median 210. Total motility 42% (40-43), median 64. Progressive motility 30% (29-31), median 55. Non-progressive 1% (1-1), median 8. Immotile 20% (19-20), median 37. Vitality 54% (50-56), median 78. Normal forms 4% (3.9-4.0), median 14. |
| Reference-limit trap | Non-progressive 1% and immotile 20% are **distributional centiles, not lower limits**. A low immotile count is good. |
| Not pass/fail | Section 8.1.3, verbatim: "The lower fifth percentile of data from men in the reference population (Table 8.3) does not represent a limit between fertile and infertile men." Section 1.3: "these percentiles do not represent distinct limits between fertile and subfertile men." Clinical decision limits "still need to be developed." The editorial board's suggested phrasing for a value below a limit is "not typical" for a highly fertile man. |
| Morphology criteria | Table 2.6, p. 50. Normal head: "smooth, regularly contoured and generally oval... well-defined acrosomal region comprising 40-70% of the head area. The acrosomal region should contain no large vacuoles, and not more than two small vacuoles, which should not occupy more than one fifth of the sperm head. The post-acrosomal region should not contain any vacuoles." Abnormal head: acrosome <40% or >70%; length-to-width ratio <1.5 (round) or >2 (elongated); pyriform, amorphous or asymmetrical; vacuoles over one fifth of head area or in the post-acrosomal region; double heads. Midpiece normal: slender, regular, about the same length as the head, axis aligned with the head axis. Tail normal: uniform calibre, thinner than the midpiece, ~45 um (about 10x head length); may loop back provided there is no sharp angulation, which indicates a broken flagellum. Cytoplasmic droplets under one third of a normal head size are **normal**. |
| Morphometry | 77 Papanicolaou-stained spermatozoa **already classified normal**: head length median 4.1 um (95% CI 3.7-4.7), width 2.8 um (2.5-3.2), L:W 1.5 (1.3-1.8); midpiece length 4.0 um (3.3-5.2), width 0.6 um (0.5-0.7). |
| Morphometry trap | **These are descriptive statistics of already-normal cells, not the classification rule.** The rule is L/W <1.5 or >2 *plus* the shape, acrosome and vacuole criteria. An AI thresholding on "4.1 x 2.8 um" is not implementing WHO strict criteria. And Papanicolaou means fixed, air-dried and stained: dimensions from unstained phase or brightfield images are not comparable, since "each stain provides quite different results down to the level of sperm sizes". |
| Normal forms 4% | Section 2.5: "reference limits and thresholds of 3-5% normal forms have been found in studies of IVF, IUI and in vivo fertility." The Tygerberg/Kruger criteria originate from sperm that penetrated cervical mucus and bind the zona pellucida. The 4% of the 6th edition and the 30% of the 3rd are **different classification systems**, not a decline in sperm quality. |
| Per-region defect reporting | WHO **requires** %H (head), %NM (neck and midpiece), %T (tail) and %C (excess residual cytoplasm): "It is common that the only reported parameter is the proportion of 'normal' spermatozoa and that the distribution of the defects is ignored." **This project's four aspects (head / acrosome / vacuole / tail) are MHSMA's decomposition, not WHO's.** Acrosome and vacuole are sub-features of WHO's head category, and there is no midpiece or cytoplasm aspect here at all. This is documented in `docs/safety_and_claims.md` and must not be presented as WHO reporting. |
| CASA definitions | Section 4.5.1.4, pp. 156-157, verbatim. VCL: "time-averaged velocity of the sperm head moving along the path traced out by the sperm as described in two dimensions under a microscope". VSL: "velocity calculated along a straight line between the first and last points". VAP: "time-averaged velocity calculated along the average path... a smooth curved path, calculated according to the algorithm embedded in the CASA system; these algorithms are different in different systems, so the values may not be comparable between systems, or with different acquisition parameters such as framerate." ALH: "magnitude of the lateral displacement of the sperm head about the average path... Different CASA systems calculate ALH using different algorithms, so the values may not be comparable." MAD: "time-averaged absolute values of the instantaneous angle of rotation of the curvilinear path", and it does **not** measure the turning angle of the direction the head points. LIN = VSL/VCL; WOB = VAP/VCL; STR = VSL/VAP. D is the fractal dimension. |
| BCF | "average frequency at which the curvilinear path crosses the average path. However, it should be noted that BCF **has been shown to not correlate with flagellar beat frequency**" (citing Gallagher et al., *Hum Reprod* 2019;34:1173-85). |
| Cross-instrument comparability | Figure 4.4 caption: "Different CASA instruments use different mathematical algorithms... The comparability of measurements across all instruments **is not yet known**." |
| Annexin V MACS | Section 5.6.2: "20 ul of Annexin V-conjugated microspheres", 15 minutes, load column, "Non-labelled fraction is recovered and processed." Section 5.6 on the evidence: "A Cochrane systematic review (358) did not see differences in clinical or live birth between MACS and sperm selected by hyaluronic acid binding (HA-ICSI) or other selection techniques, on live birth." |
| sDF and semen quality | Section 3.2.1: "Since sDF is **only partially related** to semen quality..." Section 2.5: "Abnormal spermatozoa generally have a lower fertilizing potential ... and **may also** have abnormal DNA. Morphological defects have been **associated with** increased DNA fragmentation..." |

### 4.1 Secondary literature checked against the WHO text

| Source | Status | What it contributes |
|---|---|---|
| Björndahl & Kirkman-Brown, *Fertil Steril* 2022;117:246-51 | **Verified**, open access, CC BY | "the exact velocity of each individual spermatozoon does not need to be assessed - this is only possible by CASA. Put simply, a rapidly progressive spermatozoon is one that moves >5 head-lengths per second". Note 4.1 um x 5 = 20.5 um/s versus half a tail length 22.5 um versus the manual's 25 um/s: WHO's three "definitions" agree with each other only approximately. Also the paragraph on emerging technologies asserting equivalence, quoted verbatim in `docs/safety_and_claims.md`. |
| Mortimer, van der Horst & Mortimer, *Asian J Androl* 2015;17:545-53 | **Verified**, open access, PMC4492043 | Source of the ~60 Hz figure (see C3). Also: "the points that make up the average path are themselves calculated from the average of the points on the curvilinear path"; "ALH values are not consistent between instruments and so cannot be standardized across CASA platforms"; "Older CASA systems that still use FIXED FIVE-POINT SMOOTHING to derive the average path will provide inadequate smoothing and hence widely aberrant ALH values." This is why `MotionConfig.vap_window_ms` is specified as a duration and converted per track. |
| Castellini et al., *Fertil Steril* 2011 (frame rate 12-200 Hz; human, rabbit, bull, ram) | **Partially verified** -- the primary PDF was **403-blocked** | Direction of the frame-rate error: higher fps raises VCL and lowers LIN and ALH (VSL roughly unchanged), so at low fps VCL is under-estimated while LIN and ALH are over-estimated. Fast non-linear cells are worst affected. Because this could not be read from the primary source, it is used only as a *direction*, never as a correction factor. |
| CASA guidance recommending 50-60 Hz, *Fertil Steril* S0015-0282(11)00766-7 | **Not verified -- paywalled** | Not used. The 50-60 Hz figure is attributed to Mortimer 2015 instead. |
| Cochrane CD010461 (Garg et al., updated 2026; 8 RCTs, 4147 women) | **Verified** | "We are uncertain whether MACS improves live birth", certainty **very low**. MACS live birth RR 1.95 (0.89-4.29) from one RCT of 62 women; clinical pregnancy RR 1.05 (0.84-1.31), 413 women, I² = 81%. |
| Martinez MG et al., *J Assist Reprod Genet* 2018;35(12):2215-21 (WHO reference 184) | **Verified via the WHO reference list** | Title states the finding: "Magnetic-activated cell sorting is not completely effective at reducing sperm DNA fragmentation." |
| Morphology-vs-DFI meta-correlation, *Reprod Sci* 2025, PMID 40312558 | **Verified** | Normal morphology vs DFI r = -0.30; abnormal vs DFI r = +0.39, i.e. ~9-15% of variance. "None of the morphological indices independently predicted SDF after adjustment for sperm concentration and progressive motility." |
| Human flagellar beat frequency (~10-20 Hz) | **Not verified** | Used only to observe that 160 fps gives roughly 8-16 samples per beat -- adequate for a crossing count, marginal for waveform reconstruction. Marked unverified wherever it appears. |
| "CASA VCL > 50-60 um/s healthy", "grade a VCL > 40 um/s" | **Not verified** -- secondary snippets only | **Not used anywhere.** The thresholds in this project are WHO's own VSL-like limits. |
| CASA VCL/VAP population means | **Not verified** -- paywalled | Not used. |

---

## 5. Regulatory

| Topic | Verified from the primary source |
|---|---|
| EU IVDR (Regulation (EU) 2017/746) | An IVD is intended for the in vitro examination of specimens to provide information on a physiological state, disease, or **fertility**. Semen is explicitly a specimen type. **The intended purpose is the trigger, not the technology.** |
| Annex VIII implementing rule 1.4 | Verbatim, via MDCG 2020-16 rev.4: "Software which drives a device or influences the use of the device shall fall within the same class as the device. If the software is independent of any other device, it shall be classified in its own right." |
| Annex VIII rule 6 | Verbatim: "Devices not covered by the above-mentioned classification rules are classified as class B." |
| Annex VIII rule 4(a) | Self-tests are Class C "except for devices for the detection of pregnancy, **for fertility testing** and for determining cholesterol level... class B". |
| Research carve-out | Article 1(3) excludes products genuinely intended for research **without a medical purpose**. Article 2(45): "A device intended to be used for research purposes, without any medical objective, shall not be deemed to be a device for performance study." |
| MDCG 2020-16 rev.4 | https://health.ec.europa.eu/document/download/12f9756a-1e0d-4aed-9783-d948553f1705_en |
| US FDA | Automated semen analysers are **Class II under 21 CFR 864.5220** (automated differential cell counter), **product code POV**, cleared via 510(k); review panels Hematology (81) and OB-GYN (85). Verified against the 510(k) records K071737, K220828 (SQA-iO), K242830 (LensHooke X3 PRO) and K183602 (SwimCount). |
| Classification caveat | The Class B reading above is a **reasoned reading of the rules, not an official determination**. A professional-use semen-analysis IVD most plausibly falls to rule 6; a lay self-test falls to rule 4(a); either way Class B, and Class B and above require Notified Body involvement. |
| Not verified | Whether the US Clinical Decision Support exemption (FD&C Act 520(o)(1)(E)) covers a black-box CNN. The reasoning that it generally does *not* cover software whose basis a clinician cannot independently review is **general knowledge, not a verified citation**, and is marked as such in `docs/safety_and_claims.md`. |

**The conclusion that matters, and it is not a technical one:** a research-use-only
label does not protect a project whose documentation, marketing or actual use
asserts a diagnostic claim. Intent is inferred from the whole record, not from
the disclaimer.

---

## 6. Summary of everything left unverified

Repeated here in one place so that nothing needs to be hunted for. **None of
these may be presented as established anywhere in this repository.**

**Datasets**
- MIaMIA-SVDS native video resolution and on-disk annotation format.
- UEye UI-2210C: monochrome versus colour, rolling versus global shutter.
- HSMA-DS staining protocol and microscope model.
- VISEM's exact CSV filenames (`datasets.simula.no` SSL certificate error).
- VISEM-Tracking's own resolution statement (640x480 is inherited from VISEM).

**Hardware and optics**
- PLN 100X Oil cover-glass thickness (Edmund lists "N/A"; 0.17 mm expected, unconfirmed).
- Edmund's depth-of-field 0.27 um and depth-of-focus 1760 um (retrieved once, re-fetch bot-blocked). Not used.
- Per-SKU chunk availability for the a2A1920-160umPRO.
- Sustained USB 3.0 throughput for this model against Basler's own bandwidth figures. The ~378 MB/s figure in `docs/assumptions.md` is **computed**, not read from a vendor table.
- Dynamic range 71.7 dB (shop page, not documentation).
- The 164 vs 160 fps discrepancy is unresolved by the vendor's own material.

**Physiology and CASA**
- Human flagellar beat frequency (~10-20 Hz).
- CASA VCL/VAP population means (paywalled).
- "CASA VCL > 50-60 um/s healthy" and "grade a VCL > 40 um/s" (secondary snippets; not used).
- Castellini et al. 2011 frame-rate error magnitudes (403-blocked; direction only).

**Regulatory**
- Scope of the US CDS exemption with respect to black-box models.
- The EU class assignment is a reasoned reading, not an official determination.
