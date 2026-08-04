# Assumptions

Every assumption this prototype rests on, its status, and what breaks if it is
wrong. This is the most important document in `docs/`, because a research
instrument's honesty is entirely a function of how clearly it states what it has
*not* measured.

Three status values are used throughout and they mean exactly what they say:

| Status | Meaning |
|---|---|
| **VERIFIED** | Read from a primary source, or computed here from values that were. The source is named. |
| **UNVERIFIED** | Not read from a primary source, for a stated reason. Never presented as established, anywhere. |
| **UNMEASURED** | A property of the built instrument that has no value yet. `None` or `calibrated: false` in configuration, by design. |

A fourth category is worth naming separately: **DESIGN CHOICE** -- a decision
this project made that is stricter than, or simply different from, an external
standard. Those are flagged so they are never mistaken for citations.

---

## 1. The optical budget, worked through

### 1.1 Inputs

| Quantity | Value | Status | Source |
|---|---|---|---|
| Sensor pixel pitch | 3.45 x 3.45 um | VERIFIED | Basler docs, Sony IMX392LLR-C |
| Sensor format | 1936 x 1216 full, 1920 x 1200 default | VERIFIED | Basler docs |
| Shutter | global | VERIFIED | Basler docs |
| Colour | monochrome only | VERIFIED | Basler docs |
| Objective magnification | 100x | VERIFIED | Olympus PLN 100X Oil (Edmund 29225) |
| Numerical aperture | 1.25 | VERIFIED | same |
| Tube lens | infinity-corrected, Olympus 180 mm | VERIFIED | same (**not** the obsolete 160 mm finite standard) |
| Field number | 22 mm | VERIFIED | same |
| Working distance | 0.15 mm | VERIFIED | same |
| Published resolving power | 0.27 um | VERIFIED | same |
| Coupler magnification | 1.0 **assumed** | **UNMEASURED** | `OpticsConfig.coupler_magnification`; see 1.7 |
| Illumination wavelength | 550 nm | assumption for the calculation | green-centred white light |

### 1.2 Sampling

```
s = pixel_pitch / (objective_mag x coupler_mag)
  = 3.45 um / (100 x 1.0)
  = 0.0345 um/px   =   34.5 nm/px
```

This is the single most leveraged number in the product. Everything below scales
with it, and so does every velocity the system reports.

### 1.3 Diffraction limit and oversampling

```
Abbe      d = lambda / (2 NA)      = 550 / (2 x 1.25) = 220.0 nm = 0.2200 um
Rayleigh  d = 0.61 lambda / NA     = 0.61 x 550 / 1.25 = 268.4 nm = 0.2684 um
```

The Rayleigh figure matches the objective's published 0.27 um resolving power,
which is a genuine cross-check: two independent routes to the same number.

Nyquist requires `s <= d / 2`:

| Criterion | `d` | `d / s` (px per resolvable distance) | Oversampling `(d/2)/s` |
|---|---|---|---|
| Abbe | 0.2200 um | 6.38 px | **3.19x** |
| Rayleigh | 0.2684 um | 7.78 px | **3.89x** |

So the sensor resolves roughly **3.2-3.9x finer than the optics can**. That is
not waste; it is spendable headroom, and three things can be bought with it:

- **2x2 binning** gives 0.069 um/px, which is still **1.59x** (Abbe) to **1.94x**
  (Rayleigh) above the Nyquist requirement -- compliant -- while cutting the data
  rate fourfold, improving SNR, and raising the ROI-limited frame rate.
- **Phase-contrast rings or DIC prisms**, which reduce effective NA. See 5.2.
- **A dry condenser**, which does the same. See 5.1.

*(Note: the `BaslerConfig.binning` docstring describes 2x2 binning as "~1.6x above
the Rayleigh-limited Nyquist requirement". 1.6x is the Abbe figure; the Rayleigh
figure is 1.94x. The conclusion -- still Nyquist-compliant -- is unaffected.)*

### 1.4 Field of view

```
H:  1920 px x 0.0345 um/px  =  66.240 um
V:  1200 px x 0.0345 um/px  =  41.400 um
diagonal                     =  78.113 um
```

Cross-check against the published sensing area of 6.6 x 4.1 mm: 66.0 x 41.0 um.
Consistent.

The sensor diagonal is 7.81 mm against a field number of 22 mm, so **the sensor
is the limiting aperture**. There is no vignetting, but only
`(7.81 / 22)^2 = 12.6%` of the available field *area* is used. The objective is
delivering an image circle almost eight times larger in area than the camera can
see.

### 1.5 A spermatozoon in pixels

Using WHO 6th-edition morphometry (Table 2.6 -- and note carefully that those
are descriptive statistics of 77 Papanicolaou-stained cells *already classified
normal*, not a classification rule):

| Structure | Physical | Pixels at 0.0345 um/px |
|---|---|---|
| Head length | 4.1 um (95% CI 3.7-4.7) | **118.8 px** (107.2-136.2) |
| Head width | 2.8 um (2.5-3.2) | **81.2 px** (72.5-92.8) |
| Midpiece length | 4.0 um | 115.9 px |
| Midpiece width | 0.6 um | 17.4 px |
| Tail | ~45 um | ~1304 px |
| **Whole cell** | **~53.1 um** | **~1539 px** |

### 1.6 The finding: a whole spermatozoon does not fit

| Comparison | Result |
|---|---|
| 53.1 um vs H 66.24 um | fits, using **80.2%** of the frame width |
| 53.1 um vs V 41.40 um | **does not fit** |
| 60 um vs H 66.24 um | fits with **10%** slack |

So at most about one whole sperm can be in frame at a time, and full-flagellum
capture is **orientation-dependent and will frequently fail**. The pipeline's own
feasibility check reports this at startup:

```
whole 53 um sperm fits along flow: True, across flow: False
```

**Consequence: detect the head, not the cell.** This is a decision, and it is
recorded in `DetectionConfig`'s docstring and in `configs/default.yaml`. Three
independent reasons converge on it:

1. **Geometry.** A head at 4.1 x 2.8 um is ~119 x 81 px and always fits, in any
   orientation, anywhere in the frame.
2. **CASA already does this.** WHO section 4.5.1.4 defines VCL, VSL, VAP, ALH and
   MAD on the motion of the **sperm head**. Tracking the head centroid is not an
   approximation to what CASA measures; it *is* what CASA measures.
3. **MHSMA already does this.** Its crops are head-centred with the tail not
   entirely visible, so a morphology model trained on MHSMA is a head-centred
   model whether or not anyone intended it.

**And the cost is real and must be stated.** Tail morphology is judged from a
partial tail. Together with the tail aspect's 4.6% abnormal prevalence in MHSMA's
train split (and only 7 abnormal tails in the whole validation split), the tail
is the least trustworthy of the four aspects. `CropRecord.tail_complete` exists
so that this is visible per crop rather than assumed, and the feasibility report
warns to expect `tail_complete=False` routinely.

**If whole-cell or flagellar imaging ever matters**, the fix is optical, not
algorithmic: 40x oil gives 0.0863 um/px and a 166 x 104 um field while remaining
~1.55x oversampled, and 60x sits between. 100x is, for head-centroid CASA,
heavily overkill at 119 px per head.

### 1.7 The assumption most likely to be wrong: the coupler

**The 100x sample-plane magnification is an assumption.** It holds only with a
1x C-mount adapter and the Olympus 180 mm tube lens. A **0.5x or 0.63x reducing
coupler is very common, easy to overlook, and changes everything**:

| Coupler | `um_per_px` | Field of view | Head length | Rayleigh oversampling |
|---|---|---|---|---|
| 1.0x (assumed) | 0.0345 | 66.24 x 41.40 um | 118.8 px | 3.89x |
| 0.63x | 0.0548 | 105.1 x 65.7 um | 74.8 px | 2.45x |
| **0.5x** | **0.069** | **132.5 x 82.8 um** | **59.4 px** | **1.94x** |

A 0.5x coupler puts **every reported velocity out by exactly a factor of two** --
which moves every cell across the 25 um/s and 5 um/s WHO boundaries in the same
direction, changes the shot ratio, and changes the sort. Nothing raises. The
images look fine.

Two defences are in the code:

- `OpticalCalibration._plausible` rejects a measured scale differing from the
  nominal one by more than `max_nominal_discrepancy` (default 1.5x). A 0.5x
  coupler produces exactly 2.0x and is caught.
- `calibration/optics.py::_check_against_nominal` raises the same error at
  measurement time, with a message that names the reducing coupler as the most
  likely cause.

Neither defence works if the calibration is never performed, which is why
`calibrated` defaults to `false` and `require_calibrated()` raises rather than
substituting the nominal value.

---

## 2. The shot-throughput budget

The shot definition and the optics impose competing demands. Whether both can be
met is arithmetic, and `shots/feasibility.py` does it at startup so the answer
appears as a warning rather than as a puzzling stream of `INDETERMINATE` shots
weeks later.

### 2.1 The tension

A shot must contain 20-30 uniquely trackable sperm and must close within one
second, so the imaging region has to **deliver at least 20 trackable sperm per
second**. But each of those sperm must be observed long enough to measure its
velocity, and residence time is `field_length / flow_speed` -- so a faster flow
delivers more sperm and gives less evidence about each one. And the field is
66.24 x 41.40 um.

### 2.2 The default operating point

Working forward from `SyntheticSourceConfig`'s defaults, which were chosen to be
physically coherent rather than round:

| Step | Value |
|---|---|
| Field along the flow axis | 66.24 um |
| Chosen residence time `r` | **0.200 s** |
| Frames of evidence per sperm at 160 fps | **32** (the track-quality bar demands 6) |
| Implied bulk flow | 66.24 / 0.200 = **331.2 um/s** = 9600 px/s |
| Sperm needed simultaneously to deliver 25 in 1 s | 25 x (0.200 / 1.0) = **5.0** |
| Optical section depth assumed | **20 um** (UNMEASURED placeholder) |
| Sampled volume | 66.24 x 41.40 x 20 um³ = 5.485 x 10^4 um³ = 5.485 x 10^-8 mL |
| **Implied concentration** | 5.0 / 5.485e-8 = **91.2 million/mL** |

Reproduced verbatim by `sperm-sorting feasibility` on `configs/default.yaml`:

```
  sampling            : 0.03450 um/px (NOMINAL, uncalibrated)
  field of view       : 66.24 x 41.40 um
  flow speed          : 331.2 um/s
  residence time      : 200.0 ms (32.0 frames, 6 required)
  visible sperm needed: 5.00 simultaneously
  implied concentration: 91.2 million/mL (assuming 20 um chamber depth)
```

### 2.3 What 91 million/mL means

The **WHO 6th-edition 5th centile for concentration is 16 million/mL**, with a
median of 66 and a 95th centile of 208 (Table 8.3, p. 213). So the default
operating point needs a sample at roughly the **75th-90th centile** of the WHO
reference distribution -- a good sample, not a typical one, and far above the
threshold below which a value is "not typical for a highly fertile man".

This is an honest constraint, not a defect: it says the prototype's default
geometry is sized for good samples and will time out on thin ones. But it means
**a thin sample will produce `INDETERMINATE` shots, and that must not be read as
a fault in the models.** The feasibility check warns above 200 million/mL; 91
million/mL passes without a warning while still being demanding.

### 2.4 What to change if a sample is thinner

Every lever and its cost. The scaling is exact: required concentration is
proportional to `target / (duration x field_area x depth)` times residence, and
residence is `field_length / flow_speed`.

| Lever | Effect on required concentration | Cost |
|---|---|---|
| **Lower the magnification** (40x oil: 0.0863 um/px) | Field area scales with `s²`. 0.0863/0.0345 = 2.50x, so area x6.26 → **~15 million/mL**. Field length also x2.5, so residence x2.5 unless flow rises. | Head drops to ~48 px, still ample. Loses vacuole/acrosome detail. The single most effective lever. |
| **Enlarge the ROI** | Concentration scales inversely with field area | The ROI is already the full sensor; there is no headroom without changing optics. |
| **Raise the flow speed** | Residence falls, so required simultaneous count falls proportionally | Fewer frames of evidence. At 32 frames there is room: doubling the flow leaves 16 frames, still well above the 6-frame bar. Motion blur also doubles (section 3). |
| **Lengthen `maximum_shot_duration_seconds`** | Required count falls proportionally | Latency rises, and the fluid segment a shot describes gets longer -- the field must be held for the whole span, so sorting precision falls. |
| **Lower `target_trackable_sperm`** | Proportional | The ratio is estimated from fewer sperm; below 20 the rule refuses to decide at all. |
| **Deepen the chamber** | Inversely proportional | Only counts if the added depth is actually in focus. At NA 1.25 the depth of field is very shallow (section 5.3), so this lever is largely unavailable at 100x -- another argument for 40x. |
| **Concentrate the sample** | Directly | A pre-processing step outside this software; changes what is being sorted. |

The 20 um "chamber depth" is a **placeholder**, exposed as
`assess_feasibility(chamber_depth_um=...)` and as `--chamber-depth-um` on the CLI.
The implied concentration scales **inversely** with it, so a 10 um chamber would
need 182 million/mL. It is not a measured property of any built kit.

### 2.5 Crowding, checked twice

`assess_feasibility` judges crowding for heads and tails separately, because they
crowd differently. Heads are what the detector localises, so head occupancy
decides whether detection is possible at all; tails are what a morphology crop
contains, so tail occupancy decides whether that crop is contaminated by a
neighbour. At 5 simultaneous sperm in a 2742 um² field, head occupancy is a few
percent and the warning does not fire -- but the flagella sweep a much larger
fraction, which is why the best-frame `overlap` term (weight 0.15) exists.

---

## 3. The motion-blur budget

Blur in pixels = `v x t_exposure / s`.

| Exposure | 25 um/s | 50 um/s | 100 um/s |
|---|---|---|---|
| 1 us (Ultra Short mode floor) | 0.001 px | 0.001 px | 0.003 px |
| **19 us** (Common mode floor) | 0.014 px | 0.028 px | **0.055 px** |
| **1 ms** | 0.72 px | 1.45 px | **2.90 px** |
| 6.1 ms (one frame at 164 fps) | 4.42 px | 8.84 px | 17.67 px |

The conclusion is unambiguous: **keep the exposure well under a millisecond** to
stay sub-pixel at 100 um/s. `BaslerConfig.exposure_time_us` defaults to 200 us,
which at 100 um/s gives 0.58 px -- sub-pixel, with margin.

Two supporting facts. The camera has a **global shutter**, so there is no
rolling-shutter skew to correct on top of the blur. And the exposure floor is
19 us in Common mode (1 us in Ultra Short mode, which caps at 14 us), so the
hardware is not the constraint.

**The illumination is the constraint, and this is the practical risk item for
this build.** Exposing usefully in tens of microseconds, through an NA 1.25 oil
objective, on a near-transparent phase object, requires bright illumination. If
the illumination cannot deliver it, the exposure must lengthen, and blur grows
linearly. There is no software fix: a blurred head is a blurred head, and the
best-frame `motion_blur` term (weight 0.20) can only choose the least-bad frame,
not create a sharp one.

### 3.1 The related budget: per-frame displacement

At 164 fps the frame period is 6.0976 ms, so at 0.0345 um/px:

| Speed | um/frame | px/frame |
|---|---|---|
| 5 um/s | 0.0305 | 0.88 |
| 25 um/s | 0.1524 | 4.42 |
| 50 um/s | 0.3049 | 8.84 |
| 100 um/s | 0.6098 | 17.67 |

Displacement per frame (0.9-17.7 px) is far smaller than the head (119 x 81 px),
so **consecutive detections of the same cell overlap by more than 85% even at
100 um/s**. Nearest-neighbour and IoU-based association are therefore
unambiguous, which is why an IoU-driven tracker like ByteTrack is a good fit here
and why `match_iou_threshold` can be as low as 0.20 without inviting ID switches.

### 3.2 Temporal sampling

160 fps is **~2.7-3.3x above the ~50-60 Hz** that the CASA methodology literature
recommends. **That figure is Mortimer et al. (2015), not WHO** -- WHO 6th ed.
specifies no numeric minimum frame rate at all (see `docs/source_audit.md` C3).

Flagellar beat frequency is reported around 10-20 Hz -- **UNVERIFIED**, no primary
source obtained. If that range is right, 160 fps gives roughly 8-16 samples per
beat: fine for a crossing count, marginal for waveform reconstruction. This is
used only as an order-of-magnitude observation and never as a specification.

Note also WHO's own finding that BCF **does not correlate with flagellar beat
frequency** (citing Gallagher et al., *Hum Reprod* 2019;34:1173-85), so "enough
samples per beat" is a weaker requirement than it first appears.

---

## 4. The USB 3.0 bandwidth ceiling

```
1920 x 1200 x 1 byte x 164 fps  =  377.9 MB/s     (Mono8, full frame, docs rate)
1920 x 1200 x 1 byte x 160 fps  =  368.6 MB/s     (Mono8, full frame, design rate)
1920 x 1200 x 1.5 bytes x 164   =  566.8 MB/s     (Mono12p)
```

USB 3.0's nominal 5 Gbit/s becomes roughly **350-400 MB/s in practice** after
protocol overhead. So Mono8 at full frame and full rate sits **right at the
practical limit**, and **Mono12 cannot sustain full frame rate at all**.

This figure is **computed here and UNVERIFIED against Basler's own bandwidth
table**. The practical-throughput range is a general property of USB 3.0, not a
vendor statement for this model.

Three mitigations, in order of preference:

1. **Mono8** -- the default (`BaslerConfig.pixel_format`). With 3.2-3.9x spatial
   oversampling and a near-binary object against a bright field, the extra bit
   depth buys little.
2. **Reduce the ROI height.** Readout is row-limited, so **height is the dominant
   term**; reducing it both cuts bandwidth and *raises* the achievable frame rate.
   Read the result from `BslResultingAcquisitionFrameRate` rather than computing
   it.
3. **2x2 binning** -- cuts the rate fourfold to ~94 MB/s, improves SNR, raises the
   ROI-limited rate, and remains Nyquist-compliant at 1.59-1.94x oversampling
   (section 1.3). This is the cheapest fix if the link binds.

The failure mode if the link is oversubscribed is that **the camera reports
skipped images**. `acquisition/basler.py` surfaces those as
`FramePacket.dropped_before` rather than letting them vanish -- which is the same
reason `grab_strategy` is `OneByOne` and not `LatestImageOnly` (see section 6.2).

---

## 5. Optical assumptions that are not about geometry

### 5.1 The condenser must match the objective

In brightfield the effective resolution limit is `lambda / (NA_obj + NA_cond)`,
not `lambda / (2 NA_obj)`. A **dry condenser at NA ~0.9 is worse than 0.22 um**,
so the Abbe figure in section 1.3 assumes a matched oil condenser.

Separately: **NA 1.25 requires immersion oil actually being present.** Dry, the
effective NA collapses to about 1.0 and the objective does not form a proper
image at all. With a working distance of 0.15 mm the coverslip gap is tiny, and
the cover-glass thickness the objective expects is **UNVERIFIED** -- Edmund lists
"N/A"; Olympus oil objectives are normally 0.17 mm, but that was not confirmed.

**Status: UNMEASURED.** Both the condenser NA and the coverslip specification
are properties of the built rig.

### 5.2 Brightfield contrast on unstained live sperm

**This is the second most likely assumption to be wrong, after the coupler.**

Unstained live spermatozoa are near-transparent **phase objects**. They absorb
very little, so brightfield gives them poor contrast. **Phase contrast (or DIC)
is the standard for CASA** -- and it is what every public dataset in this project
uses: VISEM and VISEM-Tracking are 400x phase contrast on unstained wet
preparations, on a 37 C heated stage.

If the built rig is brightfield, the risk is not subtle: the detector may simply
not see enough contrast to localise heads reliably, and the failure appears as
low recall and short, fragmented tracks -- which then fail the track-quality bar,
shrink the shot, and produce `INDETERMINATE`.

The mitigations, in order:

- **Switch to phase contrast or DIC.** Both reduce effective NA, which the
  3.2-3.9x oversampling margin can absorb. This also moves the instrument
  *towards* the public datasets' domain rather than away from it.
- `PreprocessConfig.invert` exists because brightfield gives dark objects on a
  bright field and some detectors train better on the inverted convention. The
  simulator renders the brightfield convention explicitly
  (`RenderConfig.dark_objects`), stated once and implemented once, because
  getting the sign backwards silently would poison every model trained on it.
- `PreprocessConfig.normalize` offers CLAHE, which raises local contrast at the
  cost of amplifying noise in flat regions.

None of these creates contrast that the optics did not deliver.

### 5.3 Depth of field

At NA 1.25 the depth of field is **very shallow** -- far shallower than the 400x
dry setups the public datasets were captured on. Two consequences:

- A far larger fraction of cells in a chamber of any depth will be out of focus.
  `SyntheticSourceConfig.defocus_rate` defaults to 0.15 specifically to exercise
  the quality gate against this.
- The "20 um chamber depth" in section 2.2 assumes that all 20 um contributes
  countable sperm. At NA 1.25 it almost certainly does not, which makes the
  91 million/mL figure an **under**-estimate of the concentration actually
  required.

**Numeric depth-of-field values are deliberately absent from this document.**
Edmund's published DOF (0.27 um) and depth of focus (1760 um) were retrieved once
and could not be re-fetched -- the page subsequently returned a bot block -- so
they are UNVERIFIED and are not used.

### 5.4 37 C is required for WHO-comparable velocity

WHO section 2.4.6: "The velocity of motile spermatozoa is temperature
dependent." **A velocity-threshold classifier at uncontrolled room temperature
does not produce a WHO-comparable grading.** The same cell graded cold is slower
and falls into a lower category, so a room-temperature run systematically
under-counts progressive sperm, depresses the shot ratio, and rejects segments it
should accept.

Every public dataset in this project was captured on a **37 C heated stage**
(VISEM, VISEM-Tracking).

`MotilityThresholds.sample_temperature_c` defaults to 37.0 with a tolerance of
0.5 C. Out of specification, the grade is **still produced** -- a bench test at
room temperature is a legitimate thing to run -- but a non-comparability note is
appended to every reason string, and only to the branches that actually applied a
velocity threshold.

**Status: UNMEASURED.** Nothing in this software reads a thermometer. The
configured temperature is what the operator asserts, and the audit log records
the assertion, not a measurement.

---

## 6. Assumptions in the software

### 6.1 Frame rate: 160, not 164

The Basler documentation says 164 fps "at default settings"; the shop page and
the model number say 160. The vendor's own material does not reconcile them.
**160 is used as the conservative design number** throughout; 164 appears only
where the *worst-case* bandwidth is being computed, which is the direction that
makes the discrepancy safe.

### 6.2 `OneByOne`, never `LatestImageOnly`

`LatestImageOnly` silently discards older frames under load. Tracking
reconstructs a trajectory from **consecutive** observations, so a discarded frame
fragments tracks and corrupts every velocity derived from them -- and a broken
track is two tracks, i.e. one sperm counted twice in the shot denominator.

`OneByOne` delivers every image in arrival order and is the only correct choice
here. Under overload the drop then happens where the pipeline can *see* it, as a
reported `dropped_before` and a `frames_dropped_source` counter, rather than
invisibly inside the driver. `LatestImageOnly` is retained in the config
`Literal` only for a live-display path that measures nothing.

(`UpcomingImage` is **not supported on USB devices** -- Basler's own sample guards
it with `if not camera.IsUsb()` -- and is therefore not offered.)

### 6.3 Timestamps

The camera's own tick counter is the only timestamp that reflects when the
exposure actually started; a host timestamp taken at grab time includes USB
transfer and scheduling jitter. On ace 2 the device clock is **1 GHz, so one tick
is 1 ns**, and it resets on power cycle.

Preference order: chunk timestamp (`BslChunkTimestampValue` on ace 2 -- Basler
explicitly recommends the `Bsl` name over the legacy `ChunkTimestamp`), then the
grab-result tick, then the host monotonic clock. Whichever was obtained is
recorded in `FramePacket.timestamp_source`, because it changes how much the
velocity can be trusted.

Two traps, both read from primary sources:

- `GetTimeStamp()`'s own docstring: "Cameras that do not support this feature
  return **ZERO**." A zero timestamp must be treated as *absent*, not as t=0.
- `TimestampLatch` / `TimestampLatchValue` exist, but the documentation warns of
  an "unspecified and variable delay". They **must not** be used for sub-
  millisecond host-to-camera alignment without characterising that delay first.

Chunk support was verified **at ace 2 USB family level, not per SKU** -- probe the
actual camera with `ChunkSelector.GetSettableValues()`. USB cameras provide **no
Framecounter chunk**.

### 6.4 The LIN floor is stricter than WHO

`min_lin_for_progressive = 0.35` is a **DESIGN CHOICE**, not a WHO criterion.
WHO's own wording admits progression "either linearly or in a large circle",
which a linearity floor rejects. It is applied because the downstream action is a
physical sort and a large-circle swimmer does not reliably leave the imaging
region. Setting it to 0.0 disables the criterion entirely -- including the
demotion applied when LIN cannot be computed -- for anyone who wants WHO's
wording followed literally.

By contrast, the 25 um/s and 5 um/s cut-points and the four-grade enum are
**WHO 6th ed. section 2.4.6.1 verbatim**, not inspired-by. The distinction is
recorded in `MOTILITY_PROFILE_VERSION = "who6-2021-s2.4.6.1-v1"`, stamped into
every audit record.

### 6.5 The four morphology aspects are MHSMA's, not WHO's

`head`, `acrosome`, `vacuole`, `tail` is **MHSMA's decomposition**. WHO 6th ed.
*requires* per-region defect reporting as **%H (head), %NM (neck+midpiece),
%T (tail), %C (excess residual cytoplasm)** -- and explicitly criticises reporting
only the proportion of normal forms.

The two schemes do not line up: acrosome and vacuole are *sub-features of WHO's
head category*, and there is **no midpiece and no cytoplasm aspect here at all**.
This project's morphology output is therefore not WHO defect reporting and must
never be presented as such. See `docs/safety_and_claims.md`.

### 6.6 The average-path window is a duration

`vap_window_ms = 100.0`, converted per track against that track's own measured
frame rate -- 17 frames at 160 fps, 5 frames at 50 fps. A **fixed frame count** is
what Mortimer et al. (2015) identify as producing "widely aberrant ALH values"
when the frame rate changes: five frames is 100 ms at 50 fps but 31 ms at 160
fps, so the same nominal setting smooths a third as much trajectory.

More fundamentally: the average path is an **algorithm, not a measurement**, so
VAP and everything defined relative to it -- STR, WOB, ALH, BCF -- are
algorithm-dependent and not comparable across CASA systems. WHO Fig. 4.4 says the
cross-instrument comparability "is not yet known". The resolved smoothing
parameters are appended to the profile version of every record for this reason.

### 6.7 The simulator's ground truth is a model, not a measurement

The simulator closes the one gap no public dataset closes: per-cell boxes *and*
per-cell morphology for the same cell. But its normal ranges come from WHO strict
criteria where a WHO number exists and are **documented modelling choices where
one does not**, and its timestamps are exact by construction. Velocity estimates
are therefore *better* on synthetic data than on hardware, and **a model that
only works on synthetic data should be assumed not to work on a camera.**

Nothing on the decision path reads the ground truth: it travels in
`FramePacket.meta` and is picked up only by the oracle detector and the
evaluation harness. That separation is what keeps the measurement honest.

---

## 7. The unmeasured physical constants

Every one of these is a property of the built instrument. Every one is `None` or
`calibrated: false` in configuration **by design**, and the system refuses rather
than substituting a plausible number.

| Constant | Config field | Default | What the system does without it |
|---|---|---|---|
| **Micrometres per pixel** | `calibration.optical.um_per_px` | `None`, `calibrated: false` | `require_calibrated()` raises. Velocities stay in px/s; the motility grade is `UNDETERMINED` with a reason naming the missing calibration. |
| **Transport delay** (imaging region → magnetic region) | `scheduling.transport_delay_ms` | `0.0`, `calibrated: false` | `ActuationScheduler.arm()` raises `CalibrationError`. Nothing is driven. |
| **Transport-delay spread** | `scheduling.transport_delay_std_ms` | `0.0` | Sets the pre/post activation margin (3 sigma). Without it the window cannot be sized. |
| **Field rise time** | `scheduling.field_rise_time_ms` | `0.0` | A FIELD_ON dispatched with no lead arrives late; the segment's leading edge passes un-diverted. |
| **Field fall time** | `scheduling.field_fall_time_ms` | `0.0` | A FIELD_OFF arrives late; the following segment's leading edge is diverted in error. |
| **Bulk flow vector** | `motion.flow_correction.fixed_vx_px_s` / `fixed_vy_px_s` | `None` | `mode=fixed_vector` is a **validation error**, not a silent zero. The default `robust_estimate` measures it live and reports *unavailable* below `robust_min_tracks`. |
| **Chamber / optical-section depth** | not a config field; `assess_feasibility(chamber_depth_um=...)` | `20.0` placeholder | Only affects the advisory concentration estimate, which scales inversely with it. |
| **Sample temperature** | `motion.thresholds.sample_temperature_c` | `37.0` asserted | Not measured by this software. Out of tolerance, grades carry a non-comparability note. |
| **Condenser NA, coverslip thickness** | not modelled | — | Affects the achieved resolution, not any computed value. |

Why this is enforced rather than encouraged:

> A wrong transport delay is the worst class of failure available in this device.
> Nothing raises, nothing looks wrong, and every shot gates the wrong segment of
> fluid.

and, for the optical scale:

> Reporting micrometres per second from an uncalibrated system would be a
> fabricated physical measurement.

The one exception is `configs/synthetic.yaml`, where the simulator's geometry and
timing are known *exactly by construction*. It asserts calibration -- and marks it
`calibration_id: simulator-exact-not-a-real-instrument`, so no audit log from it
can be mistaken for an instrument's.

---

## 8. Ranked: the assumptions most likely to be wrong

| Rank | Assumption | Why it is likely wrong | What it costs | Detection |
|---|---|---|---|---|
| **1** | **The C-mount coupler is 1x** | A 0.5x or 0.63x reducing coupler is very common and easy not to notice | **Every velocity out by a factor of 2** (0.5x). Every cell moves across the WHO boundaries in the same direction; the ratio and the sort change. Nothing raises. | Stage micrometer, and the 1.5x nominal-discrepancy check (`docs/calibration.md` §2.4). A 2.0x ratio is diagnostic. |
| **2** | **Brightfield gives adequate contrast on unstained live sperm** | Sperm are near-transparent phase objects; **phase contrast is the CASA norm** and is what every public dataset used | Low detector recall, short fragmented tracks, quality-bar failures, `INDETERMINATE` shots. Looks like a model problem and is not. | Compare detection recall on the rig against the simulator at matched density. Try phase contrast. |
| **3** | **37 C is actually held** | Requires a heated stage and a working controller; easily assumed | **No WHO-comparable velocity grading.** Cold cells are slower, so progressive counts and the ratio are systematically depressed. | Not detectable in software -- it is asserted, not measured. Instrument the stage. |
| **4** | The illumination can expose in tens of microseconds | Bright enough for <1 ms exposure through NA 1.25 on a phase object is demanding | Exposure lengthens; blur grows linearly (2.9 px at 1 ms, 100 um/s). No software fix. | Measure the achievable exposure at working illumination; check the best-frame `motion_blur` term distribution. |
| **5** | 20 um of chamber depth contributes countable sperm | At NA 1.25 the depth of field is very shallow | The 91 million/mL requirement is an **under**-estimate; shots time out on samples that "should" be dense enough. | `defocus_rate` in synthetic runs; the quality gate's `n_degraded` count on real data. |
| **6** | USB 3.0 sustains ~378 MB/s for this model | Computed here; **UNVERIFIED** against Basler's bandwidth table | Camera-reported skipped images, visible as `dropped_before` | `BslResultingAcquisitionFrameRate`; `frames_dropped_source`. Fix with Mono8 + ROI or binning. |
| **7** | 164 vs 160 fps | Vendor's own material disagrees with itself | Small; 160 is used conservatively | Read the achievable rate from the camera. |
| **8** | Flagellar beat is 10-20 Hz | **UNVERIFIED**, no primary source | Only affects an order-of-magnitude observation about BCF sampling. WHO notes BCF does not correlate with beat frequency anyway. | — |

---

## 9. Assumptions this project explicitly does **not** make

Stated because their absence is a design decision:

- That a shot's ratio means anything about a **patient**. A shot is 20-30 sperm in
  one second of flow.
- That `ai_eligible` means **healthy**. It means "progressive and all four aspects
  normal" -- observed phenotype and nothing else.
- That morphology and motility carry information about **DNA fragmentation,
  apoptosis or Annexin-V status** for an individual cell. They do not, and
  `docs/safety_and_claims.md` sets out why this is a structural limit rather than
  a data-volume problem.
- That the WHO **5th centile is a pass/fail line**. WHO section 8.1.3 says it is
  not: "The lower fifth percentile ... does not represent a limit between fertile
  and infertile men."
- That any public-dataset weights are **device-validated**. They are
  `public-research-baseline`, stamped as such into every checkpoint and every
  audit manifest.
- That **any model here has been trained or evaluated**. None has. Every threshold
  in `configs/` is a placeholder, `MorphologyConfig.model_id` and
  `weights_provenance` both default to the literal string `"unset"`, and there is
  no performance figure anywhere in this repository.
