# Safety and claims

What this system measures, what it does not measure, what it cannot in principle
measure, and how it must and must not be described.

Read this before writing anything user-facing about the project -- a README, a
paper abstract, a grant application, a slide, a product page. The regulatory
position set out in section 6 turns on *claims*, not on technology, so the words
chosen in those documents are the thing under discussion, not commentary on it.

---

## 1. What the system measures

Visible phenotype from monochrome microscopy video, and nothing else:

| Measured | From |
|---|---|
| Presence of a sperm head | detector output |
| Trajectory | tracked head centroid across frames |
| Velocity -- VCL, VSL, VAP | `motion/features.py`, from measured points and real timestamps |
| Direction of travel and its stability | net heading; circular standard deviation of per-step headings |
| Progression (the WHO four-category motility grade) | flow-corrected VSL against the 25 and 5 um/s limits, plus a linearity floor |
| Linearity ratios -- LIN, STR, WOB | derived from the above |
| ALH, BCF | derived from the average path, and **refused** below the configured sampling rate |
| Morphology, four binary aspects: head, acrosome, vacuole, tail | a CNN over a single crop of that tracked cell |

That is the whole list. It is exactly the list in the package docstring, and it
is deliberately short.

## 2. What the system does not measure

**It does not measure any of the following, and it must never be described as
doing so:**

- DNA fragmentation or DNA integrity
- Phosphatidylserine (PS) externalisation
- Annexin V binding
- Apoptosis, or apoptotic signalling
- Whether a given sperm is magnetically labelled or bead-bound
- Viability or vitality (which requires a membrane-integrity assay)
- Acrosome reaction status
- Chromatin condensation, oxidative stress, aneuploidy
- Fertility potential, fertilisation probability, or pregnancy rate
- Live birth rate

This list is not a stylistic preference. Sections 4 and 5 set out why several of
these are not merely unmeasured but **unmeasurable by this method**, and why more
data would not change that.

## 3. Why the term is `ai_eligible`, never "healthy"

Internally, a sperm satisfying the combined rule is `ai_eligible`
(`TrackRecord.ai_eligible`). The rule is:

```
ai_eligible = valid unique gated track
          AND passed the track-quality bar
          AND flow-corrected motility is progressive (rapid OR slow)
          AND head, acrosome, vacuole and tail are ALL normal
          AND the evaluation completed before its deadline
```

Read that literally: **the term names the pipeline's own decision about the
evidence it collected.** It says "this software, from these pixels, under this
configuration, found nothing disqualifying". It is a statement about an
observation, not about a cell.

"Healthy" would be a statement about the cell -- and would smuggle in four claims
the system has not made:

1. **A biological claim.** Health is a property of a cell's genome, membranes and
   metabolism. This system observes shape and movement. Section 4 shows that the
   link between the two is weak at the population level and absent at the cell
   level.
2. **A clinical claim.** "Healthy" implies fitness for fertilisation. That is a
   fertility-potential claim, which is precisely what makes something an IVD
   under EU IVDR (section 6).
3. **A completeness claim.** It implies nothing else is wrong. Two of the aspects
   WHO *requires* be reported -- midpiece and excess residual cytoplasm -- have no
   aspect here at all (section 5.3).
4. **A transitivity claim.** It invites the reader to conclude that a "healthy"
   sperm is Annexin-V-negative and DNA-intact -- which is the specific inference
   section 4 shows to be unsupported.

Two places in the code make the boundary visible rather than relying on
convention. The `IneligibilityReason` enum records exactly *how* a sperm failed
(`NOT_PROGRESSIVE`, `ABNORMAL_TAIL`, `DEADLINE_MISSED`, ...), so a decision is
always explainable as a set of observations rather than a verdict. And the
simulator, which does have a genuine generative ground truth, calls its label
`overall_label` and notes explicitly that the runtime calls the same property
`ai_eligible` because the pipeline observes phenotype only.

Note also the third status. A shot with fewer than 20 trackable sperm is
`INDETERMINATE`, not "bad" -- the system distinguishes "we looked and found few
qualifying sperm" from "we could not look properly", and the field stays off in
the second case.

---

## 4. Why a visual AI cannot infer Annexin-V positivity or DNA fragmentation

This is the most important section in the document, because it is the claim the
project's context most invites: the physical sort acts on Annexin V binding, this
software decides which segments to divert, so it is tempting to describe the
software as predicting what the magnet would do.

**It does not, and it cannot.** Four independent reasons, of increasing severity.
The fourth is not a limitation of current models; it is structural.

### 4.1 WHO's own register is associative, not causal

The manual never asserts a mechanism. Its wording throughout is
"associated with", "may also have", "only partially related":

- Section 3.2.1: sperm DNA fragmentation is described as **"only partially
  related"** to semen quality.
- Section 2.5: abnormal spermatozoa "generally have a lower fertilizing potential
  ... and **may also** have abnormal DNA"; morphological defects have been
  **"associated with"** increased DNA fragmentation.

A source that consistently declines to assert a mechanism is not a source that
supports predicting one cell's molecular state from its appearance.

### 4.2 The effect sizes are weak

Meta-correlation of morphology against the DNA fragmentation index
(*Reprod Sci* 2025, PMID 40312558):

| Relationship | r | Variance explained |
|---|---|---|
| Normal morphology vs DFI | **-0.30** | ~9% |
| Abnormal morphology vs DFI | **+0.39** | ~15% |

So between 85% and 91% of the variance in DNA fragmentation is **not** explained
by morphology. And in multivariable analysis the association does not survive
adjustment at all:

> "None of the morphological indices independently predicted SDF after adjustment
> for sperm concentration and progressive motility."

A predictor that explains a tenth of the variance and loses significance under
adjustment is not a basis for a per-cell label.

### 4.3 The level-of-inference gap: an ecological fallacy

This is the reason that is most often missed, and it cannot be fixed by more
data.

**Every published morphology-to-DNAf and motility-to-DNAf correlation is between
per-ejaculate summary percentages, across men.** The unit of observation is a
man's sample: "this ejaculate has 6% normal forms and a DFI of 22%". The
correlation is over a set of such pairs.

Inferring a *single cell's* status from a relationship between *group-level
summaries* is an **ecological fallacy**. Two ejaculates with identical 4% normal
forms can have very different DFI, and nothing in a between-men correlation
constrains which individual cells within either sample are fragmented.

Concretely: even a hypothetical perfect between-men correlation of r = 1.0
between "% normal forms" and "% DNA-fragmented" would tell you nothing about
whether *this* normal-looking spermatozoon is one of the fragmented ones. The
correlation is a fact about population means; the question is about a member.

More data does not help, because the data has the wrong unit. What would be
needed is per-cell paired observations -- and section 4.4 is why those cannot
exist.

### 4.4 The physical problem: the cell you assay is never the cell you use

WHO strict morphology requires a **fixed, air-dried, stained** cell --
Papanicolaou. The cell is dead before it is assessed.

Every DNA and apoptosis assay is likewise destructive or label-dependent:

| Assay | What it requires |
|---|---|
| Annexin V | binding a Ca²⁺-dependent PS-binding protein to the cell surface; in MACS, conjugated to superparamagnetic microbeads |
| TUNEL | fixation and permeabilisation |
| Comet | lysis and electrophoresis |
| SCSA | acid denaturation and staining |
| WHO strict morphology | fixation, air-drying, staining |

> **The cell you assay is never the cell you use.**

So a per-cell visual model **cannot even be ground-truthed** against Annexin-V
positivity on the same spermatozoon. There is no procedure -- not now, not with
more funding -- that yields "this exact cell looked like *X* and was
Annexin-V-positive", because determining the second destroys or labels the cell
whose unlabelled appearance was the first.

This is not an argument that the correlation is weak. It is an argument that the
training pair does not exist.

(Annexin V is also not specific to apoptosis. It marks PS exposure, which arises
from **either** activated apoptotic signalling **or** simple membrane damage.
And DNA fragmentation occurs *later* than PS externalisation, so PS-positive and
DNA-fragmented are non-identical populations by construction. Even a perfect
Annexin-V predictor would not be a DNA-fragmentation predictor.)

### 4.5 Even the biochemical sort is of uncertain benefit

For completeness, because it bounds what the whole device can claim -- not only
the software.

WHO section 5.6, on MACS: a Cochrane systematic review "did not see differences
in clinical or live birth between MACS and sperm selected by hyaluronic acid
binding (HA-ICSI) or other selection techniques, on live birth."

Cochrane CD010461 (Garg et al., updated 2026; 8 RCTs, 4147 women): **"We are
uncertain whether MACS improves live birth"**, certainty **very low**. Live birth
RR 1.95 (0.89-4.29) from a single RCT of 62 women; clinical pregnancy RR 1.05
(0.84-1.31) over 413 women with I² = 81%.

And WHO's own reference 184 -- Martinez MG et al., *J Assist Reprod Genet*
2018;35(12):2215-21 -- states the finding in its title: "Magnetic-activated cell
sorting is not completely effective at reducing sperm DNA fragmentation."

So the physical mechanism this software gates is itself of unproven clinical
benefit. Any claim about outcomes would have to clear that bar first.

### 4.6 The bottom line

> Motility and morphology correlate with DNA fragmentation at the **population**
> level, **weakly** (|r| ~ 0.2-0.4). There is **no** validated causal pathway and
> **no** per-cell predictive model permitting a microscopy AI to output
> "Annexin-V positive", "DNA-fragmented" or "apoptotic" for an individual
> spermatozoon.

Those labels must not appear in the output schema, in the UI, in a README, or in
a plot legend, **except as an explicitly-flagged research hypothesis**. They do
not appear in `schemas/`, and `IneligibilityReason` contains no such member.

The software's relationship to the magnet is *sequential*, not *predictive*: it
decides which segments of fluid to divert, on visual grounds. The magnet then
acts on Annexin V binding, on biochemical grounds. The two mechanisms are
**complementary, not equivalent**, and the software does not and cannot observe
the second.

---

## 5. What the kinematic and morphological outputs actually are

### 5.1 VAP, STR, WOB, ALH and BCF are algorithm-dependent

The **average path** is an algorithm, not a measurement. It is a smoothed version
of the observed track, and its smoothing method and window length are part of the
*definition* of everything derived from it.

The dependency chain:

```
average path  (an algorithmic choice)
     |-- VAP   = speed along it
     |-- STR   = VSL / VAP
     |-- WOB   = VAP / VCL
     |-- ALH   = lateral displacement about it
     `-- BCF   = crossings of it
```

**All five inherit the smoother.** WHO says so directly in its own definitions:
VAP is "calculated according to the algorithm embedded in the CASA system; these
algorithms are different in different systems, so the values may not be
comparable between systems, or with different acquisition parameters such as
framerate", and ALH likewise -- "Different CASA systems calculate ALH using
different algorithms, so the values may not be comparable."

And the caption of **WHO Figure 4.4** states the general position:

> "Different CASA instruments use different mathematical algorithms... The
> comparability of measurements across all instruments **is not yet known**."

Only **VCL** and **VSL** are defined without reference to a smoothed path -- and
VCL is still frame-rate dependent, since summing step lengths over a finer
sampling picks up more of the true path.

This is why the progressive-motility rule in this project is built on **VSL**,
which is the most transportable of the quantities available, and why the resolved
smoothing parameters are appended to `MOTILITY_PROFILE_VERSION` per track. A bare
threshold version would not fully identify how a number was produced.

Mortimer et al. (2015) make the practical consequence concrete: "ALH values are
not consistent between instruments and so cannot be standardized across CASA
platforms", and systems using a **fixed five-point** smoother "will provide
inadequate smoothing and hence widely aberrant ALH values". Hence
`MotionConfig.vap_window_ms` -- a duration, converted per track against that
track's own measured frame rate.

### 5.2 BCF does not measure what its name suggests

WHO defines BCF as the "average frequency at which the curvilinear path crosses
the average path" -- and then adds, citing Gallagher et al., *Hum Reprod*
2019;34:1173-85:

> **BCF "has been shown to not correlate with flagellar beat frequency".**

So a BCF value must never be described as a flagellar beat frequency, however
strongly the name suggests it. This project reports BCF as `bcf_hz` with an
explicit `bcf_unavailable_reason` field, and refuses to compute it below the
configured sampling rate rather than emitting a number that would be aliased
downward.

### 5.3 The four morphology aspects are not WHO's defect categories

**WHO 6th ed. requires per-region defect reporting** as %H (head), %NM (neck and
midpiece), %T (tail) and %C (excess residual cytoplasm), and criticises the
common practice of reporting only the proportion of normal forms.

**This project's four aspects are MHSMA's decomposition, not WHO's:**

| This project | WHO category |
|---|---|
| head | %H |
| acrosome | a **sub-feature** of %H |
| vacuole | a **sub-feature** of %H |
| tail | %T |
| *(none)* | **%NM -- neck and midpiece: not covered** |
| *(none)* | **%C -- excess residual cytoplasm: not covered** |

The two schemes do not correspond. Three of this project's four aspects live
inside WHO's head category, and two of WHO's four categories have no aspect at
all. **The morphology output is therefore not WHO defect reporting and must not
be presented as such.**

Two further gaps in the same direction:

- WHO's normal-head criteria are a **conjunction of shape, acrosome coverage
  (40-70% of head area) and vacuole rules**, plus a length-to-width ratio outside
  1.5-2.0 being abnormal. The morphometric medians (head 4.1 x 2.8 um) are
  **descriptive statistics of 77 cells already classified normal** -- they are not
  the rule. A model thresholding on "4.1 x 2.8 um" is not implementing WHO strict
  criteria.
- WHO's dimensions come from **Papanicolaou-stained** cells: fixed, air-dried,
  stained. Dimensions measured from unstained brightfield or phase-contrast
  images are not comparable, because "each stain provides quite different results
  down to the level of sperm sizes".

### 5.4 The velocity limits are WHO's; the linearity floor is not

Worth stating precisely, because half of this is a citation and half is a design
choice.

**WHO 6th ed. section 2.4.6.1, verbatim in substance:** rapidly progressive
>= 25 um/s; slowly progressive 5 to <25 um/s; non-progressive <5 um/s; immotile.
Four categories, reinstated in the 6th edition because "presence (or absence) of
rapid progressive spermatozoa is clinically important" -- the 5th edition had
collapsed grading to PR/NP/IM. WHO's PR is a+b, NP is c, IM is d.

`MotilityClass` **is** that four-category system, and 25/5 um/s **are** those
numbers, not values inspired by them. `MOTILITY_PROFILE_VERSION =
"who6-2021-s2.4.6.1-v1"` names the source rather than claiming mere inspiration.

**`min_lin_for_progressive = 0.35` is this project's own criterion.** WHO's
wording admits progression "either linearly or in a large circle", which a
linearity floor rejects. It is applied because the downstream action is a
physical sort and a large-circle swimmer does not reliably leave the imaging
region. Setting it to 0.0 follows WHO's wording literally.

Three caveats WHO itself attaches and this code cannot fix for the user:

- The limits are **approximate**, and are stated alongside two other
  formulations -- half a tail length per second, and one head length per second --
  that agree only roughly. Björndahl & Kirkman-Brown give a third: "a rapidly
  progressive spermatozoon is one that moves >5 head-lengths per second". At a
  4.1 um head that is 20.5 um/s, against half a tail length at 22.5 um and the
  manual's 25 um/s.
- WHO's editorial board notes that a human grader is **not expected to measure
  velocity at all**, and that per-cell velocity is "only possible by CASA".
- **37 C is required.** WHO section 2.4.6: "The velocity of motile spermatozoa is
  temperature dependent." Applying these thresholds at uncontrolled room
  temperature does not produce a WHO-comparable grading. This software records
  the asserted temperature and flags out-of-specification runs; it does not
  measure the temperature.

### 5.5 The reference limits are not a pass/fail line

If a report ever places a measured value against WHO Table 8.3, it must carry
WHO's own qualification. Section 8.1.3:

> "The lower fifth percentile of data from men in the reference population (Table
> 8.3) does not represent a limit between fertile and infertile men."

Section 1.3 repeats it: "these percentiles do not represent distinct limits
between fertile and subfertile men." Clinical decision limits "still need to be
developed."

The editorial board's suggested phrasing for a value below a limit is that it is
**"not typical" for a highly fertile man** -- not "abnormal", not "infertile".

Two traps in the table itself: non-progressive 1% and immotile 20% are
**distributional centiles, not lower limits** (a low immotile count is *good*),
and the 95th centile for concentration is **208** x 10⁶/mL, which several vendor
tables mis-transcribe as 200.

---

## 6. Regulatory framing

Not legal advice. An engineering record of what the instruments say, so that the
claims made about this project are made with the consequences in view.

### 6.1 EU -- IVDR (Regulation (EU) 2017/746)

An in vitro diagnostic medical device is one intended for the in vitro
examination of specimens to provide information on a physiological state,
disease, or **fertility**. **Semen is explicitly a specimen type.**

> **The intended purpose is the trigger, not the technology.**

The same code is or is not an IVD depending entirely on what it is claimed to do.

**Classification.** Annex VIII implementing rule 1.4, verbatim via MDCG 2020-16
rev.4:

> "Software which drives a device or influences the use of the device shall fall
> within the same class as the device. If the software is independent of any
> other device, **it shall be classified in its own right**."

The second sentence is the one that matters here: standalone analysis software is
not exempt by virtue of not being the instrument.

Rule 6, verbatim: "Devices not covered by the above-mentioned classification
rules are classified as class B."

- A **professional-use** semen-analysis IVD most plausibly falls to rule 6 →
  **Class B**.
- A **lay self-test** falls to rule 4(a), under which self-tests are Class C
  "except for devices for the detection of pregnancy, **for fertility testing**
  and for determining cholesterol level... class B".

Either route lands on **Class B**, and **Class B and above require Notified Body
involvement**.

**This class assignment is a reasoned reading of the rules, not an official
determination.**

**The research carve-out, and its limit.** Article 1(3) excludes products
genuinely intended for research **without a medical purpose**; Article 2(45): "A
device intended to be used for research purposes, without any medical objective,
shall not be deemed to be a device for performance study."

### 6.2 US -- FDA

Automated semen analysers are regulated as **Class II** devices under
**21 CFR 864.5220** (automated differential cell counter), **product code POV**,
cleared through the **510(k)** pathway. Review panels: Hematology (81) and
OB-GYN (85). Verified against 510(k) records **K071737**, **K220828** (SQA-iO),
**K242830** (LensHooke X3 PRO) and **K183602** (SwimCount).

An AI/ML component makes such a product **Software as a Medical Device**.

Whether the Clinical Decision Support exemption (FD&C Act 520(o)(1)(E)) could
apply is **UNVERIFIED** -- the reasoning that it generally does not cover software
whose basis a clinician cannot independently review, i.e. a black-box CNN, is
**general knowledge, not a verified citation**, and is recorded as such.

### 6.3 A research-use-only label does not protect a project that makes a claim

The single most important sentence in this section:

> **An RUO label does not protect you if marketing, documentation or actual use
> asserts a diagnostic claim. Intent is inferred from the whole record, not from
> the disclaimer.**

The "whole record" includes: the README; the paper abstract; the conference
slide; the demo video; the grant application; the field names in the output
schema; the labels in the UI; what the sales conversation said; and what users
were observed doing with it.

Concretely, each of these would assert a claim that a disclaimer at the bottom of
the page does not retract:

- Calling `ai_eligible` sperm "healthy", "viable" or "high quality".
- Reporting a value against the WHO 5th centile as a pass or fail.
- Describing the output as a "semen analysis" or a "sperm quality score".
- Outputting, or plotting, an estimated DNA-fragmentation or apoptosis figure.
- Claiming equivalence to, or automation of, WHO manual assessment.
- Implying a fertility, fertilisation or pregnancy outcome.

The remedy is not more disclaimers. It is to make no such claim anywhere.

---

## 7. Required disclaimers

**These are mandatory. They belong in the README, in the CLI and API banner, and
in every exported report.**

### 7.1 Research use only

> **Research use only. Not a medical device. Not for diagnostic or treatment
> decisions, and not for clinical sperm selection.**

Present in the CLI help text (`cli.py`), the package docstring (`__init__.py`),
and the audit manifest.

### 7.2 Not validated against WHO manual assessment

> **Not validated against WHO 6th edition manual assessment.**

State the comparison that was actually performed, if any. WHO's own Figure 7.3
uses a **Bland-Altman plot of manual-versus-CASA agreement** as the expected form
of evidence; a correlation coefficient is not that evidence.

At the time of writing, **no model in this repository has been trained and no
validation of any kind has been performed.** There are no performance figures
anywhere in this project, and there must be none until there are measurements to
report.

### 7.3 No pass/fail verdict

Do not report a verdict. Report values with the 5th centile as *context*, and use
WHO's own language: below a limit means **"not typical for a highly fertile
man"**, not "abnormal" and not "infertile" (section 5.5).

Note that the pipeline's own `ACCEPT` / `REJECT` / `INDETERMINATE` are statuses
**of a shot** -- 20-30 sperm in one second of flow, a sorting decision about a
segment of fluid. They are not a verdict about a sample, and still less about a
person.

### 7.4 No molecular labels from image features

Do not output Annexin-V, apoptosis or DNA-fragmentation labels derived from image
features. Section 4.

### 7.5 Declare the acquisition parameters with every kinematic output

Frame rate, **temperature**, chamber depth, magnification, and the smoothing
window. And **suppress or flag VAP, STR, WOB, ALH and BCF when the frame rate is
unknown, variable, or below ~60 Hz.**

Implemented: `MotionFeatures` carries `timestamp_source`,
`mean_frame_interval_s`, `flow_correction_mode`, `optically_calibrated`,
`um_per_px` and `profile_version` on every record, and `alh_um` / `bcf_hz` carry
their own `*_unavailable_reason` strings rather than a silent zero.

**Attribution note:** the ~60 Hz figure is **Mortimer et al. (2015)**, not WHO.
WHO 6th ed. specifies **no numeric minimum frame rate**; its only temporal
specification is that "at least 1 second is enough for basic CASA measurements".
Citing WHO for that number would be an attribution error, and a common one.

### 7.6 State the algorithm dependence explicitly

Quote WHO Figure 4.4 (section 5.1) wherever VAP, STR, WOB, ALH or BCF are
reported.

### 7.7 Do not claim equivalence to WHO methods

Section 8.

---

## 8. How this project positions itself

Björndahl & Kirkman-Brown, *Fertil Steril* 2022;117:246-51 (open access, CC BY),
on the temptation this project is subject to, quoted verbatim:

> "There are many present and emerging technologies for semen analysis which for
> commercial and/or simplicity reasons wish to assert equivalence to these core
> methods. As described in the manual, it may be better to focus on the potential
> of those technologies as separate alternative diagnostics, rather than the
> pretense that they are a universal, accurate, and appropriate answer to
> automate this core method."

**This project takes the authors' advice.**

It positions itself as a **separate research instrument**, not as an automation
of the WHO method. Specifically:

- It **is** a real-time, per-cell, phenotype-based gating system for a
  microfluidic sorter, whose output is one bit -- energise the magnet, or do not.
- It **uses** WHO's four-category motility grading and WHO's own approximate
  velocity limits, because they are the best-documented thresholds available, and
  it names them as WHO's rather than presenting them as its own.
- It **does not** claim to reproduce a WHO semen analysis, to substitute for one,
  or to be comparable with one. It does not produce the parameters a WHO analysis
  produces -- there is no concentration, no total count, no vitality, no percentage
  normal forms computed the way WHO computes it -- and its morphology
  decomposition is MHSMA's, not WHO's (section 5.3).
- It **does not** claim equivalence to any commercial CASA instrument. WHO Figure
  4.4 says cross-instrument comparability "is not yet known", which applies to
  this instrument too.
- It **does not** claim to predict, detect or estimate anything molecular.

If a future version of this work is to be compared with manual assessment, the
comparison must be an actual study, with an actual protocol, reported as
Bland-Altman agreement, on samples assessed by trained technicians against WHO
6th-edition criteria at 37 C. Until that exists, the honest statement is the one
in section 7.2.

---

## 9. Checklist before publishing anything

| Check | Where the answer is |
|---|---|
| Does any text call `ai_eligible` sperm "healthy", "viable", "good" or "high quality"? | §3 |
| Does any output, plot or field name mention DNA fragmentation, apoptosis, Annexin V or PS? | §4, §7.4 |
| Is any WHO reference limit presented as a pass/fail line? | §5.5, §7.3 |
| Is the morphology output described as WHO defect reporting (%H/%NM/%T/%C)? | §5.3 |
| Are VAP/STR/WOB/ALH/BCF reported without the algorithm-dependence caveat? | §5.1, §7.6 |
| Is BCF described as a flagellar beat frequency? | §5.2 |
| Is the ~60 Hz frame rate attributed to WHO? | §7.5 |
| Is the 25 um/s limit described as "WHO-inspired" rather than as WHO's own? | §5.4 |
| Is the LIN floor presented as a WHO criterion? | §5.4 |
| Does any performance figure appear for a model that has not been trained? | §7.2 |
| Is equivalence to WHO methods or to a commercial CASA system claimed or implied? | §8 |
| Are the RUO and not-validated disclaimers present in the README, the banner, and every exported report? | §7.1, §7.2 |
| Is `weights_provenance` set to something other than `"unset"`, and is it accurate? | `docs/domain_shift.md` §2 |
| Was the run optically calibrated, and does the report say so? | `docs/calibration.md` |
| Was the sample at 37 C, and does the report say so? | §5.4 |
