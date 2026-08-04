# sperm_sorting

Real-time AI-guided sperm analysis and conditional magnetic sorting.

**Research prototype. Not a medical device. Not clinically validated.**

This software analyses *visible phenotype* from microscopy video — presence,
trajectory, velocity, direction, progression, linearity and morphology — and
emits a two-state `FIELD_ON` / `FIELD_OFF` command. It does **not** measure DNA
integrity, phosphatidylserine exposure, Annexin V binding, apoptosis, magnetic
labelling, fertility potential or pregnancy rate, and no output of this system
should be presented as measuring any of them. See
[safety_and_claims.md](docs/safety_and_claims.md).

---

## What it does

A semen sample flows continuously through the kit. Before entering, it is
incubated with Annexin V-conjugated magnetic microbeads. Two independent
mechanisms then act on it:

* **The magnetic layer** separates on Annexin V binding — a biochemical
  property this software cannot see.
* **The AI layer** — this repository — watches the flow through a microscope
  and decides, segment by segment, whether to energise the field.

The pipeline runs in exactly this order, and the order is enforced by the code
and by the tests:

```
microscopic semen video
  → sperm detection with bounding boxes
  → multi-object tracking, one persistent unique ID per sperm
  → velocity, direction, trajectory and linearity, flow-corrected
  → identification of progressive sperm
  → best-quality frame selected for each progressive sperm
  → crop of that exact tracked sperm
  → morphology model (MHSMA-pretrained, four independent heads)
  → head, acrosome, vacuole and tail evaluated separately
  → count sperm with BOTH acceptable progressive motility
    AND normal morphology in all four components
  → shot-level aggregation
  → FIELD_ON / FIELD_OFF
```

A sperm is `ai_eligible` — the internal term, deliberately not "healthy" — only
when every one of these holds:

```python
ai_eligible = (
    valid_unique_track
    and track_quality_pass
    and progressive_motility          # rapid OR slow, both count
    and head_normal
    and acrosome_normal
    and vacuole_normal
    and tail_normal
    and evaluation_complete_before_deadline
)
```

A **shot** is a software-defined segment of the continuous flow containing
25 ± 5 uniquely trackable sperm, closed after at most one second. Its ratio is

```
ai_eligible_ratio = unique ai_eligible sperm / total unique valid trackable sperm
```

— divided by *every* counted sperm, not only the progressive ones, and with
morphologically abnormal sperm deliberately left in the denominator.

### The decision rule

```python
if trackable_count < 20:
    status, command = "INDETERMINATE", "FIELD_OFF"
elif ai_eligible_count / trackable_count > 0.60:
    status, command = "ACCEPT", "FIELD_OFF"
else:
    status, command = "REJECT", "FIELD_ON"
```

Two things about this invert the product if misread, so they are stated plainly
and asserted in [test_decision_rule.py](tests/test_decision_rule.py):

* **Exactly 60% REJECTS.** The comparison is a strict `>`. 15/25 is a reject;
  16/25 is an accept. It uses exact rational arithmetic rather than floats,
  because 0.60 has no exact binary representation and the boundary is a hard
  product definition rather than an approximation.
* **`FIELD_ON` is the rejection.** Energising the magnet diverts the
  bead-bound population toward the waste channel. An accepted segment passes
  through with the field *off*. `FIELD_OFF` is also the safe state: every
  fault, watchdog expiry and shutdown path ends there, because an unsorted
  sample is a degraded outcome while a stuck-on field silently diverts
  everything.

---

## Quick start

```bash
git clone https://github.com/hannan72/sperm_sorting.git
cd sperm_sorting
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# What is installed, what is calibrated, what is missing
sperm-sorting doctor

# Can the optics and flow actually deliver 25 sperm/second?
sperm-sorting feasibility -c configs/synthetic.yaml

# Run the whole pipeline against the built-in simulator
sperm-sorting run -c configs/synthetic.yaml -n 500

# Replay a recording through the identical production graph
sperm-sorting run -c configs/replay.yaml --video clip.mp4

# The interactive demo
uvicorn web.app:app --reload

pytest -q
```

`make help` lists the shortcuts.

---

## Why there is a simulator

The public datasets that cover this problem do not overlap. **MHSMA** has
per-sperm morphology labels but no video. **VISEM-Tracking** has bounding boxes
and track IDs but no morphology. **VISEM** has only sample-level WHO
percentages. No public dataset provides per-sperm boxes *and* per-sperm
morphology for the same cell, so nothing public can validate the combined rule
this product rests on — only its two halves, separately.

[`src/sperm_sorting/simulator/`](src/sperm_sorting/simulator) closes that gap.
It samples a ground-truth health state and emits *both* a rendered image *and*
a trajectory for the same virtual sperm, so every sample is jointly labelled
for free. It therefore serves three purposes: the synthetic frame source that
lets the whole real-time pipeline be measured end to end against known truth, a
bootstrap training set for morphology before device data exists, and the demo's
data generator.

Synthetic data is not a substitute for real data, and a model that only works
on it should be assumed not to work on a camera. Its value is that it makes the
*pipeline* falsifiable.

---

## What the hardware implies

The specified build is a Basler a2A1920-160umPRO (mono, global shutter, 3.45 µm
pixels, ~160 fps) behind an Olympus PLN 100X oil objective (NA 1.25). That
fixes the geometry, and the geometry drives design decisions that are not
obvious until the arithmetic is done. `sperm-sorting feasibility` recomputes all
of it for any configuration; [assumptions.md](docs/assumptions.md) works
through it.

| Quantity | Value | Consequence |
|---|---|---|
| Sample-plane sampling | 0.0345 µm/px | 3.9× above Nyquist for NA 1.25 — 2×2 binning is available if USB bandwidth binds |
| Field of view | 66.2 × 41.4 µm | **Smaller than a whole spermatozoon (50–60 µm)** |
| Sperm head | 4.1 × 2.8 µm → 119 × 81 px | Detection targets the **head**, as CASA kinematics do |
| Motion blur at 100 µm/s | 0.055 px at 19 µs, 2.9 px at 1 ms | Exposure must stay well under a millisecond |
| Full-rate data | ~378 MB/s Mono8 | At the practical USB 3.0 ceiling; Mono12 cannot sustain 164 fps |

The field of view being shorter than a sperm is the load-bearing finding. It is
why the detector targets the head rather than the whole cell — which turns out
to align with both CASA (kinematics are defined on the head centroid) and
MHSMA (whose crops are head-centred with the tail not entirely visible). The
cost is that tail morphology is judged from a partial tail, and the code
reports `tail_complete=False` rather than pretending otherwise.

A second constraint emerged from actually running the pipeline: **the total
analysis latency must be shorter than the transport delay** between the imaging
region and the magnet. If it is not, every decision arrives after its fluid has
already passed, the scheduler drops every command, and nothing raises an error
— each component behaved correctly in isolation. `assess_feasibility` now
checks that arithmetic at startup.

---

## Nothing physical is guessed

Transport delay, field rise and fall times, micrometres-per-pixel and the bulk
flow vector are properties of a *built instrument*. They have no defaults here.
They are `None` or `calibrated: false` in configuration until measured, and the
system refuses rather than substituting a plausible number:

* Without optical calibration, velocities stay in pixels per second and
  motility grading returns `UNDETERMINED` — it will not compare µm/s
  thresholds against pixel values.
* Without scheduling calibration, `ActuationScheduler.arm()` raises and no
  command is driven.

This is not caution for its own sake. A wrong transport delay applies the field
to the wrong segment of fluid; a wrong µm/px rescales every velocity across the
WHO 25 and 5 µm/s boundaries. Both corrupt every result while raising nothing.
See [calibration.md](docs/calibration.md).

Optical calibration additionally cross-checks against the scale implied by the
optical train and refuses a result more than 1.5× away from it, because a 0.5×
reducing C-mount coupler is easy to overlook and puts every velocity out by
exactly a factor of two.

---

## Motility thresholds are WHO's, not ours

The four grades and both velocity limits come verbatim from the **WHO
laboratory manual, 6th edition (2021), §2.4.6.1**, which reinstated the
four-category a/b/c/d system that the 5th edition had collapsed into PR/NP/IM:

| Grade | Corrected VSL | Passes the motility filter |
|---|---|---|
| Rapidly progressive | ≥ 25 µm/s | yes |
| Slowly progressive | 5 to < 25 µm/s | yes |
| Non-progressive | < 5 µm/s with local movement | no |
| Immotile | no active tail movement | no |

WHO's PR is rapid + slow, so **both** progressive grades pass. Two caveats the
manual is explicit about and this code cannot fix for you: the limits are
*approximate*, and **velocity is temperature-dependent — WHO requires 37 °C**.
A run at another temperature is graded and flagged as not WHO-comparable rather
than silently reported as if it were.

One deviation is ours and is marked as such: `min_lin_for_progressive` is
*stricter* than WHO, whose wording admits progression "either linearly or in a
large circle". Set it to `0.0` to follow WHO literally.

The outputs derived from the smoothed average path — VAP, STR, WOB, ALH, BCF —
are algorithm-dependent and not comparable across CASA systems; WHO Fig. 4.4
states that cross-instrument comparability "is not yet known". The smoothing
window is therefore specified in **milliseconds, not frames**, because a fixed
frame count smooths over different durations at different frame rates, which
Mortimer et al. (2015) show produces aberrant ALH. On one synthetic trajectory
here the difference between a 100 ms window and a nominal five-frame window was
15×. ALH and BCF are refused entirely below ~50 Hz effective sampling and
reported as unavailable with a reason.

---

## Repository layout

```
src/sperm_sorting/
  acquisition/   Basler (pypylon), video replay, synthetic — one interface
  preprocessing/ ROI, normalisation, background subtraction, quality gate
  detection/     TOD-CNN-inspired and P2 high-resolution detectors, ONNX, oracle
  tracking/      ByteTrack, OC-SORT, BoT-SORT over a shared identity store
  motion/        CASA kinematics, flow correction, WHO motility grading
  quality/       Per-crop scoring and best-frame selection
  cropping/      Padded, aspect-preserving crop extraction
  morphology/    Shared backbone, four independent heads, calibration, metrics
  shots/         Counting gate, shot assembly, feasibility budget
  decision/      The 60% rule — small, pure, exhaustively tested
  scheduling/    Monotonic clocks, the future-command scheduler
  actuation/     Mock, GPIO and serial actuators, plus the watchdog
  calibration/   Optics, flow and transport-delay measurement
  monitoring/    Structured logging, audit records, metrics, health
  runtime/       Bounded queues, the pipeline, the threaded topology
  simulator/     Procedural generator: jointly-labelled image + trajectory
datasets/        Adapters, converters, validators (no data is redistributed)
training/        Training and evaluation, including end-to-end pipeline eval
web/             FastAPI demo, vanilla JS + canvas
docs/            Audits, architecture, assumptions, calibration, claims
tests/           Unit and integration tests
```

---

## Status

Built and verified: the full runtime path, the decision logic, the schemas,
calibration, scheduling, actuation, and the simulator. **No model has been
trained.** The detectors and the morphology network are implemented and
verified for shape, geometry, determinism and numerical correctness — not for
accuracy, because no weights exist. Any accuracy figure would be invented, so
none is quoted anywhere in this repository.

On this machine (CPU only, no CUDA) the detectors are far outside the real-time
budget at full sensor resolution: a 160 fps target allows ~6 ms per frame, and
P2Net measured ~634 ms per frame at 1920×1200 on CPU. These architectures
assume a GPU. See [engineering_report.md](docs/engineering_report.md) for the
full account of what runs, what does not, and what remains uncalibrated.

---

## Licence and data

Code is Apache-2.0. **No dataset is redistributed here.** Several of the public
datasets are non-commercial or legally unclear, and MHSMA additionally carries
a ShareAlike clause whose propagation to trained weights is
jurisdiction-dependent. Weights trained on public data are labelled
`public-research-baseline` and must not be presented as device-validated.
[license_audit.md](docs/license_audit.md) sets out each licence and a path to a
commercially clean system.

## Documentation

| Document | Contents |
|---|---|
| [assumptions.md](docs/assumptions.md) | Every assumption, its status, and the consequence if wrong |
| [safety_and_claims.md](docs/safety_and_claims.md) | What is and is not measured; regulatory framing |
| [pipeline.md](docs/pipeline.md) | Stage-by-stage order and the invariants each protects |
| [architecture.md](docs/architecture.md) | Components, data contracts, concurrency |
| [calibration.md](docs/calibration.md) | How to perform each calibration |
| [dataset_audit.md](docs/dataset_audit.md) | Exact formats, splits, quirks, licences |
| [source_audit.md](docs/source_audit.md) | What was verified from primary sources, and what was not |
| [license_audit.md](docs/license_audit.md) | Code, dataset and weight licensing |
| [domain_shift.md](docs/domain_shift.md) | Why public weights are baseline research weights |
| [engineering_report.md](docs/engineering_report.md) | Completed components, measurements, limitations, next steps |
| [README_FA.md](README_FA.md) | خلاصهٔ فارسی |
