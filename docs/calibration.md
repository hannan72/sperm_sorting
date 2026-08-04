# Calibration

Four calibrations turn this software from a plausible-looking simulation into a
measuring instrument. Until they are done, the system deliberately refuses to
report physical units and refuses to actuate.

| # | Calibration | Produces | Gate it opens |
|---|---|---|---|
| 1 | **Optical** | `um_per_px` | Velocities in um/s; the motility grade at all |
| 2 | **Flow** | bulk `(vx, vy)` or an `(H, W, 2)` field | Flow-corrected kinematics |
| 3 | **Transport delay** | `transport_delay_ms` ± spread | The scheduler will arm |
| 4 | **Field rise/fall** | `field_rise_time_ms`, `field_fall_time_ms` | Commands land on time |

They are largely independent, but 1 must precede 2 if flow is to be reported in
physical units, and 3 and 4 are naturally measured in one session.

**None of them has a default.** Every value is `None` or `calibrated: false`
until measured, and `docs/assumptions.md` section 7 lists what the system does
instead of guessing.

---

## 1. Before starting

- Work on the assembled instrument, in its final optical and fluidic
  configuration. A calibration performed with a different coupler, a different
  objective, or the tubing rearranged is a calibration of a different instrument.
- Bring the stage to **37 C** and let it stabilise, for anything involving live
  sperm. WHO section 2.4.6 requires it because velocity is temperature-dependent.
- Record a `calibration_id` for every result. It is written into the audit
  manifest of every run that uses it, which is how a suspect run is later traced
  to a suspect calibration.
- Check what is already configured:

```bash
sperm-sorting doctor -c configs/device_v1.yaml
```

This prints, in colour, which calibrations exist and what the system will refuse
to do without them.

---

## 2. Optical calibration: micrometres per pixel

### 2.1 Why this one matters most

Every velocity reported in physical units passes through this single number.
Getting it wrong scales every VSL, which moves every sperm across the 25 um/s and
5 um/s WHO boundaries **in the same direction**, which changes the shot ratio,
which changes the sort. It is the most leveraged constant in the product.

It is also the one with a specific, common, silent failure: **a reducing C-mount
coupler**. A 0.5x adapter puts every velocity out by exactly a factor of two, and
the images look perfectly normal.

### 2.2 Equipment

- A **stage micrometer** (a graticule with certified rulings, usually 10 um
  pitch). This is the reference standard; nothing else substitutes.
- Immersion oil, if the objective is an oil objective. **NA 1.25 requires oil
  actually being present** -- dry, the effective NA collapses to about 1.0 and
  the objective does not form a proper image at all.
- The instrument's own camera, at the ROI, binning and pixel format the
  instrument will actually run at. Binning changes `um_per_px` by exactly the
  binning factor; a calibration at 1x1 is not valid at 2x2.

### 2.3 Route A: two-point (`calibrate_from_known_distance`)

The simple route, and the fallback when automatic detection fails on a
low-contrast image.

1. Focus the graticule. Fill the field with rulings if possible.
2. Capture a single frame at the operating ROI and pixel format.
3. Identify two rulings **as far apart as the field allows** -- marking error is a
   fixed number of pixels, so a longer span divides it down.
4. Measure the pixel distance between the two marks and note the physical
   distance between them (e.g. 20 rulings at 10 um = 200 um).
5. Run:

```python
from sperm_sorting.calibration import calibrate_from_known_distance
from sperm_sorting.config import OpticsConfig

result = calibrate_from_known_distance(
    pixel_distance=5797.0,        # measured
    physical_distance_um=200.0,   # certified
    optics=OpticsConfig(),        # for the nominal cross-check
    marking_uncertainty_px=2.0,   # how precisely each mark can be placed
)
```

`marking_uncertainty_px` is not decorative. Two independent marks means the
errors add in quadrature, and the resulting relative uncertainty
(`sqrt(2) x marking_uncertainty / pixel_distance`) is recorded and propagates
into every reported velocity. Record it honestly.

### 2.4 Route B: FFT graticule (`calibrate_from_graticule`)

Preferred when the rulings are visible across the field, because it uses **every**
ruling rather than two and so averages the marking error down.

1. Capture a frame with the rulings running perpendicular to one image axis.
2. Run:

```python
from sperm_sorting.calibration import calibrate_from_graticule

result = calibrate_from_graticule(
    image,                 # 2-D array
    ruling_pitch_um=10.0,  # certified pitch
    axis=1,                # 1 = rulings run vertically (period measured across x)
    min_period_px=4.0,
)
```

How it works, and why each step is there: the image is collapsed along the ruling
direction into a 1-D profile; a linear trend is removed, because an illumination
ramp across the field puts a huge spike at low frequency that would swamp the
rulings; a Hann window is applied; the dominant spatial frequency is found from
the real FFT, restricted to periods between `min_period_px` and a third of the
field; and the peak bin is refined by **parabolic interpolation**, because the
true period almost never lands exactly on a bin and this recovers most of the
sub-bin accuracy for free.

The uncertainty is derived from the peak's signal-to-noise against the rest of
the valid band, so it needs no repeated measurements.

### 2.5 The nominal cross-check — the coupler catcher

Both routes call `_check_against_nominal`, which compares the measurement with
the scale implied by the configured optical train
(`pixel_pitch / (objective_mag x coupler_mag)`, 0.0345 um/px for the reference
build) and **raises** if they disagree by more than `max_discrepancy` (default
1.5x):

```
measured scale 0.06900 um/px disagrees with the nominal 0.03450 um/px by 2.00x.
The most common cause is a reducing C-mount coupler (a 0.5x adapter gives
exactly 2.0x). Check the adapter, the objective, and the graticule pitch before
accepting this calibration.
```

The ratio is diagnostic, not just a flag:

| `nominal_ratio` | Almost certainly |
|---|---|
| ~1.00 | The optical train is as described |
| **~2.00** | A **0.5x** reducing coupler |
| ~1.59 | A **0.63x** reducing coupler |
| ~0.50 | A magnifying coupler, or a 200x objective in the turret |
| ~10 or ~0.1 | The graticule pitch was entered in the wrong unit |

**Do not raise `max_discrepancy` to make the error go away.** The check is the
only automatic defence against the single most likely wrong assumption in the
whole build (`docs/assumptions.md` section 8, rank 1). If the discrepancy is
real, fix the optical train or correct `OpticsConfig.coupler_magnification` to
the coupler that is actually fitted -- and then the check passes legitimately.

`OpticalCalibration._plausible` applies the same rule again at configuration
load, so a hand-edited YAML cannot smuggle a bad value past it.

### 2.6 Acceptance criteria

| Criterion | Target | If it fails |
|---|---|---|
| `nominal_ratio` within 1/1.5 to 1.5 | pass | Investigate the coupler and objective before anything else |
| `relative_uncertainty` | **< 0.01** (1%) | Use a longer span (Route A) or more rulings (Route B) |
| Route A vs Route B agreement | within 1% | A larger disagreement means the graticule detection locked onto the wrong periodicity |
| Repeat after refocusing | within 0.5% | Focus-dependent scale means the tube-lens spacing is wrong |
| Repeat at each ROI/binning in use | scales exactly with binning | Otherwise the ROI is being applied after a resize somewhere |

A 1% scale error moves the 25 um/s boundary by 0.25 um/s. Cells sit near that
boundary by construction, so 1% is a meaningful target rather than a nominal one.

### 2.7 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `no plausible ruling frequency in the spectrum` | Rulings out of focus, too low contrast, or running along the wrong axis | Refocus; try `axis=0`; fall back to Route A |
| Detected period is half or double the truth | The FFT locked onto a harmonic or a sub-harmonic of the ruling pattern | Tighten `min_period_px`; cross-check against Route A |
| Uncertainty near the 0.5 ceiling | Peak SNR near 1 -- the rulings are barely above the noise floor | Improve illumination and contrast; re-image |
| Calibration valid, velocities still wrong by a constant factor | Binning or ROI differs between the calibration frame and the run | Recalibrate at the operating configuration |
| `objective does not form a proper image` | No immersion oil on an oil objective | Apply oil |

### 2.8 Storing it

```python
from sperm_sorting.calibration import save_calibration
save_calibration(result, "models/calibration/optical-2026-08-04.json")

cfg_block = result.to_config(
    calibration_id="optical-2026-08-04", optics=OpticsConfig()
)
```

Then set in `configs/device_v1.yaml`:

```yaml
calibration:
  optical:
    calibrated: true
    calibration_id: optical-2026-08-04
    um_per_px: 0.03452
    relative_uncertainty: 0.004
```

---

## 3. Flow calibration: the bulk transport vector

### 3.1 Why

What the camera sees is not swimming:

```
observed motion = self-propulsion + bulk transport by the fluid
```

and in a microfluidic channel the second term is usually the larger one. A dead,
entirely immotile sperm carried at 331 um/s traces a long, arrow-straight track
-- high VSL, LIN near 1 -- and uncorrected it is graded **rapid progressive**.
That is the single most consequential silent error available in this pipeline,
because it pushes non-viable cells into the accepted fraction.

### 3.2 The rule that governs the whole procedure

> **Estimate flow from objects that are not swimming.**

Debris, or a control run with a non-motile sample (heat-treated or fixed).

The reason is worth stating plainly, because the shortcut is tempting. If flow
is estimated from live sperm, the estimate contains part of their own motility:
the population's mean velocity is `flow + mean(swimming)`, so subtracting it
removes the flow *and* the average swimming component, systematically
under-reporting VSL for every cell and, in the limit, driving the whole
population toward `NON_PROGRESSIVE`. The correction would be measuring the sample
against itself.

The runtime's default `ROBUST_ESTIMATE` mode is a mitigation of this, not an
exception to it: it takes the **slowest quantile** of live tracks (default 25%)
on the assumption that they are passively transported, and uses the **median**
rather than the mean so a handful of fast swimmers surviving the quantile cut
cannot drag the estimate. It is still an approximation, and a dedicated
calibration against genuinely non-motile objects is better.

### 3.3 Route A: fixed vector

Adequate when the imaging region sits in the middle of a wide channel.

1. Load a **non-motile control** -- fixed or heat-treated sperm, or a bead
   suspension at a similar size -- at the operating flow rate. Debris in a normal
   run also works, if there is enough of it.
2. Record 20-60 s through the ordinary pipeline, with
   `motion.flow_correction.mode: disabled` so nothing is subtracted twice.
3. Collect the resulting tracks and run:

```python
from sperm_sorting.calibration import calibrate_fixed_vector

result = calibrate_fixed_vector(tracks, quantile=0.25, min_tracks=8)
```

Per-track velocity is computed from **observed points only**, over a positive
duration, from first to last observed position. The estimate is the **median** of
the slowest `quantile` fraction; the spread is a median absolute deviation scaled
by 1.4826 (the consistent estimator of sigma for a normal), for the same
robustness reason.

### 3.4 Route B: flow map

Pressure-driven flow in a microchannel is **parabolic across the section**
(Poiseuille): fluid near a wall moves markedly slower than fluid at the centre.
A single vector therefore over-corrects at the walls and under-corrects at the
centre, and the sign of the residual error depends on where the cell was -- which
maps directly onto a spatially-varying bias in the motility grade.

```python
from sperm_sorting.calibration import calibrate_flow_map, save_flow_map

field, summary = calibrate_flow_map(
    tracks, height=1200, width=1920,
    grid=16, quantile=0.35, min_tracks_per_cell=3, smooth_sigma=1.5,
)
save_flow_map(field, "models/calibration/flow_map_v1.npy")
```

Slow tracks are binned onto a coarse 16x16 grid, a median is taken per cell,
empty cells are filled by nearest neighbour, the grid is Gaussian-smoothed, and
the result is bilinearly resampled to full resolution. **The coarse grid is
deliberate**: a per-pixel fit from a few hundred tracks would be mostly noise,
and the underlying profile is smooth anyway.

Then:

```yaml
motion:
  flow_correction:
    mode: flow_map
    flow_map_path: models/calibration/flow_map_v1.npy
```

### 3.5 Acceptance criteria

| Criterion | Target | Meaning |
|---|---|---|
| `n_tracks` used | >= 30 for a vector; >= 16 x `min_tracks_per_cell` for a map | Below `min_tracks` the function raises rather than guessing |
| `vx_std` / `|vx|` | **< 0.10** | A larger spread means pump pulsation, bubbles, or that live swimmers are contaminating the "slow" set |
| Direction | within a few degrees of the channel axis | A large cross-axis component means the ROI or the gate axis is misconfigured |
| Speed vs the pump setting | consistent | Order-of-magnitude disagreement means a unit error or a leak |
| Map cells populated directly | > 60% of 256 | Below that, most of the map is nearest-neighbour fill and a fixed vector is more honest |
| Residual after correction, on the control | median corrected VSL near 0 | The whole point: a non-motile control should grade `IMMOTILE` |

That last check is the one that actually validates the calibration. Replay the
control recording with the calibration applied; if non-motile objects still grade
progressive, the correction is wrong.

### 3.6 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `only N usable tracks; at least 8 are needed` | Too few objects, or tracks too short | Record longer; use a denser control; check the track-quality bar |
| `no grid cell had enough tracks` | Grid too fine for the data | Reduce `grid` or `min_tracks_per_cell` |
| Corrected VSL is systematically negative | Over-correction: live swimmers contaminated the slow quantile | Use a genuinely non-motile control; lower `quantile` |
| Corrected VSL is systematically positive | Under-correction: the true flow is faster than the slow quantile suggests | Same remedy |
| Flow drifts over a run | Pump pulsation, a bubble, a partial blockage | Use `robust_estimate` (which tracks drift with `robust_smoothing`) rather than a fixed vector, and fix the fluidics |
| Correction looks right mid-channel and wrong at the edges | Parabolic profile with a single vector | Use a flow map |

Note the interaction with the tracker: **do not enable BoT-SORT's camera-motion
compensation as a substitute.** The camera is rigidly mounted, so the global
motion CMC removes *is* the flow. Enabling it would pre-correct the trajectories
by an unrecorded amount, after which this stage would subtract a second, wrong
correction, and the resulting velocity would be neither raw nor corrected.

---

## 4. Transport-delay calibration

### 4.1 Why it cannot be guessed

The imaging region and the magnetic region are physically separated, so a
decision about the fluid under the microscope must be applied to the moment that
fluid arrives at the magnet.

> A wrong transport delay is the worst class of failure available in this device:
> nothing raises, nothing looks wrong, and **every shot gates the wrong segment of
> fluid**.

The scheduler therefore refuses to arm until this measurement exists
(`SchedulingConfig.require_calibrated()`), and `Application.setup()` arms it last.

### 4.2 Route A: tracer bolus — the method of record

1. Prepare a visible tracer: a dye bolus, a bead suspension, or an air gap. It
   must be visible at **both** observation points.
2. Establish steady flow at the operating rate. Let it settle -- the delay is
   only meaningful for the flow the instrument will run at.
3. Inject one bolus.
4. Record the instant its **leading edge** reaches the imaging region, and the
   instant it reaches the magnetic region. Use the same edge definition at both
   points, on the same clock. A monotonic clock, never wall time.
5. **Repeat at least three times.** This is enforced:

```
at least 3 trials are needed to estimate the spread of the transport delay;
got 1. The spread sets the activation margin, so it cannot be skipped.
```

```python
from sperm_sorting.calibration import estimate_from_tracer

result = estimate_from_tracer(
    imaging_times_s=[1.204, 5.881, 10.442, 15.007, 19.633],
    magnet_times_s =[1.327, 6.002, 10.566, 15.128, 19.759],
)
```

**Why three is a floor and not a ritual.** A single measurement gives a mean and
no way to size the margin. The pre- and post-activation margins default to
**three standard deviations** of the measured delay, so that the field is on
before the segment arrives in essentially every case rather than only on average.
With one trial there is no standard deviation, so there is no principled margin,
so the field would be commanded to the mean instant and would be late half the
time. Three is the minimum for a defensible spread; five to ten is better, and
the spread itself is the number that determines how precisely the device can
sort.

### 4.3 Route B: plug-flow geometry — a cross-check only

```python
from sperm_sorting.calibration import estimate_from_geometry

nominal_ms = estimate_from_geometry(
    channel_length_mm=12.0,
    channel_width_um=500.0,
    channel_height_um=100.0,
    volumetric_flow_ul_min=30.0,
)
```

This computes `length / mean_velocity` from the cross-section and the volumetric
flow rate.

**It is explicitly not a substitute**, and the reason is physical rather than
procedural. It assumes uniform **plug flow**, and pressure-driven flow in a
microchannel is not plug flow -- it is parabolic, and the centreline moves about
1.5-2x the mean. It also ignores Taylor dispersion (which is what turns a sharp
bolus into a smear, and is therefore the main contributor to the *spread* the
margin is sized from) and any dead volume in connectors and fittings. Every one
of those effects makes the real delay differ from the geometric one, and none of
them is small.

Its two legitimate uses:

- **A sanity check.** If the measured delay differs from the geometric estimate
  by more than about 2x, something is wrong -- a wrong flow rate, a leak, a
  blockage, or the two observation points are not where they are thought to be.
- **A starting estimate for the search window**, so the first tracer trial knows
  roughly when to start looking.

Trust the measurement.

### 4.4 Acceptance criteria

| Criterion | Target | If it fails |
|---|---|---|
| Trials | >= 3 (enforced), 5-10 preferred | The function raises below 3 |
| `transport_delay_std_ms / transport_delay_ms` | **< 0.10** | Above 0.5 the code logs a warning naming pump pulsation, bubbles or a leak. A wide spread forces a wide activation window, which directly reduces sorting precision. |
| Agreement with the geometry cross-check | within ~2x | Investigate the fluidics before accepting |
| Every delta positive | enforced | "every magnet arrival must be later than its imaging arrival; check that the two time series are paired and in the same order" |
| Repeat at each operating flow rate | delay scales inversely with flow | It is not a constant of the device; it is a constant of the device *at a flow rate* |

### 4.5 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `at least 3 trials are needed` | Only one or two boluses timed | Repeat |
| `every magnet arrival must be later` | Series swapped or mis-paired | Check the pairing and the order |
| Highly variable delay | Pump pulsation, a bubble, a leak, unsteady flow | Fix the fluidics. A syringe pump with a stepper is a common source of periodic pulsation |
| Delay measured, but commands still land late in a run | The *field* rise time was not measured (section 5) | Measure it |
| Commands dropped as late | `drop_if_late_by_ms` (default 50 ms) exceeded | The pipeline is not keeping up; check the stage latency percentiles. A dropped command is correct behaviour: firing it would divert the wrong segment |

### 4.6 Storing it

```python
from sperm_sorting.calibration import save_transport_calibration
save_transport_calibration(result, "models/calibration/transport-2026-08-04.json")
sched = result.to_config("transport-2026-08-04")
```

`to_config` sets `calibrated: true`, copies the delay and its spread, and sets
both `pre_activation_margin_ms` and `post_activation_margin_ms` to
`max(existing, 3 x std)`.

---

## 5. Field rise and fall time

### 5.1 Why it is separate

The transport delay says **when** the field must be on. The rise time says **how
much earlier to command it**:

```
t_activate  = t_gate + transport_delay
t_dispatch  = t_activate - settle_time - pre_activation_margin
```

`settle_time` is the **rise** time for a FIELD_ON and the **fall** time for a
FIELD_OFF. Using one for both biases every command in one direction, which is
why they are measured and stored separately.

An electromagnet is an inductor. Current -- and therefore field -- rises on an
`L/R` time constant, so rise and fall are generally **not equal**, particularly
if the driver has a flyback or snubber path that differs from its drive path.

### 5.2 Equipment and procedure

- A **Hall-effect probe or pickup coil** placed at the **magnetic region**, not at
  the driver. What matters is the field where the sperm are.
- An oscilloscope or DAQ capturing both the command line and the probe output on
  the same time base.

1. Command FIELD_ON. Record the command instant and the instant the probe first
   crosses the **settling threshold** -- the field level at which separation
   actually works, not the asymptote.
2. Repeat 5-10 times.
3. Repeat for FIELD_OFF, recording the fall.

```python
from sperm_sorting.calibration import estimate_field_switching

rise_ms, rise_std = estimate_field_switching(cmd_on_s, reached_on_s, rising=True)
fall_ms, fall_std = estimate_field_switching(cmd_off_s, reached_off_s, rising=False)
```

The threshold-crossing definition matters. An exponential approach asymptotes
slowly, so "fully settled" is not well defined and would overstate the required
lead. What is needed is the instant the field is *sufficient*, which is a
property of the separation, not of the coil.

### 5.3 Acceptance criteria

| Criterion | Target | If it fails |
|---|---|---|
| Trials | >= 5 per edge | — |
| `std / mean` per edge | < 0.10 | A variable switching time means an unstable supply or a thermally drifting coil |
| Rise + fall vs shot duration | `rise + fall << 1.0 s` | If switching is a significant fraction of a shot, consecutive shots interfere and the device cannot resolve them |
| Rise vs fall | may differ; both recorded | Do not assume symmetry |
| Field cannot settle before it was commanded | enforced | "the field cannot settle before it was commanded; check pairing" |
| Coil temperature after sustained operation | stable | Resistance rises with temperature, which changes `L/R` and therefore the rise time |

### 5.4 Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Rise time far longer than expected | High coil inductance, or a current-limited supply | A higher drive voltage with current limiting reduces the `L/R` rise; consider a different driver |
| Fall time far longer than rise | No flyback path, or a poorly chosen one | Add or resize the snubber. A slow collapse means the *following* segment is diverted in error |
| Rise time drifts during a run | Coil heating | Characterise hot and cold; use the worst case |
| Commands acknowledged but no field | Driver enable inverted, or `gpio_active_high` wrong | Check `ActuationConfig.gpio_active_high`; verify with the probe |
| Acknowledgement mismatch | Hardware reports a state other than the one commanded | The actuator forces the safe state and fails the command. Fix the firmware or the wiring; do not disable `require_acknowledgement` |

An actuator that **cannot** report its state returns `None` from
`_read_acknowledgement()`, in which case checking is skipped -- and that fact is
logged once at open time and appears in `describe()["can_acknowledge"]`, because
an unverifiable actuator is a real limitation that belongs in the audit log
rather than being silently assumed good.

---

## 6. After calibrating

Verify what the system now believes:

```bash
sperm-sorting doctor -c configs/device_v1.yaml
```

Expect green lines for both the optical and the scheduling blocks. Then check the
throughput budget against the *measured* scale rather than the nominal one:

```bash
sperm-sorting feasibility -c configs/device_v1.yaml --chamber-depth-um 20
```

The first line changes from `NOMINAL, uncalibrated` to `measured` once the
optical calibration is loaded, and every downstream figure -- field of view, flow
speed, residence time, implied concentration -- is recomputed from it.

Finally, replay a control recording end to end and confirm that a non-motile
control grades `IMMOTILE`, that a known-motile control grades progressive, and
that the shot ratios are what the control's composition implies. A calibration
that passes its own acceptance criteria and fails this is a calibration of
something other than the instrument.

### 6.1 When to recalibrate

| Change | Optical | Flow | Transport | Field |
|---|---|---|---|---|
| Objective, coupler, tube lens, camera swapped | **yes** | no | no | no |
| ROI or binning changed | **yes** | **yes** (px/s scale) | no | no |
| Pump rate or fluidics changed | no | **yes** | **yes** | no |
| Tubing, connectors or chamber replaced | no | **yes** | **yes** | no |
| Magnet, driver or wiring changed | no | no | no | **yes** |
| Routine interval | on any optical disturbance | per session | per session | periodically |

Each result carries its own `calibration_id`, and every audit manifest records
which ones were in force. That is what makes it possible to decide, months later,
whether a suspect run used a suspect calibration.
