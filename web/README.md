# Web demo

A small FastAPI service and a dependency-free HTML/JS page that make three
properties of this prototype tangible: that the synthetic generator *is* the
ground truth, that the decision rule works the way the specification says (and
not the way it is usually misremembered), and that the optical budget is tight
enough to constrain what the device can see.

It is a demo of the reasoning, not a demo of a working classifier. **No
morphology weights exist yet**, so `/classify` is served by
`RandomMorphologyEngine` and every prediction on the page is labelled
`untrained — predictions are not meaningful`. See
[What this does and does not demonstrate](#what-this-does-and-does-not-demonstrate).

---

## Running it

From the repository root:

```bash
pip install -e '.[web]'            # fastapi, uvicorn; the package itself must be importable
uvicorn web.app:app --reload       # http://127.0.0.1:8000
```

Or, without remembering the incantation:

```bash
python -m web.app                  # 127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/>.

There is no build step, no `npm install` and no bundler. `web/static/` holds
three files that are served as they are. The device runs offline, so the page
makes **no external requests of any kind** — no CDN, no web font, no remote
image. A test asserts this (`test_page_makes_no_external_requests`).

Run the API tests with:

```bash
pytest web/test_api.py
```

---

## Endpoints

| Method | Path        | What it does |
|--------|-------------|--------------|
| `GET`  | `/`         | The demo page (`static/index.html`), served with `Cache-Control: no-store`. |
| `GET`  | `/health`   | Liveness. Reports which morphology engine is loaded and whether it is trained. |
| `GET`  | `/aspects`  | The canonical vocabulary: aspect order, the 0/1 label convention, motility classes, CASA feature order, slider ranges, the polarity contract and the five mandated decision cases. The frontend hard-codes none of these; it reads them from here. |
| `GET`  | `/config`   | Resolved `AppConfig` summary plus the full feasibility report from `assess_feasibility`, warnings included. |
| `POST` | `/generate` | Samples one virtual sperm and observes it. |
| `POST` | `/classify` | Predicts the four aspects and the motility grade. |
| `POST` | `/decide`   | Applies the real decision rule to one (eligible, trackable) pair. |

### `POST /generate`

Body (all fields optional; shown with defaults):

```jsonc
{
  "seed": 1234,
  "prevalences": {"head": 0.25, "acrosome": 0.35, "vacuole": 0.20, "tail": 0.30},
  "motility": null,            // null samples a grade; or "rapid_progressive", ...
  "progressive_rate": 0.6,     // used only when "motility" is null
  "aspects": null,             // or [head, acrosome, vacuole, tail] to force the flags
  "knobs": {},                 // continuous overrides, e.g. {"head_axis_ratio": 2.8}
  "image_size": 128,           // 64 or 128 (MHSMA parity)
  "image_um_per_px": null,     // null derives the scale from the 25.6 um crop field
  "blur_px": 0.6,
  "noise_sigma": 4.0,
  "n_points": 96,
  "fps": 160.0,
  "track_um_per_px": 0.5,
  "flow_vx_px_s": 0.0,
  "flow_vy_px_s": 0.0
}
```

Returns the base64 PNG (no `data:` prefix), the trajectory in pixels, the CASA
kinematics both as observed and flow-corrected, the complete `HealthState`, and
the ground-truth labels: `true_label`, `true_aspects`, `true_motility`.

**The image and the trajectory come from the same `HealthState`.** Three
independent child generators are spawned from the one seed — one for the state,
one for the render, one for the track — so changing the frame rate cannot
silently change which cell you are looking at, and the same seed reproduces the
response byte for byte.

### `POST /classify`

Body is either `{"image": "<base64 png>"}` or `{"seed": N, "params": {...}}`.
The two are not equivalent and the response says which was used
(`input_kind`). An image on its own carries no trajectory, so `pred_motility`
comes back `undetermined` with the reason attached rather than being guessed.

Returns `pred_label`, `pred_aspects`, `pred_motility`, `probs` (P(normal) per
aspect), `all_four_normal`, and a `model` block whose `provenance`,
`untrained`, `untrained_warning` and `reads_the_image` fields the page renders
prominently.

The overall verdict is not assembled in the handler. A `TrackRecord` is
populated and `compute_eligibility()` is called, because that method is the one
place the per-sperm rule is allowed to live — and it also returns *why* a sperm
was rejected, which a hand-rolled `and` chain would not.

### `POST /decide`

```jsonc
{"ai_eligible_count": 15, "trackable_count": 25, "threshold": null, "minimum_trackable": null}
```

`threshold` and `minimum_trackable` default to the resolved configuration.
The handler calls `sperm_sorting.decision.engine.decide` and does no arithmetic
on the counts itself. A numerator larger than the denominator is a `422`, not a
`500`.

Returns the `Decision` fields — `status`, `field_command`, `ratio`,
`rationale` — plus `field_command_meaning`, `is_rejection`,
`exactly_at_threshold`, `boundary_rule` and `minimum_rule`.

---

## The two things people get wrong

**Exactly 60% REJECTS.** The comparison is a strict `>`, evaluated in exact
rational arithmetic (`Fraction(n, d) > Fraction("0.60")`) because 0.60 has no
binary representation and the boundary must not depend on rounding. 15/25 is a
REJECT; 16/25 is the first accepting count.

**FIELD_ON is the rejection.** Energising the magnet diverts the segment to the
waste channel. FIELD_OFF passes it through to collection, and is also the safe
state used when no decision could be made. Reading FIELD_ON as "good" inverts
the product.

The five mandated cases, which the demo preloads as buttons and the test suite
asserts exactly:

| eligible / trackable | ratio | status | field command |
|----------------------|-------|--------|---------------|
| 15 / 25 | 60.00% | REJECT | `FIELD_ON` |
| 16 / 25 | 64.00% | ACCEPT | `FIELD_OFF` |
| 12 / 20 | 60.00% | REJECT | `FIELD_ON` |
| 13 / 20 | 65.00% | ACCEPT | `FIELD_OFF` |
| 19 / 19 (timeout) | 100.00% | INDETERMINATE | `FIELD_OFF` |

The last row is the interesting one: a shot that timed out with only 19
trackable sperm is INDETERMINATE *even when every one of them was eligible*.
The minimum is a precondition for trusting the ratio at all, not a tie-breaker.

---

## What this does and does not demonstrate

### It does demonstrate

- **That the generator is the ground truth.** The four binary labels are drawn
  first; they drive the continuous shape and motion parameters; those are the
  only inputs to the renderer and the trajectory simulator. Flip one aspect to
  abnormal and the label, the picture and (for the tail) the kinematics all
  change together.
- **The health rule, conjunctively.** All four aspects normal *and* a
  progressive grade. Any single defect disqualifies. The per-aspect table makes
  a single wrong aspect visible instead of averaging it into an accuracy figure.
- **The exact decision rule**, computed server-side by the shipping
  implementation, with a boundary ladder in which every row is a separate
  `/decide` response.
- **The optical budget**, straight from `assess_feasibility`: the sampling in
  um/px, the field of view, the ~119 px a sperm head spans, and the fact that a
  whole 53 um spermatozoon does *not* fit across the 41.4 um short axis of the
  field — with the warnings rendered rather than summarised away.
- **The real motility grading rule.** `pred_motility` comes from
  `sperm_sorting.motion.classifier.classify_motility` — the production
  WHO-threshold classifier — applied to the simulated kinematics. That half of
  the prediction is the shipping implementation. `motility_source` says so.

### It does not demonstrate

- **A working morphology classifier.** There are no trained weights. The four
  aspect probabilities come from `RandomMorphologyEngine`, which draws from a
  seeded generator and *never looks at the pixels*. Classifying the same crop
  twice gives different answers. Any apparent agreement with the ground truth is
  chance. The moment real weights are configured at `morphology.weights`, the
  service loads them instead and the banner changes accordingly — the fallback
  is not hard-coded.
- **Anything about real sperm.** Every image and every track is procedurally
  generated. The renderer is a stylised ellipse-plus-flagellum model, not a
  microscope simulation, and the simulator's own CASA implementation is a clean
  textbook one operating on gap-free, evenly spaced points — conditions the
  production estimator does not enjoy.
- **Detection, tracking, best-frame selection, cropping, shot assembly,
  scheduling or actuation.** The demo hands the decision engine counts you type
  in. It does not produce those counts from a video.
- **DNA integrity, apoptosis, or fertility.** The AI observes visible phenotype
  only — motility and morphology. The magnetic separation the device performs
  acts on Annexin V binding, which this software does not and cannot observe.
  The two mechanisms are complementary, not equivalent.
- **Clinical validity of any kind.** Research prototype, not a medical device.
  Nothing here has been validated on human samples.

---

## Design notes

- **All product logic is server-side.** The 60% rule, the health rule, the
  aspect order, the label convention, the motility grades and even the slider
  ranges arrive from `/aspects`, `/config` and `/decide`. Where `app.js` appears
  to make a decision, it is choosing which server-returned rows to display,
  never what they say.
- **Continuous-knob overrides are announced.** The knobs are normally *caused*
  by the binary flags. Setting one by hand keeps the label and changes the
  evidence, so the label and the picture can then disagree. `/generate` echoes
  `overridden_knobs` and `label_pixel_link_intact`, and the page shows a warning
  strip while any override is active.
- **Nothing is signalled by colour alone.** Agreement, warnings and the
  ACCEPT/REJECT verdict each carry a symbol and a word as well as a colour.
- **Light and dark** both supported via `prefers-color-scheme`, plus
  `prefers-reduced-motion` (the trajectory is drawn statically instead of
  animated).
- **The crop is upscaled by an integer factor with smoothing disabled**, so
  every visible block is one real sample. A fractional factor would resample,
  and a resampled pixel is a pixel the generator never produced.
- **The morphology engine is built once**, in a FastAPI lifespan handler (not
  the deprecated `@app.on_event`), and closed on shutdown. Access is guarded by
  a lock because `numpy.random.Generator` is not thread-safe and FastAPI runs
  `def` endpoints in a worker thread pool.

## Known limitations

- The random morphology engine's `p_normal_rate` is set to 0.5 rather than the
  class default of 0.87. A 0.87 rate would produce a table that agrees with the
  ground truth most of the time and would read as a model that mostly works.
  A coin flip reads as what it is.
- With a non-zero bulk flow, `casa` (used for grading) is the flow-corrected
  reading and `casa_observed` is what a camera would see. The correction here is
  exact because the simulator knows the flow it injected; the real pipeline must
  *estimate* it, and that estimate is a genuine source of error this demo does
  not model.
- `MotionFeatures.optically_calibrated` is set true for simulated tracks because
  the simulator knows its own scale exactly. That is a property of the
  simulation, not a claim about the device — `/config` reports the device's
  calibration state separately, and it is `false`.
- The trajectory `track_um_per_px` slider defaults to 0.5 um/px, a round number
  chosen for legibility. The reference build's actual sample-plane sampling is
  0.0345 um/px, as reported in the optical-budget panel.
