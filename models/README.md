# models/

Trained weights, exported graphs and calibration artefacts. **Nothing in here
is committed** — see `.gitignore`.

## Layout

```
models/
  detector/
    p2net_visem.pt              detector trained on VISEM-Tracking
    p2net_device.pt             fine-tuned on device captures
    p2net.onnx                  exported for the ONNX Runtime backend
  morphology/
    mhsma_mobilenetv3.pt        four-head morphology model
    mhsma_mobilenetv3.calibration.json
                                per-aspect temperature and threshold
  calibration/
    optics.json                 micrometres per pixel, measured
    transport.json              transport delay and field rise/fall, measured
    flow_map.npy                position-dependent flow field, measured
```

## Weights provenance

Every checkpoint records a `weights_provenance` string, and it is stamped into
every audit log the model contributes to. The distinction is not cosmetic:

| Value | Meaning |
|---|---|
| `public-research-baseline` | Trained on public data (MHSMA, VISEM-Tracking). **Not device-validated.** The public sets were captured at 400× phase contrast or on stained smears, not at 100× oil brightfield, so these weights have never seen an image from this instrument. Treat their outputs as a starting point for fine-tuning, never as a measurement. |
| `synthetic-bootstrap` | Trained on simulator output. Useful for wiring up and smoke-testing the pipeline; a model that only works here should be assumed not to work on a camera. |
| `device-finetuned` | Trained or fine-tuned on captures from the actual instrument. The only provenance suitable for a real evaluation. |

Weights trained on MHSMA may inherit its CC BY-NC-SA terms, which are
non-commercial *and* share-alike. Whether that propagates to model parameters
is jurisdiction- and interpretation-dependent; the safe reading is that it
does. See `docs/license_audit.md` before any commercial use, and note that the
route to a clean system is retraining end to end on device-captured or
commercially licensed data.

## Calibration artefacts

`calibration/` holds measurements of *this* instrument, not of the software.
They are not portable between builds — a different coupler, objective, channel
or pump invalidates them. Each carries a `calibration_id` that is written into
every audit log, so a decision can always be traced to the calibration in force
when it was made.
