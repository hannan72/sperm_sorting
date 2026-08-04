#!/usr/bin/env python3
"""Evaluate a morphology checkpoint on one split, at its shipped operating point.

    python training/eval_morphology.py --checkpoint runs/morph/best.pt \\
        --calibration runs/morph/calibration.json --split test \\
        --source synthetic -o runs/morph/eval_test

What it reports, per aspect: sensitivity, specificity, precision, NPV,
macro-F1, balanced accuracy, MCC, ROC-AUC, PR-AUC, expected calibration error
and the 2x2 confusion matrix. Plus a macro row, and -- separately and
deliberately -- the **all-four-normal joint accuracy**.

Three things this script insists on
-----------------------------------

**It evaluates the operating point that will actually ship.** The calibration
bundle's temperatures and thresholds are applied, so the numbers describe the
model *plus* its decision rule. Evaluating at a bare 0.5 would describe neither:
on the 4.6%-prevalence tail aspect, 0.5 predicts "normal" for everything.

**Raw accuracy is not reported.** Not once, not as a footnote. A model that
calls every tail normal scores 95.4% and catches nothing, and any table
containing that number will eventually be quoted without its context.
Balanced accuracy, which is prevalence-invariant, is the accuracy-like figure
on offer.

**Low positive counts are called out by name.** Any aspect with fewer than
:data:`~training.common.morphology_report.LOW_POSITIVE_WARNING` positives in
the evaluated split gets an explicit warning naming the count, both on the
console and in the JSON. The MHSMA validation split has **7** abnormal tails
out of 240; a sensitivity computed from seven positives moves by 0.14 when one
of them changes side, and that has to be visible next to the number, not
buried in a methods section.

The all-four-normal number
--------------------------
The product does not consume four probabilities, it consumes one boolean:
:attr:`sperm_sorting.schemas.morphology.MorphologyResult.all_four_normal`. That
is a conjunction, so the four error rates compound and the joint accuracy is
**not** the mean of the four per-aspect accuracies. Both are printed, adjacent,
so the gap between them is impossible to miss.

Outputs, in ``--out``: ``eval_<split>.json``, ``experiment.json`` and
``plots/``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.bootstrap import ensure_importable

ensure_importable()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from sperm_sorting.constants import MORPHOLOGY_ASPECTS  # noqa: E402
from sperm_sorting.errors import SpermSortingError  # noqa: E402
from sperm_sorting.morphology.calibration import (  # noqa: E402
    CalibrationBundle,
    expected_calibration_error,
    maximum_calibration_error,
    sigmoid,
)
from sperm_sorting.morphology.metrics import (  # noqa: E402
    evaluate_aspects,
    format_metrics_table,
    metrics_to_json_dict,
)
from sperm_sorting.morphology.model import load_checkpoint  # noqa: E402
from sperm_sorting.morphology.polarity import POLARITY_CONVENTION  # noqa: E402
from training.common.args import (  # noqa: E402
    build_parser,
    describe_device,
    dump_json,
    resolve_config,
    resolve_device,
)
from training.common.experiment import ExperimentRecord  # noqa: E402
from training.common.logging_utils import console_table, print_block  # noqa: E402
from training.common.morphology_data import (  # noqa: E402
    SOURCE_KINDS,
    SPLIT_NAMES,
    MorphologyArrayDataset,
    load_morphology_source,
)
from training.common.morphology_report import (  # noqa: E402
    LOW_POSITIVE_WARNING,
    all_four_normal_agreement,
    low_positive_warnings,
    per_aspect_confusion,
    write_morphology_plots,
)
from training.common.seeding import seed_everything  # noqa: E402


def build_argument_parser() -> Any:
    parser = build_parser(
        description="Evaluate a morphology checkpoint on one split.",
        epilog=(
            "Examples:\n"
            "  python training/eval_morphology.py --checkpoint runs/m/best.pt \\\n"
            "      --calibration runs/m/calibration.json --split test --source synthetic\n"
            "  python training/eval_morphology.py --checkpoint runs/m/best.pt \\\n"
            "      --split valid --source mhsma --data-root data/mhsma\n"
        ),
    )
    model = parser.add_argument_group("model")
    model.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint written by train_morphology.py (best.pt or last.pt).",
    )
    model.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Calibration bundle. Defaults to <checkpoint dir>/calibration.json when "
            "that exists. Without one, the raw sigmoid and a 0.5 threshold are used "
            "and the output says so loudly -- that is not a shippable operating "
            "point on the low-prevalence aspects."
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument("--source", choices=SOURCE_KINDS, default="synthetic")
    data.add_argument("--data-root", type=Path, default=None)
    data.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default="test",
        help="Which official split to evaluate.",
    )
    data.add_argument("--n-train", type=int, default=2000)
    data.add_argument("--n-valid", type=int, default=500)
    data.add_argument("--n-test", type=int, default=500)
    data.add_argument("--image-size", type=int, default=None)
    data.add_argument("--batch-size", type=int, default=64)
    data.add_argument("--num-workers", type=int, default=0)

    output = parser.add_argument_group("output")
    output.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    output.add_argument(
        "--min-positive",
        type=int,
        default=LOW_POSITIVE_WARNING,
        help="Warn when an aspect has fewer than this many positives in the split.",
    )
    return parser


def load_calibration(
    checkpoint: Path, explicit: Path | None
) -> tuple[CalibrationBundle | None, str]:
    """Find and load the calibration bundle, or explain its absence.

    The search mirrors
    :meth:`sperm_sorting.morphology.inference.MorphologyEngine.find_calibration_sidecar`
    so that evaluating a checkpoint uses the same bundle the runtime would pick
    up for it. An evaluation at a different operating point from the deployed
    one is a measurement of something the product does not do.
    """
    if explicit is not None:
        return CalibrationBundle.load_json(explicit), str(explicit)
    for candidate in (
        checkpoint.with_suffix(".calibration.json"),
        checkpoint.parent / "calibration.json",
    ):
        if candidate.exists():
            return CalibrationBundle.load_json(candidate), str(candidate)
    return None, ""


def collect_logits(*, model: Any, loader: Any, device: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-aspect ``P(abnormal)`` **logits** and MHSMA labels over a split."""
    import torch

    model.eval()
    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model.logits_tensor(images.to(device))
            logit_chunks.append(logits.detach().float().cpu().numpy())
            label_chunks.append(targets.detach().cpu().numpy())

    if not logit_chunks:
        empty = {name: np.zeros(0, dtype=np.float64) for name in MORPHOLOGY_ASPECTS}
        return empty, dict(empty)

    logits_all = np.concatenate(logit_chunks, axis=0)
    labels_all = np.concatenate(label_chunks, axis=0)
    return (
        {n: logits_all[:, i].astype(np.float64) for i, n in enumerate(MORPHOLOGY_ASPECTS)},
        {n: labels_all[:, i].astype(np.int64) for i, n in enumerate(MORPHOLOGY_ASPECTS)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        common = resolve_config(args)
    except SpermSortingError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    out_dir = common.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentRecord(script="eval_morphology", out_dir=out_dir)
    record.args = {
        **common.to_json_dict(),
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration) if args.calibration else None,
        "source": args.source,
        "split": args.split,
        "batch_size": args.batch_size,
        "min_positive": args.min_positive,
    }

    with record:
        try:
            _run(args, common, record)
        except SpermSortingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            record.finish("failed", str(exc))
            record.save()
            return 1
    return 0


def _run(args: Any, common: Any, record: ExperimentRecord) -> None:
    from torch.utils.data import DataLoader

    cfg = common.cfg
    out_dir = common.out_dir

    record.determinism = seed_everything(cfg.run.seed, cfg.run.deterministic)
    record.set_config(cfg)
    device = resolve_device(common.device)
    record.hardware = describe_device(device)

    # --- model -----------------------------------------------------------
    # load_checkpoint enforces the polarity contract: a checkpoint trained
    # under the opposite convention is refused rather than silently inverting
    # every number below.
    model, info = load_checkpoint(args.checkpoint, map_location=str(device))
    model = model.to(device).eval()
    record.model = {
        **model.describe(),
        "checkpoint": str(args.checkpoint),
        "model_id": info.get("model_id", ""),
        "weights_provenance": info.get("weights_provenance", "unset"),
        "created_utc": info.get("created_utc", ""),
        "label_polarity": info.get("label_polarity", ""),
    }

    bundle, bundle_path = load_calibration(Path(args.checkpoint), args.calibration)

    # --- data ------------------------------------------------------------
    image_edge = int(args.image_size or model.input_size[0])
    source = load_morphology_source(
        args.source,
        root=args.data_root,
        seed=cfg.run.seed,
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        image_size=image_edge,
    )
    split = source.splits[args.split]
    reserved = {"name", "licence", "source", "splits"}
    record.set_dataset(
        name=str(source.info.get("name", args.source)),
        licence=str(source.info.get("licence", "unrecorded")),
        splits={name: len(s) for name, s in source.splits.items()},
        source=str(source.info.get("source", "")),
        evaluated_split=args.split,
        split_detail={name: s.to_json_dict() for name, s in source.splits.items()},
        **{k: v for k, v in source.to_json_dict().items() if k not in reserved},
    )

    dataset = MorphologyArrayDataset(split, augmentation=None, base_seed=cfg.run.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    # --- inference -------------------------------------------------------
    logits, labels = collect_logits(model=model, loader=loader, device=device)

    if bundle is not None:
        probabilities = bundle.apply(logits)
        thresholds = dict(bundle.thresholds)
        operating_point = f"calibrated bundle: {bundle_path}"
        temperatures = dict(bundle.temperatures)
    else:
        probabilities = {name: sigmoid(values) for name, values in logits.items()}
        thresholds = dict.fromkeys(MORPHOLOGY_ASPECTS, 0.5)
        temperatures = dict.fromkeys(MORPHOLOGY_ASPECTS, 1.0)
        operating_point = (
            "UNCALIBRATED: no bundle found, using the raw sigmoid at a 0.5 threshold. "
            "On the 4.6%-prevalence tail aspect this predicts 'normal' for "
            "everything; the numbers below do not describe a shippable operating "
            "point."
        )
        record.note(operating_point)

    # --- metrics ---------------------------------------------------------
    results = evaluate_aspects(labels, probabilities, thresholds)
    for aspect in MORPHOLOGY_ASPECTS:
        if aspect in results:
            results[aspect]["mce"] = maximum_calibration_error(
                probabilities[aspect], labels[aspect], 15
            )
            results[aspect]["ece"] = expected_calibration_error(
                probabilities[aspect], labels[aspect], 15
            )

    confusion = per_aspect_confusion(labels, probabilities, thresholds)
    joint = all_four_normal_agreement(labels, probabilities, thresholds)
    warnings = low_positive_warnings(labels, args.split, minimum=int(args.min_positive))

    plot_paths: dict[str, str] = {}
    if args.plots:
        plot_paths = write_morphology_plots(
            labels=labels,
            probabilities=probabilities,
            thresholds=thresholds,
            results=results,
            plot_dir=out_dir / "plots",
            prefix=args.split,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "calibration_bundle": bundle_path or None,
        "operating_point": operating_point,
        "label_polarity": POLARITY_CONVENTION,
        "probability_space": "P(abnormal); thresholds compared as P(abnormal) >= t",
        "split": args.split,
        "n": len(split),
        "weights_provenance": record.model.get("weights_provenance", "unset"),
        "temperatures": temperatures,
        "thresholds_p_abnormal": thresholds,
        "per_aspect": metrics_to_json_dict(results),
        "confusion": confusion,
        "all_four_normal": joint,
        "positive_counts": split.positive_counts(),
        "prevalence": split.prevalence(),
        "low_positive_warnings": warnings,
        "raw_accuracy_reported": False,
        "raw_accuracy_note": (
            "Raw accuracy is deliberately not computed anywhere in this report. "
            "A constant 'normal' predictor scores 95.4% on the MHSMA tail aspect. "
            "Use balanced_accuracy, which is prevalence-invariant."
        ),
    }
    metrics_path = dump_json(out_dir / f"eval_{args.split}.json", payload)
    record.metrics = payload
    record.artifact("metrics", metrics_path)
    for key, path in plot_paths.items():
        record.artifact(f"plot_{key}", path)
    for warning in warnings:
        record.note(warning)

    # --- console ---------------------------------------------------------
    table = format_metrics_table(
        results,
        title=(
            f"morphology metrics on the '{args.split}' split "
            "(positive class = ABNORMAL, MHSMA label 1)"
        ),
    )
    joint_row = [
        {
            "quantity": "all-four-normal joint accuracy",
            "value": joint["joint_accuracy"],
            "detail": "predicted all-normal vs true all-normal",
        },
        {
            "quantity": "mean per-aspect accuracy",
            "value": joint["mean_per_aspect_accuracy"],
            "detail": "NOT the same thing -- shown for contrast",
        },
        {
            "quantity": "sensitivity (all-normal)",
            "value": joint["sensitivity_all_normal"],
            "detail": "of truly all-normal sperm, how many we would accept",
        },
        {
            "quantity": "specificity (all-normal)",
            "value": joint["specificity_all_normal"],
            "detail": "of sperm with any defect, how many we correctly reject",
        },
        {
            "quantity": "precision (all-normal)",
            "value": joint["precision_all_normal"],
            "detail": "of the sperm we accept, how many really are all-normal",
        },
        {
            "quantity": "true all-normal rate",
            "value": joint["true_all_normal_rate"],
            "detail": "prevalence of the conjunction in this split",
        },
    ]

    lines = [
        "",
        table,
        "",
        console_table(
            joint_row,
            ("value", "detail"),
            index_column="quantity",
            title=(
                "the conjunctive rule the product actually uses "
                "(all_four_normal); this is NOT the average of the four rows above"
            ),
        ),
        "",
        f"operating point : {operating_point}",
        f"checkpoint      : {args.checkpoint}",
        f"provenance      : {record.model.get('weights_provenance', 'unset')}",
    ]
    if warnings:
        lines += ["", *warnings]
    lines += ["", f"wrote {metrics_path}"]
    print_block(lines)


if __name__ == "__main__":
    raise SystemExit(main())
