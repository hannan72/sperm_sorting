"""Reporting shared by the morphology trainer and evaluator.

Three things live here because both scripts need them and a second copy of any
of them would eventually disagree with the first:

* the **low-positive warning**, which is the difference between a table a
  reader can act on and one they will over-read;
* the **all-four-normal joint accuracy**, which is the quantity the product
  actually depends on and is emphatically *not* the mean of the four per-aspect
  numbers;
* the **plot set**, so a training run and a later evaluation of the same
  checkpoint produce figures that can be laid side by side.

Everything here is in **abnormal-positive space**: probabilities are
``P(abnormal)``, labels are MHSMA integers verbatim, thresholds are compared as
``P(abnormal) >= t``. There is no polarity flip in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sperm_sorting.constants import MORPHOLOGY_ASPECTS

__all__ = [
    "LOW_POSITIVE_WARNING",
    "all_four_normal_agreement",
    "low_positive_warnings",
    "per_aspect_confusion",
    "write_morphology_plots",
]

#: Below this many positives in a split, a per-aspect metric is reported with
#: an explicit warning rather than as a number a reader can act on. Twenty is
#: not a statistical threshold, it is a legibility one: at n = 20 a single
#: flipped example moves sensitivity by 0.05, which is already larger than most
#: of the differences anyone would try to read from the table. The MHSMA
#: validation split has **7** abnormal tails out of 240, so this fires on real
#: data, not only on degenerate folds.
LOW_POSITIVE_WARNING: int = 20


def low_positive_warnings(
    labels: Mapping[str, np.ndarray],
    split_name: str,
    *,
    minimum: int = LOW_POSITIVE_WARNING,
) -> list[str]:
    """One warning line per aspect whose positive count is too small to read.

    The count is *named* in every message. "Tail metrics are unreliable" is a
    sentence people learn to skip; "tail has 7 abnormal examples out of 240, so
    one changing side moves sensitivity by 0.14" is a number they cannot.
    """
    warnings: list[str] = []
    for aspect in MORPHOLOGY_ASPECTS:
        values = labels.get(aspect)
        if values is None:
            continue
        values = np.asarray(values).ravel()
        n_positive = int(np.sum(values == 1))
        if n_positive < minimum:
            step = 1.0 / max(n_positive, 1)
            warnings.append(
                f"WARNING: aspect '{aspect}' has only {n_positive} abnormal example(s) "
                f"out of {values.size} in the {split_name} split. Sensitivity, F1, MCC "
                f"and PR-AUC for '{aspect}' are all estimated from those {n_positive} "
                "and must not be reported as performance figures -- one example "
                f"changing side moves sensitivity by {step:.2f}."
            )
    return warnings


def per_aspect_confusion(
    labels: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """``2x2`` confusion matrix and counts per aspect, in JSON-safe form.

    The matrix is ``[[tn, fp], [fn, tp]]`` with **abnormal as the positive
    class**, matching
    :func:`sperm_sorting.morphology.metrics.confusion_matrix`. ``fn`` is called
    out separately in the output because it is the dangerous cell on this
    product: a missed abnormality promotes a sperm into ``all_four_normal``.
    """
    from sperm_sorting.morphology.metrics import confusion_counts

    out: dict[str, dict[str, Any]] = {}
    for aspect in MORPHOLOGY_ASPECTS:
        if aspect not in labels or aspect not in probabilities:
            continue
        threshold = float(thresholds.get(aspect, 0.5))
        y_true = np.asarray(labels[aspect]).ravel().astype(np.int64)
        y_pred = (np.asarray(probabilities[aspect]).ravel() >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        out[aspect] = {
            "threshold_p_abnormal": threshold,
            "matrix_true_by_pred": [[tn, fp], [fn, tp]],
            "row_order": ["true_normal", "true_abnormal"],
            "column_order": ["pred_normal", "pred_abnormal"],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "missed_abnormal": fn,
        }
    return out


def all_four_normal_agreement(
    labels: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Agreement on the conjunctive ``all_four_normal`` rule.

    **This is the number the product depends on, and it is not the average of
    the four per-aspect accuracies.** The runtime rule
    (:attr:`sperm_sorting.schemas.morphology.MorphologyResult.all_four_normal`)
    is a conjunction, so the four error rates compound: four heads each 90%
    accurate can agree with the truth on the conjunction far less than 90% of
    the time, and how much less depends on how correlated their errors are --
    which no per-aspect table shows.

    The positive class here is **normal-on-all-four**, because that is what
    makes a sperm ``ai_eligible``. Note the deliberate inversion relative to
    the per-aspect metrics, where the positive class is abnormal: here the
    conjunction of "not abnormal" is the event of interest, and stating it the
    other way round would leave the reader to work out the complement.
    """
    aspects = [a for a in MORPHOLOGY_ASPECTS if a in labels and a in probabilities]
    if not aspects:
        return {"n": 0, "joint_accuracy": float("nan"), "note": "no aspects present"}

    length = int(np.asarray(labels[aspects[0]]).ravel().size)
    predicted = np.ones(length, dtype=bool)
    truth = np.ones(length, dtype=bool)
    for aspect in aspects:
        threshold = float(thresholds.get(aspect, 0.5))
        predicted &= np.asarray(probabilities[aspect]).ravel() < threshold
        truth &= np.asarray(labels[aspect]).ravel() == 0

    agree = int(np.sum(predicted == truth))
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    per_aspect_accuracy = [
        float(
            np.mean(
                (np.asarray(probabilities[a]).ravel() >= float(thresholds.get(a, 0.5)))
                == (np.asarray(labels[a]).ravel() == 1)
            )
        )
        for a in aspects
    ]

    return {
        "definition": (
            "predicted all-four-normal vs true all-four-normal; the positive class "
            "here is NORMAL-on-all-four, because that is what makes a sperm "
            "ai_eligible"
        ),
        "aspects": aspects,
        "n": length,
        "joint_accuracy": ratio(agree, length),
        "mean_per_aspect_accuracy": float(np.mean(per_aspect_accuracy)),
        "true_all_normal_rate": ratio(int(np.sum(truth)), length),
        "predicted_all_normal_rate": ratio(int(np.sum(predicted)), length),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "sensitivity_all_normal": ratio(tp, tp + fn),
        "specificity_all_normal": ratio(tn, tn + fp),
        "precision_all_normal": ratio(tp, tp + fp),
        "f1_all_normal": ratio(2 * tp, 2 * tp + fp + fn),
        "note": (
            "joint_accuracy is NOT mean_per_aspect_accuracy. The rule is a "
            "conjunction, so the four error rates compound; the two are reported "
            "together precisely so the gap between them is visible."
        ),
    }


def write_morphology_plots(
    *,
    labels: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    results: Mapping[str, Mapping[str, float]],
    plot_dir: Path,
    history: Sequence[Mapping[str, Any]] = (),
    prefix: str = "",
) -> dict[str, str]:
    """Write the standard figure set. Returns ``{key: path}`` for what landed.

    Missing figures are simply absent from the return value: matplotlib is an
    extra, and a curve is undefined on a single-class split. Neither is worth
    failing an evaluation over, and both are visible in the returned dict.
    """
    from sperm_sorting.morphology.calibration import reliability_curve
    from sperm_sorting.morphology.metrics import confusion_matrix
    from training.common.plots import (
        plot_confusion_matrix,
        plot_pr_curve,
        plot_reliability_curve,
        plot_roc_curve,
        plot_training_curves,
    )

    written: dict[str, str] = {}
    tag = f"{prefix}_" if prefix else ""

    for aspect in MORPHOLOGY_ASPECTS:
        if aspect not in labels or aspect not in probabilities:
            continue
        y_true = np.asarray(labels[aspect]).ravel().astype(np.int64)
        y_prob = np.asarray(probabilities[aspect]).ravel().astype(np.float64)
        threshold = float(thresholds.get(aspect, 0.5))
        y_pred = (y_prob >= threshold).astype(np.int64)
        row = dict(results.get(aspect, {}))

        path = plot_confusion_matrix(
            confusion_matrix(y_true, y_pred),
            plot_dir / f"{tag}confusion_{aspect}.png",
            title=f"{aspect}: confusion @ P(abnormal) >= {threshold:.3f}",
        )
        if path:
            written[f"confusion_{aspect}"] = str(path)

        path = plot_reliability_curve(
            reliability_curve(y_prob, y_true, 15),
            plot_dir / f"{tag}reliability_{aspect}.png",
            title=f"{aspect}: reliability",
            ece=row.get("ece"),
        )
        if path:
            written[f"reliability_{aspect}"] = str(path)

        path = plot_pr_curve(
            y_true,
            y_prob,
            plot_dir / f"{tag}pr_{aspect}.png",
            title=f"{aspect}: precision-recall",
            ap=row.get("pr_auc"),
        )
        if path:
            written[f"pr_{aspect}"] = str(path)

        path = plot_roc_curve(
            y_true,
            y_prob,
            plot_dir / f"{tag}roc_{aspect}.png",
            title=f"{aspect}: ROC",
            auc=row.get("roc_auc"),
        )
        if path:
            written[f"roc_{aspect}"] = str(path)

    if history:
        path = plot_training_curves(
            list(history),
            plot_dir / f"{tag}training_curves.png",
            loss_keys=("train_loss", "val_loss"),
            metric_keys=(*tuple(f"val_{aspect}_macro_f1" for aspect in MORPHOLOGY_ASPECTS), "val_macro_f1"),
            title="morphology",
        )
        if path:
            written["training_curves"] = str(path)
    return written
