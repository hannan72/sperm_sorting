"""Per-aspect classification metrics. Pure numpy/sklearn -- no torch.

**The positive class is ABNORMAL (MHSMA label 1) in every metric in this
module.** That single sentence is repeated in every docstring below on purpose:
sign errors in a four-aspect, 0-is-normal, 1-is-abnormal problem do not announce
themselves, they just quietly report 95% accuracy while the product keeps every
sperm it was built to reject. Inputs here are ``P(abnormal)`` and MHSMA labels
verbatim; there is no polarity flip in this module (see :mod:`.polarity`).

Raw accuracy is deliberately **not** reported. MHSMA abnormal prevalence in the
train split is acrosome 30.1%, head 27.3%, vacuole 17.0% and tail **4.6%**, so a
model that predicts "normal" for every tail scores 95.4% accuracy and has zero
value. The headline numbers here are balanced accuracy, MCC and per-class
sensitivity/specificity, all of which collapse to chance level for that model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..constants import LABEL_ABNORMAL, LABEL_NORMAL, MORPHOLOGY_ASPECTS
from .calibration import expected_calibration_error

__all__ = [
    "METRIC_COLUMNS",
    "aspect_metrics",
    "balanced_accuracy",
    "confusion_counts",
    "confusion_matrix",
    "evaluate_aspects",
    "expected_calibration_error",
    "f1_score_abnormal",
    "f1_score_normal",
    "format_metrics_table",
    "macro_f1",
    "matthews_corrcoef",
    "metrics_to_json_dict",
    "negative_predictive_value",
    "pr_auc",
    "precision",
    "roc_auc",
    "sensitivity",
    "specificity",
]

#: Column order used by :func:`format_metrics_table`. Sensitivity first because
#: on this product a missed abnormality is the costly error: it lets an abnormal
#: sperm through the ``all_four_normal`` gate.
METRIC_COLUMNS: tuple[str, ...] = (
    "n",
    "prevalence",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1_abnormal",
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
    "ece",
)

_NAN = float("nan")


# ==========================================================================
# Confusion counts
# ==========================================================================


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    if t.shape != p.shape:
        raise ValueError(f"y_true shape {t.shape} != y_pred shape {p.shape}")
    for name, arr in (("y_true", t), ("y_pred", p)):
        if arr.size and not np.all(np.isin(arr, (LABEL_NORMAL, LABEL_ABNORMAL))):
            raise ValueError(
                f"{name} must contain only MHSMA labels {LABEL_NORMAL} (normal) and "
                f"{LABEL_ABNORMAL} (abnormal)"
            )
    return t, p


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """``(tn, fp, fn, tp)`` with **abnormal (label 1) as the positive class**.

    So ``tp`` counts aspects that really were abnormal and were called abnormal,
    and ``fn`` -- the dangerous cell for this product -- counts abnormal aspects
    that were called normal.
    """
    t, p = _check(y_true, y_pred)
    positive_true = t == LABEL_ABNORMAL
    positive_pred = p == LABEL_ABNORMAL
    tp = int(np.sum(positive_true & positive_pred))
    fp = int(np.sum(~positive_true & positive_pred))
    fn = int(np.sum(positive_true & ~positive_pred))
    tn = int(np.sum(~positive_true & ~positive_pred))
    return tn, fp, fn, tp


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """``2x2`` matrix indexed ``[true, pred]`` with rows/cols ordered
    ``(normal=0, abnormal=1)``, i.e. ``[[tn, fp], [fn, tp]]``.

    Positive class is **abnormal**.
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else _NAN


# ==========================================================================
# Scalar metrics
# ==========================================================================


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall of the **abnormal** class: ``tp / (tp + fn)``.

    "How many genuinely abnormal aspects did we catch?" This is the metric the
    product's safety story rests on, because a false negative here promotes an
    abnormal sperm into ``all_four_normal``. Undefined (NaN) when the split
    contains no abnormal examples.
    """
    _tn, _fp, fn, tp = confusion_counts(y_true, y_pred)
    return _ratio(tp, tp + fn)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall of the **normal** class: ``tn / (tn + fp)``.

    "How many genuinely normal aspects did we leave alone?" The positive class
    is still abnormal; specificity is simply the negative class' recall.
    Undefined (NaN) when the split contains no normal examples.
    """
    tn, fp, _fn, _tp = confusion_counts(y_true, y_pred)
    return _ratio(tn, tn + fp)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive predictive value for the **abnormal** class: ``tp / (tp + fp)``.

    "When we called an aspect abnormal, how often was it?" Undefined (NaN) when
    nothing was predicted abnormal -- which is exactly what a 0.5 threshold does
    on the 4.6%-prevalence tail aspect, and is why this metric must never be
    read on its own.
    """
    _tn, fp, _fn, tp = confusion_counts(y_true, y_pred)
    return _ratio(tp, tp + fp)


def negative_predictive_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``tn / (tn + fn)``: of the aspects we called **normal**, how many were.

    Positive class is still abnormal. This is the number that matters for the
    conjunctive ``all_four_normal`` rule, since that rule only ever acts on
    "normal" verdicts.
    """
    tn, _fp, fn, _tp = confusion_counts(y_true, y_pred)
    return _ratio(tn, tn + fn)


def f1_score_abnormal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 of the **abnormal** class (harmonic mean of precision and sensitivity)."""
    _tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    denominator = 2 * tp + fp + fn
    return _ratio(2 * tp, denominator)


def f1_score_normal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 of the **normal** class. Reported only as an input to :func:`macro_f1`."""
    tn, fp, fn, _tp = confusion_counts(y_true, y_pred)
    denominator = 2 * tn + fn + fp
    return _ratio(2 * tn, denominator)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the normal-class and abnormal-class F1 scores.

    Macro over the *two classes* of one aspect, not over the four aspects --
    see the ``"macro"`` row of :func:`evaluate_aspects` for the latter. Macro-F1
    is reported instead of plain F1 because it refuses to ignore the minority
    class: a model that never predicts abnormal gets F1_abnormal = 0 and a
    macro-F1 near 0.49 regardless of how good its accuracy looks.
    """
    scores = [f1_score_normal(y_true, y_pred), f1_score_abnormal(y_true, y_pred)]
    finite = [s for s in scores if not np.isnan(s)]
    return float(np.mean(finite)) if finite else _NAN


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``(sensitivity + specificity) / 2``. Positive class is **abnormal**.

    Prevalence-invariant, so 0.5 always means chance regardless of how skewed
    the aspect is. This is the accuracy-like number that may be quoted; raw
    accuracy is not computed anywhere in this module.
    """
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    if np.isnan(sens) or np.isnan(spec):
        return _NAN
    return float((sens + spec) / 2.0)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient. Positive class is **abnormal**.

    In ``[-1, 1]``; 0 is chance. MCC uses all four confusion cells, so unlike
    F1 it cannot be inflated by ignoring the minority class, and unlike
    balanced accuracy it degrades when precision collapses. It is the single
    most honest scalar for these aspects.

    Returns 0.0 when any confusion-matrix margin is zero -- the conventional
    definition when the coefficient is otherwise 0/0, and the honest reading:
    a constant predictor has no correlation with the labels.
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    numerator = float(tp) * tn - float(fp) * fn
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / np.sqrt(denominator))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC with ``y_score`` interpreted as **``P(abnormal)``**.

    Threshold-free, so it measures ranking quality independently of the
    threshold chosen in :mod:`.calibration`. Undefined (NaN) when the split has
    only one class -- which happens on real tail validation splits, so callers
    must handle NaN rather than assume a number.
    """
    return _sklearn_score("roc_auc_score", y_true, y_score)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision with ``y_score`` interpreted as **``P(abnormal)``**.

    Preferred over ROC-AUC on the low-prevalence aspects: ROC-AUC's specificity
    axis is dominated by the huge normal class and stays flattering, while the
    precision axis of PR-AUC does not. Its chance level is the prevalence, not
    0.5, so compare it against the reported ``prevalence`` column.
    """
    return _sklearn_score("average_precision_score", y_true, y_score)


def _sklearn_score(function_name: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    t = np.asarray(y_true).ravel().astype(np.int64)
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if t.shape != s.shape:
        raise ValueError(f"y_true shape {t.shape} != y_score shape {s.shape}")
    if t.size == 0 or np.unique(t).size < 2:
        return _NAN
    try:
        import sklearn.metrics as sk
    except ImportError:  # pragma: no cover - scikit-learn is an optional extra
        return _NAN
    return float(getattr(sk, function_name)(t, s))


# ==========================================================================
# Aggregation
# ==========================================================================


def aspect_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    *,
    n_bins: int = 15,
) -> dict[str, float]:
    """Full metric set for one aspect.

    Parameters
    ----------
    y_true
        MHSMA labels verbatim: 0 normal, 1 abnormal.
    y_prob
        **``P(abnormal)``**, calibrated if a :class:`~.calibration.CalibrationBundle`
        was applied.
    threshold
        Decision threshold in ``P(abnormal)`` space: predict abnormal when
        ``y_prob >= threshold``. This is the same space
        :func:`~.calibration.fit_thresholds` fits in, and the opposite space
        from :attr:`sperm_sorting.schemas.morphology.AspectResult.threshold`.
    """
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"y_true shape {t.shape} != y_prob shape {p.shape}")
    pred = (p >= float(threshold)).astype(np.int64)

    n = int(t.size)
    return {
        "n": float(n),
        "n_abnormal": float(np.sum(t == LABEL_ABNORMAL)),
        "prevalence": _ratio(float(np.sum(t == LABEL_ABNORMAL)), float(n)),
        "threshold": float(threshold),
        "sensitivity": sensitivity(t, pred),
        "specificity": specificity(t, pred),
        "precision": precision(t, pred),
        "npv": negative_predictive_value(t, pred),
        "f1_abnormal": f1_score_abnormal(t, pred),
        "f1_normal": f1_score_normal(t, pred),
        "macro_f1": macro_f1(t, pred),
        "balanced_accuracy": balanced_accuracy(t, pred),
        "mcc": matthews_corrcoef(t, pred),
        "roc_auc": roc_auc(t, p),
        "pr_auc": pr_auc(t, p),
        "ece": expected_calibration_error(p, t, n_bins),
    }


def evaluate_aspects(
    y_true: Mapping[str, np.ndarray],
    y_prob: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float] | float = 0.5,
    *,
    aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS,
    n_bins: int = 15,
) -> dict[str, dict[str, float]]:
    """Per-aspect metrics plus a ``"macro"`` aggregate row.

    The ``"macro"`` row is the unweighted mean over aspects of each metric,
    skipping NaNs. Unweighted on purpose: weighting by sample count would let
    the three easy aspects bury the tail, which is the one the model is worst
    at and the one a reviewer most needs to see.

    All probabilities are **``P(abnormal)``** and all thresholds are in
    ``P(abnormal)`` space.
    """
    results: dict[str, dict[str, float]] = {}
    for name in aspects:
        if name not in y_true or name not in y_prob:
            continue
        threshold = (
            float(thresholds)
            if isinstance(thresholds, (int, float))
            else float(thresholds.get(name, 0.5))
        )
        results[name] = aspect_metrics(
            y_true[name], y_prob[name], threshold, n_bins=n_bins
        )

    if results:
        keys = sorted({k for row in results.values() for k in row})
        macro: dict[str, float] = {}
        for key in keys:
            values = [row[key] for row in results.values() if not np.isnan(row.get(key, _NAN))]
            macro[key] = float(np.mean(values)) if values else _NAN
        # Counts are totals, not means; a "mean n" would be meaningless.
        macro["n"] = float(sum(row["n"] for row in results.values()))
        macro["n_abnormal"] = float(sum(row["n_abnormal"] for row in results.values()))
        results["macro"] = macro
    return results


# ==========================================================================
# Reporting
# ==========================================================================


def format_metrics_table(
    results: Mapping[str, Mapping[str, float]],
    *,
    columns: tuple[str, ...] = METRIC_COLUMNS,
    title: str = "morphology metrics (positive class = ABNORMAL, MHSMA label 1)",
) -> str:
    """Render :func:`evaluate_aspects` output as a fixed-width console table.

    Rows follow :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS` order with
    ``macro`` last, so two reports are always visually diffable. The title
    restates the positive class because a metrics table pasted into a ticket
    outlives the context that explained it.
    """
    aspect_order = [n for n in MORPHOLOGY_ASPECTS if n in results]
    extra = [n for n in results if n not in aspect_order and n != "macro"]
    rows = aspect_order + sorted(extra) + (["macro"] if "macro" in results else [])
    if not rows:
        return f"{title}\n(no results)"

    name_width = max(len("aspect"), *(len(r) for r in rows))
    widths = {c: max(len(c), 7) for c in columns}

    def cell(value: float | None, column: str) -> str:
        """Right-aligned cell; missing or undefined metrics render as ``n/a``."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a".rjust(widths[column])
        if column in ("n", "n_abnormal"):
            return f"{int(value):d}".rjust(widths[column])
        return f"{value:.4f}".rjust(widths[column])

    header = "aspect".ljust(name_width) + "  " + "  ".join(c.rjust(widths[c]) for c in columns)
    rule = "-" * len(header)
    lines = [title, rule, header, rule]
    for name in rows:
        row = results[name]
        cells = "  ".join(cell(row.get(c), c) for c in columns)
        lines.append(name.ljust(name_width) + "  " + cells)
        if name == rows[-2] and rows[-1] == "macro":
            lines.append(rule)
    lines.append(rule)
    lines.append(
        "sensitivity = recall of ABNORMAL; specificity = recall of NORMAL; "
        "raw accuracy is intentionally not reported"
    )
    return "\n".join(lines)


def metrics_to_json_dict(results: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """JSON-safe copy of a results dict (NaN becomes ``None``)."""
    return {
        aspect: {
            key: (None if isinstance(v, float) and np.isnan(v) else float(v))
            for key, v in row.items()
        }
        for aspect, row in results.items()
    }
