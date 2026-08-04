"""Diagnostic plots. Every function is a no-op when matplotlib is absent.

matplotlib is an extra, and a missing extra must never fail a training run that
has already done the expensive part. Every function here therefore returns
``Path | None`` -- the path it wrote, or ``None`` with the reason recorded in
:data:`LAST_SKIP_REASON` -- and the caller registers whatever came back.

The plot set is chosen for a severely imbalanced binary problem, which is what
each of the four morphology aspects is:

* A **confusion matrix** with counts *and* row-normalised rates. Counts alone
  hide that 7 of 240 validation tails are abnormal; rates alone hide that the
  denominator is 7.
* A **reliability curve** with the bin populations drawn underneath. An ECE of
  0.02 means nothing if thirteen of fifteen bins are empty, and the histogram
  is what makes that visible.
* A **precision-recall curve** with the prevalence drawn as the chance line.
  PR-AUC's baseline is the prevalence, not 0.5, so a PR curve without that line
  is routinely misread as good.
* A **ROC curve**, which is prevalence-invariant and therefore the one to
  compare across aspects -- with the caveat, printed on the axis label, that it
  flatters low-prevalence aspects.
* **Training curves** for loss and the selection metric, per aspect, because a
  falling total loss on a four-head model says nothing about whether the
  4.6%-prevalence tail head is learning.

Every figure is closed explicitly. A training loop that plots per epoch and
does not close leaks figures until matplotlib warns and then until the process
dies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "LAST_SKIP_REASON",
    "matplotlib_available",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "plot_reliability_curve",
    "plot_roc_curve",
    "plot_training_curves",
]

#: Why the most recent plot call did nothing. Empty when it succeeded.
LAST_SKIP_REASON: str = ""

_DPI = 130


def _pyplot() -> Any:
    """Import pyplot with a non-interactive backend, or return ``None``.

    ``Agg`` is forced before the first pyplot import: these scripts run over
    SSH and in CI where no display exists, and the default backend would either
    fail or block waiting for a window.
    """
    global LAST_SKIP_REASON
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:
        LAST_SKIP_REASON = f"matplotlib unavailable: {type(exc).__name__}: {exc}"
        return None
    LAST_SKIP_REASON = ""
    return plt


def matplotlib_available() -> bool:
    """Whether plots can be produced at all, for the experiment record."""
    return _pyplot() is not None


def _prepare(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _finish(plt: Any, fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


# ==========================================================================
# Classification diagnostics
# ==========================================================================


def plot_confusion_matrix(
    matrix: np.ndarray,
    path: str | Path,
    *,
    title: str = "confusion matrix",
    class_names: Sequence[str] = ("normal (0)", "abnormal (1)"),
) -> Path | None:
    """Plot a ``[true, pred]`` confusion matrix with counts and row rates.

    ``matrix`` follows
    :func:`sperm_sorting.morphology.metrics.confusion_matrix`: rows are the
    truth, columns the prediction, both ordered ``(normal, abnormal)``. Cells
    are annotated ``count`` over ``row-rate``, because on this problem the two
    tell opposite stories and a reader needs both.
    """
    plt = _pyplot()
    if plt is None:
        return None

    counts = np.asarray(matrix, dtype=np.float64)
    row_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    image = ax.imshow(rates, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)), labels=list(class_names))
    ax.set_yticks(range(len(class_names)), labels=list(class_names))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            # White text on dark cells, black on light: at rate 1.0 the cell is
            # near-black and black annotation is unreadable.
            colour = "white" if rates[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{int(counts[i, j])}\n{rates[i, j]:.1%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, label="row-normalised rate")
    return _finish(plt, fig, _prepare(path))


def plot_reliability_curve(
    curve: Any,
    path: str | Path,
    *,
    title: str = "reliability",
    ece: float | None = None,
) -> Path | None:
    """Plot a :class:`~sperm_sorting.morphology.calibration.ReliabilityCurve`.

    Two stacked axes: the calibration diagram on top, the per-bin sample count
    underneath. The count panel is the important half -- a curve that hugs the
    diagonal across three populated bins and wanders across twelve empty ones
    looks identical to a well-calibrated model unless the populations are drawn.
    """
    plt = _pyplot()
    if plt is None:
        return None

    centers = np.asarray(curve.bin_centers, dtype=np.float64)
    accuracies = np.asarray(curve.accuracies, dtype=np.float64)
    confidences = np.asarray(curve.confidences, dtype=np.float64)
    counts = np.asarray(curve.counts, dtype=np.int64)
    populated = counts > 0

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(4.6, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1, label="perfect calibration")
    if populated.any():
        ax.plot(
            confidences[populated],
            accuracies[populated],
            "o-",
            color="#1f77b4",
            markersize=4,
            label="observed",
        )
    label = title if ece is None else f"{title}  (ECE {ece:.4f})"
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("observed fraction abnormal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    width = float(centers[1] - centers[0]) if centers.size > 1 else 0.05
    ax_hist.bar(centers, counts, width=width * 0.9, color="#7f7f7f")
    ax_hist.set_xlabel("predicted P(abnormal)")
    ax_hist.set_ylabel("count")
    ax_hist.grid(alpha=0.25)
    return _finish(plt, fig, _prepare(path))


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: str | Path,
    *,
    title: str = "precision-recall",
    ap: float | None = None,
) -> Path | None:
    """Precision-recall curve for ``P(abnormal)`` against MHSMA labels.

    The prevalence is drawn as a horizontal chance line, because PR-AUC's
    baseline *is* the prevalence: an AP of 0.30 is excellent on the 4.6%
    abnormal-tail aspect and poor on the 30.1% acrosome aspect, and the number
    alone does not say which.
    """
    plt = _pyplot()
    if plt is None:
        return None
    precision, recall, prevalence = _pr_points(y_true, y_score)
    if precision is None:
        return None

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.step(recall, precision, where="post", color="#d62728")
    ax.axhline(
        prevalence,
        linestyle="--",
        color="0.6",
        linewidth=1,
        label=f"chance = prevalence ({prevalence:.3f})",
    )
    ax.set_xlabel("recall (sensitivity, abnormal)")
    ax.set_ylabel("precision (abnormal)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title if ap is None else f"{title}  (AP {ap:.4f})", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    return _finish(plt, fig, _prepare(path))


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: str | Path,
    *,
    title: str = "ROC",
    auc: float | None = None,
) -> Path | None:
    """ROC curve for ``P(abnormal)``.

    Labelled with the caveat that ROC is dominated by the majority class on the
    low-prevalence aspects, so that a 0.9 AUC on the tail aspect is not read as
    a usable operating point.
    """
    plt = _pyplot()
    if plt is None:
        return None
    fpr, tpr = _roc_points(y_true, y_score)
    if fpr is None:
        return None

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot(fpr, tpr, color="#2ca02c")
    ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity, abnormal)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title if auc is None else f"{title}  (AUC {auc:.4f})", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    return _finish(plt, fig, _prepare(path))


def plot_training_curves(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    loss_keys: Sequence[str] = ("train_loss", "val_loss"),
    metric_keys: Sequence[str] = (),
    title: str = "training",
) -> Path | None:
    """Loss and selection-metric curves over epochs.

    Missing keys are skipped rather than raising: a run that had no validation
    split still has a train-loss curve worth plotting, and a caller should not
    have to pre-filter the key list to get it.
    """
    plt = _pyplot()
    if plt is None:
        return None
    if not history:
        globals()["LAST_SKIP_REASON"] = "empty history"
        return None

    epochs = [int(row.get("epoch", i)) for i, row in enumerate(history)]

    def series(key: str) -> list[float] | None:
        values = [row.get(key) for row in history]
        if all(v is None for v in values):
            return None
        return [float("nan") if v is None else float(v) for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    plotted_loss = False
    for key in loss_keys:
        values = series(key)
        if values is None:
            continue
        axes[0].plot(epochs, values, marker="o", markersize=3, label=key)
        plotted_loss = True
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title}: loss", fontsize=10)
    axes[0].grid(alpha=0.25)
    if plotted_loss:
        axes[0].legend(fontsize=8)

    plotted_metric = False
    for key in metric_keys:
        values = series(key)
        if values is None:
            continue
        axes[1].plot(epochs, values, marker="o", markersize=3, label=key)
        plotted_metric = True
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].set_title(f"{title}: validation metrics", fontsize=10)
    axes[1].grid(alpha=0.25)
    if plotted_metric:
        axes[1].legend(fontsize=8)

    return _finish(plt, fig, _prepare(path))


# ==========================================================================
# Curve computation (no sklearn dependency)
# ==========================================================================


def _pr_points(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Precision/recall at every distinct score, computed directly.

    Implemented here rather than via sklearn so the plots work in an
    environment that has matplotlib but not scikit-learn -- and so the curve
    drawn is provably the same quantity the metrics module reports, in
    abnormal-positive space.
    """
    global LAST_SKIP_REASON
    labels = np.asarray(y_true).ravel().astype(np.int64)
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if labels.size == 0 or np.unique(labels).size < 2:
        LAST_SKIP_REASON = "split contains a single class; a PR curve is undefined"
        return None, None, float("nan")

    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    n_positive = int(labels.sum())
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_positive, 1)
    # Start at recall 0 with the precision of the top-ranked sample, so the
    # curve does not begin in mid-air.
    precision = np.concatenate([[precision[0]], precision])
    recall = np.concatenate([[0.0], recall])
    prevalence = n_positive / labels.size
    return precision, recall, float(prevalence)


def _roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """FPR/TPR at every distinct score."""
    global LAST_SKIP_REASON
    labels = np.asarray(y_true).ravel().astype(np.int64)
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if labels.size == 0 or np.unique(labels).size < 2:
        LAST_SKIP_REASON = "split contains a single class; a ROC curve is undefined"
        return None, None

    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    tpr = np.concatenate([[0.0], np.cumsum(labels == 1) / max(n_positive, 1)])
    fpr = np.concatenate([[0.0], np.cumsum(labels == 0) / max(n_negative, 1)])
    return fpr, tpr
