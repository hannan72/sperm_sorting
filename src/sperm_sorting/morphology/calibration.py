"""Post-hoc calibration of the morphology heads. Pure numpy -- no torch.

Everything in this module lives in **abnormal-positive space**: inputs are
logits or probabilities for ``P(abnormal)`` and labels are the MHSMA integers
(``0 = normal``, ``1 = abnormal``) verbatim. There is no polarity flip here;
see :mod:`sperm_sorting.morphology.polarity` for the one place it happens.

Two separate problems are solved here and it is worth keeping them apart:

**Calibration** (:class:`TemperatureScaler`) fixes the *probabilities*. A
network trained with a large ``pos_weight`` on a 4.6%-prevalence aspect emits
wildly over-confident scores; temperature scaling divides the logits by a single
learned constant per aspect, which cannot change the ranking and therefore
cannot change AUC, but does make the numbers mean what they say. That matters
because the audit log records ``p_normal`` and a human reads it.

**Thresholding** (:func:`fit_thresholds`) fixes the *decision*. With 4.6% of
tails abnormal, the 0.5 default classifies everything as normal and scores 95%
accuracy while catching none of the abnormal tails. The threshold must be
chosen per aspect, on a validation split, against a criterion that is not
fooled by prevalence -- Youden's J by default.

Both artefacts are deliberately kept **out of the weights**. A
:class:`CalibrationBundle` is a small JSON file that travels next to the
checkpoint and can be re-fitted on new device data in seconds without touching
the network. It carries the polarity string for the same reason the checkpoint
does: a bundle fitted under the other convention would invert every decision.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..constants import MORPHOLOGY_ASPECTS
from ..errors import ConfigurationError
from .polarity import POLARITY_CONVENTION

__all__ = [
    "CRITERIA",
    "CalibrationBundle",
    "ReliabilityCurve",
    "TemperatureScaler",
    "ThresholdFit",
    "expected_calibration_error",
    "fit_calibration_bundle",
    "fit_threshold",
    "fit_thresholds",
    "logit",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "reliability_curve",
    "sigmoid",
]

Criterion = Literal["youden", "f1", "balanced_accuracy", "mcc"]

#: Selection criteria supported by :func:`fit_thresholds`. All are computed
#: with **abnormal as the positive class**.
CRITERIA: tuple[str, ...] = ("youden", "f1", "balanced_accuracy", "mcc")

#: Probabilities are clipped this far from 0 and 1 before any log is taken.
_EPS = 1e-12

#: Thresholds are clipped this far from 0 and 1 so that both the value and its
#: polarity-flipped counterpart satisfy the strict ``0 < t < 1`` required by
#: :class:`sperm_sorting.config.MorphologyConfig`.
_THRESHOLD_EPS = 1e-6


# ==========================================================================
# Numeric primitives
# ==========================================================================


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function.

    Branch-free: ``exp(-|x|)`` never overflows, so this is safe for the very
    large logits a ``pos_weight``-trained head produces on easy negatives.
    """
    arr = np.asarray(x, dtype=np.float64)
    z = np.exp(-np.abs(arr))
    return np.where(arr >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


def logit(p: np.ndarray | float, eps: float = 1e-7) -> np.ndarray:
    """Inverse of :func:`sigmoid`, with probabilities clipped away from 0/1."""
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def negative_log_likelihood(
    probs: np.ndarray, labels: np.ndarray, eps: float = _EPS
) -> float:
    """Mean binary NLL (log loss) of ``P(abnormal)`` against MHSMA labels.

    Lower is better. This is the objective :class:`TemperatureScaler` minimises
    and it is a *proper* scoring rule, which is why it is used instead of
    accuracy: it is minimised only by the true probabilities, so it cannot be
    gamed by a model that ranks well but is over-confident.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"probs shape {p.shape} != labels shape {y.shape}")
    if p.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _as_binary_labels(labels: np.ndarray | Sequence[int], name: str) -> np.ndarray:
    y = np.asarray(labels).ravel()
    if y.size and not np.all(np.isin(y, (0, 1))):
        raise ValueError(
            f"{name} must contain only MHSMA labels 0 (normal) and 1 (abnormal); "
            f"found {sorted(set(np.unique(y).tolist()))[:5]}"
        )
    return y.astype(np.int64)


def _golden_section_min(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Minimise a 1-D unimodal function on ``[lo, hi]`` without scipy.

    Golden-section search: derivative-free, no dependency, and deterministic,
    which matters because the fitted temperature ends up in a config file that
    must be reproducible. Applied to the *inverse* temperature (see
    :meth:`TemperatureScaler.fit`), where the NLL is convex and therefore
    unimodal, so convergence to the global minimum is guaranteed.
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = func(c), func(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = func(d)
    return (a + b) / 2.0


# ==========================================================================
# Temperature scaling
# ==========================================================================


class TemperatureScaler:
    """One temperature per aspect, fitted by minimising validation NLL.

    ``p = sigmoid(z / T)``. ``T > 1`` softens an over-confident head, ``T < 1``
    sharpens an under-confident one, and ``T == 1`` is a no-op. Because the map
    is strictly monotone in ``z``, temperature scaling cannot change the ranking
    of any two samples: ROC-AUC, PR-AUC and the *optimal* threshold's operating
    point are all invariant. It fixes only the numbers, which is exactly the
    scope it should have.

    Fitted per aspect, never globally, because the four heads see very different
    prevalences and are trained with very different ``pos_weight``s, so their
    over-confidence differs by more than a factor of two in practice.
    """

    def __init__(
        self,
        aspects: Sequence[str] = MORPHOLOGY_ASPECTS,
        *,
        min_temperature: float = 0.05,
        max_temperature: float = 20.0,
        min_per_class: int = 2,
    ) -> None:
        self.aspects = tuple(aspects)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.min_per_class = int(min_per_class)
        self.temperatures: dict[str, float] = dict.fromkeys(self.aspects, 1.0)
        #: Per-aspect human-readable note; populated when an aspect was skipped.
        self.notes: dict[str, str] = {}
        self.fitted: bool = False

    # ---------------------------------------------------------------- fitting

    def fit(
        self,
        logits: Mapping[str, np.ndarray],
        labels: Mapping[str, np.ndarray],
    ) -> TemperatureScaler:
        """Fit one temperature per aspect on a held-out split.

        The search is done over the *inverse* temperature ``w = 1 / T`` because
        ``NLL(w)`` is the log loss of a one-parameter logistic model in ``w`` and
        is therefore convex, making a golden-section search provably correct.
        Searching ``T`` directly is not convex and can stall on a plateau for
        near-perfectly-separated aspects.

        An aspect with fewer than ``min_per_class`` examples of either class is
        **skipped** with ``T = 1`` and a recorded note, rather than fitted. The
        MHSMA validation split contains only 7 abnormal tails out of 240; a
        temperature fitted on a handful of positives is noise dressed as
        calibration, and silently shipping it would be worse than shipping the
        uncalibrated head.
        """
        for name in self.aspects:
            if name not in logits or name not in labels:
                raise ConfigurationError(
                    f"TemperatureScaler.fit: aspect '{name}' missing from "
                    f"{'logits' if name not in logits else 'labels'}"
                )
            z = np.asarray(logits[name], dtype=np.float64).ravel()
            y = _as_binary_labels(labels[name], f"labels['{name}']")
            if z.shape != y.shape:
                raise ValueError(
                    f"aspect '{name}': {z.shape} logits vs {y.shape} labels"
                )

            n_pos = int(y.sum())
            n_neg = int(y.size - n_pos)
            if min(n_pos, n_neg) < self.min_per_class:
                self.temperatures[name] = 1.0
                self.notes[name] = (
                    f"not fitted: only {n_pos} abnormal / {n_neg} normal examples "
                    f"(need >= {self.min_per_class} of each); temperature left at 1.0"
                )
                continue

            def objective(inv_t: float, _z: np.ndarray = z, _y: np.ndarray = y) -> float:
                """Validation NLL at inverse temperature ``inv_t``; convex in it."""
                return negative_log_likelihood(sigmoid(_z * inv_t), _y)

            best_inv = _golden_section_min(
                objective,
                1.0 / self.max_temperature,
                1.0 / self.min_temperature,
            )
            # Never make things worse: T = 1 is always an admissible answer.
            if objective(best_inv) > objective(1.0):
                best_inv = 1.0
                self.notes[name] = "search found no improvement over T = 1.0"
            self.temperatures[name] = float(1.0 / best_inv)

        self.fitted = True
        return self

    # -------------------------------------------------------------- application

    def transform_logits(self, logits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Temperature-scaled logits, still in ``P(abnormal)`` space."""
        return {
            name: np.asarray(logits[name], dtype=np.float64) / self.temperatures[name]
            for name in self.aspects
            if name in logits
        }

    def transform(self, logits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Calibrated ``P(abnormal)`` probabilities per aspect."""
        return {
            name: sigmoid(z) for name, z in self.transform_logits(logits).items()
        }

    def report(
        self,
        logits: Mapping[str, np.ndarray],
        labels: Mapping[str, np.ndarray],
        n_bins: int = 15,
    ) -> dict[str, dict[str, float]]:
        """Before/after NLL and ECE per aspect, for the calibration report."""
        out: dict[str, dict[str, float]] = {}
        after = self.transform(logits)
        for name in self.aspects:
            if name not in logits:
                continue
            y = _as_binary_labels(labels[name], f"labels['{name}']")
            raw = sigmoid(np.asarray(logits[name], dtype=np.float64))
            out[name] = {
                "temperature": self.temperatures[name],
                "nll_before": negative_log_likelihood(raw, y),
                "nll_after": negative_log_likelihood(after[name], y),
                "ece_before": expected_calibration_error(raw, y, n_bins),
                "ece_after": expected_calibration_error(after[name], y, n_bins),
            }
        return out


# ==========================================================================
# Threshold selection
# ==========================================================================


@dataclass(frozen=True)
class ThresholdFit:
    """Outcome of a threshold search for one aspect, in ``P(abnormal)`` space."""

    threshold: float
    criterion: str
    score: float
    sensitivity: float
    specificity: float
    n_positive: int
    n_negative: int
    note: str = ""


def _sweep(probs: np.ndarray, labels: np.ndarray, cuts: np.ndarray) -> dict[str, np.ndarray]:
    """Confusion counts for every candidate threshold at once.

    Vectorised over candidates rather than looped, because a full sweep over
    every distinct validation score is O(n_cuts * n) and n_cuts == n; at a few
    thousand validation crops the loop version is noticeably slower and no
    clearer. Prediction rule: **abnormal when ``p >= cut``**.
    """
    predicted = probs[None, :] >= cuts[:, None]
    positive = labels[None, :] == 1
    tp = np.sum(predicted & positive, axis=1).astype(np.float64)
    fp = np.sum(predicted & ~positive, axis=1).astype(np.float64)
    fn = np.sum(~predicted & positive, axis=1).astype(np.float64)
    tn = np.sum(~predicted & ~positive, axis=1).astype(np.float64)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _criterion_scores(counts: Mapping[str, np.ndarray], criterion: str) -> np.ndarray:
    """Vectorised criterion value per candidate threshold.

    Positive class is **abnormal** throughout: ``sensitivity`` is the fraction
    of abnormal aspects caught, ``specificity`` the fraction of normal aspects
    correctly left alone.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)

        if criterion == "youden":
            return sensitivity + specificity - 1.0
        if criterion == "balanced_accuracy":
            return 0.5 * (sensitivity + specificity)
        if criterion == "f1":
            denom = precision + sensitivity
            return np.divide(
                2.0 * precision * sensitivity,
                denom,
                out=np.zeros_like(tp),
                where=denom > 0,
            )
        if criterion == "mcc":
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return np.divide(
                numerator,
                denominator,
                out=np.zeros_like(tp),
                where=denominator > 0,
            )
    raise ConfigurationError(
        f"unknown threshold criterion '{criterion}'; available: {', '.join(CRITERIA)}"
    )


def fit_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    criterion: Criterion = "youden",
) -> ThresholdFit:
    """Choose the decision threshold on ``P(abnormal)`` for one aspect.

    Candidates are the midpoints between consecutive distinct observed scores
    (plus the two extremes), which is the smallest set that contains an optimum
    of any threshold-monotone criterion. Ties are broken by taking the
    **median** of the tied candidates rather than the first: on a small
    validation split whole ranges of thresholds score identically, and the edge
    of such a range sits right next to a training sample, which is the least
    robust place to put a decision boundary.

    The default criterion is Youden's J (``sensitivity + specificity - 1``)
    because it is invariant to prevalence. With 4.6% abnormal tails, accuracy
    and even F1 reward a threshold that simply never predicts abnormal; J does
    not.

    The result is clipped into ``(0, 1)`` so that it -- and its polarity-flipped
    counterpart used by :class:`~sperm_sorting.config.MorphologyConfig` -- are
    both strictly inside the open interval that config validation requires.
    """
    if criterion not in CRITERIA:
        raise ConfigurationError(
            f"unknown threshold criterion '{criterion}'; available: {', '.join(CRITERIA)}"
        )
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = _as_binary_labels(labels, "labels")
    if p.shape != y.shape:
        raise ValueError(f"probs shape {p.shape} != labels shape {y.shape}")

    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        # No information to fit on. Falling back to 0.5 is the honest answer;
        # silently returning an extreme threshold would look like a decision.
        return ThresholdFit(
            threshold=0.5,
            criterion=criterion,
            score=float("nan"),
            sensitivity=float("nan"),
            specificity=float("nan"),
            n_positive=n_pos,
            n_negative=n_neg,
            note=(
                f"single-class split ({n_pos} abnormal / {n_neg} normal); threshold "
                "left at 0.5 -- it is not calibrated and must not be reported as such"
            ),
        )

    distinct = np.unique(p)
    midpoints = (distinct[:-1] + distinct[1:]) / 2.0 if distinct.size > 1 else distinct
    cuts = np.unique(
        np.concatenate([[0.0], midpoints, distinct, [1.0]])
    )

    counts = _sweep(p, y, cuts)
    scores = _criterion_scores(counts, criterion)
    best = float(np.nanmax(scores))
    tied = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
    chosen_index = int(tied[len(tied) // 2])
    threshold = float(
        np.clip(cuts[chosen_index], _THRESHOLD_EPS, 1.0 - _THRESHOLD_EPS)
    )

    tp = counts["tp"][chosen_index]
    fn = counts["fn"][chosen_index]
    tn = counts["tn"][chosen_index]
    fp = counts["fp"][chosen_index]
    return ThresholdFit(
        threshold=threshold,
        criterion=criterion,
        score=best,
        sensitivity=float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        specificity=float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        n_positive=n_pos,
        n_negative=n_neg,
        note=f"{len(tied)} candidate threshold(s) tied at the optimum",
    )


def fit_thresholds(
    probs: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    criterion: Criterion = "youden",
) -> dict[str, float]:
    """Per-aspect thresholds on ``P(abnormal)``. See :func:`fit_threshold`.

    Each aspect is fitted **independently**. Sharing one threshold across the
    four would be dominated by whichever aspect has the most positives, and with
    prevalences from 30.1% (acrosome) down to 4.6% (tail) that is not a rounding
    error, it is the difference between catching abnormal tails and never
    predicting one.

    Use :func:`fit_calibration_bundle` when you also want the temperatures and a
    saveable artefact.
    """
    return {
        name: fit_threshold(probs[name], labels[name], criterion).threshold
        for name in probs
    }


# ==========================================================================
# Calibration error
# ==========================================================================


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned reliability data, ready to plot as a calibration diagram."""

    bin_lower: np.ndarray
    bin_upper: np.ndarray
    bin_centers: np.ndarray
    #: Empirical fraction of abnormal labels in each bin ("accuracy" axis).
    accuracies: np.ndarray
    #: Mean predicted ``P(abnormal)`` in each bin ("confidence" axis).
    confidences: np.ndarray
    counts: np.ndarray

    @property
    def nonempty(self) -> np.ndarray:
        """Mask of bins that hold samples; the rest carry no calibration signal."""
        return self.counts > 0

    def to_json_dict(self) -> dict[str, list[float]]:
        """Plain-list form, ready to serialise into a calibration report."""
        return {
            "bin_lower": self.bin_lower.tolist(),
            "bin_upper": self.bin_upper.tolist(),
            "bin_centers": self.bin_centers.tolist(),
            "accuracies": self.accuracies.tolist(),
            "confidences": self.confidences.tolist(),
            "counts": self.counts.tolist(),
        }


def _bin_edges(
    probs: np.ndarray, n_bins: int, strategy: Literal["uniform", "quantile"]
) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ConfigurationError(
        f"unknown binning strategy '{strategy}'; use 'uniform' or 'quantile'"
    )


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    *,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> ReliabilityCurve:
    """Bin ``P(abnormal)`` and compare predicted against observed frequency.

    This is the *binary* reliability diagram: the vertical axis is the observed
    fraction of **abnormal** labels in the bin, not top-label accuracy. For a
    two-class problem with one reported probability the two definitions differ,
    and the binary one is what a reviewer of a 4.6%-prevalence aspect needs to
    see -- top-label accuracy would show a flat, flattering line dominated by
    confident negatives.

    ``strategy='quantile'`` is worth using on the imbalanced aspects: with
    uniform bins almost every prediction lands in the first bin and thirteen of
    the fifteen bins are empty.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0.0, 1.0)
    y = _as_binary_labels(labels, "labels").astype(np.float64)
    if p.shape != y.shape:
        raise ValueError(f"probs shape {p.shape} != labels shape {y.shape}")
    if n_bins < 1:
        raise ConfigurationError(f"n_bins must be >= 1, got {n_bins}")

    edges = _bin_edges(p, n_bins, strategy)
    lower, upper = edges[:-1], edges[1:]
    n = lower.size

    accuracies = np.zeros(n, dtype=np.float64)
    confidences = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # Right-closed bins, with the first bin also closed on the left so that
        # a prediction of exactly 0.0 is counted somewhere.
        in_bin = (p > lower[i]) & (p <= upper[i]) if i > 0 else (p >= lower[i]) & (p <= upper[i])
        counts[i] = int(np.sum(in_bin))
        if counts[i]:
            accuracies[i] = float(np.mean(y[in_bin]))
            confidences[i] = float(np.mean(p[in_bin]))

    return ReliabilityCurve(
        bin_lower=lower,
        bin_upper=upper,
        bin_centers=(lower + upper) / 2.0,
        accuracies=accuracies,
        confidences=confidences,
        counts=counts,
    )


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    *,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """Sample-weighted mean ``|observed - predicted|`` over the bins.

    ECE is a summary, not a verdict: it can be small simply because almost all
    predictions sit in one well-populated bin. Read it next to
    :func:`maximum_calibration_error` and the reliability curve itself.
    """
    curve = reliability_curve(probs, labels, n_bins, strategy=strategy)
    total = int(curve.counts.sum())
    if total == 0:
        return float("nan")
    gaps = np.abs(curve.accuracies - curve.confidences)
    return float(np.sum(curve.counts * gaps) / total)


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    *,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_count: int = 1,
) -> float:
    """Worst per-bin calibration gap.

    Bins with fewer than ``min_count`` samples are excluded, because a single
    sample in a bin produces an observed frequency of exactly 0 or 1 and a gap
    of up to 1.0 that says nothing about the model.
    """
    curve = reliability_curve(probs, labels, n_bins, strategy=strategy)
    usable = curve.counts >= max(1, min_count)
    if not np.any(usable):
        return float("nan")
    return float(np.max(np.abs(curve.accuracies[usable] - curve.confidences[usable])))


# ==========================================================================
# Bundle
# ==========================================================================


@dataclass
class CalibrationBundle:
    """Per-aspect temperature and threshold, serialisable to JSON.

    Kept separate from the weights on purpose. Re-fitting calibration on a new
    batch of device data is a two-second job on a CPU and does not change a
    single network parameter; forcing it to be a new checkpoint would make
    re-calibration look like re-training in the audit log, which it is not.

    ``thresholds`` are in **``P(abnormal)`` space** -- the same space
    :func:`fit_thresholds` works in -- and :attr:`threshold_space` records that
    explicitly so the value can never be read as a ``P(normal)`` threshold. The
    conversion to the schema's ``P(normal)`` space happens once, in
    :class:`~.inference.MorphologyEngine`, via
    :func:`~.polarity.flip_polarity`.
    """

    temperatures: dict[str, float]
    thresholds: dict[str, float]
    aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS
    criterion: str = "youden"
    threshold_space: str = "p_abnormal"
    label_polarity: str = POLARITY_CONVENTION
    #: Free-form identifier of the split the calibration was fitted on.
    fitted_on: str = ""
    #: Identifier of the checkpoint these numbers belong to. A bundle fitted
    #: against different weights is meaningless; recording it lets a loader warn.
    model_id: str = ""
    n_samples: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    created_utc: str = ""
    schema_version: str = "1"

    def __post_init__(self) -> None:
        self.aspects = tuple(self.aspects)
        for name in self.aspects:
            if name not in self.temperatures:
                raise ConfigurationError(f"calibration bundle missing temperature for '{name}'")
            if name not in self.thresholds:
                raise ConfigurationError(f"calibration bundle missing threshold for '{name}'")
            if self.temperatures[name] <= 0.0:
                raise ConfigurationError(
                    f"temperature for '{name}' must be positive, got {self.temperatures[name]}"
                )
            if not 0.0 < self.thresholds[name] < 1.0:
                raise ConfigurationError(
                    f"threshold for '{name}' must lie in (0, 1), got {self.thresholds[name]}"
                )
        if not self.created_utc:
            self.created_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    # --------------------------------------------------------------- applying

    def apply(self, logits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Temperature-scaled ``P(abnormal)`` per aspect. Still not ``P(normal)``."""
        return {
            name: sigmoid(np.asarray(logits[name], dtype=np.float64) / self.temperatures[name])
            for name in self.aspects
            if name in logits
        }

    def predict_abnormal(self, logits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Boolean abnormality decisions, ``P(abnormal) >= threshold``."""
        probs = self.apply(logits)
        return {name: probs[name] >= self.thresholds[name] for name in probs}

    # -------------------------------------------------------------- persistence

    def to_json_dict(self) -> dict[str, Any]:
        """Serialisable form, including the polarity string that guards loading."""
        return {
            "schema_version": self.schema_version,
            "label_polarity": self.label_polarity,
            "threshold_space": self.threshold_space,
            "criterion": self.criterion,
            "aspects": list(self.aspects),
            "temperatures": {k: float(v) for k, v in self.temperatures.items()},
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "fitted_on": self.fitted_on,
            "model_id": self.model_id,
            "n_samples": dict(self.n_samples),
            "metrics": {k: dict(v) for k, v in self.metrics.items()},
            "notes": dict(self.notes),
            "created_utc": self.created_utc,
        }

    def save_json(self, path: str | Path) -> Path:
        """Write the bundle as pretty JSON (atomically)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    @classmethod
    def load_json(
        cls, path: str | Path, *, expected_polarity: str = POLARITY_CONVENTION
    ) -> CalibrationBundle:
        """Read a bundle, refusing one written under a different polarity.

        Raises
        ------
        ConfigurationError
            If the file is unreadable, or its ``label_polarity`` /
            ``threshold_space`` disagree with this build. Both mismatches invert
            or mis-scale every decision, and neither produces any other symptom.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"calibration bundle not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"could not read calibration bundle {path}: {exc}") from exc

        recorded = data.get("label_polarity")
        if recorded != expected_polarity:
            raise ConfigurationError(
                "calibration bundle label-polarity mismatch -- refusing to load.\n"
                f"  bundle:   {path}\n"
                f"  recorded: {recorded!r}\n"
                f"  expected: {expected_polarity!r}\n"
                "Applying it would invert every morphology decision. Re-fit the "
                "calibration under the current convention."
            )
        space = data.get("threshold_space", "p_abnormal")
        if space != "p_abnormal":
            raise ConfigurationError(
                f"calibration bundle {path} records threshold_space={space!r}; this "
                "build only understands 'p_abnormal' thresholds"
            )

        return cls(
            temperatures={k: float(v) for k, v in data["temperatures"].items()},
            thresholds={k: float(v) for k, v in data["thresholds"].items()},
            aspects=tuple(data.get("aspects") or MORPHOLOGY_ASPECTS),
            criterion=str(data.get("criterion", "youden")),
            threshold_space=space,
            label_polarity=str(recorded),
            fitted_on=str(data.get("fitted_on", "")),
            model_id=str(data.get("model_id", "")),
            n_samples={k: int(v) for k, v in (data.get("n_samples") or {}).items()},
            metrics={
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in (data.get("metrics") or {}).items()
            },
            notes={k: str(v) for k, v in (data.get("notes") or {}).items()},
            created_utc=str(data.get("created_utc", "")),
            schema_version=str(data.get("schema_version", "1")),
        )


def fit_calibration_bundle(
    logits: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    aspects: Sequence[str] = MORPHOLOGY_ASPECTS,
    criterion: Criterion = "youden",
    fitted_on: str = "",
    model_id: str = "",
    n_bins: int = 15,
) -> CalibrationBundle:
    """Fit temperatures then thresholds, in that order, and package the result.

    Order matters: the threshold is chosen on the *calibrated* probabilities, so
    that the number stored in the bundle is the number the engine will actually
    compare against at run time. Fitting the threshold on raw probabilities and
    then scaling them afterwards would move the operating point.
    """
    aspects = tuple(aspects)
    scaler = TemperatureScaler(aspects).fit(logits, labels)
    calibrated = scaler.transform(logits)

    thresholds: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    notes = dict(scaler.notes)
    n_samples: dict[str, int] = {}

    for name in aspects:
        y = _as_binary_labels(labels[name], f"labels['{name}']")
        fit = fit_threshold(calibrated[name], y, criterion)
        thresholds[name] = fit.threshold
        n_samples[name] = int(y.size)
        metrics[name] = {
            "temperature": scaler.temperatures[name],
            "criterion_score": fit.score,
            "sensitivity": fit.sensitivity,
            "specificity": fit.specificity,
            "n_abnormal": float(fit.n_positive),
            "n_normal": float(fit.n_negative),
            "nll": negative_log_likelihood(calibrated[name], y),
            "ece": expected_calibration_error(calibrated[name], y, n_bins),
            "mce": maximum_calibration_error(calibrated[name], y, n_bins),
        }
        if fit.note and fit.n_positive == 0:
            notes[name] = f"{notes.get(name, '')} {fit.note}".strip()

    return CalibrationBundle(
        temperatures=dict(scaler.temperatures),
        thresholds=thresholds,
        aspects=aspects,
        criterion=criterion,
        fitted_on=fitted_on,
        model_id=model_id,
        n_samples=n_samples,
        metrics=metrics,
        notes=notes,
    )
