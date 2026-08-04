#!/usr/bin/env python3
"""End-to-end product evaluation: does the *pipeline* work, not each model.

    python training/eval_pipeline.py -c configs/synthetic.yaml \\
        -s run.max_frames=600 -o runs/pipeline_eval

Why this script is the important one
------------------------------------
``eval_morphology`` answers "does the morphology head classify crops".
``eval_detector`` answers "does the detector find heads". ``eval_tracking``
answers "does one cell keep one id". None of them answers the question the
product is judged on, which is:

    for a segment of fluid, does this system energise the magnet when it
    should and leave it alone when it should not?

That question only has an answer where per-sperm ground truth exists, and the
only source with it is the simulator: VISEM-Tracking has boxes and identities
but no morphology, MHSMA has morphology but no video, and VISEM has only
sample-level percentages. The simulator samples a ground-truth
:class:`~sperm_sorting.simulator.params.HealthState` first and derives both
appearance and trajectory from it, so a single virtual cell is jointly labelled
for morphology *and* motion. That is what makes the conjunctive rule -- which
is what the product implements -- measurable at all.

What is measured
----------------
``per-sperm eligibility agreement``
    Predicted :attr:`TrackRecord.ai_eligible` against the truth, for every
    track that was actually gated into a shot. Reported with the confusion
    matrix and the breakdown of
    :class:`~sperm_sorting.schemas.enums.IneligibilityReason` against the true
    reason, because *why* the system disagreed determines which stage to fix.

``shot-ratio error``
    Predicted ``ai_eligible_ratio`` against the ratio computed from the true
    states of the same shot's members. Signed mean first: a systematic bias
    moves every decision the same way, and the 60% rule is a threshold on
    exactly this number.

``shot-decision confusion matrix``
    ACCEPT / REJECT / INDETERMINATE against the decision the same rule gives on
    the true ratio. The two off-diagonal cells are not equivalent: a shot that
    should have been REJECTed and was ACCEPTed passes a poor segment to
    collection, which is the failure that matters clinically.

``indeterminate rate``
    Predicted and true. A high true rate means the *optics and flow* cannot
    deliver 20 trackable sperm per shot; a high predicted rate with a low true
    one means the pipeline is losing tracks. The two have completely different
    fixes, which is why both are reported.

``command-alignment error``
    Did the field command actually scheduled match what the truth-derived
    decision required, and did it arrive on time. This is where the answer
    stops being about models: a correct decision dispatched after the fluid has
    passed the magnet is a wrong outcome.

Requirements
------------
The frame source must publish ``FramePacket.meta['gt_detections']`` and
``meta['gt_states']``; the synthetic source does. If the simulator's scene
generator is not importable in this checkout, the script says so precisely and
exits rather than reporting an empty run as a result.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from training.bootstrap import ensure_importable

ensure_importable()

import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from sperm_sorting.constants import LABEL_NORMAL  # noqa: E402
from sperm_sorting.errors import SpermSortingError  # noqa: E402
from sperm_sorting.schemas.enums import (  # noqa: E402
    FieldCommandKind,
    MotilityClass,
    ShotStatus,
)
from sperm_sorting.schemas.shot import exceeds_threshold  # noqa: E402
from training.common.args import (  # noqa: E402
    build_parser,
    describe_device,
    dump_json,
    resolve_config,
)
from training.common.experiment import ExperimentRecord  # noqa: E402
from training.common.logging_utils import console_table, print_block  # noqa: E402
from training.common.seeding import seed_everything  # noqa: E402
from training.eval_tracking import TrackSet, evaluate_tracking  # noqa: E402

__all__ = [
    "PipelineRun",
    "command_alignment",
    "eligibility_agreement",
    "map_predicted_to_ground_truth",
    "shot_decision_confusion",
    "shot_ratio_error",
    "true_eligibility",
]

#: Shot statuses, in the order they appear in the confusion matrix.
SHOT_STATUSES: tuple[str, ...] = ("accept", "reject", "indeterminate")


# ==========================================================================
# Ground-truth interpretation
# ==========================================================================


def true_eligibility(state: Mapping[str, Any]) -> bool | None:
    """Whether a ground-truth state describes an eligible sperm.

    The rule is the conjunction the runtime applies -- all four morphology
    aspects normal **and** progressive motility -- and it is deliberately
    re-derived here from the raw fields rather than trusted from a single
    summary key, with the summary used only as the fast path. The simulator's
    :func:`sperm_sorting.simulator.label.overall_label` already encodes it; if
    the two ever disagree, that disagreement is a bug worth surfacing rather
    than a formatting detail worth papering over.

    Returns ``None`` when the record carries neither the summary nor enough
    detail to reconstruct it. ``None`` is propagated, never defaulted: guessing
    "eligible" or "ineligible" for an unlabelled cell would silently move every
    number in this report.
    """
    if "overall" in state:
        return int(state["overall"]) == LABEL_NORMAL

    aspects = state.get("aspects")
    motility = state.get("motility")
    if aspects is None or motility is None:
        return None

    all_normal = all(int(v) == LABEL_NORMAL for v in aspects)
    if isinstance(motility, MotilityClass):
        progressive = motility.is_progressive
    else:
        text = str(motility)
        progressive = text in (
            str(MotilityClass.RAPID_PROGRESSIVE),
            str(MotilityClass.SLOW_PROGRESSIVE),
        )
    return bool(all_normal and progressive)


def _normalise_states(raw: Any) -> dict[int, dict[str, Any]]:
    """Accept either ``{track_id: state}`` or a list of records with ``track_id``."""
    out: dict[int, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                out[int(key)] = dict(value)
        return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for record in raw:
            if isinstance(record, Mapping) and "track_id" in record:
                out[int(record["track_id"])] = dict(record)
    return out


# ==========================================================================
# Running the pipeline
# ==========================================================================


@dataclass
class PipelineRun:
    """Everything one pipeline run produced, plus the ground truth beside it."""

    frames_processed: int = 0
    frames_with_ground_truth: int = 0
    tracks: dict[int, Any] = field(default_factory=dict)
    shots: list[Any] = field(default_factory=list)
    decisions: list[Any] = field(default_factory=list)
    commands: list[Any] = field(default_factory=list)
    gt_tracks: TrackSet = field(default_factory=TrackSet)
    gt_states: dict[int, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    wall_clock_s: float = 0.0


def _check_source_available(cfg: Any) -> None:
    """Fail early and precisely when the ground-truth source cannot be built."""
    if cfg.acquisition.kind.value != "synthetic":
        raise SpermSortingError(
            f"acquisition.kind is '{cfg.acquisition.kind}', but per-sperm ground "
            "truth only exists for the synthetic source. Run with "
            "-c configs/synthetic.yaml, or -s acquisition.kind=synthetic."
        )
    try:
        import sperm_sorting.simulator.scene  # noqa: F401
    except ImportError as exc:
        raise SpermSortingError(
            "the synthetic frame source needs 'sperm_sorting.simulator.scene', "
            f"which is not importable in this checkout ({exc}). That module is the "
            "procedural scene generator that publishes FramePacket.meta"
            "['gt_detections'] and meta['gt_states']; without it there is no "
            "per-sperm ground truth to evaluate against, and this script will not "
            "report an empty run as a result. The rest of the training harness "
            "does not depend on it."
        ) from exc


def run_pipeline(cfg: Any, *, max_frames: int | None = None) -> PipelineRun:
    """Drive the real pipeline frame by frame and capture the ground truth.

    :class:`~sperm_sorting.app.Application` builds every component -- detector,
    tracker, morphology engine, scheduler, actuator, audit log -- exactly as a
    production run does, so this measures the assembled system rather than a
    re-wiring of it. What is *not* used is
    :class:`~sperm_sorting.runtime.workers.PipelineRunner`: frames are pushed
    through :meth:`Pipeline.process_frame` directly, for two reasons. The
    threaded runner may drop frames under back-pressure, which would make the
    measurement depend on this machine's scheduling; and the ground truth has
    to be read off each packet as it goes past, which the runner does not
    expose.
    """
    from sperm_sorting.app import Application

    _check_source_available(cfg)
    run = PipelineRun()

    application = Application(cfg)
    started = time.monotonic()
    try:
        application.setup()
        source = application.source
        pipeline = application.pipeline
        if source is None or pipeline is None:  # pragma: no cover - setup raises first
            raise SpermSortingError("Application.setup() did not build a source and pipeline")

        limit = max_frames if max_frames is not None else cfg.run.max_frames
        while True:
            if limit is not None and run.frames_processed >= limit:
                break
            packet = source.read()
            if packet is None:
                break

            gt_detections = packet.meta.get("gt_detections")
            if gt_detections:
                run.frames_with_ground_truth += 1
                for record in gt_detections:
                    track_id = record.get("track_id")
                    if track_id is None:
                        continue
                    run.gt_tracks.add(packet.frame_id, int(track_id), record["box_xyxy"])
            states = packet.meta.get("gt_states")
            if states:
                run.gt_states.update(_normalise_states(states))

            result = pipeline.process_frame(packet)
            run.frames_processed += 1
            run.decisions.extend(result.decisions)
            run.commands.extend(result.commands)

        run.decisions.extend(pipeline.flush())
        run.tracks = dict(pipeline.tracks_by_id)
        run.shots = list(pipeline.shots.history)
        # The scheduler's own history, not the per-frame FrameResult lists. Shots
        # decided during flush() -- which is most of the last few on any bounded
        # run -- issue their commands after the last FrameResult exists, so
        # collecting from the frames alone silently loses them and makes those
        # shots look like they were never commanded.
        run.commands = list(pipeline.scheduler.history)
        run.summary = pipeline.summary()
        run.gt_tracks.fps = float(cfg.acquisition.synthetic.fps)
    finally:
        application.close()

    run.wall_clock_s = time.monotonic() - started
    return run


def predicted_track_set(run: PipelineRun) -> TrackSet:
    """Observed positions of every predicted track, for identity matching.

    Only *observed* points are used. An interpolated point is the tracker's
    prediction, not a measurement, and matching on predictions would credit the
    tracker for identity it maintained through a gap in which it saw nothing --
    which is exactly the situation where identity is most likely to be wrong.
    """
    tracks = TrackSet(fps=run.gt_tracks.fps)
    for track in run.tracks.values():
        for point in track.points:
            if point.observed:
                tracks.add(point.frame_id, track.track_id, point.box.as_xyxy())
    return tracks


# ==========================================================================
# Predicted <-> ground-truth identity
# ==========================================================================


def map_predicted_to_ground_truth(
    predicted: TrackSet, truth: TrackSet, *, iou_threshold: float = 0.5
) -> dict[int, int | None]:
    """Map each predicted track to the ground-truth sperm it mostly observed.

    Deliberately **many-to-one**, not a bijection. If the tracker split one
    cell into two ids, both must map to that cell: that is what makes the
    duplicate visible as two counted sperm with the same truth, which is
    precisely the error being measured. A one-to-one assignment would hide it
    by declaring the second id unmatched.

    A predicted track is mapped to the ground-truth track with which it shares
    the most frames at ``IoU >= iou_threshold``, and to ``None`` when it shares
    none -- a phantom, which is scored as a counted sperm that does not exist.
    """
    overlap: dict[int, dict[int, int]] = {}
    for frame_id in sorted(set(predicted.frame_ids) & set(truth.frame_ids)):
        pred_frame = predicted.observations[frame_id]
        gt_frame = truth.observations[frame_id]
        for pred_id, pred_box in pred_frame.items():
            for gt_id, gt_box in gt_frame.items():
                from training.eval_tracking import _iou

                if _iou(pred_box, gt_box) >= iou_threshold:
                    overlap.setdefault(pred_id, {})
                    overlap[pred_id][gt_id] = overlap[pred_id].get(gt_id, 0) + 1

    mapping: dict[int, int | None] = {}
    for pred_id in predicted.track_ids:
        candidates = overlap.get(pred_id)
        if not candidates:
            mapping[pred_id] = None
            continue
        # Ties broken by the smaller ground-truth id, so the mapping is a
        # deterministic function of the data rather than of dict ordering.
        best = max(sorted(candidates.items()), key=lambda item: item[1])
        mapping[pred_id] = int(best[0])
    return mapping


# ==========================================================================
# Product metrics
# ==========================================================================


def eligibility_agreement(
    run: PipelineRun, mapping: Mapping[int, int | None]
) -> dict[str, Any]:
    """Predicted ``ai_eligible`` against the truth, per gated sperm.

    Only tracks that were **gated into a shot** are scored, because only those
    are in a denominator. A track that never crossed the counting gate changed
    no decision and is not part of the product's output; including it would
    dilute the number with observations the system deliberately ignored.

    The positive class is *eligible*, so a false positive is a poor sperm
    counted as good -- the error that pushes a shot over the 60% line and lets
    a bad segment through.
    """
    tp = fp = fn = tn = 0
    unknown = 0
    phantom = 0
    reason_confusion: dict[str, dict[str, int]] = {}

    for track in run.tracks.values():
        if track.shot_id is None:
            continue
        gt_id = mapping.get(track.track_id)
        if gt_id is None:
            phantom += 1
            continue
        state = run.gt_states.get(gt_id)
        truth = true_eligibility(state) if state is not None else None
        if truth is None:
            unknown += 1
            continue

        predicted = bool(track.ai_eligible)
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
        else:
            tn += 1

        if predicted != truth:
            reason = str(track.ineligibility_reason)
            bucket = reason_confusion.setdefault(reason, {"false_positive": 0, "false_negative": 0})
            bucket["false_positive" if predicted else "false_negative"] += 1

    n = tp + fp + fn + tn

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    mcc_numerator = float(tp) * tn - float(fp) * fn
    mcc_denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n_scored": n,
        "n_phantom_tracks_gated": phantom,
        "n_unknown_truth": unknown,
        "agreement": ratio(tp + tn, n),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": float(mcc_numerator / np.sqrt(mcc_denominator)) if mcc_denominator > 0 else 0.0,
        "true_eligible_rate": ratio(tp + fn, n),
        "predicted_eligible_rate": ratio(tp + fp, n),
        "disagreement_by_predicted_reason": reason_confusion,
        "positive_class": "eligible (progressive AND all four morphology aspects normal)",
        "note": (
            "a false positive here is a poor sperm counted as good, which pushes a "
            "shot toward ACCEPT and lets a bad segment through"
        ),
    }


def shot_ratio_error(
    run: PipelineRun, mapping: Mapping[int, int | None]
) -> dict[str, Any]:
    """Predicted shot ratio against the truth-derived ratio, per shot.

    The true ratio is computed over **the shot's own members**, not over the
    population: the product's decision is about this segment of fluid, and
    comparing against a population average would measure sampling noise instead
    of the pipeline.

    A track whose truth is unknown is excluded from that shot's true numerator
    *and* denominator, and the exclusion is counted, so a shot scored on half
    its members is visible as such.
    """
    per_shot: list[dict[str, Any]] = []
    for shot in run.shots:
        eligible_true = 0
        denominator = 0
        unknown = 0
        for track_id in shot.track_ids:
            gt_id = mapping.get(track_id)
            state = run.gt_states.get(gt_id) if gt_id is not None else None
            truth = true_eligibility(state) if state is not None else None
            if truth is None:
                unknown += 1
                continue
            denominator += 1
            eligible_true += int(truth)

        predicted_ratio = (
            float(shot.ai_eligible_ratio)
            if shot.ai_eligible_ratio is not None
            else shot.compute_ratio()
        )
        true_ratio = float(eligible_true / denominator) if denominator else float("nan")
        per_shot.append(
            {
                "shot_id": int(shot.shot_id),
                "trackable_count": int(shot.trackable_count),
                "predicted_eligible": int(shot.ai_eligible_count),
                "true_eligible": eligible_true,
                "true_denominator": denominator,
                "n_unknown_truth": unknown,
                "predicted_ratio": predicted_ratio,
                "true_ratio": true_ratio,
                "error": predicted_ratio - true_ratio if denominator else float("nan"),
            }
        )

    errors = np.array(
        [row["error"] for row in per_shot if row["error"] == row["error"]], dtype=np.float64
    )
    return {
        "n_shots": len(per_shot),
        "n_shots_scored": int(errors.size),
        "mean_signed_error": float(np.mean(errors)) if errors.size else float("nan"),
        "mean_absolute_error": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan"),
        "max_absolute_error": float(np.max(np.abs(errors))) if errors.size else float("nan"),
        "per_shot": per_shot,
        "note": (
            "mean_signed_error is the one that decides outcomes: the rule is a "
            "threshold at 0.60, so a constant bias flips every marginal shot the "
            "same way"
        ),
    }


def _truth_status(
    eligible: int, trackable: int, threshold: float, minimum: int
) -> ShotStatus:
    """The decision the product's own rule gives on truth-derived counts.

    Uses :func:`sperm_sorting.schemas.shot.exceeds_threshold`, the same exact
    rational comparison the decision engine uses, rather than a float ``>``.
    The rule is that exactly 60% REJECTs, and binary floating point cannot
    represent 0.60; re-deriving the comparison here with ``/`` would make the
    reference decision differ from the product's on precisely the boundary
    cases this report exists to check.
    """
    if trackable < minimum:
        return ShotStatus.INDETERMINATE
    return ShotStatus.ACCEPT if exceeds_threshold(eligible, trackable, threshold) else ShotStatus.REJECT


def shot_decision_confusion(
    run: PipelineRun, ratios: Mapping[str, Any], cfg: Any
) -> dict[str, Any]:
    """ACCEPT / REJECT / INDETERMINATE against the truth-derived decision."""
    threshold = float(cfg.decision.threshold)
    minimum = int(cfg.decision.minimum_trackable_sperm)

    matrix = {t: dict.fromkeys(SHOT_STATUSES, 0) for t in SHOT_STATUSES}
    rows: list[dict[str, Any]] = []
    shots_by_id = {int(shot.shot_id): shot for shot in run.shots}

    for row in ratios["per_shot"]:
        shot = shots_by_id[int(row["shot_id"])]
        predicted = shot.status
        if predicted is None:
            continue
        truth = _truth_status(
            int(row["true_eligible"]), int(row["true_denominator"]), threshold, minimum
        )
        matrix[str(truth)][str(predicted)] += 1
        rows.append(
            {
                "shot_id": row["shot_id"],
                "true_status": str(truth),
                "predicted_status": str(predicted),
                "agree": str(truth) == str(predicted),
                "true_ratio": row["true_ratio"],
                "predicted_ratio": row["predicted_ratio"],
            }
        )

    total = sum(sum(inner.values()) for inner in matrix.values())
    agree = sum(matrix[s][s] for s in SHOT_STATUSES)
    predicted_indeterminate = sum(matrix[t]["indeterminate"] for t in SHOT_STATUSES)
    true_indeterminate = sum(matrix["indeterminate"].values())

    # The dangerous cell: truth says REJECT (poor segment, field should be ON)
    # but the pipeline said ACCEPT (field OFF, segment goes to collection).
    reject_called_accept = matrix["reject"]["accept"]
    accept_called_reject = matrix["accept"]["reject"]

    return {
        "n_decided_shots": total,
        "matrix_true_by_predicted": matrix,
        "row_order": list(SHOT_STATUSES),
        "column_order": list(SHOT_STATUSES),
        "agreement": float(agree / total) if total else float("nan"),
        "indeterminate_rate_predicted": float(predicted_indeterminate / total) if total else float("nan"),
        "indeterminate_rate_true": float(true_indeterminate / total) if total else float("nan"),
        "reject_called_accept": reject_called_accept,
        "accept_called_reject": accept_called_reject,
        "asymmetry_note": (
            "reject_called_accept is the costly cell: a segment the truth says is "
            "poor was passed to collection with the field off. accept_called_reject "
            "wastes a good segment, which is a yield loss rather than a quality one."
        ),
        "per_shot": rows,
        "decision_rule": (
            f"INDETERMINATE below {minimum} trackable; otherwise ACCEPT iff "
            f"ratio > {threshold} by exact rational comparison (exactly {threshold} "
            "REJECTs)"
        ),
    }


def command_alignment(run: PipelineRun, decisions: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    """Did the field command match the truth-derived requirement, and arrive in time.

    Two separate failures are counted separately because they have different
    causes and different fixes:

    * a **wrong** command -- the decision itself disagreed with the truth;
    * a **late or dropped** command -- the decision was right but reached the
      actuator after the fluid segment had passed the magnet, which the
      scheduler records as :class:`CommandOutcome.LATE` or drops outright.

    A correct decision delivered late has exactly the same physical outcome as
    a wrong one, so a report that only counted decision errors would overstate
    the system.
    """
    required: dict[int, FieldCommandKind] = {}
    for row in decisions["per_shot"]:
        truth = row["true_status"]
        # FIELD_ON is the rejection: energising the magnet diverts the segment.
        required[int(row["shot_id"])] = (
            FieldCommandKind.FIELD_ON if truth == str(ShotStatus.REJECT) else FieldCommandKind.FIELD_OFF
        )

    commanded: dict[int, FieldCommandKind] = {}
    outcome_by_shot: dict[int, str] = {}
    for command in run.commands:
        if command.shot_id is None:
            continue
        # A rejected shot gets FIELD_ON and then a release FIELD_OFF. The state
        # that characterises the shot is the first command it issued.
        shot_id = int(command.shot_id)
        if shot_id not in commanded:
            commanded[shot_id] = command.kind
            outcome_by_shot[shot_id] = str(command.outcome)

    matched = mismatched = 0
    without_explicit_command = 0
    not_delivered = 0
    per_shot: list[dict[str, Any]] = []

    for shot_id, want in required.items():
        issued = commanded.get(shot_id)
        # No command is not the same as no field state. `Pipeline._schedule_for`
        # omits a command only when the field is already FIELD_OFF -- an accepted
        # shot needs no actuator traffic to be handled correctly -- so the
        # *effective* state for an uncommanded shot is FIELD_OFF, and scoring it
        # as a failure would penalise the pipeline for the one optimisation it
        # is explicitly documented to make.
        effective = issued if issued is not None else FieldCommandKind.FIELD_OFF
        if issued is None:
            without_explicit_command += 1

        if effective == want:
            matched += 1
            state = "aligned"
        else:
            mismatched += 1
            state = "misaligned"

        outcome = outcome_by_shot.get(shot_id)
        # A command that was superseded or never dispatched did not reach the
        # magnet. For a shot that needs FIELD_ON that is a real failure, even
        # though the decision itself was right; it is counted separately so the
        # decision error and the delivery error are not blended.
        delivered = issued is None or outcome in ("dispatched", "acknowledged")
        if want is FieldCommandKind.FIELD_ON and not delivered:
            not_delivered += 1

        per_shot.append(
            {
                "shot_id": shot_id,
                "required": str(want),
                "commanded": str(issued) if issued is not None else None,
                "effective": str(effective),
                "first_command_outcome": outcome,
                "delivered": delivered,
                "state": state,
            }
        )

    total = len(required)
    timing_errors = [
        float(c.timing_error_s)
        for c in run.commands
        if c.timing_error_s is not None
    ]
    outcomes: dict[str, int] = {}
    for command in run.commands:
        key = str(command.outcome)
        outcomes[key] = outcomes.get(key, 0) + 1

    scheduler = run.summary
    return {
        "n_shots_with_a_decision": total,
        "n_aligned": matched,
        "n_misaligned": mismatched,
        "n_without_explicit_command": without_explicit_command,
        "n_field_on_not_delivered": not_delivered,
        "command_alignment_error": float(mismatched / total) if total else float("nan"),
        "alignment_rate": float(matched / total) if total else float("nan"),
        "delivery_failure_rate": float(not_delivered / total) if total else float("nan"),
        "commands_dispatched": int(scheduler.get("commands_dispatched", 0)),
        "commands_late": int(scheduler.get("commands_late", 0)),
        "commands_dropped_late": int(scheduler.get("commands_dropped_late", 0)),
        "command_outcomes": outcomes,
        "uncommanded_shot_policy": (
            "a shot with no command is scored as FIELD_OFF: Pipeline._schedule_for "
            "omits the command only when the field is already in the safe state, so "
            "an accepted shot correctly needs no actuator traffic"
        ),
        "timing_error_s": {
            "n": len(timing_errors),
            "mean": float(np.mean(timing_errors)) if timing_errors else float("nan"),
            "p95": float(np.percentile(timing_errors, 95)) if timing_errors else float("nan"),
            "max": float(np.max(timing_errors)) if timing_errors else float("nan"),
        },
        "per_shot": per_shot,
        "polarity_reminder": (
            "FIELD_ON is the REJECTION: energising the magnet diverts the labelled "
            "population toward waste. Reading FIELD_ON as 'good' inverts the product."
        ),
        "scheduling_calibrated": bool(cfg.scheduling.calibrated),
        "transport_delay_ms": float(cfg.scheduling.transport_delay_ms),
    }


# ==========================================================================
# CLI
# ==========================================================================


def build_argument_parser() -> Any:
    parser = build_parser(
        description="End-to-end pipeline evaluation against per-sperm ground truth.",
        epilog=(
            "Examples:\n"
            "  python training/eval_pipeline.py -c configs/synthetic.yaml -o runs/pipeline\n"
            "  python training/eval_pipeline.py -c configs/synthetic.yaml \\\n"
            "      -s detection.oracle_miss_rate=0.10 -o runs/pipeline_noisy\n"
        ),
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Stop after N frames. Defaults to run.max_frames from the config.",
    )
    run.add_argument("--iou-threshold", type=float, default=0.5)
    run.add_argument(
        "--no-tracking-metrics",
        dest="tracking_metrics",
        action="store_false",
        default=True,
        help="Skip the HOTA/IDF1 block, which is the slow part on a long run.",
    )
    return parser


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
    record = ExperimentRecord(script="eval_pipeline", out_dir=out_dir)
    record.args = {**common.to_json_dict(), "frames": args.frames, "iou_threshold": args.iou_threshold}

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
    import torch

    cfg = common.cfg
    out_dir = common.out_dir

    record.determinism = seed_everything(cfg.run.seed, cfg.run.deterministic)
    record.set_config(cfg)
    record.hardware = describe_device(torch.device("cpu"))

    run = run_pipeline(cfg, max_frames=args.frames)
    if run.frames_with_ground_truth == 0:
        raise SpermSortingError(
            f"the source produced {run.frames_processed} frame(s) but none carried "
            "meta['gt_detections']. Per-sperm ground truth is what this script "
            "measures against; without it there is nothing to report."
        )
    if not run.gt_states:
        raise SpermSortingError(
            "the source published gt_detections but no meta['gt_states'], so the "
            "true eligibility of each sperm is unknown. Detection and tracking can "
            "still be scored with eval_detector.py and eval_tracking.py, but the "
            "end-to-end product question cannot be answered."
        )

    record.set_dataset(
        name="sperm_sorting simulator (synthetic frame source)",
        licence="generated in-repo; no third-party terms apply",
        splits={"frames": run.frames_processed},
        source="sperm_sorting.acquisition.synthetic.SyntheticFrameSource",
        frames_with_ground_truth=run.frames_with_ground_truth,
        n_gt_sperm=len(run.gt_states),
        n_gt_tracks=len(run.gt_tracks.track_ids),
    )

    predicted = predicted_track_set(run)
    mapping = map_predicted_to_ground_truth(
        predicted, run.gt_tracks, iou_threshold=float(args.iou_threshold)
    )

    eligibility = eligibility_agreement(run, mapping)
    ratios = shot_ratio_error(run, mapping)
    decisions = shot_decision_confusion(run, ratios, cfg)
    commands = command_alignment(run, decisions, cfg)

    metrics: dict[str, Any] = {
        "run": {
            "frames_processed": run.frames_processed,
            "frames_with_ground_truth": run.frames_with_ground_truth,
            "wall_clock_s": round(run.wall_clock_s, 3),
            "n_predicted_tracks": len(run.tracks),
            "n_gt_sperm": len(run.gt_states),
            "n_shots": len(run.shots),
            "pipeline_summary": run.summary,
        },
        "eligibility_agreement": eligibility,
        "shot_ratio_error": ratios,
        "shot_decision": decisions,
        "command_alignment": commands,
        "identity_mapping": {
            "predicted_to_gt": {str(k): v for k, v in sorted(mapping.items())},
            "iou_threshold": float(args.iou_threshold),
            "policy": (
                "many-to-one by maximum overlapping-frame count; a duplicate track "
                "maps to the same ground-truth sperm so that double counting is "
                "visible rather than hidden as an unmatched track"
            ),
        },
    }
    if args.tracking_metrics:
        metrics["tracking"] = evaluate_tracking(
            run.gt_tracks, predicted, iou_threshold=float(args.iou_threshold)
        )

    metrics_path = dump_json(out_dir / "eval_pipeline.json", metrics)
    record.metrics = metrics
    record.artifact("metrics", metrics_path)
    if not cfg.scheduling.calibrated:
        record.note(
            "scheduling.calibrated is false, so the scheduler refused to arm and no "
            "field command was driven. The command-alignment figures below describe "
            "commands that were computed, not commands that were delivered."
        )

    _print_report(metrics, cfg)
    print_block(["", f"wrote {metrics_path}"])


def _print_report(metrics: Mapping[str, Any], cfg: Any) -> None:
    eligibility = metrics["eligibility_agreement"]
    ratios = metrics["shot_ratio_error"]
    decisions = metrics["shot_decision"]
    commands = metrics["command_alignment"]

    headline = [
        {"quantity": "per-sperm eligibility agreement", "value": eligibility["agreement"]},
        {"quantity": "  sensitivity (eligible)", "value": eligibility["sensitivity"]},
        {"quantity": "  specificity (eligible)", "value": eligibility["specificity"]},
        {"quantity": "  MCC", "value": eligibility["mcc"]},
        {"quantity": "  n scored", "value": eligibility["n_scored"]},
        {"quantity": "shot-ratio mean signed error", "value": ratios["mean_signed_error"]},
        {"quantity": "shot-ratio mean absolute error", "value": ratios["mean_absolute_error"]},
        {"quantity": "shot-decision agreement", "value": decisions["agreement"]},
        {"quantity": "  REJECT called ACCEPT (costly)", "value": decisions["reject_called_accept"]},
        {"quantity": "  ACCEPT called REJECT (yield loss)", "value": decisions["accept_called_reject"]},
        {"quantity": "indeterminate rate (predicted)", "value": decisions["indeterminate_rate_predicted"]},
        {"quantity": "indeterminate rate (true)", "value": decisions["indeterminate_rate_true"]},
        {"quantity": "command-alignment error", "value": commands["command_alignment_error"]},
        {"quantity": "  FIELD_ON not delivered", "value": commands["n_field_on_not_delivered"]},
        {"quantity": "  commands late", "value": commands["commands_late"]},
        {"quantity": "  commands dropped as late", "value": commands["commands_dropped_late"]},
    ]

    matrix_rows = [
        {
            "true \\ predicted": status,
            **{
                predicted: decisions["matrix_true_by_predicted"][status][predicted]
                for predicted in SHOT_STATUSES
            },
        }
        for status in SHOT_STATUSES
    ]

    print_block(
        [
            "",
            console_table(
                headline,
                ("value",),
                index_column="quantity",
                title=(
                    "end-to-end product metrics over "
                    f"{metrics['run']['frames_processed']} frames, "
                    f"{metrics['run']['n_shots']} shot(s)"
                ),
                footer=decisions["decision_rule"],
            ),
            "",
            console_table(
                matrix_rows,
                SHOT_STATUSES,
                index_column="true \\ predicted",
                title="shot-decision confusion matrix",
                footer=decisions["asymmetry_note"],
            ),
            "",
            commands["polarity_reminder"],
        ]
    )
    del cfg


if __name__ == "__main__":
    raise SystemExit(main())
