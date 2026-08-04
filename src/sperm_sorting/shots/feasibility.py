"""Shot-throughput feasibility budget.

The shot definition and the optics impose competing demands, and whether they
can both be met is a matter of arithmetic rather than opinion. This module does
that arithmetic so the answer is checkable, is re-checked whenever the
configuration changes, and shows up as a startup warning rather than as a
puzzling stream of ``INDETERMINATE`` shots weeks later.

The tension
-----------
A shot must contain 20-30 uniquely trackable sperm and must close within one
second. So the imaging region has to *deliver* at least 20 trackable sperm per
second.

At the same time, each of those sperm must be observed long enough to measure
its velocity. Residence time in the field is ``field_length / flow_speed``, so
a faster flow delivers more sperm but gives less evidence about each one.

And the field is small. On the reference build the sample-plane sampling is
0.0345 um/px, which puts the field of view at 66.2 x 41.4 um -- shorter than a
whole 50-60 um spermatozoon. That is comfortable for head-centroid CASA
tracking (a head is ~119 x 81 px) but means whole-flagellum imaging is
orientation-dependent and frequently impossible.

Putting those together gives a required simultaneous density, and from the
chamber depth, a required sperm concentration. If that concentration exceeds
what the sample actually has, the configuration cannot produce full shots and
will time out into ``INDETERMINATE`` no matter how good the models are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import AppConfig


@dataclass(slots=True)
class FeasibilityReport:
    """Whether the configured geometry can deliver the configured shot rate."""

    #: Sample-plane sampling actually assumed, um/px.
    um_per_px: float
    #: Whether that came from a real calibration or from nominal optics.
    um_per_px_is_measured: bool
    field_width_um: float
    field_height_um: float
    #: Extent of the field along the flow axis.
    field_length_um: float
    #: Assumed bulk flow speed along that axis, um/s.
    flow_speed_um_s: float
    #: How long a passively transported object stays in the field.
    residence_time_s: float
    #: Frames observed during residence at the configured frame rate.
    frames_per_transit: float
    #: Frames the track-quality bar demands.
    min_frames_required: int
    #: Simultaneous visible sperm needed to deliver the shot target in time.
    required_visible_sperm: float
    #: Implied sperm concentration, per millilitre.
    required_concentration_per_ml: float
    #: Chamber depth used for the concentration estimate, um.
    chamber_depth_um: float
    #: Whole-sperm length used for the fit check, um.
    sperm_length_um: float
    whole_sperm_fits_along_flow: bool
    whole_sperm_fits_across_flow: bool

    #: Worst-case interval from a sperm crossing the counting gate to its
    #: shot's decision being available, in milliseconds.
    decision_latency_budget_ms: float
    #: Transport delay the decision has to beat, in milliseconds.
    transport_delay_ms: float
    #: True when the decision can reach the magnet before the fluid does.
    decision_arrives_in_time: bool

    warnings: list[str]

    @property
    def feasible(self) -> bool:
        return not self.warnings

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "um_per_px": self.um_per_px,
            "um_per_px_is_measured": self.um_per_px_is_measured,
            "field_width_um": self.field_width_um,
            "field_height_um": self.field_height_um,
            "field_length_um": self.field_length_um,
            "flow_speed_um_s": self.flow_speed_um_s,
            "residence_time_s": self.residence_time_s,
            "frames_per_transit": self.frames_per_transit,
            "min_frames_required": self.min_frames_required,
            "required_visible_sperm": self.required_visible_sperm,
            "required_concentration_per_ml": self.required_concentration_per_ml,
            "chamber_depth_um": self.chamber_depth_um,
            "whole_sperm_fits_along_flow": self.whole_sperm_fits_along_flow,
            "whole_sperm_fits_across_flow": self.whole_sperm_fits_across_flow,
            "decision_latency_budget_ms": self.decision_latency_budget_ms,
            "transport_delay_ms": self.transport_delay_ms,
            "decision_arrives_in_time": self.decision_arrives_in_time,
            "feasible": self.feasible,
            "warnings": list(self.warnings),
        }

    def format_report(self) -> str:
        lines = [
            "Shot throughput feasibility",
            "---------------------------",
            f"  sampling            : {self.um_per_px:.5f} um/px "
            f"({'measured' if self.um_per_px_is_measured else 'NOMINAL, uncalibrated'})",
            f"  field of view       : {self.field_width_um:.2f} x "
            f"{self.field_height_um:.2f} um",
            f"  flow speed          : {self.flow_speed_um_s:.1f} um/s",
            f"  residence time      : {self.residence_time_s * 1000:.1f} ms "
            f"({self.frames_per_transit:.1f} frames, "
            f"{self.min_frames_required} required)",
            f"  visible sperm needed: {self.required_visible_sperm:.2f} "
            "simultaneously",
            f"  implied concentration: "
            f"{self.required_concentration_per_ml / 1e6:.1f} million/mL "
            f"(assuming {self.chamber_depth_um:.0f} um chamber depth)",
            f"  whole {self.sperm_length_um:.0f} um sperm fits along flow: "
            f"{self.whole_sperm_fits_along_flow}, across flow: "
            f"{self.whole_sperm_fits_across_flow}",
            f"  decision latency    : {self.decision_latency_budget_ms:.0f} ms "
            f"worst case vs {self.transport_delay_ms:.0f} ms transport "
            f"({'in time' if self.decision_arrives_in_time else 'TOO LATE'})",
        ]
        if self.warnings:
            lines.append("  WARNINGS:")
            lines.extend(f"    - {w}" for w in self.warnings)
        else:
            lines.append("  no warnings")
        return "\n".join(lines)


def assess_feasibility(
    cfg: AppConfig,
    *,
    flow_speed_um_s: float | None = None,
    chamber_depth_um: float = 20.0,
    sperm_length_um: float = 53.1,
    head_length_um: float = 4.1,
) -> FeasibilityReport:
    """Compute the throughput budget for a configuration.

    Parameters
    ----------
    flow_speed_um_s
        Bulk flow along the gate axis. Defaults to the calibrated fixed flow
        vector when one is configured, otherwise to the synthetic source's
        flow, converted through the sampling scale.
    chamber_depth_um
        Optical section thickness contributing countable sperm. This is a
        property of the kit and of the depth of field, and is *not* known for
        the prototype -- 20 um is a placeholder for the estimate, and the
        resulting concentration scales inversely with it.
    sperm_length_um, head_length_um
        WHO 6th-edition morphometry: head 4.1 um median, whole cell ~53 um.
    """
    optical = cfg.calibration.optical
    measured = optical.calibrated and optical.um_per_px is not None
    um_per_px = optical.um_per_px if measured else optical.nominal_um_per_px

    roi = cfg.preprocess.roi
    if roi is not None:
        width_px, height_px = int(roi[2]), int(roi[3])
    elif cfg.acquisition.kind.value == "basler":
        width_px, height_px = cfg.acquisition.basler.width, cfg.acquisition.basler.height
    else:
        width_px = cfg.acquisition.synthetic.width
        height_px = cfg.acquisition.synthetic.height

    field_w_um = width_px * um_per_px
    field_h_um = height_px * um_per_px
    axis = cfg.shots.gate.axis
    field_length_um = field_w_um if axis == "x" else field_h_um
    field_cross_um = field_h_um if axis == "x" else field_w_um

    # Frame rate actually expected from the configured source.
    if cfg.acquisition.kind.value == "basler":
        fps = cfg.acquisition.basler.acquisition_frame_rate or 160.0
    elif cfg.acquisition.kind.value == "video":
        fps = cfg.acquisition.video.fallback_fps
    else:
        fps = cfg.acquisition.synthetic.fps

    # Flow speed.
    if flow_speed_um_s is None:
        flow_cfg = cfg.motion.flow_correction
        if flow_cfg.fixed_vx_px_s is not None and flow_cfg.fixed_vy_px_s is not None:
            px_s = (
                abs(flow_cfg.fixed_vx_px_s)
                if axis == "x"
                else abs(flow_cfg.fixed_vy_px_s)
            )
        else:
            syn = cfg.acquisition.synthetic
            px_s = abs(syn.flow_vx_px_s) if axis == "x" else abs(syn.flow_vy_px_s)
        flow_speed_um_s = px_s * um_per_px

    warnings: list[str] = []

    if flow_speed_um_s <= 0:
        # A still fluid means residence is unbounded; the budget is vacuous
        # but the configuration is not necessarily wrong (bench tests run
        # with no flow), so this is a note rather than a failure.
        residence_s = float("inf")
        frames_per_transit = float("inf")
        required_visible = 0.0
        warnings.append(
            "bulk flow is zero, so the shot-throughput budget does not apply; "
            "this is expected only for a static bench recording"
        )
    else:
        residence_s = field_length_um / flow_speed_um_s
        frames_per_transit = residence_s * fps
        # To deliver `target` sperm within `max_duration`, the field must hold
        # on average target * (residence / duration) sperm at any instant.
        duration = cfg.shots.maximum_shot_duration_seconds
        required_visible = cfg.shots.target_trackable_sperm * (residence_s / duration)

    min_frames = cfg.track_quality.min_observed_points
    if frames_per_transit < min_frames:
        warnings.append(
            f"a sperm crosses the field in {frames_per_transit:.1f} frames but "
            f"track_quality.min_observed_points requires {min_frames}; every "
            "track will fail the quality bar. Reduce the flow speed, raise the "
            "frame rate, or lower the minimum."
        )

    # Concentration implied by the required simultaneous count.
    volume_um3 = field_w_um * field_h_um * chamber_depth_um
    volume_ml = volume_um3 * 1e-12  # 1 um^3 = 1e-12 mL
    required_conc = (required_visible / volume_ml) if volume_ml > 0 else float("inf")

    # WHO 6th-edition 5th-centile concentration is 16 million/mL. Requiring
    # much more than a healthy sample provides means the shot will time out.
    if required_conc > 2e8:
        warnings.append(
            f"the configuration needs ~{required_conc / 1e6:.0f} million "
            "sperm/mL to fill a shot in time, which is above what samples "
            "typically provide (WHO 6th-ed 5th centile is 16 million/mL). "
            "Shots will usually time out below the minimum and be reported "
            "INDETERMINATE. Consider a lower magnification, a larger ROI, a "
            "faster flow, or a longer maximum shot duration."
        )

    # ---- decision latency vs transport delay -------------------------
    # A decision is only useful if it reaches the magnet before the fluid it
    # describes does. The worst case runs from the *first* member of a shot
    # crossing the gate to that shot's decision being available:
    #
    #   shot assembly   the shot may stay open for its full duration limit
    # + track drain     the last member must leave the field to be analysed
    # + morphology      the finalisation deadline
    #
    # This is the constraint that decides whether the device can sort at all.
    # If it is violated every command is dispatched after its fluid has passed,
    # the scheduler drops them all, and the run produces decisions that were
    # never acted on -- with no error raised anywhere, because each component
    # behaved correctly in isolation.
    residence_ms = (residence_s * 1000.0) if flow_speed_um_s > 0 else 0.0
    assembly_ms = cfg.shots.maximum_shot_duration_seconds * 1000.0
    decision_latency_ms = assembly_ms + residence_ms + cfg.morphology.deadline_ms
    transport_ms = cfg.scheduling.transport_delay_ms
    in_time = transport_ms > decision_latency_ms

    if cfg.scheduling.calibrated and not in_time:
        warnings.append(
            f"the worst-case decision latency is {decision_latency_ms:.0f} ms "
            f"(shot assembly {assembly_ms:.0f} + field transit "
            f"{residence_ms:.0f} + morphology deadline "
            f"{cfg.morphology.deadline_ms:.0f}) but the transport delay is only "
            f"{transport_ms:.0f} ms. Decisions will arrive after their fluid "
            "has passed the magnet and every command will be dropped as late. "
            "Move the magnet further downstream, slow the flow, shorten the "
            "maximum shot duration, or reduce the morphology deadline."
        )

    fits_along = sperm_length_um <= field_length_um
    fits_across = sperm_length_um <= field_cross_um
    if not fits_across:
        warnings.append(
            f"a whole {sperm_length_um:.0f} um spermatozoon does not fit across "
            f"the {field_cross_um:.1f} um field dimension, so full-flagellum "
            "imaging depends on cell orientation and will often be truncated. "
            "Head-centroid tracking is unaffected (a head spans "
            f"{head_length_um / um_per_px:.0f} px), but tail morphology "
            "will frequently be uncertain and should be expected to report "
            "tail_complete=False."
        )

    # Crowding is judged twice, because heads and tails crowd differently.
    # Heads are what the detector localises, so head crowding decides whether
    # detection is even possible; tails are what a morphology crop contains,
    # so tail crowding decides whether that crop is contaminated.
    field_area = field_w_um * field_h_um
    head_area = head_length_um * (head_length_um / 1.5)  # WHO L:W ratio ~1.5
    if required_visible > 0:
        head_occupancy = required_visible * head_area / field_area
        tail_occupancy = required_visible * (sperm_length_um * head_length_um) / field_area
        if head_occupancy > 0.10:
            warnings.append(
                f"{required_visible:.1f} simultaneous sperm heads would occupy "
                f"{head_occupancy:.0%} of the field; detection and association "
                "will degrade from overlap. Reduce the flow speed or the "
                "target shot size."
            )
        elif tail_occupancy > 0.5:
            warnings.append(
                f"{required_visible:.1f} simultaneous sperm leave heads well "
                f"separated ({head_occupancy:.1%} of the field) but their "
                f"flagella sweep ~{tail_occupancy:.0%} of it, so morphology "
                "crops should be expected to contain neighbouring tails. The "
                "best-frame overlap term is what keeps those crops out of the "
                "morphology model."
            )

    return FeasibilityReport(
        um_per_px=um_per_px,
        um_per_px_is_measured=measured,
        field_width_um=field_w_um,
        field_height_um=field_h_um,
        field_length_um=field_length_um,
        flow_speed_um_s=flow_speed_um_s,
        residence_time_s=residence_s,
        frames_per_transit=frames_per_transit,
        min_frames_required=min_frames,
        required_visible_sperm=required_visible,
        required_concentration_per_ml=required_conc,
        chamber_depth_um=chamber_depth_um,
        sperm_length_um=sperm_length_um,
        whole_sperm_fits_along_flow=fits_along,
        whole_sperm_fits_across_flow=fits_across,
        decision_latency_budget_ms=decision_latency_ms,
        transport_delay_ms=transport_ms,
        decision_arrives_in_time=in_time,
        warnings=warnings,
    )
