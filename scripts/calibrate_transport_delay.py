#!/usr/bin/env python3
"""Measure the transport delay between the imaging region and the magnet.

A decision made about fluid under the microscope has to be applied to that same
fluid when it reaches the magnetic region. The interval is a property of the
built kit and cannot be guessed: a wrong value applies the field to the wrong
segment, and nothing anywhere raises an error. The scheduler therefore refuses
to arm until this measurement exists.

Procedure
---------
1. Set the pump to the flow rate the instrument will actually run at. The delay
   scales inversely with flow, so a calibration at a different rate is not a
   calibration.
2. Inject a visible tracer bolus -- dye, a bead suspension, or a small air gap.
3. Record the instant it passes the imaging region and the instant it reaches
   the magnetic region. A second camera, a photodiode or a manual keypress all
   work; whatever you use, both timestamps must come from the same clock.
4. Repeat at least three times. The *spread* sets the activation margin, so a
   single trial gives no way to size it and the script refuses fewer than three.

Also measure the field rise and fall times with a Hall probe or pickup coil at
the magnetic region, and pass them with --rise and --fall.

Examples
--------
    python scripts/calibrate_transport_delay.py \
        --imaging 0.0 1.0 2.0 3.0 --magnet 0.452 1.449 2.455 3.447 \
        --rise 8.2 --fall 6.4 --id kit-A-2026-08-04

    # Plug-flow cross-check from geometry (a sanity check, never a substitute)
    python scripts/calibrate_transport_delay.py --geometry 15 500 50 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sperm_sorting.calibration.transport import (  # noqa: E402
    estimate_from_geometry,
    estimate_from_tracer,
    save_transport_calibration,
)
from sperm_sorting.errors import CalibrationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--imaging", type=float, nargs="+", help="Arrival times at the imaging region."
    )
    parser.add_argument(
        "--magnet", type=float, nargs="+", help="Arrival times at the magnet."
    )
    parser.add_argument("--rise", type=float, help="Measured field rise time, ms.")
    parser.add_argument("--fall", type=float, help="Measured field fall time, ms.")
    parser.add_argument(
        "--geometry",
        type=float,
        nargs=4,
        metavar=("LENGTH_MM", "WIDTH_UM", "HEIGHT_UM", "FLOW_UL_MIN"),
        help="Plug-flow cross-check from channel geometry.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/calibration/transport.json"),
    )
    parser.add_argument("--id", default="", help="Calibration identifier.")
    args = parser.parse_args()

    if args.geometry:
        length, width, height, flow = args.geometry
        try:
            delay = estimate_from_geometry(length, width, height, flow)
        except CalibrationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"plug-flow estimate: {delay:.1f} ms")
        print(
            "\nThis is a CROSS-CHECK ONLY. It assumes uniform plug flow and so\n"
            "ignores the parabolic velocity profile of pressure-driven flow (the\n"
            "centreline runs about 1.5-2x the mean), Taylor dispersion, and dead\n"
            "volume in connectors. Expect the measured delay to differ, and\n"
            "trust the measurement."
        )
        if not args.imaging:
            return 0

    if not args.imaging or not args.magnet:
        parser.error("--imaging and --magnet are both required for a measurement")

    try:
        result = estimate_from_tracer(args.imaging, args.magnet)
    except CalibrationError as exc:
        print(f"\nCALIBRATION REJECTED\n  {exc}", file=sys.stderr)
        return 1

    if args.rise is not None:
        result.field_rise_time_ms = args.rise
    if args.fall is not None:
        result.field_fall_time_ms = args.fall

    print("\nmeasured")
    print(
        f"  transport delay : {result.transport_delay_ms:.2f} "
        f"+/- {result.transport_delay_std_ms:.2f} ms  (n={result.n_trials})"
    )
    print(f"  field rise      : {result.field_rise_time_ms}")
    print(f"  field fall      : {result.field_fall_time_ms}")
    print(f"  notes           : {result.notes}")

    cfg = result.to_config(args.id or "CHANGE-ME")
    save_transport_calibration(result, args.out)
    print(f"\nwrote {args.out}")

    print("\nAdd to your config:")
    print("  scheduling:")
    print("    calibrated: true")
    print(f"    calibration_id: {args.id or 'CHANGE-ME'}")
    print(f"    transport_delay_ms: {result.transport_delay_ms:.2f}")
    print(f"    transport_delay_std_ms: {result.transport_delay_std_ms:.2f}")
    if result.field_rise_time_ms is not None:
        print(f"    field_rise_time_ms: {result.field_rise_time_ms:.2f}")
    if result.field_fall_time_ms is not None:
        print(f"    field_fall_time_ms: {result.field_fall_time_ms:.2f}")
    print(f"    pre_activation_margin_ms: {cfg.pre_activation_margin_ms:.2f}")
    print(f"    post_activation_margin_ms: {cfg.post_activation_margin_ms:.2f}")
    print(
        "\nThe margins are three standard deviations of the measured spread, so\n"
        "the field is in state before the segment arrives in essentially every\n"
        "case rather than only on average."
    )
    print(
        "\nNow run `sperm-sorting feasibility` with this configuration. The\n"
        "worst-case decision latency must be SHORTER than the transport delay,\n"
        "or every command will be dropped as late."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
