#!/usr/bin/env python3
"""Check whether a configuration can physically do what it is asked to do.

Three budgets have to close simultaneously, and none of them announces itself
when it does not. Each failure mode looks like a working system producing
disappointing results:

* **Throughput.** The field of view, the flow speed and the sample
  concentration together determine how many trackable sperm pass per second.
  Too few and every shot times out below the minimum, reported INDETERMINATE.
* **Observation.** A faster flow delivers more sperm but gives fewer frames of
  evidence about each. Below the track-quality bar every track is discarded.
* **Latency.** The decision must reach the magnet before the fluid does. If it
  cannot, the scheduler drops every command -- each component having behaved
  correctly in isolation.

Run this after any change to the optics, the flow rate, the shot sizing or the
morphology deadline.

Examples
--------
    python scripts/check_feasibility.py -c configs/device_v1.yaml
    python scripts/check_feasibility.py -c configs/device_v1.yaml \
        --flow-um-s 500 --chamber-depth-um 30
    python scripts/check_feasibility.py -c configs/default.yaml --sweep-flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sperm_sorting.config import load_config  # noqa: E402
from sperm_sorting.shots.feasibility import assess_feasibility  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("-s", "--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--flow-um-s", type=float, default=None, help="Override the bulk flow speed."
    )
    parser.add_argument(
        "--chamber-depth-um",
        type=float,
        default=20.0,
        help="Optical section thickness contributing countable sperm. UNKNOWN "
        "for the prototype; the implied concentration scales inversely with it.",
    )
    parser.add_argument(
        "--sweep-flow",
        action="store_true",
        help="Show the trade-off across a range of flow speeds.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    if args.sweep_flow:
        print(
            f"{'flow um/s':>10} {'residence':>10} {'frames':>8} "
            f"{'visible':>8} {'conc M/mL':>10}  status"
        )
        print("-" * 62)
        for flow in (50, 100, 200, 331, 500, 800, 1200, 2000):
            report = assess_feasibility(
                cfg, flow_speed_um_s=float(flow),
                chamber_depth_um=args.chamber_depth_um,
            )
            status = "ok" if report.feasible else f"{len(report.warnings)} warning(s)"
            print(
                f"{flow:>10} {report.residence_time_s * 1000:>9.0f}ms "
                f"{report.frames_per_transit:>8.1f} "
                f"{report.required_visible_sperm:>8.1f} "
                f"{report.required_concentration_per_ml / 1e6:>10.1f}  {status}"
            )
        print(
            "\nFaster flow delivers more sperm per second but leaves fewer frames\n"
            "of evidence about each. The usable window is where the frame count\n"
            "clears the track-quality bar and the implied concentration is one a\n"
            "real sample can provide (the WHO 6th-ed 5th centile is 16 M/mL)."
        )
        return 0

    report = assess_feasibility(
        cfg,
        flow_speed_um_s=args.flow_um_s,
        chamber_depth_um=args.chamber_depth_um,
    )
    print(report.format_report())
    return 0 if report.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
