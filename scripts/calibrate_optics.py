#!/usr/bin/env python3
"""Measure micrometres per pixel from an imaged stage micrometer.

This is the most leveraged number in the system. It scales every velocity, and
therefore moves every sperm across the WHO 25 and 5 um/s boundaries, and
therefore changes the shot ratio and the sort. Get it wrong and nothing raises;
every result is simply wrong by a constant factor.

Procedure
---------
1. Place a stage micrometer (a certified graticule, usually 10 um rulings) on
   the stage, under the same objective, immersion oil and coupler that the
   instrument will run with. The coupler matters more than anything else here:
   a 0.5x reducing C-mount adapter is easy to overlook and puts the answer out
   by exactly a factor of two.
2. Focus carefully. A defocused graticule broadens the rulings but does not
   move them, so the period survives mild defocus -- but a tilted slide does
   shift it, so check that both ends of the field are in focus together.
3. Capture a still image with the graticule rulings running vertically.
4. Run this script on that image.

Two methods are available. The FFT method uses every ruling in the field and is
preferred; the two-point method is the fallback when the rulings are too faint
for the spectrum to show a clean peak.

Examples
--------
    python scripts/calibrate_optics.py graticule.png --pitch-um 10
    python scripts/calibrate_optics.py graticule.png --two-point 1000 34.5
    python scripts/calibrate_optics.py graticule.png --coupler 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sperm_sorting.calibration.optics import (  # noqa: E402
    calibrate_from_graticule,
    calibrate_from_known_distance,
    save_calibration,
)
from sperm_sorting.config import OpticsConfig  # noqa: E402
from sperm_sorting.errors import CalibrationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path, help="Image of the stage micrometer.")
    parser.add_argument(
        "--pitch-um",
        type=float,
        default=10.0,
        help="Certified ruling pitch of the graticule (default: 10 um).",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default="x",
        help="Axis the ruling period is measured along (default: x, for "
        "vertical rulings).",
    )
    parser.add_argument(
        "--two-point",
        nargs=2,
        type=float,
        metavar=("PIXELS", "MICROMETRES"),
        help="Fallback: a measured span instead of the FFT method.",
    )
    parser.add_argument(
        "--pixel-pitch-um", type=float, default=3.45, help="Camera pixel pitch."
    )
    parser.add_argument(
        "--magnification", type=float, default=100.0, help="Objective magnification."
    )
    parser.add_argument(
        "--coupler",
        type=float,
        default=1.0,
        help="Coupler magnification: 1.0 direct C-mount, 0.5 or 0.63 reducing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/calibration/optics.json"),
        help="Where to write the result.",
    )
    parser.add_argument(
        "--id",
        default="",
        help="Calibration identifier stamped into every audit log.",
    )
    args = parser.parse_args()

    optics = OpticsConfig(
        pixel_pitch_um=args.pixel_pitch_um,
        objective_magnification=args.magnification,
        coupler_magnification=args.coupler,
    )
    print(f"optical train implies a nominal {optics.nominal_um_per_px:.5f} um/px")

    try:
        if args.two_point:
            pixels, micrometres = args.two_point
            result = calibrate_from_known_distance(pixels, micrometres, optics)
        else:
            if not args.image.exists():
                print(f"image not found: {args.image}", file=sys.stderr)
                return 2
            image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"could not read {args.image}", file=sys.stderr)
                return 2
            result = calibrate_from_graticule(
                image, args.pitch_um, optics, axis=1 if args.axis == "x" else 0
            )
    except CalibrationError as exc:
        print(f"\nCALIBRATION REJECTED\n  {exc}", file=sys.stderr)
        return 1

    print("\nmeasured")
    print(f"  um per pixel        : {result.um_per_px:.6f}")
    print(f"  nominal             : {result.nominal_um_per_px:.6f}")
    print(f"  measured / nominal  : {result.nominal_ratio:.4f}")
    print(f"  relative uncertainty: {result.relative_uncertainty:.4%}")
    print(f"  method              : {result.method} (n={result.n_samples})")
    print(f"  notes               : {result.notes}")

    width, height = 1920, 1200
    print("\nimplied geometry at 1920 x 1200")
    print(
        f"  field of view       : {width * result.um_per_px:.2f} x "
        f"{height * result.um_per_px:.2f} um"
    )
    print(f"  sperm head (4.1 um) : {4.1 / result.um_per_px:.0f} px")

    save_calibration(result, args.out)
    print(f"\nwrote {args.out}")
    print("\nAdd to your config:")
    print("  calibration:")
    print("    optical:")
    print("      calibrated: true")
    print(f"      calibration_id: {args.id or 'CHANGE-ME'}")
    print(f"      um_per_px: {result.um_per_px:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
