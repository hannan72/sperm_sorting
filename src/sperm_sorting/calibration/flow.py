"""Bulk-flow calibration.

Observed sperm motion is swimming plus transport. Without removing the
transport component, a dead sperm drifting at 300 um/s in the channel would be
graded rapidly progressive, and the shot ratio would measure the pump rather
than the sample. This module measures the transport component so it can be
subtracted.

Two products are produced:

* a **fixed vector**, the mean flow across the field, adequate when the
  imaging region sits in the middle of a wide channel;
* a **flow map**, a per-pixel velocity field, which matters because pressure-
  driven flow in a microchannel is parabolic across the section (Poiseuille),
  so fluid near a wall moves markedly slower than fluid at the centre. A
  single vector over-corrects at the walls and under-corrects at the centre.

Both are estimated from objects that are *not* swimming: debris, or a control
run with a non-motile (e.g. heat-treated or fixed) sample. Estimating flow
from live sperm would subtract part of their own motility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CalibrationError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FlowCalibrationResult:
    """Measured bulk flow."""

    vx_px_s: float
    vy_px_s: float
    #: Robust spread of the per-track estimates, in px/s.
    vx_std_px_s: float
    vy_std_px_s: float
    n_tracks: int
    method: str
    #: Populated when a spatial map was fitted.
    map_shape: tuple[int, int] | None = None
    notes: str = ""

    @property
    def speed_px_s(self) -> float:
        return float(np.hypot(self.vx_px_s, self.vy_px_s))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "vx_px_s": self.vx_px_s,
            "vy_px_s": self.vy_px_s,
            "vx_std_px_s": self.vx_std_px_s,
            "vy_std_px_s": self.vy_std_px_s,
            "speed_px_s": self.speed_px_s,
            "n_tracks": self.n_tracks,
            "method": self.method,
            "map_shape": list(self.map_shape) if self.map_shape else None,
            "notes": self.notes,
        }


def _track_velocities(tracks: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-track mean velocity in px/s, plus the mean position of each track.

    Uses only observed points, and requires at least two of them spanning a
    positive duration; anything else has no defined velocity.
    """
    velocities: list[tuple[float, float]] = []
    positions: list[tuple[float, float]] = []
    for track in tracks:
        points = [p for p in track.points if p.observed]
        if len(points) < 2:
            continue
        dt = points[-1].capture_time_s - points[0].capture_time_s
        if dt <= 0:
            continue
        velocities.append(
            ((points[-1].x - points[0].x) / dt, (points[-1].y - points[0].y) / dt)
        )
        positions.append(
            (
                float(np.mean([p.x for p in points])),
                float(np.mean([p.y for p in points])),
            )
        )
    if not velocities:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(velocities, dtype=np.float64), np.asarray(
        positions, dtype=np.float64
    )


def calibrate_fixed_vector(
    tracks: list[Any],
    *,
    quantile: float = 0.25,
    min_tracks: int = 8,
) -> FlowCalibrationResult:
    """Estimate one flow vector from the slowest fraction of tracks.

    The slowest tracks are taken to be passively transported. The median is
    used rather than the mean so that a handful of fast swimmers that survive
    the quantile cut cannot drag the estimate; the spread is reported as a
    median absolute deviation, scaled to be comparable with a standard
    deviation, for the same reason.
    """
    velocities, _ = _track_velocities(tracks)
    if len(velocities) < min_tracks:
        raise CalibrationError(
            f"only {len(velocities)} usable tracks; at least {min_tracks} are "
            "needed for a flow estimate. Record a longer clip, or use a "
            "non-motile control sample."
        )

    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    cutoff = float(np.quantile(speeds, quantile))
    slow = velocities[speeds <= cutoff]
    if len(slow) < 3:
        slow = velocities[np.argsort(speeds)[:3]]

    vx = float(np.median(slow[:, 0]))
    vy = float(np.median(slow[:, 1]))
    # MAD * 1.4826 is the consistent estimator of sigma for a normal.
    vx_std = float(np.median(np.abs(slow[:, 0] - vx)) * 1.4826)
    vy_std = float(np.median(np.abs(slow[:, 1] - vy)) * 1.4826)

    return FlowCalibrationResult(
        vx_px_s=vx,
        vy_px_s=vy,
        vx_std_px_s=vx_std,
        vy_std_px_s=vy_std,
        n_tracks=len(slow),
        method="fixed_vector_quantile",
        notes=(
            f"median of the slowest {quantile:.0%} of {len(velocities)} tracks "
            f"(speed cutoff {cutoff:.1f} px/s)"
        ),
    )


def calibrate_flow_map(
    tracks: list[Any],
    height: int,
    width: int,
    *,
    grid: int = 16,
    quantile: float = 0.35,
    min_tracks_per_cell: int = 3,
    smooth_sigma: float = 1.5,
) -> tuple[np.ndarray, FlowCalibrationResult]:
    """Fit a position-dependent flow field.

    Bins slow tracks onto a coarse grid, takes a median per cell, fills empty
    cells by nearest-neighbour, smooths, then resamples to full resolution.
    The coarse grid is deliberate: a per-pixel fit from a few hundred tracks
    would be mostly noise, and the underlying profile is smooth anyway.

    Returns the ``(H, W, 2)`` field and a summary.
    """
    velocities, positions = _track_velocities(tracks)
    if len(velocities) < grid:
        raise CalibrationError(
            f"only {len(velocities)} usable tracks for a {grid}x{grid} flow "
            "map; record more data or reduce the grid"
        )

    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    keep = speeds <= float(np.quantile(speeds, quantile))
    velocities, positions = velocities[keep], positions[keep]

    cell_h, cell_w = height / grid, width / grid
    coarse = np.full((grid, grid, 2), np.nan, dtype=np.float64)
    counts = np.zeros((grid, grid), dtype=int)

    rows = np.clip((positions[:, 1] / cell_h).astype(int), 0, grid - 1)
    cols = np.clip((positions[:, 0] / cell_w).astype(int), 0, grid - 1)
    for r in range(grid):
        for c in range(grid):
            sel = (rows == r) & (cols == c)
            counts[r, c] = int(sel.sum())
            if counts[r, c] >= min_tracks_per_cell:
                coarse[r, c, 0] = float(np.median(velocities[sel, 0]))
                coarse[r, c, 1] = float(np.median(velocities[sel, 1]))

    filled = int(np.isfinite(coarse[..., 0]).sum())
    if filled == 0:
        raise CalibrationError(
            "no grid cell had enough tracks; reduce grid or min_tracks_per_cell"
        )

    # Nearest-neighbour fill for empty cells, so smoothing has no holes.
    known = np.argwhere(np.isfinite(coarse[..., 0]))
    for r in range(grid):
        for c in range(grid):
            if not np.isfinite(coarse[r, c, 0]):
                d = np.hypot(known[:, 0] - r, known[:, 1] - c)
                nr, nc = known[int(np.argmin(d))]
                coarse[r, c] = coarse[nr, nc]

    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            for k in (0, 1):
                coarse[..., k] = gaussian_filter(coarse[..., k], smooth_sigma)
        except ImportError:
            logger.warning("scipy unavailable; skipping flow-map smoothing")

    # Bilinear resample to full resolution.
    yy = np.linspace(0, grid - 1, height)
    xx = np.linspace(0, grid - 1, width)
    field = np.empty((height, width, 2), dtype=np.float32)
    for k in (0, 1):
        rows_interp = np.empty((grid, width), dtype=np.float64)
        for r in range(grid):
            rows_interp[r] = np.interp(xx, np.arange(grid), coarse[r, :, k])
        for x in range(width):
            field[:, x, k] = np.interp(yy, np.arange(grid), rows_interp[:, x])

    summary = FlowCalibrationResult(
        vx_px_s=float(np.mean(field[..., 0])),
        vy_px_s=float(np.mean(field[..., 1])),
        vx_std_px_s=float(np.std(field[..., 0])),
        vy_std_px_s=float(np.std(field[..., 1])),
        n_tracks=int(len(velocities)),
        method="flow_map_grid",
        map_shape=(height, width),
        notes=(
            f"{grid}x{grid} grid, {filled}/{grid * grid} cells populated "
            f"directly, remainder filled by nearest neighbour"
        ),
    )
    return field, summary


def save_flow_map(field: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, field.astype(np.float32))
    logger.info("wrote flow map %s to %s", field.shape, path)


def load_flow_map(path: Path | str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"flow map not found: {path}")
    field = np.load(path)
    if field.ndim != 3 or field.shape[2] != 2:
        raise CalibrationError(
            f"flow map must have shape (H, W, 2), got {field.shape}"
        )
    return field.astype(np.float32)


def save_flow_calibration(result: FlowCalibrationResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
