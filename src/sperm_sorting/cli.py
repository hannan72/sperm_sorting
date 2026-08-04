"""Command-line interface.

Every entry point takes the same ``--config`` plus ``--set key.path=value``
pair, so any field in the configuration tree can be changed without editing
YAML, and the exact invocation that produced a run is recoverable from the
shell history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .config import AppConfig, load_config
from .errors import SpermSortingError

app = typer.Typer(
    name="sperm-sorting",
    help=(
        "Real-time AI-guided sperm analysis and conditional magnetic sorting "
        "(research prototype). Not a medical device; not clinically validated. "
        "This software analyses visible phenotype only -- it does not measure "
        "DNA integrity, apoptosis, Annexin V binding or fertility potential."
    ),
    add_completion=False,
    no_args_is_help=True,
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML configuration file."),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a config field, e.g. -s decision.threshold=0.6 "
        "(repeatable).",
    ),
]


def _load(config: Path | None, overrides: list[str] | None) -> AppConfig:
    try:
        return load_config(config, overrides or [])
    except SpermSortingError as exc:
        typer.secho(f"configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"sperm-sorting-ai {__version__}")


@app.command()
def run(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", "-m", help="live | replay | synthetic."),
    ] = None,
    frames: Annotated[
        int | None, typer.Option("--frames", "-n", help="Stop after N frames.")
    ] = None,
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Replay this file (implies --mode replay)."),
    ] = None,
) -> None:
    """Run the pipeline end to end."""
    overrides = list(set_ or [])
    if video is not None:
        overrides += [
            "acquisition.kind=video",
            f"acquisition.video.path={video}",
            "run.mode=replay",
            "runtime.backpressure=block",
        ]
    if mode is not None:
        overrides.append(f"run.mode={mode}")
    if frames is not None:
        overrides.append(f"run.max_frames={frames}")

    cfg = _load(config, overrides)

    from .app import Application

    application = Application(cfg)
    try:
        application.setup()
        application.install_signal_handlers()
        application.run()
    except SpermSortingError as exc:
        typer.secho(f"run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    finally:
        application.close()

    typer.echo(application.format_report())


@app.command("show-config")
def show_config(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit JSON instead of YAML.")
    ] = False,
) -> None:
    """Print the fully-resolved configuration."""
    cfg = _load(config, set_)
    if as_json:
        typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))
    else:
        typer.echo(cfg.to_yaml())


@app.command()
def feasibility(
    config: ConfigOpt = None,
    set_: SetOpt = None,
    chamber_depth_um: Annotated[
        float,
        typer.Option(help="Optical section thickness contributing countable sperm."),
    ] = 20.0,
) -> None:
    """Check whether the optics and flow can deliver the configured shot rate.

    Answers the question that otherwise only surfaces after a long run of
    INDETERMINATE shots: can this field of view, at this flow speed, actually
    present 20-30 trackable sperm per second?
    """
    cfg = _load(config, set_)
    from .shots.feasibility import assess_feasibility

    report = assess_feasibility(cfg, chamber_depth_um=chamber_depth_um)
    typer.echo(report.format_report())
    raise typer.Exit(0 if report.feasible else 1)


@app.command()
def doctor(
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Report what is installed, what is calibrated, and what is missing."""
    cfg = _load(config, set_)
    from .backends.runtime_backend import describe_environment

    env = describe_environment()
    typer.echo("environment")
    typer.echo(f"  python            : {sys.version.split()[0]}")
    typer.echo(f"  torch             : {env['torch'] or 'not installed'}")
    typer.echo(f"  cuda              : {env['cuda_available']} {env['cuda_devices']}")
    typer.echo(f"  onnxruntime       : {env['onnxruntime'] or 'not installed'}")
    typer.echo(f"  onnx providers    : {env['onnx_providers']}")
    typer.echo(f"  tensorrt          : {env['tensorrt'] or 'not installed'}")

    try:
        from pypylon import pylon  # noqa: F401

        pylon_status = "installed"
    except ImportError:
        pylon_status = "not installed (live camera unavailable)"
    typer.echo(f"  pypylon           : {pylon_status}")

    typer.echo("")
    typer.echo("calibration")
    optical = cfg.calibration.optical
    if optical.calibrated:
        typer.secho(
            f"  optical           : {optical.um_per_px:.5f} um/px "
            f"(nominal {optical.nominal_um_per_px:.5f})",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"  optical           : NOT CALIBRATED (nominal would be "
            f"{optical.nominal_um_per_px:.5f} um/px). Velocities stay in "
            "pixels/second and motility cannot be graded.",
            fg=typer.colors.YELLOW,
        )
    if cfg.scheduling.calibrated:
        typer.secho(
            f"  scheduling        : transport {cfg.scheduling.transport_delay_ms:.1f} "
            f"+/- {cfg.scheduling.transport_delay_std_ms:.1f} ms",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            "  scheduling        : NOT CALIBRATED. The scheduler will refuse "
            "to arm, so no field command will be driven.",
            fg=typer.colors.YELLOW,
        )

    flow = cfg.motion.flow_correction
    typer.echo(f"  flow correction   : {flow.mode}")

    typer.echo("")
    typer.echo("models")
    typer.echo(f"  detector          : {cfg.detection.architecture} "
               f"weights={cfg.detection.weights or 'none (untrained)'}")
    typer.echo(f"  tracker           : {cfg.tracking.algorithm}")
    typer.echo(f"  morphology        : {cfg.morphology.backbone} "
               f"weights={cfg.morphology.weights or 'none (untrained)'}")
    typer.echo(f"  weights provenance: {cfg.morphology.weights_provenance}")

    typer.echo("")
    typer.echo("decision rule")
    typer.echo(f"  threshold         : ratio > {cfg.decision.threshold} accepts "
               "(exactly at threshold rejects)")
    typer.echo(f"  minimum trackable : {cfg.decision.minimum_trackable_sperm}")
    typer.echo(f"  shot size         : {cfg.shots.minimum_trackable_sperm}-"
               f"{cfg.shots.maximum_trackable_sperm}, target "
               f"{cfg.shots.target_trackable_sperm}, max "
               f"{cfg.shots.maximum_shot_duration_seconds}s")

    from .shots.feasibility import assess_feasibility

    report = assess_feasibility(cfg)
    typer.echo("")
    typer.echo(report.format_report())


@app.command("generate-data")
def generate_data(
    n: Annotated[int, typer.Option("--n", help="Number of samples.")] = 20000,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("data"),
    image_size: Annotated[int, typer.Option(help="64 or 128.")] = 128,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 1234,
) -> None:
    """Build a synthetic morphology + motility training set."""
    from .simulator.generate import build_dataset

    build_dataset(n=n, out_dir=out, image_size=image_size, seed=seed)
    typer.secho(f"wrote {n} samples to {out}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
