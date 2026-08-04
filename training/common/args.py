"""The argument pattern every training and evaluation script shares.

The contract is the one already established by
:mod:`sperm_sorting.cli`: a ``--config`` YAML file plus repeatable
``--set key.path=value`` overrides, resolved through
:func:`sperm_sorting.config.load_config` so that a training run and a runtime
run are validated by exactly the same Pydantic models. Anything a script needs
that is *not* part of the runtime configuration (epochs, learning rate, output
directory) is a plain argparse flag, because putting it in ``AppConfig`` would
mean extending a fixed contract.

Why argparse rather than typer, which the runtime CLI uses: these are six
independent scripts, not one command tree, and argparse gives them a
``--help`` that works without importing torch. That matters -- ``--help`` is
the cheapest smoke test there is, and it should not take four seconds and a CUDA
probe to answer.

The device string is resolved late and loudly. ``--device auto`` picks CUDA
when it is genuinely available and CPU otherwise; ``--device cuda`` on a
machine with no CUDA is an *error*, not a silent fallback, because a run that
was meant to take twenty minutes on a GPU and instead takes nine hours on a CPU
should say so immediately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sperm_sorting.config import AppConfig, load_config
from sperm_sorting.errors import SpermSortingError

__all__ = [
    "CommonArgs",
    "add_common_arguments",
    "build_parser",
    "describe_device",
    "resolve_config",
    "resolve_device",
]


@dataclass(slots=True)
class CommonArgs:
    """The resolved common arguments, kept together for the experiment record."""

    config_path: Path | None
    overrides: list[str]
    out_dir: Path
    resume: Path | None
    device: str
    seed: int
    deterministic: bool
    #: The fully-resolved configuration, already validated.
    cfg: AppConfig = field(repr=False, default_factory=AppConfig)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialisable view, for ``experiment.json``."""
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "overrides": list(self.overrides),
            "out_dir": str(self.out_dir),
            "resume": str(self.resume) if self.resume else None,
            "device": self.device,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }


def build_parser(description: str, *, epilog: str = "") -> argparse.ArgumentParser:
    """Create a parser with the shared options already installed."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Install ``--config/--set/--out/--resume/--device/--seed`` on ``parser``.

    ``--set`` is ``append``-based rather than ``nargs='+'`` so that a value
    containing spaces cannot swallow the next flag, and so that the order of
    overrides is preserved -- later overrides win, which is only meaningful if
    the order survives parsing.
    """
    group = parser.add_argument_group("configuration")
    group.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="YAML configuration file. Omit for the built-in defaults.",
    )
    group.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help=(
            "Override one configuration field, e.g. "
            "-s morphology.backbone=simplecnn. Repeatable; later wins."
        ),
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "-o",
        "--out",
        dest="out_dir",
        type=Path,
        default=Path("runs/training"),
        metavar="DIR",
        help="Directory for checkpoints, metrics, plots and experiment.json.",
    )
    run.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CKPT",
        help=(
            "Resume from a training checkpoint (normally <out>/last.pt). "
            "Restores model, optimizer, scheduler, AMP scaler, epoch counter "
            "and best-metric state."
        ),
    )
    run.add_argument(
        "--device",
        default="auto",
        metavar="DEV",
        help=(
            "auto | cpu | cuda | cuda:N. 'auto' selects CUDA when available. "
            "Naming a CUDA device that does not exist is an error, not a "
            "silent fallback."
        ),
    )
    run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override run.seed from the configuration.",
    )
    run.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        default=None,
        help=(
            "Allow cuDNN autotuning and non-deterministic kernels. Faster on "
            "CUDA, but two runs with the same seed may then differ."
        ),
    )


def resolve_config(args: argparse.Namespace) -> CommonArgs:
    """Turn parsed arguments into a validated :class:`CommonArgs`.

    Configuration errors are raised as :class:`SpermSortingError` so the caller
    can print one clean line instead of a Pydantic traceback; an operator
    mistyping ``morphology.backbon`` should be told which key is wrong, not
    shown twenty frames of validation internals.
    """
    overrides = list(getattr(args, "overrides", None) or [])
    if getattr(args, "seed", None) is not None:
        overrides.append(f"run.seed={int(args.seed)}")
    if getattr(args, "deterministic", None) is False:
        overrides.append("run.deterministic=false")

    cfg = load_config(args.config, overrides)

    out_dir = Path(args.out_dir).expanduser().resolve()
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume is not None and not resume.exists():
        raise SpermSortingError(f"--resume checkpoint does not exist: {resume}")

    return CommonArgs(
        config_path=Path(args.config).resolve() if args.config else None,
        overrides=overrides,
        out_dir=out_dir,
        resume=resume,
        device=str(args.device),
        seed=int(cfg.run.seed),
        deterministic=bool(cfg.run.deterministic),
        cfg=cfg,
    )


def resolve_device(spec: str) -> Any:
    """Turn a device string into a ``torch.device`` that exists.

    Unlike :func:`sperm_sorting.detection.torch_base.resolve_device`, which
    falls back to CPU with a warning because a deployed pipeline must keep
    running, this raises. A training script has no reason to silently spend a
    day doing what was meant to take an hour.
    """
    import torch

    text = str(spec).strip().lower()
    if text in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise SpermSortingError(f"--device '{spec}' is not a valid torch device: {exc}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SpermSortingError(
                f"--device '{spec}' was requested but torch reports no CUDA device "
                f"(torch {torch.__version__}). Use --device cpu, or --device auto "
                "to let the script choose."
            )
        index = 0 if device.index is None else int(device.index)
        count = torch.cuda.device_count()
        if index >= count:
            raise SpermSortingError(
                f"--device '{spec}' names CUDA device {index}, but only {count} "
                "are visible."
            )
    return device


def describe_device(device: Any) -> dict[str, Any]:
    """Hardware facts for the experiment record.

    Captured at run time rather than assumed, because "it was slow" and "it ran
    on the wrong device" look identical in a log that does not say which device
    was used.
    """
    import platform

    import torch

    info: dict[str, Any] = {
        "device": str(device),
        "type": device.type,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        info["cpu_count"] = len(__import__("os").sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        import os as _os

        info["cpu_count"] = _os.cpu_count()

    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        props = torch.cuda.get_device_properties(index)
        info["cuda_device_name"] = props.name
        info["cuda_capability"] = f"{props.major}.{props.minor}"
        info["cuda_total_memory_mb"] = round(props.total_memory / 1024**2, 1)
        info["cuda_version"] = str(torch.version.cuda)
    return info


def dump_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON, atomically.

    Atomic because an interrupted metrics write leaves a truncated file that
    ``json.load`` rejects, and a half-written result is indistinguishable from
    a crashed run when someone reads it a month later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
