"""The experiment record: what makes a number reproducible.

``experiment.json`` is a mandatory output of every script in this package, not
an optional extra. A metrics file on its own says *what* was measured; it does
not say which commit produced it, which configuration was in force, which
dataset and split it ran on, which library versions were linked, or which
machine it ran on -- and a result missing any of those cannot be reproduced or
challenged.

What is captured, and why each item is not optional:

``git``
    Commit, branch and whether the tree was dirty. A dirty tree does not
    invalidate a result, but a result from a dirty tree that *claims* a commit
    is worse than one that admits it.
``config``
    The fully-resolved :class:`~sperm_sorting.config.AppConfig`, after YAML and
    every ``--set`` override. Not the file path -- the file may have changed
    since -- and not the overrides alone, which are meaningless without the
    base.
``packages``
    Versions of every library whose numerics can move a metric.
``dataset``
    Name, licence, split sizes and per-aspect positive counts. The licence
    belongs here because public research data carries terms, and a weights file
    whose provenance is recorded as ``public-research-baseline`` needs a record
    saying which public data that was.
``seed`` / ``determinism``
    The dict returned by :func:`training.common.seeding.seed_everything`, which
    records the knobs actually set rather than the ones requested.
``hardware`` / ``timing``
    Device, thread count, wall-clock start/end. Needed to interpret any
    latency figure at all.
``metrics``
    The final numbers, exactly as reported.

The record is written incrementally: :meth:`ExperimentRecord.save` is safe to
call at any point and is called on the failure path too, so a crashed run still
leaves a record saying how far it got and why it stopped.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.common.args import dump_json
from training.common.logging_utils import _jsonable

__all__ = ["ExperimentRecord", "collect_git_info", "collect_package_versions"]

#: Bumped when the record's schema changes.
EXPERIMENT_SCHEMA_VERSION: str = "1"

#: Libraries whose version can move a metric. Recorded even when absent, with
#: the value ``"not installed"``, because "numpy 2.4 vs 1.26" and "sklearn was
#: missing so ROC-AUC came back NaN" are both things a reader needs to see.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "sklearn",
    "matplotlib",
    "tensorboard",
    "cv2",
    "PIL",
    "pydantic",
    "yaml",
)


def collect_git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirty flag for ``repo_root``.

    Every failure mode is reported rather than raised: training in an exported
    tarball with no ``.git`` is legitimate, and a missing git binary must not
    be able to fail a training run. What is *not* acceptable is silently
    recording a commit that is not the one that ran, so an unavailable commit
    is recorded as unavailable.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    info: dict[str, Any] = {"repo_root": str(root)}

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            info.setdefault("error", f"{type(exc).__name__}: {exc}")
            return None
        if out.returncode != 0:
            info.setdefault("error", (out.stderr or "git returned non-zero").strip())
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    info["commit"] = commit
    info["commit_short"] = commit[:12] if commit else None
    info["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    info["dirty"] = bool(status) if status is not None else None
    if status:
        # Truncated: the point is to flag that the tree was modified and give a
        # hint which files, not to embed a diff in a metrics file.
        info["dirty_files"] = status.splitlines()[:20]
    info["describe"] = run("describe", "--always", "--dirty", "--tags")
    return info


def collect_package_versions() -> dict[str, str]:
    """Version string for each tracked package, or ``"not installed"``."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        try:
            module = __import__(name)
        except Exception:  # ImportError, or a broken optional dependency
            versions[name] = "not installed"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    try:
        import training

        versions["training_harness"] = training.__version__
    except Exception:  # pragma: no cover - only if the package is broken
        versions["training_harness"] = "unknown"
    try:
        import sperm_sorting

        versions["sperm_sorting"] = str(getattr(sperm_sorting, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        versions["sperm_sorting"] = "unknown"
    return versions


@dataclass
class ExperimentRecord:
    """One run's reproducibility record.

    Parameters
    ----------
    script
        Which entry point produced this, e.g. ``"train_morphology"``.
    out_dir
        Where ``experiment.json`` is written.
    """

    script: str
    out_dir: Path
    filename: str = "experiment.json"

    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    started_utc: str = ""
    finished_utc: str = ""
    duration_s: float | None = None
    status: str = "running"
    failure: str = ""

    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    args: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    packages: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    determinism: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    _monotonic_start: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------ start

    def start(self) -> ExperimentRecord:
        """Stamp the start time and collect the static environment facts."""
        import time

        self.started_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self._monotonic_start = time.monotonic()
        self.git = collect_git_info()
        self.packages = collect_package_versions()
        self.status = "running"
        return self

    # ---------------------------------------------------------------- populate

    def set_config(self, cfg: Any) -> None:
        """Record the fully-resolved configuration and its audit summary."""
        self.config = _jsonable(cfg.model_dump(mode="json"))
        self.config_summary = _jsonable(cfg.summary())

    def set_dataset(
        self,
        *,
        name: str,
        licence: str,
        splits: dict[str, int],
        source: str = "",
        **extra: Any,
    ) -> None:
        """Record dataset identity, licence and split sizes.

        ``licence`` is required rather than optional. Weights stamped
        ``public-research-baseline`` inherit the terms of whatever produced
        them, and a record that omits the licence cannot answer the only
        question that matters when those weights leave the building.
        """
        self.dataset = {
            "name": name,
            "licence": licence,
            "source": source,
            "splits": dict(splits),
            **_jsonable(extra),
        }

    def note(self, text: str) -> None:
        """Attach a caveat to the record.

        Used for things a reader must not miss, such as a validation split with
        seven positives for an aspect. Notes are part of the result, not
        commentary on it.
        """
        self.notes.append(str(text))

    def artifact(self, key: str, path: str | Path) -> None:
        """Register an output file so the record lists everything produced."""
        self.artifacts[key] = str(path)

    # ------------------------------------------------------------------ finish

    def finish(self, status: str = "completed", failure: str = "") -> ExperimentRecord:
        """Stamp the end time and outcome."""
        import time

        self.finished_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        if self._monotonic_start:
            self.duration_s = round(time.monotonic() - self._monotonic_start, 3)
        self.status = status
        self.failure = failure
        return self

    # ------------------------------------------------------------------- save

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "script": self.script,
            "status": self.status,
            "failure": self.failure,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "duration_s": self.duration_s,
            "argv": self.argv,
            "args": self.args,
            "git": self.git,
            "packages": self.packages,
            "hardware": self.hardware,
            "determinism": self.determinism,
            "dataset": self.dataset,
            "model": self.model,
            "training": self.training,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "config_summary": self.config_summary,
            "config": self.config,
        }
        return _jsonable(payload)

    def save(self) -> Path:
        """Write ``experiment.json``. Safe to call repeatedly."""
        path = Path(self.out_dir) / self.filename
        dump_json(path, self.to_json_dict())
        return path

    # ------------------------------------------------------------ context use

    def __enter__(self) -> ExperimentRecord:
        return self.start()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        """Always write a record, including on the failure path.

        A run that crashed at epoch 12 still produced twelve epochs of evidence
        and a reason for stopping. Writing nothing would throw both away.
        """
        if exc_type is None:
            self.finish("completed")
        elif exc_type is KeyboardInterrupt:
            self.finish("interrupted", "KeyboardInterrupt")
        else:
            self.finish("failed", f"{exc_type.__name__}: {exc}")
        self.save()
