"""Audit logging.

Every decision this device makes must be reconstructable afterwards from the
log alone: which sperm were counted, which qualified, why the others did not,
what ratio resulted, what command was issued, and when it actually fired.

The format is JSON Lines -- one self-describing record per line, appendable,
streamable, and readable with standard tools. A run directory contains:

* ``manifest.json``  -- configuration, versions, calibration state, git commit
* ``events.jsonl``   -- shots, decisions, commands, faults
* ``tracks.jsonl``   -- per-sperm records (optional; large)
* ``metrics.jsonl``  -- periodic runtime metrics

Nothing here is on the hot path. Records are written at shot rate (about 1 Hz),
not frame rate.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from .. import __version__
from ..constants import SCHEMA_VERSION

logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    """Best-effort git revision of the working tree, for traceability."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


class AuditLogger:
    """Writes the decision record for one run.

    Files are opened line-buffered and flushed after every record. That costs
    a syscall per shot, which is nothing at ~1 Hz, and means a power loss
    leaves a complete log up to the last decision rather than an empty buffer.
    """

    def __init__(
        self,
        directory: Path | str | None,
        *,
        run_name: str = "run",
        audit_tracks: bool = True,
        audit_track_points: bool = False,
    ) -> None:
        self.enabled = directory is not None
        self.audit_tracks = audit_tracks
        self.audit_track_points = audit_track_points
        self.run_dir: Path | None = None
        self._events: TextIO | None = None
        self._tracks: TextIO | None = None
        self._metrics: TextIO | None = None
        self.n_events = 0
        self.n_tracks = 0

        if not self.enabled:
            logger.warning(
                "audit logging is disabled; decisions will not be "
                "reconstructable after this run"
            )
            return

        base = Path(directory)  # type: ignore[arg-type]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = base / f"{stamp}-{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")
        if audit_tracks:
            self._tracks = (self.run_dir / "tracks.jsonl").open("a", encoding="utf-8")
        self._metrics = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        logger.info("audit log: %s", self.run_dir)

    # -------------------------------------------------------------- manifest

    def write_manifest(self, config_summary: dict[str, Any], **extra: Any) -> None:
        """Record everything needed to interpret the rest of the log."""
        if not self.enabled or self.run_dir is None:
            return
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package_version": __version__,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "config": config_summary,
            **extra,
        }
        path = self.run_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    # ---------------------------------------------------------------- events

    def _write(self, stream: TextIO | None, record: dict[str, Any]) -> None:
        if stream is None:
            return
        stream.write(json.dumps(record, ensure_ascii=False, default=str))
        stream.write("\n")
        stream.flush()

    def event(self, kind: str, **fields: Any) -> None:
        """Append one event record."""
        if not self.enabled:
            return
        self._write(
            self._events,
            {"t": time.monotonic(), "kind": kind, **fields},
        )
        self.n_events += 1

    def shot_decided(self, shot: Any, decision: Any) -> None:
        """The central record: one shot, its members, and its verdict."""
        self.event(
            "shot_decided",
            shot=shot.to_json_dict(),
            decision=decision.to_json_dict(),
        )

    def command_dispatched(self, command: Any) -> None:
        self.event("command", command=command.to_json_dict())

    def fault(self, source: str, message: str, **fields: Any) -> None:
        self.event("fault", source=source, message=message, **fields)

    def track(self, track: Any) -> None:
        """Append one per-sperm record."""
        if not self.enabled or not self.audit_tracks:
            return
        self._write(
            self._tracks,
            track.to_json_dict(include_points=self.audit_track_points),
        )
        self.n_tracks += 1

    def metrics(self, snapshot: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._write(self._metrics, {"t": time.monotonic(), **snapshot})

    # ------------------------------------------------------------- lifecycle

    def close(self, summary: dict[str, Any] | None = None) -> None:
        """Flush, write the summary, and release the files. Idempotent."""
        if not self.enabled:
            return
        if summary is not None and self.run_dir is not None:
            with (self.run_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=str)
        for stream in (self._events, self._tracks, self._metrics):
            if stream is not None:
                try:
                    stream.flush()
                    stream.close()
                except Exception:
                    logger.exception("failed to close an audit stream")
        self._events = self._tracks = self._metrics = None
        self.enabled = False

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: Path | str) -> list[dict[str, Any]]:
    """Read an ``events.jsonl`` back. Used by the replay-determinism test.

    Malformed trailing lines are skipped rather than fatal: a log truncated by
    a power loss should still be readable up to the cut.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed audit line %d in %s", line_no, path)
    return records
