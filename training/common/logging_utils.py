"""Metric logging: TensorBoard, JSON Lines, and a console table.

Three sinks because they answer three different questions and none of them
substitutes for the others:

* **TensorBoard** answers "is this run going anywhere" while it is still
  running. It is optional -- ``tensorboard`` is an extra, and a missing extra
  must not be able to kill a training run three hours in -- so
  :class:`TensorBoardWriter` degrades to a no-op that records *why* it is a
  no-op.
* **JSON Lines** answers "what exactly happened on epoch 37" afterwards, from a
  script. One flat JSON object per line, appended and flushed per epoch, so a
  killed run still leaves every epoch it completed.
* **The console table** answers "should I keep watching this" for a human.
  Fixed-width and column-stable, so two runs can be diffed by eye.

TensorBoard is imported lazily inside the constructor. Importing it at module
scope costs a second or two and pulls in a large dependency tree even for
``--help``, which is the one command that has to stay instant.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

__all__ = ["JsonlLogger", "TensorBoardWriter", "console_table", "format_duration"]


class TensorBoardWriter:
    """Thin wrapper over ``torch.utils.tensorboard.SummaryWriter``.

    Every method is safe to call whether or not TensorBoard is installed and
    whether or not logging was requested. :attr:`enabled` and
    :attr:`unavailable_reason` say which, so the experiment record can state
    "TensorBoard logging was off because the package is missing" rather than
    leaving a reader to wonder where the event files went.

    Non-finite values are dropped rather than written. A NaN in a scalar series
    makes the whole TensorBoard chart render blank from that point on, which
    hides the very epochs a reader needs to see -- and NaN is a normal outcome
    here for an aspect with no positives in a validation fold.
    """

    def __init__(self, log_dir: str | Path | None, *, enabled: bool = True) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.enabled = False
        self.unavailable_reason = ""
        self._writer: Any = None
        self.n_dropped_nonfinite = 0

        if not enabled:
            self.unavailable_reason = "disabled by the caller"
            return
        if self.log_dir is None:
            self.unavailable_reason = "no log directory was given"
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # ImportError, or a protobuf/setuptools clash
            self.unavailable_reason = f"tensorboard unavailable: {type(exc).__name__}: {exc}"
            return

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except Exception as exc:  # pragma: no cover - filesystem dependent
            self.unavailable_reason = f"could not open a SummaryWriter: {exc}"

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Log one scalar, skipping non-finite values."""
        if not self.enabled:
            return
        numeric = float(value)
        if not math.isfinite(numeric):
            self.n_dropped_nonfinite += 1
            return
        self._writer.add_scalar(tag, numeric, int(step))

    def add_scalars(self, prefix: str, values: Mapping[str, float], step: int) -> None:
        """Log a dict of scalars under ``prefix/key``.

        Separate ``add_scalar`` calls rather than ``add_scalars``: the plural
        form writes an extra event-file directory per tag group, which makes a
        run directory with four aspects x six metrics unpleasant to move
        around, and it cannot be overlaid with a scalar from another run.
        """
        for key, value in values.items():
            self.add_scalar(f"{prefix}/{key}", value, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        """Log a block of text (the resolved config, the metrics table)."""
        if not self.enabled:
            return
        self._writer.add_text(tag, text, int(step))

    def flush(self) -> None:
        if self.enabled:
            self._writer.flush()

    def close(self) -> None:
        """Close the writer. Safe to call more than once."""
        if self.enabled and self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
            self.enabled = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "unavailable_reason": self.unavailable_reason,
            "dropped_nonfinite": self.n_dropped_nonfinite,
        }

    def __enter__(self) -> TensorBoardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JsonlLogger:
    """Append-only JSON Lines metric log.

    Opened in append mode so a ``--resume`` extends the previous run's log
    instead of truncating it; the epoch number in each record is what
    distinguishes the segments. Flushed after every record because the whole
    point is to survive a run that gets killed.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.n_records = 0
        self._handle: TextIO | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        """Write one record. Non-JSON-safe values are stringified, not dropped."""
        if not self.enabled or self._handle is None:
            return
        self._handle.write(json.dumps(_jsonable(record), sort_keys=True, default=str) + "\n")
        self._handle.flush()
        self.n_records += 1

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.enabled = False

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    """Recursively convert to something ``json.dumps`` accepts.

    NaN and infinity become ``None``: strict JSON has no literal for either,
    and ``json.dumps`` emitting bare ``NaN`` produces a file that most other
    parsers reject. ``None`` is the honest encoding of "this metric was not
    defined on this split".
    """
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def console_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    title: str = "",
    float_format: str = "{:.4f}",
    index_column: str | None = None,
    footer: str = "",
) -> str:
    """Render rows as a fixed-width table.

    Columns are sized to their widest cell, numbers are right-aligned and text
    is left-aligned, and undefined values render as ``n/a`` rather than
    ``nan``. The distinction matters: ``nan`` reads like a bug, ``n/a`` reads
    like "this metric is not defined on this split", which is what it means
    when an aspect has no positive examples.
    """
    if not rows:
        return f"{title}\n(no rows)" if title else "(no rows)"

    headers = ([index_column] if index_column else []) + list(columns)
    widths: dict[str, int] = {h: len(h) for h in headers}
    rendered: list[dict[str, str]] = []

    for row in rows:
        cells: dict[str, str] = {}
        for header in headers:
            cells[header] = _format_cell(row.get(header), float_format)
            widths[header] = max(widths[header], len(cells[header]))
        rendered.append(cells)

    def line(cells: Mapping[str, str]) -> str:
        parts = []
        for i, header in enumerate(headers):
            text = cells.get(header, "")
            parts.append(text.ljust(widths[header]) if i == 0 and index_column else text.rjust(widths[header]))
        return "  ".join(parts)

    header_line = line({h: h for h in headers})
    rule = "-" * len(header_line)
    out: list[str] = []
    if title:
        out += [title, rule]
    out += [header_line, rule]
    out += [line(cells) for cells in rendered]
    out.append(rule)
    if footer:
        out.append(footer)
    return "\n".join(out)


def _format_cell(value: Any, float_format: str) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        # Integral floats (counts that came back as float from numpy) read
        # better without four zeros after the point.
        if value == int(value) and abs(value) < 1e9:
            return str(int(value))
        return float_format.format(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def format_duration(seconds: float) -> str:
    """``1h 03m 07s`` style duration, for the end-of-run line."""
    total = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def print_block(lines: Iterable[str], stream: TextIO | None = None) -> None:
    """Print an iterable of lines to ``stream`` (default stdout), flushed.

    Flushed because these scripts are usually watched through a pipe or a CI
    log, where a buffered stdout means the table appears minutes after the work
    that produced it.
    """
    handle = stream or sys.stdout
    for text in lines:
        handle.write(text + "\n")
    handle.flush()
