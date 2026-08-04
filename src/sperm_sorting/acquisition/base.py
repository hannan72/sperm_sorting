"""Frame-source interface.

Three sources -- a live Basler camera, a recorded video, and the procedural
simulator -- feed the *identical* downstream graph. Nothing after this module
knows or cares which one is running. That is what makes replay a genuine test
of the production path rather than a parallel implementation of it.

The one thing a source must be honest about is time. Motion analysis divides
by elapsed time, so a source that invents plausible-looking timestamps
produces plausible-looking velocities that are wrong. Every packet therefore
carries a :class:`TimestampSource` saying where its timestamp came from, and
drops are reported explicitly rather than left to be inferred from a gap in
frame IDs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from ..schemas.enums import SourceKind
from ..schemas.frame import FramePacket


class FrameSource(ABC):
    """Produces :class:`FramePacket`s."""

    kind: SourceKind

    def __init__(self) -> None:
        self._open = False
        self._frame_id = 0
        self._session_id = 0
        self.n_dropped = 0

    # -------------------------------------------------------------- lifecycle

    @abstractmethod
    def open(self) -> None:
        """Acquire the source. Raises :class:`CameraError` on failure."""

    @abstractmethod
    def close(self) -> None:
        """Release the source. Idempotent; must not raise."""

    @property
    def is_open(self) -> bool:
        return self._open

    # ----------------------------------------------------------------- frames

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Return the next frame, or ``None`` when the source is exhausted.

        A live camera blocks up to its configured timeout and raises on a
        disconnect; a file source returns ``None`` at end of stream.
        """

    def frames(self, max_frames: int | None = None) -> Iterator[FramePacket]:
        """Iterate until exhaustion, or until ``max_frames`` have been read."""
        count = 0
        while max_frames is None or count < max_frames:
            packet = self.read()
            if packet is None:
                return
            count += 1
            yield packet

    # ------------------------------------------------------------- metadata

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Source metadata for the audit manifest."""

    @property
    def nominal_fps(self) -> float | None:
        """Expected frame rate, if the source knows it.

        Nominal only. Motion analysis must use the per-frame timestamps, since
        the achieved rate differs from the nominal one under load and the
        difference goes straight into every velocity.
        """
        return None

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FramePacket]:
        return self.frames()
