"""Tracker construction from configuration.

One place that maps ``TrackingConfig.algorithm`` onto a class, so nothing else
in the pipeline imports a concrete tracker. Swapping the algorithm is then a
config change and an audit-log line, not a code change.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import TrackingConfig
from ..errors import ConfigurationError
from .base import Tracker
from .botsort import BoTSortTracker
from .bytetrack import ByteTracker
from .ocsort import OCSortTracker

#: Values are typed as factories rather than as ``type[Tracker]`` because the
#: :class:`Tracker` ABC deliberately says nothing about construction; what this
#: registry actually promises is "call it with a TrackingConfig, get a Tracker".
_REGISTRY: dict[str, Callable[[TrackingConfig], Tracker]] = {
    ByteTracker.name: ByteTracker,
    OCSortTracker.name: OCSortTracker,
    BoTSortTracker.name: BoTSortTracker,
}


def available_trackers() -> tuple[str, ...]:
    """Names accepted by :func:`build_tracker`, in registration order."""
    return tuple(_REGISTRY)


def build_tracker(cfg: TrackingConfig) -> Tracker:
    """Instantiate the tracker named by ``cfg.algorithm``.

    Raises :class:`~sperm_sorting.errors.ConfigurationError` for an unknown
    name. ``TrackingConfig.algorithm`` is a ``Literal`` so a typo is normally
    caught at config-load time; this check covers a config built in code,
    where the type is only a hint.
    """
    algorithm = str(cfg.algorithm)
    tracker_class = _REGISTRY.get(algorithm)
    if tracker_class is None:
        raise ConfigurationError(
            f"unknown tracking.algorithm {algorithm!r}; "
            f"expected one of {', '.join(available_trackers())}"
        )
    return tracker_class(cfg)
