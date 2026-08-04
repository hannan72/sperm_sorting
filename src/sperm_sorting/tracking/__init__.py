"""Multi-object tracking.

Three interchangeable trackers behind one interface. All of them share the
identity machinery that the rest of the pipeline depends on: an ID is issued
once and never reused, one physical sperm owns one
:class:`~sperm_sorting.schemas.track.TrackRecord` that grows in place, and a
position produced by the motion model rather than by the detector is flagged
``observed=False`` so no velocity is ever computed from a guess.

Choosing between them:

* :class:`ByteTracker` -- the default. Its low-score second pass is what keeps
  dim and partly-occluded sperm on the same ID.
* :class:`OCSortTracker` -- better through longer occlusions and direction
  changes, because association and re-update are driven by real observations
  rather than by the filter's own extrapolation.
* :class:`BoTSortTracker` -- ByteTrack plus camera-motion compensation and
  appearance fusion, both off by default. Read its module docstring before
  enabling camera-motion compensation: on a rigidly-mounted camera the global
  motion it removes is the sample's fluid flow, which belongs to the flow
  correction stage, not to the tracker.
"""

from __future__ import annotations

from .base import Tracker
from .botsort import BoTSortTracker
from .bytetrack import ByteTracker
from .factory import available_trackers, build_tracker
from .ocsort import OCSortTracker

__all__ = [
    "BoTSortTracker",
    "ByteTracker",
    "OCSortTracker",
    "Tracker",
    "available_trackers",
    "build_tracker",
]
