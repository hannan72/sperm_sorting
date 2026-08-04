"""VISEM-Tracking-graphs: graph representations of VISEM-Tracking. **Optional.**

Source: Hugging Face ``SimulaMet-HOST/visem-tracking-graphs``. Data CC BY 4.0,
generator code MIT. Derived from VISEM-Tracking, whose attribution requirement
it therefore also carries.

Layout and content
------------------
Five spatial thresholds -- ``spatial_threshold_{0.1,0.2,0.3,0.4,0.5}/`` -- and,
per video, ``frame_graphs/frame_graph_{i}.graphml`` plus one
``video_graph.graphml``. Nodes are keyed by ``sperm_id`` and carry
``frame_number``, ``class_name`` (the YOLO class index, as a *string*),
``x_center``, ``y_center``, ``width`` and ``height``, all YOLO-normalised to
``[0, 1]``. Edges are either spatial (attribute ``weight`` = Euclidean distance
between normalised centres, present when that distance is below the threshold)
or temporal (``edge_type="temporal"``). Frame graphs are undirected; the video
graph is directed.

The upstream defect this module works around
--------------------------------------------
``video_graph.graphml`` keys its nodes by ``sperm_id`` **alone**, not by
``(sperm_id, frame)``. Since a sperm appears in many frames, every one of its
per-frame nodes collapses onto a single node, and the loop that adds temporal
edges then connects that node to itself: the released video graphs contain
self-loops and have lost the temporal structure they exist to represent. A
node's ``frame_number`` attribute ends up holding whichever frame was written
last.

The per-frame graphs are unaffected -- within one frame each ``sperm_id``
appears once, so the key is unique there -- so this adapter treats the per-frame
graphs as the source of truth and offers
:meth:`VisemGraphsAdapter.regenerate_video_graph`, which rebuilds the video-level
graph from them using ``(video_id, frame_id, track_id)`` as the node identity.
:meth:`VisemGraphsAdapter.inspect_upstream_video_graph` measures the defect on
your copy rather than asking you to take this paragraph on trust.

Optional by design
------------------
networkx is not a dependency of this project. Importing this module without it
works; constructing the adapter raises an ``ImportError`` that names the install
command. The MVP does not depend on graph representations, and nothing in
:mod:`sperm_sorting` imports this module.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from sperm_sorting.errors import DatasetValidationError

from ..validators.integrity import CheckStatus, ValidationReport, check_non_empty
from .base import CaptureConditions, DatasetAdapter, DatasetInfo

__all__ = ["SPATIAL_THRESHOLDS", "VisemGraphsAdapter", "node_key"]

#: Spatial thresholds published upstream, as they appear in directory names.
SPATIAL_THRESHOLDS: Final[tuple[str, ...]] = ("0.1", "0.2", "0.3", "0.4", "0.5")

_THRESHOLD_DIR_RE: Final[re.Pattern[str]] = re.compile(r"spatial_threshold_([0-9.]+)$")
_FRAME_GRAPH_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)$")
_VIDEO_ID_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)")


def node_key(video_id: int, frame_id: int, track_id: int | str) -> str:
    """Canonical node identity for a rebuilt video graph.

    ``(video_id, frame_id, track_id)`` -- the triple the upstream generator
    should have used. Returned as a string rather than a tuple because GraphML
    node ids must be scalars; the three components are also written back as node
    attributes so nothing has to parse this string to recover them.
    """
    return f"v{int(video_id)}_f{int(frame_id)}_t{track_id}"


class VisemGraphsAdapter(DatasetAdapter):
    """Reader for the per-frame GraphML files, plus a correct video-graph rebuild.

    Parameters
    ----------
    root
        Directory containing the ``spatial_threshold_*`` folders (i.e. the
        downloaded Hugging Face dataset root).
    threshold
        Which spatial threshold to read. Defaults to ``"0.3"``, the middle of
        the published range; there is no upstream recommendation, and the choice
        materially changes graph density, so it is recorded in every report.
    require_present
        See :class:`~datasets.adapters.base.DatasetAdapter`.

    Raises
    ------
    ImportError
        If networkx is not installed.
    """

    info = DatasetInfo(
        name="visem_graphs",
        title="VISEM-Tracking-graphs (graph representations of VISEM-Tracking)",
        url="https://huggingface.co/datasets/SimulaMet-HOST/visem-tracking-graphs",
        license_key="visem_graphs",
        annotation_level="graph nodes = detections (YOLO-normalised boxes) + track ids",
        approximate_size="a few GB (five thresholds x 20 videos x ~1500 frame graphs)",
        capture=CaptureConditions(
            objective_magnification=None,
            total_magnification=None,
            contrast_mode="brightfield, unstained wet preparation",
            stained=False,
            camera="UEye UI-2210C on an Olympus CX31 (inherited from VISEM-Tracking)",
            fps_range=(45.0, 50.0),
            fps_uniform=False,
            resolution=(640, 480),
            um_per_px=None,
            notes=(
                "Not an independent capture: derived entirely from VISEM-Tracking, so "
                "every capture condition and every domain-shift note of that dataset "
                "applies unchanged."
            ),
        ),
        domain_shift_notes=[
            "Purely derived from VISEM-Tracking -- it inherits that dataset's optics, "
            "resolution, frame-rate variability and upper-left spatial prior, and adds "
            "no new imaging conditions.",
            "Coordinates are YOLO-normalised, so a graph carries no absolute scale at "
            "all; any physical interpretation needs the frame size and a calibration "
            "neither this dataset nor its parent provides.",
            "The spatial edge threshold is a modelling choice baked into the released "
            "files. A model trained at threshold 0.3 has learned a neighbourhood "
            "definition, not a property of sperm.",
        ],
        expected_layout=(
            "  <root>/spatial_threshold_{0.1,0.2,0.3,0.4,0.5}/\n"
            "      <video_id>/frame_graphs/frame_graph_{i}.graphml\n"
            "      <video_id>/video_graph.graphml   (see the module docstring: the\n"
            "                                        released file is defective)"
        ),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        threshold: str = "0.3",
        require_present: bool = True,
    ) -> None:
        self._nx = _import_networkx()
        self._threshold = str(threshold)
        if self._threshold not in SPATIAL_THRESHOLDS:
            raise ValueError(
                f"unknown spatial threshold {threshold!r}; published values are "
                f"{list(SPATIAL_THRESHOLDS)}"
            )
        self._video_dirs: dict[int, Path] | None = None
        super().__init__(root, require_present=require_present)

    # ------------------------------------------------------------ discovery

    @classmethod
    def _resolve_root(cls, given: Path) -> Path | None:
        """First candidate holding at least one ``spatial_threshold_*`` folder.

        No threshold folder means this is not a copy of the graph dataset, so
        ``None`` makes the constructor raise with the Hugging Face URL rather
        than deferring the confusion to the first read.
        """
        for candidate in (given, given / "visem-tracking-graphs", given / "data"):
            if candidate.is_dir() and any(
                _THRESHOLD_DIR_RE.match(p.name) for p in candidate.iterdir() if p.is_dir()
            ):
                return candidate
        return None

    @property
    def threshold(self) -> str:
        """The spatial threshold this adapter is reading."""
        return self._threshold

    def threshold_dir(self) -> Path:
        """Directory for the configured threshold."""
        path = self.root / f"spatial_threshold_{self._threshold}"
        return self.require_path(path, f"spatial_threshold_{self._threshold}/")

    def available_thresholds(self) -> list[str]:
        """Thresholds actually present on this disk."""
        out: list[str] = []
        for entry in sorted(self.root.iterdir()):
            match = _THRESHOLD_DIR_RE.match(entry.name) if entry.is_dir() else None
            if match:
                out.append(match.group(1))
        return out

    def _discover_videos(self) -> dict[int, Path]:
        if self._video_dirs is not None:
            return self._video_dirs
        base = self.threshold_dir()
        found: dict[int, Path] = {}
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            match = _VIDEO_ID_RE.search(entry.name)
            if match is None:
                continue
            found[int(match.group(1))] = entry
        if not found:
            raise DatasetValidationError(
                f"VISEM-Tracking-graphs: no video folders under {base}. Expected "
                f"<video_id>/frame_graphs/frame_graph_*.graphml"
            )
        self._video_dirs = found
        return found

    def videos(self) -> list[int]:
        """Video IDs present at the configured threshold."""
        return sorted(self._discover_videos())

    def frame_graph_paths(self, video_id: int) -> list[Path]:
        """Frame-graph files for one video, sorted by frame index."""
        directory = self._discover_videos()[int(video_id)] / "frame_graphs"
        self.require_path(directory, f"frame_graphs/ for video {video_id}")
        paths = list(directory.glob("*.graphml"))
        # A file whose stem carries no trailing integer sorts first at -1 rather
        # than being dropped: an unexpected filename is something to notice, not
        # something to hide.
        paths.sort(key=lambda p: _frame_index_or(p.stem, -1))
        return paths

    def frame_ids(self, video_id: int) -> list[int]:
        """Frame indices present for one video."""
        out = [_frame_index(p.stem) for p in self.frame_graph_paths(video_id)]
        return sorted(i for i in out if i is not None)

    # ---------------------------------------------------------------- reading

    def frame_graph(self, video_id: int, frame_id: int) -> Any:
        """Read one ``frame_graph_{i}.graphml`` as a networkx graph."""
        directory = self._discover_videos()[int(video_id)] / "frame_graphs"
        path = directory / f"frame_graph_{int(frame_id)}.graphml"
        if not path.exists():
            matches = [p for p in self.frame_graph_paths(video_id) if _frame_index(p.stem) == int(frame_id)]
            if not matches:
                raise DatasetValidationError(
                    f"VISEM-Tracking-graphs: no frame graph for video {video_id} frame "
                    f"{frame_id} under {directory}"
                )
            path = matches[0]
        return self._nx.read_graphml(path)

    def iter_frame_graphs(self, video_id: int) -> Iterator[tuple[int, Any]]:
        """Yield ``(frame_id, graph)`` for one video, in frame order."""
        for path in self.frame_graph_paths(video_id):
            frame_id = _frame_index(path.stem)
            if frame_id is None:
                continue
            yield frame_id, self._nx.read_graphml(path)

    def upstream_video_graph_path(self, video_id: int) -> Path:
        return self._discover_videos()[int(video_id)] / "video_graph.graphml"

    def upstream_video_graph(self, video_id: int) -> Any:
        """Read the **defective** released ``video_graph.graphml``.

        Provided so the defect can be measured (see
        :meth:`inspect_upstream_video_graph`), not so it can be used. Use
        :meth:`regenerate_video_graph` for anything that depends on temporal
        structure.
        """
        path = self.upstream_video_graph_path(video_id)
        self.require_path(path, f"video_graph.graphml for video {video_id}")
        return self._nx.read_graphml(path)

    # ------------------------------------------------------------- the fix

    def regenerate_video_graph(
        self,
        video_id: int,
        *,
        spatial_edges: bool = True,
        temporal_edges: bool = True,
        max_temporal_gap: int = 1,
    ) -> Any:
        """Rebuild the video-level graph correctly from the per-frame graphs.

        **The defect being worked around.** The released
        ``video_graph.graphml`` uses ``sperm_id`` as the node key. A sperm that
        appears in 1,400 frames therefore contributes one node, not 1,400, its
        ``frame_number`` attribute holds whichever frame happened to be written
        last, and the temporal-edge loop -- which wants to connect a sperm in
        frame ``i`` to the same sperm in frame ``i+1`` -- connects that single
        node to itself. The result is a graph full of self-loops carrying none
        of the temporal information it was built to carry.

        **The fix.** Node identity here is ``(video_id, frame_id, track_id)``
        via :func:`node_key`, so:

        * one node per detection, as the data actually contains;
        * spatial edges stay within a frame, copied from the frame graph
          together with their ``weight``;
        * temporal edges run from ``(f, t)`` to ``(f + gap, t)`` for the same
          track, directed forwards in time, with ``edge_type="temporal"`` and
          the gap recorded;
        * no self-loop is possible, because source and target differ in
          ``frame_id`` by construction. A post-condition check asserts that.

        Parameters
        ----------
        video_id
            Video to rebuild.
        spatial_edges, temporal_edges
            Include each edge family. Both on by default.
        max_temporal_gap
            Connect a track across up to this many missing frames. 1 means
            strictly consecutive frames. Larger values bridge frames where a
            detection was missed, at the cost of asserting continuity that was
            not observed -- so the gap is recorded on every edge.

        Returns
        -------
        networkx.DiGraph
            Directed, matching the upstream intent. Graph-level attributes
            record the video, the spatial threshold, the source of the rebuild
            and the defect it works around, so a serialised graph explains
            itself.
        """
        if max_temporal_gap < 1:
            raise ValueError(f"max_temporal_gap must be >= 1, got {max_temporal_gap}")

        graph = self._nx.DiGraph(
            video_id=int(video_id),
            spatial_threshold=self._threshold,
            node_identity="(video_id, frame_id, track_id)",
            rebuilt_from="frame_graphs",
            upstream_defect=(
                "the released video_graph.graphml keys nodes by sperm_id alone, "
                "collapsing every frame of a track onto one node and producing "
                "self-loops instead of temporal edges"
            ),
            generator="datasets.adapters.visem_graphs.VisemGraphsAdapter.regenerate_video_graph",
        )

        # track_id -> sorted list of frames in which it was seen.
        appearances: dict[str, list[int]] = {}

        for frame_id, frame_graph in self.iter_frame_graphs(video_id):
            for raw_id, attrs in frame_graph.nodes(data=True):
                track_id = str(raw_id)
                key = node_key(video_id, frame_id, track_id)
                graph.add_node(
                    key,
                    video_id=int(video_id),
                    frame_id=int(frame_id),
                    track_id=track_id,
                    # frame_number is copied verbatim when present so that a
                    # disagreement with the file's own index stays visible
                    # rather than being silently overwritten.
                    upstream_frame_number=attrs.get("frame_number"),
                    class_name=attrs.get("class_name"),
                    x_center=_as_float(attrs.get("x_center")),
                    y_center=_as_float(attrs.get("y_center")),
                    width=_as_float(attrs.get("width")),
                    height=_as_float(attrs.get("height")),
                )
                appearances.setdefault(track_id, []).append(int(frame_id))

            if spatial_edges:
                for source, target, attrs in frame_graph.edges(data=True):
                    if attrs.get("edge_type") == "temporal":
                        # A temporal edge inside a frame graph is meaningless;
                        # skip rather than promote it to a spatial one.
                        continue
                    graph.add_edge(
                        node_key(video_id, frame_id, str(source)),
                        node_key(video_id, frame_id, str(target)),
                        edge_type="spatial",
                        weight=_as_float(attrs.get("weight")),
                        frame_id=int(frame_id),
                    )

        if temporal_edges:
            for track_id, frames in appearances.items():
                ordered = sorted(set(frames))
                for i, frame_id in enumerate(ordered[:-1]):
                    next_frame = ordered[i + 1]
                    gap = next_frame - frame_id
                    if gap > max_temporal_gap:
                        continue
                    graph.add_edge(
                        node_key(video_id, frame_id, track_id),
                        node_key(video_id, next_frame, track_id),
                        edge_type="temporal",
                        gap=int(gap),
                        weight=float(gap),
                    )

        n_self_loops = self._nx.number_of_selfloops(graph)
        if n_self_loops:  # pragma: no cover - unreachable by construction
            raise DatasetValidationError(
                f"regenerated video graph for {video_id} contains {n_self_loops} "
                "self-loop(s); node identity must include the frame, so this indicates "
                "a bug in regenerate_video_graph, not in the upstream data"
            )
        return graph

    def inspect_upstream_video_graph(self, video_id: int) -> dict[str, Any]:
        """Measure the upstream defect on this copy.

        Returns node/edge counts, the self-loop count, and the same figures for
        the regenerated graph, so the difference is a number rather than a claim
        in a docstring.
        """
        upstream = self.upstream_video_graph(video_id)
        rebuilt = self.regenerate_video_graph(video_id)
        n_detections = sum(
            graph.number_of_nodes() for _, graph in self.iter_frame_graphs(video_id)
        )
        return {
            "video_id": int(video_id),
            "spatial_threshold": self._threshold,
            "upstream": {
                "n_nodes": upstream.number_of_nodes(),
                "n_edges": upstream.number_of_edges(),
                "n_self_loops": self._nx.number_of_selfloops(upstream),
            },
            "regenerated": {
                "n_nodes": rebuilt.number_of_nodes(),
                "n_edges": rebuilt.number_of_edges(),
                "n_self_loops": self._nx.number_of_selfloops(rebuilt),
            },
            "n_detections_in_frame_graphs": n_detections,
            "nodes_lost_upstream": max(0, n_detections - upstream.number_of_nodes()),
        }

    def write_video_graph(self, video_id: int, path: str | Path, **kwargs: Any) -> Path:
        """Regenerate and write a corrected video graph to GraphML."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        graph = self.regenerate_video_graph(video_id, **kwargs)
        # GraphML cannot serialise None; drop those attributes rather than
        # writing a string "None" that a later reader would parse as data.
        for _, attrs in graph.nodes(data=True):
            for key in [k for k, v in attrs.items() if v is None]:
                del attrs[key]
        for _, _, attrs in graph.edges(data=True):
            for key in [k for k, v in attrs.items() if v is None]:
                del attrs[key]
        self._nx.write_graphml(graph, out)
        return out

    # ------------------------------------------------------------- contract

    def splits(self) -> list[str]:
        """Inherits VISEM-Tracking's split; see that adapter's ``official_split``."""
        return ["train", "val"]

    def __len__(self) -> int:
        """Total frame graphs at the configured threshold."""
        return sum(len(self.frame_graph_paths(v)) for v in self.videos())

    def validate(self, *, sample_videos: int = 2) -> ValidationReport:
        """Structural checks plus a measurement of the upstream video-graph defect.

        ``sample_videos`` limits how many videos are inspected deeply -- reading
        every frame graph of all 20 videos means tens of thousands of XML files.
        """
        report = self._new_report()
        report.context["spatial_threshold"] = self._threshold

        thresholds = self.available_thresholds()
        report.context["available_thresholds"] = thresholds
        missing = sorted(set(SPATIAL_THRESHOLDS) - set(thresholds))
        report.add(
            "layout:thresholds",
            CheckStatus.PASS if not missing else CheckStatus.WARN,
            f"thresholds present: {thresholds}"
            + (f"; missing {missing}" if missing else ""),
        )
        if self._threshold not in thresholds:
            report.add(
                "layout:selected_threshold",
                CheckStatus.FAIL,
                f"the selected threshold {self._threshold} is not present under {self.root}",
            )
            return report

        videos = self.videos()
        report.checks.append(check_non_empty(len(videos), name="videos", what="video folders"))

        for video_id in videos[: max(0, sample_videos)]:
            paths = self.frame_graph_paths(video_id)
            if not paths:
                report.add(
                    f"frames:{video_id}",
                    CheckStatus.FAIL,
                    f"video {video_id}: frame_graphs/ contains no .graphml files",
                )
                continue
            report.add(
                f"frames:{video_id}",
                CheckStatus.PASS,
                f"video {video_id}: {len(paths)} frame graphs",
                n_frame_graphs=len(paths),
            )
            if not self.upstream_video_graph_path(video_id).exists():
                report.add(
                    f"video_graph:{video_id}",
                    CheckStatus.SKIPPED,
                    f"video {video_id}: no video_graph.graphml to inspect",
                )
                continue
            try:
                measured = self.inspect_upstream_video_graph(video_id)
            except Exception as exc:
                report.add(
                    f"video_graph:{video_id}",
                    CheckStatus.FAIL,
                    f"video {video_id}: could not inspect the video graph: {exc!r}",
                )
                continue
            n_loops = measured["upstream"]["n_self_loops"]
            report.add(
                f"video_graph:{video_id}",
                CheckStatus.WARN if n_loops else CheckStatus.PASS,
                (
                    f"video {video_id}: the released video_graph.graphml has "
                    f"{measured['upstream']['n_nodes']} nodes and {n_loops} self-loop(s) "
                    f"for {measured['n_detections_in_frame_graphs']} detections in the "
                    "frame graphs -- the documented sperm_id-only keying defect. Use "
                    "regenerate_video_graph(); the regenerated graph has "
                    f"{measured['regenerated']['n_nodes']} nodes and no self-loops."
                )
                if n_loops
                else (
                    f"video {video_id}: no self-loops in the released video graph "
                    "(unexpected -- the upstream defect may have been fixed; re-check "
                    "before relying on the released file)"
                ),
                **measured,
            )
        return report


# ==========================================================================
# helpers
# ==========================================================================


def _import_networkx() -> Any:
    """Import networkx or raise with the install command."""
    try:
        import networkx
    except ImportError as exc:
        raise ImportError(
            "VisemGraphsAdapter needs networkx, which is not installed "
            f"({exc}). Install it with `pip install networkx`. This adapter is an "
            "optional extension -- nothing in sperm_sorting depends on it, so the MVP "
            "runs without networkx."
        ) from exc
    return networkx


def _frame_index(stem: str) -> int | None:
    match = _FRAME_GRAPH_RE.search(stem)
    return int(match.group(1)) if match else None


def _frame_index_or(stem: str, fallback: int) -> int:
    """:func:`_frame_index` with a total order, for use as a sort key."""
    index = _frame_index(stem)
    return fallback if index is None else index


def _as_float(value: Any) -> float | None:
    """GraphML attributes arrive as strings; ``None`` stays ``None``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
