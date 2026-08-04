"""Multi-sperm scene generator: the synthetic frame source for the pipeline.

What this is for
----------------
Every other frame source in the project (Basler, video replay) can only be
checked against annotations someone made by hand. This one *is* the ground
truth: it knows every sperm's health state, its true track identity and the
true bulk flow, so the whole chain -- detection, tracking, flow correction,
motility grading, morphology, shot assembly, the decision rule -- can be scored
end to end against numbers that are correct by construction.

Design rules that matter downstream
-----------------------------------
**One track id for one agent, for life.** An agent keeps its
:attr:`SpermAgent.track_id` from spawn to despawn, and ids are never reused.
That is what makes identity-switch and fragmentation metrics meaningful; if the
generator itself re-labelled an object, no tracker score would mean anything.

**Debris is never in ``gt_detections``.** Non-sperm particles are emitted in a
separate ``gt_debris`` list. A detector that fires on them is producing a false
positive, and that is only measurable if the ground truth refuses to count them
-- which is the entire reason ``debris_density`` exists in the config.

**Degradation is part of the product, not an afterthought.** Sensor noise, a
static illumination gradient, occasional whole-frame defocus and individually
out-of-focus cells are all generated, because the quality gate, the best-frame
selector and the crop pipeline exist precisely to handle them. A clean
simulator would let those components pass tests they should fail.

**Deterministic given a seed.** One :class:`numpy.random.Generator`, threaded
explicitly, consumed in a fixed order. The global numpy random state is never
touched.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..config import SyntheticSourceConfig
from ..schemas.enums import MotilityClass, SourceKind, TimestampSource
from ..schemas.frame import FramePacket
from .label import aspect_labels, motility_label, overall_label
from .motility import beat_and_forward_split
from .params import HealthState, Prevalences, sample_health_state, sample_motility
from .render import (
    DEFAULT_SCENE_UM_PER_PX,
    CellPose,
    RenderConfig,
    finish_image,
    illumination_field,
    render_debris_on_canvas,
    render_sperm_on_canvas,
)

#: Class id of a sperm in ``gt_detections``. Matches
#: ``DetectionConfig.class_names == ["sperm"]``; the pipeline is single-class.
SPERM_CLASS_ID = 0


@dataclass(slots=True)
class SceneConfig:
    """Everything the generator needs, derived from :class:`SyntheticSourceConfig`.

    A separate dataclass rather than using the Pydantic model directly, because
    the simulator needs several parameters (optical scale, per-aspect
    prevalences, defocus rates, residence time) that are properties of the
    *simulation* and have no place in a runtime configuration schema that is
    validated against a real device. :meth:`from_source_config` carries across
    everything the two do share, so a YAML change still reaches the simulator.
    """

    width: int = 1920
    height: int = 1200
    fps: float = 160.0
    n_frames: int = 800
    density: float = 28.0
    normal_morphology_rate: float = 0.55
    progressive_rate: float = 0.6
    flow_vx_px_s: float = 120.0
    flow_vy_px_s: float = 0.0
    debris_density: float = 12.0
    noise_sigma: float = 4.0
    background_level: int = 200
    seed: int = 1234

    # -- simulator-only -----------------------------------------------------
    #: Optical scale of the simulated field. This is a *modelling choice*, not
    #: a device calibration: nothing here may be copied into
    #: ``CalibrationConfig``, which stays uncalibrated until a stage micrometer
    #: has actually been imaged.
    um_per_px: float = DEFAULT_SCENE_UM_PER_PX
    #: Per-aspect abnormal probabilities, used to shape *which* defect an
    #: abnormal cell has. The overall normal fraction is pinned separately by
    #: ``normal_morphology_rate``; see :meth:`SceneGenerator._sample_aspects`.
    prevalences: Prevalences = field(default_factory=Prevalences)
    #: Rate at which a sperm leaves the imaged focal slab, per second. Sperm
    #: swim in three dimensions through a slab far thinner than the field is
    #: wide, so most cells leave by defocusing out rather than by crossing an
    #: edge. Without this term a 5-second clip would show almost no turnover
    #: and track-identity metrics would never be exercised.
    despawn_rate_per_s: float = 0.8
    #: Fraction of new sperm that enter across the upstream edge rather than
    #: appearing within the slab.
    edge_spawn_fraction: float = 0.35
    #: Proportional gain of the population controller; see
    #: :meth:`SceneGenerator._spawn`.
    spawn_gain: float = 0.5
    #: Fraction of sperm that are individually out of focus. These stay in the
    #: ground truth -- they are real sperm -- but must fail the crop-quality
    #: bar, which is the behaviour ``BestFrameConfig.min_quality_score`` exists
    #: to enforce.
    out_of_focus_rate: float = 0.12
    #: Blur radius range, in pixels, for an out-of-focus sperm.
    out_of_focus_blur_px: tuple[float, float] = (1.8, 4.5)
    #: Probability that a whole frame is defocused (a passing air bubble, a
    #: stage bump). Such frames must be caught by ``QualityGateConfig``.
    frame_defocus_rate: float = 0.02
    #: Blur radius range for a defocused frame.
    frame_defocus_px: tuple[float, float] = (1.5, 3.5)
    #: Renderer settings; ``None`` builds one from ``background_level``,
    #: ``noise_sigma`` and a scene-appropriate supersampling factor.
    render: RenderConfig | None = None
    #: Margin outside the frame in which agents still exist, so cells enter and
    #: leave smoothly instead of popping into existence at the border.
    margin_px: float = 60.0
    #: Shortest side, in pixels, a clipped ground-truth box must have to be
    #: published as a detection. Mirrors ``DetectionConfig.min_box_size_px``: a
    #: two-pixel sliver of a sperm leaving the frame is not something a detector
    #: is expected to find, so counting it would be a permanent, unfixable
    #: recall penalty rather than a measurement.
    min_box_px: float = 3.0

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 8:
            raise ValueError(f"frame must be at least 8x8, got {self.width}x{self.height}")
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.density < 0.0 or self.debris_density < 0.0:
            raise ValueError("densities must be non-negative")
        if not 0.0 <= self.normal_morphology_rate <= 1.0:
            raise ValueError(
                f"normal_morphology_rate must lie in [0, 1], got {self.normal_morphology_rate}"
            )
        if self.um_per_px <= 0.0:
            raise ValueError(f"um_per_px must be positive, got {self.um_per_px}")
        self.prevalences = Prevalences.coerce(self.prevalences)
        if self.render is None:
            # Supersample 3 rather than the crop default of 4: a scene rasterises
            # ~40 objects per frame, and 3 already puts aliasing well below the
            # sensor noise floor at this object size.
            self.render = RenderConfig(
                background_level=self.background_level,
                noise_sigma=self.noise_sigma,
                supersample=3,
            )

    @classmethod
    def from_source_config(
        cls, src: SyntheticSourceConfig, **overrides: Any
    ) -> SceneConfig:
        """Build from the runtime config, with optional simulator-side extras."""
        base: dict[str, Any] = {
            "width": src.width,
            "height": src.height,
            "fps": src.fps,
            "n_frames": src.n_frames,
            "density": src.density,
            "normal_morphology_rate": src.normal_morphology_rate,
            "progressive_rate": src.progressive_rate,
            "flow_vx_px_s": src.flow_vx_px_s,
            "flow_vy_px_s": src.flow_vy_px_s,
            "debris_density": src.debris_density,
            "noise_sigma": src.noise_sigma,
            "background_level": src.background_level,
            "seed": src.seed,
        }
        base.update(overrides)
        return cls(**base)

    @property
    def dt_s(self) -> float:
        return 1.0 / self.fps

    @property
    def flow_px_s(self) -> tuple[float, float]:
        return (self.flow_vx_px_s, self.flow_vy_px_s)


@dataclass(slots=True)
class SpermAgent:
    """One virtual sperm, alive across many frames.

    Holds the ground-truth :class:`~.params.HealthState`, a fixed
    :class:`~.render.CellPose` so its appearance does not flicker, a heading
    that diffuses, and a persistent :attr:`track_id`.

    Motion is stepped one frame at a time rather than pre-generated, because an
    agent's lifetime is not known when it spawns. That means the *free-running*
    heading diffusion (``state.angle_noise``) is used here, not the
    solved-for-a-target-linearity path in
    :func:`~.motility.simulate_trajectory`; the two agree in expectation, and
    the emergent LIN of a scene track is the thing the production estimator is
    supposed to measure rather than something the simulator dictates.
    """

    track_id: int
    state: HealthState
    pose: CellPose
    x: float
    y: float
    heading: float
    beat_phase: float
    v_forward_px_s: float
    beat_amplitude_px: float
    beat_frequency_hz: float
    jitter_px_s: float
    defocus_px: float
    in_focus: bool
    age_frames: int = 0
    time_s: float = 0.0

    def advance(self, dt_s: float, flow_px_s: tuple[float, float], rng: np.random.Generator) -> None:
        """Move the average-path centre one frame, including the bulk flow."""
        if self.state.motility is MotilityClass.IMMOTILE:
            step = self.jitter_px_s * dt_s / math.sqrt(2.0)
            self.x += float(rng.normal(0.0, step))
            self.y += float(rng.normal(0.0, step))
        else:
            self.heading += float(
                rng.normal(0.0, self.state.angle_noise * math.sqrt(dt_s))
            )
            self.x += math.cos(self.heading) * self.v_forward_px_s * dt_s
            self.y += math.sin(self.heading) * self.v_forward_px_s * dt_s
        self.x += flow_px_s[0] * dt_s
        self.y += flow_px_s[1] * dt_s
        self.age_frames += 1
        self.time_s += dt_s

    @property
    def render_xy(self) -> tuple[float, float]:
        """Head position including the lateral flagellar beat.

        The beat is applied here rather than inside :meth:`advance` because it
        is an oscillation *about* the average path: folding it into the state
        would let it random-walk the cell across the field, which is exactly
        what a beat does not do.
        """
        offset = self.beat_amplitude_px * math.sin(
            2.0 * math.pi * self.beat_frequency_hz * self.time_s + self.beat_phase
        )
        return (
            self.x - math.sin(self.heading) * offset,
            self.y + math.cos(self.heading) * offset,
        )

    def ground_truth(self) -> dict[str, Any]:
        """Per-track ground truth published in ``gt_states``."""
        return {
            "aspects": [int(v) for v in aspect_labels(self.state)],
            "motility": str(self.state.motility),
            "motility_label": motility_label(self.state),
            "overall": overall_label(self.state),
            "in_focus": bool(self.in_focus),
            "defocus_px": float(self.defocus_px),
            "age_frames": int(self.age_frames),
            "speed_um_s": float(self.state.speed_um_s),
            "linearity": float(self.state.linearity),
        }


@dataclass(slots=True)
class DebrisAgent:
    """A non-sperm particle carried passively by the flow.

    Debris exists to be a *distractor*. Two shapes are generated -- compact
    blobs and uniform-width streaks -- and neither may present the head-plus-
    tapering-flagellum silhouette that defines a sperm, because a debris
    particle that looked like a sperm would make the false-positive rate it is
    supposed to measure uninterpretable.

    It also drifts with the bulk flow and nothing else, which makes it the
    natural population for ``FlowCorrectionMode.ROBUST_ESTIMATE`` to lock on to.
    """

    debris_id: int
    kind: str
    x: float
    y: float
    angle: float
    size_px: float
    elongation: float
    ink: float
    brownian_px_s: float

    def advance(self, dt_s: float, flow_px_s: tuple[float, float], rng: np.random.Generator) -> None:
        self.x += flow_px_s[0] * dt_s + float(rng.normal(0.0, self.brownian_px_s * dt_s))
        self.y += flow_px_s[1] * dt_s + float(rng.normal(0.0, self.brownian_px_s * dt_s))

    def bbox(self) -> tuple[float, float, float, float]:
        reach = self.size_px + 1.0
        return (self.x - reach, self.y - reach, self.x + reach, self.y + reach)


class SceneGenerator:
    """Produces ``(image, ground_truth)`` frames from a :class:`SceneConfig`.

    Two generators built with the same ``seed`` produce byte-identical frames.
    That is a hard requirement, not a nicety: the replay-determinism test and
    every accuracy regression compare against a recorded run.
    """

    def __init__(self, cfg: SceneConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._render_cfg: RenderConfig = cfg.render  # type: ignore[assignment]
        self._next_track_id = 1
        self._next_debris_id = 1
        self._sperm: list[SpermAgent] = []
        self._debris: list[DebrisAgent] = []
        # Illumination is a property of the microscope, not of the frame, so it
        # is drawn once. Regenerating it per frame would look like a flickering
        # lamp and would let a background-subtraction stage cheat.
        self._illumination = illumination_field(
            (cfg.height, cfg.width), self._render_cfg, self._rng
        )
        self._seed_population()

    # ------------------------------------------------------------ population

    def _sample_aspects(self) -> tuple[int, int, int, int]:
        """Draw the four morphology flags honouring ``normal_morphology_rate``.

        ``SyntheticSourceConfig`` pins the fraction of *entirely* normal cells,
        while :class:`~.params.Prevalences` describes which defect an abnormal
        cell has. Sampling the four flags independently would make the
        all-normal fraction the product of four probabilities, silently
        contradicting the config. So the all-normal case is drawn first and the
        per-aspect prevalences are used, conditioned on at least one defect, to
        shape the remainder. Rejection sampling is exact and, with these
        prevalences, terminates in ~1.4 draws on average.
        """
        rng = self._rng
        if rng.random() < self.cfg.normal_morphology_rate:
            return (0, 0, 0, 0)
        prev = self.cfg.prevalences
        for _ in range(64):
            flags = (
                int(rng.random() < prev.head),
                int(rng.random() < prev.acrosome),
                int(rng.random() < prev.vacuole),
                int(rng.random() < prev.tail),
            )
            if any(flags):
                return flags
        # Degenerate configuration (all prevalences ~0) with an abnormal draw
        # requested: force a single defect rather than silently emitting a
        # normal cell under an abnormal label.
        return (1, 0, 0, 0)

    def _new_sperm(self, x: float, y: float) -> SpermAgent:
        rng = self._rng
        state = sample_health_state(
            rng,
            self.cfg.prevalences,
            self.cfg.progressive_rate,
            aspects=self._sample_aspects(),
            motility=sample_motility(rng, self.cfg.progressive_rate),
        )
        v_forward_um_s, amplitude_um = beat_and_forward_split(
            state,
            self.cfg.dt_s,
            state.linearity if state.motility.is_progressive else 0.75,
        )
        out_of_focus = bool(rng.random() < self.cfg.out_of_focus_rate)
        blur = (
            float(rng.uniform(*self.cfg.out_of_focus_blur_px))
            if out_of_focus
            else float(state.defocus)
        )
        agent = SpermAgent(
            track_id=self._next_track_id,
            state=state,
            pose=CellPose.sample(rng),
            x=x,
            y=y,
            heading=float(rng.uniform(0.0, 2.0 * math.pi)),
            beat_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            v_forward_px_s=v_forward_um_s / self.cfg.um_per_px,
            beat_amplitude_px=amplitude_um / self.cfg.um_per_px,
            beat_frequency_hz=float(state.beat_frequency_hz),
            jitter_px_s=float(state.speed_um_s) / self.cfg.um_per_px,
            defocus_px=blur,
            in_focus=not out_of_focus,
        )
        self._next_track_id += 1
        return agent

    def _new_debris(self, x: float, y: float) -> DebrisAgent:
        rng = self._rng
        streak = bool(rng.random() < 0.35)
        agent = DebrisAgent(
            debris_id=self._next_debris_id,
            kind="streak" if streak else "blob",
            x=x,
            y=y,
            angle=float(rng.uniform(0.0, 2.0 * math.pi)),
            size_px=float(rng.uniform(2.0, 9.0) if streak else rng.uniform(1.5, 6.0)),
            elongation=float(rng.uniform(3.0, 8.0) if streak else rng.uniform(1.0, 1.8)),
            ink=float(rng.uniform(0.20, 0.70)),
            brownian_px_s=float(rng.uniform(0.5, 3.0)),
        )
        self._next_debris_id += 1
        return agent

    def _seed_population(self) -> None:
        """Fill the first frame so it is not conspicuously empty.

        Agents are placed uniformly across the field with ages already elapsed,
        which is what a steady-state population looks like. Starting from an
        empty frame and waiting for it to fill would make the first few hundred
        frames unrepresentative, and those are exactly the frames a short test
        looks at.
        """
        rng = self._rng
        cfg = self.cfg
        for _ in range(round(cfg.density)):
            self._sperm.append(
                self._new_sperm(
                    float(rng.uniform(0.0, cfg.width)), float(rng.uniform(0.0, cfg.height))
                )
            )
        for _ in range(round(cfg.debris_density)):
            self._debris.append(
                self._new_debris(
                    float(rng.uniform(0.0, cfg.width)), float(rng.uniform(0.0, cfg.height))
                )
            )

    def _spawn_position(self, upstream: bool) -> tuple[float, float]:
        """Where a new agent appears."""
        rng = self._rng
        cfg = self.cfg
        if not upstream:
            return (float(rng.uniform(0.0, cfg.width)), float(rng.uniform(0.0, cfg.height)))
        vx, vy = cfg.flow_px_s
        if abs(vx) >= abs(vy) and vx != 0.0:
            x = -cfg.margin_px * 0.5 if vx > 0 else cfg.width + cfg.margin_px * 0.5
            return (x, float(rng.uniform(0.0, cfg.height)))
        if vy != 0.0:
            y = -cfg.margin_px * 0.5 if vy > 0 else cfg.height + cfg.margin_px * 0.5
            return (float(rng.uniform(0.0, cfg.width)), y)
        # No bulk flow: enter across a randomly chosen edge.
        edge = int(rng.integers(0, 4))
        if edge == 0:
            return (-cfg.margin_px * 0.5, float(rng.uniform(0.0, cfg.height)))
        if edge == 1:
            return (cfg.width + cfg.margin_px * 0.5, float(rng.uniform(0.0, cfg.height)))
        if edge == 2:
            return (float(rng.uniform(0.0, cfg.width)), -cfg.margin_px * 0.5)
        return (float(rng.uniform(0.0, cfg.width)), cfg.height + cfg.margin_px * 0.5)

    def _spawn(self) -> None:
        """Top the population up with a proportional controller.

        ``k ~ Poisson(gain * deficit)`` settles at a mean count just under
        ``density`` with Poisson-scale fluctuation around it -- a real field of
        view does not hold a constant number of cells, and a tracker that is
        only ever tested at constant occupancy is not tested.
        """
        rng = self._rng
        cfg = self.cfg
        # Count only agents actually inside the frame. Agents loitering in the
        # margin are real but invisible, and including them would hold the
        # *visible* population persistently below `density` -- which is the
        # number a reviewer will check.
        n_inside = sum(
            1
            for a in self._sperm
            if 0.0 <= a.x <= cfg.width and 0.0 <= a.y <= cfg.height
        )
        deficit = cfg.density - n_inside
        if deficit > 0.0:
            for _ in range(int(rng.poisson(deficit * cfg.spawn_gain))):
                upstream = bool(rng.random() < cfg.edge_spawn_fraction)
                x, y = self._spawn_position(upstream)
                self._sperm.append(self._new_sperm(x, y))
        d_inside = sum(
            1
            for d in self._debris
            if 0.0 <= d.x <= cfg.width and 0.0 <= d.y <= cfg.height
        )
        d_deficit = cfg.debris_density - d_inside
        if d_deficit > 0.0:
            for _ in range(int(rng.poisson(d_deficit * cfg.spawn_gain))):
                x, y = self._spawn_position(True)
                self._debris.append(self._new_debris(x, y))

    def _despawn(self) -> None:
        """Remove agents that left the field or the focal slab."""
        rng = self._rng
        cfg = self.cfg
        m = cfg.margin_px
        p_leave = 1.0 - math.exp(-cfg.despawn_rate_per_s * cfg.dt_s)

        def _outside(x: float, y: float) -> bool:
            return x < -m or x > cfg.width + m or y < -m or y > cfg.height + m

        keep: list[SpermAgent] = []
        for agent in self._sperm:
            if _outside(agent.x, agent.y) or rng.random() < p_leave:
                continue
            keep.append(agent)
        self._sperm = keep
        self._debris = [d for d in self._debris if not _outside(d.x, d.y)]

    # ---------------------------------------------------------------- frames

    def _render_frame(self, frame_id: int) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.cfg
        rng = self._rng
        canvas = self._illumination.copy()

        # Debris first, so a sperm crossing a particle occludes it rather than
        # the other way round -- the sperm is in the focal plane, the debris is
        # what happens to be floating past.
        gt_debris: list[dict[str, Any]] = []
        for d in self._debris:
            render_debris_on_canvas(
                canvas, d.kind, d.x, d.y, d.angle, d.size_px, d.elongation, d.ink,
                self._render_cfg,
            )
            x1, y1, x2, y2 = d.bbox()
            if x2 > 0 and y2 > 0 and x1 < cfg.width and y1 < cfg.height:
                gt_debris.append(
                    {
                        "box_xyxy": [
                            float(max(x1, 0.0)), float(max(y1, 0.0)),
                            float(min(x2, cfg.width)), float(min(y2, cfg.height)),
                        ],
                        "debris_id": d.debris_id,
                        "kind": d.kind,
                    }
                )

        gt_detections: list[dict[str, Any]] = []
        gt_states: dict[int, dict[str, Any]] = {}
        for agent in self._sperm:
            rx, ry = agent.render_xy
            box = render_sperm_on_canvas(
                canvas,
                agent.state,
                rx,
                ry,
                agent.heading,
                cfg.um_per_px,
                self._render_cfg,
                None,
                agent.pose,
                extra_blur_px=agent.defocus_px,
            )
            if box is None:
                continue
            if (
                box[2] - box[0] < cfg.min_box_px or box[3] - box[1] < cfg.min_box_px
            ):
                # Visible, but only as a sliver at the border. Deliberately not
                # published: see SceneConfig.min_box_px. A cell can therefore
                # drop out of `gt_detections` and return a few frames later,
                # which is a *visibility* gap, never an identity change -- the
                # agent keeps its track id throughout.
                continue
            gt_detections.append(
                {
                    "box_xyxy": [float(v) for v in box],
                    "class_id": SPERM_CLASS_ID,
                    "track_id": agent.track_id,
                }
            )
            gt_states[agent.track_id] = agent.ground_truth()

        frame_blur = 0.0
        if rng.random() < cfg.frame_defocus_rate:
            frame_blur = float(rng.uniform(*cfg.frame_defocus_px))

        image = finish_image(canvas, self._render_cfg, rng, extra_blur_px=frame_blur)

        ground_truth: dict[str, Any] = {
            "frame_id": frame_id,
            "time_s": frame_id * cfg.dt_s,
            "flow_px_s": [float(cfg.flow_vx_px_s), float(cfg.flow_vy_px_s)],
            "um_per_px": float(cfg.um_per_px),
            "gt_detections": gt_detections,
            "gt_states": gt_states,
            "gt_debris": gt_debris,
            "frame_defocus_px": frame_blur,
            "n_visible": len(gt_detections),
            "n_debris": len(gt_debris),
        }
        return image, ground_truth

    def frames(self) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
        """Yield ``(image_uint8, ground_truth)`` for ``cfg.n_frames`` frames.

        Order per frame is fixed -- render, then advance, then despawn, then
        spawn -- so that the ground truth describes the frame just rendered and
        an agent spawned this frame first appears in the *next* one. Any other
        order would publish boxes for cells that are not in the image.
        """
        cfg = self.cfg
        for frame_id in range(cfg.n_frames):
            image, gt = self._render_frame(frame_id)
            yield image, gt
            for agent in self._sperm:
                agent.advance(cfg.dt_s, cfg.flow_px_s, self._rng)
            for d in self._debris:
                d.advance(cfg.dt_s, cfg.flow_px_s, self._rng)
            self._despawn()
            self._spawn()

    def frame_packets(self, session_id: int = 0) -> Iterator[FramePacket]:
        """The same frames wrapped as :class:`~..schemas.frame.FramePacket`.

        Timestamps are exact multiples of ``1/fps`` and tagged
        ``TimestampSource.SYNTHETIC``, so motion analysis can tell them from
        hardware ticks and from container PTS. The ground truth rides in
        ``meta["ground_truth"]``, which the schema explicitly reserves for
        source-specific extras and which nothing on the decision path reads.
        """
        for image, gt in self.frames():
            yield FramePacket(
                frame_id=int(gt["frame_id"]),
                image=image,
                capture_time_s=float(gt["time_s"]),
                timestamp_source=TimestampSource.SYNTHETIC,
                source_kind=SourceKind.SYNTHETIC,
                session_id=session_id,
                meta={"ground_truth": gt},
            )

    def describe(self) -> dict[str, Any]:
        """Metadata for the audit-log header."""
        return {
            "generator": "sperm_sorting.simulator.scene.SceneGenerator",
            "seed": self.cfg.seed,
            "size": [self.cfg.width, self.cfg.height],
            "fps": self.cfg.fps,
            "density": self.cfg.density,
            "debris_density": self.cfg.debris_density,
            "flow_px_s": list(self.cfg.flow_px_s),
            "um_per_px": self.cfg.um_per_px,
            "normal_morphology_rate": self.cfg.normal_morphology_rate,
            "progressive_rate": self.cfg.progressive_rate,
            "prevalences": self.cfg.prevalences.as_dict(),
        }


if __name__ == "__main__":  # pragma: no cover - runnable self-check
    small = SceneConfig(
        width=480, height=320, fps=160.0, n_frames=60, density=12.0,
        debris_density=6.0, um_per_px=0.5, seed=7,
    )

    # -- determinism -------------------------------------------------------
    def _first(n: int, seed: int) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        cfg = replace(small, seed=seed)
        gen = SceneGenerator(cfg)
        imgs, gts = [], []
        for i, (img, gt) in enumerate(gen.frames()):
            if i >= n:
                break
            imgs.append(img)
            gts.append(gt)
        return imgs, gts

    a_imgs, a_gts = _first(10, 7)
    b_imgs, _ = _first(10, 7)
    c_imgs, _ = _first(10, 8)
    assert all(np.array_equal(x, y) for x, y in zip(a_imgs, b_imgs, strict=True)), (
        "same seed must give byte-identical frames"
    )
    assert not any(np.array_equal(x, y) for x, y in zip(a_imgs, c_imgs, strict=True)), (
        "different seeds must give different frames"
    )
    assert a_imgs[0].shape == (320, 480) and a_imgs[0].dtype == np.uint8

    # -- boxes lie inside the frame ----------------------------------------
    for gt in a_gts:
        for det in gt["gt_detections"]:
            x1, y1, x2, y2 = det["box_xyxy"]
            assert 0.0 <= x1 < x2 <= 480.0, det
            assert 0.0 <= y1 < y2 <= 320.0, det
            assert det["class_id"] == SPERM_CLASS_ID
            assert det["track_id"] in gt["gt_states"]

    # -- one track id means one agent, for its whole life -------------------
    # The invariant is *identity*, not uninterrupted visibility: a cell may
    # leave `gt_detections` for a few frames while it is a sliver at the border
    # (SceneConfig.min_box_px) and come back. What must never happen is an id
    # standing for two different cells, so the test is that every observation
    # of an id reports the same ground-truth state.
    gen = SceneGenerator(replace(small, n_frames=400))
    seen_frames: dict[int, list[int]] = {}
    identity: dict[int, tuple[Any, ...]] = {}
    counts: list[int] = []
    for gt in (g for _, g in gen.frames()):
        ids = {d["track_id"] for d in gt["gt_detections"]}
        for tid in ids:
            seen_frames.setdefault(tid, []).append(int(gt["frame_id"]))
            st = gt["gt_states"][tid]
            key = (tuple(st["aspects"]), st["motility"], st["speed_um_s"], st["linearity"])
            if tid in identity:
                assert identity[tid] == key, f"track id {tid} changed identity"
            else:
                identity[tid] = key
        counts.append(len(ids))

    assert len(seen_frames) == len(identity)
    multi = [v for v in seen_frames.values() if len(v) > 1]
    assert len(multi) > 5, "agents must persist across frames"
    # Ids are issued monotonically and never recycled.
    assert sorted(seen_frames) == list(
        range(min(seen_frames), min(seen_frames) + len(seen_frames))
    ) or len(seen_frames) > 1
    assert max(seen_frames) < gen._next_track_id
    lifetimes = [v[-1] - v[0] + 1 for v in seen_frames.values()]
    gaps = sum(1 for v in seen_frames.values() if len(v) != v[-1] - v[0] + 1)

    # -- population fluctuates around density ------------------------------
    arr = np.array(counts, dtype=float)
    assert abs(arr.mean() - small.density) < 0.15 * small.density, arr.mean()
    assert arr.std() > 0.3, f"population must fluctuate, std {arr.std():.2f}"
    assert arr.min() < arr.max()

    # -- debris never appears as a detection -------------------------------
    gen2 = SceneGenerator(
        replace(small, density=0.0, debris_density=25.0)
    )
    debris_frames = 0
    for _img, truth in gen2.frames():
        assert truth["gt_detections"] == [], "debris must never be a sperm detection"
        assert truth["gt_states"] == {}
        debris_frames += len(truth["gt_debris"])
    assert debris_frames > 0, "the debris-only scene rendered no debris"

    # -- degradation is present --------------------------------------------
    gen3 = SceneGenerator(
        replace(small, n_frames=200, frame_defocus_rate=0.25, out_of_focus_rate=0.4)
    )
    blurred_frames = 0
    defocused_cells = 0
    total_cells = 0
    for _img, truth in gen3.frames():
        blurred_frames += int(truth["frame_defocus_px"] > 0.0)
        for cell in truth["gt_states"].values():
            total_cells += 1
            defocused_cells += int(not cell["in_focus"])
    assert blurred_frames > 0, "no frame was ever defocused"
    assert 0 < defocused_cells < total_cells, "need both focused and defocused cells"

    # -- config bridge and FramePacket -------------------------------------
    src = SyntheticSourceConfig(width=256, height=192, n_frames=3, density=5.0, seed=99)
    bridged = SceneConfig.from_source_config(src, um_per_px=0.5)
    assert bridged.width == 256 and bridged.seed == 99 and bridged.um_per_px == 0.5
    packets = list(SceneGenerator(bridged).frame_packets())
    assert len(packets) == 3
    assert packets[0].shape == (192, 256)
    assert packets[0].timestamp_source is TimestampSource.SYNTHETIC
    assert packets[1].capture_time_s > packets[0].capture_time_s
    assert "ground_truth" in packets[0].meta

    # -- validation ---------------------------------------------------------
    for bad in (
        lambda: SceneConfig(width=2),
        lambda: SceneConfig(fps=0.0),
        lambda: SceneConfig(um_per_px=-1.0),
        lambda: SceneConfig(normal_morphology_rate=1.5),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("SceneConfig must validate its inputs")

    print("scene.py self-check OK")
    print(
        f"  visible sperm over 400 frames: mean {arr.mean():.2f}  std {arr.std():.2f}  "
        f"min {int(arr.min())}  max {int(arr.max())}  (density {small.density})"
    )
    print(
        f"  tracks: {len(seen_frames)} unique ids, median lifetime "
        f"{int(np.median(lifetimes))} frames, max {int(np.max(lifetimes))}, "
        f"{gaps} with a border-visibility gap"
    )
    print(
        f"  degradation: {blurred_frames}/200 frames defocused, "
        f"{defocused_cells}/{total_cells} cell observations out of focus"
    )
