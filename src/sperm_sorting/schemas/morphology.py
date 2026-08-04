"""Morphology schemas.

Four independent binary decisions, never averaged into one score. The model
emits ``P(normal)`` per aspect; each aspect has its own calibrated threshold
because their prevalences differ by roughly an order of magnitude in MHSMA and
a single shared threshold would be dominated by the majority aspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..constants import LABEL_ABNORMAL, LABEL_NORMAL, MORPHOLOGY_ASPECTS, SCHEMA_VERSION
from .enums import MorphologyStatus


@dataclass(slots=True)
class AspectResult:
    """One morphology aspect's outcome.

    Attributes
    ----------
    p_normal
        Model probability that the aspect is **normal**. Note the polarity:
        MHSMA labels ``0 = normal``, so a model trained with "abnormal" as the
        positive class must have its output flipped exactly once, in the
        inference adapter. See ``tests/test_mhsma_label_direction.py``.
    threshold
        Calibrated decision threshold for this aspect. ``normal`` is
        ``p_normal >= threshold``.
    """

    name: str
    p_normal: float
    threshold: float

    @property
    def normal(self) -> bool:
        return self.p_normal >= self.threshold

    @property
    def label(self) -> int:
        """MHSMA-convention integer label: 0 normal, 1 abnormal."""
        return LABEL_NORMAL if self.normal else LABEL_ABNORMAL

    #: Distance from the threshold; small values mean a borderline call.
    @property
    def margin(self) -> float:
        return abs(self.p_normal - self.threshold)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "p_normal": float(self.p_normal),
            "threshold": float(self.threshold),
            "normal": self.normal,
            "label": self.label,
            "margin": float(self.margin),
        }


@dataclass(slots=True)
class MorphologyResult:
    """Four-aspect morphology verdict for one tracked sperm."""

    track_id: int
    status: MorphologyStatus
    head: AspectResult | None = None
    acrosome: AspectResult | None = None
    vacuole: AspectResult | None = None
    tail: AspectResult | None = None

    #: Which frame the winning crop came from, duplicated for auditability.
    frame_id: int | None = None
    #: Inference wall time in milliseconds.
    latency_ms: float = 0.0
    #: Identifier of the weights used, so a decision can be traced to a model.
    model_id: str = ""
    #: One of the ``WEIGHTS_PROVENANCE_*`` constants.
    weights_provenance: str = ""
    #: Populated when ``status`` is not ``COMPLETE``.
    failure_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def is_complete(self) -> bool:
        return self.status is MorphologyStatus.COMPLETE and all(
            getattr(self, name) is not None for name in MORPHOLOGY_ASPECTS
        )

    @property
    def aspects(self) -> tuple[AspectResult | None, ...]:
        """Aspects in the canonical order ``(head, acrosome, vacuole, tail)``."""
        return tuple(getattr(self, name) for name in MORPHOLOGY_ASPECTS)

    @property
    def all_four_normal(self) -> bool:
        """The rule. All four aspects normal, conjunctively.

        A missing aspect is *not* normal. Incomplete evaluation can never
        produce an acceptable sperm.
        """
        if not self.is_complete:
            return False
        return (
            self.head.normal  # type: ignore[union-attr]
            and self.acrosome.normal  # type: ignore[union-attr]
            and self.vacuole.normal  # type: ignore[union-attr]
            and self.tail.normal  # type: ignore[union-attr]
        )

    def first_abnormal_aspect(self) -> str | None:
        """Name of the first abnormal aspect in canonical order, or ``None``.

        Used to explain a rejection in the audit log. Returns ``None`` only
        when every aspect is present and normal.
        """
        for name in MORPHOLOGY_ASPECTS:
            aspect: AspectResult | None = getattr(self, name)
            if aspect is None or not aspect.normal:
                return name
        return None

    def probabilities(self) -> dict[str, float | None]:
        return {
            name: (a.p_normal if a is not None else None)
            for name, a in zip(MORPHOLOGY_ASPECTS, self.aspects, strict=True)
        }

    def labels(self) -> dict[str, int | None]:
        return {
            name: (a.label if a is not None else None)
            for name, a in zip(MORPHOLOGY_ASPECTS, self.aspects, strict=True)
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "status": str(self.status),
            "frame_id": self.frame_id,
            "head": self.head.to_json_dict() if self.head else None,
            "acrosome": self.acrosome.to_json_dict() if self.acrosome else None,
            "vacuole": self.vacuole.to_json_dict() if self.vacuole else None,
            "tail": self.tail.to_json_dict() if self.tail else None,
            "all_four_normal": self.all_four_normal,
            "latency_ms": float(self.latency_ms),
            "model_id": self.model_id,
            "weights_provenance": self.weights_provenance,
            "failure_reason": self.failure_reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def failed(
        cls,
        track_id: int,
        status: MorphologyStatus,
        reason: str,
        *,
        frame_id: int | None = None,
    ) -> MorphologyResult:
        """Construct a result representing a non-completion."""
        return cls(
            track_id=track_id,
            status=status,
            frame_id=frame_id,
            failure_reason=reason,
        )
