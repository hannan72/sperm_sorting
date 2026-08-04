"""Multi-task morphology network: one trunk, four independent binary heads.

Label polarity
--------------
**Every logit this module produces is a logit for P(abnormal)**, so the
training target is the MHSMA integer label verbatim and no flip appears
anywhere in this file. The full contract, and the single function that performs
the one permitted flip, live in :mod:`sperm_sorting.morphology.polarity`.
:data:`~sperm_sorting.morphology.polarity.POLARITY_CONVENTION` is written into
every checkpoint and :func:`load_checkpoint` refuses to load a checkpoint whose
string differs.

Why one trunk and four heads
----------------------------
The four aspects are different judgements about the same object: head shape,
acrosome, vacuole and tail all depend on the same low-level texture and
silhouette features, so sharing the trunk is a 4x saving on the expensive part
of the forward pass -- which matters because morphology runs inside a per-track
deadline. But the aspects are *not* jointly labelled classes: a sperm can be
abnormal in any subset of the four, so this is four binary problems, not one
four-way classification, and each head gets its own logit, its own
``pos_weight`` and its own decision threshold. Nothing in this package ever
averages the four probabilities; the conjunctive rule lives in
:attr:`sperm_sorting.schemas.morphology.MorphologyResult.all_four_normal`.

``forward`` returns a ``dict[str, Tensor]`` keyed by aspect name rather than a
``(B, 4)`` tensor. The dict costs a negligible amount per call and makes an
ordering bug -- the worst kind of bug available here, because it silently swaps
the 4.6%-prevalence tail head with the 27%-prevalence head head -- impossible to
write. :meth:`MultiTaskMorphologyNet.logits_tensor` provides the packed form
for the loss and for ONNX export, always in
:data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS` order.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..constants import MORPHOLOGY_ASPECTS
from ..errors import InferenceError
from .backbones import build_backbone
from .polarity import POLARITY_CONVENTION

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "AspectHead",
    "MorphologyLoss",
    "MultiTaskMorphologyNet",
    "export_onnx",
    "load_checkpoint",
    "pos_weight_from_prevalence",
    "save_checkpoint",
]

#: Bumped when the checkpoint dict gains or re-interprets a key.
CHECKPOINT_FORMAT_VERSION: str = "1"


# ==========================================================================
# Network
# ==========================================================================


class AspectHead(nn.Module):
    """Small MLP mapping the pooled trunk feature to one abnormality logit.

    Deliberately shallow (one hidden layer). With only ~1000 MHSMA training
    images per aspect and severe imbalance, a deeper head memorises the
    minority class rather than learning it; the capacity belongs in the shared
    trunk, where all four tasks' gradients regularise it.
    """

    def __init__(
        self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return ``(B,)`` logits for ``P(abnormal)``."""
        return self.net(features).squeeze(-1)


class MultiTaskMorphologyNet(nn.Module):
    """Shared backbone plus four independent binary heads.

    Parameters
    ----------
    backbone
        Name accepted by :func:`~.backbones.build_backbone`.
    in_channels
        1 for the monochrome microscope.
    hidden_dim, dropout
        Per-head MLP shape.
    pretrained
        Passed to the backbone builder.
    input_size
        ``(h, w)`` the model is *intended* to receive. Recorded in the
        checkpoint and in :meth:`describe` so that a crop-size mismatch is
        visible in the audit log; the network itself tolerates other sizes
        because the backbone global-average-pools.
    aspects
        Head names, in canonical order. Overridable only for tests; production
        always uses :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS`.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv3_small",
        *,
        in_channels: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
        input_size: tuple[int, int] = (128, 128),
        aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS,
        backbone_width: int = 32,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.aspects = tuple(aspects)
        self.backbone_width = int(backbone_width)
        #: Restated on the instance so ``model.label_polarity`` is inspectable
        #: from a debugger or a notebook without importing anything.
        self.label_polarity = POLARITY_CONVENTION

        trunk, feature_dim = build_backbone(
            backbone,
            in_channels=in_channels,
            pretrained=pretrained,
            width=backbone_width,
        )
        self.backbone = trunk
        self.feature_dim = int(feature_dim)
        # ModuleDict keeps the heads addressable by name in the state dict, so a
        # checkpoint remains readable and a head can be reset independently.
        self.heads = nn.ModuleDict(
            {
                name: AspectHead(self.feature_dim, hidden_dim, dropout)
                for name in self.aspects
            }
        )

    # ------------------------------------------------------------------ forward

    def features(self, x: Tensor) -> Tensor:
        """Pooled trunk features, ``(B, feature_dim)``. Useful for embeddings."""
        return self.backbone(x)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Per-aspect ``P(abnormal)`` logits, ``(B,)`` each, keyed by aspect."""
        feats = self.features(x)
        return {name: head(feats) for name, head in self.heads.items()}

    def logits_tensor(self, x: Tensor) -> Tensor:
        """Packed ``(B, n_aspects)`` logits in :attr:`aspects` order.

        Used by the loss and by ONNX export. Any consumer of this tensor must
        index it with :attr:`aspects`, never with a hard-coded integer.
        """
        out = self(x)
        return torch.stack([out[name] for name in self.aspects], dim=1)

    # ------------------------------------------------------------------- admin

    @torch.no_grad()
    def predict_abnormal(self, x: Tensor) -> dict[str, Tensor]:
        """Per-aspect ``P(abnormal)`` probabilities. **Not** ``P(normal)``.

        Named for what it returns so that a caller who wants ``p_normal``
        cannot mistake this for it. The flip lives in
        :func:`~.polarity.flip_polarity` and is applied by
        :class:`~.inference.MorphologyEngine`.
        """
        return {name: torch.sigmoid(logit) for name, logit in self(x).items()}

    def config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this network."""
        return {
            "backbone": self.backbone_name,
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "input_size": list(self.input_size),
            "aspects": list(self.aspects),
            "backbone_width": self.backbone_width,
        }

    def describe(self) -> dict[str, Any]:
        """Metadata for the audit-log header."""
        return {
            **self.config(),
            "feature_dim": self.feature_dim,
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "label_polarity": self.label_polarity,
        }


# ==========================================================================
# Loss
# ==========================================================================


def pos_weight_from_prevalence(prevalence: dict[str, float]) -> dict[str, float]:
    """``pos_weight`` per aspect from the abnormal prevalence of the train split.

    ``pos_weight = (1 - p) / p`` makes the positive (abnormal) class contribute
    the same total gradient mass as the negative class, which is the standard
    remedy when the minority class is the one you actually care about.

    Verified MHSMA train prevalences are acrosome 0.301, head 0.273,
    vacuole 0.170, tail 0.046 -- so the tail head needs ``pos_weight`` near 20
    while the acrosome head needs about 2.3. A single shared weight would either
    ignore the tail or drown the acrosome, which is why this is per aspect.

    Raises
    ------
    ValueError
        If a prevalence is not strictly inside ``(0, 1)``; a prevalence of 0
        means the split contains no positives at all and no weight can rescue
        it -- that must be fixed in the split, not in the loss.
    """
    out: dict[str, float] = {}
    for name, p in prevalence.items():
        if not 0.0 < p < 1.0:
            raise ValueError(
                f"prevalence for aspect '{name}' must lie in (0, 1), got {p}; a "
                "split with no positive examples for an aspect cannot be "
                "trained or evaluated on that aspect"
            )
        out[name] = (1.0 - p) / p
    return out


class MorphologyLoss(nn.Module):
    """Sum of four independent BCE-with-logits terms, one per aspect.

    Summed rather than averaged so that the reported number is comparable to
    the sum of the per-aspect logs, and so that adding a fifth aspect later does
    not silently rescale the learning rate of the existing four.

    Parameters
    ----------
    aspects
        Head names; must match the network's.
    pos_weight
        Per-aspect weight on the positive (abnormal) class. See
        :func:`pos_weight_from_prevalence`.
    aspect_weights
        Optional fixed per-aspect scaling applied on top, for the case where
        one aspect is known to be more label-noisy than the others.
    uncertainty_weighting
        Learn the per-aspect balance instead of fixing it, following the
        homoscedastic-uncertainty formulation of Kendall et al.: with
        ``s_i = log sigma_i^2`` as free parameters the total becomes
        ``sum_i exp(-s_i) * L_i + s_i``. The ``+ s_i`` term is what stops the
        optimiser from driving every weight to zero. Useful when the four
        aspects converge at very different rates (they do -- tail is by far the
        hardest), but it is off by default because it adds four parameters
        whose values must then be reported alongside the metrics for the run to
        be reproducible.

    Notes
    -----
    Targets are the **MHSMA labels verbatim** (``0 = normal``, ``1 = abnormal``).
    There is no flip in this class; see :mod:`.polarity`.
    """

    def __init__(
        self,
        aspects: tuple[str, ...] = MORPHOLOGY_ASPECTS,
        *,
        pos_weight: dict[str, float] | None = None,
        aspect_weights: dict[str, float] | None = None,
        uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.aspects = tuple(aspects)
        self.uncertainty_weighting = bool(uncertainty_weighting)
        self.aspect_weights = {
            name: float((aspect_weights or {}).get(name, 1.0)) for name in self.aspects
        }
        for name in self.aspects:
            weight = float((pos_weight or {}).get(name, 1.0))
            if weight <= 0.0:
                raise ValueError(
                    f"pos_weight['{name}'] must be positive, got {weight}"
                )
            # Buffers so that .to(device) and state_dict round-trip carry them.
            self.register_buffer(
                f"pos_weight_{name}", torch.tensor(weight, dtype=torch.float32)
            )
        if self.uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(len(self.aspects)))
        else:
            self.register_parameter("log_vars", None)
        self._last_per_aspect: dict[str, float] = {}

    # ------------------------------------------------------------------ compute

    def _as_dict(
        self, value: dict[str, Tensor] | Tensor, what: str
    ) -> dict[str, Tensor]:
        """Accept either a per-aspect dict or a packed ``(B, n_aspects)`` tensor."""
        if isinstance(value, dict):
            missing = [n for n in self.aspects if n not in value]
            if missing:
                raise ValueError(f"{what} is missing aspect(s): {', '.join(missing)}")
            return {name: value[name] for name in self.aspects}
        if value.ndim != 2 or value.shape[1] != len(self.aspects):
            raise ValueError(
                f"packed {what} must have shape (B, {len(self.aspects)}); got "
                f"{tuple(value.shape)}"
            )
        return {name: value[:, i] for i, name in enumerate(self.aspects)}

    def per_aspect_losses(
        self,
        logits: dict[str, Tensor] | Tensor,
        targets: dict[str, Tensor] | Tensor,
    ) -> dict[str, Tensor]:
        """Differentiable per-aspect BCE terms, before any task weighting.

        Exposed so a training loop can log four curves instead of one: with
        prevalences spanning 27% to 4.6%, a falling total tells you almost
        nothing about whether the tail head is learning.
        """
        logit_map = self._as_dict(logits, "logits")
        target_map = self._as_dict(targets, "targets")
        out: dict[str, Tensor] = {}
        for name in self.aspects:
            logit = logit_map[name]
            target = target_map[name].to(dtype=logit.dtype)
            if target.shape != logit.shape:
                raise ValueError(
                    f"aspect '{name}': target shape {tuple(target.shape)} does not "
                    f"match logit shape {tuple(logit.shape)}"
                )
            out[name] = nn.functional.binary_cross_entropy_with_logits(
                logit,
                target,
                pos_weight=getattr(self, f"pos_weight_{name}"),
            )
        return out

    def forward(
        self,
        logits: dict[str, Tensor] | Tensor,
        targets: dict[str, Tensor] | Tensor,
    ) -> Tensor:
        """Total loss. Per-aspect values are cached in :attr:`last_per_aspect_losses`."""
        losses = self.per_aspect_losses(logits, targets)
        terms: list[Tensor] = []
        for index, name in enumerate(self.aspects):
            term = losses[name] * self.aspect_weights[name]
            if self.uncertainty_weighting and self.log_vars is not None:
                s = self.log_vars[index]
                term = torch.exp(-s) * term + s
            terms.append(term)
        total = torch.stack(terms).sum()
        self._last_per_aspect = {
            name: float(losses[name].detach()) for name in self.aspects
        }
        return total

    @property
    def last_per_aspect_losses(self) -> dict[str, float]:
        """Detached per-aspect losses from the most recent :meth:`forward`."""
        return dict(self._last_per_aspect)

    @property
    def task_weights(self) -> dict[str, float]:
        """Effective per-aspect weight, including learned uncertainty terms."""
        if not self.uncertainty_weighting or self.log_vars is None:
            return dict(self.aspect_weights)
        with torch.no_grad():
            scale = torch.exp(-self.log_vars)
        return {
            name: self.aspect_weights[name] * float(scale[i])
            for i, name in enumerate(self.aspects)
        }


# ==========================================================================
# Checkpoints
# ==========================================================================


def save_checkpoint(
    model: MultiTaskMorphologyNet,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    model_id: str = "",
    weights_provenance: str = "",
) -> Path:
    """Write a self-describing checkpoint.

    The dict carries everything needed to rebuild the network *and* everything
    needed to prove it is being interpreted correctly:

    ``state_dict``, ``backbone``, ``input_size``, ``aspects``,
    ``weights_provenance``, ``model_id`` and ``label_polarity``.

    ``label_polarity`` is the important one. A checkpoint trained under the
    opposite convention would load cleanly, run at full speed and invert every
    decision the product makes; :func:`load_checkpoint` compares the string and
    raises rather than allowing that.

    ``metadata`` is serialised to JSON so the whole file loads under
    ``torch.load(..., weights_only=True)``. That is a security property, not a
    style choice: checkpoints are the kind of artefact that gets copied between
    machines, and ``weights_only=True`` means loading one cannot execute
    arbitrary pickled code. Non-JSON-serialisable metadata values are stringified.

    The write is done to a temporary file and renamed, so an interrupted save
    cannot leave a truncated checkpoint that fails half-way through a later load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "label_polarity": POLARITY_CONVENTION,
        "state_dict": model.state_dict(),
        "model_config": model.config(),
        "backbone": model.backbone_name,
        "input_size": list(model.input_size),
        "aspects": list(model.aspects),
        "feature_dim": model.feature_dim,
        "model_id": str(model_id),
        "weights_provenance": str(weights_provenance),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        # str() matters: torch.__version__ is a TorchVersion object, which is
        # not on torch.load's weights_only allow-list and would make the whole
        # checkpoint unreadable without disabling that protection.
        "torch_version": str(torch.__version__),
        "metadata_json": json.dumps(metadata or {}, sort_keys=True, default=str),
    }

    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_polarity: str = POLARITY_CONVENTION,
    strict: bool = True,
) -> tuple[MultiTaskMorphologyNet, dict[str, Any]]:
    """Rebuild a network from a checkpoint, refusing a polarity mismatch.

    Returns ``(model, info)`` where ``info`` holds every non-tensor key of the
    checkpoint plus the decoded ``metadata`` dict. The model is returned in
    ``eval()`` mode, because a checkpoint is loaded for inference far more often
    than for resuming training and a forgotten ``.eval()`` silently changes the
    answer (dropout, BatchNorm).

    Raises
    ------
    InferenceError
        If the file is missing, malformed, has no recorded polarity, or records
        a polarity different from ``expected_polarity``. A checkpoint with no
        polarity string is treated as a mismatch rather than assumed
        compatible: pre-convention checkpoints are exactly the ones that might
        be inverted.
    """
    path = Path(path)
    if not path.exists():
        raise InferenceError(f"morphology checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise InferenceError(f"could not read morphology checkpoint {path}: {exc}") from exc

    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise InferenceError(
            f"{path} is not a morphology checkpoint written by save_checkpoint "
            "(no 'state_dict' key)"
        )

    recorded = payload.get("label_polarity")
    if recorded != expected_polarity:
        raise InferenceError(
            "morphology checkpoint label-polarity mismatch -- refusing to load.\n"
            f"  checkpoint: {path}\n"
            f"  recorded:   {recorded!r}\n"
            f"  expected:   {expected_polarity!r}\n"
            "Loading it would invert every morphology decision without any "
            "visible error. Re-export the checkpoint under the current "
            "convention (see sperm_sorting.morphology.polarity) instead of "
            "relaxing this check."
        )

    config = dict(payload.get("model_config") or {})
    config.pop("aspects", None)  # restored from the top-level key below
    input_size = payload.get("input_size") or config.pop("input_size", (128, 128))
    config.pop("input_size", None)
    aspects = tuple(payload.get("aspects") or MORPHOLOGY_ASPECTS)

    model = MultiTaskMorphologyNet(
        backbone=str(payload.get("backbone", config.pop("backbone", "mobilenetv3_small"))),
        in_channels=int(config.pop("in_channels", 1)),
        hidden_dim=int(config.pop("hidden_dim", 128)),
        dropout=float(config.pop("dropout", 0.2)),
        pretrained=False,  # weights come from the checkpoint, never the network
        input_size=(int(input_size[0]), int(input_size[1])),
        aspects=aspects,
        backbone_width=int(config.pop("backbone_width", 32)),
    )
    try:
        model.load_state_dict(payload["state_dict"], strict=strict)
    except Exception as exc:
        raise InferenceError(
            f"morphology checkpoint {path} does not fit the reconstructed "
            f"architecture: {exc}"
        ) from exc
    model.eval()

    info = {k: v for k, v in payload.items() if k != "state_dict"}
    try:
        info["metadata"] = json.loads(payload.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        info["metadata"] = {}
    return model, info


# ==========================================================================
# Export
# ==========================================================================


def export_onnx(
    model: MultiTaskMorphologyNet,
    path: str | Path,
    *,
    batch_size: int = 1,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export the packed ``(B, n_aspects)`` logits to ONNX.

    The packed form is exported rather than four named outputs because ONNX
    Runtime's output ordering is then unambiguous: column ``i`` is
    ``model.aspects[i]``, which is
    :data:`~sperm_sorting.constants.MORPHOLOGY_ASPECTS` order.
    :class:`~.inference.MorphologyEngine` relies on exactly that.

    The exported graph emits **logits for P(abnormal)**, matching the torch
    path, so the same single flip applies to both backends.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class _Packed(nn.Module):
        def __init__(self, inner: MultiTaskMorphologyNet) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Tensor) -> Tensor:
            """Packed ``(B, n_aspects)`` P(abnormal) logits, in aspect order."""
            return self.inner.logits_tensor(x)

    wrapper = _Packed(model).eval()
    height, width = model.input_size
    dummy = torch.zeros(batch_size, model.in_channels, height, width)
    dynamic_axes = {"crops": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["crops"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return path
