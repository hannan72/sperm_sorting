"""Mixed precision, enabled only where it helps.

``torch.amp`` autocast on CPU exists, but for the small convolutional models
here it is at best neutral and usually a slowdown: bf16 CPU kernels for
depthwise and grouped convolutions fall back to fp32 with an extra pair of
casts around them. So AMP is gated on the device actually being CUDA, and a
request for AMP on CPU is honoured by *reporting* that it was ignored rather
than by silently pretending it was on. A latency figure recorded under
"amp=true" that was actually fp32 is a fabricated measurement.

The gradient scaler is only meaningful with fp16. bf16 has fp32's exponent
range, so gradients do not underflow and the scaler is a no-op; constructing
one anyway costs a per-step ``inf`` check on every gradient tensor.
:class:`AmpContext` therefore builds a scaler only for fp16 on CUDA.

The class deliberately exposes the *same* call sequence in every mode, so the
training loop has no ``if amp:`` branches -- the one place a mixed-precision
bug reliably hides is in the branch that only runs when AMP is off.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

__all__ = ["AmpContext"]


class AmpContext:
    """Autocast plus gradient scaling, or an exact no-op.

    Usage is identical in both cases::

        amp = AmpContext(device, enabled=args.amp)
        with amp.autocast():
            loss = criterion(model(x), y)
        amp.backward(loss)
        amp.step(optimizer, clip_grad_norm=1.0, parameters=model.parameters())

    Parameters
    ----------
    device
        The ``torch.device`` training will run on.
    enabled
        Whether AMP was *requested*. Whether it is actually active is
        :attr:`enabled` after construction, which may be ``False`` with
        :attr:`reason` explaining why.
    dtype
        ``"float16"`` or ``"bfloat16"``. fp16 is the default because it is what
        tensor cores accelerate on every CUDA generation this would run on;
        bf16 is offered for Ampere and later where its wider exponent removes
        the need for loss scaling altogether.
    """

    def __init__(self, device: Any, *, enabled: bool = True, dtype: str = "float16") -> None:
        import torch

        self._torch = torch
        self.device_type = getattr(device, "type", str(device))
        self.requested = bool(enabled)
        self.reason = ""
        self.dtype_name = str(dtype)

        if not self.requested:
            self.enabled = False
            self.reason = "not requested"
        elif self.device_type != "cuda":
            self.enabled = False
            self.reason = (
                f"device is '{self.device_type}', not CUDA; CPU autocast does not "
                "accelerate these models and would make any latency figure "
                "misleading"
            )
        else:
            self.enabled = True

        self.dtype = torch.float16 if self.dtype_name == "float16" else torch.bfloat16
        if self.dtype_name not in ("float16", "bfloat16"):
            raise ValueError(f"amp dtype must be 'float16' or 'bfloat16', got {dtype!r}")

        # Only fp16 underflows; bf16 keeps fp32's exponent range.
        use_scaler = self.enabled and self.dtype is torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        """Autocast region, or a null context when AMP is off."""
        if not self.enabled:
            yield
            return
        with self._torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            yield

    def backward(self, loss: Any) -> None:
        """Scale (when scaling) and back-propagate."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(
        self,
        optimizer: Any,
        *,
        clip_grad_norm: float | None = None,
        parameters: Any = None,
    ) -> float | None:
        """Unscale, clip, step and update the scaler. Returns the grad norm.

        Order matters and is the reason this is a method rather than three
        lines in the loop: gradients must be **unscaled before clipping**, or
        the clip threshold is applied to gradients inflated by the loss scale
        and effectively does nothing. Returning the measured norm lets the
        caller log it, which is how a clipping threshold gets chosen from
        evidence instead of from habit.
        """
        grad_norm: float | None = None

        if self.scaler is not None:
            if clip_grad_norm is not None and parameters is not None:
                self.scaler.unscale_(optimizer)
                grad_norm = float(
                    self._torch.nn.utils.clip_grad_norm_(parameters, float(clip_grad_norm))
                )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if clip_grad_norm is not None and parameters is not None:
                grad_norm = float(
                    self._torch.nn.utils.clip_grad_norm_(parameters, float(clip_grad_norm))
                )
            optimizer.step()
        return grad_norm

    def state_dict(self) -> dict[str, Any] | None:
        """Scaler state for the checkpoint, or ``None`` when there is no scaler."""
        return self.scaler.state_dict() if self.scaler is not None else None

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore the scaler's loss scale on resume."""
        if self.scaler is not None and state is not None:
            self.scaler.load_state_dict(state)

    def to_json_dict(self) -> dict[str, Any]:
        """What AMP actually did, for the experiment record."""
        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "reason": self.reason,
            "dtype": self.dtype_name if self.enabled else None,
            "grad_scaler": self.scaler is not None,
            "device_type": self.device_type,
        }
