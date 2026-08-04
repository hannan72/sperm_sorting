"""GPIO actuator.

Drives a single digital line that enables the magnet's power stage. Uses
``libgpiod`` v2 (the ``gpiod`` Python bindings) rather than the deprecated
sysfs interface or the Raspberry-Pi-specific ``RPi.GPIO``, because the target
board is not fixed and the character-device interface is the portable one.

The import is deferred to :meth:`open` so that the module can be imported --
and the rest of the package tested -- on a machine with no GPIO at all.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import ActuationConfig
from ..errors import ActuatorError
from ..schemas.enums import FieldCommandKind
from .base import MagneticActuator

logger = logging.getLogger(__name__)


class GpioActuator(MagneticActuator):
    """One digital output line, active-high or active-low.

    Polarity matters for safety. If the power stage is enabled by a *low*
    level, then a line that floats or is released on process exit would
    energise the magnet. :attr:`ActuationConfig.gpio_active_high` must match
    the hardware, and the line is requested with an explicit initial value of
    "inactive" so that the field is off from the instant the line is claimed.
    """

    name = "gpio"

    def __init__(self, cfg: ActuationConfig) -> None:
        super().__init__(cfg)
        if cfg.gpio_pin is None:
            raise ActuatorError(
                "actuation.kind=gpio requires actuation.gpio_pin to be set"
            )
        self._request: Any = None
        self._gpiod: Any = None

    def _level_for(self, kind: FieldCommandKind) -> int:
        """Physical level for a logical state, honouring polarity."""
        energised = kind is FieldCommandKind.FIELD_ON
        if self.cfg.gpio_active_high:
            return 1 if energised else 0
        return 0 if energised else 1

    def open(self) -> None:
        try:
            import gpiod
        except ImportError as exc:
            raise ActuatorError(
                "the gpiod package is required for the GPIO actuator. Install "
                "it with 'pip install sperm-sorting-ai[gpio]', or use "
                "actuation.kind=mock for a bench test."
            ) from exc

        self._gpiod = gpiod
        pin = int(self.cfg.gpio_pin)  # type: ignore[arg-type]
        inactive = self._level_for(FieldCommandKind.FIELD_OFF)

        try:
            settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT,
                output_value=(
                    gpiod.line.Value.ACTIVE if inactive else gpiod.line.Value.INACTIVE
                ),
            )
            self._request = gpiod.request_lines(
                self.cfg.gpio_chip,
                consumer="sperm-sorting-field",
                config={pin: settings},
            )
        except Exception as exc:
            raise ActuatorError(
                f"could not claim GPIO line {pin} on {self.cfg.gpio_chip}: {exc}"
            ) from exc

        self._open = True
        self._state = FieldCommandKind.FIELD_OFF
        logger.info(
            "GPIO actuator opened on %s line %d (active_%s), field off",
            self.cfg.gpio_chip,
            pin,
            "high" if self.cfg.gpio_active_high else "low",
        )

    def close(self) -> None:
        if not self._open:
            return
        try:
            self.safe_state()
        finally:
            if self._request is not None:
                try:
                    self._request.release()
                except Exception:
                    logger.exception("failed to release the GPIO line")
                self._request = None
            self._open = False
            logger.info("GPIO actuator closed in the safe state")

    def _write(self, kind: FieldCommandKind) -> bool:
        if self._request is None or self._gpiod is None:
            return False
        level = self._level_for(kind)
        value = (
            self._gpiod.line.Value.ACTIVE if level else self._gpiod.line.Value.INACTIVE
        )
        try:
            self._request.set_value(int(self.cfg.gpio_pin), value)  # type: ignore[arg-type]
        except Exception:
            logger.exception("GPIO write failed")
            return False
        return True

    def _read_acknowledgement(self) -> FieldCommandKind | None:
        """Read the line back.

        This confirms the *line* is where we set it, which is weaker than
        confirming the *field* is where we want it -- nothing downstream of
        the pin is verified. A production build should sense the coil current
        instead; see docs/assumptions.md.
        """
        if self._request is None or self._gpiod is None:
            return None
        try:
            value = self._request.get_value(int(self.cfg.gpio_pin))  # type: ignore[arg-type]
        except Exception:
            return None
        level = 1 if value == self._gpiod.line.Value.ACTIVE else 0
        energised = (level == 1) if self.cfg.gpio_active_high else (level == 0)
        return FieldCommandKind.FIELD_ON if energised else FieldCommandKind.FIELD_OFF
