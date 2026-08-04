"""Serial actuator.

Sends a text command to a microcontroller that owns the magnet driver. This is
the more likely production arrangement than raw GPIO: a dedicated MCU can hold
its own hardware watchdog, ramp the coil current, and report back, none of
which a bare pin can do.

The protocol is deliberately trivial and configurable -- a line out, a line
back -- so it can be matched to whatever firmware exists without changing code.

``pyserial`` is imported lazily so this module can be imported anywhere.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import ActuationConfig
from ..errors import ActuatorError
from ..schemas.enums import FieldCommandKind
from .base import MagneticActuator

logger = logging.getLogger(__name__)


class SerialActuator(MagneticActuator):
    """Line-oriented serial control of the field driver.

    Acknowledgement, when enabled, expects the firmware to echo a line
    containing the state it entered. A mismatch or a timeout is treated as a
    failure and drives the safe state, because an unverified magnet is exactly
    the situation the acknowledgement exists to prevent.
    """

    name = "serial"

    def __init__(self, cfg: ActuationConfig) -> None:
        super().__init__(cfg)
        if not cfg.serial_port:
            raise ActuatorError(
                "actuation.kind=serial requires actuation.serial_port to be set"
            )
        self._port: Any = None

    def open(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ActuatorError(
                "the pyserial package is required for the serial actuator. "
                "Install it with 'pip install sperm-sorting-ai[serial]', or "
                "use actuation.kind=mock for a bench test."
            ) from exc

        try:
            self._port = serial.Serial(
                port=self.cfg.serial_port,
                baudrate=self.cfg.serial_baudrate,
                timeout=self.cfg.serial_timeout_s,
                write_timeout=self.cfg.serial_timeout_s,
            )
        except Exception as exc:
            raise ActuatorError(
                f"could not open serial port {self.cfg.serial_port}: {exc}"
            ) from exc

        # Discard whatever the firmware said before we attached, so the first
        # acknowledgement we read belongs to a command we actually sent.
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except Exception:
            logger.debug("could not flush serial buffers", exc_info=True)

        self._open = True
        self._write(FieldCommandKind.FIELD_OFF)
        self._state = FieldCommandKind.FIELD_OFF
        logger.info(
            "serial actuator opened on %s at %d baud, field off",
            self.cfg.serial_port,
            self.cfg.serial_baudrate,
        )

    def close(self) -> None:
        if not self._open:
            return
        try:
            self.safe_state()
        finally:
            if self._port is not None:
                try:
                    self._port.close()
                except Exception:
                    logger.exception("failed to close the serial port")
                self._port = None
            self._open = False
            logger.info("serial actuator closed in the safe state")

    def _write(self, kind: FieldCommandKind) -> bool:
        if self._port is None:
            return False
        payload = (
            self.cfg.on_command
            if kind is FieldCommandKind.FIELD_ON
            else self.cfg.off_command
        )
        try:
            self._port.write(payload.encode("ascii"))
            self._port.flush()
        except Exception:
            logger.exception("serial write failed")
            return False
        return True

    def _read_acknowledgement(self) -> FieldCommandKind | None:
        if self._port is None:
            return None
        try:
            line = self._port.readline().decode("ascii", errors="replace").strip()
        except Exception:
            logger.exception("serial acknowledgement read failed")
            return None
        if not line:
            # A timeout is not "the field is fine"; report unknown and let the
            # caller decide. Returning the commanded state here would defeat
            # the entire purpose of acknowledgement.
            logger.warning("serial acknowledgement timed out")
            return None
        upper = line.upper()
        if "FIELD_ON" in upper:
            return FieldCommandKind.FIELD_ON
        if "FIELD_OFF" in upper:
            return FieldCommandKind.FIELD_OFF
        logger.warning("unrecognised serial acknowledgement: %r", line)
        return None
