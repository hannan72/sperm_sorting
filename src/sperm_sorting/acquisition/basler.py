"""Basler camera frame source (pypylon).

Targets the a2A1920-160umPRO: monochrome, global shutter, 1936 x 1216 native,
3.45 um pixels, USB 3.0, roughly 160-164 FPS at the default ROI.

Three things here are easy to get wrong and expensive to discover later.

**Grab strategy.** ``OneByOne`` is used, not ``LatestImageOnly``. Tracking
reconstructs a trajectory from consecutive observations, so a silently
discarded frame fragments tracks and corrupts every velocity derived from
them. Under load we would rather see reported drops than invisible ones.

**Timestamps.** The camera's own tick counter is the only timestamp that
reflects when the exposure actually started; a host timestamp taken at grab
time includes USB transfer and scheduling jitter. On ace 2 the device clock
runs at 1 GHz, so one tick is one nanosecond. The chunk timestamp is preferred
(Basler recommends ``BslChunkTimestampValue`` on this family), with the
grab-result tick as a fallback and the host clock as a last resort -- and the
packet records which one it got, because that changes how much the velocity
can be trusted.

**Bandwidth.** 1920 x 1200 Mono8 at 164 FPS is about 378 MB/s, which is at the
practical ceiling of USB 3.0. Mono12 at full rate is not achievable. If the
link cannot sustain the configured mode, the camera reports skipped images and
this module surfaces them rather than letting them vanish.

``pypylon`` is imported lazily so the package imports on machines with no
camera and no SDK.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import numpy as np

from ..config import BaslerConfig
from ..errors import CameraError
from ..schemas.enums import SourceKind, TimestampSource
from ..schemas.frame import FramePacket
from .base import FrameSource

logger = logging.getLogger(__name__)

#: ace 2 device clock; one tick is one nanosecond.
_DEFAULT_TICK_HZ = 1e9


class BaslerFrameSource(FrameSource):
    """Live acquisition from a Basler USB3 camera."""

    kind = SourceKind.BASLER

    def __init__(self, cfg: BaslerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._pylon: Any = None
        self._camera: Any = None
        self._tick_hz: float = cfg.timestamp_tick_frequency_hz or _DEFAULT_TICK_HZ
        self._timestamp_source = TimestampSource.HOST_MONOTONIC
        self._chunk_parameter: str | None = None
        #: Camera tick and host time captured together, so camera ticks can be
        #: mapped onto the host monotonic timeline the rest of the system uses.
        self._tick_origin: int | None = None
        self._host_origin: float = 0.0
        self.n_skipped_total = 0

    # ------------------------------------------------------------------ open

    def _import_pylon(self) -> Any:
        try:
            from pypylon import pylon
        except ImportError as exc:
            raise CameraError(
                "pypylon is required for live Basler acquisition. Install it "
                "with 'pip install sperm-sorting-ai[camera]', or use "
                "acquisition.kind=video / synthetic."
            ) from exc
        return pylon

    def _create_device(self, pylon: Any) -> Any:
        factory = pylon.TlFactory.GetInstance()
        if self.cfg.serial_number:
            devices = factory.EnumerateDevices()
            for info in devices:
                if info.GetSerialNumber() == self.cfg.serial_number:
                    return factory.CreateDevice(info)
            found = ", ".join(d.GetSerialNumber() for d in devices) or "none"
            raise CameraError(
                f"no Basler camera with serial {self.cfg.serial_number!r}; "
                f"found: {found}"
            )
        # CreateFirstDevice works across pypylon 4.x and 26.x; the newer
        # pylon.FirstFound shortcut does not exist on 4.x.
        try:
            return factory.CreateFirstDevice()
        except Exception as exc:
            raise CameraError(f"no Basler camera found: {exc}") from exc

    @staticmethod
    def _set(node: Any, value: Any, name: str) -> None:
        """Assign through the ``.Value`` API, tolerating absent nodes.

        Direct attribute assignment is deprecated in current pypylon, and not
        every node exists on every model, so a missing one is logged rather
        than fatal.
        """
        try:
            node.Value = value
        except Exception as exc:
            logger.warning("could not set %s=%r: %s", name, value, exc)

    def open(self) -> None:
        pylon = self._import_pylon()
        self._pylon = pylon
        camera = pylon.InstantCamera(self._create_device(pylon))
        try:
            camera.Open()
        except Exception as exc:
            raise CameraError(f"could not open the camera: {exc}") from exc
        self._camera = camera

        info = camera.GetDeviceInfo()
        logger.info(
            "opened %s (serial %s)", info.GetModelName(), info.GetSerialNumber()
        )

        try:
            self._configure(camera)
        except Exception as exc:
            self.close()
            raise CameraError(f"camera configuration failed: {exc}") from exc

        strategy = (
            pylon.GrabStrategy_OneByOne
            if self.cfg.grab_strategy == "OneByOne"
            else pylon.GrabStrategy_LatestImageOnly
        )
        if self.cfg.grab_strategy != "OneByOne":
            logger.warning(
                "grab strategy is %s, which discards frames under load. "
                "Trajectories will fragment and velocities will be unreliable. "
                "Use OneByOne for any run that measures motion.",
                self.cfg.grab_strategy,
            )
        camera.StartGrabbing(strategy)

        self._open = True
        self._session_id += 1
        self._host_origin = time.monotonic()
        self._tick_origin = None

    def _configure(self, camera: Any) -> None:
        cfg = self.cfg

        if cfg.binning > 1:
            self._set(camera.BinningHorizontal, cfg.binning, "BinningHorizontal")
            self._set(camera.BinningVertical, cfg.binning, "BinningVertical")

        # Offsets are zeroed first: a large existing offset can make the new
        # width illegal, since OffsetX + Width must not exceed WidthMax.
        self._set(camera.OffsetX, 0, "OffsetX")
        self._set(camera.OffsetY, 0, "OffsetY")
        self._set(camera.Width, cfg.width, "Width")
        self._set(camera.Height, cfg.height, "Height")
        self._set(camera.OffsetX, cfg.offset_x, "OffsetX")
        self._set(camera.OffsetY, cfg.offset_y, "OffsetY")
        self._set(camera.PixelFormat, cfg.pixel_format, "PixelFormat")
        self._set(camera.ExposureTime, float(cfg.exposure_time_us), "ExposureTime")
        self._set(camera.Gain, float(cfg.gain_db), "Gain")

        if cfg.acquisition_frame_rate is not None:
            self._set(
                camera.AcquisitionFrameRateEnable, True, "AcquisitionFrameRateEnable"
            )
            self._set(
                camera.AcquisitionFrameRate,
                float(cfg.acquisition_frame_rate),
                "AcquisitionFrameRate",
            )
        else:
            self._set(
                camera.AcquisitionFrameRateEnable, False, "AcquisitionFrameRateEnable"
            )

        try:
            camera.MaxNumBuffer.Value = cfg.max_num_buffer
        except Exception:
            logger.debug("MaxNumBuffer is not settable", exc_info=True)

        try:
            achievable = camera.BslResultingAcquisitionFrameRate.Value
            logger.info("camera reports %.2f FPS achievable", achievable)
            if (
                cfg.acquisition_frame_rate is not None
                and achievable < cfg.acquisition_frame_rate * 0.95
            ):
                logger.warning(
                    "requested %.1f FPS but the camera can only achieve %.1f "
                    "with this ROI, pixel format and exposure. Reduce the ROI "
                    "height, shorten the exposure, or enable binning.",
                    cfg.acquisition_frame_rate,
                    achievable,
                )
        except Exception:
            logger.debug("BslResultingAcquisitionFrameRate unavailable", exc_info=True)

        if cfg.enable_chunk_timestamp:
            self._enable_timestamp_chunk(camera)

        self._detect_tick_frequency(camera)

    def _enable_timestamp_chunk(self, camera: Any) -> None:
        """Turn on the timestamp chunk, probing for the right parameter name."""
        try:
            camera.ChunkModeActive.Value = True
        except Exception:
            logger.warning(
                "chunk mode is unavailable; falling back to the grab-result "
                "tick for timestamps"
            )
            return

        try:
            settable = set(camera.ChunkSelector.GetSettableValues())
        except Exception:
            settable = set()

        if "Timestamp" in settable or not settable:
            try:
                camera.ChunkSelector.Value = "Timestamp"
                camera.ChunkEnable.Value = True
            except Exception:
                logger.warning("could not enable the Timestamp chunk", exc_info=True)
                return

        # Basler recommends BslChunkTimestampValue on ace 2; ChunkTimestamp is
        # the legacy name and is still present. Prefer the configured one and
        # fall back.
        for name in (self.cfg.chunk_timestamp_parameter, "ChunkTimestamp"):
            try:
                if hasattr(camera, "BslChunkTimestampSelector"):
                    camera.BslChunkTimestampSelector.Value = (
                        self.cfg.chunk_timestamp_selector
                    )
            except Exception:
                logger.debug("BslChunkTimestampSelector not settable", exc_info=True)
            self._chunk_parameter = name
            break

        # Pre-allocate the chunk node-map pool: constructing it per frame is a
        # real cost at 160 FPS.
        try:
            camera.StaticChunkNodeMapPoolSize.Value = camera.MaxNumBuffer.Value
        except Exception:
            logger.debug("StaticChunkNodeMapPoolSize not settable", exc_info=True)

    def _detect_tick_frequency(self, camera: Any) -> None:
        if self.cfg.timestamp_tick_frequency_hz is not None:
            self._tick_hz = float(self.cfg.timestamp_tick_frequency_hz)
            return
        for name in ("GevTimestampTickFrequency", "TimestampTickFrequency"):
            try:
                value = float(getattr(camera, name).Value)
                if value > 0:
                    self._tick_hz = value
                    logger.info("camera tick frequency: %.0f Hz", value)
                    return
            except Exception:
                continue
        self._tick_hz = _DEFAULT_TICK_HZ
        logger.info("assuming a %.0f Hz tick (ace 2 default: 1 ns)", self._tick_hz)

    # ------------------------------------------------------------------ read

    def _extract_ticks(self, result: Any) -> int | None:
        """Camera tick for this frame, preferring the chunk over the result."""
        if self._chunk_parameter is not None:
            try:
                param = self._pylon.IntegerParameter(
                    result.ChunkDataNodeMap, self._chunk_parameter
                )
                if param.IsReadable():
                    return int(param.Value)
            except Exception:
                pass
        try:
            ticks = int(result.TimeStamp)
            # GetTimeStamp() returns zero on cameras without the feature, so
            # zero means "unsupported", not "time zero".
            if ticks > 0:
                return ticks
        except Exception:
            pass
        return None

    def read(self) -> FramePacket | None:
        if self._camera is None or not self._open:
            raise CameraError("camera is not open")

        try:
            result = self._camera.RetrieveResult(
                self.cfg.grab_timeout_ms, self._pylon.TimeoutHandling_ThrowException
            )
        except Exception as exc:
            raise CameraError(f"grab failed or timed out: {exc}") from exc

        try:
            if not result.GrabSucceeded():
                code = getattr(result, "ErrorCode", "?")
                desc = getattr(result, "ErrorDescription", "")
                raise CameraError(f"grab error {code}: {desc}")

            image = np.array(result.Array, copy=True)
            host_now = time.monotonic()

            skipped = int(getattr(result, "NumberOfSkippedImages", 0) or 0)
            if skipped:
                self.n_skipped_total += skipped
                self.n_dropped += skipped
                logger.warning(
                    "camera skipped %d frame(s) before frame %d; the link or "
                    "the host cannot keep up",
                    skipped,
                    self._frame_id,
                )

            ticks = self._extract_ticks(result)
            if ticks is not None:
                if self._tick_origin is None:
                    self._tick_origin = ticks
                    self._host_origin = host_now
                # Map camera ticks onto the host monotonic timeline, anchored
                # at the first frame. Relative intervals stay camera-accurate,
                # which is what velocity needs, while the absolute value stays
                # comparable with every other clock in the system.
                capture_time = self._host_origin + (ticks - self._tick_origin) / self._tick_hz
                ts_source = TimestampSource.HARDWARE
            else:
                capture_time = host_now
                ts_source = TimestampSource.HOST_MONOTONIC

            if ts_source is not self._timestamp_source:
                self._timestamp_source = ts_source
                if ts_source is TimestampSource.HOST_MONOTONIC:
                    logger.warning(
                        "no hardware timestamp available; using the host clock. "
                        "Velocities will carry USB and scheduling jitter."
                    )

            block_id = int(getattr(result, "BlockID", self._frame_id) or self._frame_id)

            packet = FramePacket(
                frame_id=self._frame_id,
                image=image,
                capture_time_s=capture_time,
                timestamp_source=ts_source,
                source_kind=self.kind,
                received_time_s=host_now,
                dropped_before=skipped,
                session_id=self._session_id,
                meta={
                    "block_id": block_id,
                    "exposure_us": self.cfg.exposure_time_us,
                    "gain_db": self.cfg.gain_db,
                    "pixel_format": self.cfg.pixel_format,
                    "camera_ticks": ticks,
                },
            )
            self._frame_id += 1
            return packet
        finally:
            with contextlib.suppress(Exception):
                result.Release()

    # ----------------------------------------------------------------- close

    def close(self) -> None:
        if self._camera is not None:
            try:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
            except Exception:
                logger.debug("StopGrabbing failed", exc_info=True)
            try:
                if self._camera.IsOpen():
                    self._camera.Close()
            except Exception:
                logger.debug("Close failed", exc_info=True)
            self._camera = None
        self._open = False

    @property
    def nominal_fps(self) -> float | None:
        return self.cfg.acquisition_frame_rate

    def describe(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "serial_number": self.cfg.serial_number,
            "roi": [self.cfg.offset_x, self.cfg.offset_y, self.cfg.width, self.cfg.height],
            "binning": self.cfg.binning,
            "pixel_format": self.cfg.pixel_format,
            "exposure_us": self.cfg.exposure_time_us,
            "gain_db": self.cfg.gain_db,
            "requested_fps": self.cfg.acquisition_frame_rate,
            "grab_strategy": self.cfg.grab_strategy,
            "timestamp_source": str(self._timestamp_source),
            "tick_frequency_hz": self._tick_hz,
            "chunk_parameter": self._chunk_parameter,
            "n_skipped_total": self.n_skipped_total,
        }
