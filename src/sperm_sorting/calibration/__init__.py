"""Physical calibration.

Nothing in this package has a default. Micrometres per pixel, the bulk flow
vector and the transport delay are properties of a built instrument, and the
system refuses to report physical units or to actuate until they are measured.
"""

from __future__ import annotations

from .flow import (
    FlowCalibrationResult,
    calibrate_fixed_vector,
    calibrate_flow_map,
    load_flow_map,
    save_flow_map,
)
from .optics import (
    OpticalCalibrationResult,
    calibrate_from_graticule,
    calibrate_from_known_distance,
    load_calibration,
    px_s_to_um_s,
    save_calibration,
)
from .transport import (
    TransportCalibrationResult,
    estimate_field_switching,
    estimate_from_geometry,
    estimate_from_tracer,
    load_transport_calibration,
    save_transport_calibration,
)

__all__ = [
    "FlowCalibrationResult",
    "OpticalCalibrationResult",
    "TransportCalibrationResult",
    "calibrate_fixed_vector",
    "calibrate_flow_map",
    "calibrate_from_graticule",
    "calibrate_from_known_distance",
    "estimate_field_switching",
    "estimate_from_geometry",
    "estimate_from_tracer",
    "load_calibration",
    "load_flow_map",
    "load_transport_calibration",
    "px_s_to_um_s",
    "save_calibration",
    "save_flow_map",
    "save_transport_calibration",
]
