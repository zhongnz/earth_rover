from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SensorPacket:
    """Unified snapshot from SDK camera + telemetry endpoints."""

    received_at: float
    frame_ts: float
    data_ts: float
    source_latency_ms: float
    frame_bgr: Optional[np.ndarray]
    raw_data: Dict[str, Any]


@dataclass
class DriveCommand:
    """Velocity command in SDK-compatible ranges."""

    linear: float
    angular: float
    lamp: int = 0


@dataclass
class StateEstimate:
    """Filtered robot state in a local ENU frame."""

    ts: float
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float
    gps_valid: bool
    pose_cov: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )

