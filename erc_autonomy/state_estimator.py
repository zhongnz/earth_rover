from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ERCConfig
from .types import StateEstimate


@dataclass
class _Origin:
    lat_deg: float
    lon_deg: float


def _wrap_angle_rad(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _lerp_angle(prev: float, nxt: float, alpha: float) -> float:
    delta = _wrap_angle_rad(nxt - prev)
    return _wrap_angle_rad(prev + alpha * delta)


class StateEstimator:
    """
    Lightweight filter for noisy GPS/heading streams.

    This is intentionally simple for Week 1-2 scaffolding and can be replaced
    by a full EKF/UKF without changing downstream interfaces.
    """

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self.origin: Optional[_Origin] = None
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._speed = 0.0
        self._last_ts = 0.0

    def _latlon_to_local(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        assert self.origin is not None
        r_earth_m = 6_378_137.0
        lat0 = math.radians(self.origin.lat_deg)
        dlat = math.radians(lat_deg - self.origin.lat_deg)
        dlon = math.radians(lon_deg - self.origin.lon_deg)
        x = dlon * math.cos(lat0) * r_earth_m
        y = dlat * r_earth_m
        return x, y

    def update(self, telemetry: dict) -> StateEstimate:
        ts = float(telemetry.get("timestamp", time.time()))
        lat = float(telemetry.get("latitude", 0.0))
        lon = float(telemetry.get("longitude", 0.0))
        heading_deg = float(telemetry.get("orientation", 0.0))
        speed = float(telemetry.get("speed", 0.0))

        gps_valid = bool(
            np.isfinite(lat) and np.isfinite(lon) and (lat != 0.0 or lon != 0.0)
        )

        if gps_valid and self.origin is None:
            self.origin = _Origin(lat_deg=lat, lon_deg=lon)

        if gps_valid and self.origin is not None:
            obs_x, obs_y = self._latlon_to_local(lat, lon)
            jump = math.hypot(obs_x - self._x, obs_y - self._y)
            if self._last_ts <= 0.0 or jump <= self.cfg.max_gps_jump_m:
                a = self.cfg.position_alpha
                self._x = (1.0 - a) * self._x + a * obs_x
                self._y = (1.0 - a) * self._y + a * obs_y

        obs_yaw = math.radians(heading_deg)
        if self._last_ts <= 0.0:
            self._yaw = obs_yaw
            self._speed = speed
        else:
            self._yaw = _lerp_angle(self._yaw, obs_yaw, self.cfg.yaw_alpha)
            self._speed = (1.0 - self.cfg.speed_alpha) * self._speed + (
                self.cfg.speed_alpha * speed
            )

        self._last_ts = ts

        cov = np.eye(3, dtype=np.float32)
        cov[0, 0] = 4.0 if gps_valid else 25.0
        cov[1, 1] = 4.0 if gps_valid else 25.0
        cov[2, 2] = 0.15

        return StateEstimate(
            ts=ts,
            x_m=float(self._x),
            y_m=float(self._y),
            yaw_rad=float(self._yaw),
            speed_mps=float(self._speed),
            gps_valid=gps_valid,
            pose_cov=cov,
        )

