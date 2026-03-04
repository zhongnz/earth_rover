from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def _wrap_angle_rad(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def _bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(y, x)


@dataclass
class Checkpoint:
    sequence: int
    latitude: float
    longitude: float
    checkpoint_id: int = 0


@dataclass
class GoalHint:
    turn_hint: float  # [-1, 1], +left/-right
    distance_m: float
    target_sequence: int


class GoalManager:
    """Tracks mission checkpoints and computes heading error to active goal."""

    def __init__(self):
        self._checkpoints: List[Checkpoint] = []
        self._active_sequence: int = 0
        self._latest_scanned_sequence: int = 0

    @property
    def has_goal(self) -> bool:
        return self.current_checkpoint() is not None

    def current_checkpoint(self) -> Optional[Checkpoint]:
        if not self._checkpoints:
            return None
        if self._active_sequence <= 0:
            return self._checkpoints[0]
        for cp in self._checkpoints:
            if cp.sequence == self._active_sequence:
                return cp
        # Fallback to first pending by sequence
        for cp in self._checkpoints:
            if cp.sequence > self._latest_scanned_sequence:
                return cp
        return None

    def update_from_checkpoints_payload(self, payload: Dict) -> bool:
        items = payload.get("checkpoints_list")
        if not isinstance(items, list) or not items:
            return False

        parsed: List[Checkpoint] = []
        for item in items:
            try:
                seq = int(item.get("sequence", 0))
                lat = float(item.get("latitude", 0.0))
                lon = float(item.get("longitude", 0.0))
                cp_id = int(item.get("id", 0))
            except Exception:
                continue
            if seq <= 0:
                continue
            parsed.append(
                Checkpoint(
                    sequence=seq,
                    latitude=lat,
                    longitude=lon,
                    checkpoint_id=cp_id,
                )
            )

        if not parsed:
            return False
        parsed.sort(key=lambda c: c.sequence)
        self._checkpoints = parsed

        latest = payload.get("latest_scanned_checkpoint", self._latest_scanned_sequence)
        try:
            self._latest_scanned_sequence = int(latest)
        except Exception:
            pass

        if self._active_sequence <= 0:
            self._active_sequence = self._latest_scanned_sequence + 1
        else:
            self._active_sequence = max(self._active_sequence, self._latest_scanned_sequence + 1)
        return True

    def update_from_checkpoint_reached_response(self, payload: Dict) -> bool:
        next_seq = payload.get("next_checkpoint_sequence")
        if next_seq is None:
            return False
        try:
            self._active_sequence = int(next_seq)
            self._latest_scanned_sequence = max(self._latest_scanned_sequence, self._active_sequence - 1)
            return True
        except Exception:
            return False

    def compute_turn_hint(self, telemetry: Dict) -> Optional[GoalHint]:
        cp = self.current_checkpoint()
        if cp is None:
            return None

        try:
            lat = float(telemetry.get("latitude", 0.0))
            lon = float(telemetry.get("longitude", 0.0))
            heading_deg = float(telemetry.get("orientation", 0.0))
        except Exception:
            return None

        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None
        if lat == 0.0 and lon == 0.0:
            return None

        dist_m = _haversine_m(lat, lon, cp.latitude, cp.longitude)
        goal_bearing = _bearing_rad(lat, lon, cp.latitude, cp.longitude)
        heading = math.radians(heading_deg)
        error = _wrap_angle_rad(goal_bearing - heading)
        turn_hint = float(np.clip(error / math.pi, -1.0, 1.0))
        return GoalHint(
            turn_hint=turn_hint,
            distance_m=float(dist_m),
            target_sequence=cp.sequence,
        )

    def status(self) -> Dict:
        cp = self.current_checkpoint()
        return {
            "has_goal": bool(cp),
            "active_sequence": int(cp.sequence if cp else 0),
            "latest_scanned_sequence": int(self._latest_scanned_sequence),
            "total_checkpoints": int(len(self._checkpoints)),
        }

