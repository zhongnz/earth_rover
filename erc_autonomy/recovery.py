from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .config import ERCConfig
from .types import DriveCommand


@dataclass
class RecoveryStatus:
    mode: str = "idle"  # idle|backtrack|rotate|pause
    active: bool = False
    stuck_elapsed_s: float = 0.0
    recoveries: int = 0


class RecoveryManager:
    """Stuck detector and explicit recovery state machine."""

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self._stuck_since: Optional[float] = None
        self._mode: str = "idle"
        self._mode_started_at: float = 0.0
        self._last_recovery_end: float = 0.0
        self._recoveries: int = 0
        self._next_rotate_sign: int = 1

    @property
    def is_active(self) -> bool:
        return self._mode != "idle"

    @property
    def mode(self) -> str:
        return self._mode

    def note_observation(
        self,
        *,
        now: float,
        speed_mps: float,
        cmd_linear: float,
        cmd_angular: float,
        traversability_confidence: float,
    ) -> None:
        if not self.cfg.recovery_enabled:
            return

        moving = abs(speed_mps) >= self.cfg.recovery_min_speed_mps
        commanded = (abs(cmd_linear) >= self.cfg.recovery_min_cmd) or (
            abs(cmd_angular) >= self.cfg.recovery_min_cmd
        )
        conf_ok = traversability_confidence >= self.cfg.recovery_trav_conf_floor

        if self.is_active:
            return

        if commanded and (not moving) and conf_ok:
            if self._stuck_since is None:
                self._stuck_since = now
        else:
            self._stuck_since = None

    def maybe_start(self, now: float, preferred_turn_hint: float = 0.0) -> bool:
        if not self.cfg.recovery_enabled:
            return False
        if self.is_active:
            return False
        if self._stuck_since is None:
            return False
        if (now - self._stuck_since) < self.cfg.recovery_stuck_timeout_s:
            return False
        if (now - self._last_recovery_end) < self.cfg.recovery_cooldown_s:
            return False

        self._mode = "backtrack"
        self._mode_started_at = now
        self._recoveries += 1
        if abs(preferred_turn_hint) > 0.1:
            self._next_rotate_sign = 1 if preferred_turn_hint >= 0 else -1
        self._stuck_since = None
        return True

    def command_override(self, now: float) -> Optional[DriveCommand]:
        if not self.cfg.recovery_enabled:
            return None
        if self._mode == "idle":
            return None

        elapsed = now - self._mode_started_at
        if self._mode == "backtrack":
            if elapsed < self.cfg.recovery_backtrack_s:
                return DriveCommand(
                    linear=self.cfg.recovery_backtrack_linear,
                    angular=0.0,
                    lamp=1,
                )
            self._mode = "rotate"
            self._mode_started_at = now
            elapsed = 0.0

        if self._mode == "rotate":
            if elapsed < self.cfg.recovery_rotate_s:
                return DriveCommand(
                    linear=0.0,
                    angular=self.cfg.recovery_rotate_angular * float(self._next_rotate_sign),
                    lamp=1,
                )
            self._mode = "pause"
            self._mode_started_at = now
            elapsed = 0.0

        if self._mode == "pause":
            if elapsed < self.cfg.recovery_pause_s:
                return DriveCommand(linear=0.0, angular=0.0, lamp=1)
            self._mode = "idle"
            self._last_recovery_end = now
            self._next_rotate_sign *= -1
            return None

        return None

    def status(self, now: Optional[float] = None) -> RecoveryStatus:
        if now is None:
            now = time.monotonic()
        stuck_elapsed = 0.0
        if self._stuck_since is not None and not self.is_active:
            stuck_elapsed = max(0.0, now - self._stuck_since)
        return RecoveryStatus(
            mode=self._mode,
            active=self.is_active,
            stuck_elapsed_s=stuck_elapsed,
            recoveries=self._recoveries,
        )

