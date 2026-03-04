"""
Recovery behaviors for when the robot gets stuck.

Provides a hierarchy of increasingly aggressive maneuvers to free the
robot from stuck states (dead ends, collisions, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from indoor_nav.configs.config import RecoveryConfig
from indoor_nav.modules.sdk_client import RoverSDKClient

logger = logging.getLogger(__name__)


class RecoveryBehavior(Enum):
    BACK_UP = auto()
    RANDOM_TURN = auto()
    WALL_FOLLOW = auto()
    FULL_ROTATION = auto()


class RecoveryManager:
    """
    Manages stuck detection and recovery maneuver execution.

    Stuck detection: if the robot's speed stays near zero for `stuck_timeout`
    despite non-zero commands, we trigger recovery.

    Recovery hierarchy:
    1. Back up for ~1.5s
    2. Random turn for ~1s
    3. Wall follow for ~3s
    4. Full 360° rotation to re-localize

    Each failure escalates to the next behavior. After exhausting all, reset.
    """

    def __init__(self, cfg: RecoveryConfig, sdk: RoverSDKClient):
        self.cfg = cfg
        self.sdk = sdk
        self._stuck_start: Optional[float] = None
        self._recovery_level: int = 0
        self._total_recoveries: int = 0
        self._is_recovering: bool = False
        self._last_commanded_nonzero: float = 0.0

    @property
    def is_recovering(self) -> bool:
        return self._is_recovering

    def note_command(self, linear: float, angular: float):
        """Called every control tick to track if we're commanding motion."""
        if abs(linear) > 0.05 or abs(angular) > 0.05:
            self._last_commanded_nonzero = time.time()

    def check_stuck(self, speed: float, linear_cmd: float) -> bool:
        """
        Returns True if the robot appears stuck.

        Stuck = speed near zero while we've been commanding forward motion
        for longer than stuck_timeout.
        """
        if not self.cfg.enabled:
            return False
        if self._is_recovering:
            return False

        commanding_motion = abs(linear_cmd) > 0.05
        barely_moving = abs(speed) < 0.05  # speed from telemetry

        if commanding_motion and barely_moving:
            if self._stuck_start is None:
                self._stuck_start = time.time()
            elif time.time() - self._stuck_start > 8.0:  # seconds
                return True
        else:
            self._stuck_start = None

        return False

    async def execute_recovery(self) -> str:
        """
        Execute the next recovery behavior in the hierarchy.
        Returns the name of the behavior executed.
        """
        if not self.cfg.enabled:
            return "disabled"

        self._is_recovering = True
        self._total_recoveries += 1

        behaviors = self.cfg.behaviors
        if self._recovery_level >= len(behaviors):
            self._recovery_level = 0  # cycle back

        behavior_name = behaviors[self._recovery_level]
        logger.info(
            "RECOVERY [%d/%d] executing: %s (total recoveries: %d)",
            self._recovery_level + 1,
            len(behaviors),
            behavior_name,
            self._total_recoveries,
        )

        try:
            if behavior_name == "back_up":
                await self._back_up()
            elif behavior_name == "random_turn":
                await self._random_turn()
            elif behavior_name == "wall_follow":
                await self._wall_follow()
            elif behavior_name == "full_rotation":
                await self._full_rotation()
            else:
                logger.warning("Unknown recovery behavior: %s", behavior_name)
        except Exception as e:
            logger.error("Recovery behavior %s failed: %s", behavior_name, e)
        finally:
            # Always stop after recovery
            await self.sdk.stop(duration=0.5)
            self._recovery_level += 1
            self._stuck_start = None
            self._is_recovering = False

        return behavior_name

    def reset(self):
        """Reset recovery state (e.g. after reaching a checkpoint)."""
        self._recovery_level = 0
        self._stuck_start = None

    async def _back_up(self):
        """Reverse at moderate speed."""
        duration = self.cfg.backup_duration
        speed = self.cfg.backup_speed
        end = time.time() + duration
        while time.time() < end:
            await self.sdk.send_control(speed, 0.0)
            await asyncio.sleep(0.05)

    async def _random_turn(self):
        """Turn in a random direction."""
        direction = random.choice([-1, 1])
        angular = self.cfg.turn_speed * direction
        duration = self.cfg.turn_duration
        end = time.time() + duration
        while time.time() < end:
            await self.sdk.send_control(0.0, angular)
            await asyncio.sleep(0.05)

    async def _wall_follow(self):
        """
        Simple wall-following: go forward with a slight turn.
        Direction chosen to steer away from the most recent obstacle side.
        """
        direction = random.choice([-1, 1])
        angular = 0.3 * direction
        linear = 0.3
        duration = self.cfg.wall_follow_duration
        end = time.time() + duration
        while time.time() < end:
            await self.sdk.send_control(linear, angular)
            await asyncio.sleep(0.05)

    async def _full_rotation(self):
        """
        Perform a full 360° rotation for re-localization.
        The navigation agent can monitor goal similarity during this.
        """
        angular = self.cfg.rotation_speed
        duration = self.cfg.rotation_duration
        end = time.time() + duration
        while time.time() < end:
            await self.sdk.send_control(0.0, angular)
            await asyncio.sleep(0.05)
