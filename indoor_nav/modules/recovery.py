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
        self._motion_ref_orientation: Optional[float] = None
        self._last_stuck_detail: str = "no stuck condition recorded"

    @property
    def is_recovering(self) -> bool:
        return self._is_recovering

    @property
    def last_stuck_detail(self) -> str:
        return self._last_stuck_detail

    def note_command(self, linear: float, angular: float):
        """Called every control tick to track if we're commanding motion."""
        if abs(linear) > 0.05 or abs(angular) > 0.05:
            self._last_commanded_nonzero = time.time()

    @staticmethod
    def _angle_delta_deg(current: Optional[float], reference: Optional[float]) -> float:
        if current is None or reference is None:
            return 0.0
        try:
            delta = (float(current) - float(reference) + 180.0) % 360.0 - 180.0
        except Exception:
            return 0.0
        return abs(delta)

    @staticmethod
    def _mean_abs_rpm(rpms) -> Optional[float]:
        if not rpms:
            return None
        values = []
        for item in rpms:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            try:
                values.extend(abs(float(v)) for v in item[:4])
            except Exception:
                continue
        if not values:
            return None
        return sum(values) / len(values)

    def _clear_stuck_window(self, *, orientation: Optional[float] = None, detail: str = ""):
        self._stuck_start = None
        self._motion_ref_orientation = float(orientation) if orientation is not None else None
        if detail:
            self._last_stuck_detail = detail

    def check_stuck(
        self,
        speed: float,
        linear_cmd: float,
        angular_cmd: float = 0.0,
        orientation: Optional[float] = None,
        rpms=None,
    ) -> bool:
        """
        Returns True if the robot appears stuck.

        Robust stuck detection uses more than one telemetry channel:
        - translational progress: reported speed
        - rotational progress: heading change while commanding turns
        - drivetrain engagement: wheel RPMs, if present

        We only escalate to recovery when commands persist and the robot shows
        no evidence of translational or rotational progress for the configured
        timeout. If RPM telemetry is present and shows no wheel activity, we
        bias toward "not enough evidence yet" instead of false-positive stuck.
        """
        if not self.cfg.enabled:
            return False
        if self._is_recovering:
            return False

        now = time.time()
        commanding_translation = abs(linear_cmd) > self.cfg.stuck_linear_cmd_thresh
        commanding_rotation = abs(angular_cmd) > self.cfg.stuck_angular_cmd_thresh

        if not (commanding_translation or commanding_rotation):
            self._clear_stuck_window(
                orientation=orientation,
                detail="commands below stuck thresholds",
            )
            return False

        translational_progress = abs(speed) >= self.cfg.stuck_speed_thresh
        heading_delta = self._angle_delta_deg(orientation, self._motion_ref_orientation)
        rotational_progress = commanding_rotation and (
            heading_delta >= self.cfg.stuck_heading_delta_thresh
        )

        mean_abs_rpm = self._mean_abs_rpm(rpms)
        drivetrain_engaged = (
            mean_abs_rpm is None
            or mean_abs_rpm >= self.cfg.stuck_rpm_active_thresh
        )

        if translational_progress or rotational_progress:
            progress_bits = []
            if translational_progress:
                progress_bits.append(f"speed={speed:.3f}")
            if rotational_progress:
                progress_bits.append(f"heading_delta={heading_delta:.1f}deg")
            self._clear_stuck_window(
                orientation=orientation,
                detail="progress observed: " + ", ".join(progress_bits),
            )
            return False

        if self._motion_ref_orientation is None and orientation is not None:
            self._motion_ref_orientation = float(orientation)

        if not drivetrain_engaged:
            self._clear_stuck_window(
                orientation=orientation,
                detail=(
                    f"commanded motion but drivetrain not engaged "
                    f"(mean_abs_rpm={mean_abs_rpm:.2f})"
                ),
            )
            return False

        if self._stuck_start is None:
            self._stuck_start = now
            self._last_stuck_detail = (
                "candidate stuck window started: "
                f"speed={speed:.3f}, linear={linear_cmd:.3f}, angular={angular_cmd:.3f}, "
                f"heading_delta={heading_delta:.1f}deg, "
                f"mean_abs_rpm={mean_abs_rpm if mean_abs_rpm is not None else 'n/a'}"
            )
            return False

        elapsed = now - self._stuck_start
        if elapsed >= self.cfg.stuck_timeout:
            self._last_stuck_detail = (
                "stuck confirmed: "
                f"elapsed={elapsed:.1f}s, speed={speed:.3f}, linear={linear_cmd:.3f}, "
                f"angular={angular_cmd:.3f}, heading_delta={heading_delta:.1f}deg, "
                f"mean_abs_rpm={mean_abs_rpm if mean_abs_rpm is not None else 'n/a'}"
            )
            return True

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

    async def execute_relocalize_rotate(
        self,
        duration: Optional[float] = None,
        angular: Optional[float] = None,
    ) -> str:
        """
        SLAM-aware recovery behavior for lost tracking.

        This is intentionally separate from the stuck-recovery escalation ladder:
        - it does not advance `_recovery_level`
        - it performs a bounded in-place rotation to help a visual SLAM backend
          reacquire features and relocalize
        """
        if not self.cfg.enabled:
            return "disabled"
        if self._is_recovering:
            return "busy"

        self._is_recovering = True
        self._total_recoveries += 1
        behavior_name = "relocalize_rotate"

        try:
            spin_speed = float(angular if angular is not None else self.cfg.rotation_speed)
            spin_duration = float(duration if duration is not None else min(self.cfg.rotation_duration, 3.0))
            logger.info(
                "RECOVERY [slam] executing: %s (duration=%.1fs, angular=%.2f, total recoveries: %d)",
                behavior_name,
                spin_duration,
                spin_speed,
                self._total_recoveries,
            )

            end = time.time() + spin_duration
            while time.time() < end:
                await self.sdk.send_control(0.0, spin_speed)
                await asyncio.sleep(0.05)
        except Exception as e:
            logger.error("Recovery behavior %s failed: %s", behavior_name, e)
        finally:
            await self.sdk.stop(duration=0.5)
            self._stuck_start = None
            self._is_recovering = False

        return behavior_name

    async def execute_pose_backtrack(
        self,
        angular_bias: float = 0.0,
        duration: Optional[float] = None,
        linear: Optional[float] = None,
    ) -> str:
        """
        Pose-aware reverse maneuver for SLAM-backed recovery.

        This uses a caller-provided steering bias derived from recent pose
        history, but keeps the behavior itself simple and bounded.
        """
        if not self.cfg.enabled:
            return "disabled"
        if self._is_recovering:
            return "busy"

        self._is_recovering = True
        self._total_recoveries += 1
        behavior_name = "pose_backtrack"

        try:
            reverse_speed = float(linear if linear is not None else self.cfg.backup_speed)
            reverse_duration = float(duration if duration is not None else self.cfg.backup_duration)
            steer = max(-self.cfg.turn_speed, min(self.cfg.turn_speed, float(angular_bias)))
            logger.info(
                "RECOVERY [slam] executing: %s (duration=%.1fs, linear=%.2f, angular=%.2f, total recoveries: %d)",
                behavior_name,
                reverse_duration,
                reverse_speed,
                steer,
                self._total_recoveries,
            )

            end = time.time() + reverse_duration
            while time.time() < end:
                await self.sdk.send_control(reverse_speed, steer)
                await asyncio.sleep(0.05)
        except Exception as e:
            logger.error("Recovery behavior %s failed: %s", behavior_name, e)
        finally:
            await self.sdk.stop(duration=0.5)
            self._stuck_start = None
            self._is_recovering = False

        return behavior_name

    def reset(self):
        """Reset recovery state (e.g. after reaching a checkpoint)."""
        self._recovery_level = 0
        self._clear_stuck_window(detail="recovery state reset")

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
