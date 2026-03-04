from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Dict, Optional

import numpy as np

from .bev_mapper import BEVMapper, BEVResult
from .config import ERCConfig
from .goal_manager import GoalHint, GoalManager
from .mission_fsm import MissionFSM, MissionState
from .planner import PathFusionPlanner, PlannerOutput
from .recovery import RecoveryManager
from .sdk_io import SDKIO
from .state_estimator import StateEstimator
from .traversability import TraversabilityEngine, TraversabilityResult
from .types import DriveCommand, SensorPacket, StateEstimate
from .watchdog import StaleSensorWatchdog

logger = logging.getLogger(__name__)


class AutonomousMissionRunner:
    """
    Week 1-6 mission runner:
    - hardened SDK I/O
    - mission state machine
    - stale-sensor watchdog
    - filtered state estimation
    - traversability + BEV projection
    - candidate rollout + path fusion planner
    - checkpoint-bearing integration
    - explicit recovery behaviors (backtrack + rotate)
    """

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self.fsm = MissionFSM()
        self.sdk = SDKIO(cfg)
        self.estimator = StateEstimator(cfg)
        self.traversability = TraversabilityEngine(cfg)
        self.bev_mapper = BEVMapper(cfg)
        self.planner = PathFusionPlanner(cfg)
        self.goal_manager = GoalManager()
        self.recovery = RecoveryManager(cfg)
        self.stop_event = asyncio.Event()
        self.ticks = 0
        self.last_packet: Optional[SensorPacket] = None
        self.last_estimate: Optional[StateEstimate] = None
        self.last_traversability: Optional[TraversabilityResult] = None
        self.last_bev: Optional[BEVResult] = None
        self.last_plan: Optional[PlannerOutput] = None
        self.last_goal_hint: Optional[GoalHint] = None
        self._last_checkpoint_refresh_monotonic: float = 0.0
        self._last_checkpoint_attempt_monotonic: float = 0.0
        self._checkpoint_failure_distance_m: Optional[float] = None
        self._checkpoint_failure_monotonic: float = 0.0
        self._checkpoint_failure_count: int = 0
        self._last_checkpoint_speed_factor: float = 1.0
        self._last_checkpoint_failure_factor: float = 1.0
        self._last_checkpoint_angular_factor: float = 1.0
        self._last_checkpoint_failure_angular_factor: float = 1.0
        self.watchdog = StaleSensorWatchdog(
            stale_after_ms=cfg.stale_sensor_ms,
            on_stale=self._handle_stale_sensor,
        )

    async def _handle_stale_sensor(self) -> None:
        logger.warning("stale sensor input detected; issuing safe stop")
        await self.sdk.safe_stop(self.cfg.stop_duration_s, self.cfg.stop_hz)

    def _register_signals(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                # add_signal_handler is unavailable on some platforms.
                pass

    def request_stop(self) -> None:
        self.stop_event.set()

    async def _startup(self) -> None:
        self.fsm.on_start()
        logger.info("mission startup", extra={"extra_data": {"state": self.fsm.state.name}})
        if self.cfg.start_mission_on_boot:
            mission = await self.sdk.start_mission()
            logger.info(
                "start-mission response",
                extra={"extra_data": {"message": mission.get("message", ""), "ok": bool(mission)}},
            )
        await self._refresh_checkpoints(force=True)
        self.fsm.on_started()

    def _goal_turn_hint(self) -> float:
        self.last_goal_hint = None
        if self.last_packet is None:
            return 0.0
        hint = self.goal_manager.compute_turn_hint(self.last_packet.raw_data)
        if hint is None:
            return 0.0
        self.last_goal_hint = hint
        return hint.turn_hint

    async def _refresh_checkpoints(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_checkpoint_refresh_monotonic) < self.cfg.checkpoint_refresh_interval_s:
            return
        self._last_checkpoint_refresh_monotonic = now
        payload = await self.sdk.get_checkpoints()
        if self.goal_manager.update_from_checkpoints_payload(payload):
            status = self.goal_manager.status()
            logger.info(
                "checkpoints synced",
                extra={"extra_data": status},
            )

    @staticmethod
    def _extract_checkpoint_error(payload: Dict) -> str:
        detail = payload.get("detail")
        if isinstance(detail, dict):
            err = detail.get("error", "")
            dist = detail.get("proximate_distance_to_checkpoint")
            if dist is not None:
                return f"{err} dist={dist}"
            return str(err)
        return ""

    @staticmethod
    def _extract_checkpoint_proximate_distance(payload: Dict) -> Optional[float]:
        detail = payload.get("detail")
        if not isinstance(detail, dict):
            return None
        value = detail.get("proximate_distance_to_checkpoint")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _distance_taper(
        distance_m: float, start_m: float, hard_m: float, min_factor: float
    ) -> float:
        min_factor = float(np.clip(min_factor, 0.0, 1.0))
        if start_m <= hard_m:
            return min_factor if distance_m <= hard_m else 1.0
        if distance_m >= start_m:
            return 1.0
        if distance_m <= hard_m:
            return min_factor
        alpha = (distance_m - hard_m) / max(1e-6, start_m - hard_m)
        return float(min_factor + alpha * (1.0 - min_factor))

    def _checkpoint_motion_factors(self, now: float) -> tuple[float, float, float, float]:
        distance_factor = 1.0
        failure_factor = 1.0
        distance_angular_factor = 1.0
        failure_angular_factor = 1.0
        hint = self.last_goal_hint
        if hint is None:
            return (
                distance_factor,
                failure_factor,
                distance_angular_factor,
                failure_angular_factor,
            )

        distance_factor = self._distance_taper(
            distance_m=hint.distance_m,
            start_m=max(
                self.cfg.checkpoint_slowdown_start_m,
                self.cfg.checkpoint_slowdown_hard_m + 0.1,
            ),
            hard_m=self.cfg.checkpoint_slowdown_hard_m,
            min_factor=self.cfg.checkpoint_slowdown_min_factor,
        )
        distance_angular_factor = self._distance_taper(
            distance_m=hint.distance_m,
            start_m=max(
                self.cfg.checkpoint_slowdown_start_m,
                self.cfg.checkpoint_slowdown_hard_m + 0.1,
            ),
            hard_m=self.cfg.checkpoint_slowdown_hard_m,
            min_factor=self.cfg.checkpoint_angular_min_factor,
        )

        failed_dist = self._checkpoint_failure_distance_m
        if failed_dist is None:
            return (
                distance_factor,
                failure_factor,
                distance_angular_factor,
                failure_angular_factor,
            )

        age_s = now - self._checkpoint_failure_monotonic
        if age_s > self.cfg.checkpoint_failure_effect_s:
            self._checkpoint_failure_distance_m = None
            self._checkpoint_failure_monotonic = 0.0
            self._checkpoint_failure_count = 0
            return (
                distance_factor,
                failure_factor,
                distance_angular_factor,
                failure_angular_factor,
            )

        spatial_factor = self._distance_taper(
            distance_m=hint.distance_m,
            start_m=failed_dist + max(0.1, self.cfg.checkpoint_failure_buffer_m),
            hard_m=failed_dist,
            min_factor=self.cfg.checkpoint_failure_min_factor,
        )
        spatial_angular_factor = self._distance_taper(
            distance_m=hint.distance_m,
            start_m=failed_dist + max(0.1, self.cfg.checkpoint_failure_buffer_m),
            hard_m=failed_dist,
            min_factor=self.cfg.checkpoint_failure_angular_min_factor,
        )
        time_mix = float(
            1.0 - np.clip(age_s / max(1e-6, self.cfg.checkpoint_failure_effect_s), 0.0, 1.0)
        )
        failure_factor = float(1.0 - (time_mix * (1.0 - spatial_factor)))
        failure_angular_factor = float(1.0 - (time_mix * (1.0 - spatial_angular_factor)))
        return (
            distance_factor,
            failure_factor,
            distance_angular_factor,
            failure_angular_factor,
        )

    async def _maybe_report_checkpoint(self) -> None:
        if not self.cfg.auto_checkpoint_report:
            return
        hint = self.last_goal_hint
        if hint is None:
            return
        if hint.distance_m > self.cfg.checkpoint_attempt_distance_m:
            return

        now = time.monotonic()
        if (now - self._last_checkpoint_attempt_monotonic) < self.cfg.checkpoint_attempt_interval_s:
            return
        self._last_checkpoint_attempt_monotonic = now

        result = await self.sdk.checkpoint_reached()
        if not result:
            return
        updated = self.goal_manager.update_from_checkpoint_reached_response(result)
        proximate_dist = self._extract_checkpoint_proximate_distance(result)
        if updated:
            self._checkpoint_failure_distance_m = None
            self._checkpoint_failure_monotonic = 0.0
            self._checkpoint_failure_count = 0
        elif proximate_dist is not None:
            self._checkpoint_failure_distance_m = max(0.0, float(proximate_dist))
            self._checkpoint_failure_monotonic = now
            self._checkpoint_failure_count += 1

        extra = {
            "attempt_dist_m": round(hint.distance_m, 2),
            "updated_sequence": bool(updated),
            "next_checkpoint_sequence": result.get("next_checkpoint_sequence"),
            "proximate_distance_m": round(proximate_dist, 2) if proximate_dist is not None else None,
            "failure_count": self._checkpoint_failure_count,
            "error": self._extract_checkpoint_error(result),
        }
        logger.info("checkpoint report", extra={"extra_data": extra})

    def _decide_command(
        self, estimate: Optional[StateEstimate], packet: Optional[SensorPacket]
    ) -> DriveCommand:
        self._last_checkpoint_speed_factor = 1.0
        self._last_checkpoint_failure_factor = 1.0
        self._last_checkpoint_angular_factor = 1.0
        self._last_checkpoint_failure_angular_factor = 1.0
        now = time.monotonic()
        if self.recovery.is_active:
            override = self.recovery.command_override(now=now)
            if override is not None:
                return override

        # Safety-first default.
        if not self.cfg.enable_motion:
            _ = self._goal_turn_hint()
            return DriveCommand(
                linear=self.cfg.default_linear,
                angular=self.cfg.default_angular,
                lamp=self.cfg.default_lamp,
            )

        speed_mps = estimate.speed_mps if estimate is not None else 0.0
        _ = packet
        trav = self.last_traversability
        bev = self.last_bev
        if trav is None or bev is None:
            self.last_plan = None
            return DriveCommand(linear=0.0, angular=0.0, lamp=0)

        if trav.confidence < self.cfg.traversability_confidence_floor:
            self.last_plan = None
            return DriveCommand(linear=0.0, angular=0.0, lamp=1)

        center_clear = np.clip(
            0.5 * trav.center_clearance + 0.5 * bev.center_score, 0.0, 1.0
        )
        left_clear = np.clip(
            0.5 * trav.left_clearance + 0.5 * bev.left_score, 0.0, 1.0
        )
        right_clear = np.clip(
            0.5 * trav.right_clearance + 0.5 * bev.right_score, 0.0, 1.0
        )

        if center_clear < 0.28:
            # Too risky straight ahead: stop and bias turn toward clear side.
            turn = np.clip((left_clear - right_clear) * self.cfg.reactive_turn_gain, -1, 1)
            self.last_plan = None
            return DriveCommand(linear=0.0, angular=float(turn * self.cfg.max_angular), lamp=1)

        goal_hint = self._goal_turn_hint()
        plan = self.planner.plan(bev, goal_turn_hint=goal_hint)
        self.last_plan = plan
        if plan is None:
            turn = np.clip((left_clear - right_clear) * self.cfg.reactive_turn_gain, -1, 1)
            return DriveCommand(linear=0.0, angular=float(turn * self.cfg.max_angular), lamp=1)

        if plan.mode == "stop":
            turn = np.clip((left_clear - right_clear) * self.cfg.reactive_turn_gain, -1, 1)
            return DriveCommand(linear=0.0, angular=float(turn * self.cfg.max_angular * 0.7), lamp=1)

        reactive_turn = np.clip(
            (left_clear - right_clear) * self.cfg.reactive_turn_gain, -1, 1
        )
        blended_turn = float(np.clip((0.75 * plan.angular_hint) + (0.25 * reactive_turn), -1.0, 1.0))
        speed_hint = float(np.clip(plan.speed_hint, 0.0, 1.0))
        linear = float(self.cfg.max_linear * min(center_clear, speed_hint))
        (
            checkpoint_speed_factor,
            failure_speed_factor,
            checkpoint_angular_factor,
            failure_angular_factor,
        ) = self._checkpoint_motion_factors(now)
        self._last_checkpoint_speed_factor = checkpoint_speed_factor
        self._last_checkpoint_failure_factor = failure_speed_factor
        self._last_checkpoint_angular_factor = checkpoint_angular_factor
        self._last_checkpoint_failure_angular_factor = failure_angular_factor
        linear *= checkpoint_speed_factor * failure_speed_factor
        angular = float(
            self.cfg.max_angular
            * blended_turn
            * checkpoint_angular_factor
            * failure_angular_factor
        )
        lamp = 1 if linear < 0.03 else 0
        base_cmd = DriveCommand(
            linear=linear,
            angular=angular,
            lamp=lamp,
        )

        self.recovery.note_observation(
            now=now,
            speed_mps=speed_mps,
            cmd_linear=base_cmd.linear,
            cmd_angular=base_cmd.angular,
            traversability_confidence=trav.confidence,
        )
        self.recovery.maybe_start(now=now, preferred_turn_hint=goal_hint)
        override = self.recovery.command_override(now=now)
        if override is not None:
            return override
        return base_cmd

    async def _shutdown(self) -> None:
        self.fsm.on_stop()
        await self.sdk.safe_stop(self.cfg.stop_duration_s, self.cfg.stop_hz)
        if self.cfg.end_mission_on_shutdown:
            result = await self.sdk.end_mission()
            logger.info(
                "end-mission response",
                extra={"extra_data": {"message": result.get("message", ""), "ok": bool(result)}},
            )
        await self.sdk.close()
        self.fsm.on_stopped()
        logger.info("mission stopped", extra={"extra_data": {"state": self.fsm.state.name}})

    async def run(self) -> None:
        self._register_signals()
        interval = 1.0 / max(1.0, self.cfg.loop_hz)
        await self._startup()

        try:
            while not self.stop_event.is_set():
                tick_start = time.monotonic()

                packet = await self.sdk.poll()
                if packet is not None:
                    self.watchdog.mark_sensor()
                    self.last_packet = packet
                    self.last_estimate = self.estimator.update(packet.raw_data)
                    self.last_traversability = self.traversability.infer(packet.frame_bgr)
                    if self.last_traversability is not None:
                        self.last_bev = self.bev_mapper.project(self.last_traversability.mask)
                _ = self._goal_turn_hint()

                command = self._decide_command(self.last_estimate, self.last_packet)
                await self.sdk.send_control(command)
                await self._maybe_report_checkpoint()
                await self._refresh_checkpoints(force=False)
                stale_triggered = await self.watchdog.tick()

                if stale_triggered or self.recovery.is_active:
                    self.fsm.on_recover()
                elif self.fsm.state == MissionState.RECOVERING:
                    self.fsm.on_resume()

                self.ticks += 1
                if self.ticks % max(1, self.cfg.log_every_n_ticks) == 0:
                    latency = self.last_packet.source_latency_ms if self.last_packet else -1
                    trav = self.last_traversability
                    bev = self.last_bev
                    plan = self.last_plan
                    hint = self.last_goal_hint
                    goal_status = self.goal_manager.status()
                    recovery_status = self.recovery.status()
                    logger.info(
                        "runner status",
                        extra={
                            "extra_data": {
                                "state": self.fsm.state.name,
                                "ticks": self.ticks,
                                "latency_ms": round(latency, 2),
                                "gps_valid": bool(self.last_estimate.gps_valid) if self.last_estimate else False,
                                "x_m": round(self.last_estimate.x_m, 2) if self.last_estimate else 0.0,
                                "y_m": round(self.last_estimate.y_m, 2) if self.last_estimate else 0.0,
                                "trav_conf": round(trav.confidence, 3) if trav else -1.0,
                                "trav_risk": round(trav.risk, 3) if trav else -1.0,
                                "bev_center": round(bev.center_score, 3) if bev else -1.0,
                                "plan_score": round(plan.score, 3) if plan else -1.0,
                                "plan_mode": plan.mode if plan else "none",
                                "plan_speed_hint": round(plan.speed_hint, 3) if plan else -1.0,
                                "plan_ang_hint": round(plan.angular_hint, 3) if plan else -1.0,
                                "goal_seq": int(hint.target_sequence) if hint else goal_status.get("active_sequence", 0),
                                "goal_dist_m": round(hint.distance_m, 2) if hint else -1.0,
                                "goal_turn_hint": round(hint.turn_hint, 3) if hint else 0.0,
                                "goal_total": goal_status.get("total_checkpoints", 0),
                                "goal_speed_factor": round(self._last_checkpoint_speed_factor, 3),
                                "goal_failure_factor": round(self._last_checkpoint_failure_factor, 3),
                                "goal_ang_factor": round(self._last_checkpoint_angular_factor, 3),
                                "goal_failure_ang_factor": round(
                                    self._last_checkpoint_failure_angular_factor, 3
                                ),
                                "goal_failure_count": self._checkpoint_failure_count,
                                "goal_failure_dist_m": round(self._checkpoint_failure_distance_m, 2)
                                if self._checkpoint_failure_distance_m is not None
                                else -1.0,
                                "recovery_mode": recovery_status.mode,
                                "recovery_active": recovery_status.active,
                                "recovery_stuck_s": round(recovery_status.stuck_elapsed_s, 2),
                                "recovery_count": recovery_status.recoveries,
                                "cmd_lin": round(command.linear, 3),
                                "cmd_ang": round(command.angular, 3),
                            }
                        },
                    )

                elapsed = time.monotonic() - tick_start
                await asyncio.sleep(max(0.0, interval - elapsed))

        except Exception:
            self.fsm.on_error()
            logger.exception("runner crashed")
            raise
        finally:
            await self._shutdown()
