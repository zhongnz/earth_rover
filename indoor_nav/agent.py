"""
Indoor Navigation Agent — Main orchestrator — SOTA 2025.

Ties together:
  - SDK client (camera frames, telemetry, control)
  - Goal-conditioned navigation policy (VLM-hybrid / VLA / NoMaD)
  - Checkpoint manager (image-goal matching with DINOv2-VLAD/SigLIP2)
  - Obstacle avoidance (Depth Anything V2 / Depth Pro)
  - Topological memory (visual graph for backtracking & planning)
  - Recovery behaviors (stuck detection & escape)
  - Telemetry logging

The agent runs an async control loop at ~10 Hz:
  1. Fetch frame + telemetry
  2. Update topological memory (build visual graph)
  3. Compute goal similarity → check arrival
  4. Run obstacle detection
  5. Query navigation policy → get (linear, angular)
  6. Apply smoothing + safety limits
  7. Send control command
  8. Check for stuck → trigger recovery if needed
"""

from __future__ import annotations

import asyncio
import math
import logging
import os
import signal
import time
from collections import deque
from enum import Enum, auto
from typing import List, Optional

import cv2
import numpy as np

from indoor_nav.configs.config import IndoorNavConfig
from indoor_nav.modules.sdk_client import RoverSDKClient
from indoor_nav.modules.checkpoint_manager import CheckpointManager
from indoor_nav.modules.obstacle_avoidance import ObstacleDetector, ObstacleInfo
from indoor_nav.modules.recovery import RecoveryManager
from indoor_nav.modules.topological_memory import TopologicalMemory, TopoMapConfig
from indoor_nav.policies.base_policy import BasePolicy, PolicyInput, PolicyOutput
from indoor_nav.slam.orbslam3_client import ORBSLAM3Client
from indoor_nav.slam.types import SlamStatus

logger = logging.getLogger(__name__)


class AgentState(Enum):
    INIT = auto()
    NAVIGATING = auto()
    APPROACHING_GOAL = auto()
    CHECKPOINT_REACHED = auto()
    RECOVERING = auto()
    PAUSED = auto()
    MISSION_COMPLETE = auto()
    ERROR = auto()


class IndoorNavigationAgent:
    """
    Main navigation agent for the ICRA 2025 Indoor Track.

    Usage:
        cfg = IndoorNavConfig()
        agent = IndoorNavigationAgent(cfg)
        await agent.run(goal_images=["goal1.jpg", "goal2.jpg", ...])
    """

    def __init__(self, cfg: IndoorNavConfig):
        self.cfg = cfg
        self.state = AgentState.INIT

        # Core modules
        self.sdk = RoverSDKClient(cfg.sdk)
        self.checkpoint_mgr = CheckpointManager(cfg.goal)
        self.obstacle_det = ObstacleDetector(cfg.obstacle) if cfg.obstacle.enabled else None
        self.recovery_mgr = RecoveryManager(cfg.recovery, self.sdk)
        self.slam: Optional[ORBSLAM3Client] = None
        self._slam_status = SlamStatus()
        self._slam_pose_history: deque = deque(maxlen=max(10, int(cfg.control.loop_hz * 6)))
        self._last_slam_ok_time: float = 0.0
        self._last_slam_push_time: float = 0.0
        self._last_slam_failure_log_time: float = 0.0
        self._last_slam_gate_log_time: float = 0.0
        self._last_slam_relocalize_time: float = 0.0
        self._topo_suppressed_by_slam: bool = bool(cfg.slam.enabled and cfg.topo_memory.enabled)
        self._event_driven_topo: bool = bool(
            cfg.policy.backend == "maze_search" and cfg.topo_memory.enabled and not cfg.slam.enabled
        )

        if cfg.slam.enabled:
            if cfg.slam.backend != "orbslam3":
                raise ValueError(f"Unsupported SLAM backend: {cfg.slam.backend}")
            self.slam = ORBSLAM3Client(cfg.slam)

        # Topological memory (SOTA 2025: visual graph for planning)
        self.topo_memory: Optional[TopologicalMemory] = None
        if cfg.topo_memory.enabled and not cfg.slam.enabled:
            topo_cfg = TopoMapConfig(
                min_node_distance=cfg.topo_memory.min_node_distance,
                scene_change_threshold=cfg.topo_memory.scene_change_threshold,
                max_nodes=cfg.topo_memory.max_nodes,
                loop_closure_threshold=cfg.topo_memory.loop_closure_threshold,
                loop_closure_min_gap=cfg.topo_memory.loop_closure_min_gap,
                feature_method=cfg.topo_memory.feature_method,
            )
            self.topo_memory = TopologicalMemory(topo_cfg)

        # Navigation policy (loaded on setup)
        self.policy: Optional[BasePolicy] = None

        # Control state
        self._prev_linear: float = 0.0
        self._prev_angular: float = 0.0
        self._context_images: deque = deque(maxlen=cfg.policy.context_length)

        # Timing
        self._loop_interval: float = 1.0 / cfg.control.loop_hz
        self._tick_count: int = 0
        self._start_time: float = 0.0

        # Logging
        self._data_logger = None
        self._run_id: str = ""
        self._topo_export_dir: Optional[str] = None

        # Shutdown
        self._stop_event = asyncio.Event()

    def _create_policy(self) -> BasePolicy:
        """Instantiate the navigation policy based on config."""
        backend = self.cfg.policy.backend

        if backend == "nomad":
            from indoor_nav.policies.nomad_policy import NoMaDPolicy
            return NoMaDPolicy(self.cfg.policy)

        elif backend == "vint":
            # ViNT shares the same interface as NoMaD
            from indoor_nav.policies.nomad_policy import NoMaDPolicy
            return NoMaDPolicy(self.cfg.policy)

        elif backend == "gnm":
            from indoor_nav.policies.nomad_policy import NoMaDPolicy
            return NoMaDPolicy(self.cfg.policy)

        elif backend == "vlm_hybrid":
            from indoor_nav.policies.vlm_hybrid_policy import VLMHybridPolicy
            return VLMHybridPolicy(self.cfg.policy)

        elif backend == "vla":
            from indoor_nav.policies.vla_policy import VLAPolicy
            return VLAPolicy(self.cfg.policy)

        elif backend == "heuristic":
            from indoor_nav.policies.nomad_policy import NoMaDPolicy
            policy = NoMaDPolicy(self.cfg.policy)
            # Force heuristic mode by not providing model path
            policy._model = None
            return policy

        elif backend == "maze_search":
            from indoor_nav.policies.maze_search_policy import MazeSearchPolicy
            return MazeSearchPolicy(self.cfg.policy)

        else:
            raise ValueError(f"Unknown policy backend: {backend}")

    async def setup(self, goal_images: List[str]):
        """
        Initialize all modules and prepare for navigation.

        Args:
            goal_images: ordered list of goal image file paths.
        """
        logger.info("=" * 60)
        logger.info("INDOOR NAVIGATION AGENT — SETUP")
        logger.info("=" * 60)

        # 1. Load navigation policy
        logger.info("Loading navigation policy: %s", self.cfg.policy.backend)
        self.policy = self._create_policy()
        self.policy.setup()

        # 2. Load goal images
        if len(goal_images) == 1 and os.path.isdir(goal_images[0]):
            self.checkpoint_mgr.load_goals_from_dir(goal_images[0])
        else:
            self.checkpoint_mgr.load_goals(goal_images)

        if len(self.checkpoint_mgr.checkpoints) == 0:
            raise RuntimeError("No goal images loaded!")

        # 3. Initialize obstacle detector
        if self.obstacle_det:
            logger.info("Obstacle detection: %s", self.cfg.obstacle.method)

        # 3a. Initialize SLAM client
        if self.slam:
            logger.info(
                "SLAM: %s (%s @ %s)",
                self.cfg.slam.backend,
                self.cfg.slam.mode,
                self.cfg.slam.endpoint,
            )
            try:
                await self.slam.start()
                self._slam_status = await self.slam.status()
            except Exception as exc:
                raise RuntimeError(
                    f"SLAM sidecar startup failed at {self.cfg.slam.endpoint}: {exc}"
                ) from exc
            self._last_slam_ok_time = time.time()
            logger.info("SLAM sidecar ready. Tracking state: %s", self._slam_status.tracking_state)

        # 3b. Initialize topological memory
        if self.topo_memory:
            if self._event_driven_topo:
                logger.info(
                    "Topological memory: ON (event-driven maze mode, max %d nodes)",
                    self.cfg.topo_memory.max_nodes,
                )
            else:
                logger.info("Topological memory: ON (max %d nodes)", self.cfg.topo_memory.max_nodes)
        elif self._topo_suppressed_by_slam:
            logger.info("Topological memory: OFF (suppressed because SLAM is enabled)")

        # 4. Initialize data logger
        if self.cfg.log.enabled:
            self._setup_logger()

        # 5. Verify SDK connectivity
        logger.info("Testing SDK connectivity at %s...", self.cfg.sdk.base_url)
        state = await self.sdk.get_data()
        logger.info("SDK connected. Battery: %.0f%%, Signal: %.0f", state.battery, state.signal_level)

        frame, ts = await self.sdk.get_front_frame()
        if frame is not None:
            logger.info("Front camera OK: %dx%d", frame.shape[1], frame.shape[0])
        else:
            logger.warning("Front camera not available yet!")

        self.state = AgentState.NAVIGATING
        logger.info("Setup complete. %d checkpoints loaded.", len(self.checkpoint_mgr.checkpoints))
        logger.info("=" * 60)

    def _setup_logger(self):
        """Initialize HDF5 data logger."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
            from examples.utils.data_logger import H5DataLogger

            if not self._run_id:
                self._run_id = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.cfg.log.log_dir, exist_ok=True)
            path = os.path.join(self.cfg.log.log_dir, f"indoor_nav_{self._run_id}.h5")
            self._topo_export_dir = os.path.join(
                self.cfg.log.log_dir,
                f"indoor_nav_{self._run_id}_topo",
            )
            self._data_logger = H5DataLogger(
                path,
                compression="gzip",
                compression_level=self.cfg.log.h5_compression,
                mode="w",
            )
            logger.info("Logging to: %s", path)
        except Exception as e:
            logger.warning("Could not initialize data logger: %s", e)
            self._data_logger = None

    def _export_topo_debug_bundle(self):
        """Persist a topo snapshot with JSON + HTML debug artifacts."""
        if not self.topo_memory or self.topo_memory.num_nodes == 0:
            return

        if not self._run_id:
            self._run_id = time.strftime("%Y%m%d_%H%M%S")
        if not self._topo_export_dir:
            self._topo_export_dir = os.path.join(
                self.cfg.log.log_dir,
                f"indoor_nav_{self._run_id}_topo",
            )

        try:
            bundle = self.topo_memory.export_debug_bundle(self._topo_export_dir)
            logger.info("Topo debug bundle: %s", bundle["html"])
        except Exception as exc:
            logger.warning("Failed to export topo debug bundle: %s", exc)

    def _compute_topo_guidance(self) -> tuple[Optional[int], Optional[str]]:
        """Return the next-hop topo guidance toward the nearest known frontier."""
        if not self.topo_memory:
            return None, None

        current_node_id = self.topo_memory.current_node_id
        if current_node_id is None:
            return None, None

        path = self.topo_memory.plan_to_nearest_frontier(current_node_id)
        if not path or len(path) < 2:
            return None, None

        next_hop_id = path[1]
        return next_hop_id, self.topo_memory.get_exit_label(current_node_id, next_hop_id)

    async def run(self, goal_images: List[str]):
        """
        Main entry point. Setup + run the navigation loop until all
        checkpoints are reached or shutdown is requested.
        """
        await self.setup(goal_images)
        self._start_time = time.time()

        logger.info("Starting navigation loop at %.0f Hz", self.cfg.control.loop_hz)
        try:
            while not self._stop_event.is_set():
                loop_start = time.time()

                if self.state == AgentState.MISSION_COMPLETE:
                    logger.info("ALL CHECKPOINTS REACHED — Mission complete!")
                    break

                if self.state == AgentState.ERROR:
                    logger.error("Agent in ERROR state. Stopping.")
                    break

                try:
                    await self._tick()
                except Exception as e:
                    logger.error("Tick error: %s", e, exc_info=True)

                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self._loop_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                self._tick_count += 1

        except asyncio.CancelledError:
            logger.info("Navigation cancelled.")
        finally:
            await self.shutdown()

    async def _tick(self):
        """Single iteration of the control loop."""

        # ---- 1. Perception: fetch frame + telemetry concurrently ----
        frame_task = asyncio.create_task(self.sdk.get_front_frame())
        data_task = asyncio.create_task(self.sdk.get_data())

        (front_image, frame_ts), bot_state = await asyncio.gather(
            frame_task, data_task
        )

        if front_image is None:
            # No frame available — send zero command and retry
            await self.sdk.send_control(0.0, 0.0)
            return

        # Update context
        self._context_images.append(front_image)

        # ---- 1b. SLAM update ----
        slam_status = self._slam_status
        if self.slam:
            slam_status = await self._update_slam(front_image, frame_ts, bot_state)
            if self._should_run_slam_relocalization(slam_status):
                logger.warning(
                    "SLAM tracking lost too long. Initiating relocalization rotate (state=%s).",
                    slam_status.tracking_state,
                )
                self.state = AgentState.RECOVERING
                self._last_slam_relocalize_time = time.time()
                await self.recovery_mgr.execute_relocalize_rotate()
                return

        # ---- 1c. Topological memory update ----
        if self.topo_memory and not self._event_driven_topo:
            self.topo_memory.update(
                front_image,
                orientation=bot_state.orientation,
            )

        # ---- 2. Goal matching ----
        goal_sim = self.checkpoint_mgr.compute_goal_similarity(front_image)
        goal_trend = self.checkpoint_mgr.get_similarity_trend()

        # Check if current checkpoint is reached
        if self.checkpoint_mgr.check_arrival(goal_sim):
            await self._on_checkpoint_reached()
            if self.checkpoint_mgr.all_done:
                self.state = AgentState.MISSION_COMPLETE
                await self.sdk.stop(self.cfg.control.stop_duration)
                return
            # Reset policy for next checkpoint
            self.policy.reset()
            self.recovery_mgr.reset()

        # Update state
        if goal_sim >= self.cfg.goal.approach_threshold:
            self.state = AgentState.APPROACHING_GOAL
        else:
            self.state = AgentState.NAVIGATING

        # ---- 3. Obstacle detection ----
        obstacle_info = ObstacleInfo()
        if self.obstacle_det and self._tick_count % max(1, int(self.cfg.control.loop_hz / self.cfg.obstacle.check_hz)) == 0:
            obstacle_info = self.obstacle_det.detect(front_image)

        if obstacle_info.emergency_stop:
            logger.warning("EMERGENCY STOP — obstacle too close!")
            await self.sdk.send_control(0.0, 0.0)
            # Trigger immediate recovery
            if not self.recovery_mgr.is_recovering:
                await self.recovery_mgr.execute_recovery()
            return

        # ---- 4. Navigation policy ----
        goal = self.checkpoint_mgr.current_goal
        topo_target_node_id = None
        topo_target_exit_label = None
        if self.topo_memory:
            topo_target_node_id, topo_target_exit_label = self._compute_topo_guidance()

        policy_input = PolicyInput(
            front_image=front_image,
            goal_image=goal.image if goal else front_image,
            goal_similarity=goal_sim,
            goal_trend=goal_trend,
            context_images=list(self._context_images),
            orientation=bot_state.orientation,
            speed=bot_state.speed,
            obstacle_speed_factor=obstacle_info.speed_factor,
            obstacle_steer_bias=obstacle_info.steer_bias,
            left_clearance=obstacle_info.left_clearance,
            center_clearance=obstacle_info.center_clearance,
            right_clearance=obstacle_info.right_clearance,
            near_field_occupancy=obstacle_info.near_field_occupancy,
            topo_node_id=self.topo_memory.current_node_id if self.topo_memory else None,
            topo_target_node_id=topo_target_node_id,
            topo_target_exit_label=topo_target_exit_label,
            slam_tracking_state=slam_status.tracking_state,
            slam_pose=slam_status.pose,
            slam_keyframe_id=slam_status.keyframe_id,
        )

        action = self.policy.predict(policy_input)

        if self.topo_memory and self._event_driven_topo:
            if self.topo_memory.current_node_id is None or action.force_topo_node:
                self.topo_memory.update(
                    front_image,
                    orientation=bot_state.orientation,
                    force_new_node=True,
                    exit_label=action.topo_exit_label,
                )

        # ---- 5. VLM async query (if applicable) ----
        from indoor_nav.policies.vlm_hybrid_policy import VLMHybridPolicy
        if isinstance(self.policy, VLMHybridPolicy) and self.policy.needs_vlm_query:
            # Fire-and-forget async VLM query
            asyncio.create_task(
                self.policy.query_vlm(front_image, goal.image if goal else front_image)
            )

        # ---- 6. Smoothing + safety ----
        linear = action.linear * self.cfg.control.max_linear
        angular = action.angular * self.cfg.control.max_angular

        # Exponential smoothing
        alpha = self.cfg.control.smoothing_alpha
        linear = alpha * linear + (1 - alpha) * self._prev_linear
        angular = alpha * angular + (1 - alpha) * self._prev_angular

        self._prev_linear = linear
        self._prev_angular = angular

        # ---- 7. SLAM gating + send command ----
        slam_gate_reason = self._slam_motion_block_reason()
        if slam_gate_reason:
            now = time.time()
            if now - self._last_slam_gate_log_time >= 2.0:
                logger.warning("SLAM gating motion: %s", slam_gate_reason)
                self._last_slam_gate_log_time = now
            self.state = AgentState.PAUSED
            linear = 0.0
            angular = 0.0

        await self.sdk.send_control(linear, angular)
        self.recovery_mgr.note_command(linear, angular)

        # ---- 8. Stuck detection ----
        if self.recovery_mgr.check_stuck(
            bot_state.speed,
            linear,
            angular_cmd=angular,
            orientation=bot_state.orientation,
            rpms=bot_state.rpms,
        ):
            self.state = AgentState.RECOVERING
            slam_backtrack_angular = self._compute_slam_backtrack_angular(slam_status)
            if slam_backtrack_angular is not None:
                logger.warning(
                    "Robot appears STUCK. Initiating SLAM pose backtrack... (%s, angular_bias=%.2f)",
                    self.recovery_mgr.last_stuck_detail,
                    slam_backtrack_angular,
                )
                await self.recovery_mgr.execute_pose_backtrack(
                    angular_bias=slam_backtrack_angular
                )
            else:
                logger.warning(
                    "Robot appears STUCK. Initiating recovery... (%s)",
                    self.recovery_mgr.last_stuck_detail,
                )
                await self.recovery_mgr.execute_recovery()
            self.state = AgentState.NAVIGATING

        # ---- 9. Logging ----
        if self._data_logger and self._tick_count % max(1, int(self.cfg.control.loop_hz / self.cfg.log.log_hz)) == 0:
            try:
                self._data_logger.log_payload(bot_state.raw)
                self._data_logger.log_control(linear, angular, time.time())
            except Exception:
                pass

        # ---- 10. Status display ----
        if self._tick_count % int(self.cfg.control.loop_hz * 2) == 0:  # every 2s
            elapsed = time.time() - self._start_time
            topo_str = ""
            if self.topo_memory:
                topo_str = f" | {self.topo_memory.status_str()}"
            slam_str = ""
            if self.slam:
                slam_str = f" | SLAM:{slam_status.tracking_state}"
                if slam_status.keyframe_id is not None:
                    slam_str += f" kf={slam_status.keyframe_id}"
            logger.info(
                "[%.0fs] %s | %s | lin=%.2f ang=%.2f | speed=%.1f bat=%.0f%%%s%s",
                elapsed,
                self.state.name,
                self.checkpoint_mgr.status_str(),
                linear,
                angular,
                bot_state.speed,
                bot_state.battery,
                topo_str,
                slam_str,
            )

    async def _on_checkpoint_reached(self):
        """Handle checkpoint reached event."""
        done, total = self.checkpoint_mgr.progress
        logger.info("=" * 40)
        logger.info("CHECKPOINT %d/%d REACHED!", done, total)
        logger.info("=" * 40)

        # Stop briefly
        await self.sdk.stop(duration=0.5)

        # Try to report to SDK (for mission tracking)
        try:
            result = await self.sdk.report_checkpoint()
            logger.info("Checkpoint reported to SDK: %s", result)
        except Exception as e:
            logger.debug("Checkpoint report failed (may be expected): %s", e)

    def _build_slam_imu(self, bot_state) -> Optional[dict]:
        if self.cfg.slam.mode != "mono_inertial":
            return None
        return {
            "timestamp": float(bot_state.timestamp),
            "accels": bot_state.accels,
            "gyros": bot_state.gyros,
        }

    async def _update_slam(self, front_image: np.ndarray, frame_ts: float, bot_state) -> SlamStatus:
        if not self.slam:
            return self._slam_status

        now = time.time()
        min_interval = 1.0 / max(0.1, float(self.cfg.slam.push_hz))
        if self._last_slam_push_time and (now - self._last_slam_push_time) < min_interval:
            return self._slam_status

        try:
            slam_status = await self.slam.track(
                front_image,
                frame_ts,
                imu=self._build_slam_imu(bot_state),
            )
            self._slam_status = slam_status
            self._last_slam_push_time = now
            if slam_status.is_tracking:
                self._last_slam_ok_time = now
                if slam_status.pose is not None:
                    self._slam_pose_history.append(
                        (float(frame_ts), slam_status.pose, slam_status.keyframe_id)
                    )
            return slam_status
        except Exception as exc:
            if now - self._last_slam_failure_log_time >= 2.0:
                logger.warning("SLAM track update failed: %s", exc)
                self._last_slam_failure_log_time = now
            self._last_slam_push_time = now
            self._slam_status = SlamStatus(
                ok=False,
                tracking_state="LOST",
                frame_ts=float(frame_ts),
                raw={"error": str(exc)},
            )
            return self._slam_status

    def _slam_motion_block_reason(self) -> Optional[str]:
        if not self.slam or not self.cfg.slam.require_tracking_for_motion:
            return None

        status = self._slam_status
        if status.is_tracking:
            return None

        now = time.time()
        if self._last_slam_ok_time > 0.0 and (now - self._last_slam_ok_time) < self.cfg.slam.lost_stop_timeout:
            return None

        if status.frame_ts > 0.0 and (now - status.frame_ts) > self.cfg.slam.pose_stale_timeout:
            return f"stale SLAM status ({now - status.frame_ts:.1f}s old)"

        return f"tracking_state={status.tracking_state}"

    def _should_run_slam_relocalization(self, status: SlamStatus) -> bool:
        if not self.slam or not self.cfg.slam.use_for_recovery:
            return False
        if self.recovery_mgr.is_recovering:
            return False
        if status.is_tracking:
            return False

        now = time.time()
        if self._last_slam_ok_time > 0.0 and (now - self._last_slam_ok_time) < self.cfg.slam.lost_stop_timeout:
            return False

        cooldown = max(1.0, min(self.cfg.recovery.rotation_duration, 3.0))
        if self._last_slam_relocalize_time > 0.0 and (now - self._last_slam_relocalize_time) < cooldown:
            return False

        return status.tracking_state in {"LOST", "NOT_INITIALIZED"}

    @staticmethod
    def _normalize_angle_rad(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _slam_yaw_rad(self, pose) -> float:
        return math.atan2(
            2.0 * (pose.qw * pose.qz + pose.qx * pose.qy),
            1.0 - 2.0 * (pose.qy * pose.qy + pose.qz * pose.qz),
        )

    def _compute_slam_backtrack_angular(self, status: SlamStatus) -> Optional[float]:
        if (
            not self.slam
            or not self.cfg.slam.use_for_recovery
            or not status.is_tracking
            or status.pose is None
            or len(self._slam_pose_history) < 2
        ):
            return None

        current_pose = status.pose
        history = list(self._slam_pose_history)
        min_backtrack_dist = 0.15
        target_pose = None
        for _, pose, _ in reversed(history[:-1]):
            dx = float(pose.tx) - float(current_pose.tx)
            dz = float(pose.tz) - float(current_pose.tz)
            if math.hypot(dx, dz) >= min_backtrack_dist:
                target_pose = pose
                break

        if target_pose is None:
            return None

        desired_yaw = math.atan2(
            float(target_pose.tz) - float(current_pose.tz),
            float(target_pose.tx) - float(current_pose.tx),
        )
        current_yaw = self._slam_yaw_rad(current_pose)
        yaw_error = self._normalize_angle_rad(desired_yaw - current_yaw)

        if abs(yaw_error) < math.radians(5.0):
            return 0.0

        scaled = (yaw_error / (math.pi / 2.0)) * float(self.cfg.recovery.turn_speed)
        return max(-self.cfg.recovery.turn_speed, min(self.cfg.recovery.turn_speed, scaled))

    async def shutdown(self):
        """Graceful shutdown: stop robot, close connections, save logs."""
        logger.info("Shutting down navigation agent...")
        try:
            await self.sdk.stop(duration=self.cfg.control.stop_duration)
        except Exception:
            pass

        self._export_topo_debug_bundle()

        if self._data_logger:
            try:
                self._data_logger.close()
            except Exception:
                pass

        if self.slam:
            try:
                await self.slam.close()
            except Exception:
                pass

        await self.sdk.close()

        elapsed = time.time() - self._start_time if self._start_time else 0
        done, total = self.checkpoint_mgr.progress
        logger.info(
            "Navigation ended. Checkpoints: %d/%d. Time: %.1fs. Ticks: %d",
            done, total, elapsed, self._tick_count,
        )

    def request_stop(self):
        """Signal the agent to stop gracefully."""
        self._stop_event.set()
