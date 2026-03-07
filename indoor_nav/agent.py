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

        # Topological memory (SOTA 2025: visual graph for planning)
        self.topo_memory: Optional[TopologicalMemory] = None
        if cfg.topo_memory.enabled:
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

        # 3b. Initialize topological memory
        if self.topo_memory:
            logger.info("Topological memory: ON (max %d nodes)", self.cfg.topo_memory.max_nodes)

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

            os.makedirs(self.cfg.log.log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.cfg.log.log_dir, f"indoor_nav_{ts}.h5")
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

        # ---- 1b. Topological memory update ----
        if self.topo_memory:
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
        )

        action = self.policy.predict(policy_input)

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

        # ---- 7. Send command ----
        await self.sdk.send_control(linear, angular)
        self.recovery_mgr.note_command(linear, angular)

        # ---- 8. Stuck detection ----
        if self.recovery_mgr.check_stuck(bot_state.speed, linear):
            logger.warning("Robot appears STUCK. Initiating recovery...")
            self.state = AgentState.RECOVERING
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
            logger.info(
                "[%.0fs] %s | %s | lin=%.2f ang=%.2f | speed=%.1f bat=%.0f%%%s",
                elapsed,
                self.state.name,
                self.checkpoint_mgr.status_str(),
                linear,
                angular,
                bot_state.speed,
                bot_state.battery,
                topo_str,
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

    async def shutdown(self):
        """Graceful shutdown: stop robot, close connections, save logs."""
        logger.info("Shutting down navigation agent...")
        try:
            await self.sdk.stop(duration=self.cfg.control.stop_duration)
        except Exception:
            pass

        if self._data_logger:
            try:
                self._data_logger.close()
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
