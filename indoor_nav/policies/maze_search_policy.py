"""
Latency-safe maze search policy for simple indoor environments.

The controller is deliberately conservative:
  - move in short forward bursts
  - pause to refresh perception
  - trigger scan sweeps at junction-like views or dead ends
  - keep lightweight per-node exit memory when topo nodes are available

This is designed for the current rover stack where goal matching and camera
updates can be significantly slower than the motor response.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from indoor_nav.configs.config import PolicyConfig
from indoor_nav.policies.base_policy import BasePolicy, PolicyInput, PolicyOutput

logger = logging.getLogger(__name__)


class MazePhase(Enum):
    BOOT = auto()
    PAUSE = auto()
    SCAN_LEFT = auto()
    SCAN_RIGHT = auto()
    ALIGN = auto()
    BACKTRACK_TURN = auto()
    BURST = auto()
    APPROACH = auto()


@dataclass
class ScanSample:
    offset: float
    similarity: float
    left_clearance: float
    center_clearance: float
    right_clearance: float


@dataclass
class NodeMemory:
    tried_bins: set[str] = field(default_factory=set)
    scans: int = 0
    last_scan_time: float = 0.0
    exhausted: bool = False


class MazeSearchPolicy(BasePolicy):
    """Burst-stop-search controller tuned for simple mazes and corridor grids."""

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self._phase = MazePhase.BOOT
        self._phase_started = 0.0
        self._scan_samples: List[ScanSample] = []
        self._align_duration = 0.0
        self._align_direction = 0.0
        self._pending_force_topo_node = False
        self._node_memory: Dict[int, NodeMemory] = {}
        self._last_scanned_node_id: Optional[int] = None
        self._last_topo_node_id: Optional[int] = None
        self._node_changed_this_tick = False
        self._last_scan_finished = 0.0
        self._last_scan_started = 0.0
        self._last_similarity = 0.0
        self._backtrack_direction = 1.0
        self._active_exit_label: Optional[str] = None

    def setup(self):
        logger.info(
            "Maze search policy initialized "
            "(burst=%.2fs, pause=%.2fs, scan_leg=%.2fs)",
            self.cfg.maze_burst_seconds,
            self.cfg.maze_pause_seconds,
            self.cfg.maze_scan_leg_seconds,
        )

    def predict(self, obs: PolicyInput) -> PolicyOutput:
        now = time.time()
        self._ensure_node_memory(obs, now)
        self._handle_node_transition(obs)

        if self._phase == MazePhase.BOOT:
            self._start_pause(now)
            self._pending_force_topo_node = obs.topo_node_id is None

        if obs.goal_similarity >= 0.82:
            self._phase = MazePhase.APPROACH
            self._phase_started = now

        if self._phase == MazePhase.APPROACH:
            action = self._predict_approach(obs, now)
        elif self._phase == MazePhase.PAUSE:
            action = self._predict_pause(obs, now)
        elif self._phase == MazePhase.SCAN_LEFT:
            action = self._predict_scan_left(obs, now)
        elif self._phase == MazePhase.SCAN_RIGHT:
            action = self._predict_scan_right(obs, now)
        elif self._phase == MazePhase.ALIGN:
            action = self._predict_align(obs, now)
        elif self._phase == MazePhase.BACKTRACK_TURN:
            action = self._predict_backtrack_turn(obs, now)
        else:
            action = self._predict_burst(obs, now)

        action.force_topo_node = action.force_topo_node or self._pending_force_topo_node
        if action.topo_exit_label is None:
            action.topo_exit_label = self._active_exit_label
        self._pending_force_topo_node = False
        self._last_similarity = obs.goal_similarity
        return action

    def _predict_pause(self, obs: PolicyInput, now: float) -> PolicyOutput:
        if now - self._phase_started < self.cfg.maze_pause_seconds:
            return PolicyOutput(0.0, 0.0, confidence=0.4)

        node_mem = self._node_memory.get(obs.topo_node_id) if obs.topo_node_id is not None else None
        if (
            node_mem is not None
            and node_mem.exhausted
            and obs.goal_similarity < 0.78
            and obs.topo_target_exit_label in (None, "back")
        ):
            self._start_backtrack_turn(obs, now)
            return self._predict_backtrack_turn(obs, now)

        if self._should_scan(obs, now):
            self._start_scan(obs, now)
            return self._predict_scan_left(obs, now)

        self._start_burst(now)
        return self._predict_burst(obs, now)

    def _predict_scan_left(self, obs: PolicyInput, now: float) -> PolicyOutput:
        elapsed = now - self._phase_started
        leg = max(0.1, self.cfg.maze_scan_leg_seconds)
        offset = min(1.0, elapsed / leg)
        self._record_scan_sample(obs, offset)
        if elapsed >= leg:
            self._phase = MazePhase.SCAN_RIGHT
            self._phase_started = now
        return PolicyOutput(0.0, +self.cfg.maze_scan_turn_rate, confidence=0.55)

    def _predict_scan_right(self, obs: PolicyInput, now: float) -> PolicyOutput:
        elapsed = now - self._phase_started
        leg = max(0.1, self.cfg.maze_scan_leg_seconds)
        total = leg * 2.0
        offset = 1.0 - 2.0 * min(1.0, elapsed / total)
        self._record_scan_sample(obs, offset)
        if elapsed >= total:
            self._finish_scan(obs, now)
            return self._predict_align(obs, now)
        return PolicyOutput(0.0, -self.cfg.maze_scan_turn_rate, confidence=0.55)

    def _predict_align(self, obs: PolicyInput, now: float) -> PolicyOutput:
        if now - self._phase_started >= self._align_duration:
            self._start_burst(now)
            return self._predict_burst(obs, now)
        return PolicyOutput(0.0, self._align_direction, confidence=0.6)

    def _predict_backtrack_turn(self, obs: PolicyInput, now: float) -> PolicyOutput:
        if self._node_changed_this_tick:
            self._start_pause(now)
            return PolicyOutput(0.0, 0.0, confidence=0.5)

        if now - self._phase_started >= self.cfg.maze_backtrack_turn_seconds:
            self._start_burst(now)
            return self._predict_burst(obs, now)

        return PolicyOutput(
            0.0,
            float(np.clip(self._backtrack_direction * self.cfg.maze_scan_turn_rate, -1.0, 1.0)),
            confidence=0.6,
        )

    def _predict_burst(self, obs: PolicyInput, now: float) -> PolicyOutput:
        if obs.goal_similarity >= 0.82:
            self._phase = MazePhase.APPROACH
            self._phase_started = now
            return self._predict_approach(obs, now)

        elapsed = now - self._phase_started
        node_like = self._looks_like_node(obs)
        dead_end = obs.center_clearance <= self.cfg.maze_dead_end_clearance
        if (
            elapsed >= self.cfg.maze_burst_seconds
            or dead_end
            or (node_like and elapsed >= self.cfg.maze_burst_seconds * 0.5)
        ):
            self._start_pause(now)
            return PolicyOutput(0.0, 0.0, confidence=0.45)

        linear = self.cfg.maze_forward_rate
        if obs.goal_similarity > 0.65 and obs.goal_trend > 0.0:
            linear *= 0.7
        linear *= obs.obstacle_speed_factor
        linear *= self._corridor_speed_factor(obs)
        angular = self._corridor_steering(obs)
        return PolicyOutput(
            linear=float(np.clip(linear, -1.0, 1.0)),
            angular=float(np.clip(angular, -1.0, 1.0)),
            confidence=0.6,
        )

    def _predict_approach(self, obs: PolicyInput, now: float) -> PolicyOutput:
        if obs.goal_similarity < 0.58 and obs.goal_trend <= 0.0:
            self._start_pause(now)
            return PolicyOutput(0.0, 0.0, confidence=0.45)

        if obs.center_clearance <= self.cfg.maze_dead_end_clearance:
            self._start_pause(now)
            return PolicyOutput(0.0, 0.0, confidence=0.45)

        linear = self.cfg.maze_approach_rate * obs.obstacle_speed_factor
        linear *= self._corridor_speed_factor(obs, approach_mode=True)
        angular = self._corridor_steering(obs, approach_mode=True)
        return PolicyOutput(
            linear=float(np.clip(linear, -1.0, 1.0)),
            angular=float(np.clip(angular, -1.0, 1.0)),
            confidence=0.85,
        )

    def _start_pause(self, now: float):
        self._phase = MazePhase.PAUSE
        self._phase_started = now

    def _start_burst(self, now: float):
        self._phase = MazePhase.BURST
        self._phase_started = now

    def _start_scan(self, obs: PolicyInput, now: float):
        self._phase = MazePhase.SCAN_LEFT
        self._phase_started = now
        self._last_scan_started = now
        self._scan_samples = []
        self._record_scan_sample(obs, 0.0)
        if (
            obs.topo_node_id is None
            or self._looks_like_node(obs)
            or obs.center_clearance <= self.cfg.maze_dead_end_clearance
        ):
            self._pending_force_topo_node = True

    def _finish_scan(self, obs: PolicyInput, now: float):
        target = self._choose_scan_target(obs)
        if target is None:
            self._start_backtrack_turn(obs, now)
            return
        target_offset, target_label = target
        self._active_exit_label = target_label
        self._phase = MazePhase.ALIGN
        self._phase_started = now
        self._align_duration = abs(target_offset) * max(0.1, self.cfg.maze_scan_leg_seconds)
        if target_offset > 0:
            self._align_direction = +self.cfg.maze_scan_turn_rate
        elif target_offset < 0:
            self._align_direction = -self.cfg.maze_scan_turn_rate
        else:
            self._align_direction = 0.0

        self._last_scan_finished = now
        self._last_scanned_node_id = obs.topo_node_id

    def _start_backtrack_turn(self, obs: PolicyInput, now: float):
        self._phase = MazePhase.BACKTRACK_TURN
        self._phase_started = now
        self._last_scan_finished = now
        self._last_scanned_node_id = obs.topo_node_id
        self._backtrack_direction = 1.0 if obs.left_clearance >= obs.right_clearance else -1.0
        self._active_exit_label = "back"
        node_mem = self._node_memory.get(obs.topo_node_id) if obs.topo_node_id is not None else None
        if node_mem is not None:
            node_mem.exhausted = True

    def _record_scan_sample(self, obs: PolicyInput, offset: float):
        self._scan_samples.append(
            ScanSample(
                offset=float(np.clip(offset, -1.0, 1.0)),
                similarity=float(obs.goal_similarity),
                left_clearance=float(obs.left_clearance),
                center_clearance=float(obs.center_clearance),
                right_clearance=float(obs.right_clearance),
            )
        )

    def _choose_scan_target(self, obs: PolicyInput) -> Optional[tuple[float, str]]:
        node_mem = self._node_memory.get(obs.topo_node_id) if obs.topo_node_id is not None else None
        if node_mem is not None:
            node_mem.scans += 1
            node_mem.last_scan_time = time.time()

        if not self._scan_samples:
            return None

        best_by_bin: Dict[str, ScanSample] = {}
        for sample in self._scan_samples:
            if sample.center_clearance < self.cfg.maze_dead_end_clearance:
                continue
            turn_bin = self._offset_bin(sample.offset)
            current = best_by_bin.get(turn_bin)
            if current is None or self._sample_score(sample, node_mem, apply_tried_penalty=False) > self._sample_score(
                current,
                node_mem,
                apply_tried_penalty=False,
            ):
                best_by_bin[turn_bin] = sample

        if not best_by_bin:
            if node_mem is not None:
                node_mem.exhausted = True
            return None

        preferred_label = obs.topo_target_exit_label if obs.topo_target_exit_label in {"left", "straight", "right"} else None
        untried = [
            (turn_bin, sample)
            for turn_bin, sample in best_by_bin.items()
            if not node_mem or turn_bin not in node_mem.tried_bins
        ]
        if preferred_label is not None:
            preferred_sample = best_by_bin.get(preferred_label)
            if preferred_sample is not None and (
                node_mem is None or preferred_label not in node_mem.tried_bins
            ):
                best_label = preferred_label
                best = preferred_sample
            elif untried:
                best_label, best = max(untried, key=lambda item: self._sample_score(item[1], node_mem))
            elif preferred_sample is not None:
                best_label = preferred_label
                best = preferred_sample
            else:
                if node_mem is not None:
                    node_mem.exhausted = True
                return None
        else:
            if not untried:
                if node_mem is not None:
                    node_mem.exhausted = True
                return None
            best_label, best = max(untried, key=lambda item: self._sample_score(item[1], node_mem))

        if node_mem is not None:
            node_mem.tried_bins.add(best_label)
            node_mem.exhausted = False
        return best.offset, best_label

    def _sample_score(
        self,
        sample: ScanSample,
        node_mem: Optional[NodeMemory],
        *,
        apply_tried_penalty: bool = True,
    ) -> float:
        turn_bin = self._offset_bin(sample.offset)
        tried_penalty = 0.16 if apply_tried_penalty and node_mem and turn_bin in node_mem.tried_bins else 0.0
        straight_bonus = 0.05 * max(0.0, 1.0 - abs(sample.offset))
        clearance_bonus = 0.28 * sample.center_clearance
        return sample.similarity + clearance_bonus + straight_bonus - tried_penalty

    def _corridor_steering(self, obs: PolicyInput, *, approach_mode: bool = False) -> float:
        """Blend obstacle bias with explicit corridor-centering from side clearances."""
        imbalance = float(obs.left_clearance - obs.right_clearance)
        deadband = max(0.0, self.cfg.maze_centering_deadband)
        if abs(imbalance) < deadband:
            centering = 0.0
        else:
            centering = imbalance * self.cfg.maze_centering_gain

        if obs.center_clearance < max(0.35, self.cfg.maze_open_clearance * 0.85):
            centering *= self.cfg.maze_blocked_turn_gain

        blended = 0.65 * centering + 0.55 * float(obs.obstacle_steer_bias)
        if approach_mode:
            blended *= 0.8
        return float(np.clip(blended, -1.0, 1.0))

    def _corridor_speed_factor(self, obs: PolicyInput, *, approach_mode: bool = False) -> float:
        """Slow slightly when corridor balance is poor or the center lane tightens."""
        imbalance = abs(float(obs.left_clearance - obs.right_clearance))
        center_penalty = max(0.0, 0.55 - float(obs.center_clearance))
        occupancy_penalty = min(0.35, max(0.0, float(obs.near_field_occupancy) - 0.08))

        factor = 1.0
        factor -= min(0.22, imbalance * 0.35)
        factor -= center_penalty * 0.7
        factor -= occupancy_penalty
        if approach_mode:
            factor = min(factor, 0.9)
        return float(np.clip(factor, 0.45, 1.0))

    def _should_scan(self, obs: PolicyInput, now: float) -> bool:
        node_mem = self._node_memory.get(obs.topo_node_id)
        if (
            node_mem is not None
            and node_mem.exhausted
            and obs.topo_target_exit_label in (None, "back")
        ):
            return False

        dead_end = obs.center_clearance <= self.cfg.maze_dead_end_clearance
        node_like = self._looks_like_node(obs)
        if dead_end:
            return True

        if node_like:
            if obs.topo_node_id != self._last_scanned_node_id:
                return True
            if now - self._last_scan_finished >= self.cfg.maze_goal_rescan_interval:
                return True

        return False

    def _looks_like_node(self, obs: PolicyInput) -> bool:
        open_thresh = self.cfg.maze_open_clearance
        left_open = obs.left_clearance >= open_thresh
        right_open = obs.right_clearance >= open_thresh
        straight_open = obs.center_clearance >= max(0.35, open_thresh * 0.85)
        return (straight_open and (left_open or right_open)) or (left_open and right_open)

    def _ensure_node_memory(self, obs: PolicyInput, now: float):
        if obs.topo_node_id is None:
            return
        if obs.topo_node_id not in self._node_memory:
            self._node_memory[obs.topo_node_id] = NodeMemory(last_scan_time=now)

    def _handle_node_transition(self, obs: PolicyInput):
        current = obs.topo_node_id
        previous = self._last_topo_node_id
        self._node_changed_this_tick = bool(
            current is not None and previous is not None and current != previous
        )
        self._last_topo_node_id = current

    def _offset_bin(self, offset: float) -> str:
        threshold = max(0.05, self.cfg.maze_turn_bin_threshold)
        if offset >= threshold:
            return "left"
        if offset <= -threshold:
            return "right"
        return "straight"

    def reset(self):
        self._phase = MazePhase.BOOT
        self._phase_started = 0.0
        self._scan_samples.clear()
        self._align_duration = 0.0
        self._align_direction = 0.0
        self._pending_force_topo_node = False
        self._node_memory.clear()
        self._last_scanned_node_id = None
        self._last_topo_node_id = None
        self._node_changed_this_tick = False
        self._last_scan_finished = 0.0
        self._last_scan_started = 0.0
        self._last_similarity = 0.0
        self._backtrack_direction = 1.0
        self._active_exit_label = None
