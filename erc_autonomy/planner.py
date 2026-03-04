from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .bev_mapper import BEVResult
from .config import ERCConfig


@dataclass
class PathCandidate:
    """Single rollout candidate in the local robot frame."""

    curvature: float
    points_xy: np.ndarray  # shape [N,2], x=left(m), y=forward(m)
    score: float
    mean_trav: float
    min_trav: float


@dataclass
class PlannerOutput:
    """Planner result after top-k selection and path fusion."""

    fused_points_xy: np.ndarray  # shape [N,2]
    selected_curvatures: List[float]
    score: float
    mean_trav: float
    min_trav: float
    speed_hint: float  # [0,1]
    angular_hint: float  # [-1,1]
    mode: str  # "drive" | "stop"


class PathFusionPlanner:
    """
    Week 4 local planner:
    1) sample kinematically-feasible arc rollouts
    2) score by traversability + smoothness + goal-turn hint
    3) fuse top-k paths for stability
    """

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg

    def plan(self, bev: Optional[BEVResult], goal_turn_hint: float = 0.0) -> Optional[PlannerOutput]:
        if bev is None:
            return None

        curvatures = np.linspace(
            -self.cfg.planner_max_curvature,
            self.cfg.planner_max_curvature,
            max(3, self.cfg.planner_num_curvatures),
            dtype=np.float32,
        )
        desired_k = float(np.clip(goal_turn_hint, -1.0, 1.0) * self.cfg.planner_max_curvature)

        candidates: List[PathCandidate] = []
        for curvature in curvatures:
            points = self._rollout_arc(float(curvature))
            trav_samples = self._sample_traversability(bev.traversability, points)
            mean_trav = float(np.mean(trav_samples))
            min_trav = float(np.min(trav_samples))
            score = self._score_candidate(curvature=float(curvature), mean_trav=mean_trav, min_trav=min_trav, desired_k=desired_k)
            candidates.append(
                PathCandidate(
                    curvature=float(curvature),
                    points_xy=points,
                    score=score,
                    mean_trav=mean_trav,
                    min_trav=min_trav,
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.score, reverse=True)
        top_k = candidates[: max(1, min(self.cfg.planner_fuse_top_k, len(candidates)))]
        fused = self._fuse_paths(top_k)

        mean_trav = float(np.mean([c.mean_trav for c in top_k]))
        min_trav = float(np.mean([c.min_trav for c in top_k]))
        angular_hint = float(
            np.clip(
                np.mean([c.curvature for c in top_k]) / max(1e-6, self.cfg.planner_max_curvature),
                -1.0,
                1.0,
            )
        )
        speed_hint = float(np.clip((0.65 * mean_trav) + (0.35 * min_trav), 0.0, 1.0))
        mode = "drive" if min_trav >= self.cfg.planner_min_trav_for_motion else "stop"
        score = float(top_k[0].score)

        return PlannerOutput(
            fused_points_xy=fused,
            selected_curvatures=[c.curvature for c in top_k],
            score=score,
            mean_trav=mean_trav,
            min_trav=min_trav,
            speed_hint=speed_hint,
            angular_hint=angular_hint,
            mode=mode,
        )

    def _score_candidate(self, curvature: float, mean_trav: float, min_trav: float, desired_k: float) -> float:
        max_k = max(1e-6, self.cfg.planner_max_curvature)
        goal_align = 1.0 - min(1.0, abs(curvature - desired_k) / (2.0 * max_k))
        curvature_cost = abs(curvature) / max_k
        return float(
            (self.cfg.planner_score_mean_w * mean_trav)
            + (self.cfg.planner_score_min_w * min_trav)
            + (self.cfg.planner_score_goal_w * goal_align)
            - (self.cfg.planner_score_curvature_penalty_w * curvature_cost)
        )

    def _rollout_arc(self, curvature: float) -> np.ndarray:
        s_vals = np.linspace(
            0.0,
            max(0.5, self.cfg.planner_horizon_m),
            max(6, self.cfg.planner_num_points),
            dtype=np.float32,
        )

        points = np.zeros((s_vals.shape[0], 2), dtype=np.float32)
        if abs(curvature) < 1e-5:
            points[:, 0] = 0.0
            points[:, 1] = s_vals
            return points

        radius = 1.0 / curvature
        theta = s_vals * curvature
        points[:, 0] = radius * (1.0 - np.cos(theta))
        points[:, 1] = radius * np.sin(theta)
        return points

    def _sample_traversability(self, bev_t: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        h, w = bev_t.shape[:2]
        width_m = max(0.5, self.cfg.bev_width_m)
        depth_m = max(0.5, self.cfg.bev_depth_m)

        samples = np.zeros((points_xy.shape[0],), dtype=np.float32)
        for i, (x_m, y_m) in enumerate(points_xy):
            # Map local metric (x left, y forward) to BEV raster:
            # y=0 -> bottom row (near); y=depth -> top row (far)
            col = ((x_m + (width_m * 0.5)) / width_m) * (w - 1)
            row = (1.0 - (y_m / depth_m)) * (h - 1)

            if row < 0 or row > (h - 1) or col < 0 or col > (w - 1):
                samples[i] = 0.0
                continue

            samples[i] = self._bilinear(bev_t, row, col)
        return samples

    @staticmethod
    def _bilinear(grid: np.ndarray, row: float, col: float) -> float:
        r0 = int(np.floor(row))
        c0 = int(np.floor(col))
        r1 = min(r0 + 1, grid.shape[0] - 1)
        c1 = min(c0 + 1, grid.shape[1] - 1)
        dr = row - r0
        dc = col - c0

        v00 = float(grid[r0, c0])
        v10 = float(grid[r1, c0])
        v01 = float(grid[r0, c1])
        v11 = float(grid[r1, c1])

        v0 = v00 * (1.0 - dr) + v10 * dr
        v1 = v01 * (1.0 - dr) + v11 * dr
        return float(v0 * (1.0 - dc) + v1 * dc)

    @staticmethod
    def _fuse_paths(candidates: List[PathCandidate]) -> np.ndarray:
        if len(candidates) == 1:
            return candidates[0].points_xy.copy()

        scores = np.array([c.score for c in candidates], dtype=np.float32)
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / max(1e-6, np.sum(weights))

        fused = np.zeros_like(candidates[0].points_xy, dtype=np.float32)
        for w, cand in zip(weights, candidates):
            fused += w * cand.points_xy
        return fused

