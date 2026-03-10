"""
Goal / checkpoint manager for image-goal navigation.

Manages the sequence of image goals, computes visual similarity between
the current observation and the goal, and decides when a checkpoint is reached.

Supports multiple SOTA feature backends (2025):
  - DINOv2-VLAD  (AnyLoc-style: DINOv2-reg4 patch tokens + VLAD aggregation —
                   SOTA VPR, arXiv:2308.00688 + arXiv:2309.16588)
  - DINOv3-VLAD  (DINOv3 patch tokens + VLAD aggregation, arXiv:2508.10104)
  - dinov2_direct (DINOv2 mean-pooled patch tokens — image matching, not VPR)
  - wall_crop_direct (detect rectangular wall-image crops, then DINOv2 direct match)
  - wall_rectify_direct (detect wall-image quadrilaterals, rectify, then DINOv2 direct match)
  - SigLIP2      (Google 2025, best open vision encoder, arXiv:2502.14786)
  - DINOv2       (strong spatial features via CLS token, arXiv:2304.07193)
  - CLIP         (semantic matching baseline, Radford et al. 2021)
  - EigenPlaces  (ICCV 2023, viewpoint-robust VPR, arXiv:2308.10832)
  - CosPlace     (CVPR 2022, compact trained VPR descriptor, arXiv:2204.02287)
  - SIFT         (geometric matching verification, Lowe 2004)

The recommended pipeline for competition:
  Primary: DINOv2-VLAD or SigLIP2 (high recall)
  Verification: SIFT geometric check (high precision)

References:
  [1] Oquab et al., "DINOv2: Learning Robust Visual Features without
      Supervision", arXiv:2304.07193, 2023.
  [2] Darcet et al., "Vision Transformers Need Registers", ICLR 2024,
      arXiv:2309.16588.
  [3] Keetha et al., "AnyLoc: Towards Universal Visual Place Recognition",
      IEEE RA-L 2023, arXiv:2308.00688.
  [4] Tschannen et al., "SigLIP 2", arXiv:2502.14786, Feb 2025.
  [5] Berton et al., "EigenPlaces", ICCV 2023, arXiv:2308.10832.
  [6] Berton et al., "Rethinking Visual Geo-localization for Large-Scale
      Applications", CVPR 2022, arXiv:2204.02287.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.registry import build_backend

logger = logging.getLogger(__name__)


@dataclass
class GoalCheckpoint:
    """A single image-goal checkpoint."""
    index: int                     # sequence number (1-based)
    image_path: str                # path to goal image on disk
    image: Optional[np.ndarray] = None       # loaded BGR image
    feature: Optional[Any] = None            # backend-specific prepared goal
    reached: bool = False
    reached_time: float = 0.0


class GoalMatcher:
    """
    Computes similarity between current observation and goal image.

    SOTA backends (2025):
      - dinov2_vlad: AnyLoc-style multi-scale DINOv2 patch tokens + VLAD aggregation.
        Purpose-built for visual place recognition. Best for indoor navigation.
      - dinov3_vlad: DINOv3 patch tokens + VLAD aggregation.
        Stronger dense features, useful for A/B against DINOv2-VLAD.
      - dinov2_direct: Mean-pooled DINOv2 patch tokens (no VLAD). Compares images
        directly rather than recognising places. Best for finding the matching image.
      - wall_crop_direct: Contour-based wall crop proposals scored with DINOv2-direct.
        Best when the target is a poster, framed print, or wall-mounted screen.
      - wall_rectify_direct: Quadrilateral wall proposals rectified before DINOv2-direct.
        Best when the target is visible on the wall at a strong angle.
      - siglip2: Google's SigLIP2 (2025) — superior to CLIP with sigmoid loss.
        Excellent semantic understanding for goal matching.
      - dinov2: DINOv2 CLS token baseline (strong but not VPR-specific).
      - eigenplaces: Trained encoder specifically for place recognition.
      - cosplace: Compact, trained place-recognition descriptor from CVPR 2022.
      - sift: Classical geometric matching (great as verification stage).

    Recommended for image matching: dinov2_direct.
    Recommended for place recognition: dinov2_vlad (primary) + sift (verification).
    """

    def __init__(self, cfg: GoalConfig):
        self.cfg = cfg
        self._backend = build_backend(cfg)

    def prepare_goal(self, image: np.ndarray) -> Any:
        return self._backend.prepare_goal(image)

    def prepare_query(self, image: np.ndarray) -> Any:
        return self._backend.prepare_query(image)

    def score_prepared(self, query: Any, goal: Any) -> float:
        return self._backend.score(query, goal)

    def extract_feature(self, image: np.ndarray) -> Any:
        """Compatibility helper for tooling that expects a direct extractor."""
        return self.prepare_query(image).payload

    def compute_similarity(self, obs_image: np.ndarray, goal: GoalCheckpoint) -> float:
        if obs_image is None or goal.image is None:
            return 0.0
        query = self.prepare_query(obs_image)
        if goal.feature is None:
            goal.feature = self.prepare_goal(goal.image)
        return self.score_prepared(query, goal.feature)


class CheckpointManager:
    """
    Manages the ordered sequence of image-goal checkpoints.

    Tracks which checkpoint is the current target, computes similarity,
    and decides when to advance to the next checkpoint.
    """

    def __init__(self, cfg: GoalConfig):
        self.cfg = cfg
        self.matcher = GoalMatcher(cfg)
        self.checkpoints: List[GoalCheckpoint] = []
        self.current_idx: int = 0
        self._similarity_history: deque = deque(maxlen=30)
        self._above_threshold_count: int = 0

    def load_goals(self, goal_images: List[str]):
        """
        Load goal images from file paths.

        Args:
            goal_images: Ordered list of image file paths for each checkpoint.
        """
        self.checkpoints = []
        for i, path in enumerate(goal_images):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Could not load goal image: %s", path)
                continue
            cp = GoalCheckpoint(index=i + 1, image_path=path, image=img)
            self.checkpoints.append(cp)
            logger.info("Loaded goal %d: %s (%dx%d)", i + 1, path, img.shape[1], img.shape[0])

        # Precompute features for all goals
        logger.info("Precomputing features for %d goals...", len(self.checkpoints))
        for cp in self.checkpoints:
            cp.feature = self.matcher.prepare_goal(cp.image)
        logger.info("Goal features ready.")

        self.current_idx = 0
        self._above_threshold_count = 0

    def load_goals_from_dir(self, directory: str):
        """Load goal images from a directory, sorted by filename."""
        if not os.path.isdir(directory):
            logger.error("Goal directory not found: %s", directory)
            return
        files = sorted(
            f for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        paths = [os.path.join(directory, f) for f in files]
        logger.info("Found %d goal images in %s", len(paths), directory)
        self.load_goals(paths)

    @property
    def current_goal(self) -> Optional[GoalCheckpoint]:
        if 0 <= self.current_idx < len(self.checkpoints):
            return self.checkpoints[self.current_idx]
        return None

    @property
    def all_done(self) -> bool:
        return self.current_idx >= len(self.checkpoints)

    @property
    def progress(self) -> Tuple[int, int]:
        """Return (completed, total)."""
        return self.current_idx, len(self.checkpoints)

    def compute_goal_similarity(self, observation: np.ndarray) -> float:
        """
        Compute similarity of the current observation to the active goal.
        Returns 0.0 if no active goal.
        """
        goal = self.current_goal
        if goal is None or observation is None:
            return 0.0

        sim = self.matcher.compute_similarity(observation, goal)
        self._similarity_history.append(sim)
        return sim

    def check_arrival(self, similarity: float) -> bool:
        """
        Check if we've arrived at the current checkpoint.

        Requires `match_patience` consecutive frames above `match_threshold`.
        Returns True if checkpoint is reached (and advances to next).
        """
        if similarity >= self.cfg.match_threshold:
            self._above_threshold_count += 1
        else:
            self._above_threshold_count = 0

        if self._above_threshold_count >= self.cfg.match_patience:
            return self._mark_reached()
        return False

    def _mark_reached(self) -> bool:
        """Mark current checkpoint as reached and advance."""
        goal = self.current_goal
        if goal is None:
            return False

        goal.reached = True
        goal.reached_time = time.time()
        logger.info(
            "CHECKPOINT %d REACHED (similarity history: %s)",
            goal.index,
            [f"{s:.2f}" for s in list(self._similarity_history)[-5:]],
        )
        self.current_idx += 1
        self._above_threshold_count = 0
        self._similarity_history.clear()
        return True

    def get_similarity_trend(self) -> float:
        """
        Return the trend of similarity over recent frames.
        Positive = getting closer, negative = moving away.
        """
        hist = list(self._similarity_history)
        if len(hist) < 4:
            return 0.0
        recent = np.mean(hist[-3:])
        older = np.mean(hist[-6:-3]) if len(hist) >= 6 else np.mean(hist[:3])
        return float(recent - older)

    def status_str(self) -> str:
        done, total = self.progress
        goal = self.current_goal
        recent_sim = list(self._similarity_history)[-1] if self._similarity_history else 0.0
        return (
            f"Goal {done + 1}/{total} | "
            f"Sim: {recent_sim:.3f} (thresh: {self.cfg.match_threshold:.2f}) | "
            f"Patience: {self._above_threshold_count}/{self.cfg.match_patience}"
        )
