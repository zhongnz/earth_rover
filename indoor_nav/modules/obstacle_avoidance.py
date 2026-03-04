"""
Obstacle avoidance module using monocular depth estimation — SOTA 2025.

Supports multiple depth backends:
  - Depth Anything V2 (TikTok/ByteDance, NeurIPS 2024, arXiv:2406.09414):
    Fast relative depth, excellent for real-time obstacle detection. Default.
  - Depth Pro (Apple, ICLR 2025, arXiv:2410.02073):
    Metric depth with absolute scale. Best for precise distance estimation.
  - simple_edge: Fallback using edge density (no GPU needed).

Uses the depth map to detect near-field obstacles and outputs:
  - speed_factor: multiplicative speed reduction [0, 1]
  - steer_bias: angular correction to avoid obstacles [-1, 1]
  - zone clearances: per-zone (L/C/R) obstacle density

Enhanced features (2025):
  - Temporal smoothing of depth maps (EMA, reduces flickering)
  - Configurable detection zones (near/mid/far field)
  - Dynamic slowdown curve (smooth exponential, not step-function)
  - Narrow passage detection (both sides blocked → center only)
  - Anticipatory mid-field slowdown for early obstacle avoidance

References:
  [1] Yang et al., \"Depth Anything V2\", NeurIPS 2024, arXiv:2406.09414.
  [2] Bochkovskii et al., \"Depth Pro: Sharp Monocular Metric Depth in
      Less Than a Second\", ICLR 2025, arXiv:2410.02073.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from indoor_nav.configs.config import ObstacleConfig

logger = logging.getLogger(__name__)


@dataclass
class ObstacleInfo:
    """Result of obstacle detection for one frame."""
    has_obstacle: bool = False
    emergency_stop: bool = False
    speed_factor: float = 1.0          # multiply into forward speed [0, 1]
    steer_bias: float = 0.0            # added angular offset [-1, 1]
    near_field_occupancy: float = 0.0  # fraction of near-field pixels that are close
    left_clearance: float = 1.0        # 0 = blocked, 1 = clear
    right_clearance: float = 1.0
    center_clearance: float = 1.0
    depth_map: Optional[np.ndarray] = None
    narrow_passage: bool = False       # both sides have obstacles


class ObstacleDetector:
    """
    Monocular depth-based obstacle avoidance with temporal smoothing.

    Splits the near-field (bottom portion) of the depth map into
    left / center / right zones and steers away from obstacles.
    Includes narrow-passage detection and dynamic slowdown curves.
    """

    def __init__(self, cfg: ObstacleConfig):
        self.cfg = cfg
        self._model = None
        self._device = None
        self._transform = None
        self._depth_pipe = None
        self._depth_history: deque = deque(maxlen=3)  # temporal smoothing

    def _ensure_model(self):
        if self._model is not None:
            return

        if self.cfg.method == "depth_anything":
            self._load_depth_anything()
        elif self.cfg.method == "depth_pro":
            self._load_depth_pro()
        elif self.cfg.method == "simple_edge":
            self._model = "simple"
        else:
            raise ValueError(f"Unknown obstacle method: {self.cfg.method}")

    def _load_depth_anything(self):
        import torch
        from transformers import pipeline

        device_str = self.cfg.depth_device if torch.cuda.is_available() else "cpu"
        logger.info("Loading Depth Anything: %s on %s", self.cfg.depth_model, device_str)
        self._depth_pipe = pipeline(
            "depth-estimation",
            model=self.cfg.depth_model,
            device=device_str,
        )
        self._model = "depth_anything"
        logger.info("Depth model loaded.")

    def _load_depth_pro(self):
        """
        Load Depth Pro (Apple, 2024) for metric monocular depth.

        Depth Pro produces absolute metric depth (meters) without camera
        intrinsics. Best accuracy but slower than Depth Anything.
        Falls back to Depth Anything if not available.
        """
        try:
            import torch
            # Depth Pro is available via transformers or the Apple repo
            from transformers import pipeline

            model_name = self.cfg.depth_model or "apple/DepthPro"
            device_str = self.cfg.depth_device if torch.cuda.is_available() else "cpu"
            logger.info("Loading Depth Pro: %s on %s", model_name, device_str)
            self._depth_pipe = pipeline(
                "depth-estimation",
                model=model_name,
                device=device_str,
            )
            self._model = "depth_pro"
            logger.info("Depth Pro loaded (metric depth).")
        except Exception as e:
            logger.warning("Depth Pro not available (%s). Falling back to Depth Anything.", e)
            self.cfg.method = "depth_anything"
            self.cfg.depth_model = "depth-anything/Depth-Anything-V2-Base-hf"
            self._load_depth_anything()

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Return a depth map (H, W) with values in [0, 1].
        0 = close, 1 = far.

        Applies temporal smoothing across recent frames to reduce flickering.
        """
        self._ensure_model()

        if self.cfg.method in ("depth_anything", "depth_pro"):
            from PIL import Image as PILImage
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            result = self._depth_pipe(pil_img)
            depth = np.array(result["depth"], dtype=np.float32)
            # Normalize to [0, 1] where 0 = near, 1 = far
            dmin, dmax = depth.min(), depth.max()
            if dmax - dmin > 1e-6:
                depth = (depth - dmin) / (dmax - dmin)
            else:
                depth = np.ones_like(depth)

            # Temporal smoothing
            self._depth_history.append(depth)
            if len(self._depth_history) > 1:
                # Exponential moving average
                alpha = 0.6
                smoothed = self._depth_history[-1] * alpha
                for prev in list(self._depth_history)[:-1]:
                    if prev.shape == smoothed.shape:
                        smoothed += prev * (1 - alpha) / max(1, len(self._depth_history) - 1)
                depth = smoothed

            return depth

        elif self.cfg.method == "simple_edge":
            return self._simple_edge_depth(image)

        return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

    def _simple_edge_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback: use edge density in grid cells as a rough obstacle proxy.
        More edges in the bottom = closer obstacles.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Pseudo depth: more edges → closer → lower depth value
        blurred = cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)
        blurred = blurred / (blurred.max() + 1e-6)
        # Invert: high edge density → low depth (close)
        depth = 1.0 - blurred
        return depth

    def detect(self, image: np.ndarray) -> ObstacleInfo:
        """
        Analyze a front camera frame for obstacles.

        Returns ObstacleInfo with speed modulation, steering bias,
        narrow-passage detection, and per-zone clearances.
        """
        if not self.cfg.enabled:
            return ObstacleInfo()

        self._ensure_model()
        depth = self.estimate_depth(image)
        h, w = depth.shape

        # Near-field: bottom portion of image
        near_y = int(h * (1.0 - self.cfg.min_clearance_frac))
        near_field = depth[near_y:, :]

        # Mid-field: middle portion (for look-ahead planning)
        mid_y_start = int(h * 0.4)
        mid_y_end = near_y
        mid_field = depth[mid_y_start:mid_y_end, :]

        # Obstacle threshold: pixels with depth < 0.3 are "close"
        obstacle_mask = near_field < 0.3
        occupancy = float(obstacle_mask.sum()) / max(1, obstacle_mask.size)

        # Mid-field obstacles (for early slowdown)
        mid_obstacle_mask = mid_field < 0.25
        mid_occupancy = float(mid_obstacle_mask.sum()) / max(1, mid_obstacle_mask.size)

        # Split into left / center / right thirds
        third = w // 3
        left_occ = float(obstacle_mask[:, :third].sum()) / max(1, obstacle_mask[:, :third].size)
        center_occ = float(obstacle_mask[:, third:2*third].sum()) / max(1, obstacle_mask[:, third:2*third].size)
        right_occ = float(obstacle_mask[:, 2*third:].sum()) / max(1, obstacle_mask[:, 2*third:].size)

        left_clear = 1.0 - left_occ
        center_clear = 1.0 - center_occ
        right_clear = 1.0 - right_occ

        # Narrow passage detection: both sides have obstacles but center is clear
        narrow_passage = (left_occ > 0.15 and right_occ > 0.15 and center_occ < 0.1)

        # Emergency stop
        emergency = occupancy >= self.cfg.emergency_stop_frac

        # Dynamic speed factor: smooth curve instead of step function
        if emergency:
            speed_factor = 0.0
        elif center_occ > 0.05:
            # Smooth exponential slowdown based on center obstacle density
            speed_factor = max(
                self.cfg.obstacle_slowdown,
                np.exp(-3.0 * center_occ)  # smooth decay
            )
            # Also consider mid-field for anticipatory slowdown
            speed_factor *= max(0.5, 1.0 - mid_occupancy)
        else:
            # Anticipatory slowdown from mid-field
            speed_factor = max(0.7, 1.0 - mid_occupancy * 0.5)

        # Narrow passage: slow down but go straight
        if narrow_passage:
            speed_factor = min(speed_factor, 0.4)

        # Steering bias: steer away from obstacles
        steer_bias = 0.0
        if not narrow_passage:
            # Normal mode: steer toward the clearer side
            if left_occ > 0.1 or right_occ > 0.1:
                # Positive angular = turn left, so if left is blocked, steer right (negative)
                steer_bias = (left_occ - right_occ) * -0.6
        else:
            # Narrow passage: stay centered, only small corrections
            imbalance = (left_occ - right_occ)
            steer_bias = imbalance * -0.2  # gentle centering

        has_obstacle = occupancy > 0.05

        return ObstacleInfo(
            has_obstacle=has_obstacle,
            emergency_stop=emergency,
            speed_factor=speed_factor,
            steer_bias=steer_bias,
            near_field_occupancy=occupancy,
            left_clearance=left_clear,
            right_clearance=right_clear,
            center_clearance=center_clear,
            depth_map=depth,
            narrow_passage=narrow_passage,
        )
