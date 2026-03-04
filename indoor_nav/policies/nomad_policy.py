"""
NoMaD (Goal-Conditioned Diffusion Policy) integration — SOTA 2025.

NoMaD [1] is the state-of-the-art for image-goal navigation. It:
  - Takes current obs + goal image → predicts a sequence of waypoints
  - Uses a diffusion model over actions for multi-modal planning
  - Was trained on large-scale navigation data (GNM dataset + more)
  - Builds on ViNT [2], a Vision-based Navigation Transformer that provides
    the backbone architecture for goal-conditioned navigation.

This module wraps the NoMaD model for use with the Earth Rovers SDK.

References:
  [1] Sridhar et al., "NoMaD: Goal Masked Diffusion Policies for Navigation
      and Exploration", ICRA 2024, arXiv:2310.07896.
  [2] Shah et al., "ViNT: A Foundation Model for Visual Navigation", CoRL 2023,
      arXiv:2306.14846.

Setup:
  git clone https://github.com/robodhruv/visualnav-transformer.git
  # Follow their setup instructions for model weights
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from indoor_nav.configs.config import PolicyConfig
from indoor_nav.policies.base_policy import BasePolicy, PolicyInput, PolicyOutput

logger = logging.getLogger(__name__)


class NoMaDPolicy(BasePolicy):
    """
    NoMaD goal-conditioned diffusion policy.

    Predicts waypoints in the robot's egocentric frame, which are then
    converted to (linear, angular) velocities via a simple pursuit controller.
    """

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self._model = None
        self._context_queue: deque = deque(maxlen=cfg.context_length)
        self._device = None

    def setup(self):
        """Load the NoMaD model."""
        try:
            import torch
            self._device = torch.device(
                self.cfg.device if torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            logger.warning("PyTorch not installed. Using heuristic fallback.")
            self._model = None
            return

        logger.info("Loading NoMaD model from %s on %s", self.cfg.model_path, self._device)

        try:
            # Try to load the pre-trained NoMaD model
            # The model expects: (context_images, goal_image) → waypoints
            self._model = torch.jit.load(self.cfg.model_path, map_location=self._device)
            self._model.eval()
            logger.info("NoMaD model loaded successfully.")
        except (FileNotFoundError, RuntimeError, ValueError, OSError):
            logger.warning(
                "NoMaD model not found at %s. "
                "Using heuristic fallback. Download weights from: "
                "https://github.com/robodhruv/visualnav-transformer",
                self.cfg.model_path,
            )
            self._model = None

    def predict(self, obs: PolicyInput) -> PolicyOutput:
        """Predict next action from current observation + goal."""
        # Update context queue
        self._context_queue.append(obs.front_image.copy())

        if self._model is not None:
            return self._predict_nomad(obs)
        else:
            return self._predict_heuristic(obs)

    def _predict_nomad(self, obs: PolicyInput) -> PolicyOutput:
        """Run the actual NoMaD model."""
        import torch

        # Prepare context stack: list of recent frames
        context = list(self._context_queue)
        while len(context) < self.cfg.context_length:
            context.insert(0, context[0])  # pad with first frame

        # Resize and normalize
        size = self.cfg.image_size  # (W, H)
        context_tensors = []
        for img in context:
            resized = cv2.resize(img, size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            context_tensors.append(t)

        # Goal image
        goal_resized = cv2.resize(obs.goal_image, self.cfg.goal_image_size)
        goal_rgb = cv2.cvtColor(goal_resized, cv2.COLOR_BGR2RGB)
        goal_tensor = torch.from_numpy(goal_rgb).permute(2, 0, 1).float() / 255.0

        # Stack context: (1, T, C, H, W)
        context_batch = torch.stack(context_tensors).unsqueeze(0).to(self._device)
        goal_batch = goal_tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            # NoMaD outputs waypoints in egocentric frame: (1, N, 2) = (x, y)
            waypoints = self._model(context_batch, goal_batch)

        waypoints = waypoints.cpu().numpy()[0]  # (N, 2)

        # Convert waypoints to velocity commands via pursuit controller
        linear, angular = self._waypoints_to_velocity(waypoints)

        # Apply obstacle modulation
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=0.8,
            waypoints=waypoints,
        )

    def _predict_heuristic(self, obs: PolicyInput) -> PolicyOutput:
        """
        Heuristic fallback when no trained model is available.

        Strategy: Use goal similarity + trend to decide basic motions.
        - If similarity is increasing → go forward
        - If similarity is decreasing → turn to search
        - If high similarity → slow down and center
        """
        sim = obs.goal_similarity
        trend = obs.goal_trend

        # Base forward speed — higher when we're making progress
        if sim > 0.7:
            # Close to goal — approach cautiously
            linear = 0.2
            angular = 0.0
            confidence = 0.8
        elif trend > 0.01:
            # Getting closer — go forward
            linear = 0.4
            angular = 0.0
            confidence = 0.5
        elif trend < -0.01:
            # Moving away — need to search
            linear = 0.1
            angular = 0.3  # turn to search
            confidence = 0.3
        else:
            # Neutral — explore forward
            linear = 0.3
            angular = 0.0
            confidence = 0.4

        # Apply obstacle modulation
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=confidence,
        )

    def _waypoints_to_velocity(self, waypoints: np.ndarray) -> tuple:
        """
        Pure-pursuit controller: follow the first predicted waypoint.

        Waypoints are in egocentric frame: +x = forward, +y = left.
        """
        if waypoints.shape[0] == 0:
            return 0.0, 0.0

        # Look at the 2nd waypoint for smoother control (skip immediate one)
        idx = min(1, waypoints.shape[0] - 1)
        target_x = waypoints[idx, 0]  # forward distance
        target_y = waypoints[idx, 1]  # lateral offset

        # Steering: proportional to lateral offset
        angular = np.clip(target_y * 2.0, -1.0, 1.0)

        # Speed: proportional to forward distance, reduced when turning
        linear = np.clip(target_x * 1.5, -1.0, 1.0)
        linear *= max(0.3, 1.0 - abs(angular) * 0.5)  # slow down in turns

        return float(linear), float(angular)

    def reset(self):
        self._context_queue.clear()
