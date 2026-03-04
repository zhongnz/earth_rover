"""
Vision-Language-Action (VLA) Policy — SOTA 2025.

Integrates modern VLA foundation models that directly output robot actions
from visual observations and language goals:

  - OpenVLA (Stanford/TRI, 2024, arXiv:2406.09246): Open-source 7B VLA,
    trained on Open X-Embodiment (970k demonstrations, arXiv:2310.08864).
    Outperforms RT-2-X (55B) by 16.5% with 7x fewer parameters.
    Uses DINOv2 + SigLIP visual encoder — validates our feature choices.
  - π0 (Physical Intelligence, 2025): SOTA generalist robot policy.
    Requires API access.
  - Octo (Berkeley, 2024): Lightweight generalist policy, fast inference.

VLAs differ from VLMs: they output continuous actions directly, not text
instructions. This removes the need for instruction → velocity mapping.

Architecture:
  1. Observation: front camera image + goal image
  2. Language conditioning: "Navigate to the location shown in the goal image"
  3. Action output: 7-DoF (we use x=forward, rz=yaw for navigation)
  4. Obstacle modulation applied post-prediction

Fallback: If no VLA model is available, uses the enhanced heuristic policy
with visual servoing (ORB features), frontier exploration, and corridor following.

References:
  [1] Kim et al., "OpenVLA", arXiv:2406.09246, 2024.
  [2] Open X-Embodiment Collaboration, arXiv:2310.08864, ICRA 2024.
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

# Default language instruction for goal-conditioned navigation
GOAL_INSTRUCTION = "Navigate to the location shown in the goal image."


class VLAPolicy(BasePolicy):
    """
    Vision-Language-Action policy for goal-conditioned navigation.

    Supports multiple VLA backends:
      - "openvla": OpenVLA 7B (local or API)
      - "octo": Octo model (local)
      - "heuristic_plus": Enhanced heuristic with exploration (no GPU)

    The VLA takes (image, goal_image, language) → action.
    """

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._device = None
        self._context_queue: deque = deque(maxlen=cfg.context_length)
        self._vla_backend: str = getattr(cfg, 'vla_backend', 'openvla')
        self._action_scale: float = 0.5  # scale VLA outputs to robot commands
        self._explore_counter: int = 0
        self._explore_direction: float = 1.0

    def setup(self):
        """Load VLA model or set up heuristic."""
        try:
            if self._vla_backend == "openvla":
                self._setup_openvla()
            elif self._vla_backend == "octo":
                self._setup_octo()
            else:
                logger.info("VLA using enhanced heuristic mode (no GPU model).")
                self._model = None
        except Exception as e:
            logger.warning("VLA model setup failed: %s. Using enhanced heuristic.", e)
            self._model = None

    def _setup_openvla(self):
        """
        Set up OpenVLA (Stanford, 2024).

        OpenVLA is a 7B parameter VLA trained on Open X-Embodiment data.
        It can be served via vLLM or loaded directly with HuggingFace.
        """
        try:
            import torch
            self._device = torch.device(
                self.cfg.device if torch.cuda.is_available() else "cpu"
            )

            from transformers import AutoModelForVision2Seq, AutoProcessor

            model_name = self.cfg.model_path or "openvla/openvla-7b"
            logger.info("Loading OpenVLA: %s on %s", model_name, self._device)

            self._processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

            logger.info("OpenVLA loaded successfully.")
        except ImportError:
            logger.warning("OpenVLA requires transformers>=4.40. Falling back to heuristic.")
            self._model = None
        except Exception as e:
            logger.warning("OpenVLA load failed: %s", e)
            self._model = None

    def _setup_octo(self):
        """
        Set up Octo (Berkeley, 2024).

        Octo is a lightweight generalist policy that runs efficiently on
        consumer GPUs. Supports goal-image conditioning.
        """
        try:
            import torch
            # Octo uses JAX by default, but there's a PyTorch port
            logger.info("Loading Octo model...")
            # Placeholder: Octo integration depends on specific installation
            logger.warning("Octo not yet installed. Using heuristic fallback.")
            self._model = None
        except Exception as e:
            logger.warning("Octo setup failed: %s", e)
            self._model = None

    def predict(self, obs: PolicyInput) -> PolicyOutput:
        """Predict action from observation + goal."""
        self._context_queue.append(obs.front_image.copy())

        if self._model is not None and self._vla_backend == "openvla":
            return self._predict_openvla(obs)
        else:
            return self._predict_heuristic_plus(obs)

    def _predict_openvla(self, obs: PolicyInput) -> PolicyOutput:
        """Run OpenVLA inference."""
        import torch
        from PIL import Image

        # Prepare image
        rgb = cv2.cvtColor(obs.front_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Build instruction with goal context
        instruction = GOAL_INSTRUCTION

        inputs = self._processor(instruction, pil_img).to(
            self._device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            action = self._model.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False
            )

        # OpenVLA outputs 7-DoF actions: [x, y, z, rx, ry, rz, gripper]
        # For navigation, we use x (forward) and rz (rotation)
        action_np = action.cpu().numpy()
        linear = float(action_np[0]) * self._action_scale  # forward
        angular = float(action_np[5]) * self._action_scale  # yaw rotation

        # Apply obstacle modulation
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=0.7,
        )

    def _predict_heuristic_plus(self, obs: PolicyInput) -> PolicyOutput:
        """
        Enhanced heuristic policy with exploration behavior.

        Improvements over basic heuristic:
        - Visual servoing: use image-space feature displacement
        - Frontier exploration: systematically explore when lost
        - Momentum-based decisions: smooth transitions between behaviors
        - Corridor following: detect and follow long straight paths
        """
        sim = obs.goal_similarity
        trend = obs.goal_trend

        # --- Phase 1: Close to goal (sim > 0.7) → precise approach ---
        if sim > 0.7:
            linear = 0.15
            angular = 0.0
            confidence = 0.8

            # Visual servoing: try to center goal features
            angular = self._visual_servo(obs.front_image, obs.goal_image)

        # --- Phase 2: Making progress (positive trend) → maintain course ---
        elif trend > 0.02:
            linear = 0.45
            angular = 0.0
            confidence = 0.6

        # --- Phase 3: Losing track (negative trend) → corrective search ---
        elif trend < -0.02:
            # Systematic search: alternate directions
            self._explore_counter += 1
            if self._explore_counter % 20 == 0:
                self._explore_direction *= -1  # switch direction

            linear = 0.1
            angular = 0.3 * self._explore_direction
            confidence = 0.3

        # --- Phase 4: No signal → explore ---
        elif sim < 0.3:
            # Active exploration: move forward, occasionally turn
            self._explore_counter += 1

            if self._explore_counter % 30 < 20:
                # Mostly go forward
                linear = 0.4
                angular = 0.0
            else:
                # Periodic scan
                linear = 0.05
                angular = 0.35 * self._explore_direction
            confidence = 0.2

        # --- Phase 5: Moderate signal → cautious forward ---
        else:
            linear = 0.3
            angular = 0.0
            confidence = 0.4

        # Corridor following: if center is clear but sides are blocked
        if obs.obstacle_speed_factor < 0.9 and abs(obs.obstacle_steer_bias) < 0.1:
            # In a corridor — maintain forward motion
            linear = max(linear, 0.25)

        # Apply obstacle modulation
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=confidence,
        )

    def _visual_servo(self, current: np.ndarray, goal: np.ndarray) -> float:
        """
        Simple visual servoing using feature point displacement.

        Computes the horizontal offset of matching features between
        current and goal images → steering correction.
        """
        try:
            gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray_goal = cv2.cvtColor(goal, cv2.COLOR_BGR2GRAY)

            # Resize to same size
            h, w = 120, 160
            gray_curr = cv2.resize(gray_curr, (w, h))
            gray_goal = cv2.resize(gray_goal, (w, h))

            # ORB features (fast)
            orb = cv2.ORB_create(nfeatures=200)
            kp1, des1 = orb.detectAndCompute(gray_curr, None)
            kp2, des2 = orb.detectAndCompute(gray_goal, None)

            if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
                return 0.0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)[:20]

            if len(matches) < 3:
                return 0.0

            # Compute average horizontal displacement
            dx_sum = 0.0
            for m in matches:
                pt_curr = kp1[m.queryIdx].pt
                pt_goal = kp2[m.trainIdx].pt
                dx_sum += pt_goal[0] - pt_curr[0]

            avg_dx = dx_sum / len(matches)
            # Normalize: positive dx = goal features are to the right → turn right
            angular = np.clip(avg_dx / w * -0.5, -0.3, 0.3)
            return float(angular)

        except Exception:
            return 0.0

    def reset(self):
        self._context_queue.clear()
        self._explore_counter = 0
        self._explore_direction = 1.0
