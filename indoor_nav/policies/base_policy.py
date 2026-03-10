"""
Abstract base class for navigation policies.

All policies receive the current observation + goal image and output
(linear, angular) velocity commands.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PolicyInput:
    """Observation bundle passed to the policy each tick."""
    front_image: np.ndarray                          # current front camera (BGR)
    goal_image: np.ndarray                           # target image (BGR)
    goal_similarity: float                           # current similarity to goal
    goal_trend: float                                # similarity trend (+ = closer)
    context_images: List[np.ndarray]                 # recent past frames
    orientation: float = 0.0                         # IMU heading
    speed: float = 0.0                               # current speed from telemetry
    obstacle_speed_factor: float = 1.0               # from obstacle detector
    obstacle_steer_bias: float = 0.0                 # from obstacle detector
    slam_tracking_state: str = "NOT_INITIALIZED"     # SLAM sidecar tracking state
    slam_pose: Optional["SlamPose"] = None           # latest SLAM pose if available
    slam_keyframe_id: Optional[int] = None           # latest SLAM keyframe id


@dataclass
class PolicyOutput:
    """Action output from the policy."""
    linear: float = 0.0                              # forward/back [-1, 1]
    angular: float = 0.0                             # left/right [-1, 1]
    confidence: float = 0.0                          # policy confidence [0, 1]
    waypoints: Optional[np.ndarray] = None           # predicted waypoints if available


class BasePolicy(abc.ABC):
    """Abstract navigation policy interface."""

    @abc.abstractmethod
    def setup(self):
        """Load model weights, warm up inference, etc."""
        ...

    @abc.abstractmethod
    def predict(self, obs: PolicyInput) -> PolicyOutput:
        """
        Given current observation + goal, return velocity commands.

        This is called at every control loop tick (~10 Hz).
        Must complete within ~50ms for real-time operation.
        """
        ...

    def reset(self):
        """Reset any internal state (called between checkpoints)."""
        pass
