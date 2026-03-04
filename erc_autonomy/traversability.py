from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import ERCConfig


@dataclass
class TraversabilityResult:
    """Per-frame traversability prediction summary."""

    mask: np.ndarray  # HxW float32 in [0, 1], 1=traversable
    confidence: float
    risk: float  # 0=safe, 1=unsafe
    left_clearance: float
    center_clearance: float
    right_clearance: float


class TraversabilityEngine:
    """
    Traversability frontend with a stable fallback backend.

    Backends:
    - simple_edge: fast heuristic using edges + texture density.
    - sam2: placeholder that currently falls back to simple_edge if SAM2 is
      not wired in this environment.
    """

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self.backend = cfg.traversability_backend.lower()

    def infer(self, frame_bgr: Optional[np.ndarray]) -> Optional[TraversabilityResult]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if self.backend == "sam2":
            # Placeholder path for future SAM2 integration.
            return self._infer_simple_edge(frame_bgr)
        return self._infer_simple_edge(frame_bgr)

    def _infer_simple_edge(self, frame_bgr: np.ndarray) -> TraversabilityResult:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # More edges in near-field generally imply clutter/obstacles.
        edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0

        # Use lower image region for drivability estimate.
        y0 = int(h * 0.45)
        near = edges[y0:, :]
        near_occ = float(np.mean(near))

        # Convert occupancy proxy into traversability.
        near_trav = np.clip(1.0 - 2.5 * near_occ, 0.0, 1.0)

        # Build dense mask with near-field emphasis and light smoothing.
        mask = np.ones((h, w), dtype=np.float32) * near_trav
        mask[:y0, :] = min(1.0, near_trav + 0.1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        third = w // 3
        left = mask[y0:, :third]
        center = mask[y0:, third : 2 * third]
        right = mask[y0:, 2 * third :]
        left_clear = float(np.mean(left)) if left.size else 0.0
        center_clear = float(np.mean(center)) if center.size else 0.0
        right_clear = float(np.mean(right)) if right.size else 0.0

        confidence = float(np.clip(1.0 - near_occ * 3.0, 0.15, 0.95))
        risk = float(np.clip(1.0 - center_clear, 0.0, 1.0))
        return TraversabilityResult(
            mask=mask,
            confidence=confidence,
            risk=risk,
            left_clearance=left_clear,
            center_clearance=center_clear,
            right_clearance=right_clear,
        )

