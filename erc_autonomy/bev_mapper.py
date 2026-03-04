from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import ERCConfig


@dataclass
class BEVResult:
    """Bird's-eye-view traversability/cost projection."""

    traversability: np.ndarray  # [depth_cells, width_cells], 1=traversable
    cost: np.ndarray  # [depth_cells, width_cells], 0=free, 1=blocked
    left_score: float
    center_score: float
    right_score: float


class BEVMapper:
    """
    Lightweight image-space to BEV projection.

    This is a geometric approximation for Week 3 integration. It keeps module
    interfaces stable so a calibrated perspective mapping can replace it later.
    """

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self.width_cells = max(8, int(cfg.bev_width_m / cfg.bev_resolution_m))
        self.depth_cells = max(8, int(cfg.bev_depth_m / cfg.bev_resolution_m))

    def project(self, traversability_mask: Optional[np.ndarray]) -> Optional[BEVResult]:
        if traversability_mask is None or traversability_mask.size == 0:
            return None

        h, w = traversability_mask.shape[:2]
        roi = traversability_mask[int(h * 0.35) :, :]
        if roi.size == 0:
            return None

        # Resize to BEV raster as a monotonic far-to-near proxy.
        bev_t = cv2.resize(
            roi,
            (self.width_cells, self.depth_cells),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        # Near field should weigh more in decision making.
        row_weights = np.linspace(0.6, 1.0, self.depth_cells, dtype=np.float32).reshape(
            -1, 1
        )
        bev_t = np.clip(bev_t * row_weights, 0.0, 1.0)
        cost = 1.0 - bev_t

        third = self.width_cells // 3
        left = bev_t[:, :third]
        center = bev_t[:, third : 2 * third]
        right = bev_t[:, 2 * third :]
        left_score = float(np.mean(left)) if left.size else 0.0
        center_score = float(np.mean(center)) if center.size else 0.0
        right_score = float(np.mean(right)) if right.size else 0.0

        return BEVResult(
            traversability=bev_t,
            cost=cost,
            left_score=left_score,
            center_score=center_score,
            right_score=right_score,
        )

