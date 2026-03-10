from __future__ import annotations

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.backends.base import MatchBackend
from indoor_nav.goal_matching.backends.dino_vlad import DinoVladBackend
from indoor_nav.goal_matching.backends.sift import SiftBackend
from indoor_nav.goal_matching.backends.torchvision_global import (
    CosPlaceBackend,
    EigenPlacesBackend,
)
from indoor_nav.goal_matching.backends.transformers import (
    ClipBackend,
    Dinov2ClsBackend,
    Dinov2DirectBackend,
    Siglip2Backend,
)
from indoor_nav.goal_matching.backends.wall_crop import WallCropDirectBackend
from indoor_nav.goal_matching.backends.wall_rectify import WallRectifyDirectBackend


def build_backend(cfg: GoalConfig) -> MatchBackend:
    method = cfg.match_method

    if method == "dinov2_vlad":
        return DinoVladBackend(
            cfg,
            default_model="facebook/dinov2-with-registers-base",
            label="DINOv2-VLAD",
        )
    if method == "dinov3_vlad":
        return DinoVladBackend(
            cfg,
            default_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
            label="DINOv3-VLAD",
        )
    if method == "dinov2_direct":
        return Dinov2DirectBackend(cfg)
    if method == "wall_crop_direct":
        return WallCropDirectBackend(cfg)
    if method == "wall_rectify_direct":
        return WallRectifyDirectBackend(cfg)
    if method == "siglip2":
        return Siglip2Backend(cfg)
    if method == "dinov2":
        return Dinov2ClsBackend(cfg)
    if method == "clip":
        return ClipBackend(cfg)
    if method == "eigenplaces":
        return EigenPlacesBackend(cfg)
    if method == "cosplace":
        return CosPlaceBackend(cfg)
    if method in {"sift", "superglue"}:
        return SiftBackend(cfg)

    raise ValueError(f"Unknown match_method: {method}")
