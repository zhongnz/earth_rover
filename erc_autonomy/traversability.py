from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

from .config import ERCConfig

logger = logging.getLogger(__name__)

_SAM2_CFG_MAP = {
    "sam2.1_hiera_t.yaml": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s.yaml": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+.yaml": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l.yaml": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_t.yaml": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_s.yaml": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_b+.yaml": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_l.yaml": "configs/sam2/sam2_hiera_l.yaml",
}


@dataclass
class TraversabilityResult:
    """Per-frame traversability prediction summary."""

    mask: np.ndarray  # HxW float32 in [0, 1], 1=traversable
    confidence: float
    risk: float  # 0=safe, 1=unsafe
    left_clearance: float
    center_clearance: float
    right_clearance: float


@dataclass
class _SAM2Runtime:
    mask_generator: Any
    max_side: int
    device: str


class TraversabilityEngine:
    """
    Traversability frontend with a stable fallback backend.

    Backends:
    - simple_edge: fast heuristic using edges + texture density.
    - sam2: automatic-mask backend using Segment Anything 2; gracefully
      falls back to simple_edge if SAM2 runtime is unavailable.
    """

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self.backend = cfg.traversability_backend.lower()
        self._sam2_runtime: Optional[_SAM2Runtime] = None
        self._sam2_disabled_reason: Optional[str] = None
        self._sam2_warned_fallback = False

    def infer(self, frame_bgr: Optional[np.ndarray]) -> Optional[TraversabilityResult]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if self.backend == "sam2":
            sam2_result = self._infer_sam2(frame_bgr)
            if sam2_result is not None:
                return sam2_result
        return self._infer_simple_edge(frame_bgr)

    def _warn_sam2_fallback_once(self, reason: str) -> None:
        if self._sam2_warned_fallback:
            return
        self._sam2_warned_fallback = True
        logger.warning(
            "SAM2 backend unavailable (%s); falling back to simple_edge",
            reason,
        )

    def _resolve_sam2_device(self) -> str:
        requested = (self.cfg.sam2_device or "auto").strip().lower()
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _build_sam2_mask_generator(self) -> Any:
        try:
            from sam2.build_sam import build_sam2  # type: ignore
        except Exception:
            try:
                from sam2.build_sam2 import build_sam2  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"could not import SAM2 builder ({exc})"
                ) from exc

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        except Exception:
            try:
                from sam2.sam2_automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"could not import SAM2 automatic mask generator ({exc})"
                ) from exc

        device = self._resolve_sam2_device()
        model_cfg = self._normalize_sam2_model_cfg(self.cfg.sam2_model_cfg)
        model = build_sam2(
            model_cfg,
            self.cfg.sam2_checkpoint,
            device=device,
        )

        common_kwargs = {
            "points_per_side": max(4, int(self.cfg.sam2_points_per_side)),
            "pred_iou_thresh": float(np.clip(self.cfg.sam2_pred_iou_thresh, 0.0, 1.0)),
            "stability_score_thresh": float(
                np.clip(self.cfg.sam2_stability_score_thresh, 0.0, 1.0)
            ),
            "min_mask_region_area": max(0, int(self.cfg.sam2_min_mask_region_area)),
        }

        # Different SAM2 builds expose slightly different constructor signatures.
        for kwargs in (common_kwargs, {"points_per_side": common_kwargs["points_per_side"]}, {}):
            try:
                return SAM2AutomaticMaskGenerator(model=model, **kwargs)
            except TypeError:
                continue

        raise RuntimeError("failed to construct SAM2AutomaticMaskGenerator")

    @staticmethod
    def _normalize_sam2_model_cfg(model_cfg: str) -> str:
        cfg = (model_cfg or "").strip()
        if not cfg:
            return cfg
        if cfg in _SAM2_CFG_MAP.values():
            return cfg
        basename = os.path.basename(cfg)
        return _SAM2_CFG_MAP.get(basename, cfg)

    def _ensure_sam2_runtime(self) -> Optional[_SAM2Runtime]:
        if self._sam2_runtime is not None:
            return self._sam2_runtime
        if self._sam2_disabled_reason is not None:
            return None

        if not self.cfg.sam2_model_cfg or not self.cfg.sam2_checkpoint:
            self._sam2_disabled_reason = "missing sam2_model_cfg/sam2_checkpoint"
            self._warn_sam2_fallback_once(self._sam2_disabled_reason)
            return None

        try:
            mask_generator = self._build_sam2_mask_generator()
        except Exception as exc:
            self._sam2_disabled_reason = str(exc)
            self._warn_sam2_fallback_once(self._sam2_disabled_reason)
            return None

        self._sam2_runtime = _SAM2Runtime(
            mask_generator=mask_generator,
            max_side=max(256, int(self.cfg.sam2_max_side)),
            device=self._resolve_sam2_device(),
        )
        logger.info(
            "SAM2 traversability backend enabled (device=%s, max_side=%d)",
            self._sam2_runtime.device,
            self._sam2_runtime.max_side,
        )
        return self._sam2_runtime

    @staticmethod
    def _decode_mask(raw_segmentation: Any, h: int, w: int) -> Optional[np.ndarray]:
        if isinstance(raw_segmentation, np.ndarray):
            if raw_segmentation.ndim == 3:
                raw_segmentation = raw_segmentation[..., 0]
            if raw_segmentation.ndim != 2:
                return None
            if raw_segmentation.shape != (h, w):
                return None
            return raw_segmentation.astype(bool)

        if isinstance(raw_segmentation, dict):
            try:
                from pycocotools import mask as mask_utils  # type: ignore

                decoded = mask_utils.decode(raw_segmentation)
                if decoded.ndim == 3:
                    decoded = decoded[..., 0]
                if decoded.shape != (h, w):
                    return None
                return decoded.astype(bool)
            except Exception:
                return None
        return None

    @staticmethod
    def _resize_for_inference(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
        h, w = frame_bgr.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            return frame_bgr, 1.0

        scale = max_side / float(long_side)
        nh = max(2, int(round(h * scale)))
        nw = max(2, int(round(w * scale)))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _build_traversable_mask_from_sam2(self, masks: list[dict], h: int, w: int) -> tuple[Optional[np.ndarray], float]:
        if not masks:
            return None, 0.0

        y0 = int(h * 0.62)
        x0 = int(w * 0.28)
        x1 = int(w * 0.72)
        roi = np.zeros((h, w), dtype=bool)
        roi[y0:, x0:x1] = True
        roi_area = max(1, int(np.count_nonzero(roi)))
        image_area = float(h * w)

        best_score = -1.0
        best_mask: Optional[np.ndarray] = None

        for item in masks:
            seg = self._decode_mask(item.get("segmentation"), h, w)
            if seg is None:
                continue
            area = float(item.get("area", np.count_nonzero(seg)))
            if area < (0.01 * image_area):
                continue

            roi_overlap = float(np.count_nonzero(seg & roi)) / roi_area
            near_fraction = float(np.count_nonzero(seg[y0:, :])) / max(1.0, area)
            score = (0.65 * roi_overlap) + (0.35 * near_fraction)
            if score > best_score:
                best_score = score
                best_mask = seg

        if best_mask is None or best_score < 0.08:
            return None, 0.0

        traversable = best_mask.astype(np.float32)
        for item in masks:
            seg = self._decode_mask(item.get("segmentation"), h, w)
            if seg is None:
                continue
            inter = float(np.count_nonzero(seg & best_mask))
            union = float(np.count_nonzero(seg | best_mask))
            iou = inter / max(1.0, union)
            roi_overlap = float(np.count_nonzero(seg & roi)) / roi_area
            if iou >= 0.28 or roi_overlap >= 0.40:
                traversable = np.maximum(traversable, seg.astype(np.float32))

        kernel = np.ones((5, 5), dtype=np.uint8)
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_CLOSE, kernel)
        traversable = cv2.GaussianBlur(traversable, (11, 11), 0)
        traversable = np.clip(traversable, 0.0, 1.0).astype(np.float32)
        return traversable, float(np.clip(best_score, 0.0, 1.0))

    def _summarize_mask(self, mask: np.ndarray, confidence: float) -> TraversabilityResult:
        h, w = mask.shape[:2]
        y0 = int(h * 0.45)
        third = max(1, w // 3)
        left = mask[y0:, :third]
        center = mask[y0:, third : 2 * third]
        right = mask[y0:, 2 * third :]
        left_clear = float(np.mean(left)) if left.size else 0.0
        center_clear = float(np.mean(center)) if center.size else 0.0
        right_clear = float(np.mean(right)) if right.size else 0.0
        risk = float(np.clip(1.0 - center_clear, 0.0, 1.0))
        return TraversabilityResult(
            mask=mask,
            confidence=float(np.clip(confidence, 0.15, 0.98)),
            risk=risk,
            left_clearance=left_clear,
            center_clearance=center_clear,
            right_clearance=right_clear,
        )

    def _infer_sam2(self, frame_bgr: np.ndarray) -> Optional[TraversabilityResult]:
        runtime = self._ensure_sam2_runtime()
        if runtime is None:
            return None

        frame_small, scale = self._resize_for_inference(frame_bgr, runtime.max_side)
        h_small, w_small = frame_small.shape[:2]
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        try:
            masks = runtime.mask_generator.generate(rgb)
        except Exception as exc:
            self._sam2_runtime = None
            self._sam2_disabled_reason = f"SAM2 inference failed: {exc}"
            self._warn_sam2_fallback_once(self._sam2_disabled_reason)
            return None

        trav_small, score = self._build_traversable_mask_from_sam2(masks, h_small, w_small)
        if trav_small is None:
            return None

        if scale < 1.0:
            h, w = frame_bgr.shape[:2]
            traversable = cv2.resize(trav_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            traversable = trav_small
        traversable = np.clip(traversable, 0.0, 1.0).astype(np.float32)

        h, w = traversable.shape[:2]
        y0 = int(h * 0.62)
        x0 = int(w * 0.28)
        x1 = int(w * 0.72)
        center_support = float(np.mean(traversable[y0:, x0:x1])) if y0 < h else float(np.mean(traversable))
        confidence = float(np.clip((0.25 + (0.55 * score) + (0.20 * center_support)), 0.15, 0.98))
        return self._summarize_mask(traversable, confidence)

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
        confidence = float(np.clip(1.0 - near_occ * 3.0, 0.15, 0.95))
        return self._summarize_mask(mask, confidence)
