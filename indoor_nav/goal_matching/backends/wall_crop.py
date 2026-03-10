from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import cv2
import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.backends.base import MatchBackend
from indoor_nav.goal_matching.backends.transformers import Dinov2DirectBackend
from indoor_nav.goal_matching.schemas import PreparedImage


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [int(x1), int(y1), int(x2), int(y2)]


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding_frac: float,
) -> list[int]:
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * padding_frac))
    pad_y = int(round((y2 - y1) * padding_frac))
    return _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _box_iou(box_a: Iterable[int], box_b: Iterable[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
    bx1, by1, bx2, by2 = [int(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter_area / max(area_a + area_b - inter_area, 1.0)


def _score_embeddings(query_embedding: np.ndarray, goal_embedding: np.ndarray) -> float:
    cosine = float(np.dot(query_embedding, goal_embedding))
    return (cosine + 1.0) / 2.0


class WallCropDirectBackend(MatchBackend):
    """
    Two-stage matcher for wall-mounted images.

    Stage 1 proposes rectangular wall-image crops from contours.
    Stage 2 scores those crops with the existing DINOv2-direct encoder.
    """

    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        direct_cfg = replace(cfg, match_method="dinov2_direct")
        self._direct = Dinov2DirectBackend(direct_cfg)

    def prepare_query(self, image: np.ndarray) -> PreparedImage:
        scene_embedding = self._direct.extract_embedding(image)
        candidate_boxes = self._detect_candidate_boxes(image)
        embedded_boxes = []
        candidate_embeddings = []
        for x1, y1, x2, y2 in candidate_boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            embedded_boxes.append([x1, y1, x2, y2])
            candidate_embeddings.append(self._direct.extract_embedding(crop))

        return PreparedImage(
            payload={
                "scene_embedding": scene_embedding,
                "candidate_embeddings": candidate_embeddings,
            },
            image=image,
            metadata={
                "candidate_boxes": embedded_boxes,
                "candidate_count": len(candidate_embeddings),
            },
        )

    def score(self, query: PreparedImage, goal: PreparedImage) -> float:
        scene_score = _score_embeddings(
            query.payload["scene_embedding"],
            goal.payload["scene_embedding"],
        )

        best_crop_score = 0.0
        query_embeddings = [query.payload["scene_embedding"], *query.payload["candidate_embeddings"]]
        goal_embeddings = [goal.payload["scene_embedding"], *goal.payload["candidate_embeddings"]]
        for q_idx, query_embedding in enumerate(query_embeddings):
            for g_idx, goal_embedding in enumerate(goal_embeddings):
                if q_idx == 0 and g_idx == 0:
                    continue
                best_crop_score = max(
                    best_crop_score,
                    _score_embeddings(query_embedding, goal_embedding),
                )

        blend_weight = float(np.clip(self.cfg.wall_crop_score_weight, 0.0, 1.0))
        blended_score = (blend_weight * best_crop_score) + ((1.0 - blend_weight) * scene_score)
        return max(scene_score, blended_score)

    def _detect_candidate_boxes(self, image: np.ndarray) -> list[list[int]]:
        max_candidates = max(0, int(self.cfg.wall_crop_max_candidates))
        if max_candidates == 0:
            return []

        height, width = image.shape[:2]
        image_area = float(height * width)
        if image_area <= 0:
            return []

        min_area = image_area * max(0.0, self.cfg.wall_crop_min_area_frac)
        max_area = image_area * max(self.cfg.wall_crop_min_area_frac, self.cfg.wall_crop_max_area_frac)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        proposals: list[tuple[float, list[int]]] = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4:
                continue

            rect = cv2.minAreaRect(contour)
            rect_w, rect_h = rect[1]
            if rect_w <= 1.0 or rect_h <= 1.0:
                continue

            rect_area = rect_w * rect_h
            if rect_area < min_area or rect_area > max_area:
                continue

            aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)
            if aspect_ratio > self.cfg.wall_crop_max_aspect_ratio:
                continue

            fill_ratio = contour_area / max(rect_area, 1.0)
            if fill_ratio < self.cfg.wall_crop_min_fill_ratio:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            box = _expand_box(
                (x, y, x + w, y + h),
                width=width,
                height=height,
                padding_frac=self.cfg.wall_crop_padding_frac,
            )
            box_area = float((box[2] - box[0]) * (box[3] - box[1]))
            if box_area < min_area or box_area > max_area:
                continue
            if any(_box_iou(box, existing_box) >= 0.7 for _, existing_box in proposals):
                continue
            proposals.append((box_area, box))

        proposals.sort(key=lambda item: item[0], reverse=True)
        return [box for _, box in proposals[:max_candidates]]
