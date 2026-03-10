from __future__ import annotations

from dataclasses import replace

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


def _box_iou(box_a: list[int], box_b: list[int]) -> float:
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


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def _quad_box(points: np.ndarray, width: int, height: int, padding_frac: float) -> list[int]:
    xs = points[:, 0]
    ys = points[:, 1]
    x1 = int(np.floor(xs.min()))
    y1 = int(np.floor(ys.min()))
    x2 = int(np.ceil(xs.max()))
    y2 = int(np.ceil(ys.max()))
    pad_x = int(round(max(1.0, x2 - x1) * padding_frac))
    pad_y = int(round(max(1.0, y2 - y1) * padding_frac))
    return _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _expand_quad(points: np.ndarray, width: int, height: int, padding_frac: float) -> np.ndarray:
    if padding_frac <= 0.0:
        return points.astype(np.float32)
    center = points.mean(axis=0, keepdims=True)
    expanded = center + ((points - center) * (1.0 + padding_frac))
    expanded[:, 0] = np.clip(expanded[:, 0], 0.0, width - 1.0)
    expanded[:, 1] = np.clip(expanded[:, 1], 0.0, height - 1.0)
    return expanded.astype(np.float32)


def _rectify_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    ordered = _order_quad(quad)
    tl, tr, br, bl = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = int(round(max(width_a, width_b)))
    max_height = int(round(max(height_a, height_b)))
    if max_width < 16 or max_height < 16:
        return None

    dest = np.array(
        [
            [0.0, 0.0],
            [max_width - 1.0, 0.0],
            [max_width - 1.0, max_height - 1.0],
            [0.0, max_height - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, dest)
    rectified = cv2.warpPerspective(image, transform, (max_width, max_height))
    if rectified.size == 0:
        return None
    return rectified


class WallRectifyDirectBackend(MatchBackend):
    """
    Detect rectangular wall-image candidates, rectify them, then score with DINOv2-direct.

    This is a stronger variant of wall_crop_direct for angled posters, paintings,
    or screens because each candidate is perspective-warped before comparison.
    """

    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        direct_cfg = replace(cfg, match_method="dinov2_direct")
        self._direct = Dinov2DirectBackend(direct_cfg)

    def prepare_query(self, image: np.ndarray) -> PreparedImage:
        scene_embedding = self._direct.extract_embedding(image)
        candidates = self._detect_candidates(image)

        candidate_boxes: list[list[int]] = []
        candidate_quads: list[list[list[float]]] = []
        candidate_embeddings: list[np.ndarray] = []

        for candidate in candidates:
            rectified = _rectify_quad(image, candidate["quad"])
            if rectified is None:
                continue
            candidate_boxes.append(candidate["box"])
            candidate_quads.append(candidate["quad"].astype(float).tolist())
            candidate_embeddings.append(self._direct.extract_embedding(rectified))

        return PreparedImage(
            payload={
                "scene_embedding": scene_embedding,
                "candidate_embeddings": candidate_embeddings,
            },
            image=image,
            metadata={
                "candidate_boxes": candidate_boxes,
                "candidate_quads": candidate_quads,
                "candidate_count": len(candidate_embeddings),
            },
        )

    def score(self, query: PreparedImage, goal: PreparedImage) -> float:
        scene_score = _score_embeddings(
            query.payload["scene_embedding"],
            goal.payload["scene_embedding"],
        )

        best_rectified_score = 0.0
        query_embeddings = [query.payload["scene_embedding"], *query.payload["candidate_embeddings"]]
        goal_embeddings = [goal.payload["scene_embedding"], *goal.payload["candidate_embeddings"]]
        for q_idx, query_embedding in enumerate(query_embeddings):
            for g_idx, goal_embedding in enumerate(goal_embeddings):
                if q_idx == 0 and g_idx == 0:
                    continue
                best_rectified_score = max(
                    best_rectified_score,
                    _score_embeddings(query_embedding, goal_embedding),
                )

        blend_weight = float(np.clip(self.cfg.wall_crop_score_weight, 0.0, 1.0))
        blended_score = (blend_weight * best_rectified_score) + ((1.0 - blend_weight) * scene_score)
        return max(scene_score, blended_score)

    def _detect_candidates(self, image: np.ndarray) -> list[dict[str, np.ndarray | list[int]]]:
        max_candidates = max(0, int(self.cfg.wall_crop_max_candidates))
        if max_candidates == 0:
            return []

        height, width = image.shape[:2]
        image_area = float(height * width)
        if image_area <= 0.0:
            return []

        min_area = image_area * max(0.0, self.cfg.wall_crop_min_area_frac)
        max_area = image_area * max(self.cfg.wall_crop_min_area_frac, self.cfg.wall_crop_max_area_frac)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        proposals: list[tuple[float, dict[str, np.ndarray | list[int]]]] = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            quad: np.ndarray | None = None
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)

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

            if quad is None:
                quad = cv2.boxPoints(rect).astype(np.float32)

            quad = _expand_quad(quad, width, height, self.cfg.wall_crop_padding_frac)
            box = _quad_box(quad, width, height, 0.0)
            box_area = float((box[2] - box[0]) * (box[3] - box[1]))
            if box_area < min_area or box_area > max_area:
                continue
            if any(_box_iou(box, existing["box"]) >= 0.7 for _, existing in proposals):
                continue

            proposals.append(
                (
                    box_area,
                    {
                        "box": box,
                        "quad": quad,
                    },
                )
            )

        proposals.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in proposals[:max_candidates]]
