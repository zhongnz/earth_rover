from __future__ import annotations

import cv2
import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.backends.base import MatchBackend
from indoor_nav.goal_matching.schemas import PreparedImage


class SiftBackend(MatchBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._sift = cv2.SIFT_create(nfeatures=2000)
        self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def prepare_query(self, image: np.ndarray) -> PreparedImage:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        return PreparedImage(
            payload={"keypoints": keypoints, "descriptors": descriptors},
            image=image,
        )

    def score(self, query: PreparedImage, goal: PreparedImage) -> float:
        keypoints_q = query.payload["keypoints"]
        keypoints_g = goal.payload["keypoints"]
        descriptors_q = query.payload["descriptors"]
        descriptors_g = goal.payload["descriptors"]

        if (
            descriptors_q is None
            or descriptors_g is None
            or len(keypoints_q) < 4
            or len(keypoints_g) < 4
        ):
            return 0.0

        raw = self._bf.knnMatch(descriptors_q, descriptors_g, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return 0.0

        src = np.float32([keypoints_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints_g[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return min(1.0, inliers / 50.0)
