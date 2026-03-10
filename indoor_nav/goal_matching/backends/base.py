from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.schemas import PreparedImage


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False).flatten()
    return vector / (np.linalg.norm(vector) + 1e-8)


class MatchBackend(ABC):
    """Interface shared by all goal-matching backends."""

    def __init__(self, cfg: GoalConfig):
        self.cfg = cfg

    def prepare_goal(self, image: np.ndarray) -> PreparedImage:
        return self.prepare_query(image)

    @abstractmethod
    def prepare_query(self, image: np.ndarray) -> PreparedImage:
        raise NotImplementedError

    @abstractmethod
    def score(self, query: PreparedImage, goal: PreparedImage) -> float:
        raise NotImplementedError


class VectorEmbeddingBackend(MatchBackend):
    """Shared cosine-similarity scoring for learned global descriptors."""

    def prepare_query(self, image: np.ndarray) -> PreparedImage:
        return PreparedImage(payload=self.extract_embedding(image))

    def score(self, query: PreparedImage, goal: PreparedImage) -> float:
        cosine = float(np.dot(query.payload, goal.payload))
        return (cosine + 1.0) / 2.0

    @abstractmethod
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
