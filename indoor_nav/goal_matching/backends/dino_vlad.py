from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.backends.base import VectorEmbeddingBackend

logger = logging.getLogger(__name__)


class DinoVladBackend(VectorEmbeddingBackend):
    """AnyLoc-style DINO patch-token VLAD backend."""

    def __init__(self, cfg: GoalConfig, *, default_model: str, label: str):
        super().__init__(cfg)
        self._default_model = default_model
        self._label = label
        self._device = None
        self._model = None
        self._processor = None
        self._vlad_centers = None
        self._vlad_initialized = False
        self._patch_cache: List[np.ndarray] = []

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._device = torch.device(self.cfg.feature_device if torch.cuda.is_available() else "cpu")
        model_name = self.cfg.feature_model or self._default_model
        logger.info("Loading %s (AnyLoc-style): %s on %s", self._label, model_name, self._device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        hidden_dim = self._model.config.hidden_size
        n_clusters = getattr(self.cfg, "vlad_clusters", 32)
        self._vlad_centers = torch.randn(n_clusters, hidden_dim, device=self._device)
        self._vlad_centers = torch.nn.functional.normalize(self._vlad_centers, dim=1)
        logger.info("%s ready (hidden=%d, clusters=%d)", self._label, hidden_dim, n_clusters)

    def _extract_vlad(self, patch_tokens: "torch.Tensor") -> np.ndarray:
        import torch

        sims = torch.mm(patch_tokens, self._vlad_centers.T)
        assignments = torch.argmax(sims, dim=1)

        n_clusters = self._vlad_centers.shape[0]
        dim = patch_tokens.shape[1]
        vlad = torch.zeros(n_clusters, dim, device=self._device)

        for cluster_idx in range(n_clusters):
            mask = assignments == cluster_idx
            if mask.any():
                residuals = patch_tokens[mask] - self._vlad_centers[cluster_idx].unsqueeze(0)
                vlad[cluster_idx] = residuals.sum(dim=0)

        vlad = torch.nn.functional.normalize(vlad, dim=1)
        flat = vlad.flatten()
        flat = torch.nn.functional.normalize(flat, dim=0)
        return flat.cpu().numpy().astype(np.float32)

    def _initialize_vlad_centers(self) -> None:
        import torch

        all_patches = np.concatenate(self._patch_cache, axis=0)
        if all_patches.shape[0] > 5000:
            indices = np.random.choice(all_patches.shape[0], 5000, replace=False)
            all_patches = all_patches[indices]

        patches_t = torch.from_numpy(all_patches).to(self._device)
        n_clusters = self._vlad_centers.shape[0]

        centers = patches_t[np.random.choice(patches_t.shape[0], n_clusters, replace=False)]
        for _ in range(10):
            dists = torch.cdist(patches_t, centers)
            assignments = torch.argmin(dists, dim=1)
            for cluster_idx in range(n_clusters):
                mask = assignments == cluster_idx
                if mask.any():
                    centers[cluster_idx] = patches_t[mask].mean(dim=0)
            centers = torch.nn.functional.normalize(centers, dim=1)

        self._vlad_centers = centers
        self._vlad_initialized = True
        self._patch_cache.clear()
        logger.info("VLAD centers initialized via k-means on %d patches.", all_patches.shape[0])

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        self._ensure_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        patch_tokens = outputs.last_hidden_state[0, 1:, :]
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=1)

        if not self._vlad_initialized and len(self._patch_cache) < 20:
            self._patch_cache.append(patch_tokens.cpu().numpy())
            if len(self._patch_cache) >= 10:
                self._initialize_vlad_centers()

        return self._extract_vlad(patch_tokens)
