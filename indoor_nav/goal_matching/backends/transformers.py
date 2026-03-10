from __future__ import annotations

import logging

import cv2
import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.backends.base import VectorEmbeddingBackend, normalize_vector

logger = logging.getLogger(__name__)


class Dinov2DirectBackend(VectorEmbeddingBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._device = None
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._device = torch.device(self.cfg.feature_device if torch.cuda.is_available() else "cpu")
        model_name = self.cfg.feature_model or "facebook/dinov2-with-registers-base"
        logger.info("Loading DINOv2-direct (mean-pool): %s on %s", model_name, self._device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        logger.info("DINOv2-direct ready (hidden=%d)", self._model.config.hidden_size)

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
        feat = patch_tokens.mean(dim=0)
        feat = torch.nn.functional.normalize(feat, dim=0)
        return feat.cpu().numpy().astype(np.float32)


class Siglip2Backend(VectorEmbeddingBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._device = None
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = torch.device(self.cfg.feature_device if torch.cuda.is_available() else "cpu")
        model_name = self.cfg.feature_model or "google/siglip2-base-patch16-224"
        logger.info("Loading SigLIP2: %s on %s", model_name, self._device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        logger.info("SigLIP2 loaded.")

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image
        import torch

        self._ensure_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        return normalize_vector(outputs.cpu().numpy())


class Dinov2ClsBackend(VectorEmbeddingBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._device = None
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._device = torch.device(self.cfg.feature_device if torch.cuda.is_available() else "cpu")
        model_name = self.cfg.feature_model or "facebook/dinov2-base"
        logger.info("Loading DINOv2 model: %s on %s", model_name, self._device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image
        import torch

        self._ensure_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return normalize_vector(feat)


class ClipBackend(VectorEmbeddingBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._device = None
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._device = torch.device(self.cfg.feature_device if torch.cuda.is_available() else "cpu")
        model_name = self.cfg.feature_model or "openai/clip-vit-base-patch32"
        logger.info("Loading CLIP model: %s on %s", model_name, self._device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image
        import torch

        self._ensure_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feat = self._model.get_image_features(**inputs)
        return normalize_vector(feat.cpu().numpy())
