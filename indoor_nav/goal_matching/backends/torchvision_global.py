from __future__ import annotations

import logging

import cv2
import numpy as np

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.backends.base import VectorEmbeddingBackend, normalize_vector

logger = logging.getLogger(__name__)


class TorchvisionGlobalDescriptorBackend(VectorEmbeddingBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._device = None
        self._model = None
        self._transform = None

    def _ensure_device(self) -> None:
        if self._device is not None:
            return
        import torch

        self._device = torch.device(self.cfg.feature_device if torch.cuda.is_available() else "cpu")

    def _prepare_tensor(self, image: np.ndarray) -> "torch.Tensor":
        import torchvision.transforms as T
        from PIL import Image

        self._ensure_device()
        if self._transform is None:
            resize_hw = (self.cfg.feature_image_size[1], self.cfg.feature_image_size[0])
            self._transform = T.Compose([
                T.Resize(resize_hw),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return self._transform(pil_img).unsqueeze(0).to(self._device)

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        import torch

        self._ensure_model()
        tensor = self._prepare_tensor(image)
        with torch.no_grad():
            feat = self._model(tensor)
        return normalize_vector(feat.cpu().numpy())

    def _ensure_model(self) -> None:
        raise NotImplementedError


class EigenPlacesBackend(TorchvisionGlobalDescriptorBackend):
    def __init__(self, cfg: GoalConfig):
        super().__init__(cfg)
        self._fallback_backend = None

    def _ensure_model(self) -> None:
        if self._model is not None or self._fallback_backend is not None:
            return

        import torch

        self._ensure_device()
        try:
            model = torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )
            self._model = model.to(self._device).eval()
            logger.info("EigenPlaces loaded (ResNet50, dim=2048)")
        except Exception as exc:
            logger.warning("EigenPlaces not available (%s). Falling back to DINOv2.", exc)
            from indoor_nav.goal_matching.backends.transformers import Dinov2ClsBackend

            self._fallback_backend = Dinov2ClsBackend(self.cfg)

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        self._ensure_model()
        if self._fallback_backend is not None:
            return self._fallback_backend.extract_embedding(image)
        return super().extract_embedding(image)


class CosPlaceBackend(TorchvisionGlobalDescriptorBackend):
    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch

        self._ensure_device()
        backbone = getattr(self.cfg, "cosplace_backbone", "ResNet50")
        fc_output_dim = getattr(self.cfg, "cosplace_fc_output_dim", 2048)
        logger.info(
            "Loading CosPlace: repo=gmberton/cosplace backbone=%s dim=%d on %s",
            backbone,
            fc_output_dim,
            self._device,
        )
        self._model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=fc_output_dim,
        ).to(self._device).eval()
        logger.info("CosPlace loaded (%s, dim=%d)", backbone, fc_output_dim)
