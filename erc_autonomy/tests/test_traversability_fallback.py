from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from erc_autonomy.config import ERCConfig
from erc_autonomy.traversability import TraversabilityEngine


class _ExplodingMaskGenerator:
    def generate(self, _image):
        raise RuntimeError("forced generate failure")


class TraversabilityFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    def test_sam2_missing_paths_falls_back_to_simple_edge(self) -> None:
        cfg = ERCConfig()
        cfg.traversability_backend = "sam2"
        cfg.sam2_model_cfg = ""
        cfg.sam2_checkpoint = ""

        engine = TraversabilityEngine(cfg)
        result = engine.infer(self.frame)

        self.assertIsNotNone(result)
        self.assertEqual(result.mask.shape[:2], self.frame.shape[:2])
        self.assertEqual(engine._sam2_disabled_reason, "missing sam2_model_cfg/sam2_checkpoint")

    def test_sam2_init_failure_falls_back_to_simple_edge(self) -> None:
        cfg = ERCConfig()
        cfg.traversability_backend = "sam2"
        cfg.sam2_model_cfg = "/tmp/fake_cfg.yaml"
        cfg.sam2_checkpoint = "/tmp/fake_ckpt.pt"

        engine = TraversabilityEngine(cfg)
        with patch.object(
            TraversabilityEngine,
            "_build_sam2_mask_generator",
            side_effect=RuntimeError("forced init failure"),
        ):
            result = engine.infer(self.frame)

        self.assertIsNotNone(result)
        self.assertIn("forced init failure", engine._sam2_disabled_reason or "")

    def test_sam2_inference_failure_falls_back_and_disables_runtime(self) -> None:
        cfg = ERCConfig()
        cfg.traversability_backend = "sam2"
        cfg.sam2_model_cfg = "/tmp/fake_cfg.yaml"
        cfg.sam2_checkpoint = "/tmp/fake_ckpt.pt"

        engine = TraversabilityEngine(cfg)
        engine._sam2_runtime = SimpleNamespace(
            mask_generator=_ExplodingMaskGenerator(),
            max_side=1024,
            device="cpu",
        )

        result = engine.infer(self.frame)

        self.assertIsNotNone(result)
        self.assertIsNone(engine._sam2_runtime)
        self.assertIn("forced generate failure", engine._sam2_disabled_reason or "")


if __name__ == "__main__":
    unittest.main()
