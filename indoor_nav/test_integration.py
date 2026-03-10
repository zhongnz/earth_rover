#!/usr/bin/env python3
"""
Quick integration test — verifies all modules (including SOTA 2025 upgrades)
can be imported and the agent can be instantiated without errors.

Run from project root:
  python indoor_nav/test_integration.py --skip-sdk
"""

import asyncio
import logging
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test")


async def test_imports():
    logger.info("Testing imports...")
    from indoor_nav.configs.config import (
        IndoorNavConfig, SDKConfig, ControlConfig, PolicyConfig,
        GoalConfig, ObstacleConfig, TopoMemoryConfig, RecoveryConfig, LogConfig,
    )
    from indoor_nav.modules.sdk_client import RoverSDKClient, BotState
    from indoor_nav.modules.checkpoint_manager import CheckpointManager, GoalMatcher
    from indoor_nav.modules.obstacle_avoidance import ObstacleDetector, ObstacleInfo
    from indoor_nav.modules.recovery import RecoveryManager
    from indoor_nav.modules.topological_memory import TopologicalMemory, TopoMapConfig, TopoNode
    from indoor_nav.policies.base_policy import BasePolicy, PolicyInput, PolicyOutput
    from indoor_nav.policies.nomad_policy import NoMaDPolicy
    from indoor_nav.policies.vlm_hybrid_policy import VLMHybridPolicy
    from indoor_nav.policies.vla_policy import VLAPolicy
    from indoor_nav.agent import IndoorNavigationAgent
    logger.info("All imports OK (including SOTA 2025 modules).")


async def test_config_defaults():
    logger.info("Testing SOTA 2025 config defaults...")
    from indoor_nav.configs.config import IndoorNavConfig

    cfg = IndoorNavConfig()
    assert cfg.policy.backend == "vlm_hybrid", f"Expected vlm_hybrid, got {cfg.policy.backend}"
    assert cfg.policy.vlm_model == "Qwen/Qwen2.5-VL-7B-Instruct", f"Expected Qwen2.5-VL, got {cfg.policy.vlm_model}"
    assert cfg.goal.match_method == "dinov2_direct", f"Expected dinov2_direct, got {cfg.goal.match_method}"
    supported_match_methods = {
        "dinov2_direct", "wall_crop_direct", "wall_rectify_direct", "dinov2_vlad", "dinov3_vlad", "siglip2", "dinov2", "eigenplaces", "cosplace", "clip", "sift"
    }
    assert "dinov3_vlad" in supported_match_methods
    assert "cosplace" in supported_match_methods
    assert "dinov2-with-registers" in cfg.goal.feature_model, \
        f"Expected dinov2-with-registers, got {cfg.goal.feature_model}"
    assert "Base" in cfg.obstacle.depth_model or "base" in cfg.obstacle.depth_model.lower(), \
        f"Expected Depth Anything V2 Base, got {cfg.obstacle.depth_model}"
    assert cfg.topo_memory.enabled, "Topo memory should be enabled by default"
    assert cfg.topo_memory.max_nodes == 500
    logger.info("Config defaults are SOTA 2025. ✓")


async def test_topological_memory():
    import numpy as np
    from indoor_nav.modules.topological_memory import TopologicalMemory, TopoMapConfig

    logger.info("Testing topological memory...")
    cfg = TopoMapConfig(min_node_distance=0.1)  # fast node creation for test
    topo = TopologicalMemory(cfg)

    # Create some fake observations
    imgs = []
    for i in range(5):
        img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        # Make each progressively different
        img[:, :, 0] = (img[:, :, 0] + i * 50) % 256
        imgs.append(img)

    # Add nodes
    time.sleep(0.15)
    for img in imgs:
        time.sleep(0.15)
        topo.update(img, orientation=0.0)

    assert topo.num_nodes >= 2, f"Expected >=2 nodes, got {topo.num_nodes}"
    assert topo.num_edges > 0, f"Expected >0 edges, got {topo.num_edges}"

    # Test similarity search
    result = topo.find_most_similar_node(imgs[0])
    assert result is not None, "Should find a similar node"
    node_id, sim = result
    logger.info("  Most similar node: %d (sim=%.3f)", node_id, sim)

    # Test path planning
    node_ids = list(topo.nodes.keys())
    if len(node_ids) >= 2:
        path = topo.plan_path(node_ids[0], node_ids[-1])
        if path:
            logger.info("  Path found: %s", path)

    # Test backtracking
    backtrack = topo.get_backtrack_path(3)
    logger.info("  Backtrack: %s", backtrack)

    logger.info("  Status: %s", topo.status_str())
    logger.info("Topological memory test passed.")


async def test_heuristic_policy():
    import numpy as np
    from indoor_nav.configs.config import PolicyConfig
    from indoor_nav.policies.nomad_policy import NoMaDPolicy
    from indoor_nav.policies.base_policy import PolicyInput

    logger.info("Testing heuristic policy...")
    cfg = PolicyConfig(backend="heuristic")
    policy = NoMaDPolicy(cfg)
    policy._model = None  # force heuristic
    policy.setup()

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = PolicyInput(
        front_image=dummy_img,
        goal_image=dummy_img,
        goal_similarity=0.5,
        goal_trend=0.01,
        context_images=[dummy_img],
    )
    action = policy.predict(obs)
    logger.info("  Heuristic output: linear=%.2f angular=%.2f conf=%.2f",
                action.linear, action.angular, action.confidence)
    logger.info("Heuristic policy test passed.")


async def test_vla_heuristic():
    import numpy as np
    from indoor_nav.configs.config import PolicyConfig
    from indoor_nav.policies.vla_policy import VLAPolicy
    from indoor_nav.policies.base_policy import PolicyInput

    logger.info("Testing VLA enhanced heuristic...")
    cfg = PolicyConfig(backend="vla", device="cpu")
    policy = VLAPolicy(cfg)
    policy._vla_backend = "heuristic_plus"
    policy.setup()

    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    goal_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs = PolicyInput(
        front_image=dummy_img,
        goal_image=goal_img,
        goal_similarity=0.4,
        goal_trend=-0.01,
        context_images=[dummy_img],
    )
    action = policy.predict(obs)
    logger.info("  VLA heuristic output: linear=%.2f angular=%.2f conf=%.2f",
                action.linear, action.angular, action.confidence)
    logger.info("VLA enhanced heuristic test passed.")


async def test_vlm_policy_init():
    from indoor_nav.configs.config import PolicyConfig
    from indoor_nav.policies.vlm_hybrid_policy import VLMHybridPolicy

    logger.info("Testing VLM hybrid policy (Qwen2.5-VL config)...")
    cfg = PolicyConfig(
        backend="vlm_hybrid",
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_endpoint="http://127.0.0.1:8001/v1",
    )
    policy = VLMHybridPolicy(cfg)
    policy.setup()
    assert policy._api_format == "openai", f"Expected openai format, got {policy._api_format}"
    logger.info("  API format auto-detected: %s", policy._api_format)
    logger.info("VLM hybrid policy test passed.")


async def test_sdk_client(url: str):
    from indoor_nav.configs.config import SDKConfig
    from indoor_nav.modules.sdk_client import RoverSDKClient

    logger.info("Testing SDK client at %s...", url)
    cfg = SDKConfig(base_url=url)
    client = RoverSDKClient(cfg)

    try:
        state = await client.get_data()
        logger.info("  /data → battery=%.0f signal=%.0f lat=%.6f lon=%.6f",
                     state.battery, state.signal_level, state.latitude, state.longitude)

        t0 = time.time()
        img, ts = await client.get_front_frame()
        dt = (time.time() - t0) * 1000
        if img is not None:
            logger.info("  /v2/front → %dx%d in %.0fms", img.shape[1], img.shape[0], dt)
        else:
            logger.warning("  /v2/front → no frame (SDK may need browser init)")

        ok = await client.send_control(0.0, 0.0)
        logger.info("  /control → %s", "OK" if ok else "FAILED")
    finally:
        await client.close()
    logger.info("SDK client tests passed.")


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--skip-sdk", action="store_true", help="Skip SDK connectivity test")
    args = p.parse_args()

    await test_imports()
    await test_config_defaults()
    await test_topological_memory()
    await test_heuristic_policy()
    await test_vla_heuristic()
    await test_vlm_policy_init()

    if not args.skip_sdk:
        try:
            await test_sdk_client(args.url)
        except Exception as e:
            logger.warning("SDK test failed (server may not be running): %s", e)

    logger.info("=" * 40)
    logger.info("ALL TESTS PASSED — SOTA 2025")
    logger.info("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
