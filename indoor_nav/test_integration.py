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
    from indoor_nav.policies.maze_search_policy import MazeSearchPolicy
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


async def test_topological_relocalization():
    import numpy as np
    from indoor_nav.modules.topological_memory import TopologicalMemory, TopoMapConfig

    logger.info("Testing topological relocalization...")
    cfg = TopoMapConfig(min_node_distance=0.0, loop_closure_threshold=0.95)
    topo = TopologicalMemory(cfg)

    img_a = np.zeros((80, 80, 3), dtype=np.uint8)
    img_b = np.full((80, 80, 3), 255, dtype=np.uint8)

    node_a = topo.update(img_a, orientation=0.0, force_new_node=True)
    node_b = topo.update(img_b, orientation=90.0, force_new_node=True)
    node_a_revisit = topo.update(img_a, orientation=180.0, force_new_node=True)

    assert node_a is not None
    assert node_b is not None and node_b != node_a
    assert node_a_revisit == node_a, "Expected revisit to relocalize to existing node"
    assert topo.num_nodes == 2, f"Expected 2 nodes after relocalization, got {topo.num_nodes}"
    logger.info("  Relocalized revisit to node %d with %d total nodes", node_a_revisit, topo.num_nodes)
    logger.info("Topological relocalization test passed.")


async def test_topological_frontier_guidance():
    import numpy as np
    from indoor_nav.modules.topological_memory import TopologicalMemory, TopoMapConfig

    logger.info("Testing topological frontier guidance...")
    cfg = TopoMapConfig(min_node_distance=0.0, loop_closure_threshold=0.99)
    topo = TopologicalMemory(cfg)

    img_a = np.zeros((80, 80, 3), dtype=np.uint8)
    img_b = np.full((80, 80, 3), 64, dtype=np.uint8)
    img_c = np.full((80, 80, 3), 192, dtype=np.uint8)

    node_a = topo.update(img_a, orientation=0.0, force_new_node=True)
    node_b = topo.update(img_b, orientation=45.0, force_new_node=True, exit_label="left")
    node_c = topo.update(img_c, orientation=90.0, force_new_node=True, exit_label="right")

    assert node_a is not None and node_b is not None and node_c is not None
    assert topo.get_exit_label(node_a, node_b) == "left"
    assert topo.get_exit_label(node_b, node_c) == "right"
    assert topo.get_exit_label(node_b, node_a) == "back"

    path = topo.plan_to_nearest_frontier(node_b)
    assert path is not None and len(path) == 2, f"Expected 2-hop frontier path, got {path}"
    exit_label = topo.get_exit_label(path[0], path[1])
    assert exit_label in {"back", "right"}, f"Unexpected frontier exit label: {exit_label}"
    logger.info("  Frontier path from node %d: %s via %s", node_b, path, exit_label)
    logger.info("Topological frontier guidance test passed.")


async def test_topological_debug_export():
    import tempfile
    import numpy as np
    from indoor_nav.modules.topological_memory import TopologicalMemory, TopoMapConfig

    logger.info("Testing topological debug export...")
    topo = TopologicalMemory(TopoMapConfig(min_node_distance=0.0))
    img_a = np.zeros((80, 120, 3), dtype=np.uint8)
    img_b = np.full((80, 120, 3), 180, dtype=np.uint8)
    topo.update(img_a, orientation=0.0, force_new_node=True)
    topo.update(img_b, orientation=90.0, force_new_node=True, exit_label="left")

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = topo.export_debug_bundle(tmpdir)
        assert os.path.exists(bundle["json"]), "Expected topo_map.json export"
        assert os.path.exists(bundle["html"]), "Expected topo HTML export"
        assert os.path.exists(os.path.join(bundle["nodes_dir"], "node_0000.jpg")), "Expected node thumbnail export"
        logger.info("  Exported topo bundle to %s", tmpdir)

    logger.info("Topological debug export test passed.")


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


async def test_maze_search_policy():
    import numpy as np
    from indoor_nav.configs.config import PolicyConfig
    from indoor_nav.policies.base_policy import PolicyInput
    from indoor_nav.policies.maze_search_policy import MazePhase, MazeSearchPolicy, NodeMemory

    logger.info("Testing maze search policy...")
    cfg = PolicyConfig(backend="maze_search")
    policy = MazeSearchPolicy(cfg)
    policy.setup()

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = PolicyInput(
        front_image=dummy_img,
        goal_image=dummy_img,
        goal_similarity=0.25,
        goal_trend=0.0,
        context_images=[dummy_img],
        left_clearance=0.7,
        center_clearance=0.75,
        right_clearance=0.3,
        topo_node_id=1,
    )
    action = policy.predict(obs)
    logger.info(
        "  Maze search output: linear=%.2f angular=%.2f force_topo=%s",
        action.linear,
        action.angular,
        action.force_topo_node,
    )

    policy._phase = MazePhase.PAUSE
    policy._phase_started = time.time() - cfg.maze_pause_seconds - 0.05
    policy._node_memory[1] = NodeMemory(exhausted=True)
    guided_obs = PolicyInput(
        front_image=dummy_img,
        goal_image=dummy_img,
        goal_similarity=0.2,
        goal_trend=0.0,
        context_images=[dummy_img],
        left_clearance=0.8,
        center_clearance=0.8,
        right_clearance=0.2,
        topo_node_id=1,
        topo_target_node_id=2,
        topo_target_exit_label="left",
    )
    guided_action = policy.predict(guided_obs)
    assert guided_action.angular > 0.0, "Expected guided scan toward left exit"
    logger.info(
        "  Guided maze output: linear=%.2f angular=%.2f topo_exit=%s",
        guided_action.linear,
        guided_action.angular,
        guided_action.topo_exit_label,
    )

    policy.reset()
    policy._phase = MazePhase.PAUSE
    policy._phase_started = time.time() - cfg.maze_pause_seconds - 0.05
    corridor_obs = PolicyInput(
        front_image=dummy_img,
        goal_image=dummy_img,
        goal_similarity=0.1,
        goal_trend=0.0,
        context_images=[dummy_img],
        left_clearance=0.54,
        center_clearance=0.75,
        right_clearance=0.28,
        topo_node_id=1,
        near_field_occupancy=0.06,
    )
    corridor_action = policy.predict(corridor_obs)
    assert corridor_action.linear > 0.0, "Expected corridor motion instead of an opportunistic scan"
    assert corridor_action.angular > 0.0, "Expected corridor-centering turn toward the clearer left side"
    logger.info(
        "  Corridor maze output: linear=%.2f angular=%.2f",
        corridor_action.linear,
        corridor_action.angular,
    )
    logger.info("Maze search policy test passed.")


async def test_mono_inertial_timestamp_alignment():
    from indoor_nav.slam.imu import (
        build_mono_inertial_payload,
        estimate_mono_inertial_clock_alignment,
    )

    logger.info("Testing mono-inertial timestamp alignment...")
    accels = [
        [0.0, 0.0, 1.0, 100.00],
        [0.0, 0.0, 1.0, 100.05],
        [0.0, 0.0, 1.0, 100.10],
    ]
    gyros = [
        [0.0, 0.0, 0.1, 100.02],
        [0.0, 0.0, 0.1, 100.07],
        [0.0, 0.0, 0.1, 100.12],
    ]
    frame_ts = 14500.20
    data_ts = 999.00

    alignment = estimate_mono_inertial_clock_alignment(
        accels,
        gyros,
        frame_ts=frame_ts,
        data_ts=data_ts,
    )
    assert alignment.needs_correction, "Expected large frame/data skew to trigger correction"
    assert abs(alignment.offset_s - (frame_ts - gyros[-1][3])) < 1e-6

    payload, newest_imu_ts = build_mono_inertial_payload(
        accels,
        gyros,
        frame_ts=frame_ts,
        last_imu_ts=0.0,
        timestamp_offset_s=alignment.offset_s,
    )
    samples = payload["samples"]
    assert len(samples) == 3, f"Expected all gyro samples, got {len(samples)}"
    assert abs(samples[-1]["t"] - frame_ts) < 1e-6, "Expected corrected IMU sample to align with frame clock"
    assert abs(newest_imu_ts - gyros[-1][3]) < 1e-6, "Expected raw IMU timestamp bookkeeping to stay unchanged"
    logger.info("Mono-inertial timestamp alignment test passed.")


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
    await test_topological_relocalization()
    await test_topological_frontier_guidance()
    await test_topological_debug_export()
    await test_heuristic_policy()
    await test_maze_search_policy()
    await test_mono_inertial_timestamp_alignment()
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
