#!/usr/bin/env python3
"""
run_indoor.py — Launch the Indoor Navigation Agent (SOTA 2025).

Usage:
  # With goal images directory (RECOMMENDED — Qwen2.5-VL + DINOv2-VLAD):
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \\
      --policy vlm_hybrid --vlm-endpoint http://localhost:8000/v1 \\
      --vlm-model Qwen/Qwen2.5-VL-7B-Instruct --match-method dinov2_vlad

  # DINOv3-VLAD evaluation mode:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \\
      --policy vlm_hybrid --match-method dinov3_vlad

  # With explicit goal image files:
  python indoor_nav/run_indoor.py --goals goal1.jpg goal2.jpg goal3.jpg

  # Heuristic mode (no GPU needed):
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy heuristic

  # VLA mode (OpenVLA):
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vla

  # VLM with GPT-4o:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid \\
      --vlm-endpoint https://api.openai.com/v1 --vlm-model gpt-4o \\
      --vlm-api-key sk-...

  # VLM with local Ollama (legacy):
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid \\
      --vlm-endpoint http://localhost:11434/api/generate --vlm-model llava:13b

  # NoMaD with DINOv2 baseline matching:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy nomad \\
      --match-method dinov2

  # Full competition run with mission:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid \\
      --url http://127.0.0.1:8000 --mission-slug indoor-mission-1 \\
      --vlm-endpoint http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from indoor_nav.configs.config import IndoorNavConfig
from indoor_nav.agent import IndoorNavigationAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICRA 2025 Indoor Navigation — Earth Rover Challenge (SOTA 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--goals",
        nargs="+",
        required=True,
        help="Goal image files or a single directory containing them (in order)",
    )

    # SDK
    p.add_argument("--url", default="http://127.0.0.1:8000", help="SDK base URL")
    p.add_argument("--mission-slug", default="", help="Mission slug (if running a scored mission)")

    # Policy
    p.add_argument(
        "--policy",
        choices=["nomad", "vint", "vlm_hybrid", "vla", "heuristic"],
        default="vlm_hybrid",
        help="Navigation policy backend (default: vlm_hybrid)",
    )
    p.add_argument("--model-path", default="", help="Path to policy model weights")
    p.add_argument("--device", default="cuda", help="Compute device (cuda/cpu)")

    # VLM options (SOTA: Qwen2.5-VL 7B)
    p.add_argument("--vlm-endpoint", default="", help="VLM API endpoint (e.g., http://localhost:8000/v1)")
    p.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model name")
    p.add_argument("--vlm-api-key", default="", help="API key for VLM endpoint")

    # Goal matching (SOTA: DINOv2-VLAD / DINOv3-VLAD)
    p.add_argument(
        "--match-method",
        choices=["dinov2_vlad", "dinov3_vlad", "siglip2", "dinov2", "eigenplaces", "clip", "sift"],
        default="dinov2_vlad",
        help="Image goal matching method (default: dinov2_vlad)",
    )
    p.add_argument("--match-threshold", type=float, default=0.78, help="Similarity threshold for arrival")

    # Control
    p.add_argument("--max-speed", type=float, default=0.6, help="Max linear speed [0-1]")
    p.add_argument("--loop-hz", type=float, default=10.0, help="Control loop frequency")

    # Obstacle avoidance
    p.add_argument("--no-obstacle", action="store_true", help="Disable obstacle avoidance")
    p.add_argument(
        "--obstacle-method",
        choices=["depth_anything", "depth_pro", "simple_edge"],
        default="depth_anything",
        help="Obstacle detection method",
    )

    # Topological memory
    p.add_argument("--no-topo", action="store_true", help="Disable topological memory")
    p.add_argument("--topo-max-nodes", type=int, default=500, help="Max topo graph nodes")

    # Recovery
    p.add_argument("--no-recovery", action="store_true", help="Disable recovery behaviors")

    # Logging
    p.add_argument("--no-log", action="store_true", help="Disable HDF5 logging")
    p.add_argument("--log-dir", default="indoor_nav/logs", help="Log output directory")

    # Debug
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> IndoorNavConfig:
    """Build IndoorNavConfig from CLI arguments."""
    cfg = IndoorNavConfig()

    # SDK
    cfg.sdk.base_url = args.url
    cfg.mission_slug = args.mission_slug

    # Policy
    cfg.policy.backend = args.policy
    cfg.policy.device = args.device
    if args.model_path:
        cfg.policy.model_path = args.model_path
    if args.vlm_endpoint:
        cfg.policy.vlm_endpoint = args.vlm_endpoint
    cfg.policy.vlm_model = args.vlm_model
    if args.vlm_api_key:
        cfg.policy.vlm_api_key = args.vlm_api_key

    # Goal matching
    cfg.goal.match_method = args.match_method
    cfg.goal.match_threshold = args.match_threshold
    cfg.goal.feature_device = args.device

    # Auto-select feature model based on match method
    if args.match_method == "siglip2":
        cfg.goal.feature_model = "google/siglip2-base-patch16-224"
    elif args.match_method == "dinov2_vlad":
        # Use registers variant for cleaner patch features (Darcet et al., 2024)
        cfg.goal.feature_model = "facebook/dinov2-with-registers-base"
    elif args.match_method == "dinov3_vlad":
        # DINOv3 ViT-B/16 with LVD-1689M pretrain (released Aug 2025).
        cfg.goal.feature_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    elif args.match_method == "dinov2":
        cfg.goal.feature_model = "facebook/dinov2-base"

    # Control
    cfg.control.max_linear = args.max_speed
    cfg.control.max_angular = args.max_speed
    cfg.control.loop_hz = args.loop_hz

    # Obstacle avoidance
    cfg.obstacle.enabled = not args.no_obstacle
    cfg.obstacle.method = args.obstacle_method
    cfg.obstacle.depth_device = args.device
    if args.obstacle_method == "depth_pro":
        cfg.obstacle.depth_model = "apple/DepthPro"
    elif args.obstacle_method == "depth_anything":
        cfg.obstacle.depth_model = "depth-anything/Depth-Anything-V2-Base-hf"

    # Topological memory
    cfg.topo_memory.enabled = not args.no_topo
    cfg.topo_memory.max_nodes = args.topo_max_nodes

    # Recovery
    cfg.recovery.enabled = not args.no_recovery

    # Logging
    cfg.log.enabled = not args.no_log
    cfg.log.log_dir = args.log_dir

    return cfg


async def main():
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(args)
    agent = IndoorNavigationAgent(cfg)

    # Handle SIGINT/SIGTERM gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.request_stop)

    # Determine goal images
    goal_images = args.goals

    logger = logging.getLogger("run_indoor")
    logger.info("=" * 60)
    logger.info("ICRA 2025 INDOOR NAVIGATION — EARTH ROVER CHALLENGE")
    logger.info("SOTA 2025 Configuration")
    logger.info("=" * 60)
    logger.info("Policy:     %s", args.policy)
    logger.info("VLM:        %s (%s)", args.vlm_model, args.vlm_endpoint or "no endpoint")
    logger.info("Goals:      %s", goal_images)
    logger.info("Match:      %s (threshold=%.2f)", args.match_method, args.match_threshold)
    logger.info("SDK URL:    %s", args.url)
    logger.info("Max speed:  %.2f", args.max_speed)
    logger.info("Loop rate:  %.0f Hz", args.loop_hz)
    logger.info("Obstacles:  %s (%s)", "ON" if cfg.obstacle.enabled else "OFF", args.obstacle_method)
    logger.info("Topo map:   %s (max %d nodes)", "ON" if cfg.topo_memory.enabled else "OFF", args.topo_max_nodes)
    logger.info("Recovery:   %s", "ON" if cfg.recovery.enabled else "OFF")
    logger.info("Logging:    %s", "ON" if cfg.log.enabled else "OFF")
    logger.info("=" * 60)

    await agent.run(goal_images)


if __name__ == "__main__":
    asyncio.run(main())
