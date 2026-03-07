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

  # ViNT with official visualnav-transformer checkpoint:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vint \\
      --model-path /path/to/vint.pth \\
      --nomad-repo-root /path/to/visualnav-transformer \\
      --match-method sift --obstacle-method simple_edge

  # GNM with official visualnav-transformer checkpoint:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy gnm \\
      --model-path /path/to/gnm_large.pth \\
      --nomad-repo-root /path/to/visualnav-transformer \\
      --match-method sift --obstacle-method simple_edge

  # NoMaD with an official visualnav-transformer checkpoint:
  python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy nomad \\
      --model-path /path/to/nomad.pth \\
      --nomad-repo-root /path/to/visualnav-transformer \\
      --match-method sift --obstacle-method simple_edge

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

from indoor_nav.cli_common import add_common_args, build_config
from indoor_nav.agent import IndoorNavigationAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICRA 2025 Indoor Navigation — Earth Rover Challenge (SOTA 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(p)
    return p.parse_args()


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
