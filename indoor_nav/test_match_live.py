#!/usr/bin/env python3
"""
Live goal-matching probe for the Earth Rover front camera.

This script never sends motion commands. It only:
  1. loads one or more goal images
  2. polls the SDK front camera
  3. computes live similarity against the current goal
  4. prints status and optionally shows a preview window
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

# Prevent OpenBLAS/OpenMP warnings from swamping probe output on CPU builds.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv(*args, **kwargs):
        return False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

from indoor_nav.configs.config import GoalConfig, SDKConfig
from indoor_nav.modules.checkpoint_manager import CheckpointManager
from indoor_nav.modules.sdk_client import RoverSDKClient


MATCH_METHODS = [
    "dinov2_direct",
    "wall_crop_direct",
    "wall_rectify_direct",
    "dinov2_vlad",
    "dinov3_vlad",
    "siglip2",
    "dinov2",
    "eigenplaces",
    "cosplace",
    "clip",
    "sift",
]

logger = logging.getLogger("test_match_live")


def parse_args() -> argparse.Namespace:
    default_sdk_url = os.getenv("INDOOR_SDK_URL") or os.getenv("SDK_BASE_URL") or "http://127.0.0.1:8000"
    parser = argparse.ArgumentParser(
        description="Live front-camera goal matcher probe. Never sends motion commands."
    )
    parser.add_argument(
        "--goals",
        nargs="+",
        required=True,
        help="Goal image files or a single directory containing them.",
    )
    parser.add_argument("--url", default=default_sdk_url, help="SDK base URL")
    parser.add_argument("--device", default="cuda", help="Compute device (cuda/cpu)")
    parser.add_argument(
        "--match-method",
        choices=MATCH_METHODS,
        default="wall_crop_direct",
        help="Goal matching backend to probe live (default: wall_crop_direct)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.78,
        help="Arrival similarity threshold (default: 0.78)",
    )
    parser.add_argument(
        "--match-patience",
        type=int,
        default=3,
        help="Consecutive frames above threshold to count as arrived (default: 3)",
    )
    parser.add_argument("--loop-hz", type=float, default=4.0, help="Polling rate (default: 4 Hz)")
    parser.add_argument(
        "--exit-on-arrival",
        action="store_true",
        help="Exit as soon as a checkpoint is reached.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on processed frames (0 = unlimited).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview window with overlay text.",
    )
    parser.add_argument(
        "--wall-crop-min-area-frac",
        type=float,
        default=0.015,
        help="Minimum image area fraction for wall-aware proposals.",
    )
    parser.add_argument(
        "--wall-crop-max-area-frac",
        type=float,
        default=0.85,
        help="Maximum image area fraction for wall-aware proposals.",
    )
    parser.add_argument(
        "--wall-crop-max-aspect-ratio",
        type=float,
        default=2.5,
        help="Maximum aspect ratio for wall-aware proposals.",
    )
    parser.add_argument(
        "--wall-crop-min-fill-ratio",
        type=float,
        default=0.55,
        help="Minimum contour fill ratio for wall_crop_direct proposals.",
    )
    parser.add_argument(
        "--wall-crop-padding-frac",
        type=float,
        default=0.04,
        help="Padding fraction applied to wall-aware proposals.",
    )
    parser.add_argument(
        "--wall-crop-max-candidates",
        type=int,
        default=6,
        help="Maximum number of wall-aware proposals per frame.",
    )
    parser.add_argument(
        "--wall-crop-score-weight",
        type=float,
        default=0.9,
        help="How strongly wall-aware matchers prefer region matches over full-scene matches.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


def build_goal_config(args: argparse.Namespace) -> GoalConfig:
    cfg = GoalConfig()
    cfg.match_method = args.match_method
    cfg.match_threshold = args.match_threshold
    cfg.match_patience = max(1, int(args.match_patience))
    cfg.feature_device = args.device
    cfg.wall_crop_min_area_frac = args.wall_crop_min_area_frac
    cfg.wall_crop_max_area_frac = args.wall_crop_max_area_frac
    cfg.wall_crop_max_aspect_ratio = args.wall_crop_max_aspect_ratio
    cfg.wall_crop_min_fill_ratio = args.wall_crop_min_fill_ratio
    cfg.wall_crop_padding_frac = args.wall_crop_padding_frac
    cfg.wall_crop_max_candidates = args.wall_crop_max_candidates
    cfg.wall_crop_score_weight = args.wall_crop_score_weight

    if args.match_method == "siglip2":
        cfg.feature_model = "google/siglip2-base-patch16-224"
    elif args.match_method in {"dinov2_vlad", "dinov2_direct", "wall_crop_direct", "wall_rectify_direct"}:
        cfg.feature_model = "facebook/dinov2-with-registers-base"
    elif args.match_method == "dinov3_vlad":
        cfg.feature_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    elif args.match_method == "dinov2":
        cfg.feature_model = "facebook/dinov2-base"
    elif args.match_method == "cosplace":
        cfg.feature_model = (
            f"gmberton/cosplace ({cfg.cosplace_backbone}, "
            f"dim={cfg.cosplace_fc_output_dim})"
        )
    return cfg


def draw_overlay(frame, text_lines: list[str]) -> None:
    y = 28
    for line in text_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
        y += 28


async def run_probe(args: argparse.Namespace) -> int:
    sdk = RoverSDKClient(SDKConfig(base_url=args.url))
    checkpoint_mgr = CheckpointManager(build_goal_config(args))
    if len(args.goals) == 1 and os.path.isdir(args.goals[0]):
        checkpoint_mgr.load_goals_from_dir(args.goals[0])
    else:
        checkpoint_mgr.load_goals(args.goals)

    if not checkpoint_mgr.checkpoints:
        raise RuntimeError("No goal images loaded.")

    logger.info(
        "Live matcher ready: method=%s goals=%d sdk=%s",
        args.match_method,
        len(checkpoint_mgr.checkpoints),
        args.url,
    )
    logger.info("This probe never sends motion commands.")

    stop_event = asyncio.Event()

    def request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    frame_count = 0
    last_frame_warning = 0.0
    gui_enabled = bool(args.show)

    try:
        while not stop_event.is_set():
            tick_start = time.perf_counter()
            frame, frame_ts = await sdk.get_front_frame()
            if frame is None:
                now = time.time()
                if now - last_frame_warning > 2.0:
                    logger.warning("No decodable front frame yet from %s", args.url)
                    last_frame_warning = now
                await asyncio.sleep(1.0 / max(0.5, args.loop_hz))
                continue

            frame_count += 1
            similarity = checkpoint_mgr.compute_goal_similarity(frame)
            trend = checkpoint_mgr.get_similarity_trend()
            reached = checkpoint_mgr.check_arrival(similarity)
            goal = checkpoint_mgr.current_goal
            goal_name = os.path.basename(goal.image_path) if goal is not None else "none"

            logger.info(
                "frame=%d ts=%.3f | %s | trend=%+.3f | goal=%s",
                frame_count,
                frame_ts,
                checkpoint_mgr.status_str(),
                trend,
                goal_name,
            )

            if gui_enabled:
                preview = frame.copy()
                draw_overlay(
                    preview,
                    [
                        checkpoint_mgr.status_str(),
                        f"Trend: {trend:+.3f}",
                        f"Goal: {goal_name}",
                        "Press q to quit",
                    ],
                )
                try:
                    cv2.imshow("Live Goal Match", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop_event.set()
                except cv2.error as exc:
                    logger.warning(
                        "OpenCV GUI preview is unavailable (%s). Continuing without --show.",
                        exc,
                    )
                    gui_enabled = False

            if reached:
                logger.info("Checkpoint reached.")
                if args.exit_on_arrival or checkpoint_mgr.all_done:
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("Reached max frames: %d", args.max_frames)
                break

            elapsed = time.perf_counter() - tick_start
            await asyncio.sleep(max(0.0, (1.0 / max(0.5, args.loop_hz)) - elapsed))
    finally:
        await sdk.close()
        if gui_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(run_probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
