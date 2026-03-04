from __future__ import annotations

import argparse
import asyncio
import logging

from erc_autonomy.config import ERCConfig
from erc_autonomy.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ERC Autonomous GPS runner (Week 1-6 scaffold)."
    )
    p.add_argument("--url", default="http://127.0.0.1:8000", help="SDK base URL")
    p.add_argument("--loop-hz", type=float, default=10.0, help="Main control loop rate")
    p.add_argument("--stale-ms", type=int, default=1200, help="Watchdog stale threshold")
    p.add_argument("--request-timeout", type=float, default=3.0, help="HTTP timeout (s)")
    p.add_argument(
        "--traversability-backend",
        default="simple_edge",
        choices=["simple_edge", "sam2"],
        help="Traversability backend",
    )
    p.add_argument(
        "--enable-motion",
        action="store_true",
        help="Enable reactive motion (disabled by default for safety)",
    )
    p.add_argument("--max-linear", type=float, default=0.25, help="Max linear command")
    p.add_argument("--max-angular", type=float, default=0.4, help="Max angular command")
    p.add_argument(
        "--checkpoint-distance",
        type=float,
        default=16.0,
        help="Distance threshold (m) for /checkpoint-reached attempts",
    )
    p.add_argument(
        "--checkpoint-refresh",
        type=float,
        default=6.0,
        help="Seconds between /checkpoints-list refreshes",
    )
    p.add_argument(
        "--no-auto-checkpoint",
        action="store_true",
        help="Disable automatic /checkpoint-reached reporting",
    )
    p.add_argument(
        "--checkpoint-slowdown-start",
        type=float,
        default=28.0,
        help="Distance (m) where speed taper starts near checkpoint",
    )
    p.add_argument(
        "--checkpoint-slowdown-hard",
        type=float,
        default=8.0,
        help="Distance (m) where minimum checkpoint speed factor applies",
    )
    p.add_argument(
        "--checkpoint-slowdown-min-factor",
        type=float,
        default=0.45,
        help="Minimum speed factor near checkpoint",
    )
    p.add_argument(
        "--checkpoint-angular-min-factor",
        type=float,
        default=0.55,
        help="Minimum angular factor near checkpoint",
    )
    p.add_argument(
        "--checkpoint-failure-effect",
        type=float,
        default=12.0,
        help="Seconds that failed checkpoint feedback affects speed",
    )
    p.add_argument(
        "--checkpoint-failure-buffer",
        type=float,
        default=6.0,
        help="Extra distance window (m) around failed proximate distance for slowdown",
    )
    p.add_argument(
        "--checkpoint-failure-min-factor",
        type=float,
        default=0.3,
        help="Minimum speed factor applied after failed checkpoint reports",
    )
    p.add_argument(
        "--checkpoint-failure-angular-min-factor",
        type=float,
        default=0.45,
        help="Minimum angular factor applied after failed checkpoint reports",
    )
    p.add_argument(
        "--no-recovery",
        action="store_true",
        help="Disable stuck recovery behavior",
    )
    p.add_argument(
        "--recovery-stuck-timeout",
        type=float,
        default=4.0,
        help="Seconds of commanded-but-not-moving before recovery starts",
    )
    p.add_argument("--start-mission", action="store_true", help="Call /start-mission on boot")
    p.add_argument("--end-mission", action="store_true", help="Call /end-mission on shutdown")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ERCConfig:
    cfg = ERCConfig()
    cfg.base_url = args.url
    cfg.loop_hz = args.loop_hz
    cfg.stale_sensor_ms = args.stale_ms
    cfg.request_timeout_s = args.request_timeout
    cfg.traversability_backend = args.traversability_backend
    cfg.enable_motion = args.enable_motion
    cfg.max_linear = args.max_linear
    cfg.max_angular = args.max_angular
    cfg.checkpoint_attempt_distance_m = args.checkpoint_distance
    cfg.checkpoint_refresh_interval_s = args.checkpoint_refresh
    cfg.auto_checkpoint_report = not args.no_auto_checkpoint
    cfg.checkpoint_slowdown_start_m = args.checkpoint_slowdown_start
    cfg.checkpoint_slowdown_hard_m = args.checkpoint_slowdown_hard
    cfg.checkpoint_slowdown_min_factor = args.checkpoint_slowdown_min_factor
    cfg.checkpoint_angular_min_factor = args.checkpoint_angular_min_factor
    cfg.checkpoint_failure_effect_s = args.checkpoint_failure_effect
    cfg.checkpoint_failure_buffer_m = args.checkpoint_failure_buffer
    cfg.checkpoint_failure_min_factor = args.checkpoint_failure_min_factor
    cfg.checkpoint_failure_angular_min_factor = args.checkpoint_failure_angular_min_factor
    cfg.recovery_enabled = not args.no_recovery
    cfg.recovery_stuck_timeout_s = args.recovery_stuck_timeout
    cfg.start_mission_on_boot = args.start_mission
    cfg.end_mission_on_shutdown = args.end_mission
    return cfg


async def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = build_config(args)
    logging.getLogger(__name__).info("starting erc gps runner")
    try:
        from erc_autonomy.mission_runner import AutonomousMissionRunner
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "")
        if missing:
            raise SystemExit(
                f"Missing dependency: {missing}. Install project requirements "
                "before running the mission runner."
            ) from exc
        raise

    runner = AutonomousMissionRunner(cfg)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
