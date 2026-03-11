#!/usr/bin/env python3
"""
Preflight checks for indoor navigation runs.

Examples:
  python indoor_nav/check_indoor.py --goals indoor_nav/goals/
  python indoor_nav/check_indoor.py --goals indoor_nav/goals/ --policy heuristic --skip-sdk
  python indoor_nav/check_indoor.py --goals indoor_nav/goals/ --probe-model-load
  python indoor_nav/check_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid \
      --vlm-endpoint http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import cv2
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv(*args, **kwargs):
        return False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

from indoor_nav.cli_common import add_common_args, build_config, capture_cli_flags
from indoor_nav.modules.checkpoint_manager import GoalMatcher
from indoor_nav.modules.obstacle_avoidance import ObstacleDetector
from indoor_nav.modules.sdk_client import decode_b64_image
from indoor_nav.policies.nomad_policy import NoMaDPolicy
from indoor_nav.policies.vla_policy import VLAPolicy
from indoor_nav.slam.orbslam3_client import ORBSLAM3Client

logger = logging.getLogger("check_indoor")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fatal: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preflight checks for the indoor navigation stack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(p)
    p.add_argument(
        "--skip-sdk",
        action="store_true",
        help="Skip SDK /data and /v2/front checks",
    )
    p.add_argument(
        "--probe-control",
        action="store_true",
        help="Also send a zero-velocity /control command",
    )
    p.add_argument(
        "--probe-model-load",
        action="store_true",
        help="Attempt to instantiate the selected matcher/obstacle/policy backends",
    )
    args = p.parse_args()
    args._cli_flags = capture_cli_flags()
    return args


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _format_module_list(modules: Iterable[str]) -> str:
    return ", ".join(modules)


def _resolve_goal_paths(goal_args: List[str]) -> List[Path]:
    if len(goal_args) == 1:
        candidate = Path(goal_args[0]).expanduser()
        if candidate.is_dir():
            return sorted(
                path for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return [Path(arg).expanduser() for arg in goal_args]


def _check_goals(goal_args: List[str]) -> tuple[Check, List[Path], Optional[object]]:
    goal_paths = _resolve_goal_paths(goal_args)
    if not goal_paths:
        return Check("Goals", "FAIL", "no goal images found", fatal=True), [], None

    loaded_count = 0
    first_image = None
    for path in goal_paths:
        if not path.exists():
            return Check("Goals", "FAIL", f"missing file: {path}", fatal=True), [], None
        if not path.is_file():
            return Check("Goals", "FAIL", f"path is not a file: {path}", fatal=True), [], None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return Check("Goals", "FAIL", f"unable to decode image: {path}", fatal=True), [], None
        loaded_count += 1
        if first_image is None:
            first_image = img

    detail = f"{loaded_count} image(s) loaded; first={goal_paths[0].name} {first_image.shape[1]}x{first_image.shape[0]}"
    return Check("Goals", "OK", detail), goal_paths, first_image


async def _probe_sdk(base_url: str, timeout_s: float, probe_control: bool) -> List[Check]:
    results: List[Check] = []
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    base = base_url.rstrip("/") + "/"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        data_url = urljoin(base, "data")
        try:
            async with session.get(data_url) as resp:
                if resp.status != 200:
                    results.append(Check("SDK /data", "FAIL", f"HTTP {resp.status}", fatal=True))
                else:
                    payload = await resp.json()
                    keys = sorted(payload.keys())
                    sample = ", ".join(keys[:5]) if keys else "no keys"
                    results.append(Check("SDK /data", "OK", f"HTTP 200; keys={sample}"))
        except Exception as exc:
            results.append(Check("SDK /data", "FAIL", f"{type(exc).__name__}: {exc}", fatal=True))

        frame_url = urljoin(base, "v2/front")
        try:
            async with session.get(frame_url) as resp:
                if resp.status != 200:
                    results.append(Check("SDK /v2/front", "FAIL", f"HTTP {resp.status}", fatal=True))
                else:
                    payload = await resp.json()
                    b64 = payload.get("front_frame", "")
                    img = decode_b64_image(b64) if b64 else None
                    if img is None:
                        results.append(
                            Check(
                                "SDK /v2/front",
                                "FAIL",
                                "no decodable front frame; browser/Agora may not be initialized",
                                fatal=True,
                            )
                        )
                    else:
                        results.append(Check("SDK /v2/front", "OK", f"{img.shape[1]}x{img.shape[0]} frame"))
        except Exception as exc:
            results.append(Check("SDK /v2/front", "FAIL", f"{type(exc).__name__}: {exc}", fatal=True))

        if probe_control:
            control_url = urljoin(base, "control")
            try:
                payload = {"command": {"linear": 0.0, "angular": 0.0}}
                async with session.post(control_url, json=payload) as resp:
                    if resp.status == 200:
                        results.append(Check("SDK /control", "OK", "zero command accepted"))
                    else:
                        results.append(Check("SDK /control", "WARN", f"HTTP {resp.status}"))
            except Exception as exc:
                results.append(Check("SDK /control", "WARN", f"{type(exc).__name__}: {exc}"))

    return results


def _check_matcher(cfg, probe_image, probe_load: bool) -> Check:
    method = cfg.goal.match_method
    if method == "sift":
        return Check("Goal matcher", "OK", "sift uses OpenCV only")

    required = ["torch", "PIL"]
    if method in {"dinov2_vlad", "dinov3_vlad", "dinov2_direct", "wall_crop_direct", "wall_rectify_direct", "siglip2", "dinov2", "clip"}:
        required.append("transformers")
    elif method in {"eigenplaces", "cosplace"}:
        required.append("torchvision")

    missing = [name for name in required if not _module_available(name)]
    if missing:
        return Check(
            "Goal matcher",
            "FAIL",
            f"{method} missing dependencies: {_format_module_list(missing)}",
            fatal=True,
        )

    detail = f"{method} configured ({cfg.goal.feature_model})"
    if method == "eigenplaces":
        detail += "; may fall back to DINOv2 if torch.hub load fails"
    elif method == "cosplace":
        detail += "; torch.hub will fetch gmberton/cosplace weights on first use"

    if not probe_load:
        detail += "; model load not probed"
        return Check("Goal matcher", "OK", detail)

    try:
        matcher = GoalMatcher(cfg.goal)
        matcher.extract_feature(probe_image)
        return Check("Goal matcher", "OK", f"{detail}; probe feature extraction succeeded")
    except Exception as exc:
        return Check("Goal matcher", "FAIL", f"{detail}; probe failed: {exc}", fatal=True)


def _check_obstacles(cfg, probe_image, probe_load: bool) -> Check:
    if not cfg.obstacle.enabled:
        return Check("Obstacle backend", "OK", "disabled")

    method = cfg.obstacle.method
    if method == "simple_edge":
        return Check("Obstacle backend", "OK", "simple_edge uses OpenCV only")

    required = ["torch", "transformers", "PIL"]
    missing = [name for name in required if not _module_available(name)]
    if missing:
        return Check(
            "Obstacle backend",
            "FAIL",
            f"{method} missing dependencies: {_format_module_list(missing)}",
            fatal=True,
        )

    detail = f"{method} configured ({cfg.obstacle.depth_model})"
    if method == "depth_pro":
        detail += "; runtime falls back to Depth Anything if load fails"

    if not probe_load:
        detail += "; model load not probed"
        return Check("Obstacle backend", "OK", detail)

    try:
        detector = ObstacleDetector(cfg.obstacle)
        detector.estimate_depth(probe_image)
        return Check("Obstacle backend", "OK", f"{detail}; depth probe succeeded")
    except Exception as exc:
        return Check("Obstacle backend", "FAIL", f"{detail}; probe failed: {exc}", fatal=True)


async def _probe_vlm_endpoint(api_format: str, endpoint: str, api_key: str) -> Check:
    timeout = aiohttp.ClientTimeout(total=5)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    parsed = urlparse(endpoint)
    base = endpoint.rstrip("/")
    probe_url = base

    if api_format == "openai":
        probe_url = f"{base}/models" if parsed.path.endswith("/v1") else urljoin(f"{base}/", "models")
    elif api_format == "ollama":
        root = f"{parsed.scheme}://{parsed.netloc}"
        probe_url = urljoin(f"{root}/", "api/tags")

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(probe_url) as resp:
                if resp.status in {200, 401, 403}:
                    detail = f"{api_format} endpoint reachable via {probe_url} (HTTP {resp.status})"
                    if resp.status in {401, 403}:
                        return Check("VLM endpoint", "WARN", detail)
                    return Check("VLM endpoint", "OK", detail)
                return Check("VLM endpoint", "WARN", f"{probe_url} returned HTTP {resp.status}")
    except Exception as exc:
        return Check("VLM endpoint", "WARN", f"{probe_url} probe failed: {type(exc).__name__}: {exc}")


def _detect_vlm_api_format(endpoint: str) -> str:
    if "11434" in endpoint or "/api/generate" in endpoint:
        return "ollama"
    if "anthropic" in endpoint:
        return "anthropic"
    return "openai"


async def _check_policy(cfg, probe_load: bool) -> List[Check]:
    backend = cfg.policy.backend
    results: List[Check] = []

    if backend == "heuristic":
        return [Check("Policy backend", "OK", "heuristic mode selected")]

    if backend == "maze_search":
        return [Check("Policy backend", "OK", "maze_search mode selected")]

    if backend in {"nomad", "vint", "gnm"}:
        missing = [name for name in ["torch"] if not _module_available(name)]
        if missing:
            return [
                Check(
                    "Policy backend",
                    "WARN",
                    f"{backend} missing {_format_module_list(missing)}; runtime will use heuristic fallback",
                )
            ]

        model_path = Path(cfg.policy.model_path).expanduser()
        if not model_path.exists():
            detail = f"{backend} weights missing at {model_path}; runtime will use heuristic fallback"
            if probe_load:
                detail += " (probe confirms fallback path only)"
            return [Check("Policy backend", "WARN", detail)]

        if model_path.suffix.lower() == ".pth":
            if backend == "nomad":
                required = ["torch", "torchvision", "PIL", "yaml", "diffusers", "efficientnet_pytorch", "einops"]
            elif backend == "vint":
                required = ["torch", "torchvision", "PIL", "yaml", "efficientnet_pytorch", "warmup_scheduler"]
            else:
                required = ["torch", "PIL", "yaml"]
            missing = [name for name in required if not _module_available(name)]
            if missing:
                return [
                    Check(
                        "Policy backend",
                        "WARN",
                        f"{backend} official .pth checkpoint found but missing {_format_module_list(missing)}; runtime will use heuristic fallback",
                    )
                ]

        if not probe_load:
            mode = "official .pth" if model_path.suffix.lower() == ".pth" else "TorchScript"
            return [Check("Policy backend", "OK", f"{backend} {mode} weights found at {model_path}; model load not probed")]

        try:
            policy = NoMaDPolicy(cfg.policy)
            policy.setup()
            if policy._model is None:
                return [Check("Policy backend", "WARN", f"{backend} setup fell back to heuristic")]
            return [Check("Policy backend", "OK", f"{backend} model probe succeeded")]
        except Exception as exc:
            return [Check("Policy backend", "FAIL", f"{backend} probe failed: {exc}", fatal=True)]

    if backend == "vla":
        missing = [name for name in ["torch", "transformers"] if not _module_available(name)]
        if missing:
            return [
                Check(
                    "Policy backend",
                    "WARN",
                    f"vla missing {_format_module_list(missing)}; runtime will use heuristic_plus fallback",
                )
            ]

        if not probe_load:
            detail = f"vla configured ({cfg.policy.vla_backend}:{cfg.policy.model_path or 'openvla/openvla-7b'})"
            detail += "; model load not probed"
            return [Check("Policy backend", "OK", detail)]

        try:
            policy = VLAPolicy(cfg.policy)
            policy.setup()
            if policy._model is None:
                return [Check("Policy backend", "WARN", "vla setup fell back to heuristic_plus")]
            return [Check("Policy backend", "OK", f"vla model probe succeeded ({policy._vla_backend})")]
        except Exception as exc:
            return [Check("Policy backend", "FAIL", f"vla probe failed: {exc}", fatal=True)]

    if backend == "vlm_hybrid":
        api_format = _detect_vlm_api_format(cfg.policy.vlm_endpoint or "")
        results.append(
            Check(
                "Policy backend",
                "OK",
                f"vlm_hybrid configured ({api_format}, interval={cfg.policy.vlm_query_interval:.1f}s)",
            )
        )
        if not cfg.policy.vlm_endpoint:
            results.append(Check("VLM endpoint", "WARN", "unset; runtime will stay on reactive defaults"))
            return results
        results.append(await _probe_vlm_endpoint(api_format, cfg.policy.vlm_endpoint, cfg.policy.vlm_api_key))
        return results

    return [Check("Policy backend", "FAIL", f"unknown backend: {backend}", fatal=True)]


async def _check_slam(cfg, probe_load: bool) -> List[Check]:
    if not cfg.slam.enabled:
        return [Check("SLAM", "OK", "disabled")]

    results: List[Check] = []
    results.append(Check("SLAM backend", "OK", f"{cfg.slam.backend} ({cfg.slam.mode})"))

    vocab_path = Path(cfg.slam.vocab_path).expanduser()
    if not vocab_path.exists():
        results.append(Check("SLAM vocab", "FAIL", f"missing file: {vocab_path}", fatal=True))
    else:
        results.append(Check("SLAM vocab", "OK", str(vocab_path)))

    settings_path = Path(cfg.slam.settings_path).expanduser()
    if not settings_path.exists():
        results.append(Check("SLAM settings", "FAIL", f"missing file: {settings_path}", fatal=True))
    else:
        results.append(Check("SLAM settings", "OK", str(settings_path)))

    if cfg.slam.backend != "orbslam3":
        results.append(Check("SLAM endpoint", "FAIL", f"unsupported backend: {cfg.slam.backend}", fatal=True))
        return results

    if not probe_load:
        results.append(Check("SLAM endpoint", "WARN", f"{cfg.slam.endpoint} not probed (use --probe-model-load)"))
        return results

    try:
        client = ORBSLAM3Client(cfg.slam)
        try:
            await client.start()
            status = await client.status()
        finally:
            await client.close()
        detail = f"{cfg.slam.endpoint} reachable; tracking_state={status.tracking_state}"
        results.append(Check("SLAM endpoint", "OK", detail))
    except Exception as exc:
        results.append(Check("SLAM endpoint", "FAIL", f"probe failed: {type(exc).__name__}: {exc}", fatal=True))

    return results


def _print_check(check: Check) -> None:
    print(f"- {check.name}: {check.status} ({check.detail})")


def _render_run_command(args: argparse.Namespace) -> str:
    parts = ["python", "indoor_nav/run_indoor.py", "--goals", *args.goals]

    option_pairs = [
        ("--url", args.url),
        ("--policy", args.policy),
        ("--device", args.device),
        ("--match-method", args.match_method),
        ("--match-threshold", str(args.match_threshold)),
        ("--max-speed", str(args.max_speed)),
        ("--loop-hz", str(args.loop_hz)),
        ("--obstacle-method", args.obstacle_method),
        ("--topo-max-nodes", str(args.topo_max_nodes)),
    ]
    if args.mission_slug:
        option_pairs.append(("--mission-slug", args.mission_slug))
    if args.model_path:
        option_pairs.append(("--model-path", args.model_path))
    if args.nomad_repo_root:
        option_pairs.append(("--nomad-repo-root", args.nomad_repo_root))
    if args.nomad_config_path:
        option_pairs.append(("--nomad-config-path", args.nomad_config_path))
    if args.nomad_samples != 1:
        option_pairs.append(("--nomad-samples", str(args.nomad_samples)))
    if args.vlm_endpoint:
        option_pairs.append(("--vlm-endpoint", args.vlm_endpoint))
    if args.vlm_model:
        option_pairs.append(("--vlm-model", args.vlm_model))
    if args.vlm_api_key:
        option_pairs.append(("--vlm-api-key", "***"))
    if args.no_obstacle:
        parts.append("--no-obstacle")
    if args.no_topo:
        parts.append("--no-topo")
    if args.no_recovery:
        parts.append("--no-recovery")
    if args.no_log:
        parts.append("--no-log")
    if args.verbose:
        parts.append("--verbose")
    if args.no_slam:
        parts.append("--no-slam")
    if args.slam_backend != "off":
        parts.extend(["--slam-backend", args.slam_backend])
        parts.extend(["--slam-mode", args.slam_mode])
        parts.extend(["--slam-endpoint", args.slam_endpoint])
        parts.extend(["--slam-vocab", args.slam_vocab])
        parts.extend(["--slam-settings", args.slam_settings])
    for flag, value in option_pairs:
        parts.extend([flag, value])
    return " ".join(shlex.quote(part) for part in parts)


async def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(args)
    print("Indoor preflight")
    print(f"- Configuration: policy={cfg.policy.backend}, match={cfg.goal.match_method}, obstacle={cfg.obstacle.method}")
    print(f"- SDK URL: {cfg.sdk.base_url}")
    print(f"- SLAM: {'disabled' if not cfg.slam.enabled else f'{cfg.slam.backend} ({cfg.slam.mode}) @ {cfg.slam.endpoint}'}")

    checks: List[Check] = []

    goal_check, _, probe_image = _check_goals(args.goals)
    checks.append(goal_check)

    if goal_check.fatal:
        for check in checks:
            _print_check(check)
        print("")
        print("Fix goal image inputs first.")
        return 1

    if not args.skip_sdk:
        checks.extend(await _probe_sdk(cfg.sdk.base_url, cfg.sdk.request_timeout, args.probe_control))
    else:
        checks.append(Check("SDK", "WARN", "skipped by --skip-sdk"))

    checks.append(_check_matcher(cfg, probe_image, args.probe_model_load))
    checks.append(_check_obstacles(cfg, probe_image, args.probe_model_load))
    checks.extend(await _check_policy(cfg, args.probe_model_load))
    checks.extend(await _check_slam(cfg, args.probe_model_load))
    if cfg.slam.enabled and cfg.topo_memory.enabled:
        checks.append(Check("Topological memory", "WARN", "enabled in config, but runtime suppresses it when SLAM is active"))
    else:
        checks.append(Check("Topological memory", "OK", "enabled" if cfg.topo_memory.enabled else "disabled"))
    checks.append(Check("Recovery", "OK", "enabled" if cfg.recovery.enabled else "disabled"))

    fatal = any(check.fatal for check in checks if check.status == "FAIL")
    warn = any(check.status == "WARN" for check in checks)

    for check in checks:
        _print_check(check)

    print("")
    if fatal:
        print("Overall: NOT READY")
        return 1

    if warn:
        print("Overall: READY WITH WARNINGS")
    else:
        print("Overall: READY")

    print("Run command:")
    print(f"  {_render_run_command(args)}")
    if not args.probe_model_load:
        print("Optional deeper validation:")
        print("  python indoor_nav/check_indoor.py --goals ... --probe-model-load")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
