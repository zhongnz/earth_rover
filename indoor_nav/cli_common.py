from __future__ import annotations

import argparse
import os
import sys

from indoor_nav.configs.config import IndoorNavConfig


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    require_goals: bool = True,
    include_verbose: bool = True,
) -> argparse.ArgumentParser:
    """Add the shared indoor CLI surface used by run and preflight commands."""
    parser.add_argument(
        "--goals",
        nargs="+",
        required=require_goals,
        help="Goal image files or a single directory containing them (in order)",
    )

    default_sdk_url = os.getenv("INDOOR_SDK_URL") or os.getenv("SDK_BASE_URL") or "http://127.0.0.1:8000"
    default_vlm_endpoint = os.getenv("VLM_ENDPOINT", "")
    default_vlm_model = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    default_vlm_api_key = os.getenv("VLM_API_KEY", "")
    default_slam_backend = os.getenv("SLAM_BACKEND", "off")
    default_slam_mode = os.getenv("SLAM_MODE", "mono")
    default_slam_endpoint = os.getenv("SLAM_ENDPOINT", "http://127.0.0.1:8765")
    default_slam_vocab = os.getenv("SLAM_VOCAB_PATH", "external/ORB_SLAM3/Vocabulary/ORBvoc.txt")
    default_slam_settings = os.getenv("SLAM_SETTINGS_PATH", "indoor_nav/slam/calib/front_camera.yaml")

    # SDK
    parser.add_argument("--url", default=default_sdk_url, help="SDK base URL")
    parser.add_argument("--mission-slug", default="", help="Mission slug (if running a scored mission)")
    parser.add_argument(
        "--slam-backend",
        choices=["off", "orbslam3"],
        default=default_slam_backend,
        help="Optional SLAM localization backend (default: off)",
    )
    parser.add_argument(
        "--slam-mode",
        choices=["mono", "mono_inertial"],
        default=default_slam_mode,
        help="SLAM tracking mode (default: mono)",
    )
    parser.add_argument("--slam-endpoint", default=default_slam_endpoint, help="SLAM sidecar base URL")
    parser.add_argument("--slam-vocab", default=default_slam_vocab, help="Path to ORB-SLAM3 vocabulary file")
    parser.add_argument("--slam-settings", default=default_slam_settings, help="Path to SLAM camera settings YAML")
    parser.add_argument("--no-slam", action="store_true", help="Disable SLAM even if configured")

    # Policy
    parser.add_argument(
        "--policy",
        choices=["nomad", "vint", "gnm", "vlm_hybrid", "vla", "heuristic", "maze_search"],
        default="vlm_hybrid",
        help="Navigation policy backend (default: vlm_hybrid)",
    )
    parser.add_argument("--model-path", default="", help="Path to policy model weights")
    parser.add_argument("--device", default="cuda", help="Compute device (cuda/cpu)")
    parser.add_argument(
        "--nomad-repo-root",
        default="",
        help="Optional path to a visualnav-transformer checkout for official GNM-family .pth checkpoints",
    )
    parser.add_argument(
        "--nomad-config-path",
        default="",
        help="Optional path to official GNM-family config YAML (defaults to <repo>/train/config/<backend>.yaml)",
    )
    parser.add_argument(
        "--nomad-samples",
        type=int,
        default=1,
        help="Diffusion samples per tick for official NoMaD checkpoints (default: 1)",
    )

    # VLM options
    parser.add_argument(
        "--vlm-endpoint",
        default=default_vlm_endpoint,
        help="VLM API endpoint (e.g., http://127.0.0.1:8001/v1)",
    )
    parser.add_argument("--vlm-model", default=default_vlm_model, help="VLM model name")
    parser.add_argument("--vlm-api-key", default=default_vlm_api_key, help="API key for VLM endpoint")
    parser.add_argument(
        "--maze-burst-seconds",
        type=float,
        default=0.45,
        help="Forward burst duration for maze_search policy.",
    )
    parser.add_argument(
        "--maze-pause-seconds",
        type=float,
        default=0.25,
        help="Pause duration between bursts for maze_search policy.",
    )
    parser.add_argument(
        "--maze-scan-leg-seconds",
        type=float,
        default=1.0,
        help="One-way scan sweep duration for maze_search policy.",
    )
    parser.add_argument(
        "--maze-backtrack-turn-seconds",
        type=float,
        default=1.7,
        help="Turn-around duration used when maze_search backtracks.",
    )
    parser.add_argument(
        "--maze-open-clearance",
        type=float,
        default=0.58,
        help="Clearance threshold for treating a side or forward lane as open in maze_search.",
    )
    parser.add_argument(
        "--maze-dead-end-clearance",
        type=float,
        default=0.24,
        help="Center-clearance threshold for treating the current view as a dead end in maze_search.",
    )
    parser.add_argument(
        "--maze-goal-rescan-interval",
        type=float,
        default=2.0,
        help="Minimum seconds between repeat scans at the same node-like location in maze_search.",
    )
    parser.add_argument(
        "--maze-turn-bin-threshold",
        type=float,
        default=0.28,
        help="Offset threshold that separates left/straight/right scan bins in maze_search.",
    )
    parser.add_argument(
        "--maze-centering-gain",
        type=float,
        default=0.85,
        help="Corridor-centering gain from left/right clearances in maze_search.",
    )
    parser.add_argument(
        "--maze-blocked-turn-gain",
        type=float,
        default=1.15,
        help="Extra steering gain when the center lane is partially blocked in maze_search.",
    )
    parser.add_argument(
        "--maze-centering-deadband",
        type=float,
        default=0.04,
        help="Ignore tiny left/right clearance differences below this threshold in maze_search.",
    )

    # Goal matching
    parser.add_argument(
        "--match-method",
        choices=["dinov2_direct", "wall_crop_direct", "wall_rectify_direct", "dinov2_vlad", "dinov3_vlad", "siglip2", "dinov2", "eigenplaces", "cosplace", "clip", "sift"],
        default="dinov2_direct",
        help="Image goal matching method (default: dinov2_direct)",
    )
    parser.add_argument("--match-threshold", type=float, default=0.78, help="Similarity threshold for arrival")
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
        help="Minimum contour fill ratio for wall-aware proposals.",
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
        help="Maximum number of wall-aware proposals per image.",
    )
    parser.add_argument(
        "--wall-crop-score-weight",
        type=float,
        default=0.9,
        help="How strongly wall-aware matchers prefer region matches over full-scene matches.",
    )

    # Control
    parser.add_argument("--max-speed", type=float, default=0.6, help="Max linear speed [0-1]")
    parser.add_argument("--loop-hz", type=float, default=10.0, help="Control loop frequency")

    # Obstacle avoidance
    parser.add_argument("--no-obstacle", action="store_true", help="Disable obstacle avoidance")
    parser.add_argument(
        "--obstacle-method",
        choices=["depth_anything", "depth_pro", "simple_edge"],
        default="depth_anything",
        help="Obstacle detection method",
    )

    # Topological memory
    parser.add_argument("--no-topo", action="store_true", help="Disable topological memory")
    parser.add_argument("--topo-max-nodes", type=int, default=500, help="Max topo graph nodes")

    # Recovery
    parser.add_argument("--no-recovery", action="store_true", help="Disable recovery behaviors")

    # Logging
    parser.add_argument("--no-log", action="store_true", help="Disable HDF5 logging")
    parser.add_argument("--log-dir", default="indoor_nav/logs", help="Log output directory")

    if include_verbose:
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser


def capture_cli_flags(argv: list[str] | None = None) -> set[str]:
    """Return the set of option flags explicitly present on the CLI."""
    if argv is None:
        argv = sys.argv[1:]
    return {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    }


def apply_policy_presets(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply low-dependency defaults for specific policy modes unless the user
    explicitly overrode those knobs on the command line.
    """
    cli_flags = getattr(args, "_cli_flags", set())
    presets: list[str] = []

    if args.policy == "heuristic":
        if "--match-method" not in cli_flags:
            args.match_method = "sift"
            presets.append("match_method=sift")
        if "--obstacle-method" not in cli_flags and "--no-obstacle" not in cli_flags:
            args.obstacle_method = "simple_edge"
            presets.append("obstacle_method=simple_edge")
        if "--device" not in cli_flags:
            args.device = "cpu"
            presets.append("device=cpu")

    if args.policy == "maze_search":
        if "--match-method" not in cli_flags:
            args.match_method = "wall_rectify_direct"
            presets.append("match_method=wall_rectify_direct")
        if "--obstacle-method" not in cli_flags and "--no-obstacle" not in cli_flags:
            args.obstacle_method = "simple_edge"
            presets.append("obstacle_method=simple_edge")
        if "--max-speed" not in cli_flags:
            args.max_speed = 0.18
            presets.append("max_speed=0.18")
        if "--loop-hz" not in cli_flags:
            args.loop_hz = 6.0
            presets.append("loop_hz=6")

    args._applied_presets = presets
    return args


def build_config(args: argparse.Namespace) -> IndoorNavConfig:
    """Build IndoorNavConfig from CLI arguments."""
    args = apply_policy_presets(args)
    cfg = IndoorNavConfig()

    # SDK
    cfg.sdk.base_url = args.url
    cfg.mission_slug = args.mission_slug
    cfg.slam.enabled = args.slam_backend != "off" and not args.no_slam
    cfg.slam.backend = "orbslam3" if args.slam_backend == "off" else args.slam_backend
    cfg.slam.mode = args.slam_mode
    cfg.slam.endpoint = args.slam_endpoint
    cfg.slam.vocab_path = args.slam_vocab
    cfg.slam.settings_path = args.slam_settings

    # Policy
    cfg.policy.backend = args.policy
    cfg.policy.device = args.device
    if args.model_path:
        cfg.policy.model_path = args.model_path
    if args.nomad_repo_root:
        cfg.policy.nomad_repo_root = args.nomad_repo_root
    if args.nomad_config_path:
        cfg.policy.nomad_config_path = args.nomad_config_path
    cfg.policy.nomad_num_samples = max(1, int(args.nomad_samples))
    if args.vlm_endpoint:
        cfg.policy.vlm_endpoint = args.vlm_endpoint
    cfg.policy.vlm_model = args.vlm_model
    if args.vlm_api_key:
        cfg.policy.vlm_api_key = args.vlm_api_key
    cfg.policy.maze_burst_seconds = max(0.1, float(args.maze_burst_seconds))
    cfg.policy.maze_pause_seconds = max(0.0, float(args.maze_pause_seconds))
    cfg.policy.maze_scan_leg_seconds = max(0.1, float(args.maze_scan_leg_seconds))
    cfg.policy.maze_backtrack_turn_seconds = max(0.1, float(args.maze_backtrack_turn_seconds))
    cfg.policy.maze_open_clearance = float(args.maze_open_clearance)
    cfg.policy.maze_dead_end_clearance = float(args.maze_dead_end_clearance)
    cfg.policy.maze_goal_rescan_interval = max(0.0, float(args.maze_goal_rescan_interval))
    cfg.policy.maze_turn_bin_threshold = max(0.01, float(args.maze_turn_bin_threshold))
    cfg.policy.maze_centering_gain = max(0.0, float(args.maze_centering_gain))
    cfg.policy.maze_blocked_turn_gain = max(0.0, float(args.maze_blocked_turn_gain))
    cfg.policy.maze_centering_deadband = max(0.0, float(args.maze_centering_deadband))

    # Goal matching
    cfg.goal.match_method = args.match_method
    cfg.goal.match_threshold = args.match_threshold
    cfg.goal.feature_device = args.device
    cfg.goal.wall_crop_min_area_frac = args.wall_crop_min_area_frac
    cfg.goal.wall_crop_max_area_frac = args.wall_crop_max_area_frac
    cfg.goal.wall_crop_max_aspect_ratio = args.wall_crop_max_aspect_ratio
    cfg.goal.wall_crop_min_fill_ratio = args.wall_crop_min_fill_ratio
    cfg.goal.wall_crop_padding_frac = args.wall_crop_padding_frac
    cfg.goal.wall_crop_max_candidates = args.wall_crop_max_candidates
    cfg.goal.wall_crop_score_weight = args.wall_crop_score_weight

    # Auto-select feature model based on match method
    if args.match_method == "siglip2":
        cfg.goal.feature_model = "google/siglip2-base-patch16-224"
    elif args.match_method in ("dinov2_vlad", "dinov2_direct", "wall_crop_direct", "wall_rectify_direct"):
        cfg.goal.feature_model = "facebook/dinov2-with-registers-base"
    elif args.match_method == "dinov3_vlad":
        cfg.goal.feature_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    elif args.match_method == "dinov2":
        cfg.goal.feature_model = "facebook/dinov2-base"
    elif args.match_method == "cosplace":
        cfg.goal.feature_model = (
            f"gmberton/cosplace ({cfg.goal.cosplace_backbone}, "
            f"dim={cfg.goal.cosplace_fc_output_dim})"
        )

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
