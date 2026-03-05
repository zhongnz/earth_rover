"""
Configuration for the Indoor Navigation Agent — SOTA 2025.

All tunable parameters in one place. Override via YAML or CLI.

SOTA defaults (2025):
  - Goal matching: DINOv2-VLAD with registers (AnyLoc-style VPR)
    Refs: Oquab et al. arXiv:2304.07193, Darcet et al. arXiv:2309.16588,
          Keetha et al. arXiv:2308.00688
  - Optional toggle: DINOv3-VLAD (released 2025, arXiv:2508.10104)
  - VLM: Qwen2.5-VL 7B (best open VLM, native multi-image)
    Ref: Bai et al. arXiv:2502.13923
  - Depth: Depth Anything V2 Base (fast + accurate relative depth)
    Ref: Yang et al. arXiv:2406.09414 (NeurIPS 2024)
  - Policy: VLM-hybrid (semantic reasoning) or VLA (direct actions)
    Refs: NoMaD arXiv:2310.07896, OpenVLA arXiv:2406.09246,
          Mobility VLA arXiv:2407.07775

Full technical specs: see SPECIFICATIONS.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# SDK / connectivity
# ---------------------------------------------------------------------------
@dataclass
class SDKConfig:
    base_url: str = "http://127.0.0.1:8000"
    request_timeout: float = 3.0          # seconds per HTTP call
    frame_endpoint: str = "/v2/front"     # primary camera
    rear_endpoint: str = "/v2/rear"       # rear camera (zero bots only)
    data_endpoint: str = "/data"
    control_endpoint: str = "/control"
    checkpoint_endpoint: str = "/checkpoint-reached"
    checkpoints_list_endpoint: str = "/checkpoints-list"
    start_mission_endpoint: str = "/start-mission"
    end_mission_endpoint: str = "/end-mission"


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
@dataclass
class ControlConfig:
    loop_hz: float = 10.0                 # main control loop frequency
    max_linear: float = 0.6              # max forward/backward speed [-1, 1]
    max_angular: float = 0.6             # max turning speed [-1, 1]
    smoothing_alpha: float = 0.4         # exponential smoothing on commands
    stop_duration: float = 1.0           # seconds of stop commands after halt
    stop_hz: float = 20.0               # frequency of stop commands
    stuck_timeout: float = 8.0           # seconds before declaring "stuck"
    stuck_linear_thresh: float = 0.05    # if speed stays below this → stuck


# ---------------------------------------------------------------------------
# Navigation policy
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    # Which policy backend to use
    # Options: "nomad", "vint", "vlm_hybrid", "vla", "heuristic"
    backend: str = "vlm_hybrid"           # CHANGED: VLM-hybrid is now default

    # Model paths / endpoints
    model_path: str = "models/nomad/nomad_indoor.pt"
    device: str = "cuda"                  # "cuda" or "cpu"

    # NoMaD / ViNT specific
    context_length: int = 5               # number of past frames for context
    image_size: tuple = (160, 120)        # (W, H) resize for policy input
    action_horizon: int = 8               # predict N future waypoints
    waypoint_spacing: float = 0.25        # meters between waypoints

    # Goal conditioning
    goal_image_size: tuple = (160, 120)   # resize for goal image input

    # VLM hybrid settings (SOTA: Qwen2.5-VL 7B)
    vlm_endpoint: str = ""                # e.g. "http://localhost:8000/v1" for vLLM
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # CHANGED: from llava:13b
    vlm_query_interval: float = 2.5       # seconds between VLM queries
    vlm_api_format: str = "openai"        # "openai", "ollama", "anthropic"
    vlm_api_key: str = ""                 # API key if needed

    # VLA settings
    vla_backend: str = "openvla"          # "openvla", "octo", "heuristic_plus"


# ---------------------------------------------------------------------------
# Goal / checkpoint management
# ---------------------------------------------------------------------------
@dataclass
class GoalConfig:
    # Image-goal matching (SOTA: DINOv2-VLAD / DINOv3-VLAD for visual place recognition)
    match_method: str = "dinov2_vlad"     # CHANGED: "dinov2_vlad", "dinov3_vlad", "siglip2", "dinov2",
                                          # "eigenplaces", "clip", "sift"
    match_threshold: float = 0.78         # cosine similarity to declare "arrived"
    approach_threshold: float = 0.60      # start slowing down
    match_patience: int = 3               # consecutive frames above threshold → done
    goal_image_dir: str = "indoor_nav/goals"  # where goal images are stored

    # Feature extraction — DINOv2 with registers (smoother patch tokens for VLAD)
    # See: "Vision Transformers Need Registers" (Darcet et al., 2024)
    feature_model: str = "facebook/dinov2-with-registers-base"  # UPGRADED: reg4 variant
    # Alternatives: "facebook/dinov2-base", "google/siglip2-base-patch16-224"
    feature_device: str = "cuda"
    feature_image_size: tuple = (224, 224)

    # VLAD aggregation (for dinov2_vlad / dinov3_vlad methods)
    vlad_clusters: int = 32              # number of VLAD cluster centers


# ---------------------------------------------------------------------------
# Obstacle avoidance
# ---------------------------------------------------------------------------
@dataclass
class ObstacleConfig:
    enabled: bool = True
    method: str = "depth_anything"        # "depth_anything", "depth_pro", "simple_edge"
    depth_model: str = "depth-anything/Depth-Anything-V2-Base-hf"  # CHANGED: upgraded to Base
    depth_device: str = "cuda"
    min_clearance_frac: float = 0.15      # fraction of image height for near-field
    obstacle_slowdown: float = 0.3        # reduce speed when obstacles detected
    emergency_stop_frac: float = 0.25     # fraction of near-field occupied → stop
    check_hz: float = 5.0                # obstacle check frequency


# ---------------------------------------------------------------------------
# Topological memory
# ---------------------------------------------------------------------------
@dataclass
class TopoMemoryConfig:
    enabled: bool = True                  # build topological map during navigation
    min_node_distance: float = 2.0        # seconds between new nodes
    scene_change_threshold: float = 0.25  # visual difference to force new node
    max_nodes: int = 500                  # maximum graph size
    loop_closure_threshold: float = 0.85  # similarity for loop closure
    loop_closure_min_gap: int = 5         # minimum node gap for loop closure
    feature_method: str = "histogram"     # "histogram" (fast), "dinov2" (accurate)
    use_for_recovery: bool = True         # use graph for backtracking during recovery


# ---------------------------------------------------------------------------
# Recovery behaviors
# ---------------------------------------------------------------------------
@dataclass
class RecoveryConfig:
    enabled: bool = True
    max_retries: int = 3
    # Behavior sequence when stuck
    behaviors: List[str] = field(default_factory=lambda: [
        "back_up",           # reverse for 1-2 seconds
        "random_turn",       # turn random direction
        "wall_follow",       # follow nearest wall
        "full_rotation",     # 360° scan to find goal
    ])
    backup_duration: float = 1.5          # seconds to reverse
    backup_speed: float = -0.4
    turn_duration: float = 1.0            # seconds for random turn
    turn_speed: float = 0.5
    wall_follow_duration: float = 3.0
    rotation_speed: float = 0.4
    rotation_duration: float = 6.0       # seconds for full 360°


# ---------------------------------------------------------------------------
# Logging / telemetry
# ---------------------------------------------------------------------------
@dataclass
class LogConfig:
    enabled: bool = True
    log_dir: str = "indoor_nav/logs"
    log_hz: float = 5.0                  # telemetry logging rate
    save_frames: bool = True
    save_goal_matches: bool = True       # save goal similarity over time
    h5_compression: int = 4


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class IndoorNavConfig:
    sdk: SDKConfig = field(default_factory=SDKConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    topo_memory: TopoMemoryConfig = field(default_factory=TopoMemoryConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # Mission
    mission_slug: str = ""
    bot_type: str = "mini"                # "mini" or "zero"
    use_rear_camera: bool = False         # auto-set for zero bots
