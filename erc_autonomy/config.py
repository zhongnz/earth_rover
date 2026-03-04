from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ERCConfig:
    """Runtime configuration for the autonomous GPS mission runner."""

    base_url: str = "http://127.0.0.1:8000"

    # Control/runtime loop
    loop_hz: float = 10.0
    log_every_n_ticks: int = 20

    # HTTP and freshness
    request_timeout_s: float = 3.0
    stale_sensor_ms: int = 1200
    min_command_hz: float = 10.0

    # Mission lifecycle
    start_mission_on_boot: bool = False
    end_mission_on_shutdown: bool = False

    # Safety stop behavior
    stop_duration_s: float = 0.8
    stop_hz: float = 20.0

    # Estimator filtering
    position_alpha: float = 0.35
    yaw_alpha: float = 0.25
    speed_alpha: float = 0.4
    max_gps_jump_m: float = 20.0

    # Motion safety
    enable_motion: bool = False
    max_linear: float = 0.25
    max_angular: float = 0.4
    reactive_turn_gain: float = 0.8
    traversability_confidence_floor: float = 0.25

    # Traversability + BEV (Week 3)
    traversability_backend: str = "simple_edge"
    bev_width_m: float = 4.0
    bev_depth_m: float = 6.0
    bev_resolution_m: float = 0.1

    # Week 4 planner (candidate rollouts + path fusion)
    planner_num_curvatures: int = 11
    planner_max_curvature: float = 0.9
    planner_horizon_m: float = 2.8
    planner_num_points: int = 20
    planner_fuse_top_k: int = 3
    planner_min_trav_for_motion: float = 0.2
    planner_score_mean_w: float = 0.65
    planner_score_min_w: float = 0.25
    planner_score_goal_w: float = 0.12
    planner_score_curvature_penalty_w: float = 0.08

    # Week 5 checkpoint + goal bearing integration
    auto_checkpoint_report: bool = True
    checkpoint_refresh_interval_s: float = 6.0
    checkpoint_attempt_distance_m: float = 16.0
    checkpoint_attempt_interval_s: float = 2.0
    checkpoint_slowdown_start_m: float = 28.0
    checkpoint_slowdown_hard_m: float = 8.0
    checkpoint_slowdown_min_factor: float = 0.45
    checkpoint_angular_min_factor: float = 0.55
    checkpoint_failure_effect_s: float = 12.0
    checkpoint_failure_buffer_m: float = 6.0
    checkpoint_failure_min_factor: float = 0.3
    checkpoint_failure_angular_min_factor: float = 0.45

    # Week 6 recovery behavior
    recovery_enabled: bool = True
    recovery_stuck_timeout_s: float = 4.0
    recovery_min_speed_mps: float = 0.03
    recovery_min_cmd: float = 0.08
    recovery_backtrack_s: float = 1.2
    recovery_backtrack_linear: float = -0.18
    recovery_rotate_s: float = 1.6
    recovery_rotate_angular: float = 0.32
    recovery_pause_s: float = 0.6
    recovery_cooldown_s: float = 2.0
    recovery_trav_conf_floor: float = 0.35

    # Default command values
    default_linear: float = 0.0
    default_angular: float = 0.0
    default_lamp: int = 0
