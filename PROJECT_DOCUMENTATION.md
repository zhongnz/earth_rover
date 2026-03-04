# NYU Earth Rover Project Documentation

## 1) Project Purpose

This repository is an Earth Rover software stack for:

- Remote robot operation through the FrodoBots SDK pattern.
- Data collection and replay workflows for robotics experiments.
- Autonomous navigation development for competition tracks:
  - GPS/checkpoint-based urban autonomy (`erc_autonomy`).
  - Image-goal indoor autonomy (`indoor_nav`).

The codebase is built around a local FastAPI bridge that talks to FrodoBots cloud APIs and browser-based video/RTM streams, then exposes a stable local HTTP interface for control, telemetry, and camera frames.

## 2) Repository Layout

Top-level structure:

- `main.py`: FastAPI server exposing SDK-compatible endpoints.
- `browser_service.py`: Playwright-based browser automation for stream join, frame extraction, and message sending.
- `rtm_client.py`: Legacy direct Agora RTM sender.
- `README.md`: Primary SDK and baseline usage documentation.
- `examples/`: Teleop, logging, replay, and utility scripts.
- `erc_autonomy/`: GPS mission autonomy scaffold (Week 1-6 implementation).
- `indoor_nav/`: Indoor image-goal autonomy agent and modules.
- `static/`, `index.html`: Browser UI and SDK web assets.
- `assets/`, `screenshots/`, `logs/`: media and runtime outputs.
- `Dockerfile`, `docker-compose.yml`: containerized deployment.

## 3) High-Level Architecture

Runtime data/control flow:

1. Local client code (scripts or autonomy modules) calls local HTTP endpoints at `http://127.0.0.1:8000`.
2. `main.py` handles endpoint routing and mission state checks.
3. `browser_service.py` drives a browser session to `/sdk`, joins the stream, reads `window.rtm_data`, captures frame buffers, and sends control messages into the web app runtime.
4. `main.py` bridges mission/checkpoint/intervention calls to FrodoBots cloud APIs.
5. Autonomy modules consume `/data` + `/v2/*` and send `/control`.

Key design consequence:

- If browser stream data is unavailable, `/data` and `/v2/*` may be empty or return unavailable-frame errors even while the server process itself is up.

## 4) Core Subsystems

### 4.1 FastAPI SDK Bridge

Primary file: `main.py`.

Main responsibilities:

- Authentication/token acquisition from env vars or FrodoBots API.
- Mission lifecycle control:
  - `/start-mission`
  - `/end-mission`
  - `/checkpoints-list`
  - `/checkpoint-reached`
  - `/missions-history`
- Intervention lifecycle:
  - `/interventions/start`
  - `/interventions/end`
  - `/interventions/history`
- Robot I/O endpoints:
  - `/control`
  - `/data`
  - `/screenshot`
  - `/v2/screenshot`
  - `/v2/front`
  - `/v2/rear`
- Serving web UI:
  - `/` (spectator)
  - `/sdk` (controller)

### 4.2 Browser Runtime Adapter

Primary file: `browser_service.py`.

Responsibilities:

- Starts Playwright WebKit browser.
- Navigates to local `/sdk` page.
- Clicks join flow and waits for video elements.
- Reads:
  - `window.rtm_data` for telemetry.
  - front/rear base64 frame helpers for camera output.
- Sends control messages through page JavaScript (`window.sendMessage`).

Important behavior:

- First telemetry/frame request may trigger browser init.
- If stream join fails, endpoints may return `null` or "frame not available".

### 4.3 Baseline Tooling and Data Pipeline

Primary files: `examples/` and `examples/utils/`.

Capabilities:

- `examples/exploration.py`: keyboard driving + background logging.
- `examples/navigation.py`: image-match target and replay logged controls.
- `examples/utils/data_logger.py`: HDF5 recorder for telemetry, IMU, RPMs, controls, and frames.
- `examples/utils/analyze_log.py`: offline plotting and quick diagnostics.
- `examples/utils/export_images.py`: frame extraction from HDF5 logs.
- `examples/utils/image_match.py`: ORB/SIFT matching against logged frames.
- `examples/utils/extract_controls.py`: control extraction up to target frame index.
- `examples/utils/keyboard_control.py`: standalone keyboard teleop loop.

### 4.4 GPS Autonomy Stack (`erc_autonomy`)

Entry point:

- `python -m erc_autonomy.run_gps ...`

Design:

- Async mission loop with watchdog, state estimation, traversability, BEV projection, path-fusion planning, checkpoint logic, and explicit recovery.

Main modules:

- `config.py`: centralized runtime parameters.
- `sdk_io.py`: robust async SDK I/O wrapper.
- `mission_fsm.py`: lifecycle state machine.
- `watchdog.py`: stale-sensor emergency-stop trigger.
- `state_estimator.py`: lightweight filtered local pose from GPS/heading/speed.
- `traversability.py`: simple-edge traversability backend with SAM2 placeholder path.
- `bev_mapper.py`: local BEV approximation from image-space traversability.
- `planner.py`: arc rollout sampling + top-k path fusion.
- `goal_manager.py`: checkpoint parsing, bearing/turn hint generation.
- `recovery.py`: stuck detection and staged backtrack/rotate/pause recovery.
- `mission_runner.py`: orchestrator integrating all components.
- `logging_utils.py`: line-delimited JSON logging format.

Recent control features:

- Checkpoint-distance linear speed taper.
- Failed-checkpoint feedback taper using proximate distance.
- Independent angular taper near checkpoints and after failed reports.

### 4.5 Indoor Image-Goal Autonomy (`indoor_nav`)

Entry point:

- `python indoor_nav/run_indoor.py ...`

Design:

- Goal-conditioned loop for indoor checkpoints using image matching, policy inference, obstacle handling, optional topological memory, and recovery.

Main modules:

- `configs/config.py`: all indoor config dataclasses.
- `agent.py`: async orchestrator and control loop.
- `modules/sdk_client.py`: indoor SDK HTTP client.
- `modules/checkpoint_manager.py`: goal image loading, feature extraction, and similarity tracking.
- `modules/obstacle_avoidance.py`: depth/edge-based obstacle estimation.
- `modules/topological_memory.py`: visual graph, loop closure, A* path support.
- `modules/recovery.py`: stuck-recovery behavior manager.
- `policies/base_policy.py`: policy interface and I/O structures.
- `policies/nomad_policy.py`: NoMaD/heuristic policy wrapper.
- `policies/vlm_hybrid_policy.py`: VLM-guided high-level policy + reactive control.
- `policies/vla_policy.py`: VLA-style policy + heuristic fallback.
- `test_integration.py`: import/config/module and optional SDK sanity checks.

## 5) API Surface (Local Server)

Main endpoint groups:

- Control and telemetry:
  - `POST /control`
  - `GET /data`
  - `GET /screenshot`
  - `GET /v2/screenshot`
  - `GET /v2/front`
  - `GET /v2/rear`
- Mission:
  - `POST /start-mission`
  - `POST|GET /checkpoints-list`
  - `POST /checkpoint-reached`
  - `POST /end-mission`
  - `GET /missions-history`
- Interventions:
  - `POST /interventions/start`
  - `POST /interventions/end`
  - `GET /interventions/history`
- UI:
  - `GET /`
  - `GET /sdk`
  - `GET /docs`
  - `GET /openapi.json`

Control payload shape:

```json
{
  "command": {
    "linear": 0.0,
    "angular": 0.0,
    "lamp": 0
  }
}
```

## 6) Configuration

### 6.1 Environment Variables (Server)

Primary variables:

- `SDK_API_TOKEN`: FrodoBots SDK API token.
- `BOT_SLUG`: target bot slug.
- `CHROME_EXECUTABLE_PATH`: browser path when needed.
- `MAP_ZOOM_LEVEL`: map zoom for UI.
- `MISSION_SLUG`: mission identifier for mission mode.
- `IMAGE_QUALITY`: frame compression quality.
- `IMAGE_FORMAT`: `jpeg|png|webp`.
- `DEBUG`: optional request/response debug logging.

Mission behavior:

- If `MISSION_SLUG` is set, mission endpoints are enforced and `/start-mission` is required for mission-mode workflows.
- Without `MISSION_SLUG`, general SDK usage is still possible for control/stream experimentation.

### 6.2 `erc_autonomy` Runtime Flags

Notable flags in `erc_autonomy/run_gps.py`:

- Core loop/safety: `--loop-hz`, `--request-timeout`, `--stale-ms`.
- Motion: `--enable-motion`, `--max-linear`, `--max-angular`.
- Checkpoint control:
  - `--checkpoint-distance`, `--checkpoint-refresh`
  - `--checkpoint-slowdown-start`, `--checkpoint-slowdown-hard`
  - `--checkpoint-slowdown-min-factor`
  - `--checkpoint-angular-min-factor`
  - `--checkpoint-failure-effect`
  - `--checkpoint-failure-buffer`
  - `--checkpoint-failure-min-factor`
  - `--checkpoint-failure-angular-min-factor`
- Recovery: `--no-recovery`, `--recovery-stuck-timeout`.

### 6.3 `indoor_nav` Runtime Flags

Notable flags in `indoor_nav/run_indoor.py`:

- Required goals: `--goals`.
- Policy backend: `--policy` (`vlm_hybrid`, `vla`, `nomad`, `vint`, `heuristic`).
- Goal matcher: `--match-method` (`dinov2_vlad`, `siglip2`, `dinov2`, `eigenplaces`, `clip`, `sift`).
- VLM options: `--vlm-endpoint`, `--vlm-model`, `--vlm-api-key`.
- Obstacle options: `--obstacle-method`, `--no-obstacle`.
- Memory/recovery toggles: `--no-topo`, `--no-recovery`.
- Mission routing: `--mission-slug`.

## 7) Setup and Execution

### 7.1 Local Development

1. Create env:

```bash
conda create -n erv python=3.11
conda activate erv
pip install -r requirements.txt
```

2. Start server:

```bash
hypercorn main:app --reload
```

3. Open UI and join:

- `http://localhost:8000` (or `/sdk`) and click Join.

### 7.2 Baseline Examples

Exploration logging:

```bash
python examples/exploration.py --url http://127.0.0.1:8000 --rate 10 --out logs/run.h5
```

Navigation replay:

```bash
python examples/navigation.py logs/run.h5 --target assets/axis.jpg --url http://127.0.0.1:8000
```

### 7.3 GPS Autonomy

Safe startup (motion disabled by default):

```bash
python -m erc_autonomy.run_gps --url http://127.0.0.1:8000 --loop-hz 10
```

Enable motion:

```bash
python -m erc_autonomy.run_gps --enable-motion --max-linear 0.2 --max-angular 0.35
```

### 7.4 Indoor Autonomy

Install additional dependencies:

```bash
pip install -r indoor_nav/requirements_indoor.txt
```

Run:

```bash
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid
```

## 8) Testing and Validation

Baseline checks:

- `python -m compileall -q erc_autonomy`
- `python -m erc_autonomy.run_gps --help`
- `python indoor_nav/test_integration.py --skip-sdk`

Live SDK checks:

- `curl http://127.0.0.1:8000/openapi.json`
- `curl http://127.0.0.1:8000/data`
- `curl http://127.0.0.1:8000/v2/front`
- `curl -X POST http://127.0.0.1:8000/control -H 'Content-Type: application/json' -d '{"command":{"linear":0,"angular":0}}'`

## 9) Logging and Artifacts

### 9.1 HDF5 Logs

Generated by `examples/utils/data_logger.py` and indoor logging hooks.

Typical datasets/groups:

- `telemetry`
- `accels`
- `gyros`
- `mags`
- `rpms`
- `controls`
- `front_frames`
- `rear_frames`

### 9.2 `erc_autonomy` JSON Logs

Structured fields include:

- Loop state and latency.
- Traversability and planner outputs.
- Goal/checkpoint status.
- Checkpoint speed/angular taper factors.
- Recovery mode and command outputs.

## 10) Operational Failure Modes and Debugging

Common issues:

- `Missing required environment variables: MISSION_SLUG`:
  - Mission mode requested, but `MISSION_SLUG` not configured.
- `Front frame not available` or repeated stale watchdog stops:
  - Stream/browser join path not initialized or robot stream unavailable.
- `/data` returns `null`:
  - No active telemetry payload in browser runtime.
- `/control` 400 `Command not provided`:
  - Invalid JSON shape; must include `{"command": {...}}`.

Quick triage sequence:

1. Verify server alive: `GET /`.
2. Check API schema: `GET /openapi.json`.
3. Validate stream endpoints: `GET /data`, `GET /v2/front`.
4. Send explicit zero command with valid payload shape.
5. Confirm mission env configuration if running mission APIs.

## 11) Extension Guide

Recommended extension points:

- New traversability backend:
  - Implement in `erc_autonomy/traversability.py`.
  - Keep output as `TraversabilityResult`.
- New local planner:
  - Extend or replace `PathFusionPlanner` in `erc_autonomy/planner.py`.
- Better state estimation:
  - Replace internal filter in `erc_autonomy/state_estimator.py` with EKF/UKF while preserving `StateEstimate`.
- Advanced indoor policies:
  - Add new policy class implementing `BasePolicy` in `indoor_nav/policies/`.
- Improved indoor place recognition:
  - Extend `GoalMatcher` backends in `indoor_nav/modules/checkpoint_manager.py`.

## 12) Project Status Summary

- SDK server and baseline tooling are production-usable for teleop, logging, and replay.
- `erc_autonomy` provides a complete modular autonomy scaffold with safety and checkpoint behaviors.
- `indoor_nav` provides a rich modular framework for image-goal indoor autonomy with multiple policy backends.
- Real-world performance is constrained by live stream availability, mission configuration, and external API conditions.
