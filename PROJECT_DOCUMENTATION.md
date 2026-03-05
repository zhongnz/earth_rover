# NYU Earth Rover Project Documentation

This document is the canonical technical briefing for teammates working in this repository.
It consolidates architecture, tools, design rationale, and research references used in the codebase.

## 1) Project Overview

This repository implements an end-to-end Earth Rover software stack for:

- Remote robot operation through the FrodoBots SDK pattern.
- Data collection, replay, and offline analysis.
- Competition-oriented autonomous navigation in two modes:
  - GPS/checkpoint missions (`erc_autonomy`).
  - Indoor image-goal missions (`indoor_nav`).

At the center of the system is a local FastAPI bridge that standardizes robot control and sensing as HTTP endpoints for all higher-level modules.

## 2) Repository Structure

Top-level components:

- `main.py`: FastAPI SDK bridge, mission APIs, intervention APIs, and local UI routes.
- `browser_service.py`: Playwright runtime adapter for stream join, telemetry reads, frame extraction, and control message injection.
- `rtm_client.py`: legacy direct RTM sender.
- `examples/`: teleop, logging, replay, and analysis tools.
- `erc_autonomy/`: modular GPS autonomy stack for mission checkpoints.
- `indoor_nav/`: image-goal autonomy stack with VPR/VLM/policy backends.
- `static/`, `index.html`: web runtime assets used by the SDK bridge.
- `Dockerfile`, `docker-compose.yml`: containerized runtime.

## 3) System Architecture

### 3.1 Data and Control Path

1. A client (script, autonomy runner, or external app) calls local SDK endpoints at `http://127.0.0.1:8000`.
2. `main.py` validates mission state and routes requests.
3. `browser_service.py` drives a browser session to `/sdk`, joins the stream, and reads:
   - telemetry from `window.rtm_data`
   - base64 camera frames via JS helper functions
4. `main.py` proxies mission/checkpoint/intervention calls to FrodoBots cloud APIs.
5. Autonomy modules consume local endpoints (`/data`, `/v2/front`, `/v2/screenshot`) and send commands to `/control`.

### 3.2 Core Design Decision

The autonomy layers never talk directly to the cloud/video stack. They only use local HTTP contracts. This keeps policy code isolated from stream/browser details and supports faster iteration.

## 4) Core Subsystems

### 4.1 FastAPI SDK Bridge (`main.py`)

Responsibilities:

- Token/auth lifecycle (`/start-mission`, `/end-mission` and SDK token retrieval).
- Mission checkpoint APIs (`/checkpoints-list`, `/checkpoint-reached`, `/missions-history`).
- Intervention APIs (`/interventions/start`, `/interventions/end`, `/interventions/history`).
- Robot control and sensing (`/control`, `/data`, `/v2/front`, `/v2/rear`, `/v2/screenshot`).
- UI hosting (`/`, `/sdk`).

Runtime hardening in current implementation:

- External HTTP calls are executed off the event loop (`asyncio.to_thread`) to avoid blocking async request handling.
- Telemetry-null safeguards return clear HTTP errors (e.g., `503`) instead of internal crashes when stream data is temporarily unavailable.

### 4.2 Browser Runtime Adapter (`browser_service.py`)

Responsibilities:

- Launches Playwright browser engine and joins the SDK UI stream.
- Configures image quality/format for frame extraction.
- Reads live telemetry and front/rear frames.
- Sends control commands into `window.sendMessage(...)`.

Configurable runtime options:

- `BROWSER_ENGINE` (`webkit`, `chromium`, `firefox`)
- `BROWSER_HEADLESS` (`true`/`false`)
- `SDK_BASE_URL` (defaults to `http://127.0.0.1:8000`)
- `CHROME_EXECUTABLE_PATH` (used only for Chromium mode)

### 4.3 Baseline Tooling (`examples/`)

Primary capabilities:

- `exploration.py`: keyboard drive plus logging.
- `navigation.py`: target-image match plus logged-control replay.
- `utils/data_logger.py`: HDF5 logging for telemetry, controls, IMU, RPM, and frames.
- `utils/analyze_log.py`: offline plots and quick diagnostics.
- `utils/export_images.py`: frame export from HDF5.
- `utils/image_match.py`: ORB/SIFT retrieval utilities.
- `utils/extract_controls.py`: control extraction by timestamp/frame index.

## 5) Autonomy Stacks

### 5.1 GPS Autonomy (`erc_autonomy`)

Entry point:

```bash
python -m erc_autonomy.run_gps --url http://127.0.0.1:8000
```

Architecture:

- Async IO wrapper (`sdk_io.py`)
- Mission FSM (`mission_fsm.py`)
- Stale-sensor watchdog (`watchdog.py`)
- Filtered state estimator (`state_estimator.py`)
- Traversability inference (`traversability.py`)
- BEV mapper (`bev_mapper.py`)
- Candidate rollout + path fusion planner (`planner.py`)
- Goal/checkpoint manager (`goal_manager.py`)
- Recovery manager (`recovery.py`)
- Orchestrator (`mission_runner.py`)

Safety and control policy highlights:

- Motion disabled by default (`--enable-motion` required).
- Watchdog-triggered safe-stop on stale sensing.
- Checkpoint-aware speed taper and angular taper.
- Failure-aware slowdown after rejected `/checkpoint-reached` attempts.

Current limitation:

- `sam2` traversability backend is a placeholder interface and currently falls back to the built-in `simple_edge` backend.

### 5.2 Indoor Image-Goal Autonomy (`indoor_nav`)

Entry point:

```bash
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid
```

Architecture:

- Async orchestrator (`agent.py`) at ~10 Hz.
- SDK client (`modules/sdk_client.py`).
- Goal matcher (`modules/checkpoint_manager.py`):
  - `dinov2_vlad` (default)
  - `dinov3_vlad` (A/B toggle)
  - `siglip2`, `dinov2`, `eigenplaces`, `clip`, `sift`
- Obstacle module (`modules/obstacle_avoidance.py`).
- Topological memory (`modules/topological_memory.py`).
- Recovery manager (`modules/recovery.py`).
- Policy backends (`policies/`):
  - `vlm_hybrid`
  - `vla`
  - `nomad` / `vint`
  - `heuristic`

## 6) API Surface (Local SDK)

Control/sensing:

- `POST /control`
- `GET /data`
- `GET /v2/front`
- `GET /v2/rear`
- `GET /v2/screenshot`

Mission:

- `POST /start-mission`
- `GET|POST /checkpoints-list`
- `POST /checkpoint-reached`
- `POST /end-mission`
- `GET /missions-history`

Interventions:

- `POST /interventions/start`
- `POST /interventions/end`
- `GET /interventions/history`

UI/spec:

- `GET /`
- `GET /sdk`
- `GET /docs`
- `GET /openapi.json`

Control payload:

```json
{
  "command": {
    "linear": 0.0,
    "angular": 0.0,
    "lamp": 0
  }
}
```

## 7) Configuration and Environment

Use `.env.sample` as the source-of-truth template.

Core variables:

- `SDK_API_TOKEN`
- `BOT_SLUG`
- `MISSION_SLUG` (required for scored mission workflows)
- `MAP_ZOOM_LEVEL`
- `IMAGE_QUALITY`
- `IMAGE_FORMAT`
- `DEBUG`

Browser runtime variables:

- `BROWSER_ENGINE`
- `BROWSER_HEADLESS`
- `SDK_BASE_URL`
- `CHROME_EXECUTABLE_PATH` (Chromium-only override)

## 8) Tooling and Dependencies

Base runtime (`requirements.txt`):

- FastAPI, Hypercorn, Pydantic
- Playwright, python-dotenv
- requests, aiohttp
- OpenCV, NumPy
- pynput, h5py, matplotlib

Indoor extras (`indoor_nav/requirements_indoor.txt`):

- torch, torchvision, transformers
- Pillow, scipy
- optional serving/inference integrations documented inline (`vLLM`, `openai`, `ollama`)

## 9) Design Rationale and Tradeoffs

### 9.1 Why modular instead of single end-to-end policy?

- Better observability and debugging under real network latency/jitter.
- Easier safety interlocks (watchdog stop, explicit recovery, mission guards).
- Faster A/B iteration on individual modules (matcher, planner, policy, recovery).

### 9.2 Why DINOv2-VLAD default, DINOv3 as toggle?

- `dinov2_vlad` is the most stable baseline in this codebase for indoor VPR thresholds and behavior tuning.
- `dinov3_vlad` is included for controlled A/B evaluation and future migration.

### 9.3 Why Topological Memory?

- Provides long-horizon structure and backtracking without requiring metric-SLAM robustness in challenging monocular indoor conditions.

### 9.4 Why depth-based obstacle gating?

- Lightweight monocular obstacle cues improve safety and command stability with low implementation overhead.

### 9.5 Why checkpoint tapering in GPS missions?

- Reduces overshoot and repeated failed checkpoint reports by decelerating and moderating turn aggressiveness near checkpoint radius.

## 10) Academic References Used by the Project

Primary reference list is maintained in `indoor_nav/SPECIFICATIONS.md`.
Core references used to justify architecture and model choices:

1. Oquab et al., **DINOv2**, arXiv:2304.07193 (2023)
2. Darcet et al., **Vision Transformers Need Registers**, arXiv:2309.16588 (ICLR 2024)
3. Keetha et al., **AnyLoc**, arXiv:2308.00688 (RA-L 2023)
4. Tschannen et al., **SigLIP 2**, arXiv:2502.14786 (2025)
5. Berton et al., **EigenPlaces**, arXiv:2308.10832 (ICCV 2023)
6. Bai et al., **Qwen2.5-VL**, arXiv:2502.13923 (2025)
7. Yang et al., **Depth Anything V2**, arXiv:2406.09414 (NeurIPS 2024)
8. Bochkovskii et al., **Depth Pro**, arXiv:2410.02073 (ICLR 2025)
9. Sridhar et al., **NoMaD**, arXiv:2310.07896 (ICRA 2024)
10. Shah et al., **ViNT**, arXiv:2306.14846 (CoRL 2023)
11. Kim et al., **OpenVLA**, arXiv:2406.09246 (2024)
12. Chiang et al., **Mobility VLA**, arXiv:2407.07775 (2024)
13. Open X-Embodiment Collaboration, **Open X-Embodiment**, arXiv:2310.08864 (ICRA 2024)

## 11) Setup, Validation, and Operations

### 11.1 Local Setup

```bash
conda create -n erv python=3.11
conda activate erv
pip install -r requirements.txt
cp .env.sample .env
hypercorn main:app --reload
```

Then open `http://localhost:8000` and click Join.

### 11.2 Optional Indoor Setup

```bash
pip install -r indoor_nav/requirements_indoor.txt
python indoor_nav/test_integration.py --skip-sdk
```

### 11.3 Quick Runtime Checks

- `curl http://127.0.0.1:8000/openapi.json`
- `curl http://127.0.0.1:8000/data`
- `curl http://127.0.0.1:8000/v2/front`
- `curl -X POST http://127.0.0.1:8000/control -H 'Content-Type: application/json' -d '{"command":{"linear":0,"angular":0}}'`

## 12) Known Limitations and Next Steps

Current limitations:

- Stream availability governs endpoint freshness (`/data`, `/v2/*`).
- GPS traversability backend is currently heuristic-first (`simple_edge`) with `sam2` placeholder interface.
- Indoor stack requires heavyweight ML dependencies for full policy coverage.

Recommended next steps:

- Integrate a real SAM2 traversability backend in `erc_autonomy/traversability.py`.
- Expand formal tests for mission/intervention edge cases and stream loss.
- Continue DINOv2-vs-DINOv3 A/B evaluation on indoor goals before changing defaults.
