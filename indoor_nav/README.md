# Indoor Navigation Agent — ICRA 2025 Earth Rover Challenge

**SOTA 2025 autonomous indoor navigation for the Earth Rover Challenge Indoor Track.**

## Overview

This system remotely controls an Earth Rover Mini/Zero ground robot through
unknown indoor environments, navigating to sequential image-goal checkpoints
using only onboard cameras and IMU.

### Key Components

| Component | Technology | Paper |
|-----------|-----------|-------|
| Goal Matching | DINOv2-VLAD (with registers) | AnyLoc (RA-L 2023, arXiv:2308.00688) + Registers (ICLR 2024, arXiv:2309.16588) |
| VLM Reasoning | Qwen2.5-VL 7B | arXiv:2502.13923 |
| Depth Estimation | Depth Anything V2 Base | NeurIPS 2024, arXiv:2406.09414 |
| Topological Memory | Visual graph + A* | Inspired by NTS (NeurIPS 2020), Mobility VLA (arXiv:2407.07775) |
| Navigation Policy | VLM-Hybrid (default), NoMaD, VLA | NoMaD: arXiv:2310.07896, OpenVLA: arXiv:2406.09246 |

## Quick Start

```bash
# Install
conda create -n erv python=3.11 && conda activate erv
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt -r indoor_nav/requirements_indoor.txt

# Start SDK server
hypercorn main:app --reload

# Run (full SOTA: Qwen2.5-VL + DINOv2-VLAD)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vlm_hybrid --vlm-endpoint http://localhost:8000/v1

# Run (no GPU, heuristic mode)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy heuristic --obstacle-method simple_edge --device cpu

# Run tests
python indoor_nav/test_integration.py --skip-sdk
```

## Architecture

```
Agent Orchestrator (10 Hz)
├── SDK Client → Camera frames + Telemetry + Control
├── Goal Matcher (DINOv2-VLAD) → Checkpoint arrival detection
├── Navigation Policy (VLM-Hybrid) → Action commands
├── Obstacle Avoidance (Depth Anything V2) → Speed/steer modulation
├── Topological Memory (Visual Graph) → Backtracking & planning
└── Recovery Manager → Stuck detection & escape
```

## Documentation

- **[SPECIFICATIONS.md](SPECIFICATIONS.md)** — Full technical specifications with references
- **[configs/config.py](configs/config.py)** — All configurable parameters
- **[run_indoor.py](run_indoor.py)** — CLI usage and examples

## File Structure

```
indoor_nav/
├── configs/config.py           # Configuration dataclasses
├── modules/
│   ├── sdk_client.py           # Async HTTP client for Earth Rovers SDK
│   ├── checkpoint_manager.py   # Goal matching (6 backends)
│   ├── obstacle_avoidance.py   # Monocular depth obstacle detection
│   ├── topological_memory.py   # Visual graph for backtracking
│   └── recovery.py             # Stuck detection & recovery
├── policies/
│   ├── base_policy.py          # Abstract policy interface
│   ├── vlm_hybrid_policy.py    # VLM + reactive controller
│   ├── vla_policy.py           # Vision-Language-Action model
│   └── nomad_policy.py         # NoMaD diffusion policy
├── agent.py                    # Main orchestrator
├── run_indoor.py               # CLI entry point
├── test_integration.py         # Integration tests
├── requirements_indoor.txt     # Dependencies
├── SPECIFICATIONS.md           # Full technical specs
└── README.md                   # This file
```

## DINOv3?

**DINOv3 does not exist** as of February 2025. DINOv2 (Meta/FAIR, 2023) is the
latest in the DINO family. We use the improved **DINOv2 with Registers** variant
(Darcet et al., ICLR 2024) which provides smoother patch features for our VLAD
aggregation. See [SPECIFICATIONS.md](SPECIFICATIONS.md#appendix-a-why-there-is-no-dinov3) for details.
