# Indoor Navigation Agent — ICRA 2025 Earth Rover Challenge

**SOTA 2025 autonomous indoor navigation for the Earth Rover Challenge Indoor Track.**

## Overview

This system remotely controls an Earth Rover Mini/Zero ground robot through
unknown indoor environments, navigating to sequential image-goal checkpoints
using only onboard cameras and IMU.

### Key Components

| Component | Technology | Paper |
|-----------|-----------|-------|
| Goal Matching | DINOv2-VLAD (VPR baseline) / DINOv3-VLAD (toggle) / CosPlace (A/B) | AnyLoc (RA-L 2023, arXiv:2308.00688) + Registers (ICLR 2024, arXiv:2309.16588) + DINOv3 (arXiv:2508.10104) + CosPlace (CVPR 2022, arXiv:2204.02287) |
| VLM Reasoning | Qwen2.5-VL 7B | arXiv:2502.13923 |
| Depth Estimation | Depth Anything V2 Base | NeurIPS 2024, arXiv:2406.09414 |
| Topological Memory | Visual graph + A* | Inspired by NTS (NeurIPS 2020), Mobility VLA (arXiv:2407.07775) |
| Navigation Policy | VLM-Hybrid (default), NoMaD, ViNT, GNM, VLA | NoMaD: arXiv:2310.07896, ViNT: arXiv:2306.14846, GNM: ICRA 2023, OpenVLA: arXiv:2406.09246 |

## Quick Start

```bash
# Install
conda create -n erv python=3.11 && conda activate erv
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt -r indoor_nav/requirements_indoor.txt

# Start SDK server
hypercorn main:app --reload

# Preflight before a run
# First place ordered checkpoint images in indoor_nav/goals/ or pass explicit file paths.
python indoor_nav/check_indoor.py --goals indoor_nav/goals/

# Run (full SOTA: Qwen2.5-VL + DINOv2-VLAD)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vlm_hybrid --vlm-endpoint http://127.0.0.1:8001/v1 \
    --match-method dinov2_vlad

# Run (DINOv3-VLAD toggle)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vlm_hybrid --match-method dinov3_vlad

# A/B compare DINO backends plus CosPlace
python indoor_nav/eval_match_ab.py \
    --goals-dir indoor_nav/goals \
    --queries-dir indoor_nav/eval_queries \
    --methods dinov2_vlad,dinov3_vlad,cosplace

# Benchmark exact image matching for wall-mounted images
python indoor_nav/eval_match_ab.py \
    --goals-dir indoor_nav/goals \
    --queries-dir indoor_nav/eval_queries \
    --methods dinov2_direct,wall_crop_direct,wall_rectify_direct \
    --viz-dir indoor_nav/match_reports/wall_crop

# Import a public wall-image benchmark subset from Stanford Mobile Visual Search
python indoor_nav/import_smvs.py \
    --output-root indoor_nav/datasets/smvs_wall \
    --categories museum_paintings,print

# Benchmark + write a visual report bundle
python indoor_nav/eval_match_ab.py \
    --goals-dir indoor_nav/goals \
    --queries-dir indoor_nav/eval_queries \
    --methods dinov2_vlad,cosplace \
    --viz-dir indoor_nav/match_reports/latest

# Quick smoke test when you do not have eval_queries yet
python indoor_nav/eval_match_ab.py \
    --goals-dir indoor_nav/goals \
    --self-query \
    --methods dinov2_vlad,cosplace \
    --viz-dir indoor_nav/match_reports/self_query

# Run with CosPlace as the goal matcher
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vlm_hybrid --match-method cosplace

# Run with crop-aware wall-image matching
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vlm_hybrid --match-method wall_crop_direct

# Run with perspective-rectified wall-image matching
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vlm_hybrid --match-method wall_rectify_direct

# Live front-camera matcher probe with no motion commands
python indoor_nav/test_match_live.py \
    --goals "indoor_nav/goals/Screenshot 2026-03-08 133552.png" \
    --match-method wall_crop_direct \
    --device cpu

# Run (no GPU, heuristic mode)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy heuristic --obstacle-method simple_edge --device cpu

# Run NoMaD with an official visualnav-transformer checkpoint (.pth)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy nomad --model-path /path/to/nomad.pth \
    --nomad-repo-root /path/to/visualnav-transformer \
    --match-method sift --obstacle-method simple_edge

# Run ViNT with an official visualnav-transformer checkpoint (.pth)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy vint --model-path /path/to/vint.pth \
    --nomad-repo-root /path/to/visualnav-transformer \
    --match-method sift --obstacle-method simple_edge

# Run GNM with an official visualnav-transformer checkpoint (.pth)
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ \
    --policy gnm --model-path /path/to/gnm_large.pth \
    --nomad-repo-root /path/to/visualnav-transformer \
    --match-method sift --obstacle-method simple_edge

# Run tests
python indoor_nav/test_integration.py --skip-sdk
```

The visual report bundle in `--viz-dir` includes:

- `index.html` for a quick browser view
- `report.json` with per-query rankings and scores
- `*_queries.csv` per-method exports for filtering and tuning
- `summary.png` when plotting deps are installed, otherwise `summary.txt`
- `*_failures.png` contact sheets when image-rendering deps are installed and top-1 failures exist

When a backend emits crop proposals, the HTML report and failure sheets also show candidate counts and box overlays on the query, ground-truth goal, and predicted goal images.

For `wall_crop_direct` and `wall_rectify_direct`, the benchmark also exposes wall-stage tuning flags such as
`--wall-crop-min-area-frac`, `--wall-crop-max-candidates`, `--wall-crop-padding-frac`,
and `--wall-crop-score-weight`.

For a real benchmark, populate `indoor_nav/eval_queries/` with images whose filenames match a goal stem
or start with `<goal_stem>__...`, or provide `--mapping-csv query,goal`.

If you want a public dataset in that layout, `import_smvs.py` downloads the Stanford Mobile Visual Search
categories most relevant to wall-image matching and writes:

- `goals/`
- `eval_queries/`
- `mapping.csv`

Then run:

```bash
python indoor_nav/eval_match_ab.py \
    --goals-dir indoor_nav/datasets/smvs_wall/goals \
    --queries-dir indoor_nav/datasets/smvs_wall/eval_queries \
    --mapping-csv indoor_nav/datasets/smvs_wall/mapping.csv \
    --methods dinov2_direct,wall_crop_direct \
    --viz-dir indoor_nav/match_reports/smvs_wall
```

`dinov3_vlad` is available for A/B runs if you have Hugging Face access to
`facebook/dinov3-vitb16-pretrain-lvd1689m`; otherwise the benchmark will mark it as failed and continue.

## VLM Endpoint

`vlm_hybrid` only performs real VLM reasoning when a model server is set.
Keep the SDK bridge on `http://127.0.0.1:8000` and run the VLM server on a
different port, typically `8001`.

You can provide the VLM config on the CLI or through `.env`:

```env
VLM_ENDPOINT=http://127.0.0.1:8001/v1
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLM_API_KEY=
```

With those variables set, this is enough:

```bash
python indoor_nav/run_indoor.py --goals indoor_nav/goals/ --policy vlm_hybrid
```

## Architecture

```
Agent Orchestrator (10 Hz)
├── SDK Client → Camera frames + Telemetry + Control
├── Goal Matcher (DINOv2-VLAD / DINOv3-VLAD) → Checkpoint arrival detection
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
│   ├── checkpoint_manager.py   # Goal matching (7 backends incl. DINOv3-VLAD)
│   ├── obstacle_avoidance.py   # Monocular depth obstacle detection
│   ├── topological_memory.py   # Visual graph for backtracking
│   └── recovery.py             # Stuck detection & recovery
├── policies/
│   ├── base_policy.py          # Abstract policy interface
│   ├── vlm_hybrid_policy.py    # VLM + reactive controller
│   ├── vla_policy.py           # Vision-Language-Action model
│   └── nomad_policy.py         # NoMaD diffusion policy
├── agent.py                    # Main orchestrator
├── cli_common.py               # Shared indoor CLI/config builder
├── check_indoor.py             # Operator preflight for goals, SDK, and backends
├── run_indoor.py               # CLI entry point
├── eval_match_ab.py            # A/B matcher benchmark script
├── eval_queries/               # Optional query set for A/B matching
├── goals/                      # Placeholder directory for mission-specific goal images
├── test_integration.py         # Integration tests
├── requirements_indoor.txt     # Dependencies
├── SPECIFICATIONS.md           # Full technical specs
└── README.md                   # This file
```

## DINOv3?

As of August 2025, **DINOv3 exists** and is available in this repo via:

- `--match-method dinov3_vlad`
- `--match-method cosplace`

Recommended VPR baseline remains `dinov2_vlad` for stability and compatibility.
