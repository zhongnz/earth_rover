# ERC Autonomy Scaffold

This package provides the Week 1-6 implementation scaffold for the autonomous
driving track:

- Hardened async SDK IO (`sdk_io.py`)
- Mission lifecycle state machine (`mission_fsm.py`)
- Stale-sensor watchdog with emergency stop (`watchdog.py`)
- Filtered GPS/heading state estimator (`state_estimator.py`)
- Traversability inference (`traversability.py`)
- BEV projection interface (`bev_mapper.py`)
- Candidate rollouts + path fusion planner (`planner.py`)
- Goal/checkpoint manager with bearing hinting (`goal_manager.py`)
- Recovery manager with backtrack/rotate behavior (`recovery.py`)
- Runnable mission loop (`mission_runner.py`, `run_gps.py`)

## Quick Start

From project root:

```bash
python -m erc_autonomy.run_gps --url http://127.0.0.1:8000 --loop-hz 10
```

Start/end mission via API lifecycle:

```bash
python -m erc_autonomy.run_gps --start-mission --end-mission
```

Enable guarded reactive motion (optional):

```bash
python -m erc_autonomy.run_gps --enable-motion --max-linear 0.2 --max-angular 0.35
```

Tune checkpoint and recovery behavior:

```bash
python -m erc_autonomy.run_gps \
  --enable-motion \
  --checkpoint-distance 15.5 \
  --checkpoint-refresh 4.0 \
  --recovery-stuck-timeout 3.5
```

Enable adaptive checkpoint speed policy (distance + failure feedback):

```bash
python -m erc_autonomy.run_gps \
  --enable-motion \
  --checkpoint-slowdown-start 30 \
  --checkpoint-slowdown-hard 9 \
  --checkpoint-angular-min-factor 0.6 \
  --checkpoint-failure-effect 14 \
  --checkpoint-failure-buffer 7 \
  --checkpoint-failure-angular-min-factor 0.5
```

Optional SAM2 setup (reproducible):

```bash
# 1) Install torch for your platform first (CPU or CUDA build)
# 2) Install pinned SAM2 extras for this repo
pip install -r erc_autonomy/requirements_sam2.txt

# 3) Download a pinned SAM2 config + checkpoint into .models/sam2/
scripts/setup_sam2.sh --variant sam2.1_hiera_large

# 4) Validate local paths/env before mission run
python -m erc_autonomy.check_sam2
```

Run with SAM2 backend:

```bash
python -m erc_autonomy.run_gps \
  --traversability-backend sam2 \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /path/to/sam2_hiera_large.pt
```

Or use env vars instead of flags:

```bash
export SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
export SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt
export SAM2_DEVICE=auto
python -m erc_autonomy.run_gps --traversability-backend sam2
```

`run_gps.py` loads `.env` automatically, so SAM2 vars in `.env` are picked up
without manual `export`.

Notes:

- Motion remains disabled unless `--enable-motion` is explicitly set.
- `sam2` backend is optional and requires both `--sam2-model-cfg` and
  `--sam2-checkpoint`; if unavailable at runtime it falls back to
  `simple_edge`.
- CLI flags can also be provided via environment variables:
  `SAM2_MODEL_CFG`, `SAM2_CHECKPOINT`, `SAM2_DEVICE`, `SAM2_MAX_SIDE`,
  `SAM2_POINTS_PER_SIDE`, `SAM2_PRED_IOU_THRESH`,
  `SAM2_STABILITY_SCORE_THRESH`, `SAM2_MIN_MASK_REGION_AREA`.
- Speed is automatically tapered as the active checkpoint gets closer, and
  additionally tapered after failed `/checkpoint-reached` attempts using the
  returned `proximate_distance_to_checkpoint`.
- Angular aggressiveness can be tapered independently near checkpoints and
  after failed `/checkpoint-reached` attempts to reduce overshoot.
- Structured JSON logs are emitted to stdout for downstream metric ingestion.

SAM2 benchmark example:

```bash
python -m erc_autonomy.bench_traversability \
  --images-dir ./logs/exported_frames \
  --pattern '*.jpg' \
  --backend both \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /path/to/sam2_hiera_large.pt
```

Fallback regression tests:

```bash
python -m unittest erc_autonomy.tests.test_traversability_fallback
```

CI safety gates run on PR/push and include:

- `python -m compileall ...`
- `python -m unittest erc_autonomy.tests.test_traversability_fallback`
- CLI smoke checks (`run_gps --help`, `bench_traversability --help`, `check_sam2 --help`)
