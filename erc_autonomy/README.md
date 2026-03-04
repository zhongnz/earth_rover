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

Notes:

- Motion remains disabled unless `--enable-motion` is explicitly set.
- `sam2` backend is currently an interface placeholder and falls back to the
  built-in `simple_edge` inference path.
- Speed is automatically tapered as the active checkpoint gets closer, and
  additionally tapered after failed `/checkpoint-reached` attempts using the
  returned `proximate_distance_to_checkpoint`.
- Angular aggressiveness can be tapered independently near checkpoints and
  after failed `/checkpoint-reached` attempts to reduce overshoot.
- Structured JSON logs are emitted to stdout for downstream metric ingestion.
