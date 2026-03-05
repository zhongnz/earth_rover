# Design Decisions Since Upstream

This file is a structured decision ledger for changes made after the upstream baseline.

## Comparison Metadata

```yaml
generated_on: 2026-03-05
repo_root: /home/ptz/dev/nyusd/nyu-earthrover
baseline:
  upstream_ref: db3a455bedd5d1a50979443bd26a870ade5e67ec
  branch: upstream/main
head:
  branch: main
  commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
notes:
  - Comparison is against the local upstream/main ref.
  - Remote fetch for upstream was auth-blocked in this environment.
diff_summary:
  files_changed: 58
  files_added: 48
  files_deleted: 0
  insertions: 9193
  deletions: 106
```

## Decision Records

### DD-0001

```yaml
id: DD-0001
title: Split project into two autonomy subsystems
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: architecture
status: accepted
what: Added distinct GPS autonomy and indoor autonomy packages.
where:
  - path: erc_autonomy/
  - path: indoor_nav/
  - path: PROJECT_DOCUMENTATION.md
how: Created separate runners, configs, modules, and docs for each mode.
why: Outdoor GPS and indoor image-goal missions have different constraints and failure modes.
rationale: Separation improves velocity, testability, and risk isolation.
supporting_info:
  - Upstream had 44 tracked files; head has 92.
references:
  - https://fastapi.tiangolo.com/
```

### DD-0002

```yaml
id: DD-0002
title: Enforce local bridge API boundary for autonomy modules
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: architecture
status: accepted
what: Autonomy code consumes local HTTP endpoints instead of browser/cloud internals.
where:
  - path: erc_autonomy/sdk_io.py
  - path: indoor_nav/modules/sdk_client.py
  - path: PROJECT_DOCUMENTATION.md
how: Wrapped all IO through /data, /v2/front, /control, and mission endpoints.
why: Reduce coupling and keep runtime dependencies stable.
rationale: Allows autonomy iteration without touching UI, Agora wiring, or cloud API internals.
supporting_info:
  - Control/data/mission planes are documented in PROJECT_DOCUMENTATION.md.
references:
  - https://docs.aiohttp.org/en/stable/
  - https://fastapi.tiangolo.com/
```

### DD-0003

```yaml
id: DD-0003
title: Build modular GPS mission pipeline
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: architecture
status: accepted
what: Introduced mission FSM, watchdog, estimator, traversability, BEV, planner, goal, recovery modules.
where:
  - path: erc_autonomy/mission_runner.py
  - path: erc_autonomy/mission_fsm.py
  - path: erc_autonomy/state_estimator.py
  - path: erc_autonomy/traversability.py
  - path: erc_autonomy/bev_mapper.py
  - path: erc_autonomy/planner.py
  - path: erc_autonomy/goal_manager.py
  - path: erc_autonomy/recovery.py
how: Runner composes narrow modules and executes a fixed-rate control loop.
why: Keep each concern replaceable and debuggable.
rationale: Faster tuning and lower blast radius when changing one subsystem.
supporting_info:
  - Runner status logs include subsystem outputs for observability.
references:
  - https://numpy.org/doc/stable/
  - https://docs.opencv.org/4.x/
```

### DD-0004

```yaml
id: DD-0004
title: Default safety posture for GPS motion
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: safety
status: accepted
what: Motion is opt-in; stale sensor watchdog forces safe stop.
where:
  - path: erc_autonomy/config.py
  - path: erc_autonomy/watchdog.py
  - path: erc_autonomy/mission_runner.py
how: enable_motion defaults false; stale callback sends repeated zero commands.
why: Prevent unsafe movement during integration and stream gaps.
rationale: Fail-safe behavior takes priority over throughput.
supporting_info:
  - stale_sensor_ms and stop pulse parameters are configurable.
references:
  - https://docs.aiohttp.org/en/stable/
```

### DD-0005

```yaml
id: DD-0005
title: Dual traversability backend with graceful SAM2 fallback
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
  - commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
    date: 2026-03-05
type: perception
status: accepted
what: Kept simple_edge as default; added optional sam2 backend with automatic fallback.
where:
  - path: erc_autonomy/traversability.py
  - path: erc_autonomy/config.py
  - path: erc_autonomy/run_gps.py
how: Runtime checks for SAM2 config/checkpoint/import/build failures and falls back to simple_edge.
why: Preserve robustness when optional model assets are missing or unsupported.
rationale: Optional high-capability backend without breaking baseline runtime.
supporting_info:
  - Fallback behavior covered by unit tests.
references:
  - https://github.com/facebookresearch/sam2
  - https://arxiv.org/abs/2408.00714
```

### DD-0006

```yaml
id: DD-0006
title: Checkpoint-aware speed and steering tapering
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: planning
status: accepted
what: Added distance-based and failure-feedback factors for motion near checkpoints.
where:
  - path: erc_autonomy/mission_runner.py
  - path: erc_autonomy/goal_manager.py
  - path: erc_autonomy/config.py
how: Computes goal turn hint + distance taper; applies extra damping after failed checkpoint reports.
why: Reduce overshoot and repeated near-miss behavior in mission scoring.
rationale: Mission scoring depends on reliable checkpoint transitions.
supporting_info:
  - Stores proximate distance from failed checkpoint responses and decays its effect over time.
references:
  - https://numpy.org/doc/stable/
```

### DD-0007

```yaml
id: DD-0007
title: Explicit GPS recovery state machine
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: safety
status: accepted
what: Added deterministic stuck-recovery sequence (backtrack -> rotate -> pause).
where:
  - path: erc_autonomy/recovery.py
  - path: erc_autonomy/mission_runner.py
how: Detects commanded-but-not-moving conditions and overrides commands during recovery windows.
why: Avoid deadlock when planner output cannot progress robot state.
rationale: Structured recovery is easier to reason about than ad hoc retries.
supporting_info:
  - Recovery status is emitted in runner logs.
references:
  - https://numpy.org/doc/stable/
```

### DD-0008

```yaml
id: DD-0008
title: Indoor orchestrator with policy plugin architecture
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: architecture
status: accepted
what: Added pluggable policy backends (nomad, vint, vlm_hybrid, vla, heuristic).
where:
  - path: indoor_nav/agent.py
  - path: indoor_nav/run_indoor.py
  - path: indoor_nav/policies/
how: Agent creates policy by backend key and runs common control loop contract.
why: Enable side-by-side experimentation across policy classes.
rationale: Competition research cadence requires fast A/B without control-loop rewrites.
supporting_info:
  - Each policy backend has a separate module implementing the same interface.
references:
  - https://huggingface.co/docs/transformers/index
  - https://pytorch.org/docs/stable/
```

### DD-0009

```yaml
id: DD-0009
title: Indoor goal matching baseline and matcher diversity
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: perception
status: accepted
what: Added image-goal checkpoint manager with multiple matcher backends.
where:
  - path: indoor_nav/modules/checkpoint_manager.py
  - path: indoor_nav/configs/config.py
how: Central GoalMatcher supports learned and classical methods under one config knob.
why: Different environments need different recall/precision tradeoffs.
rationale: Model heterogeneity is useful for robust image-goal navigation.
supporting_info:
  - Includes dinov2_vlad, siglip2, dinov2, eigenplaces, clip, sift.
references:
  - https://arxiv.org/abs/2304.07193
  - https://arxiv.org/abs/2308.00688
  - https://arxiv.org/abs/2502.14786
```

### DD-0010

```yaml
id: DD-0010
title: Add DINOv3-VLAD toggle and A/B benchmark flow
introduced_in:
  - commit: c8a4a2e46cc74f617c15cf862ebaf848f25b525b
    date: 2026-03-04
type: experimentation
status: accepted
what: Added dinov3_vlad option and scripted matcher benchmark for retrieval metrics/latency.
where:
  - path: indoor_nav/configs/config.py
  - path: indoor_nav/modules/checkpoint_manager.py
  - path: indoor_nav/eval_match_ab.py
  - path: indoor_nav/README.md
how: Shared VLAD loader + new method value + standalone evaluation script.
why: Compare new representation quality against existing baseline before default switch.
rationale: Empirical model selection is safer than unvalidated upgrades.
supporting_info:
  - Benchmark reports top-1, top-k, MRR, mean and p95 latency.
references:
  - https://arxiv.org/abs/2308.00688
```

### DD-0011

```yaml
id: DD-0011
title: Depth-based indoor obstacle avoidance with fallback hierarchy
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: safety
status: accepted
what: Added depth_anything/depth_pro/simple_edge obstacle backends.
where:
  - path: indoor_nav/modules/obstacle_avoidance.py
  - path: indoor_nav/configs/config.py
how: Zone-based occupancy maps depth outputs to speed_factor and steer_bias.
why: Keep obstacle handling available across GPU/no-GPU setups.
rationale: Safety behavior must remain available even when advanced models are unavailable.
supporting_info:
  - Includes temporal smoothing and narrow-passage handling.
references:
  - https://arxiv.org/abs/2406.09414
  - https://arxiv.org/abs/2410.02073
```

### DD-0012

```yaml
id: DD-0012
title: Add indoor topological memory graph
introduced_in:
  - commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04
type: planning
status: accepted
what: Added online node-edge graph with loop closure and A* pathing.
where:
  - path: indoor_nav/modules/topological_memory.py
  - path: indoor_nav/agent.py
how: Creates keyframe nodes, connects traversal edges, detects visual loop closures.
why: Improve recovery and global route quality in complex indoor layouts.
rationale: Pure reactive control is weak in dead-end/cycle-rich environments.
supporting_info:
  - Configurable graph size and closure thresholds.
references:
  - https://arxiv.org/abs/2407.07775
```

### DD-0013

```yaml
id: DD-0013
title: Harden FastAPI mission backend against blocking and transient failures
introduced_in:
  - commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
    date: 2026-03-04
type: reliability
status: accepted
what: Wrapped sync requests calls in thread offload and improved HTTP error mapping.
where:
  - path: main.py
how: Added asyncio.to_thread wrappers, request exception handling, and telemetry null checks.
why: Prevent event-loop blocking and provide clearer client-side failure semantics.
rationale: Mission endpoints should degrade predictably under partial outages.
supporting_info:
  - /data returns 503 when telemetry is unavailable.
  - checkpoint/intervention lat-lon validation now reports 503 for missing telemetry.
references:
  - https://requests.readthedocs.io/en/latest/
  - https://fastapi.tiangolo.com/
```

### DD-0014

```yaml
id: DD-0014
title: Parameterize browser runtime and SDK URL
introduced_in:
  - commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
    date: 2026-03-04
type: deployment
status: accepted
what: Added browser engine/headless/base URL environment controls.
where:
  - path: browser_service.py
  - path: .env.sample
  - path: docker-compose.yml
how: Selects Playwright engine dynamically and launches with configurable flags.
why: Support local interactive runs and container/CI runs without code edits.
rationale: Runtime environment should be data-driven, not hardcoded.
supporting_info:
  - Added BROWSER_ENGINE, BROWSER_HEADLESS, SDK_BASE_URL.
references:
  - https://playwright.dev/python/docs/intro
```

### DD-0015

```yaml
id: DD-0015
title: Modernize container/runtime defaults and env hygiene
introduced_in:
  - commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
    date: 2026-03-04
  - commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
    date: 2026-03-05
type: operations
status: accepted
what: Moved to python:3.11-slim, simplified runtime command, sanitized .env template.
where:
  - path: Dockerfile
  - path: .env.sample
  - path: docker-compose.yml
  - path: .gitignore
how: Removed hardcoded sample secrets and added explicit browser/SAM2 config knobs.
why: Improve reproducibility and reduce setup mistakes.
rationale: Safer defaults lower onboarding and operational risk.
supporting_info:
  - Added .models/ to gitignore for local model artifacts.
references:
  - https://hypercorn.readthedocs.io/en/latest/index.html
  - https://pypi.org/project/python-dotenv/
```

### DD-0016

```yaml
id: DD-0016
title: Align examples to v2 frame API
introduced_in:
  - commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
    date: 2026-03-04
type: compatibility
status: accepted
what: Updated example clients from /v1/front,/v1/rear to /v2/front,/v2/rear.
where:
  - path: examples/front_example.py
  - path: examples/rear_example.py
  - path: examples/rear_test.html
how: Endpoint string updates in sample scripts/html.
why: Keep examples aligned with current local bridge contract.
rationale: Working examples reduce confusion during bring-up.
supporting_info:
  - Also corrected front window label in front_example.py.
references:
  - https://docs.aiohttp.org/en/stable/
```

### DD-0017

```yaml
id: DD-0017
title: Operationalize SAM2 with reproducible setup, preflight, benchmark, and tests
introduced_in:
  - commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
    date: 2026-03-05
type: reliability
status: accepted
what: Added setup script, preflight CLI, benchmark CLI, optional deps file, and fallback tests.
where:
  - path: scripts/setup_sam2.sh
  - path: erc_autonomy/check_sam2.py
  - path: erc_autonomy/bench_traversability.py
  - path: erc_autonomy/requirements_sam2.txt
  - path: erc_autonomy/tests/test_traversability_fallback.py
how: Pin SAM2 commit and expose deterministic setup/probe workflow.
why: SAM2 assets are heavy and optional; runtime needs clear diagnostics and safe fallback.
rationale: Avoid mission-time surprises from missing model/config assets.
supporting_info:
  - setup script supports variant selection and optional sha256 verification.
references:
  - https://github.com/facebookresearch/sam2
  - https://github.com/cocodataset/cocoapi
```

### DD-0018

```yaml
id: DD-0018
title: Add CI safety gates for syntax, fallback behavior, and CLI surfaces
introduced_in:
  - commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
    date: 2026-03-05
type: quality
status: accepted
what: Added GitHub Actions workflow running compileall, fallback unit test, and CLI help checks.
where:
  - path: .github/workflows/ci.yml
how: PR/push job installs deps and executes minimal but targeted checks.
why: Catch obvious regressions before merge.
rationale: Cheap automated checks prevent breakage in core runner interfaces.
supporting_info:
  - Covers run_gps, bench_traversability, check_sam2 CLIs.
references:
  - https://docs.github.com/actions
```

### DD-0019

```yaml
id: DD-0019
title: Treat documentation as a maintained technical spec
introduced_in:
  - commit: fb5aa0a28b2f0acbb1110d619875e95fd9c60655
    date: 2026-03-04
  - commit: 3d7071c06e11b687ab8e0c0635ece270f996425f
    date: 2026-03-04
  - commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
    date: 2026-03-05
type: process
status: accepted
what: Expanded docs to a full technical specification with explicit change-control guidance.
where:
  - path: PROJECT_DOCUMENTATION.md
  - path: README.md
how: Added architecture, runtime contracts, risk register, CI/testing runbooks, and references.
why: Keep implementation and design intent synchronized as the stack evolves.
rationale: Shared documentation reduces onboarding time and architecture drift.
supporting_info:
  - Includes dependency/tool rationale and reference bibliography.
references:
  - https://fastapi.tiangolo.com/
  - https://playwright.dev/python/docs/intro
```

## Anchor Index (Per-Decision Evidence Pointers)

```yaml
DD-0001:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - PROJECT_DOCUMENTATION.md:61
DD-0002:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - erc_autonomy/sdk_io.py:19
    - indoor_nav/modules/sdk_client.py:77
    - PROJECT_DOCUMENTATION.md:93
DD-0003:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - erc_autonomy/mission_runner.py:26
    - erc_autonomy/mission_fsm.py:7
    - erc_autonomy/state_estimator.py:29
    - erc_autonomy/traversability.py:46
    - erc_autonomy/bev_mapper.py:23
    - erc_autonomy/planner.py:37
    - erc_autonomy/goal_manager.py:48
    - erc_autonomy/recovery.py:19
DD-0004:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - erc_autonomy/config.py:36
    - erc_autonomy/watchdog.py:7
    - erc_autonomy/mission_runner.py:71
DD-0005:
  commits:
    - d1bada7a5905cc92abd61e1447e0840bce5e03de
    - 47d8ee897a3bea180298ff68877ed04ac4c30958
  anchors:
    - erc_autonomy/config.py:43
    - erc_autonomy/run_gps.py:58
    - erc_autonomy/traversability.py:46
    - erc_autonomy/tests/test_traversability_fallback.py:22
DD-0006:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - erc_autonomy/goal_manager.py:127
    - erc_autonomy/mission_runner.py:160
    - erc_autonomy/mission_runner.py:238
DD-0007:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - erc_autonomy/recovery.py:19
    - erc_autonomy/mission_runner.py:367
DD-0008:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - indoor_nav/run_indoor.py:80
    - indoor_nav/agent.py:112
DD-0009:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - indoor_nav/configs/config.py:99
    - indoor_nav/modules/checkpoint_manager.py:64
DD-0010:
  commit: c8a4a2e46cc74f617c15cf862ebaf848f25b525b
  anchors:
    - indoor_nav/configs/config.py:99
    - indoor_nav/modules/checkpoint_manager.py:172
    - indoor_nav/eval_match_ab.py:58
DD-0011:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - indoor_nav/configs/config.py:123
    - indoor_nav/modules/obstacle_avoidance.py:59
DD-0012:
  commit: d1bada7a5905cc92abd61e1447e0840bce5e03de
  anchors:
    - indoor_nav/modules/topological_memory.py:104
    - indoor_nav/agent.py:275
DD-0013:
  commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
  anchors:
    - main.py:97
    - main.py:483
    - main.py:503
    - main.py:519
    - main.py:797
DD-0014:
  commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
  anchors:
    - browser_service.py:12
    - browser_service.py:35
    - browser_service.py:54
    - .env.sample:14
    - docker-compose.yml:13
DD-0015:
  commits:
    - 53cefc30eca9560c60f0c6b2a89e4c00796c5233
    - 47d8ee897a3bea180298ff68877ed04ac4c30958
  anchors:
    - Dockerfile:1
    - .env.sample:1
    - .gitignore:13
DD-0016:
  commit: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
  anchors:
    - examples/front_example.py:9
    - examples/rear_example.py:9
    - examples/rear_test.html:31
DD-0017:
  commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
  anchors:
    - scripts/setup_sam2.sh:4
    - erc_autonomy/check_sam2.py:130
    - erc_autonomy/bench_traversability.py:14
    - erc_autonomy/requirements_sam2.txt:15
    - erc_autonomy/tests/test_traversability_fallback.py:22
DD-0018:
  commit: 47d8ee897a3bea180298ff68877ed04ac4c30958
  anchors:
    - .github/workflows/ci.yml:1
DD-0019:
  commits:
    - fb5aa0a28b2f0acbb1110d619875e95fd9c60655
    - 3d7071c06e11b687ab8e0c0635ece270f996425f
    - 47d8ee897a3bea180298ff68877ed04ac4c30958
  anchors:
    - PROJECT_DOCUMENTATION.md:1
    - PROJECT_DOCUMENTATION.md:783
    - PROJECT_DOCUMENTATION.md:828
    - README.md:7
```

## Commit Timeline Since Upstream Baseline

```yaml
commits:
  - hash: d1bada7a5905cc92abd61e1447e0840bce5e03de
    date: 2026-03-04T18:58:30-05:00
    subject: Initial import (clean history)
  - hash: c8a4a2e46cc74f617c15cf862ebaf848f25b525b
    date: 2026-03-04T19:11:46-05:00
    subject: indoor_nav: add DINOv3-VLAD toggle and A/B matcher eval
  - hash: 7e17c967750ebbfebf567616308b1f10f8493810
    date: 2026-03-04T19:13:23-05:00
    subject: Merge branch 'feature/dinov3-ab'
  - hash: 53cefc30eca9560c60f0c6b2a89e4c00796c5233
    date: 2026-03-04T19:32:56-05:00
    subject: Harden SDK runtime and setup defaults
  - hash: fb5aa0a28b2f0acbb1110d619875e95fd9c60655
    date: 2026-03-04T20:00:20-05:00
    subject: docs: refresh project architecture, rationale, and references
  - hash: 3d7071c06e11b687ab8e0c0635ece270f996425f
    date: 2026-03-04T20:34:01-05:00
    subject: docs: clarify architecture and add teammate onboarding guide
  - hash: 47d8ee897a3bea180298ff68877ed04ac4c30958
    date: 2026-03-05T14:28:01-05:00
    subject: Full technical specification added
```
