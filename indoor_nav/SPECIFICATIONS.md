# Indoor Navigation Agent — Technical Specifications Report

**ICRA 2025 Earth Rover Challenge — Indoor Track**
**Version:** 2.0 (SOTA 2025)
**Date:** February 2025
**Authors:** NYU Earthrover Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Competition Context](#2-competition-context)
3. [System Architecture](#3-system-architecture)
4. [Module Specifications](#4-module-specifications)
   - 4.1 [SDK Client](#41-sdk-client)
   - 4.2 [Goal Matching (Checkpoint Manager)](#42-goal-matching-checkpoint-manager)
   - 4.3 [Navigation Policies](#43-navigation-policies)
   - 4.4 [Obstacle Avoidance](#44-obstacle-avoidance)
   - 4.5 [Topological Memory](#45-topological-memory)
   - 4.6 [Recovery Behaviors](#46-recovery-behaviors)
   - 4.7 [Agent Orchestrator](#47-agent-orchestrator)
5. [Model Selection Justification](#5-model-selection-justification)
6. [Academic References](#6-academic-references)
7. [Deployment Guide](#7-deployment-guide)
8. [Configuration Reference](#8-configuration-reference)
9. [File Structure](#9-file-structure)

---

## 1. Executive Summary

This document specifies the design of an autonomous indoor navigation agent for the ICRA 2025 Earth Rover Challenge Indoor Track. The agent remotely controls a small ground robot (Earth Rover Mini/Zero) through unknown indoor environments, navigating to sequential image-goal checkpoints using only onboard cameras and IMU.

### Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Goal Matching** | DINOv2-VLAD (with registers) | SOTA visual place recognition; AnyLoc-style aggregation on DINOv2-reg4 patch tokens |
| **High-Level Reasoning** | Qwen2.5-VL 7B | Best open VLM (Feb 2025); matches GPT-4o on spatial understanding |
| **Depth Estimation** | Depth Anything V2 Base | Best speed/accuracy tradeoff; 10x faster than diffusion-based alternatives |
| **Topological Memory** | Visual graph + A\* | Enables backtracking, loop closure, and global planning without a metric map |
| **Low-Level Control** | Reactive controller | Converts VLM instructions + obstacle info into smooth velocity commands at 10 Hz |
| **Recovery** | Hierarchical behaviors | Escalating maneuvers (back-up → random turn → wall follow → 360° scan) |

### Performance Targets

- **Control loop:** 10 Hz (100 ms per tick)
- **VLM query rate:** 0.3–0.5 Hz (2–3 seconds per query, async)
- **Goal matching:** < 50 ms per frame (DINOv2-VLAD inference)
- **Depth estimation:** < 30 ms per frame (Depth Anything V2 Base)
- **Latency budget:** 200–400 ms end-to-end (dominated by network RTT to robot)

---

## 2. Competition Context

### 2.1 Indoor Track Rules

- **Objective:** Navigate a ground robot through an indoor environment to reach a sequence of image-goal checkpoints.
- **Input:** Goal images are provided as photographs of target locations. No GPS is available indoors.
- **Robot:** Earth Rover Mini (single front camera, IMU, 4G connectivity, ~3 km/h max speed).
- **Control:** Remote control via SDK — teams host their own AI server and send velocity commands.
- **Scoring:** Points awarded per checkpoint reached; difficulty rated 1–10.
- **Constraints:**
  - No prior environment map.
  - No SLAM or LiDAR (camera-only).
  - Network latency is a factor (~100–300 ms RTT).
  - Battery life limits total mission time.

### 2.2 Technical Challenges

1. **Visual Place Recognition (VPR):** Must match current camera view to goal images under viewpoint, lighting, and scale changes.
2. **Exploration:** Unknown environment layout — the robot must systematically search for goals.
3. **Obstacle Avoidance:** Narrow corridors, furniture, doors — using monocular depth only.
4. **Recovery:** Dead ends, wrong turns, U-turns — must detect and recover from mistakes.
5. **Latency:** Control loop must remain responsive despite network delays.

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌───────────────────────────────────────────────────────┐
│                    Agent Orchestrator                  │
│                   (10 Hz async loop)                  │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │   SDK    │  │  Goal    │  │   Navigation      │  │
│  │  Client  │  │ Matcher  │  │   Policy          │  │
│  │          │  │ (DINOv2  │  │ (VLM-Hybrid /     │  │
│  │ /v2/front│  │  -VLAD)  │  │  VLA / NoMaD)     │  │
│  │ /data    │  │          │  │                    │  │
│  │ /control │  └──────────┘  └───────────────────┘  │
│  └──────────┘                                        │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Obstacle │  │   Topo   │  │    Recovery       │   │
│  │ Avoid    │  │  Memory  │  │    Manager        │   │
│  │ (Depth   │  │  (Graph  │  │   (Escalating     │   │
│  │  Any V2) │  │   + A*)  │  │    behaviors)     │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└───────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  ┌─────────────┐           ┌─────────────┐
  │ Earth Rover │◄─ 4G/WiFi─│  SDK Server  │
  │   Robot     │           │  (FastAPI +  │
  │             │           │  Playwright) │
  └─────────────┘           └─────────────┘
```

### 3.2 Control Flow (Per Tick)

1. **Fetch** frame + telemetry from SDK (`/v2/front`, `/data`)
2. **Topological Memory** update — add node if scene changed significantly
3. **Goal Matching** — compute DINOv2-VLAD similarity to current goal
4. **Check Arrival** — if similarity > threshold for N consecutive frames → checkpoint reached
5. **Obstacle Detection** — run Depth Anything V2 → depth map → zone analysis
6. **Policy Query** — get `(linear, angular)` from active navigation policy
7. **Smoothing** — exponential smoothing on commands to prevent jitter
8. **Safety Limits** — clamp to max speed, apply obstacle speed factor
9. **Send Control** — `POST /control {linear, angular}`
10. **Stuck Detection** — if no motion for > 8s despite commands → trigger recovery

### 3.3 State Machine

```
INIT → NAVIGATING → APPROACHING_GOAL → CHECKPOINT_REACHED → (next goal)
          ↓                                     ↑
       RECOVERING ──────────────────────────────┘
          ↓
        ERROR → (manual intervention)
```

---

## 4. Module Specifications

### 4.1 SDK Client

**File:** `modules/sdk_client.py`
**Purpose:** Async HTTP client for Earth Rovers SDK v4.9.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v2/front` | GET | Front camera frame (base64 JPEG) |
| `/v2/rear` | GET | Rear camera (Zero bots only) |
| `/data` | GET | Telemetry (battery, IMU, orientation, speed) |
| `/control` | POST | Send velocity commands `{linear, angular}` |
| `/checkpoint-reached` | POST | Report checkpoint arrival to scoring system |
| `/start-mission` | POST | Start a scored mission |
| `/end-mission` | POST | End mission |

**Key Design:**
- Uses `aiohttp` for non-blocking HTTP requests
- Configurable timeout (default 3s)
- Automatic base64 → OpenCV image decoding
- Structured `BotState` dataclass for telemetry

### 4.2 Goal Matching (Checkpoint Manager)

**File:** `modules/checkpoint_manager.py`
**Purpose:** Image-goal matching with multiple SOTA backends. Determines when the robot has reached a checkpoint.

#### 4.2.1 Supported Backends

| Method | Model | Dim | Speed | Quality | Use Case |
|--------|-------|-----|-------|---------|----------|
| **dinov2_vlad** | DINOv2-reg4 + VLAD | 24,576 | ~40ms | ★★★★★ | **Default** — best for indoor VPR |
| siglip2 | SigLIP 2 Base | 768 | ~20ms | ★★★★☆ | Semantic matching |
| dinov2 | DINOv2 Base CLS | 768 | ~15ms | ★★★☆☆ | Fast baseline |
| eigenplaces | ResNet50+EigenPlaces | 2048 | ~25ms | ★★★★☆ | Viewpoint-robust VPR |
| clip | CLIP ViT-B/32 | 512 | ~15ms | ★★☆☆☆ | Semantic baseline |
| sift | SIFT + RANSAC | — | ~80ms | ★★★☆☆ | Geometric verification |

#### 4.2.2 DINOv2-VLAD Pipeline (Recommended)

```
Image → DINOv2-reg4 ViT-B/14 → 256 patch tokens (768-d each)
     → L2-normalize patches
     → VLAD aggregation (32 clusters × 768-d = 24,576-d)
     → Intra-normalize per cluster → L2-normalize final
     → Cosine similarity vs. goal descriptor
```

**Why DINOv2 with Registers?**
Standard DINOv2 produces high-norm artifact tokens in background regions
(Darcet et al., 2024). These artifacts contaminate VLAD aggregation by
creating spurious residuals. The register variant adds 4 extra tokens that
absorb this computation, producing cleaner patch features ≈ 2–4% VPR
improvement.

**VLAD Initialization:**
- Online k-means on first 10–20 frames (10 iterations)
- 32 cluster centers learned from the operational environment
- One-time cost ~100ms, then frozen

#### 4.2.3 Checkpoint Arrival Logic

```python
if similarity >= threshold (0.78):
    patience_counter += 1
else:
    patience_counter = 0

if patience_counter >= patience (3 consecutive frames):
    → checkpoint reached → advance to next goal
```

### 4.3 Navigation Policies

**Files:** `policies/vlm_hybrid_policy.py`, `policies/vla_policy.py`, `policies/nomad_policy.py`

All policies implement `BasePolicy.predict(obs) → PolicyOutput(linear, angular)`.

#### 4.3.1 VLM-Hybrid Policy (Recommended)

**Architecture:** Two-tier — VLM (slow, semantic) + reactive controller (fast, local).

**High-Level (VLM, 0.3–0.5 Hz):**
- Sends current frame + goal image to Qwen2.5-VL 7B
- Receives structured JSON: `{action, reasoning, goal_visible, landmarks, estimated_distance, direction_to_goal, confidence}`
- 14 possible actions: `forward`, `forward_slow`, `forward_fast`, `veer_left`, `veer_right`, `turn_left`, `turn_right`, `turn_left_sharp`, `turn_right_sharp`, `search_left`, `search_right`, `stop`, `reverse`, `approach`

**Low-Level (Reactive, 10 Hz):**
- Maps VLM instruction → base `(linear, angular)` velocities
- Modulates based on: goal similarity, VLM-reported distance/direction, obstacle state
- Exponential smoothing prevents jitter

**Adaptive Query Frequency:**
- Scene change detection via HSV histogram difference
- If scene changes significantly → trigger early VLM query
- Otherwise waits the configured interval (2.5s default)

**Supported API Formats:**
| Format | Endpoint Pattern | Use With |
|--------|-----------------|----------|
| `openai` | `*/v1/*` | Qwen2.5-VL (vLLM), GPT-4o, Gemma 3 |
| `ollama` | `*:11434*` | LLaVA, Llama-Vision (legacy) |
| `anthropic` | `*anthropic*` | Claude 3.5 Sonnet |

#### 4.3.2 VLA Policy

**Architecture:** Direct vision → action model.

- **OpenVLA (7B):** Trained on Open X-Embodiment (970k demonstrations). Outputs 7-DoF actions; we use `[0]` (forward) and `[5]` (yaw) for navigation.
- **Heuristic+:** Enhanced fallback with visual servoing (ORB feature displacement), frontier exploration (alternating search directions), corridor-following, and 5-phase behavior selection.

#### 4.3.3 NoMaD Policy

**Architecture:** Diffusion model over action sequences.

- Takes context stack (5 frames) + goal image → predicts 8 future waypoints
- Pure-pursuit controller converts waypoints to velocity commands
- Trained on large-scale navigation data (GNM dataset)

### 4.4 Obstacle Avoidance

**File:** `modules/obstacle_avoidance.py`
**Purpose:** Monocular depth-based obstacle detection with temporal smoothing.

#### 4.4.1 Depth Backends

| Method | Model | Speed | Metric? | Notes |
|--------|-------|-------|---------|-------|
| **depth_anything** | Depth Anything V2 Base | ~25ms | Relative | **Default** — best speed/quality tradeoff |
| depth_pro | Apple Depth Pro | ~300ms | Absolute | Most accurate; use for precise navigation |
| simple_edge | Canny edge density | ~5ms | Proxy | No GPU needed; emergency fallback |

#### 4.4.2 Detection Pipeline

```
Depth map → [Temporal EMA smoothing (α=0.6, 3-frame window)]
          → Near-field (bottom 15%): obstacle detection
          → Mid-field (40%–85%): anticipatory slowdown
          → Split into L/C/R zones (thirds)
          → Narrow passage detection (both sides blocked, center clear)
          → Dynamic speed factor: exponential decay based on center occupancy
          → Steering bias: proportional to (left - right) occupancy imbalance
```

**Outputs per frame:**
- `speed_factor` ∈ [0, 1] — multiplied into forward velocity
- `steer_bias` ∈ [-1, 1] — added to angular velocity
- `emergency_stop` — if > 25% of near-field is occupied
- `narrow_passage` — detective flag for corridor behavior

### 4.5 Topological Memory

**File:** `modules/topological_memory.py`
**Purpose:** Online visual graph for exploration, backtracking, and global planning.

#### 4.5.1 Graph Structure

- **Nodes:** Keyframe images with visual features (HSV histogram + spatial grid)
- **Edges:** Bidirectional with traversal cost (time-based); reverse edges have 1.2x cost penalty
- **Loop Closures:** Detected via feature similarity when a new node matches a distant past node (≥ 5 node gap, ≥ 0.85 similarity)

#### 4.5.2 Node Creation Policy

A new node is created when ANY of:
1. Time since last node ≥ `min_node_distance` (default 2.0s)
2. Visual scene change > `scene_change_threshold` (default 0.25)
3. Forced (e.g., at checkpoint locations)

Maximum graph size: 500 nodes (oldest, least-visited nodes pruned).

#### 4.5.3 Operations

| Operation | Algorithm | Complexity |
|-----------|-----------|------------|
| **Update** | Feature extraction + node creation + loop closure check | O(N) per frame |
| **Path Planning** | A\* / Dijkstra (no spatial heuristic) | O(E log N) |
| **Backtracking** | Path stack reversal (last N nodes) | O(1) |
| **Frontier Detection** | Find nodes with ≤ 1 outgoing edge (dead ends) | O(N) |
| **Similar Node Search** | Brute-force cosine over all nodes | O(N) |

### 4.6 Recovery Behaviors

**File:** `modules/recovery.py`
**Purpose:** Escalating maneuvers when the robot is stuck.

**Stuck Detection:** Speed near zero for > 8s despite non-zero commands.

**Behavior Hierarchy (escalating):**

| Level | Behavior | Duration | Description |
|-------|----------|----------|-------------|
| 1 | Back up | 1.5s | Reverse at -0.4 speed |
| 2 | Random turn | 1.0s | Turn ±0.5 angular (random direction) |
| 3 | Wall follow | 3.0s | Follow nearest wall |
| 4 | Full rotation | 6.0s | 360° scan at 0.4 angular (re-localize) |

After exhausting all levels, the cycle restarts.

### 4.7 Agent Orchestrator

**File:** `agent.py`
**Purpose:** Main control loop, state machine, and module coordination.

**Key Parameters:**
- Loop frequency: 10 Hz
- Command smoothing: EMA with α = 0.4
- Max speed: 0.6 (configurable)
- Status logging: every 5 seconds

---

## 5. Model Selection Justification

### 5.1 Why DINOv2-VLAD over alternatives?

**Problem:** Visual Place Recognition (VPR) under viewpoint, lighting, and scale changes in unfamiliar indoor environments.

| Approach | Pros | Cons | Indoor VPR Recall@1 |
|----------|------|------|---------------------|
| **DINOv2-VLAD (ours)** | Universal (no VPR training needed), uses rich patch tokens, SOTA on indoor/outdoor/aerial | Higher-dim descriptor (24K) | ~90–95% (AnyLoc benchmarks) |
| NetVLAD | Classic VPR-trained | Requires training data, less universal | ~70–80% |
| EigenPlaces | Viewpoint-robust training | Trained on outdoor data | ~80–85% |
| CLIP cosine | Semantic understanding | No spatial awareness | ~60–70% |
| SuperGlue | Geometric verification | Slow (~200ms), fails on textureless walls | ~75% (as standalone) |

**Decision:** DINOv2-VLAD is the best zero-shot VPR method available. Upgrading to the register variant (Darcet et al., 2024) eliminates patch token artifacts, improving VLAD quality by 2–4%.

### 5.2 Why Qwen2.5-VL over alternatives?

**Problem:** High-level navigation reasoning from visual observations.

| Model | Size | Open? | Multi-Image | Structured Output | Quality |
|-------|------|-------|-------------|-------------------|---------|
| **Qwen2.5-VL 7B (ours)** | 7B | ✅ | ✅ Native | ✅ JSON | ★★★★★ |
| GPT-4o | ~200B? | ❌ | ✅ | ✅ | ★★★★★ |
| Claude 3.5 Sonnet | ~100B? | ❌ | ✅ | ✅ | ★★★★★ |
| Gemma 3 | 4B/12B | ✅ | ✅ | ✅ | ★★★★☆ |
| LLaVA 1.5 13B | 13B | ✅ | ❌ | ❌ | ★★★☆☆ |
| InternVL2 | 8B | ✅ | ✅ | ✅ | ★★★★☆ |

**Decision:** Qwen2.5-VL 7B is the best open-source VLM as of Feb 2025. It supports native multi-image input (current + goal), structured JSON output, and spatial reasoning. Serveable via vLLM for fast inference (~300ms/query on A100). The system also supports GPT-4o as a drop-in alternative.

### 5.3 Why Depth Anything V2 over alternatives?

**Problem:** Real-time monocular depth for obstacle avoidance.

| Model | Speed (GPU) | Metric? | Quality | Notes |
|-------|-------------|---------|---------|-------|
| **Depth Anything V2 Base (ours)** | ~25ms | Relative | ★★★★☆ | Best speed/quality tradeoff |
| Depth Anything V2 Large | ~100ms | Relative | ★★★★★ | Slower; unnecessary for obstacle avoidance |
| Depth Pro (Apple) | ~300ms | Absolute (metric) | ★★★★★ | Best accuracy but 10x slower |
| Marigold | ~2000ms | High-quality | ★★★★★ | Diffusion-based; far too slow for real-time |
| MiDaS v3.1 | ~50ms | Relative | ★★★☆☆ | Older; superseded by Depth Anything |

**Decision:** Depth Anything V2 Base at ~25ms per frame allows obstacle checking at 5+ Hz within our control loop. For precise metric navigation scenarios, Depth Pro is available as an alternative.

### 5.4 Why Topological Memory?

**Problem:** The robot needs to backtrack from dead ends and avoid revisiting explored areas.

**Alternatives considered:**
- **Metric SLAM** (ORB-SLAM3, etc.): Requires stereo or depth sensors; monocular SLAM is fragile indoors. Too complex for our constraint set.
- **No memory:** Robot wanders aimlessly after wrong turns; no ability to backtrack.
- **Topological graph (ours):** Lightweight, visual-only, enables A\* planning and backtracking without metric accuracy.

**Decision:** Topological memory provides the right abstraction level — enough structure for planning and recovery, without the fragility of metric SLAM. Inspired by works on visual topological navigation (VTN, SLING, Neural Topological SLAM).

---

## 6. Academic References

### 6.1 Visual Features & Place Recognition

1. **DINOv2** — Oquab, M., Darcet, T., et al. "DINOv2: Learning Robust Visual Features without Supervision." *arXiv:2304.07193*, 2023.
   - *Foundation of our visual feature extraction. Provides universal visual representations without task-specific training.*

2. **DINOv2 with Registers** — Darcet, T., Oquab, M., Mairal, J., Bojanowski, P. "Vision Transformers Need Registers." *ICLR 2024*, *arXiv:2309.16588*.
   - *Eliminates high-norm artifact tokens in ViTs by adding register tokens. Produces smoother, cleaner patch features — critical for our VLAD aggregation.*
   - **Update:** DINOv3 was released in August 2025; this repo now includes a `dinov3_vlad` toggle for A/B evaluation against this DINOv2 baseline.

3. **AnyLoc** — Keetha, N., Mishra, A., et al. "AnyLoc: Towards Universal Visual Place Recognition." *IEEE RA-L 2023*, *arXiv:2308.00688*. Presented at ICRA 2024.
   - *Demonstrates that DINOv2 patch tokens + VLAD aggregation achieves SOTA VPR across indoor, outdoor, aerial, underwater, and subterranean environments — up to 4x better than prior methods.*

4. **SigLIP 2** — Tschannen, M., et al. "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features." *arXiv:2502.14786*, Feb 2025.
   - *Google's SOTA vision encoder with sigmoid loss. Outperforms CLIP and original SigLIP on retrieval and dense prediction. Our secondary matching backend.*

5. **EigenPlaces** — Berton, G., et al. "EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition." *ICCV 2023*, *arXiv:2308.10832*.
   - *Place recognition model trained for viewpoint robustness. Available as an alternative backend.*

### 6.2 Vision-Language Models

6. **Qwen2.5-VL** — Bai, S., Chen, K., et al. "Qwen2.5-VL Technical Report." *arXiv:2502.13923*, Feb 2025.
   - *SOTA open VLM with native dynamic-resolution ViT, multi-image support, and structured output. Matches GPT-4o on document and diagram understanding.*

### 6.3 Depth Estimation

7. **Depth Anything V2** — Yang, L., et al. "Depth Anything V2." *NeurIPS 2024*, *arXiv:2406.09414*.
   - *SOTA monocular depth with synthetic training data. 10x faster than diffusion-based methods. Our primary depth model.*

8. **Depth Pro** — Bochkovskii, A., et al. "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second." *ICLR 2025*, *arXiv:2410.02073*.
   - *Apple's zero-shot metric depth model with absolute scale. Our high-accuracy alternative.*

### 6.4 Navigation Policies

9. **NoMaD** — Sridhar, A., Shah, D., Glossop, C., Levine, S. "NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration." *ICRA 2024*, *arXiv:2310.07896*.
   - *Diffusion policy over action sequences for goal-conditioned navigation. Handles both goal-reaching and exploration.*

10. **ViNT** — Shah, D., et al. "ViNT: A Foundation Model for Visual Navigation." *CoRL 2023*, *arXiv:2306.14846*.
    - *Foundation model for visual navigation trained on diverse robot data. Our NoMaD policy builds on this architecture.*

11. **OpenVLA** — Kim, M. J., et al. "OpenVLA: An Open-Source Vision-Language-Action Model." *arXiv:2406.09246*, 2024.
    - *7B VLA trained on Open X-Embodiment (970k demonstrations). Outperforms RT-2-X (55B) by 16.5%. Uses DINOv2 + SigLIP visual encoder — validates our feature choices.*

12. **Mobility VLA** — Chiang, H.-T. L., et al. "Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs." *arXiv:2407.07775*, 2024.
    - *Combines long-context VLM with topological graph navigation — validates our VLM + topo memory architecture.*

13. **Open X-Embodiment** — Open X-Embodiment Collaboration. "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." *ICRA 2024*, *arXiv:2310.08864*.
    - *Standardized robot learning dataset powering VLA training. Demonstrates positive transfer across robot embodiments.*

### 6.5 Topological Navigation

14. **SLING** — Chane-Sane, E., et al. "Goal-Conditioned Reinforcement Learning with Imagined Subgoals." *CoRL 2022*.
    - *Topological graph-based navigation with learned subgoal proposals.*

15. **Neural Topological SLAM** — Chaplot, D. S., et al. "Neural Topological SLAM for Visual Navigation." *NeurIPS 2020*.
    - *Builds explicit topological maps for long-horizon navigation in unseen environments.*

---

## 7. Deployment Guide

### 7.1 Hardware Requirements

**Minimum (heuristic mode):**
- CPU: Any modern x86_64
- RAM: 4 GB
- GPU: None
- Network: 4G/WiFi to robot

**Recommended (full SOTA):**
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (RTX 3070+)
- Network: Low-latency WiFi

**Optimal (VLM + VLA):**
- GPU: NVIDIA A100/H100 (for Qwen2.5-VL 7B via vLLM)
- Or: API access to GPT-4o/Claude

### 7.2 Software Setup

```bash
# 1. Clone and setup environment
conda create -n erv python=3.11
conda activate erv
conda install pytorch torchvision -c pytorch

# 2. Install dependencies
pip install -r requirements.txt
pip install -r indoor_nav/requirements_indoor.txt

# 3. Start the SDK server (connects to robot)
hypercorn main:app --reload

# 4. (Optional) Start VLM server
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 4096

# 5. Run the navigation agent
python indoor_nav/run_indoor.py \
    --goals indoor_nav/goals/ \
    --policy vlm_hybrid \
    --vlm-endpoint http://localhost:8000/v1 \
    --match-method dinov2_vlad
```

### 7.3 Quick Start Commands

| Scenario | Command |
|----------|---------|
| **Full SOTA** (Qwen2.5-VL + DINOv2-VLAD) | `python indoor_nav/run_indoor.py --goals goals/ --policy vlm_hybrid --vlm-endpoint http://localhost:8000/v1` |
| **No GPU** (heuristic only) | `python indoor_nav/run_indoor.py --goals goals/ --policy heuristic --obstacle-method simple_edge --device cpu` |
| **DINOv3 toggle** | `python indoor_nav/run_indoor.py --goals goals/ --policy vlm_hybrid --match-method dinov3_vlad` |
| **A/B matcher eval** | `python indoor_nav/eval_match_ab.py --goals-dir goals/ --queries-dir eval_queries/ --methods dinov2_vlad,dinov3_vlad` |
| **GPT-4o** (cloud VLM) | `python indoor_nav/run_indoor.py --goals goals/ --policy vlm_hybrid --vlm-endpoint https://api.openai.com/v1 --vlm-model gpt-4o --vlm-api-key sk-...` |
| **NoMaD** (diffusion policy) | `python indoor_nav/run_indoor.py --goals goals/ --policy nomad --match-method dinov2` |
| **VLA** (OpenVLA) | `python indoor_nav/run_indoor.py --goals goals/ --policy vla` |

---

## 8. Configuration Reference

### 8.1 IndoorNavConfig (top-level)

| Dataclass | Purpose |
|-----------|---------|
| `SDKConfig` | SDK server URL, endpoints, timeout |
| `ControlConfig` | Loop frequency, speed limits, smoothing, stuck detection |
| `PolicyConfig` | Policy backend, VLM settings, model paths |
| `GoalConfig` | Matching method, thresholds, feature model |
| `ObstacleConfig` | Depth model, detection zones, slowdown parameters |
| `TopoMemoryConfig` | Graph parameters, loop closure, max nodes |
| `RecoveryConfig` | Behavior sequence, durations, speeds |
| `LogConfig` | HDF5 logging, frame saving |

### 8.2 Key Parameters to Tune

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `goal.match_threshold` | 0.78 | 0.6–0.9 | Higher = fewer false arrivals, may miss goals |
| `goal.match_patience` | 3 | 1–10 | Frames above threshold needed; higher = more reliable |
| `control.max_linear` | 0.6 | 0.2–1.0 | Robot speed; lower = safer, higher = faster completion |
| `policy.vlm_query_interval` | 2.5s | 1.0–5.0 | VLM query frequency; lower = more responsive |
| `obstacle.min_clearance_frac` | 0.15 | 0.1–0.3 | Size of near-field detection zone |
| `obstacle.emergency_stop_frac` | 0.25 | 0.15–0.4 | Occupancy fraction that triggers emergency stop |
| `topo_memory.min_node_distance` | 2.0s | 0.5–5.0 | Graph density; lower = more nodes |
| `topo_memory.loop_closure_threshold` | 0.85 | 0.7–0.95 | Similarity for loop closure detection |
| `control.smoothing_alpha` | 0.4 | 0.1–0.8 | Command smoothing; lower = smoother |

---

## 9. File Structure

```
indoor_nav/
├── SPECIFICATIONS.md           ← This document
├── README.md                   ← Quick-start guide
├── __init__.py
├── configs/
│   └── config.py               ← All configuration dataclasses
├── modules/
│   ├── __init__.py
│   ├── sdk_client.py           ← Async HTTP client for Earth Rovers SDK
│   ├── checkpoint_manager.py   ← Goal matching (DINOv2-VLAD, DINOv3-VLAD, SigLIP2, etc.)
│   ├── obstacle_avoidance.py   ← Monocular depth obstacle detection
│   ├── topological_memory.py   ← Visual graph for backtracking & planning
│   └── recovery.py             ← Stuck detection & recovery maneuvers
├── policies/
│   ├── __init__.py
│   ├── base_policy.py          ← Abstract policy interface
│   ├── vlm_hybrid_policy.py    ← VLM reasoning + reactive controller
│   ├── vla_policy.py           ← Vision-Language-Action model integration
│   └── nomad_policy.py         ← NoMaD diffusion policy
├── goals/                      ← Goal images (per mission)
├── eval_queries/               ← Optional query set for matcher A/B benchmarks
├── logs/                       ← HDF5 telemetry logs
├── run_indoor.py               ← CLI entry point
├── eval_match_ab.py            ← A/B matcher benchmark
├── test_integration.py         ← Integration test suite
└── requirements_indoor.txt     ← Python dependencies
```

---

## Appendix A: DINOv3 Update and Migration Notes

As of August 2025, **DINOv3 exists** and is publicly available (with accepted-access model weights and a dedicated DINOv3 license). This repository now supports:

1. `dinov2_vlad` (default baseline)
2. `dinov3_vlad` (new toggle)

We keep `dinov2_vlad` as default for stability and reproducibility because existing indoor thresholds and recovery behavior were tuned on DINOv2-registers features.

Migration notes for `dinov3_vlad`:

- Re-tune `match_threshold` and `match_patience` before competition runs.
- Re-benchmark runtime on your deployment hardware (model size and latency differ).
- Use the provided A/B script (`indoor_nav/eval_match_ab.py`) before switching defaults.

---

## Appendix B: Comparison to Mobility VLA Architecture

Our design closely mirrors the architecture validated by Google's Mobility VLA (Chiang et al., 2024), which also uses:
- A **high-level VLM** for goal interpretation and reasoning
- A **topological graph** for low-level navigation
- **Image-goal conditioning** for checkpoint matching

Key differences:
| Aspect | Mobility VLA | Our System |
|--------|-------------|------------|
| VLM | PaLM 2 (proprietary) | Qwen2.5-VL 7B (open) |
| Topo graph | Pre-built from demo video | Built online during exploration |
| Low-level | Graph-based controller | Reactive + obstacle avoidance |
| Scale | 836 m² indoor office | Competition-specific venues |

Our open-source, online-construction approach is better suited to the competition's zero-prior-knowledge constraint.

---

*End of Specifications Report*
