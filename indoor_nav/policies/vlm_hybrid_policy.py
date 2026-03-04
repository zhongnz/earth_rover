"""
VLM-Hybrid Navigation Policy — SOTA 2025.

Combines a Vision-Language Model for high-level reasoning with a reactive
low-level controller. Supports multiple VLM backends:

  - Qwen2.5-VL 7B (Alibaba, Feb 2025, arXiv:2502.13923): Best open VLM,
    native multi-image, structured output. RECOMMENDED for competition.
  - GPT-4o / Claude (via OpenAI-compatible API): Best overall quality,
    requires internet. Good for testing.
  - Gemma 3 (Google, 2025): Excellent and efficient.
  - LLaVA (legacy): Ollama-only fallback.

Architecture:
  HIGH-LEVEL (VLM, ~0.3-0.5 Hz): Analyzes current + goal image(s), outputs
    structured JSON with action, spatial reasoning, and landmark detection.
  LOW-LEVEL (reactive, 10 Hz): Converts high-level instruction + obstacle
    state into smooth velocity commands.

Key improvements over LLaVA baseline:
  - Multi-image native support (current frame + goal + optional context)
  - Structured JSON output (more reliable than free-text parsing)
  - Spatial reasoning prompts (distance/direction estimation)
  - Chain-of-thought for complex indoor scenes
  - Adaptive query frequency based on scene change (HSV histogram)

Design validated by Mobility VLA (Chiang et al., arXiv:2407.07775), which
demonstrates that hierarchical VLM + topological controller achieves high
success rates on real-world navigation tasks.

References:
  [1] Bai et al., \"Qwen2.5-VL Technical Report\", arXiv:2502.13923, 2025.
  [2] Chiang et al., \"Mobility VLA\", arXiv:2407.07775, 2024.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from collections import deque
from typing import Dict, List, Optional

import aiohttp
import cv2
import numpy as np
from PIL import Image

from indoor_nav.configs.config import PolicyConfig
from indoor_nav.policies.base_policy import BasePolicy, PolicyInput, PolicyOutput

logger = logging.getLogger(__name__)


# High-level instruction set the VLM can output
INSTRUCTION_SET = {
    "forward": (0.4, 0.0),
    "forward_slow": (0.2, 0.0),
    "forward_fast": (0.55, 0.0),
    "turn_left": (0.1, 0.4),
    "turn_left_sharp": (0.0, 0.6),
    "turn_right": (0.1, -0.4),
    "turn_right_sharp": (0.0, -0.6),
    "veer_left": (0.3, 0.2),      # gentle curve left
    "veer_right": (0.3, -0.2),    # gentle curve right
    "stop": (0.0, 0.0),
    "reverse": (-0.3, 0.0),
    "search_left": (0.0, 0.3),
    "search_right": (0.0, -0.3),
    "approach": (0.15, 0.0),       # very slow final approach
}


SYSTEM_PROMPT = """You are a navigation controller for a small indoor ground robot competing in the ICRA 2025 Earth Rover Challenge.

You receive images of what the robot sees NOW and the TARGET location it must reach.

Output ONLY a JSON object (no markdown, no commentary):
{{{{
  "action": "<one of: {actions}>",
  "reasoning": "<1 sentence: what you see and why this action>",
  "goal_visible": <true/false>,
  "landmarks": "<key visual features you observe (doors, signs, hallway features)>",
  "estimated_distance": "<near/medium/far/unknown>",
  "direction_to_goal": "<left/right/ahead/behind/unknown>",
  "confidence": <0.0 to 1.0>
}}}}

Navigation strategy:
- In HALLWAYS: prefer forward/forward_fast, follow the corridor direction
- At JUNCTIONS: compare landmarks between current view and goal to choose direction
- Near DOORS: slow down (forward_slow), check if the goal is through this door
- APPROACHING GOAL: use "approach" when very close, "stop" only when exactly at goal
- LOST: use search_left/search_right to scan environment
- Compare spatial layout, colors, textures, signage between current and goal images
- If current view shows similar architecture/features as goal → you're close""".format(
    actions=", ".join(INSTRUCTION_SET.keys())
)


class VLMHybridPolicy(BasePolicy):
    """
    VLM-guided navigation with reactive low-level control.

    The VLM runs asynchronously at low frequency; its latest instruction
    guides the reactive controller that runs at full control-loop rate.

    Supports three API formats:
      - "openai": OpenAI-compatible chat completions (Qwen2.5-VL, GPT-4o, vLLM)
      - "ollama": Ollama /api/generate format (LLaVA, Llama-Vision)
      - "anthropic": Anthropic messages API (Claude)
    """

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self._current_instruction: str = "forward"
        self._instruction_confidence: float = 0.0
        self._goal_visible: bool = False
        self._estimated_distance: str = "unknown"
        self._direction_to_goal: str = "unknown"
        self._landmarks: str = ""
        self._last_vlm_time: float = 0.0
        self._vlm_session: Optional[aiohttp.ClientSession] = None
        self._pending_vlm: bool = False
        self._context_queue: deque = deque(maxlen=cfg.context_length)
        self._scene_change_threshold: float = 0.15  # Adaptive query frequency
        self._last_frame_feature: Optional[np.ndarray] = None
        self._vlm_query_count: int = 0
        self._api_format: str = getattr(cfg, 'vlm_api_format', 'openai')

    def setup(self):
        # Auto-detect API format from endpoint
        endpoint = self.cfg.vlm_endpoint or ""
        if "11434" in endpoint or "/api/generate" in endpoint:
            self._api_format = "ollama"
        elif "anthropic" in endpoint:
            self._api_format = "anthropic"
        else:
            self._api_format = "openai"

        logger.info(
            "VLM Hybrid policy initialized (endpoint: %s, model: %s, format: %s, interval: %.1fs)",
            self.cfg.vlm_endpoint,
            self.cfg.vlm_model,
            self._api_format,
            self.cfg.vlm_query_interval,
        )

    def predict(self, obs: PolicyInput) -> PolicyOutput:
        """
        Main prediction: use latest VLM instruction + reactive adjustments.
        """
        self._context_queue.append(obs.front_image.copy())

        # Check if we should query the VLM
        now = time.time()
        time_ok = now - self._last_vlm_time >= self.cfg.vlm_query_interval
        scene_changed = self._check_scene_change(obs.front_image)

        if (
            (time_ok or scene_changed)
            and not self._pending_vlm
            and self.cfg.vlm_endpoint
        ):
            self._pending_vlm = True
            self._last_vlm_time = now

        # Get base action from current instruction
        linear, angular = INSTRUCTION_SET.get(
            self._current_instruction, (0.3, 0.0)
        )

        # Adaptive speed based on VLM context
        if self._estimated_distance == "near":
            linear = min(linear, 0.2)  # slow approach
        elif self._estimated_distance == "far":
            linear = max(linear, 0.35)  # speed up

        # Direction hints from VLM
        if self._direction_to_goal == "left" and angular >= 0:
            angular += 0.1
        elif self._direction_to_goal == "right" and angular <= 0:
            angular -= 0.1

        # Modulate based on goal similarity
        if obs.goal_similarity > 0.75:
            linear *= 0.4  # very slow near goal
        elif obs.goal_similarity > 0.6 and obs.goal_trend > 0:
            linear *= 0.7  # approaching — careful

        # Apply obstacle avoidance
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=self._instruction_confidence,
        )

    def _check_scene_change(self, frame: np.ndarray) -> bool:
        """Detect if scene changed significantly → trigger early VLM query."""
        # Use simple color histogram difference
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)

        if self._last_frame_feature is None:
            self._last_frame_feature = hist
            return False

        diff = np.sum(np.abs(hist - self._last_frame_feature))
        self._last_frame_feature = hist
        return diff > self._scene_change_threshold

    async def query_vlm(self, front_image: np.ndarray, goal_image: np.ndarray):
        """
        Query the VLM with current + goal images.
        Supports OpenAI-compatible, Ollama, and Anthropic API formats.
        """
        if not self.cfg.vlm_endpoint:
            self._pending_vlm = False
            return

        try:
            current_b64 = self._encode_image(front_image, max_size=768)
            goal_b64 = self._encode_image(goal_image, max_size=768)

            if self._vlm_session is None or self._vlm_session.closed:
                self._vlm_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )

            if self._api_format == "openai":
                await self._query_openai_format(current_b64, goal_b64)
            elif self._api_format == "ollama":
                await self._query_ollama_format(current_b64, goal_b64)
            elif self._api_format == "anthropic":
                await self._query_anthropic_format(current_b64, goal_b64)

            self._vlm_query_count += 1

        except Exception as e:
            logger.debug("VLM query error: %s", e)
        finally:
            self._pending_vlm = False

    async def _query_openai_format(self, current_b64: str, goal_b64: str):
        """
        OpenAI-compatible chat completions API.
        Works with: Qwen2.5-VL (vLLM/SGLang), GPT-4o, Gemma 3, local servers.
        """
        payload = {
            "model": self.cfg.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "CURRENT VIEW (robot's front camera):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"}
                        },
                        {
                            "type": "text",
                            "text": "GOAL (target location to reach):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{goal_b64}"}
                        },
                        {
                            "type": "text",
                            "text": "What action should the robot take? Output JSON only."
                        },
                    ],
                },
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        # Add response_format for models that support it (Qwen2.5-VL, GPT-4o)
        if any(m in self.cfg.vlm_model.lower() for m in ("qwen", "gpt-4", "gemma")):
            payload["response_format"] = {"type": "json_object"}

        endpoint = self.cfg.vlm_endpoint
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"

        headers = {"Content-Type": "application/json"}
        # Add API key if configured
        api_key = getattr(self.cfg, 'vlm_api_key', '')
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with self._vlm_session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                text = result["choices"][0]["message"]["content"]
                self._parse_vlm_response(text)
            else:
                body = await resp.text()
                logger.debug("VLM API error %d: %s", resp.status, body[:200])

    async def _query_ollama_format(self, current_b64: str, goal_b64: str):
        """Ollama /api/generate format (legacy support for LLaVA, etc.)."""
        prompt = (
            "CURRENT VIEW (robot's front camera):\n"
            "[See first image]\n\n"
            "GOAL (target location to navigate to):\n"
            "[See second image]\n\n"
            "What action should the robot take? Output JSON only."
        )

        payload = {
            "model": self.cfg.vlm_model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "images": [current_b64, goal_b64],
            "stream": False,
        }

        async with self._vlm_session.post(self.cfg.vlm_endpoint, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                response_text = result.get("response", "")
                self._parse_vlm_response(response_text)

    async def _query_anthropic_format(self, current_b64: str, goal_b64: str):
        """Anthropic Messages API for Claude models."""
        payload = {
            "model": self.cfg.vlm_model,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "CURRENT VIEW (robot's front camera):"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": current_b64,
                            },
                        },
                        {"type": "text", "text": "GOAL (target location to reach):"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": goal_b64,
                            },
                        },
                        {"type": "text", "text": "What action should the robot take? Output JSON only."},
                    ],
                },
            ],
        }

        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        api_key = getattr(self.cfg, 'vlm_api_key', '')
        if api_key:
            headers["x-api-key"] = api_key

        async with self._vlm_session.post(self.cfg.vlm_endpoint, json=payload, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                text = result["content"][0]["text"]
                self._parse_vlm_response(text)

    def _parse_vlm_response(self, text: str):
        """Parse VLM JSON response and update instruction."""
        try:
            text = text.strip()
            # Strip markdown code fences
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if "```" in text:
                    text = text[:text.rindex("```")]
            if text.startswith("json"):
                text = text[4:]

            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            data = json.loads(text)
            action = data.get("action", "forward")
            if action in INSTRUCTION_SET:
                self._current_instruction = action
                self._instruction_confidence = float(data.get("confidence", 0.5))
                self._goal_visible = bool(data.get("goal_visible", False))
                self._estimated_distance = str(data.get("estimated_distance", "unknown"))
                self._direction_to_goal = str(data.get("direction_to_goal", "unknown"))
                self._landmarks = str(data.get("landmarks", ""))
                logger.info(
                    "VLM[%d]: %s (conf=%.2f, goal=%s, dist=%s, dir=%s) — %s",
                    self._vlm_query_count,
                    action,
                    self._instruction_confidence,
                    self._goal_visible,
                    self._estimated_distance,
                    self._direction_to_goal,
                    data.get("reasoning", ""),
                )
            else:
                logger.warning("VLM returned unknown action: %s", action)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Failed to parse VLM response: %s | text: %s", e, text[:200])

    def _encode_image(self, image: np.ndarray, max_size: int = 768) -> str:
        """Resize and encode image to base64 JPEG."""
        h, w = image.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @property
    def needs_vlm_query(self) -> bool:
        return self._pending_vlm

    def reset(self):
        self._current_instruction = "forward"
        self._instruction_confidence = 0.0
        self._goal_visible = False
        self._estimated_distance = "unknown"
        self._direction_to_goal = "unknown"
        self._landmarks = ""
        self._context_queue.clear()
        self._pending_vlm = False
        self._last_frame_feature = None

    async def close(self):
        if self._vlm_session and not self._vlm_session.closed:
            await self._vlm_session.close()
