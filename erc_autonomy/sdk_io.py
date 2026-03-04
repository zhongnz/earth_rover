from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any, Dict, Optional

import aiohttp
import cv2
import numpy as np

from .config import ERCConfig
from .types import DriveCommand, SensorPacket

logger = logging.getLogger(__name__)


class SDKIO:
    """Async wrapper around Earth Rovers SDK endpoints with safety checks."""

    def __init__(self, cfg: ERCConfig):
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_frame_ts = 0.0
        self._last_data_ts = 0.0
        self._last_cmd_monotonic = 0.0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self.base}{path}"
        try:
            if method == "GET":
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            if method == "POST":
                async with session.post(url, json=payload, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            raise ValueError(f"Unsupported method: {method}")
        except aiohttp.ClientError as exc:
            logger.debug("request failed", extra={"extra_data": {"url": url, "err": str(exc)}})
            return {}
        except asyncio.TimeoutError:
            logger.debug("request timeout", extra={"extra_data": {"url": url}})
            return {}

    async def start_mission(self) -> Dict[str, Any]:
        return await self._request_json("POST", "/start-mission")

    async def end_mission(self) -> Dict[str, Any]:
        return await self._request_json("POST", "/end-mission")

    async def get_checkpoints(self) -> Dict[str, Any]:
        return await self._request_json("GET", "/checkpoints-list")

    async def checkpoint_reached(self) -> Dict[str, Any]:
        return await self._request_json("POST", "/checkpoint-reached")

    @staticmethod
    def _decode_frame(frame_b64: str) -> Optional[np.ndarray]:
        if not frame_b64:
            return None
        try:
            if frame_b64.startswith("data:"):
                frame_b64 = frame_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_b64, validate=False)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            if arr.size == 0:
                return None
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    async def poll(self) -> Optional[SensorPacket]:
        now = time.time()
        screenshot_task = asyncio.create_task(self._request_json("GET", "/v2/screenshot"))
        data_task = asyncio.create_task(self._request_json("GET", "/data"))
        screenshot, telemetry = await asyncio.gather(screenshot_task, data_task)

        if not screenshot and not telemetry:
            return None

        frame_ts = float(screenshot.get("timestamp", now)) if screenshot else now
        data_ts = float(telemetry.get("timestamp", now)) if telemetry else now

        # Drop obviously stale regressions while allowing equal timestamps once.
        if frame_ts < self._last_frame_ts:
            logger.debug(
                "dropping out-of-order frame",
                extra={"extra_data": {"frame_ts": frame_ts, "last_frame_ts": self._last_frame_ts}},
            )
            return None
        if data_ts < self._last_data_ts:
            logger.debug(
                "dropping out-of-order telemetry",
                extra={"extra_data": {"data_ts": data_ts, "last_data_ts": self._last_data_ts}},
            )
            return None

        self._last_frame_ts = max(self._last_frame_ts, frame_ts)
        self._last_data_ts = max(self._last_data_ts, data_ts)

        frame = None
        if screenshot:
            frame = self._decode_frame(screenshot.get("front_frame", ""))

        source_ts = max(frame_ts, data_ts)
        latency_ms = max(0.0, (now - source_ts) * 1000.0)
        return SensorPacket(
            received_at=now,
            frame_ts=frame_ts,
            data_ts=data_ts,
            source_latency_ms=latency_ms,
            frame_bgr=frame,
            raw_data=telemetry or {},
        )

    async def send_control(self, cmd: DriveCommand) -> bool:
        min_interval = 1.0 / max(1.0, self.cfg.min_command_hz)
        now_mono = time.monotonic()
        if (now_mono - self._last_cmd_monotonic) < (min_interval * 0.5):
            # Soft-rate limiting for accidental command bursts.
            await asyncio.sleep(min_interval * 0.5)
        self._last_cmd_monotonic = time.monotonic()

        payload = {
            "command": {
                "linear": float(np.clip(cmd.linear, -1.0, 1.0)),
                "angular": float(np.clip(cmd.angular, -1.0, 1.0)),
                "lamp": int(1 if cmd.lamp else 0),
            }
        }
        response = await self._request_json("POST", "/control", payload=payload)
        return bool(response)

    async def safe_stop(self, duration_s: float, hz: float) -> None:
        interval = 1.0 / max(1.0, hz)
        stop_cmd = DriveCommand(linear=0.0, angular=0.0, lamp=0)
        end_t = time.monotonic() + max(0.1, duration_s)
        while time.monotonic() < end_t:
            await self.send_control(stop_cmd)
            await asyncio.sleep(interval)

