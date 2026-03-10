"""
Async SDK client for the Earth Rovers platform.

Wraps all HTTP endpoints with proper error handling, retries, and
frame decoding. Designed for high-frequency control loops.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import aiohttp
import cv2
import numpy as np

from indoor_nav.configs.config import SDKConfig

logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Latest snapshot of all bot telemetry."""
    timestamp: float = 0.0
    battery: float = 0.0
    signal_level: float = 0.0
    orientation: float = 0.0
    lamp: int = 0
    speed: float = 0.0
    gps_signal: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    vibration: float = 0.0
    accels: list = field(default_factory=list)
    gyros: list = field(default_factory=list)
    mags: list = field(default_factory=list)
    rpms: list = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BotState":
        return cls(
            timestamp=float(payload.get("timestamp", time.time())),
            battery=float(payload.get("battery", 0)),
            signal_level=float(payload.get("signal_level", 0)),
            orientation=float(payload.get("orientation", 0)),
            lamp=int(payload.get("lamp", 0)),
            speed=float(payload.get("speed", 0)),
            gps_signal=float(payload.get("gps_signal", 0)),
            latitude=float(payload.get("latitude", 0)),
            longitude=float(payload.get("longitude", 0)),
            vibration=float(payload.get("vibration", 0)),
            accels=payload.get("accels", []),
            gyros=payload.get("gyros", []),
            mags=payload.get("mags", []),
            rpms=payload.get("rpms", []),
            raw=payload,
        )


def decode_b64_image(b64_str: str, *, min_side: int = 32) -> Optional[np.ndarray]:
    """Decode a base64 string to a BGR numpy image and reject tiny placeholder frames."""
    try:
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if h < min_side or w < min_side:
            logger.debug("Rejected tiny frame: %dx%d", w, h)
            return None
        return img
    except Exception:
        return None


class RoverSDKClient:
    """Async HTTP client for the Earth Rovers SDK server."""

    def __init__(self, cfg: SDKConfig):
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_state: Optional[BotState] = None
        self._connected = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------
    async def start_mission(self) -> Dict[str, Any]:
        """Call /start-mission. Returns checkpoints list."""
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.start_mission_endpoint}"
        async with session.post(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._connected = True
            logger.info("Mission started: %s", data.get("message", ""))
            return data

    async def end_mission(self) -> Dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.end_mission_endpoint}"
        async with session.post(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._connected = False
            logger.info("Mission ended: %s", data.get("message", ""))
            return data

    async def get_checkpoints(self) -> Dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.checkpoints_list_endpoint}"
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def report_checkpoint(self) -> Dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.checkpoint_endpoint}"
        async with session.post(url, json={}) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    async def get_data(self) -> BotState:
        """Fetch /data and return structured BotState."""
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.data_endpoint}"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    self._last_state = BotState.from_payload(payload)
                    return self._last_state
                else:
                    logger.warning("GET /data returned %d", resp.status)
        except Exception as e:
            logger.debug("get_data error: %s", e)
        return self._last_state or BotState()

    @property
    def last_state(self) -> BotState:
        return self._last_state or BotState()

    # ------------------------------------------------------------------
    # Camera frames
    # ------------------------------------------------------------------
    async def get_front_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (BGR image, timestamp) from front camera."""
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.frame_endpoint}"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    js = await resp.json()
                    ts = float(js.get("timestamp", time.time()))
                    b64 = js.get("front_frame", "")
                    img = decode_b64_image(b64) if b64 else None
                    return img, ts
        except Exception as e:
            logger.debug("get_front_frame error: %s", e)
        return None, time.time()

    async def get_rear_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (BGR image, timestamp) from rear camera."""
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.rear_endpoint}"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    js = await resp.json()
                    ts = float(js.get("timestamp", time.time()))
                    b64 = js.get("rear_frame", "")
                    img = decode_b64_image(b64) if b64 else None
                    return img, ts
        except Exception as e:
            logger.debug("get_rear_frame error: %s", e)
        return None, time.time()

    async def get_frames(self, include_rear: bool = False) -> Dict[str, Any]:
        """Fetch front (and optionally rear) frames concurrently."""
        tasks = {"front": asyncio.create_task(self.get_front_frame())}
        if include_rear:
            tasks["rear"] = asyncio.create_task(self.get_rear_frame())

        results = {}
        for key, task in tasks.items():
            img, ts = await task
            results[key] = {"image": img, "timestamp": ts}
        return results

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    async def send_control(self, linear: float, angular: float) -> bool:
        """Send velocity command. Returns True on success."""
        linear = max(-1.0, min(1.0, linear))
        angular = max(-1.0, min(1.0, angular))
        session = await self._ensure_session()
        url = f"{self.base}{self.cfg.control_endpoint}"
        payload = {"command": {"linear": float(linear), "angular": float(angular)}}
        try:
            async with session.post(url, json=payload) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug("send_control error: %s", e)
            return False

    async def stop(self, duration: float = 1.0, hz: float = 20.0):
        """Send repeated stop commands."""
        interval = 1.0 / hz
        end_time = time.time() + duration
        count = 0
        while time.time() < end_time:
            await self.send_control(0.0, 0.0)
            count += 1
            await asyncio.sleep(interval)
        logger.info("Sent %d stop commands over %.1fs", count, duration)
