from __future__ import annotations

import json
import logging
from typing import Any, Optional

import aiohttp
import cv2
import numpy as np

from indoor_nav.configs.config import SlamConfig
from indoor_nav.slam.base import SlamBackend
from indoor_nav.slam.types import SlamStatus

logger = logging.getLogger(__name__)


class ORBSLAM3Client(SlamBackend):
    """Async HTTP client for a local ORB-SLAM3 sidecar process."""

    def __init__(self, cfg: SlamConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_status = SlamStatus()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=max(2.0, self.cfg.pose_stale_timeout * 4.0))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _url(self, path: str) -> str:
        return f"{self.cfg.endpoint.rstrip('/')}{path}"

    def _prepare_frame(self, image: np.ndarray) -> np.ndarray:
        width = int(self.cfg.resize_width or 0)
        height = int(self.cfg.resize_height or 0)
        if width <= 0 or height <= 0:
            return image
        if image.shape[1] == width and image.shape[0] == height:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    async def start(self) -> None:
        session = await self._ensure_session()
        url = self._url("/health")
        async with session.get(url) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            if not payload.get("ok", False):
                raise RuntimeError(f"SLAM backend unhealthy: {payload}")

    async def status(self) -> SlamStatus:
        session = await self._ensure_session()
        url = self._url("/status")
        async with session.get(url) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        self._last_status = SlamStatus.from_payload(payload)
        return self._last_status

    async def track(
        self,
        image: np.ndarray,
        timestamp: float,
        imu: Optional[Any] = None,
    ) -> SlamStatus:
        session = await self._ensure_session()
        image = self._prepare_frame(image)
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame for SLAM sidecar")

        form = aiohttp.FormData()
        form.add_field("timestamp", f"{float(timestamp):.6f}")
        form.add_field(
            "frame_jpeg",
            encoded.tobytes(),
            filename="frame.jpg",
            content_type="image/jpeg",
        )
        if imu is not None:
            form.add_field("imu_json", json.dumps(imu), content_type="application/json")

        url = self._url("/track")
        async with session.post(url, data=form) as resp:
            resp.raise_for_status()
            payload = await resp.json()

        self._last_status = SlamStatus.from_payload(payload)
        return self._last_status

    async def reset(self) -> None:
        session = await self._ensure_session()
        url = self._url("/reset")
        async with session.post(url, json={}) as resp:
            resp.raise_for_status()
            await resp.read()
        self._last_status = SlamStatus()

    async def shutdown(self) -> None:
        session = await self._ensure_session()
        url = self._url("/shutdown")
        async with session.post(url, json={}) as resp:
            resp.raise_for_status()
            await resp.read()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
