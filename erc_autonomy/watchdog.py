from __future__ import annotations

import time
from typing import Awaitable, Callable


class StaleSensorWatchdog:
    """
    Triggers emergency-stop callback if sensor updates stop arriving.

    The callback is rate-limited so bursty timeout checks do not spam control.
    """

    def __init__(
        self,
        stale_after_ms: int,
        on_stale: Callable[[], Awaitable[None]],
        min_trigger_interval_s: float = 0.8,
    ):
        self.stale_after_s = max(0.05, stale_after_ms / 1000.0)
        self.on_stale = on_stale
        self.min_trigger_interval_s = max(0.1, min_trigger_interval_s)
        self.last_sensor_monotonic = time.monotonic()
        self.last_trigger_monotonic = 0.0

    def mark_sensor(self) -> None:
        self.last_sensor_monotonic = time.monotonic()

    async def tick(self) -> bool:
        now = time.monotonic()
        stale = (now - self.last_sensor_monotonic) > self.stale_after_s
        if not stale:
            return False

        if (now - self.last_trigger_monotonic) < self.min_trigger_interval_s:
            return False

        self.last_trigger_monotonic = now
        await self.on_stale()
        return True

