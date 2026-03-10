from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from indoor_nav.slam.types import SlamStatus


class SlamBackend(ABC):
    """Abstract client interface for an external SLAM backend."""

    @abstractmethod
    async def start(self) -> None:
        """Verify the backend is reachable and ready."""

    @abstractmethod
    async def status(self) -> SlamStatus:
        """Return the latest SLAM tracking status."""

    @abstractmethod
    async def track(
        self,
        image: np.ndarray,
        timestamp: float,
        imu: Optional[Any] = None,
    ) -> SlamStatus:
        """Push one frame into the backend and return the latest pose/tracking state."""

    @abstractmethod
    async def reset(self) -> None:
        """Reset the backend map/tracking state."""

    @abstractmethod
    async def close(self) -> None:
        """Release backend resources."""
