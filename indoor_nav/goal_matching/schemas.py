from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PreparedImage:
    """Backend-specific prepared representation of an image."""

    payload: Any
    image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
