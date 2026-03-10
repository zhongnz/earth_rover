from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SlamPose:
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SlamPose":
        return cls(
            tx=float(payload.get("tx", 0.0)),
            ty=float(payload.get("ty", 0.0)),
            tz=float(payload.get("tz", 0.0)),
            qx=float(payload.get("qx", 0.0)),
            qy=float(payload.get("qy", 0.0)),
            qz=float(payload.get("qz", 0.0)),
            qw=float(payload.get("qw", 1.0)),
        )


@dataclass
class SlamStatus:
    ok: bool = False
    tracking_state: str = "NOT_INITIALIZED"
    frame_ts: float = 0.0
    pose: Optional[SlamPose] = None
    keyframe_id: Optional[int] = None
    loop_closure_count: int = 0
    map_id: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tracking(self) -> bool:
        return self.tracking_state == "OK"

    @property
    def is_lost(self) -> bool:
        return self.tracking_state == "LOST"

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SlamStatus":
        pose_payload = payload.get("pose")
        pose = SlamPose.from_payload(pose_payload) if isinstance(pose_payload, dict) else None
        keyframe_id = payload.get("keyframe_id")
        return cls(
            ok=bool(payload.get("ok", False)),
            tracking_state=str(payload.get("tracking_state", "NOT_INITIALIZED")),
            frame_ts=float(payload.get("frame_ts", 0.0)),
            pose=pose,
            keyframe_id=int(keyframe_id) if keyframe_id is not None else None,
            loop_closure_count=int(payload.get("loop_closure_count", 0)),
            map_id=int(payload.get("map_id", 0)),
            raw=payload,
        )
