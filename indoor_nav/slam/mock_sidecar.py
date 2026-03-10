from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np
from aiohttp import web

from indoor_nav.slam.types import SlamPose, SlamStatus


@dataclass
class MockState:
    tracking_state: str = "NOT_INITIALIZED"
    frame_ts: float = 0.0
    pose: Optional[SlamPose] = None
    keyframe_id: Optional[int] = None
    loop_closure_count: int = 0
    map_id: int = 0
    frame_count: int = 0


STATE = MockState()


def _status_payload(status: SlamStatus, **extra) -> dict:
    payload = {
        "ok": status.ok,
        "tracking_state": status.tracking_state,
        "frame_ts": status.frame_ts,
        "pose": asdict(status.pose) if status.pose else None,
        "keyframe_id": status.keyframe_id,
        "loop_closure_count": status.loop_closure_count,
        "map_id": status.map_id,
    }
    payload.update(extra)
    return payload


def _current_status(ok: bool = True) -> SlamStatus:
    return SlamStatus(
        ok=ok,
        tracking_state=STATE.tracking_state,
        frame_ts=STATE.frame_ts,
        pose=STATE.pose,
        keyframe_id=STATE.keyframe_id,
        loop_closure_count=STATE.loop_closure_count,
        map_id=STATE.map_id,
    )


async def health(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "backend": "orbslam3",
            "mode": "mono",
            "vocab_loaded": True,
            "settings_loaded": True,
        }
    )


async def status(_request: web.Request) -> web.Response:
    return web.json_response(_status_payload(_current_status()))


async def track(request: web.Request) -> web.Response:
    reader = await request.multipart()
    timestamp = None
    frame_raw = None

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "timestamp":
            timestamp = float(await part.text())
        elif part.name == "frame_jpeg":
            frame_raw = await part.read(decode=False)
        else:
            await part.release()

    if timestamp is None:
        return web.json_response(
            _status_payload(_current_status(ok=False), error="missing timestamp"),
            status=400,
        )
    if not frame_raw:
        return web.json_response(
            _status_payload(_current_status(ok=False), error="missing frame_jpeg"),
            status=400,
        )

    arr = np.frombuffer(frame_raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        status_obj = _current_status(ok=False)
        status_obj.tracking_state = "LOST"
        return web.json_response(_status_payload(status_obj, error="failed to decode JPEG"), status=400)

    STATE.frame_count += 1
    STATE.frame_ts = timestamp

    if STATE.tracking_state == "NOT_INITIALIZED":
        STATE.tracking_state = "OK"
        STATE.pose = SlamPose()
        STATE.keyframe_id = 0

    mean_intensity = float(img.mean())
    tx = STATE.frame_count * 0.05
    yaw_phase = (STATE.frame_count % 24) / 24.0
    qz = float(np.sin(np.pi * yaw_phase))
    qw = float(np.cos(np.pi * yaw_phase))
    STATE.pose = SlamPose(tx=tx, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=qz, qw=qw)

    if STATE.frame_count % 5 == 0:
        STATE.keyframe_id = (STATE.keyframe_id or 0) + 1
    if STATE.frame_count % 25 == 0:
        STATE.loop_closure_count += 1

    return web.json_response(_status_payload(_current_status(), mean_intensity=mean_intensity))


async def reset(_request: web.Request) -> web.Response:
    STATE.tracking_state = "NOT_INITIALIZED"
    STATE.frame_ts = 0.0
    STATE.pose = None
    STATE.keyframe_id = None
    STATE.loop_closure_count = 0
    STATE.frame_count = 0
    STATE.map_id += 1
    return web.json_response(_status_payload(_current_status()))


async def shutdown(_request: web.Request) -> web.Response:
    return web.json_response(
        {"ok": True, "note": "mock sidecar stays alive; stop the process manually when finished"}
    )


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/status", status)
    app.router.add_post("/track", track)
    app.router.add_post("/reset", reset)
    app.router.add_post("/shutdown", shutdown)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mock ORB-SLAM3 sidecar.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    web.run_app(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
