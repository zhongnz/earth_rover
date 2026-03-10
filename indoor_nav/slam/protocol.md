# ORB-SLAM3 Sidecar Protocol

This document defines the local HTTP contract expected by
`indoor_nav/slam/orbslam3_client.py`.

The protocol is intentionally narrow:

- Python owns control-loop orchestration.
- The sidecar owns SLAM state, pose estimation, and map lifecycle.
- The contract must stay stable even if the native ORB-SLAM3 wrapper changes.

## Base URL

Default:

```text
http://127.0.0.1:8765
```

## Tracking States

Allowed values:

- `NOT_INITIALIZED`
- `OK`
- `LOST`

## Endpoints

### `GET /health`

Purpose:
- service liveness
- configuration sanity
- asset readiness

Response:

```json
{
  "ok": true,
  "backend": "orbslam3",
  "mode": "mono",
  "vocab_loaded": true,
  "settings_loaded": true
}
```

### `GET /status`

Purpose:
- latest sidecar state without pushing a new frame

Response:

```json
{
  "ok": true,
  "tracking_state": "OK",
  "frame_ts": 1710000000.12,
  "pose": {
    "tx": 0.43,
    "ty": -0.02,
    "tz": 1.18,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.12,
    "qw": 0.99
  },
  "keyframe_id": 17,
  "loop_closure_count": 1,
  "map_id": 0
}
```

### `POST /track`

Purpose:
- push one frame into the SLAM backend
- receive the latest pose/tracking result

Request:
- content type: `multipart/form-data`
- fields:
  - `timestamp`: float
  - `frame_jpeg`: binary JPEG image
  - `imu_json`: optional JSON payload for future `mono_inertial` support

Success response:

```json
{
  "ok": true,
  "tracking_state": "OK",
  "frame_ts": 1710000000.12,
  "pose": {
    "tx": 0.43,
    "ty": -0.02,
    "tz": 1.18,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.12,
    "qw": 0.99
  },
  "keyframe_id": 17,
  "loop_closure_count": 1,
  "map_id": 0
}
```

Failure response:

```json
{
  "ok": false,
  "tracking_state": "LOST",
  "frame_ts": 1710000000.12,
  "pose": null,
  "keyframe_id": null,
  "loop_closure_count": 1,
  "map_id": 0,
  "error": "tracking lost"
}
```

### `POST /reset`

Purpose:
- clear map and reinitialize tracking

Request body:

```json
{}
```

Response:

```json
{
  "ok": true,
  "tracking_state": "NOT_INITIALIZED",
  "frame_ts": 0.0,
  "pose": null,
  "keyframe_id": null,
  "loop_closure_count": 0,
  "map_id": 1
}
```

### `POST /shutdown`

Purpose:
- graceful shutdown hook

Request body:

```json
{}
```

Response:

```json
{
  "ok": true
}
```

## Semantics

### Pose frame

The sidecar returns the SLAM-native camera pose as translation + quaternion.

For `mono` mode:
- pose is relative
- scale may not be metric

For `mono_inertial` mode:
- metric scale is expected if calibration and timing are correct

## Safety requirements

- If `tracking_state != "OK"`, the Python agent must treat the pose as invalid.
- If `/track` is unreachable or stale, the Python agent must degrade safely.
- A `200` response does not imply usable tracking; only `tracking_state == "OK"` does.

## Mock sidecar

The repository includes a mock implementation at:

```text
indoor_nav/slam/mock_sidecar.py
```

Use it to validate:
- preflight
- client wiring
- startup/shutdown

Launch it directly with:

```bash
python -m indoor_nav.slam.mock_sidecar --host 127.0.0.1 --port 8765
```

Do not treat it as a real SLAM backend.
