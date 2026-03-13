from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


SensorSample = tuple[float, tuple[float, float, float]]


@dataclass(frozen=True)
class MonoInertialClockAlignment:
    offset_s: float = 0.0
    latest_sensor_ts: float | None = None
    frame_sensor_delta_s: float | None = None
    data_ts: float | None = None
    frame_data_delta_s: float | None = None

    @property
    def needs_correction(self) -> bool:
        return abs(self.offset_s) > 1e-9


def _normalize_stream(
    rows: Iterable[Sequence[float]] | None,
    *,
    min_timestamp: float,
    max_timestamp: float,
) -> list[SensorSample]:
    samples: list[SensorSample] = []
    if not rows:
        return samples

    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        try:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            timestamp = float(row[3])
        except (TypeError, ValueError):
            continue

        if not all(math.isfinite(value) for value in (x, y, z, timestamp)):
            continue
        if timestamp <= min_timestamp or timestamp > max_timestamp:
            continue

        samples.append((timestamp, (x, y, z)))

    samples.sort(key=lambda item: item[0])
    deduped: list[SensorSample] = []
    for timestamp, values in samples:
        if deduped and math.isclose(timestamp, deduped[-1][0], rel_tol=0.0, abs_tol=1e-9):
            deduped[-1] = (timestamp, values)
        else:
            deduped.append((timestamp, values))
    return deduped


def _latest_timestamp(rows: Iterable[Sequence[float]] | None) -> float | None:
    latest: float | None = None
    if not rows:
        return None

    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        try:
            timestamp = float(row[3])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(timestamp):
            continue
        latest = timestamp if latest is None else max(latest, timestamp)

    return latest


def estimate_mono_inertial_clock_alignment(
    accels: Iterable[Sequence[float]] | None,
    gyros: Iterable[Sequence[float]] | None,
    *,
    frame_ts: float,
    data_ts: float | None = None,
    skew_threshold_s: float = 2.0,
) -> MonoInertialClockAlignment:
    """
    Detect a large timestamp-base mismatch between camera frames and IMU telemetry.

    The rover SDK is expected to report both `/data` and `/v2/front` timestamps in
    the same epoch. When that contract is broken, the IMU samples must be shifted
    into the camera clock before they are sent to ORB-SLAM3.
    """
    frame_ts = float(frame_ts)
    latest_sensor_ts = None
    for candidate in (_latest_timestamp(gyros), _latest_timestamp(accels)):
        if candidate is None:
            continue
        latest_sensor_ts = candidate if latest_sensor_ts is None else max(latest_sensor_ts, candidate)

    frame_sensor_delta_s = None
    if latest_sensor_ts is not None:
        frame_sensor_delta_s = frame_ts - latest_sensor_ts

    valid_data_ts: float | None = None
    frame_data_delta_s: float | None = None
    if data_ts is not None:
        try:
            candidate = float(data_ts)
        except (TypeError, ValueError):
            candidate = math.nan
        if math.isfinite(candidate):
            valid_data_ts = candidate
            frame_data_delta_s = frame_ts - candidate

    offset_s = 0.0
    if frame_sensor_delta_s is not None and abs(frame_sensor_delta_s) >= skew_threshold_s:
        offset_s = frame_sensor_delta_s
    elif frame_data_delta_s is not None and abs(frame_data_delta_s) >= skew_threshold_s and latest_sensor_ts is None:
        offset_s = frame_data_delta_s

    return MonoInertialClockAlignment(
        offset_s=offset_s,
        latest_sensor_ts=latest_sensor_ts,
        frame_sensor_delta_s=frame_sensor_delta_s,
        data_ts=valid_data_ts,
        frame_data_delta_s=frame_data_delta_s,
    )


def _lerp_vec3(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return (
        left[0] + (right[0] - left[0]) * alpha,
        left[1] + (right[1] - left[1]) * alpha,
        left[2] + (right[2] - left[2]) * alpha,
    )


def build_mono_inertial_payload(
    accels: Iterable[Sequence[float]] | None,
    gyros: Iterable[Sequence[float]] | None,
    *,
    frame_ts: float,
    last_imu_ts: float = 0.0,
    timestamp_offset_s: float = 0.0,
) -> tuple[dict, float]:
    """
    Build ORB-SLAM3 IMU samples for one camera frame.

    The rover SDK exposes accel and gyro samples as separate timestamped streams.
    ORB-SLAM3 expects fused IMU points, so this helper filters out already-used
    samples, clips them to the current frame timestamp, and interpolates accel
    values onto gyro timestamps. `last_imu_ts` stays in the raw IMU clock; use
    `timestamp_offset_s` to shift emitted sample timestamps into the frame clock.
    """
    raw_frame_ts = float(frame_ts) - float(timestamp_offset_s)
    accel_stream = _normalize_stream(accels, min_timestamp=float("-inf"), max_timestamp=raw_frame_ts)
    gyro_stream = _normalize_stream(gyros, min_timestamp=last_imu_ts, max_timestamp=raw_frame_ts)

    if not accel_stream or not gyro_stream:
        return {"samples": []}, last_imu_ts

    samples: list[dict] = []
    accel_index = 0
    newest_timestamp = last_imu_ts

    for gyro_timestamp, gyro_values in gyro_stream:
        while accel_index + 1 < len(accel_stream) and accel_stream[accel_index + 1][0] <= gyro_timestamp:
            accel_index += 1

        if len(accel_stream) == 1 or gyro_timestamp <= accel_stream[0][0]:
            accel_values = accel_stream[0][1]
        elif accel_index + 1 >= len(accel_stream):
            accel_values = accel_stream[-1][1]
        else:
            left_timestamp, left_values = accel_stream[accel_index]
            right_timestamp, right_values = accel_stream[accel_index + 1]
            if right_timestamp <= left_timestamp:
                accel_values = right_values
            else:
                alpha = (gyro_timestamp - left_timestamp) / (right_timestamp - left_timestamp)
                alpha = max(0.0, min(1.0, alpha))
                accel_values = _lerp_vec3(left_values, right_values, alpha)

        samples.append(
            {
                "t": gyro_timestamp + float(timestamp_offset_s),
                "ax": accel_values[0],
                "ay": accel_values[1],
                "az": accel_values[2],
                "gx": gyro_values[0],
                "gy": gyro_values[1],
                "gz": gyro_values[2],
            }
        )
        newest_timestamp = max(newest_timestamp, gyro_timestamp)

    return {"samples": samples}, newest_timestamp
