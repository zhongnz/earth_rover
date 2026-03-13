from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from indoor_nav.configs.config import IndoorNavConfig
from indoor_nav.modules.sdk_client import RoverSDKClient
from indoor_nav.slam.imu import build_mono_inertial_payload, estimate_mono_inertial_clock_alignment
from indoor_nav.slam.orbslam3_client import ORBSLAM3Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live ORB-SLAM3 probe with HTML trajectory output.")
    parser.add_argument("--sdk-url", default="http://127.0.0.1:8000", help="Earth Rover SDK base URL")
    parser.add_argument("--slam-endpoint", default="http://127.0.0.1:8766", help="ORB-SLAM3 sidecar base URL")
    parser.add_argument("--slam-mode", default="mono", choices=["mono", "mono_inertial"])
    parser.add_argument("--resize-width", type=int, default=1024)
    parser.add_argument("--resize-height", type=int, default=576)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--poll-hz", type=float, default=6.0, help="SLAM push frequency")
    parser.add_argument("--history", type=int, default=500, help="Maximum pose points kept in memory")
    parser.add_argument("--preview-max-width", type=int, default=960, help="Max width for the saved camera preview")
    parser.add_argument(
        "--out-dir",
        default="indoor_nav/logs/slam_live_latest",
        help="Directory for live HTML/JSON/CSV output",
    )
    return parser.parse_args()


def _prepare_frame(image, width: int, height: int):
    if image is None:
        return None
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 or height <= 0:
        return image
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _tracking_color(tracking_state: str, ok: bool) -> str:
    if ok or tracking_state == "OK":
        return "#2da44e"
    if tracking_state in {"RECENTLY_LOST", "NOT_INITIALIZED"}:
        return "#bf8700"
    return "#d1242f"


def _recent_state_dots(history: Iterable[dict], limit: int = 80) -> str:
    dots: list[str] = []
    for row in list(history)[-limit:]:
        title = html.escape(f"frame={row['frame_index']} state={row['tracking_state']}")
        color = _tracking_color(str(row.get("tracking_state", "UNKNOWN")), bool(row.get("ok")))
        dots.append(f'<span class="state-dot" style="background:{color}" title="{title}"></span>')
    if not dots:
        return '<span class="muted">No samples yet.</span>'
    return "".join(dots)


def _trajectory_stats(history: Iterable[dict]) -> dict:
    points = [(float(p["tx"]), float(p["tz"])) for p in history if p.get("tx") not in ("", None) and p.get("tz") not in ("", None)]
    if not points:
        return {
            "count": 0,
            "path_length": 0.0,
            "span_x": 0.0,
            "span_z": 0.0,
            "latest_tx": None,
            "latest_tz": None,
        }

    path_length = 0.0
    for idx in range(1, len(points)):
        dx = points[idx][0] - points[idx - 1][0]
        dz = points[idx][1] - points[idx - 1][1]
        path_length += math.hypot(dx, dz)

    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    return {
        "count": len(points),
        "path_length": path_length,
        "span_x": max(xs) - min(xs),
        "span_z": max(zs) - min(zs),
        "latest_tx": points[-1][0],
        "latest_tz": points[-1][1],
    }


def _pose_points(history: Iterable[dict], width: int = 720, height: int = 520, margin: int = 48) -> tuple[str, str, str]:
    points = [
        (float(p["tx"]), float(p["tz"]))
        for p in history
        if p.get("tx") not in ("", None) and p.get("tz") not in ("", None)
    ]
    if not points:
        empty = (
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            'font-size="20" fill="#6e7781">Waiting for tracked poses...</text>'
        )
        return "", "", empty

    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = max(max_x - min_x, 1e-6)
    span_z = max(max_z - min_z, 1e-6)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_z)
    draw_width = span_x * scale
    draw_height = span_z * scale
    offset_x = margin + (width - 2 * margin - draw_width) / 2.0
    offset_y = margin + (height - 2 * margin - draw_height) / 2.0

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, z = point
        px = offset_x + (x - min_x) * scale
        py = height - offset_y - (z - min_z) * scale
        return px, py

    projected = [project(p) for p in points]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in projected)
    sx, sy = projected[0]
    cx, cy = projected[-1]
    markers = (
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#1f883d" />'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#d1242f" stroke="#ffffff" stroke-width="2" />'
    )
    return polyline, markers, ""


def _write_preview_image(out_dir: Path, frame, latest: dict, max_width: int) -> None:
    if frame is None:
        return

    image = frame.copy()
    if max_width > 0 and image.shape[1] > max_width:
        scale = max_width / float(image.shape[1])
        new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    overlay = image.copy()
    band_height = min(max(72, image.shape[0] // 7), image.shape[0])
    cv2.rectangle(overlay, (0, 0), (image.shape[1], band_height), (18, 24, 33), thickness=-1)
    image = cv2.addWeighted(overlay, 0.48, image, 0.52, 0.0)

    pose = latest.get("pose") or {}
    keyframe = latest.get("keyframe_id")
    line_one = f"state={latest.get('tracking_state', 'UNKNOWN')}  frame={latest.get('frame_index', '-')}"
    line_two = (
        f"kf={keyframe if keyframe is not None else '-'}  "
        f"loop={latest.get('loop_closure_count', 0)}  "
        f"tx={_format_number(pose.get('tx'))} tz={_format_number(pose.get('tz'))}"
    )
    cv2.putText(image, line_one, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 247, 250), 2, cv2.LINE_AA)
    cv2.putText(image, line_two, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (209, 215, 224), 2, cv2.LINE_AA)

    cv2.imwrite(
        str(out_dir / "latest_frame.jpg"),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
    )


def _write_html(out_dir: Path, latest: dict, history: Iterable[dict]) -> None:
    polyline, markers, empty_state = _pose_points(history)
    stats = _trajectory_stats(history)
    state = html.escape(str(latest.get("tracking_state", "UNKNOWN")))
    pose = latest.get("pose")
    pose_text = "pose=None"
    if latest.get("pose"):
        pose_text = f"tx={pose['tx']:.3f}, ty={pose['ty']:.3f}, tz={pose['tz']:.3f}"
    state_color = _tracking_color(str(latest.get("tracking_state", "UNKNOWN")), bool(latest.get("ok")))
    state_dots = _recent_state_dots(history)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="1">
  <title>ORB-SLAM3 Live Visualization</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #0f1720;
      --panel: #fffaf2;
      --ink: #15202b;
      --muted: #5f6b76;
      --border: #d5c6b8;
      --accent: #006d77;
      --accent-soft: #83c5be;
      --path: #1d4ed8;
      --danger: #c2410c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(131, 197, 190, 0.45), transparent 28%),
        radial-gradient(circle at top right, rgba(238, 155, 0, 0.20), transparent 24%),
        linear-gradient(180deg, #f6efe6 0%, #efe7dc 100%);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: end;
      margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2.2rem; letter-spacing: 0.02em; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      border-radius: 999px;
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--border);
      font-weight: 700;
    }}
    .badge-dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: {state_color};
      box-shadow: 0 0 0 4px rgba(255,255,255,0.7);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 12px 32px rgba(15, 23, 32, 0.08);
    }}
    .label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .value {{ font-size: 1.25rem; font-weight: 700; margin-top: 6px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 1.2fr) minmax(340px, 0.9fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 16px;
      box-shadow: 0 18px 48px rgba(15, 23, 32, 0.10);
    }}
    .panel h2 {{ margin: 0 0 12px; font-size: 1.05rem; letter-spacing: 0.04em; text-transform: uppercase; }}
    .panel p {{ margin: 10px 0 0; color: var(--muted); }}
    .muted {{ color: var(--muted); }}
    .state-strip {{ display: flex; flex-wrap: wrap; gap: 6px; min-height: 20px; }}
    .state-dot {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid rgba(21, 32, 43, 0.08);
    }}
    code {{
      background: rgba(255,255,255,0.65);
      padding: 2px 6px;
      border-radius: 6px;
      border: 1px solid rgba(21, 32, 43, 0.06);
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 16px;
      border: 1px solid rgba(21, 32, 43, 0.08);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.9), rgba(246,239,230,0.95));
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 16px;
      border: 1px solid rgba(21, 32, 43, 0.08);
      background: #d8dee4;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .links a:hover {{ text-decoration: underline; }}
    @media (max-width: 1080px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .hero {{ flex-direction: column; align-items: start; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <h1>ORB-SLAM3 Live Visualization</h1>
        <p>Output directory: <code>{html.escape(str(out_dir))}</code></p>
      </div>
      <div class="badge"><span class="badge-dot"></span>{state}</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">Frame</div><div class="value">{latest.get('frame_index', '-')}</div></div>
      <div class="card"><div class="label">Timestamp</div><div class="value">{latest.get('frame_ts', 0.0):.3f}</div></div>
      <div class="card"><div class="label">Keyframe ID</div><div class="value">{latest.get('keyframe_id')}</div></div>
      <div class="card"><div class="label">Loop Closures</div><div class="value">{latest.get('loop_closure_count', 0)}</div></div>
      <div class="card"><div class="label">Pose Samples</div><div class="value">{stats['count']}</div></div>
      <div class="card"><div class="label">Path Length</div><div class="value">{stats['path_length']:.3f}</div></div>
      <div class="card"><div class="label">Map Span X</div><div class="value">{stats['span_x']:.3f}</div></div>
      <div class="card"><div class="label">Map Span Z</div><div class="value">{stats['span_z']:.3f}</div></div>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>Trajectory</h2>
        <svg viewBox="0 0 720 520" aria-label="SLAM trajectory">
          <defs>
            <pattern id="grid" width="56" height="56" patternUnits="userSpaceOnUse">
              <path d="M 56 0 L 0 0 0 56" fill="none" stroke="#d7cec5" stroke-width="1" />
            </pattern>
            <marker id="arrowhead" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
              <polygon points="0 0, 7 3.5, 0 7" fill="var(--path)" />
            </marker>
          </defs>
          <rect x="0" y="0" width="720" height="520" fill="url(#grid)" />
          <rect x="48" y="48" width="624" height="424" fill="none" stroke="#b7aca0" stroke-width="1.2" />
          <polyline fill="none" stroke="rgba(29, 78, 216, 0.18)" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" points="{polyline}" />
          <polyline fill="none" stroke="var(--path)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowhead)" points="{polyline}" />
          {markers}
          {empty_state}
        </svg>
        <p>Trajectory uses ORB-SLAM3 map coordinates (<code>x</code> vs <code>z</code>). They are relative and, in monocular mode, scale-dependent.</p>
      </section>

      <section class="panel">
        <h2>Tracked Camera Input</h2>
        <img src="latest_frame.jpg?frame={latest.get('frame_index', '-')}" alt="Latest frame sent to ORB-SLAM3" />
        <p>Preview shows the resized frame that is sent to the sidecar on each update.</p>
      </section>
    </div>

    <div class="layout" style="margin-top: 18px;">
      <section class="panel">
        <h2>Recent Tracking Health</h2>
        <div class="state-strip">{state_dots}</div>
        <p>Green means tracking is healthy, amber means initializing or recently lost, red means lost.</p>
      </section>

      <section class="panel">
        <h2>Latest Pose</h2>
        <div class="grid" style="margin-bottom: 0;">
          <div class="card"><div class="label">X</div><div class="value">{_format_number(pose.get('tx') if pose else None)}</div></div>
          <div class="card"><div class="label">Y</div><div class="value">{_format_number(pose.get('ty') if pose else None)}</div></div>
          <div class="card"><div class="label">Z</div><div class="value">{_format_number(pose.get('tz') if pose else None)}</div></div>
          <div class="card"><div class="label">Pose</div><div class="value" style="font-size: 0.98rem;">{html.escape(pose_text)}</div></div>
        </div>
        <div class="links">
          <a href="status.json">status.json</a>
          <a href="trajectory.csv">trajectory.csv</a>
          <a href="latest_frame.jpg">latest_frame.jpg</a>
        </div>
      </section>
    </div>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_json = out_dir / "status.json"
    history_csv = out_dir / "trajectory.csv"

    cfg = IndoorNavConfig()
    cfg.sdk.base_url = args.sdk_url
    cfg.slam.enabled = True
    cfg.slam.backend = "orbslam3"
    cfg.slam.endpoint = args.slam_endpoint
    cfg.slam.mode = args.slam_mode
    cfg.slam.resize_width = args.resize_width
    cfg.slam.resize_height = args.resize_height
    cfg.slam.jpeg_quality = args.jpeg_quality

    sdk = RoverSDKClient(cfg.sdk)
    slam = ORBSLAM3Client(cfg.slam)
    history: deque[dict] = deque(maxlen=max(10, int(args.history)))
    last_imu_ts = 0.0
    last_imu_clock_log_time = 0.0

    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "frame_ts", "tracking_state", "ok", "tx", "ty", "tz", "keyframe_id"],
        )
        writer.writeheader()

        await slam.start()
        print(f"SLAM health OK | endpoint={args.slam_endpoint}")
        print(f"Live visualization: file://{out_dir / 'index.html'}")

        interval = 1.0 / max(0.1, float(args.poll_hz))
        frame_index = 0
        try:
            while True:
                loop_start = time.time()
                frame, ts = await sdk.get_front_frame()
                if frame is None:
                    print("no frame")
                    await asyncio.sleep(interval)
                    continue

                slam_frame = _prepare_frame(frame, args.resize_width, args.resize_height)
                imu_payload = None
                if args.slam_mode == "mono_inertial":
                    bot_state = await sdk.get_data()
                    alignment = estimate_mono_inertial_clock_alignment(
                        bot_state.accels,
                        bot_state.gyros,
                        frame_ts=float(ts),
                        data_ts=bot_state.timestamp,
                    )
                    if alignment.needs_correction and (time.time() - last_imu_clock_log_time) >= 5.0:
                        frame_sensor_delta = (
                            alignment.frame_sensor_delta_s
                            if alignment.frame_sensor_delta_s is not None
                            else float("nan")
                        )
                        frame_data_delta = (
                            alignment.frame_data_delta_s
                            if alignment.frame_data_delta_s is not None
                            else float("nan")
                        )
                        print(
                            "IMU clock skew detected:"
                            f" shifting samples by {alignment.offset_s:.3f}s"
                            f" (frame-sensor={frame_sensor_delta:.3f}s,"
                            f" frame-data={frame_data_delta:.3f}s)"
                        )
                        last_imu_clock_log_time = time.time()
                    imu_payload, last_imu_ts = build_mono_inertial_payload(
                        bot_state.accels,
                        bot_state.gyros,
                        frame_ts=float(ts),
                        last_imu_ts=last_imu_ts,
                        timestamp_offset_s=alignment.offset_s,
                    )

                status = await slam.track(slam_frame, ts, imu=imu_payload)
                pose = status.pose
                latest = {
                    "frame_index": frame_index,
                    "frame_ts": float(ts),
                    "tracking_state": status.tracking_state,
                    "ok": bool(status.ok),
                    "keyframe_id": status.keyframe_id,
                    "loop_closure_count": int(status.loop_closure_count),
                    "pose": None,
                }
                row = {
                    "frame_index": frame_index,
                    "frame_ts": float(ts),
                    "tracking_state": status.tracking_state,
                    "ok": bool(status.ok),
                    "tx": "",
                    "ty": "",
                    "tz": "",
                    "keyframe_id": status.keyframe_id,
                }
                if pose is not None:
                    latest["pose"] = {"tx": pose.tx, "ty": pose.ty, "tz": pose.tz}
                    row["tx"] = pose.tx
                    row["ty"] = pose.ty
                    row["tz"] = pose.tz

                history.append(row.copy())
                writer.writerow(row)
                f.flush()
                latest_json.write_text(json.dumps(latest, indent=2), encoding="utf-8")
                _write_preview_image(out_dir, slam_frame, latest, int(args.preview_max_width))
                _write_html(out_dir, latest, history)

                pose_text = "pose=None" if pose is None else f"t=({pose.tx:.2f},{pose.ty:.2f},{pose.tz:.2f})"
                print(f"frame={frame_index} state={status.tracking_state} ok={status.ok} {pose_text}")
                frame_index += 1

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, interval - elapsed))
        finally:
            await slam.close()
            await sdk.close()


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
