from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import cv2

from erc_autonomy.config import ERCConfig
from erc_autonomy.traversability import TraversabilityEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark traversability backends.")
    p.add_argument("--images-dir", required=True, help="Directory with test images")
    p.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern for images (example: '*.jpg' or '*.png')",
    )
    p.add_argument(
        "--backend",
        default="both",
        choices=["simple_edge", "sam2", "both"],
        help="Which backend to benchmark",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum number of images to evaluate",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup frames per backend (not included in timing stats)",
    )

    p.add_argument("--sam2-model-cfg", default="", help="SAM2 model config path")
    p.add_argument("--sam2-checkpoint", default="", help="SAM2 checkpoint path")
    p.add_argument(
        "--sam2-device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="SAM2 inference device",
    )
    p.add_argument(
        "--sam2-max-side",
        type=int,
        default=1024,
        help="Resize input so longest side <= this value for SAM2",
    )
    p.add_argument(
        "--sam2-points-per-side",
        type=int,
        default=24,
        help="SAM2 automatic mask generator points per side",
    )
    p.add_argument(
        "--sam2-pred-iou-thresh",
        type=float,
        default=0.8,
        help="SAM2 predicted IoU threshold",
    )
    p.add_argument(
        "--sam2-stability-score-thresh",
        type=float,
        default=0.9,
        help="SAM2 stability score threshold",
    )
    p.add_argument(
        "--sam2-min-mask-region-area",
        type=int,
        default=0,
        help="SAM2 minimum mask region area in pixels",
    )
    return p.parse_args()


def collect_images(images_dir: str, pattern: str, max_images: int) -> list[Path]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"images dir not found: {root}")

    paths = sorted(root.rglob(pattern))
    paths = [p for p in paths if p.is_file()]
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"no images found in {root} with pattern '{pattern}'")
    return paths


def build_cfg(args: argparse.Namespace, backend: str) -> ERCConfig:
    cfg = ERCConfig()
    cfg.traversability_backend = backend
    cfg.sam2_model_cfg = args.sam2_model_cfg
    cfg.sam2_checkpoint = args.sam2_checkpoint
    cfg.sam2_device = args.sam2_device
    cfg.sam2_max_side = args.sam2_max_side
    cfg.sam2_points_per_side = args.sam2_points_per_side
    cfg.sam2_pred_iou_thresh = args.sam2_pred_iou_thresh
    cfg.sam2_stability_score_thresh = args.sam2_stability_score_thresh
    cfg.sam2_min_mask_region_area = args.sam2_min_mask_region_area
    return cfg


def benchmark_backend(backend: str, cfg: ERCConfig, images: list[Path], warmup: int) -> None:
    engine = TraversabilityEngine(cfg)
    latency_ms: list[float] = []
    confidence: list[float] = []
    risk: list[float] = []
    center_clearance: list[float] = []
    left_clearance: list[float] = []
    right_clearance: list[float] = []
    total = 0
    valid = 0

    for idx, path in enumerate(images):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        total += 1

        t0 = time.perf_counter()
        result = engine.infer(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if idx < warmup:
            continue

        latency_ms.append(dt_ms)
        if result is not None:
            valid += 1
            confidence.append(float(result.confidence))
            risk.append(float(result.risk))
            center_clearance.append(float(result.center_clearance))
            left_clearance.append(float(result.left_clearance))
            right_clearance.append(float(result.right_clearance))

    evaluated = max(0, total - warmup)
    avg_ms = mean(latency_ms) if latency_ms else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 1e-6 else 0.0

    print("=" * 72)
    print(f"backend: {backend}")
    print(f"evaluated_frames: {evaluated} (warmup={warmup})")
    print(f"valid_outputs: {valid}")
    print(f"avg_latency_ms: {avg_ms:.2f}")
    print(f"fps_estimate: {fps:.2f}")
    if confidence:
        print(f"mean_confidence: {mean(confidence):.3f}")
        print(f"mean_risk: {mean(risk):.3f}")
        print(f"mean_center_clearance: {mean(center_clearance):.3f}")
        print(f"mean_left_clearance: {mean(left_clearance):.3f}")
        print(f"mean_right_clearance: {mean(right_clearance):.3f}")
    else:
        print("no valid traversability outputs produced")


def main() -> None:
    args = parse_args()
    images = collect_images(args.images_dir, args.pattern, args.max_images)

    backends = [args.backend] if args.backend != "both" else ["simple_edge", "sam2"]
    for backend in backends:
        cfg = build_cfg(args, backend)
        benchmark_backend(backend, cfg, images, args.warmup)


if __name__ == "__main__":
    main()
