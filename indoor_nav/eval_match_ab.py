#!/usr/bin/env python3
"""
A/B matcher benchmark for indoor image-goal retrieval.

Compares multiple match backends (default: dinov2_vlad vs dinov3_vlad) on a
goal/query dataset and reports retrieval metrics and per-query latency.

Expected data layout:
  - goals directory: one image per goal, filename stem is goal id.
  - queries directory: query images whose stem matches a goal id, or starts
    with "<goal_id>__..." (prefix before "__" is used as ground truth).

Optional mapping CSV:
  query,goal
  q_001.jpg,goal_a
  q_002.jpg,goal_b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Add project root so script can run as:
#   python indoor_nav/eval_match_ab.py ...
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indoor_nav.configs.config import GoalConfig
from indoor_nav.modules.checkpoint_manager import GoalCheckpoint, GoalMatcher

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_METHODS = "dinov2_vlad,dinov3_vlad"
DEFAULT_MODELS = {
    "dinov2_vlad": "facebook/dinov2-with-registers-base",
    "dinov3_vlad": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}
LEARNED_METHODS = {"dinov2_vlad", "dinov3_vlad", "siglip2", "dinov2", "clip", "eigenplaces"}


@dataclass
class QuerySample:
    path: Path
    goal_stem: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B benchmark for indoor goal matchers.")
    p.add_argument("--goals-dir", required=True, help="Directory of goal images.")
    p.add_argument("--queries-dir", required=True, help="Directory of query images.")
    p.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help="Comma-separated matcher list (default: dinov2_vlad,dinov3_vlad).",
    )
    p.add_argument("--device", default="cuda", help="Feature device (cuda/cpu).")
    p.add_argument(
        "--mapping-csv",
        default="",
        help="Optional CSV with columns: query,goal for explicit query->goal labels.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k accuracy cutoff (default: 3).",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional max number of queries to evaluate (0 = all).",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="Optional path to write full results as JSON.",
    )
    return p.parse_args()


def list_images(directory: Path) -> List[Path]:
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def infer_goal_stem_from_query(query_stem: str, goal_stems: set[str]) -> Optional[str]:
    candidates = [query_stem]
    if "__" in query_stem:
        candidates.append(query_stem.split("__", 1)[0])
    if "_" in query_stem:
        candidates.append(query_stem.split("_", 1)[0])
    for candidate in candidates:
        if candidate in goal_stems:
            return candidate
    return None


def load_mapping_csv(mapping_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with mapping_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            g = (row.get("goal") or "").strip()
            if not q or not g:
                continue
            mapping[Path(q).stem] = Path(g).stem
    return mapping


def build_query_set(queries: List[Path], goal_stems: set[str], mapping_csv: str) -> List[QuerySample]:
    csv_map: Dict[str, str] = {}
    if mapping_csv:
        csv_map = load_mapping_csv(Path(mapping_csv))

    samples: List[QuerySample] = []
    for q in queries:
        q_stem = q.stem
        goal_stem = csv_map.get(q_stem)
        if goal_stem is None:
            goal_stem = infer_goal_stem_from_query(q_stem, goal_stems)
        if goal_stem is None or goal_stem not in goal_stems:
            continue
        samples.append(QuerySample(path=q, goal_stem=goal_stem))
    return samples


def build_matcher(method: str, device: str) -> GoalMatcher:
    cfg = GoalConfig()
    cfg.match_method = method
    cfg.feature_device = device
    if method in DEFAULT_MODELS:
        cfg.feature_model = DEFAULT_MODELS[method]
    matcher = GoalMatcher(cfg)
    return matcher


def load_goal_checkpoints(matcher: GoalMatcher, goal_paths: List[Path]) -> List[GoalCheckpoint]:
    cps: List[GoalCheckpoint] = []
    for i, path in enumerate(goal_paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cp = GoalCheckpoint(index=i + 1, image_path=str(path), image=img)
        cp.feature = matcher.extract_feature(img)
        cps.append(cp)
    return cps


def evaluate_method(
    method: str,
    query_samples: List[QuerySample],
    goal_paths: List[Path],
    topk: int,
    device: str,
) -> Dict:
    matcher = build_matcher(method=method, device=device)
    goal_cps = load_goal_checkpoints(matcher, goal_paths)

    goal_by_stem: Dict[str, GoalCheckpoint] = {
        Path(cp.image_path).stem: cp for cp in goal_cps
    }
    goal_stems = list(goal_by_stem.keys())
    if not goal_stems:
        raise RuntimeError("No valid goals loaded.")

    hits_1 = 0
    hits_k = 0
    reciprocal_rank = 0.0
    lat_ms: List[float] = []
    evaluated = 0

    for sample in query_samples:
        if sample.goal_stem not in goal_by_stem:
            continue
        query_img = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if query_img is None:
            continue

        t0 = time.perf_counter()
        scores: Dict[str, float] = {}

        if method in LEARNED_METHODS:
            q_feat = matcher.extract_feature(query_img)
            for stem, cp in goal_by_stem.items():
                if cp.feature is None:
                    cp.feature = matcher.extract_feature(cp.image)
                cosine = float(np.dot(q_feat, cp.feature))
                scores[stem] = (cosine + 1.0) / 2.0
        else:
            for stem, cp in goal_by_stem.items():
                scores[stem] = matcher.compute_similarity(query_img, cp)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(elapsed_ms)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_stems = [stem for stem, _ in ranked]
        if sample.goal_stem in ranked_stems:
            rank = ranked_stems.index(sample.goal_stem) + 1
            reciprocal_rank += 1.0 / rank
            if rank == 1:
                hits_1 += 1
            if rank <= max(1, topk):
                hits_k += 1
        evaluated += 1

    if evaluated == 0:
        return {
            "method": method,
            "evaluated": 0,
            "top1_acc": 0.0,
            f"top{topk}_acc": 0.0,
            "mrr": 0.0,
            "latency_ms_mean": 0.0,
            "latency_ms_p95": 0.0,
        }

    lat_array = np.array(lat_ms, dtype=np.float64)
    return {
        "method": method,
        "evaluated": evaluated,
        "top1_acc": hits_1 / evaluated,
        f"top{topk}_acc": hits_k / evaluated,
        "mrr": reciprocal_rank / evaluated,
        "latency_ms_mean": float(np.mean(lat_array)),
        "latency_ms_p95": float(np.percentile(lat_array, 95)),
    }


def parse_methods(value: str) -> List[str]:
    methods = [m.strip() for m in value.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods provided.")
    return methods


def print_summary(results: List[Dict], topk: int) -> None:
    print("\nA/B Matching Results")
    print("-" * 84)
    print(
        f"{'method':<16} {'n':>6} {'top1':>8} {('top' + str(topk)):>8} {'mrr':>8} {'lat_ms':>12} {'lat_p95':>12}"
    )
    print("-" * 84)
    for row in results:
        print(
            f"{row['method']:<16} "
            f"{row['evaluated']:>6d} "
            f"{row['top1_acc'] * 100:>7.2f}% "
            f"{row[f'top{topk}_acc'] * 100:>7.2f}% "
            f"{row['mrr']:>8.3f} "
            f"{row['latency_ms_mean']:>12.2f} "
            f"{row['latency_ms_p95']:>12.2f}"
        )
    print("-" * 84)


def main() -> int:
    args = parse_args()
    goals_dir = Path(args.goals_dir)
    queries_dir = Path(args.queries_dir)
    if not goals_dir.is_dir():
        raise SystemExit(f"goals dir not found: {goals_dir}")
    if not queries_dir.is_dir():
        raise SystemExit(f"queries dir not found: {queries_dir}")

    methods = parse_methods(args.methods)
    goal_paths = list_images(goals_dir)
    query_paths = list_images(queries_dir)

    goal_stems = {p.stem for p in goal_paths}
    query_samples = build_query_set(query_paths, goal_stems, args.mapping_csv)
    if args.max_queries > 0:
        query_samples = query_samples[: args.max_queries]

    if not goal_paths:
        raise SystemExit(f"no goal images found in {goals_dir}")
    if not query_samples:
        raise SystemExit(
            "no labeled query samples found; add mapping CSV or rename queries to match goal stems"
        )

    results: List[Dict] = []
    for method in methods:
        print(f"Evaluating {method} on {len(query_samples)} queries ...")
        row = evaluate_method(
            method=method,
            query_samples=query_samples,
            goal_paths=goal_paths,
            topk=args.topk,
            device=args.device,
        )
        results.append(row)

    print_summary(results, topk=args.topk)

    if args.out_json:
        payload = {
            "methods": methods,
            "goals_dir": str(goals_dir),
            "queries_dir": str(queries_dir),
            "n_goals": len(goal_paths),
            "n_queries": len(query_samples),
            "results": results,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON report to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
