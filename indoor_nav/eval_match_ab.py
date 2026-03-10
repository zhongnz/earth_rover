#!/usr/bin/env python3
"""
A/B matcher benchmark for indoor image-goal retrieval.

Compares multiple match backends (default: dinov2_vlad vs dinov3_vlad vs cosplace) on a
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
from typing import Dict, List, Optional

# Prevent OpenBLAS/OpenMP warnings from swamping benchmark output on CPU builds.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np

# Add project root so script can run as:
#   python indoor_nav/eval_match_ab.py ...
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indoor_nav.configs.config import GoalConfig
from indoor_nav.goal_matching.visualize import generate_visual_report
from indoor_nav.modules.checkpoint_manager import GoalCheckpoint, GoalMatcher

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_METHODS = "dinov2_vlad,cosplace"
DEFAULT_MODELS = {
    "dinov2_vlad": "facebook/dinov2-with-registers-base",
    "dinov3_vlad": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


@dataclass
class QuerySample:
    path: Path
    goal_stem: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B benchmark for indoor goal matchers.")
    p.add_argument("--goals-dir", required=True, help="Directory of goal images.")
    p.add_argument(
        "--queries-dir",
        default="",
        help="Directory of query images. Optional when --self-query is set.",
    )
    p.add_argument(
        "--self-query",
        action="store_true",
        help="Use goal images as their own queries for a fast pipeline smoke test.",
    )
    p.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help="Comma-separated matcher list (default: dinov2_vlad,cosplace).",
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
    p.add_argument(
        "--report-topn",
        type=int,
        default=5,
        help="How many top-ranked goals to retain per query in the JSON report (default: 5).",
    )
    p.add_argument(
        "--viz-dir",
        default="",
        help="Optional directory to write summary plots, CSVs, and failure contact sheets.",
    )
    p.add_argument(
        "--failure-limit",
        type=int,
        default=12,
        help="Maximum number of failure triplets to render per method (default: 12).",
    )
    p.add_argument(
        "--wall-crop-min-area-frac",
        type=float,
        default=None,
        help="Override GoalConfig.wall_crop_min_area_frac for wall-aware matchers.",
    )
    p.add_argument(
        "--wall-crop-max-area-frac",
        type=float,
        default=None,
        help="Override GoalConfig.wall_crop_max_area_frac for wall-aware matchers.",
    )
    p.add_argument(
        "--wall-crop-max-aspect-ratio",
        type=float,
        default=None,
        help="Override GoalConfig.wall_crop_max_aspect_ratio for wall-aware matchers.",
    )
    p.add_argument(
        "--wall-crop-min-fill-ratio",
        type=float,
        default=None,
        help="Override GoalConfig.wall_crop_min_fill_ratio for wall-aware matchers.",
    )
    p.add_argument(
        "--wall-crop-padding-frac",
        type=float,
        default=None,
        help="Override GoalConfig.wall_crop_padding_frac for wall-aware matchers.",
    )
    p.add_argument(
        "--wall-crop-max-candidates",
        type=int,
        default=None,
        help="Override GoalConfig.wall_crop_max_candidates for wall-aware matchers.",
    )
    p.add_argument(
        "--wall-crop-score-weight",
        type=float,
        default=None,
        help="Override GoalConfig.wall_crop_score_weight for wall-aware matchers.",
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


def describe_stems(paths: List[Path], *, limit: int = 5) -> str:
    stems = [path.stem for path in paths[:limit]]
    if not stems:
        return "[]"
    suffix = "" if len(paths) <= limit else ", ..."
    return "[" + ", ".join(stems) + suffix + "]"


def build_matcher(method: str, device: str, args: argparse.Namespace) -> GoalMatcher:
    cfg = GoalConfig()
    cfg.match_method = method
    cfg.feature_device = device
    if method in DEFAULT_MODELS:
        cfg.feature_model = DEFAULT_MODELS[method]
    elif method == "cosplace":
        cfg.feature_model = (
            f"gmberton/cosplace ({cfg.cosplace_backbone}, dim={cfg.cosplace_fc_output_dim})"
        )
    wall_crop_overrides = {
        "wall_crop_min_area_frac": args.wall_crop_min_area_frac,
        "wall_crop_max_area_frac": args.wall_crop_max_area_frac,
        "wall_crop_max_aspect_ratio": args.wall_crop_max_aspect_ratio,
        "wall_crop_min_fill_ratio": args.wall_crop_min_fill_ratio,
        "wall_crop_padding_frac": args.wall_crop_padding_frac,
        "wall_crop_max_candidates": args.wall_crop_max_candidates,
        "wall_crop_score_weight": args.wall_crop_score_weight,
    }
    for attr, value in wall_crop_overrides.items():
        if value is not None:
            setattr(cfg, attr, value)
    matcher = GoalMatcher(cfg)
    return matcher


def load_goal_checkpoints(matcher: GoalMatcher, goal_paths: List[Path]) -> List[GoalCheckpoint]:
    cps: List[GoalCheckpoint] = []
    for i, path in enumerate(goal_paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cp = GoalCheckpoint(index=i + 1, image_path=str(path), image=img)
        cp.feature = matcher.prepare_goal(img)
        cps.append(cp)
    return cps


def evaluate_method(
    method: str,
    query_samples: List[QuerySample],
    goal_paths: List[Path],
    topk: int,
    device: str,
    report_topn: int,
    args: argparse.Namespace,
) -> Dict:
    matcher = build_matcher(method=method, device=device, args=args)
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
    query_candidate_counts: List[int] = []
    goal_candidate_counts: List[int] = []
    evaluated = 0
    query_details: List[Dict] = []

    for sample in query_samples:
        if sample.goal_stem not in goal_by_stem:
            continue
        query_img = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if query_img is None:
            continue

        t0 = time.perf_counter()
        query_prepared = matcher.prepare_query(query_img)
        scores: Dict[str, float] = {}
        for stem, cp in goal_by_stem.items():
            if cp.feature is None:
                cp.feature = matcher.prepare_goal(cp.image)
            scores[stem] = matcher.score_prepared(query_prepared, cp.feature)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(elapsed_ms)
        query_candidate_count = int(query_prepared.metadata.get("candidate_count", 0))
        true_goal_candidate_count = int(goal_by_stem[sample.goal_stem].feature.metadata.get("candidate_count", 0))
        query_candidate_counts.append(query_candidate_count)
        goal_candidate_counts.append(true_goal_candidate_count)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_stems = [stem for stem, _ in ranked]
        true_rank = None
        pred_goal_stem = ranked[0][0] if ranked else None
        pred_goal_path = str(goal_by_stem[pred_goal_stem].image_path) if pred_goal_stem else ""
        pred_goal_feature = goal_by_stem[pred_goal_stem].feature if pred_goal_stem else None
        true_goal_path = str(goal_by_stem[sample.goal_stem].image_path)
        true_score = float(scores.get(sample.goal_stem, 0.0))
        top1_score = float(ranked[0][1]) if ranked else 0.0
        if sample.goal_stem in ranked_stems:
            true_rank = ranked_stems.index(sample.goal_stem) + 1
            reciprocal_rank += 1.0 / true_rank
            if true_rank == 1:
                hits_1 += 1
            if true_rank <= max(1, topk):
                hits_k += 1
        evaluated += 1

        top_results = []
        for stem, score in ranked[: max(1, report_topn)]:
            top_results.append(
                {
                    "goal_stem": stem,
                    "goal_path": str(goal_by_stem[stem].image_path),
                    "score": float(score),
                }
            )
        query_details.append(
            {
                "query_path": str(sample.path),
                "true_goal_stem": sample.goal_stem,
                "true_goal_path": true_goal_path,
                "pred_goal_stem": pred_goal_stem,
                "pred_goal_path": pred_goal_path,
                "true_rank": true_rank,
                "true_score": true_score,
                "top1_score": top1_score,
                "margin_pred_minus_true": top1_score - true_score,
                "latency_ms": elapsed_ms,
                "correct_top1": true_rank == 1,
                "query_candidate_count": query_candidate_count,
                "true_goal_candidate_count": true_goal_candidate_count,
                "pred_goal_candidate_count": int(pred_goal_feature.metadata.get("candidate_count", 0)) if pred_goal_feature is not None else 0,
                "query_candidate_boxes": query_prepared.metadata.get("candidate_boxes", []),
                "true_goal_candidate_boxes": goal_by_stem[sample.goal_stem].feature.metadata.get("candidate_boxes", []),
                "pred_goal_candidate_boxes": pred_goal_feature.metadata.get("candidate_boxes", []) if pred_goal_feature is not None else [],
                "top_results": top_results,
            }
        )

    if evaluated == 0:
        return {
            "method": method,
            "status": "ok",
            "error": "",
            "evaluated": 0,
            "top1_acc": 0.0,
            f"top{topk}_acc": 0.0,
            "mrr": 0.0,
            "latency_ms_mean": 0.0,
            "latency_ms_p95": 0.0,
            "queries": [],
        }

    lat_array = np.array(lat_ms, dtype=np.float64)
    return {
        "method": method,
        "status": "ok",
        "error": "",
        "evaluated": evaluated,
        "top1_acc": hits_1 / evaluated,
        f"top{topk}_acc": hits_k / evaluated,
        "mrr": reciprocal_rank / evaluated,
        "latency_ms_mean": float(np.mean(lat_array)),
        "latency_ms_p95": float(np.percentile(lat_array, 95)),
        "avg_query_candidates": float(np.mean(np.array(query_candidate_counts, dtype=np.float64))),
        "avg_goal_candidates": float(np.mean(np.array(goal_candidate_counts, dtype=np.float64))),
        "queries": query_details,
    }


def parse_methods(value: str) -> List[str]:
    methods = [m.strip() for m in value.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods provided.")
    return methods


def summarize_exception(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    if "gated repo" in text.lower():
        return "gated repo; authenticate or remove this method"
    if "401" in text and "huggingface.co" in text:
        return "huggingface auth required for this model"
    return text[:240] if text else exc.__class__.__name__


def build_error_result(method: str, topk: int, exc: Exception) -> Dict:
    return {
        "method": method,
        "status": "error",
        "error": summarize_exception(exc),
        "evaluated": 0,
        "top1_acc": 0.0,
        f"top{topk}_acc": 0.0,
        "mrr": 0.0,
        "latency_ms_mean": 0.0,
        "latency_ms_p95": 0.0,
        "queries": [],
    }


def print_summary(results: List[Dict], topk: int) -> None:
    print("\nA/B Matching Results")
    print("-" * 132)
    print(
        f"{'method':<16} {'status':<8} {'n':>6} {'top1':>8} {('top' + str(topk)):>8} "
        f"{'mrr':>8} {'lat_ms':>12} {'lat_p95':>12} {'error':<40}"
    )
    print("-" * 132)
    for row in results:
        error = (row.get("error") or "")[:40]
        print(
            f"{row['method']:<16} "
            f"{row.get('status', 'ok'):<8} "
            f"{row['evaluated']:>6d} "
            f"{row['top1_acc'] * 100:>7.2f}% "
            f"{row[f'top{topk}_acc'] * 100:>7.2f}% "
            f"{row['mrr']:>8.3f} "
            f"{row['latency_ms_mean']:>12.2f} "
            f"{row['latency_ms_p95']:>12.2f} "
            f"{error:<40}"
        )
    print("-" * 132)


def main() -> int:
    args = parse_args()
    goals_dir = Path(args.goals_dir)
    if not goals_dir.is_dir():
        raise SystemExit(f"goals dir not found: {goals_dir}")

    methods = parse_methods(args.methods)
    goal_paths = list_images(goals_dir)

    goal_stems = {p.stem for p in goal_paths}
    query_paths: List[Path] = []
    queries_dir: Path | None = None
    if args.queries_dir:
        queries_dir = Path(args.queries_dir)
        if not queries_dir.is_dir():
            raise SystemExit(f"queries dir not found: {queries_dir}")
        query_paths = list_images(queries_dir)

    if args.self_query:
        query_samples = [QuerySample(path=path, goal_stem=path.stem) for path in goal_paths]
    else:
        if queries_dir is None:
            raise SystemExit("queries dir is required unless --self-query is set")
        if not query_paths:
            raise SystemExit(
                f"no query images found in {queries_dir}; add images or run with --self-query for a smoke test"
            )
        query_samples = build_query_set(query_paths, goal_stems, args.mapping_csv)
    if args.max_queries > 0:
        query_samples = query_samples[: args.max_queries]

    if not goal_paths:
        raise SystemExit(f"no goal images found in {goals_dir}")
    if not query_samples:
        query_hint = describe_stems(query_paths)
        goal_hint = describe_stems(goal_paths)
        raise SystemExit(
            "no labeled query samples found; "
            f"goal stems={goal_hint}, query stems={query_hint}. "
            "Rename queries to '<goal_stem>__anything.png', provide --mapping-csv, "
            "or run with --self-query for a smoke test."
        )

    results: List[Dict] = []
    for method in methods:
        print(f"Evaluating {method} on {len(query_samples)} queries ...")
        try:
            row = evaluate_method(
                method=method,
                query_samples=query_samples,
                goal_paths=goal_paths,
                topk=args.topk,
                device=args.device,
                report_topn=args.report_topn,
                args=args,
            )
        except Exception as exc:
            row = build_error_result(method=method, topk=args.topk, exc=exc)
        results.append(row)

    print_summary(results, topk=args.topk)

    payload = {
        "methods": methods,
        "goals_dir": str(goals_dir),
        "queries_dir": str(queries_dir) if queries_dir is not None else "",
        "self_query": bool(args.self_query),
        "n_goals": len(goal_paths),
        "n_queries": len(query_samples),
        "topk": args.topk,
        "wall_crop_overrides": {
            "min_area_frac": args.wall_crop_min_area_frac,
            "max_area_frac": args.wall_crop_max_area_frac,
            "max_aspect_ratio": args.wall_crop_max_aspect_ratio,
            "min_fill_ratio": args.wall_crop_min_fill_ratio,
            "padding_frac": args.wall_crop_padding_frac,
            "max_candidates": args.wall_crop_max_candidates,
            "score_weight": args.wall_crop_score_weight,
        },
        "results": results,
    }

    report_json_path = args.out_json
    if args.viz_dir and not report_json_path:
        report_json_path = str(Path(args.viz_dir) / "report.json")

    if report_json_path:
        Path(report_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON report to {report_json_path}")

    if args.viz_dir:
        generate_visual_report(
            results,
            topk=args.topk,
            outdir=args.viz_dir,
            failure_limit=args.failure_limit,
            report_json_path=report_json_path or "",
        )
        print(f"Wrote visualization bundle to {args.viz_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
