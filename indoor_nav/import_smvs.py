#!/usr/bin/env python3
"""
Import a subset of the Stanford Mobile Visual Search dataset into the repo's
goal-matching benchmark layout:

  <output-root>/
    goals/
    eval_queries/
    mapping.csv
    manifest.json

The importer defaults to the two most relevant categories for wall-image
matching: museum_paintings and print.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://web.cs.wpi.edu/~claypool/mmsys-dataset/2011/stanford/mvs_images/"
DEFAULT_CATEGORIES = ("museum_paintings", "print")
DEFAULT_DEVICES = ("Canon", "Droid", "E63", "Palm")
USER_AGENT = "nyu-earthrover-smvs-importer/1.0"


def log(message: str) -> None:
    print(message, flush=True)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_binary(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as response:
        return response.read()


def parse_links(index_url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(fetch_text(index_url))
    return parser.links


def natural_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return (int(stem), name)
    return (sys.maxsize, name)


def list_remote_jpgs(index_url: str) -> list[str]:
    links = parse_links(index_url)
    return sorted(
        {Path(link).name for link in links if link.lower().endswith(".jpg")},
        key=natural_key,
    )


def list_remote_dirs(index_url: str) -> list[str]:
    links = parse_links(index_url)
    dirs = []
    for link in links:
        if link in {"../", "Parent Directory", "../index.html"}:
            continue
        if link.endswith("/"):
            dirs.append(link.rstrip("/"))
            continue
        if link.endswith("/index.html"):
            dirs.append(Path(link).parent.name)
    return sorted(set(dirs))


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def choose_subset(items: list[str], limit: int, rng: random.Random) -> list[str]:
    if limit <= 0 or limit >= len(items):
        return list(items)
    chosen = rng.sample(items, limit)
    return sorted(chosen, key=natural_key)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, *, force: bool) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(fetch_binary(url))


def build_output_name(category: str, stem: str, suffix: str = ".jpg") -> str:
    return f"{category}__{stem}{suffix}"


def write_mapping(mapping_rows: list[dict[str, str]], mapping_path: Path) -> None:
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "goal"])
        writer.writeheader()
        for row in sorted(mapping_rows, key=lambda row: row["query"]):
            writer.writerow(row)


def write_manifest(
    *,
    manifest_path: Path,
    categories: list[str],
    devices: list[str],
    results: list[dict],
    mapping_path: Path,
) -> None:
    manifest = {
        "source": BASE_URL,
        "categories": categories,
        "devices": devices,
        "results": results,
        "mapping_csv": str(mapping_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def import_category(
    *,
    category: str,
    devices: Iterable[str],
    output_root: Path,
    cache_root: Path,
    max_goals_per_category: int,
    max_queries_per_goal: int,
    rng: random.Random,
    force: bool,
    progress_every: int,
) -> dict:
    category_url = urljoin(BASE_URL, f"{category}/")
    available_dirs = list_remote_dirs(category_url)
    if "Reference" not in available_dirs:
        raise RuntimeError(f"{category}: remote Reference directory not found")

    selected_devices = [device for device in devices if device in available_dirs]
    if not selected_devices:
        raise RuntimeError(
            f"{category}: none of the requested devices exist remotely; "
            f"available={available_dirs}"
        )

    reference_files = list_remote_jpgs(urljoin(category_url, "Reference/"))
    if not reference_files:
        raise RuntimeError(f"{category}: no reference images found")

    chosen_refs = choose_subset(reference_files, max_goals_per_category, rng)
    chosen_ref_set = set(chosen_refs)

    goals_dir = output_root / "goals"
    queries_dir = output_root / "eval_queries"
    cache_category_dir = cache_root / category
    ensure_dir(goals_dir)
    ensure_dir(queries_dir)
    ensure_dir(cache_category_dir)

    mapping_rows: list[dict[str, str]] = []
    downloaded_goals = 0
    downloaded_queries = 0

    log(
        f"[{category}] references={len(reference_files)} selected_goals={len(chosen_refs)} "
        f"devices={','.join(selected_devices)}"
    )

    for ref_name in chosen_refs:
        goal_stem = build_output_name(category, Path(ref_name).stem, suffix="")
        goal_filename = f"{goal_stem}.jpg"
        goal_url = urljoin(category_url, f"Reference/{ref_name}")
        goal_cache = cache_category_dir / "Reference" / ref_name
        goal_output = goals_dir / goal_filename
        download_file(goal_url, goal_cache, force=force)
        goal_output.write_bytes(goal_cache.read_bytes())
        downloaded_goals += 1
        if progress_every > 0 and downloaded_goals % progress_every == 0:
            log(f"[{category}] downloaded goals: {downloaded_goals}/{len(chosen_refs)}")

    device_files: dict[str, set[str]] = {}
    for device in selected_devices:
        files = list_remote_jpgs(urljoin(category_url, f"{device}/"))
        device_files[device] = set(files)

    for ref_name in chosen_refs:
        stem = Path(ref_name).stem
        goal_stem = build_output_name(category, stem, suffix="")
        usable_devices = [device for device in selected_devices if ref_name in device_files[device]]
        if max_queries_per_goal > 0 and len(usable_devices) > max_queries_per_goal:
            usable_devices = sorted(rng.sample(usable_devices, max_queries_per_goal))

        for device in usable_devices:
            query_filename = f"{goal_stem}__{device}.jpg"
            query_url = urljoin(category_url, f"{device}/{ref_name}")
            query_cache = cache_category_dir / device / ref_name
            query_output = queries_dir / query_filename
            download_file(query_url, query_cache, force=force)
            query_output.write_bytes(query_cache.read_bytes())
            mapping_rows.append({"query": query_filename, "goal": f"{goal_stem}.jpg"})
            downloaded_queries += 1
            if progress_every > 0 and downloaded_queries % progress_every == 0:
                log(f"[{category}] downloaded queries: {downloaded_queries}")

    log(
        f"[{category}] done: goals={downloaded_goals} queries={downloaded_queries}"
    )

    return {
        "category": category,
        "devices": selected_devices,
        "n_goals": downloaded_goals,
        "n_queries": downloaded_queries,
        "mapping_rows": mapping_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Stanford Mobile Visual Search images into benchmark format."
    )
    parser.add_argument(
        "--output-root",
        default="indoor_nav/datasets/smvs_wall",
        help="Destination root for goals/, eval_queries/, mapping.csv.",
    )
    parser.add_argument(
        "--cache-root",
        default="indoor_nav/datasets/.cache/smvs",
        help="Cache for downloaded source images.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated SMVS categories (default: museum_paintings,print).",
    )
    parser.add_argument(
        "--devices",
        default=",".join(DEFAULT_DEVICES),
        help="Comma-separated device folders to use as queries.",
    )
    parser.add_argument(
        "--max-goals-per-category",
        type=int,
        default=0,
        help="Optional cap per category (0 = all references in the category).",
    )
    parser.add_argument(
        "--max-queries-per-goal",
        type=int,
        default=0,
        help="Optional cap on how many device captures to keep per goal (0 = all selected devices).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when sampling subsets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download source images even if they already exist in the cache.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print available remote categories and exit.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N downloads within a category (default: 25).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        available_categories = list_remote_dirs(BASE_URL)
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"failed to reach Stanford MVS dataset: {exc}") from exc

    if args.list_categories:
        for category in available_categories:
            print(category)
        return 0

    categories = parse_csv_list(args.categories)
    if not categories:
        raise SystemExit("no categories requested")

    invalid = [category for category in categories if category not in available_categories]
    if invalid:
        raise SystemExit(
            f"unknown categories: {', '.join(invalid)}; "
            f"available={', '.join(available_categories)}"
        )

    devices = parse_csv_list(args.devices)
    if not devices:
        raise SystemExit("no devices requested")

    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    ensure_dir(output_root)
    ensure_dir(cache_root)

    rng = random.Random(args.seed)
    results = []
    all_mapping_rows: list[dict[str, str]] = []
    mapping_path = output_root / "mapping.csv"
    manifest_path = output_root / "manifest.json"

    log(
        f"Starting SMVS import: output={output_root} categories={','.join(categories)} "
        f"devices={','.join(devices)}"
    )

    for idx, category in enumerate(categories, start=1):
        log(f"[{idx}/{len(categories)}] importing {category} ...")
        result = import_category(
            category=category,
            devices=devices,
            output_root=output_root,
            cache_root=cache_root,
            max_goals_per_category=args.max_goals_per_category,
            max_queries_per_goal=args.max_queries_per_goal,
            rng=rng,
            force=args.force,
            progress_every=args.progress_every,
        )
        results.append({k: v for k, v in result.items() if k != "mapping_rows"})
        all_mapping_rows.extend(result["mapping_rows"])
        write_mapping(all_mapping_rows, mapping_path)
        write_manifest(
            manifest_path=manifest_path,
            categories=categories,
            devices=devices,
            results=results,
            mapping_path=mapping_path,
        )
        log(
            f"[{idx}/{len(categories)}] checkpoint written: "
            f"goals={sum(item['n_goals'] for item in results)} "
            f"queries={sum(item['n_queries'] for item in results)}"
        )

    total_goals = sum(item["n_goals"] for item in results)
    total_queries = sum(item["n_queries"] for item in results)
    log(f"Imported SMVS subset into {output_root}")
    log(f"Goals: {total_goals}")
    log(f"Queries: {total_queries}")
    log(f"Mapping: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
