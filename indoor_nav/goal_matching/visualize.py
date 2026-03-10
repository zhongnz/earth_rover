from __future__ import annotations

import csv
import html
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_pil():
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    return Image, ImageDraw, ImageFont, ImageOps


def _load_font():
    _, _, image_font, _ = _load_pil()
    return image_font.load_default()


def _fit_image(path: str, size: tuple[int, int], *, boxes: list[list[int]] | None = None) -> Image.Image:
    Image, ImageDraw, _, ImageOps = _load_pil()
    width, height = size
    try:
        with Image.open(path) as src:
            image = src.convert("RGB")
    except Exception:
        image = Image.new("RGB", size, color=(230, 230, 230))
        draw = ImageDraw.Draw(image)
        draw.text((12, 12), "missing", fill=(80, 80, 80), font=_load_font())
        return image

    contained = ImageOps.contain(image, size)
    if boxes:
        draw = ImageDraw.Draw(contained)
        scale_x = contained.width / max(image.width, 1)
        scale_y = contained.height / max(image.height, 1)
        for box in boxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            draw.rectangle(
                (
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                ),
                outline=(220, 72, 57),
                width=2,
            )
    canvas = Image.new("RGB", size, color=(245, 245, 245))
    x = (width - contained.width) // 2
    y = (height - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def _draw_triplet(detail: Dict, *, thumb_size: tuple[int, int]) -> Image.Image:
    Image, ImageDraw, _, _ = _load_pil()
    font = _load_font()
    width, height = thumb_size
    padding = 12
    gutter = 8
    header_h = 56
    label_h = 18
    canvas_w = padding * 2 + width * 3 + gutter * 2
    canvas_h = padding * 2 + header_h + label_h + height
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    pred_goal = detail.get("pred_goal_stem") or "none"
    true_rank = detail.get("true_rank")
    title = (
        f"{Path(detail['query_path']).name} | gt={detail['true_goal_stem']} "
        f"| pred={pred_goal} | rank={true_rank} | "
        f"top1={detail.get('top1_score', 0.0):.3f} gt={detail.get('true_score', 0.0):.3f}"
    )
    draw.text((padding, padding), title, fill=(20, 20, 20), font=font)
    candidate_text = (
        f"cands q={detail.get('query_candidate_count', 0)} "
        f"gt={detail.get('true_goal_candidate_count', 0)} "
        f"pred={detail.get('pred_goal_candidate_count', 0)}"
    )
    draw.text((padding, padding + 18), candidate_text, fill=(80, 80, 80), font=font)

    labels = [
        ("query", detail["query_path"], detail.get("query_candidate_boxes", [])),
        ("goal", detail["true_goal_path"], detail.get("true_goal_candidate_boxes", [])),
        ("pred", detail.get("pred_goal_path") or detail["true_goal_path"], detail.get("pred_goal_candidate_boxes", [])),
    ]
    y0 = padding + header_h
    for idx, (label, path, boxes) in enumerate(labels):
        x0 = padding + idx * (width + gutter)
        draw.text((x0, y0), label, fill=(40, 40, 40), font=font)
        tile = _fit_image(path, thumb_size, boxes=boxes)
        canvas.paste(tile, (x0, y0 + label_h))

    return canvas


def _save_contact_sheet(
    details: List[Dict],
    out_path: str,
    *,
    title: str,
    thumb_size: tuple[int, int] = (180, 120),
    columns: int = 1,
) -> None:
    Image, ImageDraw, _, _ = _load_pil()
    if not details:
        return

    rows = [_draw_triplet(detail, thumb_size=thumb_size) for detail in details]
    tile_w = max(tile.width for tile in rows)
    tile_h = max(tile.height for tile in rows)
    columns = max(1, columns)
    row_count = int(np.ceil(len(rows) / columns))
    title_h = 28
    gutter = 10
    sheet_w = tile_w * columns + gutter * (columns + 1)
    sheet_h = title_h + row_count * tile_h + gutter * (row_count + 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 6), title, fill=(20, 20, 20), font=_load_font())

    for idx, tile in enumerate(rows):
        row = idx // columns
        col = idx % columns
        x = gutter + col * (tile_w + gutter)
        y = title_h + gutter + row * (tile_h + gutter)
        sheet.paste(tile, (x, y))

    sheet.save(out_path)


def _write_method_csv(method_result: Dict, out_path: str) -> None:
    fieldnames = [
        "query_path",
        "true_goal_stem",
        "pred_goal_stem",
        "true_rank",
        "true_score",
        "top1_score",
        "margin_pred_minus_true",
        "latency_ms",
        "correct_top1",
        "query_candidate_count",
        "true_goal_candidate_count",
        "pred_goal_candidate_count",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detail in method_result.get("queries", []):
            writer.writerow({key: detail.get(key) for key in fieldnames})


def _plot_summary(results: List[Dict], topk: int, out_path: str) -> str:
    ok_results = [row for row in results if row.get("status", "ok") == "ok"]
    if not ok_results:
        return _write_summary_fallback(results, topk=topk, out_path=out_path)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _write_summary_fallback(results, topk=topk, out_path=out_path)

    methods = [row["method"] for row in ok_results]
    top1 = [row["top1_acc"] * 100.0 for row in ok_results]
    topk_vals = [row[f"top{topk}_acc"] * 100.0 for row in ok_results]
    mrr = [row["mrr"] for row in ok_results]
    lat_mean = [row["latency_ms_mean"] for row in ok_results]
    lat_p95 = [row["latency_ms_p95"] for row in ok_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, top1, width=width, label="top1")
    ax.bar(x + width / 2, topk_vals, width=width, label=f"top{topk}")
    ax.set_title("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Percent")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(methods, mrr, color="#4c78a8")
    ax.set_title("MRR")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    ax.bar(x - width / 2, lat_mean, width=width, label="mean")
    ax.bar(x + width / 2, lat_p95, width=width, label="p95")
    ax.set_title("Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("ms")
    ax.legend()

    ax = axes[1, 1]
    for row in ok_results:
        ranks = [detail["true_rank"] or 0 for detail in row.get("queries", [])]
        if not ranks:
            continue
        bins = np.arange(1, max(ranks) + 2)
        hist, _ = np.histogram(ranks, bins=bins)
        ax.plot(bins[:-1], hist, marker="o", label=row["method"])
    ax.set_title("True-Rank Distribution")
    ax.set_xlabel("Ground-truth rank")
    ax.set_ylabel("Count")
    if results:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return os.path.basename(out_path)


def _write_summary_fallback(results: List[Dict], topk: int, out_path: str) -> str:
    try:
        Image, ImageDraw, _, _ = _load_pil()
    except Exception:
        txt_path = os.path.splitext(out_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Goal Matching Summary\n")
            f.write(
                f"{'method':<16} {'status':<8} {'top1':>8} {('top' + str(topk)):>8} {'mrr':>8} {'lat_ms':>12} {'lat_p95':>12} {'error':<40}\n"
            )
            for row in results:
                f.write(
                    f"{row['method']:<16} "
                    f"{row.get('status', 'ok'):<8} "
                    f"{row['top1_acc'] * 100:>7.2f}% "
                    f"{row[f'top{topk}_acc'] * 100:>7.2f}% "
                    f"{row['mrr']:>8.3f} "
                    f"{row['latency_ms_mean']:>12.2f} "
                    f"{row['latency_ms_p95']:>12.2f} "
                    f"{(row.get('error') or '')[:40]:<40}\n"
                )
        return os.path.basename(txt_path)

    font = _load_font()
    line_h = 18
    width = 920
    height = 40 + line_h * max(2, len(results) + 2)
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), "Goal Matching Summary", fill=(20, 20, 20), font=font)
    header = (
        f"{'method':<16} {'status':<8} {'top1':>8} {('top' + str(topk)):>8} "
        f"{'mrr':>8} {'lat_ms':>12} {'lat_p95':>12} {'error':<40}"
    )
    draw.text((12, 28), header, fill=(40, 40, 40), font=font)
    for idx, row in enumerate(results):
        text = (
            f"{row['method']:<16} "
            f"{row.get('status', 'ok'):<8} "
            f"{row['top1_acc'] * 100:>7.2f}% "
            f"{row[f'top{topk}_acc'] * 100:>7.2f}% "
            f"{row['mrr']:>8.3f} "
            f"{row['latency_ms_mean']:>12.2f} "
            f"{row['latency_ms_p95']:>12.2f} "
            f"{(row.get('error') or '')[:40]:<40}"
        )
        draw.text((12, 46 + idx * line_h), text, fill=(20, 20, 20), font=font)
    image.save(out_path)
    return os.path.basename(out_path)


def generate_visual_report(
    results: List[Dict],
    *,
    topk: int,
    outdir: str,
    failure_limit: int,
    report_json_path: str,
) -> None:
    _ensure_dir(outdir)
    summary_path = os.path.join(outdir, "summary.png")
    summary_asset = _plot_summary(results, topk=topk, out_path=summary_path)

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Goal Matching Report</title></head><body>",
        "<h1>Goal Matching Report</h1>",
        f"<p>Summary JSON: <code>{html.escape(report_json_path)}</code></p>",
        "<h2>Summary</h2>",
    ]
    if summary_asset.endswith(".png"):
        html_lines.append(f"<img src='{html.escape(summary_asset)}' style='max-width: 100%; border: 1px solid #ddd;' />")
    else:
        html_lines.append(f"<p><a href='{html.escape(summary_asset)}'>summary text fallback</a></p>")
    html_lines.extend([
        "<h2>Methods</h2>",
        "<table border='1' cellspacing='0' cellpadding='6'>",
        "<tr><th>Method</th><th>Status</th><th>Top1</th><th>TopK</th><th>MRR</th><th>Latency</th><th>Avg Query Cands</th><th>Error</th><th>CSV</th></tr>",
    ])

    for row in results:
        method = row["method"]
        method_csv = f"{method}_queries.csv"
        _write_method_csv(row, os.path.join(outdir, method_csv))
        html_lines.append(
            "<tr>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{html.escape(row.get('status', 'ok'))}</td>"
            f"<td>{row['top1_acc'] * 100:.2f}%</td>"
            f"<td>{row[f'top{topk}_acc'] * 100:.2f}%</td>"
            f"<td>{row['mrr']:.3f}</td>"
            f"<td>{row['latency_ms_mean']:.2f} ms</td>"
            f"<td>{row.get('avg_query_candidates', 0.0):.2f}</td>"
            f"<td>{html.escape(row.get('error', ''))}</td>"
            f"<td><a href='{html.escape(method_csv)}'>csv</a></td>"
            "</tr>"
        )

        if row.get("status", "ok") != "ok":
            continue

        failures = [detail for detail in row.get("queries", []) if not detail.get("correct_top1")]
        failures.sort(
            key=lambda detail: (
                -(detail.get("true_rank") or 0),
                -(detail.get("margin_pred_minus_true") or 0.0),
            )
        )
        if failures:
            failure_png = f"{method}_failures.png"
            try:
                _save_contact_sheet(
                    failures[:failure_limit],
                    os.path.join(outdir, failure_png),
                    title=f"{method} failures",
                )
                html_lines.extend(
                    [
                        f"<h3>{html.escape(method)} failures</h3>",
                        f"<img src='{html.escape(failure_png)}' style='max-width: 100%; border: 1px solid #ddd;' />",
                    ]
                )
            except Exception:
                html_lines.append(
                    f"<p>{html.escape(method)} failures available in CSV only; image rendering deps are missing.</p>"
                )

    html_lines.extend(["</table>", "</body></html>"])
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
