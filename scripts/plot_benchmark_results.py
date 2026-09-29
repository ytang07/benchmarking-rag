#!/usr/bin/env python3
"""Stream benchmark JSONL results into compact summaries and static plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

METRIC_FIELDS = (
    "answer_correctness", "answer_relevance", "judge_score",
    "retrieval_native_hnsw", "retrieval_bounded_sample_hnsw",
    "retrieval_native_exact", "retrieval_bounded_sample_exact",
    "citation_validity", "citation_completeness", "groundedness", "latency_total",
    "latency_embedding", "latency_ingest", "latency_search", "latency_exact_scan", "latency_answer",
)
GROUP_FIELDS = (
    "embedding_model", "chat_model", "dataset", "retrieval_mode", "evaluation_scope", "evaluation_scope_detail",
    "provenance_scope", "provenance_scope_detail",
)
MAX_CATEGORY_LENGTH = 128


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _category(value: Any) -> str:
    """Return a bounded grouping label without stringifying nested values."""
    if not isinstance(value, str):
        return "unknown"
    value = value.strip()
    if not value:
        return "unknown"
    if len(value) > MAX_CATEGORY_LENGTH:
        return value[:MAX_CATEGORY_LENGTH] + "..."
    return value


def _value(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_scalars(record: dict[str, Any]) -> dict[str, Any]:
    """Extract only bounded scalar data; unavailable metrics remain None."""
    containers = {name: record.get(name) for name in ("retrieval_metrics", "answer_metrics", "timing_seconds")}
    malformed = any(value is not None and not isinstance(value, dict) for value in containers.values())
    retrieval = containers["retrieval_metrics"] if isinstance(containers["retrieval_metrics"], dict) else {}
    answer = containers["answer_metrics"] if isinstance(containers["answer_metrics"], dict) else {}
    timing = containers["timing_seconds"] if isinstance(containers["timing_seconds"], dict) else {}
    status = _category(record.get("status"))
    evaluation_scope = _category(record.get("evaluation_scope"))
    return {
        "embedding_model": _category(record.get("embedding_model")),
        "chat_model": _category(record.get("chat_model") or record.get("chat_model_env")),
        "dataset": _category(record.get("dataset")),
        "retrieval_mode": _category(_value(record, "retrieval", "mode")),
        "evaluation_scope": evaluation_scope,
        "evaluation_scope_detail": _category(record.get("evaluation_scope_detail")),
        "provenance_scope": _category(record.get("provenance_scope")),
        "provenance_scope_detail": _category(record.get("provenance_scope_detail")),
        "malformed": malformed,
        "skipped": malformed or status.lower() in {"skipped", "error"}
        or evaluation_scope == "skipped",
        "answer_correctness": _number(_value(answer, "correctness", "value")),
        "answer_relevance": _number(_value(answer, "answer_relevance", "value")),
        "judge_score": _number(_value(record, "judge", "value")),
        "retrieval_native_hnsw": _number(retrieval.get("native_hnsw_recall@k")),
        "retrieval_bounded_sample_hnsw": _number(retrieval.get("bounded_sample_hnsw_recall@k")),
        "retrieval_native_exact": _number(retrieval.get("native_exact_recall@k")),
        "retrieval_bounded_sample_exact": _number(retrieval.get("bounded_sample_exact_recall@k")),
        "citation_validity": _number(_value(answer, "citation_validity", "value")),
        "citation_completeness": _number(_value(answer, "retrieval_passage_citation_completeness", "value")),
        "groundedness": _number(_value(answer, "groundedness", "value")),
        "latency_total": _number(timing.get("total")),
        "latency_embedding": _number(timing.get("embedding")),
        "latency_ingest": _number(timing.get("ingest")),
        "latency_search": _number(timing.get("hnsw_search")),
        "latency_exact_scan": _number(timing.get("exact_scan")),
        "latency_answer": _number(timing.get("answer")),
    }


class _Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.skipped = 0
        self.metrics: dict[str, dict[str, float | int | None]] = {
            field: {"count": 0, "sum": 0.0, "min": math.inf, "max": -math.inf, "overflow": False}
            for field in METRIC_FIELDS
        }

    def add(self, values: dict[str, Any]) -> None:
        self.count += 1
        self.skipped += int(values["skipped"])
        if values["skipped"]:
            return
        for field in METRIC_FIELDS:
            value = values[field]
            if value is not None:
                stats = self.metrics[field]
                stats["count"] += 1
                if stats["sum"] is not None:
                    candidate = stats["sum"] + value
                    stats["sum"] = candidate if math.isfinite(candidate) else None
                    stats["overflow"] = stats["overflow"] or stats["sum"] is None
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)


def stream_aggregate(handle: Iterable[str]) -> tuple[dict[tuple[str, ...], _Accumulator], dict[str, int]]:
    """Aggregate JSONL line by line without retaining records or nested fields."""
    groups: dict[tuple[str, ...], _Accumulator] = {}
    stats = {"lines": 0, "records": 0, "malformed": 0, "malformed_records": 0}
    for line in handle:
        stats["lines"] += 1
        if not line.strip():
            continue
        if any(0xDC80 <= ord(character) <= 0xDCFF for character in line):
            stats["malformed"] += 1
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["malformed"] += 1
            continue
        if not isinstance(record, dict):
            stats["malformed"] += 1
            continue
        values = extract_scalars(record)
        stats["malformed_records"] += int(values["malformed"])
        key = tuple(str(values[field]) for field in GROUP_FIELDS)
        groups.setdefault(key, _Accumulator()).add(values)
        stats["records"] += 1
    return groups, stats


def _summary_rows(groups: dict[tuple[str, ...], _Accumulator]) -> list[dict[str, Any]]:
    rows = []
    for key, accumulator in sorted(groups.items()):
        row: dict[str, Any] = dict(zip(GROUP_FIELDS, key))
        row.update({"records": accumulator.count, "skipped": accumulator.skipped})
        for field in METRIC_FIELDS:
            stats = accumulator.metrics[field]
            row[f"{field}_count"] = stats["count"]
            total = stats["sum"]
            row[f"{field}_mean"] = (
                total / stats["count"]
                if stats["count"] and total is not None and math.isfinite(total)
                else ""
            )
            row[f"{field}_min"] = stats["min"] if stats["count"] else ""
            row[f"{field}_max"] = stats["max"] if stats["count"] else ""
            row[f"{field}_overflow"] = bool(stats["overflow"])
        rows.append(row)
    return rows


def write_summary(output_dir: Path, groups: dict[tuple[str, ...], _Accumulator], stats: dict[str, int]) -> None:
    rows = _summary_rows(groups)
    fields = list(rows[0]) if rows else list(GROUP_FIELDS)
    with (output_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {"input_stats": stats, "groups": rows, "fields_used": list(METRIC_FIELDS)},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _plot(output_dir: Path, groups: dict[tuple[str, ...], _Accumulator]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install plotting dependencies with: pip install -e '.[plot]'") from exc
    rows = _summary_rows(groups)
    labels = [f"{r['embedding_model']}\n{r['evaluation_scope']}" for r in rows]

    def means(field: str) -> list[float]:
        result = []
        for row in rows:
            value = row[f"{field}_mean"]
            result.append(float(value) if value != "" and math.isfinite(float(value)) else math.nan)
        return result

    def bars(filename: str, title: str, fields: tuple[str, ...]) -> None:
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.3), 4.5))
        width = 0.8 / len(fields)
        for index, field in enumerate(fields):
            ax.bar([x + index * width for x in range(len(labels))], means(field), width, label=field.replace("_", " "))
        ax.set_xticks([x + width * (len(fields) - 1) / 2 for x in range(len(labels))], labels)
        ax.set_title(title)
        ax.set_ylabel("mean (available values)")
        if rows:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)

    bars("model_quality_comparison.png", "Answer quality comparison", ("answer_correctness", "answer_relevance", "judge_score"))
    bars(
        "retrieval_quality_comparison.png",
        "Retrieval quality comparison (separate scope and algorithm)",
        ("retrieval_native_hnsw", "retrieval_bounded_sample_hnsw", "retrieval_native_exact", "retrieval_bounded_sample_exact"),
    )
    bars("citation_groundedness_comparison.png", "Citation and groundedness comparison", ("citation_validity", "citation_completeness", "groundedness"))
    bars("latency_breakdown.png", "Latency breakdown", ("latency_embedding", "latency_ingest", "latency_search", "latency_exact_scan", "latency_answer"))

    def scatter(filename: str, x_fields: tuple[str, ...], y_field: str, title: str, xlabel: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in rows:
            y = row[f"{y_field}_mean"]
            if y != "":
                for x_field in x_fields:
                    x = row[f"{x_field}_mean"]
                    if x != "" and math.isfinite(float(x)) and math.isfinite(float(y)):
                        ax.scatter(float(x), float(y), label=f"{x_field} / {row['embedding_model']} ({row['evaluation_scope_detail']})")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        handles, labels_for_legend = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels_for_legend, fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)

    scatter("quality_vs_latency_frontier.png", ("latency_total",), "answer_correctness", "Quality vs latency", "mean total latency (seconds)", "mean answer correctness")
    scatter(
        "retrieval_vs_answer_quality.png",
        ("retrieval_native_hnsw", "retrieval_bounded_sample_hnsw", "retrieval_native_exact", "retrieval_bounded_sample_exact"),
        "answer_correctness", "Retrieval vs answer quality (labeled metrics)", "mean retrieval recall@k", "mean answer correctness",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="benchmark JSONL input")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for CSV/JSON and PNG files")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input JSONL does not exist: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8", errors="surrogateescape") as stream:
        groups, stats = stream_aggregate(stream)
    write_summary(args.output_dir, groups, stats)
    _plot(args.output_dir, groups)
    print(json.dumps({"input": str(args.input), "output_dir": str(args.output_dir), **stats, "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
