#!/usr/bin/env python3
"""Stream benchmark JSONL results into compact summaries and static plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

METRIC_FIELDS = (
    "answer_correctness", "answer_relevance", "judge_score", "retrieval_quality",
    "citation_validity", "citation_completeness", "groundedness", "latency_total",
    "latency_embedding", "latency_ingest", "latency_search", "latency_exact_scan", "latency_answer",
)
GROUP_FIELDS = (
    "embedding_model", "chat_model", "dataset", "evaluation_scope", "evaluation_scope_detail",
    "provenance_scope", "provenance_scope_detail",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _value(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_scalars(record: dict[str, Any]) -> dict[str, Any]:
    """Extract only bounded scalar data; unavailable metrics remain None."""
    retrieval = record.get("retrieval_metrics") or {}
    answer = record.get("answer_metrics") or {}
    timing = record.get("timing_seconds") or {}
    retrieval_quality = next(
        (_number(retrieval.get(name)) for name in (
            "native_hnsw_recall@k", "bounded_sample_hnsw_recall@k", "hnsw_recall@k",
            "native_exact_recall@k", "bounded_sample_exact_recall@k",
        ) if _number(retrieval.get(name)) is not None), None
    )
    return {
        "embedding_model": record.get("embedding_model") or "unknown",
        "chat_model": record.get("chat_model") or record.get("chat_model_env") or "unknown",
        "dataset": record.get("dataset") or "unknown",
        "evaluation_scope": record.get("evaluation_scope") or "unknown",
        "evaluation_scope_detail": record.get("evaluation_scope_detail") or "unknown",
        "provenance_scope": record.get("provenance_scope") or "unknown",
        "provenance_scope_detail": record.get("provenance_scope_detail") or "unknown",
        "skipped": str(record.get("status", "")).lower() in {"skipped", "error"}
        or record.get("evaluation_scope") == "skipped",
        "answer_correctness": _number(_value(answer, "correctness", "value")),
        "answer_relevance": _number(_value(answer, "answer_relevance", "value")),
        "judge_score": _number(_value(record, "judge", "value")),
        "retrieval_quality": retrieval_quality,
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
        self.metrics: dict[str, list[float]] = defaultdict(list)

    def add(self, values: dict[str, Any]) -> None:
        self.count += 1
        self.skipped += int(values["skipped"])
        for field in METRIC_FIELDS:
            if values[field] is not None:
                self.metrics[field].append(values[field])


def stream_aggregate(handle: Iterable[str]) -> tuple[dict[tuple[str, ...], _Accumulator], dict[str, int]]:
    """Aggregate JSONL line by line without retaining records or nested fields."""
    groups: dict[tuple[str, ...], _Accumulator] = {}
    stats = {"lines": 0, "records": 0, "malformed": 0}
    for line in handle:
        stats["lines"] += 1
        if not line.strip():
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
            values = accumulator.metrics[field]
            row[f"{field}_count"] = len(values)
            row[f"{field}_mean"] = sum(values) / len(values) if values else ""
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
        json.dumps({"input_stats": stats, "groups": rows, "fields_used": list(METRIC_FIELDS)}, indent=2) + "\n",
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
        return [float(r[f"{field}_mean"]) if r[f"{field}_mean"] != "" else math.nan for r in rows]

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

    bars("model_quality_comparison.png", "Model quality comparison", ("answer_correctness", "answer_relevance", "retrieval_quality"))
    bars("citation_groundedness_comparison.png", "Citation and groundedness comparison", ("citation_validity", "citation_completeness", "groundedness"))
    bars("latency_breakdown.png", "Latency breakdown", ("latency_embedding", "latency_ingest", "latency_search", "latency_exact_scan", "latency_answer"))

    def scatter(filename: str, x_field: str, y_field: str, title: str, xlabel: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in rows:
            x, y = row[f"{x_field}_mean"], row[f"{y_field}_mean"]
            if x != "" and y != "":
                ax.scatter(float(x), float(y), label=f"{row['embedding_model']} ({row['evaluation_scope']})")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        handles, labels_for_legend = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels_for_legend, fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)

    scatter("quality_vs_latency_frontier.png", "latency_total", "answer_correctness", "Quality vs latency", "mean total latency (seconds)", "mean answer correctness")
    scatter("retrieval_vs_answer_quality.png", "retrieval_quality", "answer_correctness", "Retrieval vs answer quality", "mean retrieval quality", "mean answer correctness")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="benchmark JSONL input")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for CSV/JSON and PNG files")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input JSONL does not exist: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as stream:
        groups, stats = stream_aggregate(stream)
    write_summary(args.output_dir, groups, stats)
    _plot(args.output_dir, groups)
    print(json.dumps({"input": str(args.input), "output_dir": str(args.output_dir), **stats, "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
