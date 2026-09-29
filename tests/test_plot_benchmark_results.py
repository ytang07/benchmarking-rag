import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_benchmark_results.py"
SPEC = importlib.util.spec_from_file_location("plot_benchmark_results", SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot)


def test_stream_aggregate_extracts_metrics_and_preserves_unavailable_values():
    rows = [
        {
            "embedding_model": "minilm", "chat_model": "chat", "dataset": "vidore",
            "evaluation_scope": "bounded", "provenance_scope": "bounded", "status": "ok",
            "retrieval_metrics": {"bounded_sample_hnsw_recall@k": 0.5, "native_hnsw_recall@k": None},
            "answer_metrics": {"correctness": {"value": 0.8}, "groundedness": {"value": None}},
            "timing_seconds": {"total": 2.0, "hnsw_search": 0.4},
            "retrieved_passages": [{"text": "large field should not be retained"}],
        },
        {"embedding_model": "minilm", "evaluation_scope": "skipped", "status": "skipped",
         "retrieval_metrics": None, "answer_metrics": None, "timing_seconds": None},
        {"not": "a separate group"},
        "malformed",
    ]
    groups, stats = plot.stream_aggregate([json.dumps(row) if isinstance(row, dict) else row for row in rows])
    assert stats == {"lines": 4, "records": 3, "malformed": 1, "malformed_records": 0}
    group = groups[("minilm", "chat", "vidore", "unknown", "bounded", "unknown", "bounded", "unknown")]
    assert group.count == 1 and group.metrics["retrieval_bounded_sample_hnsw"]["count"] == 1
    assert group.metrics["retrieval_bounded_sample_hnsw"]["sum"] == 0.5
    assert group.metrics["retrieval_native_hnsw"]["count"] == 0
    assert group.metrics["answer_correctness"]["min"] == 0.8
    skipped = groups[("minilm", "unknown", "unknown", "unknown", "skipped", "unknown", "unknown", "unknown")]
    assert skipped.skipped == 1


def test_summary_writes_compact_csv_and_json(tmp_path):
    groups, stats = plot.stream_aggregate([json.dumps({"embedding_model": "m", "status": "ok"})])
    plot.write_summary(tmp_path, groups, stats)
    assert "answer_correctness_count" in (tmp_path / "benchmark_summary.csv").read_text()
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert summary["input_stats"]["records"] == 1
    assert summary["groups"][0]["latency_total_mean"] == ""


def test_unexpected_metric_containers_are_unavailable_and_non_fatal():
    groups, stats = plot.stream_aggregate([
        json.dumps({"embedding_model": "m", "retrieval_metrics": ["bad"], "status": "ok"}),
        json.dumps({"embedding_model": "m", "answer_metrics": "bad", "status": "ok"}),
    ])
    assert stats["records"] == 2 and stats["malformed_records"] == 2
    accumulator = next(iter(groups.values()))
    assert accumulator.skipped == 2
    assert accumulator.metrics["answer_correctness"]["count"] == 0


def test_skipped_metrics_do_not_enter_summary_statistics():
    rows = [
        json.dumps({"embedding_model": "m", "status": "ok", "timing_seconds": {"total": 2.0},
                    "answer_metrics": {"correctness": {"value": 0.4}}}),
        json.dumps({"embedding_model": "m", "status": "skipped", "timing_seconds": {"total": 999.0},
                    "answer_metrics": {"correctness": {"value": 1.0}}}),
    ]
    groups, stats = plot.stream_aggregate(rows)
    accumulator = next(iter(groups.values()))
    assert stats["records"] == 2 and accumulator.skipped == 1
    assert accumulator.metrics["latency_total"] == {
        "count": 1, "sum": 2.0, "min": 2.0, "max": 2.0, "overflow": False,
    }
    assert accumulator.metrics["answer_correctness"]["count"] == 1
    assert accumulator.metrics["answer_correctness"]["sum"] == 0.4


def test_aggregate_sum_overflow_is_marked_unavailable_and_json_safe(tmp_path):
    rows = [json.dumps({"embedding_model": "m", "status": "ok", "timing_seconds": {"total": 1e308}})] * 2
    groups, stats = plot.stream_aggregate(rows)
    accumulator = next(iter(groups.values()))
    metric = accumulator.metrics["latency_total"]
    assert metric["count"] == 2 and metric["overflow"] is True and metric["sum"] is None
    summary = plot._summary_rows(groups)[0]
    assert summary["latency_total_mean"] == ""
    plot.write_summary(tmp_path, groups, stats)
    serialized = (tmp_path / "benchmark_summary.json").read_text()
    assert "Infinity" not in serialized and "NaN" not in serialized
    json.loads(serialized)


def test_malformed_lines_extreme_numbers_and_grouping_fields_are_safe():
    huge = 10**4000
    groups, stats = plot.stream_aggregate([
        "not json",
        json.dumps({
            "embedding_model": {"nested": "data"}, "chat_model": ["large"],
            "dataset": "d" * 1000, "evaluation_scope": 7,
            "retrieval": {"mode": {"nested": "data"}},
            "timing_seconds": {"total": huge},
        }),
    ])
    assert stats["malformed"] == 1 and stats["records"] == 1
    key = next(iter(groups))
    assert key == ("unknown", "unknown", "d" * 128 + "...", "unknown", "unknown", "unknown", "unknown", "unknown")
    accumulator = groups[key]
    assert accumulator.metrics["latency_total"]["count"] == 0


def test_cli_writes_expected_outputs_for_explicit_paths(tmp_path):
    pytest.importorskip("matplotlib")
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "nested" / "plots"
    valid_line = json.dumps({
        "embedding_model": "minilm", "evaluation_scope": "bounded",
        "evaluation_scope_detail": "bounded_sample", "provenance_scope": "bounded",
        "provenance_scope_detail": "bounded_sample", "status": "ok",
        "retrieval_metrics": {"bounded_sample_exact_recall@k": 0.75},
        "answer_metrics": {"correctness": {"value": 0.5}},
        "timing_seconds": {"total": 1.0},
    }) + "\n"
    input_path.write_bytes(b"\xff\n" + valid_line.encode())
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(input_path), "--output-dir", str(output_dir)],
        check=True, capture_output=True, text=True,
    )
    assert '"records": 1' in result.stdout and '"malformed": 1' in result.stdout
    expected = {
        "benchmark_summary.csv", "benchmark_summary.json", "model_quality_comparison.png",
        "retrieval_quality_comparison.png", "citation_groundedness_comparison.png",
        "latency_breakdown.png", "quality_vs_latency_frontier.png", "retrieval_vs_answer_quality.png",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    summary = json.loads((output_dir / "benchmark_summary.json").read_text())
    row = summary["groups"][0]
    assert row["evaluation_scope_detail"] == "bounded_sample"
    assert row["retrieval_bounded_sample_exact_count"] == 1
    assert row["retrieval_native_exact_count"] == 0
