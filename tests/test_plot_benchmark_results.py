import importlib.util
import json
from pathlib import Path


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
    assert stats == {"lines": 4, "records": 3, "malformed": 1}
    group = groups[("minilm", "chat", "vidore", "bounded", "unknown", "bounded", "unknown")]
    assert group.count == 1 and group.metrics["retrieval_quality"] == [0.5]
    assert group.metrics["answer_correctness"] == [0.8]
    skipped = groups[("minilm", "unknown", "unknown", "skipped", "unknown", "unknown", "unknown")]
    assert skipped.skipped == 1


def test_summary_writes_compact_csv_and_json(tmp_path):
    groups, stats = plot.stream_aggregate([json.dumps({"embedding_model": "m", "status": "ok"})])
    plot.write_summary(tmp_path, groups, stats)
    assert "answer_correctness_count" in (tmp_path / "benchmark_summary.csv").read_text()
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert summary["input_stats"]["records"] == 1
    assert summary["groups"][0]["latency_total_mean"] == ""
