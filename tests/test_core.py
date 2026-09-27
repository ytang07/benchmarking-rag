import math
from pathlib import Path
import pytest
from neon_rag_benchmarks.config import BenchmarkConfig, MODELS
from neon_rag_benchmarks import db
from neon_rag_benchmarks import pipeline
from neon_rag_benchmarks.datasets import _extract_pdf, prepare_records, validate_record
from neon_rag_benchmarks.metrics import answer_metrics, exact_cosine_search, mrr_at_k, recall_at_k
from neon_rag_benchmarks.pipeline import chunk_text, run_matrix
from neon_rag_benchmarks.schema import create_schema_sql, validate_dimension, validate_vectors
from neon_rag_benchmarks.smoke import run_smoke


def test_config_and_model_dimensions():
    c = BenchmarkConfig.from_env({"HNSW_M": "20"})
    assert c.hnsw_m == 20
    assert MODELS["nomic"][1] == 768


def test_schema_is_dimension_safe():
    validate_dimension("minilm", 384)
    assert "vector(384)" in create_schema_sql(384)[1]
    with pytest.raises(ValueError):
        validate_dimension("minilm", 768)


def test_metrics():
    assert recall_at_k(["a", "b"], {"b"}, 2) == 1
    assert mrr_at_k(["a", "b"], {"b"}, 2) == 0.5
    assert math.isnan(recall_at_k([], set(), 1))
    assert exact_cosine_search([1, 0], [("a", [1, 0]), ("b", [0, 1])], 1) == [("a", 1.0)]


def test_dataset_validation():
    assert validate_record("vidore", {"id": "x", "markdown": "hello"}) == "x"
    with pytest.raises(ValueError):
        validate_record("govdocs", {"id": "x", "broken_pdf": True})


def test_offline_smoke():
    result = run_smoke()
    assert result["offline"] is True and result["recall@1"] == 1.0


def test_offline_matrix_executes_all_dataset_model_paths(tmp_path):
    config = BenchmarkConfig.from_env({"RESULTS_PATH": str(tmp_path / "results.jsonl")})
    results = run_matrix(config, smoke=True, persist=True)
    assert len(results) == 27
    assert all(row["status"] == "ok" for row in results)
    assert results[0]["retrieval_metrics"]["native_hnsw_recall@k"] == 1.0
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 27
    assert len({row["run_id"] for row in results}) == 1
    assert {row["chat_endpoint_env"] for row in results} == {
        "DATABRICKS_CHAT_ENDPOINT_1",
        "DATABRICKS_CHAT_ENDPOINT_2",
        "DATABRICKS_CHAT_ENDPOINT_3",
    }


def test_chunking_is_bounded():
    assert chunk_text("abcdefgh", size=4, overlap=1) == ["abcd", "defg", "gh"]


def test_vector_and_answer_validation(tmp_path):
    with pytest.raises(ValueError):
        validate_vectors([[1.0]], 2)
    result = answer_metrics("A correct answer!", "ok", "a correct answer")
    assert result["normalized_exact_match"] is True
    source = tmp_path / "queries.jsonl"
    source.write_text('{"query_id":"q1","text":"What?","relevant_doc_ids":"d1,d2"}\n')
    data = prepare_records(
        "parsebench", [{"id": "d1", "text_content": "text"}], query_source=str(source)
    )
    assert data.queries[0]["id"] == "q1" and set(data.qrels["q1"]) == {"d1", "d2"}
    object_source = tmp_path / "queries.json"
    object_source.write_text(
        '{"queries":[{"query_id":"q2","text":"Why?"}],"qrels":{"q2":{"d3":2}}}'
    )
    object_data = prepare_records(
        "parsebench", [{"id": "d3", "text_content": "text"}], query_source=str(object_source)
    )
    assert object_data.qrels == {"q2": {"d3": 2.0}}
    with pytest.raises(ValueError, match="raw PDF exceeds"):
        _extract_pdf({"bytes": b"too large"}, max_bytes=2)
    assert validate_record("govdocs", {"id": "x", "broken_pdf": "false"}) == "x"
    with pytest.raises(ValueError):
        validate_record("govdocs", {"id": "x", "broken_pdf": "unknown"})


def test_govdocs_cumulative_raw_budget_and_nested_pdf_shape():
    data = prepare_records(
        "govdocs",
        [
            {"id": "a", "broken_pdf": "false", "pdf": {"data": b"1234"}},
            {"id": "b", "broken_pdf": 0, "pdf": {"bytes": b"5678"}},
        ],
        max_documents=10,
        max_bytes=5,
        pdf_max_document_bytes=5,
    )
    assert data.metadata["raw_pdf_bytes_consumed"] == 4
    assert data.metadata["raw_pdf_budget_exhausted"] is True
    assert any("cumulative raw PDF byte budget" in item["reason"] for item in data.skipped)


def test_run_id_is_fresh_even_with_reused_experiment_label():
    config = BenchmarkConfig.from_env({"EXPERIMENT_ID": "same-label"})
    first = pipeline.run_matrix(config, smoke=True, persist=False)
    second = pipeline.run_matrix(config, smoke=True, persist=False)
    assert {row["run_id"] for row in first}.isdisjoint({row["run_id"] for row in second})
    assert first[0]["experiment_id"] == second[0]["experiment_id"] == "same-label"


def test_connection_is_rolled_back_and_closed_on_pipeline_failure(monkeypatch):
    class Connection:
        def __init__(self):
            self.rolled_back = False
            self.closed = False

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(db, "connect", lambda _: connection)
    monkeypatch.setattr(db, "table_exists", lambda *args: False)
    monkeypatch.setattr(db, "setup", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts, model: [[0.0] * 384 for _ in texts])
    config = BenchmarkConfig.from_env({"DATABASE_URL": "postgres://redacted"})
    data = pipeline.PreparedData("parsebench", documents=[{"id": "d", "text": "text"}], queries=[])
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_benchmark(config, data, "minilm", persist=False)
    assert connection.rolled_back and connection.closed


def test_db_cleanup_is_scoped_to_run_id():
    class Cursor:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, params):
            self.calls.append((sql, params))

    class Connection:
        def __init__(self):
            self.cursor_obj = Cursor()
            self.commits = 0

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.commits += 1

    connection = Connection()
    db.clear_dataset(connection, "run-a", "vidore", "rag_chunks_minilm")
    assert connection.cursor_obj.calls[0][1] == ("run-a", "vidore")
    assert connection.commits == 1


def test_existing_schema_must_have_expected_vector_and_hnsw_cosine_index():
    class Cursor:
        def __init__(self, columns, indexes):
            self.columns, self.indexes, self.calls = columns, indexes, []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, params):
            self.calls.append((sql, params))

        def fetchall(self):
            return self.columns if len(self.calls) == 1 else self.indexes

    class Connection:
        def __init__(self, columns, indexes):
            self.cursor_obj = Cursor(columns, indexes)

        def cursor(self):
            return self.cursor_obj

    columns = [
        (name, "vector(384)" if name == "embedding" else "text")
        for name in ("run_id", "dataset", "doc_id", "content", "embedding")
    ]
    good = Connection(columns, [("CREATE INDEX USING hnsw (embedding vector_cosine_ops)",)])
    db.validate_table_schema(good, "rag_chunks_minilm", 384)
    bad = Connection(columns, [("CREATE INDEX USING hnsw (embedding vector_l2_ops)",)])
    with pytest.raises(RuntimeError, match="no HNSW cosine index"):
        db.validate_table_schema(bad, "rag_chunks_minilm", 384)


def test_vidore_bounded_sample_renames_metrics_and_preserves_retained_qrels():
    raw = {
        "corpus": [{"id": "d1", "markdown": "one"}, {"id": "d2", "markdown": "two"}],
        "queries": [{"id": "q1", "query": "one"}, {"id": "q2", "query": "two"}],
        "qrels": [
            {"query_id": "q1", "corpus_id": "d1", "score": 1},
            {"query_id": "q2", "corpus_id": "d2", "score": 1},
        ],
    }
    data = prepare_records("vidore", raw, max_rows=1)
    assert data.metadata["evaluation_scope"] == "bounded_sample"
    assert data.qrels == {"q1": {"d1": 1.0}}
    result = pipeline.run_benchmark(
        BenchmarkConfig.from_env({}), data, "minilm", smoke=True, persist=False
    )
    metrics = result[0]["retrieval_metrics"]
    assert "bounded_sample_hnsw_recall@k" in metrics
    assert "native_hnsw_recall@k" not in metrics


def test_insert_rejects_mismatched_batches_without_database():
    with pytest.raises(ValueError, match="length mismatch"):
        db.insert_chunks(object(), "smoke", [("d", "text")], [], dimension=2)


def test_notebook_offline_execution():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook = Path(__file__).parents[1] / "notebooks" / "benchmark_matrix.ipynb"
    document = nbformat.read(notebook, as_version=4)
    nbformat.validate(document)
    nbclient.NotebookClient(document, timeout=60, kernel_name="python3").execute()
