import math
from pathlib import Path
import pytest
from neon_rag_benchmarks.config import BenchmarkConfig, MODELS
from neon_rag_benchmarks import db
from neon_rag_benchmarks import pipeline
from neon_rag_benchmarks.datasets import _extract_pdf, prepare_records, validate_record
from neon_rag_benchmarks.metrics import (
    answer_metrics,
    exact_cosine_search,
    mrr_at_k,
    parse_claims,
    parse_citations,
    recall_at_k,
    validate_citations,
)
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


def test_offline_matrix_executes_vidore_first_paths(tmp_path):
    config = BenchmarkConfig.from_env({"RESULTS_PATH": str(tmp_path / "results.jsonl")})
    results = run_matrix(config, smoke=True, persist=True)
    assert len(results) == 3
    assert all(row["status"] == "ok" for row in results)
    assert results[0]["retrieval_metrics"]["synthetic_hnsw_recall@k"] == 1.0
    assert results[0]["evaluation_scope"] == "synthetic"
    assert results[0]["native_qrels"] is False
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 3
    assert len({row["run_id"] for row in results}) == 1
    assert {row["chat_model_env"] for row in results} == {"DATABRICKS_MODEL"}
    assert {row["chat_model"] for row in results} == {None}


def test_public_workflow_rejects_non_vidore_datasets():
    config = BenchmarkConfig.from_env({})
    with pytest.raises(ValueError, match="Vidore-only workflow"):
        run_matrix(config, smoke=True, persist=False, dataset_names=("parsebench",))
    with pytest.raises(ValueError, match="Vidore-only smoke"):
        run_smoke("govdocs")


def test_chunking_is_bounded():
    assert chunk_text("abcdefgh", size=4, overlap=1) == ["abcd", "defg", "gh"]


def test_vector_and_answer_validation(tmp_path):
    with pytest.raises(ValueError):
        validate_vectors([[1.0]], 2)
    result = answer_metrics("A correct answer!", "ok", "a correct answer")
    assert result["normalized_exact_match"] is True
    assert result["correctness"]["scope"] == "legacy_fallback_not_vidore_native"
    assert result["normalized_exact_match_scope"] == "legacy_fallback_not_vidore_native"
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
    from pypdf import PdfWriter
    import io

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    valid_pdf = buffer.getvalue()
    data = prepare_records(
        "govdocs",
        [
            {"id": "a", "broken_pdf": "false", "pdf": {"data": valid_pdf}},
            {"id": "b", "broken_pdf": 0, "pdf": {"bytes": valid_pdf}},
        ],
        max_documents=10,
        max_bytes=len(valid_pdf) + 1,
        pdf_max_document_bytes=len(valid_pdf) + 1,
    )
    assert data.metadata["raw_pdf_bytes_consumed"] == len(valid_pdf)
    assert data.metadata["raw_pdf_budget_exhausted"] is True
    assert any("cumulative raw PDF byte budget" in item["reason"] for item in data.skipped)


def test_malformed_pdf_is_a_stable_document_skip():
    data = prepare_records(
        "govdocs",
        [{"id": "bad", "broken_pdf": False, "pdf": {"bytes": b"not pdf"}}],
        max_bytes=100,
        pdf_max_document_bytes=100,
    )
    assert any(
        item["id"] == "bad" and item["reason"].startswith("malformed PDF bytes")
        for item in data.skipped
    )


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
    events = []
    monkeypatch.setattr(pipeline, "warm_model", lambda model: events.append("initialize"))
    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, model: (events.append("corpus_embedding") or [[0.0] * 384 for _ in texts]),
    )
    config = BenchmarkConfig.from_env({"DATABASE_URL": "postgres://redacted"})
    data = pipeline.PreparedData("vidore", documents=[{"id": "d", "text": "text"}], queries=[])
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_benchmark(config, data, "minilm", persist=False)
    assert connection.rolled_back and connection.closed
    assert events == ["initialize", "corpus_embedding"]


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
            sql = self.calls[-1][0]
            if "pg_constraint" in sql:
                return [("p", "id")]
            if "pg_attribute" in sql:
                return self.columns
            return self.indexes

    class Connection:
        def __init__(self, columns, indexes):
            self.cursor_obj = Cursor(columns, indexes)

        def cursor(self):
            return self.cursor_obj

    columns = [
        ("id", "bigint", True, "", "nextval('seq'::regclass)"),
        *[
            (name, "vector(384)" if name == "embedding" else "text", True, "", "")
            for name in ("run_id", "dataset", "doc_id", "content", "embedding")
        ],
    ]
    good = Connection(columns, [("CREATE INDEX USING hnsw (embedding vector_cosine_ops)",)])
    db.validate_table_schema(good, "rag_chunks_minilm", 384)
    bad = Connection(columns, [("CREATE INDEX USING hnsw (embedding vector_l2_ops)",)])
    with pytest.raises(RuntimeError, match="no HNSW cosine index"):
        db.validate_table_schema(bad, "rag_chunks_minilm", 384)
    bad_id_columns = [("id", "text", True, "", "nextval('seq'::regclass)"), *columns[1:]]
    with pytest.raises(RuntimeError, match="id must be BIGINT"):
        db.validate_table_schema(
            Connection(
                bad_id_columns, [("CREATE INDEX USING hnsw (embedding vector_cosine_ops)",)]
            ),
            "rag_chunks_minilm",
            384,
        )


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
    assert data.evaluation_scope == "bounded_sample" and data.native_qrels is True
    assert data.provenance_scope == "bounded_sample"
    assert len(data.native_provenance["corpus"]) == 1
    assert len(data.native_provenance["queries"]) == 1
    assert data.qrels == {"q1": {"d1": 1.0}}
    result = pipeline.run_benchmark(
        BenchmarkConfig.from_env({}), data, "minilm", smoke=True, persist=False
    )
    metrics = result[0]["retrieval_metrics"]
    assert "bounded_sample_hnsw_recall@k" in metrics
    assert "native_hnsw_recall@k" not in metrics
    assert result[0]["provenance_scope"] == "bounded_sample"


def test_native_answer_fallback_normalizes_empty_raw_answers():
    assert pipeline._native_answers({"raw_answers": [], "answer": "native"}) == "native"
    assert pipeline._native_answers({"raw_answers": [""], "answer": "native"}) == "native"
    assert pipeline._native_answers({"raw_answers": ["", "preferred"], "answer": "native"}) == [
        "preferred"
    ]
    assert (
        pipeline._native_answers({"raw_answers": {"bad": "shape"}, "answer": "native"}) == "native"
    )
    assert pipeline._native_answers({"raw_answers": ["", 3], "answer": "native"}) == "native"


def test_exact_scan_false_does_not_report_unmeasured_exact_metrics():
    config = BenchmarkConfig.from_env({"EXACT_SCAN": "false"})
    data = pipeline.PreparedData(
        "vidore",
        documents=[{"id": "d", "text": "text"}],
        queries=[{"id": "q", "text": "text"}],
        qrels={"q": {"d": 1.0}},
        evaluation_scope="bounded_sample",
    )
    result = pipeline.run_benchmark(config, data, "minilm", smoke=True, persist=False)[0]
    assert "exact_recall@k" not in result["retrieval_metrics"]
    assert result["exact_unavailable_reason"] == "EXACT_SCAN=false"


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


def test_vidore_native_answers_and_evidence_are_retained():
    data = prepare_records(
        "vidore",
        {
            "corpus": [
                {
                    "corpus_id": "doc-1",
                    "markdown": "Evidence",
                    "page_number": 4,
                    "bbox": [1, 2, 3, 4],
                    "metadata": {"native": b"bytes"},
                    "arbitrary_native_field": {"nested": [1, b"raw"]},
                }
            ],
            "queries": [
                {
                    "query_id": "q-1",
                    "query": "What?",
                    "answer": "Evidence",
                    "raw_answers": ["Evidence"],
                    "evidence": [{"page": 4}],
                    "arbitrary_query_field": {"native": True},
                }
            ],
            "qrels": [{"query_id": "q-1", "corpus_id": "doc-1", "score": 1}],
        },
    )
    assert data.queries[0]["raw_answers"] == ["Evidence"]
    assert data.documents[0]["page_number"] == 4
    assert data.documents[0]["bbox"] == [1, 2, 3, 4]
    assert data.documents[0]["metadata"] == {"native": "b'bytes'"}
    assert data.documents[0]["native_provenance"]["arbitrary_native_field"] == {
        "nested": [1, "b'raw'"]
    }
    assert data.native_provenance["corpus"][0]["arbitrary_native_field"] == {
        "nested": [1, "b'raw'"]
    }
    assert data.native_provenance["queries"][0]["arbitrary_query_field"] == {"native": True}


def test_citation_validation_and_honest_answer_metric_availability():
    passages = [{"citation_id": "doc#chunk-0|page=4", "text": "Evidence"}]
    citations = parse_citations("Evidence [doc#chunk-0|page=4] [missing]")
    checked = validate_citations(citations, passages)
    assert checked["valid"] == ["doc#chunk-0|page=4"]
    assert checked["invalid"] == ["missing"]
    metrics = answer_metrics(
        "Evidence [doc#chunk-0|page=4]", "ok", citations=citations[:1], passages=passages
    )
    assert metrics["correctness"]["available"] is False
    assert metrics["answer_relevance"]["value"] is None
    assert metrics["groundedness"]["available"] is True
    assert metrics["groundedness"]["label"] == "lexical_heuristic_not_semantic_entailment"


def test_claim_citations_do_not_pool_unrelated_passages_or_allow_uncited_claims():
    passages = [
        {"citation_id": "doc-a#chunk-0", "text": "Alpha is a color."},
        {"citation_id": "doc-b#chunk-0", "text": "Beta is a fruit."},
    ]
    claims = parse_claims("Alpha is a color [doc-a#chunk-0]. Beta is a fruit.")
    assert claims[0]["citations"] == ["doc-a#chunk-0"]
    assert claims[1]["citations"] == []
    metrics = answer_metrics(
        "Alpha is a color [doc-a#chunk-0]. Beta is a fruit.",
        "ok",
        citations=["doc-a#chunk-0"],
        passages=passages,
    )
    assert metrics["claim_level_citation_completeness"]["value"] == 0.5
    assert metrics["groundedness"]["available"] is False
    assert metrics["claims"][1]["grounded"] is None


def test_vidore_result_persists_answer_provenance_and_timing(tmp_path):
    data = pipeline.PreparedData(
        "vidore",
        documents=[{"id": "doc-1", "text": "Evidence", "page": 4}],
        queries=[
            {
                "id": "q-1",
                "text": "What?",
                "raw_answers": [],
                "answer": "Evidence",
            }
        ],
        qrels={"q-1": {"doc-1": 1.0}},
        native_qrels=True,
        evaluation_scope="full_dataset",
        native_provenance={
            "corpus": [{"id": "doc-1", "arbitrary": {"value": "kept"}}],
            "queries": [{"id": "q-1", "arbitrary": [1, 2, 3]}],
        },
    )
    config = BenchmarkConfig.from_env({"RESULTS_PATH": str(tmp_path / "results.jsonl")})
    result = pipeline.run_benchmark(config, data, "minilm", smoke=True, persist=True)[0]
    assert result["answer"] is None
    assert result["retrieved_passages"][0]["page"] == 4
    assert result["retrieved_passages"][0]["document_id"] == "doc-1"
    assert result["retrieved_passages"][0]["text"] == "Evidence"
    assert result["answer_metrics"]["correctness"]["available"] is True
    assert result["timing_seconds"]["initialization"] == 0.0
    assert result["vidore_provenance"]["corpus"]
    persisted = (tmp_path / "results.jsonl").read_text()
    assert '"arbitrary": {"value": "kept"}' in persisted
    assert result["timing_seconds"]["rag_evaluation"] >= 0
    assert '"retrieved_passages"' in (tmp_path / "results.jsonl").read_text()


def test_notebook_is_vidore_three_model_workflow():
    import json

    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "benchmark_matrix.ipynb").read_text()
    )
    source = "\n".join("\n".join(cell["source"]) for cell in notebook["cells"])
    assert "9 dataset/embedding-model" not in source
    assert "selected_count = len(dataset_names) * len(model_keys or tuple(MODELS))" in source
    assert "Answer relevance is unavailable without a judge" in source
