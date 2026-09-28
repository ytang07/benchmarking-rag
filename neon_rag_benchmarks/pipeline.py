"""End-to-end bounded benchmark orchestration for live and offline runs."""

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from . import db
from .config import BenchmarkConfig, MODELS
from .datasets import PreparedData, load_optional, prepare_records, smoke_documents
from .embeddings import embed_query, embed_texts
from .gateway import answer
from .metrics import answer_metrics, exact_cosine_search, mrr_at_k, recall_at_k
from .schema import validate_vectors


def chunk_text(text: str, size: int = 800, overlap: int = 80) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    return [text[start : start + size] for start in range(0, len(text), size - overlap)] or [""]


def _synthetic_vector(text: str, dimension: int = 8) -> list[float]:
    values = [0.0] * dimension
    for index, char in enumerate(text.encode("utf-8")):
        values[index % dimension] += (char % 31) / 31
    return values


def _document_id(chunk_id: str) -> str:
    return chunk_id.split("#chunk-", 1)[0]


def _persist(path: str, records: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _skip_records(
    data: PreparedData, model_key: str, config: BenchmarkConfig, reason: str, run_id: str
) -> list[dict]:
    return [
        {
            "dataset": data.dataset,
            "embedding_model": model_key,
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "chat_model_env": "DATABRICKS_MODEL",
            "chat_model": config.gateway_model,
            "status": "skipped",
            "reason": reason,
            "skipped": data.skipped,
            "evaluation_scope": data.evaluation_scope,
            "native_qrels": data.native_qrels,
            "timing_seconds": {
                "embedding": 0.0,
                "corpus_embedding": 0.0,
                "query_embedding": 0.0,
                "ingest": 0.0,
                "hnsw_search": 0.0,
                "exact_scan": 0.0,
                "answer": 0.0,
                "total": 0.0,
            },
        }
        for _ in (config.gateway_model,)
    ]


def _run_benchmark(
    config: BenchmarkConfig,
    data: PreparedData,
    model_key: str,
    smoke: bool = False,
    persist: bool = True,
    run_id: str | None = None,
    _connection=None,
) -> list[dict]:
    """Run one dataset/model and emit one result for the configured chat model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown embedding model {model_key}")
    run_id = run_id or str(uuid4())
    if not data.documents:
        records = _skip_records(data, model_key, config, "no documents after validation", run_id)
        if persist:
            _persist(config.results_path, records)
        return records
    chunks = [
        (doc["id"] + f"#chunk-{i}", chunk)
        for doc in data.documents
        for i, chunk in enumerate(chunk_text(doc["text"], config.chunk_size, config.chunk_overlap))
    ]
    embedding_started = perf_counter()
    vectors = (
        [_synthetic_vector(text) for _, text in chunks]
        if smoke
        else embed_texts([text for _, text in chunks], model_key)
    )
    embedding_seconds = perf_counter() - embedding_started
    expected_dimension = 8 if smoke else MODELS[model_key][1]
    validate_vectors(vectors, expected_dimension)
    connection = _connection
    table = f"rag_chunks_{model_key}"
    ingest_started = perf_counter()
    if not smoke:
        if connection is None:
            connection = db.connect(config.require_database())
        if db.table_exists(connection, table):
            db.validate_table_schema(connection, table, expected_dimension)
        db.setup(connection, expected_dimension, config.hnsw_m, config.hnsw_ef_construction, table)
        db.validate_table_schema(connection, table, expected_dimension)
        db.clear_dataset(connection, run_id, data.dataset, table)
        db.insert_chunks(
            connection, data.dataset, chunks, vectors, table, expected_dimension, run_id
        )
    ingest_seconds = perf_counter() - ingest_started
    queries = data.queries or (
        [{"id": "smoke-query", "text": "How does vector retrieval work?"}] if smoke else []
    )
    if not queries:
        records = _skip_records(
            data,
            model_key,
            config,
            "no queries supplied; provide QUERY_SOURCE for this dataset",
            run_id,
        )
        if persist:
            _persist(config.results_path, records)
        return records
    records = []
    for query in queries:
        query_embedding_started = perf_counter()
        query_vector = (
            _synthetic_vector(query["text"]) if smoke else embed_query(query["text"], model_key)
        )
        query_embedding_seconds = perf_counter() - query_embedding_started
        if smoke:
            ranked = exact_cosine_search(
                query_vector, list(zip([doc_id for doc_id, _ in chunks], vectors)), config.top_k
            )
            hnsw_ids = [item[0] for item in ranked]
            hnsw_scores = [item[1] for item in ranked]
            exact_ids = hnsw_ids if config.exact_scan else []
            hnsw_seconds = 0.0
            exact_seconds = 0.0
            retrieval_mode = (
                "synthetic_exact_emulation" if config.exact_scan else "synthetic_retrieval_only"
            )
        else:
            hnsw_started = perf_counter()
            hnsw_rows = db.search(
                connection,
                query_vector,
                run_id,
                data.dataset,
                config.top_k,
                config.hnsw_ef_search,
                table,
            )
            hnsw_seconds = perf_counter() - hnsw_started
            hnsw_ids = [row[0] for row in hnsw_rows]
            hnsw_scores = [float(row[2]) for row in hnsw_rows]
            if config.exact_scan:
                exact_started = perf_counter()
                exact_rows = db.exact_search(
                    connection, query_vector, run_id, data.dataset, config.top_k, table
                )
                exact_seconds = perf_counter() - exact_started
                exact_ids = [row[0] for row in exact_rows]
                retrieval_mode = "neon_hnsw_and_exact"
            else:
                exact_seconds = 0.0
                exact_ids = []
                retrieval_mode = "neon_hnsw_only"
        relevant = {
            doc
            for doc, score in data.qrels.get(query["id"], {}).items()
            if score >= config.qrel_min_score
        }
        hnsw_eval_ids = [_document_id(doc_id) for doc_id in hnsw_ids]
        exact_eval_ids = [_document_id(doc_id) for doc_id in exact_ids]
        retrieval_metrics = None
        if relevant:
            metric_prefix = (
                "native"
                if data.native_qrels and data.evaluation_scope == "full_dataset"
                else ("synthetic" if data.evaluation_scope == "synthetic" else "bounded_sample")
            )
            retrieval_metrics = {
                f"{metric_prefix}_hnsw_recall@k": recall_at_k(
                    hnsw_eval_ids, relevant, config.top_k
                ),
                f"{metric_prefix}_hnsw_mrr@k": mrr_at_k(hnsw_eval_ids, relevant, config.top_k),
            }
            if config.exact_scan:
                retrieval_metrics[f"{metric_prefix}_exact_recall@k"] = recall_at_k(
                    exact_eval_ids, relevant, config.top_k
                )
        context = "\n\n".join(text for doc_id, text in chunks if doc_id in hnsw_ids)
        phase_seconds = (
            embedding_seconds
            + query_embedding_seconds
            + ingest_seconds
            + hnsw_seconds
            + exact_seconds
        )
        record = {
            "dataset": data.dataset,
            "embedding_model": model_key,
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "chat_model_env": "DATABRICKS_MODEL",
            "chat_model": config.gateway_model,
            "query_id": query["id"],
            "status": "ok",
            "retrieval": {
                "mode": retrieval_mode,
                "hnsw_top_ids": hnsw_ids,
                "hnsw_scores": hnsw_scores,
                "exact_top_ids": exact_ids,
            },
            "timing_seconds": {
                "embedding": embedding_seconds + query_embedding_seconds,
                "corpus_embedding": embedding_seconds,
                "query_embedding": query_embedding_seconds,
                "ingest": ingest_seconds,
                "hnsw_search": hnsw_seconds,
                "exact_scan": exact_seconds,
                "total": phase_seconds,
            },
            "qrel_threshold": config.qrel_min_score,
            "native_qrels": data.native_qrels,
            "evaluation_scope": data.evaluation_scope,
            "retrieval_metrics": retrieval_metrics,
            "skipped_count": len(data.skipped),
            "skipped": data.skipped,
            "dataset_metadata": data.metadata,
        }
        if not config.exact_scan:
            record["exact_unavailable_reason"] = "EXACT_SCAN=false"
        if not config.gateway_model or not config.gateway_base_url or not config.gateway_token:
            record["answer_metrics"] = answer_metrics("", "skipped", query.get("reference_answer"))
            record["answer_metrics"]["reason"] = "gateway model, base URL, or token not configured"
            record["timing_seconds"]["answer"] = 0.0
        else:
            answer_started = perf_counter()
            try:
                generated = answer(
                    query["text"],
                    context,
                    config.gateway_model,
                    config.gateway_base_url,
                    config.gateway_token,
                )
                record["answer_metrics"] = answer_metrics(
                    generated, "ok", query.get("reference_answer")
                )
            except Exception as exc:
                record["answer_metrics"] = answer_metrics(
                    "", "error", query.get("reference_answer")
                )
                record["answer_metrics"]["reason"] = str(exc)
            record["timing_seconds"]["answer"] = perf_counter() - answer_started
        record["timing_seconds"]["total"] = phase_seconds + record["timing_seconds"]["answer"]
        records.append(record)
    if persist:
        _persist(config.results_path, records)
    return records


def run_benchmark(
    config: BenchmarkConfig,
    data: PreparedData,
    model_key: str,
    smoke: bool = False,
    persist: bool = True,
    run_id: str | None = None,
) -> list[dict]:
    """Run a benchmark and always rollback/close a live connection on every path."""
    if smoke or not data.documents:
        return _run_benchmark(config, data, model_key, smoke, persist, run_id)
    connection = None
    try:
        connection = db.connect(config.require_database())
        return _run_benchmark(config, data, model_key, smoke, persist, run_id, connection)
    except Exception:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            connection.close()


def run_matrix(
    config: BenchmarkConfig,
    smoke: bool = False,
    persist: bool | None = None,
    dataset_names: tuple[str, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> list[dict]:
    """Execute all dataset/embedding-model/chat-model combinations."""
    if persist is None:
        persist = config.persist_results
    if not smoke:
        config.require_gateway()
    # EXPERIMENT_ID is a human label; every invocation gets a fresh immutable run id.
    run_id = str(uuid4())
    output = []
    selected_datasets = dataset_names or ("vidore", "parsebench", "govdocs")
    selected_models = model_keys or tuple(MODELS)
    for dataset_name in selected_datasets:
        if smoke:
            data = PreparedData(
                dataset_name,
                documents=smoke_documents(),
                queries=[{"id": "smoke-query", "text": "How does vector retrieval work?"}],
                qrels={"smoke-query": {"smoke-1": 1.0}},
                native_qrels=False,
                evaluation_scope="synthetic",
            )
        else:
            raw = load_optional(
                dataset_name,
                config.parsebench_max_rows,
                config.govdocs_max_documents,
                config.govdocs_config,
                config.govdocs_split,
            )
            data = prepare_records(
                dataset_name,
                raw,
                config.seed,
                config.parsebench_max_rows,
                config.govdocs_max_documents,
                config.govdocs_max_bytes,
                config.query_source,
                config.govdocs_pdf_extractor,
                config.govdocs_max_pages,
                config.govdocs_max_text_chars,
                config.govdocs_max_document_bytes,
                config.govdocs_max_text_bytes,
            )
        for model_key in selected_models:
            output.extend(
                run_benchmark(config, data, model_key, smoke=smoke, persist=persist, run_id=run_id)
            )
    return output
