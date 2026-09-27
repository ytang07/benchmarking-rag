"""End-to-end bounded benchmark orchestration for live and offline runs."""

import json
from pathlib import Path
from time import perf_counter

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


def _endpoint_specs(config: BenchmarkConfig):
    return tuple(
        (f"DATABRICKS_CHAT_ENDPOINT_{i}", endpoint)
        for i, endpoint in enumerate(config.chat_endpoints, 1)
    )


def _skip_records(
    data: PreparedData, model_key: str, config: BenchmarkConfig, reason: str
) -> list[dict]:
    return [
        {
            "dataset": data.dataset,
            "embedding_model": model_key,
            "chat_endpoint_env": env_name,
            "chat_endpoint": endpoint,
            "status": "skipped",
            "reason": reason,
            "skipped": data.skipped,
        }
        for env_name, endpoint in _endpoint_specs(config)
    ]


def run_benchmark(
    config: BenchmarkConfig,
    data: PreparedData,
    model_key: str,
    smoke: bool = False,
    persist: bool = True,
) -> list[dict]:
    """Run one dataset/model and emit one result per configured endpoint."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown embedding model {model_key}")
    if not data.documents:
        records = _skip_records(data, model_key, config, "no documents after validation")
        if persist:
            _persist(config.results_path, records)
        return records
    started = perf_counter()
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
    connection = None
    table = f"rag_chunks_{model_key}"
    ingest_started = perf_counter()
    if not smoke:
        connection = db.connect(config.require_database())
        db.setup(connection, expected_dimension, config.hnsw_m, config.hnsw_ef_construction, table)
        db.clear_dataset(connection, data.dataset, table)
        db.insert_chunks(connection, data.dataset, chunks, vectors, table, expected_dimension)
    ingest_seconds = perf_counter() - ingest_started
    queries = data.queries or (
        [{"id": "smoke-query", "text": "How does vector retrieval work?"}] if smoke else []
    )
    if not queries:
        records = _skip_records(
            data, model_key, config, "no queries supplied; provide QUERY_SOURCE for this dataset"
        )
        if connection:
            connection.close()
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
            exact_ids = hnsw_ids
            hnsw_seconds = query_embedding_seconds
            exact_seconds = 0.0
        else:
            hnsw_started = perf_counter()
            hnsw_rows = db.search(
                connection, query_vector, data.dataset, config.top_k, config.hnsw_ef_search, table
            )
            hnsw_seconds = perf_counter() - hnsw_started
            hnsw_ids = [row[0] for row in hnsw_rows]
            hnsw_scores = [float(row[2]) for row in hnsw_rows]
            exact_started = perf_counter()
            exact_rows = db.exact_search(
                connection, query_vector, data.dataset, config.top_k, table
            )
            exact_seconds = perf_counter() - exact_started
            exact_ids = [row[0] for row in exact_rows]
        relevant = {
            doc
            for doc, score in data.qrels.get(query["id"], {}).items()
            if score >= config.qrel_min_score
        }
        hnsw_eval_ids = [_document_id(doc_id) for doc_id in hnsw_ids]
        exact_eval_ids = [_document_id(doc_id) for doc_id in exact_ids]
        retrieval_metrics = None
        if relevant:
            retrieval_metrics = {
                "hnsw_recall@k": recall_at_k(hnsw_eval_ids, relevant, config.top_k),
                "hnsw_mrr@k": mrr_at_k(hnsw_eval_ids, relevant, config.top_k),
                "exact_recall@k": recall_at_k(exact_eval_ids, relevant, config.top_k),
            }
        context = "\n\n".join(text for doc_id, text in chunks if doc_id in hnsw_ids)
        for endpoint_env, endpoint in _endpoint_specs(config):
            record = {
                "dataset": data.dataset,
                "embedding_model": model_key,
                "chat_endpoint_env": endpoint_env,
                "chat_endpoint": endpoint,
                "query_id": query["id"],
                "status": "ok",
                "retrieval": {
                    "hnsw_top_ids": hnsw_ids,
                    "hnsw_scores": hnsw_scores,
                    "exact_top_ids": exact_ids,
                },
                "timing_seconds": {
                    "embedding": embedding_seconds + query_embedding_seconds,
                    "ingest": ingest_seconds,
                    "hnsw_search": hnsw_seconds,
                    "exact_scan": exact_seconds,
                    "total": perf_counter() - started,
                },
                "qrel_threshold": config.qrel_min_score,
                "native_qrels": data.native_qrels,
                "retrieval_metrics": retrieval_metrics,
            }
            if not endpoint or not config.gateway_base_url or not config.gateway_token:
                record["answer_metrics"] = answer_metrics(
                    "", "skipped", query.get("reference_answer")
                )
                record["answer_metrics"]["reason"] = (
                    "gateway endpoint, base URL, or token not configured"
                )
            else:
                answer_started = perf_counter()
                try:
                    generated = answer(
                        query["text"],
                        context,
                        endpoint,
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
            records.append(record)
    if connection:
        connection.close()
    if persist:
        _persist(config.results_path, records)
    return records


def run_matrix(config: BenchmarkConfig, smoke: bool = False, persist: bool = True) -> list[dict]:
    """Execute all 27 dataset/model/endpoint combinations."""
    if not smoke:
        config.require_gateway()
    output = []
    for dataset_name in ("vidore", "parsebench", "govdocs"):
        if smoke:
            data = PreparedData(
                dataset_name,
                documents=smoke_documents(),
                queries=[{"id": "smoke-query", "text": "How does vector retrieval work?"}],
                qrels={"smoke-query": {"smoke-1": 1.0}},
            )
        else:
            raw = load_optional(
                dataset_name, config.parsebench_max_rows, config.govdocs_max_documents
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
            )
        for model_key in MODELS:
            output.extend(run_benchmark(config, data, model_key, smoke=smoke, persist=persist))
    return output
