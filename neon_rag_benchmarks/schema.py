"""Dimension-safe pgvector schema and SQL helpers."""


def validate_dimension(model_key: str, dimension: int) -> None:
    from .config import MODELS

    if model_key not in MODELS:
        raise ValueError(f"Unknown embedding model: {model_key}")
    expected = MODELS[model_key][1]
    if dimension != expected:
        raise ValueError(f"{model_key} produces {expected} dimensions, got {dimension}")


def validate_vectors(vectors: list[list[float]], expected_dimension: int) -> None:
    if not vectors:
        raise ValueError("cannot insert an empty vector batch")
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise ValueError(
                f"vector {index} has dimension {len(vector)}; expected {expected_dimension}"
            )


def create_schema_sql(
    dimension: int, table: str = "rag_chunks", m: int = 16, ef_construction: int = 128
) -> tuple[str, ...]:
    if dimension < 1 or not table.replace("_", "").isalnum():
        raise ValueError("invalid schema arguments")
    if m < 2 or ef_construction < 1:
        raise ValueError("HNSW parameters are invalid")
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE TABLE IF NOT EXISTS {table} (id BIGSERIAL PRIMARY KEY, run_id TEXT NOT NULL, dataset TEXT NOT NULL, doc_id TEXT NOT NULL, content TEXT NOT NULL, embedding vector({dimension}) NOT NULL)",
        f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {ef_construction})",
    )


def search_sql(table: str = "rag_chunks") -> str:
    if not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    return f"SELECT doc_id, content, 1 - (embedding <=> %s) AS score FROM {table} WHERE dataset = %s ORDER BY embedding <=> %s LIMIT %s"
