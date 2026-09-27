"""Optional Neon/pgvector persistence."""

from .schema import create_schema_sql, search_sql


def connect(database_url: str):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install the full extra for Neon") from exc
    connection = psycopg.connect(database_url)
    try:
        from pgvector.psycopg import register_vector

        register_vector(connection)
    except ImportError as exc:
        connection.close()
        raise RuntimeError("Install pgvector to register vector parameters with psycopg") from exc
    return connection


def setup(
    connection, dimension: int, m: int, ef_construction: int, table: str = "rag_chunks"
) -> None:
    with connection.cursor() as cur:
        for statement in create_schema_sql(
            dimension, table=table, m=m, ef_construction=ef_construction
        ):
            cur.execute(statement)
    connection.commit()


def insert_chunks(
    connection,
    dataset: str,
    chunks: list[tuple[str, str]],
    vectors: list[list[float]],
    table: str = "rag_chunks",
) -> None:
    if not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    with connection.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table} (dataset, doc_id, content, embedding) VALUES (%s, %s, %s, %s)",
            [(dataset, doc_id, text, vector) for (doc_id, text), vector in zip(chunks, vectors)],
        )
    connection.commit()


def search(
    connection, vector: list[float], dataset: str, k: int, ef_search: int, table: str = "rag_chunks"
):
    with connection.cursor() as cur:
        if ef_search < 1:
            raise ValueError("ef_search must be >= 1")
        # SET does not accept bind parameters; this value is validated as an integer first.
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        cur.execute(search_sql(table), (vector, dataset, vector, k))
        return cur.fetchall()


def exact_search(connection, vector: list[float], dataset: str, k: int, table: str = "rag_chunks"):
    with connection.cursor() as cur:
        # Force PostgreSQL's exact scan for a fair baseline instead of allowing HNSW.
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(search_sql(table), (vector, dataset, vector, k))
        return cur.fetchall()
