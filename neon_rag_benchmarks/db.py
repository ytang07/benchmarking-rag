"""Optional Neon/pgvector persistence."""

from .schema import create_schema_sql, search_sql


def connect(database_url: str):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install the full extra for Neon") from exc
    return psycopg.connect(database_url)


def setup(connection, dimension: int, m: int, ef_construction: int) -> None:
    with connection.cursor() as cur:
        for statement in create_schema_sql(dimension, m=m, ef_construction=ef_construction):
            cur.execute(statement)
    connection.commit()


def search(connection, vector: list[float], dataset: str, k: int, ef_search: int):
    with connection.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (ef_search,))
        cur.execute(search_sql(), (vector, dataset, vector, k))
        return cur.fetchall()
