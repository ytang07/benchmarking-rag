"""Optional Neon/pgvector persistence."""

from .schema import create_schema_sql, search_sql
from .schema import validate_vectors


def validate_table_schema(connection, table: str, dimension: int) -> None:
    """Fail closed if a preexisting table is not this benchmark's vector schema."""
    if not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    with connection.cursor() as cur:
        cur.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull, a.attidentity, "
            "pg_get_expr(d.adbin, d.adrelid) FROM pg_attribute a "
            "LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
            "WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped",
            (table,),
        )
        rows = cur.fetchall()
        columns = {row[0]: str(row[1]).lower() for row in rows}
        not_null = {row[0] for row in rows if row[2]}
        identities = {row[0] for row in rows if row[3]}
        defaults = {row[0]: str(row[4] or "").lower() for row in rows}
        required = {"id", "run_id", "dataset", "doc_id", "content", "embedding"}
        missing = required - columns.keys()
        expected_vector = f"vector({dimension})"
        bad_types = {
            name
            for name in ("run_id", "dataset", "doc_id", "content")
            if columns.get(name) not in {"text", "character varying"}
        }
        if missing or bad_types or columns.get("embedding") != expected_vector:
            raise RuntimeError(
                f"incompatible existing table {table}: missing={sorted(missing)}, bad_types={sorted(bad_types)}, "
                f"embedding={columns.get('embedding')!r}, expected={expected_vector!r}"
            )
        if not required.issubset(not_null):
            raise RuntimeError(
                f"incompatible existing table {table}: required columns must be NOT NULL"
            )
        cur.execute(
            "SELECT c.contype, a.attname FROM pg_constraint c "
            "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey) "
            "WHERE c.conrelid = %s::regclass",
            (table,),
        )
        constraints = cur.fetchall()
        primary_key_columns = {name for kind, name in constraints if kind == "p"}
        id_generated = "id" in identities or "nextval(" in defaults.get("id", "")
        if columns.get("id") != "bigint" or "id" not in primary_key_columns or not id_generated:
            raise RuntimeError(
                f"incompatible existing table {table}: id must be BIGINT with a sequence/identity default and primary key"
            )
        cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = %s", (table,))
        definitions = [str(row[0]).lower() for row in cur.fetchall()]
        if not any(
            "using hnsw" in definition
            and "vector_cosine_ops" in definition
            and "embedding" in definition
            for definition in definitions
        ):
            raise RuntimeError(
                f"incompatible existing table {table}: no HNSW cosine index "
                "(USING hnsw with vector_cosine_ops)"
            )


def table_exists(connection, table: str) -> bool:
    if not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    with connection.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        return cur.fetchone()[0] is not None


def connect(database_url: str):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install the full extra for Neon") from exc
    connection = psycopg.connect(database_url)
    try:
        with connection.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()
        from pgvector.psycopg import register_vector

        register_vector(connection)
    except ImportError as exc:
        connection.close()
        raise RuntimeError("Install pgvector to register vector parameters with psycopg") from exc
    except Exception:
        connection.rollback()
        connection.close()
        raise
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


def clear_dataset(connection, run_id: str, dataset: str, table: str = "rag_chunks") -> None:
    if not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    with connection.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE run_id = %s AND dataset = %s", (run_id, dataset))
    connection.commit()


def insert_chunks(
    connection,
    dataset: str,
    chunks: list[tuple[str, str]],
    vectors: list[list[float]],
    table: str = "rag_chunks",
    dimension: int | None = None,
    run_id: str = "default",
) -> None:
    if not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
    if dimension is not None:
        validate_vectors(vectors, dimension)
    with connection.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table} (run_id, dataset, doc_id, content, embedding) VALUES (%s, %s, %s, %s, %s)",
            [(run_id, dataset, item[0], item[1], vector) for item, vector in zip(chunks, vectors)],
        )
    connection.commit()


def search(
    connection,
    vector: list[float],
    run_id: str,
    dataset: str,
    k: int,
    ef_search: int,
    table: str = "rag_chunks",
):
    with connection.cursor() as cur:
        if ef_search < 1:
            raise ValueError("ef_search must be >= 1")
        # SET does not accept bind parameters; this value is validated as an integer first.
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        cur.execute(
            search_sql(table).replace("WHERE dataset = %s", "WHERE run_id = %s AND dataset = %s"),
            (vector, run_id, dataset, vector, k),
        )
        return cur.fetchall()


def exact_search(
    connection, vector: list[float], run_id: str, dataset: str, k: int, table: str = "rag_chunks"
):
    with connection.cursor() as cur:
        # Force PostgreSQL's exact scan for a fair baseline instead of allowing HNSW.
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(
            search_sql(table).replace("WHERE dataset = %s", "WHERE run_id = %s AND dataset = %s"),
            (vector, run_id, dataset, vector, k),
        )
        return cur.fetchall()
