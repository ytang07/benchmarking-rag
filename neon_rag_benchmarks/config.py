"""Validated environment configuration; imports no optional integration packages."""

from dataclasses import dataclass
import os

MODELS = {
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", 768),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", 768),
}
DATASETS = ("vidore", "parsebench", "govdocs")


@dataclass(frozen=True)
class BenchmarkConfig:
    database_url: str | None
    gateway_base_url: str | None
    gateway_token: str | None
    chat_endpoints: tuple[str | None, str | None, str | None]
    seed: int = 7
    parsebench_max_rows: int = 1000
    govdocs_max_documents: int = 100
    govdocs_max_bytes: int = 1_000_000_000
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    hnsw_ef_search: int = 40
    chunk_size: int = 800
    chunk_overlap: int = 80
    top_k: int = 10
    exact_scan: bool = True
    qrel_min_score: float = 1.0
    results_path: str = "results/benchmark.jsonl"
    persist_results: bool = True
    query_source: str | None = None
    govdocs_pdf_extractor: str = "pypdf"
    govdocs_config: str = "index"
    govdocs_split: str = "train"
    govdocs_max_pages: int = 20
    govdocs_max_text_chars: int = 200_000
    govdocs_max_document_bytes: int = 100_000_000
    experiment_id: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "BenchmarkConfig":
        if environ is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        e = os.environ if environ is None else environ

        def integer(name: str, default: int, minimum: int = 1) -> int:
            try:
                value = int(e.get(name, str(default)))
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            return value

        try:
            threshold = float(e.get("QREL_MIN_SCORE", "1"))
        except ValueError as exc:
            raise ValueError("QREL_MIN_SCORE must be numeric") from exc
        if threshold < 0:
            raise ValueError("QREL_MIN_SCORE must be >= 0")
        chunk_size = integer("CHUNK_SIZE", 800)
        chunk_overlap = integer("CHUNK_OVERLAP", 80, 0)
        if chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        exact_raw = e.get("EXACT_SCAN", "true").strip().lower()
        if exact_raw not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("EXACT_SCAN must be true/false")
        persist_raw = e.get("PERSIST_RESULTS", "true").strip().lower()
        if persist_raw not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("PERSIST_RESULTS must be true/false")
        return cls(
            e.get("DATABASE_URL"),
            e.get("DATABRICKS_BASE_URL"),
            e.get("DATABRICKS_TOKEN"),
            tuple(e.get(f"DATABRICKS_CHAT_ENDPOINT_{i}") for i in range(1, 4)),
            integer("BENCHMARK_SEED", 7, 0),
            integer("PARSEBENCH_MAX_ROWS", 1000),
            integer("GOVDOCS_MAX_DOCUMENTS", 100),
            integer("GOVDOCS_MAX_BYTES", 1),
            integer("HNSW_M", 16),
            integer("HNSW_EF_CONSTRUCTION", 128),
            integer("HNSW_EF_SEARCH", 1),
            chunk_size,
            chunk_overlap,
            integer("TOP_K", 10),
            exact_raw in {"true", "1", "yes"},
            threshold,
            e.get("RESULTS_PATH", "results/benchmark.jsonl"),
            persist_raw in {"true", "1", "yes"},
            e.get("QUERY_SOURCE") or None,
            e.get("GOVDOCS_PDF_EXTRACTOR", "pypdf"),
            e.get("GOVDOCS_CONFIG", "index"),
            e.get("GOVDOCS_SPLIT", "train"),
            integer("GOVDOCS_MAX_PAGES", 20),
            integer("GOVDOCS_MAX_TEXT_CHARS", 200_000),
            integer("GOVDOCS_MAX_DOCUMENT_BYTES", 100_000_000),
            e.get("EXPERIMENT_ID") or None,
        )

    def require_database(self) -> str:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for Neon operations")
        return self.database_url

    def require_gateway(self) -> tuple[str, str, tuple[str, str, str]]:
        if not self.gateway_base_url or not self.gateway_token or not all(self.chat_endpoints):
            raise RuntimeError(
                "DATABRICKS_BASE_URL, DATABRICKS_TOKEN, and all three chat endpoints are required"
            )
        return self.gateway_base_url, self.gateway_token, self.chat_endpoints  # type: ignore
