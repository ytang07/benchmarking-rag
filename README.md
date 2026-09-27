# Neon + pgvector HNSW RAG benchmarks

This Python 3.11 suite measures a reproducible 3 × 3 × 3 matrix: Vidore, ParseBench, and GovDocs × the three local sentence-transformers models × three configurable Databricks AI Gateway chat endpoints. The matrix is described in [`notebooks/benchmark_matrix.ipynb`](notebooks/benchmark_matrix.ipynb).

Vidore (`vidore/vidore_v3_industrial`) is the small, native-evaluation dataset: its corpus/test markdown, queries/test query, and qrels/test query_id/corpus_id/score are used for retrieval metrics. ParseBench (`llamaindex/ParseBench`, parse-bench config, `text_content`/`expected_markdown`) is medium (~169,011 rows) and has no native qrels, so provide or generate queries. GovDocs (`BEE-spoke-data/govdocs1-pdf-source`) is large: index/sample configs contain metadata and PDFs but no extracted text; the adapter filters `broken_pdf=false` and uses document/byte limits. ParseBench and GovDocs are intentionally opt-in; the suite never claims their full data is locally available.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[full,dev]'
cp .env.example .env   # fill DATABASE_URL and gateway values for live runs
python -m neon_rag_benchmarks.smoke
pytest -q
```

The smoke command is deterministic and uses no Neon, Hugging Face, or Databricks credentials. `BenchmarkConfig.from_env()` loads `.env` through python-dotenv. For live work, set all three `DATABRICKS_CHAT_ENDPOINT_*` values and run the notebook with `SMOKE = False`, or call `run_matrix(BenchmarkConfig.from_env(), smoke=False)`. This performs bounded dataset loading, validation, chunking, local embedding, Neon insertion, HNSW and exact retrieval, qrels evaluation, answer generation, phase timing, and JSONL result persistence; each dataset/model/endpoint combination is a separate result record. Endpoint names are passed to OpenAI-compatible `chat.completions.create`; no model names are hardcoded. The local models are exactly `nomic-ai/nomic-embed-text-v1.5` (768, `search_document:`/`search_query:`, remote code enabled), `sentence-transformers/all-MiniLM-L6-v2` (384), and `sentence-transformers/all-mpnet-base-v2` (768).

Neon setup creates `vector` before registering pgvector with psycopg, then creates a dimension-specific table and HNSW cosine index. Each model uses an isolated table and clears its dataset before ingestion. Every live query compares HNSW with an exact pgvector scan. Record embedding, ingestion, separate HNSW/exact-scan, and answer seconds. `QREL_MIN_SCORE` is an inclusive threshold (`score >= threshold`); answer metrics are reference-based normalized exact match only when a query source supplies `reference_answer`, plus length/status. No unsupported truth/quality score is claimed.

ParseBench and GovDocs accept `QUERY_SOURCE` as JSONL/CSV with `query_id,text`, optional `relevant_doc_ids`/`qrels`, and optional `reference_answer`. Without it, bounded deterministic queries are generated from extracted text and are explicitly marked as having no native qrels. GovDocs filters `broken_pdf=true` and extracts bounded PDF bytes with optional `pypdf`; missing extractor, invalid PDFs, and missing text become per-document skip records with reasons.

For reproducibility, pin the requirements and keep downloaded Hugging Face artifacts in a controlled cache. Dataset adapters should be extended with the relevant `datasets` loading configuration when doing full ingestion; the notebook's smoke mode remains the recommended first validation.
