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

The smoke command is deterministic and uses no Neon, Hugging Face, or Databricks credentials. For live work, load `.env`, select dataset/model/sample controls in the notebook, and set all three `DATABRICKS_CHAT_ENDPOINT_*` values. Endpoint names are passed to OpenAI-compatible `chat.completions.create`; no model names are hardcoded. The local models are exactly `nomic-ai/nomic-embed-text-v1.5` (768, `search_document:`/`search_query:`, remote code enabled), `sentence-transformers/all-MiniLM-L6-v2` (384), and `sentence-transformers/all-mpnet-base-v2` (768).

Neon setup creates `vector`, a dimension-specific table, and an HNSW cosine index with configurable `HNSW_M` and `HNSW_EF_CONSTRUCTION`; each search sets `HNSW_EF_SEARCH`. Small runs should compare this with an exact cosine scan. Record embedding, ingestion, search, and answer seconds. Answer metrics are explicitly per chat endpoint and must not be confused with retrieval metrics (Vidore recall/MRR only when qrels exist).

For reproducibility, pin the requirements and keep downloaded Hugging Face artifacts in a controlled cache. Dataset adapters should be extended with the relevant `datasets` loading configuration when doing full ingestion; the notebook's smoke mode remains the recommended first validation.
