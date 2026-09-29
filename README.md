# Neon + pgvector Vidore RAG benchmark

This Python 3.11 suite benchmarks Vidore (`vidore/vidore_v3_industrial`) across local embedding models and one optional Databricks AI Gateway model. Vidore is the only runnable answer-evaluation dataset in this benchmark; other dataset adapters, if present, are not answer-evaluation support.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[full,dev]'
cp .env.example .env
python -m neon_rag_benchmarks.smoke
pytest -q
```

Offline smoke and tests require no Neon, Hugging Face, or Databricks credentials. For a live Vidore run, configure `DATABASE_URL`, `DATABRICKS_BASE_URL`, `DATABRICKS_TOKEN`, and `DATABRICKS_MODEL`, then run the notebook with `SMOKE = False` or call `run_matrix(BenchmarkConfig.from_env(), smoke=False)`.

Vidore native `corpus_id`/`doc_id`, query `answer`/`raw_answers`, qrels, page, bounding-box, and evidence fields are retained where supplied. Generated answers are instructed to cite stable retrieved chunk/document/page identifiers. JSONL records contain the answer, parsed citations, retrieved passage text and provenance, native retrieval recall/MRR, and answer-quality fields.

Correctness is an explicitly labeled deterministic token-F1 against Vidore acceptable answers. Answer relevance is a separate evaluator/judge field and is `null` with `available=false` unless a judge is added. Groundedness uses a deterministic claim-overlap check against valid cited passages; citation completeness and validity are also reported. Missing references, citations, or evidence never produce fabricated scores.

Retrieval relevance (qrel recall/MRR) is separate from answer relevance. `timing_seconds` retains corpus embedding, ingestion, query embedding, HNSW, exact scan, answer, and total phases; `rag_evaluation` measures per-request query embedding + retrieval + answer and excludes dataset/model initialization.

`EXACT_SCAN=false` runs HNSW-only retrieval. `RESULTS_PATH` and `PERSIST_RESULTS` control JSONL persistence. The notebook defaults to Vidore and supports selecting an embedding model. Do not infer answer-evaluation support for any dataset not explicitly documented here.
