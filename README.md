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

Offline smoke and tests require no Neon, Hugging Face, or Databricks credentials. For a live Vidore run, configure `DATABASE_URL`, `DATABRICKS_BASE_URL`, `DATABRICKS_TOKEN`, and `DATABRICKS_MODEL`, then run the notebook with `SMOKE = False` or call `run_matrix(BenchmarkConfig.from_env(), smoke=False)`. When the gateway is configured, the LLM judge is enabled by default and uses `system.ai.qwen35-122b-a10b`; set `LLM_JUDGE_ENABLED=false` to opt out or set `LLM_JUDGE_MODEL` to override it. The judge uses the same Databricks gateway credentials and is never called by offline smoke/tests.

Vidore native `corpus_id`/`doc_id`, query `answer`/`raw_answers`, page, bounding-box, evidence, and arbitrary native fields are retained after JSON-safe conversion in `vidore_provenance`, which explicitly covers retained corpus/query rows. Qrel rows are retained as normalized retrieval annotations (`qrels`), not claimed as raw provenance. Canonical `evaluation_scope` and `provenance_scope` values are only `complete`, `bounded`, or `skipped`; detail fields preserve classifications such as `synthetic`, `full_dataset`, or `bounded_sample`. `provenance_scope=complete` only means all source corpus/query rows were retained. `evaluation_scope` can independently be `bounded` when rows were skipped or qrels/docs are incomplete. Generated answers are instructed to cite stable retrieved chunk/document/page identifiers. JSONL records contain the answer, parsed citations, retrieved passage text and provenance, native retrieval recall/MRR, and answer-quality fields.

Correctness is an explicitly labeled deterministic token-F1 against Vidore acceptable answers, with normalized exact match preserved as a separate metric. Answer relevance is a separate LLM-as-a-judge field. A successful judge record contains a validated score in `[0, 1]`, label, rationale, and model; unavailable or failed judges contain `value=null`, `available=false`, and explicit status/reason metadata. Missing references, citations, evidence, credentials, or valid judge output never produce fabricated scores. Groundedness uses a deterministic claim-overlap check against valid cited passages; citation completeness and validity are also reported.

Retrieval relevance (qrel recall/MRR) is separate from answer relevance. `retrieval_passage_citation_completeness` is retrieved-passage coverage; `claim_level_citation_completeness` requires each factual claim to have a valid citation. Groundedness is a lexical/heuristic check, not semantic entailment. `timing_seconds` retains dataset/model initialization, corpus embedding, ingestion, query embedding, HNSW, exact scan, answer, and total phases; `rag_evaluation` measures only per-request query embedding + retrieval + answer and excludes dataset/model initialization.

`EXACT_SCAN=false` runs HNSW-only retrieval. `RESULTS_PATH` and `PERSIST_RESULTS` control JSONL persistence. The notebook defaults to Vidore and supports selecting an embedding model. Do not infer answer-evaluation support for any dataset not explicitly documented here.
