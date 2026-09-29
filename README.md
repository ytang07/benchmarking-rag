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

Offline smoke and tests require no Neon, Hugging Face, or Databricks credentials. For a live Vidore retrieval run, configure `DATABASE_URL` and run the notebook with `SMOKE = False` or call `run_matrix(BenchmarkConfig.from_env(), smoke=False)`. Databricks credentials are optional: without all three gateway settings, retrieval and timing results are still produced and answer evaluation is explicitly marked skipped/unavailable. When configured, generated-answer evaluation runs through the Databricks AI Gateway.

Vidore native `corpus_id`/`doc_id`, query `answer`/`raw_answers`, page, bounding-box, evidence, and arbitrary native fields are retained after JSON-safe conversion in `vidore_provenance`, which explicitly covers retained corpus/query rows. Qrel rows are retained as normalized retrieval annotations (`qrels`), not claimed as raw provenance. Canonical `evaluation_scope` and `provenance_scope` values are only `complete`, `bounded`, or `skipped`; detail fields preserve classifications such as `synthetic`, `full_dataset`, or `bounded_sample`. `provenance_scope=complete` only means all source corpus/query rows were retained. `evaluation_scope` can independently be `bounded` when rows were skipped or qrels/docs are incomplete. Generated answers are instructed to cite stable retrieved chunk/document/page identifiers. JSONL records contain the answer, parsed citations, retrieved passage text and provenance, native retrieval recall/MRR, and answer-quality fields.

Correctness is an explicitly labeled deterministic token-F1 against Vidore acceptable answers. Answer relevance is a separate evaluator/judge field and is `null` with `available=false` unless a judge is added. Groundedness uses a deterministic claim-overlap check against valid cited passages; citation completeness and validity are also reported. Missing references, citations, or evidence never produce fabricated scores.

Retrieval relevance (qrel recall/MRR) is separate from answer relevance. `retrieval_passage_citation_completeness` is retrieved-passage coverage; `claim_level_citation_completeness` requires each factual claim to have a valid citation. Groundedness is a lexical/heuristic check, not semantic entailment. `timing_seconds` retains dataset/model initialization, corpus embedding, ingestion, query embedding, HNSW, exact scan, answer, and total phases; `rag_evaluation` measures only per-request query embedding + retrieval + answer and excludes dataset/model initialization.

`EXACT_SCAN=false` runs HNSW-only retrieval. `RESULTS_PATH` and `PERSIST_RESULTS` control JSONL persistence for both `run_matrix` and direct `run_benchmark` calls; an explicit `persist` argument overrides the config. The notebook defaults to Vidore and supports selecting an embedding model. Do not infer answer-evaluation support for any dataset not explicitly documented here.
