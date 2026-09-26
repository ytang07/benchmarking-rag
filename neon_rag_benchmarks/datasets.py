"""Dataset adapters. Only Vidore supplies native qrels; other datasets require queries."""

DATASET_INFO = {
    "vidore": "vidore/vidore_v3_industrial: corpus/test markdown, queries/test query, qrels/test query_id/corpus_id/score; native qrels.",
    "parsebench": "llamaindex/ParseBench parse-bench: text_content split/expected_markdown; no native qrels, supply/generated queries.",
    "govdocs": "BEE-spoke-data/govdocs1-pdf-source index/sample: metadata and PDFs, no extracted text; filter broken_pdf=false and supply/generated queries.",
}


def validate_record(dataset: str, record: dict) -> str:
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {dataset}; choose {tuple(DATASET_INFO)}")
    if dataset == "vidore" and not record.get("text", record.get("markdown")):
        raise ValueError("Vidore record needs markdown text")
    if dataset == "parsebench" and not record.get("text_content"):
        raise ValueError("ParseBench record needs text_content")
    if dataset == "govdocs" and record.get("broken_pdf") is True:
        raise ValueError("GovDocs broken_pdf=true is excluded")
    return str(record.get("id", record.get("doc_id", "")))


def smoke_documents() -> list[dict[str, str]]:
    return [
        {"id": "smoke-1", "text": "Neon stores vectors with pgvector."},
        {"id": "smoke-2", "text": "HNSW accelerates cosine retrieval."},
    ]


def load_optional(name: str, max_rows: int = 1000, max_documents: int = 100):
    """Load a bounded live sample; credentials/network are required by Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the full extra to load Hugging Face datasets") from exc
    if name == "vidore":
        return load_dataset("vidore/vidore_v3_industrial", trust_remote_code=True)
    if name == "parsebench":
        return load_dataset("llamaindex/ParseBench", "parse-bench", split=f"train[:{max_rows}]")
    if name == "govdocs":
        # GovDocs provides source PDFs/metadata rather than extracted text; callers must extract text.
        return load_dataset("BEE-spoke-data/govdocs1-pdf-source", split=f"train[:{max_documents}]")
    raise ValueError(f"Unknown dataset {name}; choose {tuple(DATASET_INFO)}")
