"""Dataset adapters. Only Vidore supplies native qrels; other datasets require queries."""

from dataclasses import dataclass, field
import random

DATASET_INFO = {
    "vidore": "vidore/vidore_v3_industrial: corpus/test markdown, queries/test query, qrels/test query_id/corpus_id/score; native qrels.",
    "parsebench": "llamaindex/ParseBench parse-bench: text_content split/expected_markdown; no native qrels, supply/generated queries.",
    "govdocs": "BEE-spoke-data/govdocs1-pdf-source index/sample: metadata and PDFs, no extracted text; filter broken_pdf=false and supply/generated queries.",
}


@dataclass
class PreparedData:
    dataset: str
    documents: list[dict[str, str]] = field(default_factory=list)
    queries: list[dict[str, str]] = field(default_factory=list)
    qrels: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)
    native_qrels: bool = False


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


def _rows(value):
    if isinstance(value, dict) and "train" in value:
        return value["train"]
    return value


def _get(row, *names):
    for name in names:
        if isinstance(row, dict) and row.get(name) is not None:
            return row[name]
    return None


def prepare_records(
    name: str,
    raw,
    seed: int = 7,
    max_rows: int = 1000,
    max_documents: int = 100,
    max_bytes: int = 1_000_000_000,
) -> PreparedData:
    """Normalize a bounded HF result and record honest skips instead of inventing text/queries."""
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {name}; choose {tuple(DATASET_INFO)}")
    result = PreparedData(name)
    if name == "vidore":
        corpus = _rows(raw.get("corpus", raw)) if isinstance(raw, dict) else raw
        queries = _rows(raw.get("queries", [])) if isinstance(raw, dict) else []
        qrel_rows = _rows(raw.get("qrels", [])) if isinstance(raw, dict) else []
        for row in list(corpus)[:max_rows]:
            doc_id = str(_get(row, "corpus_id", "id", "doc_id"))
            text = _get(row, "markdown", "text", "content")
            if text:
                result.documents.append({"id": doc_id, "text": str(text)})
            else:
                result.skipped.append({"id": doc_id, "reason": "missing markdown/text"})
        for row in list(queries)[:max_rows]:
            query_id = str(_get(row, "query_id", "id"))
            text = _get(row, "query", "text")
            if text:
                result.queries.append({"id": query_id, "text": str(text)})
        for row in list(qrel_rows):
            query_id = str(_get(row, "query_id"))
            doc_id = str(_get(row, "corpus_id"))
            result.qrels.setdefault(query_id, {})[doc_id] = float(_get(row, "score") or 0)
        result.native_qrels = bool(result.qrels)
        return result
    rows = list(_rows(raw)) if raw is not None else []
    rng = random.Random(seed)
    rng.shuffle(rows)
    if name == "parsebench":
        for row in rows[:max_rows]:
            text = _get(row, "text_content")
            if text:
                result.documents.append(
                    {
                        "id": str(_get(row, "id", "document_id") or len(result.documents)),
                        "text": str(text),
                    }
                )
            else:
                result.skipped.append({"reason": "missing text_content"})
        result.skipped.append(
            {"reason": "no native queries/qrels; provide generated or user queries"}
        )
        return result
    total_bytes = 0
    for row in rows:
        if len(result.documents) >= max_documents:
            break
        if _get(row, "broken_pdf") is True:
            result.skipped.append({"reason": "broken_pdf=true"})
            continue
        payload = _get(row, "text", "content", "pdf")
        if not isinstance(payload, str):
            result.skipped.append(
                {"id": str(_get(row, "id", "doc_id")), "reason": "PDF has no extracted text"}
            )
            continue
        if total_bytes + len(payload.encode()) > max_bytes:
            result.skipped.append({"reason": "max_bytes reached"})
            break
        total_bytes += len(payload.encode())
        result.documents.append(
            {"id": str(_get(row, "id", "doc_id") or len(result.documents)), "text": payload}
        )
    result.skipped.append({"reason": "no native queries/qrels; provide generated or user queries"})
    return result
