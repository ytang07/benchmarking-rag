"""Dataset adapters. Only Vidore supplies native qrels; other datasets require queries."""

from dataclasses import dataclass, field
import csv
import json
import random
from io import BytesIO
from pathlib import Path

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
    if dataset == "govdocs" and _is_broken(record.get("broken_pdf")):
        raise ValueError("GovDocs broken_pdf must be an explicit false value")
    return str(record.get("id", record.get("doc_id", "")))


def smoke_documents() -> list[dict[str, str]]:
    return [
        {"id": "smoke-1", "text": "Neon stores vectors with pgvector."},
        {"id": "smoke-2", "text": "HNSW accelerates cosine retrieval."},
    ]


def load_optional(
    name: str,
    max_rows: int = 1000,
    max_documents: int = 100,
    govdocs_config: str = "index",
    govdocs_split: str = "train",
):
    """Load a bounded live sample; credentials/network are required by Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the full extra to load Hugging Face datasets") from exc
    if name == "vidore":
        components = {}
        for component in ("corpus", "queries", "qrels"):
            try:
                components[component] = load_dataset(
                    "vidore/vidore_v3_industrial", component, split="test"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Vidore must expose configs corpus/queries/qrels with test splits; "
                    f"could not load config={component}, split=test: {exc}"
                ) from exc
        return components
    if name == "parsebench":
        return load_dataset("llamaindex/ParseBench", "parse-bench", split=f"train[:{max_rows}]")
    if name == "govdocs":
        if govdocs_config not in {"index", "sample"}:
            raise ValueError("GOVDOCS_CONFIG must be one of the documented configs: index, sample")
        if not govdocs_split:
            raise ValueError("GOVDOCS_SPLIT must be non-empty")
        try:
            return load_dataset(
                "BEE-spoke-data/govdocs1-pdf-source",
                govdocs_config,
                split=f"{govdocs_split}[:{max_documents}]",
            )
        except Exception as exc:
            raise RuntimeError(
                "GovDocs loader expects config index or sample and a valid configured split; "
                f"config={govdocs_config}, split={govdocs_split}: {exc}"
            ) from exc
    raise ValueError(f"Unknown dataset {name}; choose {tuple(DATASET_INFO)}")


def load_query_source(path: str) -> tuple[list[dict[str, str]], dict[str, dict[str, float]]]:
    """Read query_id/text and optional reference_answer/qrels from JSONL or CSV."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"QUERY_SOURCE does not exist: {path}")
    if source.suffix.lower() == ".csv":
        rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    elif source.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("QUERY_SOURCE must be .jsonl, .ndjson, or .csv")
    queries, qrels = [], {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not query_id or not text:
            raise ValueError("every query source row needs non-empty query_id and text")
        query = {"id": query_id, "text": text}
        if row.get("reference_answer"):
            query["reference_answer"] = str(row["reference_answer"])
        queries.append(query)
        raw_qrels = row.get("qrels", row.get("relevant_doc_ids", ""))
        if raw_qrels:
            ids = (
                raw_qrels
                if isinstance(raw_qrels, list)
                else str(raw_qrels).replace(";", ",").split(",")
            )
            qrels[query_id] = {str(doc_id).strip(): 1.0 for doc_id in ids if str(doc_id).strip()}
    return queries, qrels


def _is_broken(value) -> bool:
    """Return true for broken/unknown values; only explicit false is accepted."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
    return True


def _pdf_payload(value):
    """Normalize HF PDF feature values (bytes/path or {bytes,path,...}) to one payload."""
    if isinstance(value, dict):
        for key in ("bytes", "data", "content", "path", "filename"):
            if value.get(key) is not None:
                return value[key]
    return value


def _extract_pdf(
    payload,
    extractor: str = "pypdf",
    max_bytes: int = 1_000_000_000,
    max_pages: int = 20,
    max_text_chars: int = 200_000,
) -> str:
    if extractor != "pypdf":
        raise RuntimeError(f"unsupported GOVDOCS_PDF_EXTRACTOR={extractor}; supported: pypdf")
    payload = _pdf_payload(payload)
    if isinstance(payload, bytes) and len(payload) > max_bytes:
        raise ValueError(f"raw PDF exceeds GOVDOCS_MAX_BYTES ({len(payload)} > {max_bytes})")
    if (
        isinstance(payload, str)
        and Path(payload).is_file()
        and Path(payload).stat().st_size > max_bytes
    ):
        raise ValueError(
            f"raw PDF exceeds GOVDOCS_MAX_BYTES ({Path(payload).stat().st_size} > {max_bytes})"
        )
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("GovDocs PDF extraction requires optional dependency pypdf") from exc
    if isinstance(payload, bytes):
        reader = PdfReader(BytesIO(payload))
    elif isinstance(payload, str) and Path(payload).is_file():
        reader = PdfReader(payload)
    else:
        raise ValueError("GovDocs PDF field must be bytes, path, or a {bytes,path} object")
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages]).strip()
    return text[:max_text_chars]


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
    query_source: str | None = None,
    pdf_extractor: str = "pypdf",
    pdf_max_pages: int = 20,
    pdf_max_text_chars: int = 200_000,
) -> PreparedData:
    """Normalize a bounded HF result and record honest skips instead of inventing text/queries."""
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {name}; choose {tuple(DATASET_INFO)}")
    result = PreparedData(name)
    if name == "vidore":
        if not isinstance(raw, dict) or not all(
            key in raw for key in ("corpus", "queries", "qrels")
        ):
            raise RuntimeError("Vidore loader returned no corpus/queries/qrels test components")
        corpus = _rows(raw["corpus"])
        queries = _rows(raw["queries"])
        qrel_rows = _rows(raw["qrels"])
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
        for row in list(qrel_rows)[:max_rows]:
            query_id = str(_get(row, "query_id"))
            doc_id = str(_get(row, "corpus_id"))
            result.qrels.setdefault(query_id, {})[doc_id] = float(_get(row, "score") or 0)
        result.native_qrels = bool(result.qrels)
    else:
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
            if not query_source:
                # ParseBench has useful text but no native queries: deterministic first-sentence queries are explicit fallback.
                for document in result.documents[: min(10, len(result.documents))]:
                    result.queries.append(
                        {"id": f"generated-{document['id']}", "text": document["text"][:160]}
                    )
                result.skipped.append(
                    {"reason": "no native qrels; generated synthetic prefix probes"}
                )
        else:
            total_bytes = 0
            for row in rows:
                if len(result.documents) >= max_documents:
                    break
                broken = _get(row, "broken_pdf")
                if _is_broken(broken):
                    result.skipped.append({"reason": f"broken_pdf excluded (value={broken!r})"})
                    continue
                payload = _get(row, "text", "content")
                if not payload:
                    try:
                        payload = _extract_pdf(
                            _get(row, "pdf", "pdf_bytes"),
                            pdf_extractor,
                            max_bytes,
                            pdf_max_pages,
                            pdf_max_text_chars,
                        )
                    except (RuntimeError, ValueError) as exc:
                        result.skipped.append(
                            {"id": str(_get(row, "id", "doc_id")), "reason": str(exc)}
                        )
                        continue
                if not isinstance(payload, str) or not payload.strip():
                    result.skipped.append({"reason": "PDF extraction produced no text"})
                    continue
                if total_bytes + len(payload.encode()) > max_bytes:
                    result.skipped.append({"reason": "max_bytes reached"})
                    break
                total_bytes += len(payload.encode())
                doc_id = str(_get(row, "id", "doc_id") or len(result.documents))
                result.documents.append({"id": doc_id, "text": payload})
            if not query_source:
                for document in result.documents[: min(10, len(result.documents))]:
                    result.queries.append(
                        {"id": f"generated-{document['id']}", "text": document["text"][:160]}
                    )
                result.skipped.append(
                    {"reason": "no native qrels; generated synthetic prefix probes"}
                )
    if query_source:
        result.queries, source_qrels = load_query_source(query_source)
        result.qrels.update(source_qrels)
        result.skipped.append({"reason": f"queries loaded from {query_source}"})
    return result
