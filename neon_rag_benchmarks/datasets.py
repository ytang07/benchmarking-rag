"""Dataset adapters. Only Vidore supplies native qrels; other datasets require queries."""

from dataclasses import dataclass, field
import csv
import json
import random
from io import BytesIO
from itertools import islice
from pathlib import Path

DATASET_INFO = {
    "vidore": "vidore/vidore_v3_industrial: corpus/test markdown, queries/test query, qrels/test query_id/corpus_id/score; native qrels.",
    "parsebench": "llamaindex/ParseBench parse-bench: text_content split/expected_markdown; no native qrels, supply/generated queries.",
    "govdocs": "BEE-spoke-data/govdocs1-pdf-source index/sample: metadata and PDFs, no extracted text; filter broken_pdf=false and supply/generated queries.",
}


def _canonical_scope(value: str, provenance: bool = False) -> str:
    if value in {"complete", "bounded", "skipped"}:
        return value
    if value in {"full_dataset"}:
        return "complete"
    if value in {"bounded_sample"}:
        return "bounded"
    if value == "synthetic":
        return "skipped" if provenance else "bounded"
    return "skipped"


@dataclass
class PreparedData:
    dataset: str
    documents: list[dict[str, str]] = field(default_factory=list)
    queries: list[dict[str, str]] = field(default_factory=list)
    qrels: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)
    native_qrels: bool = False
    evaluation_scope: str = "skipped"
    evaluation_scope_detail: str = "incomplete"
    metadata: dict = field(default_factory=dict)
    native_provenance: dict = field(default_factory=dict)
    provenance_scope: str = "skipped"
    provenance_scope_detail: str = "not_available"

    def __post_init__(self):
        if self.evaluation_scope not in {"complete", "bounded", "skipped"}:
            if self.evaluation_scope_detail == "incomplete":
                self.evaluation_scope_detail = self.evaluation_scope
            self.evaluation_scope = _canonical_scope(self.evaluation_scope)
        if self.provenance_scope not in {"complete", "bounded", "skipped"}:
            if self.provenance_scope_detail == "not_available":
                self.provenance_scope_detail = self.provenance_scope
            self.provenance_scope = _canonical_scope(self.provenance_scope, provenance=True)
        if self.skipped:
            if self.evaluation_scope == "complete":
                self.evaluation_scope = "bounded"
                self.evaluation_scope_detail = "filtered_rows"
            if self.provenance_scope == "complete":
                self.provenance_scope = "bounded"
                self.provenance_scope_detail = "filtered_rows"


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
                    "vidore/vidore_v3_industrial", component, split="test", streaming=True
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
    object_qrels = {}
    if source.suffix.lower() == ".csv":
        rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    elif source.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
        parsed = (
            json.loads(source.read_text(encoding="utf-8")) if source.suffix == ".json" else None
        )
        if isinstance(parsed, dict) and "queries" in parsed:
            rows = parsed["queries"]
            object_qrels = parsed.get("qrels", {})
        elif isinstance(parsed, list):
            rows, object_qrels = parsed, {}
        elif parsed is not None:
            raise ValueError("JSON query source must be an array or an object with queries/qrels")
        else:
            rows = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            object_qrels = {}
    else:
        raise ValueError("QUERY_SOURCE must be .jsonl, .ndjson, or .csv")
    queries, qrels = [], {}
    for query_id, values in object_qrels.items():
        if isinstance(values, dict):
            qrels[str(query_id)] = {str(doc_id): float(score) for doc_id, score in values.items()}
        else:
            qrels[str(query_id)] = {str(doc_id): 1.0 for doc_id in values}
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
                nested = value[key]
                return _pdf_payload(nested) if isinstance(nested, dict) else nested
    return value


def _pdf_size(value) -> int | None:
    value = _pdf_payload(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str) and Path(value).is_file():
        return Path(value).stat().st_size
    return None


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
    if isinstance(payload, (bytes, bytearray, memoryview)) and len(payload) > max_bytes:
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
    try:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            reader = PdfReader(BytesIO(bytes(payload)))
        elif isinstance(payload, str) and Path(payload).is_file():
            reader = PdfReader(payload)
        else:
            raise ValueError("GovDocs PDF field must be bytes, path, or a {bytes,path} object")
    except Exception as exc:
        raise ValueError(f"malformed PDF bytes ({type(exc).__name__})") from exc
    pieces = []
    total_chars = 0
    for page_number, page in enumerate(reader.pages[:max_pages], 1):
        try:
            piece = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(
                f"PDF text extraction failed on page {page_number} ({type(exc).__name__})"
            ) from exc
        remaining = max_text_chars - total_chars
        if remaining <= 0:
            break
        pieces.append(piece[:remaining])
        total_chars += len(pieces[-1])
    return "\n".join(pieces).strip()


def _rows(value):
    if isinstance(value, dict) and "train" in value:
        return value["train"]
    return value


def _get(row, *names):
    for name in names:
        if isinstance(row, dict) and row.get(name) is not None:
            return row[name]
    return None


def _vidore_metadata(row: dict) -> dict:
    """Keep native page, box, and evidence fields available for answer provenance."""
    metadata = {}
    for key in ("page", "page_number", "bbox", "bounding_box", "evidence", "metadata"):
        value = _get(row, key)
        if value is not None:
            metadata[key] = value
    return metadata


def json_safe(value):
    """Convert native dataset objects (including bytes/features) into JSON-safe values."""
    try:
        return json.loads(json.dumps(value, default=lambda item: repr(item), allow_nan=False))
    except (TypeError, ValueError):
        return repr(value)


def _reservoir_sample(rows, limit: int, seed: int):
    """Bound memory for streaming non-Vidore rows while retaining deterministic sampling."""
    rng = random.Random(seed)
    sample = []
    for index, row in enumerate(rows):
        if index < limit:
            sample.append(row)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < limit:
                sample[replacement] = row
    return sample


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
    pdf_max_document_bytes: int = 100_000_000,
    pdf_max_text_bytes: int = 1_000_000,
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
        # Streaming avoids materializing the ~5k-page corpus. The +1 sentinel proves truncation.
        corpus_rows = list(islice(iter(corpus), max_rows + 1))
        query_rows = list(islice(iter(queries), max_rows + 1))
        corpus_truncated = len(corpus_rows) > max_rows
        query_truncated = len(query_rows) > max_rows
        corpus_rows = corpus_rows[:max_rows]
        query_rows = query_rows[:max_rows]
        result.native_provenance = {
            "corpus": [json_safe(row) for row in corpus_rows],
            "queries": [json_safe(row) for row in query_rows],
        }
        for row in corpus_rows[:max_rows]:
            doc_id = str(_get(row, "corpus_id", "id", "doc_id"))
            text = _get(row, "markdown", "text", "content")
            if text:
                result.documents.append(
                    json_safe(
                        {
                            "id": doc_id,
                            "doc_id": _get(row, "doc_id"),
                            "corpus_id": _get(row, "corpus_id"),
                            "text": str(text),
                            "native_provenance": json_safe(row),
                            **_vidore_metadata(row),
                        }
                    )
                )
            else:
                result.skipped.append({"id": doc_id, "reason": "missing markdown/text"})
        retained_query_ids = set()
        for row in query_rows[:max_rows]:
            query_id = str(_get(row, "query_id", "id"))
            text = _get(row, "query", "text")
            if text:
                query = {"id": query_id, "text": str(text), "native_provenance": json_safe(row)}
                for key in ("answer", "raw_answers", "answers", "evidence", "page", "page_number"):
                    value = _get(row, key)
                    if value is not None:
                        query[key] = json_safe(value)
                result.queries.append(query)
                retained_query_ids.add(query_id)
            else:
                result.skipped.append({"id": query_id, "reason": "missing query text"})
        qrels_seen = 0
        qrels_retained = 0
        for row in qrel_rows:
            qrels_seen += 1
            query_id = str(_get(row, "query_id"))
            doc_id = str(_get(row, "corpus_id"))
            if query_id not in retained_query_ids:
                continue
            result.qrels.setdefault(query_id, {})[doc_id] = float(_get(row, "score") or 0)
            qrels_retained += 1
        result.native_qrels = bool(result.qrels)
        result.metadata.update(
            {
                "query_count_retained": len(result.queries),
                "qrels_rows_seen": qrels_seen,
                "qrels_rows_retained": qrels_retained,
                "qrels_coverage": qrels_retained / qrels_seen if qrels_seen else 0.0,
                "qrels_truncated": False,
                "corpus_count_total": len(corpus_rows),
                "query_count_total": len(query_rows),
                "corpus_truncated": corpus_truncated,
                "query_truncated": query_truncated,
            }
        )
        result.evaluation_scope_detail = (
            "bounded_sample"
            if result.metadata["corpus_truncated"] or result.metadata["query_truncated"]
            else "full_dataset"
        )
        indexed_ids = {document["id"] for document in result.documents}
        qrel_doc_ids = {doc_id for qrels in result.qrels.values() for doc_id in qrels}
        if (
            set(result.qrels) != retained_query_ids
            or not qrel_doc_ids.issubset(indexed_ids)
            or result.skipped
        ):
            result.evaluation_scope_detail = "bounded_sample"
        result.evaluation_scope = (
            "complete" if result.evaluation_scope_detail == "full_dataset" else "bounded"
        )
        if corpus_truncated or query_truncated:
            result.provenance_scope = "bounded"
            result.provenance_scope_detail = "bounded_sample"
        elif result.skipped:
            result.provenance_scope = "bounded"
            result.provenance_scope_detail = "filtered_rows"
        else:
            result.provenance_scope = "complete"
            result.provenance_scope_detail = "full_dataset"
        result.metadata["evaluation_scope"] = result.evaluation_scope_detail
        result.metadata["evaluation_scope_detail"] = result.evaluation_scope_detail
        result.metadata["provenance_scope"] = result.provenance_scope
        result.metadata["provenance_scope_detail"] = result.provenance_scope_detail
    else:
        if name == "parsebench":
            rows = _reservoir_sample(_rows(raw) if raw is not None else (), max_rows, seed)
            for row in rows:
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
            result.evaluation_scope = "bounded_sample"
        else:
            # GovDocs stays streaming: limits are applied before requesting/parsing the next PDF.
            rows = _rows(raw) if raw is not None else ()
            extracted_text_bytes = 0
            raw_pdf_bytes = 0
            for row in rows:
                if len(result.documents) >= max_documents:
                    break
                broken = _get(row, "broken_pdf")
                if _is_broken(broken):
                    result.skipped.append({"reason": f"broken_pdf excluded (value={broken!r})"})
                    continue
                payload = _get(row, "text", "content")
                if not payload:
                    raw_pdf = _get(row, "pdf", "pdf_bytes")
                    raw_size = _pdf_size(raw_pdf)
                    if raw_size is None:
                        result.skipped.append({"reason": "PDF raw byte size unavailable; excluded"})
                        continue
                    if raw_size > pdf_max_document_bytes:
                        result.skipped.append(
                            {
                                "reason": f"per-document raw PDF limit exceeded ({raw_size} > {pdf_max_document_bytes})"
                            }
                        )
                        continue
                    if raw_pdf_bytes + raw_size > max_bytes:
                        result.skipped.append(
                            {
                                "reason": f"cumulative raw PDF byte budget exhausted ({raw_pdf_bytes} + {raw_size} > {max_bytes})"
                            }
                        )
                        break
                    raw_pdf_bytes += raw_size
                    try:
                        payload = _extract_pdf(
                            raw_pdf,
                            pdf_extractor,
                            pdf_max_document_bytes,
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
                text_size = len(payload.encode())
                if text_size > pdf_max_text_bytes:
                    result.skipped.append(
                        {
                            "reason": f"extracted text exceeds max bytes ({text_size} > {pdf_max_text_bytes})"
                        }
                    )
                    continue
                extracted_text_bytes += text_size
                doc_id = str(_get(row, "id", "doc_id") or len(result.documents))
                result.documents.append({"id": doc_id, "text": payload})
            result.metadata.update(
                {
                    "raw_pdf_bytes_consumed": raw_pdf_bytes,
                    "raw_pdf_byte_budget": max_bytes,
                    "extracted_text_bytes": extracted_text_bytes,
                    "extracted_text_byte_limit": pdf_max_text_bytes,
                    "raw_pdf_budget_exhausted": any(
                        "cumulative raw PDF byte budget" in item.get("reason", "")
                        for item in result.skipped
                    ),
                }
            )
            if not query_source:
                for document in result.documents[: min(10, len(result.documents))]:
                    result.queries.append(
                        {"id": f"generated-{document['id']}", "text": document["text"][:160]}
                    )
                result.skipped.append(
                    {"reason": "no native qrels; generated synthetic prefix probes"}
                )
            result.evaluation_scope = "bounded_sample"
    if query_source:
        result.queries, source_qrels = load_query_source(query_source)
        result.qrels.update(source_qrels)
        result.skipped.append({"reason": f"queries loaded from {query_source}"})
    return result
