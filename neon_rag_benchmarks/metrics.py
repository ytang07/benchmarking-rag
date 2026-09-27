"""Pure retrieval and timing metrics."""

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Timing:
    embedding_seconds: float = 0.0
    ingest_seconds: float = 0.0
    search_seconds: float = 0.0
    answer_seconds: float = 0.0


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()


def answer_metrics(generated: str, status: str, reference: str | None = None) -> dict:
    result = {
        "status": status,
        "answer_chars": len(generated),
        "answer_words": len(generated.split()),
        "quality_metric_support": "reference_match_only; no truth judgment",
    }
    if reference is not None:
        result["normalized_exact_match"] = normalized_text(generated) == normalized_text(reference)
    else:
        result["normalized_exact_match"] = None
        result["quality_metric_note"] = "unsupported_without_reference_answer"
    return result


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vectors must have equal non-zero dimensions")
    denom = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0


def exact_cosine_search(
    query: list[float], documents: list[tuple[str, list[float]]], k: int
) -> list[tuple[str, float]]:
    """Exact cosine baseline for small runs, without a database or approximate index."""
    if k < 1:
        raise ValueError("k must be >= 1")
    scored = [(doc_id, cosine_similarity(query, vector)) for doc_id, vector in documents]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
