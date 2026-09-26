"""Pure retrieval and timing metrics."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Timing:
    embedding_seconds: float = 0.0
    ingest_seconds: float = 0.0
    search_seconds: float = 0.0
    answer_seconds: float = 0.0


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
