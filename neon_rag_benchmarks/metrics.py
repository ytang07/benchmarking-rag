"""Pure retrieval, Vidore answer, citation, and timing metrics."""

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


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def parse_citations(answer: str) -> list[str]:
    """Extract stable ``[document#chunk-N]`` (optionally page-qualified) citations."""
    citations = []
    for match in _CITATION_RE.finditer(answer or ""):
        for value in re.split(r"\s*[,;]\s*", match.group(1)):
            value = value.strip()
            if value and value not in citations:
                citations.append(value)
    return citations


def validate_citations(citations: list[str], passages: list[dict]) -> dict:
    """Validate citations against retrieved passage identifiers and return cited passages."""
    by_id = {str(p.get("citation_id", p.get("chunk_id", p.get("id", "")))): p for p in passages}
    valid = [citation for citation in citations if citation in by_id]
    invalid = [citation for citation in citations if citation not in by_id]
    return {
        "citations": citations,
        "valid": valid,
        "invalid": invalid,
        "validity": (len(valid) / len(citations)) if citations else None,
        "completeness": (len(valid) / len(by_id)) if by_id else None,
        "cited_passages": [by_id[citation] for citation in valid],
    }


def _answer_list(acceptable_answers) -> list[str]:
    if acceptable_answers is None:
        return []
    if isinstance(acceptable_answers, str):
        return [acceptable_answers]
    return [str(answer) for answer in acceptable_answers if str(answer).strip()]


def _token_f1(prediction: str, reference: str) -> float:
    predicted = normalized_text(prediction).split()
    expected = normalized_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum(min(predicted.count(token), expected.count(token)) for token in set(predicted))
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(
    generated: str,
    status: str,
    reference: str | None = None,
    acceptable_answers=None,
    question: str | None = None,
    citations: list[str] | None = None,
    passages: list[dict] | None = None,
) -> dict:
    result = {
        "status": status,
        "answer_chars": len(generated),
        "answer_words": len(generated.split()),
        "quality_metric_support": "vidore_deterministic_token_f1; no LLM judge",
    }
    answers = _answer_list(acceptable_answers if acceptable_answers is not None else reference)
    if answers:
        result["correctness"] = {
            "value": max(_token_f1(generated, expected) for expected in answers),
            "available": True,
            "method": "deterministic_token_f1_against_vidore_acceptable_answers",
        }
        result["normalized_exact_match"] = any(
            normalized_text(generated) == normalized_text(expected) for expected in answers
        )
    else:
        result["correctness"] = {
            "value": None,
            "available": False,
            "reason": "Vidore acceptable answer/raw_answers unavailable",
        }
        result["normalized_exact_match"] = None
        result["quality_metric_note"] = "unavailable_without_vidore_acceptable_answers"
    result["answer_relevance"] = {
        "value": None,
        "available": False,
        "method": "judge_not_configured",
        "reason": "answer relevance requires an optional evaluator/judge",
    }
    citation_result = validate_citations(citations or [], passages or [])
    result["citation_completeness"] = {
        "value": citation_result["completeness"],
        "available": bool(passages),
        "method": "retrieved_passage_coverage",
    }
    result["citation_validity"] = {
        "value": citation_result["validity"],
        "available": bool(citations),
        "method": "stable_retrieved_chunk_identifier_match",
    }
    if citations and citation_result["cited_passages"]:
        text = " ".join(str(p.get("text", "")) for p in citation_result["cited_passages"])
        claims = [part.strip() for part in re.split(r"[.!?]+", generated) if part.strip()]
        supported = [claim for claim in claims if _token_f1(claim, text) >= 0.25]
        result["groundedness"] = {
            "value": len(supported) / len(claims) if claims else None,
            "available": bool(claims),
            "method": "deterministic_claim_token_overlap_with_cited_text",
        }
    else:
        result["groundedness"] = {
            "value": None,
            "available": False,
            "method": "deterministic_claim_token_overlap_with_cited_text",
            "reason": "no valid cited retrieved text",
        }
    result["citation_details"] = citation_result
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
