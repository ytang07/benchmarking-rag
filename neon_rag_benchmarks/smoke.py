"""Credential-free deterministic smoke benchmark."""

from .datasets import smoke_documents
from .metrics import cosine_similarity, mrr_at_k, recall_at_k


def run_smoke(dataset: str = "vidore") -> dict[str, float | bool | str]:
    if dataset != "vidore":
        raise ValueError(f"Vidore-only smoke does not support dataset={dataset!r}")
    docs = smoke_documents()
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    q = [0.8, 0.2]
    ranked = [
        doc["id"]
        for _, doc in sorted(zip([cosine_similarity(q, v) for v in vectors], docs), reverse=True)
    ]
    return {
        "offline": True,
        "top_doc": ranked[0],
        "recall@1": recall_at_k(ranked, {"smoke-1"}, 1),
        "mrr@2": mrr_at_k(ranked, {"smoke-1"}, 2),
    }


if __name__ == "__main__":
    print(run_smoke())
