"""Local sentence-transformers embedding generation."""

from functools import lru_cache


@lru_cache(maxsize=3)
def _model(model_key: str):
    from .config import MODELS

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install the full extra for local embeddings") from exc
    name, _ = MODELS[model_key]
    return SentenceTransformer(name, trust_remote_code=model_key == "nomic")


def warm_model(model_key: str) -> None:
    """Initialize and cache an embedding model outside per-request timing."""
    from .config import MODELS

    if model_key not in MODELS:
        raise ValueError(f"Unknown embedding model: {model_key}")
    _model(model_key)


def embed_texts(texts: list[str], model_key: str, batch_size: int = 32) -> list[list[float]]:
    from .config import MODELS

    if model_key not in MODELS:
        raise ValueError(f"Unknown embedding model: {model_key}")
    name, dimension = MODELS[model_key]
    del name
    model = _model(model_key)
    prefix = "search_document: " if model_key == "nomic" else ""
    result = model.encode(
        [prefix + text for text in texts], batch_size=batch_size, normalize_embeddings=True
    ).tolist()
    if result and len(result[0]) != dimension:
        raise RuntimeError(f"Embedding dimension mismatch: expected {dimension}")
    return result


def embed_query(query: str, model_key: str) -> list[float]:
    """Embed one query using the model's retrieval instruction."""
    from .config import MODELS

    if model_key not in MODELS:
        raise ValueError(f"Unknown embedding model: {model_key}")
    name, dimension = MODELS[model_key]
    del name
    model = _model(model_key)
    prefix = "search_query: " if model_key == "nomic" else ""
    vector = model.encode([prefix + query], normalize_embeddings=True)[0].tolist()
    if len(vector) != dimension:
        raise RuntimeError(f"Embedding dimension mismatch: expected {dimension}")
    return vector
