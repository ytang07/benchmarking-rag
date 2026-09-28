"""Matrix definitions and credential-free planning."""

from itertools import product
from .config import DATASETS, MODELS


def matrix() -> list[dict[str, str]]:
    return [
        {"dataset": dataset, "embedding_model": model, "chat_model_env": "DATABRICKS_MODEL"}
        for dataset, model in product(DATASETS, MODELS)
    ]
