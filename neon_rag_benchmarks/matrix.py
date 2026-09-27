"""Matrix definitions and credential-free planning."""

from itertools import product
from .config import DATASETS, MODELS


def matrix() -> list[dict[str, str]]:
    endpoints = (
        "DATABRICKS_CHAT_ENDPOINT_1",
        "DATABRICKS_CHAT_ENDPOINT_2",
        "DATABRICKS_CHAT_ENDPOINT_3",
    )
    return [
        {"dataset": dataset, "embedding_model": model, "chat_endpoint_env": endpoint}
        for dataset, model, endpoint in product(DATASETS, MODELS, endpoints)
    ]
