import math
import pytest
from neon_rag_benchmarks.config import BenchmarkConfig, MODELS
from neon_rag_benchmarks.datasets import validate_record
from neon_rag_benchmarks.metrics import exact_cosine_search, mrr_at_k, recall_at_k
from neon_rag_benchmarks.schema import create_schema_sql, validate_dimension
from neon_rag_benchmarks.smoke import run_smoke


def test_config_and_model_dimensions():
    c = BenchmarkConfig.from_env({"HNSW_M": "20"})
    assert c.hnsw_m == 20
    assert MODELS["nomic"][1] == 768


def test_schema_is_dimension_safe():
    validate_dimension("minilm", 384)
    assert "vector(384)" in create_schema_sql(384)[1]
    with pytest.raises(ValueError):
        validate_dimension("minilm", 768)


def test_metrics():
    assert recall_at_k(["a", "b"], {"b"}, 2) == 1
    assert mrr_at_k(["a", "b"], {"b"}, 2) == 0.5
    assert math.isnan(recall_at_k([], set(), 1))
    assert exact_cosine_search([1, 0], [("a", [1, 0]), ("b", [0, 1])], 1) == [("a", 1.0)]


def test_dataset_validation():
    assert validate_record("vidore", {"id": "x", "markdown": "hello"}) == "x"
    with pytest.raises(ValueError):
        validate_record("govdocs", {"id": "x", "broken_pdf": True})


def test_offline_smoke():
    result = run_smoke()
    assert result["offline"] is True and result["recall@1"] == 1.0
