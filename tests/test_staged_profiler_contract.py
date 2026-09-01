from __future__ import annotations

import pytest

from sideprofile.staged import _profiler_spec, _retrieval_service_spec


def test_profiler_spec_requires_connected_provider_identity() -> None:
    assert _profiler_spec(
        {
            "profiler": {
                "provider": "gpt",
                "model": "gpt-5.6-sol",
                "execution_location": "connected_preparation",
            }
        }
    ) == ("GPT", "gpt-5.6-sol", "connected_preparation")


@pytest.mark.parametrize(
    "profiler",
    [
        {},
        {"provider": "GPT", "model": "gpt-5.6-sol"},
        {"provider": "GPT", "execution_location": "connected_preparation"},
        {"model": "gpt-5.6-sol", "execution_location": "connected_preparation"},
    ],
)
def test_profiler_spec_rejects_incomplete_registry(profiler: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="requires provider, model, and execution_location"):
        _profiler_spec({"profiler": profiler})


def test_retrieval_services_are_connected_preparation_only() -> None:
    assert _retrieval_service_spec(
        {
            "retrieval_preparation": {
                "embedding": {
                    "provider": "gpt",
                    "model": "text-embedding-3-small",
                    "execution_location": "connected_preparation",
                },
                "reranker": {
                    "provider": "cohere",
                    "model": "Cohere-rerank-v4.0-pro",
                    "execution_location": "connected_preparation",
                },
            }
        }
    ) == {
        "embedding_provider": "GPT",
        "embedding_model": "text-embedding-3-small",
        "reranker_provider": "COHERE",
        "reranker_model": "Cohere-rerank-v4.0-pro",
    }
