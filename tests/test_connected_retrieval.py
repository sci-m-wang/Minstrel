from __future__ import annotations

import json
from pathlib import Path

import pytest

from sideprofile.connected_retrieval import (
    CohereReranker,
    EmbeddingSettings,
    OpenAIEmbeddingClient,
    RerankerSettings,
)


def test_embedding_settings_resolve_openai_compatible_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GPT_API_KEY=secret\n"
        "GPT_EMBEDDING_END_POINT=https://example.invalid\n"
        "GPT_EMBEDDING_MODEL=text-embedding-3-small\n",
        encoding="utf-8",
    )
    for key in ("GPT_API_KEY", "GPT_EMBEDDING_END_POINT", "GPT_EMBEDDING_MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = EmbeddingSettings.from_env(env_file)
    assert settings.endpoint == "https://example.invalid/openai/v1/embeddings"
    assert settings.model == "text-embedding-3-small"


def test_embedding_client_validates_and_traces_one_vector_per_input(tmp_path: Path) -> None:
    requests: list[dict] = []

    def transport(url: str, key: str, payload: dict) -> tuple[int, dict]:
        requests.append({"url": url, "key": key, "payload": payload})
        return 200, {
            "model": "text-embedding-3-small",
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }

    trace = tmp_path / "embedding.trace.jsonl"
    client = OpenAIEmbeddingClient(
        EmbeddingSettings(
            api_key="secret",
            endpoint="https://example.invalid/openai/v1/embeddings",
            model="text-embedding-3-small",
        ),
        trace_path=trace,
        transport=transport,
    )
    assert client.embed(["first", "第二条"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert requests[0]["payload"] == {
        "model": "text-embedding-3-small",
        "input": ["first", "第二条"],
    }
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["model"] == "text-embedding-3-small"
    assert record["input_count"] == 2
    assert "first" not in trace.read_text(encoding="utf-8")


def test_cohere_reranker_sends_all_documents_without_top_n_or_token_limit(
    tmp_path: Path,
) -> None:
    requests: list[dict] = []

    def transport(url: str, key: str, payload: dict) -> tuple[int, dict]:
        requests.append({"url": url, "key": key, "payload": payload})
        return 200, {
            "results": [
                {"index": 2, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ],
            "meta": {"billed_units": {"search_units": 1}},
        }

    trace = tmp_path / "rerank.trace.jsonl"
    reranker = CohereReranker(
        RerankerSettings(
            api_key="secret",
            endpoint="https://example.invalid/providers/cohere/v2/rerank",
            model="Cohere-rerank-v4.0-pro",
        ),
        trace_path=trace,
        transport=transport,
    )
    scores = reranker.scores("query", ["a", "b", "c"])
    assert scores == [0.9, 0.8, 0.1]
    payload = requests[0]["payload"]
    assert payload == {
        "model": "Cohere-rerank-v4.0-pro",
        "query": "query",
        "documents": ["a", "b", "c"],
    }
    assert "top_n" not in payload
    assert "max_tokens_per_doc" not in payload
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["result_indices"] == [2, 0, 1]
    assert '"query": "query"' not in trace.read_text(encoding="utf-8")


def test_cohere_reranker_rejects_incomplete_results() -> None:
    def transport(_: str, __: str, ___: dict) -> tuple[int, dict]:
        return 200, {"results": [{"index": 0, "relevance_score": 0.5}]}

    reranker = CohereReranker(
        RerankerSettings("secret", "https://example.invalid/v2/rerank", "rerank"),
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="one score per submitted document"):
        reranker.scores("q", ["a", "b"])
