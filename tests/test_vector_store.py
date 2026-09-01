from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from sideprofile.vector_store import build_vector_store, verify_vector_store


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(provider="GPT")
        self.returned_models = {"text-embedding-3-small"}
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            [float(index + 1), float(len(text) + 1)] + [0.0] * 1534
            for index, text in enumerate(texts)
        ]

    @property
    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "returned_models": sorted(self.returned_models),
        }


def write_corpus(path: Path) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "CREATE TABLE comments(comment_id TEXT PRIMARY KEY, character_id TEXT, "
            "raw_text TEXT, is_synthetic INTEGER)"
        )
        connection.executemany(
            "INSERT INTO comments VALUES(?, ?, ?, 0)",
            [("c1", "role", "first comment"), ("c2", "role", "第二条评论")],
        )
    connection.close()


def test_connected_vector_store_builds_and_verifies(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.sqlite"
    output = tmp_path / "vectors.sqlite"
    write_corpus(corpus)
    client = FakeEmbeddingClient()

    result = build_vector_store(
        corpus_db=corpus,
        output_path=output,
        client=client,  # type: ignore[arg-type]
        model_key="text-embedding-3-small",
    )

    assert result["documents"] == 2
    assert result["queries"] == 48
    assert result["dimension"] == 1536
    assert client.calls == 2
    verification = verify_vector_store(
        vector_store=output,
        corpus_db=corpus,
        expected_model_key="text-embedding-3-small",
        expected_model_revision="provider_managed",
    )
    assert verification["ok"], verification["failures"]
