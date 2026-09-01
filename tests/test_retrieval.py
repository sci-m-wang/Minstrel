from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sideprofile.anonymize import anonymize_text, find_identity_leaks
from sideprofile.probes import PROBES_BY_ID
from sideprofile.retrieval import DeterministicSmokeRetriever, VectorRerankRetriever
from sideprofile.schema import CharacterSpec, Comment
from sideprofile.vector_store import text_sha256


def make_spec() -> CharacterSpec:
    return CharacterSpec(
        character_id="x",
        character_name="Alex Example",
        work="Example World",
        anonymous_id="TARGET_X",
        panel="A",
        aliases=["Alex"],
    )


def make_comment(comment_id: str, text: str) -> Comment:
    return Comment(
        comment_id=comment_id,
        character_id="x",
        character_name="Alex Example",
        work="Example World",
        platform="forum",
        author_hash=comment_id,
        raw_text=text,
        language="en",
    )


def test_anonymization_removes_name_alias_and_work() -> None:
    spec = make_spec()
    text = anonymize_text("Alex Example in Example World is also called Alex.", spec)
    assert text == "[TARGET] in [TARGET] is also called [TARGET]."
    assert find_identity_leaks(text, spec) == []


def test_smoke_retriever_never_exposes_identity() -> None:
    spec = make_spec()
    comments = [
        make_comment("c1", "Alex restores the routine when a plan is disrupted."),
        make_comment("c2", "Alex Example writes a careful schedule every morning."),
    ]
    result = DeterministicSmokeRetriever(final_top_k=2).retrieve(
        PROBES_BY_ID["D1-Q1"], comments, spec
    )
    assert len(result) == 2
    assert all(find_identity_leaks(item.text, spec) == [] for item in result)


def make_vector_store(path: Path, comments: list[Comment]) -> None:
    np = pytest.importorskip("numpy")
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE document_embeddings (
            comment_id TEXT PRIMARY KEY,
            character_id TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            vector BLOB NOT NULL
        );
        CREATE TABLE query_embeddings (
            probe_id TEXT NOT NULL,
            language TEXT NOT NULL,
            query_sha256 TEXT NOT NULL,
            vector BLOB NOT NULL,
            PRIMARY KEY(probe_id, language)
        );
        """
    )
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        [
            ("schema_version", "2"),
            ("model_key", "text-embedding-3-small"),
            ("similarity_metric", "cosine"),
            ("embedding_dtype", "float32_le"),
            ("embedding_dimension", "2"),
        ],
    )
    vectors = {
        "c1": np.asarray([1.0, 0.0], dtype="<f4").tobytes(),
        "c2": np.asarray([0.0, 1.0], dtype="<f4").tobytes(),
    }
    connection.executemany(
        """
        INSERT INTO document_embeddings(comment_id, character_id, text_sha256, vector)
        VALUES(?, ?, ?, ?)
        """,
        [
            (
                comment.comment_id,
                comment.character_id,
                text_sha256(comment.raw_text),
                vectors[comment.comment_id],
            )
            for comment in comments
        ],
    )
    probe = PROBES_BY_ID["D1-Q1"]
    connection.execute(
        """
        INSERT INTO query_embeddings(probe_id, language, query_sha256, vector)
        VALUES(?, ?, ?, ?)
        """,
        (
            probe.probe_id,
            "en",
            text_sha256(probe.query("en")),
            np.asarray([1.0, 0.0], dtype="<f4").tobytes(),
        ),
    )
    connection.commit()
    connection.close()


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.values = scores
        self.documents: list[str] = []

    def scores(self, query: str, documents: list[str]) -> list[float]:
        assert query
        self.documents = documents
        return self.values


def test_vector_recall_then_rerank_uses_all_candidates(tmp_path: Path) -> None:
    spec = make_spec()
    comments = [
        make_comment("c1", "Alex restores the routine when a plan is disrupted."),
        make_comment("c2", "Alex enjoys tea while talking with close friends."),
    ]
    store = tmp_path / "vectors.sqlite"
    make_vector_store(store, comments)
    reranker = FakeReranker([0.1, 0.9])
    result = VectorRerankRetriever(
        vector_store=str(store),
        embedding_model_key="text-embedding-3-small",
        reranker=reranker,  # type: ignore[arg-type]
        candidate_top_k=2,
        final_top_k=2,
    ).retrieve(PROBES_BY_ID["D1-Q1"], comments, spec)
    assert [item.comment_id for item in result] == ["c2", "c1"]
    assert len(reranker.documents) == 2
    assert all("Alex" not in document for document in reranker.documents)
    assert all(item.rank_sources == ["exact_vector", "cohere_rerank"] for item in result)


def test_frozen_vector_store_rejects_changed_comment_text(tmp_path: Path) -> None:
    spec = make_spec()
    comments = [
        make_comment("c1", "Alex restores the routine when a plan is disrupted."),
        make_comment("c2", "Alex enjoys tea while talking with close friends."),
    ]
    store = tmp_path / "vectors.sqlite"
    make_vector_store(store, comments)
    changed = [comments[0], make_comment("c2", "Alex suddenly avoids all close friends today.")]
    retriever = VectorRerankRetriever(
        vector_store=str(store),
        embedding_model_key="text-embedding-3-small",
        reranker=FakeReranker([0.5, 0.4]),  # type: ignore[arg-type]
        candidate_top_k=2,
        final_top_k=2,
    )
    with pytest.raises(RuntimeError, match="text hashes differ"):
        retriever.retrieve(PROBES_BY_ID["D1-Q1"], changed, spec)
