from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sideprofile.anonymize import anonymize_text, find_identity_leaks
from sideprofile.probes import PROBES_BY_ID
from sideprofile.retrieval import HybridRetriever, bm25_scores
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


def test_bm25_prefers_relevant_document() -> None:
    scores = bm25_scores("routine disrupted plan", ["routine plan changed", "likes warm tea"])
    assert scores[0] > scores[1]


def test_retriever_never_exposes_identity() -> None:
    spec = make_spec()
    comments = [
        make_comment("c1", "Alex restores the routine when a plan is disrupted."),
        make_comment("c2", "Alex Example writes a careful schedule every morning."),
    ]
    result = HybridRetriever(mode="bm25", final_top_k=2).retrieve(
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
            ("schema_version", "1"),
            ("model_key", "qwen3-embedding-0.6b"),
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


def test_hybrid_retriever_reads_frozen_exact_vector_store(tmp_path: Path) -> None:
    spec = make_spec()
    comments = [
        make_comment("c1", "Alex restores the routine when a plan is disrupted."),
        make_comment("c2", "Alex enjoys tea while talking with close friends."),
    ]
    store = tmp_path / "vectors.sqlite"
    make_vector_store(store, comments)
    result = HybridRetriever(
        mode="hybrid",
        vector_store=str(store),
        embedding_model_key="qwen3-embedding-0.6b",
        bm25_top_k=0,
        dense_top_k=2,
        final_top_k=2,
    ).retrieve(PROBES_BY_ID["D1-Q1"], comments, spec)
    assert [item.comment_id for item in result] == ["c1", "c2"]
    assert all("dense" in item.rank_sources for item in result)


def test_frozen_vector_store_rejects_changed_comment_text(tmp_path: Path) -> None:
    spec = make_spec()
    comments = [
        make_comment("c1", "Alex restores the routine when a plan is disrupted."),
        make_comment("c2", "Alex enjoys tea while talking with close friends."),
    ]
    store = tmp_path / "vectors.sqlite"
    make_vector_store(store, comments)
    changed = [comments[0], make_comment("c2", "Alex suddenly avoids all close friends today.")]
    retriever = HybridRetriever(
        mode="hybrid",
        vector_store=str(store),
        embedding_model_key="qwen3-embedding-0.6b",
    )
    with pytest.raises(RuntimeError, match="text hashes differ"):
        retriever.retrieve(PROBES_BY_ID["D1-Q1"], changed, spec)
