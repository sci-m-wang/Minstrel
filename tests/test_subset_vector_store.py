from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np

from sideprofile.probes import PROBES
from sideprofile.vector_store import corpus_fingerprint, text_sha256, verify_vector_store


SCRIPT = Path(__file__).parents[1] / "scripts" / "subset_vector_store.py"
SPEC = importlib.util.spec_from_file_location("subset_vector_store", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_corpus(path: Path, rows: list[tuple[str, str, str]]) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "CREATE TABLE comments(comment_id TEXT PRIMARY KEY, character_id TEXT, "
            "raw_text TEXT, is_synthetic INTEGER)"
        )
        connection.executemany(
            "INSERT INTO comments VALUES(?, ?, ?, 0)", rows
        )
    connection.close()


def write_store(path: Path, rows: list[tuple[str, str, str]]) -> None:
    connection = sqlite3.connect(path)
    dimension = 2
    with connection:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE document_embeddings(
                comment_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE TABLE query_embeddings(
                probe_id TEXT NOT NULL,
                language TEXT NOT NULL,
                query_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY(probe_id, language)
            );
            """
        )
        metadata = {
            "schema_version": "2",
            "model_key": "embed",
            "model_revision": "revision",
            "similarity_metric": "cosine",
            "embedding_dtype": "float32_le",
            "embedding_dimension": str(dimension),
            "corpus_fingerprint": corpus_fingerprint(rows),
            "document_count": str(len(rows)),
            "query_count": str(len(PROBES) * 2),
            "synthetic_included": "false",
        }
        connection.executemany(
            "INSERT INTO metadata VALUES(?, ?)", sorted(metadata.items())
        )
        for index, (comment_id, character_id, raw_text) in enumerate(rows, 1):
            vector = np.asarray([index, index + 1], dtype="<f4").tobytes()
            connection.execute(
                "INSERT INTO document_embeddings VALUES(?, ?, ?, ?)",
                (comment_id, character_id, text_sha256(raw_text), vector),
            )
        query_vector = np.asarray([1, 0], dtype="<f4").tobytes()
        for probe in PROBES:
            for language in ("en", "zh"):
                query = probe.query(language)
                connection.execute(
                    "INSERT INTO query_embeddings VALUES(?, ?, ?, ?)",
                    (probe.probe_id, language, text_sha256(query), query_vector),
                )
    connection.close()


def test_subset_preserves_retained_vector_bytes_and_verifies(tmp_path: Path) -> None:
    source_rows = [("a", "role", "first"), ("b", "role", "second"), ("c", "role", "third")]
    target_rows = [source_rows[0], source_rows[2]]
    source_corpus = tmp_path / "source.sqlite"
    target_corpus = tmp_path / "target.sqlite"
    source_store = tmp_path / "source-vectors.sqlite"
    output = tmp_path / "target-vectors.sqlite"
    write_corpus(source_corpus, source_rows)
    write_corpus(target_corpus, target_rows)
    write_store(source_store, source_rows)

    result = module.subset_vector_store(
        source_store=source_store,
        source_corpus=source_corpus,
        target_corpus=target_corpus,
        output=output,
        model_key="embed",
        model_revision="revision",
    )

    assert result["target_documents"] == 2
    assert result["removed_documents"] == 1
    assert result["retained_vector_bytes_unchanged"] is True
    assert verify_vector_store(
        vector_store=output,
        corpus_db=target_corpus,
        expected_model_key="embed",
        expected_model_revision="revision",
    )["ok"]


def test_subset_rejects_changed_target_text(tmp_path: Path) -> None:
    source_rows = [("a", "role", "first")]
    source_corpus = tmp_path / "source.sqlite"
    target_corpus = tmp_path / "target.sqlite"
    source_store = tmp_path / "source-vectors.sqlite"
    write_corpus(source_corpus, source_rows)
    write_corpus(target_corpus, [("a", "role", "changed")])
    write_store(source_store, source_rows)

    try:
        module.subset_vector_store(
            source_store=source_store,
            source_corpus=source_corpus,
            target_corpus=target_corpus,
            output=tmp_path / "out.sqlite",
            model_key="embed",
            model_revision="revision",
        )
    except RuntimeError as exc:
        assert "unchanged subset" in str(exc)
    else:
        raise AssertionError("changed target text should be rejected")
