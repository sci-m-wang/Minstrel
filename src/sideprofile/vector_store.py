from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .connected_retrieval import OpenAIEmbeddingClient
from .probes import PROBES
from .schema import Comment, Probe


VECTOR_STORE_SCHEMA_VERSION = "2"
SIMILARITY_METRIC = "cosine"
EMBEDDING_DTYPE = "float32_le"
EMBEDDING_TRANSPORT_BATCH_SIZE = 256
MODEL_DIMENSIONS = {"text-embedding-3-small": 1536}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_rows(corpus_db: str | Path) -> list[tuple[str, str, str]]:
    path = Path(corpus_db).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing corpus database: {path}")
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        rows = connection.execute(
            """
            SELECT comment_id, character_id, raw_text
            FROM comments
            WHERE is_synthetic = 0
            ORDER BY comment_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def corpus_fingerprint(rows: list[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for comment_id, character_id, raw_text in rows:
        payload = json.dumps(
            [comment_id, character_id, raw_text],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in connection.execute(
            "SELECT key, value FROM metadata ORDER BY key"
        ).fetchall()
    }


def build_vector_store(
    *,
    corpus_db: str | Path,
    output_path: str | Path,
    client: OpenAIEmbeddingClient,
    model_key: str,
) -> dict[str, Any]:
    """Build an immutable exact-cosine store through the connected embedding service."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("vector-store construction requires numpy") from exc

    source = Path(corpus_db).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite vector store: {output}; build to a new path and verify it"
        )
    rows = _corpus_rows(source)
    if not rows:
        raise RuntimeError("cannot build a vector store from an empty research corpus")
    query_specs: list[tuple[Probe, str, str]] = []
    for probe in PROBES:
        for language in ("en", "zh"):
            query_specs.append((probe, language, probe.query(language)))

    output.parent.mkdir(parents=True, exist_ok=True)
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale build path exists: {building}")
    connection = sqlite3.connect(building)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE document_embeddings (
                comment_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE INDEX idx_document_embeddings_character
                ON document_embeddings(character_id, comment_id);
            CREATE TABLE query_embeddings (
                probe_id TEXT NOT NULL,
                language TEXT NOT NULL,
                query_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY(probe_id, language)
            );
            """
        )
        dimension: int | None = None
        for start in range(0, len(rows), EMBEDDING_TRANSPORT_BATCH_SIZE):
            batch = rows[start : start + EMBEDDING_TRANSPORT_BATCH_SIZE]
            vectors = client.embed([raw_text for _, _, raw_text in batch])
            encoded_rows = []
            for (comment_id, character_id, raw_text), vector in zip(
                batch, vectors, strict=True
            ):
                array = np.asarray(vector, dtype="<f4")
                norm = float(np.linalg.norm(array))
                if array.ndim != 1 or not norm:
                    raise RuntimeError("embedding model returned an invalid document vector")
                if dimension is None:
                    dimension = int(array.shape[0])
                elif array.shape[0] != dimension:
                    raise RuntimeError("document embedding dimensions differ")
                array = np.asarray(array / norm, dtype="<f4")
                encoded_rows.append(
                    (
                        comment_id,
                        character_id,
                        text_sha256(raw_text),
                        array.tobytes(),
                    )
                )
            with connection:
                connection.executemany(
                    """
                    INSERT INTO document_embeddings(
                        comment_id, character_id, text_sha256, vector
                    ) VALUES(?, ?, ?, ?)
                    """,
                    encoded_rows,
                )

        query_vectors = client.embed([query for _, _, query in query_specs])
        encoded_queries = []
        for (probe, language, query), vector in zip(
            query_specs, query_vectors, strict=True
        ):
            array = np.asarray(vector, dtype="<f4")
            norm = float(np.linalg.norm(array))
            if array.ndim != 1 or not norm or array.shape[0] != dimension:
                raise RuntimeError("query embedding dimension differs from documents")
            array = np.asarray(array / norm, dtype="<f4")
            encoded_queries.append(
                (
                    probe.probe_id,
                    language,
                    text_sha256(query),
                    array.tobytes(),
                )
            )
        with connection:
            connection.executemany(
                """
                INSERT INTO query_embeddings(
                    probe_id, language, query_sha256, vector
                ) VALUES(?, ?, ?, ?)
                """,
                encoded_queries,
            )

        if dimension is None:
            raise RuntimeError("embedding service returned no document vectors")
        expected_dimension = MODEL_DIMENSIONS.get(model_key)
        if expected_dimension is not None and dimension != expected_dimension:
            raise RuntimeError(
                f"embedding dimension {dimension} differs from {model_key} "
                f"default {expected_dimension}"
            )
        if client.returned_models != {model_key}:
            raise RuntimeError(
                "embedding endpoint returned a different model: "
                f"{sorted(client.returned_models)!r} != {[model_key]!r}"
            )
        metadata = {
            "schema_version": VECTOR_STORE_SCHEMA_VERSION,
            "embedding_provider": client.settings.provider,
            "model_key": model_key,
            "model_revision": "provider_managed",
            "similarity_metric": SIMILARITY_METRIC,
            "embedding_dtype": EMBEDDING_DTYPE,
            "embedding_dimension": str(dimension),
            "corpus_fingerprint": corpus_fingerprint(rows),
            "document_count": str(len(rows)),
            "query_count": str(len(query_specs)),
            "synthetic_included": "false",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "api_usage": json.dumps(client.usage, sort_keys=True),
        }
        with connection:
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES(?, ?)",
                sorted(metadata.items()),
            )
        connection.execute("VACUUM")
    except BaseException:
        connection.close()
        building.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    building.replace(output)
    return {
        "status": "ok",
        "path": str(output),
        "embedding_provider": client.settings.provider,
        "model_key": model_key,
        "model_revision": "provider_managed",
        "documents": len(rows),
        "queries": len(query_specs),
        "dimension": dimension,
        "api_usage": client.usage,
        "sha256": file_sha256(output),
    }


def verify_vector_store(
    *,
    vector_store: str | Path,
    corpus_db: str | Path,
    expected_model_key: str | None = None,
    expected_model_revision: str | None = None,
) -> dict[str, Any]:
    store_path = Path(vector_store).resolve()
    failures: list[str] = []
    if not store_path.is_file():
        return {"ok": False, "path": str(store_path), "failures": ["missing vector store"]}
    rows = _corpus_rows(corpus_db)
    connection = sqlite3.connect(
        f"{store_path.as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        metadata = _metadata(connection)
        expected_metadata = {
            "schema_version": VECTOR_STORE_SCHEMA_VERSION,
            "similarity_metric": SIMILARITY_METRIC,
            "embedding_dtype": EMBEDDING_DTYPE,
            "corpus_fingerprint": corpus_fingerprint(rows),
            "document_count": str(len(rows)),
            "query_count": str(len(PROBES) * 2),
            "synthetic_included": "false",
        }
        if expected_model_key is not None:
            expected_metadata["model_key"] = expected_model_key
        if expected_model_revision is not None:
            expected_metadata["model_revision"] = expected_model_revision
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                failures.append(f"metadata {key}={metadata.get(key)!r}, expected {expected!r}")
        try:
            dimension = int(metadata.get("embedding_dimension", "0"))
        except ValueError:
            dimension = 0
        if dimension <= 0:
            failures.append("invalid embedding_dimension")
        expected_dimension = MODEL_DIMENSIONS.get(str(expected_model_key or metadata.get("model_key", "")))
        if expected_dimension is not None and dimension != expected_dimension:
            failures.append(
                f"embedding_dimension={dimension}, expected {expected_dimension}"
            )
        observed_documents = connection.execute(
            """
            SELECT comment_id, character_id, text_sha256, length(vector)
            FROM document_embeddings ORDER BY comment_id
            """
        ).fetchall()
        if len(observed_documents) != len(rows):
            failures.append(
                f"document rows={len(observed_documents)}, expected {len(rows)}"
            )
        for observed, expected in zip(observed_documents, rows, strict=False):
            comment_id, character_id, raw_text = expected
            if tuple(observed[:3]) != (
                comment_id,
                character_id,
                text_sha256(raw_text),
            ):
                failures.append(f"document metadata mismatch at {comment_id}")
                break
            if int(observed[3]) != dimension * 4:
                failures.append(f"vector byte length mismatch at {comment_id}")
                break
        observed_queries = {
            (str(row[0]), str(row[1])): (str(row[2]), int(row[3]))
            for row in connection.execute(
                """
                SELECT probe_id, language, query_sha256, length(vector)
                FROM query_embeddings ORDER BY probe_id, language
                """
            ).fetchall()
        }
        for probe in PROBES:
            for language in ("en", "zh"):
                expected = (text_sha256(probe.query(language)), dimension * 4)
                if observed_queries.get((probe.probe_id, language)) != expected:
                    failures.append(
                        f"query embedding mismatch at {probe.probe_id}/{language}"
                    )
    except sqlite3.DatabaseError as exc:
        failures.append(f"invalid vector-store database: {exc}")
        metadata = {}
    finally:
        connection.close()
    return {
        "ok": not failures,
        "path": str(store_path),
        "model_key": metadata.get("model_key", ""),
        "model_revision": metadata.get("model_revision", ""),
        "documents": metadata.get("document_count", ""),
        "queries": metadata.get("query_count", ""),
        "dimension": metadata.get("embedding_dimension", ""),
        "failures": failures,
    }


class ExactVectorStore:
    """Read-only exact cosine search over the frozen connected-service embeddings."""

    def __init__(self, path: str | Path, *, expected_model_key: str) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("vector retrieval requires numpy") from exc
        self.np = np
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise RuntimeError(f"missing frozen vector store: {self.path}")
        self.connection = sqlite3.connect(
            f"{self.path.as_uri()}?mode=ro&immutable=1", uri=True
        )
        self.metadata = _metadata(self.connection)
        if self.metadata.get("schema_version") != VECTOR_STORE_SCHEMA_VERSION:
            raise RuntimeError("unsupported vector-store schema")
        if self.metadata.get("model_key") != expected_model_key:
            raise RuntimeError(
                "vector-store model mismatch: "
                f"{self.metadata.get('model_key')!r} != {expected_model_key!r}"
            )
        if self.metadata.get("similarity_metric") != SIMILARITY_METRIC:
            raise RuntimeError("vector store is not configured for exact cosine retrieval")
        if self.metadata.get("embedding_dtype") != EMBEDDING_DTYPE:
            raise RuntimeError("unsupported vector-store embedding dtype")
        self.dimension = int(self.metadata["embedding_dimension"])
        self._documents: dict[str, tuple[list[str], list[str], Any]] = {}

    def _character_matrix(self, character_id: str) -> tuple[list[str], list[str], Any]:
        cached = self._documents.get(character_id)
        if cached is not None:
            return cached
        rows = self.connection.execute(
            """
            SELECT comment_id, text_sha256, vector
            FROM document_embeddings
            WHERE character_id = ?
            ORDER BY comment_id
            """,
            (character_id,),
        ).fetchall()
        ids = [str(row[0]) for row in rows]
        hashes = [str(row[1]) for row in rows]
        vectors = [
            self.np.frombuffer(row[2], dtype="<f4", count=self.dimension) for row in rows
        ]
        matrix = (
            self.np.vstack(vectors)
            if vectors
            else self.np.empty((0, self.dimension), dtype="<f4")
        )
        cached = (ids, hashes, matrix)
        self._documents[character_id] = cached
        return cached

    def scores(
        self,
        *,
        probe: Probe,
        language: str,
        comments: list[Comment],
    ) -> list[float]:
        normalized_language = "zh" if language.startswith("zh") else "en"
        row = self.connection.execute(
            """
            SELECT query_sha256, vector FROM query_embeddings
            WHERE probe_id = ? AND language = ?
            """,
            (probe.probe_id, normalized_language),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"missing frozen query embedding: {probe.probe_id}/{normalized_language}"
            )
        query = probe.query(normalized_language)
        if str(row[0]) != text_sha256(query):
            raise RuntimeError(
                f"stale frozen query embedding: {probe.probe_id}/{normalized_language}"
            )
        query_vector = self.np.frombuffer(
            row[1], dtype="<f4", count=self.dimension
        )
        if not comments:
            return []
        character_ids = {comment.character_id for comment in comments}
        if len(character_ids) != 1:
            raise RuntimeError("dense retrieval requires comments from exactly one character")
        character_id = next(iter(character_ids))
        ids, hashes, matrix = self._character_matrix(character_id)
        expected_ids = [comment.comment_id for comment in comments]
        if ids != expected_ids:
            raise RuntimeError(f"vector store is stale for character {character_id}")
        expected_hashes = [text_sha256(comment.raw_text) for comment in comments]
        if hashes != expected_hashes:
            raise RuntimeError(f"vector-store text hashes differ for character {character_id}")
        return [float(value) for value in matrix @ query_vector]
