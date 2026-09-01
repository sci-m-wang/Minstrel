#!/usr/bin/env python3
"""Derive an exact vector-store subset after a deletion-only corpus filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sideprofile.vector_store import (
    _corpus_rows,
    corpus_fingerprint,
    file_sha256,
    text_sha256,
    verify_vector_store,
)


def _vector_hashes(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    connection = sqlite3.connect(
        f"{resolved.as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        return {
            str(comment_id): hashlib.sha256(bytes(vector)).hexdigest()
            for comment_id, vector in connection.execute(
                "SELECT comment_id, vector FROM document_embeddings ORDER BY comment_id"
            )
        }
    finally:
        connection.close()


def subset_vector_store(
    *,
    source_store: Path,
    source_corpus: Path,
    target_corpus: Path,
    output: Path,
    model_key: str,
    model_revision: str,
) -> dict[str, Any]:
    source_store = source_store.resolve()
    source_corpus = source_corpus.resolve()
    target_corpus = target_corpus.resolve()
    output = output.resolve()
    if output in {source_store, source_corpus, target_corpus}:
        raise ValueError("output must differ from every input")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite vector store: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale build path exists: {building}")

    source_verification = verify_vector_store(
        vector_store=source_store,
        corpus_db=source_corpus,
        expected_model_key=model_key,
        expected_model_revision=model_revision,
    )
    if not source_verification["ok"]:
        raise RuntimeError(
            "source vector store is not valid: "
            + "; ".join(source_verification["failures"])
        )

    source_rows = _corpus_rows(source_corpus)
    target_rows = _corpus_rows(target_corpus)
    source_by_id = {
        comment_id: (character_id, raw_text)
        for comment_id, character_id, raw_text in source_rows
    }
    for comment_id, character_id, raw_text in target_rows:
        if source_by_id.get(comment_id) != (character_id, raw_text):
            raise RuntimeError(
                f"target corpus is not an unchanged subset at comment {comment_id}"
            )

    before_vectors = _vector_hashes(source_store)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"{source_store.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    target_connection = sqlite3.connect(building)
    try:
        source_connection.backup(target_connection)
        with target_connection:
            target_connection.execute(
                "CREATE TEMP TABLE keep_comment_ids(comment_id TEXT PRIMARY KEY)"
            )
            target_connection.executemany(
                "INSERT INTO keep_comment_ids(comment_id) VALUES(?)",
                ((comment_id,) for comment_id, _, _ in target_rows),
            )
            target_connection.execute(
                "DELETE FROM document_embeddings "
                "WHERE comment_id NOT IN (SELECT comment_id FROM keep_comment_ids)"
            )
            target_connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'corpus_fingerprint'",
                (corpus_fingerprint(target_rows),),
            )
            target_connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'document_count'",
                (str(len(target_rows)),),
            )
            target_connection.execute("DROP TABLE keep_comment_ids")
        integrity = str(target_connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"derived vector store integrity failure: {integrity}")
        target_connection.execute("VACUUM")
    except BaseException:
        target_connection.close()
        source_connection.close()
        building.unlink(missing_ok=True)
        raise
    else:
        target_connection.close()
        source_connection.close()
    building.replace(output)

    verification = verify_vector_store(
        vector_store=output,
        corpus_db=target_corpus,
        expected_model_key=model_key,
        expected_model_revision=model_revision,
    )
    if not verification["ok"]:
        raise RuntimeError(
            "derived vector store failed verification: "
            + "; ".join(verification["failures"])
        )
    after_vectors = _vector_hashes(output)
    changed = [
        comment_id
        for comment_id, vector_hash in after_vectors.items()
        if before_vectors.get(comment_id) != vector_hash
    ]
    if changed:
        raise RuntimeError("retained vector bytes changed: " + ", ".join(changed[:10]))

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation": "verified_deletion_only_vector_subset",
        "model_key": model_key,
        "model_revision": model_revision,
        "source_store": str(source_store),
        "source_store_sha256": file_sha256(source_store),
        "source_corpus": str(source_corpus),
        "source_documents": len(source_rows),
        "target_corpus": str(target_corpus),
        "target_documents": len(target_rows),
        "removed_documents": len(source_rows) - len(target_rows),
        "retained_vector_bytes_unchanged": True,
        "query_embeddings_unchanged": True,
        "output": str(output),
        "output_sha256": file_sha256(output),
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-store", required=True)
    parser.add_argument("--source-corpus", required=True)
    parser.add_argument("--target-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    payload = subset_vector_store(
        source_store=Path(args.source_store),
        source_corpus=Path(args.source_corpus),
        target_corpus=Path(args.target_corpus),
        output=Path(args.output),
        model_key=args.model_key,
        model_revision=args.model_revision,
    )
    audit = Path(args.audit).resolve()
    if audit.exists():
        raise FileExistsError(f"refusing to overwrite audit: {audit}")
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
