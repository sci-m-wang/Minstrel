#!/usr/bin/env python3
"""Build a new research corpus using the frozen deterministic exclusion rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUESTION_WHERE = (
    "is_synthetic=0 AND platform='stackexchange' "
    "AND source_url NOT LIKE '%#answer-%' AND source_url NOT LIKE '%#comment%'"
)
ANSWER_WHERE = (
    "is_synthetic=0 AND platform='stackexchange' AND source_url LIKE '%#answer-%'"
)
LENGTH_WHERE = "is_synthetic=0 AND LENGTH(raw_text)>=1000"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    non_synthetic = connection.execute(
        "SELECT COUNT(*) FROM comments WHERE is_synthetic=0"
    ).fetchone()[0]
    synthetic = connection.execute(
        "SELECT COUNT(*) FROM comments WHERE is_synthetic=1"
    ).fetchone()[0]
    characters = [
        {
            "character_id": row[0],
            "comments": row[1],
            "platforms": row[2],
            "authors": row[3],
            "max_characters": row[4],
        }
        for row in connection.execute(
            "SELECT character_id, COUNT(*), COUNT(DISTINCT platform), "
            "COUNT(DISTINCT author_hash), MAX(LENGTH(raw_text)) "
            "FROM comments WHERE is_synthetic=0 GROUP BY character_id ORDER BY character_id"
        )
    ]
    return {
        "non_synthetic_rows": non_synthetic,
        "synthetic_rows": synthetic,
        "characters": characters,
    }


def matching_ids(connection: sqlite3.Connection, where: str) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            f"SELECT comment_id FROM comments WHERE {where} ORDER BY comment_id"
        )
    ]


def filter_corpus(*, source: Path, output: Path, audit_path: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    if source == output:
        raise ValueError("source and output must be different paths")
    if not source.is_file():
        raise FileNotFoundError(f"missing source corpus: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output corpus: {output}")
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {audit_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(
        f"{source.as_uri()}?mode=ro&immutable=1", uri=True
    )
    output_connection = sqlite3.connect(output)
    try:
        source_connection.backup(output_connection)
        before = row_counts(output_connection)
        removed_ids: dict[str, list[str]] = {}
        for reason, where in (
            ("stackexchange_question", QUESTION_WHERE),
            ("stackexchange_answer", ANSWER_WHERE),
            ("length_ge_1000", LENGTH_WHERE),
        ):
            ids = matching_ids(output_connection, where)
            removed_ids[reason] = ids
            with output_connection:
                output_connection.execute(f"DELETE FROM main.comments WHERE {where}")
        after = row_counts(output_connection)
        output_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        output_connection.execute("VACUUM")
        integrity = str(output_connection.execute("PRAGMA integrity_check").fetchone()[0])
        remaining_question = len(matching_ids(output_connection, QUESTION_WHERE))
        remaining_answer = len(matching_ids(output_connection, ANSWER_WHERE))
        remaining_long = len(matching_ids(output_connection, LENGTH_WHERE))
        if integrity != "ok":
            raise RuntimeError(f"filtered corpus integrity check failed: {integrity}")
        if remaining_question or remaining_answer or remaining_long:
            raise RuntimeError(
                "filtered corpus retains excluded rows: "
                f"question={remaining_question}, answer={remaining_answer}, length={remaining_long}"
            )
    finally:
        output_connection.close()
        source_connection.close()

    removed = {reason: len(ids) for reason, ids in removed_ids.items()}
    expected_after = before["non_synthetic_rows"] - sum(removed.values())
    if after["non_synthetic_rows"] != expected_after:
        raise RuntimeError(
            f"row accounting mismatch: expected {expected_after}, got {after['non_synthetic_rows']}"
        )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "stackexchange": "retain only source URLs containing #comment",
            "maximum_retained_characters": 999,
            "length_unit": "Unicode characters in raw_text",
            "synthetic_rows": "preserved and excluded from research counts",
        },
        "source": str(source),
        "output": str(output),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "before": before,
        "removed": removed,
        "removed_comment_ids": removed_ids,
        "after": after,
        "integrity_check": integrity,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/corpus/comments.sqlite")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default="data/audits/comment_filter_audit.json")
    args = parser.parse_args()
    payload = filter_corpus(
        source=Path(args.source), output=Path(args.output), audit_path=Path(args.audit)
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "source_sha256": payload["source_sha256"],
                "output_sha256": payload["output_sha256"],
                "before": payload["before"]["non_synthetic_rows"],
                "removed": payload["removed"],
                "after": payload["after"]["non_synthetic_rows"],
                "audit": str(Path(args.audit).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
