from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from .schema import CharacterSpec, Comment


SCHEMA_VERSION = "1"


def hash_author(platform: str, author_id: str, salt: str) -> str:
    payload = f"{salt}\0{platform}\0{author_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class CommentCorpus:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CommentCorpus":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS characters (
                character_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                work TEXT NOT NULL,
                platform TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                author_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                language TEXT NOT NULL,
                source_url TEXT NOT NULL,
                collection_method TEXT NOT NULL,
                license_note TEXT NOT NULL,
                is_synthetic INTEGER NOT NULL,
                collected_at TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                UNIQUE(character_id, text_hash),
                FOREIGN KEY(character_id) REFERENCES characters(character_id)
            );
            CREATE INDEX IF NOT EXISTS idx_comments_character
                ON comments(character_id);
            CREATE INDEX IF NOT EXISTS idx_comments_character_platform
                ON comments(character_id, platform);
            CREATE INDEX IF NOT EXISTS idx_comments_character_author
                ON comments(character_id, author_hash);
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self.connection.commit()

    def add_character(self, spec: CharacterSpec) -> None:
        self.initialize()
        self.connection.execute(
            "INSERT OR REPLACE INTO characters(character_id, payload) VALUES(?, ?)",
            (spec.character_id, spec.model_dump_json()),
        )
        self.connection.commit()

    def add_characters(self, specs: Iterable[CharacterSpec]) -> None:
        self.initialize()
        with self.connection:
            for spec in specs:
                self.connection.execute(
                    "INSERT OR REPLACE INTO characters(character_id, payload) VALUES(?, ?)",
                    (spec.character_id, spec.model_dump_json()),
                )

    def get_character(self, character_id: str) -> CharacterSpec:
        row = self.connection.execute(
            "SELECT payload FROM characters WHERE character_id = ?", (character_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"unknown character_id: {character_id}")
        return CharacterSpec.model_validate_json(row["payload"])

    def list_characters(self) -> list[CharacterSpec]:
        rows = self.connection.execute(
            "SELECT payload FROM characters ORDER BY character_id"
        ).fetchall()
        return [CharacterSpec.model_validate_json(row["payload"]) for row in rows]

    def add_comment(self, comment: Comment) -> bool:
        self.initialize()
        try:
            self.get_character(comment.character_id)
        except KeyError as exc:
            raise ValueError(
                f"comment references missing character {comment.character_id}; import catalog first"
            ) from exc
        values = comment.model_dump()
        values["is_synthetic"] = int(comment.is_synthetic)
        values["text_hash"] = _text_hash(comment.raw_text)
        columns = list(values)
        placeholders = ", ".join("?" for _ in columns)
        try:
            with self.connection:
                self.connection.execute(
                    f"INSERT INTO comments ({', '.join(columns)}) VALUES ({placeholders})",
                    tuple(values[column] for column in columns),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def add_comments(self, comments: Iterable[Comment]) -> tuple[int, int]:
        inserted = duplicate = 0
        for comment in comments:
            if self.add_comment(comment):
                inserted += 1
            else:
                duplicate += 1
        return inserted, duplicate

    def comments_for(self, character_id: str, *, include_synthetic: bool = False) -> list[Comment]:
        query = "SELECT * FROM comments WHERE character_id = ?"
        params: list[object] = [character_id]
        if not include_synthetic:
            query += " AND is_synthetic = 0"
        query += " ORDER BY comment_id"
        rows = self.connection.execute(query, params).fetchall()
        result = []
        fields = set(Comment.model_fields)
        for row in rows:
            payload = {key: row[key] for key in fields}
            payload["is_synthetic"] = bool(payload["is_synthetic"])
            result.append(Comment.model_validate(payload))
        return result

    def stats(self, *, include_synthetic: bool = False) -> dict:
        where = "" if include_synthetic else "WHERE is_synthetic = 0"
        total = self.connection.execute(f"SELECT COUNT(*) AS n FROM comments {where}").fetchone()["n"]
        synthetic = self.connection.execute(
            "SELECT COUNT(*) AS n FROM comments WHERE is_synthetic = 1"
        ).fetchone()["n"]
        by_character = []
        rows = self.connection.execute(
            f"""
            SELECT character_id, COUNT(*) AS comments,
                   COUNT(DISTINCT platform) AS platforms,
                   COUNT(DISTINCT author_hash) AS authors
            FROM comments {where}
            GROUP BY character_id ORDER BY character_id
            """
        ).fetchall()
        by_character.extend(dict(row) for row in rows)
        return {
            "schema_version": SCHEMA_VERSION,
            "total_comments": total,
            "synthetic_comments_excluded": synthetic if not include_synthetic else 0,
            "characters": by_character,
        }

    def validate_targets(
        self,
        *,
        min_comments: int,
        min_platforms: int,
        min_authors: int,
        include_synthetic: bool = False,
        character_ids: list[str] | None = None,
    ) -> dict:
        stats = self.stats(include_synthetic=include_synthetic)
        observed = {row["character_id"]: row for row in stats["characters"]}
        rows = []
        selected = set(character_ids or [])
        for spec in self.list_characters():
            if selected and spec.character_id not in selected:
                continue
            row = observed.get(
                spec.character_id,
                {"character_id": spec.character_id, "comments": 0, "platforms": 0, "authors": 0},
            )
            failures = []
            if row["comments"] < min_comments:
                failures.append(f"comments<{min_comments}")
            if row["platforms"] < min_platforms:
                failures.append(f"platforms<{min_platforms}")
            if row["authors"] < min_authors:
                failures.append(f"authors<{min_authors}")
            rows.append({**row, "ready": not failures, "failures": failures})
        return {"ready": all(row["ready"] for row in rows), "characters": rows}


def load_character_catalog(path: str | Path) -> list[CharacterSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload["characters"] if isinstance(payload, dict) else payload
    return [CharacterSpec.model_validate(value) for value in values]


def load_comments(path: str | Path) -> list[Comment]:
    source = Path(path)
    records: list[dict] = []
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            records.extend(csv.DictReader(handle))
    else:
        with source.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {source}:{line_no}: {exc}") from exc
    comments = []
    for index, record in enumerate(records, 1):
        if "is_synthetic" in record and isinstance(record["is_synthetic"], str):
            record["is_synthetic"] = record["is_synthetic"].casefold() in {"1", "true", "yes"}
        try:
            comments.append(Comment.model_validate(record))
        except ValidationError as exc:
            raise ValueError(f"invalid comment record {index} in {source}: {exc}") from exc
    return comments
