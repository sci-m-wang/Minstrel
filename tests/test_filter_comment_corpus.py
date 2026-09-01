from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from sideprofile.corpus import CommentCorpus
from sideprofile.schema import CharacterSpec, Comment


SCRIPT = Path(__file__).parents[1] / "scripts" / "filter_comment_corpus.py"
SPEC = importlib.util.spec_from_file_location("filter_comment_corpus", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def add_comment(
    corpus: CommentCorpus,
    comment_id: str,
    *,
    platform: str,
    source_url: str,
    length: int,
    synthetic: bool = False,
) -> None:
    text = (comment_id + " " + ("x" * length))[:length]
    assert corpus.add_comment(
        Comment(
            comment_id=comment_id,
            character_id="x",
            character_name="X",
            work="W",
            platform=platform,
            author_hash=f"author-{comment_id}",
            raw_text=text,
            language="en",
            source_url=source_url,
            is_synthetic=synthetic,
        )
    )


def build_source(path: Path) -> None:
    with CommentCorpus(path) as corpus:
        corpus.add_character(
            CharacterSpec(
                character_id="x",
                character_name="X",
                work="W",
                anonymous_id="TARGET_X",
                panel="A",
            )
        )
        add_comment(
            corpus,
            "question",
            platform="stackexchange",
            source_url="https://scifi.stackexchange.com/questions/1/x",
            length=120,
        )
        add_comment(
            corpus,
            "answer",
            platform="stackexchange",
            source_url="https://scifi.stackexchange.com/questions/1/x#answer-2",
            length=120,
        )
        add_comment(
            corpus,
            "se-comment",
            platform="stackexchange",
            source_url="https://scifi.stackexchange.com/questions/1/x#comment3_1",
            length=602,
        )
        add_comment(
            corpus,
            "se-comment-long",
            platform="stackexchange",
            source_url="https://scifi.stackexchange.com/questions/1/x#comment4_1",
            length=1000,
        )
        add_comment(
            corpus,
            "youtube-999",
            platform="youtube",
            source_url="https://youtube.test/watch?v=1&lc=1",
            length=999,
        )
        add_comment(
            corpus,
            "youtube-1000",
            platform="youtube",
            source_url="https://youtube.test/watch?v=1&lc=2",
            length=1000,
        )
        add_comment(
            corpus,
            "synthetic-long",
            platform="test",
            source_url="https://example.test/synthetic",
            length=1200,
            synthetic=True,
        )


def test_filter_builds_new_verified_database(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    output = tmp_path / "filtered.sqlite"
    audit = tmp_path / "audit.json"
    build_source(source)

    result = module.filter_corpus(source=source, output=output, audit_path=audit)

    assert result["before"]["non_synthetic_rows"] == 6
    assert result["removed"]["stackexchange_question"] == 1
    assert result["removed"]["stackexchange_answer"] == 1
    assert result["removed"]["length_ge_1000"] == 2
    assert result["after"]["non_synthetic_rows"] == 2
    assert result["after"]["synthetic_rows"] == 1
    assert result["integrity_check"] == "ok"
    assert result["source_sha256"] != result["output_sha256"]
    assert json.loads(audit.read_text(encoding="utf-8")) == result

    connection = sqlite3.connect(output)
    try:
        ids = {
            row[0]
            for row in connection.execute("SELECT comment_id FROM comments ORDER BY comment_id")
        }
        assert ids == {"se-comment", "youtube-999", "synthetic-long"}
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "1"
        assert connection.execute("SELECT COUNT(*) FROM characters").fetchone()[0] == 1
    finally:
        connection.close()


def test_filter_refuses_input_or_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    build_source(source)
    with pytest.raises(ValueError, match="different paths"):
        module.filter_corpus(source=source, output=source, audit_path=tmp_path / "audit.json")

    output = tmp_path / "filtered.sqlite"
    output.write_bytes(b"already exists")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.filter_corpus(source=source, output=output, audit_path=tmp_path / "audit.json")
