from __future__ import annotations

import hashlib

import pytest

from sideprofile.corpus import CommentCorpus
from sideprofile.schema import CharacterSpec, Comment


def spec() -> CharacterSpec:
    return CharacterSpec(
        character_id="x",
        character_name="Example Name",
        work="Example Work",
        anonymous_id="TARGET_X",
        panel="A",
    )


def comment(comment_id: str, *, synthetic: bool = False) -> Comment:
    return Comment(
        comment_id=comment_id,
        character_id="x",
        character_name="Example Name",
        work="Example Work",
        platform="forum",
        author_hash=f"author-{comment_id}",
        raw_text="Example Name carefully restores the plan after an unexpected change.",
        language="en",
        is_synthetic=synthetic,
    )


def test_corpus_deduplicates_normalized_text_and_excludes_synthetic(tmp_path) -> None:
    with CommentCorpus(tmp_path / "comments.sqlite") as corpus:
        corpus.add_character(spec())
        inserted, duplicate = corpus.add_comments([comment("c1"), comment("c2")])
        assert (inserted, duplicate) == (1, 1)
        assert corpus.stats()["total_comments"] == 1

        synthetic = comment("c3", synthetic=True).model_copy(
            update={"raw_text": "A separate synthetic observation with enough useful text."}
        )
        assert corpus.add_comment(synthetic)
        assert len(corpus.comments_for("x")) == 1
        assert len(corpus.comments_for("x", include_synthetic=True)) == 2


def test_corpus_validates_research_targets(tmp_path) -> None:
    with CommentCorpus(tmp_path / "comments.sqlite") as corpus:
        corpus.add_character(spec())
        corpus.add_comment(comment("c1"))
        result = corpus.validate_targets(min_comments=2, min_platforms=1, min_authors=1)
        assert not result["ready"]
        assert result["characters"][0]["failures"] == ["comments<2"]


def test_read_only_corpus_never_rewrites_frozen_database(tmp_path) -> None:
    path = tmp_path / "comments.sqlite"
    with CommentCorpus(path) as corpus:
        corpus.add_character(spec())
        corpus.add_comment(comment("c1"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with CommentCorpus(path, read_only=True) as corpus:
        corpus.initialize()
        assert corpus.get_character("x").character_id == "x"
        assert len(corpus.comments_for("x")) == 1
        with pytest.raises(RuntimeError, match="read-only"):
            corpus.add_character(spec())
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert after == before
    assert not path.with_name(path.name + "-wal").exists()
    assert not path.with_name(path.name + "-shm").exists()
