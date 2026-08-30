from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sideprofile.schema import CharacterSpec


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_comment_capture.py"
SPEC = importlib.util.spec_from_file_location("prepare_comment_capture", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_prepare_capture_hashes_author_and_redacts_handles() -> None:
    character = CharacterSpec(
        character_id="hp_hermione",
        character_name="Hermione Granger",
        work="Harry Potter",
        anonymous_id="TARGET_A02",
        panel="A",
        aliases=["Hermione", "赫敏"],
    )
    row = module.CaptureRow(
        platform="youtube",
        thread_id="video-1",
        source_comment_id="comment-1",
        author_source_id="public-handle",
        raw_text="@OtherUser Hermione seeks information and helps classmates even under pressure.",
        source_url="https://example.test/watch?v=1&lc=1",
        source_title="Hermione character analysis",
        timestamp="2026-08-29",
        language="en",
        license_note="Private research corpus; raw text is not redistributed.",
        target_relevant=True,
    )
    comments, decisions = module.prepare_rows(
        [row], character=character, salt="private-test-salt", llm=None
    )
    assert len(comments) == 1
    assert comments[0].author_hash != row.author_source_id
    assert "OtherUser" not in comments[0].raw_text
    assert comments[0].raw_text.startswith("[USER]")
    assert decisions[0]["relevant"] is True


def test_prepare_capture_rejects_unadjudicated_without_llm() -> None:
    character = CharacterSpec(
        character_id="x",
        character_name="X",
        work="W",
        anonymous_id="TARGET_X",
        panel="A",
    )
    row = module.CaptureRow(
        platform="forum",
        thread_id="t",
        source_comment_id="c",
        author_source_id="a",
        raw_text="This is long enough but has not been adjudicated for relevance.",
        source_url="https://example.test/t",
        source_title="Target discussion",
        license_note="test",
    )
    try:
        module.prepare_rows([row], character=character, salt="salt", llm=None)
    except ValueError as exc:
        assert "target_relevant" in str(exc)
    else:
        raise AssertionError("unadjudicated input must not enter the corpus")


def test_provider_error_audit_is_not_reused_as_a_label(tmp_path) -> None:
    character = CharacterSpec(
        character_id="x",
        character_name="X",
        work="W",
        anonymous_id="TARGET_X",
        panel="A",
    )
    row = module.CaptureRow(
        platform="forum",
        thread_id="t",
        source_comment_id="c",
        author_source_id="a",
        raw_text="X consistently protects friends even when doing so creates personal risk.",
        source_url="https://example.test/t#c",
        source_title="X character discussion",
        license_note="test",
    )
    decisions_path = tmp_path / "decisions.jsonl"
    decisions_path.write_text(
        json.dumps(
            {
                "character_id": "x",
                "source_comment_id": "c",
                "platform": "forum",
                "relevant": False,
                "classification_status": "classification_error",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeLLM:
        def chat_json(self, **kwargs):
            del kwargs
            return {
                "decisions": [
                    {
                        "source_comment_id": "c",
                        "relevant": True,
                        "evidence_scope": "motives",
                        "rationale": "The comment describes a recurring prosocial motive.",
                    }
                ]
            }

    comments, decisions = module.prepare_rows(
        [row],
        character=character,
        salt="salt",
        llm=FakeLLM(),
        decisions_path=decisions_path,
    )
    assert len(comments) == 1
    assert decisions[0]["classification_status"] == "completed"


def test_comment_identity_is_source_stable_but_character_scoped() -> None:
    row = module.CaptureRow(
        platform="forum",
        thread_id="thread-1",
        source_comment_id="comment-1",
        author_source_id="author-1",
        raw_text="The same observation compares two fictional characters in one sentence.",
        source_url="https://example.test/thread-1#comment-1",
        source_title="Character comparison",
        license_note="test",
        target_relevant=True,
    )
    first = module.stable_comment_id(row, "character_a")
    second = module.stable_comment_id(row, "character_b")
    assert first != second
    assert first.endswith(second.split(":", 1)[1])
