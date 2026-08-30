from __future__ import annotations

import pytest
from pydantic import ValidationError

from sideprofile.schema import CharacterSpec, Comment, PERSON_MODEL_SECTIONS, PersonModel


def test_character_includes_canonical_alias() -> None:
    spec = CharacterSpec(
        character_id="x",
        character_name="Example Name",
        work="Example Work",
        anonymous_id="TARGET_X",
        panel="A",
    )
    assert spec.aliases == ["Example Name"]


def test_comment_rejects_too_short_text() -> None:
    with pytest.raises(ValidationError):
        Comment(
            comment_id="c1",
            character_id="x",
            character_name="X",
            work="W",
            platform="p",
            author_hash="a",
            raw_text="short",
            language="en",
        )


def test_person_model_requires_all_sections() -> None:
    with pytest.raises(ValidationError):
        PersonModel(anonymous_id="TARGET_X", sections={})
    model = PersonModel(
        anonymous_id="TARGET_X",
        sections={name: [] for name in PERSON_MODEL_SECTIONS},
    )
    assert list(model.sections) == PERSON_MODEL_SECTIONS

