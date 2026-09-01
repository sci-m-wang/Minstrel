from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sideprofile.roleplay import generate_response
from sideprofile.schema import BenchmarkExample, PersonModel, PERSON_MODEL_SECTIONS


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_actor_input_lengths.py"
SPEC = importlib.util.spec_from_file_location("audit_actor_input_lengths", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def person_model() -> PersonModel:
    return PersonModel(
        anonymous_id="TARGET_X",
        sections={
            section: (["Protects close friends under pressure."] if section == "stable_tendencies" else [])
            for section in PERSON_MODEL_SECTIONS
        },
    )


class RecordingLLM:
    def __init__(self):
        self.request = None

    def chat(self, **kwargs):
        self.request = kwargs
        return "response"


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(sum(len(item["content"].split()) for item in messages) + 3))


def test_actor_messages_match_the_real_generation_prompt() -> None:
    example = BenchmarkExample(
        example_id="e1",
        character_id="x",
        query="What do you do?",
        context="A friend is in danger.",
    )
    model = person_model()
    llm = RecordingLLM()
    generate_response(llm, example=example, model=model, condition="ours")

    messages = module.actor_messages(
        example=example,
        condition="ours",
        conditioning_text="ignored",
        person_model=model,
    )

    assert messages == [
        {"role": "system", "content": llm.request["system"]},
        {"role": "user", "content": llm.request["user"]},
    ]
    user = json.loads(messages[1]["content"])
    assert user["anonymous_id"] == "TARGET_X"
    assert "Protects close friends" in user["conditioning"]


def test_audit_requires_positive_remaining_context() -> None:
    rows = [
        {
            "panel": "panel-a",
            "character_id": "x",
            "example_id": "e1",
            "condition": "none",
            "messages": [{"role": "user", "content": "one two three"}],
        }
    ]
    ok = module.audit_message_rows(rows, FakeTokenizer(), context_window=16)
    assert ok["ok"]
    assert ok["maximum_prompt_tokens"] == 6
    assert ok["minimum_remaining_context"] == 10

    failed = module.audit_message_rows(rows, FakeTokenizer(), context_window=6)
    assert not failed["ok"]
    assert "no generation capacity" in failed["failures"][0]

