from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sideprofile.anonymize import find_identity_leaks
from sideprofile.corpus import CommentCorpus, load_character_catalog, load_comments
from sideprofile.llm import MockLLMClient
from sideprofile.pipeline import SUPPORTED_CONDITIONS, ProfileBuilder, _condition_payload
from sideprofile.probes import select_probes
from sideprofile.profile import extract_cues, render_person_model
from sideprofile.retrieval import DeterministicSmokeRetriever
from sideprofile.roleplay import generate_response, judge_response
from sideprofile.schema import BenchmarkExample, PERSON_MODEL_SECTIONS, RetrievedComment


ROOT = Path(__file__).resolve().parent.parent


class IdentityGuessingLLM(MockLLMClient):
    def chat_json(self, *, agent: str, **kwargs) -> dict:
        self.usage.calls += 1
        if agent.startswith("cue:"):
            user = json.loads(kwargs["user"])
            return {
                "cues": [
                    {
                        "cue": "Alex Example restores structure.",
                        "context": "Example World",
                        "cue_type": "behavioral_pattern",
                        "directness": "behavior_based",
                        "support": [user["comments"][0]["comment_id"]],
                        "counterevidence": [],
                        "confidence": 0.7,
                    }
                ]
            }
        if agent == "person_model":
            return {
                "sections": {
                    section: ["Alex Example is recognizable from Example World."]
                    for section in PERSON_MODEL_SECTIONS
                }
            }
        return {}


class RecordingConditionLLM(MockLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict[str, str]] = []

    def chat(self, *, system: str, user: str, agent: str, **kwargs) -> str:
        self.requests.append({"system": system, "user": user, "agent": agent})
        return f"natural {agent} output"


class RecordingCueLLM(MockLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict] = []

    def chat_json(self, *, system: str, user: str, agent: str, **kwargs) -> dict:
        payload = json.loads(user)
        self.requests.append({"system": system, "user": payload, "agent": agent})
        comment_id = payload["comments"][0]["comment_id"]
        return {
            "cues": [
                {
                    "cue": "A supported observation.",
                    "context": "the supplied situation",
                    "cue_type": "behavioral_pattern",
                    "directness": "behavior_based",
                    "support": [comment_id],
                    "counterevidence": [],
                    "confidence": 0.7,
                }
            ]
        }


def test_identity_blind_pipeline_end_to_end(tmp_path) -> None:
    with CommentCorpus(tmp_path / "smoke.sqlite") as corpus:
        corpus.add_characters(load_character_catalog(ROOT / "data/catalog/characters.json"))
        inserted, duplicates = corpus.add_comments(load_comments(ROOT / "data/smoke/comments.jsonl"))
        assert inserted == 10
        assert duplicates == 0
        llm = MockLLMClient()
        builder = ProfileBuilder(
            corpus,
            llm,
            DeterministicSmokeRetriever(final_top_k=10),
            include_synthetic=True,
        )
        result = builder.build("demo_alex", select_probes(["D1-Q1"]))
        rendered = render_person_model(result.person_model)
        assert "Alex" not in rendered
        assert result.cues[0].support == ["demo_c01", "demo_c02"]
        assert result.cues[0].status == "CONTESTED"

        example = BenchmarkExample.model_validate_json(
            (ROOT / "data/smoke/benchmark.jsonl").read_text(encoding="utf-8").strip()
        )
        output = generate_response(
            llm, example=example, model=result.person_model, condition="ours"
        )
        score, rationale = judge_response(
            llm, spec=result.character, example=example, output=output
        )
        assert "restore" in output
        assert score == 0.75
        assert rationale


def test_profile_builder_uses_frozen_catalog_spec_instead_of_corpus_metadata(
    tmp_path,
) -> None:
    catalog_spec = load_character_catalog(ROOT / "data/catalog/characters.json")[0]
    corpus_spec = catalog_spec.model_copy(update={"gold_profile": ""})
    frozen_spec = catalog_spec.model_copy(update={"gold_profile": "Frozen official profile."})
    with CommentCorpus(tmp_path / "catalog-source.sqlite") as corpus:
        corpus.add_character(corpus_spec)
        corpus.add_comments(load_comments(ROOT / "data/smoke/comments.jsonl"))
        result = ProfileBuilder(
            corpus,
            MockLLMClient(),
            DeterministicSmokeRetriever(final_top_k=10),
            characters={frozen_spec.character_id: frozen_spec},
            include_synthetic=True,
        ).build("demo_alex", select_probes(["D1-Q1"]))
    assert result.character.gold_profile == "Frozen official profile."


def test_cue_extraction_keeps_each_probe_comment_set_isolated() -> None:
    llm = RecordingCueLLM()
    probes = select_probes(["D1-Q1", "D2-Q1"])
    spec = SimpleNamespace(
        character_id="x",
        character_name="Example Target",
        aliases=["Example Target"],
        work="Example Work",
        anonymous_id="TARGET_X",
    )
    for probe, comment_id in zip(probes, ("c1", "c2"), strict=True):
        extract_cues(
            llm,
            spec=spec,
            probe=probe,
            retrieved=[
                RetrievedComment(
                    comment_id=comment_id,
                    anonymous_id="TARGET_X",
                    text=f"evidence for {comment_id}",
                    platform="test",
                    author_hash=f"author-{comment_id}",
                    language="en",
                    score=1.0,
                )
            ],
        )
    assert [request["agent"] for request in llm.requests] == ["cue:D1-Q1", "cue:D2-Q1"]
    assert [[item["comment_id"] for item in request["user"]["comments"]] for request in llm.requests] == [["c1"], ["c2"]]


def test_raw_is_not_an_executable_condition() -> None:
    assert "raw" not in SUPPORTED_CONDITIONS
    retrieval = {
        "D1-Q1": [
            {
                "comment_id": f"c{index}",
                "text": " ".join(f"evidence{index}_{word}" for word in range(30)),
            }
            for index in range(40)
        ],
        "D2-Q1": [
            {
                "comment_id": f"c{index}",
                "text": " ".join(f"evidence{index}_{word}" for word in range(30)),
            }
            for index in range(39, 60)
        ],
    }
    with pytest.raises(ValueError, match="unknown condition"):
        _condition_payload(
            RecordingConditionLLM(),
            condition="raw",
            result=SimpleNamespace(retrieval=retrieval),
        )


def test_summary_and_personality_process_each_probe_before_aggregation() -> None:
    retrieval = {
        "D1-Q1": [
            {
                "comment_id": "c1",
                "text": "first probe evidence",
            }
        ],
        "D2-Q1": [
            {
                "comment_id": "c2",
                "text": "second probe evidence",
            }
        ],
    }
    result = SimpleNamespace(
        retrieval=retrieval,
        character=SimpleNamespace(
            character_id="x",
            character_name="Example Target",
            aliases=[],
            work="Example Work",
            gold_profile="",
            anonymous_id="TARGET_X",
            language="en",
        ),
    )
    llm = RecordingConditionLLM()
    for condition in ("summary", "personality"):
        assert _condition_payload(llm, condition=condition, result=result).startswith("natural")
    assert [request["agent"] for request in llm.requests] == [
        "condition_summary:D1-Q1",
        "condition_summary:D2-Q1",
        "condition_summary:aggregate",
        "condition_personality:D1-Q1",
        "condition_personality:D2-Q1",
        "condition_personality:aggregate",
    ]
    for condition_offset in (0, 3):
        first = json.loads(llm.requests[condition_offset]["user"])
        second = json.loads(llm.requests[condition_offset + 1]["user"])
        aggregate = json.loads(llm.requests[condition_offset + 2]["user"])
        assert [item["comment_id"] for item in first["comments"]] == ["c1"]
        assert [item["comment_id"] for item in second["comments"]] == ["c2"]
        assert "comments" not in aggregate
        assert [item["probe_id"] for item in aggregate["probe_observations"]] == [
            "D1-Q1",
            "D2-Q1",
        ]
    for request in llm.requests:
        assert "token" not in request["system"].lower()


def test_llm_identity_guesses_are_deterministically_masked(tmp_path) -> None:
    with CommentCorpus(tmp_path / "identity.sqlite") as corpus:
        corpus.add_characters(load_character_catalog(ROOT / "data/catalog/characters.json"))
        corpus.add_comments(load_comments(ROOT / "data/smoke/comments.jsonl"))
        result = ProfileBuilder(
            corpus,
            IdentityGuessingLLM(),
            DeterministicSmokeRetriever(final_top_k=2),
            include_synthetic=True,
        ).build("demo_alex", select_probes(["D1-Q1"]))
        assert all(not find_identity_leaks(cue.cue, result.character) for cue in result.cues)
        assert all(not find_identity_leaks(cue.context, result.character) for cue in result.cues)
        rendered = render_person_model(result.person_model)
        assert find_identity_leaks(rendered, result.character) == []
        assert "[TARGET]" in rendered
