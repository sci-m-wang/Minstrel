from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sideprofile.anonymize import find_identity_leaks
from sideprofile.corpus import CommentCorpus, load_character_catalog, load_comments
from sideprofile.llm import MockLLMClient
from sideprofile.pipeline import ProfileBuilder, _condition_payload, _unique_retrieved_text
from sideprofile.probes import select_probes
from sideprofile.profile import render_person_model
from sideprofile.retrieval import HybridRetriever
from sideprofile.roleplay import generate_response, judge_response
from sideprofile.schema import BenchmarkExample, PERSON_MODEL_SECTIONS


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


class RecordingBaselineLLM(MockLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict[str, str]] = []

    def chat(self, *, system: str, user: str, agent: str, **kwargs) -> str:
        self.requests.append({"system": system, "user": user, "agent": agent})
        return f"natural {agent} output"


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
            HybridRetriever(mode="bm25", final_top_k=10),
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


def test_raw_condition_uses_every_unique_retrieved_comment_without_truncation() -> None:
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
    raw = _unique_retrieved_text(SimpleNamespace(retrieval=retrieval))
    assert "[c0]" in raw
    assert "[c59]" in raw
    assert raw.count("[c39]") == 1
    assert raw.count("\n") == 59


def test_summary_and_personality_read_the_complete_comment_set_without_length_instruction() -> None:
    retrieval = {
        "D1-Q1": [
            {
                "comment_id": f"c{index}",
                "text": " ".join(f"evidence{index}_{word}" for word in range(30)),
            }
            for index in range(60)
        ]
    }
    result = SimpleNamespace(
        retrieval=retrieval,
        character=SimpleNamespace(
            character_id="x",
            character_name="Example Target",
            aliases=[],
            work="Example Work",
            gold_profile="",
        ),
    )
    llm = RecordingBaselineLLM()
    for condition in ("summary", "personality"):
        assert _condition_payload(llm, condition=condition, result=result).startswith("natural")
    assert [request["agent"] for request in llm.requests] == [
        "condition_summary",
        "condition_personality",
    ]
    for request in llm.requests:
        assert "[c0]" in request["user"]
        assert "[c59]" in request["user"]
        assert "token" not in request["system"].lower()


def test_llm_identity_guesses_are_deterministically_masked(tmp_path) -> None:
    with CommentCorpus(tmp_path / "identity.sqlite") as corpus:
        corpus.add_characters(load_character_catalog(ROOT / "data/catalog/characters.json"))
        corpus.add_comments(load_comments(ROOT / "data/smoke/comments.jsonl"))
        result = ProfileBuilder(
            corpus,
            IdentityGuessingLLM(),
            HybridRetriever(mode="bm25", final_top_k=2),
            include_synthetic=True,
        ).build("demo_alex", select_probes(["D1-Q1"]))
        assert all(not find_identity_leaks(cue.cue, result.character) for cue in result.cues)
        assert all(not find_identity_leaks(cue.context, result.character) for cue in result.cues)
        rendered = render_person_model(result.person_model)
        assert find_identity_leaks(rendered, result.character) == []
        assert "[TARGET]" in rendered
