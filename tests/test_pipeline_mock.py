from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sideprofile.anonymize import find_identity_leaks
from sideprofile.corpus import CommentCorpus, load_character_catalog, load_comments
from sideprofile.llm import MockLLMClient
from sideprofile.pipeline import ProfileBuilder, _unique_retrieved_text
from sideprofile.probes import select_probes
from sideprofile.profile import count_identity_tokens, render_person_model
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
            target_tokens=300,
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


def test_raw_baseline_uses_the_registered_conditioning_budget() -> None:
    retrieval = {
        "D1-Q1": [
            {
                "comment_id": f"c{index}",
                "text": " ".join(f"evidence{index}_{word}" for word in range(30)),
            }
            for index in range(10)
        ]
    }
    raw = _unique_retrieved_text(SimpleNamespace(retrieval=retrieval), target_tokens=100)
    observed = count_identity_tokens(raw)
    assert 100 <= observed <= 105


def test_llm_identity_guesses_are_deterministically_masked(tmp_path) -> None:
    with CommentCorpus(tmp_path / "identity.sqlite") as corpus:
        corpus.add_characters(load_character_catalog(ROOT / "data/catalog/characters.json"))
        corpus.add_comments(load_comments(ROOT / "data/smoke/comments.jsonl"))
        result = ProfileBuilder(
            corpus,
            IdentityGuessingLLM(),
            HybridRetriever(mode="bm25", final_top_k=2),
            include_synthetic=True,
            target_tokens=300,
        ).build("demo_alex", select_probes(["D1-Q1"]))
        assert all(not find_identity_leaks(cue.cue, result.character) for cue in result.cues)
        assert all(not find_identity_leaks(cue.context, result.character) for cue in result.cues)
        rendered = render_person_model(result.person_model)
        assert find_identity_leaks(rendered, result.character) == []
        assert "[TARGET]" in rendered
