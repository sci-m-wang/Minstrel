from __future__ import annotations

import json

from .llm import LLMClient, MockLLMClient
from .profile import render_person_model
from .schema import BenchmarkExample, CharacterSpec, PersonModel


ACTOR_SYSTEM = """Role-play the anonymous target using the supplied anonymous conditioning, if any.
Do not guess or reveal the target's identity. Do not mechanically list conditioning content or traits.
Do not invent missing biographical facts. Respond naturally to the situation in the target's expressive
style and behavioral pattern when those are supported by the supplied conditioning."""


def generate_response(
    llm: LLMClient | MockLLMClient,
    *,
    example: BenchmarkExample,
    model: PersonModel | None,
    condition: str,
    conditioning_text: str = "",
) -> str:
    if condition == "ours":
        if model is None:
            raise ValueError("ours condition requires a person model")
        conditioning_text = render_person_model(model)
    prompt = {
        "anonymous_id": model.anonymous_id if model else "TARGET",
        "conditioning": conditioning_text,
        "context": example.context,
        "query": example.query,
    }
    return llm.chat(
        system=ACTOR_SYSTEM,
        user=json.dumps(prompt, ensure_ascii=False),
        agent="actor",
    ).strip()


JUDGE_SYSTEM = """Evaluate a role-play response against the private target identity and available reference.
The actor was intentionally identity-blind, so do not reward explicit name-dropping. Score behavioral,
personality, and expressive fidelity from 0 to 1. Do not penalize missing encyclopedic facts when the
reference does not require them. Return JSON {"score":number,"rationale":string}."""


def judge_response(
    llm: LLMClient | MockLLMClient,
    *,
    spec: CharacterSpec,
    example: BenchmarkExample,
    output: str,
) -> tuple[float, str]:
    payload = llm.chat_json(
        system=JUDGE_SYSTEM,
        user=json.dumps(
            {
                "character_name": spec.character_name,
                "work": spec.work,
                "gold_profile": spec.gold_profile,
                "query": example.query,
                "context": example.context,
                "reference": example.reference,
                "candidate": output,
            },
            ensure_ascii=False,
        ),
        agent="judge",
    )
    score = min(1.0, max(0.0, float(payload.get("score", 0.0))))
    return score, str(payload.get("rationale", ""))
