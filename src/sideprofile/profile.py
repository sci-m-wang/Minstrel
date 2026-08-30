from __future__ import annotations

import json
import re
from collections import defaultdict

from .anonymize import anonymize_text
from .llm import LLMClient, MockLLMClient
from .schema import (
    CharacterSpec,
    PERSON_MODEL_SECTIONS,
    Cue,
    PersonModel,
    Probe,
    RetrievedComment,
)


CUE_SYSTEM = """You extract behaviorally grounded cues from third-party comments about an anonymous target.
Refer to the person only by the supplied anonymous_id. Never output or guess a real name, work, or
franchise. Never convert observations directly into personality labels.
Use only the supplied comment IDs. Separate supporting and counterevidence. If evidence is insufficient,
return no cues. Return JSON: {"cues":[{"cue":str,"context":str,"cue_type":str,
"directness":str,"support":[comment_id],"counterevidence":[comment_id],"confidence":0..1}]}.
Allowed cue_type values: behavioral_pattern, motive, value_priority, appraisal_style, affect_coping,
interpersonal_pattern, narrative_theme, expressive_signature, unknown.
Allowed directness values: behavior_based, explicit_judgment, inferred, unknown."""


MODEL_SYSTEM = """Construct an anonymous reconstructed person model from evidence-cited cues only.
Refer to the person only by the supplied anonymous_id. Never output or guess a real name, work, or
franchise. Do not add biography or missing facts. Express tendencies with context, boundary
conditions, and counterevidence. Put insufficient or disputed claims in unknown_contested.
The user supplies a target_tokens treatment budget. The final rendered conditioning includes section
headings and bullet prefixes: budget for that overhead and make the complete rendering fall within
95% to 105% of the target under your own tokenizer, without padding, omitting supported domains, or
changing the JSON schema. This is a conditioning-size requirement, not an API output cap.
Return JSON with one key, sections. sections must contain exactly these keys, each a list of concise
statements: stable_tendencies, motives_and_goals, value_priorities, cognitive_appraisal_model,
affective_dynamics, interpersonal_patterns, self_narrative_themes, situation_behavior_signatures,
expressive_signature, contradictions_boundary_conditions, unknown_contested."""


def _cue_status(support: list[str], counter: list[str], comments: dict[str, RetrievedComment]) -> str:
    authors = {comments[item].author_hash for item in support if item in comments}
    platforms = {comments[item].platform for item in support if item in comments}
    if not support:
        return "UNKNOWN"
    if counter:
        return "CONTESTED"
    if len(authors) >= 2 or len(platforms) >= 2:
        return "SUPPORTED"
    return "WEAK"


def extract_cues(
    llm: LLMClient | MockLLMClient,
    *,
    spec: CharacterSpec,
    probe: Probe,
    retrieved: list[RetrievedComment],
) -> list[Cue]:
    allowed = {comment.comment_id: comment for comment in retrieved}
    evidence = [
        {
            "comment_id": comment.comment_id,
            "platform": comment.platform,
            "observer": comment.author_hash,
            "text": comment.text,
        }
        for comment in retrieved
    ]
    payload = llm.chat_json(
        system=CUE_SYSTEM,
        user=json.dumps(
            {
                "anonymous_id": retrieved[0].anonymous_id if retrieved else "TARGET",
                "probe_id": probe.probe_id,
                "question": probe.question_en,
                "comments": evidence,
            },
            ensure_ascii=False,
        ),
        agent=f"cue:{probe.probe_id}",
    )
    cues = []
    for index, raw in enumerate(payload.get("cues", []), 1):
        support = list(dict.fromkeys(item for item in raw.get("support", []) if item in allowed))
        counter = list(
            dict.fromkeys(item for item in raw.get("counterevidence", []) if item in allowed)
        )
        if not support:
            continue
        status = _cue_status(support, counter, allowed)
        confidence = float(raw.get("confidence", 0.0))
        if status == "WEAK":
            confidence = min(confidence, 0.59)
        cues.append(
            Cue(
                cue_id=f"{probe.probe_id}-C{index:02d}",
                probe_id=probe.probe_id,
                domain_id=probe.domain_id,
                cue=anonymize_text(str(raw.get("cue", "")), spec),
                context=anonymize_text(str(raw.get("context", "")), spec),
                cue_type=raw.get("cue_type", "unknown"),
                directness=raw.get("directness", "unknown"),
                support=support,
                counterevidence=counter,
                confidence=confidence,
                status=status,
            )
        )
    return cues


def build_person_model(
    llm: LLMClient | MockLLMClient,
    *,
    spec: CharacterSpec,
    cues: list[Cue],
    target_tokens: int = 1000,
) -> PersonModel:
    usable = [cue for cue in cues if cue.status != "UNKNOWN"]
    payload = llm.chat_json(
        system=MODEL_SYSTEM,
        user=json.dumps(
            {
                "anonymous_id": spec.anonymous_id,
                "target_tokens": target_tokens,
                "cues": [cue.model_dump() for cue in usable],
            },
            ensure_ascii=False,
        ),
        agent="person_model",
    )
    raw_sections = payload.get("sections", {})
    sections = {}
    for name in PERSON_MODEL_SECTIONS:
        value = raw_sections.get(name, [])
        if isinstance(value, str):
            value = [value]
        sections[name] = [
            anonymize_text(str(item).strip(), spec)
            for item in value
            if str(item).strip()
        ]
    return PersonModel(
        anonymous_id=spec.anonymous_id,
        sections=sections,
        cue_ids=[cue.cue_id for cue in usable],
        model_name=llm.settings.model,
    )


def render_person_model(model: PersonModel) -> str:
    labels = {
        "stable_tendencies": "Stable Tendencies",
        "motives_and_goals": "Motives and Goals",
        "value_priorities": "Value Priorities",
        "cognitive_appraisal_model": "Cognitive / Appraisal Model",
        "affective_dynamics": "Affective Dynamics",
        "interpersonal_patterns": "Interpersonal Patterns",
        "self_narrative_themes": "Self and Narrative Themes",
        "situation_behavior_signatures": "Situation–Behavior Signatures",
        "expressive_signature": "Expressive Signature",
        "contradictions_boundary_conditions": "Contradictions and Boundary Conditions",
        "unknown_contested": "Unknown / Contested",
    }
    blocks = []
    for key in PERSON_MODEL_SECTIONS:
        values = model.sections[key]
        body = "\n".join(f"- {value}" for value in values) if values else "- Insufficient evidence."
        blocks.append(f"[{labels[key]}]\n{body}")
    return "\n\n".join(blocks)


def count_identity_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[\u3400-\u9fff]", text))


def cue_coverage(cues: list[Cue]) -> dict:
    by_domain: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    for cue in cues:
        by_domain[cue.domain_id] += 1
        by_status[cue.status] += 1
    return {"by_domain": dict(sorted(by_domain.items())), "by_status": dict(sorted(by_status.items()))}
