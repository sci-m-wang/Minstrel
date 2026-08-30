from __future__ import annotations

from .schema import Probe


_ROWS = [
    ("D1-Q1", "D1", "Stable Tendencies", "How does this person plan, fulfill responsibilities, and react when plans are disrupted?", "此人通常如何计划、履行责任并应对计划被打乱？", ["planning", "responsibility", "routine", "disruption"]),
    ("D1-Q2", "D1", "Stable Tendencies", "What stable differences appear with familiar people versus strangers?", "此人在熟人与陌生人面前有哪些稳定差异？", ["familiar", "stranger", "social behavior"]),
    ("D1-Q3", "D1", "Stable Tendencies", "How does this person usually approach rules, uncertainty, novelty, and risk?", "此人通常如何面对规则、不确定性、新事物与风险？", ["rules", "uncertainty", "novelty", "risk"]),
    ("D2-Q1", "D2", "Motives & Goals", "What does this person repeatedly pursue or protect?", "此人反复追求或保护什么？", ["goal", "protect", "pursue"]),
    ("D2-Q2", "D2", "Motives & Goals", "What does this person most fear losing or try to avoid?", "此人最害怕失去或避免什么？", ["fear", "loss", "avoid"]),
    ("D2-Q3", "D2", "Motives & Goals", "For which goals will this person accept substantial personal cost?", "此人愿意为哪些目标承担明显代价？", ["sacrifice", "cost", "commitment"]),
    ("D3-Q1", "D3", "Values & Moral Priorities", "When loyalty conflicts with rules or fairness, what does this person prioritize?", "忠诚与规则或公平冲突时，此人如何选择？", ["loyalty", "rules", "fairness"]),
    ("D3-Q2", "D3", "Values & Moral Priorities", "When self-interest conflicts with helping others, what does this person do?", "自身利益与帮助他人冲突时，此人如何选择？", ["self-interest", "help", "others"]),
    ("D3-Q3", "D3", "Values & Moral Priorities", "What boundaries does this person appear unwilling to cross?", "此人有哪些明显不可跨越的底线？", ["moral boundary", "refuse", "line"]),
    ("D4-Q1", "D4", "Cognitive/Appraisal Style", "When intentions are ambiguous, does this person first trust or suspect?", "面对模糊意图时，此人首先信任还是怀疑？", ["ambiguous", "trust", "suspicion"]),
    ("D4-Q2", "D4", "Cognitive/Appraisal Style", "How does this person interpret failure, criticism, and authority?", "此人如何解释失败、批评和权威？", ["failure", "criticism", "authority"]),
    ("D4-Q3", "D4", "Cognitive/Appraisal Style", "What recurring assumptions does this person make about people and the world?", "此人对人和世界有哪些反复出现的假设？", ["assumption", "people", "worldview"]),
    ("D5-Q1", "D5", "Affect & Coping", "What most reliably triggers anger, shame, anxiety, or joy in this person?", "什么最容易触发此人的愤怒、羞耻、焦虑或快乐？", ["anger", "shame", "anxiety", "joy", "trigger"]),
    ("D5-Q2", "D5", "Affect & Coping", "How does this person's outward expression differ from their apparent inner emotion?", "此人的真实情绪与外在表达是否一致？", ["emotion", "expression", "conceal"]),
    ("D5-Q3", "D5", "Affect & Coping", "How does this person regulate emotion and recover after stress?", "此人通常如何调节情绪并从压力中恢复？", ["coping", "recover", "stress"]),
    ("D6-Q1", "D6", "Interpersonal Pattern", "How does this person differ with intimates, strangers, rivals, and authorities?", "此人面对亲密者、陌生人、竞争者和权威时有何差异？", ["intimate", "stranger", "rival", "authority"]),
    ("D6-Q2", "D6", "Interpersonal Pattern", "How does this person express care, refusal, and conflict?", "此人如何表达关心、拒绝和冲突？", ["care", "refusal", "conflict"]),
    ("D6-Q3", "D6", "Interpersonal Pattern", "Does this person tend toward control, dependence, dominance, or cooperation in relationships?", "此人在人际中更偏控制、依赖、支配还是配合？", ["control", "dependence", "dominance", "cooperation"]),
    ("D7-Q1", "D7", "Self & Narrative Identity", "Which experiences do observers believe genuinely shaped this person?", "评论者认为哪些经历真正塑造了此人？", ["shaped", "experience", "past"]),
    ("D7-Q2", "D7", "Self & Narrative Identity", "What self-image does this person appear to maintain?", "此人似乎试图维持怎样的自我形象？", ["self-image", "identity", "reputation"]),
    ("D7-Q3", "D7", "Self & Narrative Identity", "Which blind spots or inner contradictions do observers repeatedly note?", "评论者反复指出哪些自我盲点或内在矛盾？", ["blind spot", "contradiction", "hypocrisy"]),
    ("D8-Q1", "D8", "Situation & Expression", "How does this person react under stress, public challenge, or threat to a close other?", "此人在压力、公开挑战或亲近者受威胁时如何反应？", ["stress", "public challenge", "threat"]),
    ("D8-Q2", "D8", "Situation & Expression", "Which situations are exceptions to this person's general tendencies?", "哪些情境构成此人一般人格规律的例外？", ["exception", "boundary condition", "unless"]),
    ("D8-Q3", "D8", "Situation & Expression", "What recurring language, humor, directness, and expressive style characterize this person?", "此人有哪些典型语言、幽默、直接程度和表达方式？", ["language", "humor", "directness", "speech style"]),
]

PROBES = [
    Probe(
        probe_id=row[0],
        domain_id=row[1],
        domain=row[2],
        question_en=row[3],
        question_zh=row[4],
        search_terms=row[5],
    )
    for row in _ROWS
]

PROBES_BY_ID = {probe.probe_id: probe for probe in PROBES}


def select_probes(probe_ids: list[str] | None = None) -> list[Probe]:
    if not probe_ids:
        return list(PROBES)
    missing = [probe_id for probe_id in probe_ids if probe_id not in PROBES_BY_ID]
    if missing:
        raise ValueError(f"unknown probe ids: {', '.join(missing)}")
    return [PROBES_BY_ID[probe_id] for probe_id in probe_ids]
