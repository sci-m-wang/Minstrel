from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_charactereval.py"
SPEC = importlib.util.spec_from_file_location("charactereval_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_official_formatter_and_aggregate() -> None:
    record = {
        "role": "华妃",
        "context": "上下文",
        "model_output": "回答",
        "metric_zh": "行为一致性",
        "metric_en": "Behavior",
        "condition": "ours",
        "replicate": 1,
        "actor_model": "actor",
        "character_rm_score": 4.5,
    }
    text = MODULE.format_input(record, {"华妃": {"姓名": "年世兰"}})
    assert text.startswith("<RoleInfo>\n\n")
    assert "<Response>\n\n回答" in text
    summary = MODULE.aggregate([record])
    assert summary[0]["mean"] == 4.5
    assert summary[0]["primary_metric"] is True
