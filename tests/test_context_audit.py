from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_context_usage.py"
SPEC = importlib.util.spec_from_file_location("context_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_context_audit_groups_probe_calls_and_checks_native_window(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 32768}), encoding="utf-8"
    )
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "agent": "cue:D1-Q1",
            "usage": {"prompt_tokens": 1200, "completion_tokens": 200},
        },
        {
            "agent": "condition_summary:D1-Q1",
            "usage": {"prompt_tokens": 1300, "completion_tokens": 180},
        },
        {
            "agent": "condition_summary:aggregate",
            "usage": {"prompt_tokens": 5000, "completion_tokens": 500},
        },
    ]
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = MODULE.audit_trace([trace], MODULE.native_context_window(model_dir))
    assert result["ok"]
    assert result["native_context_window"] == 32768
    assert {item["group"] for item in result["groups"]} == {
        "cue_per_probe",
        "summary_per_probe",
        "summary_aggregate",
    }


def test_context_audit_rejects_total_usage_beyond_native_window(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "agent": "actor",
                "usage": {"prompt_tokens": 7900, "completion_tokens": 400},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = MODULE.audit_trace([trace], 8192)
    assert not result["ok"]
    assert "native context exceeded" in result["failures"][0]
