from __future__ import annotations

import json

from sideprofile.report import build_report


def test_report_marks_synthetic_run_as_smoke_only(tmp_path) -> None:
    manifest = {
        "run_id": "smoke-1",
        "provider": "MOCK",
        "model": "mock",
        "research_valid": False,
        "characters": ["demo"],
        "conditions": ["ours"],
        "replicates": 1,
        "probe_ids": ["D1-Q1"],
        "corpus_stats": {"total_comments": 3, "synthetic_comments_excluded": 0},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text(
        json.dumps({"condition": "ours", "score": 0.8}) + "\n", encoding="utf-8"
    )
    report = build_report(tmp_path).read_text(encoding="utf-8")
    assert "SMOKE TEST ONLY" in report
    assert "0.8000" in report
