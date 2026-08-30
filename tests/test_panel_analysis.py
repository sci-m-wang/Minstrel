from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_panel_results.py"
SPEC = importlib.util.spec_from_file_location("panel_analysis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_analysis_reports_micro_and_character_macro() -> None:
    rows = []
    for example_id, character_id, raw_value in (
        ("e1", "c1", 0.0),
        ("e2", "c2", 1.0),
        ("e3", "c2", 1.0),
    ):
        rows.append(
            {
                "actor_model": "actor",
                "metric": "score",
                "condition": "ours",
                "replicate": 1,
                "example_id": example_id,
                "character_id": character_id,
                "value": 1.0,
            }
        )
        for baseline in MODULE.CONTRAST_BASELINES:
            rows.append(
                {
                    "actor_model": "actor",
                    "metric": "score",
                    "condition": baseline,
                    "replicate": 1,
                    "example_id": example_id,
                    "character_id": character_id,
                    "value": raw_value,
                }
            )

    contrasts, failures = MODULE.paired_contrasts(rows)
    assert failures == []
    raw = [
        row
        for row in contrasts
        if row["actor_model"] == "actor" and row["contrast"] == "ours-raw"
    ]
    by_aggregation = {row["aggregation"]: row for row in raw}
    assert by_aggregation["exact_unit_micro"]["mean_difference"] == 1 / 3
    assert by_aggregation["character_macro"]["mean_difference"] == 0.5
