from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_external_baselines_are_official_best_effort_and_nonblocking() -> None:
    scope = yaml.safe_load((ROOT / "configs/scope.yaml").read_text(encoding="utf-8"))

    assert set(scope["external_baselines"]) == {
        "AMADEUS",
        "RoleGPT_RoleLLM",
        "PersonaForge",
        "CoSER",
    }
    policy = scope["external_baseline_attempt_policy"]
    assert policy == {
        "mode": "best_effort_nonblocking",
        "priority": "core_preflight_and_execution",
        "methods": ["AMADEUS", "RoleGPT_RoleLLM", "PersonaForge", "CoSER"],
        "official_artifacts_only": True,
        "modify_method": False,
        "failure_effect_on_core": "none",
        "evidence_path": "runs/external-baselines/<method>/attempt-status.json",
    }


def test_external_baseline_reference_is_routed_by_the_skill() -> None:
    skill = (ROOT / "skills/side-profile-experiment/SKILL.md").read_text(encoding="utf-8")
    reference = ROOT / "skills/side-profile-experiment/references/external-baselines.md"

    assert reference.is_file()
    assert "references/external-baselines.md" in skill
