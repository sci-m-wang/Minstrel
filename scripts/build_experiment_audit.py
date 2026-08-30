#!/usr/bin/env python3
"""Persist a machine-readable audit of retained baselines, profiles, models, and benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


EXPECTED_CONDITIONS = ["none", "personality", "raw", "summary", "gold", "ours"]
FORBIDDEN_KEYS = {
    "max_tokens", "max_completion_tokens", "max_calls", "max_retries",
    "temperature", "top_p", "seed", "seeds",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def forbidden_paths(value: object, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_KEYS:
                found.append(path)
            found.extend(forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def load_catalog(path: Path) -> dict[str, dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("characters", value)
    return {row["character_id"]: row for row in rows}


def audit_panel(root: Path, config_path: Path, registry: dict) -> tuple[dict, list[str]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run = config["run"]
    data = config["data"]
    panel_key = str(run["name"]).replace("-", "_")
    selected = list(run["character_ids"])
    catalog_path = root / data["catalog"]
    benchmark_path = root / data["benchmark"]
    catalog = load_catalog(catalog_path)
    benchmark = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = []
    conditions = list(run.get("conditions", []))
    if conditions != EXPECTED_CONDITIONS:
        failures.append(f"{run['name']}: conditions differ from the six retained treatments")
    forbidden = forbidden_paths(config)
    if forbidden:
        failures.append(f"{run['name']}: forbidden settings: {', '.join(forbidden)}")
    profiles = []
    for character_id in selected:
        row = catalog.get(character_id)
        profile = str((row or {}).get("gold_profile") or "")
        if not row:
            failures.append(f"{run['name']}: missing catalog role {character_id}")
        if not profile:
            failures.append(f"{run['name']}: missing official gold profile {character_id}")
        profiles.append(
            {
                "character_id": character_id,
                "anonymous_id": (row or {}).get("anonymous_id"),
                "profile_characters": len(profile),
                "profile_sha256": sha256_bytes(profile.encode("utf-8")),
            }
        )
    counts = Counter(item["character_id"] for item in benchmark)
    missing_benchmark = [item for item in selected if counts[item] == 0]
    if missing_benchmark:
        failures.append(f"{run['name']}: benchmark missing {', '.join(missing_benchmark)}")
    actors = list(registry.get("actor_matrix", {}).get(panel_key, []))
    undefined = [item for item in actors if item not in registry.get("models", {})]
    if not actors or undefined:
        failures.append(f"{run['name']}: invalid actor matrix {undefined}")
    retrieval = config.get("retrieval", {})
    embedding_model_key = str(retrieval.get("embedding_model_key", ""))
    vector_store_value = str(retrieval.get("vector_store", ""))
    if retrieval.get("mode") == "hybrid":
        if embedding_model_key not in registry.get("models", {}):
            failures.append(
                f"{run['name']}: undefined embedding model {embedding_model_key}"
            )
        vector_store_path = root / vector_store_value
        if not vector_store_value or not vector_store_path.is_file():
            failures.append(f"{run['name']}: missing frozen vector store {vector_store_value}")
    tasks = Counter(
        str(item.get("metadata", {}).get("task", "CharacterRM")) for item in benchmark
    )
    return (
        {
            "panel": run["name"],
            "config": str(config_path.relative_to(root)),
            "config_sha256": sha256_file(config_path),
            "conditions": conditions,
            "replicates": run.get("replicates"),
            "conditioning_target": config.get("profiling", {}).get("target_tokens"),
            "controlled_conditioning_range": [950, 1050],
            "decoding": "provider_default",
            "retrieval_mode": retrieval.get("mode"),
            "retrieval": {
                "mode": retrieval.get("mode"),
                "vector_store": vector_store_value,
                "embedding_model_key": embedding_model_key,
                "reranker_model": retrieval.get("reranker_model"),
            },
            "coverage": data.get("coverage"),
            "characters": selected,
            "official_gold_profiles": profiles,
            "benchmark": {
                "path": str(benchmark_path.relative_to(root)),
                "sha256": sha256_file(benchmark_path),
                "examples": len(benchmark),
                "examples_by_character": dict(sorted(counts.items())),
                "tasks": dict(sorted(tasks.items())),
            },
            "actors": actors,
            "forbidden_config_paths": forbidden,
            "ready": not failures,
        },
        failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="data/audits/experiment_inventory.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    scope_path = root / "configs/scope.yaml"
    registry_path = root / "offline/models.yaml"
    scope = yaml.safe_load(scope_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    failures = []
    if scope.get("supported_conditions") != EXPECTED_CONDITIONS:
        failures.append("configs/scope.yaml does not define the exact six retained conditions")
    profiler = str(registry.get("profiler", {}).get("model", ""))
    if profiler != "qwen2.5-14b-instruct":
        failures.append(f"wrong fixed profiler: {profiler}")
    panels = []
    for name in ("panel-a.yaml", "panel-d.yaml"):
        panel, panel_failures = audit_panel(root, root / "configs/offline" / name, registry)
        panels.append(panel)
        failures.extend(panel_failures)
    evaluator_paths = [
        root / "data/evaluators/charactereval/character_profiles.selected.json",
        root / "data/evaluators/charactereval/id2metric.selected.json",
        root / "scripts/evaluate_roleagentbench.py",
        root / "scripts/evaluate_charactereval.py",
    ]
    missing_evaluators = [str(path.relative_to(root)) for path in evaluator_paths if not path.is_file()]
    if missing_evaluators:
        failures.append("missing evaluator assets: " + ", ".join(missing_evaluators))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": not failures,
        "failures": failures,
        "scope": {
            "path": "configs/scope.yaml",
            "sha256": sha256_file(scope_path),
            "active_panels": scope.get("active_panels"),
            "retained_conditions": scope.get("supported_conditions"),
            "baseline_definitions": scope.get("baseline_definitions"),
            "removed_external_baselines": scope.get("removed_external_baselines"),
        },
        "model_registry": {
            "path": "offline/models.yaml",
            "sha256": sha256_file(registry_path),
            "profiler": profiler,
            "decoding": registry.get("decoding"),
        },
        "panels": panels,
        "evaluators": [
            {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in evaluator_paths
            if path.is_file()
        ],
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if not failures else "failed", "output": str(output), "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
