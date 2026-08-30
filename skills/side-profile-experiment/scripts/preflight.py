#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import yaml


def load_env_keys(path: Path) -> set[str]:
    keys = set(os.environ)
    if not path.is_file():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and value.strip():
            keys.add(key.strip())
    return keys


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve(root, args.config)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    check("project", (root / "pyproject.toml").is_file() and (root / "src/sideprofile").is_dir(), str(root))
    check("config", config_path.is_file(), str(config_path))
    if not config_path.is_file():
        print(json.dumps({"ok": False, "checks": checks}, indent=2))
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    forbidden = {
        "max_tokens",
        "max_completion_tokens",
        "max_calls",
        "max_retries",
        "temperature",
        "top_p",
        "seed",
        "seeds",
    }

    def find_forbidden(value: object, prefix: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if str(key) in forbidden:
                    found.append(path)
                found.extend(find_forbidden(child, path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(find_forbidden(child, f"{prefix}[{index}]"))
        return found

    forbidden_paths = find_forbidden(config)
    check(
        "unrelated_limits",
        not forbidden_paths,
        "no API token caps, call budgets, retry limits, temperature, top_p, or seed; found: "
        + ", ".join(forbidden_paths),
    )
    provider = config.get("provider", {}).get("name", "GPT").upper()
    env_file = resolve(root, config.get("provider", {}).get("env_file", ".env"))
    env_keys = load_env_keys(env_file)
    required = {f"{provider}_API_KEY", f"{provider}_MODEL"}
    check("provider_env", required.issubset(env_keys), f"required keys present: {sorted(required)}")

    data = config.get("data", {})
    run_name = config.get("run", {}).get("name", "")
    is_smoke = "smoke" in run_name.casefold()
    for field in ("catalog", "benchmark"):
        path = resolve(root, data.get(field, "missing"))
        check(field, path.is_file(), str(path))
    catalog_path = resolve(root, data.get("catalog", "missing"))
    benchmark_path = resolve(root, data.get("benchmark", "missing"))
    selected = config.get("run", {}).get("character_ids", [])
    conditions = config.get("run", {}).get("conditions", [])
    expected_conditions = ["none", "personality", "raw", "summary", "gold", "ours"]
    check(
        "condition_scope",
        is_smoke or conditions == expected_conditions,
        f"configured={conditions}; expected={expected_conditions}",
    )
    if catalog_path.is_file() and not is_smoke:
        catalog_value = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_rows = catalog_value.get("characters", catalog_value)
        by_id = {row["character_id"]: row for row in catalog_rows}
        missing_gold = [
            character_id
            for character_id in selected
            if not str((by_id.get(character_id) or {}).get("gold_profile") or "").strip()
        ]
        missing_catalog = [character_id for character_id in selected if character_id not in by_id]
        check(
            "official_gold_profiles",
            not missing_gold and not missing_catalog,
            "missing catalog: " + ", ".join(missing_catalog)
            + "; missing official gold profile: " + ", ".join(missing_gold),
        )
    if benchmark_path.is_file() and not is_smoke:
        benchmark_characters = {
            json.loads(line)["character_id"]
            for line in benchmark_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing_benchmark = [item for item in selected if item not in benchmark_characters]
        check(
            "benchmark_character_coverage",
            not missing_benchmark,
            "missing: " + ", ".join(missing_benchmark),
        )
    model_registry = resolve(root, data.get("model_registry", "offline/models.yaml"))
    check("model_registry", model_registry.is_file(), str(model_registry))
    registry: dict = {}
    if model_registry.is_file() and not is_smoke:
        registry = yaml.safe_load(model_registry.read_text(encoding="utf-8")) or {}
        panel_key = str(run_name).strip().lower().replace("-", "_")
        actors = list((registry.get("actor_matrix") or {}).get(panel_key, []))
        defined_models = set((registry.get("models") or {}).keys())
        undefined_actors = [item for item in actors if item not in defined_models]
        check(
            "actor_matrix",
            bool(actors) and not undefined_actors,
            f"panel={panel_key}; actors={actors}; undefined={undefined_actors}",
        )
        profiler = str((registry.get("profiler") or {}).get("model") or "")
        check(
            "fixed_profiler",
            profiler == "qwen2.5-14b-instruct" and profiler in defined_models,
            profiler,
        )
    db = resolve(root, data.get("corpus_db", "missing"))
    check("corpus_db", db.is_file(), str(db))
    for index, value in enumerate(data.get("frozen_assets", [])):
        path = resolve(root, value)
        check(f"frozen_asset[{index}]", path.is_file(), str(path))

    if not is_smoke:
        manifest_value = data.get("bundle_manifest")
        manifest_path = resolve(root, manifest_value or "missing")
        manifest_failures: list[str] = []
        if not manifest_path.is_file():
            manifest_failures.append(f"missing frozen bundle manifest: {manifest_path}")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("research_ready") is not True:
                    manifest_failures.append("manifest is not research_ready")
                if manifest.get("read_only_execution") is not True:
                    manifest_failures.append("manifest is not read_only_execution")
                for item in manifest.get("artifacts", []):
                    artifact_path = resolve(root, item["path"])
                    if not artifact_path.is_file():
                        manifest_failures.append(f"missing artifact: {item['path']}")
                    elif sha256_file(artifact_path) != item["sha256"]:
                        manifest_failures.append(f"checksum mismatch: {item['path']}")
            except Exception as exc:  # noqa: BLE001
                manifest_failures.append(f"invalid manifest: {exc}")
        check(
            "frozen_bundle",
            not manifest_failures,
            "; ".join(manifest_failures) or str(manifest_path),
        )

    coverage = data.get("coverage")
    if db.is_file() and coverage:
        include_synthetic = bool(data.get("include_synthetic", False))
        failures = []
        connection = sqlite3.connect(db)
        try:
            for character_id in selected:
                synthetic_clause = "" if include_synthetic else " AND is_synthetic = 0"
                row = connection.execute(
                    "SELECT COUNT(*) AS comments, COUNT(DISTINCT platform) AS platforms, "
                    "COUNT(DISTINCT author_hash) AS authors FROM comments "
                    f"WHERE character_id = ?{synthetic_clause}",
                    (character_id,),
                ).fetchone()
                observed = dict(zip(("comments", "platforms", "authors"), row, strict=True))
                for metric, field in (
                    ("comments", "min_comments"),
                    ("platforms", "min_platforms"),
                    ("authors", "min_authors"),
                ):
                    if observed[metric] < int(coverage[field]):
                        failures.append(
                            f"{character_id}:{metric}={observed[metric]}<{coverage[field]}"
                        )
        finally:
            connection.close()
        check("corpus_coverage", not failures, "; ".join(failures) or "all selected roles pass")

    include_synthetic = bool(data.get("include_synthetic", False))
    check(
        "synthetic_gate",
        not include_synthetic or is_smoke,
        "synthetic data is allowed only for an explicitly named smoke run",
    )
    retrieval = config.get("retrieval", {})
    mode = retrieval.get("mode", "auto")
    vector_store_value = retrieval.get("vector_store")
    embedding_model_key = str(retrieval.get("embedding_model_key", ""))
    if mode == "hybrid" and not is_smoke:
        vector_store_path = resolve(root, vector_store_value or "missing")
        expected_revision = str(
            ((registry.get("models") or {}).get(embedding_model_key) or {}).get(
                "revision", ""
            )
        )
        vector_failures: list[str] = []
        try:
            from sideprofile.vector_store import verify_vector_store  # noqa: PLC0415

            vector_result = verify_vector_store(
                vector_store=vector_store_path,
                corpus_db=db,
                expected_model_key=embedding_model_key,
                expected_model_revision=expected_revision,
            )
            vector_failures.extend(vector_result["failures"])
        except Exception as exc:  # noqa: BLE001
            vector_failures.append(str(exc))
        check(
            "frozen_vector_store",
            not vector_failures and bool(expected_revision),
            "; ".join(vector_failures)
            or f"{embedding_model_key}@{expected_revision}: {vector_store_path}",
        )
    numpy_available = importlib.util.find_spec("numpy") is not None
    reranker_required = bool(retrieval.get("reranker_model"))
    reranker_available = importlib.util.find_spec("sentence_transformers") is not None
    check(
        "retrieval_runtime",
        mode != "hybrid"
        or (numpy_available and (not reranker_required or reranker_available)),
        f"mode={mode}; numpy={'available' if numpy_available else 'missing'}; "
        f"reranker_runtime={'available' if reranker_available else 'missing'}",
    )
    probes = config.get("profiling", {}).get("probe_ids")
    check(
        "probe_scope",
        probes is None or is_smoke or "ablation" in run_name.casefold(),
        "main runs use all 24 probes; subsets must be smoke or ablation",
    )
    result = {"ok": all(item["ok"] for item in checks), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
