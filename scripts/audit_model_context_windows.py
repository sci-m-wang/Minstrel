#!/usr/bin/env python3
"""Inventory native context windows from the exact local model configurations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CONTEXT_KEYS = (
    "max_position_embeddings", "n_positions", "seq_length", "max_seq_len", "model_max_length"
)
UNDERSIZED_ACTOR_MAX = 8192


def native_context(config: dict[str, Any]) -> tuple[str, int]:
    for key in CONTEXT_KEYS:
        value = int(config.get(key, 0) or 0)
        if value > 0:
            return key, value
    raise ValueError("no native context field")


def read_optional_tokenizer_limit(model_dir: Path) -> int | None:
    path = model_dir / "tokenizer_config.json"
    if not path.is_file():
        return None
    value = int(json.loads(path.read_text(encoding="utf-8")).get("model_max_length", 0) or 0)
    return value if value > 0 else None


def inventory_contexts(registry_path: Path, asset_root: Path) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    asset_root = asset_root.resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    failures = []
    models: dict[str, dict[str, Any]] = {}
    for key, spec in registry.get("models", {}).items():
        model_dir = asset_root / "models" / key
        config_path = model_dir / "config.json"
        item = {
            "purpose": spec.get("purpose"),
            "repo_id": spec.get("repo_id"),
            "revision": spec.get("revision"),
            "config_path": str(config_path),
        }
        if not config_path.is_file():
            failures.append(f"missing config.json for {key}: {config_path}")
            item["error"] = "missing config.json"
            models[key] = item
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        try:
            field, context = native_context(config)
        except ValueError as exc:
            failures.append(f"{key}: {exc} in {config_path}")
            item["error"] = str(exc)
            models[key] = item
            continue
        item.update(
            {
                "native_context_field": field,
                "native_context_tokens": context,
                "tokenizer_model_max_length": read_optional_tokenizer_limit(model_dir),
            }
        )
        models[key] = item

    actor_matrices = {}
    actor_keys = set()
    for panel, keys in registry.get("actor_matrix", {}).items():
        actor_keys.update(keys)
        contexts = [
            int(models[key]["native_context_tokens"])
            for key in keys
            if key in models and "native_context_tokens" in models[key]
        ]
        missing = [key for key in keys if key not in models or "native_context_tokens" not in models[key]]
        if missing:
            failures.append(f"{panel}: actor context missing for {', '.join(missing)}")
        actor_matrices[panel] = {
            "actors": list(keys),
            "minimum_native_context_tokens": min(contexts) if contexts else None,
        }
    undersized = sorted(
        key
        for key in actor_keys
        if key in models
        and 0 < int(models[key].get("native_context_tokens", 0)) <= UNDERSIZED_ACTOR_MAX
    )
    profiler_spec = registry.get("profiler") or {}
    profiler = str(profiler_spec.get("model", ""))
    profiler_provider = str(profiler_spec.get("provider", "")).upper()
    profiler_location = str(profiler_spec.get("execution_location", ""))
    profiler_context = None
    if profiler in models and "native_context_tokens" in models.get(profiler, {}):
        profiler_context = models[profiler]["native_context_tokens"]
    elif profiler_location != "connected_preparation" or not profiler_provider:
        failures.append(f"profiler context missing for {profiler or 'undefined'}")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": not failures,
        "registry": str(registry_path),
        "asset_root": str(asset_root),
        "undersized_actor_rule": f"native_context_tokens <= {UNDERSIZED_ACTOR_MAX}",
        "profiler": {
            "provider": profiler_provider,
            "model": profiler,
            "execution_location": profiler_location,
            "native_context_tokens": profiler_context,
            "context_evidence": (
                "local_config" if profiler_context is not None else "provider_usage_trace"
            ),
        },
        "models": models,
        "actor_matrices": actor_matrices,
        "undersized_actors": undersized,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="offline/models.yaml")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = inventory_contexts(Path(args.registry), Path(args.asset_root))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
