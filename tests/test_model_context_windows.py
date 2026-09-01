from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_model_context_windows.py"
SPEC = importlib.util.spec_from_file_location("audit_model_context_windows", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_model(root: Path, key: str, config: dict, tokenizer: dict | None = None) -> None:
    path = root / "models" / key
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if tokenizer is not None:
        (path / "tokenizer_config.json").write_text(json.dumps(tokenizer), encoding="utf-8")


def test_inventory_reports_native_fields_and_undersized_actors(tmp_path: Path) -> None:
    registry = {
        "profiler": {"model": "large"},
        "actor_matrix": {"panel_a": ["small", "large"], "panel_d": ["large"]},
        "models": {
            "small": {"purpose": "actor", "repo_id": "test/small", "revision": "a"},
            "large": {"purpose": "profiler_and_actor", "repo_id": "test/large", "revision": "b"},
            "embed": {"purpose": "frozen_vector_store_construction", "repo_id": "test/embed", "revision": "c"},
            "reranker": {"purpose": "reranking", "repo_id": "test/reranker", "revision": "d"},
            "rm": {"purpose": "official_evaluator", "repo_id": "test/rm", "revision": "e"},
        },
    }
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    write_model(
        tmp_path,
        "small",
        {"max_position_embeddings": 8192},
        {"model_max_length": 8192},
    )
    write_model(tmp_path, "large", {"n_positions": 32768})
    write_model(tmp_path, "embed", {"seq_length": 32768})
    write_model(tmp_path, "reranker", {"max_seq_len": 8194})
    write_model(tmp_path, "rm", {"model_max_length": 4096})

    result = module.inventory_contexts(registry_path, tmp_path)

    assert result["ok"]
    assert result["models"]["small"]["native_context_tokens"] == 8192
    assert result["models"]["small"]["native_context_field"] == "max_position_embeddings"
    assert result["models"]["small"]["tokenizer_model_max_length"] == 8192
    assert result["models"]["large"]["native_context_field"] == "n_positions"
    assert result["models"]["embed"]["native_context_field"] == "seq_length"
    assert result["models"]["reranker"]["native_context_field"] == "max_seq_len"
    assert result["models"]["rm"]["native_context_tokens"] == 4096
    assert result["undersized_actors"] == ["small"]
    assert result["actor_matrices"]["panel_a"]["minimum_native_context_tokens"] == 8192


def test_inventory_fails_when_native_context_is_missing(tmp_path: Path) -> None:
    registry = {
        "profiler": {"model": "broken"},
        "actor_matrix": {"panel_a": ["broken"]},
        "models": {
            "broken": {"purpose": "actor", "repo_id": "test/broken", "revision": "a"}
        },
    }
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    write_model(tmp_path, "broken", {"hidden_size": 8})

    result = module.inventory_contexts(registry_path, tmp_path)

    assert not result["ok"]
    assert "no native context field" in result["failures"][0]


def test_external_profiler_uses_provider_trace_not_local_model_asset(tmp_path: Path) -> None:
    registry = {
        "profiler": {
            "provider": "GPT",
            "model": "gpt-5.6-sol",
            "execution_location": "connected_preparation",
        },
        "actor_matrix": {"panel_a": ["actor"]},
        "models": {
            "actor": {"purpose": "actor", "repo_id": "test/actor", "revision": "a"}
        },
    }
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    write_model(tmp_path, "actor", {"max_position_embeddings": 32768})

    result = module.inventory_contexts(registry_path, tmp_path)

    assert result["ok"]
    assert result["profiler"] == {
        "provider": "GPT",
        "model": "gpt-5.6-sol",
        "execution_location": "connected_preparation",
        "native_context_tokens": None,
        "context_evidence": "provider_usage_trace",
    }
