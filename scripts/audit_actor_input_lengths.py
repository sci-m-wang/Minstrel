#!/usr/bin/env python3
"""Tokenize every exact prepared Actor input without changing or truncating it."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sideprofile.pipeline import load_benchmark
from sideprofile.profile import render_person_model
from sideprofile.roleplay import ACTOR_SYSTEM
from sideprofile.schema import BenchmarkExample, PersonModel
from sideprofile.staged import verify_prepared


CONTEXT_KEYS = (
    "max_position_embeddings", "n_positions", "seq_length", "max_seq_len", "model_max_length"
)


def actor_messages(
    *,
    example: BenchmarkExample,
    condition: str,
    conditioning_text: str,
    person_model: PersonModel | None,
) -> list[dict[str, str]]:
    if condition == "ours":
        if person_model is None:
            raise ValueError("ours condition requires a person model")
        conditioning_text = render_person_model(person_model)
    payload = {
        "anonymous_id": person_model.anonymous_id if person_model and condition == "ours" else "TARGET",
        "conditioning": conditioning_text,
        "context": example.context,
        "query": example.query,
    }
    return [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "shape"):
        return int(tokens.shape[-1])
    return len(tokens)


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def audit_message_rows(
    rows: list[dict[str, Any]], tokenizer: Any, *, context_window: int
) -> dict[str, Any]:
    measured = []
    failures = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        prompt_tokens = token_count(tokenizer, row["messages"])
        remaining = context_window - prompt_tokens
        record = {
            key: row[key]
            for key in ("panel", "character_id", "example_id", "condition")
        }
        record.update(
            {
                "prompt_tokens": prompt_tokens,
                "remaining_context": remaining,
            }
        )
        measured.append(record)
        grouped[str(row["condition"])].append(prompt_tokens)
        if remaining <= 0:
            failures.append(
                "no generation capacity: "
                f"{row['character_id']}:{row['example_id']}:{row['condition']} "
                f"uses {prompt_tokens}/{context_window} prompt tokens"
            )
    maximum = max(measured, key=lambda item: item["prompt_tokens"], default=None)
    return {
        "ok": bool(measured) and not failures,
        "inputs": len(measured),
        "native_context_window": context_window,
        "maximum_prompt_tokens": maximum["prompt_tokens"] if maximum else 0,
        "minimum_remaining_context": min(
            (item["remaining_context"] for item in measured), default=context_window
        ),
        "maximum_record": maximum,
        "by_condition": [
            {
                "condition": condition,
                "inputs": len(values),
                "maximum_prompt_tokens": max(values),
                "p95_prompt_tokens": percentile(values, 0.95),
            }
            for condition, values in sorted(grouped.items())
        ],
        "failures": failures or ([] if measured else ["no Actor inputs"]),
    }


def native_context(model_dir: Path) -> tuple[str, int]:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in CONTEXT_KEYS:
        value = int(config.get(key, 0) or 0)
        if value > 0:
            return key, value
    raise RuntimeError(f"no native context field in {config_path}")


def project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in [resolved.parent, *resolved.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/sideprofile").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate project root from {resolved}")


def build_message_rows(config_path: Path, prepared_dir: Path) -> tuple[list[dict[str, Any]], dict]:
    verified = verify_prepared(prepared_dir)
    if not verified["ok"]:
        raise RuntimeError("invalid prepared directory: " + "; ".join(verified["failures"]))
    manifest = verified["manifest"]
    root = project_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark = load_benchmark(root / config["data"]["benchmark"])
    rows = []
    for character_id in manifest["characters"]:
        character_dir = prepared_dir / "characters" / character_id
        person_model = PersonModel.model_validate_json(
            (character_dir / "person_model.json").read_text(encoding="utf-8")
        )
        conditionings = json.loads(
            (character_dir / "conditionings.json").read_text(encoding="utf-8")
        )
        for example in [item for item in benchmark if item.character_id == character_id]:
            for condition in manifest["conditions"]:
                rows.append(
                    {
                        "panel": config["run"]["name"],
                        "character_id": character_id,
                        "example_id": example.example_id,
                        "condition": condition,
                        "messages": actor_messages(
                            example=example,
                            condition=condition,
                            conditioning_text=str(conditionings[condition]),
                            person_model=person_model if condition == "ours" else None,
                        ),
                    }
                )
    return rows, config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config_path = Path(args.config).resolve()
    prepared_dir = Path(args.prepared_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    rows, config = build_message_rows(config_path, prepared_dir)
    context_field, context_window = native_context(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    audit = audit_message_rows(rows, tokenizer, context_window=context_window)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model_key,
        "model_dir": str(model_dir),
        "native_context_field": context_field,
        "replicates": int(config["run"].get("replicates", 1)),
        "unique_inputs": len(rows),
        "planned_generation_calls": len(rows) * int(config["run"].get("replicates", 1)),
        "audit": audit,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

