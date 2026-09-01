#!/usr/bin/env python3
"""Audit actual vLLM token usage and per-probe evidence size without changing inputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


CONTEXT_KEYS = (
    "max_position_embeddings", "n_positions", "seq_length", "max_seq_len", "model_max_length"
)


def native_context_window(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in CONTEXT_KEYS:
        value = int(config.get(key, 0) or 0)
        if value > 0:
            return value
    raise RuntimeError(f"no native context field in {config_path}")


def agent_group(agent: str) -> str:
    if agent.startswith("cue:"):
        return "cue_per_probe"
    if agent.startswith("condition_summary:"):
        return (
            "summary_aggregate"
            if agent.endswith(":aggregate")
            else "summary_per_probe"
        )
    if agent.startswith("condition_personality:"):
        return (
            "personality_aggregate"
            if agent.endswith(":aggregate")
            else "personality_per_probe"
        )
    return agent


def load_trace(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_trace(paths: list[Path], context_window: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    failures = []
    records = 0
    for path in paths:
        for row in load_trace(path):
            records += 1
            usage = row.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = prompt_tokens + completion_tokens
            agent = str(row.get("agent", "unknown"))
            item = {
                "trace": str(path),
                "agent": agent,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "remaining_context": context_window - total_tokens,
            }
            groups[agent_group(agent)].append(item)
            if prompt_tokens <= 0:
                failures.append(f"missing prompt token usage: {path}:{agent}")
            if total_tokens > context_window:
                failures.append(
                    f"native context exceeded: {path}:{agent} "
                    f"{total_tokens}>{context_window}"
                )
    summaries = []
    for group, items in sorted(groups.items()):
        maximum = max(items, key=lambda item: item["total_tokens"])
        summaries.append(
            {
                "group": group,
                "calls": len(items),
                "maximum_observed_prompt_tokens": max(
                    item["prompt_tokens"] for item in items
                ),
                "maximum_observed_completion_tokens": max(
                    item["completion_tokens"] for item in items
                ),
                "maximum_observed_total_tokens": maximum["total_tokens"],
                "minimum_remaining_context": min(
                    item["remaining_context"] for item in items
                ),
                "max_record": maximum,
            }
        )
    return {
        "ok": bool(records) and not failures,
        "records": records,
        "native_context_window": context_window,
        "groups": summaries,
        "failures": failures or ([] if records else ["no trace records"]),
    }


def prepared_evidence(prepared_dir: Path) -> dict:
    rows = []
    for path in sorted((prepared_dir / "characters").glob("*/retrieval.json")):
        character_id = path.parent.name
        retrieval = json.loads(path.read_text(encoding="utf-8"))
        for probe_id, comments in retrieval.items():
            rows.append(
                {
                    "character_id": character_id,
                    "probe_id": probe_id,
                    "comments": len(comments),
                    "characters": sum(len(str(item.get("text", ""))) for item in comments),
                }
            )
    return {
        "probe_sets": len(rows),
        "max_comments_per_probe": max((row["comments"] for row in rows), default=0),
        "max_characters_per_probe": max(
            (row["characters"] for row in rows), default=0
        ),
        "max_probe_set": max(rows, key=lambda row: row["characters"], default=None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--prepared-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    traces = [Path(value).resolve() for value in args.trace]
    context_window = native_context_window(model_dir)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "trace_audit": audit_trace(traces, context_window),
    }
    if args.prepared_dir:
        payload["prepared_evidence"] = prepared_evidence(
            Path(args.prepared_dir).resolve()
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["trace_audit"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
