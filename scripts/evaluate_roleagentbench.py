#!/usr/bin/env python3
"""Run the official offline Reaction accuracy portion of RoleAgentBench.

General Response was reported with human/GPT-4 pairwise win rate in the official paper and therefore
cannot be reproduced on a disconnected host. This script records that limitation rather than silently
substituting a local judge.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_choice(output: str, choices: list[str]) -> str | None:
    match = re.search(r"(?:^|\s)([A-D])(?:[.):\s]|$)", output.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    normalized = " ".join(output.casefold().split())
    for index, choice in enumerate(choices):
        text = re.sub(r"^[A-D][.):]?\s*", "", choice, flags=re.IGNORECASE)
        if " ".join(text.casefold().split()) in normalized:
            return chr(ord("A") + index)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark", default="data/benchmarks/panel_a.jsonl")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    benchmark = {row["example_id"]: row for row in load_jsonl(Path(args.benchmark))}
    records = load_jsonl(run_dir / "generations.jsonl")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    scored = []
    general_count = 0
    for record in records:
        example = benchmark[record["example_id"]]
        task = example["metadata"]["task"]
        if task == "general_response":
            general_count += 1
            continue
        choice = selected_choice(record["output"], example["metadata"]["multi_choices"])
        correct = int(choice == example["reference"])
        grouped[(record["condition"], int(record["replicate"]))].append(correct)
        scored.append(
            {
                "example_id": record["example_id"],
                "character_id": example["character_id"],
                "condition": record["condition"],
                "replicate": record["replicate"],
                "selected": choice,
                "gold": example["reference"],
                "correct": bool(correct),
            }
        )
    summary = [
        {
            "condition": condition,
            "replicate": replicate,
            "reaction_examples": len(values),
            "accuracy": sum(values) / len(values) if values else None,
        }
        for (condition, replicate), values in sorted(grouped.items())
    ]
    payload = {
        "benchmark": "RoleAgentBench",
        "reaction_evaluator": "official exact accuracy",
        "general_response_evaluator": "pending human/GPT-4 pairwise evaluation; no offline substitute used",
        "general_response_generations": general_count,
        "summary": summary,
        "records": scored,
    }
    output = run_dir / "roleagentbench-evaluation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "groups": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
