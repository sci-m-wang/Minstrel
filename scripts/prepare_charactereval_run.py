#!/usr/bin/env python3
"""Transform one SideProfile actor run into the official CharacterRM input schema."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark", default="data/benchmarks/panel_d.jsonl")
    parser.add_argument(
        "--metrics", default="data/evaluators/charactereval/id2metric.selected.json"
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    benchmark = {row["example_id"]: row for row in load_jsonl(Path(args.benchmark))}
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    transformed = []
    for record in load_jsonl(run_dir / "generations.jsonl"):
        example = benchmark[record["example_id"]]
        metadata = example["metadata"]
        official_id = str(metadata["official_id"])
        base = {
            "id": metadata["official_id"],
            "role": metadata["source_role"],
            "novel_name": metadata["source_work"],
            "context": metadata["official_context"],
            "model_output": record["output"].split("\n", 1)[0],
            "condition": record["condition"],
            "replicate": record["replicate"],
            "actor_model": record["actor_model"],
            "example_id": record["example_id"],
            "character_id": record["character_id"],
        }
        for metric_en, metric_zh in metrics.get(official_id, []):
            item = copy.deepcopy(base)
            item["metric_en"] = metric_en
            item["metric_zh"] = metric_zh
            transformed.append(item)
    output = run_dir / "charactereval-rm-input.json"
    output.write_text(
        json.dumps(transformed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(output), "records": len(transformed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
