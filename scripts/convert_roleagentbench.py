#!/usr/bin/env python3
"""Convert the pinned RoleAgentBench Core-10 subset into SideProfile inputs.

The official ``raw/role_summary.json`` files are the privileged profiles used only by the
Anonymous Gold Profile condition. Ours, Generic Summary, and Personality Only never
read these summaries. General Response and Reaction examples are retained with their official
references/choices in evaluator-only metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sideprofile.anonymize import anonymize_text
from sideprofile.corpus import load_character_catalog
from sideprofile.schema import BenchmarkExample


WORKS = {
    "Harry Potter": {
        "Harry": "hp_harry",
        "Hermione": "hp_hermione",
        "Ron": "hp_ron",
        "Malfoy": "hp_draco",
        "McGonagall": "hp_mcgonagall",
    },
    "The Big Bang Theory S1E1": {
        "Sheldon": "tbbt_sheldon",
        "Leonard": "tbbt_leonard",
        "Penny": "tbbt_penny",
        "Raj": "tbbt_raj",
        "Howard": "tbbt_howard",
    },
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def convert(root: Path, catalog_path: Path) -> tuple[list[BenchmarkExample], list[object]]:
    specs = load_character_catalog(catalog_path)
    by_id = {spec.character_id: spec for spec in specs}
    examples: list[BenchmarkExample] = []
    for work, role_ids in WORKS.items():
        work_root = root / work
        summaries = load_json(work_root / "raw" / "role_summary.json")
        if not isinstance(summaries, dict):
            raise ValueError(f"invalid role summaries: {work}")
        for role, character_id in role_ids.items():
            if role not in summaries:
                raise RuntimeError(f"official summary missing {work}/{role}")
            spec = by_id[character_id]
            spec.gold_profile = anonymize_text(str(summaries[role]), spec)

        general = load_json(work_root / "general_response.json")
        reaction = load_json(work_root / "reaction.json")
        if not isinstance(general, list) or not isinstance(reaction, list):
            raise ValueError(f"invalid benchmark arrays: {work}")
        for index, row in enumerate(general, 1):
            target = str(row["target_role"])
            if target not in role_ids:
                continue
            spec = by_id[role_ids[target]]
            examples.append(
                BenchmarkExample(
                    example_id=f"roleagent-{work.casefold().replace(' ', '-')}-gr-{index:04d}",
                    character_id=spec.character_id,
                    query=anonymize_text(str(row["question"]), spec),
                    reference=anonymize_text(str(row.get("answer", "")), spec),
                    benchmark="RoleAgentBench",
                    metadata={
                        "task": "general_response",
                        "source_role": row.get("source_role", ""),
                        "target_role": target,
                        "official_type": row.get("type", ""),
                        "official_question": str(row["question"]),
                        "official_answer": str(row.get("answer", "")),
                    },
                )
            )
        for index, row in enumerate(reaction, 1):
            target = str(row["target_role"])
            if target not in role_ids:
                continue
            spec = by_id[role_ids[target]]
            examples.append(
                BenchmarkExample(
                    example_id=f"roleagent-{work.casefold().replace(' ', '-')}-reaction-{index:04d}",
                    character_id=spec.character_id,
                    query=anonymize_text(str(row["instruction"]), spec),
                    reference=str(row.get("gt_answer", "")),
                    benchmark="RoleAgentBench",
                    metadata={
                        "task": "reaction",
                        "source_role": row.get("source_role", ""),
                        "target_role": target,
                        "question": anonymize_text(str(row.get("question", "")), spec),
                        "multi_choices": [
                            anonymize_text(str(item), spec)
                            for item in row.get("multi_choices", [])
                        ],
                        "scene_id": row.get("scene_id"),
                        "official_instruction": str(row["instruction"]),
                    },
                )
            )
    return examples, specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roleagent-root", required=True)
    parser.add_argument("--catalog", default="data/catalog/characters.json")
    parser.add_argument("--benchmark-out", required=True)
    parser.add_argument("--catalog-out", required=True)
    args = parser.parse_args()

    examples, specs = convert(Path(args.roleagent_root), Path(args.catalog))
    benchmark_out = Path(args.benchmark_out)
    benchmark_out.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json() + "\n")
    catalog_out = Path(args.catalog_out)
    catalog_out.parent.mkdir(parents=True, exist_ok=True)
    catalog_out.write_text(
        json.dumps(
            {"schema_version": 1, "characters": [spec.model_dump() for spec in specs]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    task_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for example in examples:
        task = str(example.metadata["task"])
        task_counts[task] = task_counts.get(task, 0) + 1
        role_counts[example.character_id] = role_counts.get(example.character_id, 0) + 1
    print(
        json.dumps(
            {
                "benchmark": str(benchmark_out.resolve()),
                "catalog": str(catalog_out.resolve()),
                "examples": len(examples),
                "tasks": task_counts,
                "per_role": role_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
