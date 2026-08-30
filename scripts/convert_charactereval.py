#!/usr/bin/env python3
"""Convert an authorized CharacterEval checkout into SideProfile inputs.

The official file named test_data.jsonl is a JSON array in the published release.
Target identity strings are masked in actor-visible dialogue. Official profiles are
stored only in the private catalog for the anonymous-gold baseline and evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sideprofile.anonymize import anonymize_text
from sideprofile.corpus import load_character_catalog
from sideprofile.schema import BenchmarkExample


ROLE_IDS = {
    "华妃": "ce_huafei",
    "吕子乔": "ce_luzhiqiao",
    "老默": "ce_laomo",
    "朱朝阳": "ce_zhuchaoyang",
    "孟宴臣": "ce_mengyanchen",
    "许红豆": "ce_xuhongdou",
}


def render_profile(profile: dict) -> str:
    lines = []
    for key, value in profile.items():
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                lines.append(f"{key}：" + "；".join(values))
        elif str(value).strip():
            lines.append(f"{key}：{value}")
    return "\n".join(lines)


def load_test(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--catalog", default="data/catalog/characters.json")
    parser.add_argument("--benchmark-out", required=True)
    parser.add_argument("--catalog-out", required=True)
    parser.add_argument("--per-role", type=int, default=0, help="0 keeps every official example")
    args = parser.parse_args()

    specs = load_character_catalog(args.catalog)
    by_id = {spec.character_id: spec for spec in specs}
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    selected = []
    counts = {role: 0 for role in ROLE_IDS}
    for row in sorted(load_test(Path(args.test_data)), key=lambda item: int(item["id"])):
        role = row.get("role")
        if role not in ROLE_IDS:
            continue
        if args.per_role and counts[role] >= args.per_role:
            continue
        character_id = ROLE_IDS[role]
        spec = by_id[character_id]
        selected.append(
            BenchmarkExample(
                example_id=f"charactereval-{row['id']}",
                character_id=character_id,
                query="作为匿名目标自然地继续这段对话。只输出下一句台词，不添加说话人标签。",
                context=anonymize_text(str(row.get("context", "")), spec),
                benchmark="CharacterEval",
                metadata={
                    "official_id": row["id"],
                    "split": "test",
                    "source_role": role,
                    "source_work": row.get("novel_name", ""),
                    "official_context": str(row.get("context", "")),
                },
            )
        )
        counts[role] += 1

    for role, character_id in ROLE_IDS.items():
        if role not in profiles:
            raise RuntimeError(f"official profile missing selected role: {role}")
        spec = by_id[character_id]
        spec.gold_profile = anonymize_text(render_profile(profiles[role]), spec)

    benchmark_out = Path(args.benchmark_out)
    benchmark_out.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_out.open("w", encoding="utf-8") as handle:
        for example in selected:
            handle.write(example.model_dump_json() + "\n")
    catalog_out = Path(args.catalog_out)
    catalog_out.parent.mkdir(parents=True, exist_ok=True)
    catalog_out.write_text(
        json.dumps(
            {"schema_version": 1, "characters": [spec.model_dump() for spec in specs]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "benchmark": str(benchmark_out.resolve()),
                "catalog": str(catalog_out.resolve()),
                "examples": len(selected),
                "per_role": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
