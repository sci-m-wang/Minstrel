#!/usr/bin/env python3
"""Select the evaluator-only CharacterEval assets for the frozen six-role panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLES = ["华妃", "吕子乔", "老默", "朱朝阳", "孟宴臣", "许红豆"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--id2metric", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", default="data/evaluators/charactereval")
    args = parser.parse_args()
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    id2metric = json.loads(Path(args.id2metric).read_text(encoding="utf-8"))
    official_ids = set()
    with Path(args.benchmark).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                official_ids.add(str(row["metadata"]["official_id"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_profiles = {role: profiles[role] for role in ROLES}
    selected_metrics = {key: id2metric[key] for key in sorted(official_ids) if key in id2metric}
    (output / "character_profiles.selected.json").write_text(
        json.dumps(selected_profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "id2metric.selected.json").write_text(
        json.dumps(selected_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "profiles": len(selected_profiles),
                "benchmark_ids": len(official_ids),
                "metric_ids": len(selected_metrics),
                "output_dir": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
