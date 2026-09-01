#!/usr/bin/env python3
"""Build a reproducible character-length profile of the frozen research corpus."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def profile(values: list[int]) -> dict:
    return {
        "rows": len(values),
        "mean_characters": sum(values) / len(values),
        "p50_characters": percentile(values, 0.50),
        "p75_characters": percentile(values, 0.75),
        "p90_characters": percentile(values, 0.90),
        "p95_characters": percentile(values, 0.95),
        "p99_characters": percentile(values, 0.99),
        "max_characters": max(values),
        "over_500": sum(value > 500 for value in values),
        "over_1000": sum(value > 1000 for value in values),
        "over_2000": sum(value > 2000 for value in values),
        "over_5000": sum(value > 5000 for value in values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/corpus/comments.sqlite")
    parser.add_argument("--output", default="data/audits/comment_length_distribution.json")
    args = parser.parse_args()
    db = Path(args.db).resolve()
    connection = sqlite3.connect(
        f"{db.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        records = connection.execute(
            "SELECT character_id, platform, source_url, LENGTH(raw_text) "
            "FROM comments WHERE is_synthetic = 0"
        ).fetchall()
    finally:
        connection.close()
    by_platform: dict[str, list[int]] = defaultdict(list)
    by_character: dict[str, list[int]] = defaultdict(list)
    stackexchange_kind: dict[str, list[int]] = defaultdict(list)
    for character_id, platform, source_url, length in records:
        by_platform[str(platform)].append(int(length))
        by_character[str(character_id)].append(int(length))
        if platform == "stackexchange":
            kind = (
                "answer"
                if "#answer-" in source_url
                else "comment"
                if "#comment" in source_url
                else "question"
            )
            stackexchange_kind[kind].append(int(length))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grain": "one non-synthetic imported corpus record",
        "unit": "Unicode characters in raw_text",
        "overall": profile([int(row[3]) for row in records]),
        "by_platform": {
            key: profile(values) for key, values in sorted(by_platform.items())
        },
        "by_character": {
            key: profile(values) for key, values in sorted(by_character.items())
        },
        "stackexchange_record_kind": {
            key: profile(values)
            for key, values in sorted(stackexchange_kind.items())
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "rows": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
