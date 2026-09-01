#!/usr/bin/env python3
"""Build a redistribution-safe inventory for the frozen private comment corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PANELS = {
    "A": {
        "character_ids": [
            "hp_harry", "hp_hermione", "hp_ron", "hp_draco", "hp_mcgonagall",
            "tbbt_sheldon", "tbbt_leonard", "tbbt_penny", "tbbt_raj", "tbbt_howard",
        ],
        "minimums": {"comments": 1, "platforms": 2, "authors": 100},
    },
    "D": {
        "character_ids": [
            "ce_huafei", "ce_luzhiqiao", "ce_laomo", "ce_zhuchaoyang",
            "ce_mengyanchen", "ce_xuhongdou",
        ],
        "minimums": {"comments": 1, "platforms": 2, "authors": 100},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/corpus/comments.sqlite")
    parser.add_argument("--decisions-dir", default="data/audits")
    parser.add_argument("--import-log", default="data/audits/comment_imports.jsonl")
    parser.add_argument("--output", default="data/audits/corpus_inventory.json")
    args = parser.parse_args()

    db = Path(args.db)
    decisions_dir = Path(args.decisions_dir)
    connection = sqlite3.connect(
        f"{db.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        characters = [
            dict(row)
            for row in connection.execute(
                "SELECT character_id, COUNT(*) AS comments, "
                "COUNT(DISTINCT platform) AS platforms, "
                "COUNT(DISTINCT author_hash) AS authors "
                "FROM comments WHERE is_synthetic=0 GROUP BY character_id ORDER BY character_id"
            )
        ]
        sources = [
            dict(row)
            for row in connection.execute(
                "SELECT character_id, platform, thread_id, collection_method, COUNT(*) AS comments, "
                "COUNT(DISTINCT author_hash) AS authors FROM comments WHERE is_synthetic=0 "
                "GROUP BY character_id, platform, thread_id, collection_method "
                "ORDER BY character_id, platform, thread_id, collection_method"
            )
        ]
        platforms = [
            dict(row)
            for row in connection.execute(
                "SELECT platform, collection_method, COUNT(*) AS comments, "
                "COUNT(DISTINCT author_hash) AS authors FROM comments WHERE is_synthetic=0 "
                "GROUP BY platform, collection_method ORDER BY platform, collection_method"
            )
        ]
        synthetic = connection.execute(
            "SELECT COUNT(*) FROM comments WHERE is_synthetic=1"
        ).fetchone()[0]
    finally:
        connection.close()

    by_character = {row["character_id"]: row for row in characters}
    panel_audits = {}
    for panel, spec in PANELS.items():
        rows = []
        for character_id in spec["character_ids"]:
            observed = by_character.get(
                character_id,
                {"character_id": character_id, "comments": 0, "platforms": 0, "authors": 0},
            )
            failures = [
                metric
                for metric, minimum in spec["minimums"].items()
                if observed[metric] < minimum
            ]
            rows.append({**observed, "ready": not failures, "failures": failures})
        panel_audits[panel] = {
            "minimums": spec["minimums"],
            "ready": all(row["ready"] for row in rows),
            "characters": rows,
        }

    latest_decisions = {}
    for path in sorted(decisions_dir.glob("comment_relevance*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (
                str(item.get("character_id", "")),
                str(item.get("platform", "")),
                str(item.get("source_comment_id", "")),
            )
            previous = latest_decisions.get(key)
            # A transient provider failure is not a label. Preserve a completed adjudication if
            # the same public comment appears in overlapping captures or is successfully resumed.
            if (
                previous
                and previous.get("classification_status") != "classification_error"
                and item.get("classification_status") == "classification_error"
            ):
                continue
            latest_decisions[key] = item
    decision_status = {}
    for item in latest_decisions.values():
        status = str(item.get("classification_status", "locally_adjudicated"))
        decision_status[status] = decision_status.get(status, 0) + 1

    latest_imports = {}
    import_log = Path(args.import_log)
    if import_log.is_file():
        for line in import_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            latest_imports[(item.get("character_id"), item.get("capture_file"))] = item

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_db": str(db),
        "corpus_sha256": sha256(db),
        "raw_text_redistribution": False,
        "synthetic_comments": synthetic,
        "panel_audits": panel_audits,
        "characters": characters,
        "platforms": platforms,
        "sources": sources,
        "relevance_decisions": {
            "unique_rows": len(latest_decisions),
            "latest_status_counts": dict(sorted(decision_status.items())),
        },
        "latest_import_records": sorted(
            latest_imports.values(),
            key=lambda item: (str(item.get("character_id")), str(item.get("capture_file"))),
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), **panel_audits}, ensure_ascii=False, indent=2))
    return 0 if all(item["ready"] for item in panel_audits.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
