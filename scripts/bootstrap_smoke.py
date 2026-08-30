#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sideprofile.corpus import CommentCorpus, load_character_catalog, load_comments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/corpus/smoke.sqlite")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    db = Path(args.db)
    if not db.is_absolute():
        db = root / db
    with CommentCorpus(db) as corpus:
        corpus.add_characters(load_character_catalog(root / "data/catalog/characters.json"))
        inserted, duplicates = corpus.add_comments(load_comments(root / "data/smoke/comments.jsonl"))
        print(f"smoke corpus ready: {db} (inserted={inserted}, duplicates={duplicates})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
