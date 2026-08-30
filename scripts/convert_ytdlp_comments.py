#!/usr/bin/env python3
"""Convert a yt-dlp metadata file into the audited comment-capture schema.

This is a connected preparation-side utility. It preserves canonical public comment IDs and author
identifiers only in the temporary capture; the importer later replaces author identifiers with
salted hashes and keeps raw text private.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


LICENSE_NOTE = (
    "Publicly visible YouTube comment retained privately for research audit; canonical permalink "
    "preserved; redistribution is not authorized."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="yt-dlp .info.json with comments")
    parser.add_argument("--output", required=True, help="temporary capture JSONL")
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    video_id = str(source["id"])
    video_url = str(source.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}")
    title = str(source.get("title") or video_id)
    rows = []
    for index, item in enumerate(source.get("comments") or [], 1):
        text = " ".join(str(item.get("text") or "").split())
        if not text:
            continue
        comment_id = str(item.get("id") or f"visible-{index}")
        author_id = str(
            item.get("author_id")
            or item.get("author_url")
            or item.get("author")
            or f"visible-author-{index}"
        )
        timestamp = item.get("timestamp")
        published = (
            datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
            if timestamp is not None
            else ""
        )
        rows.append(
            {
                "platform": "youtube",
                "thread_id": video_id,
                "source_comment_id": comment_id,
                "author_source_id": author_id,
                "raw_text": text,
                "source_url": f"{video_url}&lc={comment_id}",
                "source_title": title,
                "timestamp": published,
                "language": "",
                "collection_method": "youtube_public_comment_pages",
                "license_note": LICENSE_NOTE,
                "target_relevant": None,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "video_id": video_id,
                "captured": len(rows),
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
